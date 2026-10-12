"""
llm_client.py
The only module in BioSync that talks to an LLM provider.

Providers: Gemini (a key from https://aistudio.google.com/apikey) and Claude
(a key from https://console.anthropic.com). There is no shared "free tier": a
schedule is built with either a key the student pastes into the chat page
(sent per request in X-Gemini-Api-Key / X-Anthropic-Api-Key) or a key the
server owner configures in GEMINI_API_KEY / ANTHROPIC_API_KEY.

What this layer guarantees, and why it exists:

* Model fallback. Vendors retire and restrict models faster than apps get
  redeployed (Google limited the Gemini 2.5 family to existing users in Sept
  2026). Each task walks an ordered chain of current models and skips any
  that answer "not found / not available to this key". If every configured
  model fails that way it asks the provider which models the key can use and
  tries the best few of those.
* Thinking-token safety. Gemini counts "thinking" tokens against
  max_output_tokens, so a small cap can come back empty or cut off mid-JSON.
  Thinking is kept low, caps are generous, and a truncated answer is retried
  once with a bigger cap and less thinking before it is reported.
* One error vocabulary. Every failure becomes an LLMError whose `reason` is a
  stable code (bad_key, rate_limited, model_unavailable, ...) with a message
  that is safe to show a student. Raw provider text only goes to the log, and
  only after API keys have been scrubbed from it.
* One JSON extractor (extract_json) and one retry-on-bad-JSON helper
  (generate_json), shared by the structuring and scheduling steps.

API keys are held in a Credentials object that the caller passes in. This
module never stores a key (the per-model health cache is keyed by a SHA-256
fingerprint), never logs one, and builds a fresh client for every call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
GEMINI = "gemini"
CLAUDE = "claude"
PROVIDERS = (GEMINI, CLAUDE)
PROVIDER_LABEL = {GEMINI: "Gemini", CLAUDE: "Claude"}
PROVIDER_VENDOR = {GEMINI: "Google", CLAUDE: "Anthropic"}
KEY_HINT = {
    GEMINI: "a Gemini API key from aistudio.google.com/apikey",
    CLAUDE: "a Claude API key from console.anthropic.com",
}

TASK_CHAT = "chat"          # rephrase the next checklist question (a sentence or two)
TASK_EXTRACT = "extract"    # syllabus text -> JSON
TASK_SCHEDULE = "schedule"  # the weekly schedule JSON (the product)
TASK_PING = "ping"          # "does this key work?"

EFFORTS = ("low", "medium", "high")
DEFAULT_EFFORT = "medium"

# ---------------------------------------------------------------------------
# Models. Change them here, or without a redeploy via the env vars named in
# _models_for(). Order matters: first = preferred, the rest = fallbacks.
# Verified against Google's and Anthropic's model/deprecation pages on
# 2026-10-10. Those pages move quickly, which is why this is a chain and not a
# single constant. Re-check them if schedules stop coming back.
# ---------------------------------------------------------------------------
_DEFAULT_MODELS: dict[tuple[str, str], tuple[str, ...]] = {
    (GEMINI, "fast"): (
        "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash",
        "gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash",
    ),
    (GEMINI, "balanced"): (
        "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
        "gemini-flash-latest", "gemini-3.1-flash-lite", "gemini-2.5-flash",
    ),
    (GEMINI, "best"): (
        "gemini-3.8-flash", "gemini-3.6-flash", "gemini-flash-latest",
        "gemini-3.5-flash-lite", "gemini-2.5-flash",
    ),
    (CLAUDE, "fast"): ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-haiku-4-5-20251001"),
    (CLAUDE, "balanced"): ("claude-sonnet-5-5", "claude-haiku-5-5", "claude-haiku-4-5-20251001"),
    (CLAUDE, "best"): ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"),
}

# The effort selector on the chat page picks the schedule model tier and how
# much per-block detail the prompt asks for (see schedule_generator.py).
_TIER_BY_EFFORT = {"low": "fast", "medium": "balanced", "high": "best"}

# Output-token caps. Gemini's include thinking tokens, hence the headroom: a
# full-detail week is ~5k tokens of JSON and thinking can add a few thousand.
_MAX_TOKENS = {
    (GEMINI, TASK_CHAT): 1024,
    (GEMINI, TASK_EXTRACT): 4096,
    (GEMINI, TASK_PING): 256,
    (GEMINI, TASK_SCHEDULE): {"low": 8192, "medium": 12288, "high": 16384},
    (CLAUDE, TASK_CHAT): 300,
    (CLAUDE, TASK_EXTRACT): 3000,
    (CLAUDE, TASK_PING): 32,
    (CLAUDE, TASK_SCHEDULE): {"low": 4000, "medium": 8000, "high": 16000},
}
_TOKEN_GROWTH_CAP = {GEMINI: 32768, CLAUDE: 32000}

# Per-call read timeout and overall deadline (seconds) for one ask().
_CALL_TIMEOUT = {TASK_CHAT: 18, TASK_PING: 20, TASK_EXTRACT: 60, TASK_SCHEDULE: 180}
_OVERALL_TIMEOUT = {TASK_CHAT: 30, TASK_PING: 30, TASK_EXTRACT: 90, TASK_SCHEDULE: 260}
# Longest rate-limit wait we will sit through inside one request before moving on.
_MAX_INLINE_WAIT = {TASK_CHAT: 3, TASK_PING: 3, TASK_EXTRACT: 12, TASK_SCHEDULE: 20}
_MAX_HTTP_CALLS_PER_ASK = 14
# A provider-wide outage looks like several different models failing in a transient way. After this many
# models have, stop walking the chain: the student gets a clear "try again in a minute" instead of a long wait.
_MAX_TRANSIENT_MODELS = 3

_sleep = time.sleep  # indirection so tests can run without real waiting


def normalize_effort(effort: Any) -> str:
    return effort if effort in EFFORTS else DEFAULT_EFFORT


def _models_for(provider: str, tier: str) -> tuple[str, ...]:
    """Env override, e.g. GEMINI_MODELS_BALANCED="gemini-3.8-flash,gemini-3.6-flash"."""
    raw = os.getenv(f"{provider.upper()}_MODELS_{tier.upper()}", "")
    items = tuple(m.strip() for m in raw.split(",") if m.strip())
    return items or _DEFAULT_MODELS[(provider, tier)]


def tier_for(task: str, effort: str) -> str:
    if task == TASK_SCHEDULE:
        return _TIER_BY_EFFORT[normalize_effort(effort)]
    return "fast"  # chat, extraction and pings are cheap jobs; never spend Opus on them


def model_chain(provider: str, task: str, effort: str = DEFAULT_EFFORT) -> tuple[str, ...]:
    return _models_for(provider, tier_for(task, effort))


def max_tokens_for(provider: str, task: str, effort: str = DEFAULT_EFFORT) -> int:
    value = _MAX_TOKENS[(provider, task)]
    return value[normalize_effort(effort)] if isinstance(value, dict) else value


# ---------------------------------------------------------------------------
# Thinking control (Gemini only)
# ---------------------------------------------------------------------------
_GEMINI_VERSION = re.compile(r"^gemini-(\d+)")


def _gemini_major(model: str) -> int | None:
    m = _GEMINI_VERSION.match(model)
    return int(m.group(1)) if m else None


def _thinking_choice(kind: str, task: str, effort: str) -> tuple[str, Any]:
    heavy = task == TASK_SCHEDULE and effort == "high"
    if kind == "level":
        return ("level", "medium" if heavy else "low")
    return ("budget", 4096 if heavy else (1024 if task == TASK_SCHEDULE and effort == "medium" else 0))


def gemini_thinking_plan(model: str, task: str, effort: str = DEFAULT_EFFORT) -> list[tuple[str, Any] | None]:
    """
    Ordered thinking settings to try for a model; None = send no thinking config.

    Gemini 3.x takes thinking_level; Gemini 2.5 takes thinking_budget (0 = off);
    sending both is a 400. Aliases such as gemini-flash-latest point at the
    newest family, so they are treated like 3.x. The last entry is always None,
    so a model that rejects every thinking option still gets called. What a
    model accepted earlier in this process is remembered and tried first.
    """
    effort = normalize_effort(effort)
    learned = _CAPS.get(model)
    if learned == "none":
        return [None]
    kind = learned or ("budget" if _gemini_major(model) == 2 else "level")
    return [_thinking_choice(kind, task, effort), None]


def _thinking_config(choice: tuple[str, Any] | None):
    if not choice:
        return None
    kind, value = choice
    if kind == "level":
        return genai_types.ThinkingConfig(thinking_level=value)
    return genai_types.ThinkingConfig(thinking_budget=value)


# ---------------------------------------------------------------------------
# Small TTL memory of what we learned about models (never holds a key)
# ---------------------------------------------------------------------------
class _TTLCache:
    def __init__(self, max_items: int = 512):
        self._items: dict[Any, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._max = max_items

    def get(self, key: Any) -> Any:
        with self._lock:
            hit = self._items.get(key)
            if not hit:
                return None
            value, expires = hit
            if expires < time.monotonic():
                del self._items[key]
                return None
            return value

    def set(self, key: Any, value: Any, ttl: float) -> None:
        with self._lock:
            if len(self._items) >= self._max:
                now = time.monotonic()
                for k in [k for k, (_, exp) in self._items.items() if exp < now]:
                    del self._items[k]
                if len(self._items) >= self._max:
                    self._items.pop(next(iter(self._items)))
            self._items[key] = (value, time.monotonic() + ttl)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_DEAD = _TTLCache()   # (provider, key fingerprint, model) -> reason it is unusable right now
_GOOD = _TTLCache()   # (provider, key fingerprint, tier) -> last model that answered
_CAPS = _TTLCache()   # model -> "level" | "budget" | "none" (which thinking setting it accepts)


def reset_caches() -> None:
    """Forget everything learned about models. Used by tests and scripts."""
    for cache in (_DEAD, _GOOD, _CAPS):
        cache.clear()


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Credentials:
    """A provider plus its API key. The key is excluded from repr() on purpose."""
    provider: str
    api_key: str = field(repr=False)
    source: str = "user"  # "user" = pasted by the student, "server" = env var

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:12]


def clean_key(raw: Any) -> str | None:
    """Trim whitespace and stray quotes from a pasted key; None if nothing is left."""
    if raw is None:
        return None
    key = str(raw).strip().strip("'\"").strip()
    return key or None


def check_key_shape(provider: str, key: str) -> None:
    """
    Reject keys that cannot possibly work (non-ASCII, inner whitespace, absurd
    length) before they hit an HTTP header and fail in a confusing way. The
    prefix (AIza / sk-ant-) is deliberately not enforced: vendors change key
    formats, and the provider is the real judge.
    """
    if (len(key) < 16 or len(key) > 512 or re.search(r"\s", key)
            or not key.isascii() or not key.isprintable()):
        raise LLMError("bad_key", provider=provider, detail="key has an impossible shape")


def _server_key(provider: str) -> str | None:
    return clean_key(os.getenv("GEMINI_API_KEY" if provider == GEMINI else "ANTHROPIC_API_KEY"))


def _default_provider() -> str | None:
    """DEFAULT_LLM_PROVIDER, or None for 'whichever server key exists'."""
    raw = os.getenv("DEFAULT_LLM_PROVIDER", "").strip().lower()
    if raw in PROVIDERS:
        return raw
    if raw and raw not in ("auto",):
        logger.warning("DEFAULT_LLM_PROVIDER=%r is not supported (use 'gemini' or 'claude'); auto-detecting.", raw)
    return None


def server_llm_info() -> dict:
    """Which server-side keys are configured. Booleans only, never key material."""
    available = [p for p in PROVIDERS if _server_key(p)]
    preferred = _default_provider()
    provider = preferred if preferred in available else (available[0] if available else None)
    return {"available": bool(available), "provider": provider, "providers": available}


def resolve_credentials(user_claude_key: Any = None, user_gemini_key: Any = None) -> Credentials:
    """
    Pick the key for this request. Order: a Claude key pasted by the student,
    then a Gemini key pasted by the student (if both headers arrive the Claude
    one wins, as it always has), then the server's own key. Raises
    LLMError("no_key") when there is none.
    """
    claude, gemini = clean_key(user_claude_key), clean_key(user_gemini_key)
    if claude:
        check_key_shape(CLAUDE, claude)
        return Credentials(CLAUDE, claude, "user")
    if gemini:
        check_key_shape(GEMINI, gemini)
        return Credentials(GEMINI, gemini, "user")
    info = server_llm_info()
    if info["provider"]:
        return Credentials(info["provider"], _server_key(info["provider"]), "server")
    raise LLMError("no_key")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
_KEY_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{16,}"),
    re.compile(r"sk-ant-[0-9A-Za-z_\-]{6,}"),
    re.compile(r"\bsk-[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)(x-goog-api-key|x-api-key|authorization)\s*[:=]\s*\S+"),
)


def redact(text: Any, creds: Credentials | None = None) -> str:
    """Scrub API keys out of text before it reaches a log line or an exception."""
    out = str(text)
    if creds is not None and creds.api_key:
        out = out.replace(creds.api_key, "[key]")
    for pattern in _KEY_PATTERNS:
        out = pattern.sub("[key]", out)
    return out


_MESSAGES = {
    "no_key": "BioSync needs an AI key to build your schedule. Add a Gemini or Claude API key and try again.",
    "bad_key": "{vendor} didn't accept this API key. Check that you copied the whole key, and that it is {hint}.",
    "region_blocked": "{vendor} says {label} isn't available for this key in your region. Turn on billing for the key's project, or use a Claude key instead.",
    "quota_exhausted": "This {label} key has no quota left for the models BioSync uses. Wait for it to reset, add billing or credit, or try another key.",
    "rate_limited": "{label} is rate-limiting this key right now. {wait}Try again in a moment.",
    "model_unavailable": "None of the {label} models BioSync tries are available to this key right now. Try the other provider, or a key from a different project.",
    "blocked": "The AI declined to answer this time. Try again, or start over and reword your answers.",
    "truncated": "The AI's answer was cut off before your week was finished. Try again, or pick Low detail for a shorter answer.",
    "bad_response": "The AI's answer came back in a shape BioSync couldn't read. Try again.",
    "timeout": "The AI took too long to answer. Check your connection and try again.",
    "network": "BioSync couldn't reach {label}. Check your connection and try again.",
    "provider_error": "{label} had a hiccup on its side. Try again in a minute.",
    "bad_request": "{label} refused this request. Try again, or start over.",
    "generation_failed": "The AI couldn't build your schedule this time. Try again shortly.",
    "internal_error": "Something went wrong on our side while building your schedule. Try again shortly.",
}

# Reasons that make a retry pointless until the student changes something.
_FATAL = {"no_key", "bad_key", "region_blocked", "bad_request", "internal_error"}
# Error priority when several models failed differently: the most actionable wins.
_PRIORITY = ["no_key", "bad_key", "region_blocked", "quota_exhausted", "rate_limited", "model_unavailable",
             "truncated", "blocked", "timeout", "network", "provider_error", "bad_response", "bad_request",
             "internal_error"]


def public_message(reason: str, provider: str | None = None, retry_after: int | None = None) -> str:
    """The student-facing sentence for a failure reason."""
    template = _MESSAGES.get(reason, _MESSAGES["generation_failed"])
    prov = provider if provider in PROVIDERS else GEMINI
    wait = f"Wait about {int(retry_after)} seconds, then " if retry_after else ""
    text = template.format(vendor=PROVIDER_VENDOR[prov], label=PROVIDER_LABEL[prov],
                          hint=KEY_HINT[prov], wait=wait)
    return text.replace("then Try", "then try").replace("  ", " ")


class LLMError(Exception):
    """
    A failed LLM call. `reason` is a stable code (see _MESSAGES). `message` is safe
    to show a student; `detail` is redacted technical text meant for logs only.
    """

    def __init__(self, reason: str, *, provider: str | None = None, model: str | None = None,
                 status: int | None = None, retry_after: int | None = None, detail: str = "",
                 attempts: list[dict] | None = None):
        self.reason = reason
        self.provider = provider
        self.model = model
        self.status = status
        self.retry_after = retry_after
        self.detail = detail
        self.attempts = attempts or []
        super().__init__(f"{reason} [{provider or '-'}/{model or '-'} status={status}] {detail}".strip())

    @property
    def message(self) -> str:
        return public_message(self.reason, self.provider, self.retry_after)

    @property
    def retryable(self) -> bool:
        return self.reason not in _FATAL


def _retry_after_seconds(exc: Exception, text: str) -> int | None:
    """Pull a 'try again in N seconds' hint out of a provider error, if it has one."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        try:
            value = headers.get("retry-after")
            if value:
                return max(1, int(float(value) + 0.999))
        except (TypeError, ValueError):
            pass
    details = getattr(exc, "details", None)
    try:  # google.rpc.RetryInfo: {"@type": ".../google.rpc.RetryInfo", "retryDelay": "13s"}
        for item in (details or {}).get("error", {}).get("details", []):
            if str(item.get("@type", "")).endswith("RetryInfo"):
                return max(1, int(float(str(item["retryDelay"]).rstrip("s")) + 0.999))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    m = re.search(r"retry in ([\d.]+)\s*s", text, re.IGNORECASE)
    if m:
        return max(1, int(float(m.group(1)) + 0.999))
    return None


def classify_exception(provider: str, exc: Exception, *, model: str | None = None,
                       creds: Credentials | None = None) -> LLMError:
    """Turn any exception raised by an SDK call into an LLMError with a stable reason."""
    text = redact(getattr(exc, "message", None) or str(exc), creds)
    low = text.lower()
    status: int | None = None
    reason = "provider_error"

    if provider == GEMINI and isinstance(exc, genai_errors.APIError):
        status = int(exc.code or 0) or None
        api_status = str(exc.status or "").upper()
        if status == 401 or "api key not valid" in low or "api_key_invalid" in low \
                or "api key expired" in low or "reported as leaked" in low:
            reason = "bad_key"
        elif ("location is not supported" in low or "not available in your country" in low
              or ("location" in low and "not supported" in low)):
            reason = "region_blocked"
        elif status == 400 and "thinking" in low:
            reason = "bad_thinking"
        elif status == 400 and any(t in low for t in ("mime", "json mode", "response_schema", "responseschema")):
            reason = "bad_json_mode"
        elif status == 400:
            reason = "bad_request"
        elif status == 403:
            # A key whose project never enabled the Gemini API reads like a bad key to the student.
            key_problem = any(t in low for t in ("api key", "api_key", "has not been used", "is disabled",
                                                 "service_disabled", "suspended"))
            reason = "bad_key" if key_problem else "model_unavailable"
        elif status == 404:
            reason = "model_unavailable"
        elif status == 429:
            daily = any(t in low.replace(" ", "") for t in ("perday", "limit:0", "daily"))
            reason = "quota_exhausted" if daily else "rate_limited"
        elif status in (408, 504) or api_status == "DEADLINE_EXCEEDED":
            reason = "timeout"
    elif provider == CLAUDE and isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code
        if isinstance(exc, anthropic.AuthenticationError):
            reason = "bad_key"
        elif isinstance(exc, anthropic.PermissionDeniedError):
            reason = "model_unavailable" if "model" in low else "bad_key"
        elif isinstance(exc, anthropic.NotFoundError):
            reason = "model_unavailable"
        elif isinstance(exc, anthropic.RateLimitError):
            reason = "rate_limited"
        elif status == 402 or "credit balance" in low or "billing" in low:
            reason = "quota_exhausted"
        elif status == 400 and "model" in low and any(t in low for t in ("not found", "invalid", "does not exist")):
            reason = "model_unavailable"
        elif status in (400, 413, 422):
            reason = "bad_request"
    elif isinstance(exc, anthropic.APITimeoutError) or "timeout" in type(exc).__name__.lower() \
            or isinstance(exc, TimeoutError):
        reason = "timeout"
    elif isinstance(exc, anthropic.APIConnectionError) or isinstance(exc, OSError) \
            or any(t in type(exc).__name__ for t in ("Connect", "Network", "Remote", "Protocol", "SSL")):
        reason = "network"
    elif isinstance(exc, (TypeError, ValueError, AttributeError, KeyError)):
        reason = "internal_error"  # a bug on our side (bad argument, SDK misuse), not the provider's fault

    return LLMError(reason, provider=provider, model=model, status=status,
                    retry_after=_retry_after_seconds(exc, text), detail=f"{type(exc).__name__}: {text[:300]}")


# ---------------------------------------------------------------------------
# One raw provider call
# ---------------------------------------------------------------------------
@dataclass
class _Raw:
    text: str
    finish: str  # "stop" | "length" | "blocked" | "other"
    input_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    finish_raw: str = ""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    finish: str
    input_tokens: int
    output_tokens: int
    thought_tokens: int
    latency_ms: int
    attempts: list[dict] = field(default_factory=list)


def _clean_history(history: list[dict]) -> list[dict]:
    """Drop empty turns (providers reject them), merge same-role neighbours, start on a user turn."""
    cleaned: list[dict] = []
    for turn in history or []:
        role = "assistant" if turn.get("role") in ("assistant", "model") else "user"
        content = str(turn.get("content") or "").strip()[:20000]
        if not content:
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n\n" + content
        else:
            cleaned.append({"role": role, "content": content})
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    return cleaned


def _gemini_contents(history: list[dict]) -> list[dict]:
    """Gemini says "model" where Claude says "assistant"; the system prompt travels separately."""
    return [{"role": "model" if t["role"] == "assistant" else "user", "parts": [{"text": t["content"]}]}
            for t in history]


def _gemini_client(creds: Credentials, timeout_s: float):
    options = genai_types.HttpOptions(
        timeout=int(timeout_s * 1000),
        retry_options=genai_types.HttpRetryOptions(attempts=1),  # retries are ours, so they can switch models
        base_url=os.getenv("GEMINI_BASE_URL") or None,  # test/proxy hook
    )
    return genai.Client(api_key=creds.api_key, http_options=options)


def _claude_client(creds: Credentials, timeout_s: float):
    kwargs: dict[str, Any] = {}
    if os.getenv("ANTHROPIC_BASE_URL"):  # test/proxy hook
        kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
    return anthropic.Anthropic(api_key=creds.api_key, timeout=timeout_s, max_retries=0, **kwargs)


_GEMINI_BLOCKED = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION",
                   "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT"}


def _parse_gemini(resp) -> _Raw:
    usage = getattr(resp, "usage_metadata", None)
    tokens = dict(
        input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        thought_tokens=int(getattr(usage, "thoughts_token_count", 0) or 0),
    )
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        blocked = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        return _Raw("", "blocked" if blocked else "other", finish_raw=str(blocked or "NO_CANDIDATES"), **tokens)
    cand = candidates[0]
    parts = (cand.content.parts if cand.content and cand.content.parts else None) or []
    # Skip "thought" parts: they are the model's scratch work, not the answer.
    text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
    raw_reason = getattr(cand.finish_reason, "name", None) or str(cand.finish_reason or "")
    if raw_reason in ("", "STOP", "FINISH_REASON_UNSPECIFIED"):
        finish = "stop"
    elif raw_reason == "MAX_TOKENS":
        finish = "length"
    elif raw_reason in _GEMINI_BLOCKED:
        finish = "blocked"
    else:
        finish = "other"
    return _Raw(text, finish, finish_raw=raw_reason or "STOP", **tokens)


def _call_gemini(creds, model, system_prompt, history, *, max_tokens, json_mode, thinking, timeout_s) -> _Raw:
    config: dict[str, Any] = {"system_instruction": system_prompt, "max_output_tokens": max_tokens}
    if json_mode:
        config["response_mime_type"] = "application/json"
    thinking_config = _thinking_config(thinking)
    if thinking_config is not None:
        config["thinking_config"] = thinking_config
    # No temperature/top_p/top_k: Google deprecated them for current models in July 2026.
    with _gemini_client(creds, timeout_s) as client:
        resp = client.models.generate_content(
            model=model, contents=_gemini_contents(history),
            config=genai_types.GenerateContentConfig(**config))
    return _parse_gemini(resp)


def _call_claude(creds, model, system_prompt, history, *, max_tokens, timeout_s) -> _Raw:
    # No temperature/top_p/top_k and no assistant prefill: newer Claude models reject
    # sampling overrides, and the SDK raises TypeError for them.
    with _claude_client(creds, timeout_s) as client:
        resp = client.messages.create(model=model, max_tokens=max_tokens, system=system_prompt,
                                      messages=history)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    stop = resp.stop_reason or "end_turn"
    finish = ("stop" if stop in ("end_turn", "stop_sequence")
              else "length" if stop in ("max_tokens", "model_context_window_exceeded")
              else "blocked" if stop == "refusal" else "other")
    usage = resp.usage
    return _Raw(text, finish, int(usage.input_tokens or 0), int(usage.output_tokens or 0), 0, stop)


# ---------------------------------------------------------------------------
# ask(): the fallback loop
# ---------------------------------------------------------------------------
class _Runner:
    """State for one ask(): walks the model chain, switching models/settings as errors dictate."""

    def __init__(self, creds, system_prompt, history, task, effort, max_tokens, json_mode, deadline):
        self.creds = creds
        self.system_prompt = system_prompt
        self.history = history
        self.task = task
        self.effort = effort
        self.base_tokens = max_tokens
        self.json_mode = json_mode
        self.deadline = deadline
        self.errors: list[LLMError] = []
        self.attempts: list[dict] = []
        self.calls = 0
        self.tried: set[str] = set()
        self.tier = tier_for(task, effort)
        self.transient_models = 0   # models that failed with provider_error / timeout / network
        self.budget_hit = False     # the call or time budget ran out (reported only if nothing more useful is known)

    # -- bookkeeping --------------------------------------------------------
    def _key(self, model):
        return (self.creds.provider, self.creds.fingerprint, model)

    def _remaining(self) -> float:
        return self.deadline - time.monotonic()

    def _log(self, model, thinking, started, *, raw: _Raw | None = None, err: LLMError | None = None):
        ms = int((time.monotonic() - started) * 1000)
        label = f"{thinking[0]}={thinking[1]}" if thinking else "none"
        entry = {"model": model, "thinking": label, "ms": ms}
        if raw is not None:
            entry.update(outcome=raw.finish, finish=raw.finish_raw, in_tokens=raw.input_tokens,
                         out_tokens=raw.output_tokens, thought_tokens=raw.thought_tokens)
            logger.info("llm ok provider=%s model=%s task=%s thinking=%s finish=%s in=%d out=%d thoughts=%d ms=%d",
                        self.creds.provider, model, self.task, label, raw.finish_raw,
                        raw.input_tokens, raw.output_tokens, raw.thought_tokens, ms)
        else:
            entry.update(outcome=err.reason, status=err.status)
            logger.warning("llm fail provider=%s model=%s task=%s thinking=%s reason=%s status=%s retry_after=%s detail=%s",
                           self.creds.provider, model, self.task, label, err.reason, err.status,
                           err.retry_after, err.detail[:240])
        self.attempts.append(entry)

    def _chain(self) -> list[str]:
        chain = list(model_chain(self.creds.provider, self.task, self.effort))
        good = _GOOD.get((self.creds.provider, self.creds.fingerprint, self.tier))
        if good in chain:  # keep what worked last time for this key at the front
            chain.remove(good)
            chain.insert(0, good)
        return chain

    # -- the loop -----------------------------------------------------------
    def run(self) -> LLMResult:
        try:
            result = self._run_models(self._chain())
            if result is None and self.errors and all(e.reason == "model_unavailable" for e in self.errors):
                # Every configured model is gone for this key: ask the provider what it does offer.
                extra = [m for m in self._discover() if m not in self.tried]
                if extra:
                    logger.info("llm discovery provider=%s models=%s", self.creds.provider, extra)
                    result = self._run_models(extra)
        except LLMError as fatal:  # bad key, region block, ...: no point trying other models
            fatal.attempts = self.attempts
            raise
        if result is None:
            raise self._best_error()
        return result

    def _run_models(self, models: list[str]) -> LLMResult | None:
        for model in models:
            if self._remaining() <= 0 or self.calls >= _MAX_HTTP_CALLS_PER_ASK:
                self.budget_hit = True
                return None
            self.tried.add(model)
            dead = _DEAD.get(self._key(model))
            if dead is not None:
                self.errors.append(dead)  # learned a moment ago; do not spend a call re-learning it
                continue
            result = self._attempt_model(model)
            if result is not None:
                return result
            if self.transient_models >= _MAX_TRANSIENT_MODELS:
                logger.warning("llm giving up after %d models failed transiently (provider outage?)",
                               self.transient_models)
                return None
        return None

    def _sleep_or_give_up(self, seconds: float) -> bool:
        if self._remaining() <= seconds + 1:
            return False
        _sleep(seconds)
        return True

    def _attempt_model(self, model: str) -> LLMResult | None:
        provider = self.creds.provider
        plan = list(gemini_thinking_plan(model, self.task, self.effort)) if provider == GEMINI else [None]
        tokens, json_mode = self.base_tokens, self.json_mode
        plan_index, transient, grew, json_dropped = 0, 0, False, False
        cap = _TOKEN_GROWTH_CAP[provider]

        while plan_index < len(plan):
            if self._remaining() <= 0 or self.calls >= _MAX_HTTP_CALLS_PER_ASK:
                self.budget_hit = True
                return None
            thinking = plan[plan_index]
            timeout_s = max(5.0, min(_CALL_TIMEOUT[self.task], self._remaining()))
            started = time.monotonic()
            self.calls += 1
            try:
                if provider == GEMINI:
                    raw = _call_gemini(self.creds, model, self.system_prompt, self.history, max_tokens=tokens,
                                       json_mode=json_mode, thinking=thinking, timeout_s=timeout_s)
                else:
                    raw = _call_claude(self.creds, model, self.system_prompt, self.history,
                                       max_tokens=tokens, timeout_s=timeout_s)
            except Exception as exc:  # noqa: BLE001 - every SDK failure is classified below
                err = classify_exception(provider, exc, model=model, creds=self.creds)
                self._log(model, thinking, started, err=err)
                reason = err.reason
                # A 400 may be this model refusing an optional setting. Degrade before giving up:
                # first drop the thinking option, then JSON mode, and only then call it fatal.
                if reason in ("bad_thinking", "bad_request") and thinking is not None and plan_index + 1 < len(plan):
                    plan_index += 1
                    continue
                if reason in ("bad_json_mode", "bad_request") and json_mode and not json_dropped:
                    json_mode, json_dropped = False, True  # fall back to prompt-only JSON
                    continue
                if reason in _FATAL or reason in ("bad_thinking", "bad_json_mode"):
                    raise err if reason in _FATAL else LLMError("bad_request", provider=provider, model=model,
                                                               status=err.status, detail=err.detail)
                if reason == "model_unavailable":
                    _DEAD.set(self._key(model), err, 900)
                    self.errors.append(err)
                    return None
                if reason in ("rate_limited", "quota_exhausted"):
                    wait = err.retry_after
                    if (reason == "rate_limited" and wait is not None and transient < 1
                            and wait <= _MAX_INLINE_WAIT[self.task]
                            and self._sleep_or_give_up(wait + random.uniform(0, 0.5))):
                        transient += 1
                        continue
                    _DEAD.set(self._key(model), err, max(30, wait or 60) if reason == "rate_limited" else 600)
                    self.errors.append(err)
                    return None  # another model has its own quota bucket: move on
                # provider_error / network: one quick retry, but only on the first model that hits it. After
                # that a different model is a better retry than the same struggling one. A timeout is not
                # retried on the same model at all: it would most likely be slow again.
                if (reason != "timeout" and self.transient_models == 0 and transient < 1
                        and self._sleep_or_give_up(1.0 + random.uniform(0, 0.4))):
                    transient += 1
                    continue
                self.errors.append(err)
                self.transient_models += 1
                return None

            self._log(model, thinking, started, raw=raw)
            if raw.finish == "blocked":
                self.errors.append(LLMError("blocked", provider=provider, model=model, detail=raw.finish_raw))
                return None
            empty = not raw.text.strip()
            if raw.finish == "length" or (empty and raw.thought_tokens):
                # Cut off, or thinking ate the whole budget. Retry once with more room and less thinking.
                if not grew and tokens < cap:
                    grew, tokens = True, min(int(tokens * 1.6), cap)
                    if thinking and thinking[0] == "level" and thinking[1] != "low":
                        plan[plan_index] = ("level", "low")
                    elif thinking and thinking[0] == "budget" and thinking[1]:
                        plan[plan_index] = ("budget", 0)
                    continue
                self.errors.append(LLMError("truncated", provider=provider, model=model,
                                            detail=f"finish={raw.finish_raw} out={raw.output_tokens} thoughts={raw.thought_tokens}"))
                return None
            if empty:
                if transient < 1:
                    transient += 1
                    continue
                self.errors.append(LLMError("bad_response", provider=provider, model=model, detail="empty reply"))
                return None

            if provider == GEMINI and thinking is not None:
                _CAPS.set(model, thinking[0], 3600)
            elif provider == GEMINI:
                _CAPS.set(model, "none", 3600)
            _GOOD.set((provider, self.creds.fingerprint, self.tier), model, 1800)
            return LLMResult(raw.text, provider, model, raw.finish, raw.input_tokens, raw.output_tokens,
                             raw.thought_tokens, sum(a["ms"] for a in self.attempts), list(self.attempts))
        self.errors.append(LLMError("provider_error", provider=provider, model=model,
                                    detail="no thinking option was accepted"))
        return None

    # -- discovery ----------------------------------------------------------
    def _discover(self) -> list[str]:
        """Ask the provider which models this key can use; return the best few, best first."""
        try:
            if self.creds.provider == GEMINI:
                with _gemini_client(self.creds, 15) as client:
                    names = [m.name.removeprefix("models/") for m in client.models.list()
                             if "generateContent" in (getattr(m, "supported_actions", None) or [])]
                pattern = re.compile(r"^gemini-(\d+)(?:\.(\d+))?-flash(-lite)?(-preview)?(?:-\d{2}-\d{4})?$")
                scored = []
                for name in names:
                    m = pattern.match(name)
                    if not m:
                        continue
                    lite, preview = bool(m.group(3)), bool(m.group(4))
                    prefer_lite = self.tier == "fast"
                    scored.append((preview, -int(m.group(1)), -int(m.group(2) or 0), lite != prefer_lite, name))
                return [s[-1] for s in sorted(scored)][:3]
            with _claude_client(self.creds, 15) as client:
                ids = [m.id for m in client.models.list(limit=100)]
            wanted = {"fast": "haiku", "balanced": "sonnet", "best": "opus"}[self.tier]
            family_rank = [wanted] + [f for f in ("sonnet", "haiku", "opus") if f != wanted]
            pattern = re.compile(r"^claude-(haiku|sonnet|opus)-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?$")
            scored = []
            for model_id in ids:
                m = pattern.match(model_id)
                if m:  # newest version of the preferred family first
                    scored.append((family_rank.index(m.group(1)), -int(m.group(2)), -int(m.group(3) or 0), model_id))
            return [s[-1] for s in sorted(scored)][:3]
        except Exception as exc:  # noqa: BLE001 - discovery is best-effort
            logger.warning("llm discovery failed: %s", redact(exc, self.creds)[:200])
            return []

    def _best_error(self) -> LLMError:
        if not self.errors:
            reason = "timeout" if self.budget_hit else "generation_failed"
            return LLMError(reason, provider=self.creds.provider, attempts=self.attempts,
                            detail="ask() time or call budget used up" if self.budget_hit else "")
        errors = sorted(self.errors, key=lambda e: _PRIORITY.index(e.reason) if e.reason in _PRIORITY else 99)
        best = errors[0]
        best.attempts = self.attempts
        return best


def ask(system_prompt: str, conversation_history: list[dict], *, creds: Credentials,
        task: str = TASK_SCHEDULE, effort: str = DEFAULT_EFFORT, max_tokens: int | None = None,
        json_mode: bool = False, deadline: float | None = None) -> LLMResult:
    """
    Send a conversation to the provider in `creds` and return the full reply.

    conversation_history: [{"role": "user"|"assistant", "content": "..."}, ...]
    task / effort: choose the model chain, token cap and thinking level (see above).
    json_mode: ask Gemini for application/json output (Claude is told in the prompt).
    deadline: absolute time.monotonic() value; defaults to a per-task budget.

    Raises LLMError (never a raw SDK exception).
    """
    history = _clean_history(conversation_history)
    if not history:
        raise LLMError("bad_request", provider=creds.provider, detail="empty conversation")
    effort = normalize_effort(effort)
    tokens = max_tokens or max_tokens_for(creds.provider, task, effort)
    deadline = deadline or (time.monotonic() + _OVERALL_TIMEOUT[task])
    return _Runner(creds, system_prompt, history, task, effort, tokens, json_mode, deadline).run()


def check_credentials(creds: Credentials) -> LLMResult:
    """A tiny real call proving the key works end to end (fallbacks included)."""
    return ask("You are a connectivity check. Reply with the single word OK.",
               [{"role": "user", "content": "Ping."}], creds=creds, task=TASK_PING)


# ---------------------------------------------------------------------------
# JSON extraction and the retry-on-bad-JSON helper (shared by both LLM steps)
# ---------------------------------------------------------------------------
class JSONExtractError(ValueError):
    """kind: "empty" | "no_json" | "truncated" | "invalid"."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _balanced_slice(text: str, start: int) -> tuple[str, str]:
    """
    From text[start] ('{' or '['), return (slice, state) where state is "closed",
    "unclosed" (the text ended inside the object: a cut-off reply) or "mismatch"
    (a closing bracket that does not fit: malformed). String- and escape-aware.
    """
    pairs = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return text[start:i + 1], "mismatch"
            if not stack:
                return text[start:i + 1], "closed"
    return text[start:], "unclosed"


def _loads(candidate: str) -> Any:
    try:
        return json.loads(candidate, strict=False)  # strict=False tolerates raw newlines inside strings
    except json.JSONDecodeError:
        return json.loads(_TRAILING_COMMA.sub(r"\1", candidate), strict=False)


def extract_json(raw: Any, expect: type = dict) -> Any:
    """
    Pull one JSON object out of an LLM reply. Handles markdown fences, chatter
    before/after the JSON, a BOM, raw newlines inside strings, trailing commas
    and a single object wrapped in a one-element list.
    Raises JSONExtractError with a kind that says what went wrong: "truncated"
    means the object never closed (the reply hit its token limit).
    """
    text = ("" if raw is None else str(raw)).lstrip("\ufeff").strip()
    if not text:
        raise JSONExtractError("empty", "the reply was empty")

    candidates = [text]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    truncated = False
    pos = 0
    for _ in range(5):  # the first few top-level objects, in order
        starts = [i for i in (text.find("{", pos), text.find("[", pos)) if i != -1]
        if not starts:
            break
        start = min(starts)
        chunk, state = _balanced_slice(text, start)
        if state == "closed":
            candidates.append(chunk)
            pos = start + len(chunk)
            continue
        truncated = state == "unclosed"
        break

    for candidate in candidates:
        try:
            value = _loads(candidate)
        except json.JSONDecodeError:
            continue
        if expect is dict and isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            value = value[0]
        if isinstance(value, expect):
            return value
    if "{" not in text and "[" not in text:
        raise JSONExtractError("no_json", "the reply contained no JSON")
    if truncated:
        raise JSONExtractError("truncated", "the JSON object was never closed (reply cut off)")
    raise JSONExtractError("invalid", "the reply contained JSON-like text that would not parse")


@dataclass
class JsonResult:
    data: Any
    llm: LLMResult


def generate_json(system_prompt: str, conversation: list[dict], *, creds: Credentials, task: str,
                  effort: str = DEFAULT_EFFORT, max_tokens: int | None = None,
                  validate: Callable[[Any], Any] | None = None, repair_attempts: int = 1,
                  deadline: float | None = None) -> JsonResult:
    """
    ask() + extract_json() + an optional validator, retrying once with a corrective
    note when the reply is not usable JSON (or fails validation).

    validate(data) returns the cleaned data or raises ValueError("what is wrong").
    Raises LLMError: "truncated" if the reply kept getting cut off, "bad_response"
    for anything else unreadable, or whatever ask() raised.
    """
    deadline = deadline or (time.monotonic() + _OVERALL_TIMEOUT[task])
    history = list(conversation)
    last_problem, last_kind = "", "invalid"
    result: LLMResult | None = None
    for attempt in range(repair_attempts + 1):
        result = ask(system_prompt, history, creds=creds, task=task, effort=effort,
                     max_tokens=max_tokens, json_mode=True, deadline=deadline)
        try:
            data = extract_json(result.text)
            return JsonResult(validate(data) if validate else data, result)
        except JSONExtractError as exc:
            last_problem, last_kind = str(exc), exc.kind
        except ValueError as exc:
            last_problem, last_kind = str(exc), "invalid"
        logger.warning("llm json problem task=%s attempt=%d kind=%s problem=%s",
                       task, attempt + 1, last_kind, last_problem[:200])
        history = list(conversation) + [{
            "role": "user",
            "content": ("Your previous reply could not be used: " + last_problem[:300] +
                        ". Reply again with ONLY the complete JSON object in the exact shape requested "
                        "(no markdown fences, no commentary)."),
        }]
    error = LLMError("truncated" if last_kind == "truncated" else "bad_response",
                     provider=creds.provider, model=result.model if result else None,
                     detail=last_problem[:300], attempts=list(result.attempts) if result else [])
    error.last_llm = result  # lets a caller that can live with a partial answer report which model gave it
    raise error
