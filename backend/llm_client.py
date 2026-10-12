"""
llm_client.py — the ONLY module that talks to an LLM provider.

Providers: Gemini (Google AI Studio key) and Claude (Anthropic key). Groq was
removed: its free-tier tokens-per-minute cap truncated schedules mid-JSON.

Which key is used, in order:
  1. a caller-supplied BYO key (per request, never stored or logged) —
     Claude wins if both are given;
  2. LLM_PROVIDER from the environment, if that provider's server key is set;
  3. the first server key that exists (GEMINI_API_KEY, then ANTHROPIC_API_KEY).
If none exists, ask() raises LLMError("not_configured") — it never guesses.

Every failure surfaces as LLMError(reason) with reason in REASONS, so callers
and the API can show an honest, user-safe message instead of a raw SDK error.
"""

import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# stdout, not a file: Render's Logs tab only captures stdout/stderr.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Model IDs live here and nowhere else. Google retires Gemini models on a fixed
# schedule (gemini-2.5-flash is scheduled to shut down 2026-10-16), so the
# Gemini ID is overridable without a code change: set GEMINI_MODEL.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
CLAUDE_MODEL_BY_EFFORT = {
    "low": "claude-haiku-4-5-20251001",
    "medium": "claude-sonnet-5-5",
    "high": "claude-opus-5-5",
}

# Output-token ceiling for the schedule call. A 7-day schedule is ~70-90 blocks
# at ~50 tokens each (~4-5k tokens) and Gemini's thinking tokens count against
# this limit too, so even "low" needs real headroom. You only pay for tokens
# actually generated; the ceiling just stops a good answer being cut off.
MAX_TOKENS_BY_EFFORT = {"low": 6000, "medium": 10000, "high": 16000}
EFFORTS = tuple(MAX_TOKENS_BY_EFFORT)

REQUEST_TIMEOUT_S = 60
RETRY_DELAY_S = 2.0
REASONS = ("rate_limited", "invalid_api_key", "not_configured", "generation_failed")


class LLMError(Exception):
    """reason: one of REASONS. truncated: the reply hit the token ceiling."""

    def __init__(self, reason, message="", truncated=False):
        super().__init__(message or reason)
        self.reason = reason
        self.truncated = truncated


@dataclass
class LLMReply:
    text: str
    in_tokens: int = 0
    out_tokens: int = 0
    truncated: bool = False


def resolve_provider(user_api_key=None, user_gemini_api_key=None):
    """Pick the provider for one call (see module docstring for the order)."""
    if user_api_key:
        return "claude"
    if user_gemini_api_key:
        return "gemini"
    have = {"gemini": bool(os.getenv("GEMINI_API_KEY")), "claude": bool(os.getenv("ANTHROPIC_API_KEY"))}
    preferred = os.getenv("LLM_PROVIDER", "").strip().lower()
    if preferred and not have.get(preferred):
        logger.warning("LLM_PROVIDER=%r has no server key set; falling back to another provider.", preferred)
    for name in ([preferred] if have.get(preferred) else []) + ["gemini", "claude"]:
        if have.get(name):
            return name
    raise LLMError("not_configured", "No GEMINI_API_KEY or ANTHROPIC_API_KEY is set on the server.")


def _classify(exc):
    """Map any SDK exception to (reason, retryable). Never trusts exception text for control flow beyond 'key'."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if status == 429 or "ratelimit" in name or "resource_exhausted" in text:
        return "rate_limited", True
    if status in (401, 403) or "api key not valid" in text or "api_key_invalid" in text:
        return "invalid_api_key", False
    if status in (500, 502, 503, 504, 529):
        return "generation_failed", True
    return "generation_failed", False


def _call_claude(api_key, model, system_prompt, history, max_tokens):
    import anthropic  # lazy: a missing SDK only breaks that provider

    client = anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_S, max_retries=0)
    r = client.messages.create(model=model, max_tokens=max_tokens, system=system_prompt, messages=history)
    text = "".join(b.text for b in r.content if b.type == "text")
    return LLMReply(text, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason == "max_tokens")


def _call_gemini(api_key, model, system_prompt, history, max_tokens, json_mode):
    from google import genai  # lazy, same reason
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000))
    cfg = {"system_instruction": system_prompt, "max_output_tokens": max_tokens}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    contents = [
        {"role": "model" if t["role"] == "assistant" else "user", "parts": [{"text": t["content"]}]}
        for t in history
    ]
    r = client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(**cfg))
    usage = r.usage_metadata
    cand = r.candidates[0] if r.candidates else None
    return LLMReply(
        r.text or "",
        (usage.prompt_token_count or 0) if usage else 0,
        (usage.candidates_token_count or 0) if usage else 0,
        bool(cand and "MAX_TOKENS" in str(cand.finish_reason)),
    )


def ask(system_prompt, conversation_history, *, max_tokens=1024, effort="low", json_mode=False,
        user_api_key=None, user_gemini_api_key=None):
    """
    One LLM call -> LLMReply. Raises LLMError only.

    effort picks the Claude model tier (Gemini has a single model). Callers use
    "low" for chat and extraction (cheap, short tasks) and the student's chosen
    tier only for schedule generation. One automatic retry for 429/5xx.
    BYO keys are used for this call only: nothing here stores or logs them.
    """
    provider = resolve_provider(user_api_key, user_gemini_api_key)
    effort = effort if effort in EFFORTS else "medium"
    for attempt in (1, 2):
        try:
            if provider == "claude":
                key = user_api_key or os.getenv("ANTHROPIC_API_KEY")
                reply = _call_claude(key, CLAUDE_MODEL_BY_EFFORT[effort], system_prompt, conversation_history, max_tokens)
            else:
                key = user_gemini_api_key or os.getenv("GEMINI_API_KEY")
                reply = _call_gemini(key, GEMINI_MODEL, system_prompt, conversation_history, max_tokens, json_mode)
            logger.info("[%s] ok in=%s out=%s truncated=%s", provider, reply.in_tokens, reply.out_tokens, reply.truncated)
            return reply
        except Exception as exc:  # noqa: BLE001 — SDKs raise many types; all are mapped below
            reason, retryable = _classify(exc)
            logger.error("[%s] call failed (%s, attempt %d): %s: %s", provider, reason, attempt,
                         type(exc).__name__, str(exc)[:300])
            if attempt == 1 and retryable:
                time.sleep(RETRY_DELAY_S + random.random())
                continue
            raise LLMError(reason, f"{provider} call failed") from exc


def extract_json(raw_text):
    """Parse the first JSON object in model text, tolerating ``` fences and surrounding prose. Raises ValueError."""
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", (raw_text or "").strip()).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start:end + 1])


def ask_json(system_prompt, conversation_history, **kwargs):
    """ask() + extract_json(). Raises LLMError('generation_failed', truncated=...) if the reply isn't valid JSON."""
    reply = ask(system_prompt, conversation_history, json_mode=True, **kwargs)
    try:
        return extract_json(reply.text)
    except ValueError as exc:
        logger.error("Unparseable JSON (truncated=%s, out=%s): %s | tail: %r",
                     reply.truncated, reply.out_tokens, exc, reply.text[-200:])
        raise LLMError("generation_failed", "unreadable JSON", truncated=reply.truncated) from exc
