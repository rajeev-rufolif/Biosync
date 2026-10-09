"""
llm_client.py
Shared multi-provider LLM client — Claude, Groq, or Gemini, chosen per call.

Why this exists as its own file: chat_handler.py, syllabus_structurer.py,
and schedule_generator.py all need to call an LLM. Instead of each script
repeating client setup, error handling, and logging, they all import from
here. One place to fix bugs, one place to swap models.

Provider selection: each script passes its own `provider` argument
("claude", "groq", or "gemini") when it calls ask(). Today all three callers
(chat_handler, syllabus_structurer, schedule_generator) pass "groq" — Groq's
free tier needs a GROQ_API_KEY. Claude is supported as an alternate provider
(needs ANTHROPIC_API_KEY); flip any single script to it by changing that
script's PROVIDER constant — one line each.

Gemini (added alongside Claude): Google's Gemini API free tier (needs a
GEMINI_API_KEY from https://aistudio.google.com — NOT the same thing as a
consumer Gemini app subscription, which has no API key at all). Its free-tier
TPM ceiling is far above Groq's, which is the actual fix for Groq cutting the
schedule-generation response off mid-JSON. Same BYO-key pattern as Claude
(see below) via user_gemini_api_key, and the same per-script PROVIDER
one-liner ("gemini") to make it the default for a given step.

BYO-key path (Task 8, extended): a caller can pass `user_api_key` (Claude) or
`user_gemini_api_key` (Gemini) to ask(). Whichever is present overrides the
provider arg above entirely — the call always goes to that provider, using
the caller-supplied key via a fresh, one-off client (see create_client's
api_key_override). If both are somehow present, Claude wins (keeps existing
Task 8 behavior unchanged for anyone already using it). This never touches
ANTHROPIC_API_KEY/GEMINI_API_KEY, SESSIONS, or any server-side storage; see
app.py for where these keys are read from the request and
chat_handler.py/schedule_generator.py for how they're threaded through.
"""

import os
import sys
import logging
import anthropic
import groq
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

# Log to stdout, NOT to a file. This is the first basicConfig() to run in the
# process (app.py imports this module before its own basicConfig call, which
# is then a no-op), so this config applies app-wide. A file inside the
# container is invisible to Render's Logs tab (which only captures
# stdout/stderr) and is wiped on every redeploy.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Model names — change here if you want a different model, nowhere else.
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-2.5-flash"  # free-tier eligible; see GEMINI_MODEL_BY_EFFORT below

# Added for the BYO-Claude-key feature: a user who brings their own Anthropic key
# picks a model tier via the SAME Low/Medium/High effort selector used for the
# shared Groq path (see schedule_generator.py / chat_handler.py), instead of a
# second "pick your model" control. CLAUDE_MODEL (Haiku, above) is reused as-is
# for "low" — it was already the only Claude model in this file. These two are
# new; verified against Anthropic's current model listing before hardcoding.
CLAUDE_MODEL_SONNET = "claude-sonnet-5-5"
CLAUDE_MODEL_OPUS = "claude-opus-5-5"

CLAUDE_MODEL_BY_EFFORT = {
    "low": CLAUDE_MODEL,
    "medium": CLAUDE_MODEL_SONNET,
    "high": CLAUDE_MODEL_OPUS,
}

# Claude's paid API has a far higher per-response ceiling than Groq's free tier
# (which is what Task 7 hit at a self-imposed max_tokens=3000). These numbers are
# deliberately NOT a reuse of the Groq tiers' 3000/5000 — the whole point of the
# BYO-key path is removing that ceiling, so each tier gets real headroom.
CLAUDE_MAX_TOKENS_BY_EFFORT = {"low": 4000, "medium": 8000, "high": 16000}

# Gemini mirrors the Claude BYO-key pattern: same Low/Medium/High effort
# selector, its own model tier and max_tokens ceiling. Flash is used at every
# tier (not Pro) because, as of the free-tier changes through mid-2026, Flash
# is the model Google actually keeps on the free tier with real TPM headroom —
# Pro's free allowance is minimal/trial-only and would defeat the point of
# using this path for free. If a paid Gemini key is ever used, Pro can be
# swapped in per-tier the same way Claude's tiers step up models.
GEMINI_MODEL_FLASH = "gemini-2.5-flash"
GEMINI_MODEL_BY_EFFORT = {
    "low": GEMINI_MODEL_FLASH,
    "medium": GEMINI_MODEL_FLASH,
    "high": GEMINI_MODEL_FLASH,
}
# Gemini's free-tier TPM (~250,000) is far above Groq's (~8,000 on gpt-oss-120b),
# which is the actual bottleneck Groq was hitting — these ceilings are generous
# per-call budgets, not an attempt to approach the TPM limit in one request.
GEMINI_MAX_TOKENS_BY_EFFORT = {"low": 3000, "medium": 6000, "high": 10000}

DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "groq").strip().lower()


def create_client(provider, api_key_override=None):
    """
    Create and return an API client for 'claude', 'groq', or 'gemini'.

    api_key_override: used only by a BYO-key path (provider="claude" or
    provider="gemini") — when present, builds the client from this
    caller-supplied key instead of the shared ANTHROPIC_API_KEY/GEMINI_API_KEY
    env var. Never logged. This function always returns a brand-new client
    (there is no module-level/cached client here), so a BYO-key call and the
    shared-key path never share a client instance.
    """
    if provider == "claude":
        api_key = api_key_override or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY missing. Add it to your .env file, "
                "or switch this call to provider='groq' if you don't have a Claude key."
            )
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client created" + (" (user-supplied key)" if api_key_override else ""))
        return client

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing. Add it to your .env file.")
        client = groq.Groq(api_key=api_key)
        logger.info("Groq client created")
        return client

    elif provider == "gemini":
        api_key = api_key_override or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY missing. Get a free key at https://aistudio.google.com "
                "and add it to your .env file, or switch this call to provider='groq' "
                "if you don't have a Gemini key. Note: a consumer Gemini app subscription "
                "(e.g. via Jio's Google AI Pro bundle) does NOT include an API key — "
                "this must be a separate key from Google AI Studio."
            )
        client = genai.Client(api_key=api_key)
        logger.info("Gemini client created" + (" (user-supplied key)" if api_key_override else ""))
        return client

    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'claude', 'groq', or 'gemini'.")


def _gemini_contents(conversation_history):
    """
    Converts the shared {"role": "user"|"assistant", "content": "..."} history
    shape into Gemini's expected contents list. Gemini uses "model" where
    Claude/Groq (OpenAI-style) use "assistant" — everything else carries over
    as-is, since system_prompt is passed separately via GenerateContentConfig's
    system_instruction, not folded into this list (same split Claude uses).
    """
    role_map = {"user": "user", "assistant": "model"}
    return [
        {"role": role_map.get(turn["role"], turn["role"]), "parts": [{"text": turn["content"]}]}
        for turn in conversation_history
    ]


def ask(system_prompt, conversation_history, provider=None, max_tokens=1024,
        user_api_key=None, user_gemini_api_key=None, model=None):
    """
    Send a conversation to the chosen LLM and get back the full reply (no streaming —
    this is a web backend, not a CLI, so we wait for the full response and return it).

    conversation_history: list of {"role": "user"|"assistant", "content": "..."}
    provider: "claude", "groq", or "gemini". If None, uses DEFAULT_LLM_PROVIDER from
              .env (defaults to groq). Ignored when user_api_key or
              user_gemini_api_key is given (see below).
    user_api_key: optional caller-supplied Anthropic API key (BYO-key feature, Task 8).
              When present, this call ALWAYS goes to Claude, using a fresh client built
              from this key only (never the shared ANTHROPIC_API_KEY, never cached —
              `client` here is a local variable that goes out of scope when this
              function returns, so nothing about this key outlives this one call). The
              key itself is never written to a log line by this function; callers
              must not store it anywhere either. Takes precedence over
              user_gemini_api_key if both are somehow passed, so any existing Task 8
              caller is unaffected by this parameter's addition.
    user_gemini_api_key: optional caller-supplied Gemini API key (BYO-key feature,
              same pattern as user_api_key above, mirrored for Gemini). When present
              (and user_api_key is not), this call ALWAYS goes to Gemini, using a
              fresh client built from this key only. Same never-stored,
              never-logged guarantee as user_api_key.
    model: explicit model ID to use when provider is (or becomes) "claude" or
              "gemini" — e.g. one of CLAUDE_MODEL_BY_EFFORT's or
              GEMINI_MODEL_BY_EFFORT's values. Falls back to CLAUDE_MODEL /
              GEMINI_MODEL respectively.

    Returns: (reply_text, input_tokens, output_tokens)
    Raises: ValueError (bad/missing key), or the underlying API exception on failure —
            callers are expected to catch these and return a friendly error to the user.
    """
    if user_api_key:
        provider = "claude"
        client = create_client("claude", api_key_override=user_api_key)
    elif user_gemini_api_key:
        provider = "gemini"
        client = create_client("gemini", api_key_override=user_gemini_api_key)
    else:
        provider = (provider or DEFAULT_PROVIDER).strip().lower()
        client = create_client(provider)

    if provider == "claude":
        response = client.messages.create(
            model=model or CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=conversation_history
        )
        reply = "".join(
            block.text for block in response.content if block.type == "text"
        )
        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens

    elif provider == "gemini":
        response = client.models.generate_content(
            model=model or GEMINI_MODEL,
            contents=_gemini_contents(conversation_history),
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            )
        )
        # response.text is a convenience property that joins all text parts of
        # the first candidate — same "just give me the string" contract the
        # Claude/Groq branches below return. Falls back to "" (not None) on an
        # empty/blocked candidate so callers' later .strip()/json.loads calls
        # fail predictably (bad JSON -> clean "unreadable response" error)
        # rather than raising a confusing TypeError deeper in the pipeline.
        reply = response.text or ""
        usage = response.usage_metadata
        in_tok = usage.prompt_token_count if usage else 0
        out_tok = usage.candidates_token_count if usage else 0

    else:  # groq
        groq_messages = [{"role": "system", "content": system_prompt}] + conversation_history
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=max_tokens,
            messages=groq_messages
        )
        reply = response.choices[0].message.content
        usage = response.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0

    logger.info(f"[{provider}] reply ok — in:{in_tok} out:{out_tok}")
    return reply, in_tok, out_tok
