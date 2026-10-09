"""
llm_client.py
Shared dual-provider LLM client — Claude or Groq, chosen per call.

Why this exists as its own file: chat_handler.py, syllabus_structurer.py,
and schedule_generator.py all need to call an LLM. Instead of each script
repeating client setup, error handling, and logging, they all import from
here. One place to fix bugs, one place to swap models.

Provider selection: each script passes its own `provider` argument
("claude" or "groq") when it calls ask(). Today all three callers
(chat_handler, syllabus_structurer, schedule_generator) pass "groq" — Groq's
free tier needs a GROQ_API_KEY. Claude is supported as an alternate provider
(needs ANTHROPIC_API_KEY); flip any single script to it by changing that
script's PROVIDER constant — one line each.

BYO-key path (Task 8): a caller can instead pass `user_api_key` to ask().
When present it overrides the provider arg above entirely — the call always
goes to Claude, using that caller-supplied key via a fresh, one-off client
(see create_client's api_key_override). This never touches ANTHROPIC_API_KEY,
SESSIONS, or any server-side storage; see app.py for where that key is read
from the request and chat_handler.py/schedule_generator.py for how it's
threaded through.
"""

import os
import sys
import logging
import anthropic
import groq
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

DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "groq").strip().lower()


def create_client(provider, api_key_override=None):
    """
    Create and return an API client for 'claude' or 'groq'.

    api_key_override: used only by the BYO-key path (provider="claude") — when
    present, builds the client from this caller-supplied key instead of the
    shared ANTHROPIC_API_KEY env var. Never logged. This function always
    returns a brand-new client (there is no module-level/cached client here),
    so a BYO-key call and the shared-key path never share a client instance.
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

    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'claude' or 'groq'.")


def ask(system_prompt, conversation_history, provider=None, max_tokens=1024, user_api_key=None, model=None):
    """
    Send a conversation to the chosen LLM and get back the full reply (no streaming —
    this is a web backend, not a CLI, so we wait for the full response and return it).

    conversation_history: list of {"role": "user"|"assistant", "content": "..."}
    provider: "claude" or "groq". If None, uses DEFAULT_LLM_PROVIDER from .env (defaults to groq).
              Ignored when user_api_key is given (see below).
    user_api_key: optional caller-supplied Anthropic API key (BYO-key feature, Task 8).
              When present, this call ALWAYS goes to Claude, using a fresh client built
              from this key only (never the shared ANTHROPIC_API_KEY, never cached —
              `client` here is a local variable that goes out of scope when this
              function returns, so nothing about this key outlives this one call). The
              key itself is never written to a log line by this function; callers
              (chat_handler.py, syllabus_structurer.py, schedule_generator.py, app.py)
              must not store it anywhere either.
    model: explicit Claude model ID to use when provider is (or becomes) "claude" —
              e.g. one of CLAUDE_MODEL_BY_EFFORT's values. Falls back to CLAUDE_MODEL.

    Returns: (reply_text, input_tokens, output_tokens)
    Raises: ValueError (bad/missing key), or the underlying API exception on failure —
            callers are expected to catch these and return a friendly error to the user.
    """
    if user_api_key:
        provider = "claude"
        client = create_client("claude", api_key_override=user_api_key)
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
