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

DEFAULT_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "groq").strip().lower()


def create_client(provider):
    """Create and return an API client for 'claude' or 'groq'."""
    if provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY missing. Add it to your .env file, "
                "or switch this call to provider='groq' if you don't have a Claude key."
            )
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client created")
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


def ask(system_prompt, conversation_history, provider=None, max_tokens=1024):
    """
    Send a conversation to the chosen LLM and get back the full reply (no streaming —
    this is a web backend, not a CLI, so we wait for the full response and return it).

    conversation_history: list of {"role": "user"|"assistant", "content": "..."}
    provider: "claude" or "groq". If None, uses DEFAULT_LLM_PROVIDER from .env (defaults to groq).

    Returns: (reply_text, input_tokens, output_tokens)
    Raises: ValueError (bad/missing key), or the underlying API exception on failure —
            callers are expected to catch these and return a friendly error to the user.
    """
    provider = (provider or DEFAULT_PROVIDER).strip().lower()
    client = create_client(provider)

    if provider == "claude":
        response = client.messages.create(
            model=CLAUDE_MODEL,
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
