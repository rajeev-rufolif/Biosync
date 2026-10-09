"""
syllabus_structurer.py
Script 3 of 4 in the pipeline.

Job: take the messy raw text that document_parser.py extracted (which could
be from a PDF, Word doc, Excel sheet, or OCR'd image — so formatting is
inconsistent) and turn it into clean, structured subject data: subject
names, rough weekly hours/credit load if mentioned, and exam/deadline
dates if present.

This IS an AI step (unlike document_parser.py) — the task is "make sense
of messy unstructured text", which plain code can't reliably do across
such varied input formats. Groq by default: this is an extraction task,
not deep reasoning, so the cheaper/faster model is the right fit — the
quality bar that actually matters is in schedule_generator.py.

Output is forced into JSON so schedule_generator.py can consume it
directly without more parsing.
"""

import json
import logging
from llm_client import ask, CLAUDE_MODEL, GEMINI_MODEL_FLASH

logger = logging.getLogger(__name__)

PROVIDER = "groq"  # change to "claude" here if Groq's structuring quality isn't good enough

SYSTEM_PROMPT = """You extract structured data from messy, unstructured text pulled from a \
student's uploaded documents (syllabus, timetable, course list — the text may be poorly \
formatted because it came from a PDF, Word doc, Excel sheet, or OCR scan).

Respond ONLY with valid JSON. No preamble, no explanation, no markdown code fences — \
just the raw JSON object, in this exact shape:

{
  "subjects": [
    {
      "name": "subject name as found in the text",
      "weekly_hours": <number, or null if not mentioned>,
      "notes": "anything relevant: credit count, instructor, difficulty hints — or empty string"
    }
  ],
  "exams_or_deadlines": [
    {
      "description": "what it is",
      "date": "date as found in the text, or null if unclear"
    }
  ],
  "parse_confidence": "high" | "medium" | "low"
}

Rules:
- Only include subjects/dates that are actually present in the text. Do not invent courses.
- If the text is too garbled to extract anything reliable, return empty arrays and set \
parse_confidence to "low" — do not guess or fabricate data to fill the response.
- weekly_hours should be null unless a number is actually stated or clearly computable \
from the text (e.g. "3 lectures/week, 1hr each" = 3).
"""


def _extract_json(raw_text):
    """
    LLMs sometimes wrap JSON in markdown fences or add stray text even when
    told not to. This strips common wrapping before parsing, rather than
    failing outright on a minor formatting slip.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def structure_documents(raw_texts, user_api_key=None, user_gemini_api_key=None):
    """
    raw_texts: list of strings — the .text from one or more ParseResult objects
               (document_parser.py), already filtered to successful parses only.
    user_api_key: optional caller-supplied Anthropic key (Task 8, BYO Claude key).
               Threaded through so a user's whole post-chat pipeline — structuring
               AND schedule generation, both of which run inside the same
               _run_post_chat_pipeline() call in app.py — uses their own key
               consistently, not just the final schedule-building step. Always on
               CLAUDE_MODEL (Haiku) here regardless of the Low/Medium/High effort
               selector: per this module's own docstring, structuring is an
               extraction task, not deep reasoning, so there's no reason to spend
               Sonnet/Opus tokens on it even when the user has opted into their own
               key and selected a higher effort tier for schedule generation.
    user_gemini_api_key: optional caller-supplied Gemini key (Task 8b, BYO Gemini
               key), same threading rationale as user_api_key above. Always on
               GEMINI_MODEL_FLASH here for the same reason Claude is pinned to
               Haiku — extraction, not deep reasoning, so no need for a bigger
               model even at a higher effort tier. If both keys are somehow set,
               Claude wins (see llm_client.ask()).

    Returns a dict:
      {
        "success": bool,
        "data": { subjects, exams_or_deadlines, parse_confidence }  (if success)
        "error": str  (if not success)
      }
    """
    if not raw_texts:
        return {
            "success": False,
            "error": "No document text provided to structure.",
            "data": None
        }

    combined_text = "\n\n---\n\n".join(raw_texts)

    # Guard against sending something absurdly long to the LLM — truncate
    # with a note rather than silently failing or blowing past context limits.
    MAX_CHARS = 15000
    if len(combined_text) > MAX_CHARS:
        combined_text = combined_text[:MAX_CHARS] + "\n\n[...text truncated...]"

    conversation = [
        {"role": "user", "content": f"Extract structured data from this text:\n\n{combined_text}"}
    ]

    try:
        model = CLAUDE_MODEL if user_api_key else (GEMINI_MODEL_FLASH if user_gemini_api_key else None)
        reply, _, _ = ask(
            system_prompt=SYSTEM_PROMPT,
            conversation_history=conversation,
            provider=PROVIDER,
            max_tokens=2000,
            user_api_key=user_api_key,
            user_gemini_api_key=user_gemini_api_key,
            model=model
        )
    except Exception as e:
        logger.error(f"Syllabus structuring LLM call failed: {e}")
        return {
            "success": False,
            "error": f"Couldn't process the documents right now: {str(e)}",
            "data": None
        }

    try:
        cleaned = _extract_json(reply)
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Syllabus structurer returned invalid JSON: {e}\nRaw reply: {reply[:500]}")
        return {
            "success": False,
            "error": "Got an unreadable response while structuring your documents. Try again.",
            "data": None
        }

    # Basic shape validation so schedule_generator.py never receives garbage
    parsed.setdefault("subjects", [])
    parsed.setdefault("exams_or_deadlines", [])
    parsed.setdefault("parse_confidence", "low")

    logger.info(
        f"Structured {len(parsed['subjects'])} subjects, "
        f"{len(parsed['exams_or_deadlines'])} deadlines, "
        f"confidence={parsed['parse_confidence']}"
    )

    return {"success": True, "data": parsed, "error": None}
