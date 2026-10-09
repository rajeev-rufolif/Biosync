"""
schedule_generator.py
Script 4 of 4 in the pipeline — the core value-prop script.

Job: take everything collected so far (structured subjects from
syllabus_structurer.py + lifestyle answers from chat_handler.py) and
generate an actual weekly schedule, reasoned from real chronobiology/
productivity principles rather than a generic template.

Provider: Groq by default (PROVIDER below), same as the other two LLM
steps. This is the one step where output quality is the actual product —
a generic, unreasoned schedule defeats the whole point of the tool — so if
Groq's free-tier quality or rate limits become a problem, this is the step
worth moving to Claude: set PROVIDER = "claude" (needs ANTHROPIC_API_KEY).

Output is forced into the exact JSON shape results.html already expects
(see script.js MOCK_SCHEDULE) — no translation layer needed in app.py.

--- Effort tiers (Task 7) ---
This is the step that was overflowing Groq's free tier: the prompt below
asked for full per-block detail at max_tokens=3000, and Groq's own
openai/gpt-oss-120b model (confirmed max_completion_tokens=65,536 — nowhere
close to the actual problem) cut the response off exactly at that
self-imposed 3000-token ceiling, mid-JSON. Three tiers fix this by varying
BOTH how much detail is requested (less content = fewer tokens needed to
finish) AND the token budget itself:
  - low:    compact — no per-block "note", a terse focus_summary/reasoning.
            max_tokens=2000. This is the tier that MUST reliably finish on
            Groq free tier, so it asks for the least and gets the most
            relative headroom.
  - medium: today's normal level of detail, trimmed slightly (shorter note
            clauses, 2-3 sentence reasoning instead of 2-4).
  - high:   exactly the original, full-detail prompt — unchanged, still the
            one most likely to run long. Opt-in once low/medium exist.
Medium and High share max_tokens=5000 (raised from the old flat 3000, the
number requested) — real headroom per Task 7(b); they're told apart by
prompt detail rather than token budget, since High's own prompt asking for
more content is what makes it more likely to still run long even at 5000.

--- BYO Claude key (Task 8) ---
If a caller passes user_api_key, the SAME low/medium/high effort value is
reused to pick a Claude model tier instead of a Groq max_tokens tier (see
llm_client.CLAUDE_MODEL_BY_EFFORT / CLAUDE_MAX_TOKENS_BY_EFFORT) — one
control, two different meanings depending on which provider is in play.
This module never stores or logs that key; see llm_client.ask().
"""

import json
import logging
from llm_client import ask, CLAUDE_MODEL_BY_EFFORT, CLAUDE_MAX_TOKENS_BY_EFFORT

logger = logging.getLogger(__name__)

PROVIDER = "groq"  # change to "claude" here to use Claude (needs ANTHROPIC_API_KEY)

# Groq-side max_tokens per effort tier (see module docstring for why these numbers).
GROQ_MAX_TOKENS_BY_EFFORT = {"low": 2000, "medium": 5000, "high": 5000}
DEFAULT_EFFORT = "medium"

_INTRO_AND_PRINCIPLES = """You are an expert in student time management, productivity, and \
chronobiology. You build realistic, personalized weekly study schedules — not generic \
advice, but schedules reasoned from the specific student's subjects, routine, and body's \
natural patterns.

Ground every scheduling decision in these principles, applied to THIS student's actual \
answers (not generic filler):
- Circadian rhythm: most people have a natural alertness dip in the early afternoon \
(roughly 1-3pm) and a focus peak either mid-morning or early evening depending on \
chronotype — use what the student told you about when they focus best, don't assume.
- Attention/focus limits: deep-focus work (hard subjects, problem sets) belongs in the \
longest uninterrupted high-energy window the student has. Don't schedule two demanding \
subjects back to back without a break.
- Food and energy: heavy or high-carb meals before study time lead to a post-meal energy \
dip — schedule lighter food before focus blocks, and put demanding work before heavy \
meals, not right after. If the student mentioned caffeine (coffee/tea), caffeine can \
sharpen focus for 1-3 hours after intake — place it ahead of a demanding block, not right \
before sleep.
- Realistic load: respect the number of study hours/day the student said is realistic for \
them. Do not overload the schedule past what they told you they can actually do.
- Exams/deadlines: if any were extracted from their documents, give those subjects extra \
blocks in the days leading up to the date.
"""

_COMMON_RULES = """- Cover all 7 days, Monday through Sunday.
- Every activity description should be specific (name the actual subject), not vague \
("study" is bad, "Deep focus: Data Structures problem set" is good).
- Include meals and breaks, not just study blocks — this is a full day schedule, not just \
a study plan.
- Keep total daily study load realistic to what the student stated.
- Times should be in "H:MM–H:MM" 24-hour-ish readable format matching the example above.
- "energy" should reflect the REAL circadian/food-driven energy level you reasoned about for \
that slot — not a constant value. Vary it across the day (e.g. low right after a heavy meal \
or during the 1-3pm dip, high in the student's stated peak window).
- subject_breakdown should include every subject from SUBJECTS that got at least one study \
block; weekly_focus_hours should equal the sum of subject_breakdown's hours (sanity-check \
your own arithmetic before responding).
"""

# ---- HIGH: the original, full-detail prompt (what's live today). Unchanged. ----
_HIGH_JSON_SHAPE = """Respond ONLY with valid JSON, no preamble, no markdown fences, in exactly this shape:

{
  "schedule": [
    {
      "day": "Monday",
      "focus_summary": "one short sentence on what this day is built around (e.g. the day's hardest block and why it's placed there)",
      "blocks": [
        {
          "time": "7:00–8:00",
          "activity": "short, specific activity description",
          "type": "study" | "break" | "meal" | "sleep" | "class" | "other",
          "subject": "subject name if type is study or class, else empty string",
          "energy": "high" | "medium" | "low",
          "note": "one short clause on WHY this block is placed here, tied to circadian/food/focus reasoning — empty string if a block is self-explanatory (e.g. plain meals/sleep)"
        }
      ]
    }
  ],
  "reasoning_notes": "2-4 sentences explaining the key scheduling decisions you made and why, referencing this student's specific answers — this is shown to the student so they understand and trust the schedule",
  "weekly_focus_hours": <number, total hours of type="study" blocks across the whole week, computed by you from the schedule you just built>,
  "subject_breakdown": [
    { "subject": "subject name", "hours_this_week": <number, sum of study block durations for this subject> }
  ]
}

Rules:
"""
HIGH_SYSTEM_PROMPT = _INTRO_AND_PRINCIPLES + "\n" + _HIGH_JSON_SHAPE + _COMMON_RULES

# ---- MEDIUM: today's normal detail, trimmed — shorter notes, shorter reasoning. ----
_MEDIUM_JSON_SHAPE = """Respond ONLY with valid JSON, no preamble, no markdown fences, in exactly this shape:

{
  "schedule": [
    {
      "day": "Monday",
      "focus_summary": "one short sentence on what this day is built around",
      "blocks": [
        {
          "time": "7:00–8:00",
          "activity": "short, specific activity description",
          "type": "study" | "break" | "meal" | "sleep" | "class" | "other",
          "subject": "subject name if type is study or class, else empty string",
          "energy": "high" | "medium" | "low",
          "note": "a FEW WORDS on why this block is placed here (not a full sentence) — empty string if self-explanatory"
        }
      ]
    }
  ],
  "reasoning_notes": "2-3 sentences explaining the key scheduling decisions you made and why, referencing this student's specific answers",
  "weekly_focus_hours": <number, total hours of type="study" blocks across the whole week, computed by you from the schedule you just built>,
  "subject_breakdown": [
    { "subject": "subject name", "hours_this_week": <number, sum of study block durations for this subject> }
  ]
}

Keep every string field as short as you can while staying accurate and specific — this \
response needs to fit comfortably in a moderate token budget.

Rules:
"""
MEDIUM_SYSTEM_PROMPT = _INTRO_AND_PRINCIPLES + "\n" + _MEDIUM_JSON_SHAPE + _COMMON_RULES

# ---- LOW: compact — no per-block "note" at all, terse everything else. ----
# This is the tier that MUST reliably finish on Groq free tier, so it asks for
# meaningfully less content rather than just hoping a smaller max_tokens is enough.
_LOW_JSON_SHAPE = """Respond ONLY with valid JSON, no preamble, no markdown fences, in exactly this shape:

{
  "schedule": [
    {
      "day": "Monday",
      "focus_summary": "",
      "blocks": [
        {
          "time": "7:00–8:00",
          "activity": "short, specific activity description",
          "type": "study" | "break" | "meal" | "sleep" | "class" | "other",
          "subject": "subject name if type is study or class, else empty string",
          "energy": "high" | "medium" | "low",
          "note": ""
        }
      ]
    }
  ],
  "reasoning_notes": "1-2 sentences, max 30 words total, explaining the single biggest scheduling decision you made",
  "weekly_focus_hours": <number, total hours of type="study" blocks across the whole week, computed by you from the schedule you just built>,
  "subject_breakdown": [
    { "subject": "subject name", "hours_this_week": <number, sum of study block durations for this subject> }
  ]
}

This is the COMPACT tier: always return "" for every "focus_summary" and "note" field \
exactly as shown above — do not fill them in, even briefly. Keep "activity" short (a few \
words). The per-block "energy" field still needs to vary realistically (see the energy \
rule below) — that reasoning should drive your placement of blocks, just don't write it \
out in "note".

Rules:
"""
LOW_SYSTEM_PROMPT = _INTRO_AND_PRINCIPLES + "\n" + _LOW_JSON_SHAPE + _COMMON_RULES

SYSTEM_PROMPT_BY_EFFORT = {
    "low": LOW_SYSTEM_PROMPT,
    "medium": MEDIUM_SYSTEM_PROMPT,
    "high": HIGH_SYSTEM_PROMPT,
}

# Kept for anything importing the old name directly — same text as before (= "high").
SYSTEM_PROMPT = HIGH_SYSTEM_PROMPT


def _extract_json(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def _build_user_prompt(subjects, exams_or_deadlines, lifestyle_answers):
    """Assembles everything we know about the student into one clear prompt."""
    lines = ["Build a weekly schedule for this student.\n"]

    lines.append("SUBJECTS:")
    if subjects:
        for s in subjects:
            hrs = f", ~{s['weekly_hours']}h/week" if s.get("weekly_hours") else ""
            note = f" ({s['notes']})" if s.get("notes") else ""
            lines.append(f"- {s['name']}{hrs}{note}")
    else:
        lines.append("- (none extracted from documents — ask the schedule to be general-purpose)")

    if exams_or_deadlines:
        lines.append("\nUPCOMING EXAMS/DEADLINES:")
        for e in exams_or_deadlines:
            date = e.get("date") or "date unclear"
            lines.append(f"- {e['description']} ({date})")

    lines.append("\nSTUDENT'S ROUTINE AND PREFERENCES:")
    field_labels = {
        "college": "College",
        "country": "Country",
        "course": "Course/Program",
        "sleep_pattern": "Sleep schedule",
        "food_habits": "Food/eating habits",
        "focus_pattern": "Focus pattern & realistic study hours/day",
        "extra_notes": "Additional notes from student",
    }
    for field, label in field_labels.items():
        value = lifestyle_answers.get(field)
        if value:
            lines.append(f"- {label}: {value}")

    return "\n".join(lines)


def generate_schedule(subjects, exams_or_deadlines, lifestyle_answers, effort=None, user_api_key=None):
    """
    subjects: list of dicts from syllabus_structurer.py's "subjects" field (can be empty list)
    exams_or_deadlines: list of dicts from syllabus_structurer.py (can be empty list)
    lifestyle_answers: dict from chat_handler.get_collected_answers()
    effort: "low" | "medium" | "high" (Task 7's effort selector). Anything else
            (None, unrecognised) falls back to "medium" — the previous, single-tier
            behavior, just with the raised max_tokens.
    user_api_key: optional caller-supplied Anthropic key (Task 8, BYO Claude key).
            When present, this call goes to Claude instead of Groq, using the model
            tier CLAUDE_MODEL_BY_EFFORT[effort] maps to. Never stored here — passed
            straight through to llm_client.ask() for this one call only.

    Returns:
      {
        "success": bool,
        "schedule": [ {day, focus_summary, blocks: [{time, activity, type, subject,
                       energy, note}]} ...]   (if success — matches results.html's
                                                expected shape, now with per-block
                                                type/energy/note metadata for the
                                                analytics view)
        "reasoning_notes": str  (if success)
        "weekly_focus_hours": number  (if success)
        "subject_breakdown": [ {subject, hours_this_week} ... ]  (if success)
        "error": str  (if not success)
      }
    """
    effort = effort if effort in SYSTEM_PROMPT_BY_EFFORT else DEFAULT_EFFORT
    system_prompt = SYSTEM_PROMPT_BY_EFFORT[effort]

    if user_api_key:
        model = CLAUDE_MODEL_BY_EFFORT[effort]
        max_tokens = CLAUDE_MAX_TOKENS_BY_EFFORT[effort]
    else:
        model = None  # unused for Groq
        max_tokens = GROQ_MAX_TOKENS_BY_EFFORT[effort]

    user_prompt = _build_user_prompt(subjects, exams_or_deadlines, lifestyle_answers)

    conversation = [{"role": "user", "content": user_prompt}]

    try:
        reply, _, out_tok = ask(
            system_prompt=system_prompt,
            conversation_history=conversation,
            provider=PROVIDER,
            max_tokens=max_tokens,
            user_api_key=user_api_key,
            model=model
        )
    except Exception as e:
        logger.error(f"Schedule generation LLM call failed: {e}")
        return {
            "success": False,
            "error": f"Couldn't generate your schedule right now: {str(e)}",
            "schedule": None,
            "reasoning_notes": None,
            "weekly_focus_hours": None,
            "subject_breakdown": None
        }

    try:
        cleaned = _extract_json(reply)
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Task 7(c): tell a truncated response (hit max_tokens mid-JSON, the Task 7
        # root cause) apart from a genuinely malformed one in the LOG ONLY — the
        # user-facing error/UI below is unchanged from Task 5/6, on purpose.
        if out_tok is not None and out_tok >= max_tokens - 5:
            logger.error(
                f"Schedule generator JSON truncated at token limit "
                f"(effort={effort}, out_tok={out_tok}, max_tokens={max_tokens}): {e}\n"
                f"Raw reply (last 300 chars): {reply[-300:]}"
            )
        else:
            logger.error(f"Schedule generator returned malformed JSON: {e}\nRaw reply: {reply[:500]}")
        return {
            "success": False,
            "error": "Got an unreadable response while building your schedule. Try again.",
            "schedule": None,
            "reasoning_notes": None,
            "weekly_focus_hours": None,
            "subject_breakdown": None
        }

    schedule = parsed.get("schedule", [])
    reasoning = parsed.get("reasoning_notes", "")
    weekly_focus_hours = parsed.get("weekly_focus_hours")
    subject_breakdown = parsed.get("subject_breakdown", [])

    if not schedule:
        return {
            "success": False,
            "error": "The schedule came back empty. Try again.",
            "schedule": None,
            "reasoning_notes": None,
            "weekly_focus_hours": None,
            "subject_breakdown": None
        }

    logger.info(f"Generated schedule covering {len(schedule)} days (effort={effort})")
    return {
        "success": True,
        "schedule": schedule,
        "reasoning_notes": reasoning,
        "weekly_focus_hours": weekly_focus_hours,
        "subject_breakdown": subject_breakdown,
        "error": None
    }
