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
"""

import json
import logging
from llm_client import ask

logger = logging.getLogger(__name__)

PROVIDER = "groq"  # change to "claude" here to use Claude (needs ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are an expert in student time management, productivity, and \
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

Respond ONLY with valid JSON, no preamble, no markdown fences, in exactly this shape:

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
- Cover all 7 days, Monday through Sunday.
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


def generate_schedule(subjects, exams_or_deadlines, lifestyle_answers):
    """
    subjects: list of dicts from syllabus_structurer.py's "subjects" field (can be empty list)
    exams_or_deadlines: list of dicts from syllabus_structurer.py (can be empty list)
    lifestyle_answers: dict from chat_handler.get_collected_answers()

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
    user_prompt = _build_user_prompt(subjects, exams_or_deadlines, lifestyle_answers)

    conversation = [{"role": "user", "content": user_prompt}]

    try:
        reply, _, _ = ask(
            system_prompt=SYSTEM_PROMPT,
            conversation_history=conversation,
            provider=PROVIDER,
            max_tokens=3000
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
        logger.error(f"Schedule generator returned invalid JSON: {e}\nRaw reply: {reply[:500]}")
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

    logger.info(f"Generated schedule covering {len(schedule)} days")
    return {
        "success": True,
        "schedule": schedule,
        "reasoning_notes": reasoning,
        "weekly_focus_hours": weekly_focus_hours,
        "subject_breakdown": subject_breakdown,
        "error": None
    }
