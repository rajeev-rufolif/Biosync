"""
chat_handler.py — stage 1 of the pipeline (LLM).

Walks the student through a FIXED checklist, one question at a time. The LLM
only phrases the next question warmly; the order, the stored answers and the
"done" signal are plain code, so the flow is predictable. Chat always uses the
cheapest model tier (it is a one-sentence reply), regardless of schedule detail.

State is a plain dict held by app.py's session store (in memory; lost on
restart by design — see SETUP.md).
"""

import logging

from llm_client import LLMError, ask

logger = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 1500  # also enforced server-side in app.py; bounds prompt size and quota burn

# ---- The fixed checklist ----
# Each step has: the field name we're collecting, and the exact question
# text the AI is instructed to ask. The AI is told to ask ONE question at
# a time, in this order, and move to the next step once it has a usable
# answer for the current one.
CHECKLIST = [
    {"field": "college", "question": "What college or university do you attend?"},
    {"field": "country", "question": "Which country is that in?"},
    {"field": "course", "question": "What course or program are you studying?"},
    {"field": "sleep_pattern", "question": "What's your usual sleep schedule? (e.g. when you sleep and wake up)"},
    {"field": "food_habits", "question": "Tell me a bit about your eating habits — meal times, and anything like your coffee/tea intake."},
    {"field": "focus_pattern", "question": "When do you focus best — morning, afternoon, or late at night? And roughly how many hours a day can you realistically study?"},
    {"field": "extra_notes", "question": "Anything else you think I should know while building your schedule? (Deadlines, exams, commitments, anything at all — or just say 'no' to skip.)"},
]

SYSTEM_PROMPT_TEMPLATE = """You are BioSync, a friendly AI assistant that helps students build a \
personalized weekly time-management schedule. You are warm, concise, and conversational — \
not robotic, not overly formal.

You are currently working through a fixed checklist of information you need from the \
student. Right now, you need to ask them this exact question, in your own natural words \
(you may rephrase it conversationally, but do not skip or change its meaning):

"{current_question}"

Rules:
- Ask only ONE thing — the question above. Do not ask multiple questions at once.
- If the student's last message already answers this question, acknowledge it briefly \
and naturally (don't repeat their answer back at length), then you're done for this turn.
- If the student's message is unrelated or confused, gently steer back to the question.
- Keep replies short — 1 to 3 sentences. This is a chat interface, not an essay.
- Never ask about information not in your current question. Stay on this one topic.
- Do not mention "checklist", "step", or "field" — the student should experience this as \
a normal conversation, not a form.
"""


def _new_session_state():
    return {
        "step": 0,  # index into CHECKLIST — which question we're currently asking
        "answers": {},  # field -> student's answer text
        "history": [],  # full chat transcript, list of {"role", "content"}
        "uploaded_docs_text": [],  # filled in by document_parser.py via app.py
        "done": False,
    }


def get_greeting():
    """The very first message shown when a chat session starts."""
    return (
        "Hi! I'm BioSync. I'll ask a few quick questions so I can build a schedule that "
        "actually fits your life — let's start with the basics. "
        f"{CHECKLIST[0]['question']}"
    )


def handle_message(session_state, user_message, user_api_key=None, user_gemini_api_key=None):
    """
    Process one user message. Returns (reply_text, done, session_state).

    A reply shorter than 2 characters is not accepted as an answer: the same
    question is asked again and the step does not advance. If the LLM call
    fails, the plain checklist question is used so the student is never stuck.
    BYO keys are used for this call only and never stored.
    """
    if session_state is None:
        session_state = _new_session_state()

    step = session_state["step"]
    if step >= len(CHECKLIST):
        session_state["done"] = True
        return "Thanks, that's everything I need. Your schedule is ready.", True, session_state

    text = user_message.strip()
    current = CHECKLIST[step]
    if len(text) < 2:
        return f"Could you say a little more? {current['question']}", False, session_state

    session_state["history"].append({"role": "user", "content": text})
    session_state["answers"][current["field"]] = text
    session_state["step"] = step + 1

    if session_state["step"] >= len(CHECKLIST):
        session_state["done"] = True
        reply = ("Thanks, that's everything I need. I've got a full picture of your "
                 "courses and routine, so your schedule is ready.")
        session_state["history"].append({"role": "assistant", "content": reply})
        return reply, True, session_state

    next_question = CHECKLIST[session_state["step"]]["question"]
    try:
        reply = ask(
            SYSTEM_PROMPT_TEMPLATE.format(current_question=next_question),
            session_state["history"], max_tokens=200, effort="low",
            user_api_key=user_api_key, user_gemini_api_key=user_gemini_api_key,
        ).text.strip() or next_question
    except LLMError as exc:
        logger.warning("Chat LLM unavailable (%s); using the plain checklist question.", exc.reason)
        reply = next_question

    session_state["history"].append({"role": "assistant", "content": reply})
    return reply, False, session_state


def is_complete(session_state):
    return session_state is not None and session_state.get("done", False)


def get_collected_answers(session_state):
    """Used by schedule_generator.py to pull the lifestyle answers it needs."""
    if session_state is None:
        return {}
    return session_state.get("answers", {})
