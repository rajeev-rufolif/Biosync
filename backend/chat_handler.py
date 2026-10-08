"""
chat_handler.py
Script 1 of 4 in the pipeline.

Job: talk to the student, work through a FIXED checklist of questions,
and signal when enough information has been collected. This is deliberately
a fixed checklist (not "let the AI decide when it's done") — predictable
flow, easier to demo reliably, and the user can always still type freely;
we just always know which question we're on.

Provider: Groq by default (fast, free, this runs on every single message
so cost/speed matters more than peak reasoning quality here). Can be
switched to Claude via provider="claude" in ask_chat(), or by setting
DEFAULT_LLM_PROVIDER=claude in .env.

This module holds per-session chat state in memory (a dict). No database —
if the Flask process restarts, in-progress conversations are lost. That's
an accepted MVP tradeoff, noted in the README.
"""

from llm_client import ask

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

PROVIDER = "groq"  # change to "claude" here to use Claude for chat instead

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


def handle_message(session_state, user_message):
    """
    Process one incoming user message for this session.

    session_state: dict from _new_session_state() (or an existing one for this session)
    user_message: the text the student just sent

    Returns: (reply_text, done_bool, updated_session_state)
    """
    if session_state is None:
        session_state = _new_session_state()

    step = session_state["step"]
    session_state["history"].append({"role": "user", "content": user_message})

    # If we've already finished the checklist, just acknowledge — app.py
    # should stop routing here and move to document parsing / schedule gen.
    if step >= len(CHECKLIST):
        session_state["done"] = True
        return (
            "Thanks, that's everything I need. Your schedule is ready.",
            True,
            session_state
        )

    current = CHECKLIST[step]

    # Record the raw answer to the field we were just asking about.
    # (Simple approach for MVP: whatever the student just said IS the answer
    # to the current question. A stretch improvement would be a second AI
    # call to validate/extract the answer cleanly, but that's unnecessary
    # complexity for a first working version.)
    session_state["answers"][current["field"]] = user_message

    # Move to the next step
    next_step = step + 1
    session_state["step"] = next_step

    if next_step >= len(CHECKLIST):
        # Just answered the last question — wrap up, no new question to ask.
        session_state["done"] = True
        reply = (
            "Thanks, that's everything I need. I've got a full picture of your "
            "courses and routine — your schedule is ready."
        )
        session_state["history"].append({"role": "assistant", "content": reply})
        return reply, True, session_state

    next_question = CHECKLIST[next_step]["question"]
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(current_question=next_question)

    try:
        reply, _, _ = ask(
            system_prompt=system_prompt,
            conversation_history=session_state["history"],
            provider=PROVIDER,
            max_tokens=200
        )
    except Exception:
        # Fallback: if the LLM call fails for any reason (bad key, rate limit,
        # network), still move the conversation forward with the raw question
        # rather than leaving the student stuck.
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
