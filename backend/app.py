"""
app.py
Flask backend — the glue between the frontend (chat.html / results.html /
script.js, built by teammate) and the 4 pipeline scripts.

Routes match the EXACT contract already implemented in script.js:
  POST /api/chat      -> {"message": "..."}           => {"reply": "...", "done": bool}
  POST /api/upload     -> multipart form, field "file"  => {"status": "received" | "failed", "error"?: "..."}
  GET  /api/schedule   -> (no body)                     => {"status": "pending"|"ready"|"failed", "schedule": [...], ...}
                                                           (full shape documented above api_schedule())
  POST /api/schedule/retry -> (no body)                 => same shape as GET /api/schedule, after re-running
                                                           the pipeline if the last attempt failed

Session handling: single global dict keyed by Flask session cookie id.
No login system in this MVP — one browser session = one in-progress
schedule-building flow. Good enough for a hackathon demo; swapping in
real auth (we HAVE a proven auth pattern, see TEAM_MEMORY) is a
documented next step, not done here to keep the MVP scope tight.

To run:
  1. Put FLASK_SECRET_KEY, and GROQ_API_KEY and/or ANTHROPIC_API_KEY in a .env file
  2. pip install -r requirements.txt
  3. python app.py
  4. Open the frontend's chat.html (served separately, or via /  below) with
     USE_MOCK = false in script.js
"""

import os
import re
import logging
from flask import Flask, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from dotenv import load_dotenv

import chat_handler
import document_parser
import syllabus_structurer
import schedule_generator

load_dotenv()

app = Flask(__name__, static_folder="../frontend", static_url_path="")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-key-change-this")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- In-memory state, keyed by session id (see module docstring) ----
# Each entry: {
#   "chat_state": <chat_handler session dict>,
#   "parsed_docs": [ParseResult.to_dict(), ...],
#   "structured_data": {...} or None,
#   "schedule": [...] or None,
# }
SESSIONS = {}


def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def get_session_data():
    sid = get_session_id()
    if sid not in SESSIONS:
        SESSIONS[sid] = {
            "chat_state": None,
            "parsed_docs": [],
            "structured_data": None,
            "schedule": None,
        }
    return SESSIONS[sid]


# ---------------------------------------------------------------------
# Serve the frontend (optional convenience — the frontend can also just
# be opened as static files / served separately; this just makes `python
# app.py` enough to see the whole thing running from one place).
# ---------------------------------------------------------------------
@app.route("/")
def serve_landing():
    return send_from_directory(app.static_folder, "landing.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


# ---------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------
@app.errorhandler(500)
def handle_internal_error(e):
    # Safety net: anything uncaught inside an /api/* route still comes back as
    # JSON (script.js calls res.json()), never Flask's HTML error page. Flask
    # has already logged the traceback by the time this runs.
    if request.path.startswith("/api/"):
        return jsonify({"error": "Something went wrong on the server. Please try again."}), 500
    return e


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    user_message = str(data.get("message") or "").strip()

    if not user_message:
        return jsonify({"reply": "Could you type something first?", "done": False}), 400

    sdata = get_session_data()

    if sdata["chat_state"] is None:
        # First message of the session — still process it as the answer to
        # question 1, consistent with how the greeting was already shown
        # client-side (script.js shows GREETING locally before any API call).
        sdata["chat_state"] = chat_handler._new_session_state()

    reply, done, updated_state = chat_handler.handle_message(sdata["chat_state"], user_message)
    sdata["chat_state"] = updated_state

    if done:
        # Never let a pipeline crash turn the final chat reply into a 500.
        # The chat itself succeeded; if schedule building failed, that is
        # recorded in sdata["schedule"] and surfaced by /api/schedule.
        _run_pipeline_safely(sdata)

    return jsonify({"reply": reply, "done": done})


def _run_pipeline_safely(sdata):
    """
    Runs _run_post_chat_pipeline() and guarantees it never raises. The LLM
    helpers already return {"success": False, ...} for LLM/JSON failures;
    this catches everything else (bad data shapes, KeyError, etc.) and
    stores it as a failed schedule so /api/schedule can report it.
    """
    try:
        _run_post_chat_pipeline(sdata)
    except Exception:
        logger.exception("Post-chat pipeline crashed unexpectedly")
        sdata["schedule"] = {
            "success": False,
            "error": "Unexpected error while building the schedule.",
            "reason": "internal_error",
        }


def _run_post_chat_pipeline(sdata):
    """
    Once the chat checklist is complete, structure any uploaded documents
    and generate the schedule. Runs synchronously — fine for an MVP demo
    with a handful of small files; a production version would background
    this and let the frontend poll, but that's unnecessary complexity here.
    """
    raw_texts = [doc["text"] for doc in sdata["parsed_docs"] if doc.get("success")]

    structured = syllabus_structurer.structure_documents(raw_texts) if raw_texts else {
        "success": True,
        "data": {"subjects": [], "exams_or_deadlines": [], "parse_confidence": "low"},
        "error": None
    }
    sdata["structured_data"] = structured

    subjects = structured["data"]["subjects"] if structured["success"] else []
    deadlines = structured["data"]["exams_or_deadlines"] if structured["success"] else []
    lifestyle_answers = chat_handler.get_collected_answers(sdata["chat_state"])

    result = schedule_generator.generate_schedule(subjects, deadlines, lifestyle_answers)
    sdata["schedule"] = result


# ---------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"status": "failed", "error": "No file received."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "failed", "error": "Empty filename."}), 400

    original_filename = file.filename
    ext = document_parser.get_extension(original_filename)
    if ext not in document_parser.SUPPORTED_EXTENSIONS:
        return jsonify({"status": "failed", "error": f"Unsupported file type: .{ext}"}), 400

    safe_name = f"{uuid.uuid4().hex}_{secure_filename(original_filename)}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    file.save(save_path)

    result = document_parser.parse_document(save_path, original_filename=original_filename)

    sdata = get_session_data()
    sdata["parsed_docs"].append(result.to_dict())

    # Clean up the saved file now that we've extracted its text — we don't
    # need to keep the original around for this MVP (also keeps the "we
    # don't hold onto more than we need" privacy story honest).
    try:
        os.remove(save_path)
    except OSError:
        pass

    if result.success:
        return jsonify({"status": "received"})
    else:
        return jsonify({"status": "failed", "error": result.error}), 200
        # Note: 200 on purpose — script.js checks the "status" field in the
        # JSON body, not the HTTP status code, for this particular failure case.


# ---------------------------------------------------------------------
# GET /api/schedule
# Response shape (extended for the analytics view on results.html):
#   {
#     "status": "pending" | "ready" | "failed",
#     "schedule": [ {day, focus_summary, blocks:[{time, activity, type,
#                    subject, energy, note}]} ... ],
#     "reasoning_notes": "...",
#     "weekly_focus_hours": <number>,
#     "subject_breakdown": [ {subject, hours_this_week} ... ],
#     "error": null | "<user-facing message>",          (status == "failed")
#     "reason": null | "rate_limited" | "generation_failed" | "internal_error"
#   }
# Kept as one object (not a bare array) so the frontend can read the
# analytics fields alongside the day list without a second request.
#
# "pending" = the chat checklist hasn't finished, nothing has been built yet.
# "failed"  = the pipeline ran and did NOT produce a schedule (e.g. the LLM
#             provider rate-limited us). Always HTTP 200 — the signal is in the
#             body, so a proxy in front of the app can't swallow it the way it
#             could a 429/502/503 error page.
# The raw provider exception text is only logged (schedule_generator.py), never
# sent to the browser: it can contain org IDs and other internals.
# ---------------------------------------------------------------------
SCHEDULE_ERROR_MESSAGES = {
    "rate_limited": "The AI service hit its rate limit — try again shortly.",
    "generation_failed": "The AI service couldn't build your schedule this time — try again shortly.",
    "internal_error": "Something went wrong on our side while building your schedule — try again shortly.",
}
_RATE_LIMIT_PATTERN = re.compile(r"rate[ _-]?limit|too many requests|\b429\b", re.IGNORECASE)


def _classify_schedule_failure(result):
    """
    Map a failed pipeline result to a stable reason code. schedule_generator
    only hands back an error string, so rate limits are recognised from the
    provider's own error text (Groq and Anthropic both surface HTTP 429 /
    "rate_limit" in it).
    """
    reason = result.get("reason")
    if reason in SCHEDULE_ERROR_MESSAGES:
        return reason
    if _RATE_LIMIT_PATTERN.search(str(result.get("error") or "")):
        return "rate_limited"
    return "generation_failed"


def _schedule_payload(sdata):
    payload = {
        "status": "pending",
        "schedule": [],
        "reasoning_notes": "",
        "weekly_focus_hours": None,
        "subject_breakdown": [],
        "error": None,
        "reason": None,
    }
    result = sdata.get("schedule")

    if result is None:
        # Chat flow not finished yet / pipeline hasn't run.
        return payload

    if not result.get("success"):
        reason = _classify_schedule_failure(result)
        payload.update(status="failed", reason=reason, error=SCHEDULE_ERROR_MESSAGES[reason])
        return payload

    payload.update(
        status="ready",
        schedule=result["schedule"],
        reasoning_notes=result.get("reasoning_notes") or "",
        weekly_focus_hours=result.get("weekly_focus_hours"),
        subject_breakdown=result.get("subject_breakdown") or [],
    )
    return payload


@app.route("/api/schedule", methods=["GET"])
def api_schedule():
    return jsonify(_schedule_payload(get_session_data()))


# ---------------------------------------------------------------------
# POST /api/schedule/retry
# After a "failed" schedule the chat UI is already finished (input disabled),
# so without this the only way to retry would be to redo the whole chat.
# Re-runs the pipeline only if the chat is complete and the last attempt did
# not produce a schedule — never regenerates (and replaces) a good one.
# Returns the same shape as GET /api/schedule.
# ---------------------------------------------------------------------
@app.route("/api/schedule/retry", methods=["POST"])
def api_schedule_retry():
    sdata = get_session_data()
    result = sdata.get("schedule")
    last_attempt_failed = result is None or not result.get("success")
    if chat_handler.is_complete(sdata["chat_state"]) and last_attempt_failed:
        _run_pipeline_safely(sdata)
    return jsonify(_schedule_payload(sdata))


if __name__ == "__main__":
    # Render (and most PaaS hosts) inject the port to bind via the PORT env
    # var — fall back to 5000 for local dev. debug=False for any deployed
    # instance: debug mode exposes the interactive Werkzeug debugger/stack
    # traces publicly, which is a real security risk on a public URL.
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "false").strip().lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
