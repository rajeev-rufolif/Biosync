"""
app.py
Flask backend — the glue between the frontend (chat.html / results.html /
script.js, built by teammate) and the 4 pipeline scripts.

Routes match the EXACT contract already implemented in script.js:
  POST /api/chat      -> {"message": "..."}           => {"reply": "...", "done": bool}
  POST /api/upload     -> multipart form, field "file"  => {"status": "received" | "failed", "error"?: "..."}
  GET  /api/schedule   -> (no body)                     => [ {day, blocks:[{time, activity}]} ... ]

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
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()

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
        _run_post_chat_pipeline(sdata)

    return jsonify({"reply": reply, "done": done})


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
#     "schedule": [ {day, focus_summary, blocks:[{time, activity, type,
#                    subject, energy, note}]} ... ],
#     "reasoning_notes": "...",
#     "weekly_focus_hours": <number>,
#     "subject_breakdown": [ {subject, hours_this_week} ... ]
#   }
# Kept as one object (not a bare array) so the frontend can read the
# analytics fields alongside the day list without a second request.
# ---------------------------------------------------------------------
@app.route("/api/schedule", methods=["GET"])
def api_schedule():
    sdata = get_session_data()
    result = sdata.get("schedule")

    empty = {
        "schedule": [],
        "reasoning_notes": "",
        "weekly_focus_hours": None,
        "subject_breakdown": []
    }

    if result is None:
        # Chat flow not finished yet / pipeline hasn't run.
        return jsonify(empty), 200

    if not result.get("success"):
        return jsonify(empty), 200

    return jsonify({
        "schedule": result["schedule"],
        "reasoning_notes": result.get("reasoning_notes") or "",
        "weekly_focus_hours": result.get("weekly_focus_hours"),
        "subject_breakdown": result.get("subject_breakdown") or []
    })


if __name__ == "__main__":
    # Render (and most PaaS hosts) inject the port to bind via the PORT env
    # var — fall back to 5000 for local dev. debug=False for any deployed
    # instance: debug mode exposes the interactive Werkzeug debugger/stack
    # traces publicly, which is a real security risk on a public URL.
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "false").strip().lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
