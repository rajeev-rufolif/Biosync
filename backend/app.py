"""
app.py — Flask glue between the frontend and the four pipeline stages.

API contract (unchanged for the frontend; every field below is additive-only):
  POST /api/chat           {"message"}            -> {"reply", "done"}
  POST /api/upload         multipart field "file" -> {"status": "received"|"failed", "error"?}
  POST /api/effort         {"level"}              -> {"status": "ok", "level"}
  GET  /api/schedule       -> {"status": "pending"|"ready"|"failed", "schedule", "reasoning_notes",
                               "weekly_focus_hours", "subject_breakdown", "error", "reason", "warnings"}
  POST /api/schedule/retry -> same shape; re-runs only if the last attempt failed
  GET  /healthz            -> {"status": "ok"}
`reason` (when failed): rate_limited | invalid_api_key | not_configured |
generation_failed | internal_error. A failed schedule is still HTTP 200.
Other 4xx/5xx are JSON {"error": "..."}; 429 carries Retry-After.

BYO keys: optional X-Anthropic-Api-Key / X-Gemini-Api-Key headers are read per
request, format-checked, passed down as plain arguments and never stored or
logged. A malformed key is rejected (400) rather than silently ignored, so a
student's typo can never quietly fall back to the shared server key.

State: sessions live in memory (see SETUP.md for why that is the right call
today). Run ONE gunicorn worker with threads; each session has its own lock so
double-clicks cannot advance the checklist twice or run the pipeline twice.
"""

import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, session
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

load_dotenv()

import chat_handler  # noqa: E402  (after load_dotenv so env is populated)
import document_parser  # noqa: E402
import schedule_generator  # noqa: E402
import syllabus_structurer  # noqa: E402

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MAX_UPLOAD_MB = 20  # keep in sync with MAX_MB in frontend/script.js
MAX_DOCS_PER_SESSION = 8
SESSION_TTL_S = 6 * 3600
MAX_SESSIONS = 2000

APP_ENV = os.getenv("APP_ENV", "production").strip().lower()
IS_PROD = APP_ENV == "production"


def resolve_secret(app_env, key):
    """Production must have a real secret; dev gets a random per-process one (never a guessable constant)."""
    key = (key or "").strip()
    if app_env == "production":
        if len(key) < 16 or key.startswith("change-this"):
            raise RuntimeError(
                "FLASK_SECRET_KEY is missing or a placeholder (APP_ENV=production). Generate one with: "
                'python -c "import secrets; print(secrets.token_hex(32))"'
            )
        return key
    return key or secrets.token_hex(32)


app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "..", "frontend"), static_url_path="")
app.secret_key = resolve_secret(APP_ENV, os.getenv("FLASK_SECRET_KEY"))
app.config.update(
    MAX_CONTENT_LENGTH=(MAX_UPLOAD_MB + 1) * 1024 * 1024,  # hard cap on any request body, enforced before parsing
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # Explicit opt-in: a Secure cookie is never sent over plain http, which would break local Docker runs.
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "0") == "1",
)
_hops = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))
if _hops:  # only trust X-Forwarded-* when a known proxy (e.g. Render) sits in front
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=_hops, x_proto=_hops)

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------------- sessions
class SessionStore:
    """In-memory sessions with a TTL and a hard cap, so memory cannot grow without bound."""

    def __init__(self, ttl, cap):
        self._d, self._lock, self._ttl, self._cap, self._pruned = {}, threading.Lock(), ttl, cap, time.time()

    @staticmethod
    def _new():
        return {"chat_state": None, "parsed_docs": [], "structured_data": None, "structured_n": -1,
                "schedule": None, "effort": "medium", "warnings": [], "lock": threading.RLock(), "seen": time.time()}

    def get(self, sid, create=True):
        now = time.time()
        with self._lock:
            if now - self._pruned > 60:
                self._d = {k: v for k, v in self._d.items() if now - v["seen"] < self._ttl}
                self._pruned = now
            s = self._d.get(sid)
            if s is None and create:
                if len(self._d) >= self._cap:  # evict the least recently used
                    self._d.pop(min(self._d, key=lambda k: self._d[k]["seen"]))
                s = self._d[sid] = self._new()
            if s:
                s["seen"] = now
            return s


store = SessionStore(SESSION_TTL_S, MAX_SESSIONS)


def _sdata(create=True):
    if "session_id" not in session:
        if not create:
            return None
        session["session_id"] = uuid.uuid4().hex
    return store.get(session["session_id"], create=create)


# ---------------------------------------------------------------- rate limiting
class RateLimiter:
    """Sliding window per (bucket, client IP). Plenty for one instance; swap for Redis if you ever scale out."""

    def __init__(self):
        self.enabled, self._hits, self._lock = True, defaultdict(deque), threading.Lock()

    def retry_after(self, bucket, ip, limit, window):
        if not self.enabled:
            return 0
        now = time.monotonic()
        with self._lock:
            q = self._hits[(bucket, ip)]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return int(window - (now - q[0])) + 1
            q.append(now)
            if len(self._hits) > 5000:
                self._hits = defaultdict(deque, {k: v for k, v in self._hits.items() if v})
        return 0


limiter = RateLimiter()


def rate_limited(bucket, limit, window=60):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            wait = limiter.retry_after(bucket, request.remote_addr or "?", limit, window)
            if wait:
                return jsonify({"error": f"You're going a little fast. Wait {wait} seconds and try again."}), 429, {"Retry-After": str(wait)}
            return fn(*a, **kw)
        return wrapper
    return deco


# ---------------------------------------------------------------- BYO keys
_KEY_RE = re.compile(r"[A-Za-z0-9_\-]{20,300}")


class BadKey(ValueError):
    pass


def _byok():
    """(anthropic_key, gemini_key) from this request's headers. Raises BadKey on a malformed value."""
    keys = []
    for header in ("X-Anthropic-Api-Key", "X-Gemini-Api-Key"):
        value = (request.headers.get(header) or "").strip()
        if value and not _KEY_RE.fullmatch(value):
            raise BadKey()
        keys.append(value or None)
    return keys


# ---------------------------------------------------------------- errors & headers
@app.errorhandler(BadKey)
def bad_key(_):
    return jsonify({"error": "That API key doesn't look valid. Check it, or turn off 'Use my own AI API key'."}), 400


@app.errorhandler(HTTPException)
def http_error(e):
    if not request.path.startswith("/api/"):
        return e
    msg = f"That file is over {MAX_UPLOAD_MB} MB. Upload a smaller file." if e.code == 413 else e.description
    body = {"error": msg}
    if request.path == "/api/upload":
        body["status"] = "failed"
    return jsonify(body), e.code


@app.errorhandler(Exception)
def unhandled(e):
    logger.exception("Unhandled error on %s", request.path)
    if request.path.startswith("/api/"):
        return jsonify({"error": "Something went wrong on the server. Please try again."}), 500
    return "Something went wrong.", 500


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    return resp


# ---------------------------------------------------------------- pages
@app.route("/")
def serve_landing():
    return send_from_directory(app.static_folder, "landing.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


# ---------------------------------------------------------------- pipeline
SCHEDULE_ERROR_MESSAGES = {
    "rate_limited": "The AI service hit its rate limit. Wait a minute, then try again.",
    "invalid_api_key": "The AI provider rejected the API key. Check the key you pasted, or turn off 'Use my own AI API key'.",
    "not_configured": "This server has no AI key configured. Set GEMINI_API_KEY or ANTHROPIC_API_KEY, or paste your own key on the chat page.",
    "generation_failed": "The AI service couldn't build your schedule this time. Try again shortly.",
    "internal_error": "Something went wrong on our side while building your schedule. Try again shortly.",
}


def _run_pipeline(sdata, user_key, gemini_key):
    """Structure uploaded docs (if any), then build the schedule. Never raises; failures are stored as results."""
    try:
        raw = [d["text"] for d in sdata["parsed_docs"] if d.get("success")]
        sdata["warnings"] = []
        subjects, deadlines = [], []
        if raw:
            if sdata["structured_n"] != len(raw):  # reuse a good structuring result across retries
                sdata["structured_data"] = syllabus_structurer.structure_documents(raw, user_key, gemini_key)
                sdata["structured_n"] = len(raw) if sdata["structured_data"]["success"] else -1
            structured = sdata["structured_data"]
            if structured["success"]:
                subjects, deadlines = structured["data"]["subjects"], structured["data"]["exams_or_deadlines"]
            else:
                sdata["warnings"].append("We couldn't read your uploaded files, so this schedule is built from your chat answers only.")
        sdata["schedule"] = schedule_generator.generate_schedule(
            subjects, deadlines, chat_handler.get_collected_answers(sdata["chat_state"]),
            effort=sdata["effort"], user_api_key=user_key, user_gemini_api_key=gemini_key,
        )
    except Exception:  # noqa: BLE001 — anything unexpected becomes a failed schedule, not a 500
        logger.exception("Pipeline crashed")
        sdata["schedule"] = {"success": False, "reason": "internal_error"}


def _schedule_payload(sdata):
    payload = {"status": "pending", "schedule": [], "reasoning_notes": "", "weekly_focus_hours": None,
               "subject_breakdown": [], "error": None, "reason": None, "warnings": []}
    result = sdata.get("schedule") if sdata else None
    if result is None:
        return payload
    if not result.get("success"):
        reason = result.get("reason") if result.get("reason") in SCHEDULE_ERROR_MESSAGES else "generation_failed"
        payload.update(status="failed", reason=reason, error=SCHEDULE_ERROR_MESSAGES[reason])
        return payload
    payload.update(status="ready", schedule=result["schedule"], reasoning_notes=result["reasoning_notes"],
                   weekly_focus_hours=result["weekly_focus_hours"], subject_breakdown=result["subject_breakdown"],
                   warnings=sdata["warnings"])
    return payload


# ---------------------------------------------------------------- API
@app.route("/api/chat", methods=["POST"])
@rate_limited("chat", 30)
def api_chat():
    data = request.get_json(silent=True)
    message = str((data or {}).get("message") or "").strip() if isinstance(data, dict) else ""
    if not message:
        return jsonify({"error": "Could you type something first?", "reply": "Could you type something first?", "done": False}), 400
    if len(message) > chat_handler.MAX_MESSAGE_CHARS:
        err = f"That's a long one. Keep it under {chat_handler.MAX_MESSAGE_CHARS} characters."
        return jsonify({"error": err, "reply": err, "done": False}), 400
    user_key, gemini_key = _byok()

    sdata = _sdata()
    with sdata["lock"]:
        if chat_handler.is_complete(sdata["chat_state"]):
            # Chat is over: never re-run the pipeline (that would burn the student's quota on a double-click).
            return jsonify({"reply": "Your schedule is ready. Open it from 'My schedule'.", "done": True})
        if sdata["chat_state"] is None:
            sdata["chat_state"] = chat_handler._new_session_state()
        reply, done, sdata["chat_state"] = chat_handler.handle_message(
            sdata["chat_state"], message, user_api_key=user_key, user_gemini_api_key=gemini_key)
        if done:
            _run_pipeline(sdata, user_key, gemini_key)
    return jsonify({"reply": reply, "done": done})


@app.route("/api/effort", methods=["POST"])
@rate_limited("effort", 30)
def api_effort():
    data = request.get_json(silent=True)
    level = str((data or {}).get("level") or "").strip().lower() if isinstance(data, dict) else ""
    if level not in ("low", "medium", "high"):
        return jsonify({"error": "level must be 'low', 'medium', or 'high'."}), 400
    _sdata()["effort"] = level
    return jsonify({"status": "ok", "level": level})


def _upload_fail(msg, code=200):
    # 200 for user-fixable problems: script.js reads the "status" field for these.
    return jsonify({"status": "failed", "error": msg}), code


@app.route("/api/upload", methods=["POST"])
@rate_limited("upload", 10)
def api_upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return _upload_fail("No file received.", 400)
    ext = document_parser.get_extension(f.filename)
    if ext not in document_parser.SUPPORTED_EXTENSIONS:
        return _upload_fail(f"Unsupported file type: .{ext}", 400)
    sdata = _sdata()
    if len(sdata["parsed_docs"]) >= MAX_DOCS_PER_SESSION:
        return _upload_fail(f"You've added {MAX_DOCS_PER_SESSION} files already, which is the limit.")

    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.{ext}")  # server-chosen name; the client's is never used on disk
    try:
        f.save(path)
        if os.path.getsize(path) > MAX_UPLOAD_MB * 1024 * 1024:
            return _upload_fail(f"That file is over {MAX_UPLOAD_MB} MB. Upload a smaller file.", 413)
        with open(path, "rb") as fh:
            if not document_parser.looks_like(ext, fh.read(16)):
                return _upload_fail(f"This doesn't look like a real .{ext} file. Check it opens on your device.")
        result = document_parser.parse_document(path, original_filename=f.filename)
    finally:
        try:
            os.remove(path)  # we keep only the extracted text, never the original
        except OSError:
            pass

    with sdata["lock"]:
        sdata["parsed_docs"].append(result.to_dict())
    return jsonify({"status": "received"}) if result.success else _upload_fail(result.error)


@app.route("/api/schedule", methods=["GET"])
def api_schedule():
    return jsonify(_schedule_payload(_sdata(create=False)))


@app.route("/api/schedule/retry", methods=["POST"])
@rate_limited("retry", 6)
def api_schedule_retry():
    user_key, gemini_key = _byok()
    sdata = _sdata()
    with sdata["lock"]:  # a concurrent retry waits here, then sees the fresh result and does nothing
        result = sdata["schedule"]
        failed = result is None or not result.get("success")
        if chat_handler.is_complete(sdata["chat_state"]) and failed:
            _run_pipeline(sdata, user_key, gemini_key)
    return jsonify(_schedule_payload(sdata))


if __name__ == "__main__":
    # Local dev server. Production uses gunicorn (see Dockerfile); debug is never allowed in production.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)),
            debug=(os.getenv("FLASK_DEBUG", "").lower() == "true" and not IS_PROD))
