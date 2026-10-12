"""
app.py
Flask backend: the glue between the frontend (landing.html / chat.html /
results.html with script.js) and the four pipeline scripts.

Routes. The first five keep the exact shapes script.js was already written
against; everything marked NEW only adds fields or endpoints.

  POST /api/chat           {"message"}                  -> {"reply", "done"}
                           NEW fields: step, total, schedule_status, notice
  POST /api/upload         multipart, field "file"      -> {"status": "received" | "failed", "error"?}
                           NEW fields: filename, note
  GET  /api/schedule                                    -> {"status", "schedule", "reasoning_notes",
                                                           "weekly_focus_hours", "subject_breakdown", "error", "reason"}
                           NEW: status may be "generating" (+ progress), plus warnings, meta, retry_after_s
  POST /api/schedule/retry                              -> same shape as GET /api/schedule
                           NEW: returns at once with "generating"; the page polls
  POST /api/effort         {"level": low|medium|high}   -> {"status": "ok", "level"}

  NEW  GET  /api/health        liveness probe (Render's health check)
  NEW  GET  /api/config        limits, greeting and which provider the server can use
  NEW  GET  /api/chat/state    redraw the chat after a refresh
  NEW  POST /api/key/test      one tiny real call proving a pasted key works
  NEW  POST /api/reset         start over

Keys: a student's own key travels per request in X-Gemini-Api-Key or
X-Anthropic-Api-Key (if both arrive, Claude wins, as it always has). Otherwise the
server's GEMINI_API_KEY / ANTHROPIC_API_KEY is used if the operator set one; with
neither, the schedule step fails with reason "no_key". A key is read from the
request, passed down as an argument and never stored in a session, a log line or a
response.

Errors: every /api error is {"error": "<student-safe sentence>", "reason": "<code>"}.

Sessions: one dict per browser in server memory (sessions.py explains why that is a
deliberate MVP decision), with idle expiry, a size cap, per-session and per-IP rate
limits, and one build at a time per student.

To run locally:
  1. cd backend && cp .env.example .env     (the defaults work; add a key if you like)
  2. pip install -r requirements.txt
  3. python app.py                           then open http://localhost:5000
"""

from __future__ import annotations

import copy
import logging
import os
import secrets
import tempfile
import threading
import time
import uuid
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, request, send_from_directory, session  # noqa: E402
from werkzeug.exceptions import HTTPException, MethodNotAllowed, NotFound  # noqa: E402
from werkzeug.middleware.proxy_fix import ProxyFix  # noqa: E402

import chat_handler  # noqa: E402
import document_parser  # noqa: E402
import llm_client  # noqa: E402
import pipeline  # noqa: E402
from sessions import RateLimiter, Session, SessionStore  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "google_genai", "google.genai", "anthropic", "urllib3", "PIL", "pdfminer"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("biosync")

APP_VERSION = "1.0.0"
START_TIME = time.time()


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("%s=%r is not a whole number; using %d", name, raw, default)
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return default if not raw else raw in ("1", "true", "yes", "on")


IS_PRODUCTION = os.getenv("APP_ENV", "").strip().lower() == "production" or bool(os.getenv("RENDER"))
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 20)
MAX_FILES = _env_int("MAX_FILES", 10)
MAX_JSON_BYTES = 64 * 1024
MAX_CONCURRENT_PARSES = _env_int("MAX_CONCURRENT_PARSES", 2)

# ---------------------------------------------------------------------------
# App, secret key, cookies
# ---------------------------------------------------------------------------
_WEAK_SECRETS = {"", "dev-only-key-change-this", "change-me", "changeme", "replace-me", "secret", "your-secret-key"}


def _load_secret_key() -> str:
    """The signing key for the session cookie. Production refuses to start without a real one."""
    key = os.getenv("FLASK_SECRET_KEY", "").strip()
    if key.lower() in _WEAK_SECRETS or len(key) < 16:
        if IS_PRODUCTION:
            raise RuntimeError(
                "FLASK_SECRET_KEY is missing or too weak. Set it to a long random value, e.g. "
                "`python -c \"import secrets; print(secrets.token_hex(32))\"` (render.yaml generates one for you).")
        logger.warning("FLASK_SECRET_KEY is not set: using a random key for this run, so sessions reset on restart.")
        return secrets.token_hex(32)
    return key


app = Flask(__name__, static_folder=None)
app.secret_key = _load_secret_key()
app.config.update(
    MAX_CONTENT_LENGTH=(MAX_UPLOAD_MB + 1) * 1024 * 1024,   # the hard ceiling for any request body
    SESSION_COOKIE_NAME="biosync_session",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
)
app.json.ensure_ascii = False
app.json.sort_keys = False

if _env_flag("TRUST_PROXY", bool(os.getenv("RENDER"))):
    # Behind Render's proxy the client IP and scheme arrive in X-Forwarded-* headers.
    # Trusted only when asked for: otherwise anyone could fake their IP and dodge the rate limits.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

STORE = SessionStore(ttl_seconds=_env_int("SESSION_TTL_MINUTES", 360, 5) * 60,
                     max_sessions=_env_int("MAX_SESSIONS", 1000, 10))
RATE = RateLimiter()
_PARSE_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_PARSES)

# bucket: (hits allowed per session, hits allowed per IP) in RATE_WINDOW_S seconds.
# The per-IP number is deliberately generous: a whole campus can share one address.
RATE_WINDOW_S = 600
LIMITS = {
    "chat": (40, 400),
    "upload": (15, 150),
    "job": (6, 60),
    "key_test": (12, 120),
    "poll": (900, 3000),
    "reset": (10, 100),
}

CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' https://fonts.googleapis.com",
    "font-src https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "object-src 'none'",
])


# ---------------------------------------------------------------------------
# Errors: one JSON envelope for everything under /api
# ---------------------------------------------------------------------------
class ApiError(Exception):
    def __init__(self, status: int, reason: str, message: str, **extra):
        super().__init__(message)
        self.status, self.reason, self.message, self.extra = status, reason, message, extra


def _error_response(status: int, reason: str, message: str, **extra):
    body = {"error": message, "reason": reason, **extra}
    response = jsonify(body)
    response.status_code = status
    if extra.get("retry_after_s"):
        response.headers["Retry-After"] = str(extra["retry_after_s"])
    return response


@app.errorhandler(ApiError)
def _on_api_error(exc: ApiError):
    return _error_response(exc.status, exc.reason, exc.message, **exc.extra)


_HTTP_MESSAGES = {
    400: ("bad_request", "That request wasn't valid."),
    404: ("not_found", "That page or endpoint doesn't exist."),
    405: ("method_not_allowed", "That request used the wrong method."),
    413: ("too_large", "That request is too large."),
    415: ("bad_request", "That request wasn't valid."),
}


@app.errorhandler(HTTPException)
def _on_http_error(exc: HTTPException):
    if not request.path.startswith("/api/"):
        return exc  # the normal page for 404s and friends
    reason, message = _HTTP_MESSAGES.get(exc.code or 500, ("http_error", "Something went wrong with that request."))
    if exc.code == 413 and request.path == "/api/upload":
        message = f"That file is too large. The limit is {MAX_UPLOAD_MB} MB."
    return _error_response(exc.code or 500, reason, message)


@app.errorhandler(Exception)
def _on_unhandled(exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.path)
    if request.path.startswith("/api/"):
        return _error_response(500, "internal_error", "Something went wrong on our side. Please try again.")
    return "Something went wrong. Please try again.", 500


# ---------------------------------------------------------------------------
# Request plumbing
# ---------------------------------------------------------------------------
@app.before_request
def _before():
    request.environ["biosync.t0"] = time.monotonic()
    if request.path.startswith("/api/") and request.path != "/api/upload":
        request.max_content_length = MAX_JSON_BYTES  # chat messages and settings are small; uploads are not


@app.after_request
def _after(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Content-Security-Policy", CSP)
    if IS_PRODUCTION:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        if request.path not in ("/api/health", "/api/schedule"):  # polls would drown the log
            ms = int((time.monotonic() - request.environ.get("biosync.t0", time.monotonic())) * 1000)
            logger.info("%s %s -> %s %dms sid=%s", request.method, request.path, response.status_code, ms,
                        str(session.get("sid", "-"))[:8])
    return response


def _sid() -> str:
    sid = session.get("sid")
    if not isinstance(sid, str) or len(sid) != 32:
        sid = uuid.uuid4().hex
        session["sid"] = sid
        session.permanent = True
    return sid


def current_session() -> Session:
    return STORE.get(_sid())


def read_json() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def enforce_limit(bucket: str) -> None:
    """Raise a 429 when this browser (or this address) is going too fast for `bucket`."""
    per_session, per_ip = LIMITS[bucket]
    allowed, retry = RATE.hit(f"{bucket}:s:{_sid()}", per_session, RATE_WINDOW_S)
    if allowed:
        allowed, retry = RATE.hit(f"{bucket}:ip:{request.remote_addr or '-'}", per_ip, RATE_WINDOW_S)
    if not allowed:
        raise ApiError(429, "too_many_requests",
                       f"You're going a little fast. Try again in {retry} seconds.", retry_after_s=retry)


def request_credentials() -> tuple[llm_client.Credentials | None, llm_client.LLMError | None]:
    """
    The API key for THIS request: a key the student pasted (headers), else the server's own.
    Returns (creds, None), or (None, why) when there is no usable key. The key is only ever
    held in local variables of the request that carries it.
    """
    try:
        return llm_client.resolve_credentials(request.headers.get("X-Anthropic-Api-Key"),
                                              request.headers.get("X-Gemini-Api-Key")), None
    except llm_client.LLMError as err:
        return None, err


# Reasons worth interrupting the chat for: the student can fix them right now.
_NOTICE_REASONS = {"no_key", "bad_key", "region_blocked", "quota_exhausted", "rate_limited", "model_unavailable"}


def notice_for(err: llm_client.LLMError | None) -> dict | None:
    if err is None or err.reason not in _NOTICE_REASONS:
        return None
    return {"reason": err.reason, "message": err.message}


def _schedule_status(sess: Session) -> str:
    """pending | generating | ready | failed. Caller need not hold the lock (single reads)."""
    job, result = sess.job, sess.schedule
    if job is not None and job.status == "generating":
        return "generating"
    if result is None:
        return "pending"
    return "ready" if result.get("success") else "failed"


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.get("/")
def serve_landing():
    return send_from_directory(FRONTEND_DIR, "landing.html")


@app.get("/<path:filename>")
def serve_static(filename: str):
    if filename.startswith("api/"):
        # An API path answers in JSON, never with a page: 405 if it exists for another method, else 404.
        try:
            app.url_map.bind_to_environ(request.environ).match("/" + filename, method="POST")
        except HTTPException:
            raise NotFound() from None
        raise MethodNotAllowed(valid_methods=["POST"])
    return send_from_directory(FRONTEND_DIR, filename)


# ---------------------------------------------------------------------------
# Small endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def api_health():
    return jsonify(status="ok", version=APP_VERSION, uptime_s=int(time.time() - START_TIME))


@app.get("/api/config")
def api_config():
    return jsonify({
        "version": APP_VERSION,
        "total_questions": chat_handler.TOTAL_QUESTIONS,
        "greeting": chat_handler.get_greeting(),
        "max_message_chars": chat_handler.MAX_ANSWER_CHARS,
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_files": MAX_FILES,
        "allowed_extensions": sorted(document_parser.SUPPORTED_EXTENSIONS),
        "efforts": list(llm_client.EFFORTS),
        "server_llm": llm_client.server_llm_info(),
        "providers": [
            {"id": llm_client.GEMINI, "label": llm_client.PROVIDER_LABEL[llm_client.GEMINI],
             "key_url": "https://aistudio.google.com/apikey", "key_hint": llm_client.KEY_HINT[llm_client.GEMINI]},
            {"id": llm_client.CLAUDE, "label": llm_client.PROVIDER_LABEL[llm_client.CLAUDE],
             "key_url": "https://console.anthropic.com/settings/keys", "key_hint": llm_client.KEY_HINT[llm_client.CLAUDE]},
        ],
    })


@app.post("/api/effort")
def api_effort():
    level = str(read_json().get("level") or "").strip().lower()
    if level not in llm_client.EFFORTS:
        raise ApiError(400, "bad_request", "level must be 'low', 'medium', or 'high'.")
    sess = current_session()
    with sess.lock:
        sess.effort = level
    return jsonify({"status": "ok", "level": level})


@app.post("/api/key/test")
def api_key_test():
    """Prove a key works with one tiny real call, so a typo is caught before the 7 questions, not after."""
    enforce_limit("key_test")
    creds, issue = request_credentials()
    if creds is None:
        err = issue or llm_client.LLMError("no_key")
        return jsonify(ok=False, reason=err.reason, message=err.message)
    try:
        result = llm_client.check_credentials(creds)
    except llm_client.LLMError as err:
        logger.info("key test failed: provider=%s reason=%s", creds.provider, err.reason)
        return jsonify(ok=False, reason=err.reason, message=err.message, provider=creds.provider,
                       retry_after_s=err.retry_after)
    logger.info("key test ok: provider=%s model=%s source=%s %dms", creds.provider, result.model, creds.source,
                result.latency_ms)
    return jsonify(ok=True, provider=creds.provider, model=result.model, latency_ms=result.latency_ms,
                   source=creds.source)


@app.post("/api/reset")
def api_reset():
    enforce_limit("reset")
    STORE.reset(_sid())
    return jsonify(status="ok")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@app.get("/api/chat/state")
def api_chat_state():
    sess = current_session()
    with sess.lock:
        state = sess.chat_state
        return jsonify({
            **chat_handler.progress(state),
            "done": chat_handler.is_complete(state),
            "greeting": chat_handler.get_greeting(),
            "messages": chat_handler.public_transcript(state),
            "effort": sess.effort,
            "documents": [{"name": d.get("source_filename"), "status": "received" if d.get("success") else "failed",
                           "error": d.get("error"), "note": d.get("note")} for d in sess.parsed_docs],
            "schedule_status": _schedule_status(sess),
        })


@app.post("/api/chat")
def api_chat():
    enforce_limit("chat")
    message = str(read_json().get("message") or "").strip()
    if not message:
        return jsonify({"reply": chat_handler.EMPTY_REPLY, "done": False, "error": chat_handler.EMPTY_REPLY,
                        "reason": "empty_message"}), 400

    sess = current_session()
    creds, key_issue = request_credentials()

    # A double-tap or a second tab must not advance the checklist twice.
    if not sess.chat_lock.acquire(blocking=False):
        raise ApiError(409, "busy", "One moment, I'm still working on your last message.")
    try:
        with sess.lock:
            state = copy.deepcopy(sess.chat_state) if sess.chat_state else chat_handler.new_session_state()
            was_done = chat_handler.is_complete(state)
        # The AI call happens here, with no lock held, so polling stays responsive.
        turn = chat_handler.handle_message(state, message, creds)
        with sess.lock:
            sess.chat_state = turn.state
        if turn.done and not was_done:
            # Never let the build crash the final chat reply: it runs in the background and reports
            # its own failure through /api/schedule.
            pipeline.start_job(sess, creds, missing=key_issue)
    finally:
        sess.chat_lock.release()

    payload = {"reply": turn.reply, "done": turn.done, **chat_handler.progress(turn.state),
               "schedule_status": _schedule_status(sess)}
    notice = notice_for(key_issue or turn.llm_issue)
    if notice:
        payload["notice"] = notice
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
@app.post("/api/upload")
def api_upload():
    enforce_limit("upload")
    if "file" not in request.files:
        return jsonify({"status": "failed", "error": "No file received."}), 400
    file = request.files["file"]
    original_filename = file.filename or ""
    if original_filename == "":
        return jsonify({"status": "failed", "error": "Empty filename."}), 400
    ext = document_parser.get_extension(original_filename)
    if ext not in document_parser.SUPPORTED_EXTENSIONS:
        return jsonify({"status": "failed", "error": f"Unsupported file type: .{ext}"}), 400

    sess = current_session()
    with sess.lock:
        if len(sess.parsed_docs) >= MAX_FILES:
            return jsonify({"status": "failed",
                            "error": f"That's the limit of {MAX_FILES} files. Start over to change them."}), 400

    if not _PARSE_SLOTS.acquire(timeout=30):
        return jsonify({"status": "failed", "error": "BioSync is busy reading other files. Try again in a moment."}), 503
    try:
        # A private temporary folder per request: the file gets a name we chose (never the student's),
        # and it is deleted when the block ends, whether or not parsing worked.
        with tempfile.TemporaryDirectory(prefix="biosync-") as tmp:
            save_path = os.path.join(tmp, f"upload.{ext}")
            file.save(save_path)
            result = document_parser.parse_document(save_path, original_filename=original_filename)
    finally:
        _PARSE_SLOTS.release()

    with sess.lock:
        sess.parsed_docs.append(result.to_dict())

    if result.success:
        body = {"status": "received", "filename": result.source_filename}
        if result.note:
            body["note"] = result.note
        return jsonify(body)
    # 200 on purpose: script.js checks the "status" field in the body for this failure case.
    return jsonify({"status": "failed", "error": result.error, "filename": result.source_filename}), 200


# ---------------------------------------------------------------------------
# Schedule
# GET /api/schedule always answers HTTP 200: the signal is in the body, so a proxy in
# front of the app cannot swallow it the way it could a 429/502/503 error page.
#
#   status "pending"     the chat isn't finished, nothing has been built yet
#   status "generating"  the build is running (progress: {stage, elapsed_s})
#   status "ready"       schedule, reasoning_notes, weekly_focus_hours, subject_breakdown,
#                        warnings (things to know) and meta (provider, model, effort)
#   status "failed"      error (a sentence for the student) + reason (a stable code) and,
#                        for rate limits, retry_after_s
# The provider's raw error text is only logged, never sent to the browser.
# ---------------------------------------------------------------------------
def schedule_payload(sess: Session) -> dict:
    with sess.lock:
        payload = {
            "status": "pending", "schedule": [], "reasoning_notes": "", "weekly_focus_hours": None,
            "subject_breakdown": [], "error": None, "reason": None,
        }
        job, result = sess.job, sess.schedule
        if job is not None and job.status == "generating":
            payload.update(status="generating", progress=pipeline.job_progress(job))
            return payload
        if result is None:
            return payload
        payload["warnings"] = list(result.get("warnings") or [])
        if not result.get("success"):
            reason = result.get("reason") or "generation_failed"
            provider = (result.get("meta") or {}).get("provider")
            payload.update(status="failed", reason=reason,
                           error=result.get("error") or llm_client.public_message(reason, provider),
                           provider=provider, retry_after_s=result.get("retry_after_s"))
            return payload
        payload.update(
            status="ready",
            schedule=result["schedule"],
            reasoning_notes=result.get("reasoning_notes") or "",
            weekly_focus_hours=result.get("weekly_focus_hours"),
            subject_breakdown=result.get("subject_breakdown") or [],
            meta=result.get("meta") or {},
        )
        return payload


@app.get("/api/schedule")
def api_schedule():
    enforce_limit("poll")
    return jsonify(schedule_payload(current_session()))


@app.post("/api/schedule/retry")
def api_schedule_retry():
    """
    After a failed build the chat is already finished, so without this the only way to
    retry would be to redo the whole chat. Starts a new build only if the chat is complete
    and the last attempt did not produce a schedule: never while one is running, never
    over a good one. Returns at once; the page polls GET /api/schedule.
    """
    enforce_limit("job")
    sess = current_session()
    creds, key_issue = request_credentials()
    with sess.lock:
        chat_done = chat_handler.is_complete(sess.chat_state)
        can_retry = chat_done and (sess.job is None or sess.job.status == "failed")
    if can_retry:
        pipeline.start_job(sess, creds, missing=key_issue)
    return jsonify(schedule_payload(sess))


if __name__ == "__main__":
    # Render (and most PaaS hosts) inject the port to bind via the PORT env var; 5000 for local dev.
    # debug is off unless asked for, and never in production: the Werkzeug debugger exposes a
    # remote shell on a public URL.
    port = _env_int("PORT", 5000)
    debug_mode = _env_flag("FLASK_DEBUG", False) and not IS_PRODUCTION
    info = llm_client.server_llm_info()
    logger.info("BioSync %s starting on :%d (env=%s). Server AI key: %s", APP_VERSION, port,
                "production" if IS_PRODUCTION else "development",
                f"{info['provider']} (students may still use their own)" if info["available"]
                else "none, so every student pastes their own Gemini or Claude key")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
else:
    _info = llm_client.server_llm_info()
    logger.info("BioSync %s ready (env=%s). Server AI key: %s", APP_VERSION,
                "production" if IS_PRODUCTION else "development",
                f"{_info['provider']}" if _info["available"] else "none (students bring their own)")
