"""
A small fake of the Gemini and Anthropic HTTP APIs, so BioSync can be exercised end to end
(real SDK clients, real HTTP, real JSON) without real keys or internet.

It is driven by a Scenario that tests change on the fly: which models "no longer exist",
whether thinking tokens eat the output budget, queued failures (429, 500, ...), how sloppy
the schedule JSON is, and so on. Every request is recorded so tests can assert on what the
app actually sent (model, thinking setting, token cap, never the key in a body).

This is a *simulation*: it reproduces the failure modes documented by Google and Anthropic,
not their real servers. scripts/live_check.py is the one that talks to the real ones.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
KNOWN_SUBJECTS = ["Data Structures", "Linear Algebra", "Operating Systems", "Digital Logic"]
CHARS_PER_TOKEN = 3.5


@dataclass
class Scenario:
    # --- availability / auth ---
    unavailable_models: set = field(default_factory=set)        # answer 404 "no longer available to new users"
    valid_gemini_keys: set = field(default_factory=set)         # empty = accept any
    valid_claude_keys: set = field(default_factory=set)
    gemini_models: list = field(default_factory=lambda: ["gemini-3.5-flash", "gemini-3.7-flash-lite"])
    claude_models: list = field(default_factory=lambda: ["claude-sonnet-5-5", "claude-haiku-5-5"])
    region_blocked: bool = False
    # --- thinking behaviour (Gemini) ---
    thinking_tokens: int = 0                                    # tokens a thinking model burns per call
    reject_thinking_level: set = field(default_factory=lambda: {"gemini-2.5-flash", "gemini-2.0-flash"})
    reject_thinking_budget: set = field(default_factory=set)    # models answering 400 to thinking_budget
    # --- queued one-off failures: {"provider": "gemini"|"claude"|None, "status": 429, "message": str,
    #     "retry_after": int|None, "details": [...]} consumed in order by matching requests ---
    queue: list = field(default_factory=list)
    always_fail: dict | None = None                              # same shape, never consumed
    # --- content quality ---
    schedule_style: str = "clean"          # clean | ampm | five_days | fenced | dict_days | messy_types | empty_text
    broken_first_schedule: str | None = None   # style for the first schedule reply only; a repair request fixes it
    chat_reply: str | None = None          # fixed chat reply (None = derive from the question)
    safety_block: bool = False
    latency_s: float = 0.0
    max_tokens_floor_for_json: int = 0     # pretend JSON needs at least this many tokens regardless of content


class FakeLLMServer:
    def __init__(self):
        self.scenario = Scenario()
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self._schedule_calls = 0
        self._httpd: ThreadingHTTPServer | None = None
        self.base_url = ""

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "FakeLLMServer":
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence
                pass

            def _read(self) -> bytes:
                n = int(self.headers.get("content-length") or 0)
                return self.rfile.read(n) if n else b""

            def _send(self, status: int, body: dict, headers: dict | None = None):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                for k, v in (headers or {}).items():
                    self.send_header(k, str(v))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                server.handle(self, "GET", self.path, b"")

            def do_POST(self):
                server.handle(self, "POST", self.path, self._read())

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._httpd.server_port}"
        return self

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()

    def reset(self):
        with self._lock:
            self.scenario = Scenario()
            self.requests.clear()
            self._schedule_calls = 0

    # -- helpers for tests ---------------------------------------------------
    def calls(self, provider: str | None = None, task: str | None = None) -> list[dict]:
        with self._lock:
            return [r for r in self.requests
                    if (provider is None or r["provider"] == provider) and (task is None or r["task"] == task)]

    def models_called(self, provider: str, task: str | None = None) -> list[str]:
        return [r["model"] for r in self.calls(provider, task)]

    # -- request handling ----------------------------------------------------
    def handle(self, h: BaseHTTPRequestHandler, method: str, path: str, body: bytes):
        sc = self.scenario
        if sc.latency_s:
            time.sleep(sc.latency_s)
        headers = {k.lower(): v for k, v in h.headers.items()}
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        gem = re.match(r"^/v1beta/models/([^:/]+):generateContent$", path)
        if method == "POST" and gem:
            return self._gemini_generate(h, gem.group(1), payload, headers)
        if method == "GET" and path.startswith("/v1beta/models"):
            return self._gemini_list(h, headers)
        if method == "POST" and path == "/v1/messages":
            return self._claude_messages(h, payload, headers)
        if method == "GET" and path.startswith("/v1/models"):
            return self._claude_list(h, headers)
        h._send(404, {"error": {"code": 404, "message": f"no route {method} {path}", "status": "NOT_FOUND"}})

    def _next_failure(self, provider: str):
        with self._lock:
            for i, item in enumerate(self.scenario.queue):
                if item.get("provider") in (None, provider):
                    return self.scenario.queue.pop(i)
        return self.scenario.always_fail if (self.scenario.always_fail and
                                             self.scenario.always_fail.get("provider") in (None, provider)) else None

    def _record(self, **info):
        with self._lock:
            self.requests.append(info)

    # -- Gemini --------------------------------------------------------------
    def _gem_error(self, h, status, message, api_status, retry_after=None, details=None):
        body = {"error": {"code": status, "message": message, "status": api_status}}
        if details:
            body["error"]["details"] = details
        if retry_after:
            body["error"].setdefault("details", []).append(
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": f"{retry_after}s"})
        h._send(status, body)

    def _gemini_list(self, h, headers):
        key = headers.get("x-goog-api-key", "")
        if self.scenario.valid_gemini_keys and key not in self.scenario.valid_gemini_keys:
            return self._gem_error(h, 400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")
        models = [{"name": f"models/{m}", "displayName": m,
                   "supportedGenerationMethods": ["generateContent"], "supportedActions": ["generateContent"]}
                  for m in self.scenario.gemini_models]
        h._send(200, {"models": models})

    def _gemini_generate(self, h, model, payload, headers):
        sc = self.scenario
        key = headers.get("x-goog-api-key", "")
        gen = payload.get("generationConfig") or {}
        thinking = gen.get("thinkingConfig") or {}
        level = thinking.get("thinkingLevel") or thinking.get("thinking_level")
        budget = thinking.get("thinkingBudget", thinking.get("thinking_budget"))
        system = ((payload.get("systemInstruction") or {}).get("parts") or [{}])[0].get("text", "")
        contents = payload.get("contents") or []
        last_user = ((contents[-1].get("parts") or [{}])[0].get("text", "")) if contents else ""
        task = detect_task(system)
        self._record(provider="gemini", model=model, task=task, thinking_level=level, thinking_budget=budget,
                     max_tokens=gen.get("maxOutputTokens"), json_mode=gen.get("responseMimeType") == "application/json",
                     key_ok=bool(key), turns=len(contents))

        if sc.valid_gemini_keys and key not in sc.valid_gemini_keys:
            return self._gem_error(h, 400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT",
                                   details=[{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}])
        if sc.region_blocked:
            return self._gem_error(h, 400, "User location is not supported for the API use.", "FAILED_PRECONDITION")
        if model in sc.unavailable_models:
            return self._gem_error(h, 404, f"models/{model} is no longer available to new users. Please update your "
                                           "code to use a newer model for the latest features and improvements.", "NOT_FOUND")
        failure = self._next_failure("gemini")
        if failure:
            return self._gem_error(h, failure["status"], failure.get("message", "failure"),
                                   failure.get("api_status", "UNAVAILABLE"), failure.get("retry_after"))
        if level is not None and budget is not None:
            return self._gem_error(h, 400, "Cannot set both thinking_level and thinking_budget.", "INVALID_ARGUMENT")
        if level is not None and model in sc.reject_thinking_level:
            return self._gem_error(h, 400, f"Thinking level is not supported for model {model}.", "INVALID_ARGUMENT")
        if budget is not None and model in sc.reject_thinking_budget:
            return self._gem_error(h, 400, f"thinking_budget is not supported for model {model}.", "INVALID_ARGUMENT")
        if sc.safety_block:
            return h._send(200, {"candidates": [{"finishReason": "SAFETY", "index": 0}],
                                 "usageMetadata": {"promptTokenCount": 10, "totalTokenCount": 10}})

        text = self._content(task, system, last_user)
        max_out = int(gen.get("maxOutputTokens") or 8192)
        thinks = sc.thinking_tokens if (level is not None or (budget not in (None, 0))) else 0
        if budget == 0:
            thinks = 0
        need = max(int(len(text) / CHARS_PER_TOKEN), sc.max_tokens_floor_for_json if gen.get("responseMimeType") else 0)
        available = max_out - thinks
        finish, out_text = "STOP", text
        if available <= 0:
            finish, out_text = "MAX_TOKENS", ""
        elif need > available:
            finish, out_text = "MAX_TOKENS", text[: int(available * CHARS_PER_TOKEN)]
        usage = {"promptTokenCount": max(1, len(system + last_user) // 4),
                 "candidatesTokenCount": min(need, max(available, 0)), "thoughtsTokenCount": thinks,
                 "totalTokenCount": 0}
        cand = {"content": {"role": "model", "parts": ([{"text": out_text}] if out_text else [])},
                "finishReason": finish, "index": 0}
        h._send(200, {"candidates": [cand], "usageMetadata": usage, "modelVersion": model})

    # -- Claude --------------------------------------------------------------
    def _claude_error(self, h, status, etype, message, retry_after=None):
        h._send(status, {"type": "error", "error": {"type": etype, "message": message}},
                {"retry-after": retry_after} if retry_after else None)

    def _claude_list(self, h, headers):
        key = headers.get("x-api-key", "")
        if self.scenario.valid_claude_keys and key not in self.scenario.valid_claude_keys:
            return self._claude_error(h, 401, "authentication_error", "invalid x-api-key")
        data = [{"id": m, "type": "model", "display_name": m, "created_at": "2026-01-01T00:00:00Z"}
                for m in self.scenario.claude_models]
        h._send(200, {"data": data, "has_more": False, "first_id": data[0]["id"] if data else None,
                      "last_id": data[-1]["id"] if data else None})

    def _claude_messages(self, h, payload, headers):
        sc = self.scenario
        key = headers.get("x-api-key", "")
        model = payload.get("model", "")
        system = payload.get("system", "")
        if isinstance(system, list):
            system = " ".join(b.get("text", "") for b in system)
        messages = payload.get("messages") or []
        last_user = messages[-1]["content"] if messages else ""
        if isinstance(last_user, list):
            last_user = " ".join(b.get("text", "") for b in last_user)
        task = detect_task(system)
        self._record(provider="claude", model=model, task=task, max_tokens=payload.get("max_tokens"),
                     has_temperature="temperature" in payload, key_ok=bool(key), turns=len(messages),
                     first_role=messages[0]["role"] if messages else None)

        if sc.valid_claude_keys and key not in sc.valid_claude_keys:
            return self._claude_error(h, 401, "authentication_error", "invalid x-api-key")
        if model in sc.unavailable_models:
            return self._claude_error(h, 404, "not_found_error", f"model: {model}")
        failure = self._next_failure("claude")
        if failure:
            return self._claude_error(h, failure["status"], failure.get("etype", "overloaded_error"),
                                      failure.get("message", "failure"), failure.get("retry_after"))
        text = self._content(task, system, last_user)
        max_out = int(payload.get("max_tokens") or 1024)
        need = int(len(text) / CHARS_PER_TOKEN)
        stop, out = "end_turn", text
        if need > max_out:
            stop, out = "max_tokens", text[: int(max_out * CHARS_PER_TOKEN)]
        h._send(200, {"id": "msg_fake", "type": "message", "role": "assistant", "model": model,
                      "content": [{"type": "text", "text": out}], "stop_reason": stop, "stop_sequence": None,
                      "usage": {"input_tokens": max(1, len(system + last_user) // 4), "output_tokens": min(need, max_out)}})

    # -- what the "model" says -----------------------------------------------
    def _content(self, task: str, system: str, last_user: str) -> str:
        sc = self.scenario
        if task == "ping":
            return "OK"
        if task == "chat":
            if sc.chat_reply is not None:
                return sc.chat_reply
            m = re.search(r'"([^"]+)"\s*\n\s*\nRules:', system)
            question = m.group(1) if m else "Tell me more."
            return f"Got it, thanks! {question}"
        if task == "extract":
            return json.dumps(extract_from(last_user))
        if task == "schedule":
            with self._lock:
                self._schedule_calls += 1
                first = self._schedule_calls == 1
            repair = "could not be used" in last_user
            style = sc.schedule_style
            if sc.broken_first_schedule and first and not repair:
                style = sc.broken_first_schedule
            return build_schedule_reply(system, last_user, style)
        return "OK"


def detect_task(system: str) -> str:
    if "connectivity check" in system:
        return "ping"
    if "You are BioSync, a friendly AI assistant" in system:
        return "chat"
    if "You extract structured data" in system:
        return "extract"
    if "expert in student time management" in system:
        return "schedule"
    return "other"


def extract_from(user_text: str) -> dict:
    """Pretend extraction: find known subject names and dated 'Exam' lines in the documents."""
    subjects = [{"name": s, "weekly_hours": 3 + i, "notes": f"{3 + i} credits"}
                for i, s in enumerate(KNOWN_SUBJECTS) if s.lower() in user_text.lower()]
    exams = []
    for line in user_text.splitlines():
        if "exam" in line.lower() or "deadline" in line.lower():
            date = re.search(r"\d{4}-\d{2}-\d{2}", line)
            exams.append({"description": re.sub(r"\s+", " ", line.strip())[:80], "date": date.group() if date else None})
    return {"subjects": subjects, "exams_or_deadlines": exams[:5], "parse_confidence": "high" if subjects else "low"}


def _fmt_time(start: str, end: str, style: str) -> str:
    if style != "ampm":
        return f"{start}–{end}"

    def ampm(t):
        h, m = (int(x) for x in t.split(":"))
        return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"
    return f"{ampm(start)} - {ampm(end)}"


def build_schedule_reply(system: str, user_text: str, style: str) -> str:
    if style == "empty_text":
        return ""
    tier = "low" if "COMPACT tier" in system else ("high" if "2-4 sentences" in system else "medium")
    subjects = []
    in_subjects = False
    for line in user_text.splitlines():
        if line.startswith("SUBJECTS:"):
            in_subjects = True
            continue
        if in_subjects:
            if not line.strip():
                break
            m = re.match(r"^- (.+?)(?:, ~[\d.]+h/week)?(?: \(.*\))?$", line)
            if m and not m.group(1).startswith("(none"):
                subjects.append(m.group(1))
    subjects = subjects or [""]
    day_names = DAYS[:5] if style == "five_days" else DAYS
    notes = tier != "low"
    days = []
    for i, day in enumerate(day_names):
        s1, s2, s3 = (subjects[(i + k) % len(subjects)] for k in range(3))
        rows = [
            ("6:30", "7:00", "Wake up and morning routine", "other", "", "low", "gentle start"),
            ("7:00", "7:30", "Breakfast", "meal", "", "medium", ""),
            ("9:00", "11:00", f"Deep focus: {s1 or 'self-study'} problem set", "study", s1, "high", "your sharpest window"),
            ("11:00", "11:15", "Short walk", "break", "", "medium", ""),
            ("11:15", "12:30", f"{s2 or 'Reading'} revision", "study", s2, "high", "still alert before lunch"),
            ("12:30", "13:15", "Lunch", "meal", "", "low", "light meal before the dip"),
            ("13:15", "14:15", "Admin and light reading", "other", "", "low", "post-lunch dip"),
            ("14:30", "16:00", f"{s3 or 'Practice'} practice questions", "study", s3, "medium", "energy recovers"),
            ("16:00", "16:30", "Tea break", "break", "", "medium", ""),
            ("19:00", "19:45", "Dinner", "meal", "", "low", ""),
            ("20:00", "21:30", f"{s1 or 'Self-study'} recap", "study", s1, "medium", "short recap before wind-down"),
            ("23:00", "6:30", "Sleep", "sleep", "", "low", ""),
        ]
        blocks = []
        for start, end, activity, kind, subject, energy, note in rows:
            block = {"time": _fmt_time(start, end, style), "activity": activity, "type": kind,
                     "subject": subject, "energy": energy, "note": note if notes else ""}
            if style == "messy_types":
                block["type"] = {"study": "Revision", "meal": "Lunch", "break": "REST", "sleep": "bed",
                                 "other": "Admin"}.get(kind, kind)
                block["energy"] = {"high": "peak", "medium": "Moderate", "low": "LOW"}[energy]
            blocks.append(block)
        days.append({"day": day if style != "messy_types" else day[:3].upper(),
                     "focus_summary": f"{day}: built around {s1 or 'steady study'}" if notes else "", "blocks": blocks})
    reasoning = ("Hard subjects sit in your morning peak and lighter reading in the post-lunch dip." if notes
                 else "Hard subjects sit in your peak window.")
    if style == "dict_days":
        data = {"schedule": {d["day"]: {"focus_summary": d["focus_summary"], "blocks": d["blocks"]} for d in days},
                "reasoning_notes": reasoning}
    else:
        data = {"schedule": days, "reasoning_notes": reasoning}
    text = json.dumps(data, ensure_ascii=False)
    if style == "fenced":
        return "Sure! Here is your schedule:\n\n```json\n" + text + "\n```\n\nLet me know if you want changes."
    return text
