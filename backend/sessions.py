"""
sessions.py
Per-browser state held in server memory, plus a small sliding-window rate limiter.

Why memory and not a database (decision record):
BioSync has no accounts. A session is ~10 KB (seven short answers, a few
extracted syllabus texts, one schedule) and is rebuilt in about a minute, so
nothing here is precious enough to justify SQLite. On the free Render tier the
disk is wiped on every deploy and spin-down anyway, so a database file would
not survive the events it is meant to survive. Real persistence needs real
infrastructure (Postgres + accounts), which is a product decision, not a
patch. What is done instead:
  * idle sessions expire (SESSION_TTL_MINUTES) and the table is capped
    (MAX_SESSIONS), so memory cannot grow without bound;
  * the server runs ONE worker process with threads (see the Dockerfile), because
    two workers would each have their own memory and a student's requests would
    land in the wrong one;
  * the browser keeps a copy of the last finished schedule (script.js), so a
    restart costs the student nothing once their week exists;
  * after a restart, /api/chat/state tells the page the truth (step 0) instead of
    leaving a stale "Question 5 of 7" on screen.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Job:
    """One background schedule build."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "generating"   # generating | ready | failed
    stage: str = "queued"        # queued | structuring | scheduling | done
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None


@dataclass
class Session:
    sid: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    chat_state: dict | None = None
    parsed_docs: list = field(default_factory=list)   # ParseResult.to_dict() per upload
    structured_data: dict | None = None
    schedule: dict | None = None                      # result of schedule_generator (success or failure)
    effort: str = "medium"
    job: Job | None = None
    # chat_lock serialises /api/chat so a double-tap cannot advance the checklist twice;
    # lock guards every other field (the build runs on another thread).
    chat_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # Deliberately NOT stored here: any API key. Keys travel per request in headers and
    # live only in local variables of the request/build that uses them.


class SessionStore:
    def __init__(self, ttl_seconds: float = 6 * 3600, max_sessions: int = 1000, clock=time.time):
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._clock = clock
        self._items: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._last_purge = clock()

    def get(self, sid: str) -> Session:
        """Fetch (or create) the session and mark it as seen."""
        now = self._clock()
        with self._lock:
            sess = self._items.get(sid)
            if sess is None:
                sess = self._items[sid] = Session(sid=sid, created_at=now, last_seen=now)
            sess.last_seen = now
            if now - self._last_purge > 60 or len(self._items) > self._max:
                self._purge_locked(now)
            return sess

    def peek(self, sid: str) -> Session | None:
        with self._lock:
            return self._items.get(sid)

    def reset(self, sid: str) -> Session:
        """Start the student over: a fresh Session replaces the old one (a running build, if any,
        keeps writing to the orphaned object and so can no longer touch the new session)."""
        now = self._clock()
        with self._lock:
            sess = self._items[sid] = Session(sid=sid, created_at=now, last_seen=now)
            return sess

    def purge(self) -> int:
        with self._lock:
            return self._purge_locked(self._clock())

    def _purge_locked(self, now: float) -> int:
        self._last_purge = now
        before = len(self._items)
        busy = lambda s: s.job is not None and s.job.status == "generating"  # noqa: E731
        for sid in [k for k, s in self._items.items() if now - s.last_seen > self._ttl and not busy(s)]:
            del self._items[sid]
        if len(self._items) > self._max:  # still too many: drop the least recently seen
            idle_first = sorted((s for s in self._items.values() if not busy(s)), key=lambda s: s.last_seen)
            for sess in idle_first[: len(self._items) - self._max]:
                del self._items[sess.sid]
        return before - len(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class RateLimiter:
    """
    Sliding-window limiter: allow `limit` hits per `window_s` per key. In memory, like
    the sessions, which is right for one worker process. It protects the student's own
    API quota (a stuck script hammering /api/chat would burn their key) and the server.
    """

    def __init__(self, clock=time.monotonic):
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._clock = clock
        self._last_purge = clock()

    def hit(self, key: str, limit: int, window_s: float) -> tuple[bool, int]:
        """Record an attempt. Returns (allowed, retry_after_seconds)."""
        now = self._clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= window_s:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, int(window_s - (now - q[0]) + 0.999))
            q.append(now)
            if now - self._last_purge > 120:
                self._purge_locked(now, window_s)
            return True, 0

    def _purge_locked(self, now: float, window_s: float) -> None:
        self._last_purge = now
        for key in [k for k, q in self._hits.items() if not q or now - q[-1] >= max(window_s, 600)]:
            del self._hits[key]
