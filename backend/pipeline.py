"""
pipeline.py
Runs the post-chat build (read uploads -> structure them -> generate the schedule)
on a background thread.

Why a background job: the last /api/chat reply used to wait for two LLM calls
(up to a few minutes in the worst case) and then reply, which a browser, a proxy
or a nervous student could give up on mid-way. Now the last reply returns at
once, the build runs here, and the results page polls GET /api/schedule, which
reports "generating" (with the current stage and elapsed seconds), then
"ready" or "failed" with the reason.

Rules this module keeps:
  * at most one build per session at a time (start_job is idempotent), so a
    double-tap or a retry-while-running can never burn the student's quota twice;
  * at most MAX_CONCURRENT_JOBS builds across the server (extra ones wait in a queue);
  * credentials are passed in as an argument and live only in this call stack;
    they are never written to the session;
  * a build always ends in a stored result: success, or a failure with a reason.
    Nothing here can raise out of the worker thread.
"""

from __future__ import annotations

import logging
import os
import time

from concurrent.futures import ThreadPoolExecutor

import chat_handler
import llm_client
import schedule_generator
import syllabus_structurer
from llm_client import Credentials
from sessions import Job, Session

logger = logging.getLogger(__name__)

MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "4")))
JOB_BUDGET_S = 300          # wall-clock cap for one whole build
_STRUCTURE_BUDGET_S = 90
_SCHEDULE_BUDGET_S = 260

_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="biosync-build")
_submit = _executor.submit  # indirection so tests can run a build inline


def _snapshot(sess: Session) -> dict:
    """Copy what the build needs, so the student's later actions cannot change it mid-run."""
    docs = [(d.get("source_filename") or "document", d.get("text") or "")
            for d in sess.parsed_docs if d.get("success") and d.get("text")]
    return {
        "answers": chat_handler.get_collected_answers(sess.chat_state),
        "docs": docs,
        "effort": llm_client.normalize_effort(sess.effort),
    }


def _finish(sess: Session, job: Job, result: dict) -> None:
    """Store the outcome. Caller holds sess.lock."""
    sess.schedule = result
    job.status = "ready" if result.get("success") else "failed"
    job.stage = "done"
    job.finished_at = time.time()


def start_job(sess: Session, creds: Credentials | None,
              missing: llm_client.LLMError | None = None) -> tuple[Job, bool]:
    """
    Start a build unless one is already running or finished. Returns (job, started).

    With creds=None (no usable key) the job fails immediately, with the reason in `missing`
    (default "no_key"), so the results page can show the key form instead of a spinner
    that never ends.
    """
    with sess.lock:
        if sess.job is not None and sess.job.status in ("generating", "ready"):
            return sess.job, False
        job = Job()
        sess.job = job
        sess.schedule = None
        snapshot = _snapshot(sess)
        if creds is None:
            err = missing or llm_client.LLMError("no_key")
            _finish(sess, job, schedule_generator._failure(err.reason, err.message, provider=err.provider))
            return job, True
    _submit(_run, sess, job, snapshot, creds)
    return job, True


def job_progress(job: Job) -> dict:
    return {"stage": job.stage, "elapsed_s": max(0, int(time.time() - job.started_at))}


def _stage(sess: Session, job: Job, stage: str) -> None:
    with sess.lock:
        if sess.job is job:
            job.stage = stage


def _run(sess: Session, job: Job, snapshot: dict, creds: Credentials) -> None:
    tag = sess.sid[:8]  # enough to follow one student's build through the logs; not a credential
    started = time.monotonic()
    logger.info("[%s] build started: effort=%s docs=%d provider=%s key_source=%s", tag, snapshot["effort"],
                len(snapshot["docs"]), creds.provider, creds.source)
    try:
        result = _build(sess, job, snapshot, creds, started)
    except Exception:  # noqa: BLE001 - the worker thread must always store an outcome
        logger.exception("[%s] build crashed", tag)
        result = schedule_generator._failure(
            "internal_error", llm_client.public_message("internal_error", creds.provider), provider=creds.provider)
    with sess.lock:
        if sess.job is not job:
            logger.info("[%s] build finished after being replaced; result discarded", tag)
            return
        _finish(sess, job, result)
    logger.info("[%s] build %s in %.1fs reason=%s", tag, job.status, time.monotonic() - started,
                result.get("reason") or "-")


def _build(sess: Session, job: Job, snapshot: dict, creds: Credentials, started: float) -> dict:
    warnings: list[str] = []
    subjects: list[dict] = []
    deadlines: list[dict] = []
    docs = snapshot["docs"]

    if docs:
        _stage(sess, job, "structuring")
        structured = syllabus_structurer.structure_documents(
            [text for _, text in docs], creds, names=[name for name, _ in docs],
            deadline=min(time.monotonic() + _STRUCTURE_BUDGET_S, started + JOB_BUDGET_S))
        with sess.lock:
            if sess.job is job:
                sess.structured_data = {k: structured.get(k) for k in ("success", "data", "reason", "warnings")}
        if structured["success"]:
            subjects = structured["data"]["subjects"]
            deadlines = structured["data"]["exams_or_deadlines"]
            warnings.extend(structured["warnings"])
        elif structured["reason"] in syllabus_structurer.ABORT_REASONS:
            # The schedule step would hit the same wall: say so now instead of waiting for it.
            return schedule_generator._failure(structured["reason"], structured["error"], provider=creds.provider,
                                               retry_after=structured.get("retry_after_s"))
        else:
            # A glitch reading the files (unreadable reply, cut-off, timeout): still build the week
            # from the student's answers, but say plainly that the files were not used.
            warnings.append("We couldn't read your uploaded files this time, so this week isn't tailored to "
                            "your subjects. Try again to include them.")

    _stage(sess, job, "scheduling")
    result = schedule_generator.generate_schedule(
        subjects, deadlines, snapshot["answers"], effort=snapshot["effort"], creds=creds,
        deadline=min(time.monotonic() + _SCHEDULE_BUDGET_S, started + JOB_BUDGET_S))
    if result["success"]:
        result["warnings"] = warnings + result["warnings"]
        result["meta"].update(documents=len(docs), subjects_found=len(subjects))
    return result
