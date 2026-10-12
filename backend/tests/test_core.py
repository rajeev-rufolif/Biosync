"""Offline tests: no network, no API keys. Run: python -m unittest discover -s tests -v (from backend/)."""
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["APP_ENV"] = "development"

import app as app_module  # noqa: E402
import chat_handler  # noqa: E402
import document_parser  # noqa: E402
import llm_client  # noqa: E402
import schedule_generator as sg  # noqa: E402

DAYS = sg.DAYS


def fake_week(**over):
    """A valid 7-day model reply. The model's own totals are deliberately WRONG to prove they are recomputed."""
    blocks = [
        {"time": "8:00–10:00", "activity": "Deep focus: Data Structures", "type": "study", "subject": "Data Structures", "energy": "high", "note": ""},
        {"time": "13:00-14:00", "activity": "Lunch", "type": "meal", "subject": "", "energy": "low", "note": ""},
    ]
    week = {"schedule": [{"day": d, "focus_summary": "x", "blocks": [dict(b) for b in blocks]} for d in DAYS],
            "reasoning_notes": "because", "weekly_focus_hours": 999, "subject_breakdown": []}
    week.update(over)
    return week


class JsonExtraction(unittest.TestCase):
    def test_plain_fenced_and_prose_wrapped(self):
        for raw in ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here you go:\n{"a": 1}\nHope that helps.']:
            self.assertEqual(llm_client.extract_json(raw), {"a": 1})

    def test_truncated_or_empty_raises(self):
        for raw in ['{"a": [1, 2', "", None, "no json here"]:
            with self.assertRaises(ValueError):
                llm_client.extract_json(raw)


class ScheduleValidation(unittest.TestCase):
    def test_hours_recomputed_from_blocks(self):
        r = sg._normalise(fake_week())
        self.assertEqual(r["weekly_focus_hours"], 14.0)  # 2h x 7 days, not the model's 999
        self.assertEqual(r["subject_breakdown"], [{"subject": "Data Structures", "hours_this_week": 14.0}])

    def test_incomplete_week_rejected_and_bad_values_coerced(self):
        short = fake_week()
        short["schedule"].pop()
        self.assertIsNone(sg._normalise(short))
        odd = fake_week()
        odd["schedule"][0]["blocks"][0].update(type="STUDYING", energy="extreme")
        b = sg._normalise(odd)["schedule"][0]["blocks"][0]
        self.assertEqual((b["type"], b["energy"]), ("other", "medium"))

    def test_overnight_range(self):
        self.assertEqual(sg._block_hours("23:00–7:00"), 8.0)
        self.assertIsNone(sg._block_hours("whenever"))


class EffortTiers(unittest.TestCase):
    def run_with(self, effort):
        with mock.patch.object(sg, "ask_json", return_value=fake_week()) as m:
            self.assertTrue(sg.generate_schedule([], [], {}, effort=effort)["success"])
        return m.call_args

    def test_budget_and_prompt_follow_effort(self):
        for effort in ("low", "medium", "high"):
            call = self.run_with(effort)
            self.assertEqual(call.kwargs["max_tokens"], llm_client.MAX_TOKENS_BY_EFFORT[effort])
            self.assertIs(call.args[0], sg.SYSTEM_PROMPT_BY_EFFORT[effort])

    def test_unknown_effort_means_medium(self):
        self.assertEqual(self.run_with("turbo").kwargs["max_tokens"], llm_client.MAX_TOKENS_BY_EFFORT["medium"])

    def test_truncation_retries_once_rate_limit_does_not(self):
        truncated = llm_client.LLMError("generation_failed", truncated=True)
        with mock.patch.object(sg, "ask_json", side_effect=[truncated, fake_week()]) as m:
            self.assertTrue(sg.generate_schedule([], [], {}, effort="high")["success"])
            self.assertEqual(m.call_count, 2)
            self.assertIs(m.call_args.args[0], sg.SYSTEM_PROMPT_BY_EFFORT["medium"])  # retried more compact
        with mock.patch.object(sg, "ask_json", side_effect=llm_client.LLMError("rate_limited")) as m:
            r = sg.generate_schedule([], [], {})
            self.assertEqual((r["success"], r["reason"], m.call_count), (False, "rate_limited", 1))


class Documents(unittest.TestCase):
    def test_xls_that_is_really_xlsx_is_read(self):
        import openpyxl
        wb = openpyxl.Workbook()
        wb.active.append(["Data Structures", 4])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "timetable.xls")
            wb.save(path)
            r = document_parser.parse_document(path)
        self.assertTrue(r.success, r.error)
        self.assertIn("Data Structures", r.text)

    def test_magic_bytes(self):
        self.assertTrue(document_parser.looks_like("pdf", b"%PDF-1.7"))
        self.assertFalse(document_parser.looks_like("pdf", b"PK\x03\x04"))
        self.assertTrue(document_parser.looks_like("webp", b"RIFF\x00\x00\x00\x00WEBPVP8 "))


class Security(unittest.TestCase):
    def test_production_requires_real_secret(self):
        for bad in (None, "", "short", "change-this-to-any-random-string"):
            with self.assertRaises(RuntimeError):
                app_module.resolve_secret("production", bad)
        self.assertEqual(app_module.resolve_secret("production", "x" * 32), "x" * 32)
        self.assertEqual(len(app_module.resolve_secret("development", None)), 64)


class Frontend(unittest.TestCase):
    def test_greeting_asks_the_first_checklist_question(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "script.js"), encoding="utf-8") as fh:
            js = fh.read()
        line = next(l for l in js.splitlines() if l.startswith("const GREETING"))
        self.assertIn(chat_handler.CHECKLIST[0]["question"], line)


class EndToEnd(unittest.TestCase):
    def setUp(self):
        app_module.limiter.enabled = False
        self.c = app_module.app.test_client()
        self.pipeline_calls = 0

    def chat_all(self, fake_ask_json):
        reply = llm_client.LLMReply("Nice. Next question!")
        with mock.patch.object(chat_handler, "ask", return_value=reply), \
             mock.patch.object(sg, "ask_json", side_effect=fake_ask_json):
            for i in range(len(chat_handler.CHECKLIST)):
                r = self.c.post("/api/chat", json={"message": f"answer {i}"})
                self.assertEqual(r.status_code, 200)
            return r.get_json()

    def test_full_flow_survives_a_truncated_first_reply(self):
        def fake(*a, **k):
            self.pipeline_calls += 1
            if self.pipeline_calls == 1:
                raise llm_client.LLMError("generation_failed", truncated=True)
            return fake_week()
        last = self.chat_all(fake)
        self.assertTrue(last["done"])
        s = self.c.get("/api/schedule").get_json()
        self.assertEqual((s["status"], len(s["schedule"]), s["weekly_focus_hours"]), ("ready", 7, 14.0))
        # a double-click / extra message after the chat ended must NOT re-run the pipeline
        before = self.pipeline_calls
        self.c.post("/api/chat", json={"message": "hello again"})
        self.assertEqual(self.pipeline_calls, before)

    def test_failure_is_reported_then_retry_recovers(self):
        self.chat_all(lambda *a, **k: (_ for _ in ()).throw(llm_client.LLMError("rate_limited")))
        s = self.c.get("/api/schedule").get_json()
        self.assertEqual((s["status"], s["reason"]), ("failed", "rate_limited"))
        with mock.patch.object(sg, "ask_json", return_value=fake_week()):
            self.assertEqual(self.c.post("/api/schedule/retry").get_json()["status"], "ready")

    def test_server_side_validation(self):
        big = self.c.post("/api/upload", data={"file": (io.BytesIO(b"%PDF" + b"0" * (22 * 1024 * 1024)), "x.pdf")},
                          content_type="multipart/form-data")
        self.assertEqual(big.status_code, 413)
        self.assertEqual(big.get_json()["status"], "failed")
        fake = self.c.post("/api/upload", data={"file": (io.BytesIO(b"not a pdf"), "x.pdf")}, content_type="multipart/form-data")
        self.assertEqual(fake.get_json()["status"], "failed")
        self.assertEqual(self.c.post("/api/chat", json={"message": "x" * 2000}).status_code, 400)
        r = self.c.post("/api/chat", json={"message": "hi"}, headers={"X-Gemini-Api-Key": "bad key!"})
        self.assertEqual(r.status_code, 400)

    def test_rate_limit_returns_429_with_retry_after(self):
        app_module.limiter.enabled = True
        app_module.limiter._hits.clear()
        codes = [self.c.post("/api/effort", json={"level": "low"}).status_code for _ in range(31)]
        self.assertEqual(codes[-1], 429)
        self.assertIn("Retry-After", self.c.post("/api/effort", json={"level": "low"}).headers)
        app_module.limiter.enabled = False


if __name__ == "__main__":
    unittest.main()
