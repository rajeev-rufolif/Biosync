# BioSync setup

## Run locally
```
cd backend
pip install -r requirements.txt
cp .env.example .env        # add GEMINI_API_KEY and/or ANTHROPIC_API_KEY
python app.py               # http://localhost:5000
python -m unittest discover -s tests -v    # 16 offline tests, no API key needed
```
Photo/image uploads also need the `tesseract` program (`apt install tesseract-ocr`, `brew install tesseract`,
or the UB-Mannheim Windows installer). PDF/Word/Excel work without it.

## Environment variables (all of them)
| Variable | Needed | Purpose |
|---|---|---|
| `APP_ENV` | no (default `production`) | `development` locally. In production the app refuses to start without a real `FLASK_SECRET_KEY`. |
| `FLASK_SECRET_KEY` | production | Signs the session cookie. |
| `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | at least one | Shared server keys. Students can also paste their own on the chat page. |
| `LLM_PROVIDER` | no | `gemini` or `claude`; used when both keys exist. Otherwise the first key found wins. |
| `GEMINI_MODEL` | no (default `gemini-3.5-flash`) | Google retires models on a schedule (`gemini-2.5-flash` was due to shut down 2026-10-16). Verify the model is on your key's free tier in AI Studio. |
| `COOKIE_SECURE`, `TRUSTED_PROXY_HOPS` | deploy | `1` and `1` behind Render's HTTPS proxy (set in `render.yaml`). |
| `PORT`, `FLASK_DEBUG` | no | Port (default 5000); debug is ignored in production. |

## Deploy (Render)
Push the repo, create a Blueprint from `render.yaml`, set `GEMINI_API_KEY` (and optionally `ANTHROPIC_API_KEY`).
Run a single instance: sessions are in memory, so a second instance would not see the first one's sessions.

## How it works
`chat_handler` (7 fixed questions) -> `document_parser` (plain code + OCR) -> `syllabus_structurer` (LLM, cheapest model)
-> `schedule_generator` (LLM, then validated in code). `llm_client` is the only file that talks to a provider.

- **Schedule detail** (Low/Medium/High) changes how much per-block reasoning is requested. With a pasted Claude key it
  also picks Haiku/Sonnet/Opus for the schedule step only. Chat and file reading always use the cheapest tier.
- **Reliability:** every reply is validated (7 days, allowed block types, energy levels). Weekly hours and the subject
  breakdown are recomputed from block times, not trusted from the model. A truncated or incomplete reply gets one retry
  with a more compact prompt. Rate limits and rejected keys are reported honestly and not retried.
- **Limits enforced server-side:** 20 MB per file, 8 files per session, file-signature check, 40 PDF pages, 1,500
  characters per chat message, per-IP rate limits (chat 30/min, upload 10/min, retry 6/min).
- **Privacy:** uploaded files are deleted right after text extraction. Pasted API keys are used for one request and never
  stored or logged. Sessions expire after 6 hours of inactivity.

## `GET /api/schedule` shape
```json
{"status": "pending|ready|failed", "schedule": [{"day": "Monday", "focus_summary": "", "blocks": [
  {"time": "7:00–8:00", "activity": "", "type": "study|break|meal|sleep|class|other", "subject": "", "energy": "high|medium|low", "note": ""}]}],
 "reasoning_notes": "", "weekly_focus_hours": 16, "subject_breakdown": [{"subject": "", "hours_this_week": 4}],
 "error": null, "reason": null, "warnings": []}
```
`reason` when failed: `rate_limited`, `invalid_api_key`, `not_configured`, `generation_failed`, `internal_error`.
`warnings` is new and additive (e.g. uploaded files could not be read). A failure is still HTTP 200.

## Known limitations
- Sessions are in memory: a restart clears in-progress chats (deliberate, see below).
- The pipeline runs inside the final chat request (typically 20–40 s). A host with a short proxy timeout would cut it off.
- Scanned PDFs with no text layer come back empty (OCR only runs on image files). Legacy `.doc` is not supported.
