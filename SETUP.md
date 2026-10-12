# Running the backend

## 1. System requirement (for OCR)
Image/photo document parsing needs the `tesseract` OCR engine installed as a
system program — the `pytesseract` Python package is only a wrapper around it.

- Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`
- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki

Without this, `.pdf`/`.docx`/`.xlsx` uploads still work fine — only image
uploads (`.png`/`.jpg`/`.jpeg`/`.webp`) will fail.

## 2. Python dependencies
```
cd backend
pip install -r requirements.txt
```

## 3. API keys
```
cp .env.example .env
```
Then open `.env` and fill in:
- `FLASK_SECRET_KEY` — any random string
- `GROQ_API_KEY` — free, from https://console.groq.com (used by chat_handler.py
  and syllabus_structurer.py by default)
- `GROQ_API_KEY` is the only LLM key required: chat_handler.py,
  syllabus_structurer.py and schedule_generator.py all have `PROVIDER = "groq"`
  near the top of the file, so Groq (free tier) is the default everywhere.
- `ANTHROPIC_API_KEY` — optional. Only needed if you switch a step to Claude
  by changing that file's `PROVIDER = "groq"` to `PROVIDER = "claude"` (one
  line per file). schedule_generator.py is the step where Claude is most worth
  it if Groq's free-tier quality or rate limits become a problem.
- `GEMINI_API_KEY` — optional, also free. Get one at https://aistudio.google.com
  and switch a file's `PROVIDER = "groq"` to `PROVIDER = "gemini"` the same way
  as Claude above. **This is a separate developer API key — it is not the same
  thing as a consumer Gemini app subscription** (e.g. the free "Google AI Pro"
  plan some phone carriers bundle in). That consumer plan gives you the Gemini
  chat app, not an API key, so it can't be pasted in here. schedule_generator.py
  is again the step most worth moving off Groq: Gemini's free-tier tokens-per-
  minute limit is roughly 30x Groq's free-tier limit on `openai/gpt-oss-120b`,
  which is the actual cause of Groq cutting a full-detail schedule response off
  mid-JSON (not the old flat `max_tokens=3000`, which was already raised — see
  the Effort tiers section below and that file's module docstring).
- `DEFAULT_LLM_PROVIDER` — only used by `llm_client.ask()` when a caller passes
  no provider. All three scripts pass one explicitly, so changing this value
  does not by itself switch them. Accepts `groq`, `claude`, or `gemini`.

## 3b. Schedule detail (Low/Medium/High) and bring-your-own Claude key
- The chat page has a Low/Medium/High "Schedule detail" selector. It changes
  how much per-block detail `schedule_generator.py` asks the LLM for and its
  `max_tokens` budget — Low=2000, Medium/High=5000 on Groq (see that file's
  module docstring for the full reasoning; this replaced a single flat
  `max_tokens=3000` that was truncating full-detail responses mid-JSON).
  Stored server-side per session (`POST /api/effort`), defaults to Medium.
- Also on the chat page: an opt-in "Use my own AI API key" control — a
  dropdown (Off / Claude / Gemini) rather than a single checkbox, so either
  provider's key can be pasted in. When a provider is selected, that
  request's chat/structuring/schedule-generation calls use the pasted key
  instead of the shared keys — sent as the `X-Anthropic-Api-Key` header
  (Claude) or `X-Gemini-Api-Key` header (Gemini) on `/api/chat` and
  `/api/schedule/retry`. Both providers' typed-in keys are kept in the
  browser's `sessionStorage` only (cleared when the tab closes), never
  written to `SESSIONS`, a log, or disk; if a request somehow carries both
  headers, Claude wins (see `llm_client.ask()`). On this path, the SAME
  Low/Medium/High selector picks a model tier instead of a Groq token tier:
  - Claude: Low→`claude-haiku-4-5-20251001`, Medium→`claude-sonnet-5-5`,
    High→`claude-opus-5-5` (`llm_client.py`'s `CLAUDE_MODEL_BY_EFFORT`), each
    with a materially higher `max_tokens` (4000/8000/16000) than the Groq tiers.
  - Gemini: all three tiers currently use `gemini-2.5-flash` (the model
    Google keeps on a real free tier — Pro's free allowance is minimal/trial
    only) with `max_tokens` 3000/6000/10000 (`GEMINI_MODEL_BY_EFFORT` /
    `GEMINI_MAX_TOKENS_BY_EFFORT`) — generous headroom under Gemini's free-tier
    TPM ceiling, not an attempt to approach it in one call.

## 4. Run it
```
python app.py
```
Starts on http://localhost:5000 and serves the frontend directly (landing
page at `/`), so you don't need a separate static file server for local
testing/demo.

## 5. Switch the frontend off mock mode
Open `frontend/script.js` and change the line near the top:
```js
const USE_MOCK = true;
```
to
```js
const USE_MOCK = false;
```
Without this, the frontend will keep showing fake scripted responses instead
of talking to the real backend.

## What each script does (quick reference)
| File | Job | Calls an LLM? |
|---|---|---|
| `llm_client.py` | Shared Claude/Groq/Gemini client, used by the 3 scripts below | — |
| `chat_handler.py` | Asks the fixed checklist of questions | Yes (Groq) |
| `document_parser.py` | Extracts text from PDF/Word/Excel/image uploads | No — plain code + OCR |
| `syllabus_structurer.py` | Turns messy extracted text into structured subject data | Yes (Groq) |
| `schedule_generator.py` | Builds the actual weekly schedule (circadian rhythm reasoning) | Yes (Groq by default; Claude or Gemini optional) |
| `app.py` | Flask routes tying it all together, matches the frontend's exact API contract | — |

## API contract note: `/api/schedule` response shape
As of the BioSync analytics upgrade, `GET /api/schedule` returns one JSON
object (not a bare array):
```json
{
  "schedule": [ { "day": "Monday", "focus_summary": "...", "blocks": [
      { "time": "7:00–8:00", "activity": "...", "type": "study",
        "subject": "Data Structures", "energy": "high", "note": "..." }
  ] } ],
  "reasoning_notes": "...",
  "weekly_focus_hours": 16,
  "subject_breakdown": [ { "subject": "Data Structures", "hours_this_week": 4 } ]
}
```
The object also carries `status` (`"pending"` = chat not finished,
`"ready"`, or `"failed"`), plus `error` and `reason` when `status` is
`"failed"`. `reason` is `rate_limited` (the LLM provider's rate limit was hit),
`generation_failed` (any other LLM failure or unreadable reply), or
`internal_error` (unexpected backend crash); `error` is a user-safe message —
the raw provider error is only written to the server log. A failure is still
HTTP 200: the signal is in the body. `POST /api/schedule/retry` re-runs the
pipeline after a failure and returns the same shape.

`type` is one of `study` / `break` / `meal` / `sleep` / `class` / `other`,
and `energy` is `high` / `medium` / `low` — both come straight from
`schedule_generator.py`'s prompt, not computed separately. The results
page (`frontend/results.html` + `script.js`) uses these fields to draw
the Analytics tab (subject-hours chart, focus/break donut, energy
heatmap) and the per-block reasoning notes in the Weekly view. The
frontend still tolerates the old bare-array shape as a fallback, so nothing
breaks if an older backend build is swapped in.

## Known limitations (be upfront about these, don't let them surprise you mid-demo)
- State is in-memory only — restarting the Flask server loses all in-progress
  sessions. No database in this MVP.
- OCR on scanned/photographed documents is not guaranteed to be perfectly
  accurate — test with a real sample of what you plan to demo with, ahead of time.
- `document_parser.py` does not OCR-fallback for scanned PDFs with no text
  layer (only for standalone image files) — a scanned PDF syllabus may come
  back empty.
- Legacy `.doc` (not `.docx`) files are not supported — only `.docx`.
  (Legacy `.xls` *is* supported, via `xlrd`.)
- Groq's free tier is rate-limited (notably its tokens-per-minute cap, well
  below Gemini's free tier). When it rejects a request the results page says
  so and offers a retry, plus a clearly labeled example schedule. The example
  is also available any time at `results.html?example=1`. Switching
  `schedule_generator.py`'s `PROVIDER` to `"gemini"`, or using the chat page's
  BYO-key dropdown with a free Gemini key, is the most direct fix.
- The Low effort tier is the one most likely to reliably finish on Groq's
  free tier; High is opt-in and still the most likely of the three to run
  long on Groq (it asks for the most detail) — that's expected, not a bug.
