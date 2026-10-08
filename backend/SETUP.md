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
- `ANTHROPIC_API_KEY` — only needed because schedule_generator.py defaults to
  Claude (this is the one step where we deliberately chose quality over cost —
  see that file's docstring). If you don't have a Claude key, open
  schedule_generator.py and change `PROVIDER = "claude"` to `PROVIDER = "groq"`
  near the top of the file — one line, no other changes needed.

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
| `llm_client.py` | Shared Claude/Groq client, used by the 3 scripts below | — |
| `chat_handler.py` | Asks the fixed checklist of questions | Yes (Groq) |
| `document_parser.py` | Extracts text from PDF/Word/Excel/image uploads | No — plain code + OCR |
| `syllabus_structurer.py` | Turns messy extracted text into structured subject data | Yes (Groq) |
| `schedule_generator.py` | Builds the actual weekly schedule (circadian rhythm reasoning) | Yes (Claude by default) |
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
