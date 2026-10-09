/* ===== Config ===== */
// Set to false once the Flask backend exposes /api/chat, /api/upload, /api/schedule.
const USE_MOCK = false;
const STREAK_KEY = 'biosync.streak.v1'; // localStorage: { count, last: 'YYYY-MM-DD' }
const BYOK_KEY = 'biosync.byok.anthropic.v1'; // sessionStorage (Task 8 — NOT localStorage, on purpose): { enabled, key }
const OK_EXT = ["pdf", "doc", "docx", "xls", "xlsx", "png", "jpg", "jpeg", "webp"];
const MAX_MB = 20;

const sleep = ms => new Promise(r => setTimeout(r, ms));
const $ = id => document.getElementById(id);

/* ===== Icons & mascot ===== */
const ICON = {
  send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2 11 13M22 2l-7 20-4-9-9-4z"/></svg>',
  clip: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m21 11-9 9a6 6 0 0 1-8.5-8.5l9-9a4 4 0 0 1 5.7 5.7l-9 9a2 2 0 0 1-2.8-2.8l8.3-8.3"/></svg>',
  flame: '<svg viewBox="0 0 24 24"><path d="M12 2c1 4 5 6 5 11a5 5 0 0 1-10 0c0-2 1-3 2-4 0 2 1 3 2 3-1-4 0-7 1-10z" fill="#FF7A1A" stroke="#1B2340" stroke-width="1.6" stroke-linejoin="round"/></svg>',
  bulb: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3z"/><path d="M19 11a7 7 0 0 1-14 0M12 18v4"/></svg>',
  clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
  bolt: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h7l-1 8 10-12h-7l1-8z"/></svg>',
  calendar: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>'
};
function mascot() {
  return `<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <g stroke="#1B2340" stroke-width="5" stroke-linecap="round">
      <circle cx="58" cy="32" r="16" fill="#FFE066"/><circle cx="142" cy="32" r="16" fill="#FFE066"/>
      <ellipse cx="80" cy="176" rx="12" ry="8" fill="#FFE066"/><ellipse cx="120" cy="176" rx="12" ry="8" fill="#FFE066"/>
      <circle cx="100" cy="104" r="68" fill="#fff"/>
      <path d="M100 44v8M100 156v8M40 104h8M152 104h8" fill="none"/>
      <path d="M76 126q24 20 48 0" fill="none"/>
    </g>
    <circle cx="80" cy="96" r="7" fill="#1B2340"/><circle cx="120" cy="96" r="7" fill="#1B2340"/>
    <circle cx="66" cy="114" r="7" fill="#FFB4A8" opacity=".8"/><circle cx="134" cy="114" r="7" fill="#FFB4A8" opacity=".8"/>
  </svg>`;
}
document.querySelectorAll('[data-mascot]').forEach(el => el.innerHTML = mascot());

/* ===== BYO Claude key (Task 8) =====
   Lives here (not inside initChat) because retrySchedule() on results.html also
   needs to send it — sessionStorage is shared across pages of the same tab, so
   this works even though the toggle/input only exist in chat.html's DOM. Storage
   is a single JSON object (enabled + key together) specifically so results.html
   can tell "opted in" from "opted out" without needing chat.html's checkbox. */
function readByok() {
  try {
    const o = JSON.parse(sessionStorage.getItem(BYOK_KEY) || 'null');
    if (o && typeof o.enabled === 'boolean' && typeof o.key === 'string') return o;
  } catch (e) { /* storage unavailable or corrupt — treat as opted out */ }
  return { enabled: false, key: '' };
}
function writeByok(o) {
  try { sessionStorage.setItem(BYOK_KEY, JSON.stringify(o)); } catch (e) { /* in-memory only for this page load */ }
}
// Only returns a key when the user has actually opted in AND pasted something —
// never inferred from storage alone, so a key left over from an unchecked toggle
// is never silently sent.
function getUserApiKey() {
  const o = readByok();
  return (o.enabled && o.key.trim()) ? o.key.trim() : null;
}
function authHeaders() {
  const key = getUserApiKey();
  return key ? { 'X-Anthropic-Api-Key': key } : {};
}

/* ===== API (mocked until backend exists) ===== */
const GREETING = "Hi! I'm BioSync. Let's build your week. What courses are you taking this term?";
const MOCK_REPLIES = [
  "Nice. Any exams or big deadlines coming up in the next few weeks?",
  "When do you focus best: morning, afternoon, or late evening?",
  "Last one: how many hours a day can you realistically study?"
];
let mockStep = 0;

async function sendMessage(text) {
  if (USE_MOCK) {
    await sleep(900 + Math.random() * 600);
    const done = mockStep >= MOCK_REPLIES.length;
    const reply = done
      ? "Thanks, that's everything I need. Your schedule is ready."
      : MOCK_REPLIES[mockStep++];
    return { reply, done };
  }
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ message: text })
  });
  if (!res.ok) throw new Error('chat ' + res.status);
  const data = await res.json();
  // data.reply = the AI's text response to show
  // data.done = true once enough info is collected
  return data;
}

async function uploadFile(file) {
  if (USE_MOCK) { await sleep(1000); return { status: 'received' }; }
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch('/api/upload', { method: 'POST', body: formData });
  if (!res.ok) throw new Error('upload ' + res.status);
  return await res.json(); // { status: "received" }
}

/* ---- Mock schedule: matches the extended /api/schedule response shape ----
   { schedule: [{day, focus_summary, blocks:[{time, activity, type, subject,
     energy, note}]}], reasoning_notes, weekly_focus_hours, subject_breakdown } */
const MOCK_DAYS = [
  ['Monday', "Data Structures gets the morning peak window before the 1–3pm dip.", [
    ['7:00–8:00', 'Light breakfast + review notes', 'meal', '', 'medium', 'Light meal keeps energy steady going into the first focus block.'],
    ['8:00–10:00', 'Deep focus: Data Structures', 'study', 'Data Structures', 'high', "Placed in your stated morning peak, before the early-afternoon dip."],
    ['10:00–10:30', 'Break + walk', 'break', '', 'medium', 'Short reset between two demanding blocks.'],
    ['13:00–15:00', 'Calculus problem set', 'study', 'Calculus', 'low', 'Lighter problem-set work scheduled through the natural post-lunch dip.']
  ]],
  ['Tuesday', "Lecture sits in the mid-morning window; essay writing gets your second peak.", [
    ['7:30–8:30', 'Breakfast + plan the day', 'meal', '', 'medium', ''],
    ['9:00–11:00', 'Lecture: Intro to Psychology', 'class', 'Psychology', 'medium', 'Fixed class time.'],
    ['14:00–16:00', 'Deep focus: Essay draft', 'study', 'Essay Writing', 'high', 'Your focus survey flagged early afternoon as a secondary peak.'],
    ['19:00–20:00', 'Flashcards review', 'study', 'Psychology', 'medium', 'Light review pass before winding down for sleep.']
  ]],
  ['Wednesday', "Heaviest day — Data Structures exam is 5 days out, so it gets two blocks.", [
    ['8:00–10:00', 'Deep focus: Data Structures', 'study', 'Data Structures', 'high', 'Exam in 5 days — extra block added ahead of the deadline.'],
    ['10:30–12:00', 'Lab: Physics', 'class', 'Physics', 'medium', 'Fixed lab time.'],
    ['13:00–14:00', 'Lunch + rest', 'meal', '', 'low', 'Heavier midday meal — no focus work scheduled right after.'],
    ['15:00–17:00', 'Calculus practice exam', 'study', 'Calculus', 'medium', 'Afternoon block kept lighter given the post-lunch dip.']
  ]],
  ['Thursday', "Lighter academic load — used to recover before Friday's push.", [
    ['7:30–8:30', 'Breakfast + review notes', 'meal', '', 'medium', ''],
    ['9:00–11:00', 'Lecture: Intro to Psychology', 'class', 'Psychology', 'medium', 'Fixed class time.'],
    ['16:00–18:00', 'Group project meeting', 'other', '', 'medium', 'Scheduled around your stated availability window.']
  ]],
  ['Friday', "Essay revision gets the morning peak; office hours placed right after for immediate follow-up.", [
    ['8:00–10:00', 'Deep focus: Essay revision', 'study', 'Essay Writing', 'high', 'Morning peak window, same as Monday.'],
    ['10:30–12:00', 'Office hours: Data Structures', 'class', 'Data Structures', 'medium', 'Right after focus block while questions are fresh.'],
    ['15:00–16:00', 'Weekly review + plan next week', 'other', '', 'low', 'Low-energy admin task placed in the afternoon dip.']
  ]],
  ['Saturday', "Rest day — only light optional catch-up, no scheduled deep focus.", [
    ['10:00–12:00', 'Catch-up reading', 'study', 'General', 'medium', 'Optional, flexible — not counted against your weekday study load.'],
    ['14:00–16:00', 'Free time', 'break', '', 'medium', 'Deliberately unscheduled — recovery matters for next week\u2019s focus.']
  ]],
  ['Sunday', "Light review only — protects your stated realistic weekly load.", [
    ['11:00–12:00', 'Light review: all courses', 'study', 'General', 'medium', 'Keeps momentum without adding real load before the week resets.'],
    ['17:00–18:00', 'Set up next week', 'other', '', 'low', 'Quick planning pass, not counted as study time.']
  ]]
];
const MOCK_SCHEDULE = MOCK_DAYS.map(([day, focus_summary, blocks]) => ({
  day, focus_summary,
  blocks: blocks.map(([time, activity, type, subject, energy, note]) => ({ time, activity, type, subject, energy, note }))
}));
const MOCK_REASONING = "Data Structures got priority placement because it has the nearest exam (5 days out) and you said mornings are your strongest focus window. I kept Calculus work in lighter afternoon slots since you mentioned a post-lunch energy dip, and left Saturday mostly open since you said 4–5 hours/day is your realistic ceiling — Wednesday already hits that with the extra exam-prep block.";
// Derived from MOCK_SCHEDULE itself (same way the real backend derives
// subject_breakdown from the schedule it just generated) so the mock
// numbers can never drift out of sync with what's actually rendered.
function deriveSubjectBreakdown(schedule) {
  const totals = {};
  schedule.forEach(d => (d.blocks || []).forEach(b => {
    if (b.type === 'study' && b.subject) {
      const parts = String(b.time).split(/[–-]/);
      const [h1, m1] = parts[0].split(':').map(Number);
      const [h2, m2] = parts[1].split(':').map(Number);
      const hrs = ((h2 * 60 + m2) - (h1 * 60 + m1)) / 60;
      totals[b.subject] = (totals[b.subject] || 0) + hrs;
    }
  }));
  return Object.entries(totals).map(([subject, hours_this_week]) => ({ subject, hours_this_week: Math.round(hours_this_week * 10) / 10 }));
}
const MOCK_SUBJECT_BREAKDOWN = deriveSubjectBreakdown(MOCK_SCHEDULE);
const MOCK_WEEKLY_FOCUS_HOURS = MOCK_SUBJECT_BREAKDOWN.reduce((s, x) => s + x.hours_this_week, 0);

// The "example schedule" view (results.html?example=1, and the labeled fallback
// shown when the backend reports a failed schedule). Deliberately NOT gated on
// USE_MOCK: USE_MOCK only decides real-vs-fake API calls; this is static sample
// data that is always available and always labeled as an example.
function exampleScheduleData() {
  return {
    schedule: MOCK_SCHEDULE,
    reasoning_notes: MOCK_REASONING,
    weekly_focus_hours: MOCK_WEEKLY_FOCUS_HOURS,
    subject_breakdown: MOCK_SUBJECT_BREAKDOWN
  };
}

async function fetchSchedule() {
  if (USE_MOCK) {
    await sleep(700);
    return {
      schedule: MOCK_SCHEDULE,
      reasoning_notes: MOCK_REASONING,
      weekly_focus_hours: MOCK_WEEKLY_FOCUS_HOURS,
      subject_breakdown: MOCK_SUBJECT_BREAKDOWN
    };
  }
  const res = await fetch('/api/schedule');
  if (!res.ok) throw new Error('schedule ' + res.status);
  // { status: "pending" | "ready" | "failed", schedule, ..., error, reason }
  return await res.json();
}

// Re-runs schedule generation after a failure (e.g. rate limit). Same response shape as fetchSchedule().
async function retrySchedule() {
  if (USE_MOCK) return fetchSchedule();
  const res = await fetch('/api/schedule/retry', { method: 'POST', headers: authHeaders() });
  if (!res.ok) throw new Error('schedule retry ' + res.status);
  return await res.json();
}

/* ===== Chat page: effort selector (Task 7) ===== */
// Lives before/alongside the checklist (visible for the whole chat, not a
// checklist question) so it can be changed any time before the pipeline runs.
// Server already defaults new sessions to "medium", so this only needs to POST
// on an actual change — no need to sync a value on page load.
function setEffort(level, group) {
  (group || $('effortTabs'))?.querySelectorAll('button').forEach(b => {
    const active = b.dataset.effort === level;
    b.classList.toggle('active', active);
    b.setAttribute('aria-pressed', String(active));
  });
  if (!USE_MOCK) {
    fetch('/api/effort', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level })
    }).catch(() => { /* best-effort — server falls back to "medium" if this never lands */ });
  }
}
function initEffortSelector() {
  const group = $('effortTabs');
  if (!group) return;
  group.querySelectorAll('button').forEach(b => {
    b.addEventListener('click', () => setEffort(b.dataset.effort, group));
  });
}

/* ===== Chat page: BYO Claude key opt-in (Task 8) ===== */
// Pre-fills from sessionStorage so a mid-chat page refresh doesn't force a
// re-paste; writing happens on every toggle/input change.
function initByok() {
  const toggle = $('byokToggle'), panel = $('byokPanel'), input = $('byokInput');
  if (!toggle || !panel || !input) return;
  const saved = readByok();
  toggle.checked = saved.enabled;
  input.value = saved.key;
  panel.hidden = !saved.enabled;

  function sync() {
    writeByok({ enabled: toggle.checked, key: input.value });
    panel.hidden = !toggle.checked;
  }
  toggle.addEventListener('change', sync);
  input.addEventListener('input', sync);
}

/* ===== Chat page ===== */
function initChat() {
  const box = $('messages'), input = $('input'), sendBtn = $('send');
  $('attach').innerHTML = ICON.clip;
  sendBtn.innerHTML = ICON.send;
  initEffortSelector();
  initByok();
  let busy = false, finished = false;

  const scroll = () => box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' });

  function addRow(role, node) {
    const row = document.createElement('div');
    row.className = 'row ' + role;
    if (role === 'ai') {
      const av = document.createElement('div');
      av.className = 'avatar'; av.innerHTML = mascot(); av.setAttribute('aria-hidden', 'true');
      row.append(av);
    }
    row.append(node);
    box.append(row); scroll();
    return row;
  }
  function addMsg(role, text) {
    const b = document.createElement('div');
    b.className = 'bubble'; b.textContent = text;
    return addRow(role, b);
  }
  function addFile(name) {
    const b = document.createElement('div');
    b.className = 'bubble file'; b.innerHTML = ICON.clip;
    b.append(document.createTextNode(name));
    return addRow('user', b);
  }
  function showTyping() {
    const t = document.createElement('div');
    t.className = 'bubble typing'; t.innerHTML = '<i></i><i></i><i></i>';
    return addRow('ai', t);
  }
  function showDone() {
    const a = document.createElement('a');
    a.className = 'btn done-cta'; a.href = 'results.html'; a.textContent = 'See my schedule';
    box.append(a); scroll();
    input.disabled = true; input.placeholder = 'All done!';
  }
  function setBusy(v) { busy = v; sendBtn.disabled = v || finished; }

  async function handleSend() {
    const text = input.value.trim();
    if (!text || busy || finished) return;
    input.value = ''; input.style.height = 'auto';
    addMsg('user', text);
    setBusy(true);
    const typing = showTyping();
    try {
      const data = await sendMessage(text);
      typing.remove();
      addMsg('ai', data.reply);
      if (data.done) { finished = true; showDone(); }
    } catch (e) {
      typing.remove();
      addMsg('ai', "I couldn't reach the server. Check your connection and send that again.");
    }
    setBusy(false); if (!finished) input.focus();
  }

  async function handleFiles(files) {
    for (const file of files) {
      addFile(file.name);
      const ext = file.name.split('.').pop().toLowerCase();
      if (!OK_EXT.includes(ext)) { addMsg('ai', `I can't use ${file.name}. Upload a PDF, Word, Excel, or image file.`); continue; }
      if (file.size > MAX_MB * 1024 * 1024) { addMsg('ai', `${file.name} is over ${MAX_MB} MB. Upload a smaller file.`); continue; }
      const typing = showTyping();
      try {
        const r = await uploadFile(file);
        typing.remove();
        addMsg('ai', r.status === 'received'
          ? `Got ${file.name}. I'll use it when building your schedule.`
          : `I couldn't read ${file.name}. ${r.error ? String(r.error).slice(0, 200) : 'Try a PDF, Word, Excel, or image file.'}`);
      } catch (e) {
        typing.remove();
        addMsg('ai', `Upload of ${file.name} failed. Try again in a moment.`);
      }
    }
  }

  $('composer').addEventListener('submit', e => { e.preventDefault(); handleSend(); });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto'; input.style.height = (input.scrollHeight + input.offsetHeight - input.clientHeight) + 'px';
  });
  $('attach').addEventListener('click', () => $('file').click());
  $('file').addEventListener('change', e => { handleFiles([...e.target.files]); e.target.value = ''; });

  // drop files anywhere on the page (also stops the browser from navigating to a dropped file)
  let drags = 0;
  const hasFiles = e => e.dataTransfer && [...e.dataTransfer.types].includes('Files');
  document.addEventListener('dragenter', e => { if (hasFiles(e)) { e.preventDefault(); drags++; document.body.classList.add('dragging'); } });
  document.addEventListener('dragover', e => { if (hasFiles(e)) e.preventDefault(); });
  document.addEventListener('dragleave', e => { if (hasFiles(e) && --drags <= 0) { drags = 0; document.body.classList.remove('dragging'); } });
  document.addEventListener('drop', e => {
    if (!hasFiles(e)) return;
    e.preventDefault(); drags = 0; document.body.classList.remove('dragging');
    handleFiles([...e.dataTransfer.files]);
  });

  // opening message
  (async () => {
    const t = showTyping(); await sleep(700); t.remove();
    addMsg('ai', GREETING);
  })();
}

/* ===== Results page: helpers ===== */
const DAY_ORDER = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'];
const PALETTE = ['#4B4FE8', '#9A9DF2', '#FFB4A8', '#FFE066', '#6FCF97', '#56B4D3', '#B79CED'];

function timeToMinutes(t) {
  if (!t) return null;
  const start = String(t).split(/[–-]/)[0].trim();
  const m = start.match(/(\d{1,2}):(\d{2})/);
  if (!m) return null;
  return parseInt(m[1], 10) * 60 + parseInt(m[2], 10);
}
function durationHours(time) {
  if (!time) return 0;
  const parts = String(time).split(/[–-]/);
  if (parts.length < 2) return 0;
  const a = timeToMinutes(parts[0]), bRaw = parts[1].trim();
  const bm = bRaw.match(/(\d{1,2}):(\d{2})/);
  if (a == null || !bm) return 0;
  let b = parseInt(bm[1], 10) * 60 + parseInt(bm[2], 10);
  if (b <= a) b += 24 * 60; // overnight block, rare but don't break on it
  return Math.round(((b - a) / 60) * 100) / 100;
}
function computeStats(schedule, subjectBreakdown, weeklyFocusHours) {
  let studyMinutes = 0, breakMinutes = 0, totalBlocks = 0;
  let busiestDay = null, busiestMinutes = -1;
  const bySubject = {};
  schedule.forEach(d => {
    let dayMinutes = 0;
    (d.blocks || []).forEach(b => {
      totalBlocks++;
      const mins = durationHours(b.time) * 60;
      if (b.type === 'study') { studyMinutes += mins; dayMinutes += mins; }
      if (b.type === 'break') breakMinutes += mins;
      if (b.subject) bySubject[b.subject] = (bySubject[b.subject] || 0) + (b.type === 'study' ? mins / 60 : 0);
    });
    if (dayMinutes > busiestMinutes) { busiestMinutes = dayMinutes; busiestDay = d.day; }
  });
  const subjects = (subjectBreakdown && subjectBreakdown.length)
    ? subjectBreakdown
    : Object.entries(bySubject).map(([subject, hours_this_week]) => ({ subject, hours_this_week: Math.round(hours_this_week * 10) / 10 }));
  const totalFocus = (typeof weeklyFocusHours === 'number' && weeklyFocusHours > 0)
    ? weeklyFocusHours
    : Math.round((studyMinutes / 60) * 10) / 10;
  return {
    totalFocusHours: totalFocus,
    breakHours: Math.round((breakMinutes / 60) * 10) / 10,
    subjectCount: subjects.length,
    busiestDay,
    subjects: subjects.sort((a, b) => b.hours_this_week - a.hours_this_week),
    totalBlocks
  };
}

function statCard(label, value, sub) {
  const div = document.createElement('div');
  div.className = 'stat-card';
  div.innerHTML = `<div class="stat-label">${label}</div><div class="stat-value">${value}</div>${sub ? `<div class="stat-sub">${sub}</div>` : ''}`;
  return div;
}

function renderStats(container, stats) {
  container.innerHTML = '';
  container.append(
    statCard('Weekly focus time', `${stats.totalFocusHours}h`, 'across all subjects'),
    statCard('Subjects covered', stats.subjectCount, stats.subjectCount ? 'from your synced syllabus' : 'none yet'),
    statCard('Busiest day', stats.busiestDay || '—', stats.busiestDay ? 'most focus time scheduled' : ''),
    statCard('Break &amp; rest time', `${stats.breakHours}h`, 'scheduled this week')
  );
}

function renderSubjectBars(container, subjects) {
  container.innerHTML = '';
  if (!subjects.length) { container.innerHTML = '<p class="state" style="padding:20px 0">No subject data yet.</p>'; return; }
  const max = Math.max(...subjects.map(s => s.hours_this_week), 1);
  subjects.forEach((s, i) => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    const pct = Math.max(4, Math.round((s.hours_this_week / max) * 100));
    row.innerHTML = `
      <div class="bar-label" title="${s.subject}">${s.subject}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${PALETTE[i % PALETTE.length]}"></div></div>
      <div class="bar-val">${s.hours_this_week}h</div>`;
    container.append(row);
  });
}

function renderTypeDonut(wrapEl, legendEl, schedule) {
  const totals = { study: 0, class: 0, meal: 0, break: 0, other: 0 };
  schedule.forEach(d => (d.blocks || []).forEach(b => {
    const key = totals.hasOwnProperty(b.type) ? b.type : 'other';
    totals[key] += durationHours(b.time);
  }));
  const labels = { study: 'Deep focus', class: 'Class / lab', meal: 'Meals', break: 'Breaks', other: 'Other' };
  const colors = { study: '#4B4FE8', class: '#9A9DF2', meal: '#FFB4A8', break: '#6FCF97', other: '#D5DAEA' };
  const entries = Object.entries(totals).filter(([, v]) => v > 0);
  const sum = entries.reduce((s, [, v]) => s + v, 0) || 1;

  const r = 54, cx = 64, cy = 64, circumference = 2 * Math.PI * r;
  let offset = 0;
  const segs = entries.map(([key, val]) => {
    const frac = val / sum;
    const dash = frac * circumference;
    const seg = `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${colors[key]}" stroke-width="18" stroke-dasharray="${dash} ${circumference - dash}" stroke-dashoffset="${-offset}" transform="rotate(-90 ${cx} ${cy})"/>`;
    offset += dash;
    return seg;
  }).join('');
  wrapEl.innerHTML = `<svg class="donut" width="128" height="128" viewBox="0 0 128 128">${segs}<circle cx="${cx}" cy="${cy}" r="34" fill="var(--card)"/></svg>`;

  legendEl.innerHTML = '';
  entries.forEach(([key, val]) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-swatch" style="background:${colors[key]}"></span>${labels[key]} <b style="margin-left:auto">${Math.round(val * 10) / 10}h</b>`;
    legendEl.append(item);
  });
}

function renderHeatmap(container, schedule) {
  const dayAbbrev = d => String(d || '').slice(0, 3);
  const slots = [
    { label: '6–9am', from: 6 * 60, to: 9 * 60 },
    { label: '9–12pm', from: 9 * 60, to: 12 * 60 },
    { label: '12–3pm', from: 12 * 60, to: 15 * 60 },
    { label: '3–6pm', from: 15 * 60, to: 18 * 60 },
    { label: '6–9pm', from: 18 * 60, to: 21 * 60 },
    { label: '9pm+', from: 21 * 60, to: 24 * 60 + 180 }
  ];
  const ENERGY_RANK = { high: 3, medium: 2, low: 1 };

  const grid = {};
  schedule.forEach(d => {
    const key = dayAbbrev(d.day).toLowerCase();
    grid[key] = {};
    (d.blocks || []).forEach(b => {
      const start = timeToMinutes(b.time);
      if (start == null) return;
      const slot = slots.find(s => start >= s.from && start < s.to);
      if (!slot) return;
      const rank = ENERGY_RANK[b.energy] || 0;
      if (!grid[key][slot.label] || rank > ENERGY_RANK[grid[key][slot.label]]) {
        grid[key][slot.label] = b.energy || null;
      }
    });
  });

  let html = `<div class="hm-corner"></div>`;
  DAY_ORDER.forEach(d => html += `<div class="hm-day">${d[0].toUpperCase() + d[1]}</div>`);
  slots.forEach(slot => {
    html += `<div class="hm-label">${slot.label}</div>`;
    DAY_ORDER.forEach(d => {
      const energy = (grid[d] || {})[slot.label];
      const cls = energy ? `e-${energy}` : 'e-none';
      html += `<div class="hm-cell ${cls}" title="${d.toUpperCase()} ${slot.label}${energy ? ' — ' + energy + ' energy' : ''}"></div>`;
    });
  });
  container.innerHTML = html;
}

function renderReasoning(container, notes) {
  if (!notes) { container.style.display = 'none'; return; }
  container.style.display = 'flex';
  container.innerHTML = `${ICON.bulb}<p><strong>Why your week looks like this</strong>${notes}</p>`;
}

function renderWeek(container, schedule) {
  const todayKey = new Date().toLocaleDateString('en-US', { weekday: 'short' }).toLowerCase();
  container.innerHTML = '';
  schedule.forEach(d => {
    const blocks = Array.isArray(d.blocks) ? d.blocks : [];
    const isToday = String(d.day || '').slice(0, 3).toLowerCase() === todayKey;
    const card = document.createElement('article');
    card.className = 'day' + (isToday ? ' today' : '');
    const head = document.createElement('header');
    const h = document.createElement('h3'); h.textContent = d.day || '';
    const n = document.createElement('small');
    n.textContent = isToday ? 'Today' : `${blocks.length} ${blocks.length === 1 ? 'block' : 'blocks'}`;
    head.append(h, n);
    card.append(head);

    if (d.focus_summary) {
      const fs = document.createElement('div');
      fs.className = 'focus-summary';
      fs.textContent = d.focus_summary;
      card.append(fs);
    }

    const body = document.createElement('tbody');
    if (!blocks.length) {
      const td = body.insertRow().insertCell(); td.colSpan = 2; td.className = 'empty'; td.textContent = 'Nothing planned.';
    }
    blocks.forEach(b => {
      const tr = body.insertRow();
      const t = tr.insertCell(); t.className = 'time'; t.textContent = b.time || '';
      const a = tr.insertCell();
      const energyClass = b.energy && ['high', 'medium', 'low'].includes(b.energy) ? b.energy : '';
      a.innerHTML = `<div class="block-activity">${energyClass ? `<span class="energy-dot ${energyClass}" title="${energyClass} energy"></span>` : ''}<div class="block-main"><b>${(b.activity || '').replace(/[<>&]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]))}</b>${b.note ? `<div class="block-note">${b.note.replace(/[<>&]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]))}</div>` : ''}</div></div>`;
    });
    const table = document.createElement('table'); table.append(body);
    card.append(table); container.append(card);
  });
}

/* ===== Streak + daily check-in =====
   Persistence: localStorage (one entry per browser) — fits the app's current
   no-database, cookie-session setup; the backend has no user identity to key a
   streak to, so a server endpoint would just be a second source of truth.
   Day boundary: local midnight, so "once a day" and "next eligible check-in"
   are the same calendar-day rule. The countdown is recomputed from the clock
   on every tick (never a decremented counter), so refresh / sleeping tabs /
   crossing midnight all stay correct. */
let streakMemory = { count: 0, last: null }; // used only if localStorage is blocked
const pad2 = n => String(n).padStart(2, '0');
const dayKey = d => `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;

function readStreak() {
  try {
    const o = JSON.parse(localStorage.getItem(STREAK_KEY) || 'null');
    if (o && Number.isInteger(o.count) && o.count >= 0 && /^\d{4}-\d{2}-\d{2}$/.test(o.last || '')) {
      streakMemory = { count: o.count, last: o.last };
    }
  } catch (e) { /* storage unavailable or corrupt — keep the in-memory value */ }
  return streakMemory;
}
function writeStreak(s) {
  streakMemory = s;
  try { localStorage.setItem(STREAK_KEY, JSON.stringify(s)); } catch (e) { /* in-memory only for this page load */ }
}
// A streak survives only if the last check-in was today or yesterday; otherwise it has lapsed to 0.
function streakStatus(now = new Date()) {
  const s = readStreak();
  const today = dayKey(now);
  const yesterday = dayKey(new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1));
  const checkedInToday = s.last === today;
  const alive = checkedInToday || s.last === yesterday;
  return { count: alive ? s.count : 0, checkedInToday, today };
}
function checkInToday() {
  const st = streakStatus();
  if (!st.checkedInToday) writeStreak({ count: st.count + 1, last: st.today });
}
function msUntilNextMidnight(now = new Date()) {
  return new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1) - now;
}
function formatCountdown(ms) {
  const total = Math.max(0, Math.ceil(ms / 1000));
  return `${pad2(Math.floor(total / 3600))}:${pad2(Math.floor((total % 3600) / 60))}:${pad2(total % 60)}`;
}

function initStreak() {
  const streakEl = $('streak'), btn = $('checkin'), timerEl = $('timer');
  if (!streakEl || !btn || !timerEl) return;
  let shown = null; // signature of what is currently rendered; full re-render only when it changes

  function render() {
    const st = streakStatus();
    const sig = `${st.count}|${st.checkedInToday}|${st.today}`;
    if (sig !== shown) { // first render, a check-in, or the day rolled over
      shown = sig;
      streakEl.innerHTML = `${ICON.flame}<div><b>${st.count}</b> <span>day streak</span></div>`;
      btn.disabled = st.checkedInToday;
      btn.textContent = st.checkedInToday ? 'Checked in today ✓' : 'Check in today';
      timerEl.innerHTML = st.checkedInToday
        ? '<span>Next check-in in</span><b></b>'
        : '<span>Today’s check-in</span><b>Open</b>';
    }
    if (st.checkedInToday) timerEl.querySelector('b').textContent = formatCountdown(msUntilNextMidnight());
  }

  btn.addEventListener('click', () => { checkInToday(); render(); });
  render();
  setInterval(render, 1000);
  document.addEventListener('visibilitychange', render); // catch up immediately after a sleeping tab wakes
  window.addEventListener('storage', render);            // check-in made in another tab
}

/* ===== Results page ===== */
const FAILURE_FALLBACK_MSG = {
  rate_limited: 'The AI service hit its rate limit — try again shortly.'
};

function setResultsHeading(isExample) {
  $('resultsTitle').textContent = isExample ? 'Example week' : 'Your week';
  $('resultsSub').textContent = isExample
    ? 'Sample data showing what BioSync builds. This is not your schedule.'
    : 'Built from your classes, deadlines, and routine.';
  document.title = isExample ? 'Example schedule – BioSync' : 'Your schedule – BioSync';
}

function makeNotice(kind, title, text, actions) {
  const box = document.createElement('div');
  box.className = 'notice ' + kind;
  box.setAttribute('role', kind === 'error' ? 'alert' : 'note');
  const copy = document.createElement('div');
  const strong = document.createElement('strong'); strong.textContent = title;
  const p = document.createElement('p'); p.textContent = text;
  copy.append(strong, p);
  box.append(copy);
  if (actions && actions.length) {
    const wrap = document.createElement('div');
    wrap.className = 'notice-actions';
    wrap.append(...actions);
    box.append(wrap);
  }
  return box;
}

// Renders the weekly view + analytics for one schedule payload into `body`.
// Used for real schedules AND for the example data — one code path, so the
// example always looks exactly like a real result.
function renderScheduleView(body, data) {
  const schedule = Array.isArray(data) ? data : (data.schedule || []); // tolerate old bare-array shape too
  const reasoningNotes = Array.isArray(data) ? '' : (data.reasoning_notes || '');
  const subjectBreakdown = Array.isArray(data) ? [] : (data.subject_breakdown || []);
  const weeklyFocusHours = Array.isArray(data) ? null : data.weekly_focus_hours;

  const stats = computeStats(schedule, subjectBreakdown, weeklyFocusHours);

  body.innerHTML = `
    <div class="reasoning" id="reasoningBox"></div>
    <div class="view-tabs" role="tablist">
      <button type="button" class="active" data-view="week" role="tab" aria-selected="true">Weekly view</button>
      <button type="button" data-view="analytics" role="tab" aria-selected="false">Analytics</button>
    </div>
    <div class="view-panel active" id="panel-week">
      <div class="week" id="week"></div>
    </div>
    <div class="view-panel" id="panel-analytics">
      <div class="stat-grid" id="statGrid"></div>
      <div class="chart-grid">
        <div class="chart-card">
          <h3>Hours per subject</h3>
          <div class="chart-sub">Total deep-focus time scheduled this week</div>
          <div id="subjectBars"></div>
        </div>
        <div class="chart-card">
          <h3>How your time breaks down</h3>
          <div class="chart-sub">Focus vs. class, meals, and breaks</div>
          <div class="donut-wrap">
            <div id="typeDonut"></div>
            <div class="legend" id="typeLegend"></div>
          </div>
        </div>
      </div>
      <div class="heatmap-card">
        <h3>Energy map across your week</h3>
        <div class="chart-sub">Where high-, medium-, and low-energy blocks are scheduled, by time of day</div>
        <div class="heatmap" id="heatmap"></div>
        <div class="heatmap-legend">
          <span><span class="hm-cell e-high" style="width:12px;height:12px;display:inline-block"></span> High energy</span>
          <span><span class="hm-cell e-medium" style="width:12px;height:12px;display:inline-block"></span> Medium energy</span>
          <span><span class="hm-cell e-low" style="width:12px;height:12px;display:inline-block"></span> Low energy</span>
          <span><span class="hm-cell e-none" style="width:12px;height:12px;display:inline-block"></span> Nothing scheduled</span>
        </div>
      </div>
    </div>`;

  renderReasoning($('reasoningBox'), reasoningNotes);
  renderWeek($('week'), schedule);
  renderStats($('statGrid'), stats);
  renderSubjectBars($('subjectBars'), stats.subjects);
  renderTypeDonut($('typeDonut'), $('typeLegend'), schedule);
  renderHeatmap($('heatmap'), schedule);

  body.querySelectorAll('.view-tabs button').forEach(btn => {
    btn.addEventListener('click', () => {
      body.querySelectorAll('.view-tabs button').forEach(b => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
      btn.classList.add('active'); btn.setAttribute('aria-selected', 'true');
      body.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
      $('panel-' + btn.dataset.view).classList.add('active');
    });
  });
}

// Example view: results.html?example=1. Checked before any API call and independent of USE_MOCK.
function showExample() {
  setResultsHeading(true);
  const build = document.createElement('a');
  build.className = 'btn'; build.href = 'chat.html'; build.textContent = 'Build my own';
  $('resultsNotice').replaceChildren(makeNotice(
    'example', 'This is an example',
    'Sample data, not based on your answers. No AI call was made to produce it.', [build]
  ));
  renderScheduleView($('resultsBody'), exampleScheduleData());
}

// Backend said the pipeline ran and produced no schedule (rate limit, LLM error, ...).
// Show the real reason, a retry, and the labeled example — never the example as if it were theirs.
function showScheduleFailure(data) {
  setResultsHeading(true);
  const retry = document.createElement('button');
  retry.type = 'button'; retry.className = 'btn'; retry.textContent = 'Try again';
  const errorNotice = makeNotice(
    'error', "Couldn't build your schedule",
    data.error || FAILURE_FALLBACK_MSG[data.reason] || "The AI service couldn't build your schedule — try again shortly.",
    [retry]
  );
  retry.addEventListener('click', async () => {
    retry.disabled = true; retry.textContent = 'Trying again…';
    try {
      showResults(await retrySchedule());
    } catch (e) {
      retry.disabled = false; retry.textContent = 'Try again';
      errorNotice.querySelector('p').textContent = "Couldn't reach the server. Check your connection and try again.";
    }
  });
  $('resultsNotice').replaceChildren(
    errorNotice,
    makeNotice('example', 'Example schedule below', "Sample data, not your schedule — it shows what BioSync builds once the AI service is available.", [])
  );
  renderScheduleView($('resultsBody'), exampleScheduleData());
}

// Decides what to show for one /api/schedule payload: real schedule, "none yet", or a failure.
function showResults(data) {
  $('resultsNotice').replaceChildren();
  setResultsHeading(false);

  if (!Array.isArray(data) && data.status === 'failed') { showScheduleFailure(data); return; }

  const schedule = Array.isArray(data) ? data : (data.schedule || []);
  if (!schedule.length) {
    $('resultsBody').innerHTML = '<p class="state">No schedule yet. <a href="chat.html">Chat with BioSync</a> to build one.</p>';
    return;
  }
  renderScheduleView($('resultsBody'), data);
}

async function initResults() {
  initStreak();
  if (new URLSearchParams(location.search).get('example') === '1') { showExample(); return; }

  try {
    showResults(await fetchSchedule());
  } catch (e) {
    $('resultsBody').innerHTML = "<p class='state'>We couldn't load your schedule. Refresh the page to try again.</p>";
  }
}

const page = document.body.dataset.page;
if (page === 'chat') initChat();
if (page === 'results') initResults();
