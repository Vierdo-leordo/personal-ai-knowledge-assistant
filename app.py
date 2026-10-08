"""Personal AI Knowledge Assistant – Streamlit web application.

Run:  python -m streamlit run app.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.agents import ExamPrepAgent, LearningAgent, SummarizationAgent, exam_prep_markdown
from src.chat import KnowledgeAssistant
from src.config import CATEGORIES, PROJECT_ROOT, get_settings
from src.evaluation import run_full_evaluation
from src.voice import VoiceError, browser_tts_html, speech_to_text, text_to_speech

st.set_page_config(page_title="Knowledge Assistant", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

SAMPLE_DIR = PROJECT_ROOT / "sample_data"
PAGES = ["🏠  Home", "💬  Ask", "📚  My Library", "🎓  Study", "⚙️  Settings"]
TOOLS = ["Summary", "Notes", "Flashcards", "Quiz", "Exam prep"]
FILE_ICONS = {"pdf": "📕", "docx": "📘", "txt": "📄", "md": "📝", "markdown": "📝"}

# ============================================================================ look & feel
STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#2A2550; --muted:#6E6A8F; --violet:#6D5DF6; --violet-2:#8E7CFF; --lilac:#C9C2FF; --blush:#F7D9EC;
  --glass:rgba(255,255,255,.62); --glass-strong:rgba(255,255,255,.82); --edge:rgba(255,255,255,.85);
  --shadow:0 10px 30px rgba(109,93,246,.12), 0 2px 8px rgba(42,37,80,.05);
}
html, body, .stApp, p, li, label, h1, h2, h3, h4, h5, h6, input, textarea, select, button, td, th,
.stMarkdown, [data-testid="stMarkdownContainer"] { font-family:'Lexend', system-ui, sans-serif; }
/* keep Streamlit's icon font for its arrows and icons */
[data-testid="stIconMaterial"], span[class*="material-symbols"], .material-icons, .material-symbols-rounded {
  font-family:'Material Symbols Rounded' !important; }
/* hide the link icon next to headings */
[data-testid="stHeaderActionElements"], h1 a, h2 a, h3 a { display:none !important; }
.stApp {
  color: var(--ink);
  background:
    radial-gradient(900px 600px at 8% 0%, #E4DEFF 0%, transparent 60%),
    radial-gradient(800px 600px at 100% 20%, #FBE3F1 0%, transparent 55%),
    radial-gradient(700px 500px at 50% 100%, #E1E9FF 0%, transparent 60%),
    #F3F1FB;
  background-attachment: fixed;
}
h1, h2, h3 { color: var(--ink) !important; letter-spacing:-0.01em; }
h1 { font-weight:600 !important; font-size:2.2rem !important; }
h3 { font-weight:500 !important; }
#MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display:none !important; }
.topnav-wrap { margin:-8px 0 22px; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top:2rem; max-width:1120px; }

/* frosted-glass surfaces */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
[data-testid="stExpander"], [data-testid="stForm"], [data-testid="stChatMessage"], [data-testid="stPopoverBody"] {
  background: var(--glass) !important; backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--edge) !important; border-radius: 24px !important; box-shadow: var(--shadow);
}
[data-testid="stExpander"] details { border:none !important; }
[data-testid="stChatMessage"] { padding:16px 20px; margin-bottom:10px; }

/* sidebar: light frosted panel, pill navigation */
[data-testid="stSidebar"] { background: rgba(255,255,255,.55); backdrop-filter: blur(20px); border-right:1px solid var(--edge); }
[data-testid="stSidebar"] .stButton > button { justify-content:flex-start; padding:10px 18px; border:none !important;
  background: transparent !important; box-shadow:none; font-size:1rem; }
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(109,93,246,.08) !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: linear-gradient(135deg, var(--violet), var(--violet-2)) !important;
  color:#fff !important; box-shadow: 0 6px 18px rgba(109,93,246,.35); }
.brand { display:flex; align-items:center; gap:10px; font-size:1.2rem; font-weight:600; color:var(--ink); margin:2px 0 12px; }
.brand .orb-sm { width:34px; height:34px; border-radius:50%; background: radial-gradient(circle at 30% 30%, #B9AEFF, #6D5DF6 60%, #4B3FC9); box-shadow:0 4px 14px rgba(109,93,246,.45); }
.status { background:var(--glass-strong); border:1px solid var(--edge); border-radius:18px; padding:12px 14px; font-size:.85rem; color:var(--muted); line-height:1.6; margin-bottom:14px; }
.status b { color:var(--ink); font-weight:500; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }

/* inputs */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stTextArea textarea {
  border-radius:16px !important; background: rgba(255,255,255,.8) !important; border-color: rgba(109,93,246,.15) !important; }
[data-testid="stChatInput"] { border-radius:999px !important; background: var(--glass-strong) !important; border:1px solid var(--edge) !important; box-shadow: var(--shadow); }
[data-testid="stFileUploaderDropzone"] { border-radius:20px; background: rgba(255,255,255,.6); border:2px dashed var(--lilac); }

/* buttons: pills */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button, [data-testid="stPopover"] button {
  border-radius:999px !important; font-weight:500; border:1px solid rgba(109,93,246,.18) !important;
  background: rgba(255,255,255,.75) !important; color: var(--ink) !important; transition: transform .1s, box-shadow .15s; }
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stPopover"] button:hover { box-shadow: 0 6px 16px rgba(109,93,246,.18); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--violet), var(--violet-2)) !important; color:#fff !important; border:none !important;
  box-shadow: 0 8px 20px rgba(109,93,246,.35); }
.stButton > button:active { transform: scale(.98); }
button:focus-visible { outline:3px solid var(--lilac) !important; outline-offset:2px; }
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p { color: var(--violet); font-weight:600; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, var(--violet), #E29BD3) !important; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { border-radius:999px; padding:6px 16px; background: rgba(255,255,255,.55); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--violet), var(--violet-2)) !important; color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* hero orb (the focal point, as in the reference) */
.hero { display:flex; align-items:center; gap:34px; background: var(--glass); backdrop-filter: blur(16px);
        border:1px solid var(--edge); border-radius:32px; padding:30px 34px; box-shadow: var(--shadow); margin-bottom:22px; }
.orb { flex:0 0 150px; height:150px; border-radius:50%; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#fff;
       background: radial-gradient(circle at 30% 25%, #C8BEFF 0%, #8E7CFF 35%, #5B4BE0 70%, #3E33A8 100%);
       box-shadow: 0 18px 40px rgba(91,75,224,.45), inset 0 -10px 30px rgba(30,20,90,.35); }
.orb .n { font-size:2.6rem; font-weight:600; line-height:1; }
.orb .l { font-size:.78rem; opacity:.9; margin-top:6px; }
.hero h1 { margin:0 0 6px !important; }
.hero p { color: var(--muted); margin:0; font-size:1.02rem; }

/* citations: soft lilac chips */
mark.cn { background: #E7E2FF; color: #4B3FC9; padding:1px 7px; border-radius:999px; font-size:.78em; font-weight:600; }
.cite { display:inline-block; background: rgba(255,255,255,.85); border:1px solid #DCD5FF; color:#4B3FC9; border-radius:999px;
        padding:4px 12px; font-size:.82rem; margin:8px 6px 0 0; font-weight:500; }
.lead { color: var(--muted); font-size:1.05rem; margin-top:-6px; margin-bottom:18px; }
.step-n { display:inline-flex; width:34px; height:34px; border-radius:50%; color:#fff; align-items:center; justify-content:center;
          font-weight:600; margin-bottom:8px; background: linear-gradient(135deg, var(--violet), var(--violet-2)); }
.doc-name { font-weight:500; color:var(--ink); }
.doc-meta { color:var(--muted); font-size:.85rem; }
.flashcard { background: var(--glass-strong); border:1px solid var(--edge); border-radius:28px; padding:2.6rem 1.8rem; min-height:200px;
             display:flex; align-items:center; justify-content:center; text-align:center; font-size:1.25rem; color:var(--ink); box-shadow: var(--shadow); }
.flashcard small { color: var(--violet); font-size:.8rem; font-weight:500; }
.empty { color:var(--muted); padding:18px 0; }
@media (max-width: 700px) { .hero { flex-direction:column; text-align:center; } }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
</style>
"""
# Markdown ends an HTML block at a blank line, so the stylesheet must not contain any.
st.markdown(re.sub(r"\n\s*\n", "\n", STYLE), unsafe_allow_html=True)


# ============================================================================ state
def _overrides() -> dict:
    return st.session_state.setdefault("overrides", {})


@st.cache_resource(show_spinner="Getting your assistant ready…")
def _build_assistant(overrides_key: str) -> KnowledgeAssistant:
    overrides = json.loads(overrides_key)
    for env_name, value in overrides.pop("_env", {}).items():
        if value:
            os.environ[env_name] = value
    return KnowledgeAssistant(get_settings(**overrides))


def get_assistant() -> KnowledgeAssistant:
    return _build_assistant(json.dumps(_overrides(), sort_keys=True))


A = get_assistant()
PREFS = A.memory.get_preferences()
SHOW_DETAILS = PREFS.get("show_details") == "true"


def go(page: str, **state):
    """Navigate to another page (optionally carrying state). Applied before the sidebar is drawn."""
    st.session_state["_goto"] = page
    st.session_state.update(state)
    st.rerun()


def current_conversation() -> str:
    cid = st.session_state.get("conversation_id")
    if not cid or not any(c["id"] == cid for c in A.memory.list_conversations()):
        cid = A.memory.create_conversation()
        st.session_state.conversation_id = cid
    return cid


def docs_by_label() -> dict:
    return {d["file_name"]: d["doc_id"] for d in A.memory.list_documents()}


def save_output(kind: str, title: str, content: str) -> Path:
    A.memory.save_output(kind, title, content)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title)[:50]
    path = A.settings.outputs_dir / f"{stamp}_{kind}_{safe}.md"
    path.write_text(content, encoding="utf-8")
    return path


def friendly_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return ""


def ingest_uploads(files, category: str):
    for f in files:
        bar = st.progress(0.0, text=f"Reading {f.name}…")
        try:
            res = A.ingest_file(f.getvalue(), f.name, category,
                                progress=lambda p, msg, n=f.name: bar.progress(min(p, 1.0), text=f"{n}: {msg}"))
            bar.empty()
            if res.duplicate:
                st.info(f"**{f.name}** is already in your library.")
            else:
                st.success(f"Added **{res.file_name}** ({res.num_pages} pages).")
        except Exception as e:  # noqa: BLE001
            bar.empty()
            msg = str(e)
            if "No module named" in msg:
                msg += ". Start the app from the project's .venv (see README) so all packages are available."
            st.error(f"Couldn't add **{f.name}**: {msg}")


def load_samples():
    for path in sorted(SAMPLE_DIR.glob("*")):
        if path.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
            A.ingest_file(path, category="Education")


def voice_question(key: str, label: str):
    """Record a question in the browser, transcribe it and send it to the Ask page."""
    if not hasattr(st, "audio_input"):
        st.caption("Voice input needs Streamlit 1.39 or newer: python -m pip install -U streamlit")
        return
    audio = st.audio_input(label, key=key)
    if audio is None:
        return
    data = audio.getvalue()
    digest = hashlib.md5(data).hexdigest()
    if st.session_state.get(f"{key}_done") == digest:
        return
    st.session_state[f"{key}_done"] = digest
    with st.spinner("Listening…"):
        try:
            text, _engine = speech_to_text(data, A.settings)
        except VoiceError:
            st.error("Voice input needs a Groq, Gemini or OpenAI key. Add one in Settings → AI service.")
            return
    if text.strip():
        go(PAGES[1], pending_question=text.strip())
    else:
        st.warning("I couldn't hear a question. Try recording again a little closer to the microphone.")


# ============================================================================ renderers
_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def highlight_citations(md: str) -> str:
    """Turn [1] markers into highlighter marks, leaving code untouched."""
    parts = re.split(r"(```.*?```)", md, flags=re.S)
    for i in range(0, len(parts), 2):
        bits = re.split(r"(`[^`\n]*`)", parts[i])
        for j in range(0, len(bits), 2):
            bits[j] = _CITE.sub(lambda m: f"<mark class='cn'>{m.group(1)}</mark>", bits[j])
        parts[i] = "".join(bits)
    return "".join(parts)


def render_answer(text: str):
    st.markdown(highlight_citations(text), unsafe_allow_html=True)


def render_sources(sources: list[dict], retrieved: list[dict] | None = None, metrics: dict | None = None):
    if sources:
        seen, chips = set(), []
        for s in sources:
            key = (s["file_name"], s.get("page"))
            if key in seen:
                continue
            seen.add(key)
            page = f" · p.{s['page']}" if s.get("page") else ""
            chips.append(f"<span class='cite'>{s['number']} &nbsp;{s['file_name']}{page}</span>")
        st.markdown("".join(chips), unsafe_allow_html=True)
        with st.expander("See the passages this answer used"):
            for s in sources:
                where = f"page {s['page']}" if s.get("page") else ""
                if s.get("heading"):
                    where = f"{where}, {s['heading']}" if where else s["heading"]
                st.markdown(f"**{s['number']}. {s['file_name']}** {('(' + where + ')') if where else ''}")
                st.caption(s.get("excerpt", ""))
    if SHOW_DETAILS and (metrics or retrieved):
        with st.expander("Technical details"):
            if metrics:
                m = metrics
                st.caption(
                    f"Time {m.get('latency_s', '-')}s · citation coverage {m.get('citation_coverage', 0):.0%} · "
                    f"unsupported sentences {m.get('hallucination_rate', 0):.0%} · top match {m.get('top_score', 0):.2f}"
                )
                if m.get("llm_error"):
                    st.code(m["llm_error"], language=None)
            if retrieved:
                st.dataframe(pd.DataFrame([{
                    "file": r["file_name"], "page": r["page"], "score": r["score"], "vector": r["vector_score"],
                    "keyword": r["keyword_score"], "text": r["text"][:140] + "…"} for r in retrieved]),
                    hide_index=True, use_container_width=True)


def render_read_aloud(text: str):
    if PREFS.get("voice_replies") == "true":
        audio = text_to_speech(text, A.settings)
        if audio:
            st.audio(audio, format="audio/mp3", autoplay=True)
            return
    components.html(browser_tts_html(text), height=44)


def render_flashcards(cards: list[dict], key: str):
    if not cards:
        st.info("No flashcards could be made from this material. Try a different topic or document.")
        return
    idx_key, show_key, known_key = f"{key}_idx", f"{key}_show", f"{key}_known"
    st.session_state.setdefault(idx_key, 0)
    st.session_state.setdefault(show_key, False)
    st.session_state.setdefault(known_key, set())
    i = st.session_state[idx_key] % len(cards)
    card, known = cards[i], st.session_state[known_key]
    st.progress(len(known) / len(cards), text=f"Card {i + 1} of {len(cards)}  ·  {len(known)} learned")
    showing = st.session_state[show_key]
    st.markdown(f"<div class='flashcard'><div><small>{'Answer' if showing else 'Question'}</small><br><br>"
                f"{card['answer'] if showing else card['question']}</div></div>", unsafe_allow_html=True)
    if card.get("source_label"):
        st.caption(f"From {card['source_label']}")

    def move(step: int, learned: bool = False):
        if learned:
            known.add(i)
        st.session_state[idx_key] = (i + step) % len(cards)
        st.session_state[show_key] = False

    c1, c2, c3, c4 = st.columns(4)
    c1.button("Previous", key=f"{key}_prev", use_container_width=True, on_click=move, args=(-1,))
    c2.button("Hide answer" if showing else "Show answer", key=f"{key}_flip", use_container_width=True,
              type="primary", on_click=lambda: st.session_state.update({show_key: not showing}))
    c3.button("I knew this", key=f"{key}_know", use_container_width=True, on_click=move, args=(1, True))
    c4.button("Next", key=f"{key}_next", use_container_width=True, on_click=move, args=(1,))

    buf = io.StringIO()
    csv.writer(buf).writerows([[c["question"], c["answer"]] for c in cards])
    d1, d2 = st.columns(2)
    d1.download_button("Download for Anki (CSV)", buf.getvalue(), "flashcards.csv", "text/csv",
                       key=f"{key}_dl", use_container_width=True)
    if d2.button("Save to my study material", key=f"{key}_save", use_container_width=True):
        save_output("flashcards", f"Flashcards ({len(cards)})",
                    "\n\n".join(f"**Q:** {c['question']}\n\n**A:** {c['answer']}" for c in cards))
        st.success("Saved. Find it under Saved study material.")


def render_quiz(quiz: dict, key: str):
    mcq, short = quiz.get("mcq", []), quiz.get("short", [])
    if not mcq and not short:
        if quiz.get("error"):
            st.warning("The AI couldn't build a quiz this time. Press **Make quiz** to try again, "
                       "or type a more specific topic.")
            if SHOW_DETAILS:
                st.code(quiz["error"][:600], language=None)
        else:
            st.info("No questions could be made from this material. Try a different topic or document.")
        return
    done_key = f"{key}_submitted"
    with st.form(f"{key}_form"):
        for i, q in enumerate(mcq):
            st.markdown(f"**{i + 1}. {q['question']}**")
            st.radio("Your answer", q["options"], index=None, key=f"{key}_mcq_{i}", label_visibility="collapsed")
        for j, q in enumerate(short):
            st.text_area(f"{len(mcq) + j + 1}. {q['question']}", key=f"{key}_short_{j}", height=80,
                         placeholder="Type your answer")
        if st.form_submit_button("Check my answers", type="primary"):
            st.session_state[done_key] = True
    if not st.session_state.get(done_key):
        return
    correct = 0
    for i, q in enumerate(mcq):
        right = q["options"][q["answer_index"]]
        ok = st.session_state.get(f"{key}_mcq_{i}") == right
        correct += ok
        st.markdown(f"{'✅' if ok else '❌'} **Question {i + 1}:** {'correct' if ok else 'the answer is'} *{right}*")
        if q.get("explanation") and not ok:
            st.caption(q["explanation"])
    if mcq:
        st.metric("Your score", f"{correct} / {len(mcq)}")
    if short:
        learner = LearningAgent(A)
        for j, q in enumerate(short):
            grade = learner.grade_short_answer(q["question"], q.get("model_answer", ""),
                                               st.session_state.get(f"{key}_short_{j}", ""))
            st.markdown(f"**Question {len(mcq) + j + 1}:** {grade['score']:.0%}. {grade['feedback']}")
            with st.expander("Model answer"):
                st.write(q.get("model_answer", ""))

    def retake():
        st.session_state[done_key] = False
        for i in range(len(mcq)):
            st.session_state.pop(f"{key}_mcq_{i}", None)

    st.button("Try again", key=f"{key}_retake", on_click=retake)


def output_actions(kind: str, title: str, content: str, key: str):
    c1, c2 = st.columns(2)
    c1.download_button("Download", content, f"{title}.md", key=f"{key}_dl", use_container_width=True)
    if c2.button("Save to my study material", key=f"{key}_save", use_container_width=True):
        save_output(kind, title, content)
        st.success("Saved. Find it under Saved study material.")


# ============================================================================ sidebar
status = A.status()
with st.sidebar:
    st.markdown("<div class='brand'><span class='orb-sm'></span>Knowledge Assistant</div>", unsafe_allow_html=True)
    if status["llm_provider"] == "offline":
        ai_line = "<span class='dot' style='background:#F2B84B'></span><b>AI not set up</b> · basic answers"
    else:
        ai_line = f"<span class='dot' style='background:#3DD68C'></span><b>AI ready</b> · {status['llm_provider'].title()}"
    n_docs = status["documents"]
    st.markdown(f"<div class='status'>{ai_line}<br>📚 <b>{n_docs}</b> document{'s' if n_docs != 1 else ''} in your library</div>",
                unsafe_allow_html=True)
    if "_goto" in st.session_state:
        st.session_state.nav = st.session_state.pop("_goto")
    st.session_state.setdefault("nav", PAGES[0])
    page = st.session_state.nav
    for target in PAGES:
        if st.button(target, key=f"side_{target}", use_container_width=True,
                     type="primary" if target == page else "secondary") and target != page:
            go(target)
    stale = A.stale_documents()
    if stale:
        st.warning(f"{len(stale)} document(s) need refreshing after a settings change. Go to Settings → Storage.")


# ============================================================================ top menu (always visible)
def top_menu(current: str):
    cols = st.columns(len(PAGES))
    for col, target in zip(cols, PAGES):
        if col.button(target, key=f"top_{target}", use_container_width=True,
                      type="primary" if target == current else "secondary"):
            if target != current:
                go(target)


top_menu(page)


# ============================================================================ pages
def page_home():
    n = status["documents"]
    hour = datetime.now().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    st.markdown(
        f"<div class='hero'><div class='orb'><div class='n'>{n}</div><div class='l'>document{'s' if n != 1 else ''}</div></div>"
        f"<div><p>{greet} 👋</p><h1>What do you want to learn today?</h1>"
        f"<p>Ask anything about your notes, papers and books. Every answer shows where it came from.</p></div></div>",
        unsafe_allow_html=True)
    with st.form("home_ask", clear_on_submit=True, border=False):
        c1, c2 = st.columns([5, 1])
        q = c1.text_input("Your question", placeholder="e.g. Explain process scheduling with an example",
                          label_visibility="collapsed")
        if c2.form_submit_button("Ask", type="primary", use_container_width=True) and q.strip():
            go(PAGES[1], pending_question=q.strip())
    voice_question("home_voice", "🎤 Or ask by voice: press record, speak, then stop")

    docs = A.memory.list_documents()
    if not docs:
        st.subheader("Get started in three steps")
        steps = [("Add your notes", "Upload PDFs, Word files or text notes to your library."),
                 ("Ask a question", "Get an answer written from your own material, with the pages it used."),
                 ("Study smarter", "Turn any document into a summary, flashcards or a practice quiz.")]
        cols = st.columns(3)
        for n, (col, (title, body)) in enumerate(zip(cols, steps), 1):
            with col.container(border=True):
                st.markdown(f"<span class='step-n'>{n}</span>", unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                st.caption(body)
        c1, c2, _ = st.columns([1.3, 1.3, 2])
        if c1.button("Add your first document", type="primary", use_container_width=True):
            go(PAGES[2])
        if c2.button("Try with sample notes", use_container_width=True):
            with st.spinner("Adding sample notes…"):
                load_samples()
            st.rerun()
        return

    st.subheader("Quick actions")
    actions = [("📤 Add documents", PAGES[2], {}), ("💬 Chat with my notes", PAGES[1], {}),
               ("📄 Summarize a document", PAGES[3], {"study_tool": "Summary"}),
               ("📝 Make study notes", PAGES[3], {"study_tool": "Notes"}),
               ("🗂️ Make flashcards", PAGES[3], {"study_tool": "Flashcards"}),
               ("❓ Take a quiz", PAGES[3], {"study_tool": "Quiz"}),
               ("🎯 Prepare for an exam", PAGES[3], {"study_tool": "Exam prep"}),
               ("🕘 Chat history", PAGES[4], {})]
    for row in (actions[:4], actions[4:]):
        for col, (label, target, state) in zip(st.columns(4), row):
            if col.button(label, use_container_width=True, key=f"qa_{label}"):
                go(target, **state)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Recently added")
        for d in docs[:5]:
            st.markdown(f"{FILE_ICONS.get(d['file_type'], '📄')} <span class='doc-name'>{d['file_name']}</span> "
                        f"<span class='doc-meta'>&nbsp;{d['num_pages']} pages · {d['category']}</span>",
                        unsafe_allow_html=True)
        if len(docs) > 5:
            st.caption(f"and {len(docs) - 5} more in My Library")
    with right:
        st.subheader("Recent chats")
        convs = [c for c in A.memory.list_conversations() if c["n_messages"]][:4]
        if not convs:
            st.markdown("<div class='empty'>Your chats will show up here.</div>", unsafe_allow_html=True)
        for c in convs:
            if st.button(c["title"] or "Untitled chat", key=f"home_c_{c['id']}", use_container_width=True):
                go(PAGES[1], conversation_id=c["id"])


def page_ask():
    cid = current_conversation()
    head, b1, b2 = st.columns([4, 1, 1])
    head.title("Ask your notes")
    with b1.popover("Past chats", use_container_width=True):
        convs = [c for c in A.memory.list_conversations() if c["n_messages"]]
        if not convs:
            st.caption("No past chats yet.")
        for c in convs[:15]:
            if st.button(c["title"] or "Untitled chat", key=f"pc_{c['id']}", use_container_width=True):
                st.session_state.conversation_id = c["id"]
                st.rerun()
    if b2.button("New chat", use_container_width=True, type="primary"):
        st.session_state.conversation_id = A.memory.create_conversation()
        st.rerun()

    docs = docs_by_label()
    if not docs:
        st.info("Your library is empty. Add a document first so I have something to answer from.")
        if st.button("Go to My Library", type="primary"):
            go(PAGES[2])
        return

    with st.expander("Choose which documents to search", expanded=False):
        chosen = st.multiselect("Only search in these documents (leave empty to search everything)",
                                list(docs), key="ask_docs")
        compare = st.toggle("Compare across documents (pulls from several files)", key="ask_compare")
    doc_ids = [docs[d] for d in chosen] or None
    mode = "compare" if compare else "auto"

    messages = A.memory.get_messages(cid)
    if not messages:
        st.markdown("<div class='lead'>Ask a question, or try one of these:</div>", unsafe_allow_html=True)
        first = list(docs)[0]
        stem = re.sub(r"\.[a-z0-9]+$", "", first, flags=re.I).replace("_", " ")
        ideas = [f"Summarize {first}", f"What are the main topics in {stem}?",
                 f"Make flashcards on {stem}", f"Quiz me on {stem}"]
        cols = st.columns(2)
        for n, idea in enumerate(ideas):
            if cols[n % 2].button(idea, key=f"idea_{n}", use_container_width=True):
                st.session_state.pending_question = idea
                st.rerun()

    for m in messages:
        with st.chat_message(m["role"], avatar="🙂" if m["role"] == "user" else "🧠"):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_answer(m["content"])
                meta = m.get("metadata", {})
                render_sources(m.get("sources", []), meta.get("retrieved"), meta.get("metrics"))
                if meta.get("extra", {}).get("flashcards") or meta.get("extra", {}).get("quiz"):
                    st.caption("Open **Study** to practise these interactively.")

    # voice + quick upload sit just above the chat box
    t1, t2, _ = st.columns([1, 1, 4])
    with t1.popover("🎤 Speak", use_container_width=True):
        voice_question("ask_voice", "Record your question")
    with t2.popover("📎 Add file", use_container_width=True):
        files = st.file_uploader("Add documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
                                 key="chat_upload", label_visibility="collapsed")
        if files and st.button("Add to library", type="primary", key="chat_upload_go"):
            ingest_uploads(files, "General")

    question = st.chat_input("Ask about your documents…") or st.session_state.pop("pending_question", None)
    if question:
        with st.chat_message("user", avatar="🙂"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🧠"):
            with st.spinner("Reading your documents…"):
                resp = A.ask(question, cid, mode=mode, doc_ids=doc_ids)
            render_answer(resp.answer)
            render_sources([s.to_dict() for s in resp.cited], [r.to_dict() for r in resp.retrieved], resp.metrics)
            render_read_aloud(resp.answer)
            if resp.extra.get("flashcards") or resp.extra.get("quiz"):
                st.session_state.study_payload = resp.extra
                if st.button("Practise these in Study"):
                    go(PAGES[3], study_tool="Flashcards" if resp.extra.get("flashcards") else "Quiz")


def page_library():
    st.title("My Library")
    st.markdown("<div class='lead'>Everything the assistant can answer from. Add notes, books and papers here.</div>",
                unsafe_allow_html=True)
    with st.container(border=True):
        files = st.file_uploader("Drop files here: PDF, Word (.docx), text or Markdown",
                                 type=["pdf", "docx", "txt", "md", "markdown"], accept_multiple_files=True)
        c1, c2, c3 = st.columns([2, 1.2, 1.2])
        category = c1.selectbox("Put them in", CATEGORIES, key="upload_cat")
        c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        c3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        add = c2.button("Add to library", type="primary", disabled=not files, use_container_width=True)
        samples = c3.button("Add sample notes", use_container_width=True)
        if add and files:
            ingest_uploads(files, category)
        if samples:
            with st.spinner("Adding sample notes…"):
                load_samples()
            st.success("Sample notes added.")

    docs = A.memory.list_documents()
    if not docs:
        st.markdown("<div class='empty'>No documents yet. Add your first file above.</div>", unsafe_allow_html=True)
        return

    st.subheader("Find something")
    q = st.text_input("Search your library", placeholder="e.g. deadlock conditions", label_visibility="collapsed")
    if q:
        results = A.search(q, k=6)
        if not results:
            st.caption("Nothing matched. Try different words.")
        for r in results:
            with st.container(border=True):
                where = f"page {r.page}" + (f" · {r.metadata.get('heading')}" if r.metadata.get("heading") else "")
                st.markdown(f"<span class='cite'>{r.file_name} · {where}</span>", unsafe_allow_html=True)
                st.write(r.text[:600] + ("…" if len(r.text) > 600 else ""))
                if SHOW_DETAILS:
                    st.caption(f"score {r.score:.3f} (vector {r.vector_score:.3f}, keyword {r.keyword_score:.2f})")

    st.subheader(f"Your documents ({len(docs)})")
    used = ["All"] + [c for c in CATEGORIES if any(d["category"] == c for d in docs)]
    shown_cat = st.radio("Show", used, horizontal=True, label_visibility="collapsed", key="lib_cat")
    for d in docs:
        if shown_cat != "All" and d["category"] != shown_cat:
            continue
        with st.container(border=True):
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"{FILE_ICONS.get(d['file_type'], '📄')} <span class='doc-name'>{d['file_name']}</span><br>"
                        f"<span class='doc-meta'>{d['num_pages']} pages · {d['category']} · added {friendly_date(d['upload_date'])}</span>",
                        unsafe_allow_html=True)
            with c2.popover("Options", use_container_width=True):
                new_cat = st.selectbox("Category", CATEGORIES, key=f"cat_{d['doc_id']}",
                                       index=CATEGORIES.index(d["category"]) if d["category"] in CATEGORIES else 0)
                if new_cat != d["category"]:
                    A.set_category(d["doc_id"], new_cat)
                    st.rerun()
                if st.button("Summarize", key=f"sum_{d['doc_id']}", use_container_width=True):
                    go(PAGES[3], study_tool="Summary", study_docs=[d["file_name"]])
                stored = Path(d["stored_path"] or "")
                if stored.exists():
                    st.download_button("Download", stored.read_bytes(), d["file_name"], key=f"dl_{d['doc_id']}",
                                       use_container_width=True)
                if st.button("Remove from library", key=f"del_{d['doc_id']}", use_container_width=True):
                    A.delete_document(d["doc_id"])
                    st.rerun()


def page_study():
    st.title("Study")
    st.markdown("<div class='lead'>Pick your material, choose a tool, and the assistant builds it from your documents.</div>",
                unsafe_allow_html=True)
    docs = docs_by_label()
    if not docs:
        st.info("Add documents to your library first.")
        if st.button("Go to My Library", type="primary"):
            go(PAGES[2])
        return

    if "study_docs" in st.session_state:
        st.session_state.study_docs = [d for d in st.session_state.study_docs if d in docs]
    c1, c2 = st.columns(2)
    topic = c1.text_input("Topic (optional)", key="study_topic", placeholder="e.g. CPU scheduling")
    chosen = c2.multiselect("Documents (optional)", list(docs), key="study_docs",
                            placeholder="All documents")
    doc_ids = [docs[d] for d in chosen] or None

    st.session_state.setdefault("study_tool", "Summary")
    tool = st.radio("Tool", TOOLS, horizontal=True, key="study_tool", label_visibility="collapsed")
    st.divider()

    if tool == "Summary":
        length = st.radio("Length", ["Short", "Detailed"], horizontal=True, key="sum_len")
        if st.button("Write summary", type="primary"):
            if not topic and not doc_ids:
                st.warning("Choose a document or type a topic to summarize.")
            else:
                agent = SummarizationAgent(A)
                with st.spinner("Writing your summary…"):
                    if doc_ids and len(doc_ids) == 1 and not topic:
                        res = agent.summarize_document(doc_ids[0], "detailed" if length == "Detailed" else "concise")
                    else:
                        res = agent.summarize_topic(topic or "key ideas", doc_ids,
                                                    "detailed" if length == "Detailed" else "concise")
                st.session_state.summary = res
                st.session_state.summary_title = topic or ", ".join(chosen)
        if st.session_state.get("summary"):
            res = st.session_state.summary
            render_answer(res["markdown"])
            render_sources([s.to_dict() for s in res["sources"]][:8])
            output_actions("summary", f"Summary - {st.session_state.summary_title}", res["markdown"], "sum")

    elif tool == "Notes":
        style = st.radio("Kind of notes", ["Chapter Notes", "Revision Guide", "Exam Summary"], horizontal=True,
                         key="notes_style")
        if st.button("Make notes", type="primary"):
            with st.spinner("Writing your notes…"):
                st.session_state.notes = LearningAgent(A).study_notes(style, topic, doc_ids)
                st.session_state.notes_title = f"{style} - {topic or 'all'}"
        if st.session_state.get("notes"):
            render_answer(st.session_state.notes["markdown"])
            render_sources([s.to_dict() for s in st.session_state.notes["sources"]][:10])
            output_actions("notes", st.session_state.notes_title, st.session_state.notes["markdown"], "notes")

    elif tool == "Flashcards":
        n = st.slider("How many cards", 5, 30, 10, key="fc_n")
        payload = st.session_state.get("study_payload") or {}
        if payload.get("flashcards") and not st.session_state.get("flashcards"):
            st.session_state.flashcards = {"cards": payload["flashcards"]}
        if st.button("Make flashcards", type="primary"):
            with st.spinner("Making flashcards…"):
                st.session_state.flashcards = LearningAgent(A).flashcards(topic, int(n), doc_ids)
            for k in ("fc_idx", "fc_show", "fc_known"):
                st.session_state.pop(k, None)
        if st.session_state.get("flashcards"):
            if st.session_state.flashcards.get("error"):
                st.warning("The AI couldn't make flashcards this time. Press **Make flashcards** to try again.")
            else:
                render_flashcards(st.session_state.flashcards["cards"], key="fc")

    elif tool == "Quiz":
        c1, c2, c3 = st.columns(3)
        n_mcq = c1.number_input("Multiple-choice", 0, 20, 5, key="qz_mcq")
        n_short = c2.number_input("Short answer", 0, 10, 3, key="qz_short")
        diff = c3.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="qz_diff")
        payload = st.session_state.get("study_payload") or {}
        if payload.get("quiz") and not st.session_state.get("quiz"):
            st.session_state.quiz = payload["quiz"]
        if st.button("Make quiz", type="primary"):
            with st.spinner("Writing questions…"):
                st.session_state.quiz = LearningAgent(A).quiz(topic, int(n_mcq), int(n_short), diff, doc_ids)
            for k in [k for k in st.session_state if str(k).startswith(("qz_submitted", "qz_mcq_", "qz_short_"))]:
                del st.session_state[k]
        if st.session_state.get("quiz"):
            render_quiz(st.session_state.quiz, key="qz")

    else:  # Exam prep
        goal = st.text_input("What's the exam?", key="ep_goal", placeholder="e.g. Operating Systems mid-term")
        days = st.slider("Days until the exam", 1, 30, 5, key="ep_days")
        if st.button("Build my study plan", type="primary", disabled=not (goal or topic)):
            with st.status("Building your study pack…", expanded=True) as box:
                res = ExamPrepAgent(A).prepare(f"Prepare me for my {goal or topic} exam", int(days), doc_ids,
                                               progress=lambda p, msg: box.write(msg))
                box.update(label="Your study pack is ready" if "error" not in res else "Nothing found",
                           state="complete")
            st.session_state.exam = res
            for k in [k for k in st.session_state if str(k).startswith(("epfc_", "epqz_"))]:
                del st.session_state[k]
        res = st.session_state.get("exam")
        if res and "error" in res:
            st.warning("I couldn't find material for that exam in your library. Add the right notes, or pick documents above.")
        elif res:
            st.markdown("**Key topics:** " + ", ".join(res["topics"]))
            t1, t2, t3, t4 = st.tabs(["Summary", "Day-by-day plan", "Practice quiz", "Flashcards"])
            with t1:
                render_answer(res["summary"]["markdown"])
            with t2:
                render_answer(res["plan"])
            with t3:
                render_quiz(res["quiz"], key="epqz")
            with t4:
                render_flashcards(res["flashcards"]["cards"], key="epfc")
            output_actions("exam_prep", f"Exam prep - {res['subject']}", exam_prep_markdown(res), "ep")

    st.divider()
    with st.expander("Saved study material"):
        outs = A.memory.list_outputs()
        if not outs:
            st.caption("Things you save will appear here.")
        for o in outs:
            c1, c2, c3 = st.columns([5, 1, 1])
            c1.markdown(f"**{o['title']}**  \n<span class='doc-meta'>{o['kind']} · {friendly_date(o['created_at'])}</span>",
                        unsafe_allow_html=True)
            c2.download_button("Download", o["content"], f"{o['title']}.md", key=f"o_dl_{o['id']}")
            if c3.button("Delete", key=f"o_del_{o['id']}"):
                A.memory.delete_output(o["id"])
                st.rerun()


def page_settings():
    st.title("Settings")
    t1, t2, t3, t4, t5 = st.tabs(["Answers", "AI service", "Chat history", "Quality check", "Storage"])

    with t1:
        with st.form("prefs"):
            style = st.select_slider("How long should answers be?", ["Concise", "Balanced", "Detailed"],
                                     value=PREFS["answer_style"])
            level = st.radio("Explain things for a…", ["Beginner", "Student", "Expert"], horizontal=True,
                             index=["Beginner", "Student", "Expert"].index(PREFS["expertise_level"]))
            lang = st.text_input("Answer language", PREFS["language"])
            custom = st.text_area("Anything else the assistant should always do?", PREFS["custom_instructions"],
                                  placeholder="e.g. Always include a real-world example.")
            voice = st.toggle("Read answers aloud automatically", value=PREFS["voice_replies"] == "true")
            details = st.toggle("Show technical details (scores, retrieval tables)", value=SHOW_DETAILS)
            if st.form_submit_button("Save", type="primary"):
                for k, v in {"answer_style": style, "expertise_level": level, "language": lang,
                             "custom_instructions": custom, "voice_replies": str(voice).lower(),
                             "show_details": str(details).lower()}.items():
                    A.memory.set_preference(k, v)
                st.success("Saved.")
                st.rerun()
        topics = A.memory.top_topics(10)
        if topics:
            st.subheader("Topics you ask about most")
            st.bar_chart(pd.DataFrame(topics).set_index("topic")["count"], horizontal=True, color="#3B3FB6")

    with t2:
        st.markdown(f"Answers are written by **{status['llm_provider'].title()}** "
                    f"(`{status['llm_model']}`). Document search uses **{status['embedder']}**.")
        st.caption("Keys typed here last until the app restarts. To keep them, put them in the .env file.")
        ov, env = _overrides(), dict(_overrides().get("_env", {}))
        with st.form("ai"):
            providers = ["auto", "groq", "gemini", "openai", "anthropic", "offline"]
            cur = ov.get("llm_provider", A.settings.llm_provider)
            prov = st.selectbox("Which AI writes the answers", providers,
                                index=providers.index(cur) if cur in providers else 0,
                                help="'auto' uses the first service you have a key for and switches if one is busy.")
            c1, c2 = st.columns(2)
            qk = c1.text_input("Groq key (free)", type="password", value=env.get("GROQ_API_KEY", ""),
                               help="https://console.groq.com/keys")
            gk = c2.text_input("Gemini key (free)", type="password", value=env.get("GEMINI_API_KEY", ""),
                               help="https://aistudio.google.com/apikey")
            ok = c1.text_input("OpenAI key", type="password", value=env.get("OPENAI_API_KEY", ""))
            ak = c2.text_input("Anthropic key", type="password", value=env.get("ANTHROPIC_API_KEY", ""))
            with st.expander("Advanced"):
                c1, c2 = st.columns(2)
                qm = c1.text_input("Groq model", A.settings.groq_model)
                gm = c2.text_input("Gemini model", A.settings.gemini_model)
                om = c1.text_input("OpenAI model", A.settings.openai_model)
                am = c2.text_input("Claude model", A.settings.anthropic_model)
                temp = st.slider("Creativity", 0.0, 1.0, float(A.settings.temperature), 0.05)
                emb_opts = ["auto", "sentence-transformers", "gemini", "openai", "hashing"]
                emb = c1.selectbox("Search model", emb_opts, index=emb_opts.index(A.settings.embedding_provider)
                                   if A.settings.embedding_provider in emb_opts else 0,
                                   help="Changing this means refreshing your documents in Storage.")
                tk = c2.slider("Passages per answer", 2, 15, A.settings.top_k)
            if st.form_submit_button("Apply", type="primary"):
                st.session_state.overrides = {
                    "llm_provider": prov, "groq_model": qm, "gemini_model": gm, "openai_model": om,
                    "anthropic_model": am, "temperature": temp, "embedding_provider": emb, "top_k": int(tk),
                    "groq_api_key": qk or A.settings.groq_api_key, "gemini_api_key": gk or A.settings.gemini_api_key,
                    "openai_api_key": ok or A.settings.openai_api_key,
                    "anthropic_api_key": ak or A.settings.anthropic_api_key,
                    "_env": {"GROQ_API_KEY": qk, "GEMINI_API_KEY": gk, "OPENAI_API_KEY": ok, "ANTHROPIC_API_KEY": ak},
                }
                _build_assistant.clear()
                st.rerun()

    with t3:
        q = st.text_input("Search your chats", placeholder="e.g. overfitting")
        if q:
            for m in A.memory.search_messages(q)[:20]:
                st.markdown(f"**{m['title']}** · {friendly_date(m['created_at'])}  \n{m['content'][:240]}…")
            st.divider()
        convs = [c for c in A.memory.list_conversations() if c["n_messages"]]
        if not convs:
            st.caption("No chats yet.")
        for c in convs:
            msgs = A.memory.get_messages(c["id"])
            transcript = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in msgs)
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
                c1.markdown(f"**{c['title'] or 'Untitled chat'}**  \n<span class='doc-meta'>{c['n_messages'] // 2} questions · "
                            f"{friendly_date(c['updated_at'])}</span>", unsafe_allow_html=True)
                if c2.button("Open", key=f"open_{c['id']}", use_container_width=True):
                    go(PAGES[1], conversation_id=c["id"])
                c3.download_button("Export", transcript, f"chat-{c['id']}.md", key=f"exp_{c['id']}",
                                   use_container_width=True)
                if c4.button("Delete", key=f"delc_{c['id']}", use_container_width=True):
                    A.memory.delete_conversation(c["id"])
                    st.rerun()

    with t4:
        st.markdown("Check how well the assistant finds the right pages and sticks to your documents.")
        up = st.file_uploader("Question set (JSON). Leave empty to use the sample set with the sample notes.",
                              type=["json"])
        if st.button("Run quality check", type="primary"):
            items = json.loads(up.getvalue()) if up else json.loads((SAMPLE_DIR / "eval_set.json").read_text())
            with st.spinner(f"Checking {len(items)} questions…"):
                st.session_state.eval = run_full_evaluation(A, items, 5, not A.llm.is_offline)
        ev = st.session_state.get("eval")
        if ev:
            r, a = ev["retrieval"], ev["answers"]
            pct = lambda v: f"{v:.0%}" if isinstance(v, (int, float)) else "–"  # noqa: E731
            m = st.columns(4)
            m[0].metric("Found the right document", pct(r.get("hit_rate@5")))
            m[1].metric("Sentences with a citation", pct(a["citation_coverage"]))
            m[2].metric("Unsupported sentences", pct(a["hallucination_rate"]))
            m[3].metric("Answer quality (1–5)", a["response_quality_1to5"] or "–")
            with st.expander("Full results"):
                st.dataframe(pd.DataFrame(r["rows"]), hide_index=True, use_container_width=True)
                st.dataframe(pd.DataFrame(a["rows"]), hide_index=True, use_container_width=True)
            st.download_button("Download report", json.dumps(ev, indent=2, default=str), "quality-check.json")

    with t5:
        st.markdown(f"**{status['documents']}** documents, **{status['chunks']}** searchable passages.")
        if st.button("Refresh all documents", help="Re-reads every document. Needed after changing the search model."):
            bar = st.progress(0.0)
            n = A.reindex_all(progress=lambda p, name: bar.progress(p, text=name))
            st.success(f"Refreshed {n} documents.")
        with st.expander("Danger zone"):
            st.caption("Removes every document from your library. Your chats are kept.")
            if st.checkbox("I understand this can't be undone"):
                if st.button("Remove all documents", type="primary"):
                    for d in A.memory.list_documents():
                        A.delete_document(d["doc_id"])
                    A.store.reset()
                    st.rerun()


{PAGES[0]: page_home, PAGES[1]: page_ask, PAGES[2]: page_library, PAGES[3]: page_study, PAGES[4]: page_settings}[page]()
