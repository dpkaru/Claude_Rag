"""
app.py
======
RAG Studio — Frontend (Streamlit UI)
Warm rose-cream aesthetic with activity stream, file cards, and status panels.
All logic imported from backend.py.
"""

import datetime
import streamlit as st
from backend import (
    CHUNK_STRATEGIES,
    SUPPORTED_EMBED_MODELS,
    SUPPORTED_MODELS,
    build_vectorstore,
    get_groq_api_key,
    get_llm,
    load_file,
    rag_answer,
)

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  CSS — Warm Rose / Cream aesthetic
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --cream:      #fdf6f0;
    --rose-light: #f9ede8;
    --rose-mid:   #e8c4bc;
    --rose-warm:  #c97b6e;
    --rose-deep:  #a85a50;
    --terracotta: #b86455;
    --blush:      #f2d8d2;
    --text-dark:  #3d2820;
    --text-mid:   #7a4f47;
    --text-soft:  #b08880;
    --border:     #e5cdc8;
    --card-bg:    #fdf0ec;
    --white:      #ffffff;
    --teal:       #4a9ba8;
    --teal-light: #d4eff3;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--cream);
    color: var(--text-dark);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--rose-light); }
::-webkit-scrollbar-thumb { background: var(--rose-mid); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2e1a16 0%, #3d2218 50%, #2a1a12 100%) !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #d4b8b0 !important; }
[data-testid="stSidebar"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #8a5a50 !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #1e0f0c !important;
    border: 1px solid #5a2e28 !important;
    color: #e8c4bc !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.76rem !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #c97b6e !important;
    box-shadow: 0 0 0 2px rgba(201,123,110,0.2) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e0f0c !important;
    border: 1px solid #5a2e28 !important;
    color: #e8c4bc !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: #5a2e28 !important;
}

/* ── Main background ── */
.main .block-container {
    background: var(--cream);
    padding: 2rem 2.5rem 4rem 2.5rem;
    max-width: 1300px;
}

/* ── Masthead ── */
.masthead-wrap {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 2rem;
}
.masthead-left {}
.masthead-icon {
    font-size: 2rem;
    margin-bottom: 0.2rem;
    opacity: 0.7;
}
.masthead-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 600;
    color: var(--text-dark);
    line-height: 1;
    margin-bottom: 0.3rem;
}
.masthead-title .accent { color: var(--rose-warm); }
.masthead-sub {
    font-size: 0.82rem;
    color: var(--text-soft);
    letter-spacing: 0.04em;
}

/* ── Section headers ── */
.section-hdr {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--text-soft);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-hdr::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Upload card ── */
.upload-card {
    background: var(--white);
    border: 1.5px dashed var(--rose-mid);
    border-radius: 14px;
    padding: 0.5rem 1rem;
    transition: border-color 0.2s;
}
.upload-card:hover { border-color: var(--rose-warm); }

/* ── Process button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--terracotta), var(--rose-deep)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 15px rgba(184,100,85,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 22px rgba(184,100,85,0.45) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 13px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.pill-ready   { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
.pill-idle    { background: var(--blush); color: var(--text-soft); border: 1px solid var(--rose-mid); }
.pill-chunks  { background: var(--teal-light); color: var(--teal); border: 1px solid #a0d8e0; }

/* ── File cards (processed files) ── */
.file-cards { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 1rem; }
.file-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 160px;
    box-shadow: 0 2px 8px rgba(180,100,80,0.06);
}
.file-card-icon {
    width: 32px;
    height: 32px;
    background: var(--blush);
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    flex-shrink: 0;
}
.file-card-name {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-dark);
    word-break: break-all;
}
.file-card-status {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-soft);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Activity stream ── */
.activity-panel {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    height: 100%;
}
.activity-hdr {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-soft);
    margin-bottom: 0.8rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}
.activity-line {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 0.55rem;
}
.act-time {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-soft);
    flex-shrink: 0;
    margin-top: 1px;
}
.act-msg {
    font-size: 0.78rem;
    color: var(--text-mid);
    line-height: 1.4;
}
.act-ok   { color: #2e9e50; font-weight: 600; }
.act-warn { color: var(--rose-warm); font-weight: 600; }
.act-info { color: var(--teal); font-weight: 600; }

/* ── Stats panel ── */
.stats-panel {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
}
.stats-hdr {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-soft);
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-soft);
}
.stat-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dark);
    font-weight: 500;
}
.stat-bar-wrap {
    width: 100%;
    height: 5px;
    background: var(--rose-light);
    border-radius: 3px;
    margin-top: 3px;
    margin-bottom: 0.8rem;
}
.stat-bar {
    height: 5px;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--rose-warm), var(--terracotta));
}

/* ── Info notice ── */
.notice-box {
    background: var(--rose-light);
    border: 1px solid var(--rose-mid);
    border-left: 4px solid var(--rose-warm);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: var(--text-mid);
    line-height: 1.6;
}
.notice-box strong { color: var(--text-dark); }

/* ── Divider ── */
.h-rule { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── Q&A cards ── */
.qa-wrap { margin-bottom: 1rem; animation: fadeUp 0.35s ease; }
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.qa-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(180,100,80,0.06);
}
.qa-question-bar {
    background: var(--rose-light);
    padding: 0.85rem 1.2rem;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    border-bottom: 1px solid var(--border);
}
.qa-q-badge {
    background: var(--rose-warm);
    color: #fff;
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 4px;
    flex-shrink: 0;
    margin-top: 2px;
    letter-spacing: 0.06em;
}
.qa-q-text {
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--text-dark);
    line-height: 1.5;
}
.qa-answer-area { padding: 1rem 1.2rem; }
.qa-a-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-soft);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.qa-a-text {
    font-size: 0.87rem;
    color: var(--text-mid);
    line-height: 1.75;
    white-space: pre-wrap;
}
.qa-footer {
    padding: 0.6rem 1.2rem;
    border-top: 1px solid var(--border);
    background: #fdf8f6;
    display: flex;
    gap: 14px;
}
.qa-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-soft);
}
.qa-meta b { color: var(--text-mid); }

/* ── Chunk boxes ── */
.chunk-hdr {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-soft);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.35rem;
}
.chunk-body {
    background: var(--rose-light);
    border: 1px solid var(--border);
    border-left: 3px solid var(--rose-warm);
    border-radius: 7px;
    padding: 0.75rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-mid);
    line-height: 1.65;
    white-space: pre-wrap;
    margin-bottom: 0.7rem;
}

/* ── Ask input ── */
.stTextInput input {
    background: var(--white) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-dark) !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput input:focus {
    border-color: var(--rose-warm) !important;
    box-shadow: 0 0 0 3px rgba(201,123,110,0.12) !important;
}
.stTextInput input::placeholder { color: var(--text-soft) !important; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--rose-mid);
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
.empty-text {
    font-size: 0.82rem;
    color: var(--text-soft);
    letter-spacing: 0.04em;
}

/* ── Sidebar logo ── */
.sb-logo {
    padding: 1.6rem 1.4rem 1rem 1.4rem;
    border-bottom: 1px solid #5a2e2820;
}
.sb-logo-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.15rem !important;
    color: #e8c4bc !important;
}
.sb-logo-sub {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.58rem !important;
    color: #7a4038 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Sidebar section header ── */
.sb-hdr {
    padding: 1rem 1.4rem 0.3rem 1.4rem;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.58rem !important;
    color: #8a5a50 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.16em !important;
    border-top: 1px solid #3a1a1420;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--text-soft) !important;
}

/* Streamlit alerts */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
defaults = {
    "vectorstore": None,
    "chat_history": [],
    "processed_files": [],
    "total_chunks": 0,
    "embedding_model": None,
    "activity_log": [],   # list of {"time": str, "msg": str, "kind": str}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def log_activity(msg: str, kind: str = "info"):
    """Append an entry to the activity stream."""
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.activity_log.insert(0, {"time": now, "msg": msg, "kind": kind})
    st.session_state.activity_log = st.session_state.activity_log[:20]  # keep last 20


# ──────────────────────────────────────────────────────────────────────────────
#  RESOLVE API KEY (auto from .env / secrets, fallback to sidebar input)
# ──────────────────────────────────────────────────────────────────────────────
auto_key = get_groq_api_key()


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-title">✦ RAG Studio</div>
        <div class="sb-logo-sub">Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-hdr">⬡ API Configuration</div>', unsafe_allow_html=True)

    if auto_key:
        st.markdown(
            '<div style="padding:0.4rem 1.4rem 0.2rem; font-size:0.72rem; color:#7ab87a;">'
            '✓ API key loaded from environment</div>',
            unsafe_allow_html=True,
        )
        groq_api_key = auto_key
        # Still show a masked display
        st.text_input("GROQ API KEY", value="••••••••••••••••••••", disabled=True,
                      label_visibility="collapsed")
    else:
        groq_api_key = st.text_input(
            "GROQ API KEY",
            type="password",
            placeholder="gsk_... (or add to .env / secrets.toml)",
            help="Priority: .env / secrets.toml → OS env → this field",
            label_visibility="collapsed",
        )

    groq_model = st.selectbox("MODEL", SUPPORTED_MODELS, index=0)
    temperature = st.slider("TEMPERATURE", 0.0, 1.0, 0.0, 0.05)

    st.markdown('<div class="sb-hdr">⬡ Chunking Strategy</div>', unsafe_allow_html=True)
    chunk_strategy = st.selectbox("STRATEGY", CHUNK_STRATEGIES, index=0,
        help="Recursive: paragraphs→sentences→words\nCharacter: fixed chars\nSemantic: embedding similarity")
    if chunk_strategy != "Sentence (Semantic)":
        chunk_size    = st.slider("CHUNK SIZE", 100, 2000, 500, 50)
        chunk_overlap = st.slider("CHUNK OVERLAP", 0, 500, 50, 10)
    else:
        st.info("Semantic chunking uses embedding similarity — no fixed size needed.")
        chunk_size, chunk_overlap = 500, 50

    st.markdown('<div class="sb-hdr">⬡ Embedding Model</div>', unsafe_allow_html=True)
    embed_model_name = st.selectbox("HUGGINGFACE MODEL", SUPPORTED_EMBED_MODELS, index=0)

    st.markdown('<div class="sb-hdr">⬡ Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("TOP-K DOCUMENTS", 1, 20, 10, 1)

    st.markdown("---")
    if st.button("✦ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        log_activity("Chat history cleared", "warn")
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  MASTHEAD
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead-wrap">
    <div class="masthead-left">
        <div class="masthead-icon">✦</div>
        <div class="masthead-title"><span class="accent">RAG</span> Studio</div>
        <div class="masthead-sub">Upload documents · Configure chunking · Ask anything</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  UPLOAD SECTION
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">01 · Document Upload</div>', unsafe_allow_html=True)

col_up, col_proc = st.columns([3, 1], gap="medium")

with col_up:
    uploaded_files = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt", "csv", "xlsx", "html", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="PDF · DOCX · TXT · CSV · XLSX · HTML · Markdown",
    )

with col_proc:
    st.write("")
    process_btn = st.button("⚡ Process Files", use_container_width=True)
    st.write("")
    if st.session_state.vectorstore:
        st.markdown(
            f'<span class="status-pill pill-ready">● Ready</span>&nbsp;'
            f'<span class="status-pill pill-chunks">{st.session_state.total_chunks} chunks</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill pill-idle">○ Not Processed</span>',
            unsafe_allow_html=True,
        )

# ── File cards ────────────────────────────────────────────────────────────────
if uploaded_files:
    ext_icons = {"pdf": "📄", "docx": "📝", "txt": "🗒️", "csv": "📊",
                 "xlsx": "📊", "html": "🌐", "md": "📋"}
    cards_html = '<div class="file-cards">'
    for f in uploaded_files:
        ext = f.name.rsplit(".", 1)[-1].lower()
        icon = ext_icons.get(ext, "📁")
        processed = f.name in st.session_state.processed_files
        status = "Processed" if processed else "Pending"
        status_color = "#2e7d32" if processed else "#b08880"
        cards_html += f"""
        <div class="file-card">
            <div class="file-card-icon">{icon}</div>
            <div>
                <div class="file-card-name">{f.name}</div>
                <div class="file-card-status" style="color:{status_color};">{status}</div>
            </div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

st.markdown('<hr class="h-rule">', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  PROCESS FILES LOGIC
# ──────────────────────────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.warning("Please upload at least one file first.")
    elif not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar (or add it to .env / secrets.toml).")
    else:
        log_activity("Processing started", "info")
        with st.spinner("Loading documents & building FAISS index…"):
            all_docs, failed = [], []
            for uf in uploaded_files:
                try:
                    all_docs.extend(load_file(uf))
                    log_activity(f"Loaded: {uf.name}", "ok")
                except Exception as e:
                    failed.append(uf.name)
                    log_activity(f"Failed: {uf.name}", "warn")

            if all_docs:
                try:
                    vs, n_chunks, emb = build_vectorstore(
                        all_docs, chunk_strategy, chunk_size, chunk_overlap, embed_model_name
                    )
                    st.session_state.vectorstore     = vs
                    st.session_state.total_chunks    = n_chunks
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.session_state.embedding_model  = emb
                    log_activity(f"FAISS index built — {n_chunks} chunks", "ok")
                    log_activity(f"Model: {groq_model}", "info")
                    st.success(f"✓ {len(uploaded_files)} file(s) → {n_chunks} chunks → FAISS index ready!")
                except Exception as e:
                    log_activity(f"Vectorstore error: {e}", "warn")
                    st.error(f"Error building vectorstore: {e}")
            else:
                st.error("Could not load any documents. Check file formats.")
            if failed:
                st.warning(f"Failed: {', '.join(failed)}")
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  ACTIVITY STREAM + STATS  (always shown)
# ──────────────────────────────────────────────────────────────────────────────
col_act, col_stats = st.columns([3, 1], gap="medium")

with col_act:
    kind_cls = {"ok": "act-ok", "warn": "act-warn", "info": "act-info"}
    kind_prefix = {"ok": "OK", "warn": "WARN", "info": "INFO"}

    lines_html = ""
    if st.session_state.activity_log:
        for entry in st.session_state.activity_log[:8]:
            cls = kind_cls.get(entry["kind"], "act-info")
            prefix = kind_prefix.get(entry["kind"], "INFO")
            lines_html += f"""
            <div class="activity-line">
                <span class="act-time">[{entry['time']}]</span>
                <span class="act-msg"><span class="{cls}">{prefix}:</span> {entry['msg']}</span>
            </div>"""
    else:
        lines_html = '<div class="act-msg" style="color:#c8a89a; font-size:0.78rem;">No activity yet — upload and process files to begin.</div>'

    st.markdown(f"""
    <div class="activity-panel">
        <div class="activity-hdr">⬡ Activity Stream</div>
        {lines_html}
    </div>
    """, unsafe_allow_html=True)

with col_stats:
    chunks     = st.session_state.total_chunks
    files_done = len(st.session_state.processed_files)
    qa_count   = len(st.session_state.chat_history)
    chunk_pct  = min(int((chunks / 2000) * 100), 100)  # visual only

    st.markdown(f"""
    <div class="stats-panel">
        <div class="stats-hdr">⬡ Index Stats</div>
        <div class="stat-row">
            <span class="stat-label">Files</span>
            <span class="stat-val">{files_done}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Chunks</span>
            <span class="stat-val">{chunks}</span>
        </div>
        <div class="stat-bar-wrap">
            <div class="stat-bar" style="width:{chunk_pct}%"></div>
        </div>
        <div class="stat-row">
            <span class="stat-label">Strategy</span>
            <span class="stat-val">{chunk_strategy.split()[0]}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Q&A Pairs</span>
            <span class="stat-val">{qa_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="h-rule">', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  CHAT / Q&A
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="notice-box">
        <strong>Getting started —</strong>
        Upload your documents above, add your Groq API key in the sidebar
        (or set <code>GROQ_API_KEY</code> in your <code>.env</code> or
        <code>.streamlit/secrets.toml</code>), then click
        <strong>⚡ Process Files</strong> to build the knowledge base.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="section-hdr">02 · Query Interface</div>', unsafe_allow_html=True)

    col_q, col_btn = st.columns([5, 1], gap="small")
    with col_q:
        user_question = st.text_input(
            "Question",
            placeholder="Ask anything about your documents…",
            label_visibility="collapsed",
            key="question_input",
        )
    with col_btn:
        ask_btn = st.button("Ask →", use_container_width=True)

    if ask_btn and user_question.strip():
        if not groq_api_key:
            st.warning("Add your Groq API key in the sidebar.")
        else:
            with st.spinner("Retrieving context & generating answer…"):
                try:
                    llm = get_llm(groq_api_key, groq_model, temperature)
                    answer, top2_chunks = rag_answer(
                        user_question, st.session_state.vectorstore, llm, k=top_k
                    )
                    st.session_state.chat_history.insert(0, {
                        "question": user_question,
                        "answer":   answer,
                        "chunks":   top2_chunks,
                        "model":    groq_model,
                    })
                    log_activity(f"Query answered: {user_question[:40]}…", "ok")
                except Exception as e:
                    log_activity(f"Query error: {e}", "warn")
                    st.error(f"Error: {e}")
            st.rerun()

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">03 · Conversation History</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        total = len(st.session_state.chat_history)
        for i, item in enumerate(st.session_state.chat_history):
            q_num = total - i
            st.markdown(f"""
            <div class="qa-wrap">
                <div class="qa-card">
                    <div class="qa-question-bar">
                        <span class="qa-q-badge">Q{q_num}</span>
                        <span class="qa-q-text">{item['question']}</span>
                    </div>
                    <div class="qa-answer-area">
                        <div class="qa-a-label">// Response</div>
                        <div class="qa-a-text">{item['answer']}</div>
                    </div>
                    <div class="qa-footer">
                        <span class="qa-meta">model: <b>{item['model']}</b></span>
                        <span class="qa-meta">chunks retrieved: <b>{len(item['chunks'])}</b></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"📎 Source chunks — Q{q_num}"):
                for j, chunk in enumerate(item["chunks"]):
                    st.markdown(
                        f'<div class="chunk-hdr">Chunk {j+1} · {chunk["filename"]}</div>',
                        unsafe_allow_html=True,
                    )
                    preview = chunk["content"][:600] + ("…" if len(chunk["content"]) > 600 else "")
                    st.markdown(f'<div class="chunk-body">{preview}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <div class="empty-text">Ask your first question above</div>
        </div>
        """, unsafe_allow_html=True)
