"""
app.py
======
RAG Studio — Frontend (Streamlit UI)
Imports all logic from backend.py — zero business logic lives here.
"""

import streamlit as st
from backend import (
    CHUNK_STRATEGIES,
    SUPPORTED_EMBED_MODELS,
    SUPPORTED_MODELS,
    build_vectorstore,
    get_llm,
    load_file,
    rag_answer,
)

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS — "Research Terminal" aesthetic
#  Monospace + neon-green on near-black, tight grid, data-dense panels
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #080b10;
    color: #d4dbe8;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0e1218; }
::-webkit-scrollbar-thumb { background: #1e3a2f; border-radius: 2px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070a0f 0%, #080d12 100%);
    border-right: 1px solid #0f2318;
    width: 290px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="stSidebar"] * { color: #8fa3b1 !important; }
[data-testid="stSidebar"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3d6b4f !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0c1419 !important;
    border: 1px solid #142b1e !important;
    border-radius: 6px !important;
    color: #a8c5b5 !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #0c1419 !important;
    border: 1px solid #142b1e !important;
    color: #a8c5b5 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #2ef87d !important;
    box-shadow: 0 0 0 2px rgba(46,248,125,0.08) !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    padding: 0 !important;
}

/* ── Main area ── */
.main .block-container {
    background: #080b10;
    padding: 2rem 2.5rem 4rem 2.5rem;
    max-width: 1200px;
}

/* ── Masthead ── */
.masthead {
    display: flex;
    align-items: flex-end;
    gap: 0;
    margin-bottom: 0.15rem;
    line-height: 1;
}
.masthead-rag {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #2ef87d;
    letter-spacing: -0.03em;
    text-shadow: 0 0 40px rgba(46,248,125,0.25);
}
.masthead-studio {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.4rem;
    font-weight: 300;
    color: #cdd8e3;
    letter-spacing: -0.02em;
    margin-left: 10px;
}
.masthead-version {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #2a4d38;
    margin-left: 10px;
    margin-bottom: 8px;
    letter-spacing: 0.1em;
}
.masthead-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #2e5c42;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Section label ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #2a6644;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #0f2a1c, transparent);
}

/* ── Panel cards ── */
.panel {
    background: #0c1118;
    border: 1px solid #111d27;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    position: relative;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #2ef87d22, transparent);
    border-radius: 10px 10px 0 0;
}

/* ── Status badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-ready   { background: #071a0e; color: #2ef87d; border: 1px solid #1a5c30; }
.badge-idle    { background: #101520; color: #4a6880; border: 1px solid #1a2b3a; }
.badge-info    { background: #0b1828; color: #4a90d9; border: 1px solid #1a3a60; }
.badge-warn    { background: #1a1200; color: #e0a030; border: 1px solid #5a3e00; }
.badge-dot-green::before { content:'●'; font-size: 0.55rem; color:#2ef87d; animation: pulse 2s infinite; }
.badge-dot-gray::before  { content:'●'; font-size: 0.55rem; color:#2a4050; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── File pills ── */
.file-pills { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 0.7rem; }
.file-pill {
    background: #0d1a12;
    border: 1px solid #163322;
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a9e6a;
    letter-spacing: 0.03em;
}

/* ── Process button ── */
.stButton > button {
    background: linear-gradient(135deg, #0e3d22 0%, #145030 100%) !important;
    color: #2ef87d !important;
    border: 1px solid #1e6638 !important;
    border-radius: 7px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    padding: 0.5rem 1.4rem !important;
    text-transform: uppercase;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(46,248,125,0.06) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #144d2a 0%, #1a6038 100%) !important;
    border-color: #2ef87d !important;
    box-shadow: 0 0 25px rgba(46,248,125,0.18) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Info box ── */
.info-box {
    background: #08141e;
    border: 1px solid #0f2a3a;
    border-left: 3px solid #1a5580;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #4a7a9b;
    line-height: 1.7;
}
.info-box strong { color: #6aaece; }

/* ── Q&A chat cards ── */
.qa-wrap {
    margin-bottom: 1.2rem;
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.qa-card {
    background: #0c1118;
    border: 1px solid #111d27;
    border-radius: 10px;
    overflow: hidden;
}
.qa-question {
    background: #0a1610;
    border-bottom: 1px solid #111d27;
    padding: 0.9rem 1.2rem;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.qa-q-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #2ef87d;
    background: #0e2a18;
    border: 1px solid #1e5030;
    border-radius: 3px;
    padding: 2px 6px;
    flex-shrink: 0;
    margin-top: 1px;
    letter-spacing: 0.06em;
}
.qa-q-text {
    font-size: 0.88rem;
    color: #cdd8e3;
    font-weight: 500;
    line-height: 1.5;
}
.qa-answer {
    padding: 1rem 1.2rem;
}
.qa-a-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #4a6880;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.qa-a-text {
    font-size: 0.87rem;
    color: #b8c8d8;
    line-height: 1.75;
    white-space: pre-wrap;
}
.qa-footer {
    padding: 0.6rem 1.2rem;
    border-top: 1px solid #0d1822;
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
}
.qa-meta-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #2a4050;
    letter-spacing: 0.06em;
}
.qa-meta-tag span { color: #3d6070; }

/* ── Chunk box ── */
.chunk-box {
    background: #080d12;
    border: 1px solid #0f1e2a;
    border-left: 3px solid #1a5030;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #4a7060;
    line-height: 1.65;
    white-space: pre-wrap;
}
.chunk-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #2a6040;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}

/* ── Sidebar logo block ── */
.sidebar-logo {
    padding: 1.4rem 1.4rem 0.8rem 1.4rem;
    border-bottom: 1px solid #0f2318;
    margin-bottom: 0.6rem;
}
.sidebar-logo-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #2ef87d !important;
    letter-spacing: -0.02em;
}
.sidebar-logo-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: #1e4d30 !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Sidebar section header ── */
.sb-section {
    padding: 0.8rem 1.4rem 0.3rem 1.4rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: #1e4d30 !important;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    border-top: 1px solid #0a1e13;
}

/* ── Ask input ── */
.stTextInput input {
    background: #0c1520 !important;
    border: 1px solid #162435 !important;
    color: #cdd8e3 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #2ef87d !important;
    box-shadow: 0 0 0 3px rgba(46,248,125,0.07) !important;
}
.stTextInput input::placeholder { color: #2a4050 !important; }

/* ── Divider ── */
.h-rule {
    border: none;
    border-top: 1px solid #0e1a22;
    margin: 1.4rem 0;
}

/* ── History counter ── */
.history-counter {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #2a4050;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #1e3040;
}
.empty-state-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.empty-state-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
}

/* ── Status bar (after processing) ── */
.status-bar {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 1.2rem;
    padding: 0.8rem 1rem;
    background: #080e12;
    border: 1px solid #0f1e28;
    border-radius: 8px;
}

/* Streamlit override cleanups */
.stFileUploader { border-radius: 8px; }
[data-testid="stFileUploader"] section {
    background: #0a1018 !important;
    border: 1px dashed #162435 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #2ef87d55 !important;
}
[data-testid="stExpander"] {
    background: #09111a !important;
    border: 1px solid #0f1e28 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #2a5040 !important;
    letter-spacing: 0.06em;
}
div[data-testid="stAlert"] {
    background: #0a1610 !important;
    border-color: #1a4028 !important;
    color: #4a9a60 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────────
defaults = {
    "vectorstore": None,
    "chat_history": [],
    "processed_files": [],
    "total_chunks": 0,
    "embedding_model": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-title">⬡ RAG Studio</div>
        <div class="sidebar-logo-sub">Document Intelligence System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">⬡ API Configuration</div>', unsafe_allow_html=True)
    groq_api_key = st.text_input(
        "GROQ API KEY",
        type="password",
        placeholder="gsk_...",
        help="Get your key at console.groq.com",
    )
    groq_model = st.selectbox("MODEL", SUPPORTED_MODELS, index=0)
    temperature = st.slider("TEMPERATURE", 0.0, 1.0, 0.0, 0.05)

    st.markdown('<div class="sb-section">⬡ Chunking Strategy</div>', unsafe_allow_html=True)
    chunk_strategy = st.selectbox(
        "STRATEGY",
        CHUNK_STRATEGIES,
        index=0,
        help="Recursive: paragraphs→sentences→words\nCharacter: fixed character count\nSemantic: embedding-similarity based",
    )
    if chunk_strategy != "Sentence (Semantic)":
        chunk_size    = st.slider("CHUNK SIZE", 100, 2000, 500, 50)
        chunk_overlap = st.slider("CHUNK OVERLAP", 0, 500, 50, 10)
    else:
        st.markdown(
            '<div class="info-box">Semantic chunking uses embedding similarity — no fixed size required.</div>',
            unsafe_allow_html=True,
        )
        chunk_size, chunk_overlap = 500, 50

    st.markdown('<div class="sb-section">⬡ Embedding Model</div>', unsafe_allow_html=True)
    embed_model_name = st.selectbox("HUGGINGFACE MODEL", SUPPORTED_EMBED_MODELS, index=0)

    st.markdown('<div class="sb-section">⬡ Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("TOP-K DOCUMENTS", 1, 20, 10, 1)

    st.markdown("---")
    if st.button("⬡ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN — MASTHEAD
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <span class="masthead-rag">RAG</span>
    <span class="masthead-studio">Studio</span>
    <span class="masthead-version">v2.0</span>
</div>
<div class="masthead-sub">// Retrieval-Augmented Generation · Document Intelligence</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  DOCUMENT UPLOAD PANEL
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 — Document Ingestion</div>', unsafe_allow_html=True)

with st.container():
    col_up, col_proc = st.columns([3, 1], gap="medium")

    with col_up:
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt", "csv", "xlsx", "html", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="PDF · DOCX · TXT · CSV · XLSX · HTML · Markdown",
        )

    with col_proc:
        st.write("")  # vertical align
        process_btn = st.button("⚡ Process Files", use_container_width=True)
        if st.session_state.vectorstore:
            st.markdown(
                f'<div style="margin-top:8px; display:flex; gap:6px; flex-wrap:wrap;">'
                f'<span class="badge badge-ready badge-dot-green">Ready</span>'
                f'<span class="badge badge-info">{st.session_state.total_chunks} chunks</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="margin-top:8px;">'
                '<span class="badge badge-idle badge-dot-gray">Not Processed</span>'
                '</div>',
                unsafe_allow_html=True,
            )

# Show file pills
if uploaded_files:
    pills = "".join([f'<span class="file-pill">📄 {f.name}</span>' for f in uploaded_files])
    st.markdown(f'<div class="file-pills">{pills}</div>', unsafe_allow_html=True)

st.markdown('<hr class="h-rule">', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  PROCESS FILES LOGIC
# ──────────────────────────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.warning("Please upload at least one file first.")
    elif not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Loading documents & building FAISS index…"):
            all_docs, failed = [], []
            for uf in uploaded_files:
                try:
                    all_docs.extend(load_file(uf))
                except Exception:
                    failed.append(uf.name)

            if all_docs:
                try:
                    vs, n_chunks, emb = build_vectorstore(
                        all_docs, chunk_strategy, chunk_size, chunk_overlap, embed_model_name
                    )
                    st.session_state.vectorstore     = vs
                    st.session_state.total_chunks    = n_chunks
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.session_state.embedding_model  = emb
                    st.success(
                        f"✓ {len(uploaded_files)} file(s) processed → "
                        f"{n_chunks} chunks → FAISS index ready"
                    )
                except Exception as e:
                    st.error(f"Error building vectorstore: {e}")
            else:
                st.error("Could not load any documents. Check file formats.")

            if failed:
                st.warning(f"Failed to load: {', '.join(failed)}")

        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  CHAT / Q&A SECTION
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="info-box">
        <strong>Getting started —</strong>
        Upload your documents above, add your Groq API key in the sidebar,
        then click <strong>⚡ Process Files</strong> to build the knowledge base.
        Once ready, type any question to query your documents.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Status bar ────────────────────────────────────────────────────────────
    files_str = " · ".join(st.session_state.processed_files)
    st.markdown(
        f"""
        <div class="status-bar">
            <span class="badge badge-ready badge-dot-green">Vectorstore Ready</span>
            <span class="badge badge-info">FAISS · {st.session_state.total_chunks} chunks</span>
            <span class="badge badge-warn">{chunk_strategy}</span>
            <span class="qa-meta-tag" style="margin-left:4px;">— <span>{files_str}</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Question input ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">02 — Query Interface</div>', unsafe_allow_html=True)

    col_q, col_btn = st.columns([5, 1], gap="small")
    with col_q:
        user_question = st.text_input(
            "Question",
            placeholder="What does the document say about…?",
            label_visibility="collapsed",
            key="question_input",
        )
    with col_btn:
        ask_btn = st.button("ASK →", use_container_width=True)

    # ── Handle question ───────────────────────────────────────────────────────
    if ask_btn and user_question.strip():
        if not groq_api_key:
            st.warning("Add your Groq API key in the sidebar.")
        else:
            with st.spinner("Retrieving context & generating answer…"):
                try:
                    llm = get_llm(groq_api_key, groq_model, temperature)
                    answer, top2_chunks = rag_answer(
                        user_question,
                        st.session_state.vectorstore,
                        llm,
                        k=top_k,
                    )
                    st.session_state.chat_history.insert(0, {
                        "question": user_question,
                        "answer":   answer,
                        "chunks":   top2_chunks,
                        "model":    groq_model,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
            st.rerun()

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">03 — Conversation History</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        total = len(st.session_state.chat_history)
        st.markdown(
            f'<div class="history-counter">{total} exchange{"s" if total > 1 else ""} · most recent first</div>',
            unsafe_allow_html=True,
        )

        for i, item in enumerate(st.session_state.chat_history):
            q_num = total - i
            st.markdown(f"""
            <div class="qa-wrap">
                <div class="qa-card">
                    <div class="qa-question">
                        <span class="qa-q-label">Q{q_num}</span>
                        <span class="qa-q-text">{item['question']}</span>
                    </div>
                    <div class="qa-answer">
                        <div class="qa-a-label">// Answer</div>
                        <div class="qa-a-text">{item['answer']}</div>
                    </div>
                    <div class="qa-footer">
                        <span class="qa-meta-tag">model: <span>{item['model']}</span></span>
                        <span class="qa-meta-tag">·</span>
                        <span class="qa-meta-tag">chunks retrieved: <span>{len(item['chunks'])}</span></span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"📎 Top 2 source chunks — Q{q_num}"):
                for j, chunk in enumerate(item["chunks"]):
                    st.markdown(
                        f'<div class="chunk-label">Chunk {j+1} · {chunk["filename"]}</div>',
                        unsafe_allow_html=True,
                    )
                    preview = chunk["content"][:600]
                    if len(chunk["content"]) > 600:
                        preview += "…"
                    st.markdown(f'<div class="chunk-box">{preview}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">⬡</div>
            <div class="empty-state-text">Ask your first question above</div>
        </div>
        """, unsafe_allow_html=True)
