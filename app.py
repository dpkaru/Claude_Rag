"""
app.py
======
RAG Studio — Frontend (Streamlit UI)
Matches the warm cream/rose mockup exactly.
API key loaded from backend (env / secrets) — no UI input needed.
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
#  CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:         #f5ede6;
    --bg2:        #f0e4db;
    --card:       #faf3ef;
    --white:      #ffffff;
    --rose:       #c8756a;
    --rose-light: #e8c4bc;
    --rose-pale:  #f2dbd6;
    --rose-deep:  #a05548;
    --border:     #e0ccc6;
    --border2:    #d4b8b0;
    --text1:      #2e1a14;
    --text2:      #7a4f47;
    --text3:      #b08880;
    --teal:       #5aabb8;
    --teal-bg:    #e0f3f7;
    --green:      #5a9e6a;
    --green-bg:   #e4f2e8;
    --sidebar-bg: #1e0f0b;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: var(--bg) !important; }
.main .block-container {
    background: transparent !important;
    padding: 1.5rem 2rem 4rem 2rem !important;
    max-width: 1280px !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--rose-light); border-radius: 4px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid #3a1a1240 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] * { color: #c8a89a !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    color: #7a4a40 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #2e1410 !important;
    border: 1px solid #5a2a22 !important;
    border-radius: 8px !important;
    color: #e0c0b8 !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] hr { border-color: #3a1a1450 !important; margin: 0.6rem 0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #2e1410 !important;
    border: 1px solid #6a3028 !important;
    color: #c8a89a !important;
    font-size: 0.75rem !important;
    border-radius: 8px !important;
    padding: 0.45rem 1rem !important;
    box-shadow: none !important;
    transform: none !important;
    font-weight: 400 !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #3e1e16 !important;
    border-color: var(--rose) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── HERO ── */
.hero {
    position: relative;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 1.6rem;
    min-height: 140px;
    overflow: hidden;
}
.hero-left { z-index: 2; }
.hero-logo-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
}
.hero-icon-box {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #c8756a, #a05548);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 4px 16px rgba(200,117,106,0.35);
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem; font-weight: 700; line-height: 1;
    color: var(--text1); letter-spacing: -0.01em;
}
.hero-title .rag { color: var(--rose); }
.hero-subtitle { font-size: 0.82rem; color: var(--text3); letter-spacing: 0.04em; margin-top: 6px; }
.hero-graphic {
    position: absolute;
    right: -10px; top: -25px;
    width: 360px; height: 200px;
    pointer-events: none; z-index: 1; opacity: 0.9;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; text-transform: uppercase;
    letter-spacing: 0.18em; color: var(--text3);
    margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 10px;
}
.sec-label::after { content:''; flex:1; height:1px; background: linear-gradient(90deg,var(--border),transparent); }

/* ── UPLOAD TABS ── */
.upload-tabs {
    display: flex; gap: 8px; margin-top: 0.6rem;
    padding-top: 0.6rem; border-top: 1px solid var(--border);
}
.utab { display:flex; align-items:center; gap:6px; font-size:0.72rem; color:var(--text3); }
.utab-toggle {
    width: 30px; height: 15px;
    background: var(--rose-pale); border-radius: 8px;
    position: relative;
}
.utab-toggle::after {
    content:''; position:absolute;
    width:13px; height:13px; background:var(--rose);
    border-radius:50%; top:1px; right:1px;
}

/* ── PROCESS BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #c8756a 0%, #a05548 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 4px 18px rgba(200,117,106,0.4) !important;
    transition: all 0.2s !important; letter-spacing: 0.02em !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #d4857a 0%, #b06058 100%) !important;
    box-shadow: 0 6px 24px rgba(200,117,106,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── STATUS PILLS ── */
.pill-not { background:var(--rose-pale); border:1px solid var(--rose-light); border-radius:20px; padding:4px 14px; font-family:'DM Mono',monospace; font-size:0.62rem; color:var(--rose-deep); text-transform:uppercase; letter-spacing:0.08em; display:inline-block; margin-top:6px; }
.pill-ok  { background:var(--green-bg); border:1px solid #a0d4a8; border-radius:20px; padding:4px 14px; font-family:'DM Mono',monospace; font-size:0.62rem; color:var(--green); text-transform:uppercase; letter-spacing:0.08em; display:inline-block; margin-top:6px; }
.pill-cnt { background:var(--teal-bg); border:1px solid #a0d8e4; border-radius:20px; padding:4px 14px; font-family:'DM Mono',monospace; font-size:0.62rem; color:var(--teal); text-transform:uppercase; letter-spacing:0.08em; display:inline-block; margin-top:6px; margin-left:4px; }

/* ── FILE CARDS ── */
.file-cards-row { display:flex; flex-wrap:wrap; gap:10px; margin-top:1rem; }
.fcard {
    background:var(--white); border:1px solid var(--border); border-radius:12px;
    padding:10px 14px; display:flex; align-items:center; gap:10px;
    box-shadow:0 2px 10px rgba(180,100,80,0.07); min-width:180px;
}
.fcard-icon {
    width:36px; height:36px; border-radius:9px;
    background:linear-gradient(135deg,#f2dbd6,#e8c4bc);
    display:flex; align-items:center; justify-content:center;
    font-size:0.9rem; flex-shrink:0; position:relative;
}
.fcard-badge {
    position:absolute; top:-4px; right:-4px;
    background:var(--rose); color:#fff;
    font-family:'DM Mono',monospace; font-size:0.45rem;
    padding:1px 4px; border-radius:3px; letter-spacing:0.04em; text-transform:uppercase;
}
.fcard-name { font-size:0.78rem; font-weight:500; color:var(--text1); word-break:break-all; line-height:1.3; }
.fcard-status { font-family:'DM Mono',monospace; font-size:0.58rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:2px; }

/* ── PANELS ── */
.panel-card {
    background:var(--white); border:1px solid var(--border);
    border-radius:14px; padding:1.1rem 1.3rem; height:100%;
}
.panel-hdr {
    font-family:'DM Mono',monospace; font-size:0.62rem;
    text-transform:uppercase; letter-spacing:0.14em; color:var(--text3);
    padding-bottom:0.6rem; border-bottom:1px solid var(--border); margin-bottom:0.8rem;
    display:flex; align-items:center; justify-content:space-between;
}

/* Activity */
.act-line { display:flex; align-items:baseline; gap:8px; margin-bottom:0.5rem; font-family:'DM Mono',monospace; font-size:0.71rem; line-height:1.5; }
.act-time { color:var(--text3); flex-shrink:0; }
.act-msg  { color:var(--text2); }
.act-ok   { color:#3a9e5a; font-weight:600; }
.act-warn { color:var(--rose); font-weight:600; }
.act-info { color:var(--teal); font-weight:600; }
.act-empty { font-family:'DM Mono',monospace; font-size:0.7rem; color:var(--text3); text-align:center; padding:1rem 0; }

/* Resource */
.res-legend { display:flex; flex-direction:column; gap:8px; margin-bottom:1rem; }
.res-row { display:flex; align-items:center; justify-content:space-between; gap:8px; }
.res-dot-wrap { display:flex; align-items:center; gap:7px; }
.res-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
.res-label { font-size:0.75rem; color:var(--text2); }
.res-val { font-family:'DM Mono',monospace; font-size:0.7rem; color:var(--text3); }
.donut-wrap { display:flex; justify-content:center; margin-top:0.5rem; }

/* ── NOTICE ── */
.notice {
    background:var(--rose-pale); border:1px solid var(--rose-light);
    border-left:4px solid var(--rose); border-radius:10px;
    padding:0.85rem 1.1rem; font-size:0.8rem; color:var(--text2); line-height:1.65;
}
.notice strong { color:var(--text1); }

/* ── HR ── */
.hr { border:none; border-top:1px solid var(--border); margin:1.3rem 0; }

/* ── Q&A ── */
.stTextInput input {
    background:var(--white) !important; border:1.5px solid var(--border) !important;
    border-radius:10px !important; color:var(--text1) !important;
    font-size:0.9rem !important; font-family:'DM Sans',sans-serif !important;
}
.stTextInput input:focus {
    border-color:var(--rose) !important;
    box-shadow:0 0 0 3px rgba(200,117,106,0.12) !important;
}
.stTextInput input::placeholder { color:var(--text3) !important; }

.qa-card {
    background:var(--white); border:1px solid var(--border); border-radius:14px;
    overflow:hidden; margin-bottom:1rem; box-shadow:0 2px 12px rgba(180,100,80,0.06);
    animation:fadeUp 0.3s ease;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
.qa-qrow { background:var(--rose-pale); padding:0.85rem 1.2rem; display:flex; align-items:flex-start; gap:10px; border-bottom:1px solid var(--border); }
.qa-qbadge { background:var(--rose); color:#fff; font-family:'DM Mono',monospace; font-size:0.55rem; padding:2px 7px; border-radius:4px; flex-shrink:0; margin-top:2px; letter-spacing:0.06em; text-transform:uppercase; }
.qa-qtext { font-size:0.88rem; font-weight:500; color:var(--text1); line-height:1.5; }
.qa-arow { padding:1rem 1.2rem; }
.qa-alabel { font-family:'DM Mono',monospace; font-size:0.57rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.45rem; }
.qa-atext { font-size:0.87rem; color:var(--text2); line-height:1.75; white-space:pre-wrap; }
.qa-foot { padding:0.55rem 1.2rem; border-top:1px solid var(--border); background:#fdf8f6; display:flex; gap:16px; }
.qa-meta { font-family:'DM Mono',monospace; font-size:0.58rem; color:var(--text3); }
.qa-meta b { color:var(--text2); }
.chunk-lbl { font-family:'DM Mono',monospace; font-size:0.58rem; color:var(--text3); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem; }
.chunk-body { background:var(--rose-pale); border:1px solid var(--border); border-left:3px solid var(--rose); border-radius:7px; padding:0.75rem 1rem; font-family:'DM Mono',monospace; font-size:0.7rem; color:var(--text2); line-height:1.65; white-space:pre-wrap; margin-bottom:0.6rem; }
.empty-wrap { text-align:center; padding:2.5rem 1rem; }
.empty-icon { font-size:2rem; margin-bottom:0.5rem; }
.empty-txt { font-size:0.8rem; color:var(--text3); }

/* ── SIDEBAR BLOCKS ── */
.sb-brand { padding:1.4rem 1.2rem 1rem; border-bottom:1px solid #3a1a1440; }
.sb-brand-title { font-family:'Cormorant Garamond',serif; font-size:1.2rem; font-weight:700; color:#e0c0b8 !important; }
.sb-brand-sub { font-family:'DM Mono',monospace; font-size:0.55rem; color:#6a3a30 !important; text-transform:uppercase; letter-spacing:0.14em; margin-top:2px; }
.sb-sec { padding:0.9rem 1.2rem 0.2rem; font-family:'DM Mono',monospace; font-size:0.56rem; color:#7a4a40 !important; text-transform:uppercase; letter-spacing:0.18em; border-top:1px solid #3a1a1430; margin-top:0.2rem; }

[data-testid="stExpander"] { background:var(--white) !important; border:1px solid var(--border) !important; border-radius:10px !important; }
[data-testid="stExpander"] summary { font-family:'DM Mono',monospace !important; font-size:0.68rem !important; color:var(--text3) !important; }
div[data-testid="stAlert"] { border-radius:10px !important; font-size:0.82rem !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
for _k, _v in {
    "vectorstore": None, "chat_history": [], "processed_files": [],
    "total_chunks": 0, "embedding_model": None, "activity_log": [],
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def log(msg: str, kind: str = "info"):
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.activity_log.insert(0, {"time": now, "msg": msg, "kind": kind})
    st.session_state.activity_log = st.session_state.activity_log[:20]

# ── API key resolved silently ──────────────────────────────────────────────────
groq_api_key = get_groq_api_key()

# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR  — NO Groq API input field
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-title">✦ RAG Studio</div>
        <div class="sb-brand-sub">Command Center</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">API Configuration</div>', unsafe_allow_html=True)
    if groq_api_key:
        st.markdown(
            '<div style="padding:0.3rem 1.2rem 0.6rem;font-family:\'DM Mono\',monospace;'
            'font-size:0.65rem;color:#5a9e6a;">✓ API key loaded from environment</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="padding:0.3rem 1.2rem 0.6rem;font-family:\'DM Mono\',monospace;'
            'font-size:0.65rem;color:#c8756a;">✗ Key missing — add GROQ_API_KEY to .env or Streamlit secrets</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sb-sec">Model</div>', unsafe_allow_html=True)
    groq_model  = st.selectbox("Model", SUPPORTED_MODELS, index=0, label_visibility="collapsed")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)

    st.markdown('<div class="sb-sec">Chunking Strategy</div>', unsafe_allow_html=True)
    chunk_strategy = st.selectbox("Strategy", CHUNK_STRATEGIES, index=0,
        label_visibility="collapsed",
        help="Recursive: paragraphs→sentences→words\nCharacter: fixed chars\nSemantic: embedding similarity")
    if chunk_strategy != "Sentence (Semantic)":
        chunk_size    = st.slider("Chunk Size", 100, 2000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50, 10)
    else:
        st.info("Semantic chunking uses embedding similarity — no fixed size needed.")
        chunk_size, chunk_overlap = 500, 50

    st.markdown('<div class="sb-sec">Embedding Model</div>', unsafe_allow_html=True)
    embed_model_name = st.selectbox("Embedding", SUPPORTED_EMBED_MODELS, index=0, label_visibility="collapsed")

    st.markdown('<div class="sb-sec">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K Documents", 1, 20, 10, 1)

    st.markdown("---")
    if st.button("🗑  Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        log("Chat history cleared", "warn")
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
#  HERO — with decorative network SVG (top-right, matches screenshot)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-left">
    <div class="hero-logo-row">
      <div class="hero-icon-box">✦</div>
      <div class="hero-title"><span class="rag">RAG</span> Studio</div>
    </div>
    <div class="hero-subtitle">Upload documents. Configure chunking. Ask anything.</div>
  </div>

  <svg class="hero-graphic" viewBox="0 0 360 200" fill="none" xmlns="http://www.w3.org/2000/svg">
    <!-- Background wavy lines -->
    <path d="M10 100 Q70 45 140 75 Q200 105 265 55 Q310 15 355 42" stroke="#e8c4bc" stroke-width="1.8" fill="none" opacity="0.55"/>
    <path d="M10 120 Q80 68 155 95 Q215 118 270 75 Q315 48 355 65" stroke="#c8756a" stroke-width="1.1" fill="none" opacity="0.35"/>
    <path d="M10 140 Q90 100 162 118 Q222 135 278 97 Q320 70 355 88" stroke="#e8c4bc" stroke-width="0.9" fill="none" opacity="0.25"/>
    <path d="M10 80  Q65 35 130 60 Q195 88 255 42 Q305 8 355 30"  stroke="#d4a09a" stroke-width="0.7" fill="none" opacity="0.3"/>

    <!-- Connection lines from center node -->
    <line x1="255" y1="58" x2="208" y2="82"  stroke="#d4a09a" stroke-width="0.9" opacity="0.6"/>
    <line x1="255" y1="58" x2="285" y2="95"  stroke="#d4a09a" stroke-width="0.9" opacity="0.6"/>
    <line x1="255" y1="58" x2="230" y2="20"  stroke="#d4a09a" stroke-width="0.9" opacity="0.6"/>
    <line x1="255" y1="58" x2="305" y2="35"  stroke="#d4a09a" stroke-width="0.9" opacity="0.6"/>
    <line x1="255" y1="58" x2="270" y2="118" stroke="#d4a09a" stroke-width="0.7" opacity="0.35"/>

    <!-- Center glowing node -->
    <circle cx="255" cy="58" r="22" fill="#e8c4bc" opacity="0.25"/>
    <circle cx="255" cy="58" r="16" fill="#f0dbd6" opacity="0.6"/>
    <circle cx="255" cy="58" r="10" fill="url(#cg1)"/>
    <circle cx="255" cy="58" r="5"  fill="#c8756a" opacity="0.85"/>

    <!-- Doc node 1 (left-bottom) -->
    <rect x="193" y="70" width="28" height="28" rx="6" fill="#faf3ef" stroke="#e0ccc6" stroke-width="1.2"/>
    <line x1="200" y1="79" x2="214" y2="79" stroke="#c8a89a" stroke-width="1.2"/>
    <line x1="200" y1="83" x2="214" y2="83" stroke="#c8a89a" stroke-width="1.2"/>
    <line x1="200" y1="87" x2="208" y2="87" stroke="#c8a89a" stroke-width="1.2"/>

    <!-- Doc node 2 (right-bottom) -->
    <rect x="274" y="84" width="28" height="28" rx="6" fill="#faf3ef" stroke="#e0ccc6" stroke-width="1.2"/>
    <line x1="281" y1="93" x2="295" y2="93" stroke="#c8a89a" stroke-width="1.2"/>
    <line x1="281" y1="97" x2="295" y2="97" stroke="#c8a89a" stroke-width="1.2"/>
    <line x1="281" y1="101" x2="289" y2="101" stroke="#c8a89a" stroke-width="1.2"/>

    <!-- Doc node 3 (top-left) -->
    <rect x="218" y="8" width="28" height="28" rx="6" fill="#faf3ef" stroke="#e0ccc6" stroke-width="1.2"/>
    <line x1="225" y1="17" x2="239" y2="17" stroke="#c8a89a" stroke-width="1.2"/>
    <line x1="225" y1="21" x2="239" y2="21" stroke="#c8a89a" stroke-width="1.2"/>
    <line x1="225" y1="25" x2="233" y2="25" stroke="#c8a89a" stroke-width="1.2"/>

    <!-- Doc node 4 (top-right) -->
    <rect x="298" y="22" width="28" height="28" rx="6" fill="#faf3ef" stroke="#e0ccc6" stroke-width="1.2"/>
    <line x1="305" y1="31" x2="319" y2="31" stroke="#c8a89a" stroke-width="1.2"/>
    <line x1="305" y1="35" x2="319" y2="35" stroke="#c8a89a" stroke-width="1.2"/>
    <line x1="305" y1="39" x2="313" y2="39" stroke="#c8a89a" stroke-width="1.2"/>

    <!-- Small accent node (below center) -->
    <circle cx="262" cy="118" r="9"  fill="#f5ede6" stroke="#e0ccc6" stroke-width="1.2"/>
    <circle cx="262" cy="118" r="3.5" fill="#c8756a" opacity="0.55"/>

    <!-- Teal accent node (upper-left of center) -->
    <circle cx="205" cy="45" r="11" fill="#e0f3f7" stroke="#a0d8e4" stroke-width="1.2"/>
    <circle cx="205" cy="45" r="4.5" fill="#5aabb8" opacity="0.75"/>

    <defs>
      <radialGradient id="cg1" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#f0dbd6"/>
        <stop offset="100%" stop-color="#c8756a" stop-opacity="0.7"/>
      </radialGradient>
    </defs>
  </svg>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  UPLOAD SECTION
# ──────────────────────────────────────────────────────────────────────────────
col_up, col_proc = st.columns([3, 1], gap="medium")

with col_up:
    uploaded_files = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt", "csv", "xlsx", "html", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown("""
    <div class="upload-tabs">
        <span class="utab">
            <span class="utab-toggle"></span>
            Recently uploaded
        </span>
        <span class="utab" style="margin-left:auto;color:var(--text2);font-size:0.72rem;">
            📁 File Explorer
        </span>
    </div>
    """, unsafe_allow_html=True)

with col_proc:
    st.write("")
    process_btn = st.button("⚡ Process Files", use_container_width=True)
    if st.session_state.vectorstore:
        st.markdown(
            f'<span class="pill-ok">● Ready</span>'
            f'<span class="pill-cnt">{st.session_state.total_chunks} chunks</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="pill-not">Not Processed</span>', unsafe_allow_html=True)

# File cards
_ext_icons = {
    "pdf":("📄","PDF"), "docx":("📝","DOC"), "txt":("🗒","TXT"),
    "csv":("📊","CSV"), "xlsx":("📊","XLS"), "html":("🌐","HTML"), "md":("📋","MD"),
}
if uploaded_files:
    cards = ""
    for f in uploaded_files:
        ext = f.name.rsplit(".", 1)[-1].lower()
        icon, badge = _ext_icons.get(ext, ("📁", ext.upper()))
        done = f.name in st.session_state.processed_files
        sc = "#3a9e5a" if done else "#b08880"
        st_ = "Processed" if done else "Pending"
        cards += f"""<div class="fcard">
            <div class="fcard-icon">{icon}<span class="fcard-badge">{badge}</span></div>
            <div><div class="fcard-name">{f.name}</div>
            <div class="fcard-status" style="color:{sc};">{st_}</div></div>
        </div>"""
    st.markdown(f'<div class="file-cards-row">{cards}</div>', unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  PROCESS
# ──────────────────────────────────────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.warning("Please upload at least one file first.")
    elif not groq_api_key:
        st.error("API key not found. Add GROQ_API_KEY to your .env file or Streamlit secrets.")
    else:
        log("Processing started", "info")
        with st.spinner("Loading documents & building FAISS index…"):
            all_docs, failed = [], []
            for uf in uploaded_files:
                try:
                    all_docs.extend(load_file(uf))
                    log(f"Loaded: {uf.name}", "ok")
                except Exception:
                    failed.append(uf.name)
                    log(f"Failed: {uf.name}", "warn")
            if all_docs:
                try:
                    vs, n_chunks, emb = build_vectorstore(
                        all_docs, chunk_strategy, chunk_size, chunk_overlap, embed_model_name
                    )
                    st.session_state.vectorstore     = vs
                    st.session_state.total_chunks    = n_chunks
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.session_state.embedding_model = emb
                    log(f"FAISS ready — {n_chunks} chunks", "ok")
                    log(f"Model config loaded", "info")
                    st.success(f"✓ {len(uploaded_files)} file(s) → {n_chunks} chunks → ready!")
                except Exception as e:
                    log(f"Error: {e}", "warn")
                    st.error(f"Vectorstore error: {e}")
            else:
                st.error("Could not load any documents.")
            if failed:
                st.warning(f"Failed: {', '.join(failed)}")
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
#  ACTIVITY STREAM + RESOURCE PANEL
# ──────────────────────────────────────────────────────────────────────────────
col_act, col_res = st.columns([3, 1], gap="medium")

with col_act:
    _cls = {"ok":"act-ok","warn":"act-warn","info":"act-info"}
    lines = ""
    if st.session_state.activity_log:
        for e in st.session_state.activity_log[:7]:
            c = _cls.get(e["kind"], "act-info")
            lines += f'<div class="act-line"><span class="act-time">[{e["time"]}]</span><span class="act-msg"><span class="{c}">{e["kind"].upper()}:</span> {e["msg"]}</span></div>'
    else:
        lines = '<div class="act-empty">No activity yet — upload and process documents to begin.</div>'

    st.markdown(f"""
    <div class="panel-card">
        <div class="panel-hdr">Activity Stream <span>⌄</span></div>
        {lines}
    </div>""", unsafe_allow_html=True)

with col_res:
    chunks = st.session_state.total_chunks
    files_done = len(st.session_state.processed_files)
    qa_count   = len(st.session_state.chat_history)
    pct = min(int(chunks / 20), 100) if chunks else 0

    import math
    r, cx, cy = 36, 48, 48
    circ = 2 * math.pi * r
    dash = circ * pct / 100
    donut = f"""<svg viewBox="0 0 96 96" width="88" height="88">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#f0e4db" stroke-width="12"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#c8756a" stroke-width="12"
        stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round"
        transform="rotate(-90 {cx} {cy})"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#5aabb8" stroke-width="12"
        stroke-dasharray="{circ*0.18:.1f} {circ:.1f}" stroke-linecap="round"
        stroke-dashoffset="-{dash:.1f}"
        transform="rotate(-90 {cx} {cy})"/>
      <text x="{cx}" y="{cy+5}" text-anchor="middle"
        font-family="DM Mono,monospace" font-size="13" font-weight="600" fill="#2e1a14">{pct}%</text>
    </svg>"""

    st.markdown(f"""
    <div class="panel-card">
        <div class="panel-hdr">Resource Usage</div>
        <div class="res-legend">
            <div class="res-row">
                <div class="res-dot-wrap"><div class="res-dot" style="background:#3a9e5a;"></div><span class="res-label">Files</span></div>
                <span class="res-val">{files_done}</span>
            </div>
            <div class="res-row">
                <div class="res-dot-wrap"><div class="res-dot" style="background:#5aabb8;"></div><span class="res-label">Chunks</span></div>
                <span class="res-val">{chunks}</span>
            </div>
            <div class="res-row">
                <div class="res-dot-wrap"><div class="res-dot" style="background:#c8756a;"></div><span class="res-label">Q&amp;A</span></div>
                <span class="res-val">{qa_count}</span>
            </div>
        </div>
        <div class="donut-wrap">{donut}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  NOTICE / CHAT
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="notice">
        <strong>Getting started:</strong> Upload your documents above,
        then click <strong>⚡ Process Files</strong> to build the knowledge base.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="sec-label">Query Interface</div>', unsafe_allow_html=True)
    col_q, col_btn = st.columns([5, 1], gap="small")
    with col_q:
        user_question = st.text_input("Q", placeholder="Ask anything about your documents…",
                                      label_visibility="collapsed", key="q_input")
    with col_btn:
        ask_btn = st.button("Ask →", use_container_width=True)

    if ask_btn and user_question.strip():
        if not groq_api_key:
            st.error("API key not found.")
        else:
            with st.spinner("Generating answer…"):
                try:
                    llm = get_llm(groq_api_key, groq_model, temperature)
                    answer, chunks_ret = rag_answer(
                        user_question, st.session_state.vectorstore, llm, k=top_k)
                    st.session_state.chat_history.insert(0, {
                        "question": user_question, "answer": answer,
                        "chunks": chunks_ret, "model": groq_model,
                    })
                    log(f"Query: {user_question[:45]}…", "ok")
                except Exception as e:
                    log(f"Error: {e}", "warn")
                    st.error(f"Error: {e}")
            st.rerun()

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Conversation History</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        total = len(st.session_state.chat_history)
        for i, item in enumerate(st.session_state.chat_history):
            q_num = total - i
            st.markdown(f"""
            <div class="qa-card">
                <div class="qa-qrow">
                    <span class="qa-qbadge">Q{q_num}</span>
                    <span class="qa-qtext">{item['question']}</span>
                </div>
                <div class="qa-arow">
                    <div class="qa-alabel">// Response</div>
                    <div class="qa-atext">{item['answer']}</div>
                </div>
                <div class="qa-foot">
                    <span class="qa-meta">model: <b>{item['model']}</b></span>
                    <span class="qa-meta">chunks: <b>{len(item['chunks'])}</b></span>
                </div>
            </div>""", unsafe_allow_html=True)
            with st.expander(f"📎 Source chunks — Q{q_num}"):
                for j, chunk in enumerate(item["chunks"]):
                    preview = chunk["content"][:600] + ("…" if len(chunk["content"]) > 600 else "")
                    st.markdown(f'<div class="chunk-lbl">Chunk {j+1} · {chunk["filename"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chunk-body">{preview}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-wrap">
            <div class="empty-icon">✦</div>
            <div class="empty-txt">Ask your first question above</div>
        </div>""", unsafe_allow_html=True)
