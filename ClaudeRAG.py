import os
import tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Studio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Main background */
.main .block-container {
    background: #0d1117;
    padding-top: 2rem;
}
body { background: #0d1117; }

/* Header */
.rag-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.25rem;
}
.rag-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.02em;
}
.rag-subtitle {
    font-size: 0.85rem;
    color: #8b949e;
    margin-bottom: 1.5rem;
}
.accent { color: #58a6ff; }

/* Status badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-green { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
.badge-blue  { background: #0c1f3a; color: #58a6ff; border: 1px solid #1f6feb; }
.badge-gray  { background: #1c2128; color: #8b949e; border: 1px solid #30363d; }
.badge-orange{ background: #2d1e0a; color: #f0883e; border: 1px solid #9e6a03; }

/* Q&A Cards */
.qa-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.qa-q {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #8b949e;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.qa-q span { color: #58a6ff; font-weight: 600; }
.qa-a {
    font-size: 0.95rem;
    color: #e6edf3;
    line-height: 1.65;
    margin-bottom: 0.6rem;
}
.qa-meta {
    font-size: 0.72rem;
    color: #6e7681;
    font-family: 'IBM Plex Mono', monospace;
}

/* Chunk expander */
.chunk-box {
    background: #0d1117;
    border: 1px solid #1e2130;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.6;
    white-space: pre-wrap;
}
.chunk-label {
    font-size: 0.7rem;
    color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Upload area */
.upload-hint {
    font-size: 0.78rem;
    color: #6e7681;
    margin-top: 0.3rem;
}

/* Input box */
.stTextInput input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.stTextInput input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important;
}

/* Buttons */
.stButton button {
    background: #1f6feb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.45rem 1.2rem !important;
    transition: background 0.2s;
}
.stButton button:hover {
    background: #388bfd !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.2rem 0;
}

/* Sidebar section headers */
.sidebar-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e2130;
}

/* File pill */
.file-pill {
    display: inline-block;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 5px;
    padding: 3px 9px;
    font-size: 0.72rem;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
    margin: 2px;
}

/* Process button area */
.process-hint {
    font-size: 0.75rem;
    color: #6e7681;
    margin-top: 0.4rem;
}

/* Info box */
.info-box {
    background: #0c1f3a;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #8b949e;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="rag-title" style="font-size:1.2rem;"> RAG Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section"> API Configuration</div>', unsafe_allow_html=True)

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your key at console.groq.com"
    )

    groq_model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

    st.markdown('<div class="sidebar-section"> Chunking Strategy</div>', unsafe_allow_html=True)

    chunk_strategy = st.selectbox(
        "Strategy",
        ["Recursive", "Character", "Sentence (Semantic)"],
        index=0,
        help="Recursive: splits by paragraphs → sentences → words\nCharacter: splits by character count\nSentence: semantic similarity-based splitting"
    )

    if chunk_strategy != "Sentence (Semantic)":
        chunk_size = st.slider("Chunk Size", 100, 2000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50, 10)
    else:
        st.markdown('<div class="info-box"> Semantic chunking uses embedding similarity — no fixed size needed.</div>', unsafe_allow_html=True)
        chunk_size = 500
        chunk_overlap = 50

    st.markdown('<div class="sidebar-section"> Embedding Model</div>', unsafe_allow_html=True)
    embed_model_name = st.selectbox(
        "HuggingFace Model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
        ],
        index=0
    )

    st.markdown('<div class="sidebar-section">🔍 Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Top K documents", 1, 20, 10, 1)

    st.markdown("---")
    if st.button(" Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def load_file(uploaded_file):
    """Save uploaded file to temp dir and load as LangChain docs."""
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        #PDF
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

        #DOCX
        elif suffix in [".docx", ".doc"]:
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()

        #TXT / Markdown
        elif suffix in [".txt", ".md"]:
            loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()

        #CSV
        elif suffix == ".csv":
            df = pd.read_csv(tmp_path)
            text = df.to_string()
            docs = [Document(page_content=text)]

        #Excel
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(tmp_path)
            text = df.to_string()
            docs = [Document(page_content=text)]

        #HTML
        elif suffix == ".html":
            with open(tmp_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text()
            docs = [Document(page_content=text)]

        else:
            return []

        # Add metadata
        for doc in docs:
            doc.metadata["source_filename"] = uploaded_file.name

        return docs

    finally:
        os.unlink(tmp_path)


def get_splitter(strategy, size, overlap, embedding_model=None):
    if strategy == "Recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    elif strategy == "Character":
        return CharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separator="\n"
        )
    elif strategy == "Sentence (Semantic)":
        return SemanticChunker(embedding_model)


def build_vectorstore(docs, strategy, size, overlap, embed_model_name):
    """Build FAISS vectorstore from documents."""
    embedding = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    st.session_state.embedding_model = embedding

    splitter = get_splitter(strategy, size, overlap, embedding)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding)
    return vectorstore, len(chunks)


def rag_answer(question, vectorstore, llm, k=10):
    """Retrieve top-k docs and generate answer."""
    all_docs = vectorstore.similarity_search(question, k=k)
    top_docs = all_docs[:4]

    context = "\n\n".join([doc.page_content for doc in top_docs])

    prompt = """You are a knowledgeable assistant. Using ONLY the context provided below, answer the question clearly and accurately.
If the answer is not in the context, say: "I couldn't find relevant information in the uploaded documents."

Instructions:
- Be concise and direct.
- Use bullet points only when listing multiple items.
- Do not add introductions or conclusions.
- Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)
    clean_response = response.content.strip().replace("**", "")

    # Return only top 2 chunks for display
    top2_chunks = [
        {
            "filename": doc.metadata.get("source_filename", "Unknown"),
            "content": doc.page_content
        }
        for doc in top_docs[:2]
    ]

    return clean_response, top2_chunks


# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <div class="rag-title"> RAG <span class="accent">Studio</span></div>
</div>
<div class="rag-subtitle">Upload documents · Configure chunking · Ask anything</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FILE UPLOAD SECTION
# ─────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "csv", "xlsx", "html", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Supports PDF, DOCX, TXT, CSV, Excel, HTML, Markdown"
    )
    st.markdown('<div class="upload-hint">📎 Supports PDF, DOCX, TXT, CSV, Excel, HTML, Markdown — multiple files allowed</div>', unsafe_allow_html=True)

with col2:
    process_btn = st.button("⚡ Process Files", use_container_width=True)
    if st.session_state.vectorstore:
        st.markdown(f'<div style="margin-top:6px"><span class="badge badge-green">✓ READY</span> <span class="badge badge-blue">{st.session_state.total_chunks} chunks</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin-top:6px"><span class="badge badge-gray">NOT PROCESSED</span></div>', unsafe_allow_html=True)

# Show uploaded file names
if uploaded_files:
    pills = "".join([f'<span class="file-pill"> {f.name}</span>' for f in uploaded_files])
    st.markdown(f'<div style="margin-top:0.5rem">{pills}</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PROCESS FILES
# ─────────────────────────────────────────────
if process_btn:
    if not uploaded_files:
        st.warning(" Please upload at least one file first.")
    elif not groq_api_key:
        st.warning(" Please enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Loading & processing documents..."):
            all_docs = []
            failed = []
            for uf in uploaded_files:
                try:
                    docs = load_file(uf)
                    all_docs.extend(docs)
                except Exception as e:
                    failed.append(uf.name)

            if all_docs:
                try:
                    vs, n_chunks = build_vectorstore(
                        all_docs, chunk_strategy, chunk_size, chunk_overlap, embed_model_name
                    )
                    st.session_state.vectorstore = vs
                    st.session_state.total_chunks = n_chunks
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.success(f" Processed {len(uploaded_files)} file(s) → {n_chunks} chunks → FAISS index ready!")
                except Exception as e:
                    st.error(f" Error building vectorstore: {e}")
            else:
                st.error(" Could not load any documents. Check file formats.")

            if failed:
                st.warning(f" Failed to load: {', '.join(failed)}")

        st.rerun()

# ─────────────────────────────────────────────
#  CHAT SECTION
# ─────────────────────────────────────────────
if not st.session_state.vectorstore:
    st.markdown("""
    <div class="info-box">
         <strong>Getting started:</strong> Upload your documents above, add your Groq API key in the sidebar, then click <strong>⚡ Process Files</strong> to build the knowledge base.
    </div>
    """, unsafe_allow_html=True)
else:
    # Status bar
    files_str = " · ".join([f" {f}" for f in st.session_state.processed_files])
    st.markdown(f"""
    <div style="display:flex; gap:8px; align-items:center; margin-bottom:1rem; flex-wrap:wrap;">
        <span class="badge badge-green">✓ VECTORSTORE READY</span>
        <span class="badge badge-blue">FAISS · {st.session_state.total_chunks} chunks</span>
        <span class="badge badge-orange">{chunk_strategy}</span>
        <span style="font-size:0.72rem; color:#6e7681;">{files_str}</span>
    </div>
    """, unsafe_allow_html=True)

    # Question input
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        user_question = st.text_input(
            "Ask a question",
            placeholder="What does the document say about...?",
            label_visibility="collapsed",
            key="question_input"
        )
    with col_btn:
        ask_btn = st.button("Ask →", use_container_width=True)

    # Handle question
    if ask_btn and user_question.strip():
        if not groq_api_key:
            st.warning(" Add your Groq API key in the sidebar.")
        else:
            with st.spinner("Searching & generating answer..."):
                try:
                    os.environ["GROQ_API_KEY"] = groq_api_key
                    llm = ChatGroq(
                        model=groq_model,
                        temperature=temperature,
                        api_key=groq_api_key
                    )
                    answer, top2_chunks = rag_answer(
                        user_question,
                        st.session_state.vectorstore,
                        llm,
                        k=top_k
                    )
                    st.session_state.chat_history.insert(0, {
                        "question": user_question,
                        "answer": answer,
                        "chunks": top2_chunks,
                        "model": groq_model
                    })
                except Exception as e:
                    st.error(f" Error: {e}")
            st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Chat History (Q&A list) ──
    if st.session_state.chat_history:
        st.markdown(f'<div style="font-size:0.75rem; color:#6e7681; margin-bottom:0.8rem; font-family:\'IBM Plex Mono\',monospace;">CONVERSATION HISTORY · {len(st.session_state.chat_history)} exchange(s)</div>', unsafe_allow_html=True)

        for i, item in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="qa-card">
                <div class="qa-q"><span>Q{len(st.session_state.chat_history) - i}</span> · {item['question']}</div>
                <div class="qa-a">{item['answer']}</div>
                <div class="qa-meta">model: {item['model']} · {len(item['chunks'])} source chunks retrieved</div>
            </div>
            """, unsafe_allow_html=True)

            # Top 2 chunks expandable
            with st.expander(f"📎 View top 2 retrieved chunks — Q{len(st.session_state.chat_history) - i}"):
                for j, chunk in enumerate(item["chunks"]):
                    st.markdown(f'<div class="chunk-label">Chunk {j+1} · {chunk["filename"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chunk-box">{chunk["content"][:600]}{"..." if len(chunk["content"]) > 600 else ""}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center; padding:2rem; color:#6e7681; font-size:0.85rem;">Ask your first question above 👆</div>', unsafe_allow_html=True)
