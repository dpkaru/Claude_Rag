"""
backend.py
==========
RAG Studio — Backend Logic
All document loading, chunking, embedding, vectorstore, and LLM inference.
No Streamlit UI code lives here.
 
API KEY LOADING ORDER (first found wins):
  1. Streamlit secrets  → st.secrets["GROQ_API_KEY"]   (Streamlit Cloud)
  2. .env file          → GROQ_API_KEY=gsk_...          (local dev)
  3. OS environment     → export GROQ_API_KEY=gsk_...   (fallback)
  4. Manual override    → UI text input                  (last resort)
"""
 
import os
import tempfile
 
import pandas as pd
from bs4 import BeautifulSoup
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
 
# ── Load .env file if present (install python-dotenv for local dev) ───────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — other fallbacks still work
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  API KEY RESOLUTION
# ──────────────────────────────────────────────────────────────────────────────
 
def get_groq_api_key() -> str | None:
    """
    Resolve the Groq API key from available sources in priority order.
    Returns None if not found — the UI will prompt the user to enter it manually.
    """
    # 1. Streamlit Cloud secrets / local .streamlit/secrets.toml
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
 
    # 2. OS environment variable (also covers .env loaded above)
    key = os.environ.get("GROQ_API_KEY", None)
    if key:
        return key
 
    return None
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
 
SUPPORTED_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
 
SUPPORTED_EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
]
 
CHUNK_STRATEGIES = ["Recursive", "Character", "Sentence (Semantic)"]
 
RAG_PROMPT_TEMPLATE = """You are a knowledgeable assistant. Using ONLY the context provided below, answer the question clearly and accurately.
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
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  FILE LOADING
# ──────────────────────────────────────────────────────────────────────────────
 
def load_file(uploaded_file) -> list[Document]:
    """
    Save an uploaded Streamlit file to a temp path, load via LangChain loader,
    attach source metadata, and return a list of Document objects.
    Supported: PDF, DOCX, TXT, MD, CSV, XLSX, HTML
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
 
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
 
    try:
        if suffix == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
        elif suffix in [".docx", ".doc"]:
            docs = Docx2txtLoader(tmp_path).load()
        elif suffix in [".txt", ".md"]:
            docs = TextLoader(tmp_path, encoding="utf-8").load()
        elif suffix == ".csv":
            df = pd.read_csv(tmp_path)
            docs = [Document(page_content=df.to_string())]
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(tmp_path)
            docs = [Document(page_content=df.to_string())]
        elif suffix == ".html":
            with open(tmp_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                text = soup.get_text()
            docs = [Document(page_content=text)]
        else:
            docs = []
 
        for doc in docs:
            doc.metadata["source_filename"] = uploaded_file.name
 
        return docs
 
    finally:
        os.unlink(tmp_path)
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  CHUNKING
# ──────────────────────────────────────────────────────────────────────────────
 
def get_splitter(strategy: str, size: int, overlap: int, embedding_model=None):
    """Return the appropriate LangChain text splitter."""
    if strategy == "Recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    elif strategy == "Character":
        return CharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap, separator="\n",
        )
    elif strategy == "Sentence (Semantic)":
        return SemanticChunker(embedding_model)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  VECTORSTORE CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────
 
def build_vectorstore(
    docs: list[Document],
    strategy: str,
    size: int,
    overlap: int,
    embed_model_name: str,
) -> tuple:
    """
    Embed and index documents into a FAISS vectorstore.
    Returns (vectorstore, n_chunks, embedding_model).
    """
    embedding = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    splitter = get_splitter(strategy, size, overlap, embedding)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding)
    return vectorstore, len(chunks), embedding
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  RAG INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
 
def rag_answer(question: str, vectorstore, llm, k: int = 10) -> tuple[str, list[dict]]:
    """
    Retrieve top-k chunks and generate an answer.
    Returns (answer_string, top2_chunks_list).
    """
    all_docs = vectorstore.similarity_search(question, k=k)
    top_docs = all_docs[:4]
    context = "\n\n".join([doc.page_content for doc in top_docs])
    final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke(final_prompt)
    clean_response = response.content.strip().replace("**", "")
    top2_chunks = [
        {"filename": doc.metadata.get("source_filename", "Unknown"), "content": doc.page_content}
        for doc in top_docs[:2]
    ]
    return clean_response, top2_chunks
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  LLM FACTORY
# ──────────────────────────────────────────────────────────────────────────────
 
def get_llm(api_key: str, model: str, temperature: float) -> ChatGroq:
    """Instantiate and return a ChatGroq LLM client."""
    os.environ["GROQ_API_KEY"] = api_key
    return ChatGroq(model=model, temperature=temperature, api_key=api_key)
