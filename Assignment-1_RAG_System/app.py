"""
RAG System for Research Paper Q&A - Streamlit Web Application
Harshit Arora & Aditi Jha

Interactive web UI that allows users to ask questions about research papers
using a Retrieval-Augmented Generation (RAG) pipeline.
"""

import os
import time
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PDF_FOLDER = "research_papers"
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 10
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Paper Q&A",
    page_icon="📄",
    layout="wide"
)

# ──────────────────────────────────────────────
# Custom CSS - Premium Design
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ─── Global ─── */
    .block-container { padding-top: 2rem; }

    /* ─── Hero Header ─── */
    .hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem;
        margin-bottom: 1.8rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 70%, rgba(78,205,196,0.06) 0%, transparent 50%),
                    radial-gradient(circle at 70% 30%, rgba(255,107,107,0.04) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
        background: linear-gradient(135deg, #ffffff 0%, #a8d8ea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 0.95rem;
        opacity: 0.55;
        margin: 0;
        font-weight: 400;
        letter-spacing: 0.5px;
    }

    /* ─── Stats Cards ─── */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.25s ease;
    }
    .stat-card:hover {
        background: rgba(255,255,255,0.07);
        border-color: rgba(255,255,255,0.15);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .stat-icon { font-size: 1.6rem; margin-bottom: 0.3rem; }
    .stat-value {
        font-size: 1.7rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ecdc4, #a8d8ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.45;
        margin-top: 0.15rem;
        font-weight: 600;
    }

    /* ─── Input Area ─── */
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        background: rgba(255,255,255,0.04) !important;
        padding: 0.85rem 1.2rem !important;
        font-size: 1rem !important;
        transition: all 0.25s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(78,205,196,0.5) !important;
        box-shadow: 0 0 0 3px rgba(78,205,196,0.1) !important;
    }

    /* ─── Buttons ─── */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.65rem 1.5rem;
        font-size: 0.92rem;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* ─── Chat Entry ─── */
    .chat-entry {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.25s ease;
    }
    .chat-entry:hover {
        border-color: rgba(255,255,255,0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }

    /* ─── Question Badge ─── */
    .question-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, rgba(78,205,196,0.12), rgba(78,205,196,0.04));
        border: 1px solid rgba(78,205,196,0.2);
        border-radius: 10px;
        padding: 0.6rem 1.1rem;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        color: #a8d8ea;
    }

    /* ─── Answer Area ─── */
    .answer-area {
        line-height: 1.85;
        font-size: 0.96rem;
        padding: 0.5rem 0;
    }
    .answer-area h1, .answer-area h2, .answer-area h3, .answer-area h4 {
        margin-top: 1.3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .answer-area ul, .answer-area ol {
        padding-left: 1.6rem;
        margin: 0.4rem 0;
    }
    .answer-area li {
        margin-bottom: 0.35rem;
        line-height: 1.75;
    }
    .answer-area li ul, .answer-area li ol {
        margin-top: 0.25rem;
    }
    .answer-area p { margin-bottom: 0.7rem; }
    .answer-area code {
        background: rgba(255,255,255,0.08);
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.88em;
    }
    .answer-area strong { color: #a8d8ea; }

    /* ─── Meta line ─── */
    .meta-line {
        display: flex;
        align-items: center;
        gap: 1rem;
        font-size: 0.78rem;
        opacity: 0.4;
        margin-top: 1rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }

    /* ─── Source Chips ─── */
    .source-chip {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-left: 3px solid rgba(78,205,196,0.4);
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.3rem;
        margin-bottom: 0.6rem;
        font-size: 0.86rem;
        line-height: 1.65;
        transition: all 0.2s ease;
    }
    .source-chip:hover {
        background: rgba(255,255,255,0.05);
        border-left-color: rgba(78,205,196,0.7);
    }
    .source-chip .src-header {
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 0.35rem;
        color: #a8d8ea;
    }
    .source-chip .src-preview {
        opacity: 0.6;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    /* ─── Expander ─── */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
    }

    /* ─── Footer ─── */
    .app-footer {
        text-align: center;
        opacity: 0.3;
        font-size: 0.8rem;
        margin-top: 3.5rem;
        padding: 1.5rem 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        letter-spacing: 0.3px;
    }

    /* ─── Dividers ─── */
    hr { opacity: 0.06; margin: 1.5rem 0; }

    /* ─── Spinner ─── */
    .stSpinner > div { justify-content: center; }

    /* ─── Hide Streamlit defaults ─── */
    [data-testid="stMetric"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    """Load and cache the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource
def load_and_index_pdfs(_embeddings):
    """Load PDFs, chunk text, and create FAISS vector store."""
    pdf_files = sorted([f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")])

    if not pdf_files:
        st.error(f"No PDF files found in '{PDF_FOLDER}/' directory.")
        return None, [], 0, 0

    all_documents = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)

    total_pages = len(all_documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)

    vectorstore = FAISS.from_documents(chunks, _embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

    return vectorstore, pdf_files, total_pages, len(chunks)


def build_qa_chain(vectorstore, api_key, model_name):
    """Build the RetrievalQA chain with Groq LLM."""
    prompt_template = """You are a helpful research assistant. Use the following pieces of context from research papers to answer the question. 
If you don't know the answer based on the context, say "I don't have enough information in the provided papers to answer this question."

Always cite which paper(s) you're referencing in your answer.

Context:
{context}

Question: {question}

Answer (with citations):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model_name=model_name,
        temperature=0.3,
        groq_api_key=api_key
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False


# ──────────────────────────────────────────────
# Auto-initialize with API key
# ──────────────────────────────────────────────
API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")  # Set via environment variable
os.environ["GROQ_API_KEY"] = API_KEY

if not st.session_state.initialized:
    with st.spinner("Loading system..."):
        model_name = "llama-3.3-70b-versatile"
        embeddings = get_embeddings()
        vectorstore, pdf_files, total_pages, total_chunks = load_and_index_pdfs(embeddings)

    if vectorstore:
        st.session_state.qa_chain = build_qa_chain(vectorstore, API_KEY, model_name)
        st.session_state.initialized = True
        st.session_state.pdf_files = pdf_files
        st.session_state.total_pages = total_pages
        st.session_state.total_chunks = total_chunks
        st.session_state.model_name = model_name


# ──────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────

# ── Hero Header ──
st.markdown(
    '<div class="hero">'
    '<h1>📄 Research Paper Q&A</h1>'
    '<p>RAG System &nbsp;·&nbsp; Agentic AI Assignment &nbsp;·&nbsp; Harshit Arora & Aditi Jha</p>'
    '</div>',
    unsafe_allow_html=True
)

# ── System Statistics ──
if st.session_state.initialized:
    pdf_count = len(st.session_state.pdf_files)
    st.markdown(
        f'<div class="stats-row">'
        f'  <div class="stat-card">'
        f'    <div class="stat-icon">📚</div>'
        f'    <div class="stat-value">{pdf_count}</div>'
        f'    <div class="stat-label">Papers</div>'
        f'  </div>'
        f'  <div class="stat-card">'
        f'    <div class="stat-icon">📄</div>'
        f'    <div class="stat-value">{st.session_state.total_pages}</div>'
        f'    <div class="stat-label">Pages</div>'
        f'  </div>'
        f'  <div class="stat-card">'
        f'    <div class="stat-icon">🧩</div>'
        f'    <div class="stat-value">{st.session_state.total_chunks}</div>'
        f'    <div class="stat-label">Chunks</div>'
        f'  </div>'
        f'  <div class="stat-card">'
        f'    <div class="stat-icon">🤖</div>'
        f'    <div class="stat-value" style="font-size:1rem;">Llama 3.3 70B</div>'
        f'    <div class="stat-label">Model</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Query Input ──
    query = st.text_input(
        "🔍  Ask a question about the research papers",
        placeholder="e.g., What is the Transformer architecture and its key components?",
        label_visibility="visible"
    )

    col_submit, col_clear = st.columns([3, 1])
    with col_submit:
        submit = st.button("🚀  Search & Answer", use_container_width=True, type="primary")
    with col_clear:
        clear = st.button("🗑️ Clear", use_container_width=True)

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    # ── Process Query ──
    if submit and query:
        with st.spinner("🔎 Searching papers and generating answer..."):
            start_time = time.time()
            result = st.session_state.qa_chain.invoke({"query": query})
            elapsed = time.time() - start_time

        answer = result["result"]
        sources = result["source_documents"]

        st.session_state.chat_history.insert(0, {
            "query": query,
            "answer": answer,
            "sources": sources,
            "time": elapsed
        })

    # ── Display Chat History ──
    if st.session_state.chat_history:
        st.markdown("---")

    for i, entry in enumerate(st.session_state.chat_history):
        # Question badge
        st.markdown(
            f'<div class="chat-entry">'
            f'  <div class="question-badge">💬 {entry["query"]}</div>'
            f'  <div class="answer-area">{entry["answer"]}</div>'
            f'  <div class="meta-line">'
            f'    <span>⏱️ {entry["time"]:.2f}s</span>'
            f'    <span>📑 {len(entry["sources"])} source chunks</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True
        )

        with st.expander(f"📑  View Source Documents ({len(entry['sources'])} chunks)"):
            for j, doc in enumerate(entry["sources"], 1):
                source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "N/A")
                content_preview = doc.page_content[:250].replace("\n", " ")
                st.markdown(
                    f'<div class="source-chip">'
                    f'  <div class="src-header">📎 [{j}] {source_file} — Page {page}</div>'
                    f'  <div class="src-preview">{content_preview}...</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

else:
    st.info("⏳ System is initializing... Please wait.")

# ── Footer ──
st.markdown(
    '<div class="app-footer">'
    '© Harshit Arora & Aditi Jha &nbsp;·&nbsp; Agentic AI — Assignment 1 &nbsp;·&nbsp; RAG System for Research Paper Q&A'
    '</div>',
    unsafe_allow_html=True
)
