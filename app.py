"""
RAG-based Research Paper Q&A System
Agentic AI Assignment
Students: Harshit Arora, Aditi Jha
"""

import os
import glob
import streamlit as st
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# --- Page Config ---
st.set_page_config(page_title="RAG Q&A System", page_icon="📄", layout="centered")

# --- Minimal CSS ---
st.markdown("""
<style>
    .block-container { max-width: 800px; padding-top: 2rem; }
    .stTextInput > div > div > input { border-radius: 4px; }
    .source-label { font-size: 0.85rem; color: #555; }
    hr { margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

GOOGLE_API_KEY = "AIzaSyDSS-MuRRsNrbPnQoVFm8W3t9FHN7UvnDI"


def find_working_model():
    """Auto-detect a working Gemini model from the free tier."""
    genai.configure(api_key=GOOGLE_API_KEY)
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name:
                    return m.name
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                return m.name
    except Exception:
        pass

    for name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]:
        try:
            model = genai.GenerativeModel(name)
            model.generate_content("hi")
            return name
        except Exception:
            continue
    return None


@st.cache_resource
def init_rag():
    """Load PDFs, build vector store, and find a working Gemini model."""
    try:
        genai.configure(api_key=GOOGLE_API_KEY)

        model_name = find_working_model()
        if not model_name:
            st.error("No working Gemini model found. Please verify API key.")
            return None

        pdf_files = glob.glob(os.path.join("research_papers", "*.pdf"))
        if not pdf_files:
            st.error("No PDFs found in `research_papers/` folder.")
            return None

        documents = []
        for f in pdf_files:
            documents.extend(PyPDFLoader(f).load())

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        ).split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        idx_path = "faiss_index"
        if os.path.exists(idx_path):
            try:
                vs = FAISS.load_local(idx_path, embeddings, allow_dangerous_deserialization=True)
            except TypeError:
                vs = FAISS.load_local(idx_path, embeddings)
        else:
            vs = FAISS.from_documents(chunks, embeddings)
            vs.save_local(idx_path)

        return {
            "vs": vs,
            "llm": genai.GenerativeModel(model_name),
            "model": model_name.replace("models/", ""),
            "pdfs": len(pdf_files),
            "pages": len(documents),
            "chunks": len(chunks),
        }
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        return None


def query_rag(question, rag):
    """Retrieve context and generate an answer."""
    docs = rag["vs"].similarity_search(question, k=6)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = (
        "You are an expert research assistant. Based on the context from research papers below, "
        "provide a detailed, comprehensive answer. Include key findings, methods, and concepts. "
        "If the context only partially answers the question, share what is available.\n\n"
        f"Context from papers:\n{context}\n\n"
        f"Question: {question}\n\nDetailed Answer:"
    )

    resp = rag["llm"].generate_content(prompt)
    return resp.text, docs


# ── UI ──────────────────────────────────────────────────────────────────────────

st.title("📄 Research Paper Q&A")
st.caption("RAG System — Agentic AI Assignment  |  **Harshit Arora** & **Aditi Jha**")

st.divider()

# Initialise
if "rag" not in st.session_state:
    with st.spinner("Loading documents & models …"):
        st.session_state.rag = init_rag()

rag = st.session_state.get("rag")

if rag:
    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDFs", rag["pdfs"])
    c2.metric("Pages", rag["pages"])
    c3.metric("Chunks", rag["chunks"])
    c4.metric("Model", rag["model"])

    st.divider()

    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    question = st.text_input("Ask a question about the research papers")

    col_ask, col_clear = st.columns([1, 1])
    with col_ask:
        ask = st.button("Submit", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    if ask and question:
        with st.spinner("Generating answer …"):
            try:
                t0 = time.time()
                answer, sources = query_rag(question, rag)
                elapsed = time.time() - t0

                st.session_state.history.insert(0, {
                    "q": question, "a": answer,
                    "sources": sources, "t": elapsed
                })
            except Exception as e:
                st.error(f"Error: {e}")

    # Render history
    for idx, item in enumerate(st.session_state.history):
        st.markdown(f"**Q:** {item['q']}")
        st.markdown(item["a"])
        st.caption(f"Response time: {item['t']:.2f}s")

        with st.expander("View sources"):
            for j, doc in enumerate(item["sources"], 1):
                src = os.path.basename(doc.metadata.get("source", "—"))
                pg = doc.metadata.get("page", "—")
                st.markdown(f"**Source {j}** — `{src}`, page {pg}")
                st.text(doc.page_content[:300] + " …")
        st.divider()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Submitted by <b>Harshit Arora</b> &amp; <b>Aditi Jha</b> · Agentic AI Course · 2026"
    "</div>",
    unsafe_allow_html=True,
)
