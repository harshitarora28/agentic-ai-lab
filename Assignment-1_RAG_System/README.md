# RAG System for Research Paper Q&A

**Harshit Arora & Aditi Jha** | Agentic AI - Assignment 1

A Retrieval-Augmented Generation (RAG) system that answers questions about research papers by combining semantic search with Large Language Model (LLM) generation.

---

## Problem Statement

Research papers contain dense technical information that is hard to query using traditional keyword search. LLMs have limited context windows and cannot process entire papers at once. This RAG pipeline retrieves the most relevant document chunks using semantic embeddings, then augments the LLM prompt with this context so the model can generate accurate, grounded answers with source citations.

## Architecture

```
PDF Files → PyPDF Loader → Text Chunking → Embedding (MiniLM) → FAISS Index
                                                                      ↓
User Query → Embedding → Similarity Search → Top-k Chunks → Groq LLM → Answer + Citations
```

## Research Papers Included

| File | Paper |
|------|-------|
| `1706.03762v7.pdf` | Attention Is All You Need (Transformer) |
| `1810.04805v2.pdf` | BERT: Pre-training of Deep Bidirectional Transformers |
| `1908.10084v1.pdf` | Sentence-BERT: Sentence Embeddings using Siamese BERT |
| `2005.11401v4.pdf` | RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP |
| `2401.08281v4.pdf` | Recent advances in AI/NLP research |

## Technical Details

| Component | Details |
|-----------|---------|
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| **Vector Store** | FAISS (Flat L2, exact search) |
| **LLM** | Groq — Llama 3.3 70B Versatile |
| **Chunk Size** | 500 characters, 100 overlap |
| **Retrieval** | Top-k=10 similarity search |
| **Framework** | LangChain + LangChain-Groq |
| **Frontend** | Streamlit with custom premium UI |

## Project Structure

```
Assignment-1_Agentic_AI/
├── research_papers/          # 5 PDF research papers
│   ├── 1706.03762v7.pdf
│   ├── 1810.04805v2.pdf
│   ├── 1908.10084v1.pdf
│   ├── 2005.11401v4.pdf
│   └── 2401.08281v4.pdf
├── faiss_index/              # Generated vector index
├── rag_implementation.ipynb  # Notebook with full pipeline
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── Assignment_1_Report.pdf   # Project report
└── README.md                 # This file
```

## Instructions to Run

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Internet connection (for Groq API)
- Groq API key ([Get one free](https://console.groq.com/))

### Step 1: Clone and Setup Environment

```bash
git clone https://github.com/harshitarora28/agentic-ai-lab.git
cd agentic-ai-lab
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your-actual-groq-api-key-here
```

Or set it as an environment variable:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your-actual-groq-api-key-here"

# Linux/Mac
export GROQ_API_KEY="your-actual-groq-api-key-here"
```

### Step 4: Verify PDFs

Ensure 5 PDF files are present in the `research_papers/` folder.

### Step 5a: Run Jupyter Notebook

```bash
jupyter notebook rag_implementation.ipynb
```

Run all cells sequentially. Update the API key in the configuration cell if not using environment variables.

### Step 5b: Run Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Tools & Libraries

| Library | Purpose |
|---------|---------|
| Python 3.10+ | Programming language |
| LangChain | RAG framework and document processing |
| LangChain-Groq | Groq LLM integration |
| FAISS (faiss-cpu) | Vector similarity search |
| Sentence Transformers | Text embedding generation |
| PyPDF | PDF text extraction |
| Streamlit | Web UI framework |
| NumPy / Pandas | Data processing |
| PyTorch | ML backend for transformers |

## Future Improvements

- **Semantic Chunking**: Split by topic boundaries instead of fixed character counts
- **Hybrid Search**: Combine dense vector + sparse BM25 retrieval
- **Reranking**: Cross-encoder reranker (ms-marco-MiniLM) for precision improvement
- **Metadata Filtering**: Filter by title, authors, year, section headers
- **Multi-turn Memory**: Conversation context across questions
- **Evaluation Framework**: Ground-truth test set with retrieval metrics (precision@k, recall@k, MRR)
