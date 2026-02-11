# RAG System for Research Paper Q&A 📚

A complete **Retrieval-Augmented Generation (RAG)** system that enables intelligent question-answering over research papers using semantic search and large language models.

![RAG Architecture](https://img.shields.io/badge/RAG-System-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![LangChain](https://img.shields.io/badge/LangChain-Framework-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [RAG Pipeline Details](#rag-pipeline-details)
- [Future Improvements](#future-improvements)
- [Project Structure](#project-structure)
- [License](#license)

---

## 🎯 Problem Statement

**Objective**: Build a RAG system that answers questions about research papers by combining semantic search with LLM generation.

**Challenges Addressed**:
1. Research papers contain dense technical information
2. Traditional keyword search fails to capture semantic meaning
3. LLMs have limited context windows
4. Need for accurate, citation-backed answers

**Solution**: Our RAG system retrieves relevant document chunks using semantic embeddings and augments LLM prompts with this context for accurate, source-backed answers.

---

## ✨ Features

- 📄 **Multi-PDF Processing**: Load and process multiple research papers simultaneously
- 🔍 **Semantic Search**: Find relevant information using meaning, not just keywords
- 🤖 **AI-Powered Answers**: Generate accurate responses using Google Gemini Pro
- 📑 **Source Citations**: Every answer includes references to source documents
- 💬 **Interactive UI**: User-friendly Streamlit interface with chat history
- ⚡ **Fast Retrieval**: FAISS vector database for efficient similarity search
- 🎨 **Clean Interface**: Modern, responsive web design

---

## 🏗️ Architecture

### RAG Pipeline Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

1. DATA INGESTION
   📄 PDF Files → PyPDF Loader → Text Extraction

2. TEXT PROCESSING
   📝 Text Chunking
   └─ RecursiveCharacterTextSplitter
      ├─ Chunk Size: 1000 characters
      └─ Overlap: 200 characters

3. EMBEDDING GENERATION
   🧠 Sentence Transformers (all-MiniLM-L6-v2)
   └─ 384-dimensional embeddings

4. VECTOR STORAGE
   💾 FAISS Vector Database
   └─ Fast similarity search & indexing

5. QUERY PROCESSING
   ❓ User Query → Query Embedding

6. RETRIEVAL
   🔍 Similarity Search → Top-k Relevant Chunks (k=4)

7. GENERATION
   🤖 Context + Query → Gemini Pro LLM → Answer + Citations
```

### Component Details

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document Loader** | PyPDF | Extract text from PDF files |
| **Text Splitter** | LangChain RecursiveCharacterTextSplitter | Chunk documents intelligently |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Convert text to vectors |
| **Vector DB** | FAISS | Store & search embeddings |
| **LLM** | Google Gemini Pro | Generate contextual answers |
| **Framework** | LangChain | Orchestrate RAG pipeline |
| **UI** | Streamlit | Interactive web interface |

---

## 📚 Dataset

**Type**: PDF Research Papers

**Source**: Public research papers from arXiv

**Papers Included**:
1. `1706.03762v7.pdf` - "Attention Is All You Need" (Transformer)
2. `1810.04805v2.pdf` - "BERT: Pre-training of Deep Bidirectional Transformers"
3. `1908.10084v1.pdf` - NLP/ML Research Paper
4. `2005.11401v4.pdf` - AI/ML Research Paper
5. `2401.08281v4.pdf` - Recent AI Research Paper

**Total**: 5 research papers covering foundational and cutting-edge AI/NLP topics

---

## 🛠️ Technologies Used

### Core Libraries

```
langchain              # RAG framework
langchain-community    # Community integrations
langchain-google-genai # Gemini integration
google-generativeai    # Google Gemini API
pypdf                  # PDF processing
sentence-transformers  # Embedding model
faiss-cpu              # Vector database
streamlit              # Web UI
```

### Key Technologies

- **Python 3.8+**: Programming language
- **LangChain**: RAG framework for building LLM applications
- **FAISS**: Facebook AI Similarity Search for vector operations
- **Sentence Transformers**: State-of-the-art text embeddings
- **Google Gemini Pro**: Large language model for generation
- **Streamlit**: Fast web app framework

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Step 1: Clone or Download Project

```bash
cd Assignment-1_Agentic_AI
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify PDF Files

Ensure your research papers are in the `research_papers/` folder:

```
research_papers/
├── 1706.03762v7.pdf
├── 1810.04805v2.pdf
├── 1908.10084v1.pdf
├── 2005.11401v4.pdf
└── 2401.08281v4.pdf
```

---

## 🚀 Usage

### Option 1: Jupyter Notebook (Recommended for Learning)

1. **Open the notebook**:
   ```bash
   jupyter notebook rag_implementation.ipynb
   ```

2. **Run cells sequentially** to:
   - Load PDF documents
   - Create embeddings
   - Build vector database
   - Test with sample queries

3. **Experiment with queries** in the test cells

### Option 2: Streamlit UI (Best User Experience)

1. **Launch the app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the interface**:
   - Open your browser to `http://localhost:8501`
   - The UI will load automatically

3. **Start asking questions**:
   - Type your question in the input box
   - Click "Ask" or press Enter
   - View the answer and source citations
   - Try sample questions from the sidebar

### Sample Questions to Try

```
✅ "What is the Transformer architecture and what are its key components?"
✅ "What is BERT and how does it differ from previous language models?"
✅ "Explain the attention mechanism in neural networks"
✅ "What are the main contributions of the research papers?"
✅ "Compare self-attention and cross-attention mechanisms"
```

---

## 🔍 RAG Pipeline Details

### 1. Text Chunking Strategy

**Configuration**:
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Strategy**: RecursiveCharacterTextSplitter

**Rationale**:
- **1000 chars** provides 2-3 paragraphs of context
- **200 char overlap** prevents information loss at boundaries
- **Recursive splitting** respects natural text structure (paragraphs, sentences)
- Balances context preservation with retrieval precision

### 2. Embedding Model

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

**Specifications**:
- Embedding Dimensions: 384
- Max Sequence Length: 256 tokens
- Model Size: ~80MB
- License: Apache 2.0

**Why This Model?**
1. ✅ **Efficient**: Runs on CPU, no GPU required
2. ✅ **Accurate**: Excellent semantic similarity performance
3. ✅ **Pre-trained**: Ready to use, no fine-tuning needed
4. ✅ **Open Source**: No API costs
5. ✅ **Production-Ready**: Battle-tested in real applications

### 3. Vector Database

**Database**: FAISS (Facebook AI Similarity Search)

**Key Features**:
- Lightning-fast similarity search
- Supports billion-scale databases
- Multiple index types (Flat, IVF, HNSW)
- Disk persistence (save/load indexes)
- No external server needed

**Advantages**:
- ⚡ **Speed**: Optimized C++ implementation
- 📈 **Scalability**: Handles large document collections
- 💾 **Persistence**: Save index to disk for reuse
- 🔧 **Easy Setup**: Embedded, no infrastructure required

### 4. Retrieval Configuration

**Settings**:
- Search Type: Similarity search
- Top-k: 4 most relevant chunks
- Chain Type: "Stuff" (all context in one prompt)

### 5. LLM Configuration

**Model**: Google Gemini Pro

**Settings**:
- Temperature: 0.3 (more factual, less creative)
- Context Window: Large (handles 4 chunks + question)
- API: Google Generative AI

---

## 🚀 Future Improvements

### 1. Better Chunking
- **Semantic Chunking**: Split by topics, not fixed sizes
- **Hierarchical Chunks**: Multi-level (sections → paragraphs → sentences)
- **Metadata-Aware**: Preserve document structure (headings, figures)
- **Impact**: 15-20% better retrieval accuracy

### 2. Reranking / Hybrid Search
- **Hybrid Retrieval**: Combine dense (embeddings) + sparse (BM25) search
- **Cross-Encoder Reranking**: Rerank top results with more powerful model
- **Query Expansion**: Expand queries with synonyms
- **Impact**: 25-30% improvement in precision

### 3. Metadata Filtering
- **Rich Metadata**: Extract authors, dates, sections
- **Filtered Search**: "Show me papers from 2023 about transformers"
- **Citation Tracking**: Link answers to specific papers
- **Impact**: Better user control and targeted results

### 4. UI Enhancements
- ✅ Chat history (already implemented)
- 🔄 Multi-turn conversations with memory
- 📤 Document upload for new PDFs
- 📊 Visualization of similarity scores
- 📱 Mobile-responsive design

### 5. Performance Optimization
- **Quantization**: Reduce embedding memory by 50%
- **Batch Processing**: Process multiple queries in parallel
- **Caching**: Cache frequent questions
- **GPU Acceleration**: Faster embedding generation

### 6. Accuracy Enhancements
- **Few-Shot Examples**: Include Q&A examples in prompt
- **Confidence Scoring**: Return confidence levels
- **Hallucination Detection**: Detect when LLM invents information
- **Multi-Query Retrieval**: Generate query variations

---

## 📁 Project Structure

```
Assignment-1_Agentic_AI/
│
├── research_papers/              # PDF research papers
│   ├── 1706.03762v7.pdf
│   ├── 1810.04805v2.pdf
│   ├── 1908.10084v1.pdf
│   ├── 2005.11401v4.pdf
│   └── 2401.08281v4.pdf
│
├── faiss_index/                  # Generated vector database (after first run)
│   ├── index.faiss
│   └── index.pkl
│
├── rag_implementation.ipynb      # Complete notebook with explanations
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore file
```

---

## 📊 System Requirements

**Minimum**:
- CPU: 2 cores
- RAM: 4GB
- Storage: 1GB free space
- Internet: Required for Gemini API

**Recommended**:
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 2GB free space
- Internet: Stable connection

---

## 🔑 API Keys

This project uses Google Gemini Pro API. The API key is already configured in the code:

```python
GOOGLE_API_KEY = "AIzaSyDSS-MuRRsNrbPnQoVFm8W3t9FHN7UvnDI"
```

**Note**: For production use, store API keys in environment variables:

```python
import os
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'langchain'"
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: "API key not valid"
**Solution**: Verify Gemini API key is correct and active

### Issue: "Out of memory"
**Solution**: Reduce chunk size or process fewer PDFs

### Issue: "FAISS index not found"
**Solution**: Run notebook first to create index, or delete `faiss_index/` folder to rebuild

---

## 📝 Assignment Submission Checklist

✅ **Problem Statement** - Clearly defined in README and notebook
✅ **Dataset** - 5 PDF research papers included
✅ **RAG Architecture** - Block diagram and detailed explanation
✅ **Text Chunking** - 1000 chars with 200 overlap, rationale provided
✅ **Embedding Model** - all-MiniLM-L6-v2 with justification
✅ **Vector Database** - FAISS implementation
✅ **Notebook** - Complete step-by-step implementation
✅ **Test Queries** - 3+ test queries with outputs
✅ **Future Improvements** - Comprehensive list with explanations
✅ **Streamlit UI** - Fully functional web interface
✅ **README** - Complete documentation

---

## 👨‍💻 Author

**Assignment**: RAG System Implementation
**Course**: Agentic AI
**Date**: February 2026

---

## 📄 License

This project is created for educational purposes as part of an Agentic AI assignment.

---

## 🙏 Acknowledgments

- **LangChain**: For the excellent RAG framework
- **Google**: For Gemini Pro API
- **Hugging Face**: For open-source embedding models
- **Meta**: For FAISS vector search library
- **Streamlit**: For rapid UI development

---

## 📞 Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the Jupyter notebook comments
3. Check LangChain documentation: https://python.langchain.com/

---

**Happy Learning! 🚀📚**
