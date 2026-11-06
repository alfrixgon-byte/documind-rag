# 📄 DocuMind: Enterprise RAG System with Hybrid Search & Re-Ranking

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://www.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-Chroma-red.svg)](https://www.trychroma.com/)

DocuMind is an **Advanced Retrieval-Augmented Generation (RAG)** pipeline designed to provide zero-hallucination, context-aware answers from complex PDF documents and academic research papers. 

Unlike naive RAG implementations, DocuMind utilizes a **two-stage retrieval pipeline with Cross-Encoder Re-ranking** to ensure maximum precision and contextual recall.

---

## 🔑 Key Features & Innovations

- **Advanced Chunking Strategy:** Implements `RecursiveCharacterTextSplitter` with tuned overlap to preserve semantic boundaries.
- **Two-Stage Retrieval (Re-Ranking):** Uses `BAAI/bge-small-en-v1.5` embeddings for initial retrieval, followed by a `BAAI/bge-reranker-base` Cross-Encoder to re-rank chunks for high precision.
- **Strict Citation Engine:** Enforces page-level source citations for every generated answer to eliminate hallucinations.
- **RAGAS Evaluation Framework:** Integrated automated benchmark scripts evaluating *Faithfulness*, *Answer Relevance*, and *Context Recall*.
- **Interactive UI & API:** Full Streamlit front-end with file uploading and source highlighting.

---

## 🛠️ System Architecture

```text
+------------------+     +-----------------------+     +-----------------------+
|  Input PDF Doc   | --> |  Recursive Chunking   | --> | BAAI/bge Embeddings   |
+------------------+     +-----------------------+     +-----------------------+
                                                                   |
                                                                   v
+------------------+     +-----------------------+     +-----------------------+
| Streamlit / User | <-- | GPT-4o-mini Synthesis | <-- |  Chroma Vector Store  |
+------------------+     +-----------------------+     +-----------------------+
                                     ^                             |
                                     |                             v
                                  +------------------------------------+
                                  | BAAI Cross-Encoder (Re-ranker k=3) |
                                  +------------------------------------+

