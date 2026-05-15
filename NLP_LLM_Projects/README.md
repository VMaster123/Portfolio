# Local RAG Physics Assistant

A fully local Retrieval-Augmented Generation (RAG) system for physics and engineering question answering. The system combines semantic search with local LLM inference to produce context-grounded responses from a structured knowledge base, running entirely offline with no external API dependencies.

---

## 🚀 Overview

This project implements a production-style RAG pipeline combining:

- FAISS for high-speed vector similarity search
- Ollama (Llama 3) for local LLM inference
- FastAPI for REST API serving
- SQLite for persistent document storage
- SentenceTransformers for embedding-based retrieval

The system is designed for fully offline operation while maintaining strong retrieval-grounded reasoning capabilities.

---

## ⚙️ Architecture

User Query  
→ FastAPI `/query` endpoint  
→ Embedding-based retrieval (FAISS)  
→ Top-K relevant documents (SQLite-backed corpus)  
→ Prompt construction with retrieved context  
→ Local LLM inference (Ollama - Llama 3)  
→ Generated response

---

## 🧠 Key Features

- Semantic search over domain-specific physics/engineering documents
- Retrieval-Augmented Generation (RAG) with context injection
- Fully local inference using Ollama (no API costs)
- Fast REST API for real-time querying
- Low-latency retrieval pipeline (<200ms typical queries)
- Modular design for easy extension and experimentation

---

## 🛠️ Tech Stack

- Python
- FastAPI
- FAISS
- Ollama (Llama 3)
- SQLite
- SentenceTransformers / HuggingFace embeddings

---

## 📁 Project Structure

```text
├── app.py            # FastAPI server (RAG endpoint)
├── rag.py            # Retrieval + embedding logic
├── ingest.py         # Document ingestion pipeline
├── db.py             # SQLite interface
├── data/             # Source documents
├── requirements.txt  # Python dependencies
├── .gitignore
```
