# Local RAG Benchmarking

It's hard to know which local LLMs will run well on your Mac. Specs and benchmarks online don't tell the whole story—what matters is how a model performs on *your* hardware, with *your* workload.

This tool lets you spin up a RAG pipeline and immediately see the numbers that matter: **tokens/second** and **total latency** for each stage. Upload a document, ask questions, swap models, and compare.

## Quick Start

**Prerequisites:** Ollama running with at least one embedding model and one chat model.

```bash
# Pull models (if you don't have them)
ollama pull qwen3-embedding:4b
ollama pull qwen3:4b

# Start backend
pip install -r requirements.txt
python api.py

# Start frontend (new terminal)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173, upload a PDF, and start benchmarking.

## What You'll See

- **Ingestion time** — How long to chunk and embed your document
- **Retrieval time** — Vector search + reranking latency  
- **Generation speed** — Live tok/s as the model streams its response
- **Total pipeline time** — End-to-end latency

## Hardware Context

Developed and tested on an **M1 Pro MacBook (16GB RAM)**.

Models used during development:
- Embedding: `qwen3-embedding:4b`
- Generation: `qwen3:4b`, `llama3.1:8b`
- Reranker: `BAAI/bge-reranker-v2-m3`

## Stack

FastAPI · LanceDB · Ollama · React + Vite
