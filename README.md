# Local RAG Pipeline - Ingestion Module

A semantic document chunking and ingestion system using local LLMs (Ollama) and LanceDB.

## What It Does

Reads PDF/TXT files, uses `llama3.1:8b` to intelligently chunk them at semantic boundaries, and stores them in a LanceDB vector database.

## Setup

1. **Install Ollama** and pull the model:
   ```bash
   ollama pull llama3.1:8b
   ```

2. **Create virtual environment and install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Add documents**:
   ```bash
   mkdir -p documents
   # Place your PDF/TXT files in documents/
   ```

4. **Run ingestion**:
   ```bash
   python ingest.py
   ```

## Output

- **Database**: `data/lancedb/` - LanceDB vector store
- **Debug**: `data/chunks_debug.jsonl` - Inspect generated chunks

## How It Works

1. Extracts text from PDFs (with whitespace normalization) and TXT files
2. Sends text to `llama3.1:8b` with a semantic parsing prompt
3. LLM inserts `¶` tokens at topic boundaries
4. Splits on `¶` to create semantic chunks
5. Stores chunks in LanceDB with metadata (source, page, etc.)

## Requirements

- Python 3.10+
- Ollama running locally
- Mac M1 (Apple Silicon) or compatible
