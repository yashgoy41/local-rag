# Local RAG Pipeline

Semantic document chunking, vector embedding, and retrieval system using local LLMs and LanceDB.

## Features

- **Semantic Chunking**: Uses `qwen3:4b-instruct-2507-q4_K_M` to intelligently split documents at topic boundaries
- **Vector Embeddings**: Uses `qwen3-embedding:4b` (2560 dimensions) for semantic search
- **Reranking**: Uses `CrossEncoder("BAAI/bge-reranker-v2-m3")` for accurate retrieval
- **PDF Support**: Extracts and normalizes text from PDFs and TXT files
- **Local & Private**: All processing happens locally with Ollama

## Quick Start

### 1. Prerequisites

```bash
# Install Ollama and pull models
ollama pull llama3.1:8b
ollama pull qwen3-embedding:4b

# Setup Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Add your PDF/TXT files to documents/
mkdir -p documents
# Place files in documents/

# Run the complete pipeline (ingest → embed → retrieve)
python main.py
```

Or run each step individually:

```bash
python ingest.py   # Step 1: Ingest documents
python embed.py    # Step 2: Generate embeddings
python retriever.py # Step 3: Test retrieval
```

## Usage

```python
from retriever import RAGRetriever

retriever = RAGRetriever()
results = retriever.retrieve_context("your query here")

for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")
    print(f"Source: {result['source']} (Page {result['page']})")
```

## Output

- **Database**: `data/lancedb/` - Vector database with chunks and embeddings
- **Debug**: `data/chunks_debug.jsonl` - Inspect generated chunks

## Viewing Your Data

Use Lance Data Viewer to browse your database:

```bash
docker pull ghcr.io/gordonmurray/lance-data-viewer:lancedb-0.24.3
docker run --rm -p 8080:8080 \
  -v $(pwd)/data/lancedb:/data:ro \
  ghcr.io/gordonmurray/lance-data-viewer:lancedb-0.24.3
```

Open http://localhost:8080 to explore tables, schemas, and vector visualizations.

## How It Works

1. **Extract**: Reads PDF/TXT files with whitespace normalization
2. **Chunk**: LLM inserts `¶` tokens at semantic boundaries
3. **Split**: Creates chunks by splitting on `¶`
4. **Embed**: Generates 2560-dim vectors for each chunk
5. **Retrieve**: Vector search (top 25) + reranking (top 5)
6. **Store**: Saves to LanceDB with metadata (source, page)

## Requirements

- Python 3.10+
- Ollama running locally
- Docker (optional, for data viewer)
