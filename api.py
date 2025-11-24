import json
import shutil
import time
import uuid
from pathlib import Path

import lancedb
import ollama
import pyarrow as pa
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from embed import get_embeddings
from ingest import extract_text_from_pdf, semantic_chunk
from retriever import RAGRetriever

app = FastAPI(title="Local RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
DATA_DIR = Path("./data")
UPLOAD_DIR = DATA_DIR / "uploads"
LANCEDB_DIR = DATA_DIR / "lancedb"
VECTOR_DIM = 2560

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LANCEDB_DIR.mkdir(parents=True, exist_ok=True)


def init_db(table_name="docs"):
    """Initialize or open the LanceDB table."""
    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        table = db.open_table(table_name)
    except Exception:
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("page", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM))
        ])
        table = db.create_table(table_name, schema=schema)
    return db, table


# Request models
class ProcessRequest(BaseModel):
    filename: str
    chunking_model: str

class EmbedRequest(BaseModel):
    embedding_model: str

class RetrieveRequest(BaseModel):
    query: str
    embedding_model: str
    reranker_model: str
    top_k: int = 5

class GenerateRequest(BaseModel):
    query: str
    context: str
    model: str


@app.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        models = ollama.list()['models']
        embedding_models = [m for m in models if 'embedding' in m['model'].lower()]
        generation_models = [m for m in models if 'embedding' not in m['model'].lower()]
        return {
            "models": models,
            "embedding_models": embedding_models,
            "generation_models": generation_models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_database():
    """Clear the database and uploaded files."""
    try:
        if LANCEDB_DIR.exists():
            shutil.rmtree(LANCEDB_DIR)
        LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
        
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a document."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "path": str(file_path)}


@app.post("/process")
async def process_document(request: ProcessRequest):
    """Extract text and chunk a document."""
    start_time = time.time()
    file_path = UPLOAD_DIR / request.filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Extract text
    extract_start = time.time()
    if file_path.suffix.lower() == ".pdf":
        extracted_data = extract_text_from_pdf(str(file_path))
    else:
        extracted_data = [(file_path.read_text(), 1)]
    extract_time = time.time() - extract_start
    
    # Chunk
    chunk_start = time.time()
    all_chunks = []
    for text, page_num in extracted_data:
        chunks = semantic_chunk(text, model_name=request.chunking_model)
        for chunk in chunks:
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk,
                "source": request.filename,
                "page": page_num,
                "vector": [0.0] * VECTOR_DIM
            })
    chunk_time = time.time() - chunk_start
    
    # Store in DB
    db_start = time.time()
    _, table = init_db()
    if all_chunks:
        table.add(all_chunks)
    db_time = time.time() - db_start
    
    return {
        "status": "success",
        "chunks_count": len(all_chunks),
        "pages_processed": len(extracted_data),
        "metrics": {
            "extraction_time": extract_time,
            "chunking_time": chunk_time,
            "db_insertion_time": db_time,
            "total_time": time.time() - start_time
        }
    }


@app.post("/embed")
async def embed_documents(request: EmbedRequest):
    """Generate embeddings for unprocessed chunks."""
    start_time = time.time()
    _, table = init_db()
    
    all_docs = table.to_arrow().to_pylist()
    zero_vector = [0.0] * VECTOR_DIM
    rows_to_update = [doc for doc in all_docs if doc['vector'] == zero_vector]
    
    if not rows_to_update:
        return {"status": "no_updates_needed", "count": 0}
    
    texts = [doc['text'] for doc in rows_to_update]
    ids = [doc['id'] for doc in rows_to_update]
    embeddings = get_embeddings(texts, model_name=request.embedding_model)
    
    update_start = time.time()
    for doc_id, embedding in zip(ids, embeddings):
        table.update(where=f"id = '{doc_id}'", values={"vector": embedding})
    update_time = time.time() - update_start
    
    return {
        "status": "success",
        "count": len(rows_to_update),
        "metrics": {
            "embedding_time": time.time() - start_time - update_time,
            "db_update_time": update_time,
            "total_time": time.time() - start_time
        }
    }


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Retrieve and rerank documents."""
    try:
        retriever = RAGRetriever(
            embedding_model=request.embedding_model,
            reranker_model=request.reranker_model
        )
        results, metrics = retriever.retrieve_context(request.query, top_k=request.top_k)
        return {"results": results, "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Generate an answer (non-streaming)."""
    start_time = time.time()
    
    prompt = f"""Answer based on the context below.

Context:
{request.context}

Question: {request.query}

Answer:"""

    response = ollama.chat(
        model=request.model,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ]
    )
    
    total_time = time.time() - start_time
    token_count = response.get('eval_count', len(response['message']['content'].split()))
    
    return {
        "answer": response['message']['content'],
        "metrics": {
            "generation_time": total_time,
            "token_count": token_count,
            "tokens_per_sec": token_count / total_time if total_time > 0 else 0
        }
    }


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Generate an answer with streaming."""
    prompt = f"""Answer based on the context below.

Context:
{request.context}

Question: {request.query}

Answer:"""

    def stream_response():
        start_time = time.time()
        token_count = 0
        
        for chunk in ollama.chat(
            model=request.model,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            stream=True
        ):
            token_count += 1
            elapsed = time.time() - start_time
            
            yield f"data: {json.dumps({
                'content': chunk['message']['content'],
                'token_count': token_count,
                'tokens_per_sec': token_count / elapsed if elapsed > 0 else 0,
                'done': chunk.get('done', False)
            })}\n\n"
        
        total_time = time.time() - start_time
        yield f"data: {json.dumps({
            'done': True,
            'metrics': {
                'generation_time': total_time,
                'token_count': token_count,
                'tokens_per_sec': token_count / total_time if total_time > 0 else 0
            }
        })}\n\n"
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
