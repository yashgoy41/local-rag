import time
import ollama
import lancedb
from tqdm import tqdm

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Ollama.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors (2560 dimensions each)
    """
    embeddings = []
    start_time = time.time()
    for text in texts:
        response = ollama.embeddings(
            model='qwen3-embedding:4b',
            prompt=text
        )
        embeddings.append(response['embedding'])
    elapsed = time.time() - start_time
    if len(texts) > 0:
        print(f"  Embedded {len(texts)} chunks in {elapsed:.2f}s ({elapsed/len(texts):.2f}s/chunk)")
    return embeddings

def main():
    # Connect to LanceDB
    db = lancedb.connect("./data/lancedb")
    table = db.open_table("docs")
    
    # Get all documents
    all_docs = table.to_arrow().to_pylist()
    
    # Find rows where vector is all zeros (placeholder)
    zero_vector = [0.0] * 2560
    rows_to_update = [doc for doc in all_docs if doc['vector'] == zero_vector]
    
    if len(rows_to_update) == 0:
        print("No rows with placeholder vectors found. All embeddings are already generated.")
        return
    
    print(f"Found {len(rows_to_update)} rows to embed.")
    
    total_start = time.time()
    total_tokens = 0
    total_embed_time = 0
    
    # Batch processing
    BATCH_SIZE = 64
    total_batches = (len(rows_to_update) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(total_batches), desc="Embedding batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(rows_to_update))
        
        batch = rows_to_update[start_idx:end_idx]
        texts = [doc['text'] for doc in batch]
        ids = [doc['id'] for doc in batch]
        
        # Generate embeddings and track time
        batch_start = time.time()
        embeddings = get_embeddings(texts)
        batch_time = time.time() - batch_start
        total_embed_time += batch_time
        
        # Estimate tokens (approximate: 1 token ≈ 4 characters)
        batch_tokens = sum(len(text) // 4 for text in texts)
        total_tokens += batch_tokens
        
        # Update rows in LanceDB
        for doc_id, embedding in zip(ids, embeddings):
            table.update(
                where=f"id = '{doc_id}'",
                values={"vector": embedding}
            )
    
    total_time = time.time() - total_start
    avg_tps = total_tokens / total_embed_time if total_embed_time > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"EMBEDDING COMPLETE")
    print(f"{'='*60}")
    print(f"Total chunks: {len(rows_to_update)}")
    print(f"Total tokens: ~{total_tokens}")
    print(f"Embedding time: {total_embed_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: ~{avg_tps:.1f} tok/s")
    print(f"Average: {total_time/len(rows_to_update):.2f}s/chunk")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
