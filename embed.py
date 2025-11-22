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
    for text in texts:
        response = ollama.embeddings(
            model='qwen3-embedding:4b',
            prompt=text
        )
        embeddings.append(response['embedding'])
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
    
    # Batch processing
    BATCH_SIZE = 64
    total_batches = (len(rows_to_update) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(total_batches), desc="Embedding batches"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(rows_to_update))
        
        batch = rows_to_update[start_idx:end_idx]
        texts = [doc['text'] for doc in batch]
        ids = [doc['id'] for doc in batch]
        
        # Generate embeddings
        embeddings = get_embeddings(texts)
        
        # Update rows in LanceDB
        for doc_id, embedding in zip(ids, embeddings):
            table.update(
                where=f"id = '{doc_id}'",
                values={"vector": embedding}
            )
    
    print("Embedding complete.")

if __name__ == "__main__":
    main()
