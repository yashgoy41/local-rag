import gc
import time
import torch
import ollama
import lancedb
from sentence_transformers import CrossEncoder

class RAGRetriever:
    def __init__(self, db_path="./data/lancedb", table_name="docs"):
        """
        Initialize the RAG Retriever.
        
        Args:
            db_path: Path to LanceDB database
            table_name: Name of the table to search
        """
        self.db_path = db_path
        self.table_name = table_name
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
        
    def load_reranker(self):
        """
        Load the CrossEncoder reranker model with MPS (Metal) device support.
        
        Returns:
            CrossEncoder model configured for MPS with float16 precision
        """
        # Set default dtype to float16 to save RAM
        torch.set_default_dtype(torch.float16)
        
        # Load the reranker model (BAAI/bge-reranker-v2-m3 is stable and accurate)
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        
        # Move to MPS (Metal Performance Shaders) device
        reranker.model = reranker.model.to("mps")
        
        return reranker
    
    def retrieve_context(self, query: str, top_k: int = 5):
        """
        Retrieve the most relevant chunks for a given query.
        
        This method performs:
        1. Query embedding using Ollama
        2. Vector search in LanceDB (top 25 candidates)
        3. Reranking using CrossEncoder
        4. Resource cleanup
        
        Args:
            query: The search query string
            top_k: Number of top results to return (default: 5)
            
        Returns:
            List of dicts containing top_k chunks with metadata:
            [{"text": str, "source": str, "page": int, "score": float}, ...]
        """
        total_start = time.time()
        
        # Step 1: Embed the query using Ollama
        print("Step 1: Embedding query...")
        embed_start = time.time()
        response = ollama.embeddings(
            model='qwen3-embedding:4b',
            prompt=query
        )
        query_vector = response['embedding']
        embed_time = time.time() - embed_start
        
        # Estimate tokens
        query_tokens = len(query) // 4
        query_tps = query_tokens / embed_time if embed_time > 0 else 0
        print(f"  Query embedded in {embed_time:.2f}s (~{query_tokens} tokens, ~{query_tps:.1f} tok/s)")
        
        # Step 2: Search LanceDB for top 25 candidates
        print("\nStep 2: Searching LanceDB...")
        search_start = time.time()
        results = self.table.search(query_vector).limit(25).to_list()
        print(f"  Found {len(results)} candidates in {time.time() - search_start:.2f}s")
        
        # Step 3: Load reranker and rerank candidates
        print("\nStep 3: Reranking with CrossEncoder...")
        rerank_start = time.time()
        reranker = self.load_reranker()
        
        # Prepare query-document pairs for reranking
        pairs = [(query, result['text']) for result in results]
        
        # Calculate reranking scores (batch processing works with BAAI model)
        scores = reranker.predict(pairs)
        
        # Combine results with scores
        for i, result in enumerate(results):
            result['score'] = float(scores[i])
        
        # Sort by score (descending) and select top_k
        ranked_results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        print(f"  Reranking completed in {time.time() - rerank_start:.2f}s")
        
        # Step 4: Clean up reranker to free memory
        print("\nStep 4: Cleaning up resources...")
        cleanup_start = time.time()
        del reranker
        gc.collect()
        torch.mps.empty_cache()
        print(f"  Cleanup completed in {time.time() - cleanup_start:.2f}s")
        
        # Format output
        output = []
        for result in ranked_results:
            output.append({
                "text": result['text'],
                "source": result['source'],
                "page": result['page'],
                "score": result['score']
            })
        
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"Total retrieval time: {total_time:.2f}s")
        print(f"{'='*60}\n")
        
        return output

# Example usage
if __name__ == "__main__":
    retriever = RAGRetriever()
    
    # Test query
    query = "What did Rutger say about talent"
    results = retriever.retrieve_context(query)
    
    print(f"Query: {query}\n")
    print(f"Top {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   Source: {result['source']} (Page {result['page']})")
        print(f"   Text: {result['text'][:200]}...")
        print()
