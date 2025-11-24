import gc
import time

import lancedb
import ollama
import torch
from sentence_transformers import CrossEncoder


class RAGRetriever:
    def __init__(
        self,
        db_path: str = "./data/lancedb",
        table_name: str = "docs",
        embedding_model: str = 'qwen3-embedding:4b',
        reranker_model: str = 'BAAI/bge-reranker-v2-m3'
    ):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.db = lancedb.connect(db_path)
        
        try:
            self.table = self.db.open_table(table_name)
        except Exception:
            self.table = None

    def retrieve_context(self, query: str, top_k: int = 5) -> tuple[list, dict]:
        """Retrieve and rerank relevant chunks for a query."""
        metrics = {}
        total_start = time.time()
        
        if self.table is None:
            return [], metrics
        
        # Embed query
        embed_start = time.time()
        query_embedding = ollama.embeddings(model=self.embedding_model, prompt=query)['embedding']
        metrics['embedding_time'] = time.time() - embed_start
        
        # Vector search
        search_start = time.time()
        results = self.table.search(query_embedding).limit(25).to_list()
        metrics['vector_search_time'] = time.time() - search_start
        
        if not results:
            return [], metrics
        
        # Rerank
        rerank_start = time.time()
        reranker = CrossEncoder(self.reranker_model, automodel_args={"torch_dtype": torch.float16})
        pairs = [[query, r['text']] for r in results]
        scores = reranker.predict(pairs)
        
        for i, result in enumerate(results):
            result['score'] = float(scores[i])
        
        ranked_results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        metrics['reranking_time'] = time.time() - rerank_start
        
        # Cleanup
        cleanup_start = time.time()
        del reranker
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        metrics['cleanup_time'] = time.time() - cleanup_start
        
        metrics['total_time'] = time.time() - total_start
        
        return [
            {"text": r['text'], "source": r['source'], "page": r['page'], "score": r['score']}
            for r in ranked_results
        ], metrics
