import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from src.embedder_faiss import Embedder, FaissStore


class Retriever:
    def __init__(self):
        self.embedder = Embedder()
        self.store = FaissStore()
        self.index, self.chunks = self.store.load_index()
        self.bm25 = None
        if self.chunks:
            self._init_bm25(self.chunks)

    def _init_bm25(self, chunks):
        tokenized_corpus = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def update_index(self, chunks):
        """Creates embeddings and updates FAISS index."""
        embeddings = self.embedder.generate_embedding(chunks=chunks)
        self.store.save_index(embeddings, chunks)
        self.index, self.chunks = self.store.load_index()
        self._init_bm25(chunks)

    def vector_search(self, query, k=3):
        """Retrieval from vector DB using FAISS (IndexFlatIP)."""
        query = [query]
        if self.index is None:
            return []

        query_embedding = self.embedder.generate_embedding(chunks=query)
        # FAISS search expects a 2D array (num_queries, dimension),
        # which generate_embedding already provides for our list of chunks.

        # Normalize query for Inner Product search
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append(
                    {
                        "chunk": self.chunks[idx],
                        "score": float(distances[0][i]),
                        "id": int(idx),
                    }
                )
        return results

    def hybrid_search(self, query, k=3):
        """Hybrid retrieval: BM25 keyword search + Vector reranking."""
        if not self.bm25 or self.index is None:
            return []

        # 1. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top candidates for reranking
        top_n = k * 5
        if top_n > len(bm25_scores):
            top_n = len(bm25_scores)
        top_n_indices = np.argsort(bm25_scores)[::-1][:top_n]

        # 2. Vector Rerank
        # For simplicity and consistency with index changes,
        # we'll use the scores from a full vector search filtered by top_n_indices
        vector_results = self.vector_search(query, k=len(self.chunks))
        vector_scores = {res["id"]: res["score"] for res in vector_results}

        hybrid_scores = []
        for idx in top_n_indices:
            hybrid_scores.append(
                {
                    "chunk": self.chunks[idx],
                    "score": vector_scores.get(
                        idx, -1.0
                    ),  # Default low score for Inner Product
                    "id": int(idx),
                    "bm25_score": float(bm25_scores[idx]),
                }
            )

        hybrid_scores.sort(
            key=lambda x: x["score"], reverse=True
        )  # Higher score is better for IndexFlatIP
        return hybrid_scores[:k]
