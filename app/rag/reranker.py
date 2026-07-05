from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        reranked_chunks = []

        for chunk, score in zip(chunks, scores):
            new_chunk = dict(chunk)
            new_chunk["original_rank"] = chunk.get("rank")
            new_chunk["rerank_score"] = float(score)
            reranked_chunks.append(new_chunk)

        reranked_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        for idx, chunk in enumerate(reranked_chunks, start=1):
            chunk["rank"] = idx

        return reranked_chunks
