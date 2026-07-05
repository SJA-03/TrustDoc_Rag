import json
import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from app.rag.retriever import ChromaRetriever


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9가-힣_+-]+", text.lower())


def normalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = chunk.get("metadata")

    if metadata is None:
        metadata = {
            "source_file": chunk.get("source_file"),
            "page_number": chunk.get("page_number"),
            "chunk_id": chunk.get("chunk_id"),
            "chunking_strategy": chunk.get("chunking_strategy"),
        }
        if chunk.get("section_title") is not None:
            metadata["section_title"] = chunk["section_title"]

    return {
        "text": chunk["text"],
        "metadata": metadata,
    }


def chunk_key(chunk: Dict[str, Any]) -> str:
    meta = chunk["metadata"]
    return f"{meta.get('source_file')}::{meta.get('page_number')}::{meta.get('chunk_id')}"


class BM25Retriever:
    def __init__(self, chunks_path: str):
        self.chunks_path = chunks_path
        self.chunks = self._load_chunks(chunks_path)
        self.tokenized_corpus = [tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_chunks(self, chunks_path: str) -> List[Dict[str, Any]]:
        with open(chunks_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        return [normalize_chunk(chunk) for chunk in raw_chunks]

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        results = []

        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = dict(self.chunks[idx])
            chunk["rank"] = rank
            chunk["bm25_score"] = float(scores[idx])
            results.append(chunk)

        return results


class HybridRetriever:
    def __init__(
        self,
        chunks_path: str,
        collection_name: str,
        rrf_k: int = 60,
    ):
        self.bm25_retriever = BM25Retriever(chunks_path=chunks_path)
        self.dense_retriever = ChromaRetriever(collection_name=collection_name)
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 10,
        bm25_top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        dense_results = self.dense_retriever.retrieve(query, top_k=dense_top_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_top_k)

        fused: Dict[str, Dict[str, Any]] = {}

        for result in dense_results:
            key = chunk_key(result)

            if key not in fused:
                fused[key] = dict(result)
                fused[key]["dense_rank"] = result["rank"]
                fused[key]["bm25_rank"] = None
                fused[key]["hybrid_score"] = 0.0

            fused[key]["dense_rank"] = result["rank"]
            fused[key]["distance"] = result.get("distance")
            fused[key]["hybrid_score"] += 1.0 / (self.rrf_k + result["rank"])

        for result in bm25_results:
            key = chunk_key(result)

            if key not in fused:
                fused[key] = dict(result)
                fused[key]["dense_rank"] = None
                fused[key]["bm25_rank"] = result["rank"]
                fused[key]["distance"] = None
                fused[key]["hybrid_score"] = 0.0

            fused[key]["bm25_rank"] = result["rank"]
            fused[key]["bm25_score"] = result.get("bm25_score")
            if (
                not fused[key]["metadata"].get("section_title")
                and result["metadata"].get("section_title")
            ):
                fused[key]["metadata"]["section_title"] = result["metadata"]["section_title"]
            fused[key]["hybrid_score"] += 1.0 / (self.rrf_k + result["rank"])

        results = sorted(
            fused.values(),
            key=lambda chunk: chunk["hybrid_score"],
            reverse=True,
        )[:top_k]

        for rank, chunk in enumerate(results, start=1):
            chunk["rank"] = rank

        return results
