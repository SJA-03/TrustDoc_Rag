from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.rag.retriever import ChromaRetriever
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import CrossEncoderReranker
from app.rag.build_prompt import build_rag_prompt
from app.rag.llm_client import GeminiClient


app = FastAPI(
    title="TrustDoc RAG API",
    description="PDF-based RAG API with dense retrieval, hybrid retrieval, and optional reranking",
    version="0.2.0",
)


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    collection: str = Field(
        default="trustdoc_os_paragraph",
        description="Chroma collection name",
    )
    chunks_path: str = Field(
        default="data/processed/chunks_paragraph_all.json",
        description="Chunk JSON path for hybrid retrieval",
    )
    retrieval_mode: Literal["dense", "hybrid"] = Field(
        default="dense",
        description="Retrieval mode: dense or hybrid",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    initial_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Initial candidate size for dense retrieval before reranking",
    )
    dense_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Dense candidate size for hybrid retrieval",
    )
    bm25_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="BM25 candidate size for hybrid retrieval",
    )
    use_rerank: bool = Field(default=True)


class RetrievedSource(BaseModel):
    rank: int
    source_file: Optional[str]
    page_number: Optional[int]
    chunk_id: Optional[str]
    distance: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    text_preview: str


class RAGQueryResponse(BaseModel):
    query: str
    collection: str
    retrieval_mode: str
    use_rerank: bool
    answer: str
    retrieved_sources: List[RetrievedSource]


_dense_retriever_cache: Dict[str, ChromaRetriever] = {}
_hybrid_retriever_cache: Dict[str, HybridRetriever] = {}
_reranker: Optional[CrossEncoderReranker] = None
_llm_client: Optional[GeminiClient] = None


def get_dense_retriever(collection_name: str) -> ChromaRetriever:
    if collection_name not in _dense_retriever_cache:
        _dense_retriever_cache[collection_name] = ChromaRetriever(
            collection_name=collection_name
        )

    return _dense_retriever_cache[collection_name]


def get_hybrid_retriever(chunks_path: str, collection_name: str) -> HybridRetriever:
    cache_key = f"{chunks_path}::{collection_name}"

    if cache_key not in _hybrid_retriever_cache:
        _hybrid_retriever_cache[cache_key] = HybridRetriever(
            chunks_path=chunks_path,
            collection_name=collection_name,
        )

    return _hybrid_retriever_cache[cache_key]


def get_reranker() -> CrossEncoderReranker:
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoderReranker()

    return _reranker


def get_llm_client() -> GeminiClient:
    global _llm_client

    if _llm_client is None:
        _llm_client = GeminiClient()

    return _llm_client


def to_source_response(chunk: Dict[str, Any]) -> RetrievedSource:
    metadata = chunk["metadata"]

    return RetrievedSource(
        rank=chunk["rank"],
        source_file=metadata.get("source_file"),
        page_number=metadata.get("page_number"),
        chunk_id=metadata.get("chunk_id"),
        distance=chunk.get("distance"),
        dense_rank=chunk.get("dense_rank"),
        bm25_rank=chunk.get("bm25_rank"),
        hybrid_score=chunk.get("hybrid_score"),
        rerank_score=chunk.get("rerank_score"),
        text_preview=chunk["text"][:300],
    )



class RetrieveResponse(BaseModel):
    query: str
    collection: str
    retrieval_mode: str
    use_rerank: bool
    retrieved_sources: List[RetrievedSource]


def retrieve_chunks(request: RAGQueryRequest) -> List[Dict[str, Any]]:
    if request.retrieval_mode == "hybrid":
        retriever = get_hybrid_retriever(
            chunks_path=request.chunks_path,
            collection_name=request.collection,
        )
        chunks = retriever.retrieve(
            query=request.query,
            top_k=request.initial_top_k,
            dense_top_k=request.dense_top_k,
            bm25_top_k=request.bm25_top_k,
        )
    else:
        retriever = get_dense_retriever(request.collection)
        chunks = retriever.retrieve(
            query=request.query,
            top_k=request.initial_top_k if request.use_rerank else request.top_k,
        )

    if request.use_rerank:
        reranker = get_reranker()
        chunks = reranker.rerank(request.query, chunks)

    return chunks[: request.top_k]

@app.get("/")
def root():
    return {
        "message": "TrustDoc RAG API is running",
        "docs": "/docs",
        "endpoints": [
            "POST /rag/query",
        ],
    }


@app.post("/rag/retrieve", response_model=RetrieveResponse)
def rag_retrieve(request: RAGQueryRequest):
    chunks = retrieve_chunks(request)

    return RetrieveResponse(
        query=request.query,
        collection=request.collection,
        retrieval_mode=request.retrieval_mode,
        use_rerank=request.use_rerank,
        retrieved_sources=[
            to_source_response(chunk) for chunk in chunks
        ],
    )


@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(request: RAGQueryRequest):
    final_chunks = retrieve_chunks(request)

    prompt = build_rag_prompt(
        query=request.query,
        chunks=final_chunks,
    )

    llm_client = get_llm_client()
    answer = llm_client.generate(prompt)

    return RAGQueryResponse(
        query=request.query,
        collection=request.collection,
        retrieval_mode=request.retrieval_mode,
        use_rerank=request.use_rerank,
        answer=answer,
        retrieved_sources=[
            to_source_response(chunk) for chunk in final_chunks
        ],
    )
