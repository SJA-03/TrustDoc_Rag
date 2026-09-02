"""Unified, reproducible benchmark runner for retrieval pipelines."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.metrics import (  # noqa: E402
    aggregate_ranks,
    bootstrap_confidence_intervals,
    build_query_record,
    find_answer_rank,
    load_questions,
    summarize_latency,
)
from app.eval.report_utils import normalize_top_result  # noqa: E402
from app.rag.hybrid_retriever import BM25Retriever, HybridRetriever  # noqa: E402
from app.rag.reranker import CrossEncoderReranker  # noqa: E402
from app.rag.retriever import ChromaRetriever  # noqa: E402


METHOD_ORDER: Tuple[str, ...] = (
    "dense",
    "bm25",
    "dense_rerank",
    "hybrid",
    "hybrid_rerank",
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
Pipeline = Callable[[str], List[Dict[str, Any]]]


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"name", "questions", "collection", "methods", "top_k"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Benchmark config missing: {', '.join(sorted(missing))}")

    name = str(config["name"])
    if name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError("name must be a single safe directory name")

    methods = config["methods"]
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a non-empty list")
    unknown = set(methods) - set(METHOD_ORDER)
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(sorted(unknown))}")
    if len(methods) != len(set(methods)):
        raise ValueError("methods must not contain duplicates")

    top_k = int(config["top_k"])
    if top_k < 10:
        raise ValueError("top_k must be at least 10 to calculate Hit@10 fairly")

    if any(method in methods for method in ("bm25", "hybrid", "hybrid_rerank")):
        if not config.get("chunks"):
            raise ValueError("chunks is required for BM25 and hybrid methods")

    for key in ("dense_top_k", "bm25_top_k", "rerank_top_k"):
        if int(config.get(key, top_k)) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(config.get("rerank_top_k", top_k)) < top_k:
        raise ValueError("rerank_top_k must be greater than or equal to top_k")
    if any(method in methods for method in ("hybrid", "hybrid_rerank")):
        if int(config.get("dense_top_k", top_k)) < top_k:
            raise ValueError("dense_top_k must be greater than or equal to top_k")
        if int(config.get("bm25_top_k", top_k)) < top_k:
            raise ValueError("bm25_top_k must be greater than or equal to top_k")
    if int(config.get("rrf_k", 60)) < 0:
        raise ValueError("rrf_k must be non-negative")


def ordered_methods(config: Mapping[str, Any]) -> List[str]:
    selected = set(config["methods"])
    return [method for method in METHOD_ORDER if method in selected]


def _timed_initialization(
    label: str,
    factory: Callable[[], Any],
    initialization_ms: Dict[str, float],
) -> Any:
    started = time.perf_counter()
    instance = factory()
    initialization_ms[label] = (time.perf_counter() - started) * 1000.0
    return instance


def build_pipelines(
    config: Mapping[str, Any],
    methods: Sequence[str],
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Initialize shared retrieval components once, before timed query execution."""
    top_k = int(config["top_k"])
    dense_top_k = int(config.get("dense_top_k", top_k))
    bm25_top_k = int(config.get("bm25_top_k", top_k))
    rerank_top_k = int(config.get("rerank_top_k", top_k))
    rrf_k = int(config.get("rrf_k", 60))
    persist_dir = str(_resolve_project_path(config.get("persist_dir", "data/chroma")))
    embedding_model = str(config.get("embedding_model", DEFAULT_EMBEDDING_MODEL))
    reranker_model = str(config.get("reranker_model", DEFAULT_RERANKER_MODEL))
    chunks_path = (
        str(_resolve_project_path(str(config["chunks"]))) if config.get("chunks") else None
    )

    initialization_ms: Dict[str, float] = {}
    needs_dense = any(method != "bm25" for method in methods)
    needs_bm25 = any(method in {"bm25", "hybrid", "hybrid_rerank"} for method in methods)
    needs_reranker = any(method in {"dense_rerank", "hybrid_rerank"} for method in methods)

    dense = None
    if needs_dense:
        dense = _timed_initialization(
            "dense_retriever",
            lambda: ChromaRetriever(
                persist_dir=persist_dir,
                collection_name=str(config["collection"]),
                model_name=embedding_model,
            ),
            initialization_ms,
        )

    bm25 = None
    if needs_bm25:
        assert chunks_path is not None
        bm25 = _timed_initialization(
            "bm25_retriever",
            lambda: BM25Retriever(chunks_path=chunks_path),
            initialization_ms,
        )

    hybrid = None
    if any(method in {"hybrid", "hybrid_rerank"} for method in methods):
        assert dense is not None and bm25 is not None
        hybrid = HybridRetriever(
            chunks_path=chunks_path or "",
            collection_name=str(config["collection"]),
            rrf_k=rrf_k,
            dense_retriever=dense,
            bm25_retriever=bm25,
        )

    reranker = None
    if needs_reranker:
        reranker = _timed_initialization(
            "cross_encoder_reranker",
            lambda: CrossEncoderReranker(model_name=reranker_model),
            initialization_ms,
        )

    pipelines: Dict[str, Pipeline] = {}
    method_parameters: Dict[str, Dict[str, Any]] = {}

    if "dense" in methods:
        assert dense is not None
        pipelines["dense"] = lambda query: dense.retrieve(query, top_k=top_k)
        method_parameters["dense"] = {"final_top_k": top_k}

    if "bm25" in methods:
        assert bm25 is not None
        pipelines["bm25"] = lambda query: bm25.retrieve(query, top_k=top_k)
        method_parameters["bm25"] = {"final_top_k": top_k}

    if "dense_rerank" in methods:
        assert dense is not None and reranker is not None

        def dense_rerank(query: str) -> List[Dict[str, Any]]:
            candidates = dense.retrieve(query, top_k=rerank_top_k)
            return reranker.rerank(query, candidates)[:top_k]

        pipelines["dense_rerank"] = dense_rerank
        method_parameters["dense_rerank"] = {
            "dense_candidate_top_k": rerank_top_k,
            "final_top_k": top_k,
        }

    if "hybrid" in methods:
        assert hybrid is not None
        pipelines["hybrid"] = lambda query: hybrid.retrieve(
            query=query,
            top_k=top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
        )
        method_parameters["hybrid"] = {
            "dense_candidate_top_k": dense_top_k,
            "bm25_candidate_top_k": bm25_top_k,
            "final_top_k": top_k,
            "rrf_k": rrf_k,
        }

    if "hybrid_rerank" in methods:
        assert hybrid is not None and reranker is not None

        def hybrid_rerank(query: str) -> List[Dict[str, Any]]:
            candidates = hybrid.retrieve(
                query=query,
                top_k=rerank_top_k,
                dense_top_k=dense_top_k,
                bm25_top_k=bm25_top_k,
            )
            return reranker.rerank(query, candidates)[:top_k]

        pipelines["hybrid_rerank"] = hybrid_rerank
        method_parameters["hybrid_rerank"] = {
            "dense_candidate_top_k": dense_top_k,
            "bm25_candidate_top_k": bm25_top_k,
            "fusion_candidate_top_k": rerank_top_k,
            "final_top_k": top_k,
            "rrf_k": rrf_k,
        }

    return pipelines, method_parameters, initialization_ms


def evaluate_method(
    questions: Sequence[Mapping[str, Any]],
    pipeline: Pipeline,
    top_k: int,
    bootstrap_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    per_query: List[Dict[str, Any]] = []
    ranks = []
    latencies_ms = []

    for question in questions:
        started = time.perf_counter()
        results = pipeline(str(question["query"]))[:top_k]
        latency_ms = (time.perf_counter() - started) * 1000.0
        answer_rank = find_answer_rank(results, question["answers"])
        normalized_results = [normalize_top_result(result) for result in results]
        per_query.append(
            build_query_record(question, answer_rank, latency_ms, normalized_results)
        )
        ranks.append(answer_rank)
        latencies_ms.append(latency_ms)

    return {
        "metrics": aggregate_ranks(ranks),
        "confidence_intervals": bootstrap_confidence_intervals(
            per_query,
            resamples=bootstrap_resamples,
            seed=seed,
        ),
        "latency": summarize_latency(latencies_ms),
        "per_query": per_query,
    }


def run_benchmark(
    config: Mapping[str, Any],
    bootstrap_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    methods = ordered_methods(config)
    question_path = _resolve_project_path(str(config["questions"]))
    if not question_path.is_file():
        raise FileNotFoundError(f"Question set not found: {question_path}")
    if config.get("chunks"):
        chunks_path = _resolve_project_path(str(config["chunks"]))
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Chunk set not found: {chunks_path}")

    questions = load_questions(question_path)
    pipelines, method_parameters, initialization_ms = build_pipelines(config, methods)

    warmup_ms: Dict[str, float] = {}
    warmup_query = str(questions[0]["query"])
    for method in methods:
        started = time.perf_counter()
        pipelines[method](warmup_query)
        warmup_ms[method] = (time.perf_counter() - started) * 1000.0

    method_results: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        print(f"Benchmarking {method} ({len(questions)} queries)...")
        result = evaluate_method(
            questions=questions,
            pipeline=pipelines[method],
            top_k=int(config["top_k"]),
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        )
        result["parameters"] = method_parameters[method]
        method_results[method] = result

    return {
        "run_metadata": {
            "name": config["name"],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_commit": config.get("dataset_commit"),
            "questions": str(config["questions"]),
            "chunks": config.get("chunks"),
            "collection": config["collection"],
            "methods": methods,
            "question_count": len(questions),
            "top_k": int(config["top_k"]),
            "bootstrap_resamples": bootstrap_resamples,
            "random_seed": seed,
            "embedding_model": config.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
            "reranker_model": config.get("reranker_model", DEFAULT_RERANKER_MODEL),
            "persist_dir": config.get("persist_dir", "data/chroma"),
            "python_version": platform.python_version(),
            "latency_clock": "time.perf_counter",
            "initialization_ms": initialization_ms,
            "warmup_queries_per_method": 1,
            "warmup_ms": warmup_ms,
            "method_parameters": method_parameters,
        },
        "methods": method_results,
    }


def _summary_payload(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "run_metadata": report["run_metadata"],
        "methods": {
            method: {
                "parameters": result["parameters"],
                "metrics": result["metrics"],
                "confidence_intervals": result["confidence_intervals"],
                "latency": result["latency"],
            }
            for method, result in report["methods"].items()
        },
    }


def build_summary_markdown(summary: Mapping[str, Any]) -> str:
    metadata = summary["run_metadata"]
    lines = [
        f"# Benchmark: {metadata['name']}",
        "",
        "## Run Metadata",
        "",
        f"- Dataset commit: `{metadata.get('dataset_commit')}`",
        f"- Questions: `{metadata['questions']}` ({metadata['question_count']} queries)",
        f"- Chunks: `{metadata.get('chunks')}`",
        f"- Collection: `{metadata['collection']}`",
        f"- Final top-k: {metadata['top_k']}",
        f"- Bootstrap: {metadata['bootstrap_resamples']:,} resamples, seed {metadata['random_seed']}",
        "- Initialization and warm-up timings are recorded separately in `summary.json`.",
        "",
        "## Method Comparison",
        "",
        "| Method | Hit@1 | Hit@1 95% CI | Hit@3 | Hit@5 | Hit@10 | MRR | MRR 95% CI | Mean ms | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for method, result in summary["methods"].items():
        metrics = result["metrics"]
        intervals = result["confidence_intervals"]
        latency = result["latency"]
        hit_ci = intervals["hit@1"]
        mrr_ci = intervals["mrr"]
        lines.append(
            f"| {method} | {metrics['hit@1']:.4f} | "
            f"{hit_ci['lower']:.4f}–{hit_ci['upper']:.4f} | "
            f"{metrics['hit@3']:.4f} | {metrics['hit@5']:.4f} | "
            f"{metrics['hit@10']:.4f} | {metrics['mrr']:.4f} | "
            f"{mrr_ci['lower']:.4f}–{mrr_ci['upper']:.4f} | "
            f"{latency['mean_ms']:.2f} | {latency['p50_ms']:.2f} | "
            f"{latency['p95_ms']:.2f} |"
        )

    lines.extend(["", "Candidate counts and RRF settings are recorded per method in `summary.json`.", ""])
    return "\n".join(lines)


def write_artifacts(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_payload(report)
    per_query = {
        "run_name": report["run_metadata"]["name"],
        "methods": {
            method: result["per_query"]
            for method, result in report["methods"].items()
        },
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "per_query.json").write_text(
        json.dumps(per_query, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        build_summary_markdown(summary),
        encoding="utf-8",
    )

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method", "total", "hit@1", "hit@1_ci_lower", "hit@1_ci_upper",
            "hit@3", "hit@5", "hit@10", "mrr", "mrr_ci_lower", "mrr_ci_upper",
            "mean_ms", "p50_ms", "p95_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, result in summary["methods"].items():
            metrics = result["metrics"]
            intervals = result["confidence_intervals"]
            latency = result["latency"]
            writer.writerow({
                "method": method,
                "total": metrics["total"],
                "hit@1": metrics["hit@1"],
                "hit@1_ci_lower": intervals["hit@1"]["lower"],
                "hit@1_ci_upper": intervals["hit@1"]["upper"],
                "hit@3": metrics["hit@3"],
                "hit@5": metrics["hit@5"],
                "hit@10": metrics["hit@10"],
                "mrr": metrics["mrr"],
                "mrr_ci_lower": intervals["mrr"]["lower"],
                "mrr_ci_upper": intervals["mrr"]["upper"],
                "mean_ms": latency["mean_ms"],
                "p50_ms": latency["p50_ms"],
                "p95_ms": latency["p95_ms"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Benchmark JSON config")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("eval/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path)
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")

    report = run_benchmark(
        config=config,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    output_dir = output_root / str(config["name"])
    write_artifacts(report, output_dir)
    print(build_summary_markdown(_summary_payload(report)))
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
