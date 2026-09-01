"""Dependency-free metrics and statistics for retrieval evaluation."""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


HIT_CUTOFFS: Tuple[int, ...] = (1, 3, 5, 10)


def load_questions(path: Path | str) -> List[Dict[str, Any]]:
    """Load and minimally validate the existing JSONL question format."""
    questions: List[Dict[str, Any]] = []
    question_path = Path(path)

    with question_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            question = json.loads(line)
            missing = {"id", "query", "answers"} - question.keys()
            if missing:
                fields = ", ".join(sorted(missing))
                raise ValueError(
                    f"{question_path}:{line_number} is missing required field(s): {fields}"
                )
            if not isinstance(question["answers"], list) or not question["answers"]:
                raise ValueError(
                    f"{question_path}:{line_number} must contain at least one answer"
                )
            for answer in question["answers"]:
                if not isinstance(answer, dict) or not {"file", "page"} <= answer.keys():
                    raise ValueError(
                        f"{question_path}:{line_number} contains an invalid answer"
                    )
            if "metadata" in question and not isinstance(question["metadata"], dict):
                raise ValueError(
                    f"{question_path}:{line_number} metadata must be an object"
                )
            questions.append(question)

    if not questions:
        raise ValueError(f"No questions found in {question_path}")
    return questions


def _same_page(left: Any, right: Any) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def find_answer_rank(
    results: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
) -> Optional[int]:
    """Return the first 1-based rank matching source_file and page_number."""
    for index, chunk in enumerate(results):
        metadata = chunk.get("metadata", {})
        for answer in answers:
            if (
                metadata.get("source_file") == answer.get("file")
                and _same_page(metadata.get("page_number"), answer.get("page"))
            ):
                return index + 1
    return None


def reciprocal_rank(answer_rank: Optional[int]) -> float:
    return 0.0 if answer_rank is None else 1.0 / answer_rank


def rank_metrics(answer_rank: Optional[int]) -> Dict[str, float | bool]:
    metrics: Dict[str, float | bool] = {
        f"hit@{cutoff}": answer_rank is not None and answer_rank <= cutoff
        for cutoff in HIT_CUTOFFS
    }
    metrics["reciprocal_rank"] = reciprocal_rank(answer_rank)
    return metrics


def aggregate_ranks(answer_ranks: Sequence[Optional[int]]) -> Dict[str, float | int]:
    if not answer_ranks:
        raise ValueError("At least one answer rank is required")

    total = len(answer_ranks)
    metrics: Dict[str, float | int] = {"total": total}
    for cutoff in HIT_CUTOFFS:
        metrics[f"hit@{cutoff}"] = sum(
            rank is not None and rank <= cutoff for rank in answer_ranks
        ) / total
    metrics["mrr"] = statistics.fmean(reciprocal_rank(rank) for rank in answer_ranks)
    return metrics


def percentile(values: Sequence[float], probability: float) -> float:
    """Calculate a linearly interpolated percentile on a 0..1 scale."""
    if not values:
        raise ValueError("At least one value is required")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def bootstrap_confidence_intervals(
    per_query: Sequence[Mapping[str, Any]],
    resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float | int]]:
    """Calculate paired, query-level bootstrap intervals for Hit@1 and MRR."""
    if not per_query:
        raise ValueError("At least one query result is required")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    hit_values = [float(item["hit@1"]) for item in per_query]
    rr_values = [float(item["reciprocal_rank"]) for item in per_query]
    sample_size = len(per_query)
    rng = random.Random(seed)
    hit_estimates: List[float] = []
    mrr_estimates: List[float] = []

    for _ in range(resamples):
        indices = [rng.randrange(sample_size) for _ in range(sample_size)]
        hit_estimates.append(statistics.fmean(hit_values[index] for index in indices))
        mrr_estimates.append(statistics.fmean(rr_values[index] for index in indices))

    tail = (1.0 - confidence_level) / 2.0
    return {
        "hit@1": {
            "lower": percentile(hit_estimates, tail),
            "upper": percentile(hit_estimates, 1.0 - tail),
            "confidence_level": confidence_level,
            "resamples": resamples,
        },
        "mrr": {
            "lower": percentile(mrr_estimates, tail),
            "upper": percentile(mrr_estimates, 1.0 - tail),
            "confidence_level": confidence_level,
            "resamples": resamples,
        },
    }


def summarize_latency(latencies_ms: Sequence[float]) -> Dict[str, float]:
    if not latencies_ms:
        raise ValueError("At least one latency value is required")
    values = [float(value) for value in latencies_ms]
    median_ms = statistics.median(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": median_ms,
        "p50_ms": median_ms,
        "p95_ms": percentile(values, 0.95),
    }


def build_query_record(
    question: Mapping[str, Any],
    answer_rank: Optional[int],
    latency_ms: float,
    top_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build a serializable per-query record while preserving optional metadata."""
    record: Dict[str, Any] = {
        "id": question["id"],
        "query": question["query"],
        "answers": question["answers"],
        "answer_rank": answer_rank,
        **rank_metrics(answer_rank),
        "latency_ms": latency_ms,
        "top_results": list(top_results),
    }
    if "metadata" in question:
        record["metadata"] = question["metadata"]
    return record
