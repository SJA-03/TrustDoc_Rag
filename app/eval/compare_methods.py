"""Paired comparisons for retrieval per-query artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.metrics import percentile  # noqa: E402


DEFAULT_METHOD_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("dense", "dense_rerank"),
    ("hybrid", "hybrid_rerank"),
    ("dense", "hybrid"),
    ("dense_rerank", "hybrid_rerank"),
    ("dense", "bm25"),
    ("bm25", "hybrid"),
)


def _record_index(
    records: Sequence[Mapping[str, Any]],
    label: str,
) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        query_id = record.get("id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError(f"{label} record {position} has no valid query ID")
        if query_id in indexed:
            raise ValueError(f"{label} contains duplicate query ID: {query_id}")
        indexed[query_id] = record
    if not indexed:
        raise ValueError(f"{label} contains no query results")
    return indexed


def align_query_records(
    records_a: Sequence[Mapping[str, Any]],
    records_b: Sequence[Mapping[str, Any]],
    label_a: str = "A",
    label_b: str = "B",
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Align two result sets by ID and reject non-identical query sets."""
    indexed_a = _record_index(records_a, label_a)
    indexed_b = _record_index(records_b, label_b)
    ids_a = set(indexed_a)
    ids_b = set(indexed_b)
    if ids_a != ids_b:
        only_a = ", ".join(sorted(ids_a - ids_b)) or "none"
        only_b = ", ".join(sorted(ids_b - ids_a)) or "none"
        raise ValueError(
            f"Query ID mismatch between {label_a} and {label_b}; "
            f"only in {label_a}: {only_a}; only in {label_b}: {only_b}"
        )

    aligned = [(indexed_a[query_id], indexed_b[query_id]) for query_id in sorted(ids_a)]
    for record_a, record_b in aligned:
        if record_a.get("query") != record_b.get("query"):
            raise ValueError(
                f"Query text mismatch for ID {record_a['id']} between "
                f"{label_a} and {label_b}"
            )
    return aligned


def rank_values(record: Mapping[str, Any]) -> Tuple[float, float]:
    """Return Hit@1 and reciprocal rank, treating a missing rank as RR zero."""
    answer_rank = record.get("answer_rank")
    if answer_rank is None:
        return 0.0, 0.0
    if type(answer_rank) is not int or answer_rank <= 0:
        raise ValueError(
            f"Query {record.get('id', '<unknown>')} has invalid answer_rank: "
            f"{answer_rank}"
        )
    return float(answer_rank == 1), 1.0 / answer_rank


def paired_bootstrap(
    deltas: Sequence[float],
    resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Bootstrap a paired mean delta using shared sampled query indices."""
    if not deltas:
        raise ValueError("At least one paired delta is required")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    values = [float(delta) for delta in deltas]
    sample_size = len(values)
    rng = random.Random(seed)
    estimates: List[float] = []
    positive = 0
    zero = 0
    negative = 0

    for _ in range(resamples):
        estimate = statistics.fmean(
            values[rng.randrange(sample_size)] for _ in range(sample_size)
        )
        estimates.append(estimate)
        if math.isclose(estimate, 0.0, abs_tol=1e-15):
            zero += 1
        elif estimate > 0.0:
            positive += 1
        else:
            negative += 1

    tail = (1.0 - confidence_level) / 2.0
    lower = percentile(estimates, tail)
    upper = percentile(estimates, 1.0 - tail)
    return {
        "observed_delta": statistics.fmean(values),
        "confidence_interval": {
            "lower": lower,
            "upper": upper,
            "confidence_level": confidence_level,
            "resamples": resamples,
        },
        "ci_contains_zero": lower <= 0.0 <= upper,
        "bootstrap_fractions": {
            "positive": positive / resamples,
            "zero": zero / resamples,
            "negative": negative / resamples,
        },
    }


def _query_outcomes(
    query_ids: Sequence[str],
    deltas: Sequence[float],
) -> Dict[str, Any]:
    groups = {"wins": [], "ties": [], "losses": []}
    for query_id, delta in zip(query_ids, deltas):
        if delta > 0.0:
            groups["wins"].append(query_id)
        elif delta < 0.0:
            groups["losses"].append(query_id)
        else:
            groups["ties"].append(query_id)
    return {
        key: {"count": len(ids), "query_ids": ids}
        for key, ids in groups.items()
    }


def compare_query_results(
    records_a: Sequence[Mapping[str, Any]],
    records_b: Sequence[Mapping[str, Any]],
    resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
    label_a: str = "A",
    label_b: str = "B",
) -> Dict[str, Any]:
    """Compare two methods or chunking settings on an identical query set."""
    aligned = align_query_records(records_a, records_b, label_a, label_b)
    query_ids = [str(record_a["id"]) for record_a, _ in aligned]
    hit_deltas: List[float] = []
    rr_deltas: List[float] = []

    for record_a, record_b in aligned:
        hit_a, rr_a = rank_values(record_a)
        hit_b, rr_b = rank_values(record_b)
        hit_deltas.append(hit_b - hit_a)
        rr_deltas.append(rr_b - rr_a)

    hit_result = paired_bootstrap(
        hit_deltas,
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )
    rr_result = paired_bootstrap(
        rr_deltas,
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )
    hit_result["query_outcomes"] = _query_outcomes(query_ids, hit_deltas)
    rr_result["query_outcomes"] = _query_outcomes(query_ids, rr_deltas)
    return {
        "query_count": len(aligned),
        "hit@1": hit_result,
        "mrr": rr_result,
    }


def load_per_query_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    methods = payload.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError(f"{path} must contain a non-empty methods object")
    for method, records in methods.items():
        if not isinstance(records, list):
            raise ValueError(f"{path}: method {method} must contain a list")
        _record_index(records, f"{path}:{method}")
    return payload


def _candidate_case(
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    _, rr_a = rank_values(record_a)
    _, rr_b = rank_values(record_b)
    return {
        **context,
        "query_id": record_a["id"],
        "query": record_a.get("query", ""),
        "metadata": record_a.get("metadata", {}),
        "rank_a": record_a.get("answer_rank"),
        "rank_b": record_b.get("answer_rank"),
        "reciprocal_rank_a": rr_a,
        "reciprocal_rank_b": rr_b,
    }


def _append_method_candidates(
    candidates: Dict[str, List[Dict[str, Any]]],
    setting: str,
    method_a: str,
    method_b: str,
    aligned: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    category_pair = {
        ("dense", "dense_rerank"): (
            "dense_rerank_improved",
            "dense_rerank_worsened",
        ),
        ("hybrid", "hybrid_rerank"): (
            "hybrid_rerank_improved",
            "hybrid_rerank_worsened",
        ),
        ("dense", "bm25"): ("bm25_beats_dense", "dense_beats_bm25"),
        ("dense", "hybrid"): ("hybrid_beats_dense", "dense_beats_hybrid"),
    }.get((method_a, method_b))
    if category_pair is None:
        return

    positive_category, negative_category = category_pair
    context = {
        "setting": setting,
        "comparison": f"{method_a} -> {method_b}",
        "method_a": method_a,
        "method_b": method_b,
    }
    for record_a, record_b in aligned:
        _, rr_a = rank_values(record_a)
        _, rr_b = rank_values(record_b)
        if rr_b > rr_a:
            candidates[positive_category].append(
                _candidate_case(record_a, record_b, context)
            )
        elif rr_b < rr_a:
            candidates[negative_category].append(
                _candidate_case(record_a, record_b, context)
            )


def _append_chunking_candidates(
    candidates: Dict[str, List[Dict[str, Any]]],
    label: str,
    setting_a: str,
    setting_b: str,
    method: str,
    aligned: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    context = {
        "chunking_comparison": label,
        "comparison": f"{setting_a} -> {setting_b}",
        "setting_a": setting_a,
        "setting_b": setting_b,
        "method": method,
    }
    for record_a, record_b in aligned:
        _, rr_a = rank_values(record_a)
        _, rr_b = rank_values(record_b)
        if rr_a > rr_b:
            candidates["paragraph_beats_alternative"].append(
                _candidate_case(record_a, record_b, context)
            )
        elif rr_b > rr_a:
            candidates["alternative_beats_paragraph"].append(
                _candidate_case(record_a, record_b, context)
            )


def analyze_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    method_pairs: Sequence[Tuple[str, str]] = DEFAULT_METHOD_PAIRS,
    chunking_pairs: Sequence[Tuple[str, str, str]] = (),
    resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Analyze within-artifact methods and cross-artifact chunking settings."""
    candidates: Dict[str, List[Dict[str, Any]]] = {
        category: []
        for category in (
            "dense_rerank_improved",
            "dense_rerank_worsened",
            "hybrid_rerank_improved",
            "hybrid_rerank_worsened",
            "bm25_beats_dense",
            "dense_beats_bm25",
            "hybrid_beats_dense",
            "dense_beats_hybrid",
            "paragraph_beats_alternative",
            "alternative_beats_paragraph",
        )
    }
    method_comparisons: List[Dict[str, Any]] = []
    chunking_comparisons: List[Dict[str, Any]] = []

    for setting, artifact in artifacts.items():
        methods = artifact["methods"]
        for method_a, method_b in method_pairs:
            if method_a not in methods or method_b not in methods:
                raise ValueError(
                    f"{setting} does not contain method pair {method_a}, {method_b}"
                )
            result = compare_query_results(
                methods[method_a],
                methods[method_b],
                resamples=resamples,
                seed=seed,
                confidence_level=confidence_level,
                label_a=f"{setting}:{method_a}",
                label_b=f"{setting}:{method_b}",
            )
            method_comparisons.append(
                {
                    "setting": setting,
                    "method_a": method_a,
                    "method_b": method_b,
                    "comparison": f"{method_a} -> {method_b}",
                    **result,
                }
            )
            aligned = align_query_records(
                methods[method_a],
                methods[method_b],
                f"{setting}:{method_a}",
                f"{setting}:{method_b}",
            )
            _append_method_candidates(
                candidates, setting, method_a, method_b, aligned
            )

    for label, setting_a, setting_b in chunking_pairs:
        if setting_a not in artifacts or setting_b not in artifacts:
            raise ValueError(
                f"Chunking comparison {label} references an unknown setting"
            )
        methods_a = artifacts[setting_a]["methods"]
        methods_b = artifacts[setting_b]["methods"]
        if set(methods_a) != set(methods_b):
            raise ValueError(
                f"Chunking comparison {label} has different method sets"
            )
        for method in methods_a:
            result = compare_query_results(
                methods_a[method],
                methods_b[method],
                resamples=resamples,
                seed=seed,
                confidence_level=confidence_level,
                label_a=f"{setting_a}:{method}",
                label_b=f"{setting_b}:{method}",
            )
            chunking_comparisons.append(
                {
                    "label": label,
                    "setting_a": setting_a,
                    "setting_b": setting_b,
                    "method": method,
                    "comparison": f"{setting_a} -> {setting_b}",
                    **result,
                }
            )
            aligned = align_query_records(
                methods_a[method],
                methods_b[method],
                f"{setting_a}:{method}",
                f"{setting_b}:{method}",
            )
            _append_chunking_candidates(
                candidates, label, setting_a, setting_b, method, aligned
            )

    metadata = {
        "resamples": resamples,
        "seed": seed,
        "confidence_level": confidence_level,
        "method_pairs": [f"{a} -> {b}" for a, b in method_pairs],
        "chunking_pairs": [
            {"label": label, "setting_a": a, "setting_b": b}
            for label, a, b in chunking_pairs
        ],
        "probability_note": (
            "bootstrap_fractions are fractions of paired bootstrap resamples with "
            "positive, zero, or negative mean delta; they are not posterior "
            "probabilities or formal p-values"
        ),
        "candidate_basis": "strict per-query reciprocal-rank comparison",
    }
    comparisons = {
        "analysis_metadata": metadata,
        "method_comparisons": method_comparisons,
        "chunking_comparisons": chunking_comparisons,
    }
    outcomes = {
        "analysis_metadata": metadata,
        "method_outcomes": [
            {
                "setting": item["setting"],
                "comparison": item["comparison"],
                "hit@1": item["hit@1"]["query_outcomes"],
                "mrr": item["mrr"]["query_outcomes"],
            }
            for item in method_comparisons
        ],
        "chunking_outcomes": [
            {
                "label": item["label"],
                "comparison": item["comparison"],
                "method": item["method"],
                "hit@1": item["hit@1"]["query_outcomes"],
                "mrr": item["mrr"]["query_outcomes"],
            }
            for item in chunking_comparisons
        ],
        "error_analysis_candidates": candidates,
        "candidate_counts": {
            category: len(cases) for category, cases in candidates.items()
        },
    }
    return comparisons, outcomes


def _format_interval(metric: Mapping[str, Any]) -> str:
    interval = metric["confidence_interval"]
    return f"{interval['lower']:.4f}–{interval['upper']:.4f}"


def _format_wtl(metric: Mapping[str, Any]) -> str:
    outcomes = metric["query_outcomes"]
    return (
        f"{outcomes['wins']['count']}/"
        f"{outcomes['ties']['count']}/"
        f"{outcomes['losses']['count']}"
    )


def build_markdown_report(comparisons: Mapping[str, Any]) -> str:
    metadata = comparisons["analysis_metadata"]
    interval_label = f"{metadata['confidence_level']:.1%} CI"
    lines = [
        "# Paired Retrieval Comparisons",
        "",
        f"- Bootstrap: {metadata['resamples']:,} resamples, seed {metadata['seed']}",
        f"- Confidence level: {metadata['confidence_level']:.2f}",
        "- Delta direction: B minus A",
        "- W/T/L direction: method or setting B relative to A",
        "- Bootstrap fractions are descriptive resample fractions, not posterior probabilities or formal p-values.",
        "",
        "## Method Comparisons",
        "",
        f"| Setting | Comparison | Δ Hit@1 | {interval_label} | Δ MRR | {interval_label} | RR W/T/L |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons["method_comparisons"]:
        lines.append(
            f"| {item['setting']} | {item['comparison']} | "
            f"{item['hit@1']['observed_delta']:+.4f} | "
            f"{_format_interval(item['hit@1'])} | "
            f"{item['mrr']['observed_delta']:+.4f} | "
            f"{_format_interval(item['mrr'])} | "
            f"{_format_wtl(item['mrr'])} |"
        )

    lines.extend(
        [
            "",
            "## Chunking Comparisons",
            "",
            f"| Label | Comparison | Method | Δ Hit@1 | {interval_label} | Δ MRR | {interval_label} | RR W/T/L |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in comparisons["chunking_comparisons"]:
        lines.append(
            f"| {item['label']} | {item['comparison']} | {item['method']} | "
            f"{item['hit@1']['observed_delta']:+.4f} | "
            f"{_format_interval(item['hit@1'])} | "
            f"{item['mrr']['observed_delta']:+.4f} | "
            f"{_format_interval(item['mrr'])} | "
            f"{_format_wtl(item['mrr'])} |"
        )

    lines.extend(
        [
            "",
            "## Bootstrap Resample Fractions",
            "",
            "| Setting | Comparison | Metric | Positive | Zero | Negative | CI contains zero |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in comparisons["method_comparisons"]:
        for metric_name in ("hit@1", "mrr"):
            metric = item[metric_name]
            fractions = metric["bootstrap_fractions"]
            lines.append(
                f"| {item['setting']} | {item['comparison']} | {metric_name} | "
                f"{fractions['positive']:.4f} | {fractions['zero']:.4f} | "
                f"{fractions['negative']:.4f} | "
                f"{str(metric['ci_contains_zero']).lower()} |"
            )
    return "\n".join(lines) + "\n"


def write_analysis(
    comparisons: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_comparisons.json").write_text(
        json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "query_pair_outcomes.json").write_text(
        json.dumps(outcomes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "paired_comparisons.md").write_text(
        build_markdown_report(comparisons),
        encoding="utf-8",
    )


def _parse_named_path(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Artifact must use NAME=PATH: {spec}")
    name, raw_path = spec.split("=", 1)
    if not name or not raw_path:
        raise ValueError(f"Artifact must use NAME=PATH: {spec}")
    path = Path(raw_path)
    return name, path if path.is_absolute() else PROJECT_ROOT / path


def _parse_method_pair(spec: str) -> Tuple[str, str]:
    parts = spec.split(":")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Method pair must use METHOD_A:METHOD_B: {spec}")
    return parts[0], parts[1]


def _parse_chunking_pair(spec: str) -> Tuple[str, str, str]:
    if "=" not in spec:
        raise ValueError(
            f"Chunking pair must use LABEL=SETTING_A:SETTING_B: {spec}"
        )
    label, settings = spec.split("=", 1)
    parts = settings.split(":")
    if not label or len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Chunking pair must use LABEL=SETTING_A:SETTING_B: {spec}"
        )
    return label, parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Named per-query artifact as NAME=PATH; repeat for each setting",
    )
    parser.add_argument(
        "--method-pair",
        action="append",
        help="Method comparison as METHOD_A:METHOD_B; defaults to official pairs",
    )
    parser.add_argument(
        "--chunking-pair",
        action="append",
        default=[],
        help="Cross-artifact comparison as LABEL=SETTING_A:SETTING_B",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts: Dict[str, Dict[str, Any]] = {}
    artifact_paths: Dict[str, str] = {}
    for spec in args.artifact:
        name, path = _parse_named_path(spec)
        if name in artifacts:
            raise ValueError(f"Duplicate artifact name: {name}")
        artifacts[name] = load_per_query_artifact(path)
        try:
            artifact_paths[name] = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            artifact_paths[name] = str(path)

    method_pairs = (
        [_parse_method_pair(spec) for spec in args.method_pair]
        if args.method_pair
        else list(DEFAULT_METHOD_PAIRS)
    )
    chunking_pairs = [
        _parse_chunking_pair(spec) for spec in args.chunking_pair
    ]
    comparisons, outcomes = analyze_artifacts(
        artifacts,
        method_pairs=method_pairs,
        chunking_pairs=chunking_pairs,
        resamples=args.resamples,
        seed=args.seed,
        confidence_level=args.confidence,
    )
    comparisons["analysis_metadata"]["artifacts"] = artifact_paths
    outcomes["analysis_metadata"]["artifacts"] = artifact_paths
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    write_analysis(comparisons, outcomes, output_dir)
    print(build_markdown_report(comparisons))
    print(f"Analysis artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
