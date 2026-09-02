"""Validate manual retrieval-error annotations and build descriptive reports."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PIPELINE_MECHANISMS = (
    "candidate_miss",
    "lexical_rescue",
    "lexical_distraction",
    "fusion_rescue",
    "fusion_degradation",
    "reranker_rescue",
    "reranker_inversion",
    "chunking_improvement",
    "chunking_degradation",
    "mixed_or_unclear",
)

CONTENT_FACTORS = (
    "exact_terminology",
    "semantic_neighbor",
    "neighbor_page",
    "multi_page_evidence",
    "section_boundary",
    "cross_document_confusion",
    "repeated_terminology",
    "chunk_granularity",
    "table_or_layout",
    "other_or_unclear",
)

DOMAINS = ("operating_systems", "ai_papers")
QUERY_TYPES = (
    "definition",
    "comparison",
    "mechanism",
    "reason",
    "exact_terminology",
    "similar_concept",
    "multi_page",
)

CASE_REQUIRED_FIELDS = (
    "case_id",
    "query_id",
    "domain",
    "setting",
    "comparison",
    "method_a",
    "method_b",
    "query",
    "metadata",
    "expected_answers",
    "rank_a",
    "rank_b",
    "reciprocal_rank_a",
    "reciprocal_rank_b",
    "pipeline_mechanism",
    "primary_factor",
    "secondary_factor",
    "observation",
    "interpretation",
    "ground_truth_verified",
    "retrieved_candidates_reviewed",
)


def load_annotation_document(path: Path) -> Dict[str, Any]:
    """Load a manual annotation document from JSON."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("annotation document must be a JSON object")
    return payload


def _non_empty_string(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")


def _validate_rank_and_rr(
    rank: Any,
    reciprocal_rank: Any,
    location: str,
) -> None:
    if rank is not None and (type(rank) is not int or rank <= 0):
        raise ValueError(f"{location}.rank must be null or a positive integer")
    if type(reciprocal_rank) not in (int, float):
        raise ValueError(f"{location}.reciprocal_rank must be numeric")
    expected = 0.0 if rank is None else 1.0 / rank
    if not math.isclose(float(reciprocal_rank), expected, abs_tol=1e-12):
        raise ValueError(
            f"{location} has inconsistent rank and reciprocal rank: "
            f"{rank}, {reciprocal_rank}"
        )


def _validate_evidence_references(
    entries: Any,
    field_name: str,
    known_case_ids: set[str],
) -> None:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{field_name} must be a non-empty list")
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"{field_name}[{index}] must be an object")
        _non_empty_string(entry.get("id"), f"{field_name}[{index}].id")
        text_key = "answer" if field_name == "research_questions" else "text"
        if field_name == "research_questions":
            _non_empty_string(
                entry.get("question"), f"{field_name}[{index}].question"
            )
        _non_empty_string(entry.get(text_key), f"{field_name}[{index}].{text_key}")
        evidence = entry.get("evidence_case_ids")
        if not isinstance(evidence, list):
            raise ValueError(
                f"{field_name}[{index}].evidence_case_ids must be a list"
            )
        unknown = sorted(set(evidence) - known_case_ids)
        if unknown:
            raise ValueError(
                f"{field_name}[{index}] references unknown case IDs: "
                f"{', '.join(unknown)}"
            )


def validate_annotations(document: Mapping[str, Any]) -> None:
    """Validate schema, taxonomy vocabulary, and internal references."""
    metadata = document.get("analysis_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("analysis_metadata must be an object")
    for field in (
        "scope",
        "dataset",
        "selection_policy",
        "ground_truth_commit",
        "analysis_commit_base",
    ):
        _non_empty_string(metadata.get(field), f"analysis_metadata.{field}")
    if metadata.get("descriptive_analysis") is not True:
        raise ValueError("analysis_metadata.descriptive_analysis must be true")

    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list")

    case_ids: set[str] = set()
    for position, case in enumerate(cases, start=1):
        location = f"cases[{position}]"
        if not isinstance(case, dict):
            raise ValueError(f"{location} must be an object")
        missing = [field for field in CASE_REQUIRED_FIELDS if field not in case]
        if missing:
            raise ValueError(f"{location} is missing fields: {', '.join(missing)}")

        for field in (
            "case_id",
            "query_id",
            "setting",
            "comparison",
            "method_a",
            "method_b",
            "query",
            "observation",
            "interpretation",
        ):
            _non_empty_string(case[field], f"{location}.{field}")

        case_id = case["case_id"]
        if case_id in case_ids:
            raise ValueError(f"duplicate case ID: {case_id}")
        case_ids.add(case_id)

        if case["domain"] not in DOMAINS:
            raise ValueError(f"{location}.domain has invalid value: {case['domain']}")
        mechanism = case["pipeline_mechanism"]
        if mechanism not in PIPELINE_MECHANISMS:
            raise ValueError(
                f"{location}.pipeline_mechanism has invalid taxonomy label: "
                f"{mechanism}"
            )
        primary = case["primary_factor"]
        if primary not in CONTENT_FACTORS:
            raise ValueError(
                f"{location}.primary_factor has invalid taxonomy label: {primary}"
            )
        secondary = case["secondary_factor"]
        if secondary is not None and secondary not in CONTENT_FACTORS:
            raise ValueError(
                f"{location}.secondary_factor has invalid taxonomy label: "
                f"{secondary}"
            )
        if secondary == primary:
            raise ValueError(f"{location} repeats the primary factor as secondary")

        case_metadata = case["metadata"]
        if not isinstance(case_metadata, dict):
            raise ValueError(f"{location}.metadata must be an object")
        query_type = case_metadata.get("query_type")
        if query_type not in QUERY_TYPES:
            raise ValueError(
                f"{location}.metadata.query_type has invalid value: {query_type}"
            )

        answers = case["expected_answers"]
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"{location}.expected_answers must be a non-empty list")
        for answer_index, answer in enumerate(answers, start=1):
            answer_location = f"{location}.expected_answers[{answer_index}]"
            if not isinstance(answer, dict):
                raise ValueError(f"{answer_location} must be an object")
            _non_empty_string(answer.get("file"), f"{answer_location}.file")
            page = answer.get("page")
            if type(page) is not int or page <= 0:
                raise ValueError(f"{answer_location}.page must be a positive integer")

        _validate_rank_and_rr(
            case["rank_a"], case["reciprocal_rank_a"], f"{location}.method_a"
        )
        _validate_rank_and_rr(
            case["rank_b"], case["reciprocal_rank_b"], f"{location}.method_b"
        )
        if case["ground_truth_verified"] is not True:
            raise ValueError(f"{location}.ground_truth_verified must be true")
        if case["retrieved_candidates_reviewed"] is not True:
            raise ValueError(f"{location}.retrieved_candidates_reviewed must be true")

    for field in ("research_questions", "key_findings", "unexpected_results"):
        _validate_evidence_references(document.get(field), field, case_ids)

    limitations = document.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise ValueError("limitations must be a non-empty list")
    for index, limitation in enumerate(limitations, start=1):
        _non_empty_string(limitation, f"limitations[{index}]")


def _ordered_counts(values: Iterable[str], order: Sequence[str]) -> Dict[str, int]:
    counts = Counter(values)
    return {label: counts.get(label, 0) for label in order}


def _cross_tab(
    cases: Sequence[Mapping[str, Any]],
    row_field: str,
    row_order: Sequence[str],
    column_getter: Any,
    column_order: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        table[str(case[row_field])][str(column_getter(case))] += 1
    return {
        row: {column: table[row].get(column, 0) for column in column_order}
        for row in row_order
    }


def build_summary(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Build deterministic descriptive counts and cross-tabs."""
    validate_annotations(document)
    cases = document["cases"]
    by_domain = _ordered_counts((case["domain"] for case in cases), DOMAINS)
    by_mechanism = _ordered_counts(
        (case["pipeline_mechanism"] for case in cases), PIPELINE_MECHANISMS
    )
    by_primary_factor = _ordered_counts(
        (case["primary_factor"] for case in cases), CONTENT_FACTORS
    )
    by_query_type = _ordered_counts(
        (case["metadata"]["query_type"] for case in cases), QUERY_TYPES
    )

    summary = {
        "analysis_scope": "manually_audited_heldout_cases",
        "descriptive_analysis": True,
        "content_factor_distribution_basis": "primary_factor_only",
        "total_cases": len(cases),
        "by_domain": by_domain,
        "by_pipeline_mechanism": by_mechanism,
        "by_primary_factor": by_primary_factor,
        "by_query_type": by_query_type,
        "mechanism_by_domain": _cross_tab(
            cases,
            "pipeline_mechanism",
            PIPELINE_MECHANISMS,
            lambda case: case["domain"],
            DOMAINS,
        ),
        "mechanism_by_query_type": _cross_tab(
            cases,
            "pipeline_mechanism",
            PIPELINE_MECHANISMS,
            lambda case: case["metadata"]["query_type"],
            QUERY_TYPES,
        ),
        "quality_audit": {
            "ground_truth_verified_cases": sum(
                case["ground_truth_verified"] is True for case in cases
            ),
            "retrieved_candidates_reviewed_cases": sum(
                case["retrieved_candidates_reviewed"] is True for case in cases
            ),
            "unique_case_ids": len({case["case_id"] for case in cases}),
            "observation_interpretation_separated_cases": sum(
                bool(case["observation"].strip())
                and bool(case["interpretation"].strip())
                for case in cases
            ),
        },
    }
    _validate_summary_consistency(summary)
    return summary


def _validate_summary_consistency(summary: Mapping[str, Any]) -> None:
    total = summary["total_cases"]
    count_sections = (
        "by_domain",
        "by_pipeline_mechanism",
        "by_primary_factor",
        "by_query_type",
    )
    for section in count_sections:
        section_total = sum(summary[section].values())
        if section_total != total:
            raise ValueError(
                f"summary count mismatch for {section}: {section_total} != {total}"
            )
    for section in ("mechanism_by_domain", "mechanism_by_query_type"):
        section_total = sum(sum(row.values()) for row in summary[section].values())
        if section_total != total:
            raise ValueError(
                f"summary count mismatch for {section}: {section_total} != {total}"
            )


def _markdown_count_table(counts: Mapping[str, int], heading: str) -> List[str]:
    lines = [f"## {heading}", "", "| Label | Cases |", "|---|---:|"]
    lines.extend(f"| `{label}` | {count} |" for label, count in counts.items())
    return lines + [""]


def build_markdown(document: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    """Render the validated annotations and summary as a human-readable report."""
    validate_annotations(document)
    _validate_summary_consistency(summary)
    metadata = document["analysis_metadata"]
    lines = [
        "# Held-out Retrieval Error Analysis",
        "",
        "This report describes a purposive sample of manually audited held-out "
        "comparison cases. It is descriptive, not an estimate of population-level "
        "error prevalence.",
        "",
        "## Scope and audit method",
        "",
        f"- Scope: {metadata['scope']}",
        f"- Dataset: {metadata['dataset']}",
        f"- Selection: {metadata['selection_policy']}",
        f"- Cases: {summary['total_cases']} "
        f"(OS {summary['by_domain']['operating_systems']}, "
        f"AI Papers {summary['by_domain']['ai_papers']})",
        "- Every case was checked against both retrieved candidates and the "
        "original ground-truth PDF page(s).",
        "- Content-factor counts use the primary factor only; secondary factors "
        "remain available in the case annotations.",
        "",
    ]
    lines.extend(
        _markdown_count_table(
            summary["by_pipeline_mechanism"], "Pipeline mechanism distribution"
        )
    )
    lines.extend(
        _markdown_count_table(
            summary["by_primary_factor"], "Content factor distribution"
        )
    )

    lines.extend(
        [
            "## Mechanism × domain",
            "",
            "| Mechanism | OS | AI Papers |",
            "|---|---:|---:|",
        ]
    )
    for mechanism, counts in summary["mechanism_by_domain"].items():
        lines.append(
            f"| `{mechanism}` | {counts['operating_systems']} | "
            f"{counts['ai_papers']} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Mechanism × query type",
            "",
            "| Mechanism | Definition | Comparison | Mechanism | Reason | "
            "Exact Term | Similar Concept | Multi-page |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mechanism, counts in summary["mechanism_by_query_type"].items():
        values = " | ".join(str(counts[query_type]) for query_type in QUERY_TYPES)
        lines.append(f"| `{mechanism}` | {values} |")
    lines.append("")

    lines.extend(["## Research questions", ""])
    for item in document["research_questions"]:
        evidence = ", ".join(f"`{case_id}`" for case_id in item["evidence_case_ids"])
        lines.extend(
            [
                f"### {item['id']}: {item['question']}",
                "",
                item["answer"],
                "",
                f"Evidence cases: {evidence or 'none'}",
                "",
            ]
        )

    for field, heading in (
        ("key_findings", "Key findings"),
        ("unexpected_results", "Unexpected results"),
    ):
        lines.extend([f"## {heading}", ""])
        for item in document[field]:
            evidence = ", ".join(
                f"`{case_id}`" for case_id in item["evidence_case_ids"]
            )
            lines.append(f"- {item['text']} Evidence: {evidence or 'none'}. ")
        lines.append("")

    lines.extend(["## Manually audited cases", ""])
    for case in document["cases"]:
        rank_a = "not in top-k" if case["rank_a"] is None else str(case["rank_a"])
        rank_b = "not in top-k" if case["rank_b"] is None else str(case["rank_b"])
        secondary = case["secondary_factor"] or "none"
        evidence = ", ".join(
            f"{answer['file']} p.{answer['page']}"
            for answer in case["expected_answers"]
        )
        lines.extend(
            [
                f"### {case['case_id']} — {case['query_id']}",
                "",
                f"- Setting/comparison: `{case['setting']}`; "
                f"`{case['method_a']}` → `{case['method_b']}` "
                f"(rank {rank_a} → {rank_b})",
                f"- Query type: `{case['metadata']['query_type']}`",
                f"- Taxonomy: `{case['pipeline_mechanism']}` / "
                f"`{case['primary_factor']}`; secondary `{secondary}`",
                f"- Ground truth: {evidence}",
                f"- Observation: {case['observation']}",
                f"- Interpretation: {case['interpretation']}",
                "",
            ]
        )

    lines.extend(["## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in document["limitations"])
    lines.extend(
        [
            "",
            "## Annotation quality audit",
            "",
            f"- Ground-truth source/page verified: "
            f"{summary['quality_audit']['ground_truth_verified_cases']}/"
            f"{summary['total_cases']}",
            f"- Retrieved candidates reviewed: "
            f"{summary['quality_audit']['retrieved_candidates_reviewed_cases']}/"
            f"{summary['total_cases']}",
            f"- Unique case IDs: {summary['quality_audit']['unique_case_ids']}/"
            f"{summary['total_cases']}",
            f"- Observation and interpretation both present: "
            f"{summary['quality_audit']['observation_interpretation_separated_cases']}/"
            f"{summary['total_cases']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_analysis_outputs(annotation_path: Path, output_dir: Path) -> None:
    """Validate annotations and write deterministic JSON/Markdown summaries."""
    document = load_annotation_document(annotation_path)
    summary = build_summary(document)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "error_analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (output_dir / "error_analysis.md").write_text(
        build_markdown(document, summary), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_analysis_outputs(args.annotations, args.output_dir)


if __name__ == "__main__":
    main()
