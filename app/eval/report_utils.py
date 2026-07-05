import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


TOP_RESULT_FIELDS = [
    "rank",
    "source_file",
    "page_number",
    "chunk_id",
    "section_title",
    "original_rank",
    "dense_rank",
    "bm25_rank",
    "hybrid_score",
    "rerank_score",
    "distance",
    "bm25_score",
    "text_preview",
]


def ensure_output_dir(output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)


def extract_weak_cases(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        item
        for item in details
        if item.get("answer_rank") is None or item.get("answer_rank", 0) > 1
    ]


def save_evaluation_json(
    output_path: str,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    details: List[Dict[str, Any]],
) -> None:
    ensure_output_dir(output_path)
    report = build_json_report(metadata, metrics, details)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def save_markdown_report(
    output_path: str,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    details: List[Dict[str, Any]],
    title: str = "Hybrid + Rerank Evaluation Report",
) -> None:
    ensure_output_dir(output_path)
    markdown = build_markdown_report(
        title=title,
        metadata=metadata,
        metrics=metrics,
        details=details,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)


def build_json_report(
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    details: List[Dict[str, Any]],
) -> Dict[str, Any]:
    run_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
    }
    normalized_details = [normalize_detail(item) for item in details]
    weak_cases = extract_weak_cases(normalized_details)
    summary = normalize_metrics(metrics)

    return {
        "run_metadata": run_metadata,
        "summary": summary,
        "detailed_results": normalized_details,
        "weak_cases": weak_cases,
    }


def normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "total_questions": metrics["total"],
        "hit_1": metrics["hit@1"],
        "hit_3": metrics["hit@3"],
        "hit_5": metrics["hit@5"],
        "hit_10": metrics["hit@10"],
        "mrr": metrics["mrr"],
    }


def normalize_detail(item: Dict[str, Any]) -> Dict[str, Any]:
    answer_rank = item.get("answer_rank")

    return {
        "id": item["id"],
        "query": item["query"],
        "answers": item["answers"],
        "answer_rank": answer_rank,
        "hit_1": is_hit(answer_rank, 1),
        "hit_3": is_hit(answer_rank, 3),
        "hit_5": is_hit(answer_rank, 5),
        "hit_10": is_hit(answer_rank, 10),
        "top_results": [
            normalize_top_result(chunk)
            for chunk in item.get("results", [])[:10]
        ],
    }


def normalize_top_result(chunk: Dict[str, Any]) -> Dict[str, Any]:
    metadata = chunk.get("metadata", {})
    normalized = {
        "rank": chunk.get("rank"),
        "source_file": metadata.get("source_file"),
        "page_number": metadata.get("page_number"),
        "chunk_id": metadata.get("chunk_id"),
        "section_title": metadata.get("section_title") or chunk.get("section_title"),
        "original_rank": chunk.get("original_rank"),
        "dense_rank": chunk.get("dense_rank"),
        "bm25_rank": chunk.get("bm25_rank"),
        "hybrid_score": chunk.get("hybrid_score"),
        "rerank_score": chunk.get("rerank_score"),
        "distance": chunk.get("distance"),
        "bm25_score": chunk.get("bm25_score"),
        "text_preview": shorten_text(chunk.get("text", "")),
    }

    return {
        key: normalized.get(key)
        for key in TOP_RESULT_FIELDS
        if normalized.get(key) is not None
    }


def is_hit(answer_rank: Optional[int], top_k: int) -> bool:
    return answer_rank is not None and answer_rank <= top_k


def shorten_text(text: str, max_length: int = 300) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def build_markdown_report(
    title: str,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    details: List[Dict[str, Any]],
) -> str:
    normalized_metrics = normalize_metrics(metrics)
    normalized_details = [normalize_detail(item) for item in details]
    weak_cases = extract_weak_cases(normalized_details)

    lines = [
        f"# {title}",
        "",
        "## Run Metadata",
        "",
        "| Field | Value |",
        "|---|---|",
    ]

    for key, value in metadata.items():
        lines.append(f"| `{escape_markdown(str(key))}` | {escape_markdown(str(value))} |")

    lines.extend(
        [
            "",
            "## Summary Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Total questions | {normalized_metrics['total_questions']} |",
            f"| Hit@1 | {normalized_metrics['hit_1']:.4f} |",
            f"| Hit@3 | {normalized_metrics['hit_3']:.4f} |",
            f"| Hit@5 | {normalized_metrics['hit_5']:.4f} |",
            f"| Hit@10 | {normalized_metrics['hit_10']:.4f} |",
            f"| MRR | {normalized_metrics['mrr']:.4f} |",
            "",
            "## Weak Cases",
            "",
            "Weak cases are questions where the expected answer was not found or was not ranked first.",
            "",
            "| ID | Query | Expected Pages | Answer Rank | Top1 Source | Top1 Page | Top1 Section |",
            "|---|---|---|---:|---|---:|---|",
        ]
    )

    if weak_cases:
        for item in weak_cases:
            top1 = get_top1(item)
            lines.append(
                "| "
                f"{escape_markdown(str(item['id']))} | "
                f"{escape_markdown(item['query'])} | "
                f"{escape_markdown(format_answers(item['answers']))} | "
                f"{format_answer_rank(item.get('answer_rank'))} | "
                f"{escape_markdown(str(top1.get('source_file', '')))} | "
                f"{top1.get('page_number', '')} | "
                f"{escape_markdown(str(top1.get('section_title', '')))} |"
            )
    else:
        lines.append("| - | No weak cases | - | - | - | - | - |")

    return "\n".join(lines) + "\n"


def get_top1(item: Dict[str, Any]) -> Dict[str, Any]:
    top_results = item.get("top_results", [])
    if not top_results:
        return {}
    return top_results[0]


def format_answers(answers: List[Dict[str, Any]]) -> str:
    return ", ".join(
        f"{answer.get('file')} p.{answer.get('page')}"
        for answer in answers
    )


def format_answer_rank(answer_rank: Optional[int]) -> str:
    if answer_rank is None:
        return "Not found"
    return str(answer_rank)


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
