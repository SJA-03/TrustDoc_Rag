"""Validate frozen held-out retrieval question sets without running retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import fitz


VALID_DOMAINS = {"operating_systems", "ai_papers"}
VALID_LANGUAGES = {"en", "ko"}
VALID_QUERY_TYPES = {
    "definition",
    "comparison",
    "mechanism",
    "reason",
    "exact_terminology",
    "similar_concept",
    "multi_page",
}
REQUIRED_METADATA = {"split", "domain", "language", "query_type"}


class QuestionValidationError(ValueError):
    """Raised when one or more held-out question records are invalid."""


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise QuestionValidationError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise QuestionValidationError(
                    f"{path}:{line_number}: each line must be a JSON object"
                )
            record["_validation_source"] = f"{path}:{line_number}"
            records.append(record)
    if not records:
        raise QuestionValidationError(f"{path}: no question records found")
    return records


def collect_dev_queries(paths: Sequence[Path]) -> Set[str]:
    queries: Set[str] = set()
    for path in paths:
        for record in load_jsonl(path):
            query = record.get("query")
            if isinstance(query, str) and query.strip():
                queries.add(normalize_query(query))
    return queries


def build_source_page_inventory(source_root: Path) -> Dict[str, int]:
    inventory: Dict[str, int] = {}
    for path in sorted(source_root.rglob("*.pdf")):
        if path.name in inventory:
            raise QuestionValidationError(
                f"Duplicate PDF basename under {source_root}: {path.name}"
            )
        with fitz.open(path) as document:
            inventory[path.name] = document.page_count
    return inventory


def validate_records(
    records: Sequence[Mapping[str, Any]],
    dev_queries: Iterable[str] = (),
    source_pages: Optional[Mapping[str, int]] = None,
) -> None:
    errors: List[str] = []
    seen_ids: Set[str] = set()
    seen_queries: Set[str] = set()
    normalized_dev_queries = {normalize_query(query) for query in dev_queries}

    for index, record in enumerate(records, start=1):
        source = str(record.get("_validation_source", f"record {index}"))
        question_id = record.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            errors.append(f"{source}: id must be a non-empty string")
        elif question_id in seen_ids:
            errors.append(f"{source}: duplicate id: {question_id}")
        else:
            seen_ids.add(question_id)

        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            errors.append(f"{source}: query must be a non-empty string")
        else:
            normalized = normalize_query(query)
            if normalized in seen_queries:
                errors.append(f"{source}: duplicate held-out query: {query}")
            else:
                seen_queries.add(normalized)
            if normalized in normalized_dev_queries:
                errors.append(f"{source}: exact duplicate of a development query: {query}")

        answers = record.get("answers")
        if not isinstance(answers, list) or not answers:
            errors.append(f"{source}: answers must be a non-empty list")
        else:
            for answer_index, answer in enumerate(answers, start=1):
                answer_source = f"{source}: answer {answer_index}"
                if not isinstance(answer, dict):
                    errors.append(f"{answer_source} must be an object")
                    continue
                file_name = answer.get("file")
                page = answer.get("page")
                if not isinstance(file_name, str) or not file_name.strip():
                    errors.append(f"{answer_source}: file must be a non-empty string")
                if type(page) is not int or page <= 0:
                    errors.append(f"{answer_source}: page must be a positive integer")
                if source_pages is not None and isinstance(file_name, str):
                    if file_name not in source_pages:
                        errors.append(f"{answer_source}: source PDF not found: {file_name}")
                    elif type(page) is int and page > source_pages[file_name]:
                        errors.append(
                            f"{answer_source}: page {page} exceeds PDF page count "
                            f"{source_pages[file_name]}"
                        )

        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            errors.append(f"{source}: metadata must be an object")
            continue
        missing_metadata = REQUIRED_METADATA - metadata.keys()
        if missing_metadata:
            errors.append(
                f"{source}: metadata missing: {', '.join(sorted(missing_metadata))}"
            )
        if metadata.get("split") != "heldout":
            errors.append(f"{source}: metadata.split must equal heldout")
        if metadata.get("domain") not in VALID_DOMAINS:
            errors.append(f"{source}: invalid domain: {metadata.get('domain')}")
        if metadata.get("language") not in VALID_LANGUAGES:
            errors.append(f"{source}: invalid language: {metadata.get('language')}")
        if metadata.get("query_type") not in VALID_QUERY_TYPES:
            errors.append(f"{source}: invalid query_type: {metadata.get('query_type')}")

    if errors:
        raise QuestionValidationError("\n".join(errors))


def validate_files(
    heldout_paths: Sequence[Path],
    dev_paths: Sequence[Path],
    source_root: Optional[Path],
) -> int:
    records = [record for path in heldout_paths for record in load_jsonl(path)]
    dev_queries = collect_dev_queries(dev_paths)
    source_pages = (
        build_source_page_inventory(source_root) if source_root is not None else None
    )
    validate_records(records, dev_queries=dev_queries, source_pages=source_pages)
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", nargs="+", required=True, type=Path)
    parser.add_argument("--dev", nargs="*", default=[], type=Path)
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = validate_files(args.heldout, args.dev, args.source_root)
    print(f"Validated {count} held-out questions across {len(args.heldout)} file(s).")


if __name__ == "__main__":
    main()
