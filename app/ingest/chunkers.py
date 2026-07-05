from typing import List, Dict, Tuple
import re


SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "approach",
    "model",
    "experiments",
    "evaluation",
    "evaluation strategies",
    "results",
    "analysis",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
}

CAPTION_PREFIX_PATTERN = re.compile(r"^(table|figure|fig\.)\b", re.IGNORECASE)
METRIC_ROW_TERMS = {
    "gpt score",
    "gpt ranking",
    "agreement",
    "score",
    "ranking",
    "precision",
    "recall",
    "f1",
    "accuracy",
}


def fixed_size_chunking(
    pages: List[Dict],
    chunk_size: int = 600,
    overlap: int = 100,
) -> List[Dict]:
    """
    고정 길이 기반 chunking.
    장점: 구현이 쉽고 안정적.
    단점: 문단/조항/표의 의미가 중간에 끊길 수 있음.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap은 chunk_size보다 작아야 합니다.")

    chunks = []

    for page in pages:
        text = normalize_text(page["text"])
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "source_file": page["source_file"],
                        "page_number": page["page_number"],
                        "chunk_id": f"{page['page_number']}_fixed_{chunk_id}",
                        "chunking_strategy": "fixed_size",
                        "text": chunk_text,
                    }
                )

            start += chunk_size - overlap
            chunk_id += 1

    return chunks


def paragraph_chunking(
    pages: List[Dict],
    max_chunk_size: int = 900,
) -> List[Dict]:
    """
    문단 기반 chunking.
    문단 단위로 묶되, 너무 긴 문단은 적당히 분리합니다.
    """
    chunks = []

    for page in pages:
        text = normalize_text(page["text"])
        paragraphs = split_paragraphs(text)

        buffer = ""
        chunk_id = 0

        for paragraph in paragraphs:
            if not paragraph:
                continue

            if len(buffer) + len(paragraph) <= max_chunk_size:
                buffer = f"{buffer}\n{paragraph}".strip()
            else:
                if buffer:
                    chunks.append(
                        {
                            "source_file": page["source_file"],
                            "page_number": page["page_number"],
                            "chunk_id": f"{page['page_number']}_para_{chunk_id}",
                            "chunking_strategy": "paragraph",
                            "text": buffer,
                        }
                    )
                    chunk_id += 1

                if len(paragraph) > max_chunk_size:
                    split_chunks = split_long_text(paragraph, max_chunk_size)
                    for part in split_chunks:
                        chunks.append(
                            {
                                "source_file": page["source_file"],
                                "page_number": page["page_number"],
                                "chunk_id": f"{page['page_number']}_para_{chunk_id}",
                                "chunking_strategy": "paragraph",
                                "text": part,
                            }
                        )
                        chunk_id += 1
                    buffer = ""
                else:
                    buffer = paragraph

        if buffer:
            chunks.append(
                {
                    "source_file": page["source_file"],
                    "page_number": page["page_number"],
                    "chunk_id": f"{page['page_number']}_para_{chunk_id}",
                    "chunking_strategy": "paragraph",
                    "text": buffer,
                }
            )

    return chunks


def section_aware_chunking(
    pages: List[Dict],
    max_chunk_size: int = 1200,
) -> List[Dict]:
    """
    논문형 PDF를 위한 간단한 section-aware chunking.
    텍스트에서 section heading을 추적하고, section_title 메타데이터를 추가합니다.
    """
    chunks = []
    current_section = "Unknown"
    buffer = ""
    buffer_page_number = None
    buffer_source_file = None
    chunk_id = 0

    def flush_buffer() -> None:
        nonlocal buffer
        nonlocal buffer_page_number
        nonlocal buffer_source_file
        nonlocal chunk_id

        chunk_text = buffer.strip()
        if not chunk_text:
            buffer = ""
            return

        chunks.append(
            {
                "source_file": buffer_source_file,
                "page_number": buffer_page_number,
                "chunk_id": f"{buffer_page_number}_section_{chunk_id}",
                "chunking_strategy": "section",
                "section_title": current_section,
                "text": chunk_text,
            }
        )
        chunk_id += 1
        buffer = ""
        buffer_page_number = None
        buffer_source_file = None

    for page in pages:
        text = normalize_text(page["text"])
        units = split_section_units(text)

        for unit_type, value in units:
            if unit_type == "heading":
                flush_buffer()
                current_section = value
                continue

            paragraph = value.strip()
            if not paragraph:
                continue

            if len(paragraph) > max_chunk_size:
                flush_buffer()
                for part in split_long_text(paragraph, max_chunk_size):
                    if not part:
                        continue
                    chunks.append(
                        {
                            "source_file": page["source_file"],
                            "page_number": page["page_number"],
                            "chunk_id": f"{page['page_number']}_section_{chunk_id}",
                            "chunking_strategy": "section",
                            "section_title": current_section,
                            "text": part,
                        }
                    )
                    chunk_id += 1
                continue

            if not buffer:
                buffer = paragraph
                buffer_page_number = page["page_number"]
                buffer_source_file = page["source_file"]
                continue

            next_buffer = f"{buffer}\n\n{paragraph}".strip()
            if len(next_buffer) <= max_chunk_size:
                buffer = next_buffer
            else:
                flush_buffer()
                buffer = paragraph
                buffer_page_number = page["page_number"]
                buffer_source_file = page["source_file"]

    flush_buffer()
    return chunks


def normalize_text(text: str) -> str:
    """
    PDF 추출 텍스트의 과도한 공백을 정리합니다.
    """
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> List[str]:
    """
    빈 줄 기준으로 문단을 나눕니다.
    """
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_section_units(text: str) -> List[Tuple[str, str]]:
    """
    section heading과 paragraph를 순서대로 분리합니다.
    """
    units = []

    for paragraph in split_paragraphs(text):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        buffer = []
        index = 0

        while index < len(lines):
            line = lines[index]
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            combined_heading = combine_numbered_heading(line, next_line)

            if combined_heading:
                append_paragraph_unit(units, buffer)
                units.append(("heading", combined_heading))
                index += 2
                continue

            heading = detect_section_heading(line)
            if heading:
                append_paragraph_unit(units, buffer)
                units.append(("heading", heading))
                index += 1
                continue

            buffer.append(line)
            index += 1

        append_paragraph_unit(units, buffer)

    return units


def append_paragraph_unit(units: List[Tuple[str, str]], buffer: List[str]) -> None:
    paragraph = "\n".join(buffer).strip()
    if paragraph:
        units.append(("paragraph", paragraph))
    buffer.clear()


def combine_numbered_heading(line: str, next_line: str) -> str:
    if not next_line:
        return ""

    stripped_line = line.strip()
    if re.fullmatch(r"\d{1,2}\.", stripped_line):
        return ""

    number_match = re.fullmatch(r"\d{1,2}(?:\.\d{1,2})*\.?", stripped_line)
    if not number_match:
        return ""

    heading = detect_section_heading(f"{stripped_line} {next_line}")
    return heading or ""


def detect_section_heading(line: str) -> str:
    cleaned = clean_heading(line)
    if not cleaned:
        return ""

    if not is_section_heading(cleaned):
        return ""

    return cleaned


def is_section_heading(line: str) -> bool:
    cleaned = clean_heading(line)
    if not cleaned:
        return False

    if is_table_or_figure_caption(cleaned):
        return False

    if is_numeric_metric_row(cleaned):
        return False

    normalized = cleaned.lower().rstrip(":")
    if normalized in SECTION_HEADINGS:
        return True

    top_level_match = re.match(r"^\d{1,2}\s+(.+)$", cleaned)
    if top_level_match:
        title = top_level_match.group(1).strip().rstrip(":")
        if is_reasonable_heading_title(title):
            return True

    subsection_match = re.match(r"^(\d{1,2})\.\d{1,2}(?:\.\d{1,2})*\.?\s+(.+)$", cleaned)
    if subsection_match:
        major_number = int(subsection_match.group(1))
        if major_number < 1:
            return False

        title = subsection_match.group(2).strip().rstrip(":")
        if is_reasonable_heading_title(title):
            return True

    return False


def is_table_or_figure_caption(line: str) -> bool:
    return bool(CAPTION_PREFIX_PATTERN.match(line))


def is_numeric_metric_row(line: str) -> bool:
    decimal_match = re.match(r"^\d+\.\d+\s+(.+)$", line)
    if not decimal_match:
        return False

    remainder = decimal_match.group(1).lower()
    return any(term in remainder for term in METRIC_ROW_TERMS)


def clean_heading(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_reasonable_heading_title(title: str) -> bool:
    if len(title) > 140:
        return False

    words = title.split()
    if not words or len(words) > 18:
        return False

    normalized = title.lower().rstrip(":")
    if normalized in SECTION_HEADINGS:
        return True

    if title.endswith((".", ",", ";")):
        return False

    alpha_chars = [char for char in title if char.isalpha()]
    if not alpha_chars:
        return False

    uppercase_ratio = sum(char.isupper() for char in alpha_chars) / len(alpha_chars)
    starts_like_heading = title[0].isupper()

    return starts_like_heading or uppercase_ratio >= 0.5


def split_long_text(text: str, max_size: int) -> List[str]:
    """
    너무 긴 문단을 max_size 기준으로 분리합니다.
    """
    return [text[i : i + max_size].strip() for i in range(0, len(text), max_size)]
