from typing import List, Dict
import re


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


def split_long_text(text: str, max_size: int) -> List[str]:
    """
    너무 긴 문단을 max_size 기준으로 분리합니다.
    """
    return [text[i : i + max_size].strip() for i in range(0, len(text), max_size)]