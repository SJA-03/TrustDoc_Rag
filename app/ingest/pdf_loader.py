from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF


def load_pdf_by_page(pdf_path: str) -> List[Dict]:
    """
    PDF를 페이지 단위로 읽어 텍스트와 메타데이터를 반환합니다.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("PDF 파일만 지원합니다.")

    documents = []

    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc):
            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                {
                    "source_file": path.name,
                    "page_number": page_index + 1,
                    "text": text,
                }
            )

    return documents