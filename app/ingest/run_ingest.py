import argparse
import json
from pathlib import Path

from pdf_loader import load_pdf_by_page
from chunkers import fixed_size_chunking, paragraph_chunking, section_aware_chunking


def save_json(data, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="입력 PDF 경로")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "paragraph", "section"],
        default="paragraph",
        help="chunking 전략",
    )
    parser.add_argument(
        "--output",
        default="data/processed/chunks.json",
        help="chunk 결과 저장 경로",
    )

    args = parser.parse_args()

    pages = load_pdf_by_page(args.pdf)

    if args.strategy == "fixed":
        chunks = fixed_size_chunking(pages)
    elif args.strategy == "section":
        chunks = section_aware_chunking(pages)
    else:
        chunks = paragraph_chunking(pages)

    save_json(chunks, args.output)

    print(f"PDF pages loaded: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Saved to: {args.output}")

    if chunks:
        print("\n--- Sample chunk ---")
        print(f"source_file: {chunks[0]['source_file']}")
        print(f"page_number: {chunks[0]['page_number']}")
        print(f"chunk_id: {chunks[0]['chunk_id']}")
        print(f"strategy: {chunks[0]['chunking_strategy']}")
        print(chunks[0]["text"][:500])


if __name__ == "__main__":
    main()
