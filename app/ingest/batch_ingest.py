import argparse
import json
from pathlib import Path

from tqdm import tqdm

from pdf_loader import load_pdf_by_page
from chunkers import fixed_size_chunking, paragraph_chunking


def save_json(data, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw", help="PDF 폴더 경로")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "paragraph"],
        default="paragraph",
        help="chunking 전략",
    )
    parser.add_argument(
        "--output",
        default="data/processed/chunks_all.json",
        help="전체 chunk 저장 경로",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"PDF 파일이 없습니다: {input_dir}")

    all_chunks = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            pages = load_pdf_by_page(str(pdf_path))

            if args.strategy == "fixed":
                chunks = fixed_size_chunking(pages)
            else:
                chunks = paragraph_chunking(pages)

            all_chunks.extend(chunks)

            print(f"[OK] {pdf_path.name}: pages={len(pages)}, chunks={len(chunks)}")

        except Exception as e:
            print(f"[ERROR] {pdf_path.name}: {e}")

    save_json(all_chunks, args.output)

    print("\nDone")
    print(f"PDF files processed: {len(pdf_files)}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()