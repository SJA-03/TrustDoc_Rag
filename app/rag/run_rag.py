import argparse

from retriever import ChromaRetriever
from build_prompt import build_rag_prompt
from llm_client import GeminiClient


def print_retrieved_chunks(chunks):
    print("\nRetrieved Chunks")
    print("=" * 80)

    for chunk in chunks:
        meta = chunk["metadata"]
        print(
            f"[{chunk['rank']}] "
            f"distance={chunk['distance']:.4f} | "
            f"{meta['source_file']} p.{meta['page_number']} | "
            f"{meta['chunk_id']}"
        )


def print_retrieved_sources(chunks):
    print("\nRetrieved Sources")
    print("=" * 80)

    seen = set()

    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta["source_file"], meta["page_number"])

        if key in seen:
            continue

        seen.add(key)
        print(f"- {meta['source_file']} p.{meta['page_number']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="사용자 질문")
    parser.add_argument("--collection", default="trustdoc_os_paragraph")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--show_prompt", action="store_true")
    args = parser.parse_args()

    retriever = ChromaRetriever(collection_name=args.collection)
    chunks = retriever.retrieve(args.query, top_k=args.top_k)

    prompt = build_rag_prompt(args.query, chunks)

    print_retrieved_chunks(chunks)

    if args.show_prompt:
        print("\nGenerated Prompt")
        print("=" * 80)
        print(prompt)

    llm = GeminiClient()
    answer = llm.generate(prompt)

    print("\nAnswer")
    print("=" * 80)
    print(answer)

    print_retrieved_sources(chunks)


if __name__ == "__main__":
    main()