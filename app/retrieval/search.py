import argparse

import chromadb
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="검색 질문")
    parser.add_argument("--persist_dir", default="data/chroma")
    parser.add_argument("--collection", default="trustdoc_os_paragraph")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    print("Loading embedding model...")
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_collection(name=args.collection)

    query_embedding = model.encode(
        [args.query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.top_k,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nQuery: {args.query}")
    print("=" * 80)

    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"\n[{i + 1}] distance={distance:.4f}")
        print(f"file: {meta['source_file']}")
        print(f"page: {meta['page_number']}")
        print(f"chunk_id: {meta['chunk_id']}")
        print(f"strategy: {meta['chunking_strategy']}")
        print("-" * 80)
        print(doc[:1000])


if __name__ == "__main__":
    main()