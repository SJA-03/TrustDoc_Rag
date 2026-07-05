import argparse
import json

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        default="data/processed/chunks_paragraph_all.json",
        help="chunk json path",
    )
    parser.add_argument(
        "--persist_dir",
        default="data/chroma",
        help="Chroma DB persist directory",
    )
    parser.add_argument(
        "--collection",
        default="trustdoc_os_paragraph",
        help="Chroma collection name",
    )
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)

    print("Loading embedding model...")
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(path=args.persist_dir)

    try:
        client.delete_collection(args.collection)
    except Exception:
        pass

    collection = client.create_collection(name=args.collection)

    ids = []
    documents = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        ids.append(str(idx))
        documents.append(chunk["text"])
        metadatas.append(
            {
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "chunking_strategy": chunk["chunking_strategy"],
            }
        )

    batch_size = 64

    for start in tqdm(range(0, len(documents), batch_size), desc="Embedding chunks"):
        end = start + batch_size

        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_metas = metadatas[start:end]

        embeddings = model.encode(
            batch_docs,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
        )

    print("\nIndex build complete")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Persist dir: {args.persist_dir}")
    print(f"Collection: {args.collection}")


if __name__ == "__main__":
    main()