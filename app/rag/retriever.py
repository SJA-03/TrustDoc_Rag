from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


class ChromaRetriever:
    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "trustdoc_os_paragraph",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(name=collection_name)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved_chunks = []

        for i in range(len(results["documents"][0])):
            retrieved_chunks.append(
                {
                    "rank": i + 1,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return retrieved_chunks