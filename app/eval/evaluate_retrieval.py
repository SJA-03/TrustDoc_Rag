import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.metrics import aggregate_ranks, find_answer_rank, load_questions
from app.rag.retriever import ChromaRetriever


def evaluate(questions, collection_name: str, top_k: int):
    retriever = ChromaRetriever(collection_name=collection_name)

    detailed_results = []
    answer_ranks = []

    for q in questions:
        results = retriever.retrieve(q["query"], top_k=top_k)

        rank = find_answer_rank(
            results,
            answers=q["answers"],
        )

        answer_ranks.append(rank)

        detailed_results.append(
            {
                "id": q["id"],
                "query": q["query"],
                "answers": q["answers"],
                "rank": rank,
                "top_results": [
                    {
                        "rank": chunk["rank"],
                        "source_file": chunk["metadata"]["source_file"],
                        "page_number": chunk["metadata"]["page_number"],
                        "chunk_id": chunk["metadata"]["chunk_id"],
                        "distance": chunk["distance"],
                    }
                    for chunk in results
                ],
            }
        )

    metrics = aggregate_ranks(answer_ranks)

    return metrics, detailed_results


def print_results(metrics, detailed_results, collection_name: str):
    print("\nEvaluation Result")
    print("=" * 80)
    print(f"Collection: {collection_name}")
    print(f"Total questions: {metrics['total']}")
    print(f"Hit@1: {metrics['hit@1']:.4f}")
    print(f"Hit@3: {metrics['hit@3']:.4f}")
    print(f"Hit@5: {metrics['hit@5']:.4f}")
    print(f"Hit@10: {metrics['hit@10']:.4f}")
    print(f"MRR:   {metrics['mrr']:.4f}")

    print("\nDetailed Results")
    print("=" * 80)

    for item in detailed_results:
        rank_text = item["rank"] if item["rank"] is not None else "Not found"

        print(f"\n[{item['id']}] {item['query']}")
        answer_text = ", ".join(
            [f"{answer['file']} p.{answer['page']}" for answer in item["answers"]]
        )
        print(f"Answers: {answer_text}")
        print(f"Answer rank: {rank_text}")

        for result in item["top_results"]:
            print(
                f"  - rank {result['rank']}: "
                f"{result['source_file']} p.{result['page_number']} "
                f"({result['chunk_id']}), "
                f"distance={result['distance']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--collection", default="trustdoc_os_paragraph")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions(args.questions)

    metrics, detailed_results = evaluate(
        questions=questions,
        collection_name=args.collection,
        top_k=args.top_k,
    )

    print_results(metrics, detailed_results, args.collection)


if __name__ == "__main__":
    main()
