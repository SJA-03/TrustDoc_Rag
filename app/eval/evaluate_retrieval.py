import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "app" / "rag"))

from retriever import ChromaRetriever


def load_questions(path: str):
    questions = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            questions.append(json.loads(line))

    return questions


def find_answer_rank(results, answers):
    for idx, chunk in enumerate(results):
        meta = chunk["metadata"]

        for answer in answers:
            if (
                meta["source_file"] == answer["file"]
                and int(meta["page_number"]) == int(answer["page"])
            ):
                return idx + 1

    return None


def evaluate(questions, collection_name: str, top_k: int):
    retriever = ChromaRetriever(collection_name=collection_name)

    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0
    mrr_total = 0.0

    detailed_results = []

    for q in questions:
        results = retriever.retrieve(q["query"], top_k=top_k)

        rank = find_answer_rank(
            results,
            answers=q["answers"],
        )

        if rank == 1:
            hit_1 += 1

        if rank is not None and rank <= 3:
            hit_3 += 1

        if rank is not None and rank <= 5:
            hit_5 += 1

        if rank is not None and rank <= 10:
            hit_10 += 1

        if rank is not None:
            mrr_total += 1 / rank

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

    total = len(questions)

    metrics = {
        "total": total,
        "hit@1": hit_1 / total,
        "hit@3": hit_3 / total,
        "hit@5": hit_5 / total,
        "hit@10": hit_10 / total,
        "mrr": mrr_total / total,
    }

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