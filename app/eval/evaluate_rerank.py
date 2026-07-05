import argparse
import json
from typing import Any, Dict, List, Optional

from app.rag.retriever import ChromaRetriever
from app.rag.reranker import CrossEncoderReranker


def load_questions(path: str) -> List[Dict[str, Any]]:
    questions = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))

    return questions


def find_answer_rank(results: List[Dict[str, Any]], answers: List[Dict[str, Any]]) -> Optional[int]:
    for idx, chunk in enumerate(results):
        meta = chunk["metadata"]

        for answer in answers:
            if (
                meta["source_file"] == answer["file"]
                and int(meta["page_number"]) == int(answer["page"])
            ):
                return idx + 1

    return None


def evaluate(
    questions: List[Dict[str, Any]],
    collection_name: str,
    initial_top_k: int,
) -> Dict[str, Any]:
    retriever = ChromaRetriever(collection_name=collection_name)
    reranker = CrossEncoderReranker()

    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    hit_10 = 0
    mrr_total = 0.0
    details = []

    for q in questions:
        query = q["query"]
        answers = q["answers"]

        retrieved_chunks = retriever.retrieve(query, top_k=initial_top_k)
        reranked_chunks = reranker.rerank(query, retrieved_chunks)

        rank = find_answer_rank(reranked_chunks, answers)

        if rank is not None and rank <= 1:
            hit_1 += 1
        if rank is not None and rank <= 3:
            hit_3 += 1
        if rank is not None and rank <= 5:
            hit_5 += 1
        if rank is not None and rank <= 10:
            hit_10 += 1
        if rank is not None:
            mrr_total += 1.0 / rank

        details.append(
            {
                "id": q["id"],
                "query": query,
                "answers": answers,
                "answer_rank": rank,
                "results": reranked_chunks,
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

    return {
        "metrics": metrics,
        "details": details,
    }


def format_answers(answers: List[Dict[str, Any]]) -> str:
    return ", ".join([f"{a['file']} p.{a['page']}" for a in answers])


def print_results(result: Dict[str, Any], collection_name: str, initial_top_k: int):
    metrics = result["metrics"]

    print()
    print("Rerank Evaluation Result")
    print("=" * 80)
    print(f"Collection: {collection_name}")
    print(f"Initial top_k: {initial_top_k}")
    print(f"Total questions: {metrics['total']}")
    print(f"Hit@1:  {metrics['hit@1']:.4f}")
    print(f"Hit@3:  {metrics['hit@3']:.4f}")
    print(f"Hit@5:  {metrics['hit@5']:.4f}")
    print(f"Hit@10: {metrics['hit@10']:.4f}")
    print(f"MRR:     {metrics['mrr']:.4f}")

    print()
    print("Detailed Results")
    print("=" * 80)

    for item in result["details"]:
        print()
        print(f"[{item['id']}] {item['query']}")
        print(f"Answers: {format_answers(item['answers'])}")

        if item["answer_rank"] is None:
            print("Answer rank: Not found")
        else:
            print(f"Answer rank: {item['answer_rank']}")

        for chunk in item["results"][:10]:
            meta = chunk["metadata"]
            print(
                f"  - rank {chunk['rank']}: "
                f"{meta['source_file']} p.{meta['page_number']} "
                f"({meta['chunk_id']}), "
                f"original_rank={chunk.get('original_rank')}, "
                f"rerank_score={chunk['rerank_score']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--initial_top_k", type=int, default=10)

    args = parser.parse_args()

    questions = load_questions(args.questions)
    result = evaluate(
        questions=questions,
        collection_name=args.collection,
        initial_top_k=args.initial_top_k,
    )

    print_results(result, args.collection, args.initial_top_k)


if __name__ == "__main__":
    main()
