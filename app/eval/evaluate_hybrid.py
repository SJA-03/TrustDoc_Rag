import argparse
from typing import Any, Dict, List

from app.eval.metrics import aggregate_ranks, find_answer_rank, load_questions
from app.rag.hybrid_retriever import HybridRetriever


def evaluate(
    questions: List[Dict[str, Any]],
    chunks_path: str,
    collection_name: str,
    top_k: int,
    dense_top_k: int,
    bm25_top_k: int,
) -> Dict[str, Any]:
    retriever = HybridRetriever(
        chunks_path=chunks_path,
        collection_name=collection_name,
    )

    details = []
    answer_ranks = []

    for q in questions:
        query = q["query"]
        answers = q["answers"]

        results = retriever.retrieve(
            query=query,
            top_k=top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
        )

        rank = find_answer_rank(results, answers)

        answer_ranks.append(rank)

        details.append(
            {
                "id": q["id"],
                "query": query,
                "answers": answers,
                "answer_rank": rank,
                "results": results,
            }
        )

    return {
        "metrics": aggregate_ranks(answer_ranks),
        "details": details,
    }


def format_answers(answers: List[Dict[str, Any]]) -> str:
    return ", ".join([f"{a['file']} p.{a['page']}" for a in answers])


def print_results(
    result: Dict[str, Any],
    chunks_path: str,
    collection_name: str,
    top_k: int,
    dense_top_k: int,
    bm25_top_k: int,
):
    metrics = result["metrics"]

    print()
    print("Hybrid Retrieval Evaluation Result")
    print("=" * 80)
    print(f"Chunks path: {chunks_path}")
    print(f"Collection: {collection_name}")
    print(f"top_k: {top_k}")
    print(f"dense_top_k: {dense_top_k}")
    print(f"bm25_top_k: {bm25_top_k}")
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
                f"dense_rank={chunk.get('dense_rank')}, "
                f"bm25_rank={chunk.get('bm25_rank')}, "
                f"hybrid_score={chunk['hybrid_score']:.5f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dense_top_k", type=int, default=10)
    parser.add_argument("--bm25_top_k", type=int, default=10)

    args = parser.parse_args()

    questions = load_questions(args.questions)

    result = evaluate(
        questions=questions,
        chunks_path=args.chunks,
        collection_name=args.collection,
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
    )

    print_results(
        result=result,
        chunks_path=args.chunks,
        collection_name=args.collection,
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
    )


if __name__ == "__main__":
    main()
