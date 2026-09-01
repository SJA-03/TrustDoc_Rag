import argparse
from typing import Any, Dict, List

from app.eval.metrics import aggregate_ranks, find_answer_rank, load_questions
from app.rag.retriever import ChromaRetriever
from app.rag.reranker import CrossEncoderReranker


def evaluate(
    questions: List[Dict[str, Any]],
    collection_name: str,
    initial_top_k: int,
) -> Dict[str, Any]:
    retriever = ChromaRetriever(collection_name=collection_name)
    reranker = CrossEncoderReranker()

    details = []
    answer_ranks = []

    for q in questions:
        query = q["query"]
        answers = q["answers"]

        retrieved_chunks = retriever.retrieve(query, top_k=initial_top_k)
        reranked_chunks = reranker.rerank(query, retrieved_chunks)

        rank = find_answer_rank(reranked_chunks, answers)

        answer_ranks.append(rank)

        details.append(
            {
                "id": q["id"],
                "query": query,
                "answers": answers,
                "answer_rank": rank,
                "results": reranked_chunks,
            }
        )

    metrics = aggregate_ranks(answer_ranks)

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
