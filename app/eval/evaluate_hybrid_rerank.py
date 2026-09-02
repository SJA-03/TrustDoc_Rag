import argparse
from typing import Any, Dict, List

from app.eval.metrics import aggregate_ranks, find_answer_rank, load_questions
from app.eval.report_utils import save_evaluation_json, save_markdown_report
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.reranker import CrossEncoderReranker


def evaluate(
    questions: List[Dict[str, Any]],
    chunks_path: str,
    collection_name: str,
    hybrid_top_k: int,
    dense_top_k: int,
    bm25_top_k: int,
) -> Dict[str, Any]:
    hybrid_retriever = HybridRetriever(
        chunks_path=chunks_path,
        collection_name=collection_name,
    )
    reranker = CrossEncoderReranker()

    details = []
    answer_ranks = []

    for q in questions:
        query = q["query"]
        answers = q["answers"]

        hybrid_results = hybrid_retriever.retrieve(
            query=query,
            top_k=hybrid_top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
        )

        reranked_results = reranker.rerank(query, hybrid_results)

        rank = find_answer_rank(reranked_results, answers)

        answer_ranks.append(rank)

        details.append(
            {
                "id": q["id"],
                "query": query,
                "answers": answers,
                "answer_rank": rank,
                "results": reranked_results,
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
    hybrid_top_k: int,
    dense_top_k: int,
    bm25_top_k: int,
):
    metrics = result["metrics"]

    print()
    print("Hybrid + Rerank Evaluation Result")
    print("=" * 80)
    print(f"Chunks path: {chunks_path}")
    print(f"Collection: {collection_name}")
    print(f"hybrid_top_k: {hybrid_top_k}")
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
                f"original_rank={chunk.get('original_rank')}, "
                f"dense_rank={chunk.get('dense_rank')}, "
                f"bm25_rank={chunk.get('bm25_rank')}, "
                f"hybrid_score={chunk.get('hybrid_score'):.5f}, "
                f"rerank_score={chunk['rerank_score']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--hybrid_top_k", type=int, default=10)
    parser.add_argument("--dense_top_k", type=int, default=10)
    parser.add_argument("--bm25_top_k", type=int, default=10)
    parser.add_argument("--output", help="JSON evaluation report output path")
    parser.add_argument("--markdown_output", help="Markdown evaluation report output path")

    args = parser.parse_args()

    questions = load_questions(args.questions)

    result = evaluate(
        questions=questions,
        chunks_path=args.chunks,
        collection_name=args.collection,
        hybrid_top_k=args.hybrid_top_k,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
    )

    print_results(
        result=result,
        chunks_path=args.chunks,
        collection_name=args.collection,
        hybrid_top_k=args.hybrid_top_k,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
    )

    metadata = {
        "questions_path": args.questions,
        "chunks_path": args.chunks,
        "collection": args.collection,
        "evaluation_method": "hybrid_rerank",
        "hybrid_top_k": args.hybrid_top_k,
        "dense_top_k": args.dense_top_k,
        "bm25_top_k": args.bm25_top_k,
    }

    if args.output:
        save_evaluation_json(
            output_path=args.output,
            metadata=metadata,
            metrics=result["metrics"],
            details=result["details"],
        )

    if args.markdown_output:
        save_markdown_report(
            output_path=args.markdown_output,
            metadata=metadata,
            metrics=result["metrics"],
            details=result["details"],
        )


if __name__ == "__main__":
    main()
