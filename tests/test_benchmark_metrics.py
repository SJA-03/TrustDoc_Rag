import unittest

from app.eval.metrics import (
    aggregate_ranks,
    bootstrap_confidence_intervals,
    build_query_record,
    find_answer_rank,
    percentile,
    rank_metrics,
    summarize_latency,
)


def result(source_file: str, page_number: int):
    return {
        "metadata": {
            "source_file": source_file,
            "page_number": page_number,
            "chunk_id": f"{page_number}_chunk",
        }
    }


class BenchmarkMetricsTest(unittest.TestCase):
    def test_find_answer_rank(self):
        results = [result("other.pdf", 1), result("answer.pdf", 7)]
        self.assertEqual(find_answer_rank(results, [{"file": "answer.pdf", "page": 7}]), 2)

    def test_hit_at_k(self):
        metrics = rank_metrics(3)
        self.assertFalse(metrics["hit@1"])
        self.assertTrue(metrics["hit@3"])
        self.assertTrue(metrics["hit@5"])
        self.assertTrue(metrics["hit@10"])

    def test_mrr(self):
        metrics = aggregate_ranks([1, 2, None])
        self.assertAlmostEqual(metrics["mrr"], 0.5)

    def test_multiple_valid_answer_pages(self):
        results = [result("paper.pdf", 9), result("paper.pdf", 3)]
        answers = [{"file": "paper.pdf", "page": 3}, {"file": "paper.pdf", "page": 9}]
        self.assertEqual(find_answer_rank(results, answers), 1)

    def test_answer_not_found(self):
        self.assertIsNone(
            find_answer_rank(
                [result("paper.pdf", 1)],
                [{"file": "paper.pdf", "page": 2}],
            )
        )
        metrics = rank_metrics(None)
        self.assertEqual(metrics["reciprocal_rank"], 0.0)
        self.assertFalse(any(metrics[f"hit@{k}"] for k in (1, 3, 5, 10)))

    def test_bootstrap_reproducibility_with_seed(self):
        per_query = [
            {"hit@1": True, "reciprocal_rank": 1.0},
            {"hit@1": False, "reciprocal_rank": 0.5},
            {"hit@1": False, "reciprocal_rank": 0.0},
        ]
        first = bootstrap_confidence_intervals(per_query, resamples=500, seed=17)
        second = bootstrap_confidence_intervals(per_query, resamples=500, seed=17)
        self.assertEqual(first, second)

    def test_percentile_and_latency_summary(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5), 3.0)
        summary = summarize_latency([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(summary["mean_ms"], 3.0)
        self.assertEqual(summary["median_ms"], 3.0)
        self.assertEqual(summary["p50_ms"], 3.0)
        self.assertAlmostEqual(summary["p95_ms"], 4.8)

    def test_optional_metadata_preservation(self):
        question = {
            "id": "q1",
            "query": "query",
            "answers": [{"file": "paper.pdf", "page": 1}],
            "metadata": {"domain": "systems", "language": "en"},
        }
        record = build_query_record(question, 1, 2.5, [])
        self.assertEqual(record["metadata"], question["metadata"])


if __name__ == "__main__":
    unittest.main()
