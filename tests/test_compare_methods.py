import unittest

from app.eval.compare_methods import (
    align_query_records,
    analyze_artifacts,
    compare_query_results,
    paired_bootstrap,
    rank_values,
)


def query_result(query_id, answer_rank, query=None):
    return {
        "id": query_id,
        "query": query or f"Question {query_id}",
        "answer_rank": answer_rank,
        "metadata": {"query_type": "definition"},
    }


class CompareMethodsTest(unittest.TestCase):
    def test_paired_query_alignment_uses_ids(self):
        left = [query_result("q2", 2), query_result("q1", 1)]
        right = [query_result("q1", 3), query_result("q2", 1)]
        aligned = align_query_records(left, right)
        self.assertEqual([pair[0]["id"] for pair in aligned], ["q1", "q2"])
        self.assertTrue(all(a["id"] == b["id"] for a, b in aligned))

    def test_observed_hit1_delta(self):
        left = [query_result("q1", 2), query_result("q2", 1)]
        right = [query_result("q1", 1), query_result("q2", 1)]
        result = compare_query_results(left, right, resamples=100, seed=3)
        self.assertEqual(result["hit@1"]["observed_delta"], 0.5)

    def test_observed_mrr_delta(self):
        left = [query_result("q1", 2), query_result("q2", None)]
        right = [query_result("q1", 1), query_result("q2", 2)]
        result = compare_query_results(left, right, resamples=100, seed=3)
        self.assertEqual(result["mrr"]["observed_delta"], 0.5)

    def test_bootstrap_is_deterministic_with_seed(self):
        deltas = [1.0, 0.0, -0.5, 0.25]
        first = paired_bootstrap(deltas, resamples=250, seed=17)
        second = paired_bootstrap(deltas, resamples=250, seed=17)
        self.assertEqual(first, second)

    def test_confidence_interval_output(self):
        result = paired_bootstrap([1.0, 0.0, -1.0], resamples=200, seed=5)
        interval = result["confidence_interval"]
        self.assertIn("lower", interval)
        self.assertIn("upper", interval)
        self.assertLessEqual(interval["lower"], interval["upper"])
        self.assertTrue(result["ci_contains_zero"])

    def test_win_tie_loss(self):
        left = [
            query_result("loss", 1),
            query_result("tie", 2),
            query_result("win", 3),
        ]
        right = [
            query_result("loss", 2),
            query_result("tie", 2),
            query_result("win", 1),
        ]
        outcomes = compare_query_results(left, right, resamples=50)["mrr"][
            "query_outcomes"
        ]
        self.assertEqual(outcomes["wins"]["query_ids"], ["win"])
        self.assertEqual(outcomes["ties"]["query_ids"], ["tie"])
        self.assertEqual(outcomes["losses"]["query_ids"], ["loss"])

    def test_missing_query_id_detection(self):
        left = [query_result("q1", 1), query_result("q2", 2)]
        right = [query_result("q1", 1)]
        with self.assertRaisesRegex(ValueError, "Query ID mismatch"):
            align_query_records(left, right)

    def test_answer_not_found_has_zero_rr(self):
        self.assertEqual(rank_values(query_result("q1", None)), (0.0, 0.0))

    def test_chunking_artifact_comparison(self):
        paragraph = {
            "methods": {
                "dense": [query_result("q1", 1), query_result("q2", 3)]
            }
        }
        alternative = {
            "methods": {
                "dense": [query_result("q1", 2), query_result("q2", 1)]
            }
        }
        comparisons, outcomes = analyze_artifacts(
            {"paragraph": paragraph, "alternative": alternative},
            method_pairs=[],
            chunking_pairs=[("chunking", "paragraph", "alternative")],
            resamples=100,
            seed=11,
        )
        self.assertEqual(len(comparisons["chunking_comparisons"]), 1)
        self.assertEqual(
            outcomes["candidate_counts"]["paragraph_beats_alternative"], 1
        )
        self.assertEqual(
            outcomes["candidate_counts"]["alternative_beats_paragraph"], 1
        )


if __name__ == "__main__":
    unittest.main()
