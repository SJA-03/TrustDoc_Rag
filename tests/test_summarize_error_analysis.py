import copy
import unittest

from app.eval.summarize_error_analysis import (
    build_summary,
    validate_annotations,
)


def valid_document():
    case = {
        "case_id": "case_001",
        "query_id": "q1",
        "domain": "operating_systems",
        "setting": "os_paragraph",
        "comparison": "dense_to_bm25",
        "method_a": "dense",
        "method_b": "bm25",
        "query": "What is the concept?",
        "metadata": {"query_type": "definition"},
        "expected_answers": [{"file": "source.pdf", "page": 3}],
        "rank_a": 2,
        "rank_b": 1,
        "reciprocal_rank_a": 0.5,
        "reciprocal_rank_b": 1.0,
        "pipeline_mechanism": "lexical_rescue",
        "primary_factor": "exact_terminology",
        "secondary_factor": None,
        "observation": "The target moved from rank 2 to rank 1.",
        "interpretation": "Exact terminology likely helped lexical retrieval.",
        "ground_truth_verified": True,
        "retrieved_candidates_reviewed": True,
    }
    return {
        "analysis_metadata": {
            "scope": "held-out only",
            "dataset": "frozen benchmark",
            "selection_policy": "purposive manual sample",
            "ground_truth_commit": "abc123",
            "analysis_commit_base": "def456",
            "descriptive_analysis": True,
        },
        "research_questions": [
            {
                "id": "RQ1",
                "question": "Why?",
                "answer": "The case is consistent with exact-term rescue.",
                "evidence_case_ids": ["case_001"],
            }
        ],
        "key_findings": [
            {
                "id": "F1",
                "text": "Exact terms helped BM25.",
                "evidence_case_ids": ["case_001"],
            }
        ],
        "unexpected_results": [
            {
                "id": "U1",
                "text": "The change was larger than expected.",
                "evidence_case_ids": ["case_001"],
            }
        ],
        "limitations": ["One synthetic case cannot support general conclusions."],
        "cases": [case],
    }


class SummarizeErrorAnalysisTest(unittest.TestCase):
    def test_valid_annotation_schema(self):
        validate_annotations(valid_document())

    def test_invalid_taxonomy_label_is_rejected(self):
        document = valid_document()
        document["cases"][0]["pipeline_mechanism"] = "keyword_magic"
        with self.assertRaisesRegex(ValueError, "invalid taxonomy label"):
            validate_annotations(document)

    def test_duplicate_case_id_is_rejected(self):
        document = valid_document()
        document["cases"].append(copy.deepcopy(document["cases"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate case ID"):
            validate_annotations(document)

    def test_summary_counts_are_consistent(self):
        document = valid_document()
        second = copy.deepcopy(document["cases"][0])
        second.update(
            {
                "case_id": "case_002",
                "query_id": "q2",
                "domain": "ai_papers",
                "pipeline_mechanism": "reranker_inversion",
                "primary_factor": "semantic_neighbor",
            }
        )
        document["cases"].append(second)
        summary = build_summary(document)
        self.assertEqual(summary["total_cases"], 2)
        self.assertEqual(sum(summary["by_domain"].values()), 2)
        self.assertEqual(sum(summary["by_pipeline_mechanism"].values()), 2)
        self.assertEqual(sum(summary["by_primary_factor"].values()), 2)
        self.assertEqual(
            sum(sum(row.values()) for row in summary["mechanism_by_domain"].values()),
            2,
        )

    def test_rank_and_reciprocal_rank_must_agree(self):
        document = valid_document()
        document["cases"][0]["reciprocal_rank_a"] = 1.0
        with self.assertRaisesRegex(ValueError, "inconsistent rank"):
            validate_annotations(document)


if __name__ == "__main__":
    unittest.main()
