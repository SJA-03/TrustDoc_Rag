import unittest

from app.eval.validate_questions import QuestionValidationError, validate_records


def valid_record(question_id: str = "test_q01", query: str = "A valid question?"):
    return {
        "id": question_id,
        "query": query,
        "answers": [{"file": "source.pdf", "page": 1}],
        "metadata": {
            "split": "heldout",
            "domain": "operating_systems",
            "language": "en",
            "query_type": "definition",
        },
    }


class ValidateQuestionsTest(unittest.TestCase):
    def test_duplicate_id(self):
        records = [valid_record(), valid_record(query="A second valid question?")]
        with self.assertRaisesRegex(QuestionValidationError, "duplicate id"):
            validate_records(records)

    def test_empty_answers(self):
        record = valid_record()
        record["answers"] = []
        with self.assertRaisesRegex(QuestionValidationError, "answers must be"):
            validate_records([record])

    def test_invalid_query_type(self):
        record = valid_record()
        record["metadata"]["query_type"] = "made_up_type"
        with self.assertRaisesRegex(QuestionValidationError, "invalid query_type"):
            validate_records([record])

    def test_missing_metadata(self):
        record = valid_record()
        del record["metadata"]
        with self.assertRaisesRegex(QuestionValidationError, "metadata must be"):
            validate_records([record])

    def test_dev_query_exact_duplicate(self):
        record = valid_record(query="What is demand paging?")
        with self.assertRaisesRegex(QuestionValidationError, "development query"):
            validate_records([record], dev_queries={"what is demand paging?"})


if __name__ == "__main__":
    unittest.main()
