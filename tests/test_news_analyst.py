import unittest
from datetime import datetime, timezone

from agents import Evidence, NewsAnalyst


class StubModel:
    def __init__(self, evidence_refs=None):
        self.evidence_refs = evidence_refs or ["news:1"]
        self.last_input = None

    def generate_json(self, *, instructions, input_data, schema):
        self.last_input = input_data
        return {
            "claim": "Synthetic test finding", "direction": "neutral", "confidence": 0.4,
            "horizon": "test horizon", "evidence_refs": self.evidence_refs,
            "risks": ["Synthetic test uncertainty"],
        }


def article():
    return Evidence(
        ref="news:1", source="Synthetic test source", url="https://example.test/article",
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        title="Synthetic test headline", summary="Synthetic test summary",
    )


class NewsAnalystTests(unittest.TestCase):
    def test_returns_structured_finding(self):
        model = StubModel()
        finding = NewsAnalyst(model).analyze("abc", [article()])
        self.assertEqual(finding.subject, "ABC")
        self.assertEqual(finding.evidence_refs, ("news:1",))
        self.assertEqual(model.last_input["articles"][0]["url"], "https://example.test/article")

    def test_rejects_hallucinated_evidence_reference(self):
        with self.assertRaises(ValueError):
            NewsAnalyst(StubModel(["news:not-supplied"])).analyze("ABC", [article()])

    def test_requires_sourced_input(self):
        with self.assertRaises(ValueError):
            NewsAnalyst(StubModel()).analyze("ABC", [])


if __name__ == "__main__":
    unittest.main()
