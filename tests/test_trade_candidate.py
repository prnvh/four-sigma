import unittest
from datetime import datetime, timezone

from memory import InsightId, TradeCandidate, TradeCandidateId, TradeCandidateStatus, TradeSide


NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def candidate(**overrides: object) -> TradeCandidate:
    values = {
        "id": TradeCandidateId("tc:1"),
        "instrument": "abc",
        "direction": TradeSide.LONG,
        "thesis_refs": (InsightId("insight:1"),),
        "horizon": "one month",
        "confidence": 0.6,
        "entry_conditions": ("close above 10",),
        "exit_conditions": ("invalidated thesis",),
        "proposed_size": 0.02,
        "knowledge_time": NOW,
    }
    values.update(overrides)
    return TradeCandidate(**values)  # type: ignore[arg-type]


class TradeCandidateTests(unittest.TestCase):
    def test_separates_action_from_thesis_and_requires_insight_refs(self) -> None:
        trade = candidate()
        self.assertEqual(trade.instrument, "ABC")
        self.assertEqual(trade.direction, TradeSide.LONG)
        self.assertEqual(trade.thesis_refs, (InsightId("insight:1"),))
        self.assertEqual(trade.status, TradeCandidateStatus.PROPOSED)
        self.assertNotEqual(trade.direction.value, "bullish")

    def test_no_trade_is_allowed_with_zero_size(self) -> None:
        trade = candidate(direction=TradeSide.NO_TRADE, proposed_size=0, entry_conditions=())
        self.assertEqual(trade.proposed_size, 0.0)

    def test_rejects_candidate_without_thesis_refs(self) -> None:
        with self.assertRaises(ValueError):
            candidate(thesis_refs=())

    def test_rejects_string_thesis_refs(self) -> None:
        with self.assertRaises(TypeError):
            candidate(thesis_refs=("insight:1",))  # type: ignore[arg-type]

    def test_rejects_long_with_zero_size_and_no_trade_with_size(self) -> None:
        with self.assertRaises(ValueError):
            candidate(proposed_size=0)
        with self.assertRaises(ValueError):
            candidate(direction=TradeSide.NO_TRADE, proposed_size=0.1)

    def test_rejects_naive_knowledge_time(self) -> None:
        with self.assertRaises(ValueError):
            candidate(knowledge_time=datetime(2026, 3, 1))


if __name__ == "__main__":
    unittest.main()
