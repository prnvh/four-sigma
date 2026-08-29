import unittest
from datetime import datetime, timedelta, timezone

from agents import TradeRiskAnalyst
from memory import (
    Direction,
    Evidence,
    MarketTape,
    PricePrint,
    PromotedInsight,
    TimingAction,
    TradeCandidate,
    TradeCandidateId,
    TradeCandidateStatus,
    TradeSide,
)
from memory.types import InsightId


NOW = datetime(2026, 7, 28, 13, 45, tzinfo=timezone.utc)


class StubModel:
    def __init__(self, payload):
        self.payload = payload
        self.last_input = None

    def generate_json(self, *, instructions, input_data, schema):
        self.last_input = input_data
        if callable(self.payload):
            return self.payload(input_data)
        return dict(self.payload)


def insight() -> PromotedInsight:
    return PromotedInsight(
        ref="insight:1",
        symbol="AAPL",
        claim="Supplier demand is rising on new chip orders.",
        direction=Direction.BULLISH,
        confidence=0.7,
        evidence_refs=("news:1",),
        knowledge_time=NOW,
        valid_until=NOW + timedelta(days=3),
    )


def article() -> Evidence:
    return Evidence(
        ref="news:1",
        source="reuters.com",
        url="https://example.test/aapl",
        published_at=NOW,
        title="Apple supplier demand rises",
        summary="Orders increased at a key assembler.",
        symbols=("AAPL",),
        knowledge_time=NOW,
    )


def candidate() -> TradeCandidate:
    return TradeCandidate(
        id=TradeCandidateId("tc:1"),
        instrument="AAPL",
        direction=TradeSide.LONG,
        thesis_refs=(InsightId("insight:1"),),
        horizon="from_approved_insights",
        confidence=0.7,
        entry_conditions=("majority of approved insights are bullish",),
        exit_conditions=("cited bullish insights expire or are superseded",),
        proposed_size=1.0,
        knowledge_time=NOW,
        status=TradeCandidateStatus.PROPOSED,
    )


def tape() -> MarketTape:
    return MarketTape(
        (
            PricePrint(symbol="AAPL", price=100, knowledge_time=NOW - timedelta(days=10)),
            PricePrint(symbol="AAPL", price=104, knowledge_time=NOW),
        )
    )


def review(model, **overrides):
    values = {
        "tape": tape(),
        "insights": (insight(),),
        "articles": (article(),),
        "now": NOW,
        "existing_quantity": 0.0,
        "max_position_pct": 1.0,
    }
    values.update(overrides)
    return TradeRiskAnalyst(model).review(candidate(), **values)


class TradeRiskAnalystTests(unittest.TestCase):
    def test_allows_when_the_agent_sees_no_specific_problem(self) -> None:
        model = StubModel(
            {
                "action": "allow",
                "size": 1.0,
                "rationale": "Thesis is intact and tape is not stretched.",
                "evidence_refs": ["insight:1"],
            }
        )
        decision = review(model)
        self.assertEqual(decision.action, TimingAction.ALLOW)
        self.assertEqual(decision.candidate.status, TradeCandidateStatus.APPROVED)
        self.assertEqual(decision.candidate.proposed_size, 1.0)
        self.assertIn("tape_facts", model.last_input)
        self.assertEqual(model.last_input["tape_facts"]["last_price"], 104)
        stale = Evidence(
            ref="news:old",
            source="reuters.com",
            url="https://example.test/old",
            published_at=NOW - timedelta(days=5),
            title="Old headline",
            summary="Stale article",
            symbols=("AAPL",),
            knowledge_time=NOW - timedelta(days=5),
        )
        stale_model = StubModel(
            {
                "action": "allow",
                "size": 1.0,
                "rationale": "Only recent articles are in the packet.",
                "evidence_refs": ["insight:1"],
            }
        )
        review(stale_model, articles=(stale,))
        self.assertEqual(stale_model.last_input["articles"], [])

    def test_defers_only_when_the_agent_says_so(self) -> None:
        decision = review(
            StubModel(
                {
                    "action": "defer",
                    "size": 1.0,
                    "rationale": "Entry is poorly timed versus the supplied tape facts.",
                    "evidence_refs": ["news:1"],
                }
            )
        )
        self.assertEqual(decision.action, TimingAction.DEFER)
        self.assertEqual(decision.candidate.status, TradeCandidateStatus.REJECTED)

    def test_cannot_increase_size_and_must_cite_supplied_refs(self) -> None:
        decision = review(
            StubModel(
                {
                    "action": "allow",
                    "size": 2.0,
                    "rationale": "Press it.",
                    "evidence_refs": ["insight:1"],
                }
            )
        )
        self.assertEqual(decision.action, TimingAction.ALLOW)
        self.assertEqual(decision.reasons[0].startswith("risk_pass_through:"), True)

    def test_pass_through_when_context_is_empty(self) -> None:
        decision = review(StubModel({"action": "reduce", "size": 0, "rationale": "x", "evidence_refs": ["insight:1"]}), insights=(), articles=())
        self.assertEqual(decision.action, TimingAction.ALLOW)
        self.assertEqual(decision.reasons, ("no_sourced_context",))


if __name__ == "__main__":
    unittest.main()
