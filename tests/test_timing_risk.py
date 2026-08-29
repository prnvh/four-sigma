import unittest
from datetime import datetime, timedelta, timezone

from memory import (
    MarketTape,
    PricePrint,
    TimingAction,
    TradeCandidate,
    TradeCandidateId,
    TradeCandidateStatus,
    TradeSide,
    apply_trade_risk,
    packet_news,
    pass_through_timing,
    recent_as_of,
    tape_facts,
    tape_reaction,
)
from memory.types import InsightId


NOW = datetime(2026, 7, 28, 13, 45, tzinfo=timezone.utc)


def candidate(*, direction: TradeSide = TradeSide.LONG, size: float = 1.0) -> TradeCandidate:
    return TradeCandidate(
        id=TradeCandidateId("tc:1"),
        instrument="AAPL",
        direction=direction,
        thesis_refs=(InsightId("insight:1"),),
        horizon="from_approved_insights",
        confidence=0.7,
        entry_conditions=("majority of approved insights are bullish",),
        exit_conditions=("cited bullish insights expire or are superseded",),
        proposed_size=size,
        knowledge_time=NOW,
        status=TradeCandidateStatus.PROPOSED,
    )


class TapeFactsTests(unittest.TestCase):
    def test_reports_return_and_distance_from_high(self) -> None:
        tape = MarketTape(
            (
                PricePrint(symbol="AAPL", price=100, knowledge_time=NOW - timedelta(days=10)),
                PricePrint(symbol="AAPL", price=110, knowledge_time=NOW - timedelta(days=1)),
                PricePrint(symbol="AAPL", price=105, knowledge_time=NOW),
            )
        )
        facts = tape_facts(tape, "AAPL", NOW)
        self.assertEqual(facts.last_price, 105)
        self.assertEqual(facts.window_start_price, 100)
        self.assertEqual(facts.high_20d, 110)
        self.assertAlmostEqual(facts.return_20d, 0.05)
        self.assertAlmostEqual(facts.distance_from_high, 5 / 110)

    def test_reaction_uses_this_name_typical_daily_move(self) -> None:
        prints = []
        price = 100.0
        for day in range(8):
            prints.append(
                PricePrint(
                    symbol="AAPL",
                    price=price,
                    knowledge_time=NOW - timedelta(days=8 - day),
                )
            )
            price += 1
        quiet = MarketTape(
            (*prints, PricePrint(symbol="AAPL", price=price + 0.2, knowledge_time=NOW))
        )
        self.assertFalse(tape_reaction(quiet, "AAPL", NOW).triggered)
        shock = MarketTape(
            (*prints, PricePrint(symbol="AAPL", price=price + 20, knowledge_time=NOW))
        )
        reaction = tape_reaction(shock, "AAPL", NOW)
        self.assertTrue(reaction.triggered)
        self.assertGreater(abs(reaction.move or 0), 0)

    def test_packet_keeps_two_day_news_and_falls_back_to_newest(self) -> None:
        from memory import Evidence

        fresh = Evidence(
            ref="news:fresh",
            source="reuters.com",
            url="https://example.test/fresh",
            published_at=NOW,
            title="Fresh",
            symbols=("AAPL",),
            knowledge_time=NOW - timedelta(days=1),
        )
        stale = Evidence(
            ref="news:stale",
            source="reuters.com",
            url="https://example.test/stale",
            published_at=NOW - timedelta(days=10),
            title="Stale",
            symbols=("AAPL",),
            knowledge_time=NOW - timedelta(days=10),
        )
        self.assertEqual(
            [item.ref for item in recent_as_of((fresh, stale), NOW)],
            ["news:fresh"],
        )
        self.assertEqual(
            [item.ref for item in packet_news((stale,), NOW)],
            ["news:stale"],
        )

    def test_pass_through_allows_without_a_risk_agent(self) -> None:
        tape = MarketTape(
            (
                PricePrint(symbol="AAPL", price=100, knowledge_time=NOW - timedelta(days=1)),
                PricePrint(symbol="AAPL", price=101, knowledge_time=NOW),
            )
        )
        decision = pass_through_timing(candidate(), tape=tape, now=NOW, max_position_pct=1.0)
        self.assertEqual(decision.action, TimingAction.ALLOW)
        self.assertEqual(decision.candidate.status, TradeCandidateStatus.APPROVED)
        self.assertEqual(decision.candidate.proposed_size, 1.0)

    def test_apply_cannot_raise_proposed_size(self) -> None:
        facts = tape_facts(
            MarketTape(
                (
                    PricePrint(symbol="AAPL", price=100, knowledge_time=NOW - timedelta(days=1)),
                    PricePrint(symbol="AAPL", price=101, knowledge_time=NOW),
                )
            ),
            "AAPL",
            NOW,
        )
        decision = apply_trade_risk(
            candidate(size=0.4),
            action=TimingAction.ALLOW,
            size=1.0,
            facts=facts,
            reasons=("agent asked for more",),
            max_position_pct=1.0,
        )
        self.assertEqual(decision.candidate.proposed_size, 0.4)


if __name__ == "__main__":
    unittest.main()
