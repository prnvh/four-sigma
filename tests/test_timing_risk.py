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
    apply_tape_ceiling,
    apply_trend_ceiling,
    apply_trade_risk,
    hold_working_position,
    packet_news,
    pass_through_timing,
    recent_as_of,
    may_reenter_stopped_side,
    scaled_stop_pct,
    should_stop_loser,
    should_trail_winner,
    tape_facts,
    tape_opposes_side,
    tape_reaction,
    trend_opposes_side,
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

    def test_adverse_tape_cannot_be_allowed(self) -> None:
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
            price += 0.5
        tape = MarketTape(
            (*prints, PricePrint(symbol="AAPL", price=price * 0.9, knowledge_time=NOW))
        )
        reaction = tape_reaction(tape, "AAPL", NOW)
        self.assertTrue(reaction.triggered)
        self.assertTrue(tape_opposes_side(reaction, TradeSide.LONG))
        self.assertFalse(tape_opposes_side(reaction, TradeSide.SHORT))
        allowed = pass_through_timing(candidate(), tape=tape, now=NOW, max_position_pct=1.0)
        self.assertEqual(allowed.action, TimingAction.DEFER)
        self.assertIn("adverse_tape", allowed.reasons)
        facts = tape_facts(tape, "AAPL", NOW)
        forced = apply_tape_ceiling(
            apply_trade_risk(
                candidate(),
                action=TimingAction.ALLOW,
                size=1.0,
                facts=facts,
                reasons=("llm allow",),
                max_position_pct=1.0,
            ),
            reaction,
        )
        self.assertEqual(forced.action, TimingAction.DEFER)

    def test_countertrend_thesis_waits_for_price_confirmation(self) -> None:
        falling = MarketTape(
            (
                PricePrint(
                    symbol="AAPL",
                    price=110,
                    knowledge_time=NOW - timedelta(days=10),
                ),
                PricePrint(symbol="AAPL", price=100, knowledge_time=NOW),
            )
        )
        facts = tape_facts(falling, "AAPL", NOW)
        self.assertTrue(trend_opposes_side(facts, TradeSide.LONG))
        self.assertFalse(trend_opposes_side(facts, TradeSide.SHORT))
        allowed = apply_trade_risk(
            candidate(),
            action=TimingAction.ALLOW,
            size=1.0,
            facts=facts,
            reasons=("agent allow",),
            max_position_pct=1.0,
        )
        deferred = apply_trend_ceiling(allowed)
        self.assertEqual(deferred.action, TimingAction.DEFER)
        self.assertIn("countertrend_20d", deferred.reasons)

    def test_stop_fires_on_a_losing_long_or_short(self) -> None:
        self.assertTrue(
            should_stop_loser(
                quantity=10,
                average_entry=100,
                market_price=96,
                stop_loss_pct=0.03,
            )
        )
        self.assertFalse(
            should_stop_loser(
                quantity=10,
                average_entry=100,
                market_price=98,
                stop_loss_pct=0.03,
            )
        )
        self.assertTrue(
            should_stop_loser(
                quantity=-10,
                average_entry=100,
                market_price=104,
                stop_loss_pct=0.03,
            )
        )
        self.assertFalse(
            should_stop_loser(
                quantity=-10,
                average_entry=100,
                market_price=101,
                stop_loss_pct=0.03,
            )
        )
        self.assertFalse(
            should_stop_loser(
                quantity=10,
                average_entry=100,
                market_price=90,
                stop_loss_pct=0,
            )
        )

    def test_stop_widens_with_volatility_and_uses_the_ceiling_when_vol_is_missing(self) -> None:
        quiet = scaled_stop_pct(0.20, floor_pct=0.03, multiple=1.5, vol_ceiling=0.60)
        loud = scaled_stop_pct(0.60, floor_pct=0.03, multiple=1.5, vol_ceiling=0.60)
        missing = scaled_stop_pct(None, floor_pct=0.03, multiple=1.5, vol_ceiling=0.60)
        self.assertAlmostEqual(quiet, 0.03)
        self.assertGreater(loud, 0.05)
        self.assertAlmostEqual(loud, missing)
        self.assertGreater(loud, 0.60 / (252 ** 0.5))

    def test_trailing_stop_activates_only_after_a_winner_retraces(self) -> None:
        self.assertFalse(
            should_trail_winner(
                quantity=10,
                average_entry=100,
                market_price=102,
                favorable_price=103,
                activation_pct=0.04,
                trailing_stop_pct=0.02,
            )
        )
        self.assertTrue(
            should_trail_winner(
                quantity=10,
                average_entry=100,
                market_price=103,
                favorable_price=106,
                activation_pct=0.04,
                trailing_stop_pct=0.02,
            )
        )
        self.assertTrue(
            should_trail_winner(
                quantity=-10,
                average_entry=100,
                market_price=96,
                favorable_price=93,
                activation_pct=0.04,
                trailing_stop_pct=0.02,
            )
        )

    def test_stopped_side_reenters_only_after_cooldown(self) -> None:
        stopped_at = NOW - timedelta(hours=2)
        self.assertFalse(
            may_reenter_stopped_side(
                stopped_at=stopped_at,
                now=NOW,
                newest_causal_evidence=NOW,
                cooldown=timedelta(days=1),
            )
        )
        self.assertTrue(
            may_reenter_stopped_side(
                stopped_at=stopped_at,
                now=NOW + timedelta(days=1),
                newest_causal_evidence=stopped_at,
                cooldown=timedelta(days=1),
            )
        )

    def test_hold_keeps_a_winner_and_still_cuts_a_loser(self) -> None:
        self.assertTrue(
            hold_working_position(
                existing_quantity=10,
                proposed=TradeSide.LONG,
                action=TimingAction.REDUCE,
                pnl=0.04,
            )
        )
        self.assertTrue(
            hold_working_position(
                existing_quantity=10,
                proposed=TradeSide.SHORT,
                action=TimingAction.ALLOW,
                pnl=0.04,
            )
        )
        self.assertFalse(
            hold_working_position(
                existing_quantity=10,
                proposed=TradeSide.LONG,
                action=TimingAction.ALLOW,
                pnl=0.04,
            )
        )
        self.assertFalse(
            hold_working_position(
                existing_quantity=10,
                proposed=TradeSide.LONG,
                action=TimingAction.REDUCE,
                pnl=-0.04,
            )
        )

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
