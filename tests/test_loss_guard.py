import unittest
from datetime import datetime, timedelta, timezone

from memory import (
    DeterministicLossGuard,
    InsightId,
    LossGuardLimits,
    LossGuardReason,
    PortfolioSnapshot,
    Position,
    TradeCandidate,
    TradeCandidateId,
    TradeCandidateStatus,
    TradeSide,
)


NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def candidate(side=TradeSide.LONG):
    return TradeCandidate(
        id=TradeCandidateId("trade:loss-check"),
        instrument="ABC",
        direction=side,
        thesis_refs=(InsightId("insight:1"),),
        horizon="30 days",
        confidence=0.7,
        entry_conditions=("approved thesis",),
        exit_conditions=("risk veto",),
        proposed_size=0.1,
        knowledge_time=NOW,
        status=TradeCandidateStatus.PROPOSED,
    )


def portfolio(*, equity=1000, position=None):
    return PortfolioSnapshot(
        cash=equity if position is None else 900,
        positions=() if position is None else (position,),
        realized_pnl=0,
        unrealized_pnl=0 if position is None else position.unrealized_pnl,
        fees=0,
        slippage=0,
        equity=equity,
        knowledge_time=NOW,
    )


class DeterministicLossGuardTests(unittest.TestCase):
    def test_blocks_averaging_down_even_below_stop_loss(self):
        position = Position("ABC", 10, 100, 96, -40)
        decision = DeterministicLossGuard().evaluate(
            candidate(), portfolio(equity=960, position=position),
            peak_equity=1000, evaluated_at=NOW,
        )
        self.assertFalse(decision.allowed)
        self.assertFalse(decision.must_exit)
        self.assertIn(LossGuardReason.AVERAGING_DOWN, decision.reasons)

    def test_position_stop_loss_is_a_hard_exit(self):
        position = Position("ABC", 10, 100, 90, -100)
        decision = DeterministicLossGuard().evaluate(
            candidate(), portfolio(equity=900, position=position),
            peak_equity=1000, evaluated_at=NOW,
        )
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.must_exit)
        self.assertIn(LossGuardReason.POSITION_STOP_LOSS, decision.reasons)
        exits = DeterministicLossGuard().required_exits(
            portfolio(equity=900, position=position), peak_equity=1000
        )
        self.assertIn(LossGuardReason.POSITION_STOP_LOSS, exits["ABC"])

    def test_portfolio_drawdown_is_a_hard_veto(self):
        decision = DeterministicLossGuard().evaluate(
            candidate(), portfolio(equity=899),
            peak_equity=1000, evaluated_at=NOW,
        )
        self.assertFalse(decision.allowed)
        self.assertIn(LossGuardReason.PORTFOLIO_DRAWDOWN, decision.reasons)

    def test_realized_loss_blocks_reentry_during_cooldown(self):
        guard = DeterministicLossGuard()
        guard.record_closed_trade("ABC", -25, closed_at=NOW)
        blocked = guard.evaluate(
            candidate(), portfolio(), peak_equity=1000,
            evaluated_at=NOW + timedelta(days=1),
        )
        allowed = guard.evaluate(
            candidate(), portfolio(), peak_equity=1000,
            evaluated_at=NOW + timedelta(days=5),
        )
        self.assertIn(LossGuardReason.LOSS_COOLDOWN, blocked.reasons)
        self.assertTrue(allowed.allowed)

    def test_profitable_close_resets_loss_streak(self):
        guard = DeterministicLossGuard()
        guard.record_closed_trade("ABC", -10, closed_at=NOW)
        guard.record_closed_trade("ABC", 5, closed_at=NOW + timedelta(days=1))
        self.assertEqual(guard.consecutive_losses("ABC"), 0)
        self.assertTrue(guard.evaluate(
            candidate(), portfolio(), peak_equity=1000,
            evaluated_at=NOW + timedelta(days=2),
        ).allowed)

    def test_opposite_direction_is_not_averaging_down(self):
        position = Position("ABC", 10, 100, 96, -40)
        decision = DeterministicLossGuard().evaluate(
            candidate(TradeSide.SHORT), portfolio(equity=960, position=position),
            peak_equity=1000, evaluated_at=NOW,
        )
        self.assertNotIn(LossGuardReason.AVERAGING_DOWN, decision.reasons)

    def test_limits_and_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            LossGuardLimits(max_position_loss=0)
        with self.assertRaises(ValueError):
            LossGuardLimits(loss_cooldown=timedelta(0))
        with self.assertRaises(ValueError):
            DeterministicLossGuard().record_closed_trade(
                "ABC", float("nan"), closed_at=NOW
            )


if __name__ == "__main__":
    unittest.main()
