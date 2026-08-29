import unittest
from datetime import datetime, timezone

from memory import (
    AuditEventId,
    AuditEventType,
    AuditLedger,
    DeterministicPositionRiskEngine,
    InsightId,
    PortfolioSnapshot,
    Position,
    PositionRiskInput,
    PositionRiskLimits,
    RiskCheckResult,
    RiskReasonCode,
    RunId,
    TradeCandidate,
    TradeCandidateId,
    TradeSide,
)


NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def candidate(*, side=TradeSide.LONG, size=0.05):
    return TradeCandidate(
        id=TradeCandidateId(f"trade:{side.value}:{size}"),
        instrument="ABC",
        direction=side,
        thesis_refs=(InsightId("insight:1"),),
        horizon="30 days",
        confidence=0.7,
        entry_conditions=("approved thesis",),
        exit_conditions=("thesis invalidated",),
        proposed_size=size,
        knowledge_time=NOW,
    )


def portfolio(*positions, equity=100_000):
    return PortfolioSnapshot(
        cash=equity,
        positions=tuple(positions),
        realized_pnl=0,
        unrealized_pnl=0,
        fees=0,
        slippage=0,
        equity=equity,
        knowledge_time=NOW,
    )


def risk_input(*, trade=None, state=None, sectors=None, liquidity=None, volatility=None, drawdown=0.02):
    return PositionRiskInput(
        candidate=trade or candidate(),
        portfolio=state or portfolio(),
        sectors=sectors if sectors is not None else {"ABC": "Technology"},
        average_daily_dollar_volume=liquidity if liquidity is not None else {"ABC": 1_000_000},
        annualized_volatility=volatility if volatility is not None else {"ABC": 0.20},
        current_drawdown=drawdown,
    )


class PositionRiskEngineTests(unittest.TestCase):
    def evaluate(self, inputs, limits=None, ref="risk:1"):
        ledger = AuditLedger()
        engine = DeterministicPositionRiskEngine(limits or PositionRiskLimits(), ledger)
        decision = engine.evaluate(
            inputs,
            audit_event_id=AuditEventId(ref),
            run_id=RunId(f"run:{ref}"),
        )
        return decision, ledger

    def test_safe_trade_passes_and_is_audited_without_changing_portfolio(self):
        inputs = risk_input()
        before = inputs.portfolio
        decision, ledger = self.evaluate(inputs)
        self.assertEqual(decision.result, RiskCheckResult.PASS)
        self.assertEqual(decision.approved_size, 0.05)
        self.assertEqual(decision.reasons, ())
        self.assertEqual(inputs.portfolio, before)
        event = ledger.snapshot()[0]
        self.assertEqual(event.event_type, AuditEventType.RISK_CHECK_RUN)
        self.assertEqual(event.details["result"], "PASS")
        self.assertEqual(event.subject_id, inputs.candidate.id)

    def test_each_sizing_rule_can_resize_the_trade(self):
        cases = {
            "max_position_pct": RiskReasonCode.MAX_POSITION,
            "max_gross_exposure": RiskReasonCode.GROSS_EXPOSURE,
            "max_net_exposure": RiskReasonCode.NET_EXPOSURE,
            "max_sector_concentration": RiskReasonCode.SECTOR_CONCENTRATION,
            "max_single_name_concentration": RiskReasonCode.SINGLE_NAME_CONCENTRATION,
            "max_daily_liquidity_pct": RiskReasonCode.DAILY_LIQUIDITY,
        }
        for field, code in cases.items():
            with self.subTest(rule=field):
                limits = PositionRiskLimits(**{field: 0.03})
                inputs = risk_input(
                    liquidity={"ABC": 100_000}
                    if field == "max_daily_liquidity_pct"
                    else None
                )
                decision, _ = self.evaluate(inputs, limits, f"risk:{field}")
                self.assertEqual(decision.result, RiskCheckResult.RESIZE)
                self.assertAlmostEqual(decision.approved_size, 0.03)
                self.assertIn(code, {reason.code for reason in decision.reasons})

    def test_sector_check_includes_existing_positions(self):
        state = portfolio(
            Position("XYZ", quantity=100, average_entry=100, market_price=100)
        )
        limits = PositionRiskLimits(
            max_position_pct=0.50,
            max_gross_exposure=1,
            max_net_exposure=1,
            max_sector_concentration=0.20,
            max_single_name_concentration=0.50,
        )
        inputs = risk_input(
            trade=candidate(size=0.15),
            state=state,
            sectors={"ABC": "Technology", "XYZ": "Technology"},
        )
        decision, _ = self.evaluate(inputs, limits)
        self.assertEqual(decision.result, RiskCheckResult.RESIZE)
        self.assertAlmostEqual(decision.approved_size, 0.10)
        self.assertEqual(decision.reasons[0].code, RiskReasonCode.SECTOR_CONCENTRATION)

    def test_volatility_and_drawdown_are_hard_vetoes(self):
        decision, ledger = self.evaluate(
            risk_input(volatility={"ABC": 0.80}, drawdown=0.25)
        )
        self.assertEqual(decision.result, RiskCheckResult.REJECT)
        self.assertTrue(decision.vetoed)
        self.assertEqual(decision.approved_size, 0)
        self.assertEqual(
            {reason.code for reason in decision.reasons},
            {RiskReasonCode.VOLATILITY_LIMIT, RiskReasonCode.DRAWDOWN_CONSTRAINT},
        )
        self.assertEqual(ledger.snapshot()[0].details["result"], "REJECT")

    def test_missing_risk_data_fails_closed(self):
        decision, _ = self.evaluate(
            risk_input(sectors={}, liquidity={}, volatility={})
        )
        self.assertEqual(decision.result, RiskCheckResult.REJECT)
        self.assertEqual(
            {reason.code for reason in decision.reasons},
            {
                RiskReasonCode.MISSING_SECTOR,
                RiskReasonCode.MISSING_DAILY_LIQUIDITY,
                RiskReasonCode.MISSING_VOLATILITY,
            },
        )

    def test_same_inputs_always_produce_same_decision_for_long_and_short(self):
        for side in (TradeSide.LONG, TradeSide.SHORT):
            with self.subTest(side=side):
                inputs = risk_input(trade=candidate(side=side), drawdown=0.30)
                first, _ = self.evaluate(inputs, ref=f"risk:first:{side.value}")
                second, _ = self.evaluate(inputs, ref=f"risk:second:{side.value}")
                self.assertEqual(first, second)
                self.assertEqual(first.result, RiskCheckResult.REJECT)


if __name__ == "__main__":
    unittest.main()
