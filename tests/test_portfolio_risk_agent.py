import unittest
from datetime import datetime, timezone

from agents import (
    PortfolioRiskAgent,
    PortfolioRiskAgentContext,
    PortfolioRiskRecommendation,
)
from memory import (
    Action,
    CAPABILITIES,
    Direction,
    InsightId,
    PortfolioRiskCalculator,
    PortfolioRiskInput,
    PortfolioSnapshot,
    Position,
    PositionRiskDecision,
    PromotedInsight,
    RiskAnalysis,
    RiskCheckResult,
    RiskReason,
    RiskReasonCode,
    TradeCandidate,
    TradeCandidateId,
    TradeSide,
)


NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)


class StubModel:
    def __init__(self, output):
        self.output = output
        self.last_input = None

    def generate_json(self, *, instructions, input_data, schema):
        self.last_input = input_data
        return self.output


def candidate():
    return TradeCandidate(
        id=TradeCandidateId("trade:abc"), instrument="ABC", direction=TradeSide.LONG,
        thesis_refs=(InsightId("insight:1"),), horizon="30 days", confidence=0.7,
        entry_conditions=("approved",), exit_conditions=("invalidated",),
        proposed_size=0.10, knowledge_time=NOW,
    )


def insight(ref="insight:1"):
    return PromotedInsight(
        ref=ref, symbol="ABC", claim="Synthetic approved thesis",
        direction=Direction.BULLISH, confidence=0.7, evidence_refs=("source:1",),
        knowledge_time=NOW,
    )


def comparison():
    state = PortfolioSnapshot(
        cash=100_000, positions=(), realized_pnl=0, unrealized_pnl=0,
        fees=0, slippage=0, equity=100_000, knowledge_time=NOW,
    )
    risk_input = PortfolioRiskInput(
        portfolio=state, sectors={"ABC": "Tech"},
        factor_loadings={"ABC": {"market": 1.0}},
        annualized_volatility={"ABC": 0.2}, correlations={}, current_drawdown=0.02,
    )
    return state, PortfolioRiskCalculator().compare(risk_input, candidate())


def decision(result=RiskCheckResult.PASS, approved=0.10):
    reasons = () if result is RiskCheckResult.PASS else (
        RiskReason(RiskReasonCode.MAX_POSITION, 0.10, approved),
    )
    return PositionRiskDecision(result, 0.10, approved, reasons)


def advisory():
    return RiskAnalysis(
        symbol="ABC",
        horizon_days=30,
        overall_risk_score=40,
        success_probability_pct=45,
        neutral_probability_pct=35,
        failure_probability_pct=20,
        risk_factors=(),
        hidden_assumptions=(),
        second_order_effects=(),
        success_conditions=(),
        failure_conditions=(),
        coverage_gaps=(),
        evidence_refs=("company:1",),
    )


def context(result=RiskCheckResult.PASS, approved=0.10, risk_analysis=None):
    state, risk_comparison = comparison()
    return PortfolioRiskAgentContext(
        candidate(),
        state,
        decision(result, approved),
        risk_comparison,
        (insight(),),
        risk_analysis,
    )


def output(recommendation="approve", size=0.10, refs=None):
    return {
        "recommendation": recommendation,
        "recommended_size": size,
        "rationale": "Synthetic portfolio-level rationale",
        "risk_flags": ["Synthetic concentration flag"],
        "insight_refs": refs or ["insight:1"],
    }


class PortfolioRiskAgentTests(unittest.TestCase):
    def test_approves_only_when_deterministic_risk_passes(self):
        model = StubModel(output())
        result = PortfolioRiskAgent(model).analyze(context())
        self.assertEqual(result.final_recommendation, PortfolioRiskRecommendation.APPROVE)
        self.assertEqual(result.final_size, 0.10)
        self.assertNotIn("evidence_refs", model.last_input["selected_insight_summaries"][0])

    def test_deterministic_rejection_overrides_optimistic_ai(self):
        result = PortfolioRiskAgent(StubModel(output())).analyze(
            context(RiskCheckResult.REJECT, 0)
        )
        self.assertEqual(result.recommendation, PortfolioRiskRecommendation.APPROVE)
        self.assertEqual(result.final_recommendation, PortfolioRiskRecommendation.REJECT)
        self.assertEqual(result.final_size, 0)

    def test_deterministic_resize_caps_optimistic_ai(self):
        result = PortfolioRiskAgent(StubModel(output())).analyze(
            context(RiskCheckResult.RESIZE, 0.03)
        )
        self.assertEqual(result.final_recommendation, PortfolioRiskRecommendation.RESIZE)
        self.assertEqual(result.final_size, 0.03)

    def test_ai_can_be_more_conservative(self):
        resized = PortfolioRiskAgent(StubModel(output("resize", 0.02))).analyze(context())
        rejected = PortfolioRiskAgent(StubModel(output("reject", 0))).analyze(context())
        deferred = PortfolioRiskAgent(StubModel(output("defer", 0))).analyze(context())
        self.assertEqual(resized.final_size, 0.02)
        self.assertEqual(rejected.final_recommendation, PortfolioRiskRecommendation.REJECT)
        self.assertEqual(deferred.final_recommendation, PortfolioRiskRecommendation.DEFER)

    def test_unknown_insight_or_oversize_fails_safely(self):
        with self.assertRaises(ValueError):
            PortfolioRiskAgent(StubModel(output(refs=["invented:1"]))).analyze(context())
        with self.assertRaises(ValueError):
            PortfolioRiskAgent(StubModel(output("resize", 0.20))).analyze(context())

    def test_drops_unknown_insight_refs_when_one_binds(self):
        result = PortfolioRiskAgent(
            StubModel(output(refs=["invented:1", "insight:1"]))
        ).analyze(context())
        self.assertEqual(result.insight_refs, ("insight:1",))
        self.assertEqual(result.final_recommendation, PortfolioRiskRecommendation.APPROVE)

    def test_accepts_nested_insight_and_risk_evidence_refs(self):
        nested = PortfolioRiskAgent(
            StubModel(output(refs=["source:1"]))
        ).analyze(context())
        self.assertEqual(nested.insight_refs, ("source:1",))
        from_risk = PortfolioRiskAgent(
            StubModel(output(refs=["company:1"]))
        ).analyze(context(risk_analysis=advisory()))
        self.assertEqual(from_risk.insight_refs, ("company:1",))

    def test_context_rejects_future_or_wrong_instrument_insights(self):
        state, risk_comparison = comparison()
        wrong = PromotedInsight(
            ref="wrong", symbol="XYZ", claim="Wrong symbol", direction=Direction.NEUTRAL,
            confidence=0.5, evidence_refs=("source",), knowledge_time=NOW,
        )
        with self.assertRaises(ValueError):
            PortfolioRiskAgentContext(
                candidate(), state, decision(), risk_comparison, (wrong,)
            )

    def test_context_requires_the_trade_causal_insights(self):
        state, risk_comparison = comparison()
        with self.assertRaises(ValueError):
            PortfolioRiskAgentContext(
                candidate(), state, decision(), risk_comparison, (insight("other"),)
            )

    def test_agent_cannot_execute_trades(self):
        self.assertFalse(
            CAPABILITIES.authorize("portfolio_risk", Action.EXECUTE, "trades", "order")
        )


if __name__ == "__main__":
    unittest.main()
