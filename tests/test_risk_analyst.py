import unittest
from datetime import datetime, timedelta, timezone

from agents import (
    CompanyAnalysis,
    CompanyAnalysisRecord,
    Direction,
    MarketFeature,
    MarketFeatureType,
    OutcomeDefinition,
    RiskAnalyst,
    RiskCategory,
)
from memory import ContextGateway, ContextPermissionError, ResearchContextStore


NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


class StubModel:
    def __init__(self, override=None):
        self.override = override or {}
        self.last_input = None

    def generate_json(self, *, instructions, input_data, schema):
        self.last_input = input_data
        assessed = {RiskCategory.FINANCIAL.value, RiskCategory.MARKET.value}
        result = {
            "overall_risk_score": 55,
            "success_probability_pct": 35,
            "neutral_probability_pct": 40,
            "failure_probability_pct": 25,
            "risk_factors": [
                {
                    "category": "financial", "probability_pct": 30, "severity": 4,
                    "impact": "Synthetic financial impact", "evidence_refs": ["analysis:1"],
                    "mitigants": ["Synthetic financial mitigant"],
                },
                {
                    "category": "market", "probability_pct": 45, "severity": 3,
                    "impact": "Synthetic market impact", "evidence_refs": ["market:1"],
                    "mitigants": ["Synthetic market mitigant"],
                },
            ],
            "hidden_assumptions": ["Synthetic hidden assumption"],
            "second_order_effects": ["Synthetic second-order effect"],
            "success_conditions": ["Synthetic success condition"],
            "failure_conditions": ["Synthetic failure condition"],
            "coverage_gaps": [item.value for item in RiskCategory if item.value not in assessed],
            "evidence_refs": ["analysis:1", "market:1"],
        }
        result.update(self.override)
        return result


def company_analysis():
    return CompanyAnalysis(
        agent="company_analyst:v1",
        symbol="ABC", thesis="Synthetic thesis", fundamental_direction=Direction.NEUTRAL,
        fundamental_confidence=0.5, momentum_direction=Direction.BULLISH,
        momentum_score=0.25, momentum_confidence=0.4, momentum_horizon="one month",
        evidence_refs=("company:1",), strengths=("Synthetic strength",),
        weaknesses=("Synthetic weakness",), catalysts=("Synthetic catalyst",),
        momentum_drivers=("Synthetic driver",), momentum_risks=("Synthetic risk",),
        invalidation_conditions=("Synthetic invalidation",),
    )


def analysis_record(ref="analysis:1", symbol="ABC", known_at=NOW):
    analysis = company_analysis()
    if symbol != "ABC":
        analysis = CompanyAnalysis(
            agent=analysis.agent,
            symbol=symbol, thesis=analysis.thesis,
            fundamental_direction=analysis.fundamental_direction,
            fundamental_confidence=analysis.fundamental_confidence,
            momentum_direction=analysis.momentum_direction, momentum_score=analysis.momentum_score,
            momentum_confidence=analysis.momentum_confidence,
            momentum_horizon=analysis.momentum_horizon, evidence_refs=analysis.evidence_refs,
            strengths=analysis.strengths, weaknesses=analysis.weaknesses,
            catalysts=analysis.catalysts, momentum_drivers=analysis.momentum_drivers,
            momentum_risks=analysis.momentum_risks,
            invalidation_conditions=analysis.invalidation_conditions,
        )
    return CompanyAnalysisRecord(ref=ref, analysis=analysis, knowledge_time=known_at)


def feature(ref="market:1", symbol="ABC", known_at=NOW):
    return MarketFeature(
        ref=ref, symbol=symbol, source="Synthetic market source",
        url=f"https://example.test/{ref}", knowledge_time=known_at,
        feature=MarketFeatureType.RETURN_20D, value=0.05, unit="decimal_return",
    )


def context(*, analyses=(), features=(), at=NOW):
    store = ResearchContextStore()
    for item in analyses:
        store.append_company_analysis(item)
    for item in features:
        store.append_market_feature(item)
    return ContextGateway(store).for_risk_analyst(
        agent_id="risk_analyst", symbol="ABC", simulation_time=at,
        outcome=OutcomeDefinition(30, 10, -10),
    )


class RiskAnalystTests(unittest.TestCase):
    def test_returns_risk_scores_and_outcome_probabilities(self):
        model = StubModel()
        result = RiskAnalyst(model).analyze(
            context(analyses=(analysis_record(),), features=(feature(),))
        )
        self.assertEqual(result.success_probability_pct, 35)
        self.assertEqual(result.failure_probability_pct, 25)
        self.assertEqual(sum(x.probability_pct for x in result.risk_factors), 75)
        self.assertEqual(model.last_input["outcome_definition"]["horizon_days"], 30)

    def test_rejects_unpermissioned_input(self):
        with self.assertRaises(TypeError):
            RiskAnalyst(StubModel()).analyze([analysis_record()])

    def test_requires_market_features_for_probability_estimate(self):
        with self.assertRaises(ValueError):
            RiskAnalyst(StubModel()).analyze(context(analyses=(analysis_record(),)))

    def test_requires_company_evidence(self):
        with self.assertRaises(ValueError):
            RiskAnalyst(StubModel()).analyze(context(features=(feature(),)))

    def test_rejects_probabilities_not_totalling_100(self):
        with self.assertRaises(ValueError):
            RiskAnalyst(StubModel({"success_probability_pct": 50})).analyze(
                context(analyses=(analysis_record(),), features=(feature(),))
            )

    def test_rejects_unknown_evidence(self):
        with self.assertRaises(ValueError):
            RiskAnalyst(StubModel({"evidence_refs": ["invented:1"]})).analyze(
                context(analyses=(analysis_record(),), features=(feature(),))
            )

    def test_requires_every_category_assessed_or_declared_gap(self):
        with self.assertRaises(ValueError):
            RiskAnalyst(StubModel({"coverage_gaps": []})).analyze(
                context(analyses=(analysis_record(),), features=(feature(),))
            )

    def test_gateway_filters_future_and_other_symbols(self):
        selected = context(
            analyses=(
                analysis_record(),
                analysis_record("analysis:future", known_at=NOW + timedelta(days=1)),
                analysis_record("analysis:other", symbol="XYZ"),
            ),
            features=(
                feature(), feature("market:future", known_at=NOW + timedelta(days=1)),
                feature("market:other", symbol="XYZ"),
            ),
        )
        self.assertEqual([x.ref for x in selected.company_analyses], ["analysis:1"])
        self.assertEqual([x.ref for x in selected.market_features], ["market:1"])

    def test_context_excludes_portfolio_and_trade_authority(self):
        selected = context(analyses=(analysis_record(),), features=(feature(),))
        self.assertFalse(hasattr(selected, "portfolio"))
        self.assertFalse(hasattr(selected, "trade_candidate"))

    def test_gateway_checks_agent_permission(self):
        with self.assertRaises(ContextPermissionError):
            ContextGateway(ResearchContextStore()).for_risk_analyst(
                agent_id="company_analyst", symbol="ABC", simulation_time=NOW,
                outcome=OutcomeDefinition(30, 10, -10),
            )


if __name__ == "__main__":
    unittest.main()
