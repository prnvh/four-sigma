import unittest
from datetime import datetime, timedelta, timezone

from agents import (
    CompanyAnalyst,
    CompanyRecord,
    CompanyRecordType,
    Direction,
    PromotedInsight,
)
from memory import ContextGateway, ContextPermissionError, SharedMemory


UTC = timezone.utc
NOW = datetime(2026, 2, 1, tzinfo=UTC)


class StubModel:
    def __init__(self, refs=None, output_override=None):
        self.refs = refs or ["company:1"]
        self.output_override = output_override
        self.last_input = None

    def generate_json(self, *, instructions, input_data, schema):
        self.last_input = input_data
        result = {
            "thesis": "Synthetic balanced company thesis",
            "direction": "neutral",
            "confidence": 0.5,
            "horizon": "test horizon",
            "evidence_refs": self.refs,
            "strengths": ["Synthetic strength"],
            "weaknesses": ["Synthetic weakness"],
            "catalysts": ["Synthetic catalyst"],
            "invalidation_conditions": ["Synthetic invalidation condition"],
        }
        if self.output_override:
            result.update(self.output_override)
        return result


def record(ref="company:1", symbol="ABC", known_at=NOW):
    return CompanyRecord(
        ref=ref, symbol=symbol, source="Synthetic filing source",
        url=f"https://example.test/{ref}", knowledge_time=known_at,
        record_type=CompanyRecordType.REGULATORY_FILING,
        label="Synthetic filing fact", value="Synthetic value", period_end="2025-12-31",
    )


def insight(ref="insight:1", symbol="ABC", known_at=NOW, valid_until=None):
    return PromotedInsight(
        ref=ref, symbol=symbol, claim="Synthetic promoted claim", direction=Direction.NEUTRAL,
        confidence=0.5, evidence_refs=("news:1",), knowledge_time=known_at,
        valid_until=valid_until,
    )


def context(*, records=(), insights=(), symbol="ABC", at=NOW):
    shared = SharedMemory()
    for item in records:
        shared.append_company_record(item)
    for item in insights:
        shared.append_promoted_insight(item)
    return ContextGateway(shared).for_company_analyst(
        agent_id="company_analyst", symbol=symbol, simulation_time=at
    )


class CompanyAnalystTests(unittest.TestCase):
    def test_returns_structured_analysis_from_gateway_context(self):
        model = StubModel()
        analysis = CompanyAnalyst(model).analyze(context(records=(record(),)))
        self.assertEqual(analysis.symbol, "ABC")
        self.assertEqual(analysis.evidence_refs, ("company:1",))
        self.assertEqual(model.last_input["company_records"][0]["label"], "Synthetic filing fact")

    def test_accepts_promoted_insight_reference(self):
        analysis = CompanyAnalyst(StubModel(["insight:1"])).analyze(
            context(insights=(insight(),))
        )
        self.assertEqual(analysis.evidence_refs, ("insight:1",))

    def test_rejects_direct_unpermissioned_data(self):
        with self.assertRaises(TypeError):
            CompanyAnalyst(StubModel()).analyze([record()])

    def test_rejects_unknown_citation(self):
        with self.assertRaises(ValueError):
            CompanyAnalyst(StubModel(["company:invented"])).analyze(
                context(records=(record(),))
            )

    def test_rejects_malformed_output(self):
        with self.assertRaises(ValueError):
            CompanyAnalyst(StubModel(output_override={"confidence": "high"})).analyze(
                context(records=(record(),))
            )

    def test_requires_evidence_or_promoted_insight(self):
        with self.assertRaises(ValueError):
            CompanyAnalyst(StubModel()).analyze(context())

    def test_gateway_excludes_future_other_company_and_expired_insights(self):
        selected = context(
            records=(
                record("company:visible"),
                record("company:future", known_at=NOW + timedelta(days=1)),
                record("company:other", symbol="XYZ"),
            ),
            insights=(
                insight("insight:visible"),
                insight("insight:expired", valid_until=NOW - timedelta(seconds=1), known_at=NOW - timedelta(days=1)),
            ),
        )
        self.assertEqual([item.ref for item in selected.records], ["company:visible"])
        self.assertEqual([item.ref for item in selected.promoted_insights], ["insight:visible"])

    def test_context_cannot_contain_portfolio_or_raw_news(self):
        selected = context(records=(record(),))
        self.assertEqual(
            set(selected.__slots__), {"symbol", "simulation_time", "records", "promoted_insights"}
        )
        self.assertFalse(hasattr(selected, "portfolio"))
        self.assertFalse(hasattr(selected, "articles"))

    def test_gateway_checks_agent_permission(self):
        with self.assertRaises(ContextPermissionError):
            ContextGateway(SharedMemory()).for_company_analyst(
                agent_id="news_analyst", symbol="ABC", simulation_time=NOW
            )

    def test_duplicate_company_record_is_rejected(self):
        shared = SharedMemory()
        shared.append_company_record(record())
        with self.assertRaises(ValueError):
            shared.append_company_record(record())


if __name__ == "__main__":
    unittest.main()
