import unittest
from datetime import datetime, timedelta, timezone

from agents import (
    CompanyAnalyst,
    CompanyEntityRecord,
    CompanyRecord,
    CompanyRecordType,
    Direction,
    Evidence,
    MarketFeature,
    MarketFeatureType,
    PromotedInsight,
)
from memory import (
    Action,
    CAPABILITIES,
    ContextGateway,
    ContextPermissionError,
    InsightId,
    ResearchContextStore,
)


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
            "company_thesis": "Synthetic balanced company thesis",
            "bull_case": "Synthetic bull case",
            "bear_case": "Synthetic bear case",
            "catalysts": ["Synthetic catalyst"],
            "risks": ["Synthetic risk"],
            "confidence": 0.5,
            "time_horizon": "one to three months",
            "evidence_refs": self.refs,
            "supports": [],
            "contradicts": [],
            "supersedes": [],
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


def company(fundamental_references=("company:1",), known_at=NOW):
    return CompanyEntityRecord(
        ticker="ABC",
        exchange="NASDAQ",
        sector="Technology",
        industry="Software",
        identifiers={"cik": "0000000001", "isin": "US0000000001"},
        fundamental_references=fundamental_references,
        knowledge_time=known_at,
    )


def event(ref="event:1", symbol="ABC", known_at=NOW):
    return Evidence(
        ref=ref,
        source="Synthetic news source",
        url=f"https://example.test/{ref}",
        published_at=known_at,
        title="Synthetic recent event",
        symbols=(symbol,),
        knowledge_time=known_at,
    )


def feature(ref="market:1", symbol="ABC", known_at=NOW):
    return MarketFeature(
        ref=ref, symbol=symbol, source="Synthetic market source",
        url=f"https://example.test/{ref}", knowledge_time=known_at,
        feature=MarketFeatureType.RETURN_20D, value=0.05, unit="decimal_return",
    )


def context(
    *, company_record=None, records=(), insights=(), events=(), features=(),
    symbol="ABC", at=NOW
):
    shared = ResearchContextStore()
    for item in records:
        shared.append_company_record(item)
    if company_record is not None:
        shared.append_company_entity(company_record)
    for item in insights:
        shared.append_promoted_insight(item)
    for item in features:
        shared.append_market_feature(item)
    for item in events:
        shared.append_news(item)
    return ContextGateway(shared).for_company_analyst(
        agent_id="company_analyst", symbol=symbol, simulation_time=at
    )


class CompanyAnalystTests(unittest.TestCase):
    def test_returns_structured_analysis_from_gateway_context(self):
        model = StubModel()
        analysis = CompanyAnalyst(model).analyze(context(records=(record(),)))
        self.assertEqual(analysis.symbol, "ABC")
        self.assertEqual(analysis.company_thesis, "Synthetic balanced company thesis")
        self.assertEqual(analysis.evidence_refs, ("company:1",))
        self.assertEqual(analysis.bull_case, "Synthetic bull case")
        self.assertEqual(model.last_input["company_facts"][0]["label"], "Synthetic filing fact")

    def test_accepts_promoted_insight_reference(self):
        analysis = CompanyAnalyst(StubModel(["insight:1"])).analyze(
            context(insights=(insight(),))
        )
        self.assertEqual(analysis.evidence_refs, ("insight:1",))

    def test_accepts_market_feature_for_momentum(self):
        analysis = CompanyAnalyst(StubModel(["market:1"])).analyze(
            context(features=(feature(),))
        )
        self.assertEqual(analysis.time_horizon, "one to three months")

    def test_model_packet_omits_news_older_than_two_days(self):
        model = StubModel(["company:1"])
        CompanyAnalyst(model).analyze(
            context(
                records=(record(),),
                events=(
                    event("event:old", known_at=NOW - timedelta(days=5)),
                    event("event:fresh"),
                ),
            )
        )
        refs = [item["ref"] for item in model.last_input["recent_events"]]
        self.assertEqual(refs, ["event:fresh"])
        self.assertNotIn("historical_context", model.last_input)

    def test_reads_company_entity_and_recent_events(self):
        selected = context(
            company_record=company(),
            records=(record(),),
            events=(event(),),
        )
        analysis = CompanyAnalyst(StubModel(["event:1"])).analyze(selected)
        self.assertEqual(selected.company.ticker, "ABC")
        self.assertEqual(selected.recent_events[0].ref, "event:1")
        self.assertEqual(analysis.evidence_refs, ("event:1",))

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

    def test_rejects_position_sizing_output_and_has_no_trade_capability(self):
        with self.assertRaises(ValueError):
            CompanyAnalyst(
                StubModel(output_override={"position_size": 0.1})
            ).analyze(context(records=(record(),)))
        self.assertFalse(
            CAPABILITIES.authorize(
                "company_analyst",
                Action.PROPOSE_SHARED_WRITE,
                "trade_candidates",
                "size",
            )
        )

    def test_analysis_becomes_related_insight_without_creating_a_trade(self):
        analysis = CompanyAnalyst(
            StubModel(
                ["company:1"],
                output_override={"contradicts": ["insight:1"]},
            )
        ).analyze(context(records=(record(),), insights=(insight(),)))
        revision = analysis.to_insight_revision(InsightId("company-thesis"))
        self.assertEqual(revision.contradicts, (InsightId("insight:1"),))
        self.assertFalse(hasattr(analysis, "position_size"))

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
            features=(
                feature("market:visible"),
                feature("market:future", known_at=NOW + timedelta(days=1)),
                feature("market:other", symbol="XYZ"),
            ),
            events=(
                event("event:visible"),
                event("event:future", known_at=NOW + timedelta(days=1)),
                event("event:other", symbol="XYZ"),
            ),
        )
        self.assertEqual([item.ref for item in selected.records], ["company:visible"])
        self.assertEqual([item.ref for item in selected.promoted_insights], ["insight:visible"])
        self.assertEqual([item.ref for item in selected.market_features], ["market:visible"])
        self.assertEqual([item.ref for item in selected.recent_events], ["event:visible"])

    def test_context_cannot_contain_portfolio_or_trade_authority(self):
        selected = context(records=(record(),))
        self.assertEqual(
            set(selected.__slots__), {
                "symbol", "simulation_time", "company", "records",
                "promoted_insights", "recent_events", "market_features"
            }
        )
        self.assertFalse(hasattr(selected, "portfolio"))

    def test_gateway_checks_agent_permission(self):
        with self.assertRaises(ContextPermissionError):
            ContextGateway(ResearchContextStore()).for_company_analyst(
                agent_id="news_analyst", symbol="ABC", simulation_time=NOW
            )

    def test_duplicate_company_record_is_rejected(self):
        shared = ResearchContextStore()
        shared.append_company_record(record())
        with self.assertRaises(ValueError):
            shared.append_company_record(record())

    def test_company_entity_requires_known_fundamental_references(self):
        shared = ResearchContextStore()
        with self.assertRaises(ValueError):
            shared.append_company_entity(company())

    def test_company_entity_rejects_future_or_other_ticker_fundamentals(self):
        shared = ResearchContextStore()
        shared.append_company_record(record(symbol="XYZ"))
        with self.assertRaises(ValueError):
            shared.append_company_entity(company())

        future = ResearchContextStore()
        future.append_company_record(record(known_at=NOW + timedelta(days=1)))
        with self.assertRaises(ValueError):
            future.append_company_entity(company())


if __name__ == "__main__":
    unittest.main()
