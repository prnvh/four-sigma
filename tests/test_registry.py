import unittest
from datetime import datetime, timezone

from agents import (
    AgentRegistry,
    AgentSpec,
    CompanyAnalyst,
    CompanyRecord,
    CompanyRecordType,
    Evidence,
    NewsAnalyst,
    REGISTRY,
    UnknownAgentVersion,
)
from agents.registry import COMPANY_ANALYST_V1, NEWS_ANALYST_V1
from memory import ContextGateway, NewsEventStore, ResearchContextStore


UTC = timezone.utc


class NewsStub:
    def generate_json(self, *, instructions, input_data, schema):
        self.instructions = instructions
        self.last_input = input_data
        return {
            "claim": "Synthetic test finding",
            "direction": "neutral",
            "confidence": 0.4,
            "horizon": "test horizon",
            "evidence_refs": ["news:1"],
            "risks": ["Synthetic test uncertainty"],
        }


class CompanyStub:
    def generate_json(self, *, instructions, input_data, schema):
        self.instructions = instructions
        return {
            "company_thesis": "Synthetic balanced company thesis",
            "bull_case": "Synthetic bull case",
            "bear_case": "Synthetic bear case",
            "confidence": 0.5,
            "time_horizon": "test horizon",
            "evidence_refs": ["company:1"],
            "catalysts": ["Synthetic catalyst"],
            "risks": ["Synthetic risk"],
            "supports": [],
            "contradicts": [],
            "supersedes": [],
        }


def _news_context():
    store = NewsEventStore()
    store.append_news(
        Evidence(
            ref="news:1",
            source="Synthetic test source",
            url="https://example.test/news:1",
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
            title="Synthetic test headline",
            summary="Synthetic test summary",
            symbols=("ABC",),
            knowledge_time=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    return ContextGateway(store).for_news_analyst(
        agent_id="news_analyst",
        symbol="ABC",
        simulation_time=datetime(2026, 1, 2, tzinfo=UTC),
    )


def _company_context():
    store = ResearchContextStore()
    store.append_company_record(
        CompanyRecord(
            ref="company:1",
            symbol="ABC",
            source="Synthetic filing source",
            url="https://example.test/company:1",
            knowledge_time=datetime(2026, 2, 1, tzinfo=UTC),
            record_type=CompanyRecordType.REGULATORY_FILING,
            label="Synthetic filing fact",
            value="Synthetic value",
            period_end="2025-12-31",
        )
    )
    return ContextGateway(store).for_company_analyst(
        agent_id="company_analyst",
        symbol="ABC",
        simulation_time=datetime(2026, 2, 1, tzinfo=UTC),
    )


class RegistryTests(unittest.TestCase):
    def test_builtin_versions_are_registered(self) -> None:
        self.assertEqual(
            set(REGISTRY.keys()),
            {
                "news:v1",
                "news_analyst:v1",
                "company_analyst:v1",
                "market:v1",
                "risk_llm:v1",
                "portfolio_risk:v1",
            },
        )

    def test_prompt_and_config_travel_with_version(self) -> None:
        spec = REGISTRY.get("news_analyst:v1")
        self.assertEqual(spec.prompt, NEWS_ANALYST_V1.prompt)
        self.assertEqual(dict(spec.config), {"output": "finding"})
        self.assertEqual(spec.log_identity()["agent_key"], "news_analyst:v1")
        self.assertEqual(spec.log_identity()["agent_version"], "v1")

    def test_unknown_version_is_rejected(self) -> None:
        with self.assertRaises(UnknownAgentVersion):
            REGISTRY.get("news_analyst:v9")

    def test_conflicting_reregister_is_rejected(self) -> None:
        registry = AgentRegistry()
        registry.register(NEWS_ANALYST_V1)
        with self.assertRaises(ValueError):
            registry.register(
                AgentSpec(name="news_analyst", version="v1", prompt="different prompt")
            )

    def test_news_output_identifies_registered_version(self) -> None:
        model = NewsStub()
        finding = NewsAnalyst(model).analyze(_news_context())
        self.assertEqual(finding.agent, "news_analyst:v1")
        self.assertEqual(model.instructions, NEWS_ANALYST_V1.prompt)

    def test_company_output_identifies_registered_version(self) -> None:
        model = CompanyStub()
        analysis = CompanyAnalyst(model).analyze(_company_context())
        self.assertEqual(analysis.agent, "company_analyst:v1")
        self.assertEqual(model.instructions, COMPANY_ANALYST_V1.prompt)
