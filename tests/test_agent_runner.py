import unittest
from datetime import datetime, timezone

from agents import (
    AgentRunRequest,
    AgentRunner,
    CompanyAnalysisResult,
    ModelCallResult,
    NewsAnalysisResult,
    OutputValidationError,
    RiskAnalysisResult,
    TokenUsage,
    TradeProposalResult,
)
from agents.registry import NEWS_ANALYST_V1, NEWS_V1
from memory import (
    AgentId,
    AuditEventType,
    AuditLedger,
    ContextGateway,
    ContextPurpose,
    EntityId,
    Evidence,
    ResearchContextStore,
    RunId,
    SimulationTime,
    WorkingMemory,
)


NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


class DummyModel:
    def __init__(self, output):
        self.output = output
        self.called = False

    def generate_json(self, *, instructions, input_data, schema):
        self.called = True
        return self.output


def valid_output():
    return {
        "claim": "Synthetic result",
        "direction": "neutral",
        "confidence": 0.5,
        "horizon": "30 days",
        "evidence_refs": ["news:1"],
        "risks": ["Synthetic uncertainty"],
    }


class AgentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.ledger = AuditLedger()
        store = ResearchContextStore()
        store.append_news(
            Evidence(
                ref="news:1",
                source="Synthetic source",
                url="https://example.test/news-1",
                published_at=NOW,
                title="Synthetic headline",
                symbols=("ABC",),
            )
        )
        self.gateway = ContextGateway(store)
        self.memory = WorkingMemory(self.ledger)
        self.runner = AgentRunner(
            context_gateway=self.gateway,
            working_memory=self.memory,
            audit_ledger=self.ledger,
        )

    def request(self, run_id="run:1"):
        return AgentRunRequest(
            run_id=RunId(run_id),
            entity_id=EntityId("ABC"),
            purpose=ContextPurpose.NEWS_ANALYSIS,
            simulation_time=NOW,
        )

    def test_dummy_agent_completes_audited_run_end_to_end(self):
        model = DummyModel(ModelCallResult(valid_output(), TokenUsage(12, 7)))
        result = self.runner.run(
            self.request(),
            spec=NEWS_ANALYST_V1,
            model=model,
            model_name="dummy-v1",
            output_type=NewsAnalysisResult,
        )

        self.assertTrue(model.called)
        self.assertEqual(result.output.data["claim"], "Synthetic result")
        self.assertEqual(result.token_usage.as_dict()["total_tokens"], 19)
        entries = self.memory.for_agent(result.working_memory_entry.agent_id).list(
            as_of=SimulationTime(NOW), entity_id=EntityId("ABC")
        )
        self.assertEqual(entries, (result.working_memory_entry,))
        self.assertEqual(
            [event.event_type for event in self.ledger.snapshot()],
            [
                AuditEventType.AGENT_RUN_STARTED,
                AuditEventType.CONTEXT_REQUESTED,
                AuditEventType.CONTEXT_RETURNED,
                AuditEventType.WORKING_MEMORY_WRITTEN,
                AuditEventType.AGENT_RUN_FINISHED,
            ],
        )
        finished = self.ledger.snapshot()[-1]
        self.assertEqual(finished.details["status"], "succeeded")
        self.assertEqual(
            finished.details["context_snapshot_id"],
            result.context_snapshot_id.value,
        )
        self.assertEqual(finished.details["agent_version"], "v1")
        self.assertEqual(finished.details["model"], "dummy-v1")
        self.assertEqual(finished.details["prompt_version"], "v1")

    def test_malformed_output_is_audited_and_never_reaches_memory(self):
        malformed = {**valid_output(), "unexpected": "must be rejected"}
        with self.assertRaises(OutputValidationError):
            self.runner.run(
                self.request("run:bad"),
                spec=NEWS_ANALYST_V1,
                model=DummyModel(malformed),
                model_name="dummy-v1",
                output_type=NewsAnalysisResult,
            )

        entries = self.memory.for_agent(AgentId(NEWS_ANALYST_V1.name)).list(
            as_of=SimulationTime(NOW)
        )
        self.assertEqual(entries, ())
        events = self.ledger.query(run_id=RunId("run:bad"))
        self.assertNotIn(
            AuditEventType.WORKING_MEMORY_WRITTEN,
            [event.event_type for event in events],
        )
        self.assertEqual(events[-1].event_type, AuditEventType.AGENT_RUN_FINISHED)
        self.assertEqual(events[-1].details["status"], "failed")
        self.assertEqual(events[-1].details["error_type"], "OutputValidationError")

    def test_permission_denial_is_audited_before_model_call(self):
        model = DummyModel(valid_output())
        with self.assertRaises(PermissionError):
            self.runner.run(
                self.request("run:denied"),
                spec=NEWS_V1,
                model=model,
                model_name="dummy-v1",
                output_type=NewsAnalysisResult,
            )
        self.assertFalse(model.called)
        events = self.ledger.query(run_id=RunId("run:denied"))
        self.assertEqual(
            [event.event_type for event in events],
            [AuditEventType.AGENT_RUN_STARTED, AuditEventType.AGENT_RUN_FINISHED],
        )
        self.assertEqual(events[-1].details["status"], "failed")

    def test_all_contracts_reject_unstructured_prose(self):
        for output_type in (
            NewsAnalysisResult,
            CompanyAnalysisResult,
            RiskAnalysisResult,
            TradeProposalResult,
        ):
            with self.subTest(output_type=output_type.__name__):
                with self.assertRaises(OutputValidationError):
                    output_type.parse("free-form prose")


if __name__ == "__main__":
    unittest.main()
