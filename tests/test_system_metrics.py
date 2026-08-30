import unittest
from datetime import datetime, timedelta, timezone

from agents import ModelTokenPrice, SimulationWorkload, SystemMetricsCollector
from memory import (
    AgentId,
    AuditEvent,
    AuditEventId,
    AuditEventType,
    ContextSnapshot,
    ContextSnapshotId,
    CreatedAt,
    EntityId,
    ProposalId,
    RunId,
    TradeCandidateId,
)


NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)


def event(sequence, kind, milliseconds, details, *, subject=None, run=True):
    return AuditEvent(
        id=AuditEventId(f"audit:{sequence}"), sequence=sequence,
        event_type=kind, occurred_at=CreatedAt(NOW + timedelta(milliseconds=milliseconds)),
        details=details, agent_id=AgentId("agent"),
        run_id=RunId(f"run:{sequence}") if run else None, subject_id=subject,
    )


class SystemMetricsTests(unittest.TestCase):
    def test_collects_all_required_operational_metrics(self):
        candidate = TradeCandidateId("trade:1")
        proposal = ProposalId("proposal:1")
        events = (
            event(1, AuditEventType.AGENT_RUN_FINISHED, 0, {
                "latency_ms": 100, "model": "model-a", "cache_hit": False,
                "token_usage": {"input_tokens": 100, "output_tokens": 50},
            }),
            event(2, AuditEventType.AGENT_RUN_FINISHED, 1, {
                "latency_ms": 300, "model": "model-a", "cache_hit": True,
                "token_usage": None,
            }),
            event(3, AuditEventType.TRADE_CANDIDATE_CREATED, 10, {}, subject=candidate),
            event(4, AuditEventType.TRADE_STATUS_CHANGED, 60,
                  {"to_status": "approved"}, subject=candidate),
            event(5, AuditEventType.PROMOTION_REQUESTED, 100, {}, subject=proposal),
            event(6, AuditEventType.PROMOTION_APPROVED, 125, {}, subject=proposal),
        )
        contexts = (
            ContextSnapshot(ContextSnapshotId("context:1"), "agent", NOW, (), (), "a", 100),
            ContextSnapshot(ContextSnapshotId("context:2"), "agent", NOW, (), (), "b", 300),
        )
        metrics = SystemMetricsCollector({
            "model-a": ModelTokenPrice(2.0, 4.0)
        }).collect(
            events,
            contexts=contexts,
            workload=SimulationWorkload(1000, 2.0, 10),
        )
        self.assertEqual(metrics.agent_runs, 2)
        self.assertEqual(metrics.average_agent_latency_ms, 200)
        self.assertEqual(metrics.average_event_to_decision_latency_ms, 50)
        self.assertEqual(metrics.total_tokens, 150)
        self.assertEqual(metrics.tokens_per_run, 75)
        self.assertAlmostEqual(metrics.total_cost_usd, 0.0004)
        self.assertAlmostEqual(metrics.cost_per_run_usd, 0.0002)
        self.assertAlmostEqual(metrics.cost_per_day_usd, 0.00004)
        self.assertEqual(metrics.average_context_size_bytes, 200)
        self.assertEqual(metrics.maximum_context_size_bytes, 300)
        self.assertEqual(metrics.cache_hit_rate, 0.5)
        self.assertEqual(metrics.average_governance_latency_ms, 25)
        self.assertEqual(metrics.simulation_ticks_per_second, 500)

    def test_unpriced_tokens_make_cost_unknown(self):
        metrics = SystemMetricsCollector().collect((
            event(1, AuditEventType.AGENT_RUN_FINISHED, 0, {
                "latency_ms": 1, "model": "unknown", "cache_hit": False,
                "token_usage": {"input_tokens": 1, "output_tokens": 1},
            }),
        ))
        self.assertIsNone(metrics.total_cost_usd)
        self.assertIsNone(metrics.cost_per_run_usd)

    def test_empty_telemetry_returns_explicit_unknowns(self):
        metrics = SystemMetricsCollector().collect(())
        self.assertEqual(metrics.agent_runs, 0)
        self.assertIsNone(metrics.average_agent_latency_ms)
        self.assertIsNone(metrics.cache_hit_rate)
        self.assertIsNone(metrics.simulation_ticks_per_second)


if __name__ == "__main__":
    unittest.main()
