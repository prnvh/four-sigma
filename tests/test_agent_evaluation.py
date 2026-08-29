import unittest
from datetime import datetime, timezone

from agents import (
    AgentEvaluationError,
    AgentEvaluator,
    AgentPredictionOutcome,
    ModelTokenPrice,
)
from memory import (
    AgentId,
    AuditEventId,
    AuditEventType,
    AuditLedger,
    CreatedAt,
    Direction,
    EntityId,
    RunId,
)


NOW = CreatedAt(datetime(2026, 5, 1, tzinfo=timezone.utc))


def finish(
    ledger,
    run,
    *,
    agent="news_analyst",
    version="v1",
    model="model-a",
    status="succeeded",
    latency=100,
    input_tokens=1000,
    output_tokens=500,
):
    ledger.append(
        event_id=AuditEventId(f"audit:{run}:finished"),
        event_type=AuditEventType.AGENT_RUN_FINISHED,
        occurred_at=NOW,
        agent_id=AgentId(agent),
        run_id=RunId(run),
        subject_id=EntityId("ABC"),
        details={
            "agent_version": version,
            "model": model,
            "status": status,
            "latency_ms": latency,
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        },
    )


def promotion(ledger, run, approved):
    suffix = "approved" if approved else "rejected"
    ledger.append(
        event_id=AuditEventId(f"audit:{run}:{suffix}"),
        event_type=(
            AuditEventType.PROMOTION_APPROVED
            if approved else AuditEventType.PROMOTION_REJECTED
        ),
        occurred_at=NOW,
        agent_id=AgentId("news_analyst"),
        run_id=RunId(run),
    )


class AgentEvaluatorTests(unittest.TestCase):
    def test_calculates_all_required_agent_metrics(self):
        ledger = AuditLedger()
        finish(ledger, "run:1")
        promotion(ledger, "run:1", True)
        finish(
            ledger, "run:2", status="failed", latency=300,
            input_tokens=500, output_tokens=0,
        )
        promotion(ledger, "run:2", False)
        outcome = AgentPredictionOutcome(
            run_id=RunId("run:1"),
            confidence=0.8,
            predicted_direction=Direction.BULLISH,
            realized_direction=Direction.BULLISH,
        )

        result = AgentEvaluator({
            "model-a": ModelTokenPrice(2, 4)
        }).evaluate(ledger, outcomes=(outcome,))[0]

        self.assertEqual(result.runs, 2)
        self.assertEqual((result.successful_runs, result.failed_runs), (1, 1))
        self.assertEqual((result.accepted_proposals, result.rejected_proposals), (1, 1))
        self.assertAlmostEqual(result.confidence_brier_score, 0.04)
        self.assertEqual(result.direction_accuracy, 1)
        self.assertEqual(result.average_latency_ms, 200)
        self.assertEqual((result.input_tokens, result.output_tokens), (1500, 500))
        self.assertEqual(result.total_tokens, 2000)
        self.assertAlmostEqual(result.estimated_cost_usd, 0.005)

    def test_incorrect_direction_affects_accuracy_and_calibration(self):
        ledger = AuditLedger()
        finish(ledger, "run:wrong")
        outcome = AgentPredictionOutcome(
            RunId("run:wrong"), 0.9, Direction.BULLISH, Direction.BEARISH
        )
        result = AgentEvaluator().evaluate(ledger, outcomes=(outcome,))[0]
        self.assertEqual(result.direction_accuracy, 0)
        self.assertAlmostEqual(result.confidence_brier_score, 0.81)

    def test_groups_agent_versions_and_models_separately(self):
        ledger = AuditLedger()
        finish(ledger, "run:v1", version="v1", model="model-a")
        finish(ledger, "run:v2", version="v2", model="model-b")
        result = AgentEvaluator().evaluate(ledger)
        self.assertEqual(
            [(item.agent_version, item.model) for item in result],
            [("v1", "model-a"), ("v2", "model-b")],
        )

    def test_unpriced_token_usage_reports_unknown_cost(self):
        ledger = AuditLedger()
        finish(ledger, "run:1")
        result = AgentEvaluator().evaluate(ledger)[0]
        self.assertIsNone(result.estimated_cost_usd)
        self.assertIsNone(result.direction_accuracy)
        self.assertIsNone(result.confidence_brier_score)

    def test_zero_token_run_has_known_zero_cost_without_price(self):
        ledger = AuditLedger()
        finish(ledger, "run:1", input_tokens=0, output_tokens=0)
        self.assertEqual(AgentEvaluator().evaluate(ledger)[0].estimated_cost_usd, 0)

    def test_rejects_duplicate_outcomes(self):
        ledger = AuditLedger()
        finish(ledger, "run:1")
        outcome = AgentPredictionOutcome(
            RunId("run:1"), 0.5, Direction.NEUTRAL, Direction.NEUTRAL
        )
        with self.assertRaisesRegex(AgentEvaluationError, "duplicate outcome"):
            AgentEvaluator().evaluate(ledger, outcomes=(outcome, outcome))

    def test_rejects_outcome_for_unknown_run(self):
        outcome = AgentPredictionOutcome(
            RunId("run:missing"), 0.5, Direction.NEUTRAL, Direction.NEUTRAL
        )
        with self.assertRaisesRegex(AgentEvaluationError, "unknown finished runs"):
            AgentEvaluator().evaluate((), outcomes=(outcome,))

    def test_rejects_duplicate_finished_run(self):
        ledger = AuditLedger()
        finish(ledger, "run:1")
        ledger.append(
            event_id=AuditEventId("audit:duplicate-finish"),
            event_type=AuditEventType.AGENT_RUN_FINISHED,
            occurred_at=NOW,
            agent_id=AgentId("news_analyst"),
            run_id=RunId("run:1"),
            details={
                "agent_version": "v1", "model": "model-a", "status": "failed",
                "latency_ms": 1, "token_usage": None,
            },
        )
        with self.assertRaisesRegex(AgentEvaluationError, "duplicate finished run"):
            AgentEvaluator().evaluate(ledger)


if __name__ == "__main__":
    unittest.main()
