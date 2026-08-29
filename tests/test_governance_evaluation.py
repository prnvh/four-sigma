import unittest
from datetime import datetime, timedelta, timezone

from agents import (
    GovernanceEvaluationError,
    GovernanceEvaluator,
    GovernanceInsightOutcome,
)
from memory import (
    AuditEventId,
    AuditEventType,
    AuditLedger,
    CreatedAt,
    ProposalId,
    SimulationTime,
)


NOW = datetime(2026, 5, 1, tzinfo=timezone.utc)
HORIZON = timedelta(days=30)


def decision(ledger, proposal, approved, reasons=()):
    ledger.append(
        event_id=AuditEventId(f"audit:{proposal}"),
        event_type=(
            AuditEventType.PROMOTION_APPROVED
            if approved
            else AuditEventType.PROMOTION_REJECTED
        ),
        occurred_at=CreatedAt(NOW),
        subject_id=ProposalId(proposal),
        details={"reasons": reasons},
    )


def outcome(proposal, performance, polluted=False, horizon=HORIZON):
    return GovernanceInsightOutcome(
        proposal_id=ProposalId(proposal),
        performance=performance,
        evaluated_at=SimulationTime(NOW + horizon),
        horizon=horizon,
        polluted=polluted,
    )


class GovernanceEvaluatorTests(unittest.TestCase):
    def test_measures_gate_quality_and_rejected_counterfactuals(self):
        ledger = AuditLedger()
        decision(ledger, "approved:good", True)
        decision(ledger, "approved:bad", True)
        decision(ledger, "rejected:duplicate", False, ("duplicate_news_insight",))
        decision(ledger, "rejected:weak", False, ("insufficient_evidence",))

        result = GovernanceEvaluator().evaluate(
            ledger,
            outcomes=(
                outcome("approved:good", 0.10),
                outcome("approved:bad", -0.05, polluted=True),
                outcome("rejected:duplicate", 0.08),
                outcome("rejected:weak", -0.10),
            ),
        )

        self.assertEqual((result.total_proposals, result.resolved_proposals), (4, 4))
        self.assertEqual((result.approval_rate, result.rejection_rate), (0.5, 0.5))
        self.assertAlmostEqual(result.approved_insight_performance, 0.025)
        self.assertAlmostEqual(
            result.rejected_insight_counterfactual_performance, -0.01
        )
        self.assertEqual(result.memory_pollution_rate, 0.5)
        self.assertEqual(result.duplicate_rate, 0.25)
        self.assertAlmostEqual(result.gate_value_added, 0.035)

    def test_missing_outcomes_are_unresolved_not_zero(self):
        ledger = AuditLedger()
        decision(ledger, "approved", True)
        decision(ledger, "rejected", False)

        result = GovernanceEvaluator().evaluate(
            ledger, outcomes=(outcome("approved", 0.1),)
        )

        self.assertEqual((result.resolved_proposals, result.unresolved_proposals), (1, 1))
        self.assertEqual(result.approved_insight_performance, 0.1)
        self.assertIsNone(result.rejected_insight_counterfactual_performance)
        self.assertIsNone(result.gate_value_added)

    def test_empty_audit_returns_safe_undefined_rates(self):
        result = GovernanceEvaluator().evaluate(())
        self.assertEqual(result.total_proposals, 0)
        self.assertIsNone(result.approval_rate)
        self.assertIsNone(result.rejection_rate)
        self.assertIsNone(result.memory_pollution_rate)
        self.assertIsNone(result.duplicate_rate)

    def test_rejects_mixed_horizons_and_wrong_evaluation_time(self):
        ledger = AuditLedger()
        decision(ledger, "one", True)
        decision(ledger, "two", False)
        with self.assertRaisesRegex(GovernanceEvaluationError, "same horizon"):
            GovernanceEvaluator().evaluate(
                ledger,
                outcomes=(
                    outcome("one", 0.1),
                    outcome("two", 0.2, horizon=timedelta(days=10)),
                ),
            )
        wrong_time = GovernanceInsightOutcome(
            ProposalId("one"), 0.1, SimulationTime(NOW), HORIZON
        )
        with self.assertRaisesRegex(GovernanceEvaluationError, "declared horizon"):
            GovernanceEvaluator().evaluate(ledger, outcomes=(wrong_time,))

    def test_rejects_duplicate_and_unknown_outcomes(self):
        ledger = AuditLedger()
        decision(ledger, "known", True)
        known = outcome("known", 0.1)
        with self.assertRaisesRegex(GovernanceEvaluationError, "duplicate outcome"):
            GovernanceEvaluator().evaluate(ledger, outcomes=(known, known))
        with self.assertRaisesRegex(GovernanceEvaluationError, "unknown proposals"):
            GovernanceEvaluator().evaluate(
                ledger, outcomes=(outcome("unknown", 0.1),)
            )


if __name__ == "__main__":
    unittest.main()
