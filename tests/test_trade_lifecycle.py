import unittest
from datetime import datetime, timedelta, timezone

from memory import (
    AgentId,
    AuditEventId,
    AuditEventType,
    AuditLedger,
    CreatedAt,
    InsightId,
    TradeCandidate,
    TradeCandidateId,
    TradeCandidateStatus,
    TradeLifecycle,
    TradeLifecycleError,
    TradeSide,
)


NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)


def candidate(identifier="trade:1", status=TradeCandidateStatus.PROPOSED):
    return TradeCandidate(
        id=TradeCandidateId(identifier),
        instrument="ABC",
        direction=TradeSide.LONG,
        thesis_refs=(InsightId("insight:1"),),
        horizon="30 days",
        confidence=0.7,
        entry_conditions=("approved",),
        exit_conditions=("thesis invalidated",),
        proposed_size=0.1,
        knowledge_time=NOW,
        status=status,
    )


class TradeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.ledger = AuditLedger()
        self.lifecycle = TradeLifecycle(self.ledger)

    def at(self, minute):
        return CreatedAt(NOW + timedelta(minutes=minute))

    def register(self):
        return self.lifecycle.register(
            candidate(),
            event_id=AuditEventId("audit:create"),
            occurred_at=self.at(0),
            agent_id=AgentId("trade_constructor"),
        )

    def transition(self, status, minute):
        return self.lifecycle.transition(
            TradeCandidateId("trade:1"),
            status,
            event_id=AuditEventId(f"audit:{status.value}"),
            occurred_at=self.at(minute),
            reason=f"test {status.value}",
        )

    def test_complete_lifecycle_is_audited(self):
        self.register()
        path = (
            TradeCandidateStatus.RISK_REVIEWED,
            TradeCandidateStatus.APPROVED,
            TradeCandidateStatus.SUBMITTED,
            TradeCandidateStatus.FILLED,
            TradeCandidateStatus.CLOSED,
        )
        for minute, status in enumerate(path, start=1):
            self.assertEqual(self.transition(status, minute).status, status)

        events = self.ledger.query(subject_id=TradeCandidateId("trade:1"))
        self.assertEqual(events[0].event_type, AuditEventType.TRADE_CANDIDATE_CREATED)
        self.assertTrue(
            all(event.event_type is AuditEventType.TRADE_STATUS_CHANGED for event in events[1:])
        )
        self.assertEqual(events[-1].details["from_status"], "filled")
        self.assertEqual(events[-1].details["to_status"], "closed")

    def test_risk_review_can_reject_and_rejected_is_terminal(self):
        self.register()
        self.transition(TradeCandidateStatus.RISK_REVIEWED, 1)
        self.transition(TradeCandidateStatus.REJECTED, 2)
        with self.assertRaisesRegex(TradeLifecycleError, "cannot transition"):
            self.transition(TradeCandidateStatus.APPROVED, 3)
        self.assertEqual(
            self.lifecycle.get(TradeCandidateId("trade:1")).status,
            TradeCandidateStatus.REJECTED,
        )

    def test_cannot_skip_risk_review_or_submission(self):
        self.register()
        with self.assertRaises(TradeLifecycleError):
            self.transition(TradeCandidateStatus.APPROVED, 1)
        self.transition(TradeCandidateStatus.RISK_REVIEWED, 2)
        self.transition(TradeCandidateStatus.APPROVED, 3)
        with self.assertRaises(TradeLifecycleError):
            self.transition(TradeCandidateStatus.FILLED, 4)

    def test_invalid_transition_does_not_write_audit_event(self):
        self.register()
        before = self.ledger.snapshot()
        with self.assertRaises(TradeLifecycleError):
            self.transition(TradeCandidateStatus.CLOSED, 1)
        self.assertEqual(self.ledger.snapshot(), before)

    def test_duplicate_audit_id_prevents_state_change(self):
        self.register()
        with self.assertRaisesRegex(ValueError, "duplicate audit event id"):
            self.lifecycle.transition(
                TradeCandidateId("trade:1"),
                TradeCandidateStatus.RISK_REVIEWED,
                event_id=AuditEventId("audit:create"),
                occurred_at=self.at(1),
            )
        self.assertEqual(
            self.lifecycle.get(TradeCandidateId("trade:1")).status,
            TradeCandidateStatus.PROPOSED,
        )

    def test_rejects_non_proposed_registration_and_duplicate_candidate(self):
        with self.assertRaisesRegex(TradeLifecycleError, "start as proposed"):
            self.lifecycle.register(
                candidate(status=TradeCandidateStatus.APPROVED),
                event_id=AuditEventId("audit:bad"),
                occurred_at=self.at(0),
            )
        self.register()
        with self.assertRaisesRegex(TradeLifecycleError, "already registered"):
            self.lifecycle.register(
                candidate(),
                event_id=AuditEventId("audit:duplicate"),
                occurred_at=self.at(1),
            )

    def test_rejects_backdated_transition(self):
        self.register()
        self.transition(TradeCandidateStatus.RISK_REVIEWED, 2)
        with self.assertRaisesRegex(TradeLifecycleError, "move backwards"):
            self.lifecycle.transition(
                TradeCandidateId("trade:1"),
                TradeCandidateStatus.APPROVED,
                event_id=AuditEventId("audit:backdated"),
                occurred_at=self.at(1),
            )

    def test_risk_review_can_resize_candidate_and_audits_the_size(self):
        self.register()
        reviewed = self.lifecycle.transition(
            TradeCandidateId("trade:1"),
            TradeCandidateStatus.RISK_REVIEWED,
            event_id=AuditEventId("audit:resize"),
            occurred_at=self.at(1),
            proposed_size=0.05,
        )
        self.assertEqual(reviewed.proposed_size, 0.05)
        self.assertEqual(self.ledger.snapshot()[-1].details["proposed_size"], 0.05)

        with self.assertRaisesRegex(TradeLifecycleError, "only change during risk review"):
            self.lifecycle.transition(
                TradeCandidateId("trade:1"),
                TradeCandidateStatus.APPROVED,
                event_id=AuditEventId("audit:bad-resize"),
                occurred_at=self.at(2),
                proposed_size=0.01,
            )


if __name__ == "__main__":
    unittest.main()
