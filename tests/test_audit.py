import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

from memory import (
    AgentId,
    AuditEvent,
    AuditEventId,
    AuditEventType,
    AuditLedger,
    CreatedAt,
    EventId,
    KnowledgeTime,
    RunId,
)


UTC = timezone.utc


def _created(day: int = 1) -> CreatedAt:
    return CreatedAt(datetime(2025, 1, day, tzinfo=UTC))


class AuditEventTypeTests(unittest.TestCase):
    def test_initial_event_types_are_exact(self) -> None:
        self.assertEqual(
            {event_type.value for event_type in AuditEventType},
            {
                "agent_run_started",
                "agent_run_finished",
                "context_requested",
                "context_returned",
                "working_memory_written",
                "working_memory_write_rejected",
                "promotion_requested",
                "promotion_approved",
                "promotion_rejected",
                "shared_memory_updated",
                "shared_memory_write_rejected",
                "risk_check_run",
                "trade_candidate_created",
                "trade_decision_created",
            },
        )

    def test_every_initial_event_type_can_be_recorded(self) -> None:
        ledger = AuditLedger()

        for sequence, event_type in enumerate(AuditEventType, start=1):
            ledger.append(
                event_id=AuditEventId(f"audit-{sequence}"),
                event_type=event_type,
                occurred_at=_created(),
            )

        self.assertEqual(
            tuple(event.event_type for event in ledger.snapshot()),
            tuple(AuditEventType),
        )


class AuditLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = AuditLedger()

    def _append(self, number: int, event_type: AuditEventType):
        return self.ledger.append(
            event_id=AuditEventId(f"audit-{number}"),
            event_type=event_type,
            occurred_at=_created(number),
            agent_id=AgentId("news-analyst"),
            run_id=RunId("run-1"),
        )

    def test_events_are_appended_in_sequence_order(self) -> None:
        first = self._append(1, AuditEventType.AGENT_RUN_STARTED)
        second = self._append(2, AuditEventType.AGENT_RUN_FINISHED)

        self.assertEqual((first.sequence, second.sequence), (1, 2))
        self.assertEqual(self.ledger.snapshot(), (first, second))

    def test_duplicate_event_id_is_rejected_without_changing_ledger(self) -> None:
        self._append(1, AuditEventType.AGENT_RUN_STARTED)

        with self.assertRaises(ValueError):
            self.ledger.append(
                event_id=AuditEventId("audit-1"),
                event_type=AuditEventType.AGENT_RUN_FINISHED,
                occurred_at=_created(2),
            )

        self.assertEqual(len(self.ledger.snapshot()), 1)

    def test_ids_and_timestamps_must_use_the_correct_semantic_types(self) -> None:
        with self.assertRaises(TypeError):
            self.ledger.append(
                event_id=EventId("wrong-kind"),  # type: ignore[arg-type]
                event_type=AuditEventType.CONTEXT_REQUESTED,
                occurred_at=_created(),
            )
        with self.assertRaises(TypeError):
            self.ledger.append(
                event_id=AuditEventId("audit-1"),
                event_type=AuditEventType.CONTEXT_REQUESTED,
                occurred_at=KnowledgeTime(_created().value),  # type: ignore[arg-type]
            )

    def test_event_and_nested_details_are_immutable_and_detached(self) -> None:
        original = {"entities": ["AAPL"], "result": {"approved": True}}
        event = self.ledger.append(
            event_id=AuditEventId("audit-1"),
            event_type=AuditEventType.PROMOTION_APPROVED,
            occurred_at=_created(),
            details=original,
        )
        original["entities"].append("MSFT")
        original["result"]["approved"] = False

        self.assertEqual(event.details["entities"], ("AAPL",))
        self.assertEqual(event.details["result"]["approved"], True)
        with self.assertRaises(TypeError):
            event.details["new"] = "value"  # type: ignore[index]
        with self.assertRaises(TypeError):
            event.details["result"]["approved"] = False  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            event.sequence = 99  # type: ignore[misc]

    def test_direct_event_construction_also_freezes_details(self) -> None:
        details = {"evidence": ["event-1"]}
        event = AuditEvent(
            id=AuditEventId("audit-1"),
            sequence=1,
            event_type=AuditEventType.PROMOTION_REQUESTED,
            occurred_at=_created(),
            details=details,
        )
        details["evidence"].append("event-2")

        self.assertEqual(event.details["evidence"], ("event-1",))

    def test_non_json_compatible_details_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            self.ledger.append(
                event_id=AuditEventId("audit-1"),
                event_type=AuditEventType.RISK_CHECK_RUN,
                occurred_at=_created(),
                details={"bad": object()},
            )

    def test_query_filters_without_mutating_history(self) -> None:
        self._append(1, AuditEventType.CONTEXT_REQUESTED)
        second = self._append(2, AuditEventType.CONTEXT_RETURNED)

        self.assertEqual(
            self.ledger.query(event_type=AuditEventType.CONTEXT_RETURNED),
            (second,),
        )
        self.assertEqual(
            self.ledger.query(through=_created(1)),
            (self.ledger.snapshot()[0],),
        )


class AuditedStateChangeTests(unittest.TestCase):
    def test_successful_state_change_generates_event(self) -> None:
        ledger = AuditLedger()
        state: list[str] = []

        result = ledger.record_state_change(
            event_id=AuditEventId("audit-1"),
            event_type=AuditEventType.WORKING_MEMORY_WRITTEN,
            occurred_at=_created(),
            change=lambda: (state.append("observation") or "written"),
        )

        self.assertEqual(result, "written")
        self.assertEqual(state, ["observation"])
        self.assertEqual(len(ledger.snapshot()), 1)

    def test_failed_state_change_does_not_generate_success_event(self) -> None:
        ledger = AuditLedger()

        def fail() -> None:
            raise RuntimeError("write failed")

        with self.assertRaises(RuntimeError):
            ledger.record_state_change(
                event_id=AuditEventId("audit-1"),
                event_type=AuditEventType.SHARED_MEMORY_UPDATED,
                occurred_at=_created(),
                change=fail,
            )

        self.assertEqual(ledger.snapshot(), ())

    def test_invalid_audit_data_prevents_state_change(self) -> None:
        ledger = AuditLedger()
        state: list[str] = []

        with self.assertRaises(TypeError):
            ledger.record_state_change(
                event_id=AuditEventId("audit-1"),
                event_type="working_memory_written",  # type: ignore[arg-type]
                occurred_at=_created(),
                change=lambda: state.append("should not happen"),
            )

        self.assertEqual(state, [])


if __name__ == "__main__":
    unittest.main()
