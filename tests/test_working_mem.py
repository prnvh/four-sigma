import unittest
from datetime import datetime, timedelta, timezone

from memory import (
    AgentId,
    AuditEventId,
    AuditEventType,
    AuditLedger,
    CreatedAt,
    EntityId,
    SimulationClock,
    SimulationTime,
    WorkingMemory,
    WorkingMemoryCategory,
    WorkingMemoryEntry,
)


UTC = timezone.utc
START = SimulationTime(datetime(2024, 1, 1, tzinfo=UTC))
AAPL = EntityId("AAPL")


def _created(clock: SimulationClock) -> CreatedAt:
    return CreatedAt(clock.now().value)


class WorkingMemoryIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = SimulationClock(START)
        self.ledger = AuditLedger()
        self.memory = WorkingMemory(self.ledger)
        self.analyst = self.memory.for_agent(AgentId("news_analyst"))
        self.researcher = self.memory.for_agent(AgentId("company_analyst"))

    def test_workspaces_are_just_agent_ids(self) -> None:
        self.assertEqual(self.analyst.agent_id, AgentId("news_analyst"))
        self.assertIsInstance(self.analyst.agent_id, AgentId)

    def test_agent_cannot_read_another_agents_memory(self) -> None:
        self.analyst.write(
            audit_event_id=AuditEventId("audit-1"),
            entity_id=AAPL,
            category=WorkingMemoryCategory.OBSERVATION,
            value={"text": "private to news_analyst"},
            created_at=_created(self.clock),
        )
        self.researcher.write(
            audit_event_id=AuditEventId("audit-2"),
            entity_id=AAPL,
            category=WorkingMemoryCategory.HYPOTHESIS,
            value={"text": "private to company_analyst"},
            created_at=_created(self.clock),
        )

        as_of = self.clock.now()
        mine = self.researcher.list(as_of=as_of, entity_id=AAPL)
        theirs = self.analyst.list(as_of=as_of, entity_id=AAPL)

        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0].value, {"text": "private to company_analyst"})
        self.assertEqual(mine[0].agent_id, AgentId("company_analyst"))
        self.assertEqual(len(theirs), 1)
        self.assertEqual(theirs[0].agent_id, AgentId("news_analyst"))
        self.assertNotEqual(mine[0].value, theirs[0].value)

    def test_store_list_is_scoped_to_the_given_agent(self) -> None:
        self.analyst.write(
            audit_event_id=AuditEventId("audit-1"),
            entity_id=AAPL,
            category="observation",
            value=1,
            created_at=_created(self.clock),
        )
        leaked = self.memory.list(
            AgentId("company_analyst"),
            as_of=self.clock.now(),
            entity_id=AAPL,
        )
        self.assertEqual(leaked, ())

    def test_workspace_write_cannot_impersonate_another_agent(self) -> None:
        entry = self.analyst.write(
            audit_event_id=AuditEventId("audit-1"),
            entity_id=AAPL,
            category=WorkingMemoryCategory.QUESTION,
            value="why did volume spike?",
            created_at=_created(self.clock),
        )
        self.assertEqual(entry.agent_id, AgentId("news_analyst"))
        self.assertEqual(
            self.researcher.list(as_of=self.clock.now()),
            (),
        )


class WorkingMemoryWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = SimulationClock(START)
        self.ledger = AuditLedger()
        self.ws = WorkingMemory(self.ledger).for_agent(AgentId("news_analyst"))

    def test_unknown_category_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.ws.write(
                audit_event_id=AuditEventId("audit-1"),
                entity_id=AAPL,
                category="trade",
                value={},
                created_at=_created(self.clock),
            )

    def test_created_at_must_be_supplied(self) -> None:
        with self.assertRaises(TypeError):
            self.ws.write(
                audit_event_id=AuditEventId("audit-1"),
                entity_id=AAPL,
                category=WorkingMemoryCategory.OBSERVATION,
                value={},
            )  # type: ignore[call-arg]

    def test_filters_by_category_and_entity(self) -> None:
        self.ws.write(
            audit_event_id=AuditEventId("audit-1"),
            entity_id=AAPL,
            category=WorkingMemoryCategory.OBSERVATION,
            value="aapl-obs",
            created_at=_created(self.clock),
        )
        self.ws.write(
            audit_event_id=AuditEventId("audit-2"),
            entity_id=EntityId("MSFT"),
            category=WorkingMemoryCategory.HYPOTHESIS,
            value="msft-hyp",
            created_at=_created(self.clock),
        )
        as_of = self.clock.now()
        only_aapl = self.ws.list(as_of=as_of, entity_id=AAPL)
        only_hyp = self.ws.list(
            as_of=as_of, category=WorkingMemoryCategory.HYPOTHESIS
        )
        self.assertEqual([e.value for e in only_aapl], ["aapl-obs"])
        self.assertEqual([e.value for e in only_hyp], ["msft-hyp"])

    def test_as_of_hides_future_and_expired_entries(self) -> None:
        first = _created(self.clock)
        self.ws.write(
            audit_event_id=AuditEventId("audit-1"),
            entity_id=AAPL,
            category=WorkingMemoryCategory.CANDIDATE_INSIGHT,
            value="early",
            created_at=first,
            expires_at=SimulationTime(self.clock.now().value + timedelta(days=2)),
        )
        self.clock.advance_by(timedelta(days=3))
        later = _created(self.clock)
        self.ws.write(
            audit_event_id=AuditEventId("audit-2"),
            entity_id=AAPL,
            category=WorkingMemoryCategory.CANDIDATE_INSIGHT,
            value="late",
            created_at=later,
        )

        at_start = self.ws.list(as_of=START, entity_id=AAPL)
        after_expiry = self.ws.list(as_of=self.clock.now(), entity_id=AAPL)
        self.assertEqual([e.value for e in at_start], ["early"])
        self.assertEqual([e.value for e in after_expiry], ["late"])

    def test_direct_entry_requires_owner_agent_id(self) -> None:
        with self.assertRaises(TypeError):
            WorkingMemoryEntry(
                agent_id="news_analyst",  # type: ignore[arg-type]
                entity_id=AAPL,
                category=WorkingMemoryCategory.OBSERVATION,
                value={},
                created_at=_created(SimulationClock(START)),
            )

    def test_successful_write_generates_audit_event(self) -> None:
        entry = self.ws.write(
            audit_event_id=AuditEventId("audit-1"),
            entity_id=AAPL,
            category=WorkingMemoryCategory.OBSERVATION,
            value="volume increased",
            created_at=_created(self.clock),
        )

        events = self.ledger.snapshot()
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0].event_type,
            AuditEventType.WORKING_MEMORY_WRITTEN,
        )
        self.assertEqual(events[0].agent_id, entry.agent_id)
        self.assertEqual(events[0].subject_id, AAPL)
        self.assertEqual(events[0].details["category"], "observation")

    def test_rejected_write_generates_rejection_audit_event(self) -> None:
        with self.assertRaises(ValueError):
            self.ws.write(
                audit_event_id=AuditEventId("audit-1"),
                entity_id=AAPL,
                category="unknown",
                value={},
                created_at=_created(self.clock),
            )

        events = self.ledger.snapshot()
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0].event_type,
            AuditEventType.WORKING_MEMORY_WRITE_REJECTED,
        )
        self.assertEqual(events[0].agent_id, AgentId("news_analyst"))
        self.assertEqual(events[0].subject_id, AAPL)
        self.assertEqual(events[0].details["category"], "unknown")

    def test_invalid_audit_id_prevents_the_memory_write(self) -> None:
        self.ws.write(
            audit_event_id=AuditEventId("audit-1"),
            entity_id=AAPL,
            category=WorkingMemoryCategory.OBSERVATION,
            value="first",
            created_at=_created(self.clock),
        )

        with self.assertRaises(ValueError):
            self.ws.write(
                audit_event_id=AuditEventId("audit-1"),
                entity_id=AAPL,
                category=WorkingMemoryCategory.OBSERVATION,
                value="must not be stored",
                created_at=_created(self.clock),
            )

        entries = self.ws.list(as_of=self.clock.now())
        self.assertEqual([entry.value for entry in entries], ["first"])
        self.assertEqual(len(self.ledger.snapshot()), 1)
