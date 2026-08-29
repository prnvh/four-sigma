from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .audit import AuditEventType, AuditLedger
from .ids import AgentId, AuditEventId, EntityId, RunId
from .time import CreatedAt, SimulationTime


class WorkingMemoryCategory(StrEnum):
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    QUESTION = "question"
    CANDIDATE_INSIGHT = "candidate_insight"


def _category(value: object) -> WorkingMemoryCategory:
    if isinstance(value, WorkingMemoryCategory):
        return value
    if isinstance(value, str):
        try:
            return WorkingMemoryCategory(value)
        except ValueError:
            pass
    raise ValueError(f"unknown working-memory category: {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkingMemoryEntry:
    agent_id: AgentId
    entity_id: EntityId
    category: WorkingMemoryCategory
    value: object
    created_at: CreatedAt
    expires_at: SimulationTime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, AgentId):
            raise TypeError("agent_id must be AgentId")
        if not isinstance(self.entity_id, EntityId):
            raise TypeError("entity_id must be EntityId")
        object.__setattr__(self, "category", _category(self.category))
        if not isinstance(self.created_at, CreatedAt):
            raise TypeError("created_at must be CreatedAt")
        if self.expires_at is not None:
            if not isinstance(self.expires_at, SimulationTime):
                raise TypeError("expires_at must be SimulationTime")
            if self.expires_at <= self.created_at:
                raise ValueError("expires_at must be after created_at")

    def visible_as_of(self, when: SimulationTime) -> bool:
        if not isinstance(when, SimulationTime):
            raise TypeError("as_of must be SimulationTime")
        if self.created_at > when:
            return False
        if self.expires_at is not None and self.expires_at <= when:
            return False
        return True


class AgentWorkspace:
    """Private working memory for one AgentId. No Agent class required."""

    def __init__(self, store: WorkingMemory, agent_id: AgentId) -> None:
        if not isinstance(agent_id, AgentId):
            raise TypeError("agent_id must be AgentId")
        self._store = store
        self.agent_id = agent_id

    def write(
        self,
        *,
        audit_event_id: AuditEventId,
        run_id: RunId | None = None,
        entity_id: EntityId,
        category: WorkingMemoryCategory | str,
        value: object,
        created_at: CreatedAt,
        expires_at: SimulationTime | None = None,
    ) -> WorkingMemoryEntry:
        try:
            entry = WorkingMemoryEntry(
                agent_id=self.agent_id,
                entity_id=entity_id,
                category=_category(category),
                value=value,
                created_at=created_at,
                expires_at=expires_at,
            )
        except (TypeError, ValueError) as error:
            self._store._record_rejection(
                audit_event_id=audit_event_id,
                occurred_at=created_at,
                agent_id=self.agent_id,
                entity_id=entity_id,
                category=category,
                error=error,
                run_id=run_id,
            )
            raise
        return self._store.write(
            entry,
            audit_event_id=audit_event_id,
            run_id=run_id,
        )

    def list(
        self,
        *,
        as_of: SimulationTime,
        entity_id: EntityId | None = None,
        category: WorkingMemoryCategory | str | None = None,
    ) -> tuple[WorkingMemoryEntry, ...]:
        return self._store.list(
            self.agent_id,
            as_of=as_of,
            entity_id=entity_id,
            category=category,
        )


class WorkingMemory:
    """Append-only private stores, partitioned by AgentId."""

    def __init__(self, audit_ledger: AuditLedger) -> None:
        if not isinstance(audit_ledger, AuditLedger):
            raise TypeError("audit_ledger must be AuditLedger")
        self._audit_ledger = audit_ledger
        self._entries: dict[AgentId, list[WorkingMemoryEntry]] = {}

    def for_agent(self, agent_id: AgentId) -> AgentWorkspace:
        return AgentWorkspace(self, agent_id)

    def write(
        self,
        entry: WorkingMemoryEntry,
        *,
        audit_event_id: AuditEventId,
        run_id: RunId | None = None,
    ) -> WorkingMemoryEntry:
        if not isinstance(entry, WorkingMemoryEntry):
            raise TypeError("entry must be WorkingMemoryEntry")

        def store() -> WorkingMemoryEntry:
            self._entries.setdefault(entry.agent_id, []).append(entry)
            return entry

        return self._audit_ledger.record_state_change(
            event_id=audit_event_id,
            event_type=AuditEventType.WORKING_MEMORY_WRITTEN,
            occurred_at=entry.created_at,
            change=store,
            details={
                "entity_id": entry.entity_id.value,
                "category": entry.category.value,
            },
            agent_id=entry.agent_id,
            run_id=run_id,
            subject_id=entry.entity_id,
        )

    def _record_rejection(
        self,
        *,
        audit_event_id: AuditEventId,
        occurred_at: CreatedAt,
        agent_id: AgentId,
        entity_id: object,
        category: object,
        error: TypeError | ValueError,
        run_id: RunId | None = None,
    ) -> None:
        self._audit_ledger.append(
            event_id=audit_event_id,
            event_type=AuditEventType.WORKING_MEMORY_WRITE_REJECTED,
            occurred_at=occurred_at,
            details={
                "entity_id": str(entity_id),
                "category": str(category),
                "reason": type(error).__name__,
                "message": str(error),
            },
            agent_id=agent_id,
            run_id=run_id,
            subject_id=entity_id if isinstance(entity_id, EntityId) else None,
        )

    def list(
        self,
        agent_id: AgentId,
        *,
        as_of: SimulationTime,
        entity_id: EntityId | None = None,
        category: WorkingMemoryCategory | str | None = None,
    ) -> tuple[WorkingMemoryEntry, ...]:
        if not isinstance(agent_id, AgentId):
            raise TypeError("agent_id must be AgentId")
        wanted = None if category is None else _category(category)
        found: list[WorkingMemoryEntry] = []
        for entry in self._entries.get(agent_id, ()):
            if entity_id is not None and entry.entity_id != entity_id:
                continue
            if wanted is not None and entry.category != wanted:
                continue
            if not entry.visible_as_of(as_of):
                continue
            found.append(entry)
        return tuple(found)
