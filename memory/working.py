from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .ids import AgentId, EntityId
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
        entity_id: EntityId,
        category: WorkingMemoryCategory | str,
        value: object,
        created_at: CreatedAt,
        expires_at: SimulationTime | None = None,
    ) -> WorkingMemoryEntry:
        return self._store.write(
            WorkingMemoryEntry(
                agent_id=self.agent_id,
                entity_id=entity_id,
                category=_category(category),
                value=value,
                created_at=created_at,
                expires_at=expires_at,
            )
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

    def __init__(self) -> None:
        self._entries: dict[AgentId, list[WorkingMemoryEntry]] = {}

    def for_agent(self, agent_id: AgentId) -> AgentWorkspace:
        return AgentWorkspace(self, agent_id)

    def write(self, entry: WorkingMemoryEntry) -> WorkingMemoryEntry:
        if not isinstance(entry, WorkingMemoryEntry):
            raise TypeError("entry must be WorkingMemoryEntry")
        self._entries.setdefault(entry.agent_id, []).append(entry)
        return entry

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
