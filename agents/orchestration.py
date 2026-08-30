from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class TriggerType(StrEnum):
    RELEVANT_NEWS = "relevant_news"
    NEW_FILING = "new_filing"
    LARGE_PRICE_MOVE = "large_price_move"
    PORTFOLIO_CHANGE = "portfolio_change"
    INSIGHT_EXPIRY = "insight_expiry"
    SCHEDULED_REVIEW = "scheduled_review"


@dataclass(frozen=True, slots=True)
class AgentTrigger:
    trigger_type: TriggerType
    entity_id: str
    knowledge_time: datetime
    source_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trigger_type, TriggerType):
            raise TypeError("trigger_type must be TriggerType")
        entity = self.entity_id.strip().upper() if isinstance(self.entity_id, str) else ""
        if not entity:
            raise ValueError("entity_id must be non-empty")
        if not isinstance(self.knowledge_time, datetime):
            raise TypeError("knowledge_time must be datetime")
        if self.knowledge_time.tzinfo is None:
            raise ValueError("knowledge_time must be timezone-aware")
        if self.source_ref is not None and (
            not isinstance(self.source_ref, str) or not self.source_ref.strip()
        ):
            raise ValueError("source_ref must be a non-empty string or None")
        object.__setattr__(self, "entity_id", entity)
        if self.source_ref is not None:
            object.__setattr__(self, "source_ref", self.source_ref.strip())


@dataclass(frozen=True, slots=True)
class PlannedAgentRun:
    agent_key: str
    entity_id: str
    simulation_time: datetime
    triggers: tuple[TriggerType, ...]
    source_refs: tuple[str, ...]


class EventDrivenOrchestrator:
    """Convert point-in-time events into coalesced, deterministic agent work."""

    def __init__(
        self, subscriptions: Mapping[str, frozenset[TriggerType] | set[TriggerType]]
    ) -> None:
        if not isinstance(subscriptions, Mapping) or not subscriptions:
            raise ValueError("subscriptions must be a non-empty mapping")
        normalized: dict[str, frozenset[TriggerType]] = {}
        for agent_key, triggers in subscriptions.items():
            if not isinstance(agent_key, str) or not agent_key.strip():
                raise ValueError("subscription agent keys must be non-empty strings")
            selected = frozenset(triggers)
            if not selected or any(not isinstance(item, TriggerType) for item in selected):
                raise ValueError("each subscription requires TriggerType values")
            normalized[agent_key.strip()] = selected
        self._subscriptions = normalized

    def plan(
        self,
        events: Sequence[AgentTrigger],
        *,
        simulation_time: datetime,
    ) -> tuple[PlannedAgentRun, ...]:
        if not isinstance(simulation_time, datetime):
            raise TypeError("simulation_time must be datetime")
        if simulation_time.tzinfo is None:
            raise ValueError("simulation_time must be timezone-aware")
        if any(not isinstance(event, AgentTrigger) for event in events):
            raise TypeError("events must contain AgentTrigger values")
        grouped: dict[tuple[str, str], list[AgentTrigger]] = {}
        for event in events:
            if event.knowledge_time > simulation_time:
                continue
            for agent_key, subscribed in self._subscriptions.items():
                if event.trigger_type in subscribed:
                    grouped.setdefault((agent_key, event.entity_id), []).append(event)
        planned = []
        for (agent_key, entity_id), selected in sorted(grouped.items()):
            triggers = tuple(sorted({item.trigger_type for item in selected}, key=str))
            refs = tuple(sorted({item.source_ref for item in selected if item.source_ref}))
            planned.append(PlannedAgentRun(
                agent_key=agent_key,
                entity_id=entity_id,
                simulation_time=simulation_time,
                triggers=triggers,
                source_refs=refs,
            ))
        return tuple(planned)
