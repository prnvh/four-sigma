from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import TypeVar

from .types import AgentId, AuditEventId, CanonicalId, CreatedAt, Instant, RunId


class AuditEventType(str, Enum):
    AGENT_RUN_STARTED = "agent_run_started"
    AGENT_RUN_FINISHED = "agent_run_finished"
    CONTEXT_REQUESTED = "context_requested"
    CONTEXT_RETURNED = "context_returned"
    WORKING_MEMORY_WRITTEN = "working_memory_written"
    WORKING_MEMORY_WRITE_REJECTED = "working_memory_write_rejected"
    PROMOTION_REQUESTED = "promotion_requested"
    PROMOTION_APPROVED = "promotion_approved"
    PROMOTION_REJECTED = "promotion_rejected"
    SHARED_MEMORY_UPDATED = "shared_memory_updated"
    SHARED_MEMORY_WRITE_REJECTED = "shared_memory_write_rejected"
    RISK_CHECK_RUN = "risk_check_run"
    TRADE_CANDIDATE_CREATED = "trade_candidate_created"
    TRADE_DECISION_CREATED = "trade_decision_created"
    TRADE_STATUS_CHANGED = "trade_status_changed"


AuditScalar = str | int | float | bool | None
AuditValue = AuditScalar | tuple["AuditValue", ...] | Mapping[str, "AuditValue"]
T = TypeVar("T")


def _freeze(value: object, *, path: str = "details") -> AuditValue:
    """Validate and detach audit details from caller-owned mutable objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, AuditValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON-compatible values, got {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class AuditEvent:
    id: AuditEventId
    sequence: int
    event_type: AuditEventType
    occurred_at: CreatedAt
    details: Mapping[str, AuditValue]
    agent_id: AgentId | None = None
    run_id: RunId | None = None
    subject_id: CanonicalId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, AuditEventId):
            raise TypeError("id must be an AuditEventId")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool):
            raise TypeError("sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("sequence must be positive")
        if not isinstance(self.event_type, AuditEventType):
            raise TypeError("event_type must be an AuditEventType")
        if not isinstance(self.occurred_at, CreatedAt):
            raise TypeError("occurred_at must be a CreatedAt")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        if self.agent_id is not None and not isinstance(self.agent_id, AgentId):
            raise TypeError("agent_id must be an AgentId")
        if self.run_id is not None and not isinstance(self.run_id, RunId):
            raise TypeError("run_id must be a RunId")
        if self.subject_id is not None and not isinstance(
            self.subject_id, CanonicalId
        ):
            raise TypeError("subject_id must be a CanonicalId")

        frozen_details = _freeze(self.details)
        assert isinstance(frozen_details, Mapping)
        object.__setattr__(self, "details", frozen_details)


class AuditLedger:
    """Append-only system record of state changes and system activity."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._event_ids: set[AuditEventId] = set()
        self._lock = RLock()

    def append(
        self,
        *,
        event_id: AuditEventId,
        event_type: AuditEventType,
        occurred_at: CreatedAt,
        details: Mapping[str, object] | None = None,
        agent_id: AgentId | None = None,
        run_id: RunId | None = None,
        subject_id: CanonicalId | None = None,
    ) -> AuditEvent:
        with self._lock:
            event = self._prepare_event(
                event_id=event_id,
                event_type=event_type,
                occurred_at=occurred_at,
                details=details,
                agent_id=agent_id,
                run_id=run_id,
                subject_id=subject_id,
            )
            self._commit(event)
            return event

    def record_state_change(
        self,
        *,
        event_id: AuditEventId,
        event_type: AuditEventType,
        occurred_at: CreatedAt,
        change: Callable[[], T],
        details: Mapping[str, object] | None = None,
        agent_id: AgentId | None = None,
        run_id: RunId | None = None,
        subject_id: CanonicalId | None = None,
    ) -> T:
        """Run a state change through the ledger's required audit boundary.

        All event validation happens before the callback. The event is appended only
        if the callback completes successfully.
        """
        if not callable(change):
            raise TypeError("change must be callable")
        with self._lock:
            event = self._prepare_event(
                event_id=event_id,
                event_type=event_type,
                occurred_at=occurred_at,
                details=details,
                agent_id=agent_id,
                run_id=run_id,
                subject_id=subject_id,
            )
            result = change()
            self._commit(event)
            return result

    def snapshot(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def query(
        self,
        *,
        event_type: AuditEventType | None = None,
        agent_id: AgentId | None = None,
        run_id: RunId | None = None,
        subject_id: CanonicalId | None = None,
        through: Instant | None = None,
    ) -> tuple[AuditEvent, ...]:
        if event_type is not None and not isinstance(event_type, AuditEventType):
            raise TypeError("event_type must be an AuditEventType")
        if through is not None and not isinstance(through, Instant):
            raise TypeError("through must be an Instant")
        with self._lock:
            return tuple(
                event
                for event in self._events
                if (event_type is None or event.event_type is event_type)
                and (agent_id is None or event.agent_id == agent_id)
                and (run_id is None or event.run_id == run_id)
                and (subject_id is None or event.subject_id == subject_id)
                and (
                    through is None
                    or event.occurred_at.value <= through.value
                )
            )

    def _prepare_event(
        self,
        *,
        event_id: AuditEventId,
        event_type: AuditEventType,
        occurred_at: CreatedAt,
        details: Mapping[str, object] | None,
        agent_id: AgentId | None,
        run_id: RunId | None,
        subject_id: CanonicalId | None,
    ) -> AuditEvent:
        if not isinstance(event_id, AuditEventId):
            raise TypeError("event_id must be an AuditEventId")
        if event_id in self._event_ids:
            raise ValueError(f"duplicate audit event id: {event_id}")
        if not isinstance(event_type, AuditEventType):
            raise TypeError("event_type must be an AuditEventType")
        if not isinstance(occurred_at, CreatedAt):
            raise TypeError("occurred_at must be a CreatedAt")
        if details is not None and not isinstance(details, Mapping):
            raise TypeError("details must be a mapping")

        frozen_details = _freeze(details or {})
        assert isinstance(frozen_details, Mapping)
        return AuditEvent(
            id=event_id,
            sequence=len(self._events) + 1,
            event_type=event_type,
            occurred_at=occurred_at,
            details=frozen_details,
            agent_id=agent_id,
            run_id=run_id,
            subject_id=subject_id,
        )

    def _commit(self, event: AuditEvent) -> None:
        self._events.append(event)
        self._event_ids.add(event.id)
