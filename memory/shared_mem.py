from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from threading import RLock
from types import MappingProxyType

from .audit_logger import AuditEventType, AuditLedger
from .types import AgentId, AuditEventId, CanonicalId, CreatedAt, RunId


class SharedMemorySection(str, Enum):
    EVENTS = "events"
    ENTITIES = "entities"
    INSIGHTS = "insights"
    PORTFOLIO = "portfolio"
    RISK = "risk"
    TRADE_CANDIDATES = "trade_candidates"
    DECISIONS = "decisions"


class SharedMemoryValidationError(ValueError):
    """A shared-memory write does not match the fixed schema."""


class UnknownSharedMemorySection(SharedMemoryValidationError):
    """A write targeted a section outside the shared-memory schema."""


class SharedMemory:
    """Validated storage boundary for the seven shared-memory sections."""

    def __init__(self, audit_ledger: AuditLedger) -> None:
        if not isinstance(audit_ledger, AuditLedger):
            raise TypeError("audit_ledger must be an AuditLedger")
        self._audit_ledger = audit_ledger
        self._records: dict[SharedMemorySection, dict[str, dict[str, object]]] = {
            section: {} for section in SharedMemorySection
        }
        self._lock = RLock()

    def write(
        self,
        *,
        section: SharedMemorySection | str,
        record_key: str,
        value: Mapping[str, object],
        audit_event_id: AuditEventId,
        occurred_at: CreatedAt,
        agent_id: AgentId | None = None,
        run_id: RunId | None = None,
        subject_id: CanonicalId | None = None,
    ) -> Mapping[str, object]:
        try:
            target = self._section(section)
            key = self._record_key(record_key)
            record = self._record(value)
        except (SharedMemoryValidationError, TypeError) as error:
            self._record_rejection(
                audit_event_id=audit_event_id,
                occurred_at=occurred_at,
                section=section,
                record_key=record_key,
                error=error,
                agent_id=agent_id,
                run_id=run_id,
                subject_id=subject_id,
            )
            raise

        def store() -> None:
            with self._lock:
                if (
                    target is SharedMemorySection.EVENTS
                    and key in self._records[target]
                ):
                    raise SharedMemoryValidationError(
                        f"events are append-only; record already exists: {key}"
                    )
                self._records[target][key] = record

        try:
            self._audit_ledger.record_state_change(
                event_id=audit_event_id,
                event_type=AuditEventType.SHARED_MEMORY_UPDATED,
                occurred_at=occurred_at,
                change=store,
                details={"section": target.value, "record_key": key},
                agent_id=agent_id,
                run_id=run_id,
                subject_id=subject_id,
            )
        except SharedMemoryValidationError as error:
            self._record_rejection(
                audit_event_id=audit_event_id,
                occurred_at=occurred_at,
                section=section,
                record_key=record_key,
                error=error,
                agent_id=agent_id,
                run_id=run_id,
                subject_id=subject_id,
            )
            raise
        return self.read(target, key)

    def read(
        self, section: SharedMemorySection | str, record_key: str
    ) -> Mapping[str, object]:
        target = self._section(section)
        key = self._record_key(record_key)
        with self._lock:
            return MappingProxyType(deepcopy(self._records[target][key]))

    def snapshot(
        self, section: SharedMemorySection | str
    ) -> Mapping[str, Mapping[str, object]]:
        target = self._section(section)
        with self._lock:
            records = {
                key: MappingProxyType(deepcopy(value))
                for key, value in self._records[target].items()
            }
        return MappingProxyType(records)

    @staticmethod
    def _section(value: SharedMemorySection | str) -> SharedMemorySection:
        if isinstance(value, SharedMemorySection):
            return value
        if isinstance(value, str):
            try:
                return SharedMemorySection(value)
            except ValueError as error:
                raise UnknownSharedMemorySection(
                    f"unknown shared-memory section: {value}"
                ) from error
        raise TypeError("section must be a SharedMemorySection or string")

    @staticmethod
    def _record_key(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SharedMemoryValidationError("record_key must be a non-empty string")
        return value

    @staticmethod
    def _record(value: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise SharedMemoryValidationError("value must be a mapping")
        if any(not isinstance(key, str) for key in value):
            raise SharedMemoryValidationError("value keys must be strings")
        return deepcopy(dict(value))

    @staticmethod
    def _section_label(value: object) -> str:
        if isinstance(value, SharedMemorySection):
            return value.value
        return str(value)

    def _record_rejection(
        self,
        *,
        audit_event_id: AuditEventId,
        occurred_at: CreatedAt,
        section: object,
        record_key: object,
        error: TypeError | SharedMemoryValidationError,
        agent_id: AgentId | None,
        run_id: RunId | None,
        subject_id: CanonicalId | None,
    ) -> None:
        self._audit_ledger.append(
            event_id=audit_event_id,
            event_type=AuditEventType.SHARED_MEMORY_WRITE_REJECTED,
            occurred_at=occurred_at,
            details={
                "section": self._section_label(section),
                "record_key": str(record_key),
                "reason": type(error).__name__,
                "message": str(error),
            },
            agent_id=agent_id,
            run_id=run_id,
            subject_id=subject_id,
        )
