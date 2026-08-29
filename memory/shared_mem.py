from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from threading import RLock
from types import MappingProxyType

from .audit_logger import AuditEventType, AuditLedger
from .promotion import PromotionProposal
from .types import (
    AgentId,
    AuditEventId,
    CanonicalId,
    CreatedAt,
    InsightId,
    InsightRevision,
    InsightStatus,
    InsightVersion,
    ProposalId,
    RunId,
    SimulationTime,
)


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
        self._insight_versions: dict[InsightId, list[InsightVersion]] = {}
        self._insight_proposals: set[ProposalId] = set()
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
            if target is SharedMemorySection.INSIGHTS:
                raise SharedMemoryValidationError(
                    "insights can only be written by an approved promotion proposal"
                )
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

    def apply_approved(
        self, proposal: PromotionProposal, *, decided_at: SimulationTime
    ) -> InsightVersion:
        """Append the insight revision carried by a governance-approved proposal."""
        if not isinstance(proposal, PromotionProposal):
            raise TypeError("proposal must be PromotionProposal")
        if not isinstance(decided_at, SimulationTime):
            raise TypeError("decided_at must be SimulationTime")
        if proposal.target_resource != SharedMemorySection.INSIGHTS.value:
            raise SharedMemoryValidationError(
                "SharedMemory.apply_approved currently accepts insight proposals only"
            )
        revision = (
            proposal.proposed_value
            if isinstance(proposal.proposed_value, InsightRevision)
            else InsightRevision(
                insight_id=InsightId(proposal.id.value),
                value=proposal.proposed_value,
            )
        )
        with self._lock:
            if proposal.id in self._insight_proposals:
                raise SharedMemoryValidationError(
                    f"proposal already created an insight version: {proposal.id.value}"
                )
            history = self._insight_versions.get(revision.insight_id, [])
            if history and decided_at.value < history[-1].valid_from.value:
                raise SharedMemoryValidationError(
                    "an insight version cannot be backdated before its latest version"
                )
            if history and history[-1].entity_id != proposal.entity_id:
                raise SharedMemoryValidationError(
                    "an insight cannot change entity between versions"
                )
            if (
                revision.valid_until is not None
                and revision.valid_until.value <= decided_at.value
            ):
                raise SharedMemoryValidationError(
                    "valid_until must be later than valid_from"
                )
            related = {
                *revision.supports,
                *revision.contradicts,
                *revision.supersedes,
            }
            missing = sorted(
                ref.value for ref in related if ref not in self._insight_versions
            )
            if missing:
                raise SharedMemoryValidationError(
                    f"insight relationships reference unknown insights: {missing}"
                )
            version_number = len(history) + 1
            version = InsightVersion(
                insight_id=revision.insight_id,
                entity_id=proposal.entity_id,
                value=revision.value,
                version=version_number,
                supersedes_version=(
                    None if version_number == 1 else version_number - 1
                ),
                status=revision.status,
                created_by_proposal=proposal.id,
                valid_from=decided_at,
                valid_until=revision.valid_until,
                supports=revision.supports,
                contradicts=revision.contradicts,
                supersedes=revision.supersedes,
            )
            self._insight_versions.setdefault(revision.insight_id, []).append(
                version
            )
            self._insight_proposals.add(proposal.id)
            return deepcopy(version)

    def insight_history(self, insight_id: InsightId) -> tuple[InsightVersion, ...]:
        if not isinstance(insight_id, InsightId):
            raise TypeError("insight_id must be InsightId")
        with self._lock:
            return tuple(deepcopy(self._insight_versions.get(insight_id, ())))

    def insight_as_of(
        self, insight_id: InsightId, simulation_time: SimulationTime
    ) -> InsightVersion | None:
        if not isinstance(insight_id, InsightId):
            raise TypeError("insight_id must be InsightId")
        if not isinstance(simulation_time, SimulationTime):
            raise TypeError("simulation_time must be SimulationTime")
        with self._lock:
            visible = [
                version
                for version in self._insight_versions.get(insight_id, ())
                if version.valid_from.value <= simulation_time.value
            ]
            if not visible:
                return None
            latest = max(visible, key=lambda item: (item.valid_from.value, item.version))
            if latest.status is InsightStatus.RETRACTED:
                return None
            if (
                latest.valid_until is not None
                and simulation_time.value >= latest.valid_until.value
            ):
                return None
            return deepcopy(latest)

    def insights_as_of(
        self, simulation_time: SimulationTime
    ) -> Mapping[InsightId, InsightVersion]:
        if not isinstance(simulation_time, SimulationTime):
            raise TypeError("simulation_time must be SimulationTime")
        with self._lock:
            insight_ids = tuple(self._insight_versions)
        visible = {
            insight_id: version
            for insight_id in insight_ids
            if (version := self.insight_as_of(insight_id, simulation_time)) is not None
        }
        return MappingProxyType(visible)

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
