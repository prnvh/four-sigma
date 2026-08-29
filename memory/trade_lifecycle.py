from __future__ import annotations

from dataclasses import replace
from threading import RLock

from .audit_logger import AuditEventType, AuditLedger
from .types import (
    AgentId,
    AuditEventId,
    CreatedAt,
    RunId,
    TradeCandidate,
    TradeCandidateId,
    TradeCandidateStatus,
)


class TradeLifecycleError(ValueError):
    """A candidate registration or status transition is invalid."""


ALLOWED_TRADE_TRANSITIONS: dict[
    TradeCandidateStatus, frozenset[TradeCandidateStatus]
] = {
    TradeCandidateStatus.PROPOSED: frozenset({TradeCandidateStatus.RISK_REVIEWED}),
    TradeCandidateStatus.RISK_REVIEWED: frozenset(
        {TradeCandidateStatus.APPROVED, TradeCandidateStatus.REJECTED}
    ),
    TradeCandidateStatus.APPROVED: frozenset({TradeCandidateStatus.SUBMITTED}),
    TradeCandidateStatus.REJECTED: frozenset(),
    TradeCandidateStatus.SUBMITTED: frozenset({TradeCandidateStatus.FILLED}),
    TradeCandidateStatus.FILLED: frozenset({TradeCandidateStatus.CLOSED}),
    TradeCandidateStatus.CLOSED: frozenset(),
}


class TradeLifecycle:
    """Audited source of truth for immutable trade-candidate states."""

    def __init__(self, ledger: AuditLedger) -> None:
        if not isinstance(ledger, AuditLedger):
            raise TypeError("ledger must be an AuditLedger")
        self._ledger = ledger
        self._candidates: dict[TradeCandidateId, TradeCandidate] = {}
        self._changed_at: dict[TradeCandidateId, CreatedAt] = {}
        self._lock = RLock()

    def register(
        self,
        candidate: TradeCandidate,
        *,
        event_id: AuditEventId,
        occurred_at: CreatedAt,
        agent_id: AgentId | None = None,
        run_id: RunId | None = None,
    ) -> TradeCandidate:
        if not isinstance(candidate, TradeCandidate):
            raise TypeError("candidate must be a TradeCandidate")
        if candidate.status is not TradeCandidateStatus.PROPOSED:
            raise TradeLifecycleError("new candidates must start as proposed")
        if not isinstance(occurred_at, CreatedAt):
            raise TypeError("occurred_at must be CreatedAt")
        if occurred_at.value < candidate.knowledge_time:
            raise TradeLifecycleError("registration cannot predate candidate knowledge_time")

        with self._lock:
            if candidate.id in self._candidates:
                raise TradeLifecycleError(f"candidate already registered: {candidate.id}")

            def change() -> TradeCandidate:
                self._candidates[candidate.id] = candidate
                self._changed_at[candidate.id] = occurred_at
                return candidate

            return self._ledger.record_state_change(
                event_id=event_id,
                event_type=AuditEventType.TRADE_CANDIDATE_CREATED,
                occurred_at=occurred_at,
                change=change,
                details={
                    "candidate_id": candidate.id.value,
                    "instrument": candidate.instrument,
                    "status": candidate.status.value,
                },
                agent_id=agent_id,
                run_id=run_id,
                subject_id=candidate.id,
            )

    def transition(
        self,
        candidate_id: TradeCandidateId,
        to_status: TradeCandidateStatus,
        *,
        event_id: AuditEventId,
        occurred_at: CreatedAt,
        reason: str = "",
        agent_id: AgentId | None = None,
        run_id: RunId | None = None,
    ) -> TradeCandidate:
        if not isinstance(candidate_id, TradeCandidateId):
            raise TypeError("candidate_id must be TradeCandidateId")
        if not isinstance(to_status, TradeCandidateStatus):
            raise TypeError("to_status must be TradeCandidateStatus")
        if not isinstance(occurred_at, CreatedAt):
            raise TypeError("occurred_at must be CreatedAt")
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")

        with self._lock:
            current = self._candidates.get(candidate_id)
            if current is None:
                raise TradeLifecycleError(f"unknown candidate: {candidate_id}")
            if occurred_at.value < self._changed_at[candidate_id].value:
                raise TradeLifecycleError("transition time cannot move backwards")
            allowed = ALLOWED_TRADE_TRANSITIONS[current.status]
            if to_status not in allowed:
                raise TradeLifecycleError(
                    f"cannot transition {current.status.value} to {to_status.value}"
                )
            updated = replace(current, status=to_status)

            def change() -> TradeCandidate:
                self._candidates[candidate_id] = updated
                self._changed_at[candidate_id] = occurred_at
                return updated

            return self._ledger.record_state_change(
                event_id=event_id,
                event_type=AuditEventType.TRADE_STATUS_CHANGED,
                occurred_at=occurred_at,
                change=change,
                details={
                    "candidate_id": candidate_id.value,
                    "from_status": current.status.value,
                    "to_status": to_status.value,
                    "reason": reason.strip(),
                },
                agent_id=agent_id,
                run_id=run_id,
                subject_id=candidate_id,
            )

    def get(self, candidate_id: TradeCandidateId) -> TradeCandidate:
        if not isinstance(candidate_id, TradeCandidateId):
            raise TypeError("candidate_id must be TradeCandidateId")
        with self._lock:
            try:
                return self._candidates[candidate_id]
            except KeyError as error:
                raise TradeLifecycleError(f"unknown candidate: {candidate_id}") from error

    def snapshot(self) -> tuple[TradeCandidate, ...]:
        with self._lock:
            return tuple(
                self._candidates[key]
                for key in sorted(self._candidates, key=lambda item: item.value)
            )
