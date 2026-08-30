from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

from memory.audit_logger import AuditEvent, AuditEventType
from memory.context_gateway import ContextSnapshot, ContextSnapshotStore
from memory.portfolio import Fill, FillSide
from memory.shared_mem import SharedMemory
from memory.types import ContextSnapshotId, InsightId, jsonable

from .backtest import BacktestResult


class RunExplorerError(ValueError):
    """A query is invalid or its immutable source record is unavailable."""


@dataclass(frozen=True, slots=True)
class LifecycleExplanation:
    from_status: str | None
    to_status: str
    reason: str
    proposed_size: float | None


@dataclass(frozen=True, slots=True)
class TradeExplanation:
    candidate_id: str
    instrument: str
    direction: str
    final_status: str
    thesis_refs: tuple[str, ...]
    lifecycle: tuple[LifecycleExplanation, ...]
    fills: tuple[Fill, ...]
    context_snapshot_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResizeExplanation:
    candidate_id: str
    original_size: float
    reviewed_size: float
    resized: bool
    reason: str


@dataclass(frozen=True, slots=True)
class AgentContextExplanation:
    snapshot: ContextSnapshot
    view: object


@dataclass(frozen=True, slots=True)
class InsightOrigin:
    insight_id: str
    version: int
    proposal_id: str
    valid_from: datetime
    valid_until: datetime | None


class RunExplorer:
    """Read-only, structured explanations backed by completed-run records."""

    def __init__(
        self,
        result: BacktestResult,
        *,
        context_store: ContextSnapshotStore | None = None,
        shared_memory: SharedMemory | None = None,
    ) -> None:
        if not isinstance(result, BacktestResult):
            raise TypeError("result must be BacktestResult")
        if context_store is not None and not isinstance(context_store, ContextSnapshotStore):
            raise TypeError("context_store must be ContextSnapshotStore or None")
        if shared_memory is not None and not isinstance(shared_memory, SharedMemory):
            raise TypeError("shared_memory must be SharedMemory or None")
        self.result = result
        self.context_store = context_store
        self.shared_memory = shared_memory

    def why_trade(
        self, instrument: str, *, candidate_id: str | None = None
    ) -> TradeExplanation:
        symbol = instrument.strip().upper() if isinstance(instrument, str) else ""
        if not symbol:
            raise ValueError("instrument must be non-empty")
        candidates = [
            item for item in self.result.candidates
            if item.instrument == symbol
            and (candidate_id is None or item.id.value == candidate_id)
        ]
        if not candidates:
            raise RunExplorerError(f"no trade candidate found for {symbol}")
        candidate = candidates[-1]
        events = self._candidate_events(candidate.id.value)
        lifecycle = tuple(
            LifecycleExplanation(
                from_status=event.details.get("from_status"),
                to_status=str(event.details.get("to_status", event.details.get("status", ""))),
                reason=str(event.details.get("reason", "")),
                proposed_size=self._number(event.details.get("proposed_size")),
            )
            for event in events
        )
        thesis_refs = tuple(ref.value for ref in candidate.thesis_refs)
        snapshots = tuple(
            snapshot.id.value for snapshot in self.result.context_snapshots
            if set(snapshot.source_refs).intersection(thesis_refs)
        )
        filled_at = {
            event.occurred_at.value for event in events
            if event.event_type is AuditEventType.TRADE_STATUS_CHANGED
            and event.details.get("to_status") == "filled"
        }
        fills = tuple(
            fill for fill in self.result.fills
            if fill.instrument == symbol
            and fill.knowledge_time in filled_at
            and self._fill_matches(candidate.direction.value, fill)
        )
        return TradeExplanation(
            candidate_id=candidate.id.value,
            instrument=symbol,
            direction=candidate.direction.value,
            final_status=candidate.status.value,
            thesis_refs=thesis_refs,
            lifecycle=lifecycle,
            fills=fills,
            context_snapshot_ids=tuple(dict.fromkeys(snapshots)),
        )

    def context(self, snapshot_id: str) -> AgentContextExplanation:
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("snapshot_id must be non-empty")
        snapshot = next(
            (item for item in self.result.context_snapshots if item.id.value == snapshot_id),
            None,
        )
        if snapshot is None:
            raise RunExplorerError(f"snapshot is not part of this run: {snapshot_id}")
        if self.context_store is None:
            raise RunExplorerError("context view store was not supplied")
        try:
            view = self.context_store.view(ContextSnapshotId(snapshot_id))
        except KeyError as exc:
            raise RunExplorerError(f"context view is unavailable: {snapshot_id}") from exc
        return AgentContextExplanation(snapshot=snapshot, view=deepcopy(jsonable(view)))

    def insight_origin(self, insight_id: str, *, version: int | None = None) -> InsightOrigin:
        if self.shared_memory is None:
            raise RunExplorerError("shared memory was not supplied")
        if not isinstance(insight_id, str) or not insight_id.strip():
            raise ValueError("insight_id must be non-empty")
        history = self.shared_memory.insight_history(InsightId(insight_id))
        selected = [item for item in history if version is None or item.version == version]
        if not selected:
            raise RunExplorerError(f"unknown insight version: {insight_id}:{version}")
        item = selected[-1]
        return InsightOrigin(
            insight_id=item.insight_id.value,
            version=item.version,
            proposal_id=item.created_by_proposal.value,
            valid_from=item.valid_from.value,
            valid_until=item.valid_until.value if item.valid_until else None,
        )

    def why_resized(self, candidate_id: str) -> ResizeExplanation:
        events = self._candidate_events(candidate_id)
        created = next(
            (item for item in events if item.event_type is AuditEventType.TRADE_CANDIDATE_CREATED),
            None,
        )
        reviewed = next(
            (
                item for item in events
                if item.event_type is AuditEventType.TRADE_STATUS_CHANGED
                and item.details.get("to_status") == "risk_reviewed"
            ),
            None,
        )
        if created is None or reviewed is None:
            raise RunExplorerError(f"candidate lacks size-review records: {candidate_id}")
        original = self._number(created.details.get("proposed_size"))
        final = self._number(reviewed.details.get("proposed_size"))
        if original is None or final is None:
            raise RunExplorerError(f"candidate lacks audited size values: {candidate_id}")
        return ResizeExplanation(
            candidate_id=candidate_id,
            original_size=original,
            reviewed_size=final,
            resized=abs(original - final) > 1e-12,
            reason=str(reviewed.details.get("reason", "")),
        )

    def _candidate_events(self, candidate_id: str) -> tuple[AuditEvent, ...]:
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        events = tuple(
            event for event in self.result.audit_events
            if event.details.get("candidate_id") == candidate_id
        )
        if not events:
            raise RunExplorerError(f"unknown candidate: {candidate_id}")
        return events

    @staticmethod
    def _number(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _fill_matches(direction: str, fill: Fill) -> bool:
        return (
            direction == "long" and fill.side is FillSide.BUY
        ) or (
            direction == "short" and fill.side is FillSide.SELL
        )
