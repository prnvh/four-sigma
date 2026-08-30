from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from memory.audit_logger import AuditEvent, AuditEventType, AuditLedger
from memory.context_gateway import ContextSnapshot
from memory.lineage import LineageGraph, LineageNodeType

from .evaluation import ModelTokenPrice


class SystemMetricsError(ValueError):
    """The supplied telemetry is incomplete or internally inconsistent."""


def _non_negative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemMetricsError(f"{name} must be numeric")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise SystemMetricsError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class SimulationWorkload:
    ticks: int
    wall_time_seconds: float
    simulation_days: float

    def __post_init__(self) -> None:
        if isinstance(self.ticks, bool) or not isinstance(self.ticks, int) or self.ticks < 0:
            raise ValueError("ticks must be a non-negative integer")
        wall = _non_negative(self.wall_time_seconds, "wall_time_seconds")
        days = _non_negative(self.simulation_days, "simulation_days")
        if self.ticks and wall == 0:
            raise ValueError("wall_time_seconds must be positive when ticks were processed")
        if days == 0:
            raise ValueError("simulation_days must be positive")
        object.__setattr__(self, "wall_time_seconds", wall)
        object.__setattr__(self, "simulation_days", days)


@dataclass(frozen=True, slots=True)
class SystemMetrics:
    agent_runs: int
    average_agent_latency_ms: float | None
    average_event_to_decision_latency_ms: float | None
    total_tokens: int
    tokens_per_run: float | None
    total_cost_usd: float | None
    cost_per_run_usd: float | None
    cost_per_day_usd: float | None
    average_context_size_bytes: float | None
    maximum_context_size_bytes: int | None
    cache_hit_rate: float | None
    average_governance_latency_ms: float | None
    simulation_ticks_per_second: float | None


class SystemMetricsCollector:
    """Aggregate operational telemetry from immutable audit and context records."""

    def __init__(self, pricing: Mapping[str, ModelTokenPrice] | None = None) -> None:
        self.pricing = dict(pricing or {})
        if any(
            not isinstance(model, str) or not model.strip()
            or not isinstance(price, ModelTokenPrice)
            for model, price in self.pricing.items()
        ):
            raise TypeError("pricing must map model names to ModelTokenPrice")

    def collect(
        self,
        audit: AuditLedger | Sequence[AuditEvent],
        *,
        contexts: Sequence[ContextSnapshot] = (),
        workload: SimulationWorkload | None = None,
        lineage: LineageGraph | None = None,
    ) -> SystemMetrics:
        events = audit.snapshot() if isinstance(audit, AuditLedger) else tuple(audit)
        if any(not isinstance(item, AuditEvent) for item in events):
            raise TypeError("audit must contain AuditEvent values")
        if any(not isinstance(item, ContextSnapshot) for item in contexts):
            raise TypeError("contexts must contain ContextSnapshot values")
        if workload is not None and not isinstance(workload, SimulationWorkload):
            raise TypeError("workload must be SimulationWorkload or None")
        if lineage is not None and not isinstance(lineage, LineageGraph):
            raise TypeError("lineage must be LineageGraph or None")

        finished = [
            event for event in events
            if event.event_type is AuditEventType.AGENT_RUN_FINISHED
        ]
        latencies = [
            _non_negative(event.details.get("latency_ms"), "latency_ms")
            for event in finished
        ]
        total_tokens = 0
        total_cost = 0.0
        unpriced = False
        cache_hits = 0
        for event in finished:
            input_tokens, output_tokens = self._tokens(event.details.get("token_usage"))
            total_tokens += input_tokens + output_tokens
            model = event.details.get("model")
            price = self.pricing.get(model) if isinstance(model, str) else None
            if input_tokens + output_tokens and price is None:
                unpriced = True
            elif price is not None:
                total_cost += (
                    input_tokens * price.input_per_million_usd
                    + output_tokens * price.output_per_million_usd
                ) / 1_000_000
            hit = event.details.get("cache_hit", False)
            if not isinstance(hit, bool):
                raise SystemMetricsError("cache_hit must be boolean")
            cache_hits += hit

        decision_latencies = (
            self._lineage_decision_latencies(lineage)
            if lineage is not None
            else self._paired_latency(
                events,
                start=AuditEventType.TRADE_CANDIDATE_CREATED,
                finishes={AuditEventType.TRADE_STATUS_CHANGED},
                finish_filter=lambda event: event.details.get("to_status")
                in {"approved", "rejected"},
            )
        )
        governance_latencies = self._paired_latency(
            events,
            start=AuditEventType.PROMOTION_REQUESTED,
            finishes={AuditEventType.PROMOTION_APPROVED, AuditEventType.PROMOTION_REJECTED},
            finish_filter=lambda event: True,
        )
        sizes = [item.content_size_bytes for item in contexts]
        run_count = len(finished)
        known_cost = None if unpriced else total_cost
        return SystemMetrics(
            agent_runs=run_count,
            average_agent_latency_ms=self._average(latencies),
            average_event_to_decision_latency_ms=self._average(decision_latencies),
            total_tokens=total_tokens,
            tokens_per_run=(total_tokens / run_count if run_count else None),
            total_cost_usd=known_cost,
            cost_per_run_usd=(
                known_cost / run_count if known_cost is not None and run_count else None
            ),
            cost_per_day_usd=(
                known_cost / workload.simulation_days
                if known_cost is not None and workload is not None else None
            ),
            average_context_size_bytes=self._average(sizes),
            maximum_context_size_bytes=max(sizes) if sizes else None,
            cache_hit_rate=(cache_hits / run_count if run_count else None),
            average_governance_latency_ms=self._average(governance_latencies),
            simulation_ticks_per_second=(
                workload.ticks / workload.wall_time_seconds
                if workload is not None and workload.wall_time_seconds else None
            ),
        )

    @staticmethod
    def _lineage_decision_latencies(graph: LineageGraph) -> list[float]:
        decisions = (
            *graph.nodes(LineageNodeType.PORTFOLIO_DECISION),
            *graph.nodes(LineageNodeType.DECISION),
        )
        samples = []
        for decision in decisions:
            events = [
                node for node in graph.upstream(decision.id)
                if node.node_type is LineageNodeType.EVENT
            ]
            if not events:
                continue
            began = min(node.knowledge_time for node in events)
            elapsed = (decision.knowledge_time - began).total_seconds() * 1000
            if elapsed < 0:
                raise SystemMetricsError("lineage decision predates its source event")
            samples.append(elapsed)
        return samples

    @staticmethod
    def _tokens(value: object) -> tuple[int, int]:
        if value is None:
            return 0, 0
        if not isinstance(value, Mapping):
            raise SystemMetricsError("token_usage must be a mapping or null")
        values = (value.get("input_tokens"), value.get("output_tokens"))
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in values
        ):
            raise SystemMetricsError("token counts must be non-negative integers")
        return values

    @staticmethod
    def _paired_latency(
        events: Sequence[AuditEvent],
        *,
        start: AuditEventType,
        finishes: set[AuditEventType],
        finish_filter,
    ) -> list[float]:
        started = {}
        samples = []
        for event in sorted(events, key=lambda item: item.sequence):
            if event.subject_id is None:
                continue
            if event.event_type is start:
                started[event.subject_id] = event.occurred_at.value
            elif event.event_type in finishes and finish_filter(event):
                began = started.pop(event.subject_id, None)
                if began is None:
                    continue
                elapsed = (event.occurred_at.value - began).total_seconds() * 1000
                if elapsed < 0:
                    raise SystemMetricsError("latency completion predates its start")
                samples.append(elapsed)
        return samples

    @staticmethod
    def _average(values: Sequence[float | int]) -> float | None:
        return sum(values) / len(values) if values else None
