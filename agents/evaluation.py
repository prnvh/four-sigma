from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from memory.audit_logger import AuditEvent, AuditEventType, AuditLedger
from memory.types import AgentId, Direction, RunId


class AgentEvaluationError(ValueError):
    """Audit or outcome data cannot produce a trustworthy evaluation."""


def _non_negative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentEvaluationError(f"{name} must be numeric")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise AgentEvaluationError(f"{name} must be finite and non-negative")
    return number


@dataclass(frozen=True, slots=True)
class ModelTokenPrice:
    input_per_million_usd: float
    output_per_million_usd: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_per_million_usd",
            _non_negative(self.input_per_million_usd, "input_per_million_usd"),
        )
        object.__setattr__(
            self,
            "output_per_million_usd",
            _non_negative(self.output_per_million_usd, "output_per_million_usd"),
        )


@dataclass(frozen=True, slots=True)
class AgentPredictionOutcome:
    run_id: RunId
    confidence: float
    predicted_direction: Direction
    realized_direction: Direction

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, RunId):
            raise TypeError("run_id must be RunId")
        confidence = _non_negative(self.confidence, "confidence")
        if confidence > 1:
            raise AgentEvaluationError("confidence must be between 0 and 1")
        if not isinstance(self.predicted_direction, Direction):
            raise TypeError("predicted_direction must be Direction")
        if not isinstance(self.realized_direction, Direction):
            raise TypeError("realized_direction must be Direction")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True, slots=True)
class AgentEvaluation:
    agent_id: AgentId
    agent_version: str
    model: str
    runs: int
    successful_runs: int
    failed_runs: int
    accepted_proposals: int
    rejected_proposals: int
    confidence_brier_score: float | None
    direction_accuracy: float | None
    average_latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None


@dataclass(slots=True)
class _Aggregate:
    agent_id: AgentId
    version: str
    model: str
    runs: int = 0
    succeeded: int = 0
    failed: int = 0
    accepted: int = 0
    rejected: int = 0
    latency: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    priced_cost: float = 0.0
    has_unpriced_tokens: bool = False
    correctness: list[float] | None = None
    brier: list[float] | None = None

    def __post_init__(self) -> None:
        self.correctness = []
        self.brier = []


class AgentEvaluator:
    """Derive reproducible per-version agent metrics from audit and outcomes."""

    def __init__(self, pricing: Mapping[str, ModelTokenPrice] | None = None) -> None:
        self._pricing = dict(pricing or {})
        if any(
            not isinstance(model, str) or not model.strip()
            or not isinstance(price, ModelTokenPrice)
            for model, price in self._pricing.items()
        ):
            raise TypeError("pricing must map model names to ModelTokenPrice")

    def evaluate(
        self,
        audit: AuditLedger | Sequence[AuditEvent],
        *,
        outcomes: Sequence[AgentPredictionOutcome] = (),
    ) -> tuple[AgentEvaluation, ...]:
        events = audit.snapshot() if isinstance(audit, AuditLedger) else tuple(audit)
        if any(not isinstance(event, AuditEvent) for event in events):
            raise TypeError("audit must contain AuditEvent values")
        outcome_by_run: dict[RunId, AgentPredictionOutcome] = {}
        for outcome in outcomes:
            if not isinstance(outcome, AgentPredictionOutcome):
                raise TypeError("outcomes must contain AgentPredictionOutcome values")
            if outcome.run_id in outcome_by_run:
                raise AgentEvaluationError(f"duplicate outcome for run: {outcome.run_id}")
            outcome_by_run[outcome.run_id] = outcome

        groups: dict[tuple[str, str, str], _Aggregate] = {}
        run_groups: dict[RunId, tuple[str, str, str]] = {}
        for event in events:
            if event.event_type is not AuditEventType.AGENT_RUN_FINISHED:
                continue
            if event.agent_id is None or event.run_id is None:
                raise AgentEvaluationError("finished run requires agent_id and run_id")
            if event.run_id in run_groups:
                raise AgentEvaluationError(f"duplicate finished run: {event.run_id}")
            version = event.details.get("agent_version")
            model = event.details.get("model")
            status = event.details.get("status")
            if not isinstance(version, str) or not version.strip():
                raise AgentEvaluationError("finished run requires agent_version")
            if not isinstance(model, str) or not model.strip():
                raise AgentEvaluationError("finished run requires model")
            if status not in {"succeeded", "failed"}:
                raise AgentEvaluationError("finished run has invalid status")
            key = (event.agent_id.value, version, model)
            group = groups.setdefault(key, _Aggregate(event.agent_id, version, model))
            run_groups[event.run_id] = key
            group.runs += 1
            group.succeeded += status == "succeeded"
            group.failed += status == "failed"
            group.latency += _non_negative(event.details.get("latency_ms"), "latency_ms")
            input_tokens, output_tokens = self._tokens(event.details.get("token_usage"))
            group.input_tokens += input_tokens
            group.output_tokens += output_tokens
            price = self._pricing.get(model)
            if price is None and input_tokens + output_tokens:
                group.has_unpriced_tokens = True
            elif price is not None:
                group.priced_cost += (
                    input_tokens * price.input_per_million_usd
                    + output_tokens * price.output_per_million_usd
                ) / 1_000_000

        for event in events:
            if event.run_id not in run_groups:
                continue
            group = groups[run_groups[event.run_id]]
            if event.event_type is AuditEventType.PROMOTION_APPROVED:
                group.accepted += 1
            elif event.event_type is AuditEventType.PROMOTION_REJECTED:
                group.rejected += 1

        unknown = set(outcome_by_run) - set(run_groups)
        if unknown:
            raise AgentEvaluationError(
                f"outcomes reference unknown finished runs: {sorted(item.value for item in unknown)}"
            )
        for run_id, outcome in outcome_by_run.items():
            group = groups[run_groups[run_id]]
            correct = float(outcome.predicted_direction is outcome.realized_direction)
            assert group.correctness is not None and group.brier is not None
            group.correctness.append(correct)
            group.brier.append((outcome.confidence - correct) ** 2)

        results = []
        for key in sorted(groups):
            group = groups[key]
            assert group.correctness is not None and group.brier is not None
            results.append(AgentEvaluation(
                agent_id=group.agent_id,
                agent_version=group.version,
                model=group.model,
                runs=group.runs,
                successful_runs=group.succeeded,
                failed_runs=group.failed,
                accepted_proposals=group.accepted,
                rejected_proposals=group.rejected,
                confidence_brier_score=(
                    sum(group.brier) / len(group.brier) if group.brier else None
                ),
                direction_accuracy=(
                    sum(group.correctness) / len(group.correctness)
                    if group.correctness else None
                ),
                average_latency_ms=group.latency / group.runs,
                input_tokens=group.input_tokens,
                output_tokens=group.output_tokens,
                total_tokens=group.input_tokens + group.output_tokens,
                estimated_cost_usd=(
                    None if group.has_unpriced_tokens else group.priced_cost
                ),
            ))
        return tuple(results)

    @staticmethod
    def _tokens(value: object) -> tuple[int, int]:
        if value is None:
            return 0, 0
        if not isinstance(value, Mapping):
            raise AgentEvaluationError("token_usage must be a mapping or null")
        input_tokens = value.get("input_tokens")
        output_tokens = value.get("output_tokens")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (input_tokens, output_tokens)
        ):
            raise AgentEvaluationError("token counts must be non-negative integers")
        return input_tokens, output_tokens
