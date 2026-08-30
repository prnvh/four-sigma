from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Real
from time import perf_counter
from types import MappingProxyType

from memory.strategy_metrics import StrategyMetrics

from .backtest import BacktestResult


class ExperimentConfigurationError(ValueError):
    """An experiment matrix is ambiguous or cannot produce a fair comparison."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentVariant:
    name: str
    agent_versions: tuple[str, ...]
    strategy_config: Mapping[str, object]
    model_labels: Mapping[str, str] = MappingProxyType({})
    prompt_labels: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        name = self.name.strip() if isinstance(self.name, str) else ""
        if not name:
            raise ValueError("experiment name must be non-empty")
        if not isinstance(self.agent_versions, tuple) or any(
            not isinstance(item, str) or ":" not in item
            for item in self.agent_versions
        ):
            raise ValueError("agent_versions must contain name:version strings")
        if len(set(self.agent_versions)) != len(self.agent_versions):
            raise ValueError("agent_versions cannot contain duplicates")
        if not isinstance(self.strategy_config, Mapping):
            raise TypeError("strategy_config must be a mapping")
        models = self._labels(self.model_labels, "model_labels")
        prompts = self._labels(self.prompt_labels, "prompt_labels")
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self, "strategy_config", MappingProxyType(deepcopy(dict(self.strategy_config)))
        )
        object.__setattr__(self, "model_labels", MappingProxyType(models))
        object.__setattr__(self, "prompt_labels", MappingProxyType(prompts))

    def without_agent(self, agent_name: str, *, name: str | None = None) -> ExperimentVariant:
        target = agent_name.strip() if isinstance(agent_name, str) else ""
        if not target:
            raise ValueError("agent_name must be non-empty")
        selected = tuple(
            key for key in self.agent_versions if key.split(":", 1)[0] != target
        )
        if selected == self.agent_versions:
            raise ExperimentConfigurationError(
                f"cannot ablate absent agent: {target}"
            )
        return replace(
            self,
            name=name or f"without_{target}",
            agent_versions=selected,
            model_labels={
                key: value for key, value in self.model_labels.items() if key != target
            },
            prompt_labels={
                key: value for key, value in self.prompt_labels.items() if key != target
            },
        )

    def with_strategy(
        self, name: str, **overrides: object
    ) -> ExperimentVariant:
        return replace(
            self,
            name=name,
            strategy_config={**self.strategy_config, **overrides},
        )

    def with_models(self, name: str, **labels: str) -> ExperimentVariant:
        return replace(
            self,
            name=name,
            model_labels={**self.model_labels, **labels},
        )

    def with_prompts(self, name: str, **labels: str) -> ExperimentVariant:
        return replace(
            self,
            name=name,
            prompt_labels={**self.prompt_labels, **labels},
        )

    @staticmethod
    def _labels(value: Mapping[str, str], name: str) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        normalized = {}
        for key, label in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{name} keys must be non-empty strings")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{name} values must be non-empty strings")
            normalized[key.strip()] = label.strip()
        return normalized


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    variant: ExperimentVariant
    backtest: BacktestResult
    metric_deltas: Mapping[str, float | None]
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class ExperimentFailure:
    variant: ExperimentVariant
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class BatchExperimentReport:
    baseline_name: str
    succeeded: tuple[ExperimentResult, ...]
    failed: tuple[ExperimentFailure, ...]

    def result(self, name: str) -> ExperimentResult:
        for item in self.succeeded:
            if item.variant.name == name:
                return item
        raise KeyError(f"no successful experiment named {name!r}")


class BatchExperimentRunner:
    """Run a deterministic ablation matrix and compare every result to baseline."""

    def __init__(
        self,
        execute: Callable[[ExperimentVariant], BacktestResult],
        *,
        fail_fast: bool = False,
    ) -> None:
        if not callable(execute):
            raise TypeError("execute must be callable")
        if not isinstance(fail_fast, bool):
            raise TypeError("fail_fast must be bool")
        self.execute = execute
        self.fail_fast = fail_fast

    def run(
        self,
        variants: Sequence[ExperimentVariant],
        *,
        baseline_name: str = "baseline",
    ) -> BatchExperimentReport:
        selected = tuple(variants)
        if not selected or any(not isinstance(item, ExperimentVariant) for item in selected):
            raise TypeError("variants must contain ExperimentVariant values")
        names = [item.name for item in selected]
        if len(set(names)) != len(names):
            raise ExperimentConfigurationError("experiment names must be unique")
        if names.count(baseline_name) != 1:
            raise ExperimentConfigurationError(
                "the experiment matrix must contain exactly one named baseline"
            )

        completed: list[tuple[ExperimentVariant, BacktestResult, float]] = []
        failures: list[ExperimentFailure] = []
        for variant in selected:
            started = perf_counter()
            try:
                result = self.execute(variant)
                if not isinstance(result, BacktestResult):
                    raise TypeError("experiment executor must return BacktestResult")
                completed.append((variant, result, (perf_counter() - started) * 1000))
            except Exception as error:
                if self.fail_fast:
                    raise
                failures.append(ExperimentFailure(
                    variant=variant,
                    error_type=type(error).__name__,
                    message=str(error),
                ))

        baseline = next(
            (result for variant, result, _ in completed if variant.name == baseline_name),
            None,
        )
        if baseline is None:
            raise ExperimentConfigurationError("baseline experiment failed")
        succeeded = tuple(
            ExperimentResult(
                variant=variant,
                backtest=result,
                metric_deltas=MappingProxyType(
                    self._metric_deltas(result.metrics, baseline.metrics)
                ),
                elapsed_ms=elapsed,
            )
            for variant, result, elapsed in completed
        )
        return BatchExperimentReport(baseline_name, succeeded, tuple(failures))

    @staticmethod
    def _metric_deltas(
        result: StrategyMetrics, baseline: StrategyMetrics
    ) -> dict[str, float | None]:
        deltas: dict[str, float | None] = {}
        for name in StrategyMetrics.__dataclass_fields__:
            value = getattr(result, name)
            reference = getattr(baseline, name)
            deltas[name] = (
                float(value - reference)
                if isinstance(value, Real)
                and not isinstance(value, bool)
                and isinstance(reference, Real)
                and not isinstance(reference, bool)
                else None
            )
        return deltas
