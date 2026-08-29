from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Generic, TypeVar

from .orchestration import PlannedAgentRun


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ConcurrentAgentResult(Generic[T]):
    plan: PlannedAgentRun
    value: T


@dataclass(frozen=True, slots=True)
class ConcurrentAgentFailure:
    plan: PlannedAgentRun
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class ConcurrentBatchResult(Generic[T]):
    succeeded: tuple[ConcurrentAgentResult[T], ...]
    failed: tuple[ConcurrentAgentFailure, ...]


class ConcurrentAgentExecutor:
    """Parallelize model work, then serialize successful state mutations."""

    def __init__(self, max_workers: int = 4) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int):
            raise TypeError("max_workers must be an integer")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers

    def execute(
        self,
        plans: Sequence[PlannedAgentRun],
        *,
        compute: Callable[[PlannedAgentRun], T],
        commit: Callable[[ConcurrentAgentResult[T]], None] | None = None,
    ) -> ConcurrentBatchResult[T]:
        if any(not isinstance(plan, PlannedAgentRun) for plan in plans):
            raise TypeError("plans must contain PlannedAgentRun values")
        if not callable(compute):
            raise TypeError("compute must be callable")
        if commit is not None and not callable(commit):
            raise TypeError("commit must be callable or None")
        ordered = tuple(sorted(plans, key=self._plan_key))
        if len(set(ordered)) != len(ordered):
            raise ValueError("duplicate planned agent runs are not allowed")
        if not ordered:
            return ConcurrentBatchResult((), ())

        completed: dict[PlannedAgentRun, T] = {}
        failures: dict[PlannedAgentRun, ConcurrentAgentFailure] = {}
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(ordered)),
            thread_name_prefix="qfirm-agent",
        ) as pool:
            futures = {pool.submit(compute, plan): plan for plan in ordered}
            for future in as_completed(futures):
                plan = futures[future]
                try:
                    completed[plan] = future.result()
                except Exception as error:
                    failures[plan] = ConcurrentAgentFailure(
                        plan=plan,
                        error_type=type(error).__name__,
                        message=str(error),
                    )

        succeeded = tuple(
            ConcurrentAgentResult(plan, completed[plan])
            for plan in ordered
            if plan in completed
        )
        if commit is not None:
            for result in succeeded:
                commit(result)
        return ConcurrentBatchResult(
            succeeded=succeeded,
            failed=tuple(failures[plan] for plan in ordered if plan in failures),
        )

    @staticmethod
    def _plan_key(plan: PlannedAgentRun) -> tuple[object, ...]:
        return (
            plan.simulation_time,
            plan.agent_key,
            plan.entity_id,
            tuple(str(item) for item in plan.triggers),
            plan.source_refs,
        )
