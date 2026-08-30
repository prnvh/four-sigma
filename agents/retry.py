from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from time import sleep
from typing import TypeVar

from memory.types import IdempotencyConflict, IdempotencyKey


T = TypeVar("T")


class OperationKind(StrEnum):
    RESEARCH = "research"
    GOVERNANCE = "governance"
    DETERMINISTIC_RISK = "deterministic_risk"
    EXECUTION = "execution"


class TransientOperationError(RuntimeError):
    """A temporary dependency failure that may be retried."""


class OperationTimeout(TransientOperationError):
    """An operation exceeded its configured deadline."""


class RetryExhausted(RuntimeError):
    """All permitted attempts failed with a transient error."""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    timeout_seconds: float | None
    backoff_seconds: float = 0.0
    requires_idempotency_key: bool = False

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive or None")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")


DEFAULT_RETRY_POLICIES: Mapping[OperationKind, RetryPolicy] = {
    OperationKind.RESEARCH: RetryPolicy(3, 30.0, 0.25),
    OperationKind.GOVERNANCE: RetryPolicy(1, None, requires_idempotency_key=True),
    OperationKind.DETERMINISTIC_RISK: RetryPolicy(
        1, None, requires_idempotency_key=True
    ),
    OperationKind.EXECUTION: RetryPolicy(
        2, 10.0, 0.10, requires_idempotency_key=True
    ),
}


class RetryExecutor:
    """Applies bounded retries and remembers completed keyed mutations."""

    def __init__(
        self,
        policies: Mapping[OperationKind, RetryPolicy] | None = None,
        *,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self._policies = dict(policies or DEFAULT_RETRY_POLICIES)
        if set(self._policies) != set(OperationKind):
            raise ValueError("a retry policy is required for every operation kind")
        self._sleeper = sleeper
        self._completed: dict[
            tuple[OperationKind, IdempotencyKey], tuple[str, object]
        ] = {}
        self._lock = RLock()

    def reset_completed(self) -> None:
        """Start a new orchestration run without changing its policies."""
        with self._lock:
            self._completed.clear()

    def execute(
        self,
        kind: OperationKind,
        operation: Callable[[], T],
        *,
        operation_name: str,
        idempotency_key: IdempotencyKey | None = None,
    ) -> T:
        if not isinstance(kind, OperationKind):
            raise TypeError("kind must be OperationKind")
        if not callable(operation):
            raise TypeError("operation must be callable")
        if not isinstance(operation_name, str) or not operation_name.strip():
            raise ValueError("operation_name must be non-empty")
        policy = self._policies[kind]
        if policy.requires_idempotency_key and not isinstance(
            idempotency_key, IdempotencyKey
        ):
            raise ValueError(f"{kind.value} requires an idempotency key")

        cache_key = (kind, idempotency_key) if idempotency_key is not None else None
        if cache_key is not None:
            with self._lock:
                completed = self._completed.get(cache_key)
                if completed is not None:
                    if completed[0] != operation_name.strip():
                        raise IdempotencyConflict(
                            f"idempotency key reused for a different operation: "
                            f"{idempotency_key.value}"
                        )
                    return completed[1]  # type: ignore[return-value]
                result = self._attempt(operation, operation_name, policy)
                self._completed[cache_key] = (operation_name.strip(), result)
                return result
        return self._attempt(operation, operation_name, policy)

    def _attempt(
        self,
        operation: Callable[[], T],
        operation_name: str,
        policy: RetryPolicy,
    ) -> T:
        last_error: BaseException | None = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                result = self._run_once(operation, policy.timeout_seconds)
            except (TransientOperationError, TimeoutError, ConnectionError) as error:
                last_error = error
                if attempt == policy.max_attempts:
                    break
                if policy.backoff_seconds:
                    self._sleeper(policy.backoff_seconds * attempt)
                continue
            return result
        raise RetryExhausted(
            f"{operation_name.strip()} failed after {policy.max_attempts} attempt(s)"
        ) from last_error

    @staticmethod
    def _run_once(operation: Callable[[], T], timeout_seconds: float | None) -> T:
        if timeout_seconds is None:
            return operation()
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(operation)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout as error:
            future.cancel()
            raise OperationTimeout(
                f"operation exceeded {timeout_seconds:g} seconds"
            ) from error
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


_RESEARCH_RETRIES = RetryExecutor()


def generate_json_with_retry(
    model: object,
    *,
    instructions: str,
    input_data: dict[str, object],
    schema: dict[str, object],
    operation_name: str,
) -> object:
    return _RESEARCH_RETRIES.execute(
        OperationKind.RESEARCH,
        lambda: model.generate_json(  # type: ignore[attr-defined]
            instructions=instructions,
            input_data=input_data,
            schema=schema,
        ),
        operation_name=operation_name,
    )
