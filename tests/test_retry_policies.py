import unittest
from threading import Event

from agents import (
    OperationKind,
    RetryExecutor,
    RetryExhausted,
    RetryPolicy,
    TransientOperationError,
)
from memory import IdempotencyConflict, IdempotencyKey


def policies(overrides=None):
    values = {
        OperationKind.RESEARCH: RetryPolicy(3, None),
        OperationKind.GOVERNANCE: RetryPolicy(
            1, None, requires_idempotency_key=True
        ),
        OperationKind.DETERMINISTIC_RISK: RetryPolicy(
            1, None, requires_idempotency_key=True
        ),
        OperationKind.EXECUTION: RetryPolicy(
            2, None, requires_idempotency_key=True
        ),
    }
    values.update(overrides or {})
    return values


class RetryPolicyTests(unittest.TestCase):
    def test_research_retries_only_transient_failures(self):
        attempts = 0

        def research():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TransientOperationError("temporary")
            return "complete"

        result = RetryExecutor(policies(), sleeper=lambda _: None).execute(
            OperationKind.RESEARCH, research, operation_name="research"
        )
        self.assertEqual(result, "complete")
        self.assertEqual(attempts, 3)

        attempts = 0

        def invalid():
            nonlocal attempts
            attempts += 1
            raise ValueError("malformed output")

        with self.assertRaisesRegex(ValueError, "malformed"):
            RetryExecutor(policies()).execute(
                OperationKind.RESEARCH, invalid, operation_name="invalid"
            )
        self.assertEqual(attempts, 1)

    def test_governance_and_deterministic_risk_fail_after_one_attempt(self):
        for kind in (OperationKind.GOVERNANCE, OperationKind.DETERMINISTIC_RISK):
            with self.subTest(kind=kind):
                attempts = 0

                def failure():
                    nonlocal attempts
                    attempts += 1
                    raise TransientOperationError("temporary")

                with self.assertRaises(RetryExhausted):
                    RetryExecutor(policies()).execute(
                        kind,
                        failure,
                        operation_name=kind.value,
                        idempotency_key=IdempotencyKey(f"key:{kind.value}"),
                    )
                self.assertEqual(attempts, 1)

    def test_execution_requires_a_key_and_replays_one_result(self):
        executor = RetryExecutor(policies())
        calls = 0

        def execute_trade():
            nonlocal calls
            calls += 1
            return {"fill_id": "fill:1"}

        with self.assertRaisesRegex(ValueError, "idempotency key"):
            executor.execute(
                OperationKind.EXECUTION,
                execute_trade,
                operation_name="submit",
            )
        key = IdempotencyKey("trade:1")
        first = executor.execute(
            OperationKind.EXECUTION,
            execute_trade,
            operation_name="submit",
            idempotency_key=key,
        )
        second = executor.execute(
            OperationKind.EXECUTION,
            execute_trade,
            operation_name="submit",
            idempotency_key=key,
        )
        self.assertIs(first, second)
        self.assertEqual(calls, 1)
        with self.assertRaises(IdempotencyConflict):
            executor.execute(
                OperationKind.EXECUTION,
                execute_trade,
                operation_name="different trade",
                idempotency_key=key,
            )

    def test_timeout_is_bounded_and_reported_as_exhausted(self):
        release = Event()
        timed = policies({OperationKind.RESEARCH: RetryPolicy(1, 0.01)})
        with self.assertRaises(RetryExhausted):
            RetryExecutor(timed).execute(
                OperationKind.RESEARCH,
                lambda: release.wait(0.1),
                operation_name="slow research",
            )
        release.set()


if __name__ == "__main__":
    unittest.main()
