import unittest
from datetime import datetime, timezone

from memory import (
    AuditEventId,
    AuditEventType,
    AuditLedger,
    CreatedAt,
    SharedMemory,
    SharedMemorySection,
    SharedMemoryValidationError,
    UnknownSharedMemorySection,
)


UTC = timezone.utc
WRITTEN_AT = CreatedAt(datetime(2025, 1, 1, tzinfo=UTC))


class SharedMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = AuditLedger()
        self.memory = SharedMemory(self.ledger)
        self.audit_number = 0

    def _write(
        self,
        section: SharedMemorySection | str,
        record_key: str = "record-1",
        value: dict[str, object] | None = None,
    ):
        self.audit_number += 1
        return self.memory.write(
            section=section,
            record_key=record_key,
            value=value if value is not None else {"status": "active"},
            audit_event_id=AuditEventId(f"audit-{self.audit_number}"),
            occurred_at=WRITTEN_AT,
        )

    def test_schema_contains_exactly_the_seven_sections(self) -> None:
        self.assertEqual(
            {section.value for section in SharedMemorySection},
            {
                "events",
                "entities",
                "insights",
                "portfolio",
                "risk",
                "trade_candidates",
                "decisions",
            },
        )

    def test_each_known_section_accepts_a_write(self) -> None:
        for section in SharedMemorySection:
            self._write(section)
            self.assertEqual(self.memory.read(section, "record-1")["status"], "active")

    def test_unknown_section_is_rejected_and_audited(self) -> None:
        with self.assertRaises(UnknownSharedMemorySection):
            self._write("notes")

        events = self.ledger.snapshot()
        self.assertEqual(len(events), 1)
        self.assertEqual(
            events[0].event_type,
            AuditEventType.SHARED_MEMORY_WRITE_REJECTED,
        )
        self.assertEqual(events[0].details["section"], "notes")
        for section in SharedMemorySection:
            self.assertEqual(dict(self.memory.snapshot(section)), {})

    def test_section_names_are_exact(self) -> None:
        for invalid in ("event", "Events", "trade-candidates", " risk"):
            with self.subTest(section=invalid):
                with self.assertRaises(UnknownSharedMemorySection):
                    self._write(invalid)

    def test_value_must_be_a_mapping_with_string_keys(self) -> None:
        with self.assertRaises(SharedMemoryValidationError):
            self._write(
                SharedMemorySection.RISK,
                value=["high"],  # type: ignore[arg-type]
            )
        with self.assertRaises(SharedMemoryValidationError):
            self._write(
                SharedMemorySection.RISK,
                value={1: "high"},  # type: ignore[dict-item]
            )

        self.assertEqual(len(self.ledger.snapshot()), 2)
        self.assertTrue(
            all(
                event.event_type is AuditEventType.SHARED_MEMORY_WRITE_REJECTED
                for event in self.ledger.snapshot()
            )
        )

    def test_record_key_must_be_non_empty(self) -> None:
        with self.assertRaises(SharedMemoryValidationError):
            self._write(SharedMemorySection.INSIGHTS, record_key="   ")
        self.assertEqual(
            self.ledger.snapshot()[0].event_type,
            AuditEventType.SHARED_MEMORY_WRITE_REJECTED,
        )

    def test_successful_write_generates_audit_event(self) -> None:
        self._write(SharedMemorySection.INSIGHTS, record_key="insight-1")

        events = self.ledger.snapshot()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, AuditEventType.SHARED_MEMORY_UPDATED)
        self.assertEqual(events[0].details["section"], "insights")
        self.assertEqual(events[0].details["record_key"], "insight-1")

    def test_rejected_event_overwrite_adds_rejection_audit_event(self) -> None:
        self._write(SharedMemorySection.EVENTS, value={"headline": "original"})

        with self.assertRaises(SharedMemoryValidationError):
            self._write(SharedMemorySection.EVENTS, value={"headline": "changed"})

        self.assertEqual(len(self.ledger.snapshot()), 2)
        self.assertEqual(
            self.ledger.snapshot()[1].event_type,
            AuditEventType.SHARED_MEMORY_WRITE_REJECTED,
        )
        self.assertEqual(
            self.memory.read(SharedMemorySection.EVENTS, "record-1")["headline"],
            "original",
        )

    def test_non_event_records_can_be_replaced_and_each_write_is_audited(self) -> None:
        self._write(SharedMemorySection.RISK, value={"status": "pending"})
        self._write(SharedMemorySection.RISK, value={"status": "checked"})

        self.assertEqual(
            self.memory.read(SharedMemorySection.RISK, "record-1")["status"],
            "checked",
        )
        self.assertEqual(len(self.ledger.snapshot()), 2)

    def test_caller_mutation_cannot_change_stored_record(self) -> None:
        value = {"factors": ["liquidity"]}
        self._write(SharedMemorySection.RISK, value=value)
        value["factors"].append("concentration")
        returned = self.memory.read(SharedMemorySection.RISK, "record-1")
        returned["factors"].append("volatility")

        self.assertEqual(
            self.memory.read(SharedMemorySection.RISK, "record-1")["factors"],
            ["liquidity"],
        )


if __name__ == "__main__":
    unittest.main()
