import unittest
from dataclasses import dataclass
from datetime import datetime, timezone

from memory import (
    CreatedAt,
    EventTime,
    HistoricalRecord,
    Instant,
    KnowledgeTime,
    MissingKnowledgeTime,
    SimulationTime,
    parse_knowledge_time,
    visible_as_of,
)


UTC = timezone.utc


def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


class TimeTypeTests(unittest.TestCase):
    def test_naive_datetime_rejected(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeTime(datetime(2024, 1, 1))

    def test_instant_base_cannot_be_constructed(self) -> None:
        with self.assertRaises(TypeError):
            Instant(_dt(2024, 1, 1))

    def test_kinds_are_not_interchangeable(self) -> None:
        event = EventTime(_dt(2024, 6, 30))
        known = KnowledgeTime(_dt(2024, 8, 1))
        self.assertNotEqual(event, known)
        self.assertNotIsInstance(event, KnowledgeTime)

    def test_normalizes_to_utc(self) -> None:
        from datetime import timedelta

        offset = timezone(timedelta(hours=-4))
        raw = datetime(2024, 8, 1, 12, 0, tzinfo=offset)
        known = KnowledgeTime(raw)
        self.assertEqual(known.value.tzinfo, UTC)
        self.assertEqual(known.value, datetime(2024, 8, 1, 16, 0, tzinfo=UTC))


class KnowledgeTimeRequirementTests(unittest.TestCase):
    def test_historical_record_requires_knowledge_time(self) -> None:
        with self.assertRaises(TypeError):
            HistoricalRecord()  # type: ignore[call-arg]

    def test_none_knowledge_time_rejected(self) -> None:
        with self.assertRaises(MissingKnowledgeTime):
            HistoricalRecord(knowledge_time=None)  # type: ignore[arg-type]

    def test_event_time_cannot_stand_in_for_knowledge_time(self) -> None:
        with self.assertRaises(MissingKnowledgeTime):
            HistoricalRecord(knowledge_time=EventTime(_dt(2024, 6, 30)))  # type: ignore[arg-type]

    def test_created_at_cannot_stand_in_for_knowledge_time(self) -> None:
        with self.assertRaises(MissingKnowledgeTime):
            HistoricalRecord(knowledge_time=CreatedAt(_dt(2024, 8, 1)))  # type: ignore[arg-type]

    def test_subclass_cannot_omit_knowledge_time(self) -> None:
        @dataclass(frozen=True, slots=True, kw_only=True)
        class Filing(HistoricalRecord):
            source: str

        with self.assertRaises(TypeError):
            Filing(source="sec")  # type: ignore[call-arg]

        record = Filing(source="sec", knowledge_time=KnowledgeTime(_dt(2024, 8, 1)))
        self.assertEqual(record.knowledge_time, KnowledgeTime(_dt(2024, 8, 1)))

    def test_parse_rejects_missing_value(self) -> None:
        with self.assertRaises(MissingKnowledgeTime):
            parse_knowledge_time(None)


class AsOfTests(unittest.TestCase):
    def test_visible_only_when_knowable(self) -> None:
        known = KnowledgeTime(_dt(2024, 8, 1))
        before = SimulationTime(_dt(2024, 7, 31))
        on = SimulationTime(_dt(2024, 8, 1))
        after = SimulationTime(_dt(2024, 8, 2))
        self.assertFalse(visible_as_of(known, before))
        self.assertTrue(visible_as_of(known, on))
        self.assertTrue(visible_as_of(known, after))

    def test_record_as_of(self) -> None:
        record = HistoricalRecord(knowledge_time=KnowledgeTime(_dt(2024, 8, 1)))
        self.assertFalse(record.visible_as_of(SimulationTime(_dt(2024, 7, 1))))
        self.assertTrue(record.visible_as_of(SimulationTime(_dt(2024, 8, 1))))


if __name__ == "__main__":
    unittest.main()
