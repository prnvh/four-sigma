import unittest
from datetime import datetime, timedelta, timezone

from memory import (
    ClockError,
    EventTime,
    KnowledgeTime,
    SimulationClock,
    SimulationTime,
)


UTC = timezone.utc
START = SimulationTime(datetime(2024, 1, 1, tzinfo=UTC))


def _sim(year: int, month: int, day: int, hour: int = 0) -> SimulationTime:
    return SimulationTime(datetime(year, month, day, hour, tzinfo=UTC))


def _run(clock: SimulationClock) -> list[datetime]:
    stamps = [clock.now().value]
    stamps.append(clock.advance_by(timedelta(days=1)).value)
    stamps.append(clock.advance_to(_sim(2024, 1, 15)).value)
    stamps.append(clock.advance_by(timedelta(hours=6)).value)
    stamps.append(clock.now().value)
    return stamps


class SimulationClockTests(unittest.TestCase):
    def test_starts_at_explicit_time(self) -> None:
        clock = SimulationClock(START)
        self.assertEqual(clock.now(), START)

    def test_requires_simulation_time(self) -> None:
        with self.assertRaises(TypeError):
            SimulationClock(EventTime(START.value))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            SimulationClock(KnowledgeTime(START.value))  # type: ignore[arg-type]

    def test_advance_to_moves_forward(self) -> None:
        clock = SimulationClock(START)
        later = _sim(2024, 3, 1)
        self.assertEqual(clock.advance_to(later), later)
        self.assertEqual(clock.now(), later)

    def test_advance_to_same_instant_is_allowed(self) -> None:
        clock = SimulationClock(START)
        self.assertEqual(clock.advance_to(START), START)

    def test_advance_to_rejects_backward(self) -> None:
        clock = SimulationClock(_sim(2024, 6, 1))
        with self.assertRaises(ClockError):
            clock.advance_to(START)

    def test_advance_to_rejects_other_time_kinds(self) -> None:
        clock = SimulationClock(START)
        with self.assertRaises(TypeError):
            clock.advance_to(KnowledgeTime(START.value))  # type: ignore[arg-type]

    def test_advance_by(self) -> None:
        clock = SimulationClock(START)
        self.assertEqual(clock.advance_by(timedelta(days=2)), _sim(2024, 1, 3))

    def test_advance_by_zero_is_allowed(self) -> None:
        clock = SimulationClock(START)
        self.assertEqual(clock.advance_by(timedelta(0)), START)

    def test_advance_by_rejects_negative(self) -> None:
        clock = SimulationClock(START)
        with self.assertRaises(ClockError):
            clock.advance_by(timedelta(seconds=-1))

    def test_replay_yields_identical_timestamps_and_order(self) -> None:
        first = _run(SimulationClock(START))
        second = _run(SimulationClock(START))
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
