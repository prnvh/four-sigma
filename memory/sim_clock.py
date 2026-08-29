from __future__ import annotations

from datetime import timedelta

from .types import SimulationTime


class ClockError(ValueError):
    """Clock moved backward or received an invalid increment."""


class SimulationClock:
    """Deterministic simulation time. Never reads the wall clock."""

    def __init__(self, start: SimulationTime) -> None:
        if not isinstance(start, SimulationTime):
            raise TypeError(
                f"SimulationClock start must be SimulationTime, got {type(start).__name__}"
            )
        self._now = start

    def now(self) -> SimulationTime:
        return self._now

    def advance_to(self, time: SimulationTime) -> SimulationTime:
        if not isinstance(time, SimulationTime):
            raise TypeError(
                f"advance_to requires SimulationTime, got {type(time).__name__}"
            )
        if time < self._now:
            raise ClockError(
                f"cannot move clock backward from {self._now.value.isoformat()} "
                f"to {time.value.isoformat()}"
            )
        self._now = time
        return self._now

    def advance_by(self, delta: timedelta) -> SimulationTime:
        if not isinstance(delta, timedelta):
            raise TypeError(
                f"advance_by requires timedelta, got {type(delta).__name__}"
            )
        if delta < timedelta(0):
            raise ClockError("cannot advance clock by a negative duration")
        return self.advance_to(SimulationTime(self._now.value + delta))
