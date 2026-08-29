from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


class MissingKnowledgeTime(ValueError):
    """Historical data was constructed without a KnowledgeTime."""


@dataclass(frozen=True, slots=True)
class Instant:
    value: datetime

    def __post_init__(self) -> None:
        if type(self) is Instant:
            raise TypeError(
                "use EventTime, KnowledgeTime, SimulationTime, or CreatedAt"
            )
        if not isinstance(self.value, datetime):
            raise TypeError(
                f"{type(self).__name__} requires datetime, got {type(self.value).__name__}"
            )
        if self.value.tzinfo is None:
            raise ValueError(f"{type(self).__name__} must be timezone-aware")
        utc = self.value.astimezone(timezone.utc)
        object.__setattr__(self, "value", utc)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return self.value <= other.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return self.value < other.value


class EventTime(Instant):
    """When the fact occurred in the world.

    Earnings quarter-end, trade print, news occurrence.
    Never use this as the as-of filter.
    """

    __slots__ = ()


class KnowledgeTime(Instant):
    """When the fact became knowable.

    Earnings release, publication, revision availability.
    Historical queries and agent context filter on this.
    """

    __slots__ = ()


class SimulationTime(Instant):
    """Current time of the deterministic simulation clock.

    Agents may only observe records with knowledge_time <= simulation_time.
    """

    __slots__ = ()


class CreatedAt(Instant):
    """When the record was written into this system.

    Distinct from event_time and knowledge_time. Must be supplied
    explicitly — never read from the wall clock.
    """

    __slots__ = ()


def parse_knowledge_time(value: object) -> KnowledgeTime:
    if value is None:
        raise MissingKnowledgeTime("historical data requires knowledge_time")
    if isinstance(value, KnowledgeTime):
        return value
    if isinstance(value, Instant):
        raise MissingKnowledgeTime(
            f"historical data requires KnowledgeTime, got {type(value).__name__}"
        )
    if isinstance(value, datetime):
        return KnowledgeTime(value)
    raise TypeError(
        f"knowledge_time must be KnowledgeTime or datetime, got {type(value).__name__}"
    )


def visible_as_of(
    knowledge_time: KnowledgeTime, simulation_time: SimulationTime
) -> bool:
    """True when the record was knowable at simulation_time."""
    return knowledge_time <= simulation_time


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoricalRecord:
    """Any record of the past. knowledge_time has no default."""

    knowledge_time: KnowledgeTime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "knowledge_time", parse_knowledge_time(self.knowledge_time)
        )

    def visible_as_of(self, simulation_time: SimulationTime) -> bool:
        return visible_as_of(self.knowledge_time, simulation_time)
