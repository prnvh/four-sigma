from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CanonicalId:
    value: str

    def __post_init__(self) -> None:
        if type(self) is CanonicalId:
            raise TypeError("use a concrete ID type")
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError(f"{type(self).__name__} must be a non-empty string")

    def __str__(self) -> str:
        return self.value


class AgentId(CanonicalId):
    __slots__ = ()


class RunId(CanonicalId):
    __slots__ = ()


class EventId(CanonicalId):
    __slots__ = ()


class AuditEventId(CanonicalId):
    __slots__ = ()


class InsightId(CanonicalId):
    __slots__ = ()


class ProposalId(CanonicalId):
    __slots__ = ()


class TradeCandidateId(CanonicalId):
    __slots__ = ()


class DecisionId(CanonicalId):
    __slots__ = ()


class EntityId(CanonicalId):
    __slots__ = ()
