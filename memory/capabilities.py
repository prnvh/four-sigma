from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .types import AgentId


class Action(StrEnum):
    READ = "READ"
    WORKING_WRITE = "WORKING_WRITE"
    PROPOSE_SHARED_WRITE = "PROPOSE_SHARED_WRITE"
    VETO = "VETO"
    EXECUTE = "EXECUTE"


class AuthorizationError(PermissionError):
    """An agent asked for an action, resource, or field it does not have."""


def _agent_name(agent: AgentId | str) -> str:
    value = agent.value if isinstance(agent, AgentId) else agent
    if not isinstance(value, str) or not value.strip():
        raise ValueError("agent must be a non-empty string")
    return value.strip().split(":", 1)[0]


def _field_pair(resource: object, field: object) -> tuple[str, str]:
    if not isinstance(resource, str) or not resource.strip():
        raise ValueError("resource must be a non-empty string")
    if not isinstance(field, str) or not field.strip():
        raise ValueError("field must be a non-empty string")
    return resource.strip(), field.strip()


@dataclass(frozen=True, slots=True)
class AgentCapabilities:
    """Declarative grants for one agent. Missing actions are empty."""

    agent: str
    grants: Mapping[Action, frozenset[tuple[str, str]]]

    def __post_init__(self) -> None:
        name = _agent_name(self.agent)
        grants = {
            action: frozenset(_field_pair(resource, field) for resource, field in self.grants.get(action, ()))
            for action in Action
        }
        object.__setattr__(self, "agent", name)
        object.__setattr__(self, "grants", MappingProxyType(grants))

    def allows(self, action: Action, resource: str, field: str) -> bool:
        if not isinstance(action, Action):
            raise TypeError(f"action must be Action, got {type(action).__name__}")
        resource, field = _field_pair(resource, field)
        granted = self.grants[action]
        return (resource, field) in granted or (resource, "*") in granted


class CapabilityModel:
    """Deny-by-default catalog. authorize() never consults a prompt."""

    def __init__(self, agents: Sequence[AgentCapabilities]) -> None:
        catalog: dict[str, AgentCapabilities] = {}
        for spec in agents:
            if not isinstance(spec, AgentCapabilities):
                raise TypeError("catalog entries must be AgentCapabilities")
            if spec.agent in catalog:
                raise ValueError(f"duplicate capability entry: {spec.agent}")
            catalog[spec.agent] = spec
        self._agents = MappingProxyType(catalog)

    def capabilities_for(self, agent: AgentId | str) -> AgentCapabilities | None:
        return self._agents.get(_agent_name(agent))

    def authorize(
        self,
        agent: AgentId | str,
        action: Action,
        resource: str,
        field: str,
    ) -> bool:
        spec = self.capabilities_for(agent)
        if spec is None:
            if not isinstance(action, Action):
                raise TypeError(f"action must be Action, got {type(action).__name__}")
            _field_pair(resource, field)
            return False
        return spec.allows(action, resource, field)

    def require(
        self,
        agent: AgentId | str,
        action: Action,
        resource: str,
        field: str,
    ) -> None:
        if not self.authorize(agent, action, resource, field):
            name = _agent_name(agent)
            raise AuthorizationError(
                f"{name} cannot {action.value} {resource}.{field}"
            )

    def require_reads(
        self, agent: AgentId | str, fields: Sequence[tuple[str, str]]
    ) -> None:
        for resource, field in fields:
            self.require(agent, Action.READ, resource, field)


def _spec(
    agent: str,
    *,
    read: tuple[tuple[str, str], ...] = (),
    working_write: tuple[tuple[str, str], ...] = (),
    propose: tuple[tuple[str, str], ...] = (),
    veto: tuple[tuple[str, str], ...] = (),
    execute: tuple[tuple[str, str], ...] = (),
) -> AgentCapabilities:
    return AgentCapabilities(
        agent,
        {
            Action.READ: frozenset(read),
            Action.WORKING_WRITE: frozenset(working_write),
            Action.PROPOSE_SHARED_WRITE: frozenset(propose),
            Action.VETO: frozenset(veto),
            Action.EXECUTE: frozenset(execute),
        },
    )


_WORKING = (
    ("working", "observation"),
    ("working", "hypothesis"),
    ("working", "question"),
    ("working", "candidate_insight"),
)

CAPABILITIES = CapabilityModel(
    (
        _spec(
            "news",
            read=(("events", "news"),),
            working_write=(("working", "observation"),),
        ),
        _spec(
            "news_analyst",
            read=(("events", "news"), ("insights", "news"), ("entities", "basic")),
            working_write=_WORKING,
            propose=(
                ("insights", "claim"),
                ("insights", "direction"),
                ("insights", "confidence"),
            ),
        ),
        _spec(
            "company_analyst",
            read=(
                ("company", "*"),
                ("insights", "*"),
                ("events", "*"),
                ("market", "features"),
            ),
            working_write=_WORKING,
            propose=(
                ("insights", "*"),
                ("trade_candidates", "thesis"),
                ("trade_candidates", "direction"),
            ),
        ),
        _spec(
            "market",
            read=(("market", "ohlcv"),),
            working_write=(("working", "observation"),),
        ),
        _spec(
            "risk_analyst",
            read=(
                ("company", "*"),
                ("insights", "*"),
                ("events", "*"),
                ("market", "features"),
            ),
            working_write=_WORKING,
        ),
        _spec(
            "risk_llm",
            read=(
                ("company", "*"),
                ("insights", "*"),
                ("events", "*"),
                ("market", "features"),
            ),
            working_write=_WORKING,
        ),
        _spec(
            "portfolio_risk",
            read=(
                ("portfolio", "*"),
                ("trade_candidates", "*"),
                ("risk", "*"),
                ("insights", "summary"),
            ),
            working_write=(("working", "observation"),),
            propose=(
                ("trade_candidates", "size"),
                ("trade_candidates", "rejection"),
                ("portfolio", "target_exposure"),
            ),
            veto=(("trade_candidates", "*"), ("portfolio", "*")),
        ),
    )
)


def authorize(
    agent: AgentId | str,
    action: Action,
    resource: str,
    field: str,
) -> bool:
    return CAPABILITIES.authorize(agent, action, resource, field)
