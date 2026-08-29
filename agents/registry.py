from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


class UnknownAgentVersion(KeyError):
    """No registered spec for this agent version key."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSpec:
    """Prompt, config and version stay together for one agent release."""

    name: str
    version: str
    prompt: str
    config: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("agent name must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("agent version must be a non-empty string")
        if ":" in self.name or ":" in self.version:
            raise ValueError("agent name and version cannot contain ':'")
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(self.config, Mapping):
            raise TypeError("config must be a mapping")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "version", self.version.strip())
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    def log_identity(self) -> dict[str, object]:
        return {
            "agent": self.name,
            "agent_version": self.version,
            "agent_key": self.key,
            "prompt_version": self.version,
            "config": dict(self.config),
        }


class AgentRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> AgentSpec:
        if not isinstance(spec, AgentSpec):
            raise TypeError("spec must be AgentSpec")
        existing = self._specs.get(spec.key)
        if existing is not None and existing != spec:
            raise ValueError(f"agent version already registered: {spec.key}")
        self._specs[spec.key] = spec
        return spec

    def get(self, key: str) -> AgentSpec:
        if not isinstance(key, str) or ":" not in key:
            raise ValueError("agent version key must look like name:version")
        spec = self._specs.get(key)
        if spec is None:
            raise UnknownAgentVersion(f"unknown agent version: {key}")
        return spec

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))


NEWS_ANALYST_V1 = AgentSpec(
    name="news_analyst",
    version="v1",
    prompt=(
        "You are the News Analyst for a quantitative research system. Analyze only "
        "the supplied articles. Assess materiality, likely market direction, time "
        "horizon, contradictions, and uncertainty. Never add a fact, event, price, "
        "or metric absent from the input. Cite only supplied evidence_refs. When the "
        "evidence is insufficient or conflicting, choose neutral and lower confidence."
    ),
    config={"output": "finding"},
)

COMPANY_ANALYST_V1 = AgentSpec(
    name="company_analyst",
    version="v1",
    prompt=(
        "You are the Company Analyst for a quantitative research system. Build a "
        "balanced company thesis and a probabilistic stock-momentum outlook using only "
        "the supplied regulatory records, company facts, governance-promoted insights, "
        "and deterministic market features. Distinguish facts "
        "from interpretations. Never invent financial values, filings, guidance, "
        "comparables, prices, or events. Momentum score ranges from -1 (strong negative) "
        "to +1 (strong positive); it is an outlook, never a certainty or price target. "
        "Cite only supplied refs. If evidence is stale, "
        "incomplete, or contradictory, state that and reduce confidence. Do not propose "
        "position size or execute a trade."
    ),
    config={"output": "company_analysis"},
)

MARKET_V1 = AgentSpec(
    name="market",
    version="v1",
    prompt="",
    config={"source": "binance", "interval": "1m", "quote": "USDT"},
)

NEWS_V1 = AgentSpec(
    name="news",
    version="v1",
    prompt="",
    config={"role": "ingest"},
)

RISK_LLM_V1 = AgentSpec(
    name="risk_llm",
    version="v1",
    prompt="",
    config={"role": "suggest_only"},
)

PORTFOLIO_RISK_V1 = AgentSpec(
    name="portfolio_risk",
    version="v1",
    prompt="",
    config={"role": "portfolio_risk"},
)


def default_registry() -> AgentRegistry:
    registry = AgentRegistry()
    for spec in (
        NEWS_V1,
        NEWS_ANALYST_V1,
        COMPANY_ANALYST_V1,
        MARKET_V1,
        RISK_LLM_V1,
        PORTFOLIO_RISK_V1,
    ):
        registry.register(spec)
    return registry


REGISTRY = default_registry()
