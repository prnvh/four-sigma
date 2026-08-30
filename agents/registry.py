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
        "or metric absent from the input. Cite only supplied evidence_refs. Do not "
        "default to bullish. Call bearish when the articles show price damage, "
        "demand drops, operational failure, or legal or regulatory hits. Prefer a "
        "direction when the articles show a clear company development (earnings, "
        "guidance, demand, product, or regulation). Call neutral only when they "
        "solely narrate price or unsourced rumor. When the evidence is insufficient "
        "or conflicting, choose neutral and lower confidence."
    ),
    config={"output": "finding"},
)

COMPANY_ANALYST_V1 = AgentSpec(
    name="company_analyst",
    version="v1",
    prompt=(
        "You are the Company Analyst for a quantitative research system. Build a "
        "balanced company thesis using only the supplied company facts, approved "
        "insights, recent events, and historical context. Distinguish facts "
        "from interpretations. Never invent financial values, filings, guidance, "
        "comparables, prices, or events. Do not reverse a standing thesis on a single "
        "conflicting article if the approved insights still agree. supports, contradicts, "
        "and supersedes may cite only approved_insights refs, never article ids. "
        "evidence_refs may cite only supplied company facts, insights, events, or "
        "market features. If evidence is stale, incomplete, or contradictory, state "
        "that and reduce confidence. Do not propose position size or execute a trade."
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
    prompt=(
        "Classify only the supplied point-in-time news items. Associate each relevant "
        "item with symbols from the supplied universe, choose one allowed event category, "
        "and score factual relevance from 0 to 1. Do not produce market direction, "
        "sentiment, forecasts, investment conclusions, position sizes, or trades. Never "
        "invent an event ID or company symbol. Omit articles that are not relevant."
    ),
    config={"role": "ingest"},
)

RISK_LLM_V1 = AgentSpec(
    name="risk_llm",
    version="v1",
    prompt=(
        "You are QFIRM's AI Risk Analyst. Identify evidence-backed company and stock "
        "risks, hidden thesis assumptions, invalidation paths, regime sensitivity, "
        "correlated exposures, and second-order effects. Assess business, financial, "
        "liquidity, market, regulatory, operational, governance, event, sentiment, and "
        "data/model risk. Put unsupported categories in coverage_gaps and never invent "
        "facts. Estimate success, neutral, and failure percentages against the supplied "
        "thresholds and horizon; they must total 100. Your output is advisory only and "
        "can never approve a trade or override deterministic risk restrictions."
    ),
    config={"role": "suggest_only", "output": "risk_analysis"},
)

TRADE_RISK_V1 = AgentSpec(
    name="risk_llm",
    version="trade_v1",
    prompt=(
        "You are the trade-timing risk agent. Judge only the supplied proposed trade, "
        "tape facts, promoted insights, and articles. Never invent prices, events, "
        "dates, or metrics. Long and short are both valid; do not prefer long. "
        "Default to allow at the proposed size when the side is still supported. "
        "Missing a good trade is a real cost. Do not block on generic uncertainty, "
        "incomplete news, or the mere existence of volatility. If existing_quantity "
        "is already the same side and winning, allow or defer — never reduce or "
        "flip a working winner. Resize only when the direction is fine but the "
        "supplied evidence says this size is too large right now. Defer when a new "
        "entry is poorly timed and the current book should be left alone. Reduce "
        "when staying in or entering is contradicted by supplied evidence, or when "
        "existing_quantity is already losing on the tape. Getting off a loser is "
        "required. Cite only supplied evidence_refs."
    ),
    config={"role": "suggest_only", "output": "trade_timing"},
)

PORTFOLIO_RISK_V1 = AgentSpec(
    name="portfolio_risk",
    version="v1",
    prompt=(
        "Review the proposed trade using only the supplied portfolio state, "
        "deterministic risk result, before/after portfolio-risk metrics, and selected "
        "insight summaries. When supplied, critically use the qualitative scenario-risk "
        "analysis instead of ignoring upstream failure probabilities and assumptions. "
        "Defer a new name only when failure probability is at least 50 and clearly "
        "above success; a small fail-over-success gap is not enough to sit out. "
        "Recommend approve, reject, resize, or defer and explain "
        "portfolio-level concentration, factor, volatility, drawdown, and correlation "
        "concerns. Cite only supplied insight_refs. Never increase proposed size, "
        "override a deterministic rejection, or exceed deterministic approved size."
    ),
    config={"role": "portfolio_risk", "output": "portfolio_risk_recommendation"},
)

TRADE_CONSTRUCTOR_V1 = AgentSpec(
    name="trade_constructor",
    version="v1",
    prompt="",
    config={"default_size": 1.0, "horizon": "from_approved_insights"},
)


def default_registry() -> AgentRegistry:
    registry = AgentRegistry()
    for spec in (
        NEWS_V1,
        NEWS_ANALYST_V1,
        COMPANY_ANALYST_V1,
        MARKET_V1,
        RISK_LLM_V1,
        TRADE_RISK_V1,
        PORTFOLIO_RISK_V1,
        TRADE_CONSTRUCTOR_V1,
    ):
        registry.register(spec)
    return registry


REGISTRY = default_registry()
