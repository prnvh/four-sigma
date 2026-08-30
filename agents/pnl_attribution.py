from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import fsum, isfinite
from types import MappingProxyType

from memory.lineage import LineageGraph, LineageNode, LineageNodeType


class PnLAttributionError(ValueError):
    """Lineage or PnL metadata is insufficient for trustworthy attribution."""


class AttributionDimension(StrEnum):
    AGENT = "agent"
    AGENT_VERSION = "agent_version"
    INSIGHT = "insight"
    INSIGHT_CATEGORY = "insight_category"
    NEWS_SOURCE = "news_source"
    COMPANY = "company"
    SECTOR = "sector"
    CONFIDENCE_BUCKET = "confidence_bucket"
    HOLDING_HORIZON = "holding_horizon"


@dataclass(frozen=True, slots=True)
class PnLAttributionBucket:
    dimension: AttributionDimension
    key: str
    realized_pnl: float
    contributing_fills: int


@dataclass(frozen=True, slots=True)
class PnLAttributionReport:
    total_realized_pnl: float
    fill_count: int
    by_dimension: Mapping[AttributionDimension, tuple[PnLAttributionBucket, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_dimension", MappingProxyType(dict(self.by_dimension)))

    def for_dimension(
        self, dimension: AttributionDimension
    ) -> tuple[PnLAttributionBucket, ...]:
        if not isinstance(dimension, AttributionDimension):
            raise TypeError("dimension must be AttributionDimension")
        return self.by_dimension[dimension]


class PnLAttributor:
    """Allocate net realized PnL across every upstream lineage dimension."""

    def calculate(self, graph: LineageGraph) -> PnLAttributionReport:
        if not isinstance(graph, LineageGraph):
            raise TypeError("graph must be LineageGraph")
        fills = graph.nodes(LineageNodeType.FILL)
        amounts: dict[AttributionDimension, dict[str, list[float]]] = {
            dimension: defaultdict(list) for dimension in AttributionDimension
        }
        total_values: list[float] = []
        for fill in fills:
            trace = graph.trace(fill.id)
            self._require_complete(trace, fill)
            pnl = self._pnl(fill)
            total_values.append(pnl)
            values = self._dimensions(trace)
            for dimension, keys in values.items():
                share = pnl / len(keys)
                for key in keys:
                    amounts[dimension][key].append(share)

        by_dimension = {}
        for dimension in AttributionDimension:
            by_dimension[dimension] = tuple(
                PnLAttributionBucket(
                    dimension=dimension,
                    key=key,
                    realized_pnl=fsum(values),
                    contributing_fills=len(values),
                )
                for key, values in sorted(amounts[dimension].items())
            )
        return PnLAttributionReport(
            total_realized_pnl=fsum(total_values),
            fill_count=len(fills),
            by_dimension=by_dimension,
        )

    @staticmethod
    def _require_complete(trace: tuple[LineageNode, ...], fill: LineageNode) -> None:
        present = {node.node_type for node in trace}
        legacy = {
            LineageNodeType.EVENT,
            LineageNodeType.OBSERVATION,
            LineageNodeType.INSIGHT,
            LineageNodeType.TRADE_CANDIDATE,
            LineageNodeType.DECISION,
            LineageNodeType.FILL,
        }
        operational = {
            LineageNodeType.EVENT,
            LineageNodeType.OBSERVATION,
            LineageNodeType.INSIGHT_PROPOSAL,
            LineageNodeType.GOVERNANCE_APPROVAL,
            LineageNodeType.COMPANY_THESIS,
            LineageNodeType.TRADE_CANDIDATE,
            LineageNodeType.RISK_REVIEW,
            LineageNodeType.PORTFOLIO_DECISION,
            LineageNodeType.FILL,
            LineageNodeType.PNL,
        }
        if not legacy.issubset(present) and not operational.issubset(present):
            missing = min(
                (legacy - present, operational - present), key=len
            )
            raise PnLAttributionError(
                f"fill {fill.id.value} has incomplete lineage: "
                f"{sorted(item.value for item in missing)}"
            )

    @staticmethod
    def _pnl(fill: LineageNode) -> float:
        value = fill.attributes.get("realized_pnl")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PnLAttributionError(
                f"fill {fill.id.value} requires numeric realized_pnl"
            )
        pnl = float(value)
        if not isfinite(pnl):
            raise PnLAttributionError("realized_pnl must be finite")
        return pnl

    def _dimensions(
        self, trace: tuple[LineageNode, ...]
    ) -> dict[AttributionDimension, tuple[str, ...]]:
        events = self._of_type(trace, LineageNodeType.EVENT)
        insights = tuple(
            node for node in trace
            if node.node_type in {
                LineageNodeType.INSIGHT,
                LineageNodeType.INSIGHT_PROPOSAL,
                LineageNodeType.COMPANY_THESIS,
            }
        )
        trades = self._of_type(trace, LineageNodeType.TRADE_CANDIDATE)
        return {
            AttributionDimension.AGENT: self._attribute(insights, "agent"),
            AttributionDimension.AGENT_VERSION: self._attribute(
                insights, "agent_version"
            ),
            AttributionDimension.INSIGHT: tuple(
                sorted({node.id.value for node in insights})
            ),
            AttributionDimension.INSIGHT_CATEGORY: self._attribute(
                insights, "category"
            ),
            AttributionDimension.NEWS_SOURCE: self._attribute(events, "source"),
            AttributionDimension.COMPANY: self._attribute(trace, "company"),
            AttributionDimension.SECTOR: self._attribute(trace, "sector"),
            AttributionDimension.CONFIDENCE_BUCKET: self._confidence(insights),
            AttributionDimension.HOLDING_HORIZON: self._attribute(
                trades, "holding_horizon", fallback="horizon"
            ),
        }

    @staticmethod
    def _of_type(
        trace: tuple[LineageNode, ...], node_type: LineageNodeType
    ) -> tuple[LineageNode, ...]:
        return tuple(node for node in trace if node.node_type is node_type)

    @staticmethod
    def _attribute(
        nodes: tuple[LineageNode, ...], name: str, *, fallback: str | None = None
    ) -> tuple[str, ...]:
        values: set[str] = set()
        for node in nodes:
            value = node.attributes.get(name)
            if value is None and fallback is not None:
                value = node.attributes.get(fallback)
            candidates = value if isinstance(value, tuple) else (value,)
            for item in candidates:
                if isinstance(item, str) and item.strip():
                    values.add(item.strip())
        return tuple(sorted(values)) or ("unknown",)

    @staticmethod
    def _confidence(insights: tuple[LineageNode, ...]) -> tuple[str, ...]:
        buckets: set[str] = set()
        for insight in insights:
            value = insight.attributes.get("confidence")
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise PnLAttributionError("insight confidence must be between 0 and 1")
            confidence = float(value)
            if confidence < 0.25:
                buckets.add("0.00-0.25")
            elif confidence < 0.50:
                buckets.add("0.25-0.50")
            elif confidence < 0.75:
                buckets.add("0.50-0.75")
            else:
                buckets.add("0.75-1.00")
        return tuple(sorted(buckets)) or ("unknown",)
