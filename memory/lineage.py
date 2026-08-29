from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from threading import RLock
from types import MappingProxyType

from .types import LineageNodeId


class LineageError(ValueError):
    """A lineage node or relationship would make attribution unreliable."""


class LineageNodeType(StrEnum):
    EVENT = "event"
    OBSERVATION = "observation"
    INSIGHT = "insight"
    TRADE_CANDIDATE = "trade_candidate"
    DECISION = "decision"
    FILL = "fill"


_ORDER = {node_type: index for index, node_type in enumerate(LineageNodeType)}
_NEXT = {
    LineageNodeType.EVENT: LineageNodeType.OBSERVATION,
    LineageNodeType.OBSERVATION: LineageNodeType.INSIGHT,
    LineageNodeType.INSIGHT: LineageNodeType.TRADE_CANDIDATE,
    LineageNodeType.TRADE_CANDIDATE: LineageNodeType.DECISION,
    LineageNodeType.DECISION: LineageNodeType.FILL,
}


def _freeze(value: object, path: str = "attributes") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LineageError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise LineageError(f"{path} keys must be non-empty strings")
            frozen[key] = _freeze(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{path}[]") for item in value)
    raise LineageError(f"{path} contains unsupported {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class LineageNode:
    id: LineageNodeId
    node_type: LineageNodeType
    knowledge_time: datetime
    attributes: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.id, LineageNodeId):
            raise TypeError("id must be LineageNodeId")
        if not isinstance(self.node_type, LineageNodeType):
            raise TypeError("node_type must be LineageNodeType")
        if not isinstance(self.knowledge_time, datetime):
            raise TypeError("knowledge_time must be datetime")
        if self.knowledge_time.tzinfo is None:
            raise LineageError("knowledge_time must be timezone-aware")
        if not isinstance(self.attributes, Mapping):
            raise TypeError("attributes must be a mapping")
        frozen = _freeze(self.attributes)
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "attributes", frozen)


@dataclass(frozen=True, slots=True)
class LineageEdge:
    source: LineageNodeId
    target: LineageNodeId

    def __post_init__(self) -> None:
        if not isinstance(self.source, LineageNodeId):
            raise TypeError("source must be LineageNodeId")
        if not isinstance(self.target, LineageNodeId):
            raise TypeError("target must be LineageNodeId")
        if self.source == self.target:
            raise LineageError("a lineage node cannot link to itself")


class LineageGraph:
    """Append-only, stage-checked attribution graph with deterministic queries."""

    def __init__(self) -> None:
        self._nodes: dict[LineageNodeId, LineageNode] = {}
        self._edges: set[LineageEdge] = set()
        self._forward: dict[LineageNodeId, set[LineageNodeId]] = {}
        self._reverse: dict[LineageNodeId, set[LineageNodeId]] = {}
        self._lock = RLock()

    def add(self, node: LineageNode) -> LineageNode:
        if not isinstance(node, LineageNode):
            raise TypeError("node must be LineageNode")
        with self._lock:
            if node.id in self._nodes:
                raise LineageError(f"duplicate lineage node: {node.id}")
            self._nodes[node.id] = node
            self._forward[node.id] = set()
            self._reverse[node.id] = set()
            return node

    def connect(
        self, source: LineageNodeId, target: LineageNodeId
    ) -> LineageEdge:
        if not isinstance(source, LineageNodeId) or not isinstance(target, LineageNodeId):
            raise TypeError("source and target must be LineageNodeId")
        edge = LineageEdge(source, target)
        with self._lock:
            if source not in self._nodes or target not in self._nodes:
                raise LineageError("both lineage nodes must exist before connecting them")
            source_node = self._nodes[source]
            target_node = self._nodes[target]
            if _NEXT.get(source_node.node_type) is not target_node.node_type:
                raise LineageError(
                    f"cannot connect {source_node.node_type.value} to "
                    f"{target_node.node_type.value}"
                )
            if target_node.knowledge_time < source_node.knowledge_time:
                raise LineageError("lineage target cannot predate its source")
            source_company = source_node.attributes.get("company")
            target_company = target_node.attributes.get("company")
            if (
                isinstance(source_company, str)
                and isinstance(target_company, str)
                and source_company.strip().upper() != target_company.strip().upper()
            ):
                raise LineageError("lineage cannot cross company boundaries")
            if edge in self._edges:
                raise LineageError("duplicate lineage edge")
            self._edges.add(edge)
            self._forward[source].add(target)
            self._reverse[target].add(source)
            return edge

    def get(self, node_id: LineageNodeId) -> LineageNode:
        if not isinstance(node_id, LineageNodeId):
            raise TypeError("node_id must be LineageNodeId")
        with self._lock:
            try:
                return self._nodes[node_id]
            except KeyError as error:
                raise LineageError(f"unknown lineage node: {node_id}") from error

    def nodes(self, node_type: LineageNodeType | None = None) -> tuple[LineageNode, ...]:
        if node_type is not None and not isinstance(node_type, LineageNodeType):
            raise TypeError("node_type must be LineageNodeType or None")
        with self._lock:
            selected = (
                self._nodes.values()
                if node_type is None
                else (node for node in self._nodes.values() if node.node_type is node_type)
            )
            return tuple(sorted(selected, key=self._sort_key))

    def edges(self) -> tuple[LineageEdge, ...]:
        with self._lock:
            return tuple(sorted(self._edges, key=lambda edge: (edge.source.value, edge.target.value)))

    def upstream(self, node_id: LineageNodeId) -> tuple[LineageNode, ...]:
        return self._related(node_id, self._reverse)

    def downstream(self, node_id: LineageNodeId) -> tuple[LineageNode, ...]:
        return self._related(node_id, self._forward)

    def trace(self, node_id: LineageNodeId) -> tuple[LineageNode, ...]:
        center = self.get(node_id)
        related = {node.id: node for node in self.upstream(node_id)}
        related[center.id] = center
        related.update({node.id: node for node in self.downstream(node_id)})
        return tuple(sorted(related.values(), key=self._sort_key))

    def _related(
        self,
        node_id: LineageNodeId,
        adjacency: dict[LineageNodeId, set[LineageNodeId]],
    ) -> tuple[LineageNode, ...]:
        self.get(node_id)
        with self._lock:
            pending = list(adjacency[node_id])
            seen: set[LineageNodeId] = set()
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(adjacency[current])
            return tuple(sorted((self._nodes[item] for item in seen), key=self._sort_key))

    @staticmethod
    def _sort_key(node: LineageNode) -> tuple[int, datetime, str]:
        return (_ORDER[node.node_type], node.knowledge_time, node.id.value)
