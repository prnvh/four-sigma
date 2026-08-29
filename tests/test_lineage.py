import unittest
from datetime import datetime, timedelta, timezone

from memory import (
    LineageError,
    LineageGraph,
    LineageNode,
    LineageNodeId,
    LineageNodeType,
)


NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def node(node_type, name, minute, **attributes):
    return LineageNode(
        id=LineageNodeId(name),
        node_type=node_type,
        knowledge_time=NOW + timedelta(minutes=minute),
        attributes={"company": "ABC", **attributes},
    )


def complete_graph():
    graph = LineageGraph()
    nodes = (
        node(LineageNodeType.EVENT, "event:news:1", 0, source="Reuters"),
        node(LineageNodeType.OBSERVATION, "observation:1", 1, category="earnings"),
        node(LineageNodeType.INSIGHT, "insight:1", 2, agent="news_analyst"),
        node(LineageNodeType.TRADE_CANDIDATE, "trade:1", 3, horizon="30d"),
        node(LineageNodeType.DECISION, "decision:1", 4, result="approved"),
        node(LineageNodeType.FILL, "fill:1", 5, pnl=12.5),
    )
    for item in nodes:
        graph.add(item)
    for source, target in zip(nodes, nodes[1:]):
        graph.connect(source.id, target.id)
    return graph, nodes


class LineageGraphTests(unittest.TestCase):
    def test_full_attribution_chain_is_queryable_in_both_directions(self):
        graph, nodes = complete_graph()
        self.assertEqual(graph.upstream(nodes[-1].id), nodes[:-1])
        self.assertEqual(graph.downstream(nodes[0].id), nodes[1:])
        self.assertEqual(graph.trace(nodes[2].id), nodes)
        self.assertEqual(len(graph.edges()), 5)

    def test_multiple_events_can_support_one_insight_through_observations(self):
        graph, nodes = complete_graph()
        event = graph.add(node(LineageNodeType.EVENT, "event:news:2", 0))
        observation = graph.add(
            node(LineageNodeType.OBSERVATION, "observation:2", 1)
        )
        graph.connect(event.id, observation.id)
        graph.connect(observation.id, nodes[2].id)
        event_ids = {
            item.id.value
            for item in graph.upstream(nodes[2].id)
            if item.node_type is LineageNodeType.EVENT
        }
        self.assertEqual(event_ids, {"event:news:1", "event:news:2"})

    def test_cannot_skip_a_lineage_stage(self):
        graph = LineageGraph()
        event = graph.add(node(LineageNodeType.EVENT, "event:1", 0))
        insight = graph.add(node(LineageNodeType.INSIGHT, "insight:1", 1))
        with self.assertRaisesRegex(LineageError, "cannot connect event to insight"):
            graph.connect(event.id, insight.id)
        self.assertEqual(graph.edges(), ())

    def test_cannot_connect_unknown_or_duplicate_edge(self):
        graph = LineageGraph()
        event = graph.add(node(LineageNodeType.EVENT, "event:1", 0))
        observation = graph.add(node(LineageNodeType.OBSERVATION, "observation:1", 1))
        with self.assertRaisesRegex(LineageError, "both lineage nodes must exist"):
            graph.connect(event.id, LineageNodeId("missing"))
        graph.connect(event.id, observation.id)
        with self.assertRaisesRegex(LineageError, "duplicate lineage edge"):
            graph.connect(event.id, observation.id)

    def test_target_cannot_predate_source(self):
        graph = LineageGraph()
        event = graph.add(node(LineageNodeType.EVENT, "event:1", 2))
        observation = graph.add(node(LineageNodeType.OBSERVATION, "observation:1", 1))
        with self.assertRaisesRegex(LineageError, "cannot predate"):
            graph.connect(event.id, observation.id)

    def test_lineage_cannot_cross_company_boundaries(self):
        graph = LineageGraph()
        event = graph.add(node(LineageNodeType.EVENT, "event:1", 0))
        observation = graph.add(LineageNode(
            LineageNodeId("observation:1"), LineageNodeType.OBSERVATION,
            NOW + timedelta(minutes=1), {"company": "XYZ"},
        ))
        with self.assertRaisesRegex(LineageError, "cross company"):
            graph.connect(event.id, observation.id)

    def test_nodes_and_metadata_are_append_only_and_detached(self):
        metadata = {"company": "ABC", "contributors": ["news_agent"]}
        item = LineageNode(
            LineageNodeId("event:1"), LineageNodeType.EVENT, NOW, metadata
        )
        graph = LineageGraph()
        graph.add(item)
        metadata["contributors"].append("invented")
        self.assertEqual(item.attributes["contributors"], ("news_agent",))
        with self.assertRaises(TypeError):
            item.attributes["company"] = "XYZ"
        with self.assertRaisesRegex(LineageError, "duplicate lineage node"):
            graph.add(item)

    def test_type_filtered_query_is_deterministic(self):
        graph, nodes = complete_graph()
        self.assertEqual(graph.nodes(LineageNodeType.FILL), (nodes[-1],))
        self.assertEqual(graph.nodes(), nodes)


if __name__ == "__main__":
    unittest.main()
