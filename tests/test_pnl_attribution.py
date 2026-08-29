import unittest
from datetime import datetime, timedelta, timezone

from agents import AttributionDimension, PnLAttributionError, PnLAttributor
from memory import LineageGraph, LineageNode, LineageNodeId, LineageNodeType


NOW = datetime(2026, 7, 1, tzinfo=timezone.utc)


def add(graph, kind, identifier, minute, **attributes):
    item = LineageNode(
        LineageNodeId(identifier), kind, NOW + timedelta(minutes=minute), attributes
    )
    return graph.add(item)


def attributed_graph(pnl=120.0, confidence=0.8):
    graph = LineageGraph()
    event1 = add(
        graph, LineageNodeType.EVENT, "event:1", 0,
        company="ABC", sector="Technology", source="Reuters",
    )
    event2 = add(
        graph, LineageNodeType.EVENT, "event:2", 0,
        company="ABC", sector="Technology", source="Bloomberg",
    )
    observation1 = add(
        graph, LineageNodeType.OBSERVATION, "observation:1", 1, company="ABC"
    )
    observation2 = add(
        graph, LineageNodeType.OBSERVATION, "observation:2", 1, company="ABC"
    )
    insight1 = add(
        graph, LineageNodeType.INSIGHT, "insight:1", 2,
        company="ABC", agent="news_analyst", agent_version="v1",
        category="earnings", confidence=confidence,
    )
    insight2 = add(
        graph, LineageNodeType.INSIGHT, "insight:2", 2,
        company="ABC", agent="company_analyst", agent_version="v2",
        category="fundamental", confidence=0.6,
    )
    trade = add(
        graph, LineageNodeType.TRADE_CANDIDATE, "trade:1", 3,
        company="ABC", sector="Technology", holding_horizon="30 days",
    )
    decision = add(
        graph, LineageNodeType.DECISION, "decision:1", 4,
        company="ABC", result="approved",
    )
    fill = add(
        graph, LineageNodeType.FILL, "fill:1", 5,
        company="ABC", realized_pnl=pnl,
    )
    for source, target in (
        (event1, observation1), (event2, observation2),
        (observation1, insight1), (observation2, insight2),
        (insight1, trade), (insight2, trade), (trade, decision), (decision, fill),
    ):
        graph.connect(source.id, target.id)
    return graph


def bucket_map(report, dimension):
    return {item.key: item.realized_pnl for item in report.for_dimension(dimension)}


class PnLAttributionTests(unittest.TestCase):
    def test_attributes_all_required_dimensions_and_reconciles_each(self):
        report = PnLAttributor().calculate(attributed_graph())
        self.assertEqual(report.total_realized_pnl, 120)
        self.assertEqual(report.fill_count, 1)
        self.assertEqual(
            bucket_map(report, AttributionDimension.AGENT),
            {"company_analyst": 60, "news_analyst": 60},
        )
        self.assertEqual(
            bucket_map(report, AttributionDimension.AGENT_VERSION),
            {"v1": 60, "v2": 60},
        )
        self.assertEqual(
            bucket_map(report, AttributionDimension.INSIGHT),
            {"insight:1": 60, "insight:2": 60},
        )
        self.assertEqual(
            bucket_map(report, AttributionDimension.INSIGHT_CATEGORY),
            {"earnings": 60, "fundamental": 60},
        )
        self.assertEqual(
            bucket_map(report, AttributionDimension.NEWS_SOURCE),
            {"Bloomberg": 60, "Reuters": 60},
        )
        self.assertEqual(
            bucket_map(report, AttributionDimension.COMPANY), {"ABC": 120}
        )
        self.assertEqual(
            bucket_map(report, AttributionDimension.SECTOR), {"Technology": 120}
        )
        self.assertEqual(
            bucket_map(report, AttributionDimension.CONFIDENCE_BUCKET),
            {"0.50-0.75": 60, "0.75-1.00": 60},
        )
        self.assertEqual(
            bucket_map(report, AttributionDimension.HOLDING_HORIZON),
            {"30 days": 120},
        )
        for dimension in AttributionDimension:
            self.assertAlmostEqual(
                sum(bucket_map(report, dimension).values()),
                report.total_realized_pnl,
            )

    def test_losses_are_attributed_without_changing_sign(self):
        report = PnLAttributor().calculate(attributed_graph(pnl=-40))
        self.assertEqual(report.total_realized_pnl, -40)
        self.assertEqual(
            bucket_map(report, AttributionDimension.NEWS_SOURCE),
            {"Bloomberg": -20, "Reuters": -20},
        )

    def test_missing_optional_metadata_uses_explicit_unknown_bucket(self):
        graph = attributed_graph()
        for insight in graph.nodes(LineageNodeType.INSIGHT):
            # Existing graph is immutable; build a minimal graph to exercise missing labels.
            self.assertNotEqual(insight.attributes.get("agent"), None)
        minimal = LineageGraph()
        nodes = [
            add(minimal, kind, f"{kind.value}:minimal", index, realized_pnl=5)
            if kind is LineageNodeType.FILL
            else add(minimal, kind, f"{kind.value}:minimal", index)
            for index, kind in enumerate(LineageNodeType)
        ]
        for source, target in zip(nodes, nodes[1:]):
            minimal.connect(source.id, target.id)
        report = PnLAttributor().calculate(minimal)
        self.assertEqual(
            bucket_map(report, AttributionDimension.AGENT), {"unknown": 5}
        )

    def test_rejects_incomplete_fill_lineage(self):
        graph = LineageGraph()
        fill = add(
            graph, LineageNodeType.FILL, "fill:orphan", 0, realized_pnl=1
        )
        with self.assertRaisesRegex(PnLAttributionError, "incomplete lineage"):
            PnLAttributor().calculate(graph)
        self.assertEqual(fill.id.value, "fill:orphan")

    def test_rejects_missing_or_invalid_realized_pnl(self):
        graph = attributed_graph()
        bad = graph.get(LineageNodeId("fill:1"))
        self.assertEqual(bad.attributes["realized_pnl"], 120)
        invalid = attributed_graph(pnl="profit")
        with self.assertRaisesRegex(PnLAttributionError, "numeric realized_pnl"):
            PnLAttributor().calculate(invalid)

    def test_rejects_invalid_confidence_metadata(self):
        with self.assertRaisesRegex(PnLAttributionError, "confidence"):
            PnLAttributor().calculate(attributed_graph(confidence=1.5))

    def test_empty_graph_returns_complete_zero_report(self):
        report = PnLAttributor().calculate(LineageGraph())
        self.assertEqual((report.total_realized_pnl, report.fill_count), (0, 0))
        self.assertTrue(all(report.for_dimension(item) == () for item in AttributionDimension))
        with self.assertRaises(TypeError):
            report.by_dimension[AttributionDimension.AGENT] = ()


if __name__ == "__main__":
    unittest.main()
