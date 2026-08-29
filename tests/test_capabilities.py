import unittest
from datetime import datetime, timezone

from agents import NewsAnalyst
from memory import (
    Action,
    AgentId,
    AuthorizationError,
    CAPABILITIES,
    ContextGateway,
    NewsEventStore,
    authorize,
)

from test_news_analyst import StubModel, article, context


NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)


class CapabilityModelTests(unittest.TestCase):
    def test_news_analyst_can_read_and_propose_declared_fields(self) -> None:
        self.assertTrue(authorize("news_analyst", Action.READ, "events", "news"))
        self.assertTrue(authorize(AgentId("news_analyst:v1"), Action.READ, "insights", "news"))
        self.assertTrue(authorize("news_analyst", Action.WORKING_WRITE, "working", "hypothesis"))
        self.assertTrue(
            authorize("news_analyst", Action.PROPOSE_SHARED_WRITE, "insights", "claim")
        )

    def test_news_analyst_is_denied_portfolio_and_execution(self) -> None:
        self.assertFalse(authorize("news_analyst", Action.READ, "portfolio", "positions"))
        self.assertFalse(authorize("news_analyst", Action.EXECUTE, "trades", "order"))
        self.assertFalse(authorize("news_analyst", Action.VETO, "trade_candidates", "size"))

    def test_unknown_agent_and_undeclared_field_are_denied(self) -> None:
        self.assertFalse(authorize("unknown", Action.READ, "events", "news"))
        self.assertFalse(authorize("news_analyst", Action.READ, "portfolio", "cash"))
        self.assertFalse(
            authorize("news_analyst", Action.PROPOSE_SHARED_WRITE, "insights", "size")
        )

    def test_wildcard_read_does_not_grant_portfolio(self) -> None:
        self.assertTrue(authorize("company_analyst", Action.READ, "company", "records"))
        self.assertTrue(authorize("company_analyst", Action.READ, "insights", "promoted"))
        self.assertFalse(authorize("company_analyst", Action.READ, "portfolio", "positions"))
        self.assertFalse(
            authorize("company_analyst", Action.PROPOSE_SHARED_WRITE, "trade_candidates", "size")
        )

    def test_trade_constructor_reads_insights_only(self) -> None:
        self.assertTrue(authorize("trade_constructor", Action.READ, "insights", "promoted"))
        self.assertTrue(
            authorize(
                "trade_constructor",
                Action.PROPOSE_SHARED_WRITE,
                "trade_candidates",
                "direction",
            )
        )
        self.assertFalse(authorize("trade_constructor", Action.READ, "events", "news"))
        self.assertFalse(authorize("trade_constructor", Action.READ, "portfolio", "positions"))
        self.assertFalse(authorize("trade_constructor", Action.EXECUTE, "trades", "order"))

    def test_portfolio_risk_can_read_and_veto_portfolio(self) -> None:
        self.assertTrue(authorize("portfolio_risk", Action.READ, "portfolio", "positions"))
        self.assertTrue(authorize("portfolio_risk", Action.VETO, "trade_candidates", "size"))
        self.assertFalse(authorize("portfolio_risk", Action.EXECUTE, "trades", "order"))
        self.assertFalse(authorize("risk_analyst", Action.READ, "portfolio", "positions"))

    def test_require_raises_for_denied_action(self) -> None:
        with self.assertRaises(AuthorizationError):
            CAPABILITIES.require("news_analyst", Action.READ, "portfolio", "positions")


class FieldAuthorizationTests(unittest.TestCase):
    def test_gateway_rejects_portfolio_fields_before_returning_context(self) -> None:
        store = NewsEventStore()
        store.append_news(article())
        with self.assertRaises(AuthorizationError):
            ContextGateway(store).for_news_analyst(
                agent_id="news_analyst",
                symbol="ABC",
                simulation_time=NOW,
                fields=(("portfolio", "positions"),),
            )

    def test_news_analyst_rejects_portfolio_request_before_model(self) -> None:
        model = StubModel()
        with self.assertRaises(AuthorizationError):
            NewsAnalyst(model).analyze(
                context(article()),
                requested_fields=(("portfolio", "positions"),),
            )
        self.assertIsNone(model.last_input)

    def test_permitted_news_read_still_reaches_the_model(self) -> None:
        model = StubModel()
        NewsAnalyst(model).analyze(context(article()))
        self.assertIsNotNone(model.last_input)


if __name__ == "__main__":
    unittest.main()
