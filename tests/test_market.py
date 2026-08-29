import unittest
from datetime import datetime, timezone

from agents import BinanceError, MarketAgent, MarketState
from memory import ContextGateway, ContextPermissionError, SharedMemory


UTC = timezone.utc
AS_OF = datetime(2024, 6, 1, 0, 1, tzinfo=UTC)


class _FakeClient:
    def __init__(self, rows: dict[str, list[list[object]]]) -> None:
        self.rows = rows
        self.calls: list[str] = []

    def klines(self, symbol: str, **kwargs: object) -> list[list[object]]:
        self.calls.append(symbol)
        return self.rows[symbol]


def _candle(open_ms: int, close: str) -> list[object]:
    return [
        open_ms,
        "100",
        "110",
        "90",
        close,
        "12.5",
        open_ms + 59_999,
        "1000",
        42,
        "6",
        "500",
        "0",
    ]


def context(symbols=("BTC",), at=AS_OF):
    return ContextGateway(SharedMemory()).for_market_agent(
        agent_id="market",
        symbols=symbols,
        simulation_time=at,
    )


class MarketAgentTests(unittest.TestCase):
    def test_returns_state_from_gateway_context(self) -> None:
        as_of_ms = int(AS_OF.timestamp() * 1000)
        client = _FakeClient(
            {
                "BTCUSDT": [
                    _candle(as_of_ms - 120_000, "101"),
                    _candle(as_of_ms - 60_000, "105"),
                    _candle(as_of_ms, "999"),
                ]
            }
        )
        states = MarketAgent(client).snapshot(context())
        self.assertEqual(len(states), 1)
        state = states[0]
        self.assertEqual(state.symbol, "BTCUSDT")
        self.assertEqual(state.source, "binance")
        self.assertEqual(state.close, 105.0)
        self.assertEqual(state.trades, 42)
        self.assertLessEqual(state.knowledge_time, state.simulation_time)
        self.assertEqual(state.simulation_time, AS_OF)

    def test_rejects_direct_unpermissioned_data(self) -> None:
        with self.assertRaises(TypeError):
            MarketAgent(_FakeClient({})).snapshot(["BTC"])  # type: ignore[arg-type]

    def test_gateway_checks_agent_permission(self) -> None:
        with self.assertRaises(ContextPermissionError):
            ContextGateway(SharedMemory()).for_market_agent(
                agent_id="news_analyst",
                symbols=("BTC",),
                simulation_time=AS_OF,
            )

    def test_gateway_rejects_naive_simulation_time(self) -> None:
        with self.assertRaises(ValueError):
            ContextGateway(SharedMemory()).for_market_agent(
                agent_id="market",
                symbols=("BTC",),
                simulation_time=datetime(2024, 6, 1, 0, 1),
            )

    def test_rejects_lookahead_if_only_open_candle_exists(self) -> None:
        as_of_ms = int(AS_OF.timestamp() * 1000)
        client = _FakeClient({"ETHUSDT": [_candle(as_of_ms, "1")]})
        with self.assertRaises(BinanceError):
            MarketAgent(client).snapshot(context(("ETH",)))

    def test_normalizes_symbols(self) -> None:
        as_of_ms = int(AS_OF.timestamp() * 1000)
        client = _FakeClient({"ETHUSDT": [_candle(as_of_ms - 60_000, "2")]})
        state = MarketAgent(client).snapshot(context(("eth",)))[0]
        self.assertEqual(state.symbol, "ETHUSDT")
        self.assertEqual(client.calls, ["ETHUSDT"])

    def test_requires_knowledge_time(self) -> None:
        with self.assertRaises(TypeError):
            MarketState(
                ref="binance:BTCUSDT:x",
                symbol="BTCUSDT",
                source="binance",
                url="https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1m",
                event_time=AS_OF,
                simulation_time=AS_OF,
                open=1,
                high=1,
                low=1,
                close=1,
                volume=1,
                trades=1,
            )

    def test_context_contains_no_portfolio_state(self) -> None:
        selected = context()
        self.assertFalse(hasattr(selected, "portfolio"))
        self.assertEqual(set(selected.__slots__), {"symbols", "simulation_time"})
