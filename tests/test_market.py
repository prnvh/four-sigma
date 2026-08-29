import unittest
from datetime import datetime, timezone

from agents.market import BinanceError, MarketAgent, MarketState
from memory import EntityId, EventTime, KnowledgeTime, SimulationTime


AS_OF = SimulationTime(datetime(2024, 6, 1, 0, 1, tzinfo=timezone.utc))


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


class MarketAgentTests(unittest.TestCase):
    def test_uses_last_completed_candle_only(self) -> None:
        as_of_ms = int(AS_OF.value.timestamp() * 1000)
        client = _FakeClient(
            {
                "BTCUSDT": [
                    _candle(as_of_ms - 120_000, "101"),
                    _candle(as_of_ms - 60_000, "105"),
                    _candle(as_of_ms, "999"),
                ]
            }
        )
        states = MarketAgent(client).snapshot(["BTC"], as_of=AS_OF)
        self.assertEqual(len(states), 1)
        state = states[0]
        self.assertEqual(state.symbol, "BTCUSDT")
        self.assertEqual(state.close, 105.0)
        self.assertEqual(state.trades, 42)
        self.assertLessEqual(state.knowledge_time, state.as_of)
        self.assertTrue(state.visible_as_of(AS_OF))

    def test_rejects_lookahead_if_only_open_candle_exists(self) -> None:
        as_of_ms = int(AS_OF.value.timestamp() * 1000)
        client = _FakeClient({"ETHUSDT": [_candle(as_of_ms, "1")]})
        with self.assertRaises(BinanceError):
            MarketAgent(client).snapshot(["ETH"], as_of=AS_OF)

    def test_normalizes_symbols(self) -> None:
        as_of_ms = int(AS_OF.value.timestamp() * 1000)
        client = _FakeClient({"ETHUSDT": [_candle(as_of_ms - 60_000, "2")]})
        state = MarketAgent(client).snapshot(["eth"], as_of=AS_OF)[0]
        self.assertEqual(state.entity_id, EntityId("ETHUSDT"))
        self.assertEqual(client.calls, ["ETHUSDT"])

    def test_requires_knowledge_time(self) -> None:
        with self.assertRaises(TypeError):
            MarketState(
                symbol="BTCUSDT",
                entity_id=EntityId("BTCUSDT"),
                event_time=EventTime(AS_OF.value),
                as_of=AS_OF,
                open=1,
                high=1,
                low=1,
                close=1,
                volume=1,
                trades=1,
            )
