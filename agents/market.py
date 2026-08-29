from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

from memory.ids import AgentId, EntityId
from memory.time import EventTime, HistoricalRecord, KnowledgeTime, SimulationTime


AGENT_ID = AgentId("market")
_INTERVAL_MS = 60_000
_DEFAULT_QUOTE = "USDT"
_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD", "BTC", "ETH")
_HOSTS = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketState(HistoricalRecord):
    symbol: str
    entity_id: EntityId
    event_time: EventTime
    as_of: SimulationTime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    source: str = "binance"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.symbol:
            raise ValueError("symbol is required")
        if self.knowledge_time > self.as_of:
            raise ValueError("market state knowledge_time cannot be after as_of")


class BinanceError(RuntimeError):
    """Binance public market data request failed."""


class BinanceMarketClient:
    """Public Binance spot klines. No API key. Never reads the wall clock."""

    def __init__(self, hosts: tuple[str, ...] = _HOSTS, timeout: float = 10.0) -> None:
        self._hosts = hosts
        self._timeout = timeout

    def klines(
        self,
        symbol: str,
        *,
        start_ms: int,
        end_ms: int,
        interval: str = "1m",
        limit: int = 5,
    ) -> list[list[object]]:
        params = urlencode(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
        )
        last_error: Exception | None = None
        for host in self._hosts:
            url = f"{host}/api/v3/klines?{params}"
            request = Request(url, headers={"Accept": "application/json"})
            try:
                with urlopen(request, timeout=self._timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                continue
            if not isinstance(payload, list):
                raise BinanceError(f"unexpected Binance payload for {symbol}")
            return payload
        raise BinanceError(f"Binance klines failed for {symbol}: {last_error}")


class MarketAgent:
    """Pulls as-of market state for symbols from Binance public data."""

    def __init__(self, client: BinanceMarketClient | None = None) -> None:
        self.client = client or BinanceMarketClient()

    def snapshot(
        self,
        symbols: list[str] | tuple[str, ...],
        *,
        as_of: SimulationTime | datetime,
    ) -> tuple[MarketState, ...]:
        when = _as_of(as_of)
        if not symbols:
            raise ValueError("at least one symbol is required")
        return tuple(self._one(symbol, when) for symbol in symbols)

    def _one(self, raw_symbol: str, as_of: SimulationTime) -> MarketState:
        symbol = _binance_symbol(raw_symbol)
        as_of_ms = _ms(as_of.value)
        rows = self.client.klines(
            symbol,
            start_ms=as_of_ms - (3 * _INTERVAL_MS),
            end_ms=as_of_ms,
            interval="1m",
            limit=5,
        )
        candle = _last_completed(rows, as_of_ms)
        if candle is None:
            raise BinanceError(
                f"no completed Binance 1m candle for {symbol} at {as_of.value.isoformat()}"
            )
        open_time = _from_ms(int(candle[0]))
        close_time = _from_ms(int(candle[6]))
        return MarketState(
            entity_id=EntityId(symbol),
            symbol=symbol,
            event_time=EventTime(open_time),
            knowledge_time=KnowledgeTime(close_time),
            as_of=as_of,
            open=_num(candle[1]),
            high=_num(candle[2]),
            low=_num(candle[3]),
            close=_num(candle[4]),
            volume=_num(candle[5]),
            trades=int(candle[8]),
        )


def _as_of(value: SimulationTime | datetime) -> SimulationTime:
    if isinstance(value, SimulationTime):
        return value
    return SimulationTime(value)


def _binance_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper().replace("/", "").replace("-", "")
    if not cleaned:
        raise ValueError("symbol must be a non-empty string")
    for quote in _QUOTES:
        if cleaned.endswith(quote) and len(cleaned) > len(quote):
            return cleaned
    return f"{cleaned}{_DEFAULT_QUOTE}"


def _last_completed(rows: list[list[object]], as_of_ms: int) -> list[object] | None:
    completed: list[list[object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 9:
            raise BinanceError("malformed Binance kline")
        if int(row[6]) <= as_of_ms:
            completed.append(row)
    if not completed:
        return None
    return completed[-1]


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _num(value: object) -> float:
    return float(Decimal(str(value)))
