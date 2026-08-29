from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

from memory.context import MarketContext

from .schemas import MarketState


_INTERVAL_MS = 60_000
_DEFAULT_QUOTE = "USDT"
_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD", "BTC", "ETH")
_HOSTS = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)


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
    """Pulls as-of Binance market state from a permissioned context snapshot."""

    def __init__(self, client: BinanceMarketClient | None = None) -> None:
        self.client = client or BinanceMarketClient()

    def snapshot(self, context: MarketContext) -> tuple[MarketState, ...]:
        if not isinstance(context, MarketContext):
            raise TypeError("MarketAgent requires context from ContextGateway")
        if not context.symbols:
            raise ValueError("market snapshot requires at least one symbol")
        return tuple(self._one(symbol, context.simulation_time) for symbol in context.symbols)

    def _one(self, raw_symbol: str, simulation_time: datetime) -> MarketState:
        symbol = _binance_symbol(raw_symbol)
        as_of_ms = _ms(simulation_time)
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
                f"no completed Binance 1m candle for {symbol} at {simulation_time.isoformat()}"
            )
        event_time = _from_ms(int(candle[0]))
        knowledge_time = _from_ms(int(candle[6]))
        return MarketState(
            ref=f"binance:{symbol}:{knowledge_time.isoformat()}",
            symbol=symbol,
            source="binance",
            url=f"{_HOSTS[0]}/api/v3/klines?symbol={symbol}&interval=1m",
            event_time=event_time,
            knowledge_time=knowledge_time,
            simulation_time=simulation_time,
            open=_num(candle[1]),
            high=_num(candle[2]),
            low=_num(candle[3]),
            close=_num(candle[4]),
            volume=_num(candle[5]),
            trades=int(candle[8]),
        )


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
