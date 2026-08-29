from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import json
import re
import time

from memory.context_gateway import ResearchContextStore
from memory.execution import MarketTape, PricePrint
from memory.types import Evidence


HttpFetcher = Callable[[str], object]

_YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart"
_GDELT_DOC = "http://api.gdeltproject.org/api/v2/doc/doc"
_INTERVALS = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "60m": timedelta(hours=1),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}
_MAX_SPAN = {
    "1m": timedelta(days=7),
    "5m": timedelta(days=30),
    "15m": timedelta(days=30),
    "60m": timedelta(days=60),
    "1h": timedelta(days=60),
    "1d": timedelta(days=3650),
}
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0",
}


class HistoryFeedError(RuntimeError):
    """A historical equity feed request failed or returned an unusable payload."""


def equity_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper().replace("-", ".")
    if not re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z])?", cleaned):
        raise ValueError(f"universe symbols must be equity tickers, got {symbol!r}")
    return cleaned


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _http_json(url: str, *, attempts: int = 5, timeout: int = 60) -> object:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        request = Request(url, headers=_HEADERS)
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            if not raw.strip():
                last = HistoryFeedError("empty feed body")
                if attempt == attempts:
                    return {}
                time.sleep(min(5 * attempt, 20))
                continue
            break
        except (HTTPError, URLError, TimeoutError, ConnectionResetError) as exc:
            last = exc
            if attempt == attempts:
                raise HistoryFeedError(f"historical feed failed: {exc}") from exc
            time.sleep(min(5 * attempt, 20))
    else:
        raise HistoryFeedError(f"historical feed failed: {last}") from last
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HistoryFeedError("historical feed returned non-JSON") from exc


def load_equity_tape(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    interval: str = "15m",
    fetch: HttpFetcher | None = None,
) -> tuple[MarketTape, dict[str, str]]:
    start = _aware(start, "start")
    end = _aware(end, "end")
    if end < start:
        raise ValueError("end cannot precede start")
    if interval not in _INTERVALS:
        raise ValueError(f"unsupported interval: {interval}")
    getter = fetch or _http_json
    width = _INTERVALS[interval]
    span = _MAX_SPAN[interval]
    tape = MarketTape()
    names: dict[str, str] = {}
    lookback = min(width, timedelta(days=1))
    lookahead = min(span, timedelta(days=1))
    for raw in symbols:
        symbol = equity_symbol(raw)
        cursor = start - lookback
        limit = end + lookahead
        name = symbol
        found = False
        while cursor < limit:
            chunk_end = min(cursor + span, limit)
            if chunk_end <= cursor:
                break
            query = urlencode(
                {
                    "period1": int(cursor.timestamp()),
                    "period2": int(chunk_end.timestamp()),
                    "interval": interval,
                }
            )
            payload = getter(f"{_YAHOO_CHART}/{symbol}?{query}")
            prints, name = _yahoo_prints(payload, symbol, width)
            for item in prints:
                tape.add(item)
                found = True
            cursor = chunk_end
        if not found:
            raise HistoryFeedError(f"no Yahoo bars for {symbol}")
        names[symbol] = name
    return tape, names


def load_equity_news(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    names: Mapping[str, str] | None = None,
    fetch: HttpFetcher | None = None,
) -> tuple[Evidence, ...]:
    start = _aware(start, "start")
    end = _aware(end, "end")
    getter = fetch or _http_json
    labels = names or {}
    articles: dict[str, Evidence] = {}
    for raw in symbols:
        symbol = equity_symbol(raw)
        query = _gdelt_query(symbol, labels.get(symbol, ""))
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=7), end)
            print(
                f"fetching GDELT {symbol} {cursor.date()} to {chunk_end.date()}",
                flush=True,
            )
            params = urlencode(
                {
                    "query": query,
                    "mode": "ArtList",
                    "maxrecords": 250,
                    "startdatetime": _gdelt_stamp(cursor),
                    "enddatetime": _gdelt_stamp(chunk_end),
                    "format": "json",
                    "sort": "DateDesc",
                }
            )
            try:
                payload = getter(f"{_GDELT_DOC}?{params}")
            except HistoryFeedError as exc:
                print(f"GDELT chunk skipped: {exc}", flush=True)
                payload = {}
            found = 0
            for article in _gdelt_articles(payload, symbol):
                if start <= article.knowledge_time <= end:
                    articles[article.ref] = article
                    found += 1
            print(f"GDELT chunk articles: {found}", flush=True)
            cursor = chunk_end
            if cursor < end and fetch is None:
                time.sleep(6)
    return tuple(sorted(articles.values(), key=lambda item: item.knowledge_time))


def load_historical_session(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    interval: str = "15m",
    fetch: HttpFetcher | None = None,
) -> tuple[ResearchContextStore, MarketTape]:
    tape, names = load_equity_tape(
        symbols, start=start, end=end, interval=interval, fetch=fetch
    )
    print(f"loaded {len(tape.prints)} Yahoo prints for {', '.join(names)}", flush=True)
    store = ResearchContextStore()
    articles = load_equity_news(
        symbols, start=start, end=end, names=names, fetch=fetch
    )
    print(f"loaded {len(articles)} GDELT articles", flush=True)
    for article in articles:
        store.append_news(article)
    return store, tape


def _yahoo_prints(
    payload: object, symbol: str, width: timedelta
) -> tuple[list[PricePrint], str]:
    if not isinstance(payload, dict):
        raise HistoryFeedError("unexpected Yahoo chart payload")
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        raise HistoryFeedError("unexpected Yahoo chart payload")
    if chart.get("error"):
        raise HistoryFeedError(f"Yahoo chart error for {symbol}: {chart['error']}")
    results = chart.get("result")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise HistoryFeedError(f"no Yahoo chart result for {symbol}")
    result = results[0]
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    name = str(meta.get("shortName") or meta.get("longName") or symbol)
    stamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(stamps, list) or not isinstance(indicators, dict):
        raise HistoryFeedError(f"Yahoo chart missing series for {symbol}")
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or not quotes or not isinstance(quotes[0], dict):
        raise HistoryFeedError(f"Yahoo chart missing quotes for {symbol}")
    closes = quotes[0].get("close")
    if not isinstance(closes, list):
        raise HistoryFeedError(f"Yahoo chart missing closes for {symbol}")
    prints: list[PricePrint] = []
    for stamp, close in zip(stamps, closes):
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
            continue
        if close is None or isinstance(close, bool) or not isinstance(close, (int, float)):
            continue
        if close <= 0:
            continue
        known = datetime.fromtimestamp(int(stamp), tz=timezone.utc) + width
        prints.append(PricePrint(symbol=symbol, price=float(close), knowledge_time=known))
    return prints, name


def _gdelt_query(symbol: str, name: str) -> str:
    if name and name.upper() != symbol:
        return f'({symbol} OR "{name}") sourcelang:english'
    return f"{symbol} sourcelang:english"


def _gdelt_stamp(when: datetime) -> str:
    return when.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def _gdelt_articles(payload: object, symbol: str) -> list[Evidence]:
    if payload in ({}, None):
        return []
    if not isinstance(payload, dict):
        raise HistoryFeedError("unexpected GDELT payload")
    rows = payload.get("articles")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise HistoryFeedError("unexpected GDELT article list")
    articles: list[Evidence] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = row.get("title")
        url = row.get("url")
        seen = row.get("seendate")
        domain = str(row.get("domain") or "gdelt").strip() or "gdelt"
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        known = _parse_gdelt_time(seen)
        if known is None:
            continue
        articles.append(
            Evidence(
                ref=f"gdelt:{quote(url, safe='')[:80]}",
                source=domain,
                url=url,
                published_at=known,
                title=title.strip(),
                summary=str(row.get("seeninfo") or title).strip()[:500],
                symbols=(symbol,),
                knowledge_time=known,
            )
        )
    return articles


def _parse_gdelt_time(value: object) -> datetime | None:
    if not isinstance(value, str) or len(value) < 15:
        return None
    try:
        return datetime.strptime(value[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
