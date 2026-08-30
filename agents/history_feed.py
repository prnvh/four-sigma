from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import hashlib
import json
import re
import time

from math import sqrt
from statistics import fmean, pstdev

from memory.context_gateway import ResearchContextStore
from memory.execution import MarketTape, PricePrint
from memory.timing_risk import tape_facts
from memory.types import (
    CompanyEntityRecord,
    CompanyRecord,
    CompanyRecordType,
    Evidence,
    MarketFeature,
    MarketFeatureType,
)


HttpFetcher = Callable[[str], object]
_FEED_CACHE = Path(__file__).resolve().parents[1] / ".qfirm-cache" / "feeds"
LISTED_EQUITY_PROFILES = {
    "AAPL": ("NASDAQ", "Technology", "Consumer Electronics"),
    "MSFT": ("NASDAQ", "Technology", "Software—Infrastructure"),
    "NVDA": ("NASDAQ", "Technology", "Semiconductors"),
    "AMZN": ("NASDAQ", "Consumer Cyclical", "Internet Retail"),
    "META": ("NASDAQ", "Communication Services", "Internet Content & Information"),
    "GOOGL": ("NASDAQ", "Communication Services", "Internet Content & Information"),
    "GOOG": ("NASDAQ", "Communication Services", "Internet Content & Information"),
}


def listed_profile(symbol: str) -> tuple[str, str, str] | None:
    return LISTED_EQUITY_PROFILES.get(symbol.strip().upper())


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


def _cached_http_json(url: str) -> object:
    _FEED_CACHE.mkdir(parents=True, exist_ok=True)
    path = _FEED_CACHE / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
    payload = _http_json(url)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
    return payload


def _feed_getter(fetch: HttpFetcher | None) -> HttpFetcher:
    return fetch or _cached_http_json


def _workers(fetch: HttpFetcher | None, max_workers: int, jobs: int) -> int:
    if fetch is not None or jobs <= 1:
        return 1
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    return min(max_workers, jobs)


def load_equity_tape(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    interval: str = "15m",
    fetch: HttpFetcher | None = None,
    max_workers: int = 8,
) -> tuple[MarketTape, dict[str, str]]:
    start = _aware(start, "start")
    end = _aware(end, "end")
    if end < start:
        raise ValueError("end cannot precede start")
    if interval not in _INTERVALS:
        raise ValueError(f"unsupported interval: {interval}")
    getter = _feed_getter(fetch)
    width = _INTERVALS[interval]
    span = _MAX_SPAN[interval]
    lookback = min(width, timedelta(days=1))
    lookahead = min(span, timedelta(days=1))
    cleaned = [equity_symbol(raw) for raw in symbols]

    def load_symbol(symbol: str) -> tuple[str, str, list[PricePrint]]:
        cursor = start - lookback
        limit = end + lookahead
        name = symbol
        found = False
        prints: list[PricePrint] = []
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
            chunk, name = _yahoo_prints(payload, symbol, width)
            prints.extend(chunk)
            found = found or bool(chunk)
            cursor = chunk_end
        if not found:
            raise HistoryFeedError(f"no Yahoo bars for {symbol}")
        return symbol, name, prints

    workers = _workers(fetch, max_workers, len(cleaned))
    if workers == 1:
        loaded = [load_symbol(symbol) for symbol in cleaned]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            loaded = list(pool.map(load_symbol, cleaned))
    all_prints: list[PricePrint] = []
    names: dict[str, str] = {}
    for symbol, name, prints in loaded:
        names[symbol] = name
        all_prints.extend(prints)
    return MarketTape(tuple(all_prints)), names


def load_equity_news(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    names: Mapping[str, str] | None = None,
    fetch: HttpFetcher | None = None,
    max_workers: int = 8,
) -> tuple[Evidence, ...]:
    start = _aware(start, "start")
    end = _aware(end, "end")
    getter = _feed_getter(fetch)
    labels = names or {}
    jobs: list[tuple[str, datetime, datetime]] = []
    for raw in symbols:
        symbol = equity_symbol(raw)
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=7), end)
            jobs.append((symbol, cursor, chunk_end))
            cursor = chunk_end

    def fetch_chunk(
        symbol: str, cursor: datetime, chunk_end: datetime
    ) -> list[Evidence]:
        print(
            f"fetching GDELT {symbol} {cursor.date()} to {chunk_end.date()}",
            flush=True,
        )
        query = _gdelt_query(symbol, labels.get(symbol, ""))
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
        found: list[Evidence] = []
        for article in _gdelt_articles(payload, symbol):
            if start <= article.knowledge_time <= end:
                found.append(article)
        print(f"GDELT chunk articles: {len(found)}", flush=True)
        return found

    workers = _workers(fetch, max_workers, len(jobs))
    if workers == 1:
        batches = [fetch_chunk(*job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batches = list(pool.map(lambda job: fetch_chunk(*job), jobs))
    articles: dict[str, Evidence] = {}
    for batch in batches:
        for article in batch:
            articles[article.ref] = article
    return tuple(sorted(articles.values(), key=lambda item: item.knowledge_time))


def load_historical_session(
    symbols: Sequence[str],
    *,
    start: datetime,
    end: datetime,
    interval: str = "15m",
    warmup_days: int = 0,
    fetch: HttpFetcher | None = None,
    max_workers: int = 8,
) -> tuple[ResearchContextStore, MarketTape]:
    if isinstance(warmup_days, bool) or not isinstance(warmup_days, int) or warmup_days < 0:
        raise ValueError("warmup_days must be a non-negative integer")
    tape, names = load_equity_tape(
        symbols,
        start=start - timedelta(days=warmup_days),
        end=end,
        interval=interval,
        fetch=fetch,
        max_workers=max_workers,
    )
    print(f"loaded {len(tape.prints)} Yahoo prints for {', '.join(names)}", flush=True)
    store = ResearchContextStore()
    articles = load_equity_news(
        symbols,
        start=start,
        end=end,
        names=names,
        fetch=fetch,
        max_workers=max_workers,
    )
    print(f"loaded {len(articles)} GDELT articles", flush=True)
    for article in articles:
        store.append_news(article)
    publish_tape_features(store, tape)
    print("Yahoo profiles skipped", flush=True)
    for symbol in names:
        _seed_listed_profile(store, symbol, knowledge_time=start)
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
    volumes = quotes[0].get("volume")
    if not isinstance(closes, list):
        raise HistoryFeedError(f"Yahoo chart missing closes for {symbol}")
    if not isinstance(volumes, list):
        volumes = [None] * len(closes)
    prints: list[PricePrint] = []
    for stamp, close, volume in zip(stamps, closes, volumes):
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
            continue
        if close is None or isinstance(close, bool) or not isinstance(close, (int, float)):
            continue
        if close <= 0:
            continue
        shares = None
        if volume is not None and not isinstance(volume, bool) and isinstance(volume, (int, float)):
            shares = float(volume)
        known = datetime.fromtimestamp(int(stamp), tz=timezone.utc) + width
        prints.append(
            PricePrint(
                symbol=symbol,
                price=float(close),
                knowledge_time=known,
                volume=shares,
            )
        )
    return prints, name


def publish_tape_features(store: ResearchContextStore, tape: MarketTape) -> None:
    existing = {item.ref for item in store._market_evidence()}
    last_by_day: dict[tuple[str, object], datetime] = {}
    for item in tape.prints:
        key = (item.symbol, item.knowledge_time.date())
        known = last_by_day.get(key)
        if known is None or item.knowledge_time > known:
            last_by_day[key] = item.knowledge_time
    for (symbol, _), when in sorted(last_by_day.items(), key=lambda item: item[1]):
        for feature in _tape_features_as_of(tape, symbol, when):
            if feature.ref in existing:
                continue
            store.append_market_feature(feature)
            existing.add(feature.ref)


def _tape_features_as_of(
    tape: MarketTape, symbol: str, when: datetime
) -> tuple[MarketFeature, ...]:
    url = f"{_YAHOO_CHART}/{symbol}"
    found: list[MarketFeature] = []
    facts = tape_facts(tape, symbol, when)
    if facts.return_20d is not None:
        found.append(
            MarketFeature(
                ref=f"market:{symbol}:return_20d:{when.date().isoformat()}",
                symbol=symbol,
                source="Yahoo Finance",
                url=url,
                knowledge_time=when,
                feature=MarketFeatureType.RETURN_20D,
                value=facts.return_20d,
                unit="decimal_return",
            )
        )
    volatility = annualized_volatility(tape, symbol, when)
    if volatility is not None:
        found.append(
            MarketFeature(
                ref=f"market:{symbol}:volatility_20d:{when.date().isoformat()}",
                symbol=symbol,
                source="Yahoo Finance",
                url=url,
                knowledge_time=when,
                feature=MarketFeatureType.VOLATILITY_20D,
                value=volatility,
                unit="annualized_stdev",
            )
        )
    return tuple(found)


def annualized_volatility(tape: MarketTape, symbol: str, when: datetime) -> float | None:
    start = when - timedelta(days=20)
    closes: dict[object, float] = {}
    for item in tape.prints_for(symbol, start=start, end=when):
        closes[item.knowledge_time.date()] = item.price
    prices = [closes[day] for day in sorted(closes)]
    if len(prices) < 3:
        return None
    returns = [prices[index] / prices[index - 1] - 1 for index in range(1, len(prices))]
    if any(left <= -1 for left in returns):
        return None
    return pstdev(returns) * (252 ** 0.5)


def dollar_adv(tape: MarketTape, symbol: str, when: datetime) -> float | None:
    start = when - timedelta(days=20)
    daily: dict[object, float] = {}
    for item in tape.prints_for(symbol, start=start, end=when):
        if item.volume is None:
            continue
        day = item.knowledge_time.date()
        daily[day] = daily.get(day, 0.0) + item.price * item.volume
    if not daily:
        return None
    return sum(daily.values()) / len(daily)


def rolling_correlations(
    tape: MarketTape,
    symbols: Sequence[str],
    when: datetime,
    *,
    lookback_days: int = 60,
    min_overlap: int = 5,
) -> dict[str, dict[str, float]]:
    """Point-in-time daily-return correlations with a conservative sparse-data fallback."""
    if lookback_days < 2 or min_overlap < 2:
        raise ValueError("correlation lookback and overlap must be at least two")
    selected = tuple(dict.fromkeys(equity_symbol(item) for item in symbols))
    start = when - timedelta(days=lookback_days)
    closes: dict[str, dict[object, float]] = {symbol: {} for symbol in selected}
    for symbol in selected:
        for item in tape.prints_for(symbol, start=start, end=when):
            closes[item.symbol][item.knowledge_time.date()] = item.price
    returns: dict[str, dict[object, float]] = {}
    for symbol, by_day in closes.items():
        days = sorted(by_day)
        returns[symbol] = {
            days[index]: by_day[days[index]] / by_day[days[index - 1]] - 1
            for index in range(1, len(days))
        }
    matrix: dict[str, dict[str, float]] = {symbol: {} for symbol in selected}
    for index, left in enumerate(selected):
        matrix[left][left] = 1.0
        for right in selected[index + 1 :]:
            common = sorted(set(returns[left]) & set(returns[right]))
            correlation = 1.0
            if len(common) >= min_overlap:
                xs = [returns[left][day] for day in common]
                ys = [returns[right][day] for day in common]
                x_mean, y_mean = fmean(xs), fmean(ys)
                numerator = sum(
                    (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
                )
                x_scale = sqrt(sum((x - x_mean) ** 2 for x in xs))
                y_scale = sqrt(sum((y - y_mean) ** 2 for y in ys))
                if x_scale > 0 and y_scale > 0:
                    correlation = max(-1.0, min(1.0, numerator / (x_scale * y_scale)))
            matrix[left][right] = correlation
            matrix[right][left] = correlation
    return matrix


def _load_yahoo_profile(
    store: ResearchContextStore,
    symbol: str,
    *,
    knowledge_time: datetime,
    fetch: HttpFetcher,
) -> None:
    urls = (
        "https://query1.finance.yahoo.com/v10/finance/quoteSummary/"
        f"{symbol}?modules=assetProfile,price",
        "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
        f"{symbol}?modules=assetProfile,price",
    )
    payload = None
    url = urls[0]
    last_error: Exception | None = None
    for url in urls:
        try:
            payload = fetch(url)
            break
        except HistoryFeedError as exc:
            last_error = exc
    if payload is None:
        print(f"Yahoo profile skipped {symbol}: {last_error}", flush=True)
        return
    sector, industry, exchange = _yahoo_profile_fields(payload)
    if not sector or not industry or not exchange:
        return
    record = CompanyRecord(
        ref=f"yahoo:profile:{symbol}:sector",
        symbol=symbol,
        source="Yahoo Finance",
        url=url,
        knowledge_time=knowledge_time,
        record_type=CompanyRecordType.COMPANY_PROFILE,
        label="sector",
        value=sector,
    )
    if record.ref in {item.ref for item in store._company_evidence()}:
        return
    store.append_company_record(record)
    store.append_company_record(
        CompanyRecord(
            ref=f"yahoo:profile:{symbol}:industry",
            symbol=symbol,
            source="Yahoo Finance",
            url=url,
            knowledge_time=knowledge_time,
            record_type=CompanyRecordType.COMPANY_PROFILE,
            label="industry",
            value=industry,
        )
    )
    store.append_company_entity(
        CompanyEntityRecord(
            ticker=symbol,
            exchange=exchange,
            sector=sector,
            industry=industry,
            identifiers={"yahoo": symbol},
            fundamental_references=(
                f"yahoo:profile:{symbol}:sector",
                f"yahoo:profile:{symbol}:industry",
            ),
            knowledge_time=knowledge_time,
        )
    )
    print(f"company profile {symbol} {exchange} {sector}", flush=True)


def _seed_listed_profile(
    store: ResearchContextStore, symbol: str, *, knowledge_time: datetime
) -> None:
    profile = listed_profile(symbol)
    if profile is None:
        return
    exchange, sector, industry = profile
    url = f"https://finance.yahoo.com/quote/{symbol}/profile"
    sector_ref = f"listed:profile:{symbol}:sector"
    if sector_ref in {item.ref for item in store._company_evidence()}:
        return
    store.append_company_record(
        CompanyRecord(
            ref=sector_ref,
            symbol=symbol,
            source="Listed equity profile",
            url=url,
            knowledge_time=knowledge_time,
            record_type=CompanyRecordType.COMPANY_PROFILE,
            label="sector",
            value=sector,
        )
    )
    store.append_company_record(
        CompanyRecord(
            ref=f"listed:profile:{symbol}:industry",
            symbol=symbol,
            source="Listed equity profile",
            url=url,
            knowledge_time=knowledge_time,
            record_type=CompanyRecordType.COMPANY_PROFILE,
            label="industry",
            value=industry,
        )
    )
    store.append_company_entity(
        CompanyEntityRecord(
            ticker=symbol,
            exchange=exchange,
            sector=sector,
            industry=industry,
            identifiers={"yahoo": symbol},
            fundamental_references=(
                sector_ref,
                f"listed:profile:{symbol}:industry",
            ),
            knowledge_time=knowledge_time,
        )
    )
    print(f"listed profile {symbol} {exchange} {sector}", flush=True)


def _yahoo_profile_fields(payload: object) -> tuple[str, str, str]:
    if not isinstance(payload, dict):
        return "", "", ""
    summary = payload.get("quoteSummary")
    if not isinstance(summary, dict):
        return "", "", ""
    results = summary.get("result")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return "", "", ""
    row = results[0]
    profile = row.get("assetProfile") if isinstance(row.get("assetProfile"), dict) else {}
    price = row.get("price") if isinstance(row.get("price"), dict) else {}
    sector = str(profile.get("sector") or "").strip()
    industry = str(profile.get("industry") or "").strip()
    exchange = str(
        price.get("exchangeName") or price.get("fullExchangeName") or price.get("exchange") or ""
    ).strip()
    return sector, industry, exchange


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
                ref=f"gdelt:{hashlib.sha256(url.encode()).hexdigest()[:16]}",
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
