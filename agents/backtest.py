from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from memory.audit_logger import AuditEvent, AuditLedger
from memory.context_gateway import ContextGateway, ContextSnapshot
from memory.execution import ExecutionConfig, MarketTape, SimulatedExecution
from memory.governance_gate import (
    GovernanceDecision,
    GovernanceGate,
    GovernanceOutcome,
    ProposePermissions,
)
from memory.news_governance import NewsInsightGovernanceRules
from memory.portfolio import Fill, PortfolioBook, PortfolioError, PortfolioSnapshot
from memory.portfolio_risk import PortfolioRiskCalculator, PortfolioRiskError, PortfolioRiskInput
from memory.position_risk import (
    DeterministicPositionRiskEngine,
    PositionRiskInput,
    PositionRiskLimits,
    RiskCheckResult,
)
from memory.timing_risk import (
    TimingAction,
    hold_working_position,
    may_reenter_stopped_side,
    pass_through_timing,
    position_return,
    scaled_stop_pct,
    should_stop_loser,
    should_trail_winner,
    tape_reaction,
)
from memory.promotion import PromotionProposal
from memory.shared_mem import SharedMemory
from memory.strategy_metrics import StrategyMetrics, calculate_strategy_metrics
from memory.trade_lifecycle import TradeLifecycle
from memory.types import (
    AgentId,
    AuditEventId,
    CompanyAnalysis,
    CompanyAnalysisRecord,
    CreatedAt,
    Direction,
    EntityId,
    EvidenceId,
    Finding,
    InsightId,
    InsightRevision,
    OutcomeDefinition,
    PromotedInsight,
    ProposalId,
    RiskAnalysis,
    RunId,
    SimulationTime,
    TradeCandidate,
    TradeCandidateId,
    TradeCandidateStatus,
    TradeSide,
)
from memory.working_mem import WorkingMemory, WorkingMemoryCategory

from .company_analyst import CompanyAnalyst
from .history_feed import (
    annualized_volatility,
    dollar_adv,
    equity_symbol,
    listed_profile,
    load_historical_session,
    publish_tape_features,
    rolling_correlations,
)
from .news_analyst import NewsAnalyst
from .portfolio_risk_agent import (
    PortfolioRiskAgent,
    PortfolioRiskAgentContext,
    PortfolioRiskRecommendation,
)
from .registry import REGISTRY, AgentRegistry, AgentSpec
from .risk_analyst import RiskAnalyst
from .trade_constructor import TradeConstructor
from .trade_risk import TradeRiskAnalyst


def _news_due(cadence: object, last_call: datetime | None, now: datetime) -> bool:
    if last_call is None:
        return True
    if cadence == "tick":
        return True
    if cadence == "day":
        return last_call.date() != now.date()
    if isinstance(cadence, timedelta):
        return now - last_call >= cadence
    raise ValueError("news_cadence must be 'tick', 'day', or a positive timedelta")


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _held(snapshot: PortfolioSnapshot) -> str:
    if not snapshot.positions:
        return "flat"
    parts: list[str] = []
    for position in snapshot.positions:
        sign = "+" if position.quantity > 0 else ""
        parts.append(f"{position.instrument}{sign}{position.quantity:.2f}")
    return ",".join(parts)


def _log_book(kind: str, snapshot: PortfolioSnapshot, start_equity: float) -> None:
    pnl = snapshot.equity - start_equity
    ret = (snapshot.equity / start_equity) - 1 if start_equity else 0.0
    print(
        f"{kind} {snapshot.knowledge_time.isoformat()} "
        f"equity={snapshot.equity:.2f} pnl={pnl:+.2f} ret={ret:+.2%} "
        f"cash={snapshot.cash:.2f} realized={snapshot.realized_pnl:+.2f} "
        f"unrealized={snapshot.unrealized_pnl:+.2f} "
        f"fees={snapshot.fees:.2f} slip={snapshot.slippage:.2f} "
        f"{_held(snapshot)}",
        flush=True,
    )


def _defer_weak_thesis(advisory: object | None, current_qty: float) -> bool:
    if advisory is None or current_qty != 0:
        return False
    fail = getattr(advisory, "failure_probability_pct", None)
    success = getattr(advisory, "success_probability_pct", None)
    if not isinstance(fail, (int, float)) or not isinstance(success, (int, float)):
        return False
    return fail >= 50 and fail >= success + 20


def _open_pnl(
    snapshot: PortfolioSnapshot,
    symbol: str,
    tape: MarketTape,
    now: datetime,
) -> float | None:
    position = next(
        (item for item in snapshot.positions if item.instrument == symbol),
        None,
    )
    if position is None or position.average_entry <= 0:
        return None
    mark = position.market_price
    if mark is None:
        mark = tape.prices_as_of(now).get(symbol)
    if mark is None:
        return None
    return position_return(position.quantity, position.average_entry, mark)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    context_snapshots: tuple[ContextSnapshot, ...]
    audit_events: tuple[AuditEvent, ...]
    snapshots: tuple[PortfolioSnapshot, ...]
    candidates: tuple[TradeCandidate, ...]
    fills: tuple[Fill, ...]
    invocations: tuple[str, ...]
    findings: tuple[Finding, ...]
    promotions: tuple[GovernanceDecision, ...]
    company_analyses: tuple[CompanyAnalysis, ...]
    risk_analyses: tuple[RiskAnalysis, ...]
    metrics: StrategyMetrics
    final: PortfolioSnapshot


@dataclass(frozen=True, slots=True)
class _PendingFill:
    fill: Fill
    candidate_id: TradeCandidateId | None = None
    closes_candidate_id: TradeCandidateId | None = None


def _risk_review(
    candidate: TradeCandidate,
    max_position_pct: float,
    *,
    tape,
    insights,
    articles,
    now,
    existing_quantity: float,
    trade_risk: TradeRiskAnalyst | None,
    already_reviewed_today: bool,
):
    if trade_risk is None or already_reviewed_today:
        return pass_through_timing(
            candidate, tape=tape, now=now, max_position_pct=max_position_pct
        )
    return trade_risk.review(
        candidate,
        tape=tape,
        insights=insights,
        articles=articles,
        now=now,
        existing_quantity=existing_quantity,
        max_position_pct=max_position_pct,
    )


def _await_mapped(futures: dict):
    results = {}
    errors: dict[object, BaseException] = {}
    for future in as_completed(futures):
        key = futures[future]
        try:
            results[key] = future.result()
        except (ValueError, RuntimeError) as exc:
            errors[key] = exc
    return results, errors


@dataclass(frozen=True, slots=True)
class _ReadyTrade:
    symbol: str
    view: object
    proposed: TradeCandidate
    current_qty: float
    articles: tuple


@dataclass(frozen=True, slots=True)
class _SizedTrade:
    proposed: TradeCandidate
    timed: object
    invocations: tuple[str, ...]
    flatten: bool
    defer: bool


class _ArticleEvidence:
    def __init__(self, article: object) -> None:
        known = getattr(article, "knowledge_time")
        self.created_at = CreatedAt(known)
        source = str(getattr(article, "source", "")).strip().lower()
        host = urlparse(str(getattr(article, "url", ""))).hostname or ""
        approved = (
            "reuters.com",
            "apnews.com",
            "ap.org",
            "bloomberg.com",
            "ft.com",
            "wsj.com",
            "example.test",
            "cnbc.com",
            "bbc.com",
            "bbc.co.uk",
            "nytimes.com",
            "washingtonpost.com",
            "theguardian.com",
            "cnn.com",
            "npr.org",
            "forbes.com",
            "marketwatch.com",
            "barrons.com",
            "yahoo.com",
            "businessinsider.com",
            "fortune.com",
            "economist.com",
            "nikkei.com",
            "scmp.com",
            "techcrunch.com",
            "theverge.com",
            "wired.com",
            "arstechnica.com",
            "morningstar.com",
            "investopedia.com",
            "fool.com",
            "seekingalpha.com",
            "foxbusiness.com",
            "nbcnews.com",
            "cbsnews.com",
            "usatoday.com",
            "latimes.com",
            "time.com",
            "axios.com",
            "politico.com",
        )
        self.source_class = (
            "reputable_newswire"
            if any(
                source == domain
                or source.endswith(f".{domain}")
                or host == domain
                or host.endswith(f".{domain}")
                for domain in approved
            )
            else "unverified_web"
        )

    def visible_as_of(self, when: SimulationTime) -> bool:
        return self.created_at.value <= when.value


class _StoreEvidence:
    def __init__(self, store: object) -> None:
        self._store = store

    def resolve(self, agent_id: object, ref: object, *, simulation_time: SimulationTime):
        key = ref.value if hasattr(ref, "value") else str(ref)
        for article in self._store._news_records():
            if article.ref != key:
                continue
            if article.knowledge_time is None or article.knowledge_time > simulation_time.value:
                return None
            return _ArticleEvidence(article)
        return None


class _InsightMirror:
    def __init__(self, shared: SharedMemory, store: object) -> None:
        self._shared = shared
        self._store = store

    def apply_approved(self, proposal: PromotionProposal, *, decided_at: SimulationTime):
        version = self._shared.apply_approved(proposal, decided_at=decided_at)
        finding = version.value
        if isinstance(finding, Finding):
            valid_until = version.valid_until.value if version.valid_until is not None else None
            self._store.append_promoted_insight(
                PromotedInsight(
                    ref=version.insight_id.value,
                    symbol=finding.subject,
                    claim=finding.claim,
                    direction=finding.direction,
                    confidence=finding.confidence,
                    evidence_refs=finding.evidence_refs,
                    knowledge_time=decided_at.value,
                    valid_until=valid_until,
                )
            )
        return version


class BacktestRunner:
    """Walks historical time as if live. Daily steps; no wall-clock simulation time."""

    def __init__(
        self,
        *,
        store: ContextGateway | object,
        tape: MarketTape,
        registry: AgentRegistry | None = None,
        news_analyst: NewsAnalyst | None = None,
        company_analyst: CompanyAnalyst | None = None,
        risk_analyst: RiskAnalyst | None = None,
        portfolio_risk: PortfolioRiskAgent | None = None,
        trade_risk: TradeRiskAnalyst | None = None,
    ) -> None:
        if isinstance(store, ContextGateway):
            self.gateway = store
            self.store = store._shared_memory
        else:
            self.store = store
            self.gateway = ContextGateway(store)
        if not isinstance(tape, MarketTape):
            raise TypeError("tape must be MarketTape")
        self.tape = tape
        self.registry = registry or REGISTRY
        self.news_analyst = news_analyst
        self.company_analyst = company_analyst
        self.risk_analyst = risk_analyst
        self.portfolio_risk = portfolio_risk
        self.trade_risk = trade_risk
        self._invocations: list[str] = []
        self._stop_loss_pct = 0.08
        self._stop_volatility_multiple = 1.5
        self._stop_vol_ceiling = 0.60
        self._stop_reentry_cooldown = timedelta(days=1)
        self._stopped_sides: dict[str, tuple[TradeSide, datetime]] = {}
        self._trailing_stop_activation_pct = 0.04
        self._trailing_stop_floor_pct = 0.02
        self._trailing_stop_volatility_multiple = 1.0
        self._favorable_marks: dict[str, tuple[TradeSide, float]] = {}

    def run(
        self,
        start: datetime,
        end: datetime,
        universe: Sequence[str],
        agent_versions: Sequence[str],
        strategy_config: Mapping[str, object],
    ) -> BacktestResult:
        start = _aware(start, "start")
        end = _aware(end, "end")
        if end < start:
            raise ValueError("end cannot precede start")
        symbols = tuple(equity_symbol(item) for item in universe)
        if not symbols:
            raise ValueError("universe must contain at least one symbol")
        versions = tuple(agent_versions)
        names = {key.split(":", 1)[0] for key in versions}
        for key in versions:
            self.registry.get(key)
            name = key.split(":", 1)[0]
            if name not in {
                "trade_constructor",
                "news_analyst",
                "company_analyst",
                "risk_llm",
                "portfolio_risk",
            }:
                raise ValueError(
                    f"backtest runner has no historical binding for {key}"
                )
        if "news_analyst" in names and self.news_analyst is None:
            raise ValueError("news_analyst:v1 requires a NewsAnalyst binding")
        if "company_analyst" in names and self.company_analyst is None:
            raise ValueError("company_analyst:v1 requires a CompanyAnalyst binding")
        if "risk_llm:v1" in versions and self.risk_analyst is None:
            raise ValueError("risk_llm:v1 requires a RiskAnalyst binding")
        if "portfolio_risk" in names and self.portfolio_risk is None:
            raise ValueError("portfolio_risk:v1 requires a PortfolioRiskAgent binding")
        if "risk_llm:trade_v1" in versions and self.trade_risk is None:
            raise ValueError("risk_llm:trade_v1 requires a TradeRiskAnalyst binding")
        if not isinstance(strategy_config, Mapping):
            raise TypeError("strategy_config must be a mapping")
        cash = strategy_config.get("starting_cash", 1000)
        step = strategy_config.get("step", "bar")
        max_pct = strategy_config.get("max_position_pct", 0.40)
        horizon_days = strategy_config.get("insight_horizon_days", 3)
        min_evidence = strategy_config.get("min_evidence_count", 1)
        if step != "bar" and (not isinstance(step, timedelta) or step <= timedelta(0)):
            raise ValueError("strategy_config step must be 'bar' or a positive timedelta")
        if isinstance(max_pct, bool) or not isinstance(max_pct, (int, float)) or max_pct < 0:
            raise ValueError("max_position_pct must be a non-negative number")
        if (
            isinstance(horizon_days, bool)
            or not isinstance(horizon_days, int)
            or horizon_days < 1
        ):
            raise ValueError("insight_horizon_days must be a positive integer")
        if (
            isinstance(min_evidence, bool)
            or not isinstance(min_evidence, int)
            or min_evidence < 1
        ):
            raise ValueError("min_evidence_count must be a positive integer")
        stop_loss_pct = strategy_config.get("stop_loss_pct", 0.08)
        if (
            isinstance(stop_loss_pct, bool)
            or not isinstance(stop_loss_pct, (int, float))
            or stop_loss_pct < 0
        ):
            raise ValueError("stop_loss_pct must be a non-negative number")
        self._stop_loss_pct = float(stop_loss_pct)
        stop_volatility_multiple = strategy_config.get("stop_volatility_multiple", 1.5)
        if (
            isinstance(stop_volatility_multiple, bool)
            or not isinstance(stop_volatility_multiple, (int, float))
            or stop_volatility_multiple < 0
        ):
            raise ValueError("stop_volatility_multiple must be a non-negative number")
        stop_reentry_cooldown = strategy_config.get(
            "stop_reentry_cooldown", timedelta(days=1)
        )
        if (
            not isinstance(stop_reentry_cooldown, timedelta)
            or stop_reentry_cooldown < timedelta(0)
        ):
            raise ValueError("stop_reentry_cooldown must be a non-negative timedelta")
        self._stop_volatility_multiple = float(stop_volatility_multiple)
        self._stop_reentry_cooldown = stop_reentry_cooldown
        vol_ceiling = strategy_config.get("max_annualized_volatility", 0.60)
        if (
            isinstance(vol_ceiling, bool)
            or not isinstance(vol_ceiling, (int, float))
            or vol_ceiling <= 0
        ):
            raise ValueError("max_annualized_volatility must be a positive number")
        self._stop_vol_ceiling = float(vol_ceiling)
        self._stopped_sides = {}
        trailing_stop_activation_pct = strategy_config.get(
            "trailing_stop_activation_pct", 0.04
        )
        trailing_stop_floor_pct = strategy_config.get(
            "trailing_stop_floor_pct", 0.02
        )
        trailing_stop_volatility_multiple = strategy_config.get(
            "trailing_stop_volatility_multiple", 1.0
        )
        for name, value in (
            ("trailing_stop_activation_pct", trailing_stop_activation_pct),
            ("trailing_stop_floor_pct", trailing_stop_floor_pct),
            (
                "trailing_stop_volatility_multiple",
                trailing_stop_volatility_multiple,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative number")
        self._trailing_stop_activation_pct = float(trailing_stop_activation_pct)
        self._trailing_stop_floor_pct = float(trailing_stop_floor_pct)
        self._trailing_stop_volatility_multiple = float(
            trailing_stop_volatility_multiple
        )
        self._favorable_marks = {}
        constructor = None
        constructor_spec = self._spec(versions, "trade_constructor")
        if constructor_spec is not None:
            constructor = TradeConstructor(constructor_spec)
        execution = SimulatedExecution(
            ExecutionConfig(
                slippage_bps=float(strategy_config.get("slippage_bps", 0)),
                fee_bps=float(strategy_config.get("fee_bps", 0)),
            )
        )
        ledger = AuditLedger()
        lifecycle = TradeLifecycle(ledger)
        shared = SharedMemory(ledger)
        working = WorkingMemory(ledger)
        evidence = _StoreEvidence(self.store)
        gate = GovernanceGate(
            permissions=ProposePermissions(
                {AgentId("news_analyst"): frozenset({("insights", "claim")})}
            ),
            evidence=evidence,
            schema_check=lambda resource, field, value: None,
            shared=_InsightMirror(shared, self.store),
            ledger=ledger,
            rules=(
                NewsInsightGovernanceRules(
                    evidence=evidence,
                    allowed_source_classes={"reputable_newswire"},
                    min_evidence_count=min_evidence,
                ),
            ),
        )
        ticks = self._ticks(start, end, step)
        publish_tape_features(self.store, self.tape)
        requested_workers = strategy_config.get("max_workers", min(32, len(symbols)))
        if (
            isinstance(requested_workers, bool)
            or not isinstance(requested_workers, int)
            or requested_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        workers = min(32, len(symbols), requested_workers)
        print(
            f"walking {len(ticks)} ticks as if live "
            f"across {len(symbols)} tracks ({workers} workers)...",
            flush=True,
        )
        self._start_equity = float(cash)
        self._tick_count = len(ticks)
        book = PortfolioBook(float(cash), opened_at=start)
        pending: list[_PendingFill] = []
        active_candidates: dict[str, TradeCandidateId] = {}
        snapshots: list[PortfolioSnapshot] = []
        candidates: list[TradeCandidate] = []
        fills: list[Fill] = []
        invocations: list[str] = []
        self._invocations = invocations
        findings: list[Finding] = []
        promotions: list[GovernanceDecision] = []
        company_analyses: list[CompanyAnalysis] = []
        risk_analyses: list[RiskAnalysis] = []
        last_news: dict[str, datetime] = {
            symbol: datetime.min.replace(tzinfo=timezone.utc) for symbol in symbols
        }
        last_news_call: dict[str, datetime | None] = {symbol: None for symbol in symbols}
        last_insight_refs: dict[str, frozenset[str] | None] = {
            symbol: None for symbol in symbols
        }
        last_review_price: dict[str, float] = {}
        peak_equity_box = [float(cash)]
        position_risk = DeterministicPositionRiskEngine(
            _position_limits(
                float(max_pct),
                float(strategy_config.get("max_annualized_volatility", 0.60)),
            ),
            ledger,
        )
        portfolio_calc = PortfolioRiskCalculator()
        outcome = OutcomeDefinition(
            int(strategy_config.get("risk_horizon_days", 30)),
            float(strategy_config.get("success_return_pct", 10)),
            float(strategy_config.get("failure_return_pct", -10)),
        )
        news_cadence = strategy_config.get("news_cadence", "tick")
        if news_cadence not in {"tick", "day"} and (
            not isinstance(news_cadence, timedelta) or news_cadence <= timedelta(0)
        ):
            raise ValueError("news_cadence must be 'tick', 'day', or a positive timedelta")
        run_id = RunId(f"backtest:{start.isoformat()}:{end.isoformat()}")
        context_offset = len(self.gateway.snapshots.snapshot())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for tick, now in enumerate(ticks):
                self._step_tick(
                    tick=tick,
                    now=now,
                    end=end,
                    symbols=symbols,
                    names=names,
                    versions=versions,
                    constructor=constructor,
                    execution=execution,
                    lifecycle=lifecycle,
                    working=working,
                    gate=gate,
                    book=book,
                    pending=pending,
                    active_candidates=active_candidates,
                    snapshots=snapshots,
                    candidates=candidates,
                    fills=fills,
                    invocations=invocations,
                    findings=findings,
                    promotions=promotions,
                    company_analyses=company_analyses,
                    risk_analyses=risk_analyses,
                    last_news=last_news,
                    last_news_call=last_news_call,
                    last_insight_refs=last_insight_refs,
                    last_review_price=last_review_price,
                    position_risk=position_risk,
                    portfolio_calc=portfolio_calc,
                    outcome=outcome,
                    news_cadence=news_cadence,
                    run_id=run_id,
                    max_pct=max_pct,
                    horizon_days=horizon_days,
                    peak_equity_box=peak_equity_box,
                    pool=pool,
                )
        if snapshots:
            _log_book("day", snapshots[-1], self._start_equity)
            _log_book("month", snapshots[-1], self._start_equity)
        completed_snapshots = tuple(snapshots)
        completed_fills = tuple(fills)
        return BacktestResult(
            context_snapshots=self.gateway.snapshots.snapshot()[context_offset:],
            audit_events=ledger.snapshot(),
            snapshots=completed_snapshots,
            candidates=tuple(lifecycle.get(item.id) for item in candidates),
            fills=completed_fills,
            invocations=tuple(invocations),
            findings=tuple(findings),
            promotions=tuple(promotions),
            company_analyses=tuple(company_analyses),
            risk_analyses=tuple(risk_analyses),
            metrics=calculate_strategy_metrics(completed_snapshots, completed_fills),
            final=book.snapshot(),
        )

    def _step_tick(
        self,
        *,
        tick: int,
        now: datetime,
        end: datetime,
        symbols: tuple[str, ...],
        names: set[str],
        versions: tuple[str, ...],
        constructor,
        execution,
        lifecycle,
        working,
        gate,
        book,
        pending: list[_PendingFill],
        active_candidates: dict[str, TradeCandidateId],
        snapshots: list[PortfolioSnapshot],
        candidates: list[TradeCandidate],
        fills: list[Fill],
        invocations: list[str],
        findings: list[Finding],
        promotions: list[GovernanceDecision],
        company_analyses: list[CompanyAnalysis],
        risk_analyses: list[RiskAnalysis],
        last_news: dict[str, datetime],
        last_news_call: dict[str, datetime | None],
        last_insight_refs: dict[str, frozenset[str] | None],
        last_review_price: dict[str, float],
        position_risk,
        portfolio_calc,
        outcome,
        news_cadence,
        run_id: RunId,
        max_pct,
        horizon_days,
        peak_equity_box: list[float],
        pool,
    ) -> None:
        still_pending: list[_PendingFill] = []
        for pending_fill in pending:
            fill = pending_fill.fill
            if fill.knowledge_time <= now:
                try:
                    book.apply(fill)
                except PortfolioError as exc:
                    print(
                        f"fill dropped: {exc} {fill.instrument} {fill.side.value} "
                        f"qty={fill.quantity}",
                        flush=True,
                    )
                    continue
                fills.append(fill)
                print(
                    f"fill {fill.knowledge_time.isoformat()} {fill.instrument} "
                    f"{fill.side.value} qty={fill.quantity} px={fill.price} "
                    f"fee={fill.fee} slip={fill.slippage}",
                    flush=True,
                )
                _log_book("book", book.snapshot(), self._start_equity)
                if pending_fill.closes_candidate_id is not None:
                    lifecycle.transition(
                        pending_fill.closes_candidate_id,
                        TradeCandidateStatus.CLOSED,
                        event_id=self._trade_event_id(
                            run_id, pending_fill.closes_candidate_id, "closed"
                        ),
                        occurred_at=CreatedAt(fill.knowledge_time),
                        reason="position closed by simulated fill",
                        run_id=run_id,
                    )
                    active_candidates.pop(fill.instrument, None)
                if pending_fill.candidate_id is not None:
                    lifecycle.transition(
                        pending_fill.candidate_id,
                        TradeCandidateStatus.FILLED,
                        event_id=self._trade_event_id(
                            run_id, pending_fill.candidate_id, "filled"
                        ),
                        occurred_at=CreatedAt(fill.knowledge_time),
                        reason="simulated fill applied to portfolio",
                        run_id=run_id,
                    )
                    active_candidates[fill.instrument] = pending_fill.candidate_id
            else:
                still_pending.append(pending_fill)
        pending[:] = still_pending
        pending_symbols = {item.fill.instrument for item in pending}
        self._flatten_losers(
            now=now,
            end=end,
            execution=execution,
            book=book,
            pending=pending,
            pending_symbols=pending_symbols,
            active_candidates=active_candidates,
        )
        needs_review: set[str] = set()
        news_jobs: list[tuple[str, object, int]] = []
        if self.news_analyst is not None and "news_analyst" in names:
            for symbol in symbols:
                view = self.gateway.for_news_analyst(
                    agent_id="news_analyst",
                    symbol=symbol,
                    simulation_time=now,
                )
                fresh = [
                    article
                    for article in view.articles
                    if article.knowledge_time is not None
                    and article.knowledge_time > last_news[symbol]
                ]
                if not fresh:
                    continue
                if not _news_due(news_cadence, last_news_call[symbol], now):
                    continue
                news_jobs.append((symbol, view, len(fresh)))
        news_futs = {}
        if self.news_analyst is not None:
            for symbol, view, fresh_count in news_jobs:
                print(f"news {symbol} {now.isoformat()} fresh={fresh_count}", flush=True)
                news_futs[pool.submit(self.news_analyst.analyze, view)] = symbol
        tape_futs = {
            pool.submit(
                tape_reaction,
                self.tape,
                symbol,
                now,
                reference_price=last_review_price.get(symbol),
            ): symbol
            for symbol in symbols
        }
        news_findings, news_errors = _await_mapped(news_futs)
        tape_hits, _ = _await_mapped(tape_futs)
        for symbol, view, _fresh_count in news_jobs:
            if symbol in news_errors:
                print(f"news skipped: {news_errors[symbol]}", flush=True)
                last_news[symbol] = max(
                    article.knowledge_time
                    for article in view.articles
                    if article.knowledge_time is not None
                )
                last_news_call[symbol] = now
                continue
            finding = news_findings[symbol]
            invocations.append(self.news_analyst.spec.key)
            findings.append(finding)
            last_news[symbol] = max(
                article.knowledge_time
                for article in view.articles
                if article.knowledge_time is not None
            )
            last_news_call[symbol] = now
            decision = self._promote(
                finding,
                symbol=symbol,
                when=now,
                horizon_days=horizon_days,
                working=working,
                gate=gate,
                run_id=run_id,
                tick=tick,
            )
            promotions.append(decision)
            print(
                f"insight {symbol} {decision.outcome.value} "
                f"{finding.direction.value} {finding.confidence:.2f}",
                flush=True,
            )
            if (
                decision.outcome is GovernanceOutcome.APPROVED
                and finding.direction is not Direction.NEUTRAL
            ):
                needs_review.add(symbol)
        for symbol in symbols:
            if symbol in needs_review:
                continue
            reaction = tape_hits.get(symbol)
            if reaction is None or not reaction.triggered:
                continue
            needs_review.add(symbol)
            print(
                f"tape {symbol} reaction move={reaction.move:.2%} "
                f"typical={reaction.typical_daily_move:.2%}",
                flush=True,
            )
        review_names = tuple(symbol for symbol in symbols if symbol in needs_review)
        if self.company_analyst is not None and "company_analyst" in names:
            company_jobs = []
            for symbol in review_names:
                view = self.gateway.for_company_analyst(
                    agent_id="company_analyst",
                    symbol=symbol,
                    simulation_time=now,
                )
                if not (
                    view.records
                    or view.promoted_insights
                    or view.recent_events
                    or view.market_features
                ):
                    continue
                print(f"company {symbol} {now.isoformat()}", flush=True)
                company_jobs.append((symbol, view))
            company_futs = {
                pool.submit(self.company_analyst.analyze, view): symbol
                for symbol, view in company_jobs
            }
            company_hits, company_errors = _await_mapped(company_futs)
            for symbol, _view in company_jobs:
                if symbol in company_errors:
                    print(f"company skipped: {company_errors[symbol]}", flush=True)
                    continue
                analysis = company_hits[symbol]
                invocations.append(self.company_analyst.spec.key)
                company_analyses.append(analysis)
                self.store.append_company_analysis(
                    CompanyAnalysisRecord(
                        ref=f"analysis:{symbol}:{now.isoformat()}",
                        analysis=analysis,
                        knowledge_time=now,
                    )
                )
                print(
                    f"company {symbol} conf={analysis.confidence:.2f} "
                    f"{analysis.company_thesis[:160]}",
                    flush=True,
                )
        current_risk: dict[str, RiskAnalysis] = {}
        if self.risk_analyst is not None and "risk_llm:v1" in versions:
            risk_jobs = []
            for symbol in review_names:
                view = self.gateway.for_risk_analyst(
                    agent_id="risk_analyst",
                    symbol=symbol,
                    simulation_time=now,
                    outcome=outcome,
                )
                if not view.market_features or not (
                    view.company_analyses or view.records or view.promoted_insights
                ):
                    continue
                print(f"thesis-risk {symbol} {now.isoformat()}", flush=True)
                risk_jobs.append((symbol, view))
            risk_futs = {
                pool.submit(self.risk_analyst.analyze, view): symbol
                for symbol, view in risk_jobs
            }
            risk_hits, risk_errors = _await_mapped(risk_futs)
            for symbol, _view in risk_jobs:
                if symbol in risk_errors:
                    print(f"thesis-risk skipped: {risk_errors[symbol]}", flush=True)
                    continue
                advisory = risk_hits[symbol]
                current_risk[symbol] = advisory
                invocations.append(self.risk_analyst.spec.key)
                risk_analyses.append(advisory)
                print(
                    f"thesis-risk {symbol} score={advisory.overall_risk_score:.0f} "
                    f"success={advisory.success_probability_pct:.0f}% "
                    f"fail={advisory.failure_probability_pct:.0f}%",
                    flush=True,
                )
        if constructor is not None:
            self._construct_tracks(
                now=now,
                end=end,
                symbols=symbols,
                names=names,
                constructor=constructor,
                execution=execution,
                lifecycle=lifecycle,
                book=book,
                pending=pending,
                pending_symbols=pending_symbols,
                active_candidates=active_candidates,
                candidates=candidates,
                invocations=invocations,
                last_insight_refs=last_insight_refs,
                needs_review=needs_review,
                position_risk=position_risk,
                portfolio_calc=portfolio_calc,
                run_id=run_id,
                tick=tick,
                max_pct=max_pct,
                peak_equity_box=peak_equity_box,
                current_risk=current_risk,
                pool=pool,
            )
        marks = self.tape.prices_as_of(now)
        for symbol in needs_review:
            price = marks.get(symbol)
            if price is not None:
                last_review_price[symbol] = price
        if marks or book.snapshot().positions:
            book.mark(marks, knowledge_time=now)
        snap = book.snapshot()
        if snapshots:
            prior = snapshots[-1]
            if prior.knowledge_time.date() != snap.knowledge_time.date():
                print(
                    f"walk {prior.knowledge_time.date().isoformat()} "
                    f"tick={tick}/{self._tick_count}",
                    flush=True,
                )
                _log_book("day", prior, self._start_equity)
                if (
                    prior.knowledge_time.year != snap.knowledge_time.year
                    or prior.knowledge_time.month != snap.knowledge_time.month
                ):
                    _log_book("month", prior, self._start_equity)
        snapshots.append(snap)

    def _construct_tracks(
        self,
        *,
        now: datetime,
        end: datetime,
        symbols: tuple[str, ...],
        names: set[str],
        constructor,
        execution,
        lifecycle,
        book,
        pending: list[_PendingFill],
        pending_symbols: set[str],
        active_candidates: dict[str, TradeCandidateId],
        candidates: list[TradeCandidate],
        invocations: list[str],
        last_insight_refs: dict[str, frozenset[str] | None],
        needs_review: set[str],
        position_risk,
        portfolio_calc,
        run_id: RunId,
        tick: int,
        max_pct,
        peak_equity_box: list[float],
        current_risk: Mapping[str, RiskAnalysis],
        pool,
    ) -> None:
        held = {item.instrument: item.quantity for item in book.snapshot().positions}
        readies: list[_ReadyTrade] = []
        for symbol in symbols:
            if symbol in pending_symbols:
                continue
            view = self.gateway.for_trade_constructor(
                agent_id="trade_constructor",
                symbol=symbol,
                simulation_time=now,
            )
            current_qty = held.get(symbol, 0.0)
            visible_refs = frozenset(item.ref for item in view.promoted_insights)
            previous_refs = last_insight_refs[symbol]
            if not view.promoted_insights:
                last_insight_refs[symbol] = frozenset()
                self._queue_flatten(
                    execution,
                    pending,
                    pending_symbols,
                    active_candidates,
                    symbol,
                    current_qty,
                    now,
                    end,
                )
                continue
            watching_news = "news_analyst" in names
            if watching_news and symbol not in needs_review:
                continue
            if (
                not watching_news
                and previous_refs is not None
                and visible_refs == previous_refs
            ):
                continue
            last_insight_refs[symbol] = visible_refs
            proposed = constructor.propose(view)
            invocations.append(constructor.spec.key)
            lifecycle.register(
                proposed,
                event_id=self._trade_event_id(run_id, proposed.id, "created"),
                occurred_at=CreatedAt(now),
                agent_id=AgentId("trade_constructor"),
                run_id=run_id,
            )
            stopped = self._stopped_sides.get(symbol)
            if stopped is not None and proposed.direction is not TradeSide.NO_TRADE:
                stopped_side, stopped_at = stopped
                causal_refs = {ref.value for ref in proposed.thesis_refs}
                causal_times = [
                    item.knowledge_time
                    for item in view.promoted_insights
                    if item.ref in causal_refs
                ]
                newest = max(causal_times) if causal_times else None
                if not may_reenter_stopped_side(
                    stopped_at=stopped_at,
                    now=now,
                    newest_causal_evidence=newest,
                    cooldown=self._stop_reentry_cooldown,
                ):
                    self._reject_candidate(
                        lifecycle,
                        proposed,
                        now,
                        run_id,
                        reason=(
                            f"stopped {stopped_side.value} waits for "
                            "re-entry cooldown"
                        ),
                        candidates=candidates,
                    )
                    continue
                self._stopped_sides.pop(symbol, None)
            if proposed.direction is TradeSide.NO_TRADE:
                self._reject_candidate(
                    lifecycle,
                    proposed,
                    now,
                    run_id,
                    reason="constructor proposed no_trade",
                    candidates=candidates,
                )
                self._queue_flatten(
                    execution,
                    pending,
                    pending_symbols,
                    active_candidates,
                    symbol,
                    current_qty,
                    now,
                    end,
                )
                continue
            articles = tuple(
                article
                for article in self.store._news_records()
                if symbol in article.symbols
                and article.knowledge_time is not None
                and article.knowledge_time <= now
            )
            readies.append(
                _ReadyTrade(symbol, view, proposed, current_qty, articles)
            )
        snap = None
        if readies and self.portfolio_risk is not None and "portfolio_risk" in names:
            book.mark(self.tape.prices_as_of(now), knowledge_time=now)
            snap = book.snapshot()
            peak_equity_box[0] = max(peak_equity_box[0], snap.equity)
        prepared: dict[str, object] = {}
        for ready in readies:
            if snap is not None:
                prepared[ready.symbol] = self._prepare_portfolio(
                    ready.proposed,
                    snapshot=snap,
                    insights=ready.view.promoted_insights,
                    now=now,
                    peak_equity=peak_equity_box[0],
                    engine=position_risk,
                    calculator=portfolio_calc,
                    run_id=run_id,
                    tick=tick,
                    advisory=current_risk.get(ready.symbol),
                )
            else:
                prepared[ready.symbol] = ready.proposed
        sized_futs = {
            pool.submit(
                self._size_and_time,
                ready,
                prepared[ready.symbol],
                now=now,
                max_pct=max_pct,
            ): ready.symbol
            for ready in readies
        }
        sized_hits, sized_errors = _await_mapped(sized_futs)
        sized_hits = _share_simultaneous_book(readies, sized_hits)
        for ready in readies:
            if ready.symbol in sized_errors:
                print(f"track skipped: {sized_errors[ready.symbol]}", flush=True)
                continue
            sized = sized_hits[ready.symbol]
            invocations.extend(sized.invocations)
            if sized.defer:
                continue
            if sized.flatten:
                self._reject_candidate(
                    lifecycle,
                    sized.proposed,
                    now,
                    run_id,
                    reason="portfolio risk rejected",
                    candidates=candidates,
                )
                self._queue_flatten(
                    execution,
                    pending,
                    pending_symbols,
                    active_candidates,
                    ready.symbol,
                    ready.current_qty,
                    now,
                    end,
                )
                continue
            timed = sized.timed
            if self.trade_risk is not None:
                print(
                    f"risk {ready.symbol} {timed.action.value} {timed.reasons[0][:160]}",
                    flush=True,
                )
            reviewed = timed.candidate
            lifecycle.transition(
                sized.proposed.id,
                TradeCandidateStatus.RISK_REVIEWED,
                event_id=self._trade_event_id(
                    run_id, sized.proposed.id, "risk_reviewed"
                ),
                occurred_at=CreatedAt(now),
                reason="trade risk review completed",
                proposed_size=reviewed.proposed_size,
                run_id=run_id,
            )
            reviewed = lifecycle.transition(
                sized.proposed.id,
                reviewed.status,
                event_id=self._trade_event_id(
                    run_id, sized.proposed.id, reviewed.status.value
                ),
                occurred_at=CreatedAt(now),
                reason="trade risk decision",
                run_id=run_id,
            )
            candidates.append(reviewed)
            if timed.action is TimingAction.DEFER:
                continue
            if reviewed.status is not TradeCandidateStatus.APPROVED:
                self._queue_flatten(
                    execution,
                    pending,
                    pending_symbols,
                    active_candidates,
                    ready.symbol,
                    ready.current_qty,
                    now,
                    end,
                )
                continue
            marked = book.snapshot()
            held_pnl = _open_pnl(marked, ready.symbol, self.tape, now)
            if hold_working_position(
                existing_quantity=ready.current_qty,
                proposed=reviewed.direction,
                action=timed.action,
                pnl=held_pnl,
            ):
                mark_px = self.tape.prices_as_of(now).get(ready.symbol)
                cap = float(max_pct)
                over_cap = (
                    mark_px is not None
                    and marked.equity > 0
                    and abs(ready.current_qty * mark_px) / marked.equity > cap
                )
                if not over_cap:
                    print(
                        f"hold {ready.symbol} {timed.action.value} "
                        f"working pnl={held_pnl:.2%}",
                        flush=True,
                    )
                    continue
            fill = execution.submit(
                reviewed,
                tape=self.tape,
                equity=marked.equity,
                cash=marked.cash,
                existing_quantity=ready.current_qty,
            )
            if fill is None or fill.knowledge_time > end:
                continue
            lifecycle.transition(
                sized.proposed.id,
                TradeCandidateStatus.SUBMITTED,
                event_id=self._trade_event_id(run_id, sized.proposed.id, "submitted"),
                occurred_at=CreatedAt(now),
                reason="submitted to simulated execution",
                run_id=run_id,
            )
            pending.append(
                _PendingFill(
                    fill,
                    candidate_id=sized.proposed.id,
                    closes_candidate_id=active_candidates.get(ready.symbol),
                )
            )
            pending_symbols.add(fill.instrument)

    def _size_and_time(
        self,
        ready: _ReadyTrade,
        prepared: object,
        *,
        now: datetime,
        max_pct,
    ) -> _SizedTrade:
        proposed = ready.proposed
        calls: list[str] = []
        if isinstance(prepared, TradeCandidate):
            proposed = prepared
            if proposed.status is TradeCandidateStatus.REJECTED:
                return _SizedTrade(proposed, None, (), True, False)
        else:
            advisory = getattr(prepared, "risk_analysis", None)
            if _defer_weak_thesis(advisory, ready.current_qty):
                print(
                    f"portfolio {ready.symbol} defer fail>"
                    f"{advisory.failure_probability_pct:.0f}%>"
                    f"success={advisory.success_probability_pct:.0f}%",
                    flush=True,
                )
                return _SizedTrade(proposed, None, tuple(calls), True, False)
            try:
                assessment = self.portfolio_risk.analyze(prepared)
            except (ValueError, RuntimeError) as exc:
                print(f"portfolio skipped: {exc}", flush=True)
                return _SizedTrade(
                    replace(proposed, status=TradeCandidateStatus.REJECTED),
                    None,
                    tuple(calls),
                    True,
                    False,
                )
            else:
                calls.append(self.portfolio_risk.spec.key)
                print(
                    f"portfolio {ready.symbol} {assessment.final_recommendation.value} "
                    f"size={assessment.final_size} {assessment.rationale[:160]}",
                    flush=True,
                )
                proposed = self._apply_portfolio_assessment(ready.proposed, assessment)
                if proposed is None:
                    return _SizedTrade(ready.proposed, None, tuple(calls), False, True)
                if proposed.status is TradeCandidateStatus.REJECTED:
                    return _SizedTrade(proposed, None, tuple(calls), True, False)
        timed = _risk_review(
            proposed,
            float(max_pct),
            tape=self.tape,
            insights=ready.view.promoted_insights,
            articles=ready.articles,
            now=now,
            existing_quantity=ready.current_qty,
            trade_risk=self.trade_risk,
            already_reviewed_today=False,
        )
        if self.trade_risk is not None:
            calls.append(self.trade_risk.spec.key)
        return _SizedTrade(proposed, timed, tuple(calls), False, False)

    def _flatten_losers(
        self,
        *,
        now: datetime,
        end: datetime,
        execution,
        book,
        pending: list[_PendingFill],
        pending_symbols: set[str],
        active_candidates: dict[str, TradeCandidateId],
    ) -> None:
        held = book.snapshot().positions
        if not held:
            return
        marks = self.tape.prices_as_of(now)
        needed = {
            item.instrument: marks[item.instrument]
            for item in held
            if item.instrument in marks
        }
        if len(needed) != len(held):
            return
        book.mark(needed, knowledge_time=now)
        positions = book.snapshot().positions
        held_symbols = {item.instrument for item in positions}
        for symbol in tuple(self._favorable_marks):
            if symbol not in held_symbols:
                self._favorable_marks.pop(symbol, None)
        for position in positions:
            if position.instrument in pending_symbols:
                continue
            side = TradeSide.LONG if position.quantity > 0 else TradeSide.SHORT
            prior_trail = self._favorable_marks.get(position.instrument)
            if prior_trail is None or prior_trail[0] is not side:
                favorable = position.market_price
            elif side is TradeSide.LONG:
                favorable = max(prior_trail[1], position.market_price)
            else:
                favorable = min(prior_trail[1], position.market_price)
            self._favorable_marks[position.instrument] = (side, favorable)
            volatility = annualized_volatility(
                self.tape, position.instrument, now
            )
            stop_loss_pct = scaled_stop_pct(
                volatility,
                floor_pct=self._stop_loss_pct,
                multiple=self._stop_volatility_multiple,
                vol_ceiling=self._stop_vol_ceiling,
            )
            hard_stop = should_stop_loser(
                quantity=position.quantity,
                average_entry=position.average_entry,
                market_price=position.market_price,
                stop_loss_pct=stop_loss_pct,
            )
            trailing_stop_pct = scaled_stop_pct(
                volatility,
                floor_pct=self._trailing_stop_floor_pct,
                multiple=self._trailing_stop_volatility_multiple,
                vol_ceiling=self._stop_vol_ceiling,
            )
            trailing_stop = should_trail_winner(
                quantity=position.quantity,
                average_entry=position.average_entry,
                market_price=position.market_price,
                favorable_price=favorable,
                activation_pct=self._trailing_stop_activation_pct,
                trailing_stop_pct=trailing_stop_pct,
            )
            if not hard_stop and not trailing_stop:
                continue
            mark = position.market_price
            if mark is None:
                continue
            self._stopped_sides[position.instrument] = (side, now)
            loss = position_return(
                position.quantity, position.average_entry, mark
            )
            print(
                f"{'trail' if trailing_stop else 'stop'} "
                f"{position.instrument} {side.value} pnl={loss:.2%} "
                f"entry={position.average_entry} mark={position.market_price} "
                f"best={favorable}",
                flush=True,
            )
            self._favorable_marks.pop(position.instrument, None)
            self._queue_flatten(
                execution,
                pending,
                pending_symbols,
                active_candidates,
                position.instrument,
                position.quantity,
                now,
                end,
            )

    def _queue_flatten(
        self,
        execution,
        pending: list[_PendingFill],
        pending_symbols: set[str],
        active_candidates: dict[str, TradeCandidateId],
        symbol: str,
        current_qty: float,
        now: datetime,
        end: datetime,
    ) -> None:
        fill = execution.flatten(symbol, current_qty, tape=self.tape, after=now)
        if fill is not None and fill.knowledge_time <= end:
            pending.append(
                _PendingFill(
                    fill,
                    closes_candidate_id=active_candidates.get(symbol),
                )
            )
            pending_symbols.add(symbol)

    @staticmethod
    def _reject_candidate(
        lifecycle,
        proposed: TradeCandidate,
        now: datetime,
        run_id: RunId,
        *,
        reason: str,
        candidates: list[TradeCandidate],
    ) -> None:
        lifecycle.transition(
            proposed.id,
            TradeCandidateStatus.RISK_REVIEWED,
            event_id=BacktestRunner._trade_event_id(
                run_id, proposed.id, "risk_reviewed"
            ),
            occurred_at=CreatedAt(now),
            reason=reason,
            run_id=run_id,
        )
        reviewed = lifecycle.transition(
            proposed.id,
            TradeCandidateStatus.REJECTED,
            event_id=BacktestRunner._trade_event_id(run_id, proposed.id, "rejected"),
            occurred_at=CreatedAt(now),
            reason=reason,
            run_id=run_id,
        )
        candidates.append(reviewed)

    def _prepare_portfolio(
        self,
        candidate: TradeCandidate,
        *,
        snapshot: PortfolioSnapshot,
        insights: Sequence[PromotedInsight],
        now: datetime,
        peak_equity: float,
        engine: DeterministicPositionRiskEngine,
        calculator: PortfolioRiskCalculator,
        run_id: RunId,
        tick: int,
        advisory: RiskAnalysis | None = None,
    ) -> PortfolioRiskAgentContext | TradeCandidate:
        selected = tuple(
            item
            for item in insights
            if item.ref in {ref.value for ref in candidate.thesis_refs}
        )
        if not selected:
            print("portfolio skipped: no matching thesis insights", flush=True)
            return candidate
        sector = _visible_sector(self.store, candidate.instrument, now)
        vol = annualized_volatility(self.tape, candidate.instrument, now)
        adv = dollar_adv(self.tape, candidate.instrument, now)
        if vol is None or adv is None:
            print(
                f"portfolio {candidate.instrument} skipped: "
                f"missing sourced vol/ADV",
                flush=True,
            )
            return replace(candidate, status=TradeCandidateStatus.REJECTED)
        held = {
            item.instrument: _visible_sector(self.store, item.instrument, now)
            for item in snapshot.positions
        }
        held[candidate.instrument] = sector
        drawdown = 0.0 if peak_equity <= 0 else max(0.0, 1.0 - snapshot.equity / peak_equity)
        sectors = {key: value for key, value in held.items() if value is not None}
        vols = {
            item.instrument: annualized_volatility(self.tape, item.instrument, now) or vol
            for item in snapshot.positions
        }
        vols[candidate.instrument] = vol
        risk_input = PositionRiskInput(
            candidate=candidate,
            portfolio=snapshot,
            sectors=sectors,
            average_daily_dollar_volume={candidate.instrument: adv},
            annualized_volatility=vols,
            current_drawdown=drawdown,
        )
        deterministic = engine.evaluate(
            risk_input,
            audit_event_id=AuditEventId(f"{run_id.value}:pr:{tick}:{candidate.instrument}"),
            run_id=run_id,
        )
        try:
            comparison = calculator.compare(
                PortfolioRiskInput(
                    portfolio=snapshot,
                    sectors=sectors,
                    factor_loadings={
                        symbol: {"market": 1.0} for symbol in sectors
                    },
                    annualized_volatility=vols,
                    correlations=rolling_correlations(
                        self.tape, tuple(sectors), now
                    ),
                    current_drawdown=drawdown,
                ),
                candidate,
                approved_size=deterministic.approved_size
                if deterministic.result is not RiskCheckResult.REJECT
                else 0.0,
            )
        except PortfolioRiskError as exc:
            print(f"portfolio skipped: {exc}", flush=True)
            return replace(candidate, status=TradeCandidateStatus.REJECTED)
        return PortfolioRiskAgentContext(
            candidate,
            snapshot,
            deterministic,
            comparison,
            selected,
            advisory,
        )

    @staticmethod
    def _apply_portfolio_assessment(
        candidate: TradeCandidate,
        assessment,
    ) -> TradeCandidate | None:
        if assessment.final_recommendation is PortfolioRiskRecommendation.DEFER:
            return None
        if (
            assessment.final_recommendation is PortfolioRiskRecommendation.REJECT
            or assessment.final_size <= 0
        ):
            return replace(
                candidate,
                direction=TradeSide.NO_TRADE,
                status=TradeCandidateStatus.REJECTED,
                proposed_size=0,
            )
        return replace(
            candidate,
            proposed_size=assessment.final_size,
            status=TradeCandidateStatus.APPROVED,
        )

    @staticmethod
    def _trade_event_id(
        run_id: RunId, candidate_id: TradeCandidateId, transition: str
    ) -> AuditEventId:
        return AuditEventId(
            f"{run_id.value}:trade:{candidate_id.value}:{transition}"
        )

    def _promote(
        self,
        finding: Finding,
        *,
        symbol: str,
        when: datetime,
        horizon_days: int,
        working: WorkingMemory,
        gate: GovernanceGate,
        run_id: RunId,
        tick: int,
    ) -> GovernanceDecision:
        sim = SimulationTime(when)
        insight_id = InsightId(f"insight:{symbol}:{when.isoformat()}")
        revision = InsightRevision(
            insight_id=insight_id,
            value=finding,
            valid_until=SimulationTime(when + timedelta(days=horizon_days)),
        )
        entry = working.for_agent(AgentId("news_analyst")).write(
            audit_event_id=AuditEventId(f"{run_id.value}:wm:{tick}:{symbol}"),
            run_id=run_id,
            entity_id=EntityId(symbol),
            category=WorkingMemoryCategory.CANDIDATE_INSIGHT,
            value=revision,
            created_at=CreatedAt(when),
        )
        proposal = PromotionProposal.from_working_memory(
            entry,
            id=ProposalId(f"promo:{symbol}:{when.isoformat()}"),
            agent_id=AgentId("news_analyst"),
            target_resource="insights",
            target_field="claim",
            evidence_refs=tuple(EvidenceId(ref) for ref in finding.evidence_refs),
            confidence=finding.confidence,
            reasoning_summary=finding.claim,
            created_at=CreatedAt(when),
        )
        return gate.evaluate(proposal, simulation_time=sim)

    def _ticks(
        self, start: datetime, end: datetime, step: object
    ) -> tuple[datetime, ...]:
        if isinstance(step, timedelta):
            times = [start]
            now = start
            while now < end:
                now = now + step
                times.append(end if now > end else now)
            return tuple(times)
        stamps = {start, end}
        for item in self.tape.prints:
            if start <= item.knowledge_time <= end:
                stamps.add(item.knowledge_time)
        for article in self.store._news_records():
            known = article.knowledge_time
            if known is not None and start <= known <= end:
                stamps.add(known)
        return tuple(sorted(stamps))

    def _spec(self, versions: Sequence[str], name: str) -> AgentSpec | None:
        selected = [key for key in versions if key.startswith(f"{name}:")]
        if not selected:
            return None
        if len(selected) > 1:
            raise ValueError(f"only one {name} version can run")
        return self.registry.get(selected[0])


def _share_simultaneous_book(readies: Sequence[_ReadyTrade], sized_hits: dict) -> dict:
    names = []
    for ready in readies:
        sized = sized_hits.get(ready.symbol)
        if sized is None or sized.flatten or sized.defer or sized.timed is None:
            continue
        if sized.timed.action is not TimingAction.ALLOW:
            continue
        if sized.timed.candidate.proposed_size <= 0:
            continue
        names.append(ready.symbol)
    if len(names) <= 1:
        return sized_hits
    share = 1.0 / len(names)
    updated = dict(sized_hits)
    for symbol in names:
        sized = updated[symbol]
        current = sized.timed.candidate.proposed_size
        if current <= share:
            continue
        updated[symbol] = replace(
            sized,
            timed=replace(
                sized.timed,
                candidate=replace(sized.timed.candidate, proposed_size=share),
                reasons=(*sized.timed.reasons, "simultaneous_book_share"),
            ),
        )
    return updated


def _position_limits(
    max_position_pct: float, max_annualized_volatility: float = 0.60
) -> PositionRiskLimits:
    size = max(max_position_pct, 0.01)
    return PositionRiskLimits(
        max_position_pct=size,
        max_gross_exposure=max(1.2, size),
        max_net_exposure=max(1.0, size),
        max_sector_concentration=max(1.0, size),
        max_single_name_concentration=size,
        max_annualized_volatility=max(max_annualized_volatility, 0.01),
    )


def _visible_sector(store: object, symbol: str, now: datetime) -> str | None:
    companies = [
        item
        for item in store._company_entity_records()
        if item.ticker == symbol and item.knowledge_time <= now
    ]
    if companies:
        companies.sort(key=lambda item: item.knowledge_time, reverse=True)
        return companies[0].sector
    profile = listed_profile(symbol)
    return None if profile is None else profile[1]


def run_backtest(
    *,
    start: datetime,
    end: datetime,
    universe: Sequence[str],
    agent_versions: Sequence[str] | None = None,
    strategy_config: Mapping[str, object] | None = None,
    store: object | None = None,
    tape: MarketTape | None = None,
    registry: AgentRegistry | None = None,
    news_analyst: NewsAnalyst | None = None,
    company_analyst: CompanyAnalyst | None = None,
    risk_analyst: RiskAnalyst | None = None,
    portfolio_risk: PortfolioRiskAgent | None = None,
    trade_risk: TradeRiskAnalyst | None = None,
    fetch: object | None = None,
) -> BacktestResult:
    config = dict(strategy_config or {})
    versions = tuple(agent_versions or ("news_analyst:v1", "trade_constructor:v1"))
    interval = str(config.get("interval", "15m"))
    warmup_days = config.get("warmup_days", 0)
    if (
        isinstance(warmup_days, bool)
        or not isinstance(warmup_days, int)
        or warmup_days < 0
    ):
        raise ValueError("warmup_days must be a non-negative integer")
    if store is None or tape is None:
        if store is not None or tape is not None:
            raise ValueError("store and tape must be provided together")
        requested_workers = config.get("max_workers", 8)
        if (
            isinstance(requested_workers, bool)
            or not isinstance(requested_workers, int)
            or requested_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        loaded_store, loaded_tape = load_historical_session(
            universe,
            start=start,
            end=end,
            interval=interval,
            warmup_days=warmup_days,
            fetch=fetch,
            max_workers=requested_workers,
        )
        store = loaded_store
        tape = loaded_tape
    return BacktestRunner(
        store=store,
        tape=tape,
        registry=registry,
        news_analyst=news_analyst,
        company_analyst=company_analyst,
        risk_analyst=risk_analyst,
        portfolio_risk=portfolio_risk,
        trade_risk=trade_risk,
    ).run(start, end, universe, versions, config)
