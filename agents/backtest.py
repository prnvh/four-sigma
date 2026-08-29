from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

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
from memory.timing_risk import TimingAction, pass_through_timing, tape_reaction
from memory.promotion import PromotionProposal
from memory.shared_mem import SharedMemory
from memory.sim_clock import SimulationClock
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
    load_historical_session,
    publish_tape_features,
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


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


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


class _ArticleEvidence:
    def __init__(self, article: object) -> None:
        known = getattr(article, "knowledge_time")
        self.created_at = CreatedAt(known)
        self.source_class = "reputable_newswire"

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
        max_pct = strategy_config.get("max_position_pct", 1.0)
        horizon_days = strategy_config.get("insight_horizon_days", 3)
        min_evidence = strategy_config.get("min_evidence_count", 2)
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
        print(f"walking {len(ticks)} ticks as if live...", flush=True)
        clock = SimulationClock(SimulationTime(ticks[0]))
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
        last_news_day: dict[str, object] = {symbol: None for symbol in symbols}
        last_insight_refs: dict[str, frozenset[str] | None] = {
            symbol: None for symbol in symbols
        }
        last_review_price: dict[str, float] = {}
        peak_equity = float(cash)
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
        run_id = RunId(f"backtest:{start.isoformat()}:{end.isoformat()}")
        context_offset = len(self.gateway.snapshots.snapshot())
        for tick, now in enumerate(ticks):
            clock.advance_to(SimulationTime(now))
            still_pending: list[_PendingFill] = []
            open_symbols = {item.instrument for item in book.snapshot().positions}
            for pending_fill in pending:
                fill = pending_fill.fill
                if fill.knowledge_time <= now:
                    try:
                        book.apply(fill)
                    except PortfolioError as exc:
                        print(f"fill dropped: {exc} {fill.instrument} {fill.side.value} qty={fill.quantity}", flush=True)
                        continue
                    fills.append(fill)
                    open_symbols.add(fill.instrument)
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
            pending = still_pending
            pending_symbols = {item.fill.instrument for item in pending}
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
                    if news_cadence == "day" and last_news_day[symbol] == now.date():
                        continue
                    news_jobs.append((symbol, view, len(fresh)))
            workers = max(len(news_jobs) + len(symbols), 1)
            news_errors: dict[str, BaseException] = {}
            news_findings: dict[str, Finding] = {}
            tape_hits: dict[str, object] = {}
            with ThreadPoolExecutor(max_workers=min(8, workers)) as pool:
                news_futs = {}
                if self.news_analyst is not None:
                    for symbol, view, fresh_count in news_jobs:
                        print(
                            f"news {symbol} {now.isoformat()} fresh={fresh_count}",
                            flush=True,
                        )
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
                for future in as_completed(news_futs):
                    symbol = news_futs[future]
                    try:
                        news_findings[symbol] = future.result()
                    except (ValueError, RuntimeError) as exc:
                        news_errors[symbol] = exc
                for future in as_completed(tape_futs):
                    symbol = tape_futs[future]
                    tape_hits[symbol] = future.result()
            for symbol, view, _fresh_count in news_jobs:
                if symbol in news_errors:
                    print(f"news skipped: {news_errors[symbol]}", flush=True)
                    last_news[symbol] = max(
                        article.knowledge_time
                        for article in view.articles
                        if article.knowledge_time is not None
                    )
                    last_news_day[symbol] = now.date()
                    continue
                finding = news_findings[symbol]
                invocations.append(self.news_analyst.spec.key)
                findings.append(finding)
                last_news[symbol] = max(
                    article.knowledge_time
                    for article in view.articles
                    if article.knowledge_time is not None
                )
                last_news_day[symbol] = now.date()
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
                if (
                    decision.outcome is GovernanceOutcome.APPROVED
                    and finding.direction is not Direction.NEUTRAL
                ):
                    needs_review.add(symbol)
            for symbol in symbols:
                if symbol in needs_review:
                    continue
                reaction = tape_hits[symbol]
                if not reaction.triggered:
                    continue
                needs_review.add(symbol)
                print(
                    f"tape {symbol} reaction move={reaction.move:.2%} "
                    f"typical={reaction.typical_daily_move:.2%}",
                    flush=True,
                )
            if self.company_analyst is not None and "company_analyst" in names:
                for symbol in symbols:
                    if symbol not in needs_review:
                        continue
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
                    try:
                        analysis = self.company_analyst.analyze(view)
                    except (ValueError, RuntimeError) as exc:
                        print(f"company skipped: {exc}", flush=True)
                        continue
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
            if self.risk_analyst is not None and "risk_llm:v1" in versions:
                for symbol in symbols:
                    if symbol not in needs_review:
                        continue
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
                    try:
                        advisory = self.risk_analyst.analyze(view)
                    except (ValueError, RuntimeError) as exc:
                        print(f"thesis-risk skipped: {exc}", flush=True)
                        continue
                    invocations.append(self.risk_analyst.spec.key)
                    risk_analyses.append(advisory)
                    print(
                        f"thesis-risk {symbol} score={advisory.overall_risk_score:.0f} "
                        f"success={advisory.success_probability_pct:.0f}% "
                        f"fail={advisory.failure_probability_pct:.0f}%",
                        flush=True,
                    )
            if constructor is not None:
                held = {item.instrument: item.quantity for item in book.snapshot().positions}
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
                        fill = execution.flatten(
                            symbol, current_qty, tape=self.tape, after=now
                        )
                        if fill is not None and fill.knowledge_time <= end:
                            pending.append(
                                _PendingFill(
                                    fill,
                                    closes_candidate_id=active_candidates.get(symbol),
                                )
                            )
                            pending_symbols.add(symbol)
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
                    if proposed.direction is TradeSide.NO_TRADE:
                        lifecycle.transition(
                            proposed.id,
                            TradeCandidateStatus.RISK_REVIEWED,
                            event_id=self._trade_event_id(
                                run_id, proposed.id, "risk_reviewed"
                            ),
                            occurred_at=CreatedAt(now),
                            reason="constructor proposed no_trade",
                            run_id=run_id,
                        )
                        reviewed = lifecycle.transition(
                            proposed.id,
                            TradeCandidateStatus.REJECTED,
                            event_id=self._trade_event_id(
                                run_id, proposed.id, "rejected"
                            ),
                            occurred_at=CreatedAt(now),
                            reason="constructor proposed no_trade",
                            run_id=run_id,
                        )
                        candidates.append(reviewed)
                        fill = execution.flatten(
                            symbol, current_qty, tape=self.tape, after=now
                        )
                        if fill is not None and fill.knowledge_time <= end:
                            pending.append(
                                _PendingFill(
                                    fill,
                                    closes_candidate_id=active_candidates.get(symbol),
                                )
                            )
                            pending_symbols.add(symbol)
                        continue
                    if self.portfolio_risk is not None and "portfolio_risk" in names:
                        book.mark(self.tape.prices_as_of(now), knowledge_time=now)
                        snap = book.snapshot()
                        peak_equity = max(peak_equity, snap.equity)
                        sized = self._portfolio_review(
                            proposed,
                            snapshot=snap,
                            insights=view.promoted_insights,
                            now=now,
                            peak_equity=peak_equity,
                            engine=position_risk,
                            calculator=portfolio_calc,
                            run_id=run_id,
                            tick=tick,
                        )
                        if sized is None:
                            continue
                        proposed = sized
                        if proposed.status is TradeCandidateStatus.REJECTED:
                            lifecycle.transition(
                                proposed.id,
                                TradeCandidateStatus.RISK_REVIEWED,
                                event_id=self._trade_event_id(
                                    run_id, proposed.id, "risk_reviewed"
                                ),
                                occurred_at=CreatedAt(now),
                                reason="portfolio risk rejected",
                                run_id=run_id,
                            )
                            reviewed = lifecycle.transition(
                                proposed.id,
                                TradeCandidateStatus.REJECTED,
                                event_id=self._trade_event_id(
                                    run_id, proposed.id, "rejected"
                                ),
                                occurred_at=CreatedAt(now),
                                reason="portfolio risk rejected",
                                run_id=run_id,
                            )
                            candidates.append(reviewed)
                            fill = execution.flatten(
                                symbol, current_qty, tape=self.tape, after=now
                            )
                            if fill is not None and fill.knowledge_time <= end:
                                pending.append(
                                    _PendingFill(
                                        fill,
                                        closes_candidate_id=active_candidates.get(symbol),
                                    )
                                )
                                pending_symbols.add(symbol)
                            continue
                    articles = tuple(
                        article
                        for article in self.store._news_records()
                        if symbol in article.symbols
                        and article.knowledge_time is not None
                        and article.knowledge_time <= now
                    )
                    timed = _risk_review(
                        proposed,
                        float(max_pct),
                        tape=self.tape,
                        insights=view.promoted_insights,
                        articles=articles,
                        now=now,
                        existing_quantity=current_qty,
                        trade_risk=self.trade_risk,
                        already_reviewed_today=False,
                    )
                    if self.trade_risk is not None:
                        invocations.append(self.trade_risk.spec.key)
                        print(
                            f"risk {symbol} {timed.action.value} {timed.reasons[0][:160]}",
                            flush=True,
                        )
                    reviewed = timed.candidate
                    lifecycle.transition(
                        proposed.id,
                        TradeCandidateStatus.RISK_REVIEWED,
                        event_id=self._trade_event_id(run_id, proposed.id, "risk_reviewed"),
                        occurred_at=CreatedAt(now),
                        reason="trade risk review completed",
                        proposed_size=reviewed.proposed_size,
                        run_id=run_id,
                    )
                    reviewed = lifecycle.transition(
                        proposed.id,
                        reviewed.status,
                        event_id=self._trade_event_id(
                            run_id, proposed.id, reviewed.status.value
                        ),
                        occurred_at=CreatedAt(now),
                        reason="trade risk decision",
                        run_id=run_id,
                    )
                    candidates.append(reviewed)
                    if timed.action is TimingAction.DEFER:
                        continue
                    if (
                        timed.action is TimingAction.REDUCE
                        or reviewed.status is not TradeCandidateStatus.APPROVED
                    ):
                        fill = execution.flatten(
                            symbol, current_qty, tape=self.tape, after=now
                        )
                        if fill is not None and fill.knowledge_time <= end:
                            pending.append(
                                _PendingFill(
                                    fill,
                                    closes_candidate_id=active_candidates.get(symbol),
                                )
                            )
                            pending_symbols.add(symbol)
                        continue
                    snap = book.snapshot()
                    fill = execution.submit(
                        reviewed,
                        tape=self.tape,
                        equity=snap.equity,
                        cash=snap.cash,
                        existing_quantity=current_qty,
                    )
                    if fill is None or fill.knowledge_time > end:
                        continue
                    lifecycle.transition(
                        proposed.id,
                        TradeCandidateStatus.SUBMITTED,
                        event_id=self._trade_event_id(run_id, proposed.id, "submitted"),
                        occurred_at=CreatedAt(now),
                        reason="submitted to simulated execution",
                        run_id=run_id,
                    )
                    pending.append(
                        _PendingFill(
                            fill,
                            candidate_id=proposed.id,
                            closes_candidate_id=active_candidates.get(symbol),
                        )
                    )
                    pending_symbols.add(fill.instrument)
            marks = self.tape.prices_as_of(now)
            for symbol in needs_review:
                price = marks.get(symbol)
                if price is not None:
                    last_review_price[symbol] = price
            if marks or book.snapshot().positions:
                book.mark(marks, knowledge_time=now)
            snapshots.append(book.snapshot())
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

    def _portfolio_review(
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
    ) -> TradeCandidate | None:
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
        if sector is None or vol is None or adv is None:
            print(
                f"portfolio {candidate.instrument} skipped: "
                f"missing sourced sector/vol/ADV",
                flush=True,
            )
            return candidate
        held = {
            item.instrument: _visible_sector(self.store, item.instrument, now)
            for item in snapshot.positions
        }
        held[candidate.instrument] = sector
        if any(value is None for value in held.values()):
            print("portfolio skipped: missing sector for an open name", flush=True)
            return candidate
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
                    correlations=_unit_correlations(tuple(sectors)),
                    current_drawdown=drawdown,
                ),
                candidate,
                approved_size=deterministic.approved_size
                if deterministic.result is not RiskCheckResult.REJECT
                else 0.0,
            )
        except PortfolioRiskError as exc:
            print(f"portfolio skipped: {exc}", flush=True)
            return candidate
        try:
            assessment = self.portfolio_risk.analyze(
                PortfolioRiskAgentContext(
                    candidate,
                    snapshot,
                    deterministic,
                    comparison,
                    selected,
                )
            )
        except (ValueError, RuntimeError) as exc:
            print(f"portfolio skipped: {exc}", flush=True)
            return candidate
        print(
            f"portfolio {candidate.instrument} {assessment.final_recommendation.value} "
            f"size={assessment.final_size} {assessment.rationale[:160]}",
            flush=True,
        )
        self._invocations.append(self.portfolio_risk.spec.key)
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


def _position_limits(
    max_position_pct: float, max_annualized_volatility: float = 0.60
) -> PositionRiskLimits:
    size = max(max_position_pct, 0.01)
    return PositionRiskLimits(
        max_position_pct=size,
        max_gross_exposure=max(1.5, size),
        max_net_exposure=max(0.5, size),
        max_sector_concentration=max(0.3, size),
        max_single_name_concentration=max(0.15, size),
        max_annualized_volatility=max(max_annualized_volatility, 0.01),
    )


def _visible_sector(store: object, symbol: str, now: datetime) -> str | None:
    companies = [
        item
        for item in store._company_entity_records()
        if item.ticker == symbol and item.knowledge_time <= now
    ]
    if not companies:
        return None
    companies.sort(key=lambda item: item.knowledge_time, reverse=True)
    return companies[0].sector


def _unit_correlations(symbols: tuple[str, ...]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for left in symbols:
        matrix[left] = {right: (1.0 if left == right else 0.0) for right in symbols}
    return matrix


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
    if store is None or tape is None:
        if store is not None or tape is not None:
            raise ValueError("store and tape must be provided together")
        loaded_store, loaded_tape = load_historical_session(
            universe,
            start=start,
            end=end,
            interval=str(config.get("interval", "15m")),
            fetch=fetch,
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
