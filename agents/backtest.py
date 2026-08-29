from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from memory.audit_logger import AuditLedger
from memory.context_gateway import ContextGateway
from memory.execution import ExecutionConfig, MarketTape, SimulatedExecution
from memory.governance_gate import GovernanceDecision, GovernanceGate, ProposePermissions
from memory.news_governance import NewsInsightGovernanceRules
from memory.portfolio import Fill, PortfolioBook, PortfolioError, PortfolioSnapshot
from memory.promotion import PromotionProposal
from memory.shared_mem import SharedMemory
from memory.sim_clock import SimulationClock
from memory.types import (
    AgentId,
    AuditEventId,
    CreatedAt,
    EntityId,
    EvidenceId,
    Finding,
    InsightId,
    InsightRevision,
    PromotedInsight,
    ProposalId,
    RunId,
    SimulationTime,
    TradeCandidate,
    TradeCandidateStatus,
    TradeSide,
)
from memory.working_mem import WorkingMemory, WorkingMemoryCategory

from .history_feed import equity_symbol, load_historical_session
from .news_analyst import NewsAnalyst
from .registry import REGISTRY, AgentRegistry, AgentSpec
from .trade_constructor import TradeConstructor


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True)
class BacktestResult:
    snapshots: tuple[PortfolioSnapshot, ...]
    candidates: tuple[TradeCandidate, ...]
    fills: tuple[Fill, ...]
    invocations: tuple[str, ...]
    findings: tuple[Finding, ...]
    promotions: tuple[GovernanceDecision, ...]
    final: PortfolioSnapshot


def _risk_review(candidate: TradeCandidate, max_position_pct: float) -> TradeCandidate:
    if candidate.direction is TradeSide.NO_TRADE:
        return replace(candidate, status=TradeCandidateStatus.REJECTED)
    size = candidate.proposed_size
    if size > max_position_pct:
        size = max_position_pct
    if size <= 0:
        return replace(candidate, status=TradeCandidateStatus.REJECTED)
    return replace(
        candidate,
        proposed_size=size,
        status=TradeCandidateStatus.APPROVED,
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
            if name not in {"trade_constructor", "news_analyst"}:
                raise ValueError(
                    f"backtest runner has no historical binding for {key}"
                )
        if "news_analyst" in names and self.news_analyst is None:
            raise ValueError("news_analyst:v1 requires a NewsAnalyst binding")
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
        print(f"walking {len(ticks)} ticks as if live...", flush=True)
        clock = SimulationClock(SimulationTime(ticks[0]))
        book = PortfolioBook(float(cash), opened_at=start)
        pending: list[Fill] = []
        snapshots: list[PortfolioSnapshot] = []
        candidates: list[TradeCandidate] = []
        fills: list[Fill] = []
        invocations: list[str] = []
        findings: list[Finding] = []
        promotions: list[GovernanceDecision] = []
        last_news: dict[str, datetime] = {
            symbol: datetime.min.replace(tzinfo=timezone.utc) for symbol in symbols
        }
        last_news_day: dict[str, object] = {symbol: None for symbol in symbols}
        news_cadence = strategy_config.get("news_cadence", "tick")
        run_id = RunId(f"backtest:{start.isoformat()}:{end.isoformat()}")
        for tick, now in enumerate(ticks):
            clock.advance_to(SimulationTime(now))
            still_pending: list[Fill] = []
            open_symbols = {item.instrument for item in book.snapshot().positions}
            for fill in pending:
                if fill.knowledge_time <= now:
                    try:
                        book.apply(fill)
                    except PortfolioError as exc:
                        print(f"fill dropped: {exc} {fill.instrument} {fill.side.value} qty={fill.quantity}", flush=True)
                        continue
                    fills.append(fill)
                    open_symbols.add(fill.instrument)
                else:
                    still_pending.append(fill)
            pending = still_pending
            pending_symbols = {item.instrument for item in pending}
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
                    print(
                        f"news {symbol} {now.isoformat()} fresh={len(fresh)}",
                        flush=True,
                    )
                    try:
                        finding = self.news_analyst.analyze(view)
                    except (ValueError, RuntimeError) as exc:
                        print(f"news skipped: {exc}", flush=True)
                        last_news[symbol] = max(
                            article.knowledge_time
                            for article in view.articles
                            if article.knowledge_time is not None
                        )
                        last_news_day[symbol] = now.date()
                        continue
                    invocations.append(self.news_analyst.spec.key)
                    findings.append(finding)
                    last_news[symbol] = max(
                        article.knowledge_time
                        for article in view.articles
                        if article.knowledge_time is not None
                    )
                    last_news_day[symbol] = now.date()
                    promotions.append(
                        self._promote(
                            finding,
                            symbol=symbol,
                            when=now,
                            horizon_days=horizon_days,
                            working=working,
                            gate=gate,
                            run_id=run_id,
                            tick=tick,
                        )
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
                    if not view.promoted_insights:
                        fill = execution.flatten(
                            symbol, current_qty, tape=self.tape, after=now
                        )
                        if fill is not None and fill.knowledge_time <= end:
                            pending.append(fill)
                            pending_symbols.add(symbol)
                        continue
                    proposed = constructor.propose(view)
                    invocations.append(constructor.spec.key)
                    reviewed = _risk_review(proposed, float(max_pct))
                    candidates.append(reviewed)
                    if reviewed.status is not TradeCandidateStatus.APPROVED:
                        fill = execution.flatten(
                            symbol, current_qty, tape=self.tape, after=now
                        )
                        if fill is not None and fill.knowledge_time <= end:
                            pending.append(fill)
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
                    pending.append(fill)
                    pending_symbols.add(fill.instrument)
            marks = self.tape.prices_as_of(now)
            if marks or book.snapshot().positions:
                book.mark(marks, knowledge_time=now)
            snapshots.append(book.snapshot())
        return BacktestResult(
            snapshots=tuple(snapshots),
            candidates=tuple(candidates),
            fills=tuple(fills),
            invocations=tuple(invocations),
            findings=tuple(findings),
            promotions=tuple(promotions),
            final=book.snapshot(),
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
    ).run(start, end, universe, versions, config)
