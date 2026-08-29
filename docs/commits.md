## Phase 5 — News pipeline

### Commit 18 — Implement News Agent

Responsibilities:

- ingest/filter news
- identify relevant companies
- deduplicate
- classify event type
- score relevance

It should not create investment conclusions.

Output:

```text
NewsObservation
  event_id
  entities
  category
  relevance
```

**Acceptance:** historical news gets associated with relevant entities without future leakage.

---

### Commit 20 — Add news insight governance rules

Examples:

- minimum evidence count
- allowed source classes
- duplicate insight detection
- conflicting insight tagging
- expiry requirements

**Acceptance:** low-quality proposal can be rejected without touching shared memory.

---

## Phase 7 — Trade construction

### Commit 25 — Implement strategy/trade-construction agent

Its job:

```text
approved insights
        ↓
candidate trade
```

Not:

```text
news → instant BUY
```

Initially support:

- long
- short
- no-trade

Delay options until equities work.

**Acceptance:** every candidate must reference the insights that caused it.

---

## Phase 8 — Risk

### Commit 26 — Implement deterministic position-risk engine

Start simple.

Checks:

```text
max position %
gross exposure
net exposure
sector concentration
single-name concentration
daily liquidity
volatility limit
drawdown constraint
```

Return:

```text
PASS
REJECT
RESIZE
```

plus machine-readable reasons.

**Acceptance:** deterministic risk can veto any trade.

---

### Commit 27 — Implement LLM Risk Analyst

Different job from deterministic risk.

Look for:

- hidden thesis assumptions
- event risk
- regime change
- correlated exposures
- invalidation scenarios
- second-order effects

Output suggestions only.

**Acceptance:** LLM risk agent cannot bypass deterministic restrictions.

---

### Commit 28 — Implement portfolio risk snapshot

Calculate portfolio-wide metrics after every proposed change.

```text
gross
net
sector exposure
factor exposure
volatility
concentration
drawdown
correlation clusters
```

**Acceptance:** portfolio state can be evaluated before and after a proposed trade.

---

### Commit 29 — Implement Portfolio Risk Agent

Reads:

- proposed trade
- portfolio state
- deterministic risk results
- selected insight summaries

Can recommend:

```text
approve
reject
resize
defer
```

But deterministic hard limits remain final.

---

## Phase 9 — Portfolio and execution simulation


### Commit 31 — Add simulated execution

Turn approved decisions into fills.

Start with:

```text
market order at next eligible price
configurable slippage
transaction fee model
```

Avoid sophisticated execution until research quality is measurable.

---

### Commit 32 — Add trade lifecycle

Support:

```text
PROPOSED
RISK_REVIEWED
APPROVED
REJECTED
SUBMITTED
FILLED
CLOSED
```

Every transition goes into audit ledger.

---

## Phase 10 — Historical evaluation

### Commit 33 — Build historical data adapter

Unified API:

```text
events_as_of(t)
prices_as_of(t)
company_data_as_of(t)
portfolio_as_of(t)
```

Everything respects `knowledge_time`.

---

### Commit 34 — Build backtest runner

Basic interface:

```text
run_backtest(
  start,
  end,
  universe,
  agent_versions,
  strategy_config
)
```

Loop:

```text
advance clock
→ reveal new information
→ invoke eligible agents
→ governance
→ create trade candidates
→ risk
→ execution
→ mark portfolio
```

---

### Commit 35 — Add deterministic replay test

Take a short historical window.

Run it twice.

Assert equality of:

- contexts
- agent invocation order
- memory events
- trade decisions
- portfolio results

Model calls may need recorded fixtures for strict replay.

---

### Commit 36 — Add lookahead-bias test suite

Explicit tests for:

- future news
- earnings publication dates
- revised fundamentals
- future prices
- future shared-memory entries

This deserves its own commit because it is critical.

---

## Phase 11 — Evaluation

### Commit 37 — Add strategy metrics

Track:

```text
total return
CAGR
Sharpe
Sortino
max drawdown
hit rate
profit factor
turnover
transaction cost
exposure
```

---

### Commit 38 — Add agent-level evaluation

For each agent:

```text
runs
accepted proposals
rejected proposals
confidence calibration
direction accuracy
latency
tokens
cost
```

---

### Commit 39 — Add insight attribution

Connect:

```text
event
→ observation
→ insight
→ trade candidate
→ decision
→ fill
→ pnl
```

This lineage should be queryable.

---

### Commit 40 — Add PnL attribution

Calculate results by:

```text
agent
agent version
insight
insight category
news source
company
sector
confidence bucket
holding horizon
```

This is where you'll start learning which agents actually deserve to exist.

---

### Commit 41 — Add governance evaluation

Measure the gate itself.

```text
approval rate
rejection rate
approved-insight performance
rejected-insight counterfactual performance
memory pollution rate
duplicate rate
```

Particularly useful metric:

> Were rejected insights actually worse?

If not, your governance gate is hurting the system.

---

## Phase 12 — Speedrunning months of history

### Commit 42 — Add event-driven orchestration

Don't run every agent every minute.

Triggers:

```text
new relevant news
new filing
large price move
portfolio change
insight expiry
scheduled review
```

This will dramatically reduce inference cost.

---

### Commit 43 — Add agent concurrency

Parallelize safe work:

```text
        event
       /     \
news analyst company analyst
       \     /
     governance
```

Keep mutations serialized through governance/shared state.

---

### Commit 44 — Add model-response cache

Cache based on:

```text
agent_version
prompt_version
context_hash
model
parameters
```

This makes historical experimentation much cheaper.

---

### Commit 45 — Add checkpoint/resume

For 1–6 month simulations:

```text
simulation clock
portfolio state
shared memory
working memory
audit offset
pending events
```

Persist checkpoints.

---

### Commit 46 — Add batch experiment runner

Run variations such as:

```text
baseline
without news analyst
without company analyst
without LLM risk
different governance thresholds
different models
different prompts
```

This enables proper ablations.

---

## Phase 13 — Operational visibility

### Commit 47 — Add run explorer API

Queries such as:

```text
Why did we buy AAPL?
What context did the agent see?
Which proposal created this insight?
Why was this trade resized?
```

Backend first. UI later.

---

### Commit 48 — Add lineage graph

Return:

```text
News Event
   ↓
News Observation
   ↓
Insight Proposal
   ↓
Governance Approval
   ↓
Company Thesis
   ↓
Trade Candidate
   ↓
Risk Review
   ↓
Portfolio Decision
   ↓
Fill
   ↓
PnL
```

---

### Commit 49 — Add system metrics

Track:

```text
agent latency
event→decision latency
tokens/run
cost/run
cost/day
context size
cache hit rate
governance latency
simulation throughput
```

---

## Phase 14 — Hardening

### Commit 50 — Add fail-closed behavior

If any of these fail:

```text
schema validation
permissions
governance
risk engine
portfolio accounting
```

the system should produce **no trade**.

---

### Commit 51 — Add idempotency

Every state mutation gets an idempotency key.

Essential when orchestrators retry agent tasks.

---

### Commit 52 — Add agent timeout/retry policies

Different policies for:

- research agents
- governance
- deterministic risk
- execution

Never allow retry logic to accidentally generate multiple trades.

---

### Commit 53 — Add adversarial agent tests

Examples:

- agent attempts unauthorized field access
- agent invents evidence ID
- agent proposes shared-memory field it does not own
- agent references future information
- agent emits malformed trade
- agent requests absurd position size

---

### Commit 54 — Freeze v1 backtest protocol

Document the first official experimental protocol:

```text
universe
historical period
data vendors
fee model
slippage model
models
prompts
agent versions
governance rules
risk limits
benchmark
metrics
```

From here onward, architecture changes can be compared against a stable baseline.

---

# Milestones

I’d group those commits into four practical milestones.

### Milestone A — Governed memory system

Commits **1–17**

You have:

```text
agent
→ isolated context
→ working memory
→ promotion
→ governance
→ shared memory
→ audit
```

No trading yet.

### Milestone B — First research-to-trade path

Commits **18–32**

You now have:

```text
news
→ analysis
→ company thesis
→ trade candidate
→ risk
→ simulated portfolio
```

At this point you have an actual minimal prop-research system.

### Milestone C — Scientific backtesting

Commits **33–41**

You gain:

- historical replay
- no-lookahead guarantees
- attribution
- governance evaluation
- strategy evaluation

I wouldn't add many more agents before reaching this stage.

### Milestone D — 1–6 month speedruns

Commits **42–54**

Now optimize:

- parallelism
- caching
- checkpoints
- experiment batches
- observability
- reliability

---

## One sequencing rule I would enforce

Don't build:

```text
20 agents
→ then backtesting
```

Build:

```text
3–5 agents
→ backtesting
→ attribution
→ ablation
→ evidence that a new agent helps
→ add next agent
```

So the development loop becomes:

```text
Architecture
     ↓
Agent
     ↓
Historical experiment
     ↓
Measure marginal alpha / risk / cost
     ↓
Keep or delete
     ↓
Next iteration
```

For a prop system, **deleting agents that don't measurably improve the portfolio is just as important as adding new ones**.