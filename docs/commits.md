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
