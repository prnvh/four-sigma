

**Acceptance:** impossible to create historical data without `knowledge_time`.

---

### Commit 3 — Add immutable market/news event model

Define the raw input layer.

```text
Event
  id
  type
  entities[]
  event_time
  knowledge_time
  source
  payload
  metadata
```

Rules:

* events are append-only
* agents cannot edit them
* corrections produce new events

**Acceptance:** can ingest and retrieve events as-of a timestamp.

---

### Commit 4 — Build append-only audit ledger

Create the system-wide event log.

Initial event types:

```text
agent_run_started
agent_run_finished
context_requested
context_returned
working_memory_written
promotion_requested
promotion_approved
promotion_rejected
shared_memory_updated
risk_check_run
trade_candidate_created
trade_decision_created
```

**Acceptance:** every state-changing operation generates an audit event.

---

### Commit 5 — Add deterministic simulation clock

Build:

```text
SimulationClock
  now()
  advance_to()
  advance_by()
```

No agent should access wall-clock time directly.

**Acceptance:** running the same simulation twice yields identical timestamps/order.

---

## Phase 2 — Permissions and context isolation

### Commit 6 — Define agent capability model

Represent permissions declaratively.

Example:

```yaml
news_analyst:
  read:
    - events.news
    - insights.news
    - entities.basic
  propose:
    - insights.claim
    - insights.direction
    - insights.confidence
```

Support:

```text
READ
WORKING_WRITE
PROPOSE_SHARED_WRITE
VETO
EXECUTE
```

**Acceptance:** permission checks can answer whether an agent can perform an operation.

---

### Commit 7 — Implement field-level authorization

Do not rely on prompts.

Build a policy engine:

```text
authorize(
  agent,
  action,
  resource,
  field
)
```

Deny by default.

**Acceptance:** agent requesting unauthorized portfolio fields gets rejected before model invocation.

---

### Commit 8 — Build Context Gateway

Agents no longer query storage directly.

API roughly:

```text
get_context(
  agent_id,
  purpose,
  entity_ids,
  simulation_time
)
```

Gateway:

1. checks permissions
2. filters by simulation time
3. retrieves relevant data
4. returns only permitted fields

**Acceptance:** two agents requesting the same company receive different contexts based on permissions.

---

### Commit 9 — Add context snapshots and hashes

Persist exactly what every agent saw.

```text
ContextSnapshot
  id
  agent_id
  simulation_time
  fields
  source_refs
  content_hash
```

**Acceptance:** every agent output can be replayed against its exact original context.

---

## Phase 3 — Working and shared memory

### Commit 10 — Implement private agent working memory

Each agent gets isolated memory.

```text
WorkingMemoryEntry
  agent_id
  entity_id
  category
  value
  created_at
  expires_at
```

Initial categories:

```text
observation
hypothesis
question
candidate_insight
```

**Acceptance:** Agent A cannot read Agent B's private memory.

---

### Commit 11 — Define shared-memory schema

Start with:

```text
events
entities
insights
portfolio
risk
trade_candidates
decisions
```

Do not allow free-form arbitrary shared keys.

**Acceptance:** schema validation rejects unknown shared-memory writes.

---

### Commit 12 — Implement promotion proposal objects

Agents cannot mutate shared memory.

They create:

```text
PromotionProposal
  id
  agent_id
  target_resource
  target_field
  entity_id
  proposed_value
  evidence_refs[]
  confidence
  reasoning_summary
  created_at
```

**Acceptance:** working-memory insight can become a proposal without changing shared state.

---

### Commit 13 — Implement governance gate

Start deterministic.

Rules can check:

* agent has permission
* required evidence exists
* evidence predates simulation time
* confidence in range
* target schema valid
* no duplicate proposal

Possible outcomes:

```text
APPROVED
REJECTED
DEFERRED
```

**Acceptance:** only approved proposals reach shared memory.

---

### Commit 14 — Add shared-memory versioning

Do not overwrite important objects silently.

An insight should have:

```text
version
supersedes
status
created_by_proposal
valid_from
valid_until
```

**Acceptance:** historical shared state can be reconstructed at any simulation timestamp.

---

## Phase 4 — Agent runtime

### Commit 15 — Create generic AgentRunner

Standardize execution.

Pipeline:

```text
request
→ permission check
→ context build
→ model/tool call
→ output validation
→ working-memory write
→ audit
```

Record:

```text
agent_version
model
prompt_version
latency
token_usage
context_snapshot_id
```

**Acceptance:** dummy agent completes an audited run end-to-end.

---

### Commit 16 — Add structured agent output contracts

No agent should emit prose that downstream code has to interpret.

Example:

```text
NewsAnalysisResult
CompanyAnalysisResult
RiskAnalysisResult
TradeProposalResult
```

Use strict schema validation.

**Acceptance:** malformed model response fails safely and cannot reach memory.

---

### Commit 17 — Add agent registry and versions

Something like:

```text
news:v1
news_analyst:v1
company_analyst:v1
risk_llm:v1
portfolio_risk:v1
```

Keep prompt/config/version together.

**Acceptance:** backtest logs clearly identify which agent version generated each output.

---

## Phase 5 — News pipeline

### Commit 18 — Implement News Agent

Responsibilities:

* ingest/filter news
* identify relevant companies
* deduplicate
* classify event type
* score relevance

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

### Commit 19 — Implement News Analyst

Consumes:

* news observations
* permitted existing insights

Produces candidate insights:

```text
claim
direction
time_horizon
confidence
evidence_refs
```

Writes first to private memory.

**Acceptance:** candidate insight goes through working memory → promotion proposal.

---

### Commit 20 — Add news insight governance rules

Examples:

* minimum evidence count
* allowed source classes
* duplicate insight detection
* conflicting insight tagging
* expiry requirements

**Acceptance:** low-quality proposal can be rejected without touching shared memory.

---

## Phase 6 — Company research

### Commit 21 — Add company/entity knowledge model

Create company records with:

```text
ticker
exchange
sector
industry
identifiers
fundamental references
```

Keep raw data separate from inferred insights.

---

### Commit 22 — Implement Company Analyst

Reads:

* company facts
* approved insights
* recent events
* historical context

Produces:

```text
company_thesis
bull_case
bear_case
catalysts
risks
confidence
time_horizon
```

Initially write these as insights, not trades.

**Acceptance:** company analyst cannot size positions.

---

### Commit 23 — Add contradiction tracking

Allow insights to explicitly oppose earlier insights.

```text
supports[]
contradicts[]
supersedes[]
```

This will matter a lot later.

**Acceptance:** opposing analyst conclusions coexist without one silently deleting the other.

---

## Phase 7 — Trade construction

### Commit 24 — Add TradeCandidate model

Separate thesis from action.

```text
TradeCandidate
  instrument
  direction
  thesis_refs[]
  horizon
  confidence
  entry_conditions
  exit_conditions
  proposed_size
  status
```

No execution yet.

---

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

* long
* short
* no-trade

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

* hidden thesis assumptions
* event risk
* regime change
* correlated exposures
* invalidation scenarios
* second-order effects

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

* proposed trade
* portfolio state
* deterministic risk results
* selected insight summaries

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

### Commit 30 — Build portfolio accounting engine

Implement:

```text
cash
positions
average entry
realized pnl
unrealized pnl
fees
slippage
```

Make calculations deterministic.

---

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

* contexts
* agent invocation order
* memory events
* trade decisions
* portfolio results

Model calls may need recorded fixtures for strict replay.

---

### Commit 36 — Add lookahead-bias test suite

Explicit tests for:

* future news
* earnings publication dates
* revised fundamentals
* future prices
* future shared-memory entries

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

* research agents
* governance
* deterministic risk
* execution

Never allow retry logic to accidentally generate multiple trades.

---

### Commit 53 — Add adversarial agent tests

Examples:

* agent attempts unauthorized field access
* agent invents evidence ID
* agent proposes shared-memory field it does not own
* agent references future information
* agent emits malformed trade
* agent requests absurd position size

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

* historical replay
* no-lookahead guarantees
* attribution
* governance evaluation
* strategy evaluation

I wouldn't add many more agents before reaching this stage.

### Milestone D — 1–6 month speedruns

Commits **42–54**

Now optimize:

* parallelism
* caching
* checkpoints
* experiment batches
* observability
* reliability

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
