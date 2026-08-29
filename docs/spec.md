1. Data layer

Treat raw external data as immutable inputs.

You’ll probably want separate stores for:

News / events — article, source, timestamp, affected entities, raw text/reference
Market data — prices, volume, fundamentals, options data, corporate actions
Portfolio state — positions, cash, exposures, PnL
Company state — canonical facts and derived company-specific knowledge

Agents should generally not overwrite raw data. They produce derived objects from it.

2. Working memory per agent

Each agent gets its own private workspace.

For example:

news_analyst/
  observations
  hypotheses
  candidate_insights
  unresolved_questions
  confidence

This is where agents can speculate freely.

Nothing here should automatically become globally trusted state.

That separation is especially important for a quant system because otherwise one hallucinated interpretation can contaminate every downstream agent.

3. Promotion / governance layer

Your promotion-based memory idea should become a first-class subsystem rather than a helper function.

Think:

Agent Working Memory
        ↓
Promotion Proposal
        ↓
Governance Gate
        ↓
Shared Memory

A promotion proposal should contain something like:

proposal_id
agent_id
timestamp
target_field
target_entity
proposed_value
evidence_refs
confidence
reasoning_summary
dependencies
expiry / review_date

Then the governance gate determines:

ACCEPT
REJECT
ACCEPT_WITH_MODIFICATION
DEFER

And crucially, the decision itself gets logged.

I would make this an append-only event ledger:

agent_observation_created
memory_promotion_requested
memory_promotion_accepted
memory_promotion_rejected
shared_memory_updated
trade_suggestion_created
risk_check_run
trade_executed

That gives you perfect replayability for backtests.

4. Shared memory

I would change your current fields slightly.

Instead of:

news
portfolio
insights
buy/sell options

I’d separate facts, interpretations, decisions, and state.

Something closer to:

shared/
  events/
  entities/
  insights/
  portfolio/
  risk/
  trade_candidates/
  decisions/

Where:

events

event_id
event_type
entities
occurred_at
reported_at
source
raw_reference

insights/{company}

insight_id
claim
direction
time_horizon
confidence
supporting_evidence
contradicting_evidence
created_by
created_at
valid_until
status

portfolio

positions
cash
exposures
factor_exposures
sector_exposures
liquidity
pnl

risk

portfolio_var
position_limits
drawdown
concentration
correlation
scenario_results
risk_budget

trade_candidates

instrument
direction
size_range
entry_conditions
exit_conditions
thesis_refs
expected_horizon
expected_return
confidence
risk_estimate
proposed_by

Notice I’d avoid making "buy/sell" itself shared truth.

A buy/sell signal should be a proposal, not memory.

That distinction becomes very useful once several agents disagree.

Agent architecture

Your current agents map fairly cleanly into a pipeline:

                   ┌──────────────┐
Feeds ────────────→│ News Agent   │
                   └──────┬───────┘
                          ↓
                   ┌──────────────┐
                   │ News Analyst │
                   └──────┬───────┘
                          ↓
                 candidate insight
                          ↓
                   Governance Gate
                          ↓
                    Shared Insight
                          ↓
              ┌───────────┴───────────┐
              ↓                       ↓
      Company Analyst             Risk Agent
              ↓                       ↓
       Trade Candidate          Risk Assessment
              └──────────┬────────────┘
                         ↓
               Portfolio Risk Agent
                         ↓
                 Portfolio Decision

But I would introduce one more logical separation:

Research agents

They answer:

What is happening and what might it mean?

Examples:

News agent
News analyst
Company analyst
Decision agents

They answer:

Given what we know, what should we potentially do?

Potential future agent:

Strategy / trade construction agent
Risk agents

They answer:

What can go wrong, and how much can we afford?

Your split between an LLM risk analyst and a deterministic risk engine is especially good.

I would make their responsibilities very different.

LLM risk analyst:

event-driven risk
regime risk
qualitative scenario generation
hidden assumptions
thesis invalidation
cross-company contagion

Deterministic risk engine:

position limits
VaR / CVaR
volatility
drawdown
beta
correlations
gross/net exposure
sector exposure
liquidity
stop constraints
risk budgets

The deterministic system should have veto authority.

The LLM should not.

Something like:

Trade Proposal
     ↓
LLM Risk Review
     ↓
Deterministic Risk Gate
     ↓
Portfolio Construction

That keeps hard limits outside probabilistic reasoning.

Permissions

I’d make permissions field-level and action-level.

Not just:

news analyst can access insights

but:

news_analyst:
  READ:
    events.news
    entities.basic
    insights.news_related

  WORKING_WRITE:
    observations
    hypotheses

  PROPOSE_SHARED_WRITE:
    insights.claim
    insights.direction
    insights.confidence

  DENIED:
    portfolio.positions
    trade_candidates.size

Company analyst could have:

READ:
  company.*
  insights.*
  events.*

PROPOSE:
  insights.*
  trade_candidates.thesis
  trade_candidates.direction

DENIED:
  portfolio sizing

Portfolio risk:

READ:
  portfolio.*
  trade_candidates.*
  risk.*
  insights.summary

PROPOSE:
  trade_candidates.size
  trade_candidates.rejection
  portfolio.target_exposure

That gives you proper least privilege.

I would enforce this outside the model in the orchestration layer rather than relying on prompt instructions.

One architecture change I'd strongly recommend

Do not let agents directly read a giant shared-memory object.

Put a context gateway between agents and memory.

Agent
  ↓
Context Request
  ↓
Permission / Policy Engine
  ↓
Context Builder
  ↓
Memory / Data

The agent might ask:

get_context(
    entity="AAPL",
    purpose="news_analysis"
)

And the system returns only:

recent news
relevant existing insights
basic company metadata

The agent never knows what else exists.

This gives you:

smaller contexts
less leakage
cheaper inference
easier permissions
reproducible backtests

And importantly, you can log exactly what information the agent saw.

That is essential for evaluating trading agents.

Logging

I'd log significantly more than agent output.

Every invocation should roughly create:

run_id
simulation_time
agent_id
agent_version
model_version
prompt_version
input_context_refs
context_snapshot_hash
output
tool_calls
working_memory_changes
promotion_requests
governance_results
latency
token_usage
cost

Then a trade should be traceable like:

Trade #8172
  ↓
portfolio decision #612
  ↓
trade candidate #916
  ↓
insight #3114
  ↓
news analysis #6711
  ↓
news event #99271

That lineage will become one of your most valuable assets.

It lets you answer:

Why did the system buy this?

and also:

Which agent actually contributed alpha?

Backtesting

Your 1–6 month speedrun idea is exactly the right type of evaluation, but make the simulation clock fundamental.

Agents must never know anything later than:

simulation_time

So every query becomes:

WHERE created_at <= simulation_time

And ideally you preserve two timestamps:

event_time
knowledge_time

Example:

earnings quarter ended: June 30
earnings released: August 1

The agent cannot use June 30 as the knowledge date.

This is one of the easiest ways to accidentally introduce lookahead bias.

I would evaluate four levels separately
Agent quality

Example for news analyst:

precision of relevant news
insight accuracy
direction accuracy
confidence calibration
insight half-life
Promotion quality
accepted insight accuracy
rejected insight accuracy
false promotion rate
memory pollution rate

This evaluates your governance layer independently.

Strategy quality
return
Sharpe
Sortino
max drawdown
hit rate
profit factor
turnover
transaction costs
System quality
latency from event → decision
cost per decision
tokens per decision
agent invocations
duplicate work
memory size
context size

The last one matters a lot.

A multi-agent system that generates 4% more return but uses 40× inference may be a worse system.

One thing I'd add early: attribution

Every insight and trade proposal should have contributor attribution.

For example:

trade T431

contributors:
  news_agent          0.00
  news_analyst        0.22
  company_analyst     0.46
  risk_analyst       -0.08
  portfolio_agent     0.40

You don't need those weights to be perfect initially.

But eventually you want to calculate:

PnL by agent
PnL by insight type
PnL by news source
PnL by company analyst
PnL by strategy
PnL by confidence bucket

That will tell you which agents deserve more compute and which ones should be removed.

The architecture I'd aim for
                         MARKET / NEWS DATA
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Immutable Data Store│
                    └──────────┬──────────┘
                               │
                         Context Gateway
                               │
             ┌─────────────────┼────────────────┐
             │                 │                │
             ▼                 ▼                ▼
        News Agent      Company Analyst    Other Research
             │                 │
             ▼                 ▼
        Working Memory    Working Memory
             │                 │
             └──────┬──────────┘
                    ▼
             Promotion Proposals
                    │
                    ▼
             ┌───────────────┐
             │ Governance    │
             │ Gate          │
             └──────┬────────┘
                    ▼
              Shared Memory
                    │
                    ▼
             Strategy / Trade
               Construction
                    │
                    ▼
              LLM Risk Agent
                    │
                    ▼
          Deterministic Risk Engine
                    │
                    ▼
           Portfolio Risk / Sizing
                    │
                    ▼
              Execution Layer
                    │
                    ▼
                   PnL

         EVERYTHING → EVENT / AUDIT LOG

The main principle I'd hold onto is:

Agents should propose knowledge and actions. The system decides what becomes state.