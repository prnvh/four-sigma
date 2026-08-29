# Structured agent output contracts

Every runtime model response is parsed into one of four explicit contracts:
`NewsAnalysisResult`, `CompanyAnalysisResult`, `RiskAnalysisResult`, or
`TradeProposalResult`. The boundary rejects free-form prose, missing or additional
fields, wrong types, non-finite or out-of-range numbers, blank strings, and duplicate
references. It also enforces domain invariants: company relationships cannot both
support and contradict the same insight, outcome probabilities must total 100, every
risk category must be assessed or declared missing, and trade direction must agree
with proposed size. The validated result is frozen before downstream use. Because
`AgentRunner` validates before its working-memory write, malformed output is audited
as a failed run and cannot enter memory.
