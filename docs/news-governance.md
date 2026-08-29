# News insight governance

Commit 20 adds deterministic quality controls between a News Analyst promotion
proposal and shared memory. News insights must carry an explicit future expiry,
meet a configurable minimum evidence count, cite the same evidence in the finding
and proposal, and use explicitly classified sources from an allowlist. The system
does not infer whether a publisher is reputable from its name. Repeated claims are
rejected after normalization, while an opposing directional insight is retained
and tagged so downstream research can see the disagreement. Entity, confidence,
and evidence mismatches fail closed. A rejected proposal is audited by the existing
governance gate and cannot modify shared memory.

The rules operate on `InsightRevision` values containing a `Finding` and are passed
to `GovernanceGate` through its optional `rules` collection. Commit 19 can use this
policy when its working-memory promotion pipeline is wired end to end.
