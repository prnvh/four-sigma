# LLM Risk Analyst

The LLM Risk Analyst is an evidence-bound advisory model that challenges a company
or stock thesis rather than making the final trade decision. It searches for hidden
assumptions, invalidation scenarios, event and regime risk, correlated exposures,
and second-order effects. It estimates success, neutral, and failure scenarios for
an explicit horizon while reporting unsupported categories as coverage gaps. Its
structured response is validated before use and can cite only the supplied context.
The agent has no capability to execute, approve, veto, or resize a trade. A
`FinalRiskReview` pairs its analysis with the deterministic position-risk result,
but derives the final result and approved size exclusively from that deterministic
decision. Therefore an optimistic AI assessment can never bypass a hard rejection
or restore size removed by deterministic limits.
