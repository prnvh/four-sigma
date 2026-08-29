# Portfolio Risk Agent

The Portfolio Risk Agent reviews a proposed trade in the context of the whole book,
rather than assessing the company in isolation. It reads the trade, current portfolio,
deterministic position-risk decision, before/after portfolio-risk snapshot, and a
small set of approved insight summaries. It can recommend approve, reject, resize,
or defer while explaining concentration, factor, volatility, drawdown, and correlation
concerns. Its citations are restricted to the supplied insight references and it never
receives raw news evidence. The AI may be more conservative than deterministic risk,
but it can never restore a rejected trade or increase size beyond the deterministic
ceiling. It has no execution capability, so its output remains a governed recommendation.
