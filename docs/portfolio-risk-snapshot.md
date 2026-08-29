# Portfolio risk snapshot

The portfolio risk calculator produces deterministic point-in-time risk metrics for
the current portfolio and for the same portfolio after a proposed trade. It measures
gross and net exposure, signed sector exposure, factor exposure, covariance-based
annualized volatility, largest-name concentration, current drawdown, and connected
clusters of highly correlated holdings. Every input—prices, sectors, factor loadings,
volatilities, and correlations—must be supplied explicitly; missing inputs fail closed
instead of being estimated or invented. The comparison can use the requested trade
size or a smaller size approved by deterministic position risk. Replaying identical
inputs produces an identical snapshot, making it appropriate for audit and backtests.
