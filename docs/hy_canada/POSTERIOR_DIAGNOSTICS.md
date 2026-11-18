# Posterior diagnostics

Reference thresholds are rank-normalized R-hat <= 1.01 and both bulk and
tail ESS >= 400. Neither model comes close for all parameters. The reported
summaries are diagnostics only; their means, intervals and Monte Carlo errors
are not reliable inference.

| Run | Warmup + retained per chain | Max R-hat | Min bulk ESS | Min tail ESS |
|---|---:|---:|---:|---:|
| DLNS, 2001-grid pilot | 1000 + 2000 | 2.135 | 5.28 | 6.65 |
| DLNS, 2001-grid substantial | 5000 + 15000 | 1.577 | 7.00 | 8.27 |
| SVNS, offset .001 | 100 + 200 | 4.405 | 4.40 | 4.34 |
| SVNS, offset 0 | 100 + 200 | 4.595 | 4.38 | 4.34 |

DLNS 30- and 120-grid chains have no retained decay transitions. Their
decay R-hat is numerically degenerate, not a meaningful finite estimate.
Longer fine-grid DLNS chains improve the diagnostic but still do not cross
regions; this is not evidence that the full posterior was explored. The
model's diagonal measurement-error assumption and the OLS residual-complement
autocorrelation remain possible structural concerns, separate from sampling.

Parameter-level R-hat, bulk/tail ESS and MCSE are in each `summary.csv`;
chain means, transition counts, times and selected boundary/lag-one
diagnostics are in each `run.json`. Traces and ACFs cover the parameters in
their respective result directories. Boundary summaries beyond persistence
and decay, and importance-weight ESS, have not been computed.
Correction-acceptance fractions and log-ratio quantiles describe a
Metropolis transition, not an importance-weight ESS.
