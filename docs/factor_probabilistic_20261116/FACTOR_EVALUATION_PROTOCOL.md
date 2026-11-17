# Factor Evaluation Protocol

**Frozen:** 2026-11-16, before calculating any new factor-level scores  
**Version:** 1  
**Branch:** `research/factor-probabilistic-forecasting-20261116`

## 1. Locked question and estimands

The primary outcomes are the one-dimensional predictive distributions of the
extracted level, slope, and curvature factors. Evaluate each at horizons
\(h\in\{1,4\}\) for the same 130 rolling origins, producing
\(130\times4\times3\times2=3{,}120\) factor forecasts: 1,560 for Bayesian M3/M4
and 1,560 for M0/M2. Lower CRPS and Dawid–Sebastiani score (DSS) are better.
CRPS is the primary proper score; DSS is a moment-based Gaussian diagnostic,
not the primary comparison for the generally non-Gaussian Bayesian mixtures.

No forecasts, priors, maturity grid, decay parameter, rolling origins, or
accepted yield-level outputs may be changed after observing factor scores.
No MCMC is to be rerun.

## 2. Data and timing

- Use the accepted weekly panel and its recorded SHA-256 hash.
- Extract factors by cross-sectional least squares using the fixed eight
  maturities and \(\lambda=0.5160377358490565\).
- Use origins from 2016-01-08 through 2025-11-28, every four observations, and
  targets at \(t+1\) and \(t+4\). The final target is 2025-12-24.
- All forecasts use factor observations available through the origin. The
  835-observation initial window through 2015-12-31 defines the accepted
  Bayesian priors; the first rolling estimation window includes the available
  2016-01-08 origin observation (836 observations total).
- Keep archive model IDs distinct from manuscript labels: manuscript M3 is
  archive M4 (constant volatility); manuscript M4 is archive M5 (stochastic
  volatility).

## 3. Forecast construction

### M0: factor random walk

For each factor \(x_t\), estimate the innovation variance as the sample
variance of training-only first differences through the origin. Forecast
\[
x_{t+h}\mid\mathcal F_t\sim
N\!\left(x_t,\;h\,\widehat{\sigma}_{\Delta x}^{\,2}\right).
\]
Do not include post-origin differences or silently regularize a non-finite or
non-positive estimate; fail the run explicitly if one occurs.

### M2: unrestricted factor AR(1)

For each factor, fit \(x_s=a+bx_{s-1}+e_s\) by OLS using observations through
the origin. Use the existing baseline convention
\(\widehat q=n^{-1}\sum_s\widehat e_s^2\), without clipping \(b\). Forecast
\[
E_t[x_{t+h}]=b^h x_t+a\sum_{k=0}^{h-1}b^k,\qquad
\operatorname{Var}_t(x_{t+h})=\widehat q\sum_{k=0}^{h-1}b^{2k}.
\]
Invalid or non-finite estimates stop the run; they are not replaced with
plausible-looking defaults.

### M3/M4: accepted Bayesian posterior predictions

Read only the accepted posterior chains and the fixed factor histories.
Reconstruct each factor's draw-specific finite-horizon predictive mean and
variance using the archived model equations; marginal forecast moments use
the law of total variance. Generate factor-only simulations with the existing
`factor_scenarios` implementation, sampling one accepted posterior draw per
scenario and propagating the factor process. Do not project the archived
eight-yield scenarios back into factors.

Use 8,192 predictive scenarios per Bayesian factor forecast, in eight batches
of 1,024. Use separate deterministic random streams for the two independent
scenario sets required by the CRPS estimator. Derive streams from the fixed
base seed 20261117 and the origin, model, factor, horizon, and stream indices;
do not use Python's process-randomized `hash()`.

## 4. Scores and calibration

For M0/M2, compute CRPS, PIT, central predictive intervals, and interval
widths analytically from the Gaussian predictive distribution. For M3/M4:

- Compute CRPS with 8,192 independent paired simulations \(X_i,X'_i\):
  \[
  \widehat{\mathrm{CRPS}}=
  M^{-1}\sum_i|X_i-y|-(2M)^{-1}\sum_i|X_i-X'_i|.
  \]
  The reported Monte Carlo standard error is the standard deviation of the
  eight batch scores divided by \(\sqrt 8\).
- Compute the PIT from the exact posterior mixture CDF, averaging the
  draw-specific Gaussian CDFs at the realized factor.
- Estimate 50%, 80%, and 90% central interval endpoints from the 8,192
  predictive scenarios. Report empirical coverage and mean interval width;
  treat these finite-simulation intervals as descriptive calibration
  diagnostics.
- Compute DSS from the exact predictive mean and variance, omitting the
  common Gaussian constant. Also report RMSE and MAE of the predictive mean.

Report CRPS, DSS, RMSE, MAE, PIT summaries, coverage, and width separately by
model, factor, and horizon. Do not aggregate the three factors into a score
that implies a joint distribution.

## 5. Paired inference and multiplicity

The confirmatory comparison is stochastic-volatility M4 minus constant-
volatility M3 CRPS, separately for the three factors and two horizons (six
tests). Negative mean differences favor M4. Compute paired two-sided
stationary-bootstrap tests over the 130 origins, with expected block length
four origins, 9,999 replicates, and common resampled origin indices across the
six tests. Center each origin-level difference under the null. Report the raw
p-value, the 95% percentile-bootstrap interval for the mean difference, and
the Holm-adjusted p-value across the six tests. Use base seed 20261118.

Comparisons involving M0 or M2 are descriptive; do not make confirmatory
significance claims for them. The protocol does not select a model by
out-of-sample results.

## 6. Limited factor-risk illustration

For context only, transform each factor's \(h\)-step change distribution
\(x_{t+h}-x_t\) through its fixed Nelson–Siegel loading at five years. Show
one factor at a time, in basis points of five-year yield contribution. Then
map that isolated yield contribution to a stylized $1,000,000 portfolio with
five-year modified duration using the first-order relation
\(\Delta P\approx-PD_{\rm mod}\Delta y\). Report medians and central 90%
simulation intervals for the accepted Bayesian models, with the factor and
horizon named.

Label the calculation as a sensitivity illustration, not a realized bond
return or a portfolio risk forecast. Do not add factor contributions, infer
their dependence, or report joint VaR or expected shortfall. State the
omitted carry, convexity, factor dependence, and residual-yield risk.

## 7. Integrity, outputs, and exclusions

Before scoring, recheck the accepted plan, forecast records, targets,
posterior-chain hashes, and exact reconstructed factor moments. Fail closed
on any mismatch. Save new code, tests, tables, figures, and reports under
`src/`, `tests/`, `results/factor_probabilistic_20261116/`, and
`docs/factor_probabilistic_20261116/`; do not overwrite accepted output.

The analysis excludes MCMC refitting, prior or sampler changes, full
state-space estimation, maturity-grid changes, joint factor-risk estimation,
new yield-level scoring, and any use of out-of-sample results to tune the
protocol.
