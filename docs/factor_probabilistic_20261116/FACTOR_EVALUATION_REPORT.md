# Factor Probabilistic Forecast Evaluation

## Design and integrity

The frozen protocol evaluates 130 common weekly origins, horizons one and
four, and three OLS-extracted DNS factors. The complete comparison contains
3,120 factor forecasts: 1,560 for M0/M2 and 1,560 for Bayesian M3/M4. The
Bayesian predictions use only accepted posterior draws; no MCMC, prior,
sampler, or accepted yield forecast was changed.

Before evaluation, all 130 accepted origins and 3,120 posterior-chain files
passed recorded SHA-256 checks. The archived 520 Bayesian model-by-horizon
records had target dates matching the fixed panel. Reconstructed factor
predictive means and variances agreed exactly with the saved accepted
records. The new factor-scenario streams are reproducible from protocol
version 1 and seed 20261117.

M3 denotes the constant-volatility Bayesian DNS (archive M4); M4 denotes the
stochastic-volatility Bayesian DNS (archive M5). M0 is the training-only
factor random walk, and M2 is the unrestricted OLS factor AR(1).

## Primary score: CRPS

Lower values are better. Entries are mean CRPS in factor percentage-point
units across 130 origins; baseline scores are analytic Gaussian CRPS, and
Bayesian scores are Monte Carlo estimates from 8,192 predictive draws per
forecast.

| Factor | Horizon | M0 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|---:|
| Level | 1 | 0.05964 | 0.05962 | 0.05980 | 0.05968 |
| Level | 4 | 0.11277 | 0.11280 | 0.11249 | 0.11185 |
| Slope | 1 | 0.06836 | 0.06861 | 0.06870 | 0.06830 |
| Slope | 4 | 0.14587 | 0.14684 | 0.14624 | 0.14411 |
| Curvature | 1 | 0.15826 | 0.15614 | 0.15639 | 0.15102 |
| Curvature | 4 | 0.32748 | 0.31702 | 0.31744 | 0.30195 |

M0, M2, and the Bayesian models are close for level and slope. Stochastic
volatility has lower curvature CRPS than the other three models at both
horizons. That pattern is descriptive and is not evidence of general
superiority.

## Confirmatory Bayesian comparison

The paired difference is M4 minus M3, so a negative value favors
stochastic-volatility DNS. Bootstrap intervals are 95% percentile intervals
and are unadjusted; Holm adjustment applies to the six two-sided tests.

| Factor | Horizon | Mean CRPS difference | 95% interval | Raw p | Holm p |
|---|---:|---:|---:|---:|---:|
| Level | 1 | −0.00012 | [−0.00073, 0.00046] | 0.702 | 1.000 |
| Level | 4 | −0.00063 | [−0.00289, 0.00105] | 0.534 | 1.000 |
| Slope | 1 | −0.00041 | [−0.00165, 0.00085] | 0.525 | 1.000 |
| Slope | 4 | −0.00213 | [−0.00640, 0.00224] | 0.338 | 1.000 |
| Curvature | 1 | −0.00536 | [−0.01053, −0.00016] | 0.042 | 0.210 |
| Curvature | 4 | −0.01549 | [−0.02807, −0.00263] | 0.018 | 0.106 |

The 9,999-replicate stationary bootstrap uses a four-origin expected block
length and common resampled origin indices. Curvature has the largest
estimated stochastic-volatility gains. Neither curvature comparison remains
significant at 5% after the six-test Holm correction. Mean paired Monte Carlo
standard errors for the six comparisons range from 0.0011 to 0.0069 factor
percentage points; these are not negligible relative to several estimated
differences and further counsel against a strong superiority claim.

## Point forecasts and calibration

Across factors, point RMSE differences are small. For curvature at four
observations, M4 has RMSE 0.592 versus 0.589 for M3 and 0.589 for M2, despite
its lower CRPS. This illustrates that a distributional-score change need not
improve the predictive mean.

Empirical marginal 90% coverage is descriptive (nominal coverage is 90%):

| Factor | Horizon | M0 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|---:|
| Level | 1 | 0.838 | 0.838 | 0.838 | 0.862 |
| Level | 4 | 0.854 | 0.854 | 0.854 | 0.892 |
| Slope | 1 | 0.900 | 0.900 | 0.900 | 0.900 |
| Slope | 4 | 0.900 | 0.900 | 0.908 | 0.885 |
| Curvature | 1 | 0.954 | 0.954 | 0.954 | 0.885 |
| Curvature | 4 | 0.938 | 0.954 | 0.954 | 0.931 |

Coverage varies by factor and horizon: level intervals undercover at the
reported nominal levels, while curvature intervals generally overcover,
especially at four observations. M4 slope intervals also undercover at four
observations. Mean PIT values are near 0.5, but this alone does not establish
uniform PIT calibration; the PIT histograms and interval coverage are
descriptive with 130 origins.

## Interpretation

The factor-level evidence is narrower than a yield-curve forecast
validation. Marginal CRPS favors M4 most clearly for curvature, but the
adjusted paired tests do not reject equality. Level and slope scores are
nearly indistinguishable across the four specifications, and point
forecasting differences are modest. No result here establishes joint
factor dependence, calibrated eight-maturity coverage, portfolio risk, or
outperformance in a trading application.

Machine-readable outputs and figures are in
[`results/factor_probabilistic_20261116/`](../../results/factor_probabilistic_20261116/):
the origin-level forecast table, aggregated metrics, paired tests,
factor-isolated risk sensitivities, and CRPS/PIT figures.
