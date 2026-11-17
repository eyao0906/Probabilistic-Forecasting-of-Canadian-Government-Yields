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
significant at 5% after the six-test Holm correction. The mean per-origin
paired Monte Carlo standard errors range from 0.0011 to 0.0069 factor
percentage points, but the independence-based Monte Carlo standard errors of
the 130-origin average range from 0.000100 to 0.000640. For curvature, the
mean differences are 16.5 and 24.2 times those average-score MCSEs. Thus
simulation noise does not explain the curvature point estimates; uncertainty
across forecast origins and the six-test adjustment remain relevant.

The predeclared expected-block-length sensitivities preserve this conclusion:

| Factor | Horizon | Holm \(p\), length 4 | Holm \(p\), length 8 | Holm \(p\), length 12 |
|---|---:|---:|---:|---:|
| Level | 1 | 1.000 | 1.000 | 1.000 |
| Level | 4 | 1.000 | 1.000 | 1.000 |
| Slope | 1 | 1.000 | 1.000 | 1.000 |
| Slope | 4 | 1.000 | 1.000 | 1.000 |
| Curvature | 1 | 0.210 | 0.203 | 0.208 |
| Curvature | 4 | 0.106 | 0.161 | 0.208 |

Both curvature raw \(p\)-values remain below 0.05 at all three block lengths,
and their unadjusted percentile intervals exclude zero. None passes the
pre-specified six-test Holm correction. The bootstrap represents origin-level
sampling uncertainty, while the separate MCSE describes simulation error.

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

The M4 calibration summaries also report standardized-error moments and
lag-one diagnostics (origin order is the evaluation order):

| Factor | Horizon | Std. error mean | Std. error variance | PIT mean | PIT lag 1 | 90% coverage |
|---|---:|---:|---:|---:|---:|---:|
| Level | 1 | −0.101 | 1.248 | 0.473 | −0.054 | 0.862 |
| Level | 4 | 0.030 | 1.039 | 0.510 | −0.059 | 0.892 |
| Slope | 1 | 0.003 | 1.088 | 0.503 | 0.027 | 0.900 |
| Slope | 4 | −0.069 | 1.279 | 0.482 | 0.101 | 0.885 |
| Curvature | 1 | 0.066 | 0.989 | 0.531 | 0.148 | 0.885 |
| Curvature | 4 | 0.112 | 1.099 | 0.520 | 0.075 | 0.931 |

The corrected M4 PIT averages future log-volatility paths before evaluating
the path-conditional Gaussian CDF. Compared with the superseded
moment-matched-Gaussian calculation, the mean PIT changes by at most 0.00029
across the six groups; individual PIT values change by as much as 0.01884.
Forecast means, variances, CRPS, CRPS MCSE, DSS, MAE, RMSE components and
interval endpoints are identical before and after the PIT correction.
Training-only auxiliary AR residual diagnostics show substantial
non-Gaussianity (excess kurtosis 2.19, 8.58 and 3.74) and ARCH(5) effects
(LM-test \(p=6.0\times10^{-7}\), \(5.1\times10^{-21}\), and
\(3.4\times10^{-5}\)) for level, slope, and curvature. These are not
posterior-standardized M4 residuals and do not validate out-of-sample
calibration; they motivate the paper's explicit innovation-distribution
limitations.

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
the origin-level forecast table, aggregated metrics, grouped calibration
diagnostics, paired tests, factor-isolated risk sensitivities, and CRPS/PIT
figures. The corrected PIT leaves every forecast score and moment unchanged.
