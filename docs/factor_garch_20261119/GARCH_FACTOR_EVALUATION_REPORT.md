# AR(1)-GARCH(1,1) factor benchmark and historical factor plots

## Design and implementation

This is a separate, supervisor-requested extension to the frozen factor
evaluation. It uses `src.bayesian_stage_a.make_design()` and the accepted
fixed-loading least-squares factors without refitting lambda or changing the
accepted Bayesian chains, forecasts, or manuscript.

The GARCH fit at each origin uses only factor values from the first research
observation through that origin, inclusive. There are 130 accepted origins,
from 2016-01-08 through 2025-11-28, and horizons of one and four weeks. Thus
the initial expanding fit has 836 observations (through the first origin),
after the fixed 835-observation 2000-2015 prior/training window.

Each factor is fitted independently with `arch==8.0.0`, Gaussian errors,
an AR(1) conditional mean with intercept, and GARCH(1,1) variance. Factor
values are multiplied by 100 for estimation in basis points; intercepts
are divided by 100 and variance parameters by 10,000 before reporting and
scoring in percentage points. The one-week predictive distribution is
Gaussian and scored analytically. Four-week forecasts recursively simulate
8,192 complete AR-GARCH paths per factor-origin, carrying the fitted last
residual and conditional variance forward. The predictive moments use the
corresponding analytical GARCH variance recursion. CRPS is an unbiased
empirical U-statistic computed in eight independent batches of 1,024; the
reported MCSE is the batch-score standard deviation divided by the square
root of eight. Four-week PIT and intervals are simulated diagnostics.

The explicit label conversion is manuscript M0 -> accepted-file M0,
manuscript M1 -> accepted-file M2, manuscript M2 -> accepted-file M3, and
manuscript M3 -> accepted-file M4. Archive Bayesian M4/M5 remain distinct
from the manuscript's M2/M3. The added benchmark remains `AR1_GARCH11`.
All model comparisons involving GARCH below are descriptive; no new
significance tests were specified or run.

## Empirical results

Mean CRPS by factor and horizon (lower is better; factor percentage-point
units):

| Factor | Horizon | M0 | M1 | M2 | M3 | AR1-GARCH11 |
|---|---:|---:|---:|---:|---:|---:|
| Level | 1 | 0.059645 | 0.059617 | 0.059799 | 0.059681 | **0.059597** |
| Level | 4 | 0.112769 | 0.112795 | 0.112488 | 0.111854 | **0.111781** |
| Slope | 1 | 0.068360 | 0.068606 | 0.068704 | 0.068297 | **0.068221** |
| Slope | 4 | 0.145869 | 0.146843 | 0.146243 | 0.144114 | **0.144055** |
| Curvature | 1 | 0.158264 | 0.156143 | 0.156387 | 0.151023 | **0.150386** |
| Curvature | 4 | 0.327478 | 0.317023 | 0.317439 | 0.301949 | **0.301117** |

AR-GARCH has the lowest mean CRPS in all six groups. Against classical AR(1)
M1, the differences are small for level, moderate for slope, and largest for
curvature, especially at four weeks. It also has lower CRPS than Bayesian
constant-volatility M2 in all groups. The gaps against Bayesian
stochastic-volatility M3 are much smaller: 0.000058-0.000832 in absolute
terms. These are point-estimate comparisons, not evidence that GARCH is
statistically superior.

| Factor | Horizon | GARCH CRPS | GARCH DSS | RMSE | MAE | 90% coverage | Mean 90% width | CRPS mean MCSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Level | 1 | 0.059597 | -3.428 | 0.1063 | 0.0829 | 0.869 | 0.3263 | 0 |
| Level | 4 | 0.111781 | -2.266 | 0.2022 | 0.1578 | 0.900 | 0.6449 | 0.000124 |
| Slope | 1 | 0.068221 | -3.170 | 0.1241 | 0.0943 | 0.900 | 0.3971 | 0 |
| Slope | 4 | 0.144055 | -1.587 | 0.2768 | 0.1996 | 0.885 | 0.7907 | 0.000142 |
| Curvature | 1 | 0.150386 | -1.643 | 0.2796 | 0.2077 | 0.885 | 0.9792 | 0 |
| Curvature | 4 | 0.301117 | -0.220 | 0.5913 | 0.4179 | 0.931 | 1.9073 | 0.000363 |

The CRPS pattern does not uniformly transfer to point-error scores,
DSS, or calibration. Curvature's GARCH RMSE is slightly higher than M1 and
M2 at both horizons even though its CRPS is lower; its four-week RMSE is
also slightly higher than M3. The GARCH 90% intervals undercover for
one-week level and curvature, and four-week slope, while curvature
overcovers at four weeks. Calibration and interval widths should therefore
be read alongside the proper score, not as confirmation of universal
improvement.

The accepted Bayesian comparison already found lower curvature CRPS for
stochastic-volatility M3 than constant-volatility M2, but its six-test
Holm-adjusted results did not support a confirmatory claim. In this
extension GARCH's further curvature CRPS reductions relative to M3 are only
0.000637 at one week and 0.000832 at four weeks, and no GARCH-specific
inference was conducted. The existing curvature pattern is descriptively
visible; its generality is not established.

## Fit diagnostics

All **390 of 390** rolling optimizations converged, and all **780 of 780**
one- and four-week forecasts were produced. No fitted AR coefficient had
absolute value at least one, and no fitted GARCH persistence
`alpha + beta` reached one. Near-unit estimates are nevertheless common:

| Factor | AR(1) coefficient range | GARCH persistence range | Near-unit warnings |
|---|---:|---:|---|
| Level | 0.9963-0.9986 | 0.9380-0.9789 | 130 near-unit AR fits |
| Slope | 0.9956-0.9990 | 0.7729-0.9122 | 130 near-unit AR fits |
| Curvature | 0.9817-0.9883 | 0.9901-0.9968 | 130 near-unit GARCH fits |

Thus every fit carries a recorded near-unit diagnostic warning, not an
optimizer failure. Level and slope autoregressions are persistently close to
one; curvature GARCH variance persistence is close to one. These features
can make longer-horizon estimates sensitive to small parameter changes.
The CSV retains each optimizer message, parameter set, persistence value,
conditional variance, likelihood, AIC/BIC, and warning.

## Historical factor figure

The CSV and plot contain the actual 1,356 weekly extracted factor values
from 2000-01-07 through 2025-12-24, in percentage points. They do not show
conditional volatility, model innovations, forecasts, or reconstructed
yields. The plot has three shared-time-axis panels, separate vertical
ranges, and shades the 2016-2025 forecast-evaluation period. The PNG was
visually inspected; its three panels, period separator, labels, and caption
are legible and unclipped. Its raster size is 3150 by 2190 pixels at 300 DPI.

## Reproducibility and validation

- Data SHA-256:
  `0b2af5c3c7bb261973ea4ec04c151eedbf7cf1428f22d0e08eaae87a212bfd77`.
- Accepted forecasts and metrics are separately hashed in
  `results/factor_garch_20261119/experiment_manifest.json`; their files
  were not overwritten.
- Scenario seed base is 20261119. A deterministic `SeedSequence` uses
  `[base, absolute origin row index, factor index, horizon]`; the resulting
  per-forecast seeds are recorded in the forecast CSV.
- Environment: Python 3.14.7, arch 8.0.0, NumPy 2.5.4, SciPy 1.18.1,
  pandas 3.1.0, Matplotlib 3.11.2. The lockfile already pins arch 8.0.0.
- The manifest records the branch base commit, input and accepted-result
  hashes, code/test/lockfile hashes, counts, model mapping, seed scheme,
  dependency versions, outputs, and validation status.
- Eight new focused tests pass, including fixed design and origin alignment,
  independent arch parameter scaling, analytical forecast-moment agreement
  with arch's forecast API, recursive simulation moments and reproducibility,
  CRPS validation, output aggregation, and manuscript-label mapping. The
  eight existing factor-evaluation tests also pass.

No manuscript source or accepted Bayesian output was edited. A pre-existing
working-tree change to `paper/MRP_Draft.pdf` was left untouched.
