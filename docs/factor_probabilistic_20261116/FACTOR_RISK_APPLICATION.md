# Factor Forecasts as a Limited Risk Illustration

## Purpose and mapping

This note translates marginal factor forecasts into a transparent
factor-isolated yield and duration-sensitivity example. It is not a
portfolio-risk model.

For each factor forecast, the predicted change from the origin,
\(\Delta f_j=f_{t+h,j}-f_{t,j}\), is mapped through the fixed five-year
Nelson–Siegel loading at
\(\lambda=0.5160377358490565\):
\[
\Delta y_{5,j}= \Lambda_{5,j}\Delta f_j,\qquad
\Lambda_5=(1,\;0.358206,\;0.282447).
\]
Factor values and yields are in percentage points, so yield contributions
are multiplied by 100 to report basis points. The factors are varied one at
a time; the other two contributions are held at zero.

For a standardized portfolio sensitivity, apply the first-order duration
approximation to a hypothetical $1,000,000 market value with five-year
modified duration:
\[
\Delta P_j\approx-\$1{,}000{,}000\times5
\frac{\Delta y_{5,j,\mathrm{bp}}}{10{,}000}.
\]
This corresponds to $500 per basis point of five-year yield. It assumes
linear duration exposure and omits carry, convexity, transaction costs,
factor dependence, and yield-curve reconstruction error.

## Illustrative factor-isolating key-rate sensitivities

The fixed loading matrix also gives a transparent algebraic way to define
hypothetical maturity sensitivities with a chosen factor exposure. For a
vector \(d\) over the eight maturities, the implied factor exposure is
\(b=\Lambda^\top d\). Since \(\Lambda\) has full column rank, the
minimum-Euclidean-norm vector satisfying a target \(b^\ast\) is
\[
d=\Lambda(\Lambda^\top\Lambda)^{-1}b^\ast.
\]
For unit level, slope, or curvature exposure, the maturity sensitivities are:

| Maturity (years) | Level | Slope | Curvature |
|---:|---:|---:|---:|
| 0.25 | 0.203 | 0.434 | −1.721 |
| 0.50 | 0.057 | 0.416 | −0.832 |
| 1 | −0.153 | 0.371 | 0.504 |
| 2 | −0.343 | 0.257 | 1.921 |
| 5 | −0.164 | −0.081 | 1.897 |
| 7 | 0.074 | −0.248 | 1.020 |
| 10 | 0.363 | −0.416 | −0.140 |
| 30 | 0.963 | −0.732 | −2.649 |

These are normalized first-order sensitivity weights: multiplying each column
by the loading matrix gives the corresponding unit vector in factor space.
The entries include long and short signs and are not security positions or
tradability claims. They illustrate how maturity combinations can represent
factor exposures; they do not supply joint factor risk or portfolio VaR.

## Latest accepted-origin example

The following are 90% simulation intervals at the 2025-11-28 origin. The
first interval is the isolated five-year yield contribution; the second is
the corresponding linear dollar sensitivity. Values are rounded and are
conditional on the accepted Bayesian predictive draws.

| Factor | Horizon | Model | Five-year yield contribution, bp (5th, median, 95th) | Linear P&L, $ (5th, median, 95th) |
|---|---:|---|---:|---:|
| Level | 1 | M3 | [−15.88, −0.50, 15.23] | [−7,616, 250, 7,942] |
| Level | 1 | M4 | [−12.17, −0.37, 11.57] | [−5,784, 185, 6,083] |
| Level | 4 | M3 | [−31.77, −0.57, 30.52] | [−15,261, 285, 15,886] |
| Level | 4 | M4 | [−25.54, −1.39, 23.02] | [−11,512, 695, 12,770] |
| Slope | 1 | M3 | [−7.27, 0.03, 7.49] | [−3,744, −17, 3,637] |
| Slope | 1 | M4 | [−5.08, 0.26, 5.81] | [−2,904, −128, 2,540] |
| Slope | 4 | M3 | [−14.96, −0.11, 14.66] | [−7,330, 56, 7,479] |
| Slope | 4 | M4 | [−10.57, 0.96, 12.73] | [−6,367, −481, 5,287] |
| Curvature | 1 | M3 | [−16.09, 0.46, 16.88] | [−8,441, −231, 8,047] |
| Curvature | 1 | M4 | [−9.29, 0.02, 9.15] | [−4,574, −11, 4,645] |
| Curvature | 4 | M3 | [−30.17, 1.59, 33.86] | [−16,929, −796, 15,086] |
| Curvature | 4 | M4 | [−18.67, 0.20, 18.80] | [−9,400, −102, 9,334] |

## Limits

These intervals are not additive: the models estimate each factor
separately and do not estimate their joint predictive dependence. The
portfolio mapping is a stylized first-order sensitivity, not a repricing
exercise or a forecast of realized P&L. It should not be called portfolio
VaR or expected shortfall. In particular, the prior yield-level
residual-subspace exercise found substantial remaining residual dependence
and undercoverage, so the factor marginals cannot be promoted to calibrated
full-curve risk.

The complete 1,560-row set of origin-level sensitivities is saved as
[`factor_risk_illustration.csv`](../../results/factor_probabilistic_20261116/factor_risk_illustration.csv).
