# Research Scope Revision: Factor-Level Probabilistic Forecasting

**Date:** 2026-11-16  
**Branch:** `research/factor-probabilistic-forecasting-20261116`  
**Base:** `bced5fe` (`research/residual-subspace-20261116`)

## Motivation

The original study evaluated joint probabilistic forecasts for eight Canadian
government-yield maturities. The completed residual-subspace sensitivity
analysis showed that a residual correction improves the Bayesian forecasts'
Dawid–Sebastiani score, but it does not resolve material residual dependence
or undercoverage. The corrected DNS forecasts also remain behind the accepted
random-walk benchmark on the reported yield-level scores. Those findings are
retained as prior evidence; they are not results for the revised factor study.

The revised study makes the three cross-sectionally extracted
Nelson–Siegel factors—level, slope, and curvature—the primary forecast
targets. This gives a focused comparison of the predictive distributions
already produced by the accepted Bayesian models without implying that their
eight-dimensional yield distribution is calibrated.

## Research question and scope

For each extracted factor, do Bayesian DNS forecasts improve probabilistic
forecast quality over a random walk and an unrestricted AR(1) benchmark at
one- and four-observation horizons? Does allowing stochastic factor volatility
improve on the constant-volatility Bayesian specification? What limited,
factor-isolated yield and duration-sensitivity implications follow from those
forecasts?

The evaluation covers 130 rolling forecast origins from 2016-01-08 through
2025-11-28, spaced four observations apart, with horizons of one and four
weekly observations. The final four-observation target is 2025-12-24. The
panel contains eight maturities and 1,396 weekly rows. Factor extraction uses
the fixed Nelson–Siegel decay parameter
\(\lambda=0.5160377358490565\), selected by the initial training-window grid
search. The initial prior-estimation window has 835 observations through
2015-12-31. At the first rolling origin, the current factor observation is
available to the forecasters, so the estimation sample contains 836
observations through 2016-01-08.

The four evaluated models are:

| Manuscript label | Definition |
|---|---|
| M0 | Factor random walk with innovation variance estimated from training-only factor changes. |
| M2 | Factor AR(1) estimated by unrestricted OLS, with innovation variance equal to the mean squared training residual. |
| M3 | Bayesian DNS with constant factor volatility; this is internal archive model M4. |
| M4 | Bayesian DNS with stochastic factor volatility; this is internal archive model M5. |

M3 and M4 use the accepted posterior draws. The original samplers, priors,
forecast archive, and yield-level results are not changed, and MCMC is not
rerun. The accepted archive was checked before the revised scores were
computed: all 130 origins and 3,120 posterior-chain files passed hash checks,
and reconstructed factor predictive means and variances matched all 520
accepted model-by-horizon records exactly.

## Interpretation limits

The factors are OLS proxies extracted from the fixed eight-maturity
Nelson–Siegel cross-section. They are treated as observed inputs in a
two-step procedure, not as jointly estimated latent states. The three
Bayesian factor models are estimated separately. Consequently, this study
evaluates each factor's marginal predictive distribution; it does not
identify cross-factor dependence or provide a joint yield-curve predictive
distribution.

The financial illustration maps one factor forecast at a time through the
fixed Nelson–Siegel loading at a stated maturity. A duration calculation is
first-order and illustrative. The separate factor contributions must not be
added as though their joint distribution were known. The study does not claim
calibrated full-curve risk, portfolio VaR, expected shortfall, or realized
bond-return performance.

The residual-subspace experiment remains a sensitivity analysis motivating
this narrower target. Its yield-level scores, residual diagnostics, and
limitations will be described separately from the new factor-level results.
