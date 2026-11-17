# Final mathematical and statistical audit

## Executive assessment

The current paper's main empirical workflow is internally coherent as a
**two-step, marginal-factor forecasting comparison**. Its accepted posterior
draws are retained; factor histories are extracted by cross-sectional least
squares under fixed Nelson–Siegel loadings, then modeled separately in a
Gaussian AR(1) with either constant factor variance (M3) or stochastic
log-volatility (M4). This is not joint latent yield-factor estimation, and
the results do not establish joint yield-curve calibration or portfolio risk.

The main statistical qualification is substantive: initial-training AR
residuals are non-Gaussian and show ARCH effects, the factors are fit
independently, and maturity reconstruction residuals are strongly correlated
despite the retained diagonal covariance restriction. The paper now states
these limits. Corrected M4 PIT values use future volatility-path mixtures;
forecast scores, moments, and interval endpoints remain unchanged.

## Model, likelihood, and estimation

For extracted factor \(x_{j,t}\), let
\[
r_{j,t}(\mu_j,b_j)=x_{j,t}-(1-b_j)\mu_j-b_jx_{j,t-1}.
\]
The second-stage likelihood conditions on the first extracted factor and
contains only transitions \(t=2,\ldots,T\).

- **M3:** Each innovation is conditionally Gaussian with fixed variance
  \(\exp(m_j)\). Its likelihood contribution is proportional to
  \(\exp[-m_j/2-r_{j,t}^2/(2\exp(m_j))]\).
- **M4:** Each innovation has variance \(\exp(h_{j,t})\), with stationary
  AR(1) log variance \(h_{j,t}\). The exact likelihood contribution is
  \(\exp[-h_{j,t}/2-r_{j,t}^2/(2\exp(h_{j,t}))]\); a Gaussian-mixture
  approximation is used to propose volatility paths and is corrected against
  this exact target.
- The long-run mean and persistence are sampled directly; the intercept is
  derived as \(a_j=(1-b_j)\mu_j\). For M4,
  \(\omega_j=m_j-v_j/2\) and \(q_j=v_j(1-\phi_j^2)\). The transformed
  \(\operatorname{atanh}(\phi_j)\) and \(\log(v_j)\) targets include their
  Jacobians.
- The factor likelihood and priors are separate by factor. The accepted
  model has no yield-level likelihood, no estimated cross-factor dependence,
  no prior or update for \(\lambda\), and no posterior update for the
  reconstruction covariance.

The bounded \(v_j\) prior and proper coefficient priors support the documented
posterior-propriety and finite-**horizon** moment argument. Since the
persistence prior reaches arbitrarily close to one, this does not prove a
finite prior mean for the infinite-horizon stationary factor variance. An
unbounded \(v_j\) posterior and its moments have not been established.

## Prior family, support, numerical choices, and evidence

The revised numerical choices were introduced after the earlier
specification had been evaluated. The revised values were selected from
initial-training evidence; 2016–2025 scores were not used for prior selection.
The initial 835 weekly factors both determine frozen hyperparameters and
enter the likelihood, so this is empirical Bayes rather than independent
prior elicitation.

| Quantity | Family and support/parameterization | Numerical implications and training evidence | Limitation |
|---|---|---|---|
| Long-run mean \(\mu_j\) | Normal on the real line; parameterized directly, with \(a_j=(1-b_j)\mu_j\). This avoids the divergent long-run-mean moments of an independent intercept/persistence prior. | \(N(\bar f_{j,\mathrm{train}},(2s_{j,\mathrm{train}})^2)\). The multiplier 2 gives a central 90% interval of \(\pm3.29\) training SDs. In 100,000 joint prior draws, the mean falls outside the observed range in 23–34%, versus 72–89% under the original joint prior. | The mean-parameterization and persistence scale changed together; the audit does not isolate multiplier 2. |
| Factor persistence \(b_j\) | Normal centered on the initial-window OLS AR slope, truncated to \((-1,1)\) for stationarity. | SD \(0.03\) gives central 90% intervals 0.939–0.998 (level), 0.938–0.998 (slope), 0.924–0.996 (curvature); median half-lives 21.6–32.2 weeks. Prior-only SDs 0.015 and 0.06 give a combined 5th–95th half-life envelope of 14–635 and 4.8–177 weeks. | The scale is project-specific and judgmental; training diagnostics varied it jointly with the mean prior. |
| Log expected variance \(m_j\) | Normal on the real line. Its center is the log initial-sample OLS residual mean square; \(m_j\) is fixed across expanding windows. | SD 1 gives \(\exp(m_j)\) a mean/median ratio \(e^{1/2}=1.649\) and central 90% multiplicative range 0.193–5.180. The accepted training-only full-prior simulation gives one-/four-week curve-change RMS medians 0.167–0.174/0.357–0.365 percentage points, versus training 95th percentiles 0.1845/0.3849. | The movement audit changes prior components jointly and does not establish SD 1 as optimal. |
| Log-volatility persistence \(\phi_j\) | \((\phi_j+1)/2\sim\mathrm{Beta}(20,1.5)\), hence support \((-1,1)\). Kim, Shephard, and Chib (1998) motivate stationarity and discuss near-unit-root behavior; Omori et al. (2007) describe high persistence. | Mean 0.8605, median 0.8865, central 90% interval 0.6489–0.9827; at weekly frequency the mean implies a 4.61-week half-life and the interval 1.60–39.71 weeks. | Source applications use daily exchange-rate or TOPIX returns, not weekly Canadian DNS factors. The paper makes no frequency adjustment or Canadian calibration claim. |
| Stationary log-volatility variance \(v_j\) | Truncated exponential \(p(v)=e^{-v}/(1-e^{-U})\) on \((0,U)\); \(h_{j,1}\) has the stationary normal distribution. \(m_j\), \(\omega_j\), and \(q_j\) are related by \(m_j=\omega_j+v_j/2\), \(q_j=v_j(1-\phi_j^2)\). | Rate 1 and \(U=4\) give mean 0.925, median 0.675, and 95th percentile 2.697. At the bound, stationary \(\operatorname{SD}(\log\sigma)=1\), or a 95% volatility range 0.141–7.10 times its conditional median. Raising \(U\) from 4 to 8 changes the four-week training-only 99th-percentile movement by about 0.1%. | Rate and bound are regularization choices, not literature-calibrated. No yield-movement simulation varied the rate. The revised posterior was not refit above four; prior-predictive sensitivity does not establish posterior robustness. |

The prior provenance note records the source applications, exact transformed
densities, alternative hyperparameter calculations, and prior-predictive
diagnostics. The evidence supports plausibility on the training scale, not
optimality or score-based validation.

## Observation equation and covariance restrictions

The Nelson–Siegel equation maps the three extracted factors to eight
maturities using a single fixed loading matrix. The loading labels describe
the shapes across maturity; they do not formally identify economic shocks.
The fixed \(\lambda=0.5160377358490565\) was selected by the initial-window
training grid search. Factor extraction is by cross-sectional least squares,
not Kalman filtering. A numerical check at the eight maturities confirms that
the scalar loading equation plus its OLS residual equals the matrix
reconstruction; the largest normal-equation residual is \(7.6\times10^{-15}\).

Three distinct diagonal restrictions must not be conflated:

1. **Factor dynamics:** separate factor AR(1) coefficients exclude cross-factor
   lag effects.
2. **Factor innovations:** separate Bayesian fits assume no cross-factor
   innovation dependence.
3. **Yield reconstruction residuals:** the legacy common diagonal \(R_t\) is
   a maturity-independent working covariance in the yield-level forecast;
   it is not part of the current marginal factor likelihood or DSS.

For the initial 835-week sample, the mean absolute pairwise correlation among
the eight cross-sectional residual series is 0.659611; the maximum is
0.988809. Thus diagonal \(R_t\) has imperfect empirical support. The
least-squares first-order condition \(\Lambda' r_t=0\) makes the residual
orthogonal to the loading columns at that date; it does not make
maturity-specific residuals mutually uncorrelated. The diagonal restriction
is retained to preserve the accepted specification and is disclosed as a
limitation, not defended as a data finding.

## Training-only innovation diagnostics

The initial-window auxiliary AR residuals (834 transitions per factor) have
small lag-one correlations but non-Gaussian shape:

| Factor | Residual lag 1 | Skewness | Excess kurtosis | Jarque–Bera \(p\) | ARCH LM(5) \(p\) |
|---|---:|---:|---:|---:|---:|
| Level | −0.057 | 0.328 | 2.193 | \(1.38\times10^{-39}\) | \(6.00\times10^{-7}\) |
| Slope | 0.017 | −1.353 | 8.581 | \(<10^{-300}\) | \(5.09\times10^{-21}\) |
| Curvature | −0.009 | 0.524 | 3.738 | \(8.78\times10^{-113}\) | \(3.44\times10^{-5}\) |

These are auxiliary training residual checks, not tests on posterior-
standardized M4 innovations, not out-of-sample diagnostics, and not a
posterior-predictive check. They indicate that Gaussian innovations, no
leverage, and independent factor fits remain substantive assumptions.
Stochastic volatility models changing variance but does not remove skewness
or heavy tails. The findings narrow calibration claims but do not invalidate
the comparison of the specified marginal forecasts.

## Corrected PIT and forecast-score interpretation

For M3, the PIT is the average draw-specific Gaussian CDF. For M4, it averages
the Gaussian CDF conditional on each sampled posterior draw and complete
future log-volatility path. Substituting an expected variance and applying a
single Gaussian CDF would not equal the stochastic-volatility predictive
mixture CDF. The corrected value is tested at horizons one and four against
200,000 independent predictive scenarios.

The corrected PIT changes each group's mean by at most 0.00029, while
individual origins change by up to 0.01884. Forecast means, variances, CRPS,
CRPS MCSE, DSS, MAE, squared-error components, and interval endpoints match
the previous saved results bit-for-bit. The 130-origin average-score Monte
Carlo standard errors range from 0.000100 to 0.000640; curvature CRPS
differences of 0.00536 and 0.01549 are 16.5 and 24.2 times those MCSEs.
Thus simulation error does not explain the curvature point estimates.
Nevertheless, under stationary-bootstrap block lengths 4, 8, and 12, neither
curvature comparison passes the six-test Holm correction. The result supports
a possible curvature-specific benefit, not general model superiority.

M4 90% marginal coverage ranges from 0.862 to 0.931, and PIT means alone do
not demonstrate uniformity. Forecast origins are dependent observations, and
the evaluation does not estimate cross-factor dependence. The score and
calibration summaries are consequently marginal and conditional on fixed
loadings and extracted factors.

## Validation and remaining work

- The factor evaluation regenerated 3,120 forecasts from accepted posterior
  draws; 130 origins and 24 chain files per origin passed integrity checks.
  No MCMC was run and no sampler, prior, or accepted yield-level output was
  changed.
- Eight focused factor-evaluation tests pass. Python compilation,
  `git diff --check`, Quarto PDF rendering, and the no-write paper verifier
  pass. Two phase-2 experiment-gate tests still raise the strict
  evidence/source-change error; the other 27 tests in that combined targeted
  Bayesian run passed.
- The rendered PDF has 35 pages, no unresolved citations/cross-references or
  blank pages, and the configured 12-point main text. The integrated browser
  preview rendered blank, so a human should visually inspect the final PDF
  before submission.
- The analysis remains retrospective, uses a fixed historical data vintage,
  and does not propagate loading, factor-extraction, hyperparameter, or
  cross-factor uncertainty. It does not support real-time trading claims,
  portfolio VaR/expected shortfall, or calibrated joint-curve risk.

**Conclusion:** The paper's mathematical development and the factor-evaluation
outputs are consistent with the accepted two-step specification. The
stochastic-volatility PIT correction is justified, while the principal
empirical limitations are non-Gaussian training residuals, strong
cross-maturity reconstruction dependence, marginal-only factor fits, and the
lack of visual PDF inspection in this environment.
