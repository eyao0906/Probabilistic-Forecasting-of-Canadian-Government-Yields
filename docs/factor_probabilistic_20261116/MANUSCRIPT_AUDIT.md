# Manuscript, numerical, and integrity audit

## Scope and research design

This audit covers the current factor-level manuscript in `paper/MRP_Draft.qmd`,
its rendered PDF, the accepted posterior archive, and the factor-evaluation
outputs. The study uses M3 for Bayesian DNS with constant factor volatility
and M4 for Bayesian DNS with stochastic factor volatility; these correspond to
archive labels M4 and M5. The manuscript contains no legacy M5 label.

The two-step DNS design is represented accurately: fixed Nelson–Siegel
loadings extract three factors by cross-sectional least squares, then separate
Bayesian regressions treat those estimates as observations. The paper does
not describe a Kalman filter, joint latent-factor estimation, or a yield-level
likelihood in the second stage. The fixed decay is
\(\lambda=0.5160377358490565\), selected on the initial training window.

## Mathematical and statistical findings

- M3 uses the exact Gaussian AR(1) likelihood conditional on the first
  extracted factor. M4 uses a stationary latent log-volatility path and an
  exact log-\(\chi^2\) likelihood correction; the Gaussian mixture is a
  proposal, not the posterior likelihood.
- The prior is placed on the long-run mean \(\mu_j\) and persistence \(b_j\),
  with \(a_j=(1-b_j)\mu_j\). This avoids the infinite-moment ratio generated
  by an independent intercept/persistence prior. The appendix distinguishes
  finite-horizon predictive moments from an infinite-horizon stationary
  factor variance.
- \(U=4\) gives a sufficient bound for the stated posterior-moment argument,
  but is not necessary for stationary prior-predictive second moments under
  the log-expected-variance parameterization. Posterior tail behavior above
  four remains untested; the paper reports this as a limitation.
- The factor fits are independent. In the initial 835-week training window,
  the mean absolute pairwise correlation among the eight yield reconstruction
  residual series is 0.659611, with maximum 0.988809. Thus diagonal
  reconstruction covariance is a working restriction, not an empirical
  claim of uncorrelated residuals. The least-squares residual is orthogonal
  to the fitted loading columns at each date; that projection identity does
  not imply independence or diagonal covariance across maturities.
- Training-only auxiliary AR residuals show substantial non-Gaussianity and
  ARCH effects. These diagnostics are not posterior-standardized M4 residuals
  and do not establish out-of-sample misspecification tests; they appropriately
  limit claims about Gaussianity and calibration.
- The M4 PIT now averages conditional Gaussian CDFs over future volatility
  paths. This is the predictive mixture CDF, not the CDF of a
  moment-matched Gaussian. The focused check against 200,000 simulated
  predictive outcomes passes at horizons one and four.

## Numerical reconciliation

- The evaluation checked 130 origins, 3,120 forecasts, and 24 accepted
  posterior-chain files per origin. All reconstructed posterior forecast
  means and variances matched accepted records.
- Compared with the superseded PIT calculation, forecast means, variances,
  CRPS, CRPS MCSE, DSS, MAE, squared-error components, and predictive
  interval endpoints are bit-for-bit unchanged. PIT values and calibration
  diagnostics are the intended changes.
- M4 has lower CRPS than M3 in all six factor-horizon groups. Neither
  curvature comparison remains significant after six-test Holm adjustment.
  The predeclared expected block lengths 4, 8, and 12 produce the same
  family-wise conclusion.
- The 130-origin average-score MCSE is 0.000100–0.000640. It is much smaller
  than the curvature mean differences (0.00536 and 0.01549), so Monte Carlo
  noise does not account for those point estimates. Origin-level bootstrap
  uncertainty and multiplicity adjustment still prevent a confirmatory
  superiority claim.
- M4 marginal coverage ranges from 0.862 to 0.931 at 90%; calibration varies
  across factors and horizons. PIT means near 0.5 do not imply uniformity.

The detailed scores, PIT reconciliation, group diagnostics, bootstrap
sensitivities, and output hashes are recorded in
`FACTOR_EVALUATION_REPORT.md` and
`FACTOR_EVALUATION_AMENDMENT_20261116.md`.

## Current manuscript revision

The narrative now presents factor-level score differences as a descriptive
pattern, reports the relative curvature CRPS reductions (3.4% and 4.9%), and
states that no formal cross-factor heterogeneity test was conducted. It
distinguishes simulation MCSE for the 130-origin mean from origin-level
bootstrap uncertainty and model/specification uncertainty. The text also
notes that the four-observation curvature CRPS reduction does not correspond
to a lower RMSE. These are presentation and interpretation changes only:
accepted scores, tables, forecasts, priors, samplers, and evaluation
protocols were not changed.

## Build and software checks

Rendered with `quarto render paper\MRP_Draft.qmd --to pdf`. The updated PDF
has 35 pages and a 195-word abstract. The no-write paper verifier passed:
there are no unresolved citations or cross-references, duplicate labels,
missing includes, unresolved PDF reference markers, or blank pages; the
12-point main-text configuration and first-page title checks also pass.
Quarto completed both LuaLaTeX passes without emitting a build error.

Eight focused tests in `tests/test_factor_probabilistic_experiment.py` pass.
Python compilation and `git diff --check` pass. The combined Bayesian
prior/sampler test run completed 27 of 29 tests; two phase-2 experiment tests
raise the existing strict error `Phase 2 sources or evidence have changed`.
The conditional sampler, target, and moment tests in that run passed. No
MCMC or full repository suite was run.

The integrated PDF preview rendered blank, so visual inspection of the newly
rendered pages was not available in this environment. Automated PDF
verification passed, and the revised content is prose rather than a new
equation or table. The author should still inspect the final PDF visually
before submission.
