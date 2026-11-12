# Frozen main-experiment analysis and limited sensitivity

This specification is written while main production is running and before inspecting aggregate out-of-sample results. It supplements the existing training-only prior protocol; it does not change sampling, predictive acceptance thresholds, or the scientific research question.

## Main analysis

- Accept analysis only after all 130 planned common origins have accepted matched M4/M5 forecasts at both horizons, with unchanged gate, source, posterior, and predictive archive hashes. No successful-origin subset may represent the full experiment.
- Report mean DSS and mean Energy Score over origins. DSS omits the common Gaussian constant. Energy Score retains the existing off-diagonal U-statistic ensemble convention with 2,048 scenarios; units are yield percentage points.
- Calculate pooled RMSE as the square root of the mean squared error over all origins and eight maturities, not as the average of origin-level RMSE. Report yield percentage points (multiply by 100 for basis points if shown). Report marginal 90% coverage pooled over origins and maturities, with per-maturity detail separately.
- All paired differences are M5 minus M4. Negative DSS, Energy Score, or RMSE differences favor M5. A positive coverage difference is not intrinsically an improvement: interpret relative to 90% nominal coverage and interval width.
- Use the originally frozen paired stationary bootstrap of complete origin vectors, sharing resampled indices across models, maturities, metrics, and horizons. Main expected block length is eight forecast origins (approximately 32 weeks); report 95% percentile intervals, 9,999 replicates, fixed seed 271026. Recompute each bootstrap RMSE from resampled squared errors before differencing. Expected block lengths four and twelve are dependence-assumption sensitivities, not outcome-selected tuning. These intervals are descriptive, not multiplicity-adjusted hypothesis tests; historical nonstationarity and a finite number of regimes remain limitations. An initial version of this supplementary note inadvertently proposed a different moving-block scheme; it was corrected to protocol.md before aggregate forecast results were examined. The original frozen protocol controls.
- Independently reconstruct DSS, squared errors, interval endpoints, and coverage from each saved predictive archive. Independently reconstruct the U-statistic Energy Score for the three preselected representative origins using a distinct pairwise-distance implementation. Verify positive-definite covariance, finite scenarios, and exact matching targets and maturities throughout.

## Limited prior-bound sensitivity

- Keep rate one and training-only Gaussian m priors unchanged. Bounds two and eight were predeclared alongside main bound four before production. Do not select any bound using forecast scores.
- Restrict new sensitivity sampling to the same initial, stress, and late windows already preselected for Phase 2. Reuse main M4 results rather than refitting M4. Use four M5 chains per alternative bound/window with the same 3,000 warmup and 6,000 retained draws and original diagnostic thresholds. Use fixed distinct seeds, recorded in each configuration. Predictive moments must also pass derived diagnostics and positive-definiteness checks before sensitivity scores are reported.
- Report posterior mass near the upper bound, changes in analytic predictive covariance and representative scores, and clearly label this a three-window sensitivity, not a complete 130-origin robustness experiment. If conclusions materially depend on the bound, state that the full-sample robustness claim remains unresolved; do not conceal that limitation or optimize the prior.
- No new sensitivity sampling is necessary if main production stops at a scientific acceptance failure. Preserve all completed origins and issue a technical partial report instead.

## Stop and preservation rules

- Stop new production submissions on the first origin failure. Preserve both successful checkpoints and failed chains. Never discard failed chains or relax thresholds to obtain acceptance.
- Do not alter finite sampler or Phase 2 evidence while production is active. One substantive sampler repair has been used; no more than two total are allowed. Infrastructure changes do not license a scientific gate bypass.
- Leave raw data, M0-M2 artifacts, original draft, and rejected historical outputs intact. Update the existing validated manuscript only with evidence that has actually passed its applicable gates.
