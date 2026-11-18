# Final research assessment

1. **DLNS reliability:** Not established. Both four-chain runs show severe
   chain disagreement on the actual Canadian data.
2. **SVNS reliability:** Not established. The non-centered run has even
   stronger failures for several parameters.
3. **Identification:** Reliable identification of curvature persistence,
   decay and measurement variances has not been demonstrated. Level and slope
   persistence are near one in the draws, but this is not reliable inference.
4. **SV dynamics:** Not established; volatility persistence, variance and
   selected log-volatility states have poor convergence diagnostics.
5. **Joint factors versus OLS:** Exploratory smoothed factors differ from OLS
   factors at fixed decay 0.7. Based on only 25 conditional state draws per
   chain, RMS differences in level/slope/curvature are approximately
   0.288/0.316/0.642 percentage points for DLNS and 0.305/0.269/0.812 for
   SVNS. These are not defensible posterior comparisons because the chains
   have not converged, the conditional sample is small, and joint-model decay
   varies while the OLS benchmark fixes it.
6. **Parameter and measurement uncertainty:** The exploratory chains reveal
   substantial disagreement and coupling, not a reliable quantification of
   posterior uncertainty.
7. **Historical yield fit:** Not established by posterior predictive checks.
   Training-only OLS residuals show strong temporal persistence in the
   five-dimensional residual complement; OLS orthogonality also induces
   cross-maturity residual correlations.
8. **Predictive uncertainty:** No DLNS-SVNS predictive comparison is
   defensible, and no origin forecasts were produced.
9. **MRP evidence:** The mathematical checks and exploratory fits do not yet
   support an inference-focused empirical claim about the joint models.
10. **Recommendation:** Keep this as exploratory research, not a completed
    four-model comparison. Investigate the coupled decay/measurement
    geometry, initial-state and prior sensitivity, and SV path mixing using
    better-validated proposals or a mature gradient-based implementation.
    Compare methods only after reliable inference is demonstrated. Do not
    assume that more iterations, tighter priors, or joint estimation alone
    will solve the problem.

No likelihood-based comparison with differently conditioned two-step models,
Bayes factor, 130-origin backtest, or claim of improved forecast scores is made.
