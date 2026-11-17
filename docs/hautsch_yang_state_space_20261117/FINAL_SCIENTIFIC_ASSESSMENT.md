# Interim scientific assessment

**Decision: Outcome C for now.** The proposed joint-state-space MRP is not yet
computationally supported by the required validation. Retain the accepted
factor-level project as the primary MRP unless a future, independently
validated recovery study demonstrates otherwise. This is a stopping decision,
not a claim that the proposed model is inherently impossible.

1. **Original failures:** The published SV path sampler had a confirmed
   uninitialized-memory defect, invalidating the old M4 inference. Corrected
   short synthetic screens still show poor mixing, including the M3 curvature
   variance and M4 SV persistence. The relative contributions of weak
   identification, posterior geometry and specification remain unresolved.
2. **Synthetic recovery:** Neither joint model passed the declared R-hat and
   ESS gates. Independent numerical checks support individual transition
   blocks, but full repeated recovery, coverage and model-wide posterior
   correctness have not been established.
3. **Canadian estimation:** Not attempted because the synthetic gate failed.
4. **Latent versus OLS factors:** No valid Canadian joint posterior exists
   here, so no numerical comparison is available. The two approaches
   condition on different information and cannot be compared through raw
   posterior standard deviations alone.
5. **Volatility dynamics:** The corrected known-innovation conditional
   experiments reveal slow persistence mixing. They do not establish whether
   Canadian SV parameters are identifiable or economically interpretable.
6. **Effect of SV on uncertainty:** Not determined for the Canadian data.
7. **Posterior predictive fit:** Not checked on Canadian data; no claim of
   adequate yield or residual-subspace fit is justified.
8. **Regime forecasts:** Not run.
9. **Two-step comparison:** Historical benchmark archives remain intact, but
   no valid common-target joint-model comparison has been produced.
10. **MRP decision:** Do not replace the existing project on this evidence.
    A defensible next step is improved synthetic mixing and repeated recovery
    with dispersed full-model starts and realistic versus diffuse
    initialization; only then should initial-window Canadian inference be
    considered.

See [the sampler audit](SAMPLER_RECOVERY_AUDIT.md), [synthetic results](SYNTHETIC_RECOVERY_REPORT.md),
and [computational profile](COMPUTATIONAL_PROFILE.md) for evidence and
limitations. No forecast scores or missing empirical findings were invented.
