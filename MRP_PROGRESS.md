# MRP — current status

## Current manuscript update: revised priors

The revised-prior Bayesian run completed and its independent archive audit passed: 130/130 origins, 520 forecasts, no failures. Maximum rank-normalized R-hat is 1.00965; minimum bulk/tail ESS are 1,057.6/2,094.5. The manuscript's M3/M4 are internal archive M4/M5. At four weeks, stochastic minus constant volatility changes DSS by −0.3805 and Energy Score by −0.01048 (2.51%); RMSE changes by −0.1297 basis points with a 95% interval including zero. The random walk remains descriptively better on aggregate scores and RMSE.

`paper/MRP_Draft.qmd` and the rendered 35-page PDF use the revised results and passed the no-write paper verifier. The direct-mean and log-variance prior changes, training calibration, synthetic recovery misses, current outputs and reproduction commands are documented in `docs/bayesian_prior_justification.md`, `docs/PROJECT_HANDOFF.md` and `docs/reproduction_runbook.md`. The original U=2/8 checks used different priors and are not sensitivity evidence for the revised run. The sections below are retained as a **historical original-baseline record**, not current manuscript results.

## Scientific work complete

- Main finite-moment M4/M5 experiment accepted: **130/130 matched origins, 520 records, no failures**, horizons 1/4 weeks and eight maturities. Persistent pipeline `proc_3c897e1a1f02` exited **0**; no active numerical stage remains. Accepted checkpoints were reused, not refitted.
- Main read-only aggregate analysis accepted in `results/bayesian_finite_20261027T194239Z/rolling_main/analysis/`. Maximum rank R-hat 1.00760; minimum bulk/tail ESS 1,034.3 / 1,969.3.
- Frozen paired stationary bootstrap retained: **9,999 replicates, expected length 8 origins, seed 271026**, with lengths 4/12 sensitivity.
- All **six U=2/8 representative sensitivity cases accepted**, no failures. Their separate final gate and 48-row comparison CSV are in `results/bayesian_finite_20261027T194239Z/bound_sensitivity/`. The main analysis's historical sensitivity-pending flag is not rewritten; the separate completed sensitivity gate is authoritative for this stage.
- Accepted M0–M2 results (780 records), Phase 1/2 evidence, frozen protocols and source/checkpoint dependencies remain in place. No sampler/model changes, new review agents or repeated passed test suites were needed.

## Findings

At four weeks M5-minus-M4 DSS is **−0.3899** (95% interval [−0.7619, −0.0506]); Energy Score is **−0.01053** ([−0.01850, −0.00231]), a 2.52% reduction from M4. Both score intervals favor M5 at both horizons under block lengths 4/8/12. RMSE changes by only **−0.162 basis points** ([−0.464, 0.151]). M4/M5 marginal 90% coverage is 93.08%/94.33%; both overcover. The random walk remains better descriptively on aggregate scores and RMSE.

Alternative-bound DSS is locally stable, not uniformly favorable to M5: at four weeks M5 loses at the selected stress and late origins, and the late Energy Score ranking changes. Representative sensitivity is not full-period robustness and independent simulation contributes numerical variation. No prior was retuned.

## Paper and limitations

`paper/MRP_Draft.qmd` integrates accepted results, five reporting tables and two new essential figures. The PDF is successfully rendered and verified: **35 physical pages, 12-point main text, 172-word abstract**, title-first front matter, populated lists, and no unknown citations/cross-references, missing includes, duplicate labels or blank pages. Key tables, figures and derivations were visually inspected; the final horizontal-overhang check is clear. After cleanup, all 3,617 checked required scientific/source files remain present. The main text no longer presents debugging history as the research contribution.

Limitations: retrospective observation-time data and revisions; two-step extraction/fixed lambda; independent factor innovations/diagonal reconstruction noise; broad training-calibrated priors; only 130 origins; unadjusted descriptive intervals; simulation uncertainty; representative-only bound sensitivity. M3 remains excluded. The single synthetic intercept interval miss and the historical M3 fixture-preservation exception remain disclosed. Full author name, supervisor identity, permitted AI use and university sign-off still require human confirmation.

## Cleanup

After all numerical processes exited, **43,122 files / 1,438.2 MiB** were removed from the active tree: redundant logs/reports, duplicate test environments, isolated fixture copies, bytecode and scratch outputs. **358 historical files** were preserved in a verified external ZIP; additional original notes are included. No raw/processed data, accepted forecasts/parameters, essential diagnostics, frozen protocols or required source snapshots were deleted. Uncertain or referenced legacy scientific material was retained.

- Original paper/progress backup: `../MRP_archives/pre_final_manuscript_20261028T010801Z.zip`.
- Labeled historical evidence: `../MRP_archives/superseded_diagnostics_20261028T013219Z.zip` (verified original bytes).
- `.gitignore` excludes disposable runtime material. `README.md` provides the model/data summary, folder map, findings and exact minimal reproduction commands. No destructive Git operations or remote pushes occurred.
