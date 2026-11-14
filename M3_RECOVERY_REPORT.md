# M3 state-space recovery study

**Run date:** 13 November 2026  
**Branch:** `research/m3-recovery`  
**Decision:** **Not recovered.** Multistart refinement recovers the better likelihood basin at the eight targeted origins, but none of the four tested specifications removes all boundary and identification concerns. The historical state-space M3 should remain excluded as an accepted benchmark.

## 1. Scope and naming

This report concerns the historical Gaussian DNS state-space model called M3 in `src/state_space_revised.py` and `results/state_space_revised/`. It is **not** the current paper's Bayesian constant-volatility M3: the current paper uses M3 for the Bayesian constant-volatility model and M4 for Bayesian stochastic volatility. Use “state-space M3” when discussing the model in this report.

The attached task specified 1-, 5-, and 20-business-day horizons. The repository's frozen [`config/experiment.json`](config/experiment.json) and accepted outputs instead use **1 and 4 weekly observation steps** (`horizons_weeks: [1, 4]`). I retained that existing design rather than silently introducing a different panel or forecast schedule. The recovery forecasts therefore are not 1/5/20-business-day estimates.

The recovery targeted the eight known unstable origins, with the original 2000–2015 initial training sample and the existing weekly expanding-window design. No accepted Bayesian M4/M5 estimation or results were changed. The original implementation is retained byte-for-byte at [`results/state_space_revised/source_snapshot/src/state_space_revised.py`](results/state_space_revised/source_snapshot/src/state_space_revised.py); its SHA-256 matches the historical source hash in the original run metadata.

## 2. Existing failure evidence

The saved state-space M3 results provide evidence for both optimization sensitivity and structural boundary behavior:

- The original 130-origin run passed the original fit diagnostic at **122/130** origins. The eight specified origins failed the retained start-spread criterion.
- The original three-start solutions at those eight origins have negative-log-likelihood spreads from **14.328 to 36.929**, far above the existing **0.02** threshold. Start 2 repeatedly reaches an inferior basin; Start 0 or Start 1 has the better likelihood. Projected gradients at these local optima can still be small, so optimizer exit flags or gradients alone do not establish that starts found the same solution.
- In the original full 130-origin run, selected fits hit transformed persistence upper bounds at parameter indices 3 and 4 and measurement-variance lower bounds at indices 10 and 13 at **all 130 origins**. The parameter ordering is \((\mu_1,\mu_2,\mu_3,u_1,u_2,u_3,\log q_1,\log q_2,\log q_3,\log r_1,\ldots,\log r_8)\); the measurement bounds correspond to 0.5Y and 5Y.
- The original independent audit found an outward one-coordinate likelihood improvement for **all 520** active-bound profiles. This is evidence that the constrained optima lie against the box, not merely that the optimizer stopped before satisfying its projected-gradient test.

The original archived forecasts and diagnostics remain untouched in `results/state_space_revised/`.

## 3. Implementation and experiment

The active implementation is [`src/state_space_revised.py`](src/state_space_revised.py). It now supports:

- 17 free parameters when diagonal measurement covariance is estimated or regularized, and 9 when it is fixed;
- the original \(u=\operatorname{atanh}(\rho)\) persistence bound of \([-3,3]\), and a configured alternative \([-4,4]\);
- explicit fixed-diagonal-\(R\) and penalized-log-\(R\) modes;
- independent optimization of the three original starts followed by seeded, controlled perturbation restarts around distinct original basins;
- original-start and restart diagnostics kept separately, including initial/final vectors, raw NLL, penalized objective, optimizer status, iterations/evaluations, projected gradient, bound distances, restart seed, basin assignment and gaps to the best candidate;
- numerically stable stationary initialization \(q/(1-\rho^2)=q\cosh^2(u)\). Parameters are still constrained by the declared bounds; estimates are not silently clipped during optimization.

The prior implementation remains available in the verified `source_snapshot` cited above. The original `accepted` field retains the historical diagnostic meaning; the added `revised_accepted` field only records whether the best valid refined solution has a small projected gradient and is reproduced by at least two candidates in its basin. Neither field by itself establishes identification.

The experiment driver is [`src/m3_recovery.py`](src/m3_recovery.py). It compares:

| Variant | Specification |
|---|---|
| A | Original estimated diagonal \(R\), persistence bound \(u_{\max}=3\), improved multistart/restarts |
| B | As A, with \(u_{\max}=4\) |
| C | \(R\) fixed to the initial-training calibration; 9 free parameters |
| D | Estimated \(R\) with a quadratic penalty on log variances; \(\lambda_R\) selected using blocked validation inside 2000–2015 |

For C, \(R\) is the maturity-wise mean squared cross-sectional DNS residual from the initial 835 observations, floored at the existing \(10^{-6}\) variance floor. In percentage-point-squared units the diagonal is:

| Maturity (years) | 0.25 | 0.5 | 1 | 2 | 5 | 7 | 10 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Calibrated variance | 0.009976 | 0.000527 | 0.009493 | 0.006859 | 0.002447 | 0.005723 | 0.013296 | 0.010194 |

For D, the objective is raw NLL \(+\frac{\lambda_R}{2}\sum_j(\log R_{jj}-\log R^{(0)}_{jj})^2\). Raw NLL and penalized objective are stored separately. The three chronological validation blocks each contain 52 observations; each fold reselects the DNS loading and recalibrates \(R^{(0)}\) using only its training prefix. The selection criterion is pooled validation **raw conditional NLL per observation**. The selected \(\lambda_R=0.1\), equivalent to a Gaussian penalty scale of 3.162 on log \(R\). Fold-level and aggregate results are in `regularization_cv.json`.

The run used three original starts, eight restarts, L-BFGS-B, `maxiter=1000`, `ftol=1e-14`, `gtol=2e-7`, and master restart seed 271026. Restart seeds are recorded per attempt. All model variants used the same target dates, scoring functions and forecast-scenario seed rule.

## 4. Targeted results

All four variants produced a valid selected solution at all eight target origins, and all **32/32** selected solutions met the revised repeat-in-basin criterion. The retained original-start diagnostic tells a different story:

| Variant | Original starts pass strict 0.02 diagnostic | Mean original-start raw-NLL spread | Distinct basins across target origins |
|---|---:|---:|---:|
| A | 0/8 | 24.90 | 2 at each origin |
| B | 1/8 | 20.05 | 2 at 7 origins; 1 at 1 |
| C | 8/8 | effectively 0 | 1 at each origin |
| D | 0/8 | 24.91 | 2 at each origin |

The restart stage consistently found and repeated the best recorded basin, but the high likelihood solution was not consistently found by the unmodified original starts in A, B or D. Variant A's aggregate scores are numerically indistinguishable from the historical archive at the table's precision; the restart improves the search evidence, not forecast performance. Distinct basins in A, B and D can imply maximum one- and four-step forecast-mean differences of about **14.5 basis points** at these origins.

### Boundary and conditioning results

| Variant | Bounds hit at every one of the 8 origins |
|---|---|
| A | Level and slope persistence upper bounds; 0.5Y and 5Y measurement-variance lower bounds |
| B | 0.5Y and 5Y measurement-variance lower bounds |
| C | Level and slope persistence upper bounds |
| D | Level and slope persistence upper bounds |

Every one-coordinate outward profile improved the raw likelihood at each active bound: 32/32 profiles for A and 16/16 for each of B, C and D. The corresponding penalized objective also improved outward for D. Thus B removes the persistence-bound contact in this targeted sample but retains Q/R boundary behavior; C and D constrain or regularize R but retain persistence boundary behavior. No tested specification resolves both.

Stationary-initial-covariance condition numbers were finite in all fits. Their averages were 4.31 (A), 1.55 (B), 11.05 (C) and 4.31 (D). The independently verified minimum predictive-covariance eigenvalue across saved forecasts was 0.000710. These numerical checks do not resolve parameter identification.

### Forecast scores on the eight origins

These are descriptive targeted-subset averages, not estimates of full-period performance or formal comparisons. RMSE is in percentage points; smaller DSS, Energy Score and RMSE are better.

| Variant | DSS 1-step | ES 1-step | RMSE 1-step | DSS 4-step | ES 4-step | RMSE 4-step |
|---|---:|---:|---:|---:|---:|---:|
| Historical state-space M3 | −13.898 | 0.5444 | 0.2665 | −8.708 | 0.7244 | 0.3756 |
| A | −13.898 | 0.5444 | 0.2665 | −8.708 | 0.7244 | 0.3756 |
| B | −13.947 | 0.5407 | 0.2648 | −8.668 | 0.7124 | 0.3698 |
| C | 0.995 | 0.5123 | 0.2451 | 3.403 | 0.7191 | 0.3616 |
| D | −13.894 | 0.5443 | 0.2665 | −8.709 | 0.7244 | 0.3756 |
| M0 random walk | −44.766 | 0.2017 | 0.1202 | −34.199 | 0.5319 | 0.3031 |
| M1 PCA–VAR | −21.010 | 0.3093 | 0.1587 | −15.353 | 0.6116 | 0.3280 |
| M2 classical DNS | −10.384 | 0.4611 | 0.2256 | −6.955 | 0.6674 | 0.3499 |

The fixed-\(R\) variant C has poor DSS on this subset despite lower RMSE than A; the change in covariance specification materially affects its predictive distribution. The random walk and PCA–VAR outperform A–D on all three listed aggregate metrics in this targeted subset. These selected dates are not a substitute for full-sample comparison or paired uncertainty analysis.

Coverage, interval width, maturity-specific RMSE/MAE, PIT, parameter trajectories, bound hits, basin forecast differences, all optimizer attempts and score inputs are saved in the run directory. The report does not select a model by in-sample likelihood or by these eight forecast origins.

## 5. Verification

The focused test run passed **15 tests** across the new recovery tests and existing state-space tests. Coverage includes 9/17-dimensional parameterizations, fixed-\(R\) likelihood gradients and forecasts, regularized objective gradients, training-only covariance calibration, persistence-bound stability, deterministic restarts, original-start preservation and mixed-column CSV output.

[`src/verify_m3_recovery.py`](src/verify_m3_recovery.py) independently recomputed all **352** candidate likelihoods and penalized objectives using a joint maturity-vector Kalman update. It also checked the selected forecasts, targets, scenarios, scores and accepted-result hashes:

- maximum selected NLL/objective difference: \(9.1\times10^{-11}\);
- maximum forecast-mean difference: \(1.8\times10^{-15}\);
- maximum forecast-covariance difference: \(1.4\times10^{-17}\);
- maximum score difference: \(1.8\times10^{-13}\);
- accepted M0–M2 hashes unchanged; accepted Bayesian analysis hashes unchanged.

This establishes arithmetic and artifact consistency, not identification or scientific validity.

## 6. Assessment and recommendation

**Hypothesis A (primarily optimization): partly supported.** The inferior Start 2 basin is real, and the controlled restarts recovered the better basin at all eight targeted origins. However, the original three starts still fail the strict 0.02 spread diagnostic at 7/8 origins for B and 8/8 for A and D. A's chosen fit and forecasts are the same as the historical best-start solution. The optimizer issue is not the whole problem.

**Hypothesis B (structural identification): supported.** The existing 130-origin audit documents persistence and measurement-variance boundary hits at every origin. The targeted variants trade these symptoms: wider persistence resolves the persistence contact but not the R boundary; fixed or regularized R resolves its boundary but not persistence. The outward profiles and materially different basin forecasts reinforce the concern.

**Decision: Not recovered.** Do not reintroduce the historical state-space M3 as a valid benchmark on the basis of these results. The new targeted study did not proceed to a new 130-origin run: each A–D variant retained a boundary problem at every targeted origin, so none met a defensible gate for full validation. The existing rejected 130-origin output remains useful diagnostic evidence, but it does not certify any of the new specifications. A future candidate would need to specify a joint, scientifically justified resolution of persistence and Q/R identification, pass targeted diagnostics, and then be validated across all 130 origins before inclusion.

## 7. Reproduction and artifacts

From the repository root in the existing environment:

```powershell
# Unit and regression tests
.\.venv\Scripts\python.exe -B -m unittest tests.test_m3_recovery tests.test_state_space_revised -v

# Reproduce the targeted experiment into a new, empty results directory
.\.venv\Scripts\python.exe -B -m src.m3_recovery --scope targeted --restarts 8 --output results\m3_recovery_<new_timestamp>

# Export parameter, boundary and calibration summaries
.\.venv\Scripts\python.exe -B -m src.summarize_m3_recovery results\m3_recovery_<new_timestamp>

# Independently verify all fit/forecast outputs
.\.venv\Scripts\python.exe -B -m src.verify_m3_recovery results\m3_recovery_<new_timestamp>
```

The completed targeted run is [`results/m3_recovery_20261113T205639Z/`](results/m3_recovery_20261113T205639Z/). It contains configuration/environment and source hashes, blocked-CV details, initial \(R\) calibration, all fit attempts, forecast arrays and scores, parameter trajectories, bound frequencies, maturity calibration summaries, verification and run logs. [`results/m3_recovery_20261113T205426Z/INTERRUPTED.md`](results/m3_recovery_20261113T205426Z/INTERRUPTED.md) documents a separately preserved partial attempt; do not treat it as a completed result or overwrite it.

Files changed or added for this task:

- `src/state_space_revised.py`
- `src/m3_recovery.py`
- `src/summarize_m3_recovery.py`
- `src/verify_m3_recovery.py`
- `tests/test_m3_recovery.py`
- `M3_RECOVERY_REPORT.md`
- the isolated directories under `results/m3_recovery_20261113T205639Z/` and `results/m3_recovery_20261113T205426Z/`

The branch was created as requested. **No Git commit was created.** The experiment configuration records the base commit and worktree state. Pre-existing pending archive/PDF changes and the earlier handoff memo were left untouched; review `git status --short` before any further staging or cleanup.
