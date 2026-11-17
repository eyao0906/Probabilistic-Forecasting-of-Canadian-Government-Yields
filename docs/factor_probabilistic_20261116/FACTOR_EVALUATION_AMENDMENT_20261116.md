# Factor evaluation protocol amendment

**Amendment date:** 2026-11-16  
**Base protocol:** `FACTOR_EVALUATION_PROTOCOL.md`, version 1  
**Status:** Pre-analysis amendment for corrected stochastic-volatility PIT and bootstrap robustness

## Reason

An independent audit identified that the original M4 PIT implementation
averaged Gaussian CDFs using horizon variances integrated over future
stochastic-volatility paths. That computes a Gaussian moment approximation,
not the CDF of the accepted stochastic-volatility posterior predictive
distribution. This is an implementation correction, not a change to the
accepted Bayesian model or the scoring target.

## Changes

1. Retain the exact posterior-mixture Gaussian CDF for constant-volatility
   M3.
2. For stochastic-volatility M4, sample accepted posterior draws and future
   log-volatility paths with the existing forecast transition. Conditional on
   each complete path, evaluate the Gaussian factor forecast CDF using the
   path-specific variance
   \[
   V_H(\theta,h_{T+1:T+H})=
   \sum_{k=1}^H b^{2(H-k)}e^{h_{T+k}}.
   \]
   Average these conditional CDF values. This Rao–Blackwellizes over future
   factor innovations while preserving the random volatility paths.
3. Use a separate deterministic random stream for the PIT, leaving the two
   accepted CRPS streams, forecast moments, predictive scenarios, CRPS, DSS,
   interval endpoints, forecast origins, and accepted posterior draws
   unchanged. Record the PIT Monte Carlo standard error from the eight batches.
4. As a predeclared sensitivity analysis, repeat the six paired CRPS
   stationary-bootstrap comparisons with expected block lengths eight and
   twelve origins. Keep the original four-origin bootstrap as the frozen
   primary analysis; do not select a block length based on significance.

## Outputs and integrity

The amendment affects M4 PIT values, PIT histograms, PIT summaries, and
calibration prose. It does not change CRPS, DSS, forecast means or variances,
predictive samples used by the accepted CRPS streams, interval calculations,
or the original six-test inference. Any separate Monte Carlo uncertainty
summary for the paired mean CRPS difference is reported as simulation
uncertainty, distinct from time-series sampling uncertainty.

## Reconciliation and verification

The version-1 protocol remains unchanged; its SHA-256 is
`e446c30c1ba5007b0265d17d3895b9a73c7e0b78198711ed3e05f4668625502f`.
The 130-origin factor evaluation was regenerated from accepted posterior
draws; MCMC was not run. All 3,120 forecasts and accepted-chain checks passed.
The corrected PIT focused tests compare horizons one and four against
200,000 independently simulated predictive outcomes and pass. The separate
simulation check makes a further full-evaluation seed replication
unnecessary.

The origin-level mean PIT changes are:

| Factor | Horizon | Before | Corrected | Change in mean | Maximum absolute origin change |
|---|---:|---:|---:|---:|---:|
| Level | 1 | 0.473119 | 0.473168 | 0.000049 | 0.004409 |
| Level | 4 | 0.510070 | 0.509839 | −0.000231 | 0.005586 |
| Slope | 1 | 0.503297 | 0.503295 | −0.000002 | 0.013866 |
| Slope | 4 | 0.481639 | 0.481882 | 0.000244 | 0.018841 |
| Curvature | 1 | 0.530644 | 0.530934 | 0.000290 | 0.003969 |
| Curvature | 4 | 0.519823 | 0.519862 | 0.000040 | 0.005390 |

Across all forecast rows, forecast means, predictive variances, CRPS, CRPS
Monte Carlo standard errors, DSS, MAE, squared-error components, and
predictive-interval endpoints are bit-for-bit unchanged. The PIT value,
histogram, grouped PIT summary, and calibration diagnostics are the intended
changes. Bootstrap block-length 4/8/12 results are saved in
`paired_tests.json`; all six-test Holm-adjusted curvature \(p\)-values remain
above 0.20 for horizon one and range from 0.106 to 0.208 for horizon four.

SHA-256 provenance:

| Item | SHA-256 |
|---|---|
| Evaluation base commit recorded in the manifest | `23a71b147a17c6ca1ee96b6040be442c16d78235` |
| Frozen version-1 protocol | `e446c30c1ba5007b0265d17d3895b9a73c7e0b78198711ed3e05f4668625502f` |
| Accepted input manifest | `0b2af5c3c7bb261973ea4ec04c151eedbf7cf1428f22d0e08eaae87a212bfd77` |
| Corrected factor-evaluation source | `60a92f4b95caa1c07f14cf78ba4e6763ce18ec2c6c2b5fc03dbcad259a73115f` |
| Focused PIT/evaluation tests | `99f064f977f06408fc29cd9ee869d86dc35052e4d94dcbe43d4d9fee5a0ce90a` |
| Updated forecast table | `6581452e3b1aabd1a59474e0ff6da32d86fba448c481d828e9497c5c4c8c6a48` |
| Updated calibration diagnostics | `3b3d7dccb3b3ab55a32077c061bc51f53555f42f3cde48390b2ab66c777358a6` |
| Updated paired tests | `1603fd602bc6b0faa056f7bb3b149b6b829c7a169498b93e6916b683c4a3b5ad` |
| Updated experiment manifest | `639ca115f0441cac540a313ea818cd073d741ad03f7a74982ce9320fe6e92e39` |
| Updated metric table | `dc301aee69e4970f5541ee6b9075149b050b86ab5f80b45f0eda8d82af24cc96` |
| Corrected PIT figure | `2242963c2c0ada53724beae08e6a53201ac03834eb7cc32288a795181042f2ba` |

The evaluation ran with the above commit at `HEAD` and a modified working
tree. The source hash pins the exact evaluator used; the recorded commit is
the evaluation base, not a claim that the source had already been committed.
