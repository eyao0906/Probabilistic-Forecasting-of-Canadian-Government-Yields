# Structural recovery report

## Executive result

**Recommendation B: Retain the correction as a sensitivity analysis; residual adequacy remains unresolved.** The residual-subspace covariance clearly improves mean DSS and modestly improves Energy Score relative to the diagonal-reconstruction versions, but the residuals remain too dependent and undercovered to treat a static zero-mean covariance as adequate. This is a bounded covariance correction, not a new factor model, and it does not change the fixed DNS loading representation, factor sampler, priors, or accepted forecasts.

The paper labels correspond to internal archive labels M3 = M4 (constant factor volatility) and M4 = M5 (stochastic factor volatility).

For transparency, the recommendation uses this screening rule: A requires positive mean DSS improvement, joint 90% residual coverage between 0.85 and 0.95, maximum absolute residual lag-1 correlation no greater than 0.20, and maximum mean factor–residual correlation no greater than 0.30. B means DSS improves but at least one diagnostic fails; C means mean DSS does not improve. These are decision thresholds for this experiment, not previously accepted model gates.

## Forecast results

| Horizon | M3 / constant DSS | M4 / stochastic DSS | M4−M3 DSS difference (95% interval) | M3 ES | M4 ES | M4−M3 ES difference (95% interval) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -30.153 | -30.297 | -0.144 [-0.405, 0.151] | 0.2522 | 0.2491 | -0.0030 [-0.0053, -0.0008] |
| 4 | -25.489 | -25.717 | -0.227 [-0.708, 0.314] | 0.4145 | 0.4053 | -0.0091 [-0.0166, -0.0020] |

Paired stationary-bootstrap estimates use 9,999 replicates and expected block length 8 origins. Block lengths 4 and 12 are included in `metrics.json`. M0 random-walk comparisons, maturity-level RMSE/MAE/coverage/width, and paired bootstrap intervals are reported alongside these results.

## Random-walk comparison

| Horizon | Internal model | RW DSS | DNS DSS | DNS−RW DSS difference (95% interval) | DNS−RW ES difference (95% interval) | DNS−RW RMSE difference (95% interval) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | M4 | -41.223 | -30.153 | 11.070 [5.206, 17.078] | 0.0890 [0.0609, 0.1261] | 0.0391 [0.0199, 0.0596] |
| 1 | M5 | -41.223 | -30.297 | 10.926 [5.031, 16.898] | 0.0860 [0.0586, 0.1230] | 0.0392 [0.0202, 0.0598] |
| 4 | M4 | -36.006 | -25.489 | 10.517 [5.842, 15.974] | 0.0633 [0.0484, 0.0797] | 0.0220 [0.0096, 0.0337] |
| 4 | M5 | -36.006 | -25.717 | 10.289 [5.603, 15.730] | 0.0542 [0.0420, 0.0685] | 0.0207 [0.0097, 0.0309] |

The corrected DNS models remain worse than the accepted random-walk benchmark on mean DSS, Energy Score, and RMSE at both horizons. The covariance correction improves the DNS covariance specification; it does not resolve the DNS models' overall forecast ranking.

## Residual-subspace adequacy

| Horizon | Mean joint Mahalanobis² (5 df) | Joint 90% coverage | Joint 95% coverage | Max OOS residual lag-1 correlation | Max mean training factor–residual correlation | Max OOS factor-error/residual correlation (M3 / M4) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.10 | 0.723 | 0.731 | 0.864 | 0.803 | 0.397 / 0.403 |
| 4 | 9.45 | 0.731 | 0.754 | 0.914 | 0.803 | 0.434 / 0.443 |

Marginal standardized means, variances, 90% coverage by coordinate, residual-coordinate means, and window-to-window covariance estimates are in `metrics.json` and `origin_residual_diagnostics.csv`. Median residual-coordinate variances shift across these OOS periods:

| Coordinate | 2016–2019 median variance | 2020–2021 | 2022–2025 | Late/early ratio |
|---:|---:|---:|---:|---:|
| z1 | 0.00341 | 0.00344 | 0.00392 | 1.15 |
| z2 | 0.00364 | 0.00344 | 0.00612 | 1.68 |
| z3 | 0.00498 | 0.00495 | 0.00564 | 1.13 |
| z4 | 0.00406 | 0.00467 | 0.00465 | 1.14 |
| z5 | 0.03050 | 0.03125 | 0.04144 | 1.36 |

The primary estimator is centered Ledoit–Wolf. Unshrunk sample covariance and OAS are fixed sensitivity checks:

| Horizon | Internal model | Estimator | Mean DSS change from Ledoit–Wolf | Paired 95% interval |
|---:|---|---|---:|---:|
| 1 | M4 | unshrunk | 0.234 | [0.046, 0.447] |
| 1 | M4 | oas | 0.029 | [0.002, 0.063] |
| 1 | M5 | unshrunk | 0.234 | [0.046, 0.447] |
| 1 | M5 | oas | 0.029 | [0.002, 0.063] |
| 4 | M4 | unshrunk | 0.222 | [0.038, 0.427] |
| 4 | M4 | oas | 0.031 | [0.001, 0.069] |
| 4 | M5 | unshrunk | 0.222 | [0.038, 0.427] |
| 4 | M5 | oas | 0.031 | [0.001, 0.069] |

## Data integrity and limits

- All 520 corrected records use the original accepted 130 origins, two horizons, and saved 6,000-draw-per-chain posterior checkpoints.
- The 260 M0 random-walk comparator records come from the already accepted classical baseline archive. The DNS/RW covariance swaps are explicitly counterfactual diagnostics.
- All corrected covariances were symmetric and positive definite and passed Cholesky factorization; their condition numbers rise substantially under correction. Orthogonal factor/residual DSS components reproduced the full DSS.
- The original Bayesian archive, accepted M0–M2 baseline, samplers, priors, and manuscript files were hashed before and after the experiment; hashes match.
- Residuals are treated as zero-mean and serially independent at forecast time, and factor–residual cross-covariance is set to zero. This experiment diagnoses those restrictions; it does not estimate residual dynamics or joint latent states.
- Training and outcome yields are historical/revised-vintage data, not real-time publication vintages. Results are conditional on the accepted two-step DNS factor extraction.
- GPU was unavailable in the active Python environment (`torch` is not installed); NumPy/SciPy CPU execution was used.

Reproducibility details: `results\residual_subspace_experiment_20261116\experiment_manifest.json`. Main archive: `results\residual_subspace_experiment_20261116`. Supporting reports: `mathematical_specification.md` and `baseline_dss_diagnostics.md`.
