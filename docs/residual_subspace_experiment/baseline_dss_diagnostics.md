# Baseline DSS diagnostics

## Accepted diagonal-covariance forecasts

The original saved M3/M4 (internal M4/M5) mean and covariance archives were reconstructed before any correction. All original DSS and Energy Score values matched their accepted records. The decomposition shows that the diagonal reconstruction term affects both the log determinant and the Mahalanobis penalty, not just marginal interval widths.

Across the 520 Bayesian model/horizon forecasts, residual-space correction lowered DSS in 435 records and raised it in 85. The aggregate original-versus-corrected log-determinant and Mahalanobis components are:

| Horizon | Internal model | Original DSS | Corrected DSS | Corrected−original DSS (95% interval) | Original ES | Corrected ES | Corrected−original ES (95% interval) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | M4 | -25.357 | -30.153 | -4.795 [-5.981, -3.612] | 0.2557 | 0.2522 | -0.0036 [-0.0044, -0.0027] |
| 1 | M5 | -25.636 | -30.297 | -4.660 [-5.848, -3.493] | 0.2525 | 0.2491 | -0.0033 [-0.0045, -0.0021] |
| 4 | M4 | -21.454 | -25.489 | -4.035 [-5.383, -2.451] | 0.4171 | 0.4145 | -0.0027 [-0.0047, -0.0006] |
| 4 | M5 | -21.835 | -25.717 | -3.882 [-5.276, -2.308] | 0.4066 | 0.4053 | -0.0013 [-0.0029, 0.0004] |

Median covariance condition numbers and the share of the Mahalanobis penalty from the single most influential covariance eigen-direction are:

| Horizon | Internal model | Original median condition number | Corrected median condition number | Original max-direction share | Corrected max-direction share |
|---:|---|---:|---:|---:|---:|
| 1 | M4 | 48.2 | 433.9 | 0.411 | 0.417 |
| 1 | M5 | 46.2 | 388.8 | 0.399 | 0.429 |
| 4 | M4 | 155.8 | 1707.6 | 0.430 | 0.403 |
| 4 | M5 | 142.4 | 1565.5 | 0.446 | 0.420 |

The corrected covariances remain positive definite, but their smaller residual-space eigenvalues make them materially more ill-conditioned. The eigen-direction contribution arrays are retained per forecast; the sign of any individual eigenvector is arbitrary.

| Horizon | Original logdet | Original Mahalanobis | Corrected logdet | Corrected Mahalanobis |
|---:|---:|---:|---:|---:|
| 1 | -35.888 | 10.391 | -42.495 | 12.270 |
| 4 | -32.777 | 11.133 | -38.380 | 12.777 |

The corrected covariance is block diagonal in the orthonormal factor/residual coordinates. Record-level factor/residual log-determinants and Mahalanobis terms are in `dss_components.csv`; eigenvalues and standardized error contributions are in `forecast_records.json`.

## Largest origin-level DSS improvements

| Origin | Internal model | Horizon | Original DSS | Corrected DSS | Improvement |
|---|---|---:|---:|---:|---:|
| 2022-07-15 | M4 | 4 | 16.522 | -10.112 | 26.634 |
| 2022-07-15 | M5 | 4 | 12.021 | -12.496 | 24.518 |
| 2022-03-25 | M5 | 1 | 8.215 | -13.973 | 22.188 |
| 2022-03-25 | M4 | 1 | 9.688 | -12.089 | 21.778 |
| 2022-02-25 | M4 | 4 | 21.102 | -0.523 | 21.625 |

## Largest origin-level DSS degradations

| Origin | Internal model | Horizon | Original DSS | Corrected DSS | Improvement |
|---|---|---:|---:|---:|---:|
| 2022-04-22 | M5 | 1 | -14.738 | 29.162 | -43.901 |
| 2022-04-22 | M4 | 1 | -14.610 | 27.503 | -42.114 |
| 2020-12-31 | M5 | 1 | -28.303 | -9.985 | -18.318 |
| 2021-11-05 | M4 | 4 | -11.520 | 5.985 | -17.505 |
| 2021-11-05 | M5 | 4 | -11.776 | 5.003 | -16.779 |

## DNS/random-walk mean–covariance counterfactuals

These are score decompositions, not new accepted forecast models. They combine the saved M0 random-walk and M2 DNS means with M0 covariance, original diagonal M2 covariance, and residual-corrected M2 covariance. The accepted M2 scenario archive has 1,024 scenarios; corrected M2 Energy Scores use 2,048 Gaussian scenarios, so that ancillary energy comparison has unequal Monte Carlo sample sizes.

| Horizon | Mean/covariance combination | Mean DSS | Difference from RW/RW | Paired 95% block-bootstrap interval |
|---:|---|---:|---:|---:|
| 1 | rw_mean_rw_covariance | -41.223 | 0.000 | [0.000, 0.000] |
| 1 | dns_mean_dns_diagonal_covariance | -25.370 | 15.853 | [9.898, 21.733] |
| 1 | rw_mean_dns_corrected_covariance | -36.070 | 5.153 | [0.843, 8.961] |
| 1 | dns_mean_rw_covariance | 53.287 | 94.510 | [47.796, 155.837] |
| 1 | dns_mean_dns_corrected_covariance | -30.163 | 11.060 | [5.292, 16.958] |
| 4 | rw_mean_rw_covariance | -36.006 | 0.000 | [0.000, 0.000] |
| 4 | dns_mean_dns_diagonal_covariance | -21.453 | 14.553 | [10.128, 19.430] |
| 4 | rw_mean_dns_corrected_covariance | -31.756 | 4.250 | [2.594, 5.712] |
| 4 | dns_mean_rw_covariance | -5.641 | 30.365 | [13.113, 52.667] |
| 4 | dns_mean_dns_corrected_covariance | -25.478 | 10.528 | [5.777, 16.024] |

The M2 correction replaces only its historical diagonal reconstruction term. The retained M0 and M2 forecast archives are unchanged. Full rows are in `dns_rw_counterfactuals.csv`; the M2-only paired comparison is in `m2_counterfactuals.csv`.

## Figures

- `covariance_spectra.png`
- `dss_change_h1.png`
- `dss_change_h4.png`

The corrected DSS values should be interpreted alongside proper energy scores and calibration in the structural recovery report. A good DSS alone is not evidence that the five residual coordinates are adequately forecast by a zero-mean, time-invariant covariance.
