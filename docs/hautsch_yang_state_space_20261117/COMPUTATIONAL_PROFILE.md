# Computational profile

These are observed elapsed wall times on this session's local Windows
environment, not hardware-independent estimates. The full-model screens ran
four chains sequentially; their quoted times sum measured chain times and
exclude summary/IO overhead. The SV conditional screens time all four
chains plus diagnostics, and are not comparable to full joint-model fits.

| Experiment | Elapsed seconds | Minimum bulk ESS | Minimum bulk ESS per second |
|---|---:|---:|---:|
| SS_M3, dataset 1, original step | 30.32 | 7.38 | 0.243 |
| SS_M3, dataset 1, smaller step | 32.05 | 38.67 | 1.207 |
| SS_M3, dataset 2, original step | 31.89 | 19.44 | 0.610 |
| SS_M3, dataset 2, smaller step | 30.50 | 13.71 | 0.449 |
| SS_M4, dataset 1 | 35.46 | 4.69 | 0.132 |
| SS_M4, dataset 2 | 32.64 | 5.20 | 0.159 |
| Known-innovation SV, centered, persistence released | 1.12 | 7.16 | 6.407 |
| Known-innovation SV, interwoven, persistence released | 1.54 | 56.50 | 36.693 |
| Known-innovation SV, centered, persistence and variance released | 1.73 | 6.52 | 3.764 |
| Known-innovation SV, interwoven, persistence and variance released | 2.25 | 30.83 | 13.715 |

The ESS-per-second figures use the worst bulk ESS in each run, potentially for
different parameters. Since all the runs failed at least one convergence gate,
these ratios are tuning diagnostics, not validated estimates of posterior
accuracy. The screens have 120 observations and cannot establish the cost of
835-week Canadian fits. Peak memory was not measured, parallel chains were
not benchmarked, and no full rolling experiment was run.

The small screening archives contain checksummed parameter traces and, where
noted in the synthetic report, selected volatility-state traces. Full factor
and volatility paths are not retained for every draw. Large future posterior
archives should be stored outside ordinary Git history with an immutable
manifest, SHA-256 checksums, and recovery instructions.
