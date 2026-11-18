# Posterior diagnostics

The reference targets are rank-normalized R-hat <= 1.01 and both bulk and
tail ESS >= 400. These are diagnostic reference values, not procedural
barriers. Neither model yields an accepted posterior here.

| Model / run | Max R-hat | Min bulk ESS | Min tail ESS |
|---|---:|---:|---:|
| DLNS initial | 1.827 | 5.87 | 7.99 |
| DLNS pilot-informed, independent seeds | 1.803 | 6.05 | 8.08 |
| SVNS non-centered | 2.289 | 4.99 | 7.93 |

The complete parameter-wise results, prior-versus-posterior comparisons,
lag-one autocorrelations and chain-specific means are in
`results/hy_canada/dlns_diagnostics.csv` and `svns_diagnostics.csv`;
the files also divide bulk ESS by the sum of chain-loop elapsed seconds as an
efficiency indicator, not an actual CPU-time measurement.
`*_summary.csv` adds Monte Carlo errors. Traces and autocorrelations for every
parameter are plotted by block. The prior comparison uses 50,000 draws from
the training-calibrated priors and is not independent prior validation.

For DLNS the initial proposal and the fixed pilot-informed correlated proposal
both leave severe chain separation. The latter's decay R-hat is 1.747 and bulk
ESS 6.12; `rho_C`, innovation variances and many measurement variances are
also problematic. Most level and slope persistence draws lie above 0.99, but
unconverged boundary proportions are descriptive only. Decay was not near its
numerical support endpoints. SVNS level and slope persistence similarly
concentrates near one, while SV persistence diagnostics are poor even away
from their upper endpoints. See the `*_boundaries.json` files.

The chains' measurement variance and decay behavior suggests a difficult
coupled geometry, but these runs cannot distinguish sampler inefficiency,
weak identification, prior sensitivity and model misspecification. Both
under-exploration and posterior multimodality remain possible. Reported
chain-specific means and exploratory state paths demonstrate sensitivity
without quantifying a reliable posterior.
