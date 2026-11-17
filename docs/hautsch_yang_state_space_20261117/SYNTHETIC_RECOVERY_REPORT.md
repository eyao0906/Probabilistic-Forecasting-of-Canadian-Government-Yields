# Synthetic recovery: initial screens

The corrected sampler was screened on two independent datasets per model,
each with 120 observations, four independent RNG streams, 150 warmup iterations
and 300 retained draws per chain. The simulated pre-observation factor follows
the same \(N(\mu,10^6I)\) initialization used by the inference model. Both
models use decay 0.52, factor persistence (0.82, 0.72, 0.55), factor innovation
variance (0.04, 0.05, 0.03), and diagonal measurement variance 0.0036. SS_M4
uses volatility persistence 0.8 and stationary log-volatility variance 0.1.
The prior is recalibrated from each simulated panel.

| Model | Dataset | Maximum rank R-hat | Minimum bulk ESS | Minimum tail ESS | Status |
|---|---:|---:|---:|---:|---|
| SS_M3 | 1 | 1.552 | 7.4 | 8.5 | rejected |
| SS_M3 | 2 | 1.176 | 19.4 | 33.8 | rejected |
| SS_M4 | 1 | 2.948 | 4.7 | 6.6 | rejected |
| SS_M4 | 2 | 2.322 | 5.2 | 7.9 | rejected |

The unchanged thresholds are 1.01, 400 and 400, respectively. SS_M3's first
screen had particularly poor decay mixing (R-hat 1.552, bulk ESS 7.4) and
decay acceptance between 0.02 and 0.04. The third factor's persistence also
mixed poorly in both SS_M3 screens. SS_M4 additionally had poor mixing in
volatility persistence and stationary variance. These short runs establish
failure of the screen, not proof that the models are unidentifiable.

The extremely diffuse initial-state distribution produces some simulated
initial factors with magnitudes in the hundreds or thousands of percentage
points. This is the specified proper prior, not a realistic yield scenario,
and substantially changes decay information. These exact-initialization
screens must not be conflated with realistic fixed-initialization robustness
studies. The first screen's per-chain decay acceptance makes proposal scaling
a testable computational concern, but does not by itself identify the
posterior's geometry or establish a solution.

Individual posterior intervals, chain means, acceptance rates, parameter-wise
diagnostics, source hashes and checksummed chain checkpoints are retained in
`results/hautsch_yang_state_space_20261117/synthetic_screen_20261117/`.
Because the chains fail convergence gates, their interval coverage and
parameter summaries are diagnostic only. This screen does not establish
repeated-coverage rates, latent-volatility recovery, predictive recovery,
different initial-chain robustness, or performance over the full requested
range of configurations.
