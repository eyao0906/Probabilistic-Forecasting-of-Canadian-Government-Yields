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

After independently checking the collapsed Metropolis ratio, the same two
SS_M3 datasets and RNG stream seeds were rerun with the decay logit proposal
SD reduced from 0.05 to 0.005. This is a fixed proposal-scale change only; the
posterior target is unchanged. The two designs have matching generated-data
hashes. The shorter step raised decay acceptance from 0.02–0.04 to 0.20–0.26
in dataset 1, reducing decay R-hat from 1.552 to 1.046 and increasing decay
bulk ESS from 7.4 to 95.0. In dataset 2, acceptance increased from 0.06–0.14
to 0.60–0.65 but decay R-hat worsened from 1.176 to 1.220 and bulk ESS fell
from 19.7 to 15.7. The curvature persistence block also failed.

| Dataset | Tuned maximum R-hat | Tuned minimum bulk ESS | Tuned minimum tail ESS | Status |
|---|---:|---:|---:|---|
| 1 | 1.091 | 38.7 | 52.9 | rejected |
| 2 | 1.243 | 13.7 | 18.6 | rejected |

This bounded proposal change did not establish reliable inference. Acceptance
alone is not a convergence criterion, and the short runs do not establish
whether further changes or longer chains would succeed.

Individual posterior intervals, chain means, acceptance rates, parameter-wise
diagnostics, source hashes and checksummed chain checkpoints are retained in
`results/hautsch_yang_state_space_20261117/synthetic_screen_20261117/`.
The fixed-step comparison is in the sibling
`synthetic_screen_tuned_m3_20261117/` directory; its command is
`python -m src.ssdns_hy.synthetic_pilot --tuned-m3`. The original screen's
exact source files are preserved at commit `d731660`; running the newer
source against its checkpoint intentionally refuses a fingerprint mismatch.
Because the chains fail convergence gates, their interval coverage and
parameter summaries are diagnostic only. This screen does not establish
repeated-coverage rates, latent-volatility recovery, predictive recovery,
different initial-chain robustness, or performance over the full requested
range of configurations.

## SV blocks conditional on known factor innovations

The `sv_conditional_screen` holds one synthetic factor's innovations fixed,
with 120 observations, true expected log innovation variance \(\log(0.04)\),
SV persistence 0.8, stationary log-variance 0.1, and four chains started at
different volatility paths and, when released, different parameter values.
It uses 500 warmup and 1,500 retained draws per chain. The following stages
release the path and then parameters in sequence. Gates include all 120
volatility-state marginals as well as each released global parameter.

| Released | Maximum rank R-hat | Minimum bulk ESS | Minimum tail ESS | Status |
|---|---:|---:|---:|---|
| Path only | 1.015 | 613.8 | 710.0 | rejected |
| Path and expected variance | 1.032 | 227.7 | 278.4 | rejected |
| Plus SV persistence | 1.561 | 7.2 | 14.4 | rejected |
| Plus SV stationary variance | 1.640 | 6.5 | 8.7 | rejected |

With persistence released, its four chain means were 0.897, 0.946, 0.740
and 0.833 despite approximately 0.53–0.56 acceptance. This identifies a
serious mixing problem in the centered persistence block even when factor
innovations are known; it does not by itself establish weak identification.
The path-only stage came closer but still failed the unchanged R-hat gate.
The stored pointwise path-coverage proportions describe one simulated path
and are not repeated-dataset credible-interval coverage.

The command is `python -m src.ssdns_hy.sv_conditional_screen`; source hashes,
full per-state diagnostics, chain parameter means and path summaries are
stored in `results/hautsch_yang_state_space_20261117/sv_conditional_screen_20261117/`.
These conditional screens do not estimate factor dynamics, observation
variances, decay, or a full joint posterior. They retain summaries rather than
all individual path draws.
