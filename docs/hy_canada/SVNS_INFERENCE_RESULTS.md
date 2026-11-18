# SVNS inference: diagnostic pilots only

Two four-chain pilots used the same Canadian panel, seeds 77101-77104,
100 warmup + 200 retained iterations, 2001 decay points and one corrected
SV transition per factor per outer iteration. They are deliberately much
shorter than the paper and do not provide reliable posterior inference.

With the reference offset 0.001, decay chain means were 0.436, 0.605,
0.627 and 0.670, and decay R-hat was 4.405 with bulk ESS 4.40. The
correction-acceptance range across chains was 0.08-0.195 for level,
0.05-0.135 for slope, and 0.495-0.69 for curvature. Parameter-proposal
acceptance was measured separately and cannot be substituted for correction
acceptance.

With offset zero, the corresponding ranges were 0.35-0.485, 0.29-0.375,
and 0.545-0.63; decay R-hat was 4.595. These short nonconverged runs only
show that offset choice affects proposal efficiency; their differences do
not demonstrate differences in the exact target. Exact-zero innovations
with zero offset are explicitly rejected.

The code uses a Metropolis correction for the mixture proposal, **not**
Hautsch-Yang's post-hoc importance reweighting. Its fixed-innovation ratio
and a fixed-parameter kernel were checked independently, but joint-kernel
verification is limited and no importance-weight ESS is available. The
pilot output must not be interpreted as validated exact posterior inference.
See `HAUTSCH_YANG_ALGORITHM_AUDIT.md`. Pilot summaries, traces and diagnostics
are in `results/hy_canada/blocked_svns/`; full chains are local and ignored.
