# DLNS inference: not converged

The main run uses four chains, 5,000 warmup and 15,000 retained iterations,
2001 equally spaced decay values on [0.2,1.2], and seeds 67101-67104.
Conditional transition counts are 6,788, 8,381, 8,445 and 8,326. Despite
that local movement, retained decay ranges were [0.3835,0.4265],
[0.575,0.6095], [0.5765,0.611], and [0.577,0.6115]; no chain switched
between the lower and upper regions. Chain decay means were 0.401, 0.592,
0.594 and 0.593. The first chain's mean `r_0.25` was 0.0157 versus 0.0089
for the others; its mean `Q_LL` was 0.00505 versus about 0.0141.

Decay R-hat was 1.577, bulk ESS 7.00 and tail ESS 8.27. Several covariance
parameters had R-hat about 1.53 and bulk ESS about 7. These values show
failure, not reliable posterior summaries. Inspect the chain-specific
`run.json`, parameter `summary.csv`, traces, and conditional decay plot in
`results/hy_canada/blocked_dlns/substantial/`. Selected full factor paths
and all parameter draws are generated locally but ignored by Git.

The 30- and 120-point grids did not move decay in any chain. The 2001-point
pilot moved locally but also remained separated. The first saved chain-1
30-grid conditional, reevaluated on a finer local grid, had 5%-95% width
about 0.00105, far smaller than the 30-grid spacing 0.03448. The local
conditional plots are conditional on
different factor paths and do not determine the marginal posterior mode
weights. Neither the existence of multiple posterior modes nor the adequacy
of any prior region is established.
