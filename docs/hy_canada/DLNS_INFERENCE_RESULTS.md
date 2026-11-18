# Preliminary DLNS inference

The real 2000-2015 Canadian training data were used directly. The first
exploratory run used four chains with 1,000 warmup and 2,000 retained draws
each. Its maximum rank-normalized R-hat was 1.827 (`r_5`) and minimum bulk
ESS was 5.87; full parameter-wise diagnostics and chain means are in
`results/hy_canada/initial/`. In particular, decay had R-hat 1.749 and bulk
ESS 6.08. This is not an accepted posterior.

A bounded follow-up adds a fixed correlated proposal built from the
training-only exploratory draws; its covariance is recorded in
`results/hy_canada/dlns_joint_proposal.json`. A run with that proposal still
showed chain disagreement. See `POSTERIOR_DIAGNOSTICS.md` and the final
machine-readable results for the latest run. Acceptable proposal acceptance
rates do not prove exploration: several measurement variances, decay and
factor innovation variances move together, and chains can remain in distinct
regions.

The initial values are independently randomized around the same
training-calibrated center with standard deviation 15% of each prior
coordinate's scale. This is only modest dispersion, not evidence that all
plausible initializations were explored. Consequently initialization
sensitivity is only partially assessed by the chain-specific results.
