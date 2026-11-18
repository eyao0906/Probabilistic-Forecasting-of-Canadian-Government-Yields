# Preliminary SVNS inference

The same 835 Canadian observations were fitted using four independently seeded
non-centered chains, each with 1,000 warmup and 2,000 retained iterations.
This was permitted despite the earlier DLNS diagnostic failures; it is an
exploratory calculation, not an accepted posterior.

The largest parameter R-hat was 2.289 for `r_0.5`; minimum bulk ESS was 4.99
and minimum tail ESS was 7.93. Volatility persistence R-hats were 1.367,
1.617 and 1.471 for level, slope and curvature. Their bulk ESS values were
9.72, 6.64 and 8.04. Volatility stationary variances also mixed poorly.
The nine selected-date log-volatility R-hats range from 1.115 to 1.381.
Per-parameter and selected-date values, chain-specific means, traces and
autocorrelations are in `results/hy_canada/`.

Metropolis block acceptance rates ranged from 0.1215 to 0.338. Each accepted
elliptical-slice path update required approximately 7.26-8.11 likelihood
evaluations on average. These statistics do not override the poor R-hat and
ESS. Non-centering alone has not solved SV mixing. Estimated time variation
and parameter correlations from these draws cannot establish meaningful
stochastic-volatility inference.

The retained log-volatility paths are local in the ignored `chains/` folder;
tracked trajectory summaries and figures are explicitly labeled exploratory
and must not be used as credible intervals. They combine the unconverged
chains. Initial values use modest independent perturbations around a common
training-calibrated center; a wider initialization sensitivity study remains
open.
