# Stage A protocol, declared before repair trials

Scope: no baseline re-estimation, no raw data changes, no paper edits, no M3 work. Existing M4/M5 prior specifications remain fixed from 2000–2015. Existing pilot and rejected legacy scores remain untouched. Archive: original_source.py.txt; protected_before.json records raw, validated baseline, prior pilot and paper bytes.

## Diagnosis and ranked hypotheses
1. Centered latent log-volatility paths couple q to the latent scale, producing slow q mixing despite >0.91 FFBS acceptance. Prediction: an exact noncentered scale-interweaving step improves q ESS without changing the posterior target.
2. Strong negative phi/q posterior dependence slows scalar centered phi updates. Prediction: a joint collapsed mixture-conditional parameter/path proposal improves persistence/scale ESS if repair 1 fails.
3. Coefficient dependence is not the primary bottleneck: a/b have strong correlations but are sampled jointly with high cached ESS. Retain their exact truncated bivariate Gaussian update.
4. A target error remains possible despite existing tests. Re-run exact-target, stationary-density, FFBS and matched-prior tests; new parameterization requires an independent density/Jacobian regression and synthetic recovery.

At most two substantive repair variants. Do not extend chain lengths merely to cross thresholds. Same 3,000 warm-up and 6,000 retained draws per chain as the cached failed pilot; four independent seeds. First test repair 1 on the first-origin window. If it fails, evaluate repair 2 without more tuning attempts. Preserve every trial and source snapshot.

## Windows
Use existing fourth-week forecast-origin grid, h=1/4 common targets; initial representative is first origin; stress representative is the origin closest to 2020-03-20; late representative is final origin. Freeze actual dates, sample counts and target dates programmatically before sampling. Fixed lambda and priors use 2000–2015 only. This is observation-date historical-vintage work, not a real-time public-information experiment.

## Gates
All monitored a,b,omega,phi,q,h_first,h_mid,h_last (and M4 a,b,log_variance): finite rank-normalized R-hat <=1.01 and bulk/tail ESS >=400, four independent chains, all factors and all three windows. Retain traces and timings.

If the parameter gate passes, use a second independent four-chain run with the same settings. Evaluate both h=1 and h=4, posterior-predictive means, variance and DSS. Predeclared stability: absolute DSS difference <=0.5; each maturity mean difference <=0.05 predictive SD; each maturity variance relative difference <=10%; and standardized run differences <=3 combined batch MCSE where estimable (batching preserves chain boundaries). Covariance must be finite and strictly positive definite without post-hoc eigenvalue repair. Save predictive artifacts and reconstruct scores. No rolling forecasts until both runs pass all gates.

Synthetic check: fixed seed, stationary known AR-SV generator, short single-factor dataset; four independent chains; report 95% credible intervals and truth coverage for a,b,omega,phi,q without claiming universal sampler calibration from one dataset. Mathematical target tests are required separately.

## Prior comparability
Both beta priors are unscaled N((0,0.7),diag(25,0.09)), truncated |b|<1. M4 log variance and M5 long-run log variance share N(initial log residual variance,25). M5 phi and q add dispersion: conditional stationary E[exp(h)] = exp(omega+q/(2(1-phi^2))), not exp(omega). Matching omega priors does not match unconditional factor-variance priors. Document this limitation; do not change priors based on held-out performance. Prior variance_shape/variance_scale fields are not used by the operative M4 slice target.
