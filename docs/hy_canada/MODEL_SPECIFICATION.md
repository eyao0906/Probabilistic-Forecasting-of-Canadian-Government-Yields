# Model and inference specification

This is an adaptation of Hautsch and Yang's 2010 working paper (published
in 2012, DOI 10.1016/j.csda.2010.07.003). The consulted source is the
repository's working-paper copy, not a line-by-line comparison with the final
journal text. The source studies monthly U.S. yields; this study uses weekly
Canadian yields. The independent computation below is not a reproduction of
its blocked sampler.

For each date, `y_t = H(lambda) f_t + e_t`, where the row of `H` for maturity
tau is `(1, (1-exp(-lambda*tau))/(lambda*tau),
(1-exp(-lambda*tau))/(lambda*tau)-exp(-lambda*tau))`. Maturities are years
and lambda is per year. `e_t ~ N(0, diag(r))`; `f_t = mu + diag(rho)
(f_(t-1)-mu) + eta_t`; and `eta_t ~ N(0,diag(q_t))`. Conditional innovations
and observation errors are independent. There are eight measurement
variances, not an OLS reconstruction covariance. Persistence is restricted to
the positive stationary interval `(0,1)`; this is a modeling restriction,
not an implication of the data.

The first observation follows a transition from `f_0 ~ N(m0,P0)`. This is a
proper, training-calibrated initialization, not the old `10^6 I` covariance.
Set `m0` to the mean of training OLS factors computed at lambda 0.7, and set
`P0 = diag((2*s_f)^2)`, where `s_f` is their sample standard deviation floored
at 0.1. The first transition uses `q_1`, including in SVNS.

## Priors

These are empirical-Bayes, training-only choices, not literature-derived exact
hyperparameters. They use the full training window both to calibrate priors
and to fit the model. Results are conditional on that calibration and are not
a fully pre-data prior analysis. A real-time forecast must recalibrate using
only its own training window. No validation is implied by MCMC convergence.

At lambda 0.7, obtain OLS factors, their mean and standard deviation, lag-one
OLS persistence clipped to `[0.7,0.995]`, factor AR residual variances `q0`, and
cross-sectional residual variances `r0`. Floor variances at `10^-5`. In the
unconstrained coordinates used by the sampler, independently:

* `mu_j ~ N(factor_mean_j, (2*s_f_j)^2)`.
* `logit(rho_j) ~ N(logit(0.95), 1.25^2)`.
* `log(q_j) ~ N(log(q0_j), 1^2)`.
* `log(r_i) ~ N(log(r0_i), 1.5^2)`.
* `logit((lambda-0.05)/1.95) ~ N(logit(1/3), 1^2)`.

The log variance coordinates are restricted to `[-50,50]` for numerical
representability. In SVNS additionally `log(v)` is restricted to `[-10,10]`
and all realized `h_t` to `[-50,50]`; outside these supports the target is
zero. These are explicit truncations of the stated priors, not latent clipping.
No claim is made that the precise bounds or prior scales are empirically
validated. The posterior prior-comparison table is conditional on training
calibration and ignores the negligible numerical truncation in its prior
Monte Carlo.

The priors are assigned directly to the unconstrained variables, so no
Jacobian is added to their normal densities in those coordinates. Their
induced densities on `(rho,q,r,lambda)` contain the usual inverse-transform
Jacobians. `q_j` means a constant innovation variance for DLNS and the
stationary *arithmetic mean* innovation variance for SVNS.

## Stochastic volatility

For each factor, `h_t = log(q) - v/2 + sqrt(v) u_t`,
`u_1 = z_1`, `u_t = phi*u_(t-1) + sqrt(1-phi^2)*z_t`,
and independent `z_t ~ N(0,1)`. Thus `h` has stationary variance `v` and
`E(exp(h_t))=q` before numerical truncation. Priors are
`logit(phi) ~ N(logit(0.95),1.25^2)` and `log(v) ~ N(log(0.5),1)`.
The shared-parameter priors are the same as DLNS.

## Algorithm

The Kalman likelihood exactly integrates the complete latent factor path for
each parameter and volatility path. DLNS uses symmetric random-walk
Metropolis blocks in the unconstrained coordinates; scales adapt only during
warmup. A second run adds a fixed symmetric correlated random walk for
`(log(q),log(r),decay-coordinate)` estimated from the first exploratory
training-only run. It is fixed before sampling and does not change the target.
Four chains use independent seeds.

SVNS uses the same marginal likelihood and shared parameter blocks with
non-centered volatility innovations. Gaussian elliptical-slice updates each
factor's entire innovation path conditional on parameters; parameter blocks
hold these independent innovations fixed. This is exact MCMC for the stated
truncated model, not an approximate filter or variational posterior. The
Gaussian smoothing routine draws factor paths conditional on a parameter and
volatility draw; it is separate from the collapsed MCMC transition.

Diagonal measurement errors, diagonal factor persistence, the empirically
calibrated priors, positive persistence, initialization and the sampler are
adaptations. OLS residual correlations are not evidence that diagonal `R` is
true: OLS residuals obey `H' e_t = 0`, imposing five-dimensional residual
geometry and correlations by construction. A proper model-checking analysis
must separately examine dynamics and posterior predictive residuals.
