# Implementation validation

Run `python -m unittest discover -s tests\hy_canada -v` from the worktree.
Focused tests compare diagonal measurement and full-Q conditionals,
decay probabilities, Gaussian factor smoother moments, the scalar SV filter
and smoother, mixture probabilities, Student-t acceptance ratios and
Gaussian correction ratios against independent formulas. One-state corrected
mixture kernels with fixed and unknown volatility means are compared with
numerical quadrature. Weighted factor-dynamics conditionals, fixed-seed short chains,
bad data rejection and full-Q one-step forecast moments are also checked.

The standalone Kalman-likelihood test uses its documented pre-observation
initial-state convention; the blocked smoother instead takes `f_1` as the
first-state prior. The test for the latter checks its dense conditional
Gaussian moments, constant-Q equivalence with the time-varying-Q wrapper,
and independently calculated moments with two different transition Qs.
These checks do not prove the joint SV correction or the full sampler
converged. A separate importance-weight ESS and SV predictive simulator have
not been implemented.
