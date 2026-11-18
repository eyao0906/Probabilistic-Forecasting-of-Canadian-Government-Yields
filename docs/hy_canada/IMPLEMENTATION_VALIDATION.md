# Implementation validation

Run from the worktree root using the project Python environment:

```
python -m unittest discover -s tests\hy_canada -v
```

Nine focused tests passed. They verify the loading limit and dimensions;
the exact Kalman log likelihood against an independently assembled joint
Gaussian covariance with time-varying transition variances; the first
transition and time indexing; backward simulation-smoother moments against
an analytical conditional Gaussian; stationary AR log-volatility moments and
non-centered recursion; one-step predictive moments; parameter transform
inverses and a numerical Jacobian; finite posterior targets and seeded chain
reproducibility. An elliptical-slice chain also reproduces an analytically
known Gaussian conditional mean and variance. Invalid duplicate dates,
missing weeks and nonfinite yields are rejected. The smoother test uses 6,000
Monte Carlo draws and the SV and
forecast moment tests use 12,000.

These checks do not prove global posterior exploration, validate all extreme
numerical inputs, establish long-chain stationary sampling, or establish that
the model fits real data. The elementary elliptical-slice check does not prove
mixing in the high-dimensional real-data SV posterior.
