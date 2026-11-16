# Mathematical specification

## Projection and covariance

Let \(Y_t\in\mathbb{R}^8\) be the eight-maturity yield vector and \(\Lambda\in\mathbb{R}^{8\times3}\) the accepted fixed Nelson–Siegel loading matrix. The cross-sectional least-squares factor estimate is

\[
\widehat f_t=(\Lambda^\top\Lambda)^-1\Lambda^\top Y_t,
\qquad r_t=Y_t-\Lambda\widehat f_t.
\]

The residual has \(\Lambda^\top r_t=0\), so it lies in a five-dimensional complement. A complete QR factorization provides \(Q_f\) and \(U\), with \(Q_f^\top Q_f=I_3\), \(U^\top U=I_5\), and \(U^\top\Lambda=0\). The residual coordinates are \(z_t=U^\top Y_t\), and the code verifies \(Y_t=\Lambda\widehat f_t+Uz_t\) on every training window.

For each origin, the primary covariance is centered Ledoit–Wolf on \(z_s\) using only observations through that origin. The unshrunk sample covariance and OAS covariance are predeclared sensitivity estimators. The predictive residual mean is fixed at zero. Factor–residual forecast cross-covariance is fixed at zero.

For either accepted Bayesian model, the corrected predictive moments are

\[
\mu^Y_{t+h}=\Lambda\mu^f_{t+h},\qquad
\Sigma^Y_{t+h}=\Lambda\operatorname{diag}(v^f_{t+h})\Lambda^\top
+U S_z U^\top.
\]

Here \(v^f\) is the accepted posterior predictive factor variance. The same saved posterior chains and the existing factor-scenario generator are used to produce 2,048 scenarios; only the old diagonal reconstruction noise is replaced by \(U z\), where \(z\sim N(0,S_z)\).

## Fixed experiment design

- Eight maturities: 0.25, 0.5, 1, 2, 5, 7, 10, 30 years; \(\lambda=0.516037735849057\).
- 130 expanding-window origins, 2016-01-08 through 2025-11-28; horizons 1 and 4 weeks.
- Paper labels are M3 (internal M4, constant volatility) and M4 (internal M5, stochastic volatility).
- No MCMC, refitting, prior changes, OOS covariance fitting, or edits to accepted forecast files.
- The primary residual covariance is fit through the origin date; no target data enters its estimate.

## Positive definiteness and score verification

Because \([\Lambda\ U]\) is full rank and both \(\operatorname{diag}(v^f)\) and \(S_z\) are positive definite, the assembled covariance is positive definite. The run checks Cholesky factorization, symmetry, eigenvalues, condition numbers, log determinants, and Mahalanobis terms for every forecast. DSS uses Cholesky solves, not an explicit inverse. The orthogonal factor- and residual-space DSS components are checked to sum to the full corrected score.

## Results preview

| Horizon (weeks) | RW DSS | M3 / constant DSS | M4 / SV DSS | RW ES | M3 ES | M4 ES |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -41.223 | -30.153 | -30.297 | 0.1631 | 0.2522 | 0.2491 |
| 4 | -36.006 | -25.489 | -25.717 | 0.3511 | 0.4145 | 0.4053 |

Full record-level values and provenance are in `results\residual_subspace_experiment_20261116`. Source/data hashes are recorded in the experiment manifest.
