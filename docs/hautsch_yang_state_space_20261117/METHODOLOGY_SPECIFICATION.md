# Hautsch–Yang state-space DNS: methodology specification

## Source version and scope

The source inspected for this implementation is the January 2010 SFB 649
Discussion Paper 2010-004 by Hautsch and Yang, *Bayesian Inference in a
Stochastic Volatility Nelson–Siegel Model*. The repository copy is
`docs/sources/literature_verified/hautsch_yang_wp2010.pdf`; the extracted
page text is `docs/sources/literature_verified/hautsch_yang_wp2010_pages.txt`.
The inspected PDF SHA-256 is
`48e7b580354a66fa874782982087c817125b93bd64acb5a5d6122ed4435e071a`.
The 2012 journal article is identified by DOI
10.1016/j.csda.2010.07.003, but its full text was not available for a
line-by-line comparison during this specification. Accordingly, equation
references below identify the inspected 2010 working-paper version and do
not imply that every detail was rechecked against the typeset 2012 article.

The working paper studies monthly U.S. zero-coupon yields at five maturities.
The Canadian application instead has eight weekly yields, in percentage
points, at maturities 0.25, 0.5, 1, 2, 5, 7, 10 and 30 years. All conversion
and prior changes below are explicit adaptations, not replications of that
empirical design.

## Joint observation and factor equations

For observed yields \(Y_t\in\mathbb R^8\), use the fixed Nelson–Siegel loading
matrix \(\Lambda(\lambda)\in\mathbb R^{8\times3}\) and latent factor state
\(f_t=(L_t,S_t,C_t)^\top\):

\[
Y_t=\Lambda(\lambda)f_t+\epsilon_t,\qquad
\epsilon_t\sim N_8(0,R).
\]

The row for maturity \(\tau\), measured in years, is
\[
\left(1,\frac{1-e^{-\lambda\tau}}{\lambda\tau},
\frac{1-e^{-\lambda\tau}}{\lambda\tau}-e^{-\lambda\tau}\right).
\]
The implementation evaluates \(1-e^{-x}\) with `-expm1(-x)` and defines the
continuous limit at zero maturity. The factor labels describe loading shapes;
they are not restrictions that formally identify economic shocks.

The transition follows the diagonal-factor restriction discussed after
equations (3)–(5) of the working paper:
\[
f_t=\mu+A(f_{t-1}-\mu)+\eta_t,\quad
A=\operatorname{diag}(b_L,b_S,b_C),\quad
\eta_t\mid Q_t\sim N_3(0,Q_t).
\]
Equivalently, the intercept is \((I-A)\mu\). The restriction excludes
cross-factor lag effects. It is not the unrestricted VAR written before the
paper's parsimony restriction.

For SS_M3, \(Q_t=Q=\operatorname{diag}(q_L,q_S,q_C)\). For SS_M4,
\[
Q_t=\operatorname{diag}(\exp(h_{L,t}),\exp(h_{S,t}),\exp(h_{C,t})),
\]
\[
h_{j,1}\sim N(\omega_j,v_j),\qquad
h_{j,t}\mid h_{j,t-1}\sim
N\{\omega_j+\phi_j(h_{j,t-1}-\omega_j),
v_j(1-\phi_j^2)\},\quad t\ge2.
\]
Thus \(h_{j,t}\) scales the factor innovation that generates \(f_t\), not
the measurement error at maturity \(i\). Volatility persistence is restricted
to \((-1,1)\). The latent-state time index is aligned with the transition into
the corresponding observed yield date.

The working paper's equations (7)–(10) express the same factor-specific
log-variance AR(1) structure. Its Section 3.1 blocks the latent yield-factor
path with a Kalman filter/simulation smoother inside MCMC; it does not replace
Bayesian estimation with a stand-alone Kalman likelihood estimate.

## Covariance and initialization adaptations

The working paper states that measurement errors are contemporaneously
uncorrelated below its equation (6), but Appendix A's displayed precision
update is Wishart with residual cross-products. That update is a full-matrix
inverse-Wishart update, not an update confined to a diagonal covariance.
The Canadian implementation will therefore use
\[
R=\operatorname{diag}(r_1,\ldots,r_8),\qquad r_i>0,
\]
with independent proper inverse-gamma priors and conjugate updates conditional
on the sampled factor path. This preserves the declared diagonal \(R\) rather
than silently implementing an unrestricted covariance. It is a deliberate
adaptation, not an exact reproduction of the appendix update.

The source Appendix A uses a diffuse factor initialization with covariance
\(10^6 I_3\) in its filter derivation; the paper's algorithm summary separately
initializes the factors at their long-run means. The independent implementation
will make a proper initial-state distribution explicit and use the diffuse
Gaussian convention only where specified and tested. The volatility path uses
its stationary initial distribution. Any departure from these choices will be
reported with the corresponding likelihood and predictive sensitivity.

## Decay parameter and priors

Appendix A of the working paper uses a uniform decay prior between 0.055 and
0.08 when maturity is measured in months. Since
\(\lambda_{\rm month}\tau_{\rm month}
=\lambda_{\rm year}\tau_{\rm year}\), this is 0.66–0.96 per year. The
Canadian training-only loading grid instead spans 0.05–2.0 per year and selects
\(\lambda=0.5160377358490565\). Directly copying the monthly support would
exclude the training-window optimum. The experiment will use a bounded
training-calibrated prior on the per-year scale and will assess concentration
and boundary behavior; fixed-\(\lambda\) fits are pilots only.

The matched SS_M3/SS_M4 prior candidate is calibrated only with the initial
2000–2015 training window. It places the same priors on factor long-run means,
factor persistence, long-run expected innovation variance, diagonal measurement
variances and decay in both models. In SS_M3, \(q_j=\exp(m_j)\). In SS_M4,
\(m_j=\omega_j+v_j/2\), so \(E[\exp(h_{j,t})\mid m_j,v_j]=\exp(m_j)\), matching
the constant model's innovation-variance interpretation. Candidate
factor-mean, persistence, \(m_j\), volatility-persistence and bounded-\(v_j\)
scales will be recorded with their training-only source and sensitivity limits
before out-of-sample scores are examined. The existing two-step audit is
calibration evidence, not proof that its priors are optimal for latent factors.

For diagonal \(R\), a concrete proper prior candidate is
\(r_i\sim\operatorname{InvGamma}(3,2r_{i,0})\), where \(r_{i,0}\) is the
initial-window DNS least-squares residual mean square. This has prior mean
\(r_{i,0}\) and coefficient of variation one. Its posterior conditional on
the latent factor path is
\[
r_i\mid Y,f,\lambda\sim\operatorname{InvGamma}
\left(3+\frac T2,\;2r_{i,0}
+\frac12\sum_{t=1}^T(Y_{i,t}-\Lambda_i f_t)^2\right).
\]
This is not the inverse-Wishart prior in the working paper; it is an explicit
proper diagonal-covariance adaptation.

## SV likelihood and blocked update

Equations (11)–(12) of the working paper transform squared factor innovations
using \(\log(\eta_{j,t}^2+c)\) and approximate the log-\(\chi^2_1\) disturbance
with a seven-component normal mixture attributed to Kim et al. (1998). The
paper uses this approximation to construct a volatility-path proposal and
then describes posterior reweighting to address the approximation. The
mixture density is not the exact SV likelihood.

The seven-component table (working-paper PDF pp. 10–11) has weights
\((0.00730,0.10556,0.00002,0.04395,0.34001,0.24566,0.25750)\), component
means \((-11.40039,-5.24321,-9.83726,1.50746,-0.65098,0.52478,-2.35859)\),
and variances \((5.79596,2.61369,5.17950,0.16735,0.64009,0.34023,1.26261)\).

The Canadian sampler must target the exact factor-innovation density
\[
\log p(\eta_{j,t}\mid h_{j,t})
=-\frac12\{h_{j,t}+\eta_{j,t}^2e^{-h_{j,t}}+\log(2\pi)\}.
\]
The implementation uses an elliptical-slice block transition under the exact
Gaussian AR(1) prior for the complete volatility path, with the exact factor
innovation likelihood as its slice target. This is a deliberate departure from
the working paper's seven-component proposal and posterior reweighting; no
mixture approximation enters the implemented posterior. Mixture draws without
correction will not be reported as exact posterior draws. The implementation
audit records the transition, target density, and independent checks.

Section 3.2 of the working paper checks posterior predictive skewness and
kurtosis using replicated yield residuals (equations (14)–(18)). Section 4
applies the model to monthly U.S. yields, reports parameter and MCMC
diagnostics, and compares one-step predictive uncertainty. These checks are
methodological precedent only: they do not establish calibration for the
weekly Canadian panel, eight maturities, or four-observation forecasts.

## Forecast and scientific scope

Posterior predictive scenarios must propagate shared posterior parameter and
latent-state draws, future factor innovations, future M4 volatility states,
and a new target-date measurement disturbance. Measurement noise is not added
at unobserved intermediate dates solely to form a target-date forecast.
Scenarios are eight-dimensional yield vectors in percentage points.

The full research question is empirical: a coherent joint posterior does not
by itself establish that the diagonal Gaussian observation model is adequate.
One-step whitened Kalman innovations, posterior measurement-error diagnostics,
the five-dimensional orthogonal residual subspace, multivariate calibration,
and predictive scores against the preserved benchmarks are required before
recommending replacement of the existing factor-level design.
