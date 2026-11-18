# Joint Canadian DNS model specification

The observed 835 by 8 panel is in percentage points; maturity is in years.
Both models use `y_t = Lambda(lambda) f_t + epsilon_t`, where
`epsilon_t ~ N(0, diag(r))`, and
`f_t = mu + diag(rho)(f_{t-1} - mu) + eta_t`.
The sampled persistence parameters are restricted to `(-1,1)`.
The first state has the independent proper prior
`f_1 ~ N(m0,P0)`; transitions and SV paths begin at `f_1 -> f_2`.

DLNS uses `eta_t ~ N(0,Q)` with a full positive definite 3 by 3 Q.
SVNS uses `Q_t = diag(exp(h_t))`; each h component has stationary Gaussian
AR(1) dynamics with mean `a`, persistence `phi` and innovation SD `sigma`.
The different restrictions on Q mean DLNS/SVNS are not a pure
constant-versus-stochastic-volatility comparison.

## Priors and initialization

All empirical-Bayes values below are calibrated solely on the 2000-2015
training data, using OLS factors at decay 0.7 per year. They are pragmatic
research choices, not validated prior knowledge. Let `m`, `s`, `R0`, and
`S` be respectively the sample factor means, SDs (floored at 0.1), residual
variances (floored at 1e-5), and the factor AR residual covariance with
1e-5 added to its diagonal. The AR estimates used for S are clipped to
[0.7,0.995].

* `m0=m`, `P0=diag((2s)^2)`, independently of transition parameters.
* Independent `mu_j ~ N(m_j,(2s_j)^2)` and
  `rho_j ~ N(0.95,0.1^2)` truncated to (-1,1).
* `r_j ~ InvGamma(2.5, 1.5 R0_j)` using the shape/scale convention.
* DLNS `Q ~ InvWishart(df=5, scale=S)`; this is proper in dimension
  three and has prior mean S.
* Decay is discrete-uniform on a declared grid in [0.2,1.2] per year.
  The working paper's monthly [0.055,0.08] converts to [0.66,0.96]
  per year. The broader Canadian grid is a judgmental sensitivity choice;
  it does not establish that this prior is appropriate.
* For each SV factor, `(a, logit(phi), log(sigma))` has independent normal
  priors centered at `(log(S_jj), logit(0.95), log(0.2))`, with SDs
  `(2,1.25,1)`. Numerical support is restricted to `|logit(phi)|<=25`
  and `|log(sigma)|<=15`; thus `phi` is positive and stationary. These
  transformed-scale priors and bounds are adaptations, not the paper's
  exact priors. The prior can imply very persistent and volatile h paths;
  no prior predictive adequacy has been established.

The main text of Hautsch and Yang imposes diagonal measurement covariance;
its Appendix A gives a full Wishart precision update. They are not equivalent.
This implementation chooses the main-text diagonal restriction, with proper
independent inverse-gamma updates, rather than a full Wishart on eight
measurements. The paper reports DLNS factor-innovation correlations, so
the Canadian constant-Q model retains full Q, but the paper does not fully
specify a corresponding full-Q update. The inverse-Wishart update above is
an explicit adaptation. There is no claim of exact replication.
