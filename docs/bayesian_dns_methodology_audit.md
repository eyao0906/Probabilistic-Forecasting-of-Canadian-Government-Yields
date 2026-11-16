# Bayesian DNS methodology audit

## Scope and accepted design

This audit concerns the current paper's M3 (Bayesian DNS with constant factor volatility) and M4 (Bayesian DNS with stochastic factor volatility). The accepted revised-prior archive and reported forecast results are held fixed. The sampler, estimation design, and numerical experiments were not changed or rerun for this audit.

The project uses eight weekly Canadian zero-coupon yields at maturities 0.25, 0.5, 1, 2, 5, 7, 10, and 30 years. A grid search in the initial 2000–2015 window selects the fixed decay parameter \(\lambda=0.5160377358490565\). At each date, the three factors are extracted by cross-sectional least squares using the unchanged Nelson–Siegel loading matrix. The resulting factors are then treated as observed inputs to separate second-stage Bayesian factor regressions. The first-stage extraction and \(\lambda\) uncertainty are not propagated.

This is not a Kalman-filter estimate of latent yield factors and not joint yield/factor estimation. Diebold, Li, and Yue (2007) provide a related multi-step precedent, but their second stage includes latent global factors. Hautsch and Yang's 2010 working-paper model instead uses joint latent-state MCMC and a Kalman filter/simulation smoother. Those distinctions are made explicit in the manuscript.

## Exact likelihood and posterior targets

For a fixed factor \(j\), write \(x_t=\tilde f_{j,t}\), condition on the first extracted factor \(x_1\), let \(n=T-1\), and define

$$
r_t(\mu,b)=x_t-(1-b)\mu-bx_{t-1},\qquad t=2,\ldots,T.
$$

The factor likelihood is the exact Gaussian AR likelihood

$$
L_j(\mu,b,m;x)
=\prod_{t=2}^{T}
\frac{1}{\sqrt{2\pi e^m}}
\exp\!\left[-\frac{r_t(\mu,b)^2}{2e^m}\right].
$$

The coefficient prior is specified on \((\mu,b)\):

$$
p(\mu,b,m)
=N(\mu;\bar f_{j,\mathrm{train}},(2s_{j,\mathrm{train}})^2)\,
TN_{(-1,1)}(b;\hat b_{j,\mathrm{OLS}},0.03^2)\,
N(m;\mu_{0j},1),
$$

where \(TN\) is a normalized truncated normal density, and \(\mu_{0j}\) is the initial-training OLS residual log mean square defined in the paper. The intercept is \(a=(1-b)\mu\), not an independently assigned parameter. Thus the exact M3 target is

$$
\pi_{3j}(\mu,b,m\mid x)
\propto L_j(\mu,b,m;x)\,p(\mu,b,m).
$$

For M4, index the \(n\) latent log-volatility values by the \(n\) innovations, so \(h_1\) corresponds to the first transition after the conditioned factor \(x_1\). The prior densities are

$$
\begin{aligned}
p(\phi)&=
\frac{\left((1+\phi)/2\right)^{19}
\left((1-\phi)/2\right)^{0.5}}
{2\,\mathrm B(20,1.5)}\,1_{(-1,1)}(\phi),\\
p_U(v)&=\frac{e^{-v}}{1-e^{-U}}\,1_{(0,U)}(v),\qquad U=4,\\
\omega&=m-v/2,\qquad q=v(1-\phi^2),\\
h_1\mid m,v&\sim N(\omega,v),\\
h_t\mid h_{t-1},m,\phi,v&\sim
N\!\left(\omega+\phi(h_{t-1}-\omega),q\right),\quad t=2,\ldots,n.
\end{aligned}
$$

The stationary path density is

$$
p(h\mid m,\phi,v)
=N(h_1;\omega,v)
\prod_{t=2}^{n}N\!\left(h_t;\omega+\phi(h_{t-1}-\omega),q\right).
$$

The exact M4 target is

$$
\pi_{4j}(\mu,b,m,\phi,v,h\mid x)
\propto L_j(\mu,b,h;x)\,
p(\mu,b)\,p(m)\,p(\phi)\,p_U(v)\,p(h\mid m,\phi,v),
$$

where

$$
L_j(\mu,b,h;x)
=\prod_{s=1}^{n}\frac{1}{\sqrt{2\pi e^{h_s}}}
\exp\!\left[-\frac{r_{s+1}(\mu,b)^2}{2e^{h_s}}\right].
$$

There is no yield-level likelihood in either target after extraction, no prior on \(\lambda\), and no prior or MCMC update for \(R_t\). The posterior factorizes by \(j\) because both the factor likelihood and its priors are specified separately for each factor.

## Transformations and Jacobians

The sampler updates \((\mu,b)\) directly and computes \(a=(1-b)\mu\). No Jacobian is needed for this parameterization because the prior is defined directly on \((\mu,b)\). If the same density is expressed in \((a,b)\), then

$$
p(a,b)=p_\mu\!\left(\frac{a}{1-b}\right)p_b(b)\frac{1}{|1-b|},
\qquad -1<b<1.
$$

For the stochastic-volatility target, the sampler uses \(u=\operatorname{atanh}(\phi)\) and \(s=\log v\). The log-Jacobian from \((u,s)\) to \((\phi,v)\) is

$$
\log\left|\frac{\partial(\phi,v)}{\partial(u,s)}\right|
=\log(1-\phi^2)+s.
$$

The code includes these terms in the transformed target. The physical-coordinate priors on \(m,\phi,v\) are independent; \(\omega\) and \(q\) are derived and are not independently assigned priors.

## Sampling and target fidelity

The constant-volatility chain samples \(\mu\) and \(b\) from their Gaussian/truncated-normal full conditionals and updates \(m\) with an exact univariate slice step. The stochastic-volatility chain updates \(\mu,b\) conditionally, uses a ten-component Gaussian mixture only to construct volatility-path proposals, applies an in-chain Metropolis correction against the exact log-\(\chi^2\) likelihood, samples \(m\) from its Gaussian conditional, updates \(\phi\) in \(u=\operatorname{atanh}(\phi)\), and updates \(v\) with a non-centred slice step in \(s=\log v\). The mixture approximation is therefore a proposal mechanism, not the posterior likelihood. This differs from Omori et al.'s author manuscript, which describes final importance reweighting.

The focused tests already in the repository compare centered, collapsed, and non-centred target calculations with dense reference calculations, check the Jacobians and stationary path density, and test detailed balance of the mixture-based proposal. This audit did not modify those tests or execute the MCMC.

## Three distinct covariance restrictions

| Object | Role | Project restriction |
|---|---|---|
| \(B=\operatorname{diag}(b_L,b_S,b_C)\) | Factor conditional-mean dynamics | Each factor has its own AR(1) persistence; cross-factor lag effects are excluded. |
| \(Q_t=\operatorname{diag}(e^{h_{L,t}},e^{h_{S,t}},e^{h_{C,t}})\) | Conditional factor-innovation covariance | Factor innovations are independent conditional on their volatility states; posterior fits are factor-wise. |
| \(R_t=\operatorname{diag}(r_{1,t},\ldots,r_{8,t})\) | Yield reconstruction-error covariance in forecasts | Each maturity receives a plug-in expanding-window mean squared residual with floor \(10^{-6}\) squared percentage points; errors are assumed conditionally independent across maturities. |

The diagonal \(R_t\) statement is an actual Gaussian working assumption, not merely a computational choice that leaves residual dependence unspecified. It is not estimated in MCMC and is not part of \(\pi_{3j}\) or \(\pi_{4j}\). In the initial 835-observation training sample, the mean absolute pairwise correlation among the eight cross-sectional reconstruction-residual series is 0.659611; the maximum is 0.988809 for the 3-month and 1-year maturities. The diagonal covariance therefore omits material empirical residual dependence. It is retained unchanged for the accepted experiment, but the paper now identifies it as a limitation rather than implying the residuals are uncorrelated in fact.

The predictive yield covariance is \(\Lambda\operatorname{diag}(V_f)\Lambda'+R_t\). Posterior factor means and variances are computed separately and then projected through the common loading matrix. This construction induces yield co-movement through the shared DNS loadings, but not through correlated factor innovations or off-diagonal residual covariance.

## Moment and posterior-propriety conclusions

Under the stationary parameterization,

$$
E[e^{k h_{j,t}}\mid m_j,v_j]
=e^{k m_j+(k^2-k)v_j/2}.
$$

For \(k=1\), this is \(e^{m_j}\), independent of \(v_j\). Therefore an untruncated rate-one exponential prior on \(v_j\) would still give a finite stationary prior-predictive factor-innovation variance and finite finite-horizon factor second moments. For \(k=2\), the rate-one integral is at its divergence boundary, so the prior-predictive innovation fourth moment is infinite. Thus \(U=4\) is not mathematically necessary for the prior-predictive second moment under the \(m\) parameterization.

For the implemented bounded prior, the manuscript's likelihood bound and finite-path Gaussian moment-generating-function argument establish a proper posterior and finite finite-horizon predictive moments. The bound is a sufficient condition for that argument and also supplies all finite exponential moments. It has not been shown necessary for posterior propriety or for posterior-predictive second moments under an unbounded \(v\) prior; the prior-predictive calculation alone cannot establish likelihood-weighted posterior tail behavior. The paper therefore retains \(U=4\) provisionally and does not claim robustness above four. The existing all-origin posterior audit shows no sustained mass near the boundary but cannot reveal posterior behavior outside the truncated support.

Because the prior on \(b\) has support arbitrarily close to one, each fixed-draw stationary factor variance can be finite while its prior mixture over \(b\) has an infinite infinite-horizon mean proportional to \(1/(1-b^2)\). The paper correctly limits its finite-moment claim to finite forecast horizons.

## Changes integrated into the manuscript

- The classical scalar DNS yield representation now precedes the matrix equation, with factor labels explicitly described as loading shapes rather than formal economic identification.
- The literature discussion distinguishes the DLY multi-step precedent, the H&Y joint latent-state SVNS method, and this project's fixed-loading two-step procedure.
- The \(\mathrm{Beta}(20,1.5)\) rationale is split accurately between KSC's stationarity/near-unit-root discussion and Omori et al.'s high-persistence interpretation. The paper states the weekly half-life implication and makes no Canadian-yield calibration claim.
- The diagonal reconstruction covariance is described as a conditional-independence working assumption; the initial-window residual-correlation diagnostic is reported as evidence about that approximation.
- Appendix formulas specify the exact M3 and M4 targets, the stationary initial state, the transformed-coordinate Jacobians, and the difference between the mixture proposal and exact likelihood.

All changes are explanatory. Accepted posterior draws, forecast results, model priors, samplers, and experiment outputs remain unchanged.

