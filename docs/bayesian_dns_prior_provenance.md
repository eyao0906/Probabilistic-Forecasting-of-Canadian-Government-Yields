# Bayesian DNS prior provenance and calibration

## Bottom line

The accepted paper compares Bayesian DNS M3 (constant factor volatility) with M4 (stochastic factor volatility). The priors are not all literature-derived: the persistence prior for log volatility adopts a published Beta family, while the mean, factor-persistence, log-variance-scale, and finite-\(v\) choices are project-specific empirical-Bayes or regularization decisions. Their numerical calibration uses the initial 2000–2015 training sample and training-only diagnostics. The revised numerical values were introduced after the earlier specification had been evaluated; evaluation-period scores were not used to select them.

The current manuscript's public M3/M4 labels map to internal saved-run labels M4/M5. The accepted archive was not refit or modified for this review.

## Prior specification and provenance

For factor \(j\), let \(\tilde f_{j,t}\) be the factor extracted by cross-sectional least squares using the fixed Nelson–Siegel loading matrix. The factor transition is

$$
\tilde f_{j,t}=a_j+b_j\tilde f_{j,t-1}+\eta_{j,t},
\qquad a_j=(1-b_j)\mu_j .
$$

The priors are specified directly on \(\mu_j\) and \(b_j\), not independently on \(a_j\) and \(b_j\). For M4, \(\eta_{j,t}\mid h_{j,t}\sim N(0,e^{h_{j,t}})\), with \(m_j=\log E(e^{h_{j,t}})\) under stationary volatility.

| Quantity | Family and why | Support/parameterization and why | Exact numerical choice, implications, and evidence or limitation |
|---|---|---|---|
| \(\mu_j\) | Normal prior on the long-run factor mean; the direct mean has finite moments when \(b_j\) approaches one. | Real support; \(a_j=(1-b_j)\mu_j\), rather than independent priors on \(a_j,b_j\), avoids the heavy-tailed ratio \(a_j/(1-b_j)\). | \(\mu_j\sim N(\bar f_{j,\mathrm{train}},(2s_{j,\mathrm{train}})^2)\). Centers are 4.329855, \(-2.174505\), \(-1.109836\); prior SDs 2.111854, 2.812188, 3.649660. Multiplier 2 is a project-specific broad-scale choice, not literature calibrated. Its central 90% interval is \(\pm3.29\) training SDs. The 100,000-draw revised joint audit put the mean outside the observed range in 23–34% of draws versus 72–89% under the original joint prior; it does not isolate multiplier 2. |
| \(b_j\) | Normal prior centered on an initial-window auxiliary OLS estimate. | Truncation to \((-1,1)\) ensures stationary AR draws; parameterized jointly with \(\mu_j\), with \(a_j\) derived. | \(b_j\sim N(\hat b_{j,\mathrm{OLS}},0.03^2)\) truncated to \((-1,1)\). Centers are 0.997223, 0.995512, 0.976815. SD 0.03 is a judgmental, project-specific scale. At this value, central 90% intervals are 0.924–0.998 and median half-lives 21.6–32.2 weeks (factor-specific values below). Training-only joint diagnostics do not isolate this SD. |
| \(m_j\) | Normal prior on log long-run expected innovation variance; a Gaussian log scale yields a positive lognormal variance. | Real support. Centered on the log OLS residual mean square from the initial 835 factors, then fixed across windows. | \(m_j\sim N(\mu_{0j},1)\), with centers \(-4.888703,-4.122200,-1.969807\) and variance medians 0.007531, 0.016209, 0.139484 squared percentage points. SD 1 gives prior mean/median ratio \(e^{1/2}=1.649\) and central 90% multiplicative range 0.193–5.180. SD 5 would give means 162,755 times larger than SD 1. Training-only yield-change checks support the revised scale but jointly vary prior components; they do not establish SD 1 as optimal. |
| \(\phi_j\) | Transformed Beta family adopted from univariate stochastic-volatility literature. Kim, Shephard, and Chib (1998) use daily Sterling/US-dollar returns (946 observations, 1981–1985); Omori et al. (2007) explain the high-persistence motivation and use daily TOPIX returns (1,232 observations, 1998–2002). | \((\phi_j+1)/2\) maps the Beta support to the stationary region \((-1,1)\). Both source applications are univariate return-volatility models, not yield-factor DNS. This study applies the prior to weekly factors without frequency adjustment. | \(\mathrm{Beta}(20,1.5)\) gives mean 0.860465, median 0.886488, and central 90% interval [0.648933, 0.982698]. At weekly frequency the mean implies a 4.61-week half-life and that interval implies 1.60–39.71 weeks. The sources do not establish these as appropriate for Canadian yields; no alternative Beta hyperparameters were tested. |
| \(v_j\) | Exponential prior favors smaller stationary log-volatility variance rather than assigning a flat density over a selected range. | \(0<v_j<U\), with \(q_j=v_j(1-\phi_j^2)\) and \(\omega_j=m_j-v_j/2\). Finite \(U\) is sufficient for the stated posterior-moment proof, but is not necessary for the stationary prior second moment; unbounded posterior tails remain unproved. | \(p_U(v)=e^{-v}/(1-e^{-U})\), rate 1 and \(U=4\), is project-specific regularization. At \(U=4\), mean, median, and 95th percentile are 0.925, 0.675, and 2.697. Prior-only rate comparisons and training-only \(U=2,4,8\) yield-movement checks are reported below. They do not prove the bound optimal or posterior robustness above four. |
| \(a_j,\omega_j,q_j,h_{j,1}\) | Not assigned independent priors. | Derived as \(a_j=(1-b_j)\mu_j\), \(\omega_j=m_j-v_j/2\), \(q_j=v_j(1-\phi_j^2)\); \(h_{j,1}\mid m_j,v_j\sim N(\omega_j,v_j)\) is the stationary initial-state distribution. | These are induced quantities, not additional hyperparameter choices. |

Priors on \((\mu_j,b_j,m_j,\phi_j,v_j)\) factorize within each factor, subject to the model-specific absence of \((\phi_j,v_j)\) in M3. The three factors are also fitted independently. The induced priors on \((a_j,b_j)\) and on \((\omega_j,q_j)\) are dependent.

## Factor-persistence implications

The truncated-normal prior is strongly concentrated at persistent values. The following are its actual quantiles after truncation to \((-1,1)\); half-life is \(\log(0.5)/\log(b)\) in weeks.

| Factor | \(b\) 5th percentile | Median \(b\) | \(b\) 95th percentile | Median half-life | Half-life range from the 5th–95th percentiles |
|---|---:|---:|---:|---:|---:|
| Level | 0.939343 | 0.978697 | 0.997977 | 32.19 | 11.08–342.37 weeks |
| Slope | 0.938168 | 0.978003 | 0.997882 | 31.16 | 10.86–326.96 weeks |
| Curvature | 0.923946 | 0.968443 | 0.996225 | 21.62 | 8.76–183.25 weeks |

Thus, SD 0.03 is not a generic weakly informative persistence prior. It retains considerable uncertainty, including very long half-lives, but concentrates the prior on highly persistent factors. This is a material modeling choice, not a result established by the literature.

## Prior-only comparison of project-specific scales

These calculations compare prior dispersion only; they do not refit the model or use evaluation-period scores. For the factor-mean multiplier \(k\), the normal prior's central 90% interval is \(\bar f_j\pm1.645k\,s_{j,\mathrm{train}}\): \(k=1,2,3\) therefore gives half-widths 1.645, 3.290, and 4.935 training SDs. The selected \(k=2\) is deliberately broad; there is no isolated training predictive comparison establishing it as uniquely preferred.

For the persistence SD alternatives, quantiles below are calculated from each factor's truncated-normal CDF using its training OLS center; the half-life column gives the union of factor-specific 5th–95th percentile half-life ranges.

| SD for \(b_j\) | Union of factor 5th-percentile \(b\) | Union of factor 95th-percentile \(b\) | Half-life range from factor 5th–95th percentiles |
|---:|---:|---:|---:|
| 0.015 | 0.952–0.969 | 0.995–0.999 | 14.0–635.4 weeks |
| 0.030 | 0.924–0.939 | 0.996–0.998 | 8.8–342.4 weeks |
| 0.060 | 0.866–0.881 | 0.995–0.996 | 4.8–177.4 weeks |

This sensitivity shows that 0.03 still allows extremely persistent draws; it does not identify a uniquely correct SD. For the log-variance SD \(s_m\), the prior mean/median ratio of \(\exp(m_j)\) is \(\exp(s_m^2/2)\) and its central 90% multiplicative interval is \([\exp(-1.645s_m),\exp(1.645s_m)]\). Values \(s_m=0.5,1,1.5\) give mean/median ratios 1.133, 1.649, 3.080 and intervals 0.439–2.276, 0.193–5.180, 0.085–11.790. The former value \(s_m=5\) gives a ratio of about 268,337; the ratio of old to revised prior means is \(\exp(12)\approx162,755\).

For \(v_j\), the following exact truncated-exponential summaries compare rates at the fixed bound \(U=4\); rate 1 is the accepted choice. The CDF and mean are \(F_U(v)=(1-e^{-\lambda v})/(1-e^{-\lambda U})\) and \(E_U[v]=1/\lambda-U/(e^{\lambda U}-1)\). The \(U\)-sensitivity is a separate prior-predictive calculation using the revised training-conditioned design.

| Exponential rate | Truncated prior mean | Median | 95th percentile | \(P(v_j>3)\) |
|---:|---:|---:|---:|---:|
| 0.5 | 1.374 | 1.132 | 3.446 | 10.15% |
| 1 | 0.925 | 0.675 | 2.697 | 3.21% |
| 2 | 0.499 | 0.346 | 1.495 | 0.21% |

The rate controls how strongly the prior regularizes toward small \(v_j\); no corresponding yield-movement simulation varying rate was run. In the existing 100,000-draw training-only movement audit, increasing \(U\) from 4 to 8 changes the four-week 99th percentile by about 0.1%, while roughly 46% of four-week draws exceed the training 95th-percentile movement. This supports weak sensitivity of the tested predictive quantiles to \(U\), not realistic tails or posterior robustness above the hard bound.

## Log-volatility persistence and weekly interpretation

The normalized density induced by the transformed Beta prior is

$$
p(\phi)=
\frac{
\left(\frac{1+\phi}{2}\right)^{19}
\left(\frac{1-\phi}{2}\right)^{0.5}
}{
2\,\mathrm B(20,1.5)
},\qquad -1<\phi<1.
$$

Its mean is \(2(20/21.5)-1=0.860465\), and its median is 0.886488. Applied to weekly transitions, the mean corresponds to a half-life of about 4.61 weeks; the median corresponds to 5.75 weeks, and the central 90% interval implies half-lives of about 1.60–39.71 weeks. These are weekly implications of the borrowed prior, not evidence that the underlying Canadian volatility has those half-lives. Kim et al. motivate support on the stationary region and discuss the difficulty of a flat prior near nonstationarity and the nonidentification of the volatility mean at \(\phi=1\). Omori et al. describe the same numerical prior as reflecting high persistence in earlier empirical SV work. Neither source calibrates it to weekly Canadian DNS factors.

## Log-variance scale and prior-predictive evidence

The center is determined by the initial-window auxiliary OLS regression:

$$
(\hat\alpha_j,\hat\beta_j)
=\arg\min_{\alpha,\beta}\sum_{t=2}^{835}
(\tilde f_{j,t}-\alpha-\beta\tilde f_{j,t-1})^2,\qquad
\hat e_{j,t}=\tilde f_{j,t}-\hat\alpha_j-\hat\beta_j\tilde f_{j,t-1},
$$

$$
\mu_{0j}=\log\left(\frac{1}{834}\sum_{t=2}^{835}\hat e_{j,t}^{\,2}\right).
$$

The log-variance prior SD of one leaves the median innovation variance at \(\exp(\mu_{0j})\). Its prior mean is \(\exp(\mu_{0j}+1/2)\), or 0.0124, 0.0267, and 0.2300 for level, slope, and curvature. A central 90% interval for \(\exp(m_j)\) spans 0.19–5.18 times its median. Under the earlier SD of five, the corresponding means would be about 2,021, 4,349, and 37,429; changing the SD from five to one reduces these means by \(\exp(12)\), about 162,755-fold.

The accepted 100,000-draw training-only audit of the full revised prior gave median one-week curve-change RMS values of 0.167–0.174 percentage points and four-week values of 0.357–0.365, compared with training 95th-percentile thresholds 0.1845 and 0.3849. Under the original specification, medians were 4.63–4.70 and 10.32–10.44, and more than 99.9% of simulated changes exceeded those thresholds. These are joint prior revisions; the isolated lognormal comparison above is the evidence specific to changing the \(m_j\) SD. The diagnostics support keeping SD one on the observed scale, but do not show it is optimal or make the prior weak.

## What the \(v_j\) bound establishes

At stationarity,

$$
h_{j,t}\mid m_j,v_j\sim N(m_j-v_j/2,v_j),\qquad
E[e^{k h_{j,t}}\mid m_j,v_j]
=e^{k m_j+(k^2-k)v_j/2}.
$$

For \(k=1\), this expectation is \(e^{m_j}\), independent of \(v_j\). Consequently an untruncated \(v_j\sim\mathrm{Exp}(1)\) would still give finite stationary prior-predictive second moments for factor innovations and finite-horizon factor forecasts: those variances require \(E[e^{h}]\), not \(E[e^{2h}]\). But for \(k=2\), integration against an untruncated rate-one exponential prior is at the divergent boundary, so the innovation fourth moment is infinite. This distinction means a finite \(U\) is not mathematically necessary for stationary prior-predictive second moments under the \(m_j\) parameterization, but it does provide all finite exponential moments uniformly.

The implemented \(U=4\) bound also gives a convenient sufficient bound for the existing posterior-propriety and finite-horizon moment argument. An unbounded prior's prior-predictive second moment does not by itself prove a finite posterior-predictive moment after conditioning on observed factors. That likelihood-weighted tail argument has not been established here; therefore the paper does not claim that \(U=4\) is mathematically necessary, nor that an unbounded posterior would retain finite moments. The earlier bound audit found no sustained posterior concentration near four, but a posterior supported below four cannot establish behavior above it. The defensible conclusion remains that \(U=4\) is retained provisionally, not proven optimal or robust to larger support.

The stationary log-volatility SD at the boundary is \(\sqrt{U}/2=1\) for \(\log\sigma_{j,t}\). Its central 95% range is 0.141–7.10 times the conditional median volatility. This is a boundary implication, not a typical range or a bound on realized yields.

## Empirical-Bayes and information-set qualifications

The initial-sample factors determine \(\bar f_{j,\mathrm{train}}\), \(s_{j,\mathrm{train}}\), \(\hat b_{j,\mathrm{OLS}}\), and \(\mu_{0j}\), and those same initial observations also enter the likelihood. The prior is therefore empirical Bayes conditional on frozen initial-window hyperparameters, not independent prior elicitation. The revised numerical choices followed evaluation of the earlier specification, but all selection diagnostics used training information; no 2016–2025 score was used to choose them. The priors are then held fixed across expanding-window origins.

The diagonal \(R_t\) used in yield prediction is not a Bayesian prior or part of the second-stage factor likelihood. It is a plug-in, maturity-specific mean squared cross-sectional reconstruction residual, with a floor of \(10^{-6}\) squared percentage points, shared by M3 and M4. This separate approximation is assessed in the methodology audit; it should not be conflated with the diagonal factor-persistence matrix or the diagonal factor-innovation covariance.

