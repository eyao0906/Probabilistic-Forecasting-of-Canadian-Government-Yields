# Bayesian DNS prior audit

## Scope and design

This audit examines the Bayesian factor-dynamics priors using only the accepted initial training design and saved accepted posterior draws. It does not choose priors from 2016–2025 forecast scores. The accepted analysis remains intact; revised-prior outputs are kept separately in `results/bayesian_prior_revised_20261114/`.

The training sample contains 835 weekly observations from 2000-01-01 through 2015-12-31. The fixed Nelson–Siegel decay parameter is 0.5160377358490565. The training input is identified by SHA-256 `0b2af5c3c7bb261973ea4ec04c151eedbf7cf1428f22d0e08eaae87a212bfd77`. The presentation labels are M3 for internal M4 (constant volatility) and M4 for internal M5 (stochastic volatility); the revised runner retains the internal labels.

The audit script `src/audit_bayesian_priors.py` checks the initial-sample AR fits and compares the old prior with accepted M5 draws at 2025-11-28. The simulations in `src/prior_predictive_bayesian_dns.py` and `src/prior_predictive_longrun_mean.py` use 100,000 prior draws per model, the 2015-12-31 factor state, fixed training-only loadings, and the same diagonal reconstruction-error model. These are calibration diagnostics, not posterior forecasts.

## Why the original factor priors were changed

The original coefficient prior is independent across intercept and persistence:

\[
a_j\sim N(0,25),\qquad b_j\sim N(0.7,0.3^2)\ \text{truncated to }(-1,1).
\]

For a stable AR(1), the implied long-run factor mean is \(a_j/(1-b_j)\). The persistence density is positive arbitrarily close to one, so this ratio has neither a finite absolute first moment nor a finite second moment under the independent prior. This is not a failure of stationarity for any individual draw; it is a failure of the prior to control the distribution of long-run means. In 100,000 training-based draws, the implied long-run mean lay outside the observed training range with probabilities 88.9% for level, 83.8% for slope, and 71.9% for curvature.

The OLS persistence estimates are 0.997223 (level), 0.995512 (slope), and 0.976815 (curvature). These high estimates make the denominator \(1-b_j\) especially consequential. The OLS-implied long-run means are 2.936, -2.404, and -1.445, compared with training factor means 4.330, -2.175, and -1.110. The OLS intercept/persistence ratio is therefore not used as the prior center for the long-run mean.

The original log expected-variance prior \(m_j\sim N(\mu_{0j},25)\) is also too diffuse for the factor scale. Its median innovation variance is near the training residual-variance estimate, but \(E[\exp(m_j)]=\exp(\mu_{0j}+12.5)\) is dominated by an extreme upper tail: the prior means are about 2,021, 4,349, and 37,429 squared percentage points for level, slope, and curvature, respectively. Under that prior, the median prior-predictive joint yield-curve RMS change is about 4.6–4.7 percentage points at one week and 10.3–10.4 points at four weeks. The corresponding 95th percentiles in the training data are only 0.1845 and 0.3849 points, and over 99.9% of simulated changes exceed those training thresholds. This behavior is implausible for the intended short-horizon factor innovation model.

The saved posterior at the accepted 2025-11-28 origin is informative but limited evidence about the old prior. Across 24,000 M5 draws per factor, the \(b_j\) posterior medians are 0.9964, 0.9952, and 0.9814; the \(m_j\) medians are -4.707, -4.193, and -2.005. Persistence shifts substantially above the old prior median, while the \(m_j\) medians remain near their training-calibrated centers. This is one accepted origin under the old prior, not evidence that the prior is data-dominated across the rolling sample.

## Revised training-calibrated priors

The revised implementation places the prior directly on the long-run factor mean \(\mu_j\), and derives the intercept:

\[
\mu_j\sim N(\bar f_{j,\mathrm{train}},(2s_{j,\mathrm{train}})^2),\qquad
b_j\sim N(\hat b_{j,\mathrm{OLS}},0.03^2)\ \text{truncated to }(-1,1),\qquad
a_j=(1-b_j)\mu_j.
\]

Here \(\bar f_{j,\mathrm{train}}\) and \(s_{j,\mathrm{train}}\) are the mean and sample standard deviation of the initial 835 extracted factors. The factor-mean prior is deliberately broad—two training standard deviations—while having finite moments. The persistence prior is centered on the corresponding training-only OLS slope, with a scale narrow enough to reflect the observed near-unit-root persistence without using evaluation-period scores. The direct parameterization also permits exact Gaussian Gibbs updates for \(\mu_j\) and truncated-Gaussian updates for \(b_j\).

| Factor | Training mean | Training SD | OLS persistence center | \(\mu_{0j}\) |
|---|---:|---:|---:|---:|
| Level | 4.329855 | 1.055927 | 0.997223 | -4.888703 |
| Slope | -2.174505 | 1.406094 | 0.995512 | -4.122200 |
| Curvature | -1.109836 | 1.824830 | 0.976815 | -1.969807 |

The variance calibration is held fixed across expanding windows. For the initial-sample OLS residuals
\[
\hat e_{j,t}=\tilde f_{j,t}-\hat\alpha_j^{\mathrm{OLS}}-\hat\beta_j^{\mathrm{OLS}}\tilde f_{j,t-1},
\]
the code sets
\[
\mu_{0j}=\log\left(\frac{1}{834}\sum_{t=2}^{835}\hat e_{j,t}^{\,2}\right).
\]
Thus \(\mu_{0j}\) is the log of the initial-sample OLS residual mean square; its exponentials are 0.007531, 0.016209, and 0.139484 squared percentage points. The initial data calibrate these hyperparameters and also enter the likelihood, so this is an empirical-Bayes choice, not independent prior elicitation.

For both models, the revised prior is \(m_j\sim N(\mu_{0j},1)\). This retains the training residual scale as the median innovation-variance scale while reducing the log-scale SD from five to one. The stochastic-volatility priors \( (\phi_j+1)/2\sim\mathrm{Beta}(20,1.5)\) and \(v_j\sim\mathrm{Exp}(1)\) truncated to \((0,4)\) are unchanged. The earlier \(U=2\) and \(U=8\) sensitivity fits used the old mean and \(m\) priors; they are not sensitivity evidence for this revised specification, and no revised-prior bound sensitivity has yet been run.

In the stochastic-volatility model, \(\omega_j\) is the unconditional mean of log variance and \(m_j=\log E[\exp(h_{j,t})]\) is the log long-run expected innovation variance. With stationary log-volatility variance \(v_j\),
\[
h_{j,t}\sim N(\omega_j,v_j),\qquad m_j=\omega_j+v_j/2.
\]
The sampler therefore uses \(\omega_j=m_j-v_j/2\) and innovation variance \(v_j(1-\phi_j^2)\). The constant-volatility model uses \(m_j\) directly as log innovation variance.

## Revised prior-predictive checks

The direct-mean candidate was evaluated with 100,000 draws per model and no numerical failures. Its implied long-run mean falls outside the observed factor range in 33.9% of level draws, 32.1–32.3% of slope draws, and 23.4–23.9% of curvature draws. This remains a broad prior, but it no longer has divergent long-run-mean moments and is substantially more consistent with the training factor scale than the original independent prior.

| Model | Horizon | Median curve-change RMS | Training 95th percentile | Probability above training 95th percentile |
|---|---:|---:|---:|---:|
| M3, constant volatility | 1 week | 0.1741 | 0.1845 | 45.5% |
| M3, constant volatility | 4 weeks | 0.3645 | 0.3849 | 47.1% |
| M4, stochastic volatility | 1 week | 0.1673 | 0.1845 | 42.4% |
| M4, stochastic volatility | 4 weeks | 0.3567 | 0.3849 | 45.9% |

Amounts are percentage points. Candidate 95th-percentile changes are broader than the empirical training 95th percentiles, as a prior predictive distribution should allow, but their medians are on the observed scale rather than orders of magnitude larger. The 100,000-draw audit recorded no overflow, underflow, or nonfinite values. These checks do not establish that the revised prior is optimal or that all tail behavior is realistic.

## Mathematical and computational qualifications

The proper Gaussian prior on \(\mu_j\), together with \(|b_j|<1\), gives finite moments for the long-run mean and for the induced intercept \(a_j=(1-b_j)\mu_j\). The bounded prior on stationary log-volatility variance supports finite posterior-predictive moments at finite horizons. Since the persistence support still extends arbitrarily close to one, this does not establish a finite prior expectation of the infinite-horizon stationary factor variance, which contains \(1/(1-b_j^2)\).

One fixed synthetic M5 recovery series passed the stated sampling and mathematical diagnostics. Its 95% intervals contained the generating values for \(b,m,\phi,v\), but missed for \(a\) and \(\mu\). The misses are retained rather than used to tune the prior. One synthetic series is a sampler diagnostic, not repeated-sampling validation.

## Revised estimation status

The revised sampler is isolated in `src/bayesian_prior_longrun.py`; `src/bayesian_prior_longrun_experiment.py` guards the representative fits and the 130-origin run. The synthetic check and focused tests pass. All six representative fits passed the parameter, convergence, and predictive-stability gates: maximum rank-normalized \(\widehat R=1.0093\), minimum bulk ESS 960.7, and minimum tail ESS 1,136.9. The one-series synthetic misses for \(a\) and \(\mu\) remain disclosed. The phase-2 gate is accepted.

The rolling experiment completed all 130 origins with no failures; `summary.json` reports acceptance and the independent 130-origin archive audit in `analysis/summary.json` passed. Maximum rank-normalized \(\widehat R=1.00965\), minimum bulk ESS 1,057.6 and minimum tail ESS 2,094.5. The audit reconstructs DSS and intervals at every origin; independent Energy Score reconstruction was additionally run at 2016-01-08, 2020-03-27 and 2025-11-28, passing for both models and horizons.

At four weeks, paper M4 versus M3 changes mean DSS from −21.4542 to −21.8347 and Energy Score from 0.417116 to 0.406639 (a 2.51% reduction). The paired DSS difference is −0.38054 (95% interval [−0.73227, −0.05636]); the Energy Score difference is −0.01048 ([−0.01756, −0.00306]). Pooled RMSE changes by −0.1297 basis points (interval [−0.4069, 0.1453]). The random walk remains descriptively better on aggregate DSS, Energy Score and RMSE; both Bayesian models overcover pooled marginal 90% intervals. Alternative \(v\)-bound sensitivities have not been run with these revised priors.

The manuscript source and PDF now use the revised results. `src/render_revised_prior_results.py` generates the manuscript inputs under `paper/generated/revised_prior/` and `figures/finite/revised_prior/` without refitting. The 35-page PDF passes the no-write paper verifier; the KSC and Dawid–Sebastiani references appear in the rendered bibliography. The older baseline archives remain unchanged.

## Reproducible outputs

- Original prior audit and accepted-posterior comparison: `results/prior_justification_20261114/`.
- Direct long-run-mean prior-predictive simulation: `results/prior_justification_20261114/longrun_mean_candidate/`.
- Revised sampler and experiment runner: `src/bayesian_prior_longrun.py` and `src/bayesian_prior_longrun_experiment.py`.
- Fixed synthetic recovery output: `results/bayesian_prior_revised_20261114/synthetic/`.
- Focused sampler tests: `tests/test_bayesian_prior_longrun.py`.

The stochastic-volatility inference method also cites Kim, Shephard, and Chib (1998) for the Gaussian-mixture representation and Omori et al. (2007) for the exact-likelihood correction. Bibliographic metadata for the former were checked against Crossref and OpenAlex; its full text was not reviewed.
