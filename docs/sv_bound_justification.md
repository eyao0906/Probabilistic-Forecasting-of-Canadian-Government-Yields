# Training and posterior audit of the stochastic-volatility bound

## Result in brief

The revised-prior M4 model retains the rate-one exponential prior truncated to
\(0<v_j<4\). The bound is mathematically useful for establishing finite
finite-horizon posterior-predictive moments. The training-only auxiliary
volatility scales and all-origin posterior summaries do not show sustained
posterior concentration near four. Prior-predictive results are nearly
unchanged when the bound rises from four to eight, although their tails remain
substantially wider than historical training movements. These findings support
retaining four as a practical bound, not as an optimal or fully validated
choice.

No MCMC was rerun. The accepted 130-origin comparison and its forecasts,
checkpoints, scores, and validation gates were not modified. The posterior
diagnostic is conditional on \(v_j<4\); it cannot establish what would happen
to the posterior or forecasts if values above four were allowed.

## Interpretation of the bound

Under the stationary log-volatility model,
\[
h_{j,t}\sim N(\omega_j,v_j),\qquad \sigma_{j,t}=\exp(h_{j,t}/2),
\]
so
\[
\operatorname{SD}(\log\sigma_{j,t})=\frac{\sqrt{v_j}}{2}.
\]
At \(v_j=4\), this standard deviation is one. Conditional on a parameter value
at that boundary, the central 95% stationary range for innovation standard
deviation is \([\exp(-1.96),\exp(1.96)]\), or about 0.141–7.10 times its
conditional median. This is a boundary scenario, not a typical range.

The prior bounds stationary log-volatility variance; it does not bound the
realized state \(h_{j,t}\), innovation standard deviation, yield levels, or
weekly yield movements. Those remain unbounded. A larger \(U\) permits more
extreme regimes but does not make them routine. The finite upper bound is useful
for the finite-moment argument in [bayesian_finite_prior.md](bayesian_finite_prior.md);
that argument does not determine the numerical value of \(U\).

## Stage A: training-only prior predictive sensitivity

The audit simulated 100,000 four-week paths for each \(U\in\{2,4,8\}\), using
the revised priors for the long-run mean, persistence, \(m_j\), and \(\phi_j\).
Only \(U\) varied; the exponential rate remained one. Each run starts from the
last factor state and yield curve in the accepted 835-week 2000–2015 training
sample, uses the accepted decay parameter and loadings, and adds the common
training-estimated diagonal reconstruction error. The same seed and random
number design were used across bounds. Training thresholds use only observed
one- and four-week movements in those 835 curves. No 2016–2025 score was used.

The principal movement measure is the root mean square (RMS) change across all
eight maturities, in percentage points.

| \(U\) | Horizon | Predictive median | 90th pct. | 95th pct. | 99th pct. | Training 95th pct. | Training 99th pct. | Above training 95th / 99th |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 week | 0.169 | 0.346 | 0.425 | 0.621 | 0.184 | 0.274 | 43.1% / 19.1% |
| 4 | 1 week | 0.168 | 0.344 | 0.423 | 0.628 | 0.184 | 0.274 | 42.3% / 18.8% |
| 8 | 1 week | 0.167 | 0.343 | 0.423 | 0.629 | 0.184 | 0.274 | 42.1% / 18.7% |
| 2 | 4 weeks | 0.360 | 0.860 | 1.065 | 1.522 | 0.385 | 0.567 | 46.3% / 26.4% |
| 4 | 4 weeks | 0.357 | 0.859 | 1.063 | 1.533 | 0.385 | 0.567 | 45.9% / 26.2% |
| 8 | 4 weeks | 0.357 | 0.858 | 1.062 | 1.534 | 0.385 | 0.567 | 45.9% / 26.1% |

Increasing \(U\) from four to eight changes the four-week median by less than
0.2% and the 99th percentile by about 0.1%. Across all three bounds, the
one-week 95th percentile is about 0.423 percentage points, versus 0.184 in the
training sample; the four-week 95th percentile is about 1.063, versus 0.385.
About 13% of four-week simulations exceed the largest training RMS movement.
Thus, the prior predictive distribution remains broad. This is not a close fit
to the empirical movement distribution, even though its median is on the
training scale. The near-invariance across \(U\) indicates that changing this
bound alone has little effect on these central and 99th-percentile diagnostics;
it does not validate all prior components or the most extreme tails.

At \(U=4\), prior quantiles of stationary log-volatility variance \(v_j\) are
0.675, 2.697, and 3.571 at the 50th, 95th, and 99th percentiles. The corresponding
stationary innovation-standard-deviation quantiles (percentage points per
square-root week) are:

| Factor | Median | 95th pct. | 99th pct. |
|---|---:|---:|---:|
| Level | 0.072 | 0.210 | 0.332 |
| Slope | 0.105 | 0.306 | 0.483 |
| Curvature | 0.307 | 0.907 | 1.452 |

The full predictive quantiles, maturity-specific movements, and simulation
draws are retained in the results directory. The maturity table covers all
eight maturities; for example, at four weeks and \(U=4\), the 95th-percentile
absolute changes are 1.235, 1.144, 0.967, and 0.825 percentage points at 3
months, 2, 10, and 30 years, respectively. Their training-sample 95th
percentiles are 0.366, 0.470, 0.369, and 0.297.

### Training-only auxiliary scale check

For each factor, an OLS AR(1) was fitted to the 835 extracted training factors.
The audit computes the sample variance of \(\log(\hat\eta_{j,t}^2)\), then
subtracts \(\pi^2/2\), the Gaussian \(\log(\chi_1^2)\) noise variance, truncating
the result below at zero:
\[
\widehat v_j^{\,\mathrm{aux}}
=\max\{0,\widehat{\operatorname{Var}}[\log(\hat\eta_{j,t}^2)]-\pi^2/2\}.
\]

| Factor | Variance of log squared residual | Noise variance \(\pi^2/2\) | Rough auxiliary \(v_j\) |
|---|---:|---:|---:|
| Level | 4.849 | 4.935 | 0.000 |
| Slope | 5.582 | 4.935 | 0.647 |
| Curvature | 5.885 | 4.935 | 0.951 |

These are noisy auxiliary estimates, not formal Bayesian estimates or an
empirical upper confidence limit. They rely on Gaussian innovations and
stationary log volatility and provide only a rough historical scale. The direct
reconstruction differs slightly from the earlier rounded values in the
finite-prior note (especially for slope); the computations here use the
accepted training panel, its extracted factors, and the stated OLS residual
formula.

## Stage B: accepted posterior boundary audit

The audit read the accepted revised-prior M5 chains at all 130 forecast origins.
It verified the accepted run summary and plan, each origin's accepted
configuration, and the SHA-256 checksum of every M5 factor chain against its
saved forecast manifest. The audited chains contain 24,000 draws per
origin-factor: 1,560 chain files and 9,360,000 \(v_j\) draws in total. The
parameter is selected by its saved name `v`; neither latent state \(h_{j,t}\)
nor innovation variance \(q_j\) is used as a substitute.

| Factor | Median of origin medians | Median origin 95th pct. | Largest origin 99th pct. | Largest sampled \(v_j\) | Max. \(P(v>2)\) | Max. \(P(v>3)\) | Max. \(P(v>3.5)\) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Level | 0.338 | 0.536 | 0.814 | 3.707 | 0.033% | 0.013% | 0.008% |
| Slope | 0.645 | 0.854 | 1.108 | 2.058 | 0.004% | 0.000% | 0.000% |
| Curvature | 0.939 | 1.723 | 2.867 | 4.000 | 6.27% | 0.77% | 0.24% |

For comparison, the \(U=4\) truncated-exponential prior assigns probabilities
11.92%, 3.21%, and 1.21% to \(v_j>2\), \(v_j>3\), and \(v_j>3.5\),
respectively. No level or slope origin has more than 5% posterior probability
above two. Four curvature origins exceed that 5% mark; none exceeds 10%. No
factor-origin fit has more than 0.8% probability above three or more than 0.24%
above 3.5. Curvature is the least separated from the upper tail. Although the
largest individual curvature draw is 3.999993, the largest origin-specific
99th percentile is 2.867. The sample maximum is therefore not evidence of
posterior concentration at the boundary.

These summaries suggest that the bound is not obviously binding in the
accepted posterior. They cannot establish insensitivity to relaxing it: all
reported posterior draws were generated under the hard restriction \(v_j<4\).
The prior-predictive exercise varies the prior bound but is not a substitute
for fitting the less restricted posterior.

## Literature and decision

Gelman, Simpson, and Betancourt (2017) argue that priors should be assessed in
the context of the likelihood and the complete analysis, including their
implications for predictive observations. This motivates the training-only
prior-predictive check here; it does not recommend \(U=4\). Bibliographic
metadata were checked against Crossref, and the supplied paper PDF was reviewed.
No claim is made that this work, or the optional penalized-complexity-prior
literature, determines the bound's numerical value.

The evidence supports retaining \(U=4\) for the current model: the rough
training estimates are below four, posterior quantiles remain below the bound,
and raising the prior bound to eight barely changes the tested predictive
quantiles. This is a provisional modeling choice, not proof of optimality or
full-period forecasting-score robustness. Prior predictive tails remain wide,
and a sensitivity fit under a less restricted posterior would be needed to
measure posterior or forecast changes above four. That additional MCMC was not
judged necessary for the present revision because no sustained boundary
concentration appeared in the saved posterior summaries.

## Reproducibility

Run from the repository root, choosing an empty output directory:

```powershell
.\.venv\Scripts\python.exe -B -m src.audit_sv_variance_bound `
  --output results\sv_bound_justification_20261115_reproduction
```

The script refuses to overwrite a nonempty output directory. It uses 100,000
draws per bound and seed 20261115 by default. The metadata file records the input,
code, accepted-run, and plan hashes; the chain manifest records every posterior
chain hash. `prior_predictive_draws.npz` preserves all simulated yield changes,
including tail draws, without clipping or dropping them. No simulated
nonfinite values, overflow, or underflow were observed.
