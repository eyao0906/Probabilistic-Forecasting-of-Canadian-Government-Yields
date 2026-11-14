# Finite-moment Bayesian M5: specification and mathematical audit

This note records the finite-moment stochastic-volatility specification used in the accepted baseline comparison, not a reinterpretation of the old inverse-gamma model. The later training-calibrated coefficient and log-variance prior candidate is documented in [the prior audit](bayesian_prior_justification.md); it changes the factor-mean/persistence prior and the scale of $m$, while retaining the bounded $v$ and transformed SV construction described here. Historical source, chains and diagnostics remain in their original directories. Mathematical validity does not imply that a numerical run has converged.

## 1. Model, training-only calibration, and proper priors

For one extracted factor, conditional on its first observed value:

\[
x_t=a+b x_{t-1}+\exp(h_t/2)\epsilon_t,\quad \epsilon_t\stackrel{iid}{\sim}N(0,1),
\]
\[
h_1\mid m,\phi,v\sim N(\omega,v),\qquad
h_t=\omega+\phi(h_{t-1}-\omega)+\sqrt{q}\eta_t,
\quad \eta_t\stackrel{iid}{\sim}N(0,1),
\]
\[
0<v<U,\quad |\phi|<1,\qquad q=v(1-\phi^2),\quad \omega=m-v/2.
\]

The two innovation sequences are independent. Thus \(E[\exp(h_t)\mid m,\phi,v]=\exp(m)\) in the stationary log-volatility process. This is an innovation-variance interpretation, not an assertion of a finite unconditional stationary variance of factor levels under the entire coefficient-prior mixture.

Use independent priors:

- The existing coefficient prior: independent unscaled Gaussian for a,b, truncated to |b|<1; its mean and covariance are unchanged and recorded in configurations.
- \(m\sim N(\mu_0,25)\), exactly the M4 constant-log-variance prior. Each factor's \(\mu_0\) is the existing residual-variance estimate from the original 835 observations, ending 2015-12-31. It is frozen at all expanding windows. This is the inherited empirical-Bayes calibration; the initial observations also remain in the likelihood, just as in M4. It is not an independently elicited prior.
- \(z_\phi=(\phi+1)/2\sim\mathrm{Beta}(20,1.5)\), unchanged.
- \(p_U(v)=\lambda\exp(-\lambda v)/(1-\exp(-\lambda U))\), for 0<v<U, with rate \(\lambda=1\).

Main U=4, predeclared sensitivity U=2 and U=8; do not choose among them using out-of-sample scores. The prior is normalized, continuous on its positive interior, proper arbitrarily near zero, and has a hard finite upper bound. U=4 bounds stationary log-variance SD by 2. At that bound a central 95% range of stationary volatility about its conditional median spans multiplicative exp(+-1.96). The exponential prior regularizes toward v=0 instead of favoring high v merely because a wide range is available. Its untruncated mass above 4 is exp(-4).

Initial-window log-squared OLS residual variances are approximately 4.8672,5.6714,5.9074. The noise contribution Var(log chi-square_1)=pi^2/2 is approximately 4.9348. Subtraction is only a rough noisy check, not a fitted SV variance estimator; these values are consistent with latent variances of order one or smaller. No later-window data were used to choose rate or bounds.

The inherited m SD=5 is very broad. The saved prior-predictive audit reports this explicitly, including extreme but finite innovation-variance means. Matching m aligns stationary expected-variance priors, not the entire predictive distribution: M5 still has stochastic variance and additional tail flexibility.

## 2. Densities and Jacobians

Let n be the number of factor innovations, e_t=h_t-omega-phi(h_(t-1)-omega), and
\[
A(h,m,\phi,v)=(h_1-m+v/2)^2+
\frac{1}{1-\phi^2}\sum_{t=2}^n [h_t-m+v/2-\phi(h_{t-1}-m+v/2)]^2.
\]
The normalized stationary path log density is
\[
\log p(h\mid m,\phi,v)=-\frac n2\log(2\pi)-\frac n2\log v
-\frac{n-1}{2}\log(1-\phi^2)-\frac{A}{2v}.
\]
The factor likelihood is \(\prod_t N(x_t;a+bx_{t-1},\exp(h_t))\). Multiply these by all independent priors to obtain the posterior. Constants depending on proposal coordinates cannot be dropped.

For physical coordinates (omega,phi,q), m=omega+q/[2(1-phi^2)] and v=q/(1-phi^2). The determinant of the transformation from (m,phi,v) to (omega,phi,q) is 1-phi^2. The new physical-coordinate prior therefore equals \(p_m(m)p_\phi(\phi)p_U(v)/(1-\phi^2)\), not a product of old priors in omega and q.

The implemented coordinates are theta=(m,u,s), u=atanh(phi), s=log(v). Their independent-prior log density is
\[
\log p_\theta=\log p_m(m)+\log p_\phi(\phi)+\log p_U(v)
+\log(1-\phi^2)+s,
\]
with s<log U. The full theta-to-physical determinant is v(1-phi^2)^2, consistent with the previous expression. Dense finite-difference determinant tests check the complete transformation.

## 3. Valid conditional and collapsed updates

The a,b Gaussian conditional and truncation are exactly the existing ones, because the observation likelihood and coefficient prior did not change.

At fixed h,phi,v, the omega conditional is Gaussian with precision
\[
P=25^{-1}+v^{-1}+(n-1)(1-\phi)^2/q,
\]
and precision-weighted mean
\[
B=(\mu_0-v/2)/25+h_1/v+(1-\phi)\sum_{t=2}^n(h_t-\phi h_{t-1})/q.
\]
Draw omega~N(B/P,1/P), then m=omega+v/2. Equivalently reuse the validated omega helper with prior center mu_0-v/2. No independent old omega prior is introduced.

At fixed m,v,h the phi conditional includes the full path determinant and q=v(1-phi^2); using the historical fixed-q phi target would be wrong. Its u-coordinate target is the path density plus log p_phi(phi)+log(1-phi^2). The implementation evaluates the checked full prior/path expression, with constants retained harmlessly.

At fixed m,phi,h, the v conditional is proportional to
\[
p_U(v)v^{-n/2}\exp[-A(h,m,\phi,v)/(2v)]\,1_{0<v<U}.
\]
A itself depends on v through omega=m-v/2. This is not the old inverse-gamma conditional. The implementation never invokes that old M5 update.

For the exact noncentered scale move, fix z=(h-(m-v/2))/sqrt(v), m,phi and coefficients. Then h=m-v/2+sqrt(v)z; the path's v^(-n/2) factor cancels the latent-coordinate Jacobian v^(n/2). For s=log(v), the target becomes
\[
\sum_t\log f_{\log\chi^2_1}(\ell_t-h_t(s))+\log p_U(\exp(s))+s,
\]
where ell_t=log[(x_t-a-bx_(t-1))^2]. The log-squared-residual transformation constant is independent of h and the SV parameters within this update. It does not replace the Gaussian coefficient update. Update h,omega and q jointly with v. No clipping is used.

For mixture-collapsed proposals, replace log chi-square observation errors by the validated ten-component normal mixture, refresh mixture labels, integrate the stationary h path by the existing scalar Kalman likelihood, and add the new theta prior above. The stationary covariance is v*phi^|i-j|, not a fixed-q covariance when phi moves. The FFBS proposal and exact marginal-mixture correction are retained. Rejection restores theta and the entire h path jointly.

A mixture-conditional reversible theta transition is required for the existing correction. Repeated symmetric RW-MH steps meet this requirement. Targeted repair 1, if used, sandwiches a prior-independence phi step between equal powers of that RW kernel. The independence proposal has density p_phi(phi)*(1-phi^2) in u, canceling the same prior/Jacobian at fixed m,v; its MH ratio is just the Kalman likelihood ratio. The palindromic composition is reversible. A plain non-palindromic composition of different reversible steps would not justify the same joint correction.

## 4. Posterior propriety

Condition on any fixed finite observed factor sequence and the fixed empirical-Bayes hyperparameters. Every prior above is proper. For all coefficients and finite h,
\[
L(x\mid a,b,h)\le (2\pi)^{-n/2}\exp[-\tfrac12\sum_t h_t].
\]
After integrating Gaussian m and the stationary path, h has mean (mu_0-v/2)1 and covariance 25*11'+v R_phi, where every entry of R_phi has absolute value at most one. Hence, uniformly over the allowed v,phi,
\[
\log E\exp[-\tfrac12\sum_t h_t]
\le -n\mu_0/2+nU/4+n^2(25+U)/8<\infty.
\]
The likelihood normalizer is finite; integrating over the proper coefficient prior does not change the bound. Likelihood and priors are positive on a set of positive prior measure, so the normalizer is strictly positive. No spurious lower bound on q or distance of phi from one is required for this argument. The executable audit stores this finite bound on the log scale rather than exponentiating it.

## 5. Finite posterior-predictive moments

Append any finite future log-volatility path. Its joint conditional Gaussian covariance still has entries bounded by U; integrating m adds the finite covariance 25*11'. Multiplying the preceding likelihood bound by exp(c'h_extended), for any finite real vector c, has a uniformly finite Gaussian MGF. Thus every finite-order exponential moment of each finite future log-volatility is posterior-integrable.

At finite horizon H, the factor mean is b^H*x_T+a*sum_(j=0)^(H-1)b^j. Since |b|<1, it is bounded by |x_T|+H|a|. The proper Gaussian prior on a has all polynomial moments, even after weighting by the coefficient-uniform likelihood bound. Conditional Gaussian innovation moments are constants times exponential moments of h; finite-sum inequalities or Holder's inequality therefore establish every finite-order posterior-predictive factor moment. Fixed finite Gaussian yield reconstruction noise and linear loadings preserve the result. In particular, the second/fourth moments needed for predictive covariance and its stability exist. This is a finite-horizon result, not a claim about integrating 1/(1-b^2) over the coefficient prior at infinite horizon.

For an explicit prior check,
\[
\log E[\exp(kh)]=k\mu_0+k^2 25/2+
\log E_U\exp[(k^2-k)v/2].
\]
The final expectation is a finite integral over (0,U), evaluated analytically and checked by independent quadrature for U=2,4,8 and k=1,2,4. E[exp(h)]=exp(mu_0+12.5) matches M4 and explains the inherited wide prior. Finite moments must not be confused with a tight or empirically validated prior.

## 6. Forecast moments (not permission to forecast)

For a saved posterior state, at step k:
\[
E[\exp(h_(T+k))\mid\theta,h_T]=
\exp[\omega+\phi^k(h_T-\omega)+v(1-\phi^{2k})/2].
\]
Conditional factor predictive variance at horizon H is the sum of these expectations times b^(2(H-k)); conditional mean is the AR expression above. Integrate using the law of total variance, including posterior coefficient uncertainty. Independent factor posteriors yield diagonal factor covariance; yield covariance is Lambda*V_factor*Lambda'+R using the same predeclared positive diagonal reconstruction-error model in M4 and M5. Positive definiteness and score reconstruction remain numerical acceptance gates, separate from this proof.

All proofs apply equally to the declared U=2/8 sensitivities. Numerical prior-bound sensitivity of posterior forecasts is a separate experiment and must not be claimed from these proofs or prior simulations alone.
