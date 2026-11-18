# Hautsch-Yang blocked-algorithm audit

Primary source: Hautsch and Yang, *Bayesian Inference in a Stochastic
Volatility Nelson-Siegel Model*, working paper section 3.1, Table 1 and
Appendix A; [working paper](https://edoc.hu-berlin.de/bitstreams/fd4df556-9fb9-4777-8959-d47a4b7d3fb1/download),
[published article](https://doi.org/10.1016/j.csda.2010.07.003). The SV
approximation originates with Kim, Shephard and Chib (1998); the paper cites
Chib et al. (2002) for the Student-t proposal.

| Paper block | Implementation | Target / test / deviation |
|---|---|---|
| 2(a), precision | `blocked_dlns.measurement_conditional` | Independent inverse-gamma conditional for diagonal R, not the appendix's full Wishart; direct sum-of-squares test. |
| 2(b), decay | `blocked_dlns.decay_probabilities` | Grid-normalized observation likelihood with uniform discrete prior; sampled, not maximized. Tested against direct density. Canadian per-year support and grid sensitivity differ from the paper. |
| 2(c), factors | `blocked_states` and `blocked_dlns.dynamics_conditionals` | Full-path Gaussian filtering/backward simulation; mean and truncated persistence conditionals. Dense Gaussian and direct quadratic tests. Initial `f_1` prior is independent rather than fixed at the factor mean. |
| DLNS Q | `blocked_dlns.innovation_conditional` | Full-Q inverse-Wishart conditional; paper describes a simplification but does not unambiguously detail a full-Q update. Proper dimension-3 prior and direct density test. |
| 2(d)(i)-(ii), mixture | `blocked_sv.mixture_indicators` | Table 1 weights, means and variances, evaluated by stable log-sum-exp. Mixture probabilities checked against independent normal densities. |
| 2(d)(iii), h | `blocked_sv.scalar_smoother` | Full h path, conditional on mixture indicators and SV parameters. Filter likelihood and conditional moments checked against dense Gaussian formulas. |
| 2(d)(iv), SV parameters | `blocked_sv.proposal_distribution`, `parameter_log_acceptance` | Five-df Student-t with BFGS mode and inverse-Hessian approximation for the marginalized h likelihood conditional on indicators. A deterministic prior-centered optimizer is used so the same proposal applies in reverse; numerical acceptance ratio checked independently. This is not the paper's stated conditional-on-h update. |
| Approximation correction | `blocked_sv.volatility_block` | The marginalized-parameter MH move followed by a conditional h draw is a reversible proposal for the mixture target at fixed innovations and indicators. A second MH step uses the exact-Gaussian/mixture likelihood ratio. Simplified fixed-parameter and joint mean/path kernels pass numerical quadrature; the complete joint correction still needs stronger independent validation. |

## Correction and limitations

For fixed factor innovations `e`, let `z=log(e^2+c)` and
`g(z|h)=sum_k w_k N(z;h+m_k,v_k)`. The code evaluates
`log w(h)=sum_t [log N(e_t;0,exp(h_t))-log g(z_t|h_t)]`.
After an approximate reversible joint `(h,theta)` proposal, it accepts with
`min(1,w(h_new)/w(h_old))`. With e fixed, any transformation/Jacobian
factor independent of h cancels. Indicators are auxiliary, drawn from the
mixture conditional on the old h; the second MH step operates with those
indicators fixed. The sampler uses one such transition per outer iteration
in the pilots; it does not report uncorrected draws as exact.

This is **not** post-hoc importance reweighting. In particular, conditional
weights at changing factor innovations do not by themselves establish
weights for the full joint target. A coherent approximate joint posterior,
including the transformation and offset normalization, and its importance
weight concentration/ESS have not been established. The stored quantities
are MH log-ratios and acceptances, not importance weights. Treat the full
joint SV algorithm as experimental pending independent verification.

The paper describes 2,000 inner iterations per factor with 500 discarded.
`--inner-steps` can repeat the corrected transition 2,000 times at each
fixed outer state, carrying only its final draw. This is a different
corrected kernel, not reproduction of the paper's retained 1,500 inner draws
or post-hoc reweighting. The actual pilots use one inner step. The 2,000-step
fixed-data profile uses the first 500 only as a timing/acceptance burn-in;
it is not a full nested posterior run. For timing and feasibility, see
`COMPUTATIONAL_PROFILE.md`.
