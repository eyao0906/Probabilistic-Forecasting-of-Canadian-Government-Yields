# Final research assessment

The requested joint-state-space research question remains open. On the
actual Canadian panel, blocked Gibbs with a full factor smoother and a
fine-grid decay update did not converge after 20,000 DLNS iterations per
chain. The shorter corrected-mixture SVNS pilots also failed badly, with
low correction acceptance for level and slope at the reference offset.
Therefore no posterior factor estimates, model-fit claims, predictive
uncertainty comparisons or B3/B4 benchmark refits are supported.

The fine decay grid moves locally but does not bridge separated decay and
covariance configurations. One plausible next research direction is a
joint state/decay move or a carefully validated marginal/blocking scheme;
the existing results cannot establish whether the difficulty is
multimodality, weak identification, prior sensitivity, misspecification,
or a combination. Any new approach needs its own posterior-invariance and
real-data diagnostics.

This is a methodological adaptation, not a faithful numerical reproduction:
measurement covariance is diagonal despite the paper's full-Wishart appendix;
DLNS uses a proper full-Q prior; initialization, priors and decay support are
Canadian-specific; and the SV block uses a Metropolis correction rather
than the paper's post-hoc reweighting. The joint correction's mathematical
verification and importance-weight diagnostics remain incomplete.
