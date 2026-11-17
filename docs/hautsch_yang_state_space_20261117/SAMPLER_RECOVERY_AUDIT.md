# Sampler recovery audit

## Baseline and retained failed evidence

The isolated branch started this recovery at commit
`a8ed5e0fc8d6a3238136abcb0d17169ac58d70d2`, based on
`dd5b79b6c7c694826cb9e038ae44f1708d637452`. The worktree was clean and the
published branch pointed to the same commit before the correction.
The SHA-256 of the baseline `src/ssdns_hy/sampler.py` was
`b71e8c6e08f4122a9fcce10c54636ce2b41106ad9f4b15ac4cbf2d676b3d80d8`.

The earlier four-chain synthetic pilots were rejected: SS_M3 maximum
rank R-hat 1.049 and minimum bulk ESS 52; SS_M4 maximum rank R-hat 3.125
and minimum bulk ESS 4.5 in the 835-observation run. SS_M4 failed to
recover simulated volatility persistence. These numbers were reported in the
previous session; complete draws and diagnostics were not preserved and cannot
be reconstructed from the aggregate record. Their synthetic initial state was
fixed at the long-run mean, unlike the inference model's diffuse prior.
The old SS_M4 implementation also read uninitialized memory, making those
results unreliable and not necessarily reproducible. Do not compare old and
corrected values as though they were both valid posterior estimates.

The historical revised-prior Bayesian archive reports `accepted: true` and
130 completed origins in `results/bayesian_prior_revised_20261114/summary.json`.
The separate Gaussian state-space experiment at
`results/state_space_revised/verification.json` was rejected because eight
origins exhibited optimizer-start sensitivity. Neither experiment was altered
or used as the new sampler. The original processed panel SHA-256 is
`0b2af5c3c7bb261973ea4ec04c151eedbf7cf1428f22d0e08eaae87a212bfd77`,
matching its manifest. The raw panel hash
`1bbed8dbf8e2c65bc681117ad5c49971eabb8242e98e98c963ff30e6f55ddc1e`
does not match manifest hash
`ada4b4208b6a4361463181b889383869ba9bf5ad2fbca531a047f954283238a9`;
this pre-existing discrepancy is not resolved and neither file was changed.

## Confirmed defects and corrections

1. The SV elliptical-slice direction previously assigned all later entries at
   once while reading the same uninitialized array. It was not a stationary
   Gaussian AR(1) draw and could invalidate the entire SS_M4 transition.
   The direction is now generated recursively with stationary variance \(v\)
   and lag-\(k\) correlation \(\phi^k\). The likelihood and scientific model
   were not changed.
2. The volatility AR(1) likelihood hard-coded an upper limit of 4 even
   though the MH variance update receives its bound separately. The
   likelihood now checks only positive variance; the transformed proposal
   enforces the configured bound. The default bound remains 4.
3. Diagnostic aggregation formerly ignored individual NaN values, which could
   make an invalid diagnostic appear to pass. It now raises explicitly.
4. The conditional variance for the unobserved initial factor is computed as
   \(P_0q/(\rho^2P_0+q)\), algebraically the same conditional variance but
   without cancellation between nearly equal large numbers.
5. The synthetic generator now offers a draw from the inference model's
   proper diffuse initial-state distribution, and rejects nonfinite inputs.
   The old fixed-at-mean default remains available for mismatch studies.

## Conditional-target review

- The initial factor conditional follows Gaussian conditioning on \(f_1\).
  Its mean and variance are checked by repeated simulation.
- Factor means and persistence use precision-weighted normal regression
  conditionals, with the initial-state density included in the mean update
  and persistence restricted to \((-1,1)\). The persistence normalizing
  constant is fixed by the prior hyperparameters.
- Constant log factor variance has a normal log-variance prior and the
  Gaussian innovation likelihood; its slice target is checked against an
  independently written density.
- Measurement variances use inverse-gamma conditional shape
  \(3+T/2\) and scale \(2r_0+\mathrm{SSE}/2\); simulated moments are checked.
- The decay step targets the yield likelihood with \(f_{0:T}\) integrated out,
  conditional on all other variables, including the complete SV path. Its
  logit Jacobian is \((b-a)x(1-x)\). A dense multivariate-normal reference
  checks the acceptance decision on both sides of the independent probability.
  The factor path and initial state are resampled immediately after the
  collapsed decay step, before conditioning other updates on that path.
- The SV path uses a stationary AR(1) prior centered at \(m-v/2\) and the
  exact Gaussian innovation likelihood. Its stationary covariance,
  constant-likelihood invariance, independent Gaussian-likelihood posterior,
  exact one-dimensional SV posterior and near-unit-root behavior have focused
  numerical tests. Four distinct scalar starts were checked.
- The SV mean update is a Gaussian conditional using the stationary first
  density and AR(1) increments. The persistence proposal uses an atanh
  Jacobian \(1-\phi^2\) and the transformed beta prior. The stationary
  variance proposal uses a bounded-logit Jacobian and the rate-one exponential
  prior. Both are MH conditional updates; the centered SV parameterization may
  nevertheless mix slowly. The AR(1) log density is tested against an
  independently assembled multivariate normal.
- Innovations indexed by \(t\) generate \(f_t\); the predictor advances
  \(h\) before drawing the next factor innovation and adds target-date
  observation noise. Existing dense Gaussian filter/smoother and predictive
  smoke tests remain in place.

Focused tests check the factor-parameter conditional means and scales against
independently calculated weighted regression expressions, the SV mean update
against an independently assembled Gaussian conditional, and the SV MH
accept/reject decisions against dense multivariate-normal densities and
explicit transformed priors. These tests are not a proof of full sampler
posterior invariance; repeated synthetic recovery is still required.

## Reproducible bounded screening

Run `python -m src.ssdns_hy.synthetic_pilot` from the isolated worktree.
The screen uses two independently generated datasets per model and four RNG
streams per dataset, with explicit seeds and the exact diffuse initialization.
The short 120-observation runs are screening only, not the requested full
recovery study. The chains have independent random streams but currently start
from the same data-calibrated initial values, not deliberately dispersed
full-model starts. The input panel calibrates its own empirical-Bayes priors; this
is an inference procedure, not prior-free coverage.

The design records SHA-256 hashes for source files, simulation settings,
prior settings, seeds, initial states, per-parameter diagnostics and chain
checkpoint hashes. Each completed chain is written as a compressed NPZ with
an integrity sidecar. A run can resume only when its design, code, dataset
fingerprint and checkpoint checksum match. Small screen checkpoints are kept
in the experiment results directory; large production archives should be
stored externally with a manifest and checksums rather than added to Git.

The accepted gates remain R-hat at most 1.01, and bulk and tail ESS at least
400 for every reported parameter. Passing this short screen alone would not
establish repeated recovery, coverage, or suitability for Canadian inference.
The original screen is reproducible with its source at commit `d731660`; the
later targeted M3 scale comparison uses `--tuned-m3` and a separate output
directory. A different source hash cannot silently resume either run.
