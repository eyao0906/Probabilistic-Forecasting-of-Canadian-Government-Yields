# Manuscript build and results audit

## Scope

This audit covers the factor-level manuscript in `paper/MRP_Draft.qmd` and its
rendered PDF. The evaluation follows the frozen
`FACTOR_EVALUATION_PROTOCOL.md`; it reuses accepted posterior draws and does
not rerun MCMC, alter priors, or change accepted yield-level forecasts.

## Build and document checks

Rendered with:

```powershell
quarto render paper\MRP_Draft.qmd --to pdf
```

The resulting `paper/MRP_Draft.pdf` has 34 pages and a 191-word abstract. The
main text is configured at 12 pt. Automated verification found no unresolved
citations, cross-references, PDF reference markers, duplicate labels, missing
includes, or blank pages. The manuscript source and extracted PDF text contain
no legacy model label.

Representative pages covering the score tables, paired comparison, risk
illustration, and bibliography were visually inspected. The paired-comparison
table is kept together on one page. No clipped or overflowing content was
observed in those pages.

## Numerical checks

- The manuscript's CRPS, RMSE, and 90% coverage tables match all 18 displayed
  model-factor-horizon rows in `results/factor_probabilistic_20261116/factor_metrics.csv`.
- All six displayed paired differences, confidence intervals, raw p-values,
  and Holm-adjusted p-values match `paired_tests.json`.
- All six stochastic-volatility-minus-constant-volatility CRPS differences
  are negative; none is significant after Holm adjustment at 5%.
- The evaluation covers 130 origins and 3,120 forecasts. Accepted origin
  manifests and posterior checkpoint hashes passed validation; reconstructed
  posterior moments matched all 520 saved model-by-horizon records.

The numbers are marginal factor results conditional on OLS-extracted factors
and fixed loadings. The five-year duration example varies one factor at a
time; it is not joint portfolio risk or evidence of full-curve calibration.

## Software checks

- Four focused tests in `tests/test_factor_probabilistic_experiment.py` pass.
- The factor experiment, verifier, and focused test files pass Python
  compilation.
- `git diff --check` passes.

No full repository test suite or new sampler run was performed. Manual review
of the academic interpretation, references, author details, and institutional
requirements remains the author's and supervisor's responsibility.
