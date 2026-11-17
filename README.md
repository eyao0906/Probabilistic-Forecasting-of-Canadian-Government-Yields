# Canadian yield-factor forecasting — MQF MRP

**Paper:** [`paper/MRP_Draft.pdf`](paper/MRP_Draft.pdf) · **Source:** [`paper/MRP_Draft.qmd`](paper/MRP_Draft.qmd) · **Status:** [`MRP_PROGRESS.md`](MRP_PROGRESS.md)

## Current study: factor-level probabilistic forecasting

The current manuscript evaluates marginal forecasts of OLS-extracted Nelson–Siegel level, slope, and curvature factors. It compares M0 (factor random walk), M2 (unrestricted OLS AR(1)), M3 (Bayesian constant volatility), and M4 (Bayesian stochastic volatility). Paper M3/M4 map to accepted archive M4/M5. The Bayesian forecasts use the accepted posterior archive; MCMC, priors, and existing yield forecasts were not changed or rerun.

The evaluation covers 130 origins, horizons one and four, and 3,120 model-factor-horizon-origin forecasts. M4 has lower mean CRPS than M3 in all six factor-horizon groups, with the largest differences for curvature. None is significant after Holm adjustment. M0/M2 remain competitive for level and slope, and marginal coverage varies by factor. The paper does not claim full-curve calibration or joint portfolio risk.

Key deliverables:
- [`docs/factor_probabilistic_20261116/RESEARCH_SCOPE_REVISION.md`](docs/factor_probabilistic_20261116/RESEARCH_SCOPE_REVISION.md) — revised question and scope.
- [`docs/factor_probabilistic_20261116/FACTOR_EVALUATION_PROTOCOL.md`](docs/factor_probabilistic_20261116/FACTOR_EVALUATION_PROTOCOL.md) — frozen evaluation protocol.
- [`docs/factor_probabilistic_20261116/FACTOR_EVALUATION_REPORT.md`](docs/factor_probabilistic_20261116/FACTOR_EVALUATION_REPORT.md) — scores, inference, and limits.
- [`docs/factor_probabilistic_20261116/FACTOR_RISK_APPLICATION.md`](docs/factor_probabilistic_20261116/FACTOR_RISK_APPLICATION.md) — factor-isolated duration illustration.
- [`docs/factor_probabilistic_20261116/MANUSCRIPT_AUDIT.md`](docs/factor_probabilistic_20261116/MANUSCRIPT_AUDIT.md) — render, numerical, and integrity checks.

The earlier eight-yield study and residual-subspace experiment remain separate historical/sensitivity evidence; neither is presented as the new factor evaluation. For existing prior rationale and provenance, see [`docs/bayesian_prior_justification.md`](docs/bayesian_prior_justification.md).

## Folder map

- `paper/` — current QMD/PDF, citation style and saved reporting tables.
- `src/`, `tests/` — implementations and focused checks.
- `data/raw/`, `data/processed/` — preserved sources and weekly panel; retrieval verification remains separate.
- `results/factor_probabilistic_20261116/` — factor forecasts, metrics, paired tests, risk sensitivities, and figures.
- `results/bayesian_prior_revised_20261114/` — accepted revised-prior posterior draws and yield-level archive reused for M3/M4 factor predictions.
- `results/validated_baselines/` — accepted yield-level M0–M2 results; unchanged and not substituted for factor-level M0/M2.
- `results/bayesian_finite_20261027T194239Z/` — historical finite-prior run and sensitivity evidence.
- `results/residual_subspace_experiment_20261116/` — separate yield-level residual sensitivity evidence.
- `figures/finite/`, `figures/validated/` — existing paper figures.
- `docs/` — factor study reports, prior mathematics, literature/source evidence and reproduction notes.
- `references.bib`, `requirements-lock.txt` — bibliography and pinned Python packages.

Accepted-run names are unchanged to preserve frozen references. Original pre-final paper/progress bytes are externally archived in `../MRP_archives/pre_final_manuscript_20261028T010801Z.zip`. Superseded progress/review reports, isolated-test evidence and older notes are in `../MRP_archives/superseded_diagnostics_20261028T013219Z.zip`. The active `.venv` and referenced scientific dependencies remain.

## Minimal reproduction

From the repository root, using the existing Python environment and installed Quarto/LuaLaTeX:

```powershell
.\.venv\Scripts\python.exe -B -m src.factor_probabilistic_experiment
quarto render paper\MRP_Draft.qmd --to pdf
.\.venv\Scripts\python.exe -B -c "from src.verify_paper import verify; r=verify(write_outputs=False, previews=False); print(r); assert r['passed']"
```

The evaluation command rechecks the accepted archive and computes factor-only forecasts without fitting MCMC. The QMD consumes its saved tables and figures. Do not rerun MCMC or overwrite the accepted posterior or yield-level archives for a paper build. The prior audit and historical reproduction guidance remain in [`docs/reproduction_runbook.md`](docs/reproduction_runbook.md).

Full author name, supervisor identity, permitted AI assistance and university sign-off require human confirmation. The author must understand, check and revise the manuscript under the supplied 2026 guidelines.
