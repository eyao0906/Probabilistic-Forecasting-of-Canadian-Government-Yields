# Canadian yield-curve forecasting — MQF MRP

**Paper:** [`paper/MRP_Draft.pdf`](paper/MRP_Draft.pdf) · **Source:** [`paper/MRP_Draft.qmd`](paper/MRP_Draft.qmd) · **Status:** [`MRP_PROGRESS.md`](MRP_PROGRESS.md)

## Research objective and experiment

Does stochastic factor volatility improve joint probabilistic forecasting within Bayesian Dynamic Nelson–Siegel (DNS)? M4 (constant volatility) and M5 (stochastic volatility) share factor extraction, coefficient priors, long-run expected-innovation-variance priors, reconstruction noise and dates. The finite-moment M5 prior uses U=4; U=2/8 are pre-specified representative sensitivities, not test-score-selected bounds. External accepted benchmarks are M0 random walk, M1 PCA–VAR and M2 classical DNS. M3 failed existing fit-stability acceptance and is excluded.

Bank of Canada yields at eight maturities (0.25, 0.5, 1, 2, 5, 7, 10, 30 years) provide 1,356 weekly research observations in 2000–2025. The initial 2000–2015 window has 835 curves. Evaluation uses 130 common expanding-window origins during 2016–2025, every fourth weekly observation, at horizons one and four weeks. Publication delays and historical-vintage uncertainty mean these are retrospective observation-time forecasts, not live trading results.

## Accepted findings

The main Bayesian run and aggregate analysis are accepted: **130 origins, 520 records, no failed origins**. The accepted M0–M2 experiment supplies 780 unchanged records.

At four weeks M5 improves DSS from −21.450 to −21.840 and Energy Score from 0.4178 to 0.4073 (2.52%). Paired 95% intervals are [−0.762, −0.051] and [−0.0185, −0.0023]. Both score intervals favor M5 at both horizons and expected stationary-bootstrap lengths 4/8/12. Main inference is frozen at **9,999 replicates, length eight origins, seed 271026**, without multiplicity adjustment.

The RMSE difference is only −0.162 basis points and its interval contains zero. Both Bayesian models overcover marginal 90% intervals; neither beats the random walk descriptively on aggregate DSS, Energy Score or RMSE. The finding is modest within-DNS distributional value, not general dominance. All six U=2/8 representative sensitivity cases are accepted. DSS is locally stable, but M5 loses at some representative origins and the late Energy Score ranking changes; this is not full-period prior robustness.

## Folder map

- `paper/` — current QMD/PDF, citation style and saved reporting tables.
- `src/`, `tests/` — current implementations and essential checks.
- `data/raw/`, `data/processed/` — preserved sources and weekly panel; independent retrieval verification remains separate.
- `results/validated_baselines/` — accepted M0–M2.
- `results/bayesian_finite_20261027T194239Z/rolling_main/` — accepted M4/M5 checkpoints; `analysis/` holds numerical comparisons.
- `results/bayesian_finite_20261027T194239Z/bound_sensitivity/` — U=2/8 cases and acceptance/failure evidence.
- Other `results/` — historical evidence retained where code or frozen manifests require original paths, not accepted inference.
- `figures/finite/`, `figures/validated/` — paper figures.
- `docs/` — finite-prior mathematics, literature/source evidence and reproduction notes.
- `references.bib`, `requirements-lock.txt` — bibliography and pinned Python packages.
- `README.md`, `MRP_PROGRESS.md` — entry point and concise current status.

Accepted-run names are unchanged to preserve frozen references. Original pre-final paper/progress bytes are externally archived in `../MRP_archives/pre_final_manuscript_20261028T010801Z.zip`. Superseded progress/review reports, isolated-test evidence and original older notes are in `../MRP_archives/superseded_diagnostics_20261028T013219Z.zip`. Both ZIPs were read back before removal of working copies. Duplicate test environments, bytecode and scratch outputs were removed; the active `.venv` and referenced scientific dependencies remain.

## Minimal reproduction

From this repository, using the existing Windows Python 3.14.7 environment and installed Quarto/LuaLaTeX:

```bash
# Regenerate the posterior volatility figure from saved chains; no model fitting.
.venv/Scripts/python.exe -B src/plot_factor_volatility.py

# Create manuscript figures with publication-only M3/M4 labels; no model fitting.
.venv/Scripts/python.exe -B src/plot_manuscript_presentations.py

# Render saved reporting assets; no model fitting.
quarto render paper/MRP_Draft.qmd --to pdf
.venv/Scripts/python.exe -B -c "from src.verify_paper import verify; r=verify(write_outputs=False, previews=False); print(r); assert r['passed']"

# Reconstruct numerical comparisons from accepted archives; no sampling.
.venv/Scripts/python.exe -B -m src.bayesian_finite_analysis --output results/bayesian_finite_20261027T194239Z/rolling_main
```

Numerical sources are `analysis/paired_summary.csv`, `per_maturity.csv` and `summary.json`; saved Markdown tables and figures are the paper-render inputs. Prior-bound reporting uses `bound_sensitivity/comparisons.csv`. `paper/MRP_Validated.qmd` is a protected historical draft, not the current paper. Elsewhere, install a compatible environment from `requirements-lock.txt`. Do not run older all-purpose artifact builders over protected baseline metadata merely to render the paper.

Only if numerical work is genuinely unfinished, the existing checkpoint commands are:

```bash
.venv/Scripts/python.exe -u -B -m src.bayesian_finite_experiment --gate results/bayesian_finite_20261027T194239Z/phase2_gate.json --output results/bayesian_finite_20261027T194239Z/rolling_main --workers 4
.venv/Scripts/python.exe -u -B -m src.bayesian_finite_sensitivity --main results/bayesian_finite_20261027T194239Z/rolling_main --workers 2
```

Never duplicate an active process, refit accepted stages for a paper build, or change a frozen gate. `protected_before.json` is historical pre-production evidence; the explicitly authorized manuscript changes have archived originals. Scientific dependencies retain their frozen paths. See `docs/reproduction_runbook.md` before historical replay; the root full suite is not a safe paper-build step because older fixtures have written canonical results.

Full author name, supervisor identity, permitted AI assistance and university sign-off require human confirmation. The author must understand, check and revise the manuscript under the supplied 2026 guidelines. No submission, email, Git reset or remote push was performed.
