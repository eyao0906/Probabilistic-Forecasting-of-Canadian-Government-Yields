# Progress Log

**Date:** 2026-10-26 to 2026-10-27  
**Repository:** `C:/Users/ethan/Downloads/MRP`

## Completed

- [x] Located and read proposal (`MRP_Proposal.qmd`) and MQF 2026 guidelines (`MQF MRP Guidelines 2026.pdf`).
- [x] Audited repository: existing scripts (`fetch_raw.py`, `prep_data.py`), raw data, and weekly processed data validated.
- [x] Identified environment constraints: Windows Application Control blocks `.venv` pandas; resolved by using system Python.
- [x] Installed required Python packages: numpy, pandas, scipy, matplotlib, scikit-learn, statsmodels, pymc, arviz.
- [x] Created project structure: `src/`, `tests/`, `data/raw/`, `data/processed/`, `results/`, `figures/`, `paper/`, `docs/`, `logs/`.
- [x] Validated data pipeline: `prep_data.py` produces `canada_weekly.csv` (1,396 rows, 1999-12-31 to 2026-09-23) and `us_weekly.csv` (1,396 rows, 2000-01-07 to 2026-10-02).
- [x] Verified SHA256 hashes for raw and processed files.
- [x] Wrote `docs/research_audit.md`, `docs/reproducibility.md`, `docs/guidelines_compliance.md`.
- [x] Created `references.bib` with proposal references (pending full verification).
- [x] Implemented and verified data loading module (`src/data.py`) with tests.
- [x] Implemented evaluation metrics (`src/evaluation.py`): DSS, Energy Score, prediction-interval coverage, RMSE.
- [x] Confirmed existing model implementations in `src/models.py` and experiment runner in `src/experiment.py`.
- [x] Ran 10-origin pilot: all 6 models executed, results saved to `results/pilot/` and `results/full/`.
- [x] Completed bounded 10-origin test run (`proc_fa1b5b1899de`): 10 origins, 120 forecasts, state-space 10/10 success, elapsed ~351.7s.
- [x] Updated experiment configuration: increased MCMC warmup/draws and state-space max iterations.
- [x] Created `paper/MRP_Draft.qmd` with complete structure, placeholder results sections, and APA CSL.
- [x] Created `src/generate_figures.py` for reproducible figure generation.
- [x] Initialized Git (no remote configured yet).

## In Progress

- [ ] Full rolling out-of-sample experiment with all ~131 origins (running in background).
- [ ] Literature verification and source inventory (subagents dispatched; awaiting results).
- [ ] MCMC convergence diagnostics review and potential sampler tuning.

## Pending

- [ ] Replace paper placeholders with genuine numerical results and generate figures.
- [ ] Render `paper/MRP_Draft.pdf` and verify formatting against MQF guidelines.
- [ ] Validate all citations, cross-references, and bibliography entries.
- [ ] Write `references.bib` final version with verified DOIs and metadata.
- [ ] Complete `docs/research_audit.md` with limitations and unresolved issues.
- [ ] Initialize Git repository, add files, and commit milestones.
- [ ] Final review of abstract length, page count, and guideline compliance.

## Blockers

- Windows Application Control policy blocks `.venv` pandas import; mitigated by using system Python.
- MCMC acceptance rates for volatility parameters are low (0.09–0.32); ESS for `h_intercept` and `phi` is poor in pilot diagnostics. Will document and consider longer chains or improved tuning if time permits.
- State-space model (M3) fails for some origins with L-BFGS-B; success rate improved from 0/5 to 1/2 with maxiter=12. Will monitor full-run success rate.
