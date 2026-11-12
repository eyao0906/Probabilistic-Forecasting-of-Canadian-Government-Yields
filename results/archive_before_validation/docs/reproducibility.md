# Reproducibility Guide

**Project:** Probabilistic Forecasting of Canadian Government Yields  
**Date:** 2026-10-26  
**Environment:** Windows 11, Python 3.14, Quarto 1.10.18, TinyTeX 2026

## 1. Environment Setup

### Python
Use the system Python to avoid Windows Application Control restrictions on the project-local `.venv`:

```bash
C:/Users/ethan/AppData/Local/Python/pythoncore-3.14-64/python.exe -m venv .venv
.venv/Scripts/python.exe -m pip install numpy pandas scipy matplotlib scikit-learn statsmodels pymc arviz
```

If `.venv` activation fails due to Windows Application Control, call the system Python directly:

```bash
C:/Users/ethan/AppData/Local/Python/pythoncore-3.14-64/python.exe -m pip install numpy pandas scipy matplotlib scikit-learn statsmodels pymc arviz
```

### Quarto / LaTeX
Quarto 1.10.18 with TinyTeX is already installed. Verify with:

```bash
quarto check
```

## 2. Data Acquisition

Raw data files are already present in `data/raw/`:
- `boc_daily_2000_2026.csv` — Bank of Canada daily zero-coupon yield curves
- `feds200628.csv` — Federal Reserve Gurkaynak–Sack–Wright US nominal curve

To refresh raw data:

```bash
python fetch_raw.py
```

To rebuild weekly processed files:

```bash
python prep_data.py
```

Outputs:
- `canada_weekly.csv`
- `us_weekly.csv`

## 3. Running Experiments

The experimental pipeline is implemented in `src/`. The main entry point is:

```bash
C:/Users/ethan/AppData/Local/Python/pythoncore-3.14-64/python.exe src/run_experiments.py
```

This script:
1. Loads `canada_weekly.csv`.
2. Splits into initial training (2000–2015) and out-of-sample (2016–2025).
3. Selects every 4th weekly origin in the out-of-sample period.
4. Fits Models 0–5 at each origin using expanding windows.
5. Computes DSS, Energy Score, RMSE, and prediction-interval coverage.
6. Saves results to `results/forecast_scores.parquet` and scenarios to `results/scenarios.parquet`.
7. Saves MCMC diagnostics to `results/mcmc_diagnostics.parquet`.

### Computational Notes
- Models 0–3 are deterministic and fast.
- Models 4–5 require MCMC at each origin. Expect runtime of several hours for the full rolling window.
- A pilot run on a subset of origins is recommended before full execution.

## 4. Rendering the Paper

To render the Quarto manuscript:

```bash
quarto render paper/MRP_Draft.qmd --to pdf
```

Output: `paper/MRP_Draft.pdf`

Figures and tables are generated from `results/` and `figures/` by the paper's code chunks.

## 5. File Inventory

| Path | Purpose |
|---|---|
| `src/config.py` | Experimental parameters, paths, priors |
| `src/data.py` | Data loading and origin selection |
| `src/models.py` | Model implementations (0–5) |
| `src/evaluation.py` | DSS, Energy Score, coverage, RMSE |
| `src/run_experiments.py` | Main experiment runner |
| `data/raw/` | Raw downloaded data (preserved) |
| `data/processed/` | Weekly cleaned data |
| `results/` | Scores, scenarios, diagnostics |
| `figures/` | Plots saved as PNG/PDF |
| `paper/MRP_Draft.qmd` | Quarto manuscript |
| `docs/` | Guidelines compliance, reproducibility, audit |

## 6. Known Limitations

- Windows Application Control blocks the project-local `.venv` pandas import. Use the system Python.
- Model 3 (Kalman filter MLE) requires `statsmodels`; verify installation.
- Bayesian models (4–5) use PyMC v6; sampler diagnostics must be checked at each origin.
- The US data is included for reference but the primary analysis uses Canadian yields only.

---

*Update this file if the pipeline or dependencies change.*
