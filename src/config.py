"""Project-wide configuration for the MRP yield-curve forecasting experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root (one level up from src/)
ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CANADA_WEEKLY = ROOT / "canada_weekly.csv"
US_WEEKLY = ROOT / "us_weekly.csv"

# Maturities in years (must match the weekly files)
TAU = [0.25, 0.5, 1, 2, 5, 7, 10, 30]
MATURITIES = TAU
MATURITY_COLS = [f"y{t}" for t in TAU]
YIELD_COLUMNS = MATURITY_COLS

# Experimental design
INITIAL_TRAIN_END = "2015-12-31"
OUT_OF_SAMPLE_START = "2016-01-01"
OUT_OF_SAMPLE_END = "2025-12-31"
FORECAST_HORIZONS = [1, 4]
PRIMARY_HORIZON = 4
ORIGIN_STEP = 4  # every 4th weekly origin

# DNS decay parameter (held fixed after initial training)
LAMBDA = 0.0609  # standard choice for monthly-like frequency; will be optimized on initial window

# Random seed for reproducibility
SEED = 20260926

# Bayesian MCMC settings (per origin)
MCMC_CHAINS = 4
MCMC_DRAWS = 2000
MCMC_TUNE = 2000
MCMC_TARGET_ACCEPT = 0.85

# Prior defaults (univariate AR(1) + SV per factor)
PRIOR_A_MEAN = 0.0
PRIOR_A_SD = 5.0
PRIOR_B_MEAN = 0.7  # centered near stationary
PRIOR_B_SD = 0.3
PRIOR_SIGMA_ETA_SD = 0.5
PRIOR_OMEGA_MEAN = 0.0
PRIOR_OMEGA_SD = 5.0
PRIOR_PHI_MEAN = 0.7
PRIOR_PHI_SD = 0.2
PRIOR_SIGMA_U_SD = 0.5

# Results paths
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PAPER_DIR = ROOT / "paper"
DOCS_DIR = ROOT / "docs"
LOGS_DIR = ROOT / "logs"

# Output file names
RESULTS_FILE = RESULTS_DIR / "forecast_scores.parquet"
SCENARIOS_FILE = RESULTS_DIR / "scenarios.parquet"
DIAGNOSTICS_FILE = RESULTS_DIR / "mcmc_diagnostics.parquet"
