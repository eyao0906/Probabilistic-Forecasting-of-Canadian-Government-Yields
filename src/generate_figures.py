"""Generate all figures from saved experiment results."""
from __future__ import annotations

if __name__ == "__main__":
    import sys as _legacy_sys
    from pathlib import Path as _LegacyPath
    _legacy_sys.path.insert(0, str(_LegacyPath(__file__).resolve().parent))
    from legacy_guard import require_legacy_opt_in as _require_legacy_opt_in
    _require_legacy_opt_in()

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "full"
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = [
    "M0_random_walk", "M1_PCA_VAR", "M2_two_step_DNS_AR1",
    "M3_DNS_Kalman_MLE", "M4_Bayesian_DNS_constant_vol", "M5_Bayesian_DNS_stochastic_vol",
]
MODEL_SHORT = [m.split("_", 1)[1].replace("_", " ") for m in MODEL_NAMES]
MATURITIES = [0.25, 0.5, 1, 2, 5, 7, 10, 30]
PRIMARY_HORIZON = 4

plt.rcParams["figure.dpi"] = 180
plt.rcParams["font.size"] = 10


def plot_yield_curve_sample():
    canada = pd.read_csv(ROOT / "canada_weekly.csv", parse_dates=["Date"])
    train = canada[canada["Date"] <= "2015-12-31"]
    test = canada[(canada["Date"] >= "2016-01-01") & (canada["Date"] <= "2025-12-31")]
    cols = [f"y{t:g}" for t in MATURITIES]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(MATURITIES, train[cols].mean(), marker="o", label="Initial window (2000–2015)")
    ax.plot(MATURITIES, test[cols].mean(), marker="s", label="Out of sample (2016–2025)")
    ax.set(xlabel="Maturity (years)", ylabel="Yield (%)", title="Average Canadian zero-coupon yield curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "yield_curve_sample.png", dpi=180)
    plt.close(fig)


def plot_dss_comparison():
    summary = pd.read_csv(RESULTS / "model_summary.csv")
    primary = summary[summary["horizon_weeks"] == PRIMARY_HORIZON]
    if primary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar([r["model"].replace("_", "\n", 1) for _, r in primary.iterrows()],
           [r["dss"] for _, r in primary.iterrows()], color="#496a81")
    ax.set(ylabel="Mean DSS (lower is better)", title="Four-week probabilistic forecast comparison")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "dss_comparison.png", dpi=180)
    plt.close(fig)


def plot_coverage90():
    maturity = pd.read_csv(RESULTS / "maturity_metrics.csv")
    primary = maturity[(maturity["horizon_weeks"] == PRIMARY_HORIZON) &
                       (maturity["model"].isin([MODEL_NAMES[4], MODEL_NAMES[5]]))]
    if primary.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name in (MODEL_NAMES[4], MODEL_NAMES[5]):
        rows = primary[primary["model"] == name]
        ax.plot(rows["maturity_years"], rows["coverage90"], marker="o", label=name[3:])
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Maturity (years)", ylabel="Empirical 90% interval coverage", ylim=(0, 1),
           title="Marginal interval calibration at four weeks")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage90.png", dpi=180)
    plt.close(fig)


def plot_energy_score():
    summary = pd.read_csv(RESULTS / "model_summary.csv")
    primary = summary[summary["horizon_weeks"] == PRIMARY_HORIZON]
    if primary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar([r["model"].replace("_", "\n", 1) for _, r in primary.iterrows()],
           [r["energy_score"] for _, r in primary.iterrows()], color="#496a81")
    ax.set(ylabel="Energy Score (lower is better)", title="Four-week Energy Score comparison")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "energy_score_comparison.png", dpi=180)
    plt.close(fig)


def plot_rmse_by_maturity():
    maturity = pd.read_csv(RESULTS / "maturity_metrics.csv")
    primary = maturity[maturity["horizon_weeks"] == PRIMARY_HORIZON]
    if primary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for name in MODEL_NAMES:
        rows = primary[primary["model"] == name]
        if rows.empty:
            continue
        ax.plot(rows["maturity_years"], rows["rmse"], marker="o", label=name.split("_", 1)[1].replace("_", " "))
    ax.set(xlabel="Maturity (years)", ylabel="RMSE (%)", title="Four-week point-forecast RMSE by maturity")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "rmse_by_maturity.png", dpi=180)
    plt.close(fig)


def plot_risk_summary():
    risk = pd.read_csv(RESULTS / "risk_summary.csv")
    primary = risk[risk["horizon_weeks"] == PRIMARY_HORIZON]
    if primary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, tail in zip(axes, (0.05, 0.01)):
        sub = primary[primary["tail_probability"] == tail]
        labels = [r["model"].replace("_", "\n", 1) for _, r in sub.iterrows()]
        ax.bar(labels, [r["mean_VaR"] for _, r in sub.iterrows()], color="#496a81")
        ax.set(title=f"Mean VaR ({int(tail*100)}%)", ylabel="Loss (%)")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "risk_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    if not (RESULTS / "model_summary.csv").exists():
        print("Results not ready yet.")
        return
    plot_yield_curve_sample()
    plot_dss_comparison()
    plot_coverage90()
    plot_energy_score()
    plot_rmse_by_maturity()
    plot_risk_summary()
    print("Figures generated.")


if __name__ == "__main__":
    main()
