"""Publication-quality figures for the Canadian volatility-regime extension."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.canadian_volatility_regime_analysis import (
    FACTORS,
    HORIZONS,
    INTERVALS,
    MODELS,
    OUTPUT,
)

COLORS = {
    "LOW": "#1976a5",
    "MEDIUM": "#888888",
    "HIGH": "#d05a47",
    "M0": "#6a51a3",
    "M1": "#238b45",
    "M2": "#e6ab02",
    "M3": "#d95f02",
    "AR1_GARCH11": "#1b9e77",
}
SHORT_MODELS = {
    "M0": "M0",
    "M1": "M1",
    "M2": "M2",
    "M3": "M3",
    "AR1_GARCH11": "AR-GARCH",
}


def _save(fig: plt.Figure, stem: str) -> None:
    directory = OUTPUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _metrics() -> pd.DataFrame:
    return pd.read_csv(OUTPUT / "regime_metrics.csv")


def _forecasts() -> pd.DataFrame:
    return pd.read_csv(OUTPUT / "forecasts_with_regimes.csv")


def _metric_row(metrics: pd.DataFrame, regime: str, factor: str,
                horizon: int, model: str) -> pd.Series:
    rows = metrics.loc[
        (metrics["regime"] == regime) & (metrics["factor"] == factor)
        & (metrics["horizon"] == horizon) & (metrics["model"] == model)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one metric row for {regime}/{factor}/h={horizon}/{model}")
    return rows.iloc[0]


def figure_1_history() -> None:
    history = pd.read_csv(OUTPUT / "historical_volatility_index.csv", parse_dates=["Date"])
    origins = pd.read_csv(OUTPUT / "origin_volatility_regimes.csv", parse_dates=["Date"])
    thresholds = pd.read_json(OUTPUT / "regime_thresholds.json", typ="series")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(history["Date"], history["composite_volatility"],
            color="#2b2b2b", lw=1.0, alpha=0.8, label="Composite factor-change RMS")
    for regime in ("LOW", "MEDIUM", "HIGH"):
        subset = history.loc[history["regime"] == regime]
        ax.scatter(subset["Date"], subset["composite_volatility"],
                   color=COLORS[regime], s=9, alpha=0.48,
                   label=f"{regime.title()} weekly observations")
    q25 = float(thresholds["q25"])
    q75 = float(thresholds["q75"])
    ax.axhline(q25, color=COLORS["LOW"], ls="--", lw=1.3,
               label=f"Training q25 = {q25:.3f}")
    ax.axhline(q75, color=COLORS["HIGH"], ls="--", lw=1.3,
               label=f"Training q75 = {q75:.3f}")
    ax.axvline(pd.Timestamp("2016-01-01"), color="black", ls=":", lw=1.4,
               label="Evaluation period begins")
    origin_colors = origins["regime"].map(COLORS)
    ax.scatter(origins["Date"], origins["composite_volatility"],
               c=origin_colors, edgecolors="black", linewidths=0.45,
               marker="D", s=32, zorder=5, label="Forecast origins")
    counts = origins["regime"].value_counts().to_dict()
    ax.set_title("Historical Canadian DNS-factor volatility regimes")
    ax.set_ylabel("26-week standardized factor-change RMS")
    ax.set_xlabel("Observation date")
    ax.text(
        0.01, 0.97,
        "Origin counts: " + ", ".join(
            f"{name}={counts.get(name, 0)}" for name in ("LOW", "MEDIUM", "HIGH")),
        transform=ax.transAxes, ha="left", va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, fontsize=8, loc="upper right", frameon=True)
    fig.tight_layout()
    _save(fig, "figure_1_regime_history")


def figure_2_crps() -> None:
    metrics = _metrics()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharey=False)
    width = 0.36
    x = np.arange(len(MODELS))
    for row, factor in enumerate(FACTORS):
        for col, horizon in enumerate(HORIZONS):
            ax = axes[row, col]
            low = [_metric_row(metrics, "LOW", factor, horizon, model) for model in MODELS]
            high = [_metric_row(metrics, "HIGH", factor, horizon, model) for model in MODELS]
            ax.bar(x - width / 2, [item["mean_crps"] for item in low], width,
                   color=COLORS["LOW"], label="LOW")
            ax.bar(x + width / 2, [item["mean_crps"] for item in high], width,
                   color=COLORS["HIGH"], label="HIGH")
            for i, item in enumerate(low):
                ax.text(i - width / 2, item["mean_crps"], f"{int(item['n_origins'])}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
            for i, item in enumerate(high):
                ax.text(i + width / 2, item["mean_crps"], f"{int(item['n_origins'])}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
            ax.set_xticks(x, [SHORT_MODELS[model] for model in MODELS], rotation=15)
            ax.set_title(
                f"{factor.title()}, h={horizon} "
                f"(LOW n={int(low[0]['n_origins'])}, HIGH n={int(high[0]['n_origins'])})")
            ax.set_ylabel("Mean CRPS (lower is better)")
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(title="Origin regime")
    fig.suptitle("Conditional factor forecast CRPS by volatility regime", y=1.01)
    fig.tight_layout()
    _save(fig, "figure_2_conditional_crps")


def figure_3_skill() -> None:
    metrics = _metrics()
    selected = ("M2", "M3", "AR1_GARCH11")
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharey=True)
    width = 0.34
    x = np.arange(len(selected))
    for row, factor in enumerate(FACTORS):
        for col, horizon in enumerate(HORIZONS):
            ax = axes[row, col]
            low = [_metric_row(metrics, "LOW", factor, horizon, model)
                   for model in selected]
            high = [_metric_row(metrics, "HIGH", factor, horizon, model)
                    for model in selected]
            ax.bar(x - width / 2, [item["crps_skill_vs_M1"] for item in low], width,
                   color=COLORS["LOW"], label="LOW")
            ax.bar(x + width / 2, [item["crps_skill_vs_M1"] for item in high], width,
                   color=COLORS["HIGH"], label="HIGH")
            ax.axhline(0, color="black", lw=0.9)
            ax.set_xticks(x, [SHORT_MODELS[model] for model in selected])
            ax.set_title(
                f"{factor.title()}, h={horizon} "
                f"(LOW n={int(low[0]['n_origins'])}, HIGH n={int(high[0]['n_origins'])})")
            ax.set_ylabel("CRPS skill versus M1")
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(title="Origin regime")
    fig.suptitle("Relative CRPS skill by volatility regime", y=1.01)
    fig.tight_layout()
    _save(fig, "figure_3_relative_crps_skill")


def figure_4_garch_vs_sv() -> None:
    metrics = _metrics()
    differences = pd.read_csv(OUTPUT / "regime_pairwise_differences.csv")
    differences = differences.loc[
        (differences["comparison"] == "GARCH_minus_Bayesian_SV")
        & differences["regime"].isin(["LOW", "HIGH"])
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharey=False)
    x = np.arange(2)
    for row, factor in enumerate(FACTORS):
        for col, horizon in enumerate(HORIZONS):
            ax = axes[row, col]
            for index, regime in enumerate(("LOW", "HIGH")):
                record = differences.loc[
                    (differences["factor"] == factor)
                    & (differences["horizon"] == horizon)
                    & (differences["regime"] == regime)
                ].iloc[0]
                low_ci = record["bootstrap_ci95_low"]
                high_ci = record["bootstrap_ci95_high"]
                err = np.array([[record["mean_paired_crps_difference"] - low_ci],
                                [high_ci - record["mean_paired_crps_difference"]]])
                ax.errorbar(
                    index, record["mean_paired_crps_difference"], yerr=err,
                    marker="o", capsize=4, color=COLORS[regime],
                    label=f"{regime} (n={int(record['n_origins'])})",
                )
            ax.axhline(0, color="black", lw=0.9)
            ax.set_xticks(x, ["LOW", "HIGH"])
            ax.set_title(f"{factor.title()}, h={horizon}")
            ax.set_ylabel("GARCH CRPS - Bayesian SV CRPS")
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Paired GARCH versus Bayesian SV CRPS differences\n"
                 "(positive favors Bayesian SV; exploratory 95% block-bootstrap intervals)",
                 y=1.03)
    fig.tight_layout()
    _save(fig, "figure_4_garch_vs_bayesian_sv")


def figure_5_pit() -> None:
    forecasts = _forecasts()
    fig, axes = plt.subplots(2, 6, figsize=(17, 7), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 11)
    for row, regime in enumerate(("LOW", "HIGH")):
        for col, (factor, horizon) in enumerate(
                (factor, horizon) for factor in FACTORS for horizon in HORIZONS):
            ax = axes[row, col]
            for model in MODELS:
                group = forecasts.loc[
                    (forecasts["regime"] == regime)
                    & (forecasts["factor"] == factor)
                    & (forecasts["horizon"] == horizon)
                    & (forecasts["manuscript_model"] == model)
                ]
                ax.hist(group["pit"], bins=bins, density=True, histtype="step",
                        lw=1.25, color=COLORS[model], label=SHORT_MODELS[model])
            ax.axhline(1, color="black", ls="--", lw=0.9, label="Uniform")
            n = len(group) if len(group) else 0
            ax.set_title(f"{factor.title()}, h={horizon}\n{regime}, n={n}")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.grid(axis="y", alpha=0.2)
            if col == 0:
                ax.set_ylabel("PIT density")
            if row == 1:
                ax.set_xlabel("PIT")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.suptitle("Conditional PIT distributions; uniform density shown for reference", y=1.1)
    fig.tight_layout()
    _save(fig, "figure_5_calibration_pit")


def figure_5_intervals() -> None:
    metrics = _metrics()
    fig, axes = plt.subplots(4, 6, figsize=(17, 11), sharex="col")
    coverage_levels = np.array([0.50, 0.80, 0.90])
    for factor_index, factor in enumerate(FACTORS):
        for horizon_index, horizon in enumerate(HORIZONS):
            col = factor_index * 2 + horizon_index
            for regime_index, regime in enumerate(("LOW", "HIGH")):
                coverage_ax = axes[regime_index, col]
                width_ax = axes[regime_index + 2, col]
                for model in MODELS:
                    record = _metric_row(metrics, regime, factor, horizon, model)
                    coverages = [record[f"coverage{level}"] for level in INTERVALS]
                    widths = [record[f"mean_width{level}"] for level in INTERVALS]
                    coverage_ax.plot(coverage_levels, coverages, marker="o",
                                     color=COLORS[model], ms=3, lw=1,
                                     label=SHORT_MODELS[model])
                    width_ax.plot(coverage_levels, widths, marker="o",
                                  color=COLORS[model], ms=3, lw=1)
                coverage_ax.plot(coverage_levels, coverage_levels,
                                 color="black", ls="--", lw=0.8)
                coverage_ax.set_ylim(0, 1)
                coverage_ax.set_xticks(coverage_levels, ["50", "80", "90"])
                width_ax.set_xticks(coverage_levels, ["50", "80", "90"])
                coverage_ax.set_title(
                    f"{factor.title()}, h={horizon}\n{regime} (n={int(_metric_row(metrics, regime, factor, horizon, 'M0')['n_origins'])})",
                    fontsize=8,
                )
                if col == 0:
                    coverage_ax.set_ylabel(f"{regime}\nEmpirical coverage")
                    width_ax.set_ylabel(f"{regime}\nMean interval width")
                if regime_index == 1:
                    width_ax.set_xlabel("Nominal coverage (%)")
                coverage_ax.grid(alpha=0.2)
                width_ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.015), fontsize=8)
    fig.suptitle("Predictive interval calibration and width by regime", y=1.06)
    fig.tight_layout()
    _save(fig, "figure_5_calibration_intervals")


def make_all_figures() -> None:
    figure_1_history()
    figure_2_crps()
    figure_3_skill()
    figure_4_garch_vs_sv()
    figure_5_pit()
    figure_5_intervals()


if __name__ == "__main__":
    make_all_figures()
    print(f"Figures saved under {OUTPUT / 'figures'}")
