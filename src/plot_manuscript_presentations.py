"""Create presentation figures with manuscript-only model numbering."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "bayesian_finite_20261027T194239Z" / "rolling_main" / "analysis"
FIGURES = ROOT / "figures" / "finite" / "presentation"
MODEL_LABELS = {"M4": "M3", "M5": "M4"}


def plot_factor_volatility() -> None:
    data = pd.read_csv(ROOT / "figures" / "finite" / "estimated_factor_volatility.csv")
    colors = {"M3": "#c47732", "M4": "#315f78"}
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 3.8), sharex=True)

    for ax, factor in zip(axes, ("level", "slope", "curvature")):
        series = data[data["factor"] == factor].sort_values("origin")
        x = pd.to_datetime(series["origin"])
        ax.fill_between(
            x,
            series["m5_lower_90_bp"],
            series["m5_upper_90_bp"],
            color=colors["M4"],
            alpha=0.18,
            label=f"{MODEL_LABELS['M5']} 90% posterior interval",
        )
        ax.plot(
            x,
            series["m5_median_bp"],
            color=colors["M4"],
            linewidth=1.35,
            label=f"{MODEL_LABELS['M5']} posterior median",
        )
        ax.plot(
            x,
            series["m4_constant_median_bp"],
            color=colors["M3"],
            linestyle="--",
            linewidth=1.15,
            label=f"{MODEL_LABELS['M4']} posterior median (constant)",
        )
        ax.set_ylabel("SD (bp)")
        ax.set_title(f"{factor.capitalize()} factor", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.22)

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        [
            f"{MODEL_LABELS['M5']} 90% interval",
            f"{MODEL_LABELS['M5']} median",
            f"{MODEL_LABELS['M4']} median (constant)",
        ],
        frameon=False,
        ncol=3,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
    )
    axes[-1].set_xlabel("Forecast origin")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Posterior weekly factor innovation volatility", y=1.02, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(FIGURES / "estimated_factor_volatility.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paired_scores() -> None:
    data = pd.read_csv(ANALYSIS / "paired_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    colors = {1: "#1f77b4", 4: "#d95f02"}

    for ax, metric, title in zip(
        axes,
        ("dss", "energy_score"),
        ("Dawid–Sebastiani score", "Energy Score"),
    ):
        subset = data[data["metric"] == metric].sort_values("horizon")
        horizons = subset["horizon"].to_numpy()
        # Internal M5-minus-M4 equals presentation M4-minus-M3.
        estimates = subset["M5_minus_M4"].to_numpy()
        lower = subset["ci95_lower"].to_numpy()
        upper = subset["ci95_upper"].to_numpy()
        y = np.arange(len(subset))
        for yi, h, estimate, lo, hi in zip(y, horizons, estimates, lower, upper):
            ax.errorbar(
                estimate,
                yi,
                xerr=[[estimate - lo], [hi - estimate]],
                fmt="o",
                color=colors[int(h)],
                capsize=3,
                linewidth=1.4,
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_yticks(y, ["1", "4"])
        ax.set_ylabel("Horizon (weeks)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlabel("M4 − M3 (lower favors M4)", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "paired_scores.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_maturity_performance() -> None:
    data = pd.read_csv(ANALYSIS / "per_maturity.csv")
    subset = data[data["horizon"] == 4].copy()
    display_names = MODEL_LABELS
    colors = {"M3": "#1f77b4", "M4": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    for internal_name, group in subset.groupby("model", sort=True):
        display_name = display_names[internal_name]
        group = group.sort_values("maturity_years")
        label = f"{display_name} ({'constant' if display_name == 'M3' else 'stochastic'} volatility)"
        axes[0].plot(
            group["maturity_years"],
            group["rmse"] * 100,
            "-o",
            color=colors[display_name],
            label=label,
        )
        axes[1].plot(
            group["maturity_years"],
            group["coverage_90"] * 100,
            "-o",
            color=colors[display_name],
            label=label,
        )

    axes[1].axhline(90, color="black", linestyle=":", linewidth=0.8)
    axes[0].set(xlabel="Maturity (years)", ylabel="RMSE (basis points)")
    axes[1].set(xlabel="Maturity (years)", ylabel="Central-90% coverage (%)")
    maturity_ticks = [0.25, 0.5, 1, 2, 5, 7, 10, 30]
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(maturity_ticks, [str(value) for value in maturity_ticks])
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "maturity_h4.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_factor_volatility()
    plot_paired_scores()
    plot_maturity_performance()
    print(f"Wrote manuscript presentation figures to {FIGURES}")


if __name__ == "__main__":
    main()
