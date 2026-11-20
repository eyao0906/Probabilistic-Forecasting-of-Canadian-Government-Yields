"""Export and plot the accepted fixed-loading DNS factor histories."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.bayesian_stage_a import FACTORS, make_design
from src.garch_factor_experiment import LAMBDA

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "factor_probabilistic_20261116"
START = date(2000, 1, 1)
END = date(2025, 12, 31)
OOS_START = date(2016, 1, 1)
OUTPUT_COLUMNS = ("Date", "Level", "Slope", "Curvature")


def factor_history_frame(design: dict) -> pd.DataFrame:
    if float(design["lambda"]) != LAMBDA:
        raise ValueError("Factor design does not use the fixed accepted decay parameter")
    dates = design["dates"]
    factors = np.asarray(design["factors"], dtype=float)
    if factors.shape != (len(dates), 3) or not np.isfinite(factors).all():
        raise ValueError("Expected finite level, slope, and curvature values for every date")
    indices = [i for i, day in enumerate(dates) if START <= day <= END]
    if (len(indices) != 1356 or dates[indices[0]].isoformat() != "2000-01-07"
            or dates[indices[-1]].isoformat() != "2025-12-24"):
        raise ValueError("Extracted research history does not match the fixed 2000-2025 panel")
    values = factors[indices]
    return pd.DataFrame({
        "Date": [dates[i].isoformat() for i in indices],
        "Level": values[:, FACTORS.index("level")],
        "Slope": values[:, FACTORS.index("slope")],
        "Curvature": values[:, FACTORS.index("curvature")],
    }, columns=OUTPUT_COLUMNS)


def plot_factor_histories(frame: pd.DataFrame, output: Path = OUTPUT) -> tuple[Path, Path, Path]:
    if tuple(frame.columns) != OUTPUT_COLUMNS or len(frame) != 1356:
        raise ValueError("Plot input must be the complete fixed factor-history table")
    dates = pd.to_datetime(frame["Date"]).to_numpy()
    values = frame.loc[:, ["Level", "Slope", "Curvature"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Plot input contains non-finite factor values")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "extracted_dns_factors.csv"
    png_path = output / "extracted_factor_histories.png"
    pdf_path = output / "extracted_factor_histories.pdf"
    frame.to_csv(csv_path, index=False, float_format="%.17g")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.3), sharex=True)
    line_color = "#315f78"
    evaluation_color = "#eef2f4"
    for ax, factor, column in zip(axes, FACTORS, OUTPUT_COLUMNS[1:]):
        ax.axvspan(np.datetime64(OOS_START), dates[-1], color=evaluation_color, zorder=0)
        ax.axvline(np.datetime64(OOS_START), color="#4b5563", linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=3)
        ax.plot(dates, frame[column].to_numpy(dtype=float), color=line_color, linewidth=0.85)
        ax.set_ylabel(f"{factor.title()} (pp)")
        ax.grid(axis="y", color="#6b7280", alpha=0.22, linewidth=0.6)
        ax.margins(x=0.01)
    axes[0].text(0.25, 0.92, "Initial estimation (2000-2015)",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=8)
    axes[0].text(0.78, 0.92, "Out-of-sample evaluation (2016-2025)",
                 transform=axes[0].transAxes, ha="center", va="top", fontsize=8)
    axes[-1].set_xlabel("Week")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(
        "Historical Dynamic Nelson-Siegel Factors for Canadian Government Yields, 2000-2025",
        fontsize=12, y=0.975)
    fig.legend(
        handles=[
            Line2D([0], [0], color=line_color, linewidth=1.1,
                   label="Cross-sectional least-squares factor"),
            Patch(facecolor=evaluation_color, edgecolor="none",
                  label="Out-of-sample evaluation period"),
            Line2D([0], [0], color="#4b5563", linewidth=1.0,
                   linestyle=(0, (4, 3)), label="2016 evaluation start"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3,
        frameon=False, fontsize=8,
    )
    fig.text(
        0.09, 0.015,
        "Weekly level, slope, and curvature factors extracted by cross-sectional least squares "
        "from eight Canadian government zero-coupon yield maturities using fixed "
        "lambda=0.5160. The dashed line separates the 2000-2015 initial estimation period "
        "from 2016-2025 out-of-sample forecast evaluation. Factor values are in percentage points.",
        ha="left", va="bottom", fontsize=8, wrap=True,
    )
    fig.subplots_adjust(left=0.105, right=0.98, top=0.88, bottom=0.145, hspace=0.22)
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)
    return csv_path, png_path, pdf_path


def main() -> None:
    design = make_design()
    frame = factor_history_frame(design)
    paths = plot_factor_histories(frame)
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise RuntimeError("Factor history output was not written completely")
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
