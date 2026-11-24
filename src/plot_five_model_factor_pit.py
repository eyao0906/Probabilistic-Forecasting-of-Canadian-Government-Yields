"""Plot accepted PIT values for all five Canadian factor forecasting models."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTOR_FORECASTS = ROOT / "results" / "factor_probabilistic_20261116" / "factor_forecasts.csv"
FACTOR_METRICS = ROOT / "results" / "factor_probabilistic_20261116" / "factor_metrics.csv"
GARCH_FORECASTS = ROOT / "results" / "factor_garch_20261119" / "garch_factor_forecasts.csv"
GARCH_METRICS = ROOT / "results" / "factor_garch_20261119" / "garch_factor_metrics.csv"
OUTPUT = ROOT / "paper" / "generated" / "figures" / "factor_pit.png"

FACTORS = ("level", "slope", "curvature")
HORIZONS = (1, 4)
MODELS = ("M0", "M1", "M2", "M3", "AR-GARCH")
INTERNAL_TO_MANUSCRIPT = {
    "M0": "M0",
    "M2": "M1",
    "M3": "M2",
    "M4": "M3",
    "AR1_GARCH11": "AR-GARCH",
}
PIT_BINS = np.linspace(0.0, 1.0, 11)
MODEL_STYLES = {
    "M0": {"color": "#6b7280", "linestyle": "-"},
    "M1": {"color": "#2563eb", "linestyle": "-"},
    "M2": {"color": "#16a34a", "linestyle": "-"},
    "M3": {"color": "#dc2626", "linestyle": "-"},
    "AR-GARCH": {"color": "#7c3aed", "linestyle": "--"},
}
KEY = ["model", "factor", "horizon", "origin"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_metrics(records: pd.DataFrame) -> None:
    factors = pd.read_csv(FACTOR_METRICS)
    garch = pd.read_csv(GARCH_METRICS)
    inverse = {internal: manuscript
               for internal, manuscript in INTERNAL_TO_MANUSCRIPT.items()}
    factors["model"] = factors["model"].map(inverse)
    garch["model"] = "AR-GARCH"
    metrics = pd.concat([factors, garch], ignore_index=True)
    _require(len(metrics) == len(MODELS) * len(FACTORS) * len(HORIZONS),
             "Saved model-level PIT metric groups are incomplete")
    _require(not metrics.duplicated(["model", "factor", "horizon"]).any(),
             "Saved PIT metric groups contain duplicate keys")
    for metric in metrics.itertuples(index=False):
        subset = records.loc[
            (records["model"] == metric.model)
            & (records["factor"] == metric.factor)
            & (records["horizon"] == int(metric.horizon))
        ]
        _require(int(metric.n_origins) == 130 and len(subset) == 130,
                 f"Expected 130 PIT values for {metric.model}/{metric.factor}/h={metric.horizon}")
        observed_bins = np.histogram(subset["pit"].to_numpy(), bins=PIT_BINS)[0]
        saved_bins = np.array([getattr(metric, f"pit_bin_{index}") for index in range(1, 11)],
                              dtype=int)
        _require(np.array_equal(observed_bins, saved_bins),
                 f"Saved PIT histogram counts disagree for "
                 f"{metric.model}/{metric.factor}/h={metric.horizon}")


def load_pit_records() -> pd.DataFrame:
    """Load saved PIT columns, map labels, and validate the common forecast design."""
    factor = pd.read_csv(FACTOR_FORECASTS)
    garch = pd.read_csv(GARCH_FORECASTS)
    needed = {"model", "factor", "origin", "target", "horizon", "observed", "pit"}
    _require(needed.issubset(factor.columns) and needed.issubset(garch.columns),
             "Accepted forecast CSVs do not contain the required PIT fields")
    _require(set(factor["model"]) == set(INTERNAL_TO_MANUSCRIPT) - {"AR1_GARCH11"},
             "Accepted factor forecast model identifiers have changed")
    _require(set(garch["model"]) == {"AR1_GARCH11"},
             "Accepted GARCH forecast identifier has changed")
    factor["model"] = factor["model"].map(INTERNAL_TO_MANUSCRIPT)
    garch["model"] = garch["model"].map(INTERNAL_TO_MANUSCRIPT)
    records = pd.concat([factor, garch], ignore_index=True, sort=False)
    records = records.loc[:, ["model", "factor", "horizon", "origin",
                              "target", "observed", "pit"]].copy()
    records["horizon"] = pd.to_numeric(records["horizon"], errors="raise").astype(int)
    records["observed"] = pd.to_numeric(records["observed"], errors="raise")
    records["pit"] = pd.to_numeric(records["pit"], errors="raise")

    _require(len(records) == 3900, "Expected exactly 3,900 saved forecast PIT values")
    _require(set(records["model"]) == set(MODELS)
             and set(records["factor"]) == set(FACTORS)
             and set(records["horizon"]) == set(HORIZONS),
             "PIT records do not cover the expected model/factor/horizon design")
    _require(not records.duplicated(KEY).any(),
             "PIT records contain duplicate model/factor/horizon/origin keys")
    _require(np.isfinite(records["pit"].to_numpy()).all()
             and records["pit"].between(0.0, 1.0).all(),
             "PIT values must be finite and lie in [0,1]")
    counts = records.groupby(["model", "factor", "horizon"]).size()
    _require(len(counts) == 30 and (counts == 130).all(),
             "Every model/factor/horizon group must contain 130 PIT values")
    _require(records["origin"].nunique() == 130
             and records["origin"].min() == "2016-01-08"
             and records["origin"].max() == "2025-11-28",
             "Accepted forecast origins do not match the common design")

    aligned = records.groupby(["factor", "horizon", "origin"], sort=False).agg(
        target_count=("target", "nunique"),
        observed_min=("observed", "min"),
        observed_max=("observed", "max"),
        model_count=("model", "nunique"),
    )
    _require((aligned["target_count"] == 1).all()
             and (aligned["model_count"] == 5).all()
             and np.allclose(aligned["observed_min"], aligned["observed_max"],
                             rtol=0.0, atol=1e-10),
             "Forecast target dates or realized factor values differ across models")
    _validate_metrics(records)
    return records.sort_values(
        ["factor", "horizon", "origin", "model"],
        key=lambda column: column.map(
            {name: index for index, name in enumerate(MODELS)}
        ) if column.name == "model" else column,
    ).reset_index(drop=True)


def build_pit_figure(records: pd.DataFrame) -> tuple[Figure, np.ndarray]:
    """Build the 3-by-2 density histograms with common bins and a shared legend."""
    _require(len(records) == 3900 and not records.duplicated(KEY).any(),
             "Figure input must be the validated 3,900-row PIT table")
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 5.7), sharex=True, sharey=True)
    for row, factor in enumerate(FACTORS):
        for column, horizon in enumerate(HORIZONS):
            ax = axes[row, column]
            group = records.loc[
                (records["factor"] == factor) & (records["horizon"] == horizon)
            ]
            for model in MODELS:
                values = group.loc[group["model"] == model, "pit"].to_numpy()
                _require(len(values) == 130,
                         f"Incomplete PIT histogram: {model}/{factor}/h={horizon}")
                _, _, patches = ax.hist(
                    values,
                    bins=PIT_BINS,
                    density=True,
                    histtype="step",
                    linewidth=1.55,
                    **MODEL_STYLES[model],
                )
                patches[0].set_label(model)
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", zorder=0)
            ax.set_title(f"{factor.title()}, h={horizon} (n=130)", fontsize=12)
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks(np.linspace(0.0, 1.0, 6))
            ax.tick_params(labelsize=11)
            ax.grid(axis="y", alpha=0.18, linewidth=0.6)
            if column == 0:
                ax.set_ylabel("Density", fontsize=12)
            if row == len(FACTORS) - 1:
                ax.set_xlabel("PIT", fontsize=12)
    handles = [
        Line2D([0], [0], color=MODEL_STYLES[model]["color"],
               linestyle=MODEL_STYLES[model]["linestyle"], linewidth=1.7)
        for model in MODELS
    ]
    fig.legend(handles, MODELS, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.985), frameon=False, fontsize=10.5)
    fig.suptitle("PIT histograms by factor and forecast horizon",
                 fontsize=13.5, y=1.02)
    fig.tight_layout(rect=(0.04, 0.02, 0.995, 0.92), h_pad=1.2, w_pad=0.8)
    return fig, axes


def write_pit_figure(records: pd.DataFrame, output: Path = OUTPUT) -> Path:
    """Write deterministic PNG and vector PDF versions of the five-model figure."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = build_pit_figure(records)
    try:
        fig.savefig(output, dpi=300, bbox_inches="tight", metadata={"Software": "Matplotlib"})
        pdf_path = output.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight",
                    metadata={"Creator": "Matplotlib", "CreationDate": None,
                              "ModDate": None})
    finally:
        plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    records = load_pit_records()
    path = write_pit_figure(records, args.output)
    print(f"Wrote {len(records)} validated PIT observations to {path}")
    print(f"Wrote vector figure to {path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
