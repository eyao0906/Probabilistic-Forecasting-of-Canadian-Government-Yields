"""Plot rolling posterior factor innovation volatility from saved chains."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results" / "bayesian_finite_20261027T194239Z"
ROLLING = EXPERIMENT / "rolling_main"
FIGURE = ROOT / "figures" / "finite" / "estimated_factor_volatility.png"
SUMMARY = ROOT / "figures" / "finite" / "estimated_factor_volatility.csv"
FACTORS = ("level", "slope", "curvature")
CHAINS = range(4)
MODELS = ("M4", "M5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chain_directory(origin: Path, forecast: dict) -> Path:
    expected = {
        Path(path).name: checksum
        for path, checksum in forecast["chain_checkpoint_sha256"].items()
    }
    required = {
        f"{model}_{factor}_chain{chain}.npz"
        for model in MODELS
        for factor in FACTORS
        for chain in CHAINS
    }
    if set(expected) != required:
        raise ValueError(f"Unexpected checkpoint set for {origin.name}")

    candidates = [origin / "chains"]
    source = forecast["specification"].get("chain_source")
    if source:
        candidates.append(EXPERIMENT / Path(source).name)

    for candidate in candidates:
        if all((candidate / name).is_file() for name in required) and all(
            _sha256(candidate / name) == expected[name] for name in required
        ):
            return candidate

    raise FileNotFoundError(
        f"No saved chain directory matches the forecast checksums for {origin.name}"
    )


def _volatility_draws(path: Path, model: str) -> np.ndarray:
    column = "h_last" if model == "M5" else "log_variance"
    with np.load(path, allow_pickle=False) as archive:
        draws = archive["draws"]
        report = json.loads(str(archive["report"]))

    names = report.get("names", [])
    if column not in names or draws.ndim != 2 or draws.shape[1] != len(names):
        raise ValueError(f"Missing {column} posterior draws in {path}")
    volatility = np.exp(draws[:, names.index(column)] / 2) * 100
    if not np.isfinite(volatility).all():
        raise ValueError(f"Non-finite volatility draws in {path}")
    return volatility


def summarize_origin(origin: Path, forecast: dict) -> list[dict]:
    chain_dir = _chain_directory(origin, forecast)
    rows = []
    for factor in FACTORS:
        m5 = np.concatenate([
            _volatility_draws(
                chain_dir / f"M5_{factor}_chain{chain}.npz", "M5"
            )
            for chain in CHAINS
        ])
        m4 = np.concatenate([
            _volatility_draws(
                chain_dir / f"M4_{factor}_chain{chain}.npz", "M4"
            )
            for chain in CHAINS
        ])
        lower, median, upper = np.quantile(m5, [0.05, 0.5, 0.95])
        rows.append({
            "origin": forecast["specification"]["origin"],
            "factor": factor,
            "m5_lower_90_bp": float(lower),
            "m5_median_bp": float(median),
            "m5_upper_90_bp": float(upper),
            "m4_constant_median_bp": float(np.median(m4)),
        })
    return rows


def write_summary(rows: list[dict]) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 3.8), sharex=True)
    colors = {"m5": "#315f78", "m4": "#c47732"}

    for ax, factor in zip(axes, FACTORS):
        series = [row for row in rows if row["factor"] == factor]
        x = [np.datetime64(row["origin"]) for row in series]
        median = np.array([row["m5_median_bp"] for row in series])
        lower = np.array([row["m5_lower_90_bp"] for row in series])
        upper = np.array([row["m5_upper_90_bp"] for row in series])
        m4 = np.array([row["m4_constant_median_bp"] for row in series])

        ax.fill_between(
            x, lower, upper, color=colors["m5"], alpha=0.18,
            label="M5 90% posterior interval",
        )
        ax.plot(
            x, median, color=colors["m5"], linewidth=1.35,
            label="M5 posterior median",
        )
        ax.plot(
            x, m4, color=colors["m4"], linestyle="--", linewidth=1.15,
            label="M4 posterior median (constant within fit)",
        )
        ax.set_ylabel("SD (bp)")
        ax.set_title(f"{factor.capitalize()} factor", loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0.22)

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        ["M5 90% interval", "M5 median", "M4 median (constant)"],
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
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    origins = sorted(path for path in (ROLLING / "origins").iterdir() if path.is_dir())
    if len(origins) != 130:
        raise ValueError(f"Expected 130 forecast origins, found {len(origins)}")

    rows = []
    for origin in origins:
        with (origin / "forecast.json").open(encoding="utf-8") as stream:
            forecast = json.load(stream)
        rows.extend(summarize_origin(origin, forecast))

    write_summary(rows)
    plot(rows)
    print(f"Wrote {FIGURE} and {SUMMARY} from {len(origins)} saved origins.")


if __name__ == "__main__":
    main()
