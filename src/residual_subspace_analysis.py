"""Scoring, dependence-aware inference, and plots for residual-space forecasts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import chi2, norm


def energy_score(scenarios: np.ndarray, observed: np.ndarray) -> float:
    """Compute the ensemble Energy Score with the unbiased distinct-pair term."""
    scenarios = np.asarray(scenarios, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if (scenarios.ndim != 2 or observed.shape != (scenarios.shape[1],)
        or scenarios.shape[0] < 2 or not np.isfinite(scenarios).all()
        or not np.isfinite(observed).all()):
        raise ValueError("Invalid Energy Score inputs")
    first = np.linalg.norm(scenarios - observed, axis=1).mean()
    return float(first - 0.5 * pdist(scenarios).mean())


def stationary_bootstrap_indices(
    n: int, block_length: int, repetitions: int, rng: np.random.Generator
) -> np.ndarray:
    if n < 2 or not 1 <= block_length <= n or repetitions < 2:
        raise ValueError("Invalid stationary-bootstrap dimensions")
    indices = np.empty((repetitions, n), dtype=np.int32)
    indices[:, 0] = rng.integers(n, size=repetitions)
    restart_probability = 1.0 / block_length
    for j in range(1, n):
        restart = rng.random(repetitions) < restart_probability
        fresh = rng.integers(n, size=repetitions)
        indices[:, j] = np.where(restart, fresh, (indices[:, j - 1] + 1) % n)
    return indices


def _metric_estimates(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    sampled = {}
    for metric in ("dss", "energy_score"):
        sampled[metric] = np.asarray(data[metric])[indices].mean(axis=1)
    for metric in ("squared_errors", "coverage_90"):
        values = np.asarray(data[metric])[indices]
        pooled = values.mean(axis=(1, 2))
        sampled["rmse" if metric == "squared_errors" else "coverage_90"] = (
            np.sqrt(pooled) if metric == "squared_errors" else pooled
        )
    return sampled


def paired_bootstrap(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict:
    """Summarize right-minus-left paired differences for all forecast metrics."""
    required = ("dss", "energy_score", "squared_errors", "coverage_90")
    n = len(np.asarray(left["dss"]))
    if any(
        np.asarray(left[key]).shape[0] != n
        or np.asarray(right[key]).shape != np.asarray(left[key]).shape
        or not np.isfinite(left[key]).all()
        or not np.isfinite(right[key]).all()
        for key in required
    ) or indices.ndim != 2 or indices.shape[1] != n:
        raise ValueError("Unmatched or nonfinite bootstrap inputs")
    left_boot = _metric_estimates(left, indices)
    right_boot = _metric_estimates(right, indices)
    result = {}
    for metric in ("dss", "energy_score", "rmse", "coverage_90"):
        delta = right_boot[metric] - left_boot[metric]
        point_left = _metric_estimates(left, np.arange(n)[None, :])[metric][0]
        point_right = _metric_estimates(right, np.arange(n)[None, :])[metric][0]
        result[metric] = {
            "left": float(point_left),
            "right": float(point_right),
            "right_minus_left": float(point_right - point_left),
            "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "bootstrap_se": float(np.std(delta, ddof=1)),
        }
    return result


def forecast_arrays(records: list[dict], model: str, horizon: int) -> dict[str, np.ndarray]:
    rows = sorted(
        (row for row in records if row["model"] == model and row["horizon"] == horizon),
        key=lambda row: row["origin"],
    )
    if len(rows) != 130 or len({row["origin"] for row in rows}) != 130:
        raise ValueError(f"Expected 130 distinct records for {model}, horizon {horizon}")
    return {
        "dss": np.asarray([row["dss"] for row in rows], dtype=float),
        "energy_score": np.asarray([row["energy_score"] for row in rows], dtype=float),
        "squared_errors": np.asarray([row["squared_errors"] for row in rows], dtype=float),
        "absolute_errors": np.asarray([row["absolute_errors"] for row in rows], dtype=float),
        "coverage_90": np.asarray([row["coverage_90"] for row in rows], dtype=float),
        "interval_width_90": np.asarray([row["interval_width_90"] for row in rows], dtype=float),
    }


def summarize_forecasts(records: list[dict], repetitions: int = 9999, seed: int = 20261116) -> dict:
    if len(records) != 780:
        raise ValueError("Require 520 corrected forecasts and 260 matched random-walk forecasts")
    result = {"n_origins": 130, "repetitions": repetitions, "horizons": {}}
    for h in (1, 4):
        arrays = {model: forecast_arrays(records, model, h) for model in ("M4", "M5", "M0")}
        estimates = {}
        for model, values in arrays.items():
            estimates[model] = {
                "dss": float(values["dss"].mean()),
                "energy_score": float(values["energy_score"].mean()),
                "rmse": float(np.sqrt(values["squared_errors"].mean())),
                "mae": float(values["absolute_errors"].mean()),
                "coverage_90": float(values["coverage_90"].mean()),
                "mean_interval_width_90": float(values["interval_width_90"].mean()),
            }
        comparisons = {}
        for pair_index, (left_name, right_name) in enumerate((("M4", "M5"), ("M0", "M4"), ("M0", "M5"))):
            sensitivity = {}
            for block_length in (8, 4, 12):
                indices = stationary_bootstrap_indices(
                    130, block_length, repetitions,
                    np.random.default_rng(seed + 1000 * h + 100 * block_length + pair_index),
                )
                sensitivity[str(block_length)] = {
                    "block_length_origins": block_length,
                    **paired_bootstrap(arrays[left_name], arrays[right_name], indices),
                }
            comparisons[f"{left_name}_to_{right_name}"] = sensitivity
        maturity = []
        for j, tau in enumerate((0.25, 0.5, 1, 2, 5, 7, 10, 30)):
            row = {"maturity_years": tau}
            for model, values in arrays.items():
                row[model] = {
                    "rmse": float(np.sqrt(values["squared_errors"][:, j].mean())),
                    "mae": float(values["absolute_errors"][:, j].mean()),
                    "coverage_90": float(values["coverage_90"][:, j].mean()),
                    "mean_interval_width_90": float(values["interval_width_90"][:, j].mean()),
                }
            maturity.append(row)
        result["horizons"][str(h)] = {
            "model_means": estimates,
            "paired_comparisons": comparisons,
            "per_maturity": maturity,
        }
    return result


def residual_calibration(records: list[dict]) -> dict:
    output = {}
    z90 = float(norm.ppf(0.95))
    for h in (1, 4):
        residual_rows = sorted(
            (row for row in records if row["model"] == "M4" and row["horizon"] == h),
            key=lambda row: row["origin"],
        )
        observed = np.asarray([row["residual_target_coordinates"] for row in residual_rows])
        scales = np.asarray([row["residual_marginal_sd"] for row in residual_rows])
        whitened = np.asarray([row["whitened_residual_coordinates"] for row in residual_rows])
        mahalanobis = np.asarray([row["residual_mahalanobis_squared"] for row in residual_rows])
        standardized = observed / scales
        by_model = {}
        for model in ("M4", "M5"):
            rows = sorted(
                (row for row in records if row["model"] == model and row["horizon"] == h),
                key=lambda row: row["origin"],
            )
            factor_errors = np.asarray([row["factor_target_error"] for row in rows])
            oos_corr = np.asarray([
                [np.corrcoef(factor_errors[:, j], observed[:, k])[0, 1]
                 for k in range(observed.shape[1])]
                for j in range(factor_errors.shape[1])
            ])
            by_model[model] = {
                "oos_factor_error_residual_coordinate_correlations": oos_corr.tolist(),
                "max_abs_oos_factor_error_residual_correlation": float(np.max(np.abs(oos_corr))),
            }
        training_corr = np.asarray([
            row["training_factor_residual_correlation"] for row in residual_rows
        ])
        output[str(h)] = {
            "n": len(residual_rows),
            "coordinate_mean": observed.mean(axis=0).tolist(),
            "coordinate_sd": observed.std(axis=0, ddof=1).tolist(),
            "standardized_mean": standardized.mean(axis=0).tolist(),
            "standardized_variance": standardized.var(axis=0, ddof=1).tolist(),
            "marginal_90_coverage_by_coordinate": (
                np.mean(np.abs(standardized) <= z90, axis=0).tolist()
            ),
            "joint_mahalanobis_squared_mean": float(mahalanobis.mean()),
            "joint_mahalanobis_squared_90_coverage": float(
                np.mean(mahalanobis <= chi2.ppf(0.90, 5))
            ),
            "joint_mahalanobis_squared_95_coverage": float(
                np.mean(mahalanobis <= chi2.ppf(0.95, 5))
            ),
            "max_abs_lag1_autocorrelation": float(
                max(abs(np.corrcoef(whitened[:-1, j], whitened[1:, j])[0, 1])
                    for j in range(whitened.shape[1]))
            ),
            "mean_training_factor_residual_correlation": training_corr.mean(axis=0).tolist(),
            "max_abs_mean_training_factor_residual_correlation": float(
                np.max(np.abs(training_corr.mean(axis=0)))
            ),
            "max_abs_oos_factor_error_residual_correlation": max(
                item["max_abs_oos_factor_error_residual_correlation"]
                for item in by_model.values()
            ),
            "factor_residual_dependence_by_model": by_model,
        }
    return output


def make_figures(records: list[dict], output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for h in (1, 4):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for model, color in (("M4", "#1f77b4"), ("M5", "#d62728")):
            rows = sorted(
                (r for r in records if r["model"] == model and r["horizon"] == h),
                key=lambda r: r["origin"],
            )
            ax.plot(
                [r["origin"] for r in rows],
                [r["dss_original"] - r["dss"] for r in rows],
                label=model, color=color, linewidth=1.2,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title=f"DSS change after residual-space correction, h={h}",
               ylabel="Original DSS − corrected DSS", xlabel="Forecast origin")
        ax.tick_params(axis="x", labelrotation=35)
        ax.legend()
        fig.tight_layout()
        path = output / f"dss_change_h{h}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, h in zip(axes, (1, 4)):
        rows = [r for r in records if r["model"] == "M4" and r["horizon"] == h]
        old = np.asarray([r["original_eigenvalues"] for r in rows])
        new = np.asarray([r["eigenvalues_descending"] for r in rows])
        positions = np.arange(1, old.shape[1] + 1)
        ax.plot(positions, np.median(old, axis=0), marker="o", label="Original diagonal")
        ax.plot(positions, np.median(new, axis=0), marker="o", label="Residual subspace")
        ax.set(title=f"h={h}", xlabel="Eigenvalue rank", yscale="log")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Median covariance eigenvalue")
    axes[1].legend()
    fig.tight_layout()
    path = output / "covariance_spectra.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path.name)
    return paths
