"""Conditional evaluation of accepted Canadian factor forecasts by volatility regime."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.models import ns_loadings
from src.scores import stationary_bootstrap_indices

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "canadian_volatility_regime_20261120"
FACTOR_PATH = ROOT / "figures" / "factor_probabilistic_20261116" / "extracted_dns_factors.csv"
FACTOR_FORECAST_PATH = ROOT / "results" / "factor_probabilistic_20261116" / "factor_forecasts.csv"
FACTOR_METRICS_PATH = ROOT / "results" / "factor_probabilistic_20261116" / "factor_metrics.csv"
GARCH_FORECAST_PATH = ROOT / "results" / "factor_garch_20261119" / "garch_factor_forecasts.csv"
GARCH_METRICS_PATH = ROOT / "results" / "factor_garch_20261119" / "garch_factor_metrics.csv"
PANEL_PATH = ROOT / "data" / "processed" / "canada_weekly.csv"

FACTORS = ("level", "slope", "curvature")
FACTOR_COLUMNS = ("Level", "Slope", "Curvature")
HORIZONS = (1, 4)
MODELS = ("M0", "M1", "M2", "M3", "AR1_GARCH11")
MODEL_MAP = {"M0": "M0", "M1": "M2", "M2": "M3", "M3": "M4"}
MODEL_ORDER = {name: index for index, name in enumerate(MODELS)}
FACTORS_ORDER = {name: index for index, name in enumerate(FACTORS)}
PAIRWISE = {
    "Bayesian_SV_minus_Bayesian_constant_volatility": ("M3", "M2"),
    "GARCH_minus_classical_AR1": ("AR1_GARCH11", "M1"),
    "GARCH_minus_Bayesian_SV": ("AR1_GARCH11", "M3"),
}
REGIMES = ("LOW", "MEDIUM", "HIGH")
REGIME_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "ALL": 3}
INTERVALS = (50, 80, 90)
LOOKBACK = 26
TRAIN_START = "2000-01-07"
TRAIN_END = "2015-12-31"
OOS_START = "2016-01-01"
OOS_END = "2025-12-31"
LAMBDA = 0.5160377358490565
MATURITIES = np.array([0.25, 0.5, 1, 2, 5, 7, 10, 30], dtype=float)
YIELD_COLUMNS = ("y0.25", "y0.5", "y1", "y2", "y5", "y7", "y10", "y30")
DEFAULT_BOOTSTRAP_REPS = 9999
DEFAULT_BOOTSTRAP_SEED = 20261120


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_factor_history() -> tuple[pd.DataFrame, dict]:
    """Load accepted factors and check dates and values against the fixed-lambda panel."""
    history = pd.read_csv(FACTOR_PATH, parse_dates=["Date"])
    _require(list(history.columns) == ["Date", *FACTOR_COLUMNS],
             "Accepted factor inventory has unexpected columns")
    _require(len(history) == 1356, "Accepted factor history must contain 1,356 rows")
    _require(history["Date"].is_monotonic_increasing and history["Date"].is_unique,
             "Accepted factor dates must be unique and ascending")
    _require(history["Date"].iloc[0].date().isoformat() == TRAIN_START
             and history["Date"].iloc[-1].date().isoformat() == "2025-12-24",
             "Accepted factor history has unexpected sample boundaries")
    values = history.loc[:, FACTOR_COLUMNS].to_numpy(dtype=float)
    _require(np.isfinite(values).all(), "Accepted factors contain non-finite values")

    panel = pd.read_csv(PANEL_PATH, parse_dates=["Date"])
    _require(set(["Date", *YIELD_COLUMNS]).issubset(panel.columns),
             "Yield panel is missing expected maturity columns")
    research_panel = panel.loc[
        panel["Date"].between("2000-01-01", "2025-12-31")
    ].reset_index(drop=True)
    _require(len(research_panel) == len(history), "Yield and factor samples differ in size")
    _require(np.array_equal(research_panel["Date"].to_numpy(), history["Date"].to_numpy()),
             "Accepted factor dates do not match the yield panel")
    _require(panel["Date"].iloc[0].date().isoformat() == "1999-12-31",
             "Expected excluded 1999-12-31 panel row is absent or displaced")
    loadings = ns_loadings(MATURITIES, LAMBDA)
    reconstructed = np.linalg.lstsq(
        loadings,
        research_panel.loc[:, YIELD_COLUMNS].to_numpy(dtype=float).T,
        rcond=None,
    )[0].T
    max_factor_difference = float(np.max(np.abs(reconstructed - values)))
    _require(np.allclose(reconstructed, values, rtol=1e-11, atol=1e-11),
             "Accepted factors do not reproduce the fixed-loading yield extraction")
    history = history.rename(columns=dict(zip(FACTOR_COLUMNS, FACTORS)))
    history.loc[:, "research_row_index"] = np.arange(len(history), dtype=int)
    history.loc[:, "yield_panel_row_index_zero_based"] = np.arange(1, len(history) + 1)
    return history, {
        "factor_rows": len(history),
        "panel_rows": len(panel),
        "factor_dates_match_panel": True,
        "max_factor_reconstruction_abs_difference": max_factor_difference,
        "excluded_1999_observation_not_used": True,
    }


def rolling_standardized_rms(
    factors: np.ndarray, scales: np.ndarray, lookback: int = LOOKBACK
) -> tuple[np.ndarray, np.ndarray]:
    """Return factor RMS values and their equal-weight composite at each date."""
    values = np.asarray(factors, dtype=float)
    scale = np.asarray(scales, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or scale.shape != (3,)
            or lookback < 1 or not np.isfinite(values).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Expected finite three-factor data and positive training scales")
    changes = np.diff(values, axis=0)
    factor_rms = np.full(values.shape, np.nan, dtype=float)
    for row in range(lookback, len(values)):
        window = changes[row - lookback:row] / scale
        factor_rms[row] = np.sqrt(np.mean(window**2, axis=0))
    composite = np.sqrt(np.mean(factor_rms**2, axis=1))
    return factor_rms, composite


def construct_regimes(
    history: pd.DataFrame, train_end_index: int, lookback: int = LOOKBACK
) -> tuple[pd.DataFrame, dict]:
    """Freeze training-only change scales and thresholds, then label every eligible date."""
    values = history.loc[:, FACTORS].to_numpy(dtype=float)
    if not 1 < train_end_index <= len(values):
        raise ValueError("Training end index must identify a non-trivial history prefix")
    training_changes = np.diff(values[:train_end_index], axis=0)
    scales = np.std(training_changes, axis=0, ddof=1)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Training factor-change scales must be finite and positive")
    factor_rms, composite = rolling_standardized_rms(values, scales, lookback)
    train_composite = composite[:train_end_index]
    train_composite = train_composite[np.isfinite(train_composite)]
    if len(train_composite) == 0:
        raise ValueError("No complete training-period volatility windows")
    q25, q75 = np.quantile(train_composite, [0.25, 0.75], method="linear")
    labels = np.full(len(values), None, dtype=object)
    eligible = np.isfinite(composite)
    labels[eligible & (composite <= q25)] = "LOW"
    labels[eligible & (composite >= q75)] = "HIGH"
    labels[eligible & (composite > q25) & (composite < q75)] = "MEDIUM"
    inventory = history.copy()
    for index, factor in enumerate(FACTORS):
        inventory[f"volatility_{factor}"] = factor_rms[:, index]
    inventory["composite_volatility"] = composite
    inventory["q25"] = float(q25)
    inventory["q75"] = float(q75)
    inventory["regime"] = labels
    thresholds = {
        "method": "26-week standardized factor-change rolling RMS",
        "factor_order": list(FACTORS),
        "lookback_weekly_changes": lookback,
        "normalization": "sample SD (ddof=1) of within-training-sample weekly factor changes",
        "training_start": history["Date"].iloc[0].date().isoformat(),
        "training_end": history["Date"].iloc[train_end_index - 1].date().isoformat(),
        "training_observation_count": train_end_index,
        "training_change_count": len(training_changes),
        "threshold_training_window_count": len(train_composite),
        "quantile_method": "NumPy linear interpolation (method='linear')",
        "low_rule": "V_t <= q25",
        "high_rule": "V_t >= q75",
        "medium_rule": "q25 < V_t < q75",
        "normalization_factors": dict(zip(FACTORS, map(float, scales))),
        "q25": float(q25),
        "q75": float(q75),
        "composite_formula": "sqrt((V_level^2 + V_slope^2 + V_curvature^2)/3)",
        "information_cutoff": "Each index value uses factor observations dated through its own date",
    }
    return inventory, thresholds


def _read_accepted_forecasts(history: pd.DataFrame) -> pd.DataFrame:
    factors = pd.read_csv(FACTOR_FORECAST_PATH)
    garch = pd.read_csv(GARCH_FORECAST_PATH)
    factors["internal_model"] = factors["model"]
    inverse = {internal: manuscript for manuscript, internal in MODEL_MAP.items()}
    factors["manuscript_model"] = factors["internal_model"].map(inverse)
    factors = factors.loc[factors["manuscript_model"].notna()].copy()
    garch["internal_model"] = "AR1_GARCH11"
    garch["manuscript_model"] = "AR1_GARCH11"
    forecasts = pd.concat([factors, garch], ignore_index=True, sort=False)
    forecasts["horizon"] = pd.to_numeric(forecasts["horizon"], errors="raise").astype(int)
    key = ["origin", "target", "factor", "horizon", "manuscript_model"]
    _require(len(forecasts) == 3900, "Combined forecast record count is not 3,900")
    _require(set(forecasts["manuscript_model"]) == set(MODELS),
             "Combined forecasts do not contain the five expected model labels")
    _require(set(forecasts["factor"]) == set(FACTORS)
             and set(forecasts["horizon"]) == set(HORIZONS),
             "Combined forecasts have unexpected factor or horizon values")
    _require(not forecasts.duplicated(key).any(), "Forecast join key is not unique")

    history_dates = history["Date"].dt.strftime("%Y-%m-%d").tolist()
    row_by_date = {day: index for index, day in enumerate(history_dates)}
    origin_rows = []
    expected_observed = []
    for row in forecasts.itertuples(index=False):
        if row.origin not in row_by_date:
            raise ValueError(f"Unknown forecast origin {row.origin}")
        origin_index = row_by_date[row.origin]
        target_index = origin_index + int(row.horizon)
        if target_index >= len(history_dates) or history_dates[target_index] != row.target:
            raise ValueError(f"Origin/target alignment mismatch for {row.origin}, h={row.horizon}")
        if row.factor not in FACTORS:
            raise ValueError(f"Unexpected forecast factor {row.factor}")
        origin_rows.append(origin_index)
        expected_observed.append(float(history.at[target_index, row.factor]))
    forecasts["origin_row_index"] = origin_rows
    expected_observed = np.asarray(expected_observed)
    observed = pd.to_numeric(forecasts["observed"], errors="raise").to_numpy()
    _require(np.allclose(observed, expected_observed, rtol=1e-9, atol=1e-10),
             "Forecast realized values do not match accepted factor history")
    forecasts["observed"] = observed
    forecasts["model_mean"] = pd.to_numeric(forecasts["mean"], errors="raise")
    required_numeric = ["observed", "model_mean", "variance", "crps", "crps_mcse",
                        "dss", "mae", "pit", "pit_mcse"]
    for column in required_numeric:
        forecasts[column] = pd.to_numeric(forecasts[column], errors="raise")
    _require(np.isfinite(forecasts[required_numeric].to_numpy(dtype=float)).all(),
             "Forecasts contain non-finite predictive moments or scores")
    _require((forecasts["variance"] > 0).all() and (forecasts["crps"] >= 0).all(),
             "Forecast variances or CRPS values are invalid")
    _require(forecasts["pit"].between(0, 1).all(), "PIT values must be in [0,1]")
    errors = forecasts["observed"] - forecasts["model_mean"]
    _require(np.allclose(forecasts["dss"], np.log(forecasts["variance"])
                         + errors**2 / forecasts["variance"], rtol=1e-8, atol=1e-9),
             "Accepted DSS values do not match their stored moments")
    _require(np.allclose(forecasts["mae"], np.abs(errors), rtol=1e-8, atol=1e-9),
             "Accepted MAE values do not match their stored forecast means")
    forecasts["squared_error"] = errors**2
    if "rmse_component" in forecasts:
        component = pd.to_numeric(forecasts["rmse_component"], errors="coerce")
        present = component.notna()
        _require(np.allclose(component[present], forecasts.loc[present, "squared_error"],
                             rtol=1e-8, atol=1e-9),
                 "Accepted squared-error components do not match forecast means")
    for level in INTERVALS:
        low, high, width, covered = (
            f"lower{level}", f"upper{level}", f"width{level}", f"covered{level}")
        for column in (low, high, width, covered):
            forecasts[column] = pd.to_numeric(forecasts[column], errors="raise")
        _require((forecasts[low] <= forecasts[high]).all()
                 and np.isfinite(forecasts[[low, high, width]].to_numpy()).all(),
                 f"Invalid {level}% interval endpoints")
        _require(np.allclose(forecasts[width], forecasts[high] - forecasts[low],
                             rtol=1e-8, atol=1e-9),
                 f"Stored {level}% interval widths are inconsistent")
        actual_covered = ((forecasts[low] <= forecasts["observed"])
                          & (forecasts["observed"] <= forecasts[high])).astype(int)
        _require((forecasts[covered].astype(int).to_numpy() == actual_covered.to_numpy()).all(),
                 f"Stored {level}% coverage flags are inconsistent")

    counts = forecasts.groupby(["origin", "factor", "horizon"]).size()
    _require(len(counts) == 130 * 3 * 2 and (counts == 5).all(),
             "Forecast panel does not have five models per origin/factor/horizon")
    observed_spread = forecasts.groupby(["origin", "factor", "horizon"])["observed"].agg(
        lambda values: float(values.max() - values.min()))
    _require(float(observed_spread.max()) <= 1e-10,
             "Realized observations differ across models")
    origins = sorted(forecasts["origin"].unique())
    _require(len(origins) == 130 and origins[0] == "2016-01-08"
             and origins[-1] == "2025-11-28",
             "Forecast origins do not match the accepted 130-origin design")
    _require(all((row in history_dates) for row in origins),
             "Forecast origin not found in accepted factor history")

    gaussian_rows = forecasts["manuscript_model"].isin(["M0", "M1"])
    z = (forecasts.loc[gaussian_rows, "observed"]
         - forecasts.loc[gaussian_rows, "model_mean"]) / np.sqrt(
             forecasts.loc[gaussian_rows, "variance"])
    gaussian_crps = np.sqrt(forecasts.loc[gaussian_rows, "variance"]) * (
        z * (2 * ndtr(z) - 1)
        + 2 * np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        - 1 / math.sqrt(math.pi)
    )
    _require(np.allclose(gaussian_crps,
                         forecasts.loc[gaussian_rows, "crps"].to_numpy(),
                         rtol=1e-8, atol=1e-9),
             "Accepted Gaussian-model CRPS does not reproduce from forecast moments")
    return forecasts


def summarize_regimes(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize proper scores, point errors, PIT, and interval calibration."""
    group_fields = ["regime", "manuscript_model", "factor", "horizon"]
    metrics: list[dict] = []
    calibration: list[dict] = []
    source = forecasts.copy()
    all_rows = source.copy()
    all_rows["regime"] = "ALL"
    source = pd.concat([source, all_rows], ignore_index=True)
    for (regime, model, factor, horizon), group in source.groupby(
            group_fields, sort=False, dropna=False):
        n = len(group)
        errors = group["observed"].to_numpy() - group["model_mean"].to_numpy()
        pit = group["pit"].to_numpy()
        row = {
            "regime": regime,
            "model": model,
            "factor": factor,
            "horizon": int(horizon),
            "n_origins": n,
            "mean_crps": float(group["crps"].mean()),
            "mean_crps_mcse": float(group["crps_mcse"].mean()),
            "crps_mean_mcse": float(np.sqrt(np.sum(group["crps_mcse"].to_numpy()**2)) / n),
            "mean_dss": float(group["dss"].mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "pit_mean": float(np.mean(pit)),
            "pit_std": float(np.std(pit, ddof=1)) if n > 1 else None,
            "mean_pit_mcse": float(group["pit_mcse"].mean()),
            "standardized_error_mean": float(np.mean(
                errors / np.sqrt(group["variance"].to_numpy()))),
            "standardized_error_std": float(np.std(
                errors / np.sqrt(group["variance"].to_numpy()), ddof=1)) if n > 1 else None,
            "predictive_mean_mcse": _mean_available(group, "predictive_mean_mcse"),
            "predictive_variance_mcse": _mean_available(group, "predictive_variance_mcse"),
        }
        for level in INTERVALS:
            covered_count = int(group[f"covered{level}"].sum())
            row[f"covered_count{level}"] = covered_count
            row[f"coverage{level}"] = covered_count / n
            row[f"mean_width{level}"] = float(group[f"width{level}"].mean())
        for index, count in enumerate(np.histogram(pit, bins=np.linspace(0, 1, 11))[0], start=1):
            row[f"pit_bin_{index}"] = int(count)
        metrics.append(row)
        diagnostic = {key: value for key, value in row.items()
                      if key not in {"mean_crps", "mean_crps_mcse", "crps_mean_mcse",
                                     "mean_dss", "rmse", "mae"}}
        diagnostic.update({
            "pit_histogram_bins": 10,
            "pit_histogram_range": "[0,1]",
            "coverage_interpretation": "Descriptive subgroup calibration; no uniformity test",
        })
        calibration.append(diagnostic)
    result = pd.DataFrame(metrics)
    result["regime_order"] = result["regime"].map(REGIME_ORDER)
    result["model_order"] = result["model"].map(MODEL_ORDER)
    result["factor_order"] = result["factor"].map(FACTORS_ORDER)
    result = result.sort_values(["regime_order", "factor_order", "horizon", "model_order"])
    result = result.drop(columns=["regime_order", "model_order", "factor_order"])
    diagnostics = pd.DataFrame(calibration)
    diagnostics["regime_order"] = diagnostics["regime"].map(REGIME_ORDER)
    diagnostics["model_order"] = diagnostics["model"].map(MODEL_ORDER)
    diagnostics["factor_order"] = diagnostics["factor"].map(FACTORS_ORDER)
    diagnostics = diagnostics.sort_values(
        ["regime_order", "factor_order", "horizon", "model_order"])
    diagnostics = diagnostics.drop(columns=["regime_order", "model_order", "factor_order"])
    baseline = result.loc[result["model"] == "M1",
                          ["regime", "factor", "horizon", "mean_crps"]].rename(
                              columns={"mean_crps": "m1_mean_crps"})
    result = result.merge(baseline, on=["regime", "factor", "horizon"], how="left",
                          validate="many_to_one")
    result["crps_skill_vs_M1"] = 1.0 - result["mean_crps"] / result["m1_mean_crps"]
    result = result.drop(columns=["m1_mean_crps"])
    return result.reset_index(drop=True), diagnostics.reset_index(drop=True)


def _mean_available(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
    return float(values.mean()) if len(values) else None


def _validate_unconditional_metrics(forecasts: pd.DataFrame) -> dict:
    """Reaggregate saved forecasts and compare to the accepted published metric CSVs."""
    values = forecasts.copy()
    values["rmse_component"] = (values["observed"] - values["model_mean"]) ** 2
    actual = values.groupby(["manuscript_model", "factor", "horizon"]).agg(
        n_origins=("origin", "nunique"),
        mean_crps=("crps", "mean"),
        mean_dss=("dss", "mean"),
        rmse=("rmse_component", lambda x: float(np.sqrt(np.mean(x)))),
        mae=("mae", "mean"),
        pit_mean=("pit", "mean"),
    ).reset_index()
    for level in INTERVALS:
        part = values.groupby(["manuscript_model", "factor", "horizon"]).agg(
            **{f"coverage{level}": (f"covered{level}", "mean"),
               f"mean_width{level}": (f"width{level}", "mean")}
        ).reset_index()
        actual = actual.merge(part, on=["manuscript_model", "factor", "horizon"],
                              validate="one_to_one")
    accepted_factor = pd.read_csv(FACTOR_METRICS_PATH)
    accepted_garch = pd.read_csv(GARCH_METRICS_PATH)
    accepted = []
    inverse = {internal: manuscript for manuscript, internal in MODEL_MAP.items()}
    for row in accepted_factor.to_dict("records"):
        manuscript = inverse[row["model"]]
        accepted.append({
            "manuscript_model": manuscript, "factor": row["factor"],
            "horizon": int(row["horizon"]), "n_origins": int(row["n_origins"]),
            "mean_crps": float(row["mean_crps"]), "mean_dss": float(row["mean_dss"]),
            "rmse": float(row["rmse"]), "mae": float(row["mae"]),
            "pit_mean": float(row["pit_mean"]),
            **{f"coverage{level}": float(row[f"coverage{level}"]) for level in INTERVALS},
            **{f"mean_width{level}": float(row[f"mean_width{level}"]) for level in INTERVALS},
        })
    for row in accepted_garch.to_dict("records"):
        accepted.append({
            "manuscript_model": "AR1_GARCH11", "factor": row["factor"],
            "horizon": int(row["horizon"]), "n_origins": int(row["n_origins"]),
            "mean_crps": float(row["mean_crps"]), "mean_dss": float(row["mean_dss"]),
            "rmse": float(row["rmse"]), "mae": float(row["mae"]),
            "pit_mean": float(row["pit_mean"]),
            **{f"coverage{level}": float(row[f"coverage{level}"]) for level in INTERVALS},
            **{f"mean_width{level}": float(row[f"mean_width{level}"]) for level in INTERVALS},
        })
    accepted_frame = pd.DataFrame(accepted)
    compare_columns = ["mean_crps", "mean_dss", "rmse", "mae", "pit_mean",
                       *[f"coverage{level}" for level in INTERVALS],
                       *[f"mean_width{level}" for level in INTERVALS]]
    joined = actual.merge(accepted_frame, on=["manuscript_model", "factor", "horizon"],
                          suffixes=("_new", "_accepted"), validate="one_to_one")
    for field in compare_columns:
        _require(np.allclose(joined[f"{field}_new"], joined[f"{field}_accepted"],
                             rtol=1e-8, atol=1e-9),
                 f"Unconditional metric does not reproduce accepted {field}")
    _require((joined["n_origins_new"] == joined["n_origins_accepted"]).all(),
             "Unconditional sample sizes do not reproduce accepted metrics")
    return {
        "unconditional_groups_checked": len(joined),
        "unconditional_metrics_reproduced": True,
        "metrics_compared": compare_columns,
        "maximum_absolute_metric_difference": {
            field: float(np.max(np.abs(joined[f"{field}_new"] - joined[f"{field}_accepted"])))
            for field in compare_columns
        },
    }


def bootstrap_comparisons(
    forecasts: pd.DataFrame,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    expected_block_length: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Paired stationary bootstrap over the full origin sequence and labels."""
    if reps < 1 or expected_block_length < 1:
        raise ValueError("Bootstrap repetitions and expected block length must be positive")
    origin_data = forecasts[["origin", "regime", "origin_row_index"]].drop_duplicates()
    origin_data = origin_data.sort_values("origin_row_index").reset_index(drop=True)
    origins = origin_data["origin"].tolist()
    labels = origin_data["regime"].to_numpy(dtype=object)
    _require(len(origins) == 130 and pd.notna(labels).all(),
             "Bootstrap requires one regime label for every accepted origin")
    regime_sizes = {regime: int(np.sum(labels == regime)) for regime in REGIMES}
    score_arrays: dict[tuple[str, int], np.ndarray] = {}
    mcse_arrays: dict[tuple[str, int], np.ndarray] = {}
    for factor in FACTORS:
        for horizon in HORIZONS:
            block = forecasts.loc[
                (forecasts["factor"] == factor) & (forecasts["horizon"] == horizon)
            ]
            score = block.pivot(index="origin", columns="manuscript_model", values="crps")
            score = score.reindex(index=origins, columns=MODELS)
            mcse = block.pivot(index="origin", columns="manuscript_model", values="crps_mcse")
            mcse = mcse.reindex(index=origins, columns=MODELS)
            _require(score.shape == (130, len(MODELS)) and score.notna().all().all(),
                     f"Incomplete score matrix for {factor}, h={horizon}")
            score_arrays[(factor, horizon)] = score.to_numpy(dtype=float)
            mcse_arrays[(factor, horizon)] = mcse.fillna(0).to_numpy(dtype=float)

    mean_boot: dict[tuple[str, str, int, str], np.ndarray] = {}
    for comparison in PAIRWISE:
        for factor in FACTORS:
            for horizon in HORIZONS:
                for regime in (*REGIMES, "ALL"):
                    mean_boot[(comparison, factor, horizon, regime)] = np.full(reps, np.nan)
    skill_boot: dict[tuple[str, str, int, str], np.ndarray] = {}
    for model in MODELS:
        if model != "M1":
            for factor in FACTORS:
                for horizon in HORIZONS:
                    for regime in REGIMES:
                        skill_boot[(model, factor, horizon, regime)] = np.full(reps, np.nan)
    missing = {regime: 0 for regime in REGIMES}
    rng = np.random.default_rng(seed)
    for replicate in range(reps):
        indices = stationary_bootstrap_indices(len(origins), expected_block_length, rng)
        sampled_labels = labels[indices]
        masks = {regime: sampled_labels == regime for regime in REGIMES}
        for regime, mask in masks.items():
            if not np.any(mask):
                missing[regime] += 1
        for factor in FACTORS:
            for horizon in HORIZONS:
                score = score_arrays[(factor, horizon)][indices]
                for comparison, (left, right) in PAIRWISE.items():
                    difference = score[:, MODEL_ORDER[left]] - score[:, MODEL_ORDER[right]]
                    for regime, mask in masks.items():
                        if np.any(mask):
                            mean_boot[(comparison, factor, horizon, regime)][replicate] = (
                                float(np.mean(difference[mask])))
                    mean_boot[(comparison, factor, horizon, "ALL")][replicate] = float(
                        np.mean(difference))
                baseline = score[:, MODEL_ORDER["M1"]]
                for model in MODELS:
                    if model == "M1":
                        continue
                    for regime, mask in masks.items():
                        if np.any(mask) and float(np.mean(baseline[mask])) > 0:
                            skill_boot[(model, factor, horizon, regime)][replicate] = (
                                1.0 - float(np.mean(score[mask, MODEL_ORDER[model]])
                                            / np.mean(baseline[mask])))

    score_rows = []
    interaction_rows = []
    for factor in FACTORS:
        for horizon in HORIZONS:
            score = score_arrays[(factor, horizon)]
            mcse = mcse_arrays[(factor, horizon)]
            for comparison, (left, right) in PAIRWISE.items():
                difference = score[:, MODEL_ORDER[left]] - score[:, MODEL_ORDER[right]]
                paired_mcse = np.hypot(mcse[:, MODEL_ORDER[left]], mcse[:, MODEL_ORDER[right]])
                for regime in (*REGIMES, "ALL"):
                    mask = np.ones(len(labels), dtype=bool) if regime == "ALL" else labels == regime
                    sampled = mean_boot[(comparison, factor, horizon, regime)]
                    lo, hi, valid = _percentile_interval(sampled)
                    score_rows.append({
                        "comparison": comparison,
                        "left_model": left,
                        "right_model": right,
                        "regime": regime,
                        "factor": factor,
                        "horizon": horizon,
                        "n_origins": int(mask.sum()),
                        "mean_paired_crps_difference": float(np.mean(difference[mask])),
                        "median_paired_crps_difference": float(np.median(difference[mask])),
                        "paired_crps_mcse_of_mean": float(
                            np.sqrt(np.sum(paired_mcse[mask]**2)) / mask.sum()),
                        "bootstrap_ci95_low": lo,
                        "bootstrap_ci95_high": hi,
                        "bootstrap_valid_replicates": valid,
                        "bootstrap_invalid_replicates": reps - valid,
                    })
                hi = mean_boot[(comparison, factor, horizon, "HIGH")]
                low = mean_boot[(comparison, factor, horizon, "LOW")]
                interaction = hi - low
                observed = float(np.mean(difference[labels == "HIGH"])
                                 - np.mean(difference[labels == "LOW"]))
                lo_ci, hi_ci, valid = _percentile_interval(interaction)
                interaction_rows.append({
                    "statistic": "paired_crps_difference_high_minus_low",
                    "comparison": comparison,
                    "model": None,
                    "regime": "HIGH-LOW",
                    "factor": factor,
                    "horizon": horizon,
                    "n_low": int(np.sum(labels == "LOW")),
                    "n_high": int(np.sum(labels == "HIGH")),
                    "estimate": observed,
                    "bootstrap_ci95_low": lo_ci,
                    "bootstrap_ci95_high": hi_ci,
                    "bootstrap_valid_replicates": valid,
                    "bootstrap_invalid_replicates": reps - valid,
                })
            for model in MODELS:
                if model == "M1":
                    continue
                data = score_arrays[(factor, horizon)]
                baseline = data[:, MODEL_ORDER["M1"]]
                high = labels == "HIGH"
                low = labels == "LOW"
                observed = (1.0 - data[high, MODEL_ORDER[model]].mean() / baseline[high].mean()
                            - (1.0 - data[low, MODEL_ORDER[model]].mean() / baseline[low].mean()))
                for regime in REGIMES:
                    mask = labels == regime
                    regime_skill = (1.0 - data[mask, MODEL_ORDER[model]].mean()
                                    / baseline[mask].mean())
                    lo_ci, hi_ci, valid = _percentile_interval(
                        skill_boot[(model, factor, horizon, regime)])
                    interaction_rows.append({
                        "statistic": "crps_skill_vs_M1",
                        "comparison": None,
                        "model": model,
                        "regime": regime,
                        "factor": factor,
                        "horizon": horizon,
                        "n_low": int(low.sum()),
                        "n_high": int(high.sum()),
                        "estimate": float(regime_skill),
                        "bootstrap_ci95_low": lo_ci,
                        "bootstrap_ci95_high": hi_ci,
                        "bootstrap_valid_replicates": valid,
                        "bootstrap_invalid_replicates": reps - valid,
                    })
                skill_interaction = (
                    skill_boot[(model, factor, horizon, "HIGH")]
                    - skill_boot[(model, factor, horizon, "LOW")]
                )
                lo_ci, hi_ci, valid = _percentile_interval(skill_interaction)
                interaction_rows.append({
                    "statistic": "crps_skill_vs_M1_high_minus_low",
                    "comparison": None,
                    "model": model,
                    "regime": "HIGH-LOW",
                    "factor": factor,
                    "horizon": horizon,
                    "n_low": int(low.sum()),
                    "n_high": int(high.sum()),
                    "estimate": float(observed),
                    "bootstrap_ci95_low": lo_ci,
                    "bootstrap_ci95_high": hi_ci,
                    "bootstrap_valid_replicates": valid,
                    "bootstrap_invalid_replicates": reps - valid,
                })
    bootstrap_meta = {
        "purpose": "Exploratory paired conditional comparison; not original confirmatory inference",
        "method": "Circular stationary bootstrap over all 130 chronological origins",
        "replicates": reps,
        "expected_block_length_origins": expected_block_length,
        "seed": seed,
        "random_generator": "NumPy default_rng",
        "sampled_together": ["origin-level model CRPS vectors", "origin-based regime labels"],
        "confidence_interval": "95% percentile interval",
        "missing_regime_resamples": missing,
        "origin_count_by_regime": regime_sizes,
        "regimes_below_15_origins": [
            regime for regime, count in regime_sizes.items() if count < 15
        ],
        "p_values_reported": False,
        "multiplicity": "No exploratory p-values reported; intervals are descriptive",
        "sensitivity_block_lengths": "Not run; optional in protocol",
        "regime_size_note": "No threshold changes based on subgroup performance",
    }
    return (pd.DataFrame(score_rows), pd.DataFrame(interaction_rows), bootstrap_meta)


def _percentile_interval(values: np.ndarray) -> tuple[float | None, float | None, int]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return None, None, 0
    low, high = np.quantile(valid, [0.025, 0.975], method="linear")
    return float(low), float(high), int(len(valid))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="")


def _write_report(
    out: Path, thresholds: dict, regime_counts: dict, metrics: pd.DataFrame,
    pairwise: pd.DataFrame, interactions: pd.DataFrame, bootstrap: dict,
) -> None:
    report_path = ROOT / "docs" / "canadian_volatility_regime_20261120" \
        / "VOLATILITY_REGIME_EVALUATION_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    primary = metrics.loc[metrics["regime"].isin(["LOW", "HIGH"])].copy()
    table_rows = []
    for factor in FACTORS:
        for horizon in HORIZONS:
            for model in MODELS:
                low = primary.loc[(primary["regime"] == "LOW")
                                  & (primary["factor"] == factor)
                                  & (primary["horizon"] == horizon)
                                  & (primary["model"] == model)].iloc[0]
                high = primary.loc[(primary["regime"] == "HIGH")
                                   & (primary["factor"] == factor)
                                   & (primary["horizon"] == horizon)
                                   & (primary["model"] == model)].iloc[0]
                table_rows.append(
                    f"| {factor.title()} | {horizon} | {model} | {int(low.n_origins)} | "
                    f"{low.mean_crps:.5f} | {low.mean_dss:.4f} | {low.rmse:.5f} | "
                    f"{low.mae:.5f} | {low.pit_mean:.3f} ({low.pit_std:.3f}) | "
                    f"{low.coverage50:.3f}/{low.coverage80:.3f}/{low.coverage90:.3f} | "
                    f"{low.mean_width50:.4f}/{low.mean_width80:.4f}/{low.mean_width90:.4f} | "
                    f"{low.crps_skill_vs_M1:+.3f} | {int(high.n_origins)} | "
                    f"{high.mean_crps:.5f} | {high.mean_dss:.4f} | {high.rmse:.5f} | "
                    f"{high.mae:.5f} | {high.pit_mean:.3f} ({high.pit_std:.3f}) | "
                    f"{high.coverage50:.3f}/{high.coverage80:.3f}/{high.coverage90:.3f} | "
                    f"{high.mean_width50:.4f}/{high.mean_width80:.4f}/{high.mean_width90:.4f} | "
                    f"{high.crps_skill_vs_M1:+.3f} |"
                )
    sv_skill = _skill_regime_average(metrics, "M3")
    garch_skill = _skill_regime_average(metrics, "AR1_GARCH11")
    sv_vs_constant = _average_relative_reduction(metrics, "M3", "M2")
    sv_vs_garch = _average_relative_reduction(metrics, "M3", "AR1_GARCH11")
    pair_sg = pairwise.loc[
        (pairwise["comparison"] == "GARCH_minus_Bayesian_SV")
        & pairwise["regime"].isin(["LOW", "HIGH"])
    ]
    sv_wins = {
        regime: int((pair_sg.loc[pair_sg["regime"] == regime,
                                 "mean_paired_crps_difference"] > 0).sum())
        for regime in ("LOW", "HIGH")
    }
    garch_wins = {
        regime: int((pair_sg.loc[pair_sg["regime"] == regime,
                                 "mean_paired_crps_difference"] < 0).sum())
        for regime in ("LOW", "HIGH")
    }
    garch_sv_ci_excludes_zero = int(
        ((pair_sg["bootstrap_ci95_low"] > 0)
         | (pair_sg["bootstrap_ci95_high"] < 0)).sum())
    factor_dependence = _factor_dependence_summary(interactions)
    calibration = _calibration_summary(metrics)
    point_comparisons = _point_comparison_summary(metrics)
    skill_interactions = interactions.loc[
        interactions["statistic"] == "crps_skill_vs_M1_high_minus_low"
    ]
    skill_interaction_ci_excludes_zero = int(
        ((skill_interactions["bootstrap_ci95_low"] > 0)
         | (skill_interactions["bootstrap_ci95_high"] < 0)).sum())
    volatility_skill_interactions = skill_interactions.loc[
        skill_interactions["model"].isin(["M2", "M3", "AR1_GARCH11"])
    ]
    volatility_skill_ci_excludes_zero = int(
        ((volatility_skill_interactions["bootstrap_ci95_low"] > 0)
         | (volatility_skill_interactions["bootstrap_ci95_high"] < 0)).sum())
    excluded_skill_interactions = skill_interactions.loc[
        (skill_interactions["bootstrap_ci95_low"] > 0)
        | (skill_interactions["bootstrap_ci95_high"] < 0),
        ["model", "factor", "horizon"],
    ]
    excluded_skill_labels = ", ".join(
        f"{row.model}/{row.factor}/h={row.horizon}"
        for row in excluded_skill_interactions.itertuples(index=False)
    ) or "none"
    text = f"""# Canadian Yield-Factor Volatility-Regime Evaluation

## Executive assessment

This is a retrospective, observation-time regime analysis of the already accepted
130-origin forecasts. The regime at each origin uses only factor observations
available through that origin. It is not a verified real-time vintage exercise,
trading strategy, joint yield-curve forecast, or causal analysis.

The 26-week composite factor-change RMS uses frozen training-only scaling and
training-period quartile thresholds. The training change scales are
{', '.join(f'{name}={thresholds['normalization_factors'][name]:.6g}' for name in FACTORS)}.
The frozen thresholds are q25={thresholds['q25']:.6g} and
q75={thresholds['q75']:.6g}; regime origin counts are
LOW={regime_counts.get('LOW', 0)}, MEDIUM={regime_counts.get('MEDIUM', 0)},
HIGH={regime_counts.get('HIGH', 0)}.

Across the six factor-horizon groups, mean CRPS skill versus classical AR(1) is
{sv_skill['LOW']:+.3f} for Bayesian stochastic-volatility M3 in LOW regimes and
{sv_skill['HIGH']:+.3f} in HIGH regimes; corresponding AR-GARCH skills are
{garch_skill['LOW']:+.3f} and {garch_skill['HIGH']:+.3f}. These are unweighted
descriptive averages of within-group skill, not pooled forecast scores.
Bayesian SV reduces mean CRPS versus constant-volatility M2 by an average of
{sv_vs_constant['LOW']:.2%} in LOW and {sv_vs_constant['HIGH']:.2%} in HIGH
groups. For AR-GARCH versus classical M1, the corresponding relative reductions
are {garch_skill['LOW']:.2%} and {garch_skill['HIGH']:.2%}. These are
unweighted averages of six within-factor/horizon ratios.

The paired comparisons show a regime pattern, but not a decisive one. Bayesian
SV has lower CRPS than constant-volatility M2 in
{point_comparisons['sv_vs_constant']['crps_low']} of 6 LOW and
{point_comparisons['sv_vs_constant']['crps_high']} of 6 HIGH factor-horizon
groups. AR-GARCH has lower CRPS than M1 in
{point_comparisons['garch_vs_ar1']['crps_low']} of 6 LOW and
{point_comparisons['garch_vs_ar1']['crps_high']} of 6 HIGH groups. For GARCH
minus Bayesian SV, negative favors GARCH: it is negative in
{garch_wins['LOW']} of 6 LOW groups,
but only {garch_wins['HIGH']} of 6 HIGH groups; SV has lower CRPS in
{sv_wins['HIGH']} HIGH groups. None of those 12 GARCH-versus-SV subgroup
bootstrap intervals excludes zero. The average within-group relative CRPS
reduction of SV versus GARCH is {sv_vs_garch['LOW']:.2%} in LOW and
{sv_vs_garch['HIGH']:.2%} in HIGH groups. Thus the near tie overall may mask a
descriptive low-regime advantage for GARCH and high-regime advantage for SV,
but subgroup uncertainty does not establish that explanation.

## Regime construction and integrity

- Accepted factor sample: {len(pd.read_csv(FACTOR_PATH)):,} weekly observations,
  {TRAIN_START} through 2025-12-24. The 1999-12-31 row was excluded.
- Training calibration: {thresholds['training_observation_count']} factor
  observations through {thresholds['training_end']}; normalization SD uses
  {thresholds['training_change_count']} in-sample weekly changes and ddof=1.
- Volatility: RMS of exactly {LOOKBACK} standardized factor changes, then equal
  weighting across level, slope, and curvature. The rolling measure is RMS, not
  a rolling centered standard deviation.
- Quantiles: NumPy `method='linear'`, fitted only on complete training windows.
  Thresholds and labels were frozen before calculating conditional metrics.
- The factor history matches fixed-lambda least-squares extraction from the
  accepted yield panel; maximum absolute discrepancy was
  {thresholds['max_factor_reconstruction_abs_difference']:.3g}.
- Regimes are attached by origin date and apply unchanged to all factors,
  horizons, and models. Targets and observed factor values were checked against
  the accepted history.

## Conditional LOW versus HIGH results

Each cell reports LOW and HIGH separately. PIT is mean (sample SD); coverage and
width columns are ordered 50%/80%/90%. CRPS skill is relative to M1 within the
same factor, horizon, and regime; positive means better CRPS than M1. Factor
units are percentage points.

| Factor | h | Model | n Low | CRPS Low | DSS Low | RMSE Low | MAE Low | PIT Low | Coverage Low | Width Low | Skill Low | n High | CRPS High | DSS High | RMSE High | MAE High | PIT High | Coverage High | Width High | Skill High |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
{chr(10).join(table_rows)}

Complete results, including MEDIUM and ALL regimes, coverage counts, interval
widths, PIT histograms, standardized-error diagnostics, CRPS Monte Carlo
uncertainty, and all 120 model/factor/horizon/regime groups are in
`results/canadian_volatility_regime_20261120/regime_metrics.csv` and
`regime_calibration_diagnostics.csv`.

## Paired comparisons and exploratory uncertainty

Paired differences are first model minus second model; a negative mean favors
the first model. HIGH-minus-LOW interactions describe association with the
predefined regime, not a causal effect. Factor-level comparison and interaction
intervals are recorded in `regime_pairwise_differences.csv` and
`regime_interaction_summary.csv`.

Bootstrap: {bootstrap['replicates']:,} circular stationary resamples of the full
chronological 130-origin sequence, expected block length
{bootstrap['expected_block_length_origins']} origins, seed {bootstrap['seed']}.
The origin-level score vectors and labels were sampled together. Missing-regime
resamples: {bootstrap['missing_regime_resamples']}; origin group sizes are
{bootstrap['origin_count_by_regime']}, with fewer than 15 origins in
{bootstrap['regimes_below_15_origins']}. Reported intervals are
exploratory percentile 95% intervals; no exploratory p-values are reported.
The accepted original test family and Holm-adjusted results are unchanged.

## Research questions

1. **Is volatility modeling more valuable in HIGH regimes?** Not on average
   against M1 in these data: mean SV skill is {sv_skill['LOW']:.2%} LOW versus
   {sv_skill['HIGH']:.2%} HIGH, and GARCH skill is {garch_skill['LOW']:.2%}
   versus {garch_skill['HIGH']:.2%}. The high-minus-low skill intervals exclude
   zero in {skill_interaction_ci_excludes_zero} of {len(skill_interactions)}
   non-baseline model/factor/horizon comparisons; only
   {volatility_skill_ci_excludes_zero} of {len(volatility_skill_interactions)}
   comparisons involving Bayesian M2/M3 or GARCH exclude zero. The one interval
   excluding zero is {excluded_skill_labels}. Raw HIGH CRPS is not used to infer
   relative performance.
2. **Does Bayesian SV outperform GARCH?** SV is better in
   {sv_wins['HIGH']} of six HIGH groups and {sv_wins['LOW']} of six LOW groups;
   GARCH is better in {garch_wins['LOW']} LOW groups.
   The GARCH-minus-SV confidence intervals all include zero, so this is a
   descriptive reversal, not a resolved difference.
3. **Is curvature more regime-sensitive?** Across relative-skill interactions
   for M2, M3, and GARCH, the largest mean absolute HIGH-minus-LOW change is
   {factor_dependence}. This descriptive ranking does not establish a
   curvature-specific effect.
4. **Do probabilistic gains align with point forecasts?** For SV versus M2,
   CRPS improves in {point_comparisons['sv_vs_constant']['crps_low']}/6 LOW and
   {point_comparisons['sv_vs_constant']['crps_high']}/6 HIGH cells, but RMSE
   improves in {point_comparisons['sv_vs_constant']['rmse_low']}/6 and
   {point_comparisons['sv_vs_constant']['rmse_high']}/6, respectively. For
   GARCH versus M1, CRPS improves in {point_comparisons['garch_vs_ar1']['crps_low']}/6
   LOW and {point_comparisons['garch_vs_ar1']['crps_high']}/6 HIGH cells, versus
   RMSE improvement in {point_comparisons['garch_vs_ar1']['rmse_low']}/6 and
   {point_comparisons['garch_vs_ar1']['rmse_high']}/6. This is consistent with
   gains that are not solely improvements in conditional means.
5. **Does calibration differ by regime?** Mean absolute deviation of grouped
   coverage from nominal is {calibration['coverage_error_low']:.3f} LOW and
   {calibration['coverage_error_high']:.3f} HIGH; mean absolute PIT-mean
   deviation from 0.5 is {calibration['pit_error_low']:.3f} LOW and
   {calibration['pit_error_high']:.3f} HIGH. These summaries do not replace the
   histograms, counts, coverage, and width tables. A PIT mean near 0.5 alone is
   not full calibration evidence.
6. **Sensitivity to regime definition?** The primary 26-week composite
   specification was not tuned. Optional 13-week sensitivity analysis was not
   run.

## Figures

- `results/canadian_volatility_regime_20261120/figures/figure_1_regime_history.png`
  and `.pdf`
- `figure_2_conditional_crps.png` and `.pdf`
- `figure_3_relative_crps_skill.png` and `.pdf`
- `figure_4_garch_vs_bayesian_sv.png` and `.pdf`
- `figure_5_calibration_pit.png` and `.pdf`
- `figure_5_calibration_intervals.png` and `.pdf`

Figures are generated from saved outputs by
`src/plot_canadian_volatility_regimes.py`. No manuscript source, PDF, or
generated manuscript table was modified.

## Validation and limitations

The complete 3,900 records from the four accepted factor models and AR-GARCH
were joined on origin, target, factor, horizon, and manuscript model. We
verified five models, three factors, two horizons, 130 origins, unique keys,
matching realized values, valid predictive moments, intervals, and score
components. Unconditional aggregates reproduce the accepted metrics at
floating-point tolerance. Bayesian predictive simulations were not regenerated
and no forecasting model was re-estimated.

The classifications are based on observation-time factor histories, not
verified publication vintages. Regime groups are modest in size and forecasts
overlap in time; bootstrap uncertainty accounts for serial dependence only
approximately. Results are exploratory and do not support causal, portfolio,
or joint-yield-curve claims.

The largest subgroup-mean CRPS Monte Carlo standard error is
{metrics.loc[metrics['regime'].isin(['LOW', 'HIGH']),
             'crps_mean_mcse'].max():.5f}. Sampling uncertainty is generally
wider than this simulation component; the bootstrap intervals matter when
interpreting small score differences.
"""
    report_path.write_text(text, encoding="utf-8")


def _skill_regime_average(metrics: pd.DataFrame, model: str) -> dict[str, float]:
    subset = metrics.loc[(metrics["model"] == model)
                         & (metrics["regime"].isin(["LOW", "HIGH"]))]
    return {regime: float(subset.loc[subset["regime"] == regime,
                                     "crps_skill_vs_M1"].mean())
            for regime in ("LOW", "HIGH")}


def _factor_dependence_summary(interactions: pd.DataFrame) -> str:
    subset = interactions.loc[
        (interactions["statistic"] == "crps_skill_vs_M1_high_minus_low")
        & interactions["model"].isin(["M2", "M3", "AR1_GARCH11"])
    ]
    mean_abs = subset.groupby("factor")["estimate"].apply(
        lambda values: float(np.mean(np.abs(values))))
    if mean_abs.empty:
        return "not available"
    return f"{mean_abs.idxmax()} (mean absolute skill interaction {mean_abs.max():.3f})"


def _calibration_summary(metrics: pd.DataFrame) -> dict:
    subset = metrics.loc[metrics["regime"].isin(["LOW", "HIGH"])]
    return {
        "groups": int(subset.shape[0]),
        **{
            f"coverage_error_{regime.lower()}": float(np.mean(np.column_stack([
                np.abs(subset.loc[subset["regime"] == regime, f"coverage{level}"]
                       - level / 100)
                for level in INTERVALS
            ])))
            for regime in ("LOW", "HIGH")
        },
        **{
            f"pit_error_{regime.lower()}": float(np.mean(
                np.abs(subset.loc[subset["regime"] == regime, "pit_mean"] - 0.5)))
            for regime in ("LOW", "HIGH")
        },
    }


def _average_relative_reduction(
    metrics: pd.DataFrame, model: str, benchmark: str
) -> dict[str, float]:
    result = {}
    for regime in ("LOW", "HIGH"):
        subset = metrics.loc[(metrics["regime"] == regime)
                             & metrics["model"].isin([model, benchmark])]
        table = subset.pivot(index=["factor", "horizon"], columns="model",
                             values="mean_crps")
        result[regime] = float(np.mean(1.0 - table[model] / table[benchmark]))
    return result


def _point_comparison_summary(metrics: pd.DataFrame) -> dict:
    output = {"sv_vs_constant": {}, "garch_vs_ar1": {}}
    for key, left, right in (
        ("sv_vs_constant", "M3", "M2"),
        ("garch_vs_ar1", "AR1_GARCH11", "M1"),
    ):
        for regime in ("LOW", "HIGH"):
            first = metrics.loc[(metrics["model"] == left)
                                & (metrics["regime"] == regime)].set_index(
                                    ["factor", "horizon"])
            second = metrics.loc[(metrics["model"] == right)
                                 & (metrics["regime"] == regime)].set_index(
                                     ["factor", "horizon"])
            for metric, label in (("mean_crps", "crps"), ("rmse", "rmse"), ("mae", "mae")):
                output[key][f"{label}_{regime.lower()}"] = int(
                    (first[metric] < second[metric]).sum())
    return output


def run_analysis(
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history, integrity = load_factor_history()
    train_indices = np.flatnonzero(
        history["Date"].dt.date <= date.fromisoformat(TRAIN_END))
    _require(len(train_indices) == 835
             and train_indices[-1] == 834
             and history["Date"].iloc[train_indices[-1]].date().isoformat() == TRAIN_END,
             "Initial 2000-2015 training window does not match the accepted 835 rows")
    full_inventory, thresholds = construct_regimes(history, int(train_indices[-1]) + 1)
    eligible = full_inventory.loc[full_inventory["regime"].notna()].copy()
    forecasts = _read_accepted_forecasts(history)
    origin_index = full_inventory.set_index(full_inventory["Date"].dt.strftime("%Y-%m-%d"))
    origin_rows = sorted(forecasts["origin"].unique())
    origin_inventory = origin_index.loc[origin_rows].reset_index(drop=True)
    _require(len(origin_inventory) == 130 and origin_inventory["regime"].notna().all(),
             "Every forecast origin must have one eligible frozen regime")
    counts = origin_inventory["regime"].value_counts().to_dict()
    _require(set(counts).issubset(set(REGIMES)) and sum(counts.values()) == 130,
             "Origin regime counts do not reconcile")
    merge_inventory = origin_inventory[[
        "Date", "research_row_index", "yield_panel_row_index_zero_based",
        "volatility_level", "volatility_slope", "volatility_curvature",
        "composite_volatility", "q25", "q75", "regime",
    ]].rename(columns={
        "Date": "origin_date",
        "research_row_index": "research_origin_row_index",
        "yield_panel_row_index_zero_based": "yield_panel_origin_row_index_zero_based",
    })
    merge_inventory["origin_date"] = merge_inventory["origin_date"].dt.strftime("%Y-%m-%d")
    forecasts = forecasts.merge(
        merge_inventory,
        left_on="origin", right_on="origin_date", how="left", validate="many_to_one",
    )
    _require(len(forecasts) == 3900 and forecasts["regime"].notna().all(),
             "Forecast/regime join lost or duplicated rows")
    forecasts["origin_row_index"] = forecasts["research_origin_row_index"].astype(int)
    metrics, calibration = summarize_regimes(forecasts)
    pairwise, interactions, bootstrap = bootstrap_comparisons(
        forecasts, reps=bootstrap_reps, seed=bootstrap_seed, expected_block_length=4.0)
    unconditional = _validate_unconditional_metrics(forecasts)

    thresholds["origin_regime_counts"] = {regime: int(counts.get(regime, 0))
                                          for regime in REGIMES}
    thresholds["complete_eligible_history_rows"] = int(len(eligible))
    thresholds["max_factor_reconstruction_abs_difference"] = integrity[
        "max_factor_reconstruction_abs_difference"]
    thresholds["first_complete_window_date"] = (
        eligible["Date"].iloc[0].date().isoformat())
    thresholds["latest_history_date"] = history["Date"].iloc[-1].date().isoformat()
    (OUTPUT / "regime_thresholds.json").write_text(
        json.dumps(thresholds, indent=2, allow_nan=False), encoding="utf-8")
    _write_csv(OUTPUT / "historical_volatility_index.csv", eligible)
    _write_csv(OUTPUT / "origin_volatility_regimes.csv", origin_inventory)
    _write_csv(OUTPUT / "forecasts_with_regimes.csv", forecasts)
    _write_csv(OUTPUT / "regime_metrics.csv", metrics)
    _write_csv(OUTPUT / "regime_calibration_diagnostics.csv", calibration)
    _write_csv(OUTPUT / "regime_pairwise_differences.csv", pairwise)
    _write_csv(OUTPUT / "regime_interaction_summary.csv", interactions)
    (OUTPUT / "regime_bootstrap_results.json").write_text(
        json.dumps(bootstrap, indent=2, allow_nan=False), encoding="utf-8")

    from src.plot_canadian_volatility_regimes import make_all_figures

    make_all_figures()
    figure_files = sorted(
        str(path.relative_to(OUTPUT)) for path in (OUTPUT / "figures").iterdir()
        if path.is_file()
    )
    test_result = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_canadian_volatility_regime_analysis"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    test_output = (test_result.stdout + test_result.stderr).strip()
    report_relative = "docs/canadian_volatility_regime_20261120/" \
        "VOLATILITY_REGIME_EVALUATION_REPORT.md"
    _write_report(OUTPUT, thresholds, counts, metrics, pairwise, interactions, bootstrap)

    output_files = [
        "regime_thresholds.json", "historical_volatility_index.csv",
        "origin_volatility_regimes.csv", "forecasts_with_regimes.csv",
        "regime_metrics.csv", "regime_pairwise_differences.csv",
        "regime_interaction_summary.csv", "regime_calibration_diagnostics.csv",
        "regime_bootstrap_results.json",
        *figure_files, report_relative,
    ]
    validation = {
        **integrity,
        **unconditional,
        "forecast_rows": len(forecasts),
        "forecast_key_unique": True,
        "model_count": int(forecasts["manuscript_model"].nunique()),
        "factor_count": int(forecasts["factor"].nunique()),
        "horizon_count": int(forecasts["horizon"].nunique()),
        "origin_count": int(forecasts["origin"].nunique()),
        "regime_counts_sum_to_origins": sum(counts.values()) == 130,
        "scores_and_intervals_validated": True,
        "accepted_forecasts_reestimated": False,
        "legacy_make_design_hash_gate_used": False,
        "focused_test_suite": {
            "command": f"{Path(sys.executable).name} -m unittest "
                       "tests.test_canadian_volatility_regime_analysis",
            "return_code": test_result.returncode,
            "output": test_output,
        },
        "figures_rendered": len(figure_files) == 12,
    }
    manifest = {
        "experiment": "Retrospective observation-time volatility regime conditional evaluation",
        "status": "complete",
        "parent_branch": "research/factor-probabilistic-forecasting-20261116",
        "extension_branch": "research/canadian-volatility-regime-extension-20261120",
        "source_paths": {
            "factor_history": str(FACTOR_PATH.relative_to(ROOT)),
            "yield_panel": str(PANEL_PATH.relative_to(ROOT)),
            "factor_forecasts": str(FACTOR_FORECAST_PATH.relative_to(ROOT)),
            "factor_metrics": str(FACTOR_METRICS_PATH.relative_to(ROOT)),
            "garch_forecasts": str(GARCH_FORECAST_PATH.relative_to(ROOT)),
            "garch_metrics": str(GARCH_METRICS_PATH.relative_to(ROOT)),
        },
        "sample": {
            "country": "Canada",
            "research_start": TRAIN_START,
            "research_end": "2025-12-24",
            "weekly_observations": len(history),
            "initial_training_end": TRAIN_END,
            "initial_training_observations": 835,
            "origin_count": 130,
            "first_origin": str(origin_rows[0]),
            "last_origin": str(origin_rows[-1]),
            "horizons": list(HORIZONS),
            "fixed_lambda": LAMBDA,
            "maturities_years": MATURITIES.tolist(),
            "factor_units": "percentage points",
        },
        "model_label_mapping": {
            "M0": {"specification": "Factor random walk", "internal": "M0"},
            "M1": {"specification": "Classical Gaussian AR(1)", "internal": "M2"},
            "M2": {"specification": "Bayesian DNS, constant volatility", "internal": "M3"},
            "M3": {"specification": "Bayesian DNS, stochastic volatility", "internal": "M4"},
            "AR1_GARCH11": {"specification": "Independent Gaussian AR(1)-GARCH(1,1)",
                            "internal": "AR1_GARCH11"},
        },
        "regime_method": thresholds,
        "regime_counts": thresholds["origin_regime_counts"],
        "bootstrap": bootstrap,
        "software": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
            "matplotlib": importlib.metadata.version("matplotlib"),
        },
        "validation": validation,
        "outputs": output_files,
        "hash_policy": "No SHA-256 equality is required or enforced.",
    }
    (OUTPUT / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "manifest": manifest,
        "metrics": metrics,
        "pairwise": pairwise,
        "interactions": interactions,
        "origin_inventory": origin_inventory,
        "historical_inventory": eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    result = run_analysis(args.bootstrap_reps, args.bootstrap_seed)
    print(json.dumps({
        "status": result["manifest"]["status"],
        "output_dir": str(OUTPUT),
        "regime_counts": result["manifest"]["regime_counts"],
        "forecast_records": result["manifest"]["validation"]["forecast_rows"],
        "bootstrap_reps": result["manifest"]["bootstrap"]["replicates"],
    }, indent=2))


if __name__ == "__main__":
    main()
