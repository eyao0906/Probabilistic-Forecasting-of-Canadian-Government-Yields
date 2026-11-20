"""Expanding-window Gaussian AR(1)-GARCH(1,1) forecasts for extracted DNS factors."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import warnings

import numpy as np
from arch import arch_model
from scipy.stats import norm

from src.bayesian_stage_a import FACTORS, make_design
from src.experiment import select_origins
from src.factor_probabilistic_experiment import gaussian_crps

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "factor_garch_20261119"
ACCEPTED_OUTPUT = ROOT / "results" / "factor_probabilistic_20261116"
HORIZONS = (1, 4)
INTERVALS = (0.50, 0.80, 0.90)
SCENARIOS = 8192
BATCHES = 8
BASE_SEED = 20261119
SCALE_TO_BASIS_POINTS = 100.0
LAMBDA = 0.5160377358490565
MATURITIES = [0.25, 0.5, 1, 2, 5, 7, 10, 30]
MANUSCRIPT_TO_INTERNAL = {"M0": "M0", "M1": "M2", "M2": "M3", "M3": "M4"}
REPORTING_MODELS = ("M0", "M1", "M2", "M3", "AR1_GARCH11")


@dataclass(frozen=True)
class FittedGarch:
    record: dict
    intercept: float
    ar: float
    omega: float
    alpha: float
    beta: float
    last_residual: float
    last_variance: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_origins(design: dict) -> list[int]:
    selected = select_origins(
        design["dates"], date(2016, 1, 1), date(2025, 12, 31), [1, 4], 4)
    origins = sorted({origin for origin, _, _ in selected})
    if (len(origins) != 130
            or design["dates"][origins[0]].isoformat() != "2016-01-08"
            or design["dates"][origins[-1]].isoformat() != "2025-11-28"):
        raise ValueError("Accepted factor forecast origin schedule has changed")
    return origins


def fit_ar1_garch(training: np.ndarray, *, factor: str, origin: str) -> FittedGarch:
    """Fit in basis points, then return all scale-dependent values in pp units."""
    x = np.asarray(training, dtype=float)
    if x.ndim != 1 or len(x) < 20 or not np.isfinite(x).all():
        raise ValueError("GARCH training data must be a finite factor series")
    scaled = x * SCALE_TO_BASIS_POINTS
    captured: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = arch_model(
            scaled, mean="AR", lags=1, vol="GARCH", p=1, o=0, q=1,
            dist="normal", rescale=False,
        ).fit(disp="off", show_warning=True)
    captured.extend(str(item.message) for item in caught)
    params = result.params
    required = ("Const", "y[1]", "omega", "alpha[1]", "beta[1]")
    if any(name not in params for name in required):
        raise ValueError(f"Unexpected arch parameter names: {list(params.index)}")
    intercept = float(params["Const"]) / SCALE_TO_BASIS_POINTS
    ar = float(params["y[1]"])
    omega = float(params["omega"]) / SCALE_TO_BASIS_POINTS**2
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    residuals = np.asarray(result.resid, dtype=float)
    conditional_volatility = np.asarray(result.conditional_volatility, dtype=float)
    last_residual = float(residuals[-1]) / SCALE_TO_BASIS_POINTS
    last_variance = float(conditional_volatility[-1] ** 2) / SCALE_TO_BASIS_POINTS**2
    persistence = alpha + beta
    valid_parameters = bool(
        np.isfinite([intercept, ar, omega, alpha, beta, last_residual, last_variance]).all()
        and omega > 0 and alpha >= 0 and beta >= 0 and last_variance > 0)
    optimizer = result.optimization_result
    optimizer_success = bool(optimizer.success)
    converged = bool(result.convergence_flag == 0 and optimizer_success and valid_parameters)
    warnings_found = list(captured)
    if abs(ar) >= 1:
        warnings_found.append("AR coefficient is nonstationary at abs(ar) >= 1")
    elif abs(ar) >= 0.99:
        warnings_found.append("AR coefficient is near unit magnitude")
    if persistence >= 1:
        warnings_found.append("GARCH persistence is nonstationary at alpha+beta >= 1")
    elif persistence >= 0.99:
        warnings_found.append("GARCH persistence is near one")
    if not valid_parameters:
        warnings_found.append("Fitted parameters or final conditional variance are invalid")
    if not converged:
        warnings_found.append(f"Optimizer did not converge: {optimizer.message}")
    nobs = int(result.nobs)
    loglikelihood = float(result.loglikelihood + nobs * math.log(SCALE_TO_BASIS_POINTS))
    parameter_count = len(params)
    record = {
        "model": "AR1_GARCH11",
        "factor": factor,
        "origin": origin,
        "training_observations": len(x),
        "optimizer_success": optimizer_success,
        "convergence_flag": int(result.convergence_flag),
        "converged": converged,
        "optimizer_message": str(optimizer.message),
        "intercept": intercept,
        "ar1": ar,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "garch_persistence": persistence,
        "final_conditional_variance": last_variance,
        "loglikelihood_pp": loglikelihood,
        "aic_pp": float(-2 * loglikelihood + 2 * parameter_count),
        "bic_pp": float(-2 * loglikelihood + math.log(nobs) * parameter_count),
        "ar_near_unit": abs(ar) >= 0.99,
        "ar_nonstationary": abs(ar) >= 1,
        "garch_near_unit": persistence >= 0.99,
        "garch_nonstationary": persistence >= 1,
        "warnings": json.dumps(warnings_found),
    }
    return FittedGarch(record, intercept, ar, omega, alpha, beta,
                       last_residual, last_variance)


def predictive_moments(fit: FittedGarch, last_value: float, horizon: int) -> tuple[float, float]:
    if horizon not in HORIZONS or not np.isfinite(last_value):
        raise ValueError("Invalid GARCH forecast horizon or state")
    mean = float(last_value)
    for _ in range(horizon):
        mean = fit.intercept + fit.ar * mean
    innovation_variances = np.empty(horizon, dtype=float)
    innovation_variances[0] = (fit.omega + fit.alpha * fit.last_residual**2
                               + fit.beta * fit.last_variance)
    for step in range(1, horizon):
        innovation_variances[step] = (
            fit.omega + (fit.alpha + fit.beta) * innovation_variances[step - 1])
    variance = float(sum(
        fit.ar ** (2 * (horizon - step - 1)) * innovation_variances[step]
        for step in range(horizon)))
    if not np.isfinite([mean, variance]).all() or variance <= 0:
        raise ValueError("GARCH produced invalid predictive moments")
    return mean, variance


def simulate_ar1_garch(fit: FittedGarch, last_value: float, horizon: int,
                        rng: np.random.Generator, draws: int = SCENARIOS) -> np.ndarray:
    if horizon < 1 or draws < 1 or not np.isfinite(last_value):
        raise ValueError("Invalid GARCH simulation dimensions or state")
    state = np.full(draws, last_value, dtype=float)
    residual = np.full(draws, fit.last_residual, dtype=float)
    variance = np.full(draws, fit.last_variance, dtype=float)
    for _ in range(horizon):
        variance = fit.omega + fit.alpha * residual**2 + fit.beta * variance
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError("GARCH simulation generated invalid conditional variance")
        residual = np.sqrt(variance) * rng.standard_normal(draws)
        state = fit.intercept + fit.ar * state + residual
    if not np.isfinite(state).all():
        raise ValueError("GARCH simulation generated non-finite factor values")
    return state


def _sample_crps(observed: float, samples: np.ndarray) -> float:
    """Unbiased empirical CRPS using the O(n log n) pairwise U-statistic."""
    values = np.sort(np.asarray(samples, dtype=float))
    n = len(values)
    if n < 2 or not np.isfinite(values).all() or not np.isfinite(observed):
        raise ValueError("Sample CRPS requires at least two finite scenarios")
    coefficients = 2 * np.arange(n) - n + 1
    pair_sum = float(coefficients @ values)
    return float(np.mean(np.abs(values - observed)) - pair_sum / (n * (n - 1)))


def _score_forecast(mean: float, variance: float, observed: float,
                    samples: np.ndarray | None) -> dict:
    if not np.isfinite([mean, variance, observed]).all() or variance <= 0:
        raise ValueError("Invalid GARCH forecast moments or observation")
    if samples is None:
        sd = math.sqrt(variance)
        z = (observed - mean) / sd
        intervals = {
            level: (float(mean - norm.ppf((1 + level) / 2) * sd),
                    float(mean + norm.ppf((1 + level) / 2) * sd))
            for level in INTERVALS
        }
        return {
            "mean": mean, "variance": variance,
            "crps": gaussian_crps(observed, mean, variance),
            "crps_mcse": 0.0, "pit": float(norm.cdf(z)), "pit_mcse": 0.0,
            "predictive_mean_mcse": 0.0, "predictive_variance_mcse": 0.0,
            "intervals": intervals,
        }
    values = np.asarray(samples, dtype=float)
    if values.shape != (SCENARIOS,) or not np.isfinite(values).all():
        raise ValueError("GARCH forecast simulation has the wrong shape or non-finite values")
    batch = values.reshape(BATCHES, SCENARIOS // BATCHES)
    batch_crps = np.asarray([_sample_crps(observed, x) for x in batch])
    batch_pit = np.mean(batch <= observed, axis=1)
    batch_means = batch.mean(axis=1)
    batch_variances = batch.var(axis=1, ddof=1)
    intervals = {
        level: tuple(float(v) for v in np.quantile(
            values, [(1 - level) / 2, 1 - (1 - level) / 2]))
        for level in INTERVALS
    }
    return {
        "mean": mean,
        "variance": variance,
        "crps": float(batch_crps.mean()),
        "crps_mcse": float(batch_crps.std(ddof=1) / math.sqrt(BATCHES)),
        "pit": float(np.mean(batch_pit)),
        "pit_mcse": float(batch_pit.std(ddof=1) / math.sqrt(BATCHES)),
        "predictive_mean_mcse": float(batch_means.std(ddof=1) / math.sqrt(BATCHES)),
        "predictive_variance_mcse": float(batch_variances.std(ddof=1) / math.sqrt(BATCHES)),
        "intervals": intervals,
    }


def _forecast_record(factor: str, origin: str, target: str, horizon: int,
                     observed: float, current: float, fit: FittedGarch,
                     rng: np.random.Generator | None, scenario_seed: int | None) -> dict:
    mean, variance = predictive_moments(fit, current, horizon)
    samples = None if horizon == 1 else simulate_ar1_garch(
        fit, current, horizon, rng)
    scored = _score_forecast(mean, variance, observed, samples)
    row = {
        "model": "AR1_GARCH11",
        "factor": factor,
        "origin": origin,
        "target": target,
        "horizon": horizon,
        "observed": observed,
        "mean": scored["mean"],
        "variance": scored["variance"],
        "crps": scored["crps"],
        "crps_mcse": scored["crps_mcse"],
        "dss": float(math.log(variance) + (observed - mean)**2 / variance),
        "rmse_component": float((observed - mean)**2),
        "mae": float(abs(observed - mean)),
        "pit": scored["pit"],
        "pit_mcse": scored["pit_mcse"],
        "predictive_mean_mcse": scored["predictive_mean_mcse"],
        "predictive_variance_mcse": scored["predictive_variance_mcse"],
        "scenario_seed": "" if scenario_seed is None else scenario_seed,
    }
    for level, (lower, upper) in scored["intervals"].items():
        key = int(level * 100)
        row[f"lower{key}"] = lower
        row[f"upper{key}"] = upper
        row[f"width{key}"] = upper - lower
        row[f"covered{key}"] = int(lower <= observed <= upper)
    return row


def _seed(origin_index: int, factor_index: int, horizon: int) -> int:
    sequence = np.random.SeedSequence([BASE_SEED, origin_index, factor_index, horizon])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _aggregate(records: list[dict]) -> list[dict]:
    rows = []
    for factor in FACTORS:
        for horizon in HORIZONS:
            group = [r for r in records if r["factor"] == factor and r["horizon"] == horizon]
            row = {
                "model": "AR1_GARCH11",
                "factor": factor,
                "horizon": horizon,
                "n_origins": len(group),
                "mean_crps": float(np.mean([r["crps"] for r in group])) if group else None,
                "mean_crps_mcse": float(np.mean([r["crps_mcse"] for r in group])) if group else None,
                "crps_mean_mcse": (
                    float(math.sqrt(sum(r["crps_mcse"]**2 for r in group)) / len(group))
                    if group else None),
                "mean_dss": float(np.mean([r["dss"] for r in group])) if group else None,
                "rmse": float(math.sqrt(np.mean([r["rmse_component"] for r in group]))) if group else None,
                "mae": float(np.mean([r["mae"] for r in group])) if group else None,
                "pit_mean": float(np.mean([r["pit"] for r in group])) if group else None,
                "mean_pit_mcse": float(np.mean([r["pit_mcse"] for r in group])) if group else None,
                "mean_predictive_mean_mcse": float(np.mean(
                    [r["predictive_mean_mcse"] for r in group])) if group else None,
                "mean_predictive_variance_mcse": float(np.mean(
                    [r["predictive_variance_mcse"] for r in group])) if group else None,
            }
            for level in ("50", "80", "90"):
                row[f"coverage{level}"] = (
                    float(np.mean([r[f"covered{level}"] for r in group])) if group else None)
                row[f"mean_width{level}"] = (
                    float(np.mean([r[f"width{level}"] for r in group])) if group else None)
            for i, count in enumerate(np.histogram(
                    [r["pit"] for r in group], bins=np.linspace(0, 1, 11))[0], start=1):
                row[f"pit_bin_{i}"] = int(count)
            rows.append(row)
    return rows


def _calibration(records: list[dict]) -> list[dict]:
    rows = []
    for factor in FACTORS:
        for horizon in HORIZONS:
            group = sorted((r for r in records
                            if r["factor"] == factor and r["horizon"] == horizon),
                           key=lambda r: r["origin"])
            standardized = np.asarray([
                (r["observed"] - r["mean"]) / math.sqrt(r["variance"]) for r in group])
            pit = np.asarray([r["pit"] for r in group])
            row = {
                "model": "AR1_GARCH11", "factor": factor, "horizon": horizon,
                "n_origins": len(group),
                "standardized_error_mean": float(standardized.mean()) if len(group) else None,
                "standardized_error_variance": float(standardized.var(ddof=1)) if len(group) > 1 else None,
                "standardized_error_lag1": float(np.corrcoef(standardized[:-1], standardized[1:])[0, 1])
                if len(group) > 1 else None,
                "pit_mean": float(pit.mean()) if len(group) else None,
                "pit_mcse_mean": float(np.mean([r["pit_mcse"] for r in group])) if group else None,
                "pit_lag1": float(np.corrcoef(pit[:-1], pit[1:])[0, 1])
                if len(group) > 1 else None,
            }
            for level in ("50", "80", "90"):
                row[f"coverage{level}"] = (
                    float(np.mean([r[f"covered{level}"] for r in group])) if group else None)
                row[f"mean_width{level}"] = (
                    float(np.mean([r[f"width{level}"] for r in group])) if group else None)
            rows.append(row)
    return rows


def _accepted_metrics(garch_metrics: list[dict]) -> list[dict]:
    path = ACCEPTED_OUTPUT / "factor_metrics.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        accepted = list(csv.DictReader(stream))
    with (ACCEPTED_OUTPUT / "factor_forecasts.csv").open(
            newline="", encoding="utf-8") as stream:
        accepted_forecasts = list(csv.DictReader(stream))
    required_internal = set(MANUSCRIPT_TO_INTERNAL.values())
    keys = [(factor, horizon) for factor in FACTORS for horizon in HORIZONS]
    if len(accepted) != 24 or {
            row["model"] for row in accepted} != required_internal:
        raise ValueError("Accepted factor metrics do not match frozen model inventory")
    by_internal = {(r["model"], r["factor"], int(r["horizon"])): r for r in accepted}
    if len(by_internal) != len(accepted):
        raise ValueError("Accepted factor metrics contain duplicate model groups")
    mcse_by_internal = {}
    for model in required_internal:
        for factor, horizon in keys:
            group = [r for r in accepted_forecasts
                     if r["model"] == model and r["factor"] == factor
                     and int(r["horizon"]) == horizon]
            if len(group) != 130:
                raise ValueError(f"Accepted CRPS MCSE group incomplete: {model}/{factor}/h={horizon}")
            errors = np.asarray([float(r["crps_mcse"]) for r in group])
            mcse_by_internal[(model, factor, horizon)] = float(
                math.sqrt(np.sum(errors**2)) / len(group))
    by_garch = {(r["factor"], int(r["horizon"])): r for r in garch_metrics}
    rows = []
    metric_names = ("mean_crps", "mean_dss", "rmse", "coverage90", "mean_width90")
    for factor, horizon in keys:
        row = {"factor": factor, "horizon": horizon}
        model_scores = {}
        for manuscript_model in REPORTING_MODELS:
            if manuscript_model == "AR1_GARCH11":
                source = by_garch[(factor, horizon)]
            else:
                source = by_internal[(MANUSCRIPT_TO_INTERNAL[manuscript_model], factor, horizon)]
            for metric in metric_names:
                value = source.get(metric)
                row[f"{manuscript_model}_{metric}"] = (
                    float(value) if value not in (None, "") else None)
            model_scores[manuscript_model] = row[f"{manuscript_model}_mean_crps"]
            if manuscript_model == "AR1_GARCH11":
                mcse = source.get("crps_mean_mcse")
            else:
                internal_model = MANUSCRIPT_TO_INTERNAL[manuscript_model]
                mcse = mcse_by_internal[(internal_model, factor, horizon)]
            row[f"{manuscript_model}_crps_mean_mcse"] = (
                float(mcse) if mcse not in (None, "") else None)
        garch_score = model_scores["AR1_GARCH11"]
        for model in ("M1", "M2", "M3"):
            row[f"garch_minus_{model}_crps"] = (
                garch_score - model_scores[model]
                if garch_score is not None and model_scores[model] is not None else None)
        available = {name: score for name, score in model_scores.items() if score is not None}
        row["lowest_crps_specification"] = (
            min(available, key=available.get) if available else None)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError(f"Refusing to write a headerless empty CSV: {path}")
    columns = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _verify_design(design: dict, origins: list[int]) -> list[dict]:
    forecast_path = ACCEPTED_OUTPUT / "factor_forecasts.csv"
    with forecast_path.open(newline="", encoding="utf-8") as stream:
        accepted = list(csv.DictReader(stream))
    expected_dates = [design["dates"][i].isoformat() for i in origins]
    accepted_dates = sorted({row["origin"] for row in accepted})
    if (len(accepted) != 3120 or accepted_dates != expected_dates
            or float(design["lambda"]) != LAMBDA
            or len(design["factors"]) != len(design["dates"])):
        raise ValueError("Accepted factor forecast/design does not match the fixed experiment")
    by_key = {(row["origin"], int(row["horizon"]), row["model"], row["factor"]): row
              for row in accepted}
    if len(by_key) != len(accepted):
        raise ValueError("Accepted factor forecasts contain duplicate model records")
    selected = select_origins(
        design["dates"], date(2016, 1, 1), date(2025, 12, 31), [1, 4], 4)
    for origin_index, horizon, target_index in selected:
        origin = design["dates"][origin_index].isoformat()
        target = design["dates"][target_index].isoformat()
        for model in ("M0", "M2", "M3", "M4"):
            for factor in FACTORS:
                row = by_key.get((origin, horizon, model, factor))
                if row is None or row["target"] != target:
                    raise ValueError(
                        f"Accepted origin/target mismatch: {origin}, {model}/{factor}, h={horizon}")
    return accepted


def run(output: Path = OUTPUT) -> dict:
    design = make_design()
    origins = expected_origins(design)
    accepted = _verify_design(design, origins)
    records: list[dict] = []
    diagnostics: list[dict] = []
    for count, origin_index in enumerate(origins, start=1):
        origin = design["dates"][origin_index].isoformat()
        for factor_index, factor in enumerate(FACTORS):
            training = design["factors"][design["first_index"]:origin_index + 1, factor_index]
            try:
                fit = fit_ar1_garch(training, factor=factor, origin=origin)
                diagnostics.append(fit.record.copy())
            except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as error:
                diagnostics.append({
                    "model": "AR1_GARCH11", "factor": factor, "origin": origin,
                    "training_observations": len(training), "converged": False,
                    "optimizer_success": False, "error_type": type(error).__name__,
                    "error": str(error), "warnings": json.dumps([str(error)]),
                })
                continue
            if not fit.record["converged"]:
                continue
            for horizon in HORIZONS:
                target = design["dates"][origin_index + horizon].isoformat()
                observed = float(design["factors"][origin_index + horizon, factor_index])
                scenario_seed = _seed(origin_index, factor_index, horizon) if horizon > 1 else None
                rng = np.random.default_rng(scenario_seed) if scenario_seed is not None else None
                records.append(_forecast_record(
                    factor, origin, target, horizon, observed,
                    float(training[-1]), fit, rng, scenario_seed))
        if count % 10 == 0:
            print(f"Completed GARCH fits through origin {count}/{len(origins)}", flush=True)

    metrics = _aggregate(records)
    calibration = _calibration(records)
    comparison = _accepted_metrics(metrics)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "garch_factor_forecasts.csv", records, [
        "model", "factor", "origin", "target", "horizon", "observed", "mean", "variance",
        "crps", "crps_mcse", "dss", "rmse_component", "mae", "pit", "pit_mcse",
        "predictive_mean_mcse", "predictive_variance_mcse", "scenario_seed",
        "lower50", "upper50", "width50", "covered50", "lower80", "upper80",
        "width80", "covered80", "lower90", "upper90", "width90", "covered90",
    ])
    _write_csv(output / "garch_factor_metrics.csv", metrics)
    _write_csv(output / "garch_fit_diagnostics.csv", diagnostics)
    _write_csv(output / "garch_calibration_diagnostics.csv", calibration)
    _write_csv(output / "garch_vs_existing_models.csv", comparison)
    fit_success_count = sum(bool(row.get("converged")) for row in diagnostics)
    manifest = {
        "experiment": "independent Gaussian AR(1)-GARCH(1,1) models for extracted DNS factors",
        "model_identifier": "AR1_GARCH11",
        "status": "complete" if fit_success_count == 390 and len(records) == 780 else "incomplete",
        "data_sha256": _sha256(ROOT / "data" / "processed" / "canada_weekly.csv"),
        "accepted_factor_forecasts_sha256": _sha256(ACCEPTED_OUTPUT / "factor_forecasts.csv"),
        "accepted_factor_metrics_sha256": _sha256(ACCEPTED_OUTPUT / "factor_metrics.csv"),
        "repository_base_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip(),
        "git_branch": subprocess.run(
            ["git", "-C", str(ROOT), "branch", "--show-current"],
            check=True, capture_output=True, text=True).stdout.strip(),
        "code_sha256": {
            "garch_factor_experiment": _sha256(Path(__file__)),
            "plot_factor_histories": _sha256(ROOT / "src" / "plot_factor_histories.py"),
            "garch_tests": _sha256(ROOT / "tests" / "test_garch_factor_experiment.py"),
            "requirements_lock": _sha256(ROOT / "requirements-lock.txt"),
        },
        "fixed_lambda": float(design["lambda"]),
        "maturities_years": MATURITIES,
        "factor_units": "percentage points",
        "optimization_units": "basis points; intercept divided by 100 and variances by 10000 on output",
        "research_observations": 1356,
        "research_start": design["dates"][design["first_index"]].isoformat(),
        "research_end": "2025-12-24",
        "initial_training_observations": 835,
        "initial_training_end": "2015-12-31",
        "origin_count": len(origins),
        "first_origin": design["dates"][origins[0]].isoformat(),
        "last_origin": design["dates"][origins[-1]].isoformat(),
        "horizons": list(HORIZONS),
        "expected_fit_count": 390,
        "completed_fit_count": len(diagnostics),
        "successful_fit_count": fit_success_count,
        "expected_forecast_count": 780,
        "completed_forecast_count": len(records),
        "scenario_count_horizon_four": SCENARIOS,
        "scenario_batches": BATCHES,
        "scenario_seed_base": BASE_SEED,
        "seed_derivation": "[base seed, absolute origin row index, factor index, horizon]",
        "converged_fit_count_by_factor": {
            factor: sum(r.get("factor") == factor and bool(r.get("converged"))
                        for r in diagnostics) for factor in FACTORS},
        "garch_persistence_at_or_above_one": sum(
            bool(r.get("garch_nonstationary")) for r in diagnostics),
        "ar_coefficient_at_or_above_unit_magnitude": sum(
            bool(r.get("ar_nonstationary")) for r in diagnostics),
        "manuscript_to_accepted_internal_model_mapping": MANUSCRIPT_TO_INTERNAL,
        "reporting_order": list(REPORTING_MODELS),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("arch", "numpy", "scipy", "matplotlib", "pandas")
        } | {"python": platform.python_version()},
        "outputs": {
            "forecasts": "garch_factor_forecasts.csv",
            "metrics": "garch_factor_metrics.csv",
            "fits": "garch_fit_diagnostics.csv",
            "calibration": "garch_calibration_diagnostics.csv",
            "comparison": "garch_vs_existing_models.csv",
        },
        "validation": {
            "accepted_origin_and_target_alignment": True,
            "accepted_model_label_mapping_explicit": True,
            "all_390_fits_converged": fit_success_count == 390,
            "all_780_forecasts_completed": len(records) == 780,
            "six_complete_forecast_groups": all(row["n_origins"] == 130 for row in metrics),
            "fit_diagnostic_rows": len(diagnostics),
            "finite_forecasts_and_scores": bool(all(
                np.isfinite([r["mean"], r["variance"], r["crps"], r["dss"],
                             r["pit"], r["lower90"], r["upper90"]]).all()
                and r["variance"] > 0 for r in records)),
        },
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics, "comparison": comparison,
            "forecasts": records, "diagnostics": diagnostics, "accepted": accepted}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "status": result["manifest"]["status"],
        "fits": result["manifest"]["successful_fit_count"],
        "forecasts": result["manifest"]["completed_forecast_count"],
        "output": str(args.output),
    }, indent=2))
    return 0 if result["manifest"]["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
