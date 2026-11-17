"""Evaluate accepted Bayesian DNS factor predictions against M0 and M2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from src.bayesian_finite_validation import factor_conditionals, factor_scenarios
from src.bayesian_prior_longrun import validate_draws
from src.bayesian_stage_a import FACTORS, make_design
from src.scores import stationary_bootstrap_indices

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "bayesian_prior_revised_20261114"
PROTOCOL = ROOT / "docs" / "factor_probabilistic_20261116" / "FACTOR_EVALUATION_PROTOCOL.md"
OUTPUT = ROOT / "results" / "factor_probabilistic_20261116"
MODELS = ("M0", "M2", "M3", "M4")
ARCHIVE_MODELS = {"M3": "M4", "M4": "M5"}
HORIZONS = (1, 4)
INTERVALS = (0.50, 0.80, 0.90)
SCENARIOS = 8192
BATCHES = 8
BASE_SEED = 20261117
BOOTSTRAP_SEED = 20261118


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def gaussian_crps(observed: float, mean: float, variance: float) -> float:
    """Analytic CRPS for a non-degenerate Gaussian predictive distribution."""
    _require(np.isfinite([observed, mean, variance]).all() and variance > 0,
             "CRPS requires finite inputs and positive variance")
    sd = math.sqrt(variance)
    z = (observed - mean) / sd
    return float(sd * (z * (2 * ndtr(z) - 1)
                       + 2 * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
                       - 1 / math.sqrt(math.pi)))


def _baseline_forecast(training: np.ndarray, model: str, horizon: int) -> tuple[float, float]:
    if horizon not in HORIZONS or model not in ("M0", "M2"):
        raise ValueError("Invalid baseline forecast specification")
    training = np.asarray(training, dtype=float)
    if training.ndim != 1 or len(training) < 3 or not np.isfinite(training).all():
        raise ValueError("Baseline training data must be a finite factor series")
    if model == "M0":
        variance = horizon * float(np.var(np.diff(training), ddof=1))
        mean = float(training[-1])
    else:
        design = np.column_stack((np.ones(len(training) - 1), training[:-1]))
        response = training[1:]
        intercept, persistence = np.linalg.lstsq(design, response, rcond=None)[0]
        residuals = response - design @ np.array([intercept, persistence])
        innovation_variance = float(np.mean(residuals**2))
        mean = float(persistence**horizon * training[-1]
                     + intercept * sum(persistence**k for k in range(horizon)))
        variance = float(innovation_variance
                         * sum(persistence**(2 * k) for k in range(horizon)))
    if not np.isfinite([mean, variance]).all() or variance <= 0:
        raise ValueError(f"{model} produced an invalid factor forecast")
    return mean, variance


def _archive_plan(design: dict) -> tuple[dict, dict]:
    plan_path = ARCHIVE / "plan.json"
    summary_path = ARCHIVE / "summary.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(_sha256(plan_path) == summary["plan_sha256"], "Accepted plan hash mismatch")
    _require(summary["accepted"] and summary["planned_origins"] == 130
             and summary["completed_origins"] == 130
             and not summary["failed_origins"] and not summary["not_completed_dates"],
             "Accepted posterior archive is incomplete")
    _require(plan["input_sha256"] == design["input_sha256"]
             and plan["lambda"] == design["lambda"],
             "Accepted archive does not match the fixed factor panel")
    _require(plan["maturities_years"] == [0.25, 0.5, 1, 2, 5, 7, 10, 30]
             and plan["models"] == ["M4", "M5"]
             and plan["horizons"] == [1, 4]
             and plan["draws"] == 6000 and plan["chains"] == 4,
             "Accepted archive specification changed")
    _require(len(plan["origins"]) == 130, "Unexpected accepted origin count")
    return plan, summary


def _verify_origin_files(plan: dict, summary: dict, row: dict, design: dict) -> dict:
    day = row["date"]
    folder = ARCHIVE / "origins" / day
    forecast_path = folder / "forecast.json"
    _require(summary["forecasts_sha256"].get(day) == _sha256(forecast_path),
             f"Forecast hash mismatch at {day}")
    report = json.loads(forecast_path.read_text(encoding="utf-8"))
    spec = report["specification"]
    _require(report["accepted"] and spec["origin"] == day
             and spec["origin_index"] == row["index"]
             and spec["seed"] == row["seed"]
             and spec["input_sha256"] == plan["input_sha256"]
             and spec["lambda"] == plan["lambda"],
             f"Accepted forecast metadata mismatch at {day}")
    _require(spec["source_sha256"] == plan["source_sha256"],
             f"Accepted source manifest mismatch at {day}")
    prediction_path = folder / "predictions.npz"
    _require(_sha256(prediction_path) == report["prediction_archive_sha256"],
             f"Prediction archive hash mismatch at {day}")
    chain_dir = folder / "chains"
    chain_summary_path = chain_dir / "summary.json"
    chain_summary = json.loads(chain_summary_path.read_text(encoding="utf-8"))
    _require(_sha256(chain_summary_path) == report["parameter_summary_sha256"],
             f"Chain summary hash mismatch at {day}")
    _require(chain_summary["parameter_gate_passed"] and chain_summary["moment_gate_passed"]
             and all(item["passes"] for item in chain_summary["diagnostics"])
             and all(item["posterior_proper"]
                     and item["finite_predictive_second_moment"]
                     and item["finite_predictive_fourth_moment"]
                     and item["stage_a_mathematical_gate_passed"]
                     for item in chain_summary["moment_audits"]),
             f"Accepted chain diagnostics failed at {day}")
    window = chain_summary["configuration"]["window"]
    _require(window["index"] == row["index"] and window["date"] == day
             and window["n_training"] == row["index"],
             f"Training boundary mismatch at {day}")
    expected_checkpoints = {
        Path(path).name: digest for path, digest in report["chain_checkpoint_sha256"].items()
    }
    _require(chain_summary["checkpoints"] == expected_checkpoints
             and len(expected_checkpoints) == 24,
             f"Chain checkpoint manifest mismatch at {day}")
    _require(len(chain_summary["chain_seeds"]) == 24
             and len(set(chain_summary["chain_seeds"])) == 24,
             f"Chain seeds are not unique at {day}")
    for filename, digest in expected_checkpoints.items():
        _require(_sha256(chain_dir / filename) == digest,
                 f"Posterior checkpoint hash mismatch: {day}/{filename}")
    _require(len(report["records"]) == 4, f"Unexpected forecast records at {day}")
    for model in plan["models"]:
        for horizon in HORIZONS:
            target = design["dates"][row["index"] + horizon].isoformat()
            record = next((item for item in report["records"]
                           if item["model"] == model and item["horizon"] == horizon), None)
            _require(record is not None and record["origin"] == day
                     and record["target"] == target,
                     f"Accepted target mismatch at {day}, {model}, h={horizon}")
    return {"report": report, "chain_summary": chain_summary, "chain_dir": chain_dir}


def _load_origin_draws(plan: dict, row: dict, validated: dict) -> dict:
    grouped = {}
    chain_dir = validated["chain_dir"]
    chain_summary = validated["chain_summary"]
    for model in plan["models"]:
        for factor in FACTORS:
            chains = []
            names = None
            for chain in range(4):
                filename = f"{model}_{factor}_chain{chain}.npz"
                with np.load(chain_dir / filename, allow_pickle=False) as archive:
                    draws = archive["draws"].copy()
                    chain_report = json.loads(str(archive["report"]))
                _require(draws.shape[0] == plan["draws"]
                         and chain_report["signature"] == chain_summary["signature"]
                         and chain_report["seed"] in chain_summary["chain_seeds"]
                         and chain_report["seed"] == (
                             row["seed"] + 100000 * (model == "M5")
                             + 1000 * FACTORS.index(factor) + chain),
                         f"Invalid posterior checkpoint {row['date']}/{filename}")
                validate_draws(draws, chain_report["names"], model)
                _require(names is None or names == chain_report["names"],
                         f"Posterior parameter order changed at {row['date']}")
                names = chain_report["names"]
                chains.append(draws)
            grouped[(model, factor)] = (np.stack(chains), names)
    return grouped


def _verify_factor_moments(plan: dict, row: dict, design: dict,
                           validated: dict, grouped: dict) -> None:
    report = validated["report"]
    for model in plan["models"]:
        for horizon in HORIZONS:
            record = next(item for item in report["records"]
                          if item["model"] == model and item["horizon"] == horizon)
            means, variances = [], []
            for j, factor in enumerate(FACTORS):
                values, names = grouped[(model, factor)]
                conditional_mean, conditional_variance = factor_conditionals(
                    values, names, design["factors"][row["index"], j],
                    horizon, model)
                means.append(float(conditional_mean.mean()))
                variances.append(float(conditional_variance.mean()
                                       + conditional_mean.var(ddof=0)))
            _require(np.allclose(means, record["factor_mean"], rtol=0, atol=1e-12)
                     and np.allclose(variances, record["factor_variance"],
                                     rtol=0, atol=1e-12),
                     f"Accepted factor moments changed at {row['date']}, {model}, h={horizon}")


def _rng(origin_index: int, model: str, factor_index: int, horizon: int,
         stream: int) -> np.random.Generator:
    model_index = {"M3": 3, "M4": 4}[model]
    return np.random.default_rng(np.random.SeedSequence(
        [BASE_SEED, origin_index, model_index, factor_index, horizon, stream]))


def stochastic_volatility_pit(values: np.ndarray, names: list[str], factor_value: float,
                              observed: float, horizon: int,
                              rng: np.random.Generator, size: int = SCENARIOS,
                              batches: int = BATCHES) -> tuple[float, float]:
    """Rao-Blackwellized M4 PIT over posterior draws and future volatility paths."""
    if horizon not in HORIZONS or size < batches or size % batches:
        raise ValueError("Invalid stochastic-volatility PIT simulation size")
    values = np.asarray(values, dtype=float).reshape(-1, len(names))
    if not np.isfinite(values).all() or not np.isfinite([factor_value, observed]).all():
        raise ValueError("PIT inputs must be finite")
    required = ("a", "b", "m", "phi", "v", "h_last")
    if any(name not in names for name in required):
        raise ValueError("Stochastic-volatility draws are missing required parameters")

    chosen = values[rng.integers(len(values), size=size)]
    get = lambda name: chosen[:, names.index(name)]
    a, b = get("a"), get("b")
    phi, v, m, h = get("phi"), get("v"), get("m"), get("h_last")
    omega = m - v / 2
    innovation_sd = np.sqrt(v * (1 - phi**2))
    conditional_variance = np.zeros(size)
    conditional_mean = b**horizon * factor_value + a * sum(
        b**k for k in range(horizon))
    with np.errstate(over="raise", invalid="raise"):
        for step in range(1, horizon + 1):
            h = omega + phi * (h - omega) + innovation_sd * rng.normal(size=size)
            conditional_variance += (
                b ** (2 * (horizon - step)) * np.exp(h))
        cdf_values = ndtr((observed - conditional_mean)
                          / np.sqrt(conditional_variance))
    if (not np.isfinite(cdf_values).all()
            or np.any((cdf_values < 0) | (cdf_values > 1))):
        raise ValueError("Invalid stochastic-volatility PIT values")
    batch_means = cdf_values.reshape(batches, size // batches).mean(axis=1)
    return float(cdf_values.mean()), float(batch_means.std(ddof=1) / math.sqrt(batches))


def gaussian_mixture_pit(conditional_mean: np.ndarray,
                         conditional_variance: np.ndarray,
                         observed: float) -> float:
    """Exact finite-mixture CDF for conditionally Gaussian forecasts."""
    conditional_mean = np.asarray(conditional_mean, dtype=float).reshape(-1)
    conditional_variance = np.asarray(conditional_variance, dtype=float).reshape(-1)
    if (conditional_mean.ndim != 1
            or conditional_mean.shape != conditional_variance.shape
            or not len(conditional_mean)
            or not np.isfinite(conditional_mean).all()
            or not np.isfinite(conditional_variance).all()
            or np.any(conditional_variance <= 0)
            or not np.isfinite(observed)):
        raise ValueError("Invalid Gaussian-mixture PIT inputs")
    return float(np.mean(ndtr((observed - conditional_mean)
                              / np.sqrt(conditional_variance))))


def _bayesian_forecast(values: np.ndarray, names: list[str], archive_model: str,
                       factor_value: float, observed: float, horizon: int,
                       model: str, origin_index: int, factor_index: int) -> tuple[dict, np.ndarray]:
    conditional_mean, conditional_variance = factor_conditionals(
        values, names, factor_value, horizon, archive_model)
    mean = float(conditional_mean.mean())
    variance = float(conditional_variance.mean()
                     + conditional_mean.var(ddof=0))
    _require(np.isfinite([mean, variance]).all() and variance > 0,
             f"Invalid {model} predictive moments")
    samples = factor_scenarios(
        values, names, factor_value, horizon, archive_model,
        _rng(origin_index, model, factor_index, horizon, 0), SCENARIOS)
    independent = factor_scenarios(
        values, names, factor_value, horizon, archive_model,
        _rng(origin_index, model, factor_index, horizon, 1), SCENARIOS)
    batch_size = SCENARIOS // BATCHES
    batch_scores = []
    for batch in range(BATCHES):
        sl = slice(batch * batch_size, (batch + 1) * batch_size)
        batch_scores.append(float(np.mean(np.abs(samples[sl] - observed))
                                  - 0.5 * np.mean(np.abs(samples[sl] - independent[sl]))))
    crps = float(np.mean(batch_scores))
    mcse = float(np.std(batch_scores, ddof=1) / math.sqrt(BATCHES))
    if model == "M3":
        pit = gaussian_mixture_pit(
            conditional_mean, conditional_variance, observed)
        pit_mcse = 0.0
    else:
        pit, pit_mcse = stochastic_volatility_pit(
            values, names, factor_value, observed, horizon,
            _rng(origin_index, model, factor_index, horizon, 2))
    intervals = {}
    for level in INTERVALS:
        low, high = np.quantile(samples, [(1 - level) / 2, 1 - (1 - level) / 2])
        intervals[level] = (float(low), float(high))
    return {
        "mean": mean,
        "variance": variance,
        "crps": crps,
        "crps_mcse": mcse,
        "pit": pit,
        "pit_mcse": pit_mcse,
        "intervals": intervals,
    }, samples


def _gaussian_forecast(mean: float, variance: float, observed: float) -> dict:
    sd = math.sqrt(variance)
    z = float((observed - mean) / sd)
    intervals = {}
    for level in INTERVALS:
        q = float(norm.ppf((1 + level) / 2))
        intervals[level] = (float(mean - q * sd), float(mean + q * sd))
    return {
        "mean": mean,
        "variance": variance,
        "crps": gaussian_crps(observed, mean, variance),
        "crps_mcse": 0.0,
        "pit": float(ndtr(z)),
        "pit_mcse": 0.0,
        "intervals": intervals,
    }


def _add_forecast_record(records: list, risk_records: list, *, model: str,
                         factor: str, origin: str, target: str,
                         horizon: int, observed: float, current: float,
                         forecast: dict, samples: np.ndarray | None,
                         loading_5y: float) -> None:
    mean, variance = forecast["mean"], forecast["variance"]
    record = {
        "model": model,
        "factor": factor,
        "origin": origin,
        "target": target,
        "horizon": horizon,
        "observed": observed,
        "mean": mean,
        "variance": variance,
        "crps": forecast["crps"],
        "crps_mcse": forecast["crps_mcse"],
        "dss": float(math.log(variance) + (observed - mean) ** 2 / variance),
        "rmse_component": float((observed - mean) ** 2),
        "mae": float(abs(observed - mean)),
        "pit": forecast["pit"],
        "pit_mcse": forecast["pit_mcse"],
    }
    for level, (low, high) in forecast["intervals"].items():
        key = int(level * 100)
        record[f"lower{key}"] = low
        record[f"upper{key}"] = high
        record[f"width{key}"] = high - low
        record[f"covered{key}"] = int(low <= observed <= high)
    records.append(record)
    if samples is not None:
        delta_y_bp = loading_5y * (samples - current) * 100
        pnl = -1_000_000 * 5 * delta_y_bp / 10_000
        low, median, high = np.quantile(pnl, [0.05, 0.50, 0.95])
        risk_records.append({
            "model": model,
            "factor": factor,
            "origin": origin,
            "target": target,
            "horizon": horizon,
            "maturity_years": 5,
            "loading": loading_5y,
            "yield_contribution_p05_bp": float(np.quantile(delta_y_bp, 0.05)),
            "yield_contribution_median_bp": float(np.quantile(delta_y_bp, 0.50)),
            "yield_contribution_p95_bp": float(np.quantile(delta_y_bp, 0.95)),
            "linear_pnl_p05_dollars": float(low),
            "linear_pnl_median_dollars": float(median),
            "linear_pnl_p95_dollars": float(high),
            "illustration_only": True,
        })


def _paired_tests(records: list) -> dict:
    by_key = {(r["model"], r["factor"], r["horizon"], r["origin"]): r
              for r in records}
    differences = []
    keys = []
    for factor in FACTORS:
        for horizon in HORIZONS:
            series = []
            for origin in sorted({r["origin"] for r in records}):
                stochastic = by_key[("M4", factor, horizon, origin)]
                constant = by_key[("M3", factor, horizon, origin)]
                series.append(stochastic["crps"] - constant["crps"])
            differences.append(np.asarray(series, dtype=float))
            keys.append((factor, horizon))
    matrix = np.column_stack(differences)
    observed = matrix.mean(axis=0)
    def bootstrap_summary(block_length: int, seed: int) -> dict:
        bootstrap = np.empty((9999, matrix.shape[1]), dtype=float)
        rng = np.random.default_rng(seed)
        for rep in range(len(bootstrap)):
            indices = stationary_bootstrap_indices(
                len(matrix), expected_block_length=block_length, rng=rng)
            bootstrap[rep] = matrix[indices].mean(axis=0)
        null_bootstrap = bootstrap - observed
        raw_p = ((1 + np.sum(np.abs(null_bootstrap) >= np.abs(observed), axis=0))
                 / (len(bootstrap) + 1))
        order = np.argsort(raw_p)
        adjusted = np.empty_like(raw_p)
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (len(order) - rank) * raw_p[index])
            adjusted[index] = min(1.0, running)
        return {
            "expected_block_length_origins": block_length,
            "seed": seed,
            "bootstrap_ci95": [
                [float(x) for x in np.quantile(bootstrap[:, index], [0.025, 0.975])]
                for index in range(matrix.shape[1])
            ],
            "p_value_two_sided": [float(x) for x in raw_p],
            "p_value_holm": [float(x) for x in adjusted],
        }

    bootstrap_results = {
        4: bootstrap_summary(4, BOOTSTRAP_SEED),
        8: bootstrap_summary(8, BOOTSTRAP_SEED + 1),
        12: bootstrap_summary(12, BOOTSTRAP_SEED + 2),
    }
    tests = []
    for index, (factor, horizon) in enumerate(keys):
        mc_variances = [
            by_key[("M4", factor, horizon, origin)]["crps_mcse"] ** 2
            + by_key[("M3", factor, horizon, origin)]["crps_mcse"] ** 2
            for origin in sorted({r["origin"] for r in records})
        ]
        tests.append({
            "comparison": "M4 minus M3",
            "factor": factor,
            "horizon": horizon,
            "n_origins": int(matrix.shape[0]),
            "mean_crps_difference": float(observed[index]),
            "bootstrap_ci95": bootstrap_results[4]["bootstrap_ci95"][index],
            "p_value_two_sided": bootstrap_results[4]["p_value_two_sided"][index],
            "p_value_holm": bootstrap_results[4]["p_value_holm"][index],
            "mean_pair_mcse": float(np.mean([
                math.hypot(by_key[("M4", factor, horizon, origin)]["crps_mcse"],
                           by_key[("M3", factor, horizon, origin)]["crps_mcse"])
                for origin in sorted({r["origin"] for r in records})
            ])),
            "paired_mean_mcse_independence": float(
                math.sqrt(sum(mc_variances)) / matrix.shape[0]),
        })
    return {
        "bootstrap": {
            "method": "stationary bootstrap of paired origin differences",
            "repetitions": 9999,
            "expected_block_length_origins": 4,
            "seed": BOOTSTRAP_SEED,
            "common_origin_indices_across_tests": True,
            "holm_family_size": len(tests),
        },
        "bootstrap_robustness": {
            "role": "predeclared sensitivity; primary inference remains block length four",
            "repetitions": 9999,
            "common_origin_indices_across_tests": True,
            "results_by_test": [
                {
                    "factor": factor,
                    "horizon": horizon,
                    "mean_crps_difference": float(observed[index]),
                    "blocks": {
                        str(block_length): {
                            "bootstrap_ci95": result["bootstrap_ci95"][index],
                            "p_value_two_sided": result["p_value_two_sided"][index],
                            "p_value_holm": result["p_value_holm"][index],
                        }
                        for block_length, result in bootstrap_results.items()
                    },
                }
                for index, (factor, horizon) in enumerate(keys)
            ],
        },
        "tests": tests,
    }


def _aggregate(records: list) -> list:
    rows = []
    for model in MODELS:
        for factor in FACTORS:
            for horizon in HORIZONS:
                group = [r for r in records if r["model"] == model
                         and r["factor"] == factor and r["horizon"] == horizon]
                _require(len(group) == 130, f"Incomplete score group: {model}/{factor}/h={horizon}")
                row = {
                    "model": model,
                    "factor": factor,
                    "horizon": horizon,
                    "n_origins": len(group),
                    "mean_crps": float(np.mean([r["crps"] for r in group])),
                    "mean_crps_mcse": float(np.mean([r["crps_mcse"] for r in group])),
                    "mean_dss": float(np.mean([r["dss"] for r in group])),
                    "rmse": float(math.sqrt(np.mean([r["rmse_component"] for r in group]))),
                    "mae": float(np.mean([r["mae"] for r in group])),
                    "pit_mean": float(np.mean([r["pit"] for r in group])),
                    "mean_pit_mcse": float(np.mean([r["pit_mcse"] for r in group])),
                }
                for key in ("50", "80", "90"):
                    row[f"coverage{key}"] = float(np.mean([r[f"covered{key}"] for r in group]))
                    row[f"mean_width{key}"] = float(np.mean([r[f"width{key}"] for r in group]))
                bins = np.histogram([r["pit"] for r in group], bins=np.linspace(0, 1, 11))[0]
                for i, count in enumerate(bins):
                    row[f"pit_bin_{i + 1}"] = int(count)
                rows.append(row)
    return rows


def _calibration_diagnostics(records: list) -> list[dict]:
    diagnostics = []
    for model in MODELS:
        for factor in FACTORS:
            for horizon in HORIZONS:
                group = sorted(
                    (r for r in records if r["model"] == model
                     and r["factor"] == factor and r["horizon"] == horizon),
                    key=lambda r: r["origin"])
                _require(len(group) == 130,
                         f"Incomplete calibration group: {model}/{factor}/h={horizon}")
                standardized = np.asarray([
                    (r["observed"] - r["mean"]) / math.sqrt(r["variance"])
                    for r in group
                ])
                pit = np.asarray([r["pit"] for r in group])
                _require(np.isfinite(standardized).all()
                         and np.isfinite(pit).all()
                         and np.all((pit >= 0) & (pit <= 1)),
                         "Invalid calibration diagnostic inputs")
                row = {
                    "model": model,
                    "factor": factor,
                    "horizon": horizon,
                    "n_origins": len(group),
                    "standardized_error_mean": float(standardized.mean()),
                    "standardized_error_variance": float(standardized.var(ddof=1)),
                    "standardized_error_lag1": float(np.corrcoef(
                        standardized[:-1], standardized[1:])[0, 1]),
                    "pit_mean": float(pit.mean()),
                    "pit_mcse_mean": float(np.mean([r["pit_mcse"] for r in group])),
                    "pit_lag1": float(np.corrcoef(pit[:-1], pit[1:])[0, 1]),
                }
                for level in ("50", "80", "90"):
                    row[f"coverage{level}"] = float(np.mean(
                        [r[f"covered{level}"] for r in group]))
                    row[f"mean_width{level}"] = float(np.mean(
                        [r[f"width{level}"] for r in group]))
                diagnostics.append(row)
    return diagnostics


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(metrics: list, records: list, output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"M0": "#6b7280", "M2": "#2563eb", "M3": "#16a34a", "M4": "#dc2626"}
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    for row, factor in enumerate(FACTORS):
        for col, horizon in enumerate(HORIZONS):
            ax = axes[row, col]
            group = [m for m in metrics if m["factor"] == factor and m["horizon"] == horizon]
            ax.bar([m["model"] for m in group], [m["mean_crps"] for m in group],
                   color=[colors[m["model"]] for m in group])
            ax.set_title(f"{factor.title()}, h={horizon}")
            ax.set_ylabel("Mean CRPS (factor units)")
    fig.savefig(figures / "factor_crps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    for row, factor in enumerate(FACTORS):
        for col, horizon in enumerate(HORIZONS):
            ax = axes[row, col]
            for model in MODELS:
                group = [r["pit"] for r in records if r["model"] == model
                         and r["factor"] == factor and r["horizon"] == horizon]
                ax.hist(group, bins=np.linspace(0, 1, 11), histtype="step",
                        density=True, linewidth=1.5, color=colors[model], label=model)
            ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"{factor.title()}, h={horizon}")
            ax.set_xlabel("PIT")
            ax.set_ylabel("Density")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(figures / "factor_pit.png", dpi=180)
    plt.close(fig)
    return ["figures/factor_crps.png", "figures/factor_pit.png"]


def run(output: Path = OUTPUT) -> dict:
    design = make_design()
    plan, summary = _archive_plan(design)
    protocol_hash = _sha256(PROTOCOL)
    records, risk_records = [], []
    loading_5y = np.asarray(design["load"])[4, :]
    for row in plan["origins"]:
        validated = _verify_origin_files(plan, summary, row, design)
        grouped = _load_origin_draws(plan, row, validated)
        _verify_factor_moments(plan, row, design, validated, grouped)
        origin_index = row["index"]
        origin = row["date"]
        current_factors = design["factors"][origin_index]
        for horizon in HORIZONS:
            target_index = origin_index + horizon
            target = design["dates"][target_index].isoformat()
            for factor_index, factor in enumerate(FACTORS):
                observed = float(design["factors"][target_index, factor_index])
                current = float(current_factors[factor_index])
                training = design["factors"][design["first_index"]:origin_index + 1,
                                             factor_index]
                for model in ("M0", "M2"):
                    mean, variance = _baseline_forecast(training, model, horizon)
                    forecast = _gaussian_forecast(mean, variance, observed)
                    _add_forecast_record(
                        records, risk_records, model=model, factor=factor,
                        origin=origin, target=target,
                        horizon=horizon, observed=observed, current=current,
                        forecast=forecast, samples=None,
                        loading_5y=float(loading_5y[factor_index]))
                for model, archive_model in ARCHIVE_MODELS.items():
                    values, names = grouped[(archive_model, factor)]
                    forecast, samples = _bayesian_forecast(
                        values, names, archive_model, current, observed, horizon,
                        model, origin_index, factor_index)
                    _add_forecast_record(
                        records, risk_records, model=model, factor=factor,
                        origin=origin, target=target,
                        horizon=horizon, observed=observed, current=current,
                        forecast=forecast, samples=samples,
                        loading_5y=float(loading_5y[factor_index]))
    _require(len(records) == 3120 and len(risk_records) == 1560,
             "Unexpected factor forecast or risk-illustration count")
    metrics = _aggregate(records)
    calibration = _calibration_diagnostics(records)
    paired = _paired_tests(records)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "factor_forecasts.csv", records)
    _write_csv(output / "factor_metrics.csv", metrics)
    _write_csv(output / "factor_calibration_diagnostics.csv", calibration)
    _write_csv(output / "factor_risk_illustration.csv", risk_records)
    (output / "paired_tests.json").write_text(
        json.dumps(paired, indent=2, allow_nan=False), encoding="utf-8")
    figures = _write_plots(metrics, records, output)
    git_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "protocol_version": 1,
        "protocol_sha256": protocol_hash,
        "accepted_plan_sha256": _sha256(ARCHIVE / "plan.json"),
        "accepted_summary_sha256": _sha256(ARCHIVE / "summary.json"),
        "input_sha256": design["input_sha256"],
        "fixed_lambda": design["lambda"],
        "git_commit": git_commit,
        "forecast_records": len(records),
        "bayesian_factor_forecasts": sum(r["model"] in ARCHIVE_MODELS for r in records),
        "baseline_factor_forecasts": sum(r["model"] in ("M0", "M2") for r in records),
        "risk_illustration_records": len(risk_records),
        "accepted_chain_files_checked_per_origin": 24,
        "accepted_origins_checked": len(plan["origins"]),
        "scenario_count_per_bayesian_factor_forecast_and_stream": SCENARIOS,
        "simulation_batches": BATCHES,
        "scenario_seed": BASE_SEED,
        "source_sha256": {
            "experiment": _sha256(Path(__file__)),
            "bayesian_finite_validation": _sha256(
                ROOT / "src" / "bayesian_finite_validation.py"),
            "bayesian_prior_longrun": _sha256(
                ROOT / "src" / "bayesian_prior_longrun.py"),
            "bayesian_stage_a": _sha256(ROOT / "src" / "bayesian_stage_a.py"),
            "scores": _sha256(ROOT / "src" / "scores.py"),
        },
        "plots": figures,
        "calibration_diagnostics": "factor_calibration_diagnostics.csv",
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics, "paired_tests": paired}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "forecast_records": result["manifest"]["forecast_records"],
        "metrics_rows": len(result["metrics"]),
        "paired_tests": len(result["paired_tests"]["tests"]),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
