"""Training-only prior-predictive check for a direct long-run factor-mean prior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

from src import prior_predictive_bayesian_dns as audit
from src.bayesian_stage_a import FACTORS, ROOT, make_design


OUTPUT = ROOT / "results" / "prior_justification_20261114" / "longrun_mean_candidate"
SEED = 20261115
N_DRAWS = 100_000


def simulate_model(
    model: str,
    rng: np.random.Generator,
    n: int,
    factor_start: np.ndarray,
    initial_yield: np.ndarray,
    mu0: np.ndarray,
    long_run_mean_center: np.ndarray,
    long_run_mean_sd: np.ndarray,
    persistence_center: np.ndarray,
    loading: np.ndarray,
    reconstruction_variance: np.ndarray,
    failures: dict,
) -> dict:
    long_run_mean = rng.normal(long_run_mean_center, long_run_mean_sd, size=(n, 3))
    persistence_sd = 0.03
    b = np.column_stack([
        truncnorm.rvs(
            (-1 - persistence_center[j]) / persistence_sd,
            (1 - persistence_center[j]) / persistence_sd,
            loc=persistence_center[j],
            scale=persistence_sd,
            size=n,
            random_state=rng,
        )
        for j in range(3)
    ])
    a = (1 - b) * long_run_mean
    m = mu0[None, :] + rng.normal(0, 1, size=(n, 3))
    factors = np.broadcast_to(factor_start, (n, 3)).copy()
    factor_path = []

    if model == "M4":
        phi = 2 * rng.beta(20, 1.5, size=(n, 3)) - 1
        v = -np.log1p(
            -rng.random((n, 3)) * (1 - np.exp(-4.0))
        )
        omega = m - v / 2
        h = omega + np.sqrt(v) * rng.normal(size=(n, 3))
        q = v * (1 - phi**2)
    elif model != "M3":
        raise ValueError(f"Unknown model: {model}")

    for week in range(1, 5):
        if model == "M3":
            log_sigma = m / 2
        else:
            h = omega + phi * (h - omega) + np.sqrt(q) * rng.normal(size=(n, 3))
            log_sigma = h / 2
        sigma = audit.checked_exp(log_sigma, f"{model}_innovation_sd_week_{week}", failures)
        factors = a + b * factors + sigma * rng.normal(size=(n, 3))
        if not np.isfinite(factors).all():
            label = f"{model}_factor_state_week_{week}"
            failures["nonfinite_by_quantity"][label] = int(np.count_nonzero(~np.isfinite(factors)))
            raise FloatingPointError(f"Nonfinite {model} factor states")
        failures["nonfinite_by_quantity"][f"{model}_factor_state_week_{week}"] = 0
        factor_path.append(factors.copy())

    forecasts = {}
    measurement_sd = np.sqrt(reconstruction_variance)
    for horizon in (1, 4):
        future = factor_path[horizon - 1]
        levels = future @ loading.T + rng.normal(size=(n, 8)) * measurement_sd
        changes = levels - initial_yield
        if not np.isfinite(levels).all() or not np.isfinite(changes).all():
            label = f"{model}_yield_forecast_h{horizon}"
            failures["nonfinite_by_quantity"][label] = int(
                np.count_nonzero(~np.isfinite(levels)) + np.count_nonzero(~np.isfinite(changes))
            )
            raise FloatingPointError(f"Nonfinite {model} yield scenarios at horizon {horizon}")
        failures["nonfinite_by_quantity"][f"{model}_yield_forecast_h{horizon}"] = 0
        forecasts[horizon] = {
            "factor_levels": future,
            "factor_changes": future - factor_start,
            "yield_levels": levels,
            "yield_changes": changes,
        }
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(long_run_mean).all():
        raise FloatingPointError(f"Nonfinite parameter draws for {model}")
    return {
        "a": a,
        "b": b,
        "m": m,
        "long_run_mean": long_run_mean,
        "m_sd": 1.0,
        "forecasts": forecasts,
        **({"stationary_log_variance": h} if model == "M4" else {}),
    }


def parameter_rows(
    model: str,
    draws: dict,
    factor_center: np.ndarray,
    factor_sd: np.ndarray,
    persistence_center: np.ndarray,
    observed_factors: np.ndarray,
) -> list[dict]:
    rows = []
    for j, factor in enumerate(FACTORS):
        factor_min = float(np.min(observed_factors[:, j]))
        factor_max = float(np.max(observed_factors[:, j]))
        for parameter, values in [
            ("intercept_a", draws["a"][:, j]),
            ("persistence_b", draws["b"][:, j]),
            ("long_run_factor_mean", draws["long_run_mean"][:, j]),
        ]:
            q01, q05, q50, q95, q99 = audit.quantiles(values)
            rows.append({
                "model": model,
                "factor": factor,
                "parameter": parameter,
                "prior_q01": q01,
                "prior_q05": q05,
                "prior_median": q50,
                "prior_q95": q95,
                "prior_q99": q99,
                "training_factor_mean": float(factor_center[j]),
                "training_factor_sd": float(factor_sd[j]),
                "training_ols_persistence": float(persistence_center[j]),
                "training_factor_min": factor_min,
                "training_factor_max": factor_max,
                "training_factor_range": float(np.ptp(observed_factors[:, j])),
                "probability_long_run_mean_outside_training_min_max": (
                    float(np.mean((values < factor_min) | (values > factor_max)))
                    if parameter == "long_run_factor_mean" else None
                ),
                "long_run_mean_prior_second_moment_finite": (
                    True if parameter == "long_run_factor_mean" else None
                ),
                "long_run_mean_prior_absolute_mean_finite": (
                    True if parameter == "long_run_factor_mean" else None
                ),
            })
    return rows


def run(output: Path, n_draws: int = N_DRAWS, seed: int = SEED) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    design = make_design()
    initial_yields = design["y"][:design["initial_n"]]
    initial_factors = design["factors"][:design["initial_n"]]
    if len(initial_yields) != 835 or design["initial_last_date"] != "2015-12-31":
        raise ValueError("The prior-predictive conditioning window changed")
    residuals = initial_yields - initial_factors @ design["load"].T
    reconstruction_variance = np.maximum(np.mean(residuals**2, axis=0), 1e-6)
    mu0 = np.array([prior.omega_mean for prior in design["priors"]])
    ols_coefficients = np.column_stack([
        np.linalg.lstsq(
            np.column_stack((np.ones(len(initial_factors) - 1), initial_factors[:-1, j])),
            initial_factors[1:, j],
            rcond=None,
        )[0]
        for j in range(3)
    ])
    factor_center = np.mean(initial_factors, axis=0)
    factor_sd = np.std(initial_factors, axis=0, ddof=1)
    long_run_mean_sd = 2 * factor_sd
    failures = {"overflow_by_quantity": {}, "underflow_by_quantity": {}, "nonfinite_by_quantity": {}}
    factor_summary, parameter_summary = [], []
    level_summary, change_summary, curve_summary, innovation_summary = [], [], [], []
    curve_samples, training_rms = {}, {}

    for model, child_seed in zip(("M3", "M4"), np.random.SeedSequence(seed).spawn(2)):
        rng = np.random.default_rng(child_seed)
        draws = simulate_model(
            model, rng, n_draws, initial_factors[-1], initial_yields[-1], mu0,
            factor_center, long_run_mean_sd, ols_coefficients[1],
            design["load"], reconstruction_variance, failures,
        )
        parameter_summary.extend(parameter_rows(
            model, draws, factor_center, factor_sd, ols_coefficients[1], initial_factors
        ))
        for horizon in (1, 4):
            factor_summary.extend(audit.factor_rows(model, horizon, draws, initial_factors))
            levels, changes = audit.yield_rows(model, horizon, draws, initial_yields)
            level_summary.extend(levels)
            change_summary.extend(changes)
            curve, rms = audit.curve_rows(model, horizon, draws, initial_yields)
            curve_summary.append(curve)
            curve_samples[(model, horizon)] = rms
            historical = initial_yields[horizon:] - initial_yields[:-horizon]
            training_rms[horizon] = np.sqrt(np.mean(historical**2, axis=1))
        for j, factor in enumerate(FACTORS):
            log_variance = draws["m"][:, j] if model == "M3" else draws["stationary_log_variance"][:, j]
            values = audit.checked_exp(log_variance, f"{model}_{factor}_innovation_variance", failures)
            q = audit.quantiles(values)
            innovation_summary.append({
                "model": model,
                "factor": factor,
                "innovation_variance_q01": q[0],
                "innovation_variance_q05": q[1],
                "innovation_variance_median": q[2],
                "innovation_variance_q95": q[3],
                "innovation_variance_q99": q[4],
                "analytic_prior_mean_innovation_variance": float(np.exp(mu0[j] + 0.5)),
                "training_ols_residual_variance": float(np.exp(mu0[j])),
            })

    for name, rows in [
        ("prior_predictive_factor_changes.csv", factor_summary),
        ("prior_parameter_predictive.csv", parameter_summary),
        ("prior_predictive_yield_levels.csv", level_summary),
        ("prior_predictive_yield_changes.csv", change_summary),
        ("prior_predictive_curve_scale.csv", curve_summary),
        ("prior_predictive_innovation_variances.csv", innovation_summary),
    ]:
        audit.write_rows(output / name, rows)
    audit.make_plot(curve_samples, training_rms, output / "prior_predictive_curve_scale.png")
    source_files = [
        ROOT / "data" / "processed" / "canada_weekly.csv",
        ROOT / "results" / "bayesian_revised_pilot" / "extended" / "metadata.json",
        ROOT / "src" / "bayesian_stage_a.py",
        ROOT / "src" / "prior_predictive_bayesian_dns.py",
        Path(__file__),
    ]
    metadata = {
        "status": "completed",
        "prior_set": "training-calibrated direct long-run factor mean",
        "seed": seed,
        "n_draws_per_model": n_draws,
        "models": ["M3", "M4"],
        "presentation_model_mapping": {"M3": "internal M4 constant volatility", "M4": "internal M5 stochastic volatility"},
        "training_period": ["2000-01-01", "2015-12-31"],
        "training_observations": int(len(initial_yields)),
        "conditioning_date": design["initial_last_date"],
        "fixed_lambda_per_year": float(design["lambda"]),
        "factor_prior": {
            "long_run_mean": "Normal(training factor mean, (2 * training factor sd)^2)",
            "persistence": "Normal(training OLS AR coefficient, 0.03^2) truncated to (-1,1)",
            "intercept": "a = (1-b) * long_run_factor_mean",
        },
        "innovation_priors": {
            "m": "Normal(training OLS log-MSE, 1)",
            "phi": "2*Beta(20,1.5)-1",
            "v": "rate-one exponential truncated to (0,4); M4 only",
        },
        "factor_prior_centers": factor_center.tolist(),
        "factor_prior_standard_deviations": long_run_mean_sd.tolist(),
        "training_ols_coefficients": ols_coefficients.T.tolist(),
        "measurement_error_variances": reconstruction_variance.tolist(),
        "input_sha256": design["input_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)): audit.digest(path) for path in source_files},
        "numerical_failures": failures,
        "all_draws_retained": True,
        "forecast_scores_used_to_choose_priors": False,
    }
    (output / "prior_predictive_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.draws < 10_000:
        parser.error("--draws must be at least 10000 for tail diagnostics")
    result = run(args.output.resolve(), args.draws, args.seed)
    print(json.dumps({
        "status": result["status"],
        "draws_per_model": result["n_draws_per_model"],
        "seed": result["seed"],
        "output": str(args.output.resolve()),
        "numerical_failures": result["numerical_failures"],
    }))


if __name__ == "__main__":
    main()
