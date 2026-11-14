"""Prior-predictive simulation for the accepted Bayesian DNS M3/M4 specifications."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

from src.bayesian_stage_a import FACTORS, ROOT, make_design


OUTPUT = ROOT / "results" / "prior_justification_20261114"
SEED = 20261114
N_DRAWS = 100_000
V_UPPER = 4.0
V_RATE = 1.0
MATURITIES = np.array([0.25, 0.5, 1, 2, 5, 7, 10, 30])
MATURITY_LABELS = ("3m", "6m", "1y", "2y", "5y", "7y", "10y", "30y")


class NumericalAuditError(FloatingPointError):
    pass


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checked_exp(log_values: np.ndarray, label: str, failures: dict) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    over = values > np.log(np.finfo(float).max)
    under = values < np.log(np.nextafter(0.0, 1.0))
    failures["overflow_by_quantity"][label] = int(np.count_nonzero(over))
    failures["underflow_by_quantity"][label] = int(np.count_nonzero(under))
    if np.any(over):
        raise NumericalAuditError(
            f"{np.count_nonzero(over)} {label} draws exceed float64 range; none were clipped"
        )
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = np.exp(values)
    if not np.isfinite(result).all():
        failures["nonfinite_by_quantity"][label] = int(np.count_nonzero(~np.isfinite(result)))
        raise NumericalAuditError(f"Nonfinite values produced for {label}")
    return result


def quantiles(values: np.ndarray) -> tuple[float, float, float, float, float]:
    return tuple(float(x) for x in np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99]))


def prior_draws(
    rng: np.random.Generator,
    mu0: np.ndarray,
    n: int,
    prior_set: str,
    ols_coefficients: np.ndarray,
) -> dict:
    if prior_set == "current":
        a = rng.normal(0, 5, size=(n, 3))
        b_mean = np.full(3, 0.7)
        b_sd = np.full(3, 0.3)
        m_sd = 5.0
    elif prior_set == "candidate":
        a = rng.normal(ols_coefficients[0], 0.05, size=(n, 3))
        b_mean = ols_coefficients[1]
        b_sd = np.full(3, 0.02)
        m_sd = 1.0
    else:
        raise ValueError(f"Unknown prior set: {prior_set}")
    b = np.column_stack([
        truncnorm.rvs(
            (-1 - b_mean[j]) / b_sd[j],
            (1 - b_mean[j]) / b_sd[j],
            loc=b_mean[j],
            scale=b_sd[j],
            size=n,
            random_state=rng,
        )
        for j in range(3)
    ])
    m = mu0[None, :] + rng.normal(0, m_sd, size=(n, 3))
    long_run_mean = a / (1 - b)
    if not np.isfinite(long_run_mean).all():
        raise NumericalAuditError("Nonfinite prior long-run factor mean")
    return {
        "a": a,
        "b": b,
        "m": m,
        "long_run_mean": long_run_mean,
        "m_sd": m_sd,
    }


def simulate_model(
    model: str,
    rng: np.random.Generator,
    n: int,
    prior_set: str,
    ols_coefficients: np.ndarray,
    factor_start: np.ndarray,
    initial_yield: np.ndarray,
    mu0: np.ndarray,
    loading: np.ndarray,
    reconstruction_variance: np.ndarray,
    failures: dict,
) -> dict:
    parameters = prior_draws(rng, mu0, n, prior_set, ols_coefficients)
    a, b, m = parameters["a"], parameters["b"], parameters["m"]
    factors = np.broadcast_to(factor_start, (n, 3)).copy()
    factor_path = []

    if model == "M4":
        phi = 2 * rng.beta(20, 1.5, size=(n, 3)) - 1
        v = -np.log1p(
            -rng.random((n, 3)) * (1 - np.exp(-V_RATE * V_UPPER))
        ) / V_RATE
        omega = m - v / 2
        h = omega + np.sqrt(v) * rng.normal(size=(n, 3))
        q = v * (1 - phi**2)
        parameters.update({"phi": phi, "v": v})
    elif model != "M3":
        raise ValueError(f"Unknown model: {model}")

    for week in range(1, 5):
        if model == "M3":
            log_sigma = m / 2
        else:
            h = omega + phi * (h - omega) + np.sqrt(q) * rng.normal(size=(n, 3))
            log_sigma = h / 2
        sigma = checked_exp(log_sigma, f"{model}_innovation_sd_week_{week}", failures)
        innovation = sigma * rng.normal(size=(n, 3))
        if not np.isfinite(innovation).all():
            count = int(np.count_nonzero(~np.isfinite(innovation)))
            failures["nonfinite_by_quantity"][f"{model}_factor_innovation_week_{week}"] = count
            raise NumericalAuditError(f"{count} nonfinite factor innovations in {model}")
        factors = a + b * factors + innovation
        if not np.isfinite(factors).all():
            count = int(np.count_nonzero(~np.isfinite(factors)))
            failures["nonfinite_by_quantity"][f"{model}_factor_state_week_{week}"] = count
            raise NumericalAuditError(f"{count} nonfinite factor states in {model}")
        factor_path.append(factors.copy())

    forecasts = {}
    measurement_sd = np.sqrt(reconstruction_variance)
    for horizon in (1, 4):
        factor_future = factor_path[horizon - 1]
        observation_error = rng.normal(size=(n, 8)) * measurement_sd
        levels = factor_future @ loading.T + observation_error
        changes = levels - initial_yield
        if not np.isfinite(levels).all() or not np.isfinite(changes).all():
            raise NumericalAuditError(f"Nonfinite joint yield scenarios for {model}, h={horizon}")
        forecasts[horizon] = {
            "factor_levels": factor_future,
            "factor_changes": factor_future - factor_start,
            "yield_levels": levels,
            "yield_changes": changes,
        }
    parameters["factor_path"] = factor_path
    parameters["forecasts"] = forecasts
    if model == "M4":
        parameters["stationary_log_variance"] = h
    return parameters


def factor_rows(
    model: str,
    horizon: int,
    draws: dict,
    observed_factors: np.ndarray,
) -> list[dict]:
    rows = []
    for j, factor in enumerate(FACTORS):
        changes = draws["forecasts"][horizon]["factor_changes"][:, j]
        empirical = observed_factors[horizon:, j] - observed_factors[:-horizon, j]
        abs_cut_95, abs_cut_99 = np.quantile(np.abs(empirical), [0.95, 0.99])
        q01, q05, q50, q95, q99 = quantiles(changes)
        rows.append({
            "model": model,
            "horizon_weeks": horizon,
            "factor": factor,
            "predictive_change_q01": q01,
            "predictive_change_q05": q05,
            "predictive_change_median": q50,
            "predictive_change_q95": q95,
            "predictive_change_q99": q99,
            "training_abs_change_q95": float(abs_cut_95),
            "training_abs_change_q99": float(abs_cut_99),
            "prior_probability_abs_change_exceeds_training_q95": float(np.mean(np.abs(changes) > abs_cut_95)),
            "prior_probability_abs_change_exceeds_training_q99": float(np.mean(np.abs(changes) > abs_cut_99)),
        })
    return rows


def parameter_rows(
    model: str,
    draws: dict,
    ols_coefficients: np.ndarray,
    observed_factors: np.ndarray,
) -> list[dict]:
    rows = []
    for j, factor in enumerate(FACTORS):
        a = draws["a"][:, j]
        b = draws["b"][:, j]
        long_run = draws["long_run_mean"][:, j]
        factor_range = float(np.ptp(observed_factors[:, j]))
        factor_min = float(np.min(observed_factors[:, j]))
        factor_max = float(np.max(observed_factors[:, j]))
        for parameter, values in [
            ("intercept_a", a),
            ("persistence_b", b),
            ("long_run_factor_mean", long_run),
        ]:
            q01, q05, q50, q95, q99 = quantiles(values)
            rows.append({
                "model": model,
                "factor": factor,
                "parameter": parameter,
                "prior_q01": q01,
                "prior_q05": q05,
                "prior_median": q50,
                "prior_q95": q95,
                "prior_q99": q99,
                "training_ols_intercept": float(ols_coefficients[0, j]),
                "training_ols_persistence": float(ols_coefficients[1, j]),
                "training_factor_mean": float(np.mean(observed_factors[:, j])),
                "training_factor_min": factor_min,
                "training_factor_max": factor_max,
                "training_factor_range": factor_range,
                "probability_long_run_mean_outside_training_min_max": (
                    float(np.mean((long_run < factor_min) | (long_run > factor_max)))
                    if parameter == "long_run_factor_mean" else None
                ),
                "long_run_mean_prior_second_moment_finite": (
                    False if parameter == "long_run_factor_mean" else None
                ),
                "long_run_mean_prior_absolute_mean_finite": (
                    False if parameter == "long_run_factor_mean" else None
                ),
            })
    return rows


def yield_rows(
    model: str,
    horizon: int,
    draws: dict,
    observed_yields: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    level_rows = []
    change_rows = []
    levels = draws["forecasts"][horizon]["yield_levels"]
    changes = draws["forecasts"][horizon]["yield_changes"]
    training = observed_yields
    empirical_changes = observed_yields[horizon:] - observed_yields[:-horizon]
    for j, maturity in enumerate(MATURITY_LABELS):
        lq = quantiles(levels[:, j])
        cq = quantiles(changes[:, j])
        abs_cuts = np.quantile(np.abs(empirical_changes[:, j]), [0.95, 0.99])
        level_rows.append({
            "model": model,
            "horizon_weeks": horizon,
            "maturity": maturity,
            "maturity_years": float(MATURITIES[j]),
            "level_q01": lq[0], "level_q05": lq[1], "level_median": lq[2],
            "level_q95": lq[3], "level_q99": lq[4],
            "training_level_min": float(np.min(training[:, j])),
            "training_level_max": float(np.max(training[:, j])),
            "probability_below_training_min": float(np.mean(levels[:, j] < np.min(training[:, j]))),
            "probability_above_training_max": float(np.mean(levels[:, j] > np.max(training[:, j]))),
        })
        change_rows.append({
            "model": model,
            "horizon_weeks": horizon,
            "maturity": maturity,
            "maturity_years": float(MATURITIES[j]),
            "change_q01": cq[0], "change_q05": cq[1], "change_median": cq[2],
            "change_q95": cq[3], "change_q99": cq[4],
            "training_abs_change_q95": float(abs_cuts[0]),
            "training_abs_change_q99": float(abs_cuts[1]),
            "probability_abs_change_exceeds_training_q95": float(np.mean(np.abs(changes[:, j]) > abs_cuts[0])),
            "probability_abs_change_exceeds_training_q99": float(np.mean(np.abs(changes[:, j]) > abs_cuts[1])),
        })
    return level_rows, change_rows


def curve_rows(
    model: str,
    horizon: int,
    draws: dict,
    observed_yields: np.ndarray,
) -> tuple[dict, np.ndarray]:
    values = draws["forecasts"][horizon]["yield_changes"]
    rms = np.sqrt(np.mean(values**2, axis=1))
    training_changes = (
        observed_yields[horizon:] - observed_yields[:-horizon]
    )
    training_rms = np.sqrt(np.mean(training_changes**2, axis=1))
    q = quantiles(rms)
    training_q95, training_q99 = np.quantile(training_rms, [0.95, 0.99])
    return ({
        "model": model,
        "horizon_weeks": horizon,
        "curve_change_rms_q01": q[0],
        "curve_change_rms_q05": q[1],
        "curve_change_rms_median": q[2],
        "curve_change_rms_q95": q[3],
        "curve_change_rms_q99": q[4],
        "training_curve_rms_q95": float(training_q95),
        "training_curve_rms_q99": float(training_q99),
        "probability_curve_rms_exceeds_training_q95": float(np.mean(rms > training_q95)),
        "probability_curve_rms_exceeds_training_q99": float(np.mean(rms > training_q99)),
    }, rms)


def make_plot(curve_samples: dict, training_rms: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, horizon in zip(axes, (1, 4)):
        for label, values, color in [
            ("Training sample", training_rms[horizon], "black"),
            ("M3 prior predictive", curve_samples[("M3", horizon)], "#2878B5"),
            ("M4 prior predictive", curve_samples[("M4", horizon)], "#D95319"),
        ]:
            ax.hist(
                np.log10(values),
                bins=90,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=label,
                color=color,
            )
        for q, style in [(0.95, "--"), (0.99, ":")]:
            threshold = np.quantile(training_rms[horizon], q)
            ax.axvline(np.log10(threshold), color="black", linestyle=style, linewidth=1)
        ax.set_title(f"{horizon}-week joint yield-change scale")
        ax.set_xlabel(r"$\log_{10}$ RMS change across maturities (percentage points)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(
    output: Path,
    n_draws: int = N_DRAWS,
    seed: int = SEED,
    prior_set: str = "current",
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    design = make_design()
    initial_yields = design["y"][:design["initial_n"]].copy()
    initial_factors = design["factors"][:design["initial_n"]].copy()
    if len(initial_yields) != 835 or design["initial_last_date"] != "2015-12-31":
        raise ValueError("The prior-predictive conditioning window changed")
    loading = design["load"]
    residuals = initial_yields - initial_factors @ loading.T
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
    failure_counts = {
        "overflow_by_quantity": {},
        "underflow_by_quantity": {},
        "nonfinite_by_quantity": {},
    }
    seed_sequence = np.random.SeedSequence(seed)
    model_seeds = seed_sequence.spawn(2)
    factor_summary, parameter_summary = [], []
    level_summary, change_summary, curve_summary, innovation_summary = [], [], [], []
    curve_samples = {}
    training_rms = {}

    for model, child_seed in zip(("M3", "M4"), model_seeds):
        rng = np.random.default_rng(child_seed)
        draws = simulate_model(
            model, rng, n_draws, prior_set, ols_coefficients,
            initial_factors[-1], initial_yields[-1],
            mu0, loading, reconstruction_variance, failure_counts
        )
        parameter_summary.extend(
            parameter_rows(model, draws, ols_coefficients, initial_factors)
        )
        for horizon in (1, 4):
            factor_summary.extend(factor_rows(model, horizon, draws, initial_factors))
            levels, changes = yield_rows(model, horizon, draws, initial_yields)
            level_summary.extend(levels)
            change_summary.extend(changes)
            curve, rms = curve_rows(model, horizon, draws, initial_yields)
            curve_summary.append(curve)
            curve_samples[(model, horizon)] = rms
            historical = (
                initial_yields[horizon:] - initial_yields[:-horizon]
            )
            training_rms[horizon] = np.sqrt(np.mean(historical**2, axis=1))

        for j, factor in enumerate(FACTORS):
            if model == "M3":
                log_variance = draws["m"][:, j]
            else:
                # Stationary marginal h has mean m-v/2 and variance v.
                log_variance = draws["stationary_log_variance"][:, j]
            analytic_mean = float(np.exp(mu0[j] + 0.5 * draws["m_sd"]**2))
            sampled_variance = checked_exp(log_variance, f"{model}_{factor}_innovation_variance", failure_counts)
            q = quantiles(sampled_variance)
            innovation_summary.append({
                "model": model,
                "factor": factor,
                "innovation_variance_q01": q[0],
                "innovation_variance_q05": q[1],
                "innovation_variance_median": q[2],
                "innovation_variance_q95": q[3],
                "innovation_variance_q99": q[4],
                "analytic_prior_mean_innovation_variance": analytic_mean,
                "training_ols_residual_variance": float(np.exp(mu0[j])),
            })

    write_rows(output / "prior_predictive_factor_changes.csv", factor_summary)
    write_rows(output / "prior_parameter_predictive.csv", parameter_summary)
    write_rows(output / "prior_predictive_yield_levels.csv", level_summary)
    write_rows(output / "prior_predictive_yield_changes.csv", change_summary)
    write_rows(output / "prior_predictive_curve_scale.csv", curve_summary)
    write_rows(output / "prior_predictive_innovation_variances.csv", innovation_summary)
    make_plot(
        curve_samples,
        training_rms,
        output / "prior_predictive_curve_scale.png",
    )

    source_files = [
        ROOT / "data" / "processed" / "canada_weekly.csv",
        ROOT / "results" / "bayesian_revised_pilot" / "extended" / "metadata.json",
        ROOT / "src" / "bayesian_stage_a.py",
        ROOT / "src" / "bayesian_revised.py",
        ROOT / "src" / "bayesian_finite.py",
        ROOT / "src" / "experiment.py",
        ROOT / "src" / "models.py",
        Path(__file__),
    ]
    metadata = {
        "status": "completed",
        "prior_set": prior_set,
        "seed": seed,
        "n_draws_per_model": n_draws,
        "models": ["M3", "M4"],
        "presentation_model_mapping": {"M3": "internal M4 constant volatility", "M4": "internal M5 stochastic volatility"},
        "training_period": ["2000-01-01", "2015-12-31"],
        "training_observations": int(len(initial_yields)),
        "conditioning_date": design["initial_last_date"],
        "conditioning_factors": initial_factors[-1].tolist(),
        "conditioning_yield_curve_percentage_points": initial_yields[-1].tolist(),
        "fixed_lambda_per_year": float(design["lambda"]),
        "maturities_years": MATURITIES.tolist(),
        "measurement_error_variances": reconstruction_variance.tolist(),
        "priors": ({
            "a": "Normal(0, 25), independent across factors",
            "b": "Normal(0.7, 0.09) truncated to (-1,1), independent across factors",
            "m": "Normal(training OLS log-MSE, 25)",
        } if prior_set == "current" else {
            "a": "Factor-specific Normal(training OLS intercept, 0.05^2)",
            "b": "Factor-specific Normal(training OLS AR coefficient, 0.02^2) truncated to (-1,1)",
            "m": "Normal(training OLS log-MSE, 1)",
        }) | {
            "phi": "2*Beta(20,1.5)-1",
            "v": "rate-one exponential truncated to (0,4); M4 only",
        },
        "candidate_training_ols_coefficients": (
            ols_coefficients.T.tolist() if prior_set == "candidate" else None
        ),
        "sv_initial_state": "h_T sampled from the stationary N(m-v/2, v) distribution",
        "yield_reconstruction": "fixed DNS loadings plus independent Gaussian diagonal error using 2000-2015 mean squared residuals",
        "empirical_tail_thresholds": "Training-only observed level ranges and 95th/99th percentiles of absolute 1- and 4-week changes",
        "input_sha256": design["input_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_files},
        "numerical_failures": failure_counts,
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
    parser.add_argument("--prior-set", choices=("current", "candidate"), default="current")
    args = parser.parse_args()
    if args.draws < 10_000:
        parser.error("--draws must be at least 10000 for tail diagnostics")
    try:
        output = args.output.resolve()
        if args.prior_set == "candidate":
            output = output / "candidate"
        result = run(output, args.draws, args.seed, args.prior_set)
    except Exception as exc:
        out = args.output.resolve()
        if args.prior_set == "candidate":
            out = out / "candidate"
        out.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "all_draws_retained": False,
        }
        (out / "prior_predictive_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise
    print(json.dumps({
        "status": result["status"],
        "draws_per_model": result["n_draws_per_model"],
        "seed": result["seed"],
        "output": str(args.output.resolve()),
        "numerical_failures": result["numerical_failures"],
    }))


if __name__ == "__main__":
    main()
