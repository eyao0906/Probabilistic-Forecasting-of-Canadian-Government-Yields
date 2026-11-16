"""Audit the finite upper bound on stationary log-volatility variance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

from src.bayesian_stage_a import FACTORS, ROOT, make_design
from src.models import ns_loadings


DEFAULT_OUTPUT = ROOT / "results" / "sv_bound_justification_20261115"
POSTERIOR_RUN = ROOT / "results" / "bayesian_prior_revised_20261114"
MATURITIES = np.array([0.25, 0.5, 1, 2, 5, 7, 10, 30], dtype=float)
MATURITY_LABELS = ("3m", "6m", "1y", "2y", "5y", "7y", "10y", "30y")
N_DRAWS = 100_000
SEED = 20261115
V_RATE = 1.0
POSTERIOR_QUANTILES = np.array([0.50, 0.95, 0.99])
POSTERIOR_TAIL_CUTS = (2.0, 3.0, 3.5)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def truncated_exponential_quantile(probability: float, upper: float) -> float:
    if not 0.0 < probability < 1.0 or not np.isfinite(upper) or upper <= 0:
        raise ValueError("Need a probability in (0,1) and a positive finite bound")
    return float(-np.log1p(-probability * (-np.expm1(-V_RATE * upper))) / V_RATE)


def truncated_exponential_tail(cut: float, upper: float) -> float:
    if cut >= upper:
        return 0.0
    if cut <= 0:
        return 1.0
    remaining_mass = np.exp(-V_RATE * cut) * (
        -np.expm1(-V_RATE * (upper - cut))
    )
    total_mass = -np.expm1(-V_RATE * upper)
    return float(remaining_mass / total_mass)


def exp_with_counts(log_values: np.ndarray, label: str, counts: dict) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    log_max = np.log(np.finfo(float).max)
    log_min = np.log(np.nextafter(0.0, 1.0))
    overflow = int(np.count_nonzero(values > log_max))
    underflow = int(np.count_nonzero(values < log_min))
    counts[label] = {"overflow": overflow, "underflow": underflow}
    if overflow:
        raise FloatingPointError(f"{overflow} {label} values exceed float64 range")
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        result = np.exp(values)
    nonfinite = int(np.count_nonzero(~np.isfinite(result)))
    counts[label]["nonfinite"] = nonfinite
    if nonfinite:
        raise FloatingPointError(f"{nonfinite} nonfinite values in {label}")
    return result


def training_auxiliary_volatility(factors: np.ndarray) -> list[dict]:
    rows = []
    noise_variance = np.pi**2 / 2
    for j, factor in enumerate(FACTORS):
        lag = factors[:-1, j]
        response = factors[1:, j]
        design = np.column_stack((np.ones(len(lag)), lag))
        intercept, persistence = np.linalg.lstsq(design, response, rcond=None)[0]
        residual = response - (intercept + persistence * lag)
        if not np.isfinite(residual).all() or np.any(residual == 0):
            raise ValueError(f"Invalid OLS residuals for {factor}")
        log_squared = np.log(residual**2)
        observed_variance = float(np.var(log_squared, ddof=1))
        rows.append({
            "factor": factor,
            "training_observations": int(len(factors)),
            "ols_intercept": float(intercept),
            "ols_persistence": float(persistence),
            "log_squared_residual_variance": observed_variance,
            "log_chi_square_noise_variance": float(noise_variance),
            "rough_auxiliary_v_estimate": max(0.0, observed_variance - noise_variance),
        })
    return rows


def simulate_prior_predictive(
    design: dict,
    upper: float,
    n_draws: int,
    seed: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    initial_yields = design["y"][:design["initial_n"]]
    initial_factors = design["factors"][:design["initial_n"]]
    if len(initial_yields) != 835 or design["initial_last_date"] != "2015-12-31":
        raise ValueError("The fixed 2000-2015 training window changed")
    if not np.array_equal(MATURITIES, np.array([0.25, 0.5, 1, 2, 5, 7, 10, 30])):
        raise AssertionError("Unexpected maturity design")

    factor_center = np.mean(initial_factors, axis=0)
    factor_sd = np.std(initial_factors, axis=0, ddof=1)
    persistence_center = np.array([
        np.linalg.lstsq(
            np.column_stack((np.ones(834), initial_factors[:-1, j])),
            initial_factors[1:, j],
            rcond=None,
        )[0][1]
        for j in range(3)
    ])
    mean_prior = rng.normal(factor_center, 2 * factor_sd, size=(n_draws, 3))
    persistence = np.column_stack([
        truncnorm.rvs(
            (-1 - persistence_center[j]) / 0.03,
            (1 - persistence_center[j]) / 0.03,
            loc=persistence_center[j],
            scale=0.03,
            size=n_draws,
            random_state=rng,
        )
        for j in range(3)
    ])
    intercept = (1 - persistence) * mean_prior
    log_expected_variance = (
        np.array([prior.omega_mean for prior in design["priors"]])[None, :]
        + rng.normal(size=(n_draws, 3))
    )
    phi = 2 * rng.beta(20, 1.5, size=(n_draws, 3)) - 1
    uniforms = rng.random((n_draws, 3))
    stationary_v = -np.log1p(
        -uniforms * (-np.expm1(-V_RATE * upper))
    ) / V_RATE
    omega = log_expected_variance - stationary_v / 2
    state_sd = np.sqrt(stationary_v)
    q = stationary_v * (1 - phi**2)

    failures: dict[str, dict] = {}
    h = omega + state_sd * rng.normal(size=(n_draws, 3))
    stationary_sigma = exp_with_counts(h / 2, f"U{upper:g}_stationary_innovation_sd", failures)
    factor_state = np.broadcast_to(initial_factors[-1], (n_draws, 3)).copy()
    residuals = initial_yields - initial_factors @ design["load"].T
    reconstruction_variance = np.maximum(np.mean(residuals**2, axis=0), 1e-6)
    measurement_sd = np.sqrt(reconstruction_variance)
    yield_changes: dict[int, np.ndarray] = {}

    with np.errstate(over="raise", invalid="raise"):
        for week in range(1, 5):
            h = omega + phi * (h - omega) + np.sqrt(q) * rng.normal(size=(n_draws, 3))
            sigma = exp_with_counts(h / 2, f"U{upper:g}_week{week}_innovation_sd", failures)
            factor_state = (
                intercept + persistence * factor_state
                + sigma * rng.normal(size=(n_draws, 3))
            )
            if not np.isfinite(factor_state).all():
                raise FloatingPointError(f"Nonfinite factor states at U={upper}, week={week}")
            if week in (1, 4):
                future_yields = (
                    factor_state @ design["load"].T
                    + rng.normal(size=(n_draws, 8)) * measurement_sd
                )
                changes = future_yields - initial_yields[-1]
                if not np.isfinite(changes).all():
                    raise FloatingPointError(f"Nonfinite yield changes at U={upper}, week={week}")
                yield_changes[week] = changes

    draws = {
        "stationary_v": stationary_v,
        "stationary_sigma": stationary_sigma,
        "yield_changes": yield_changes,
    }
    return draws, {
        "upper_bound": upper,
        "seed": seed,
        "draws": n_draws,
        "finite_stationary_v": bool(np.isfinite(stationary_v).all()),
        "finite_stationary_sigma": bool(np.isfinite(stationary_sigma).all()),
        "finite_yield_changes": {
            str(horizon): bool(np.isfinite(values).all())
            for horizon, values in yield_changes.items()
        },
        "numeric_events": failures,
        "max_stationary_v": float(np.max(stationary_v)),
        "max_stationary_innovation_sd": float(np.max(stationary_sigma)),
        "max_abs_yield_change": {
            str(horizon): float(np.max(np.abs(values)))
            for horizon, values in yield_changes.items()
        },
    }


def prior_predictive_rows(
    draws_by_upper: dict[float, dict],
    design: dict,
    training: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    curve_rows = []
    maturity_rows = []
    for upper, draws in draws_by_upper.items():
        for horizon, changes in draws["yield_changes"].items():
            curve_magnitude = np.sqrt(np.mean(changes**2, axis=1))
            historical = training[horizon:] - training[:-horizon]
            historical_curve = np.sqrt(np.mean(historical**2, axis=1))
            threshold95, threshold99 = np.quantile(historical_curve, [0.95, 0.99])
            q50, q90, q95, q99 = np.quantile(curve_magnitude, [0.50, 0.90, 0.95, 0.99])
            curve_rows.append({
                "upper_bound": upper,
                "horizon_weeks": horizon,
                "units": "percentage_points",
                "predictive_rms_median": float(q50),
                "predictive_rms_q90": float(q90),
                "predictive_rms_q95": float(q95),
                "predictive_rms_q99": float(q99),
                "training_rms_q95": float(threshold95),
                "training_rms_q99": float(threshold99),
                "probability_exceeds_training_q95": float(np.mean(curve_magnitude > threshold95)),
                "probability_exceeds_training_q99": float(np.mean(curve_magnitude > threshold99)),
                "probability_exceeds_training_max": float(np.mean(curve_magnitude > np.max(historical_curve))),
                "count_exceeds_training_max": int(np.count_nonzero(curve_magnitude > np.max(historical_curve))),
                "maximum_predictive_rms": float(np.max(curve_magnitude)),
                "nonfinite_draws": int(np.count_nonzero(~np.isfinite(curve_magnitude))),
            })
            for j, maturity in enumerate(MATURITIES):
                magnitude = np.abs(changes[:, j])
                historical_magnitude = np.abs(historical[:, j])
                q50, q90, q95, q99 = np.quantile(magnitude, [0.50, 0.90, 0.95, 0.99])
                t95, t99 = np.quantile(historical_magnitude, [0.95, 0.99])
                maturity_rows.append({
                    "upper_bound": upper,
                    "horizon_weeks": horizon,
                    "maturity": MATURITY_LABELS[j],
                    "maturity_years": float(maturity),
                    "units": "percentage_points",
                    "absolute_change_median": float(q50),
                    "absolute_change_q90": float(q90),
                    "absolute_change_q95": float(q95),
                    "absolute_change_q99": float(q99),
                    "training_abs_change_q95": float(t95),
                    "training_abs_change_q99": float(t99),
                    "probability_exceeds_training_q95": float(np.mean(magnitude > t95)),
                    "probability_exceeds_training_q99": float(np.mean(magnitude > t99)),
                    "probability_exceeds_training_max": float(np.mean(magnitude > np.max(historical_magnitude))),
                    "maximum_predictive_absolute_change": float(np.max(magnitude)),
                    "nonfinite_draws": int(np.count_nonzero(~np.isfinite(magnitude))),
                })
    return curve_rows, maturity_rows


def posterior_diagnostics() -> tuple[list[dict], list[dict], dict]:
    run_summary_path = POSTERIOR_RUN / "summary.json"
    plan_path = POSTERIOR_RUN / "plan.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (not run_summary.get("accepted") or run_summary.get("completed_origins") != 130
            or run_summary.get("failed_origins") or digest(plan_path) != run_summary.get("plan_sha256")):
        raise ValueError("The revised-prior archive is not the accepted 130-origin run")
    dates = run_summary.get("completed_dates", [])
    if len(dates) != 130 or len(set(dates)) != 130:
        raise ValueError("Unexpected revised-prior origin list")

    detail_rows = []
    chain_rows = []
    for origin in dates:
        origin_dir = POSTERIOR_RUN / "origins" / origin
        forecast_path = origin_dir / "forecast.json"
        forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        spec = forecast["specification"]
        if not forecast.get("accepted") or spec.get("origin") != origin or "M5" not in spec.get("models", []):
            raise ValueError(f"Unaccepted or mismatched M5 archive at {origin}")
        if spec.get("input_sha256") != plan.get("input_sha256") or spec.get("chains") != 4 or spec.get("draws") != 6000:
            raise ValueError(f"Unexpected data or chain configuration at {origin}")
        prior_config = spec["prior_configuration"]
        if any(p.get("v_upper") != 4.0 or p.get("v_rate") != 1.0 for p in prior_config):
            raise ValueError(f"Prior configuration mismatch at {origin}")
        archived_hashes = {
            Path(name).name: checksum
            for name, checksum in forecast["chain_checkpoint_sha256"].items()
        }

        for j, factor in enumerate(FACTORS):
            chains = []
            for chain in range(4):
                name = f"M5_{factor}_chain{chain}.npz"
                path = origin_dir / "chains" / name
                actual_hash = digest(path)
                if archived_hashes.get(name) != actual_hash:
                    raise ValueError(f"Accepted chain hash mismatch: {origin}/{name}")
                with np.load(path, allow_pickle=False) as archive:
                    values = archive["draws"]
                    report = json.loads(str(archive["report"]))
                names = report.get("names", [])
                if (values.ndim != 2 or values.shape[0] != 6000 or len(names) != values.shape[1]
                        or "v" not in names or not np.isfinite(values).all()):
                    raise ValueError(f"Invalid chain data: {origin}/{name}")
                v = values[:, names.index("v")]
                if np.any((v <= 0) | (v >= 4)):
                    raise ValueError(f"v draw outside accepted support: {origin}/{name}")
                chains.append(v)
                chain_rows.append({
                    "origin": origin,
                    "factor": factor,
                    "chain": chain,
                    "draws": int(len(v)),
                    "v_min": float(np.min(v)),
                    "v_max": float(np.max(v)),
                    "sha256": actual_hash,
                })
            pooled = np.concatenate(chains)
            q50, q95, q99 = np.quantile(pooled, POSTERIOR_QUANTILES)
            row = {
                "origin": origin,
                "factor": factor,
                "posterior_draws": int(len(pooled)),
                "v_q50": float(q50),
                "v_q95": float(q95),
                "v_q99": float(q99),
                "max_sampled_v": float(np.max(pooled)),
                "distance_U_minus_q99": float(4.0 - q99),
                "distance_U_minus_max_sampled_v": float(4.0 - np.max(pooled)),
            }
            for cut in POSTERIOR_TAIL_CUTS:
                row[f"posterior_probability_v_gt_{cut:g}"] = float(np.mean(pooled > cut))
            detail_rows.append(row)

    aggregate_rows = []
    prior_tails = {cut: truncated_exponential_tail(cut, 4.0) for cut in POSTERIOR_TAIL_CUTS}
    for factor in FACTORS:
        rows = [r for r in detail_rows if r["factor"] == factor]
        aggregate = {
            "factor": factor,
            "origins": len(rows),
            "median_v_q50_across_origins": float(np.median([r["v_q50"] for r in rows])),
            "median_v_q95_across_origins": float(np.median([r["v_q95"] for r in rows])),
            "maximum_v_q95_across_origins": float(np.max([r["v_q95"] for r in rows])),
            "median_v_q99_across_origins": float(np.median([r["v_q99"] for r in rows])),
            "maximum_v_q99_across_origins": float(np.max([r["v_q99"] for r in rows])),
            "minimum_U_minus_q99": float(np.min([r["distance_U_minus_q99"] for r in rows])),
            "maximum_sampled_v": float(np.max([r["max_sampled_v"] for r in rows])),
            "minimum_U_minus_sampled_v": float(np.min([r["distance_U_minus_max_sampled_v"] for r in rows])),
        }
        for cut in POSTERIOR_TAIL_CUTS:
            key = f"posterior_probability_v_gt_{cut:g}"
            probs = [r[key] for r in rows]
            aggregate[f"prior_probability_v_gt_{cut:g}"] = prior_tails[cut]
            aggregate[f"median_posterior_probability_v_gt_{cut:g}"] = float(np.median(probs))
            aggregate[f"maximum_posterior_probability_v_gt_{cut:g}"] = float(np.max(probs))
            aggregate[f"origins_posterior_probability_v_gt_{cut:g}_over_005"] = int(np.count_nonzero(np.array(probs) > 0.05))
            aggregate[f"origins_posterior_probability_v_gt_{cut:g}_over_010"] = int(np.count_nonzero(np.array(probs) > 0.10))
        aggregate_rows.append(aggregate)
    provenance = {
        "accepted_run_summary_sha256": digest(run_summary_path),
        "accepted_plan_sha256": digest(plan_path),
        "origin_count": len(dates),
        "chain_count": len(chain_rows),
        "posterior_draw_count": int(sum(r["draws"] for r in chain_rows)),
    }
    return detail_rows, chain_rows, {"summary": aggregate_rows, "provenance": provenance}


def plot_prior_predictive(
    draws_by_upper: dict[float, dict],
    training: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    color_map = plt.get_cmap("viridis")
    bounds = list(draws_by_upper)
    colors = {
        upper: color_map(index / max(1, len(bounds) - 1))
        for index, upper in enumerate(bounds)
    }
    for ax, horizon in zip(axes, (1, 4)):
        empirical = training[horizon:] - training[:-horizon]
        empirical_rms = np.sqrt(np.mean(empirical**2, axis=1))
        samples = {
            upper: np.sqrt(np.mean(draws["yield_changes"][horizon]**2, axis=1))
            for upper, draws in draws_by_upper.items()
        }
        all_log = np.concatenate([np.log10(x) for x in [empirical_rms, *samples.values()]])
        bins = np.linspace(float(np.min(all_log)), float(np.max(all_log)), 90)
        ax.hist(np.log10(empirical_rms), bins=bins, density=True, histtype="step",
                color="black", linewidth=1.5, label="Training sample")
        for upper, values in samples.items():
            ax.hist(np.log10(values), bins=bins, density=True, histtype="step",
                    color=colors[upper], linewidth=1.5, label=f"U={upper:g}")
        for quantile, style in ((0.95, "--"), (0.99, ":")):
            threshold = np.quantile(empirical_rms, quantile)
            ax.axvline(np.log10(threshold), color="black", linestyle=style, linewidth=1)
        ax.set_title(f"{horizon}-week joint yield-curve movements")
        ax.set_xlabel(r"$\log_{10}$ RMS change (percentage points)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_maturity_quantiles(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, constrained_layout=True)
    for ax, maturity in zip(axes.flat, ("3m", "2y", "10y", "30y")):
        subset = [r for r in rows if r["maturity"] == maturity]
        for horizon, style in ((1, "-o"), (4, "--s")):
            curve = [r for r in subset if r["horizon_weeks"] == horizon]
            curve.sort(key=lambda r: r["upper_bound"])
            x = [r["upper_bound"] for r in curve]
            ax.plot(x, [r["absolute_change_q95"] for r in curve], style, label=f"{horizon}-week q95")
            ax.plot(x, [r["absolute_change_q99"] for r in curve], style, alpha=0.45, label=f"{horizon}-week q99")
        ax.set_title(maturity)
        ax.set_ylabel("Absolute change (percentage points)")
        ax.set_xticks(sorted({r["upper_bound"] for r in subset}))
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_posterior(detail_rows: list[dict], output: Path) -> None:
    from datetime import date
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for ax, factor in zip(axes, FACTORS):
        rows = [r for r in detail_rows if r["factor"] == factor]
        dates = [date.fromisoformat(r["origin"]) for r in rows]
        median = np.array([r["v_q50"] for r in rows])
        q95 = np.array([r["v_q95"] for r in rows])
        q99 = np.array([r["v_q99"] for r in rows])
        ax.fill_between(dates, median, q95, color="#4C78A8", alpha=0.20, label="Median to 95th percentile")
        ax.plot(dates, median, color="#4C78A8", linewidth=1.2, label="Posterior median")
        ax.plot(dates, q99, color="#F58518", linewidth=1, label="99th percentile")
        ax.axhline(truncated_exponential_quantile(0.95, 4.0), color="gray", linestyle=":", label="Prior 95th percentile")
        ax.axhline(4.0, color="black", linestyle="--", linewidth=1, label="Upper bound U=4")
        ax.set_ylabel(f"{factor.capitalize()} v")
        ax.set_ylim(bottom=0)
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def create_outputs(output: Path, n_draws: int, seed: int, uppers: list[float]) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty audit directory: {output}")
    if (n_draws < 10_000 or not uppers or len(set(uppers)) != len(uppers)
            or not np.isfinite(uppers).all() or any(u <= 0 for u in uppers)):
        raise ValueError("Use at least 10,000 draws and unique positive upper bounds")
    if 4.0 not in uppers:
        raise ValueError("The primary U=4 specification must be included")

    design = make_design()
    initial_yields = design["y"][:design["initial_n"]]
    initial_factors = design["factors"][:design["initial_n"]]
    auxiliary = training_auxiliary_volatility(initial_factors)
    draws_by_upper: dict[float, dict] = {}
    numeric_summaries = []
    prior_v_rows = []
    sigma_rows = []
    raw_arrays = {}
    for upper in uppers:
        draws, numeric = simulate_prior_predictive(design, upper, n_draws, seed)
        draws_by_upper[upper] = draws
        numeric_summaries.append(numeric)
        for probability in (0.01, 0.05, 0.50, 0.90, 0.95, 0.99):
            vq = truncated_exponential_quantile(probability, upper)
            prior_v_rows.append({
                "upper_bound": upper,
                "probability": probability,
                "stationary_log_volatility_variance_quantile": vq,
                "stationary_log_volatility_sd_quantile": float(np.sqrt(vq)),
                "stationary_log_sigma_sd_quantile": float(np.sqrt(vq) / 2),
            })
        for j, factor in enumerate(FACTORS):
            q = np.quantile(draws["stationary_sigma"][:, j], [0.01, 0.05, 0.50, 0.90, 0.95, 0.99])
            sigma_rows.append({
                "upper_bound": upper,
                "factor": factor,
                "innovation_sd_q01": float(q[0]),
                "innovation_sd_q05": float(q[1]),
                "innovation_sd_median": float(q[2]),
                "innovation_sd_q90": float(q[3]),
                "innovation_sd_q95": float(q[4]),
                "innovation_sd_q99": float(q[5]),
                "units": "percentage_points_per_sqrt_week",
                "nonfinite_draws": int(np.count_nonzero(~np.isfinite(draws["stationary_sigma"][:, j]))),
            })
        for horizon, values in draws["yield_changes"].items():
            raw_arrays[f"U{upper:g}_h{horizon}"] = values

    curve_rows, maturity_rows = prior_predictive_rows(draws_by_upper, design, initial_yields)
    posterior_rows, chain_rows, posterior = posterior_diagnostics()
    write_csv(output / "prior_v_quantiles.csv", prior_v_rows)
    write_csv(output / "stationary_innovation_sd_quantiles.csv", sigma_rows)
    write_csv(output / "prior_predictive_curve_movements.csv", curve_rows)
    write_csv(output / "prior_predictive_maturity_movements.csv", maturity_rows)
    write_csv(output / "training_auxiliary_volatility.csv", auxiliary)
    write_csv(output / "posterior_boundary_by_origin_factor.csv", posterior_rows)
    write_csv(output / "posterior_chain_manifest.csv", chain_rows)
    write_csv(output / "posterior_boundary_summary.csv", posterior["summary"])
    write_csv(output / "posterior_prior_tail_comparison.csv", [
        {
            "factor": row["factor"],
            "cut": cut,
            "prior_probability_v_above_cut": row[f"prior_probability_v_gt_{cut:g}"],
            "median_posterior_probability_across_origins": row[f"median_posterior_probability_v_gt_{cut:g}"],
            "maximum_posterior_probability_across_origins": row[f"maximum_posterior_probability_v_gt_{cut:g}"],
            "origins_posterior_probability_over_5_percent": row[f"origins_posterior_probability_v_gt_{cut:g}_over_005"],
            "origins_posterior_probability_over_10_percent": row[f"origins_posterior_probability_v_gt_{cut:g}_over_010"],
        }
        for row in posterior["summary"] for cut in POSTERIOR_TAIL_CUTS
    ])
    with (output / "prior_predictive_draws.npz").open("xb") as stream:
        np.savez_compressed(stream, **raw_arrays)
    plot_prior_predictive(draws_by_upper, initial_yields, output / "prior_predictive_curve_movements.png")
    plot_maturity_quantiles(maturity_rows, output / "prior_predictive_maturity_movements.png")
    plot_posterior(posterior_rows, output / "posterior_stationary_log_variance.png")

    source_paths = [
        ROOT / "data" / "processed" / "canada_weekly.csv",
        ROOT / "results" / "bayesian_revised_pilot" / "extended" / "metadata.json",
        POSTERIOR_RUN / "summary.json",
        POSTERIOR_RUN / "plan.json",
        ROOT / "src" / "bayesian_stage_a.py",
        ROOT / "src" / "experiment.py",
        ROOT / "src" / "bayesian_revised.py",
        ROOT / "src" / "bayesian_finite.py",
        ROOT / "src" / "prior_predictive_bayesian_dns.py",
        ROOT / "src" / "bayesian_prior_longrun.py",
        ROOT / "src" / "models.py",
        Path(__file__),
    ]
    metadata = {
        "status": "completed",
        "seed_per_bound_common_random_numbers": seed,
        "draws_per_bound": n_draws,
        "upper_bounds": uppers,
        "exponential_rate": V_RATE,
        "training_period": ["2000-01-01", "2015-12-31"],
        "training_observations": int(len(initial_yields)),
        "conditioning_date": design["initial_last_date"],
        "conditioning_factor_state": initial_factors[-1].tolist(),
        "fixed_lambda_per_year": float(design["lambda"]),
        "maturities_years": MATURITIES.tolist(),
        "factor_prior": {
            "long_run_mean": "Normal(training mean, (2 * training sample SD)^2)",
            "persistence": "Normal(initial-training OLS AR(1) coefficient, 0.03^2), truncated to (-1,1)",
            "m": "Normal(initial-training OLS log residual MSE, 1)",
            "phi": "2*Beta(20,1.5)-1",
            "v": "rate-one exponential truncated to (0,U)",
            "h_initial": "stationary Normal(m-v/2,v); simulated state h is stored separately from v",
        },
        "reconstruction_error": "training-window maturity-specific mean squared residual, diagonal, floored at 1e-6 percentage-points squared",
        "movement_reference": "2000-2015 observed 1- and 4-week yield changes only",
        "posterior_archive": {
            "path": str(POSTERIOR_RUN.relative_to(ROOT)),
            **posterior["provenance"],
            "parameter_name": "v",
            "not_h_or_q": True,
            "all_chain_sha256_verified": True,
            "upper_bound": 4.0,
        },
        "numerical_checks": numeric_summaries,
        "all_prior_predictive_draws_saved": True,
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        "forecast_scores_used": False,
        "mcmc_refit": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--upper-bounds", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    args = parser.parse_args()
    result = create_outputs(args.output.resolve(), args.draws, args.seed, args.upper_bounds)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.resolve()),
        "draws_per_bound": result["draws_per_bound"],
        "posterior_origins": result["posterior_archive"]["origin_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
