"""Reproducible, isolated recovery study for the rejected Gaussian DNS M3."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import numba

from src import state_space_revised as ss


ROOT = Path(__file__).resolve().parents[1]
TARGET_ORIGINS = (
    "2023-02-24", "2023-03-24", "2023-04-21", "2023-05-19",
    "2023-06-16", "2023-07-14", "2023-08-10", "2023-09-08",
)
RESTART_SEED = 271026
PENALTY_GRID = (0.1, 1.0, 10.0)
CV_TRAIN_FRACTIONS = (0.55, 0.65, 0.75)
CV_VALIDATION_LENGTH = 52
VARIANTS = (
    {"name": "A_original_multistart", "measurement_mode": "estimated", "persistence_bound": 3.0},
    {"name": "B_wide_persistence", "measurement_mode": "estimated", "persistence_bound": 4.0},
    {"name": "C_fixed_measurement", "measurement_mode": "fixed", "persistence_bound": 3.0},
    {"name": "D_regularized_measurement", "measurement_mode": "regularized", "persistence_bound": 3.0},
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_panel():
    from src.experiment import read_panel
    config = json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8"))
    dates, yields = read_panel(ROOT / "data" / "processed" / "canada_weekly.csv")
    design = ss.prepare_design(dates, yields, config)
    return config, dates, yields, design


def select_penalty_strength(initial_y, dates, config, penalty_grid=PENALTY_GRID,
                            maxiter=1000, ftol=1e-14, gtol=2e-7, seed=RESTART_SEED):
    """Choose lambda by chronological validation wholly inside initial training.

    Every fold re-estimates its DNS loading and measurement target from that
    fold's training prefix. Validation selection uses raw conditional NLL, not
    the penalized objective, and never reads evaluation-period observations.
    """
    from src.models import fit_lambda, ns_loadings

    initial_y = np.asarray(initial_y, dtype=float)
    if len(initial_y) <= CV_VALIDATION_LENGTH:
        raise ValueError("initial training sample is too short for blocked penalty selection")
    maturities = np.asarray(config["maturities_years"], dtype=float)
    grid = np.linspace(*config["lambda_grid_per_year"])
    folds = []
    for fold_id, fraction in enumerate(CV_TRAIN_FRACTIONS):
        train_end = int(len(initial_y)*fraction)
        validation_end = train_end+CV_VALIDATION_LENGTH
        if validation_end > len(initial_y):
            raise ValueError("a regularization CV validation block exceeds the initial sample")
        train = initial_y[:train_end]
        validation = initial_y[train_end:validation_end]
        lam, _, initial_sse = fit_lambda(train, maturities, grid)
        load = ns_loadings(maturities, lam)
        measurement_target = ss.calibrate_measurement_covariance(train, load)
        fold = {
            "fold_id": fold_id,
            "train_end_index_exclusive": train_end,
            "train_end_date": dates[train_end-1].isoformat(),
            "validation_start_date": dates[train_end].isoformat(),
            "validation_end_date": dates[validation_end-1].isoformat(),
            "validation_n": len(validation),
            "lambda_per_year": float(lam),
            "initial_sse": float(initial_sse),
            "measurement_variance_target": measurement_target.tolist(),
            "candidates": [],
        }
        joined = np.concatenate((train, validation), axis=0)
        for candidate_id, strength in enumerate(penalty_grid):
            fit = ss.fit_mle(
                train, load, n_starts=2, maxiter=maxiter, ftol=ftol, gtol=gtol,
                persistence_bound=3., measurement_mode="regularized",
                penalty_strength=float(strength),
                measurement_target=np.log(measurement_target),
                restart_count=0, restart_seed=seed+fold_id*100+candidate_id,
            )
            train_nll = ss.filter_state(
                train, load, fit["params"], "regularized")[0]
            joined_nll = ss.filter_state(
                joined, load, fit["params"], "regularized")[0]
            validation_nll = float(joined_nll-train_nll)
            fold["candidates"].append({
                "penalty_strength": float(strength),
                "train_raw_nll": float(train_nll),
                "train_penalized_objective": float(fit["objective"]),
                "validation_raw_nll": validation_nll,
                "validation_raw_nll_per_observation": validation_nll/len(validation),
                "original_start_nll_spread": fit["original_start_nll_spread"],
                "original_start_objective_spread": fit["original_start_objective_spread"],
                "best_projected_gradient_per_observation": fit["projected_gradient_per_observation"],
                "selected_parameters": fit["params"].tolist(),
            })
        folds.append(fold)

    aggregate = []
    for strength in penalty_grid:
        scored = [
            item for fold in folds for item in fold["candidates"]
            if item["penalty_strength"] == strength
        ]
        total_validation_n = sum(fold["validation_n"] for fold in folds)
        total_nll = sum(item["validation_raw_nll"] for item in scored)
        aggregate.append({
            "penalty_strength": float(strength),
            "prior_sd_on_log_R": float(1/np.sqrt(strength)),
            "validation_raw_nll": float(total_nll),
            "validation_raw_nll_per_observation": float(total_nll/total_validation_n),
        })
    selected = min(aggregate, key=lambda row: (
        row["validation_raw_nll_per_observation"], row["penalty_strength"]))
    return {
        "selection_rule": "minimum pooled blocked-validation raw conditional NLL per observation",
        "training_scope": "2000-2015 initial training sample only",
        "candidate_penalty_grid": list(map(float, penalty_grid)),
        "fold_train_fractions": list(CV_TRAIN_FRACTIONS),
        "validation_length_observations": CV_VALIDATION_LENGTH,
        "aggregate_scores": aggregate,
        "selected_penalty_strength": selected["penalty_strength"],
        "selected_prior_sd_on_log_R": selected["prior_sd_on_log_R"],
        "folds": folds,
    }


def _previous_origin_starts(origin, origin_indices, dates, archived_fits, fixed_mode):
    position = next(i for i, t in enumerate(origin_indices)
                    if dates[t].isoformat() == origin)
    if position == 0:
        return None
    previous_date = dates[origin_indices[position-1]].isoformat()
    params = np.asarray(archived_fits[previous_date]["params"], dtype=float)
    return params[:9] if fixed_mode else params.copy()


def _stationary_covariance(params):
    rho = np.tanh(params[3:6])
    q = np.exp(params[6:9])
    variance = q*np.cosh(params[3:6])**2
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise FloatingPointError("invalid stationary initial-state covariance")
    return np.diag(variance), rho


def _profile_selected_fit(y, load, fit, measurement_mode, fixed_r,
                          penalty_strength, target):
    params = np.asarray(fit["params"])
    bounds = ss.parameter_bounds(y.shape[1], fit["persistence_bound"], measurement_mode)
    raw_nll = ss.filter_state(y, load, params, measurement_mode, fixed_r)[0]
    objective = fit["objective"]
    profiles = []
    for j in np.flatnonzero((params <= bounds[:, 0]+1e-6) | (params >= bounds[:, 1]-1e-6)):
        step = .05 if j < 6 else .5
        direction = 1 if params[j] <= bounds[j, 0]+1e-6 else -1
        row = {"parameter_index": int(j), "inward_step": direction*step}
        for name, delta in (("inward", direction*step), ("outward", -direction*step)):
            candidate = params.copy()
            candidate[j] += delta
            candidate_nll = ss.filter_state(
                y, load, candidate, measurement_mode, fixed_r)[0]
            candidate_objective = candidate_nll
            if penalty_strength:
                candidate_objective += .5*penalty_strength*np.sum(
                    (candidate[9:]-target)**2)
            row[name+"_delta_raw_nll"] = float(candidate_nll-raw_nll)
            row[name+"_delta_objective"] = float(candidate_objective-objective)
        profiles.append(row)
    return profiles


def _original_scores(origins, horizons):
    paths = [
        ("historical_M3", ROOT/"results"/"state_space_revised"/"scores.csv"),
        ("external_benchmark", ROOT/"results"/"validated_baselines"/"scores.csv"),
    ]
    rows = []
    for source, path in paths:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["origin_date"] in origins and int(row["horizon"]) in horizons:
                    row["variant"] = row["model"] if source == "external_benchmark" else "M3_original_archive"
                    row["source"] = source
                    rows.append(row)
    return rows


def _summary_rows(score_rows):
    output = []
    variants = sorted({row["variant"] for row in score_rows})
    for variant in variants:
        for horizon in sorted({int(row["horizon"]) for row in score_rows}):
            subset = [row for row in score_rows
                      if row["variant"] == variant and int(row["horizon"]) == horizon]
            if not subset:
                continue
            output.append({
                "variant": variant, "horizon_observations": horizon, "n_origins": len(subset),
                **{name: float(np.mean([float(row[name]) for row in subset]))
                   for name in ("dss", "gaussian_negative_log_score", "energy_score",
                                "variogram_score", "mae", "coverage90", "width90")},
                "pooled_rmse": float(np.sqrt(np.mean([
                    float(row["squared_error_average"]) for row in subset]))),
            })
    return output


def run_recovery(output_dir=None, scope="targeted", restart_count=8, *,
                 n_starts=3, maxiter=1000, ftol=1e-14, gtol=2e-7,
                 seed=RESTART_SEED):
    """Run isolated variants on the eight failures or the full rolling design."""
    if scope not in {"targeted", "full"}:
        raise ValueError("scope must be targeted or full")
    if restart_count < 0:
        raise ValueError("restart_count must be nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(output_dir) if output_dir else ROOT/"results"/f"m3_recovery_{stamp}"
    out = out.resolve()
    if not out.is_relative_to((ROOT/"results").resolve()):
        raise ValueError("recovery outputs must remain under results/")
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):
            raise FileExistsError(f"recovery output directory is not empty: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)

    config, dates, yields, design = _load_panel()
    origin_indices = design["origin_indices"]
    all_origins = [dates[t].isoformat() for t in origin_indices]
    selected_origins = list(TARGET_ORIGINS) if scope == "targeted" else all_origins
    missing = sorted(set(selected_origins)-set(all_origins))
    if missing:
        raise ValueError(f"recovery origins are absent from the frozen design: {missing}")
    origin_index = {dates[t].isoformat(): t for t in origin_indices}
    initial_y = yields[design["initial_indices"]]
    fixed_r = np.asarray(design["initial_measurement_variance"], dtype=float)
    measurement_target = np.log(fixed_r)
    maturity_grid = np.asarray(config["maturities_years"], dtype=float)
    lambda_selection = select_penalty_strength(
        initial_y, [dates[i] for i in design["initial_indices"]], config,
        maxiter=maxiter, ftol=ftol, gtol=gtol, seed=seed)

    archive_path = ROOT/"results"/"state_space_revised"/"optimizer_diagnostics.json"
    archived_fits = {
        item["origin_date"]: item for item in json.loads(archive_path.read_text(encoding="utf-8"))
    }
    original_rows = _original_scores(set(selected_origins), set(config["horizons_weeks"]))
    archive_status = {
        item["origin_date"]: item for item in json.loads(
            (ROOT/"results"/"state_space_revised"/"optimizer_diagnostics.json").read_text(encoding="utf-8"))
    }
    for row in original_rows:
        if row["variant"] == "M3_original_archive":
            row["fit_accepted"] = str(archive_status[row["origin_date"]]["accepted"])

    baseline_dir = ROOT/"results"/"validated_baselines"
    baseline_hashes_before = {
        path.name: _sha256(path) for path in baseline_dir.iterdir() if path.is_file()
    }
    bayesian_analysis = (
        ROOT/"results"/"bayesian_finite_20261027T194239Z"/"protocol.md",
        ROOT/"results"/"bayesian_finite_20261027T194239Z"/"rolling_main"/"analysis"/"paired_summary.csv",
        ROOT/"results"/"bayesian_finite_20261027T194239Z"/"rolling_main"/"analysis"/"per_maturity.csv",
        ROOT/"results"/"bayesian_finite_20261027T194239Z"/"rolling_main"/"analysis"/"summary.json",
    )
    bayesian_hashes_before = {
        str(path.relative_to(ROOT)): _sha256(path) for path in bayesian_analysis if path.is_file()
    }
    used_files = [
        ROOT/"src"/"state_space_revised.py", ROOT/"src"/"m3_recovery.py",
        ROOT/"tests"/"test_m3_recovery.py",
        ROOT/"config"/"experiment.json", ROOT/"data"/"processed"/"canada_weekly.csv",
        archive_path, ROOT/"results"/"state_space_revised"/"scores.csv",
        ROOT/"results"/"state_space_revised"/"metadata.json",
        ROOT/"results"/"state_space_revised"/"forecasts.npz",
        ROOT/"results"/"validated_baselines"/"scores.csv",
    ]
    _write_json(out/"experiment_config.json", {
        "scope": scope, "origins": selected_origins,
        "horizons_observations": config["horizons_weeks"],
        "note": "The checked-in design uses weekly observation steps [1,4]; no 1/5/20 business-day design is substituted.",
        "variants": list(VARIANTS), "n_starts": 3, "restart_count": restart_count,
        "restart_seed_master": seed,
        "optimizer": {"method": "L-BFGS-B", "maxiter": maxiter, "ftol": ftol, "gtol": gtol},
        "penalty_selection": lambda_selection,
        "optimizer_settings": {"n_starts": n_starts, "restart_count": restart_count,
                               "maxiter": maxiter, "ftol": ftol, "gtol": gtol,
                               "restart_seed_master": seed},
        "measurement_variance_calibration": {
            "period_start": dates[design["initial_indices"][0]].isoformat(),
            "period_end": dates[design["initial_indices"][-1]].isoformat(),
            "n_observations": len(design["initial_indices"]),
            "method": "maturity-wise mean squared cross-sectional DNS residual, floored at configured variance floor",
            "variance_floor": config.get("variance_floor_percent_squared", 1e-6),
            "values": fixed_r.tolist(),
        },
        "git": {"branch": _git("branch", "--show-current"), "commit": _git("rev-parse", "HEAD"),
                "worktree_status": _git("status", "--short")},
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "numba": numba.__version__},
        "inputs_and_sources_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in used_files
        },
        "baseline_hashes_before": baseline_hashes_before,
        "accepted_bayesian_analysis_hashes_before": bayesian_hashes_before,
    })
    _write_json(out/"regularization_cv.json", lambda_selection)
    _write_json(out/"initial_measurement_covariance.json", {
        "period_start": dates[design["initial_indices"][0]].isoformat(),
        "period_end": dates[design["initial_indices"][-1]].isoformat(),
        "n_observations": len(design["initial_indices"]),
        "method": "maturity-wise mean squared cross-sectional DNS residual",
        "variance": fixed_r.tolist(), "log_variance": measurement_target.tolist(),
    })

    fits = []
    score_rows = list(original_rows)
    forecasts = []
    log_path = out/"run.log"
    for origin_position, origin in enumerate(selected_origins):
        t = origin_index[origin]
        training = np.ascontiguousarray(
            yields[design["first_training_index"]:t+1])
        target_by_horizon = {
            int(h): (t+int(h), yields[t+int(h)])
            for h in config["horizons_weeks"]
        }
        warm_original = _previous_origin_starts(
            origin, origin_indices, dates, archived_fits, fixed_mode=False)
        warm_fixed = _previous_origin_starts(
            origin, origin_indices, dates, archived_fits, fixed_mode=True)
        for variant_position, variant in enumerate(VARIANTS):
            mode = variant["measurement_mode"]
            warm = warm_fixed if mode == "fixed" else warm_original
            strength = (lambda_selection["selected_penalty_strength"]
                        if mode == "regularized" else 0.)
            fit = ss.fit_mle(
                training, design["load"], n_starts=n_starts, maxiter=maxiter,
                warm_start=warm, persistence_bound=variant["persistence_bound"],
                measurement_mode=mode,
                fixed_measurement=fixed_r if mode == "fixed" else None,
                penalty_strength=strength,
                measurement_target=measurement_target if mode == "regularized" else None,
                restart_count=restart_count,
                restart_seed=seed+variant_position*100000+origin_position,
                ftol=ftol, gtol=gtol,
            )
            params = np.asarray(fit["params"], dtype=float)
            raw_nll, state, filtered_covariance = ss.filter_state(
                training, design["load"], params, mode,
                fixed_r if mode == "fixed" else None)
            stationary_covariance, rho = _stationary_covariance(params)
            selected_forecasts = {
                horizon: ss.forecast_moments(
                    state, filtered_covariance, design["load"], params, horizon,
                    mode, fixed_r if mode == "fixed" else None)
                for horizon in target_by_horizon
            }
            basin_representatives = {}
            for attempt in fit["diagnostics"]:
                if not attempt["numerically_valid"]:
                    continue
                basin_id = attempt["basin_id"]
                if (basin_id not in basin_representatives
                        or attempt["objective"] < basin_representatives[basin_id]["objective"]):
                    basin_representatives[basin_id] = attempt
            basin_sensitivity = []
            for basin_id, attempt in sorted(basin_representatives.items()):
                candidate_params = np.asarray(attempt["params"])
                _, candidate_state, candidate_covariance = ss.filter_state(
                    training, design["load"], candidate_params, mode,
                    fixed_r if mode == "fixed" else None)
                basin_row = {
                    "basin_id": basin_id,
                    "is_selected_basin": basin_id == fit["selected_basin_id"],
                    "raw_nll_gap_to_selected": attempt["raw_nll_gap_to_best"],
                    "objective_gap_to_selected": attempt["objective_gap_to_best"],
                    "parameters": candidate_params.tolist(),
                    "max_abs_parameter_difference": float(
                        np.max(np.abs(candidate_params-params))),
                }
                for horizon, (selected_mean, selected_cov) in selected_forecasts.items():
                    candidate_mean, candidate_y_cov = ss.forecast_moments(
                        candidate_state, candidate_covariance, design["load"],
                        candidate_params, horizon, mode,
                        fixed_r if mode == "fixed" else None)
                    basin_row[f"h{horizon}_mean_max_abs_difference_bp"] = float(
                        np.max(np.abs(candidate_mean-selected_mean))*100.)
                    basin_row[f"h{horizon}_covariance_max_abs_difference"] = float(
                        np.max(np.abs(candidate_y_cov-selected_cov)))
                basin_sensitivity.append(basin_row)
            fit_record = {
                "origin_date": origin, "variant": variant["name"],
                "n_training": len(training), "latest_fit_date": origin,
                "parameters": params.tolist(),
                "rho": rho.tolist(),
                "process_variance": np.exp(params[6:9]).tolist(),
                "measurement_variance": (
                    fixed_r.tolist() if mode == "fixed" else np.exp(params[9:]).tolist()),
                "raw_nll": float(raw_nll), "raw_log_likelihood": float(-raw_nll),
                "penalized_objective": float(fit["objective"]),
                "penalty_strength": float(strength),
                "original_start_nll_spread": fit["original_start_nll_spread"],
                "original_start_objective_spread": fit["original_start_objective_spread"],
                "original_start_acceptance": fit["original_acceptance"],
                "revised_accepted": fit["revised_accepted"],
                "selected_start": fit["selected_start"],
                "selected_restart_id": fit["selected_restart_id"],
                "selected_basin_id": fit["selected_basin_id"],
                "original_distinct_basin_count": fit["original_distinct_basin_count"],
                "distinct_basin_count": fit["distinct_basin_count"],
                "selected_basin_replicates": fit["selected_basin_replicates"],
                "attempts": fit["diagnostics"],
                "basin_forecast_sensitivity": basin_sensitivity,
                "active_bounds": min(
                    (d for d in fit["diagnostics"] if d["numerically_valid"]),
                    key=lambda d: d["objective"])["active_bound_indices"],
                "projected_gradient_per_observation": fit["projected_gradient_per_observation"],
                "stationary_initial_covariance": stationary_covariance.tolist(),
                "stationary_initial_covariance_condition": float(np.linalg.cond(stationary_covariance)),
                "stationary_initial_covariance_max_diagonal": float(np.max(np.diag(stationary_covariance))),
                "terminal_filtered_covariance_min_eigenvalue": float(
                    np.linalg.eigvalsh(filtered_covariance).min()),
                "bound_profiles": _profile_selected_fit(
                    training, design["load"], fit, mode,
                    fixed_r if mode == "fixed" else None,
                    strength, measurement_target if mode == "regularized" else None),
            }
            fits.append(fit_record)
            _write_json(out/"fits.json", fits)

            for horizon, (target_index, observed) in target_by_horizon.items():
                mean, covariance = selected_forecasts[horizon]
                seed = 20261027+t*17+horizon
                forecast = ss.evaluate_forecast(
                    mean, covariance, observed, yields[t], seed)
                row = {
                    "origin_date": origin, "target_date": dates[target_index].isoformat(),
                    "horizon": horizon, "model": variant["name"],
                    "variant": variant["name"], "source": "new_recovery",
                    **forecast["metrics"],
                    "fit_revised_accepted": fit["revised_accepted"],
                    "original_start_nll_spread": fit["original_start_nll_spread"],
                    "raw_nll": float(raw_nll),
                    "penalized_objective": float(fit["objective"]),
                }
                score_rows.append(row)
                forecasts.append({
                    "origin_date": origin, "target_date": dates[target_index].isoformat(),
                    "horizon": horizon, "variant": variant["name"], "seed": seed,
                    "mean": np.asarray(mean).tolist(),
                    "covariance": np.asarray(covariance).tolist(),
                    "target": np.asarray(observed).tolist(),
                    "scenarios": np.asarray(forecast["scenarios"]).tolist(),
                })
            line = (f"{origin} {variant['name']} nll={raw_nll:.6f} "
                    f"objective={fit['objective']:.6f} original_spread="
                    f"{fit['original_start_nll_spread']:.4g} "
                    f"revised={fit['revised_accepted']} "
                    f"active={fit_record['active_bounds']}")
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(line+"\n")

    _write_json(out/"fits.json", fits)
    _write_csv(out/"scores.csv", score_rows)
    _write_csv(out/"summary.csv", _summary_rows(score_rows))
    forecast_arrays = {
        "means": np.asarray([item["mean"] for item in forecasts]),
        "covariances": np.asarray([item["covariance"] for item in forecasts]),
        "targets": np.asarray([item["target"] for item in forecasts]),
        "scenarios": np.asarray([item["scenarios"] for item in forecasts]),
        "keys": np.asarray([[item["origin_date"], str(item["horizon"]), item["variant"]]
                            for item in forecasts]),
        "target_dates": np.asarray([item["target_date"] for item in forecasts]),
        "seeds": np.asarray([item["seed"] for item in forecasts], dtype=np.int64),
        "maturities": maturity_grid,
    }
    np.savez_compressed(out/"forecasts.npz", **forecast_arrays)
    baseline_hashes_after = {
        path.name: _sha256(path) for path in baseline_dir.iterdir() if path.is_file()
    }
    if baseline_hashes_before != baseline_hashes_after:
        raise RuntimeError("accepted M0-M2 baseline files changed during M3 recovery")
    bayesian_hashes_after = {
        str(path.relative_to(ROOT)): _sha256(path) for path in bayesian_analysis if path.is_file()
    }
    if bayesian_hashes_before != bayesian_hashes_after:
        raise RuntimeError("accepted Bayesian analysis files changed during M3 recovery")
    _write_json(out/"run_summary.json", {
        "status": "targeted_complete" if scope == "targeted" else "full_complete",
        "scope": scope, "n_origins": len(selected_origins),
        "forecast_records_new_variants": len(forecasts),
        "variants": [item["name"] for item in VARIANTS],
        "fit_records": len(fits), "elapsed_seconds": None,
        "accepted_baselines_unchanged": True,
        "baseline_hashes_before": baseline_hashes_before,
        "baseline_hashes_after": baseline_hashes_after,
        "accepted_bayesian_analysis_unchanged": True,
        "accepted_bayesian_analysis_hashes_before": bayesian_hashes_before,
        "accepted_bayesian_analysis_hashes_after": bayesian_hashes_after,
        "output_sha256": {
            path.name: _sha256(path) for path in out.iterdir()
            if path.is_file() and path.name not in {"run_summary.json"}
        },
        "limitations": [
            "Historical results retain the existing weekly 1- and 4-observation horizons.",
            "Only targeted or explicitly selected rolling origins are evaluated.",
            "Gaussian forecasts condition on fitted parameters; parameter uncertainty is not propagated.",
            "Energy Score and risk scenarios use finite Monte Carlo samples.",
        ],
    })
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("targeted", "full"), default="targeted")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--ftol", type=float, default=1e-14)
    parser.add_argument("--gtol", type=float, default=2e-7)
    parser.add_argument("--seed", type=int, default=RESTART_SEED)
    args = parser.parse_args()
    started = time.perf_counter()
    output = run_recovery(
        args.output, args.scope, args.restarts, n_starts=args.starts,
        maxiter=args.maxiter, ftol=args.ftol, gtol=args.gtol, seed=args.seed)
    summary_path = output/"run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["elapsed_seconds"] = time.perf_counter()-started
    _write_json(summary_path, summary)
    print(f"Recovery outputs: {output}")


if __name__ == "__main__":
    main()
