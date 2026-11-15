"""Independently verify isolated M3 recovery fits, forecasts and score outputs."""
from __future__ import annotations

import argparse
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import norm

from src.m3_recovery import ROOT, _load_panel
from src import state_space_revised as ss


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _independent_batch_filter(y, load, params, measurement_mode, fixed_r=None):
    """Reference Kalman likelihood using a joint maturity-vector update."""
    mu = params[:3]
    rho = np.tanh(params[3:6])
    q = np.exp(params[6:9])
    r = np.asarray(fixed_r if measurement_mode == "fixed" else np.exp(params[9:]))
    state = mu.copy()
    covariance = np.diag(q/(1-rho**2))
    total = 0.
    for t, observation in enumerate(y):
        if t:
            transition = np.diag(rho)
            state = mu+transition@(state-mu)
            covariance = transition@covariance@transition.T+np.diag(q)
        innovation = observation-load@state
        innovation_covariance = load@covariance@load.T+np.diag(r)
        sign, logdet = np.linalg.slogdet(innovation_covariance)
        if sign != 1:
            raise AssertionError("independent innovation covariance is not positive definite")
        total += .5*(len(observation)*np.log(2*np.pi)+logdet
                     +innovation@np.linalg.solve(innovation_covariance, innovation))
        gain = np.linalg.solve(innovation_covariance, load@covariance).T
        state = state+gain@innovation
        residual_map = np.eye(3)-gain@load
        covariance = (residual_map@covariance@residual_map.T
                      +gain@np.diag(r)@gain.T)
        covariance = .5*(covariance+covariance.T)
    return float(total), state, covariance


def verify(output_dir, write_outputs=True):
    out = Path(output_dir).resolve()
    if not out.is_relative_to((ROOT/"results").resolve()):
        raise ValueError("verification output must be under results/")
    config = json.loads((out/"experiment_config.json").read_text(encoding="utf-8"))
    run_summary = json.loads((out/"run_summary.json").read_text(encoding="utf-8"))
    fits = json.loads((out/"fits.json").read_text(encoding="utf-8"))
    scores = _read_csv(out/"scores.csv")
    _, dates, yields, design = _load_panel()
    date_index = {day.isoformat(): i for i, day in enumerate(dates)}
    fit_map = {(row["origin_date"], row["variant"]): row for row in fits}
    expected_keys = {
        (origin, str(horizon), variant["name"])
        for origin in config["origins"]
        for horizon in config["horizons_observations"]
        for variant in config["variants"]
    }

    with np.load(out/"forecasts.npz", allow_pickle=False) as archive:
        saved = {name: archive[name].copy() for name in archive.files}
    keys = [tuple(row) for row in saved["keys"].tolist()]
    if set(keys) != expected_keys or len(keys) != len(expected_keys):
        raise AssertionError("forecast keys do not match the configured common design")
    if saved["scenarios"].shape != (len(keys), 1024, len(config["measurement_variance_calibration"]["values"])):
        raise AssertionError("unexpected forecast-scenario dimensions")
    if not all(np.isfinite(saved[name]).all()
               for name in ("means", "covariances", "targets", "scenarios")):
        raise AssertionError("non-finite forecast artifact")

    score_map = {
        (row["origin_date"], str(row["horizon"]), row["variant"]): row
        for row in scores if row.get("source") == "new_recovery"
    }
    if set(score_map) != expected_keys:
        raise AssertionError("new-variant score rows do not match forecast keys")

    config_by_name = {row["name"]: row for row in config["variants"]}
    fit_errors = []
    max_nll_error = 0.
    max_objective_error = 0.
    max_mean_error = 0.
    max_covariance_error = 0.
    max_scenario_error = 0.
    score_errors = []
    minimum_forecast_eigenvalue = np.inf
    for index, key in enumerate(keys):
        origin, horizon_text, variant_name = key
        horizon = int(horizon_text)
        t = date_index[origin]
        training = yields[design["first_training_index"]:t+1]
        if dates[t].isoformat() != origin:
            raise AssertionError("invalid origin date")
        fit = fit_map[(origin, variant_name)]
        variant = config_by_name[variant_name]
        mode = variant["measurement_mode"]
        fixed_r = np.asarray(config["measurement_variance_calibration"]["values"])
        expected_parameter_count = 9 if mode == "fixed" else 9+len(fixed_r)
        if len(fit["parameters"]) != expected_parameter_count:
            raise AssertionError(f"wrong parameter dimension for {variant_name}")
        if fit["latest_fit_date"] != origin or fit["n_training"] != len(training):
            raise AssertionError("forecast fit uses observations beyond its origin or wrong sample size")
        params = np.asarray(fit["parameters"])
        bounds = ss.parameter_bounds(
            len(fixed_r), variant["persistence_bound"], mode)
        if np.any(params < bounds[:, 0]-1e-10) or np.any(params > bounds[:, 1]+1e-10):
            raise AssertionError("selected estimate violates configured parameter bounds")
        selected_nll, state, filtered_covariance = _independent_batch_filter(
            training, design["load"], params, mode,
            fixed_r if mode == "fixed" else None)
        nll_error = abs(selected_nll-float(fit["raw_nll"]))
        max_nll_error = max(max_nll_error, nll_error)
        if nll_error > 2e-6:
            raise AssertionError(f"selected likelihood mismatch at {origin}/{variant_name}")

        penalty_strength = float(fit["penalty_strength"])
        target = np.log(fixed_r)
        penalty = (.5*penalty_strength*np.sum((params[9:]-target)**2)
                   if penalty_strength else 0.)
        objective_error = abs(selected_nll+penalty-float(fit["penalized_objective"]))
        max_objective_error = max(max_objective_error, objective_error)
        if objective_error > 2e-6:
            raise AssertionError(f"objective mismatch at {origin}/{variant_name}")

        attempt_objectives = []
        for attempt in fit["attempts"]:
            candidate = np.asarray(attempt["params"])
            raw, _, _ = _independent_batch_filter(
                training, design["load"], candidate, mode,
                fixed_r if mode == "fixed" else None)
            attempt_nll_error = abs(raw-float(attempt["nll"]))
            max_nll_error = max(max_nll_error, attempt_nll_error)
            if attempt_nll_error > 2e-6:
                raise AssertionError(f"attempt likelihood mismatch at {origin}/{variant_name}")
            candidate_penalty = (
                .5*penalty_strength*np.sum((candidate[9:]-target)**2)
                if penalty_strength else 0.)
            objective = raw+candidate_penalty
            attempt_objectives.append(objective)
            attempt_objective_error = abs(objective-float(attempt["objective"]))
            max_objective_error = max(max_objective_error, attempt_objective_error)
            if attempt_objective_error > 2e-6:
                raise AssertionError(f"attempt objective mismatch at {origin}/{variant_name}")
        valid_attempts = [
            i for i, attempt in enumerate(fit["attempts"])
            if attempt["numerically_valid"]
        ]
        best_attempt = min(valid_attempts, key=lambda i: attempt_objectives[i])
        if not fit["attempts"][best_attempt]["numerically_valid"]:
            raise AssertionError("selected objective minimum was marked invalid")
        if abs(attempt_objectives[best_attempt]-float(fit["penalized_objective"])) > 2e-6:
            raise AssertionError("selected fit is not the best valid recorded objective")

        target_index = t+horizon
        if saved["target_dates"][index] != dates[target_index].isoformat():
            raise AssertionError("forecast target date does not match frozen weekly-step horizon")
        np.testing.assert_array_equal(saved["targets"][index], yields[target_index])
        mean, covariance = ss.forecast_moments(
            state, filtered_covariance, design["load"], params, horizon,
            mode, fixed_r if mode == "fixed" else None)
        mean_error = float(np.max(np.abs(mean-saved["means"][index])))
        covariance_error = float(np.max(np.abs(covariance-saved["covariances"][index])))
        max_mean_error = max(max_mean_error, mean_error)
        max_covariance_error = max(max_covariance_error, covariance_error)
        if mean_error > 2e-8 or covariance_error > 2e-8:
            raise AssertionError(f"forecast moment mismatch at {origin}/{variant_name}/{horizon}")
        minimum_forecast_eigenvalue = min(
            minimum_forecast_eigenvalue, float(np.linalg.eigvalsh(covariance).min()))

        seed = int(saved["seeds"][index])
        rng_scenarios = rng_from_seed(
            seed, saved["means"][index], saved["covariances"][index])
        scenario_error = float(np.max(np.abs(rng_scenarios-saved["scenarios"][index])))
        max_scenario_error = max(max_scenario_error, scenario_error)
        if scenario_error > 0:
            raise AssertionError("saved Monte Carlo scenarios do not reproduce from recorded seed")
        error = yields[target_index]-mean
        sign, logdet = np.linalg.slogdet(covariance)
        if sign != 1:
            raise AssertionError("forecast covariance is not positive definite")
        dss = float(logdet+error@np.linalg.solve(covariance, error))
        sd = np.sqrt(np.diag(covariance))
        hits = abs(error) <= norm.ppf(.95)*sd
        es = float(np.linalg.norm(rng_scenarios-yields[target_index], axis=1).mean()
                   -.5*pdist(rng_scenarios).mean())
        tri = np.triu_indices(len(sd), 1)
        observed_spread = abs(yields[target_index, :, None]-yields[target_index, None, :])**.5
        simulated_spread = np.mean(
            abs(rng_scenarios[:, :, None]-rng_scenarios[:, None, :])**.5, axis=0)
        variogram = float(np.sum((observed_spread[tri]-simulated_spread[tri])**2))
        expected_metrics = {
            "dss": dss,
            "gaussian_negative_log_score": .5*(len(sd)*np.log(2*np.pi)+dss),
            "energy_score": es,
            "variogram_score": variogram,
            "squared_error_average": float(np.mean(error**2)),
            "mae": float(np.mean(abs(error))),
            "coverage90": float(hits.mean()),
            "width90": float(np.mean(2*norm.ppf(.95)*sd)),
        }
        row = score_map[(origin, horizon_text, variant_name)]
        for name, value in expected_metrics.items():
            difference = abs(float(row[name])-value)
            score_errors.append(difference)
            if difference > 2e-9:
                raise AssertionError(f"score reconstruction mismatch: {name} {key}")

    if len(fits) != len(config["origins"])*len(config["variants"]):
        raise AssertionError("fit record count does not match configured experiment")
    for name, expected_hash in run_summary["baseline_hashes_before"].items():
        path = ROOT/"results"/"validated_baselines"/name
        if _sha256(path) != expected_hash:
            raise AssertionError(f"accepted baseline changed: {name}")
    for relative_path, expected_hash in run_summary["accepted_bayesian_analysis_hashes_before"].items():
        if _sha256(ROOT/relative_path) != expected_hash:
            raise AssertionError(f"accepted Bayesian analysis changed: {relative_path}")

    report = {
        "status": "verified",
        "scope": config["scope"],
        "origins_verified": len(config["origins"]),
        "fit_records_verified": len(fits),
        "forecast_records_verified": len(keys),
        "attempt_likelihoods_independently_recomputed": sum(len(fit["attempts"]) for fit in fits),
        "maximum_selected_nll_abs_difference": max_nll_error,
        "maximum_penalized_objective_abs_difference": max_objective_error,
        "maximum_forecast_mean_abs_difference": max_mean_error,
        "maximum_forecast_covariance_abs_difference": max_covariance_error,
        "maximum_scenario_abs_difference": max_scenario_error,
        "maximum_score_abs_difference": max(score_errors, default=0.),
        "minimum_predictive_covariance_eigenvalue": minimum_forecast_eigenvalue,
        "accepted_M0_M2_hashes_unchanged": True,
        "accepted_Bayesian_analysis_hashes_unchanged": True,
        "limitations": [
            "Verification confirms saved calculations and information cutoffs, not model identification or scientific validity.",
            "The optimization variants remain conditional on fixed DNS loadings and fitted parameters.",
        ],
    }
    if write_outputs:
        (out/"verification.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def rng_from_seed(seed, mean, covariance):
    return np.random.default_rng(seed).multivariate_normal(
        mean, covariance, size=1024, check_valid="raise")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    report = verify(args.output, write_outputs=not args.no_write)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
