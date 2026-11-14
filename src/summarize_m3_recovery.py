"""Export parameter trajectories, bound frequencies and basin sensitivity."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.m3_recovery import ROOT, _load_panel
from src import state_space_revised as ss


def _write(path, rows):
    if not rows:
        raise ValueError(f"refusing to write an empty summary: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output_dir):
    out = Path(output_dir).resolve()
    if not out.is_relative_to((ROOT/"results").resolve()):
        raise ValueError("summary outputs must remain under results/")
    config = json.loads((out/"experiment_config.json").read_text(encoding="utf-8"))
    experiment, dates, _, design = _load_panel()
    design_record = {
        "maturities_years": experiment["maturities_years"],
        "horizons_observations": experiment["horizons_weeks"],
        "origin_stride_observations": experiment["origin_stride_observations"],
        "lambda_per_year": float(design["lambda_per_year"]),
        "initial_sse": float(design["initial_sse"]),
        "initial_training_n": design["initial_training_n"],
        "initial_training_start": dates[design["initial_indices"][0]].isoformat(),
        "initial_training_end": dates[design["initial_indices"][-1]].isoformat(),
        "rolling_origin_count": len(design["origin_indices"]),
        "first_rolling_origin": dates[design["origin_indices"][0]].isoformat(),
        "last_rolling_origin": dates[design["origin_indices"][-1]].isoformat(),
        "target_origins": config["origins"],
        "information_cutoff_rule": "training observations dated no later than forecast origin",
    }
    fits = json.loads((out/"fits.json").read_text(encoding="utf-8"))
    with np.load(out/"forecasts.npz", allow_pickle=False) as archive:
        forecast_arrays = {name: archive[name].copy() for name in archive.files}
    variants = {row["name"]: row for row in config["variants"]}
    dimension = len(config["measurement_variance_calibration"]["values"])
    maturities = json.loads(
        (ROOT/"config"/"experiment.json").read_text(encoding="utf-8"))["maturities_years"]
    trajectories = []
    boundaries = []
    basin_rows = []
    fit_summary = []
    maturity_rows = []
    calibration_groups = {}
    parameter_names = [
        "mu_level", "mu_slope", "mu_curvature",
        "u_level", "u_slope", "u_curvature",
        "log_q_level", "log_q_slope", "log_q_curvature",
    ]
    for fit in fits:
        variant = variants[fit["variant"]]
        mode = variant["measurement_mode"]
        bounds = ss.parameter_bounds(dimension, variant["persistence_bound"], mode)
        params = np.asarray(fit["parameters"], dtype=float)
        names = parameter_names + (
            [f"log_r_{maturity:g}y" for maturity in maturities]
            if mode != "fixed" else [])
        if len(names) != len(params):
            names = parameter_names + [f"log_r_index_{i}" for i in range(len(params)-9)]
        for index, value in enumerate(params):
            trajectories.append({
                "origin_date": fit["origin_date"], "variant": fit["variant"],
                "parameter_index": index, "parameter": names[index],
                "estimate": float(value), "lower_bound": float(bounds[index, 0]),
                "upper_bound": float(bounds[index, 1]),
            })
            tolerance = 1e-6
            boundaries.append({
                "origin_date": fit["origin_date"], "variant": fit["variant"],
                "parameter_index": index, "parameter": names[index],
                "lower_hit": bool(value <= bounds[index, 0]+tolerance),
                "upper_hit": bool(value >= bounds[index, 1]-tolerance),
                "distance_to_nearest_bound": float(min(
                    value-bounds[index, 0], bounds[index, 1]-value)),
            })
        fit_summary.append({
            "origin_date": fit["origin_date"], "variant": fit["variant"],
            "raw_nll": fit["raw_nll"], "raw_log_likelihood": fit["raw_log_likelihood"],
            "penalized_objective": fit["penalized_objective"],
            "original_start_nll_spread": fit["original_start_nll_spread"],
            "original_start_objective_spread": fit["original_start_objective_spread"],
            "original_start_acceptance": fit["original_start_acceptance"],
            "revised_accepted": fit["revised_accepted"],
            "selected_restart_id": fit["selected_restart_id"],
            "distinct_basin_count": fit["distinct_basin_count"],
            "selected_basin_replicates": fit["selected_basin_replicates"],
            "stationary_initial_covariance_condition": fit["stationary_initial_covariance_condition"],
            "stationary_initial_covariance_max_diagonal": fit["stationary_initial_covariance_max_diagonal"],
            "terminal_filtered_covariance_min_eigenvalue": fit["terminal_filtered_covariance_min_eigenvalue"],
        })
        for basin in fit["basin_forecast_sensitivity"]:
            basin_rows.append({
                "origin_date": fit["origin_date"], "variant": fit["variant"],
                **basin,
            })

    for index, key in enumerate(forecast_arrays["keys"].tolist()):
        origin, horizon, variant_name = key
        mean = forecast_arrays["means"][index]
        covariance = forecast_arrays["covariances"][index]
        target = forecast_arrays["targets"][index]
        standard_deviation = np.sqrt(np.diag(covariance))
        error = target-mean
        pit = norm.cdf(error/standard_deviation)
        coverage = abs(error) <= norm.ppf(.95)*standard_deviation
        for j, maturity in enumerate(forecast_arrays["maturities"]):
            row = {
                "origin_date": origin, "horizon_observations": int(horizon),
                "variant": variant_name, "maturity_years": float(maturity),
                "squared_error": float(error[j]**2),
                "absolute_error": float(abs(error[j])),
                "pit": float(pit[j]), "coverage90": int(coverage[j]),
                "interval_width": float(2*norm.ppf(.95)*standard_deviation[j]),
            }
            maturity_rows.append(row)
            group_key = (variant_name, int(horizon), float(maturity))
            group = calibration_groups.setdefault(group_key, [])
            group.append(row)

    calibration_summary = []
    for (variant_name, horizon, maturity), rows in calibration_groups.items():
        calibration_summary.append({
            "variant": variant_name, "horizon_observations": horizon,
            "maturity_years": maturity, "n_origins": len(rows),
            "rmse": float(np.sqrt(np.mean([row["squared_error"] for row in rows]))),
            "mae": float(np.mean([row["absolute_error"] for row in rows])),
            "coverage90": float(np.mean([row["coverage90"] for row in rows])),
            "mean_pit": float(np.mean([row["pit"] for row in rows])),
            "mean_interval_width": float(np.mean([row["interval_width"] for row in rows])),
        })

    hit_counts = {}
    for row in boundaries:
        key = (row["variant"], row["parameter_index"], row["parameter"])
        aggregate = hit_counts.setdefault(key, {
            "variant": row["variant"], "parameter_index": row["parameter_index"],
            "parameter": row["parameter"], "origins": 0, "lower_hits": 0, "upper_hits": 0,
        })
        aggregate["origins"] += 1
        aggregate["lower_hits"] += int(row["lower_hit"])
        aggregate["upper_hits"] += int(row["upper_hit"])
    frequency_rows = []
    for row in hit_counts.values():
        row["lower_hit_frequency"] = row["lower_hits"]/row["origins"]
        row["upper_hit_frequency"] = row["upper_hits"]/row["origins"]
        frequency_rows.append(row)

    _write(out/"parameter_trajectory.csv", trajectories)
    _write(out/"bound_hits_by_origin.csv", boundaries)
    _write(out/"bound_frequency.csv", frequency_rows)
    _write(out/"fit_comparison.csv", fit_summary)
    _write(out/"maturity_by_forecast.csv", maturity_rows)
    _write(out/"calibration_by_maturity.csv", calibration_summary)
    if basin_rows:
        _write(out/"basin_forecast_sensitivity.csv", basin_rows)
    (out/"design.json").write_text(
        json.dumps(design_record, indent=2, allow_nan=False), encoding="utf-8")
    return {
        "fit_rows": len(fit_summary), "parameter_rows": len(trajectories),
        "bound_summary_rows": len(frequency_rows), "basin_rows": len(basin_rows),
        "output_directory": str(out),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.output), indent=2))


if __name__ == "__main__":
    main()
