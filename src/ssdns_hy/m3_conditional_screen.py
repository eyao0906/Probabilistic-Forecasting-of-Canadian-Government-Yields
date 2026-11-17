"""Progressively release SS_M3 blocks on one synthetic dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path

import arviz as az
import numpy as np

from .data import MATURITIES_YEARS, sha256_file
from .kalman import kalman_filter
from .loadings import dns_loadings
from .model import factor_dynamics
from .priors import calibrate_training_priors
from .sampler import (
    _draw_factor_parameters,
    _draw_initial_factor,
    _draw_measurement_variance,
    _factor_log_variance_target,
    _slice_sample,
    _update_decay_marginal,
)
from .smoother import sample_state_path
from .synthetic import simulate_yields
from .synthetic_pilot import SETTINGS


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "hautsch_yang_state_space_20261117" / "m3_conditional_screen_20261117"


def main() -> None:
    config = {
        "seed": SETTINGS["seed"], "n": 120, "warmup": 150, "draws": 300,
        "chains": 4, "decay_step": .005,
        "stages": ["factor_variance", "release_dynamics", "release_measurement", "release_decay"],
    }
    source = {
        path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    metadata = {"config": config, "simulation": SETTINGS, "source_sha256": source}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "design.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("existing M3 conditional screen has different source or design")
    else:
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    truth_mean = np.asarray(SETTINGS["factor_mean"])
    truth_phi = np.asarray(SETTINGS["factor_persistence"])
    truth_m = np.log(SETTINGS["factor_innovation_variance"])
    truth_r = np.asarray(SETTINGS["measurement_variance"])
    truth_decay = SETTINGS["decay_per_year"]
    simulation = simulate_yields(
        config["n"], MATURITIES_YEARS, truth_decay, truth_mean, truth_phi, truth_r,
        np.random.default_rng(config["seed"]), "SS_M3", truth_m,
        initial_factor_variance=SETTINGS["initial_factor_variance"],
    )
    y = simulation["yields"]
    priors, _ = calibrate_training_priors(y, MATURITIES_YEARS)
    for stage_index, stage in enumerate(config["stages"]):
        started = time.perf_counter()
        stored = {
            "log_variance": np.empty((config["chains"], config["draws"], 3)),
        }
        if stage_index >= 1:
            stored["factor_mean"] = np.empty((config["chains"], config["draws"], 3))
            stored["factor_persistence"] = np.empty((config["chains"], config["draws"], 3))
        if stage_index >= 2:
            stored["measurement_variance"] = np.empty((config["chains"], config["draws"], 8))
        if stage_index >= 3:
            stored["decay"] = np.empty((config["chains"], config["draws"]))
        path_mean = np.zeros_like(simulation["factors"])
        acceptance = []
        streams = np.random.SeedSequence(config["seed"] + 200 + stage_index).spawn(config["chains"])
        for chain, stream in enumerate(streams):
            rng = np.random.default_rng(stream)
            mean = truth_mean.copy()
            persistence = truth_phi.copy()
            log_variance = truth_m + (chain - 1.5) * .5
            measurement = truth_r.copy()
            decay = truth_decay
            accepted = 0
            for iteration in range(config["warmup"] + config["draws"]):
                transition, intercept = factor_dynamics(mean, persistence)
                q = np.diag(np.exp(log_variance))
                if stage_index >= 3:
                    decay, did_accept, filtered = _update_decay_marginal(
                        decay, y, MATURITIES_YEARS, q, measurement, mean, persistence,
                        priors, rng, step=config["decay_step"],
                    )
                    if iteration >= config["warmup"]:
                        accepted += int(did_accept)
                else:
                    filtered = kalman_filter(
                        y, dns_loadings(MATURITIES_YEARS, decay), transition, intercept,
                        q, np.diag(measurement), mean, np.eye(3) * priors.initial_state_variance,
                    )
                states = sample_state_path(filtered, transition, rng)
                initial = _draw_initial_factor(
                    states[0], mean, persistence, np.diag(q), priors.initial_state_variance, rng
                )
                q_array = np.broadcast_to(np.diag(q), (config["n"], 3))
                if stage_index >= 1:
                    mean, persistence = _draw_factor_parameters(
                        states, initial, q_array, mean, persistence, priors, rng
                    )
                previous = np.vstack((initial, states[:-1]))
                innovations = states - mean - (previous - mean) * persistence
                for j in range(3):
                    log_variance[j] = _slice_sample(
                        log_variance[j],
                        lambda value, j=j: _factor_log_variance_target(
                            value, innovations[:, j], priors.log_variance_center[j],
                            priors.log_variance_sd[j],
                        ),
                        rng,
                    )
                if stage_index >= 2:
                    measurement = _draw_measurement_variance(
                        y, states, dns_loadings(MATURITIES_YEARS, decay), priors, rng
                    )
                if iteration >= config["warmup"]:
                    draw = iteration - config["warmup"]
                    stored["log_variance"][chain, draw] = log_variance
                    if stage_index >= 1:
                        stored["factor_mean"][chain, draw] = mean
                        stored["factor_persistence"][chain, draw] = persistence
                    if stage_index >= 2:
                        stored["measurement_variance"][chain, draw] = measurement
                    if stage_index >= 3:
                        stored["decay"][chain, draw] = decay
                    path_mean += states / (config["chains"] * config["draws"])
            acceptance.append(accepted / config["draws"] if stage_index >= 3 else None)
            print(f"{stage} chain {chain + 1} finished", flush=True)
        posterior = az.from_dict({"posterior": stored})
        diagnostics = {}
        for label, func, method in (
            ("rank_rhat", az.rhat, "rank"), ("bulk_ess", az.ess, "bulk"), ("tail_ess", az.ess, "tail")
        ):
            dataset = func(posterior, method=method).to_dataset()
            diagnostics[label] = {name: np.asarray(dataset[name].values).tolist() for name in dataset.data_vars}
        if not all(np.isfinite(value).all() for group in diagnostics.values() for value in group.values()):
            raise ValueError(f"nonfinite diagnostic in {stage}")
        rhat = max(np.max(value) for value in diagnostics["rank_rhat"].values())
        bulk = min(np.min(value) for value in diagnostics["bulk_ess"].values())
        tail = min(np.min(value) for value in diagnostics["tail_ess"].values())
        trace_path = OUTPUT / f"{stage}_traces.npz"
        np.savez_compressed(trace_path, **stored)
        result = {
            "status": "diagnostics_passed_not_full_recovery" if rhat <= 1.01 and bulk >= 400 and tail >= 400 else "rejected_convergence",
            "maximum_rank_rhat": float(rhat), "minimum_bulk_ess": float(bulk),
            "minimum_tail_ess": float(tail), "diagnostics": diagnostics,
            "factor_path_rmse": np.sqrt(np.mean((path_mean - simulation["factors"])**2, axis=0)).tolist(),
            "chain_means": {name: array.mean(axis=1).tolist() for name, array in stored.items()},
            "decay_acceptance": acceptance, "priors": priors.as_dict(),
            "elapsed_seconds": time.perf_counter() - started,
            "trace_sha256": sha256_file(trace_path),
        }
        result_path = OUTPUT / f"{stage}.json"
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(result_path)
        print(stage, result["status"], "Rhat", round(rhat, 3), "bulk", round(bulk, 1), "tail", round(tail, 1), flush=True)


if __name__ == "__main__":
    main()
