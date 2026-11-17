"""Screen the SV path and parameter blocks conditional on known innovations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arviz as az
import numpy as np

from .data import sha256_file
from .sampler import (
    _draw_stationary_ar1,
    _sample_volatility_path,
    _update_volatility_mean,
    _update_volatility_persistence,
    _update_volatility_variance,
)
from .sv_interweave import interweave_persistence, interweave_variance


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "results" / "hautsch_yang_state_space_20261117"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interwoven", action="store_true", help="add experimental non-centered MH moves")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / (
        "sv_conditional_interwoven_20261117" if args.interwoven else "sv_conditional_screen_20261117"
    )
    design = {
        "n": 120, "warmup": 500, "draws": 1500, "chains": 4,
        "seed": 20261118, "mean": float(np.log(.04)), "phi": .8, "variance": .1,
        "mean_prior_center": float(np.log(.04)), "mean_prior_sd": 1.,
        "phi_beta": [20., 1.5], "variance_upper": 4.,
        "stages": ["path_only", "release_mean", "release_persistence", "release_variance"],
        "interwoven": args.interwoven,
    }
    source = {
        path.name: sha256_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    metadata = {"design": design, "source_sha256": source}
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "design.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("existing SV screen uses a different design or source")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    data_rng = np.random.default_rng(design["seed"])
    true_h = (
        design["mean"] - design["variance"] / 2
        + _draw_stationary_ar1(design["n"], design["phi"], design["variance"], data_rng)
    )
    innovations = np.exp(true_h / 2) * data_rng.standard_normal(design["n"])
    for stage_index, stage in enumerate(design["stages"]):
        started = time.perf_counter()
        h_draws = np.empty((design["chains"], design["draws"], design["n"]))
        means = np.empty((design["chains"], design["draws"]))
        phis = np.empty_like(means)
        variances = np.empty_like(means)
        acceptance = []
        streams = np.random.SeedSequence(design["seed"] + 100 + stage_index).spawn(design["chains"])
        for chain, stream in enumerate(streams):
            rng = np.random.default_rng(stream)
            mean = design["mean"] + (chain - 1.5) * .5 if stage_index >= 1 else design["mean"]
            phi = [.55, .75, .9, .95][chain] if stage_index >= 2 else design["phi"]
            variance = [.05, .2, .5, 1.][chain] if stage_index >= 3 else design["variance"]
            h = np.full(design["n"], mean - variance / 2 + chain - 1.5)
            accepted_phi = accepted_variance = 0
            interwoven_phi = interwoven_variance = 0
            for iteration in range(design["warmup"] + design["draws"]):
                h = _sample_volatility_path(h, innovations, mean, phi, variance, rng)
                if stage_index >= 1:
                    mean = _update_volatility_mean(
                        h, phi, variance, design["mean_prior_center"],
                        design["mean_prior_sd"], rng,
                    )
                if stage_index >= 2:
                    phi, accepted = _update_volatility_persistence(
                        h, mean, phi, variance, *design["phi_beta"], rng,
                    )
                    if iteration >= design["warmup"]:
                        accepted_phi += int(accepted)
                if stage_index >= 3:
                    variance, accepted = _update_volatility_variance(
                        h, mean, phi, variance, design["variance_upper"], rng,
                    )
                    if iteration >= design["warmup"]:
                        accepted_variance += int(accepted)
                if args.interwoven and stage_index >= 2:
                    h, phi, accepted = interweave_persistence(
                        h, innovations, mean, phi, variance, *design["phi_beta"], rng
                    )
                    if iteration >= design["warmup"]:
                        interwoven_phi += int(accepted)
                if args.interwoven and stage_index >= 3:
                    h, variance, accepted = interweave_variance(
                        h, innovations, mean, phi, variance, design["variance_upper"], rng
                    )
                    if iteration >= design["warmup"]:
                        interwoven_variance += int(accepted)
                if iteration >= design["warmup"]:
                    index = iteration - design["warmup"]
                    h_draws[chain, index] = h
                    means[chain, index] = mean
                    phis[chain, index] = phi
                    variances[chain, index] = variance
            acceptance.append({
                "phi": accepted_phi / design["draws"] if stage_index >= 2 else None,
                "variance": accepted_variance / design["draws"] if stage_index >= 3 else None,
                "interwoven_phi": interwoven_phi / design["draws"] if args.interwoven and stage_index >= 2 else None,
                "interwoven_variance": interwoven_variance / design["draws"] if args.interwoven and stage_index >= 3 else None,
            })
        posterior = {"h": h_draws}
        if stage_index >= 1:
            posterior["mean"] = means
        if stage_index >= 2:
            posterior["phi"] = phis
        if stage_index >= 3:
            posterior["variance"] = variances
        inference = az.from_dict({"posterior": posterior})
        diagnostic = {}
        for label, func, method in (
            ("rank_rhat", az.rhat, "rank"),
            ("bulk_ess", az.ess, "bulk"),
            ("tail_ess", az.ess, "tail"),
        ):
            result = func(inference, method=method).to_dataset()
            diagnostic[label] = {
                name: np.asarray(result[name].values).tolist() for name in result.data_vars
            }
        all_diagnostics = [
            np.asarray(value) for values in diagnostic.values() for value in values.values()
        ]
        if not all(np.isfinite(value).all() for value in all_diagnostics):
            raise ValueError(f"nonfinite diagnostic in {stage}")
        max_rhat = max(np.max(value) for value in diagnostic["rank_rhat"].values())
        min_bulk = min(np.min(value) for value in diagnostic["bulk_ess"].values())
        min_tail = min(np.min(value) for value in diagnostic["tail_ess"].values())
        h_average = h_draws.mean(axis=(0, 1))
        h_interval = np.quantile(h_draws, [.025, .975], axis=(0, 1))
        result = {
            "status": "diagnostics_passed_not_full_model" if max_rhat <= 1.01 and min_bulk >= 400 and min_tail >= 400 else "rejected_convergence",
            "diagnostics": diagnostic,
            "maximum_rank_rhat": float(max_rhat),
            "minimum_bulk_ess": float(min_bulk),
            "minimum_tail_ess": float(min_tail),
            "h_rmse": float(np.sqrt(np.mean((h_average - true_h) ** 2))),
            "h_interval_pointwise_fraction_covering_truth": float(np.mean((h_interval[0] <= true_h) & (true_h <= h_interval[1]))),
            "h_true": true_h.tolist(),
            "h_mean": h_average.tolist(),
            "h_interval_95": h_interval.tolist(),
            "chain_mean": {
                name: array.mean(axis=1).tolist()
                for name, array in (("mean", means), ("phi", phis), ("variance", variances))
            },
            "acceptance": acceptance,
            "elapsed_seconds": time.perf_counter() - started,
        }
        trace_path = output_dir / f"{stage}_traces.npz"
        np.savez_compressed(
            trace_path, mean=means, phi=phis, variance=variances,
            h_selected=h_draws[:, :, [0, design["n"] // 2, design["n"] - 1]],
            h_truth=true_h, innovations=innovations,
        )
        result["trace_sha256"] = sha256_file(trace_path)
        output = output_dir / f"{stage}.json"
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(stage, result["status"], "Rhat", round(max_rhat, 3), "bulk", round(min_bulk, 1), "tail", round(min_tail, 1), flush=True)


if __name__ == "__main__":
    main()
