"""Reproducible, bounded synthetic screening; not a full recovery study."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from .data import MATURITIES_YEARS, sha256_file
from .diagnostics import summarize_chain_diagnostics
from .priors import calibrate_training_priors
from .sampler import ChainDraws, run_chain
from .synthetic import simulate_yields


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "hautsch_yang_state_space_20261117" / "synthetic_screen_20261117"
SETTINGS = {
    "n_observations": 120,
    "chains": 4,
    "warmup": 150,
    "draws": 300,
    "seed": 20261117,
    "replicates": 2,
    "initial_factor_variance": 1e6,
    "maturities_years": MATURITIES_YEARS.tolist(),
    "models": ["SS_M3", "SS_M4"],
    "factor_mean": [3.0, -1.0, 0.2],
    "factor_persistence": [0.82, 0.72, 0.55],
    "factor_innovation_variance": [0.04, 0.05, 0.03],
    "measurement_variance": [0.0036] * 8,
    "decay_per_year": 0.52,
    "volatility_persistence": [0.8] * 3,
    "stationary_log_variance": [0.1] * 3,
}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {path.name: sha256_file(path) for path in sorted(directory.glob("*.py"))}


def _save_chain(path: Path, chain: ChainDraws, fingerprint: str, seconds: float) -> None:
    arrays = {
        field.name: getattr(chain, field.name)
        for field in fields(ChainDraws)
        if field.name not in {"model", "acceptance"}
    }
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary, **arrays, model=chain.model,
        acceptance=json.dumps(chain.acceptance), fingerprint=fingerprint,
        elapsed_seconds=seconds,
    )
    temporary.replace(path)
    _atomic_json(path.with_suffix(".json"), {"sha256": sha256_file(path)})


def _load_chain(path: Path, fingerprint: str) -> tuple[ChainDraws, float]:
    checksum = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if checksum["sha256"] != sha256_file(path):
        raise ValueError(f"checkpoint checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["fingerprint"]) != fingerprint:
            raise ValueError(f"checkpoint does not match code, design, or data: {path}")
        values = {
            field.name: archive[field.name].copy()
            for field in fields(ChainDraws)
            if field.name not in {"model", "acceptance"}
        }
        chain = ChainDraws(
            model=str(archive["model"]),
            acceptance=json.loads(str(archive["acceptance"])),
            **values,
        )
        return chain, float(archive["elapsed_seconds"])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {"settings": SETTINGS, "source_sha256": _source_hashes()}
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata_path = OUTPUT / "design.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("existing synthetic screen has a different design or source hash")
    else:
        _atomic_json(metadata_path, metadata)

    for model in SETTINGS["models"]:
        for replicate in range(SETTINGS["replicates"]):
            dataset_seed = SETTINGS["seed"] + 1000 * (model == "SS_M4") + replicate
            simulation = simulate_yields(
                SETTINGS["n_observations"],
                MATURITIES_YEARS,
                SETTINGS["decay_per_year"],
                np.array(SETTINGS["factor_mean"]),
                np.array(SETTINGS["factor_persistence"]),
                np.array(SETTINGS["measurement_variance"]),
                np.random.default_rng(dataset_seed),
                model,
                np.log(SETTINGS["factor_innovation_variance"]),
                np.array(SETTINGS["volatility_persistence"]) if model == "SS_M4" else None,
                np.array(SETTINGS["stationary_log_variance"]) if model == "SS_M4" else None,
                initial_factor_variance=SETTINGS["initial_factor_variance"],
            )
            priors, _ = calibrate_training_priors(simulation["yields"], MATURITIES_YEARS)
            prefix = f"{model.lower()}_rep{replicate + 1}"
            data_hash = hashlib.sha256(simulation["yields"].tobytes()).hexdigest()
            run_fingerprint = hashlib.sha256((fingerprint + data_hash).encode()).hexdigest()
            chains = []
            times = []
            streams = np.random.SeedSequence(dataset_seed + 10000).spawn(SETTINGS["chains"])
            for index, stream in enumerate(streams):
                path = OUTPUT / f"{prefix}_chain{index + 1}.npz"
                if path.exists():
                    chain, elapsed = _load_chain(path, run_fingerprint)
                else:
                    started = time.perf_counter()
                    chain = run_chain(
                        simulation["yields"], MATURITIES_YEARS, priors, model,
                        np.random.default_rng(stream), SETTINGS["warmup"], SETTINGS["draws"],
                    )
                    elapsed = time.perf_counter() - started
                    _save_chain(path, chain, run_fingerprint, elapsed)
                chains.append(chain)
                times.append(elapsed)
                print(f"{prefix} chain {index + 1}: {elapsed:.1f}s", flush=True)

            diagnostics = summarize_chain_diagnostics(chains)
            pooled = {
                name: np.concatenate([getattr(chain, name) for chain in chains])
                for name in (
                    "factor_mean", "persistence", "log_variance",
                    "measurement_variance", "decay_per_year",
                    "volatility_persistence", "volatility_variance",
                )
            }
            truths = {
                "factor_mean": SETTINGS["factor_mean"],
                "persistence": SETTINGS["factor_persistence"],
                "log_variance": np.log(SETTINGS["factor_innovation_variance"]).tolist(),
                "measurement_variance": SETTINGS["measurement_variance"],
                "decay_per_year": SETTINGS["decay_per_year"],
            }
            if model == "SS_M4":
                truths["volatility_persistence"] = SETTINGS["volatility_persistence"]
                truths["volatility_variance"] = SETTINGS["stationary_log_variance"]
            estimates = {}
            for name, truth in truths.items():
                draws = pooled[name]
                lower, upper = np.quantile(draws, [.025, .975], axis=0)
                estimates[name] = {
                    "truth": truth,
                    "mean": np.mean(draws, axis=0).tolist(),
                    "interval_95": [lower.tolist(), upper.tolist()],
                    "covered": ((lower <= truth) & (truth <= upper)).tolist(),
                    "chain_means": [np.mean(getattr(chain, name), axis=0).tolist() for chain in chains],
                }
            factors = np.mean([chain.factor_path_mean for chain in chains], axis=0)
            passed = (
                diagnostics["maximum_rank_rhat"] <= 1.01
                and diagnostics["minimum_bulk_ess"] >= 400
                and diagnostics["minimum_tail_ess"] >= 400
            )
            result = {
                "status": "screen_passed_not_recovery_validation" if passed else "rejected_convergence",
                "dataset_seed": dataset_seed,
                "data_sha256": data_hash,
                "initial_factor": simulation["initial_factor"].tolist(),
                "priors": priors.as_dict(),
                "diagnostics": diagnostics,
                "estimates": estimates,
                "factor_path_rmse": np.sqrt(np.mean((factors - simulation["factors"]) ** 2, axis=0)).tolist(),
                "chain_seconds": times,
                "acceptance": [chain.acceptance for chain in chains],
                "checkpoints_sha256": {
                    f"chain{index + 1}": sha256_file(OUTPUT / f"{prefix}_chain{index + 1}.npz")
                    for index in range(SETTINGS["chains"])
                },
            }
            _atomic_json(OUTPUT / f"{prefix}_summary.json", result)
            print(f"{prefix}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
