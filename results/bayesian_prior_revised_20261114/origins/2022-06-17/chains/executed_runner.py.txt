"""Versioned representative and rolling experiment for the direct-mean prior."""
from __future__ import annotations

import contextlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import date
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from src import bayesian_finite as bf
from src import bayesian_finite_analysis as analysis
from src import bayesian_finite_gate as baseline_gate
from src import bayesian_finite_validation as validation
from src import bayesian_prior_longrun as sampler
from src.bayesian_stage_a import FACTORS, ROOT, diagnose, make_design
from src.experiment import select_origins, write_csv


VERSION = "bayesian_prior_revised_20261114"
DEFAULT_OUTPUT = ROOT / "results" / VERSION
MODELS = ["M4", "M5"]
HORIZONS = [1, 4]
WARMUP = 3000
DRAWS = 6000
CHAINS = 4
SCENARIOS = 2048
REPRESENTATIVE_SEEDS = [9102701, 10102701]
SOURCE_FILES = [
    "src/bayesian_prior_longrun.py",
    "src/bayesian_prior_longrun_experiment.py",
    "src/bayesian_finite.py",
    "src/bayesian_revised.py",
    "src/bayesian_stage_a.py",
    "src/bayesian_stage_a_checks.py",
    "src/bayesian_finite_validation.py",
    "src/bayesian_finite_analysis.py",
    "src/bayesian_finite_gate.py",
    "src/experiment.py",
    "src/models.py",
    "src/scores.py",
    "src/prior_predictive_bayesian_dns.py",
    "src/prior_predictive_longrun_mean.py",
]
PROTECTED_FILES = [
    "results/bayesian_finite_20261027T194239Z/protocol.md",
    "results/bayesian_finite_20261027T194239Z/phase2_gate.json",
    "results/bayesian_finite_20261027T194239Z/rolling_main/summary.json",
    "results/bayesian_finite_20261027T194239Z/rolling_main/analysis/summary.json",
]


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: digest(ROOT / name) for name in SOURCE_FILES}


def _configuration(design: dict, window: dict, seed: int, priors: tuple) -> dict:
    return {
        "window": window,
        "warmup": WARMUP,
        "draws": DRAWS,
        "chains": CHAINS,
        "models": MODELS,
        "seed": seed,
        "input_sha256": design["input_sha256"],
        "lambda": design["lambda"],
        "priors": sampler.prior_configuration(priors),
        "parameterization": "mu=a/(1-b), prior specified directly on mu and b",
        "source_sha256": _source_hashes(),
    }


def _chain_seed(seed: int, model_index: int, factor_index: int, chain: int) -> int:
    return seed + model_index * 100000 + factor_index * 1000 + chain


def run_window(output: Path, window_name: str | dict, seed: int) -> dict:
    """Fit a fixed origin and checkpoint every chain."""
    design = make_design()
    if isinstance(window_name, dict):
        window = window_name
    else:
        window = next(row for row in design["windows"] if row["name"] == window_name)
    window_label = window.get("name", window["date"])
    priors = sampler.build_priors(design)
    configuration = _configuration(design, window, seed, priors)
    signature = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "configuration.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Cannot resume a different prior, source, or sampling configuration")
    else:
        config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
        (output / "executed_sampler.py.txt").write_bytes(
            (ROOT / "src/bayesian_prior_longrun.py").read_bytes()
        )
        (output / "executed_runner.py.txt").write_bytes(Path(__file__).read_bytes())

    start = design["first_index"]
    factors = design["factors"][start:window["index"] + 1]
    diagnostics = []
    checkpoints = {}
    chain_seeds = []
    for model_index, model in enumerate(MODELS):
        for factor_index, factor in enumerate(FACTORS):
            chain_values = []
            names = None
            for chain in range(CHAINS):
                filename = f"{model}_{factor}_chain{chain}.npz"
                path = output / filename
                chain_seed = _chain_seed(seed, model_index, factor_index, chain)
                if path.exists():
                    with np.load(path, allow_pickle=False) as archive:
                        values = archive["draws"].copy()
                        report = json.loads(str(archive["report"]))
                    if report.get("signature") != signature or report.get("seed") != chain_seed:
                        raise ValueError(f"Changed checkpoint configuration: {filename}")
                else:
                    report = sampler.sample_chain(
                        factors[:, factor_index], priors[factor_index], model,
                        np.random.default_rng(chain_seed), WARMUP, DRAWS,
                    )
                    values = report.pop("draws")
                    sampler.validate_draws(values, report["names"], model)
                    report.update(signature=signature, seed=chain_seed)
                    temporary = path.with_suffix(".partial.npz")
                    with temporary.open("xb") as stream:
                        np.savez_compressed(
                            stream, draws=values,
                            report=json.dumps(report, allow_nan=False),
                        )
                    temporary.replace(path)
                if values.shape != (DRAWS, len(report["names"])):
                    raise ValueError(f"Invalid posterior checkpoint shape: {filename}")
                sampler.validate_draws(values, report["names"], model)
                if names is not None and names != report["names"]:
                    raise ValueError(f"Inconsistent parameter names: {filename}")
                names = report["names"]
                chain_values.append(values)
                chain_seeds.append(chain_seed)
                checkpoints[filename] = digest(path)
            diagnostics.extend(
                {"model": model, "factor": factor, **row}
                for row in diagnose(np.stack(chain_values), names)
            )
            print(window_label, seed, model, factor, "complete", flush=True)

    moment_reports = sampler.moment_audits(priors, window["n_training"] - 1)
    summary = {
        "configuration": configuration,
        "signature": signature,
        "diagnostics": diagnostics,
        "parameter_gate_passed": bool(diagnostics and all(row["passes"] for row in diagnostics)),
        "moment_audits": moment_reports,
        "moment_gate_passed": all(
            row["stage_a_mathematical_gate_passed"]
            and row["long_run_mean_second_moment_finite"]
            and row["long_run_mean_absolute_mean_finite"]
            for row in moment_reports
        ),
        "chain_seeds": chain_seeds,
        "checkpoints": checkpoints,
        "accepted_for_rolling": False,
        "status": "parameter fit only; predictive validation is separate",
    }
    write_csv(output / "diagnostics.csv", diagnostics)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary


def _posterior_factors(chain_dir: Path, summary: dict) -> list:
    factors = []
    for model in MODELS:
        for factor in FACTORS:
            chains = []
            names = None
            for chain in range(CHAINS):
                filename = f"{model}_{factor}_chain{chain}.npz"
                path = chain_dir / filename
                if digest(path) != summary["checkpoints"][filename]:
                    raise ValueError(f"Changed posterior checkpoint: {filename}")
                with np.load(path, allow_pickle=False) as archive:
                    values = archive["draws"].copy()
                    report = json.loads(str(archive["report"]))
                sampler.validate_draws(values, report["names"], model)
                if names is not None and names != report["names"]:
                    raise ValueError(f"Changed posterior parameter order: {filename}")
                names = report["names"]
                chains.append(values)
            factors.append((np.stack(chains), names))
    return factors


def validate_representative(chain_dir: Path) -> dict:
    """Generate score-reconstruction and predictive-stability records from saved draws."""
    chain_dir = Path(chain_dir)
    summary = json.loads((chain_dir / "summary.json").read_text(encoding="utf-8"))
    configuration = json.loads((chain_dir / "configuration.json").read_text(encoding="utf-8"))
    if (configuration != summary["configuration"]
        or configuration["source_sha256"] != _source_hashes()
        or not summary["parameter_gate_passed"]
        or not summary["moment_gate_passed"]):
        raise ValueError("Unaccepted or changed representative posterior")
    factors = _posterior_factors(chain_dir, summary)
    design = make_design()
    index = configuration["window"]["index"]
    start = design["first_index"]
    ff = design["factors"][start:index + 1]
    yy = design["y"][start:index + 1]
    reconstruction = np.maximum(np.mean((yy - ff @ design["load"].T)**2, axis=0), 1e-6)
    records = []
    archive = {}
    for model_index, model in enumerate(MODELS):
        model_factors = [
            factors[model_index * len(FACTORS) + j]
            for j in range(len(FACTORS))
        ]
        for horizon in HORIZONS:
            seed = configuration["seed"] + 700000 + 1000 * (model == "M5") + horizon * 100
            record = validation.predictive_record(
                model_factors, ff[-1], design["load"], reconstruction,
                design["y"][index + horizon], model, horizon, np.random.default_rng(seed),
                size=SCENARIOS,
            )
            samples = record.pop("scenarios")
            lower, upper = np.quantile(samples, [.05, .95], axis=0)
            observed = design["y"][index + horizon]
            record.update(
                origin=str(design["dates"][index]),
                target=str(design["dates"][index + horizon]),
                scenario_seed=seed,
                squared_errors=((observed - np.asarray(record["mean"]))**2).tolist(),
                interval_lower_90=lower.tolist(),
                interval_upper_90=upper.tolist(),
                coverage_90=((observed >= lower) & (observed <= upper)).tolist(),
            )
            tag = f"{model}_h{horizon}"
            archive[f"{tag}_scenarios"] = samples
            archive[f"{tag}_mean"] = np.asarray(record["mean"])
            archive[f"{tag}_covariance"] = np.asarray(record["covariance"])
            archive[f"{tag}_observed"] = observed
            records.append(record)

    validation_dir = chain_dir / "predictive_validation"
    validation_dir.mkdir(exist_ok=True)
    prediction_path = validation_dir / "predictions.npz"
    temporary = validation_dir / "predictions.partial.npz"
    np.savez_compressed(temporary, **archive)
    temporary.replace(prediction_path)
    report = {
        "configuration": configuration,
        "chain_seeds": summary["chain_seeds"],
        "records": records,
        "reconstruction_variance": reconstruction.tolist(),
        "parameter_diagnostics": summary["diagnostics"],
        "moment_audits": summary["moment_audits"],
        "passes": bool(len(records) == 4 and all(row["passes"] for row in records)),
        "prediction_archive_sha256": digest(prediction_path),
        "chain_checkpoint_sha256": summary["checkpoints"],
        "status": "representative forecast diagnostics only; not rolling evidence",
    }
    (validation_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def run_synthetic(output: Path) -> dict:
    """Run a fixed-data four-chain recovery check without tuning to interval hits."""
    from src.bayesian_stage_a_checks import simulate_sv
    from src import bayesian_revised as br

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    truth = {"a": 0.04, "b": 0.85, "omega": -3.0, "phi": 0.94, "q": 0.08}
    truth["v"] = truth["q"] / (1 - truth["phi"]**2)
    truth["m"] = truth["omega"] + truth["v"] / 2
    truth["long_run_mean"] = truth["a"] / (1 - truth["b"])
    x, h = simulate_sv(n=240, seed=30261027, truth=truth)
    base = br.freeze_priors(np.column_stack((x, x, x)))[0]
    factor = np.asarray(x)
    lag, response = factor[:-1], factor[1:]
    beta = np.linalg.lstsq(np.column_stack((np.ones(len(lag)), lag)), response, rcond=None)[0]
    base = replace(base, omega_variance=1.0)
    prior = sampler.LongRunPrior(
        bf.FinitePrior(base, v_upper=4.0, v_rate=1.0),
        mean_center=float(np.mean(factor)),
        mean_variance=float((2 * np.std(factor, ddof=1))**2),
        persistence_center=float(beta[1]),
    )
    configuration = {
        "truth": truth,
        "data_seed": 30261027,
        "chain_seeds": [81261027 + chain for chain in range(CHAINS)],
        "warmup": 1000,
        "draws": 2000,
        "prior": sampler.prior_record(prior),
        "source_sha256": _source_hashes(),
    }
    (output / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output / "data.npz", x=x, h=h)
    values = []
    reports = []
    for seed in configuration["chain_seeds"]:
        result = sampler.sample_chain(
            x, prior, "M5", np.random.default_rng(seed),
            configuration["warmup"], configuration["draws"],
        )
        values.append(result.pop("draws"))
        reports.append({"seed": seed, **result})
    values = np.stack(values)
    names = reports[0]["names"]
    diagnostics = diagnose(values, names)
    recovery = []
    for parameter in ["a", "b", "long_run_mean", "m", "phi", "v"]:
        sample = values[:, :, names.index(parameter)].reshape(-1)
        lower, median, upper = np.quantile(sample, [.025, .5, .975])
        recovery.append({
            "parameter": parameter,
            "truth": truth[parameter],
            "lower": float(lower),
            "median": float(median),
            "upper": float(upper),
            "contains_truth": bool(lower <= truth[parameter] <= upper),
        })
    np.savez_compressed(output / "draws.npz", draws=values, names=np.array(names))
    moment_report = sampler.moment_audits((prior,), len(x) - 1)[0]
    report = {
        "configuration": configuration,
        "diagnostics": diagnostics,
        "recovery": recovery,
        "parameter_gate_passed": all(row["passes"] for row in diagnostics),
        "moment_audit": moment_report,
        "data_sha256": digest(output / "data.npz"),
        "draws_sha256": digest(output / "draws.npz"),
        "synthetic_accepted": bool(
            all(row["passes"] for row in diagnostics)
            and moment_report["stage_a_mathematical_gate_passed"]
        ),
        "status": "one fixed synthetic series; interval misses are reported, not tuned away",
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def run_representative_phase(output: Path, workers: int = 4) -> dict:
    """Fit two independent seed sets at initial, stress and late origins."""
    if not 1 <= workers <= 8:
        raise ValueError("Choose between one and eight workers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    design = make_design()
    jobs = [
        (str(output / "representative" / window["name"] / str(seed)), window["name"], seed)
        for window in design["windows"]
        for seed in REPRESENTATIVE_SEEDS
    ]
    failures = []
    if workers == 1:
        for path, name, seed in jobs:
            try:
                run_window(Path(path), name, seed)
            except Exception:
                failures.append({"window": name, "seed": seed, "traceback": traceback.format_exc()})
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(run_window, Path(path), name, seed): (name, seed) for path, name, seed in jobs}
            for future in pending:
                try:
                    future.result()
                except Exception:
                    name, seed = pending[future]
                    failures.append({"window": name, "seed": seed, "traceback": traceback.format_exc()})
    if failures:
        (output / "representative_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError("Representative posterior fit failed; rolling experiment is forbidden")

    synthetic = run_synthetic(output / "synthetic")
    if synthetic["configuration"]["source_sha256"] != _source_hashes():
        raise ValueError("Synthetic recovery was generated by changed source")
    evidence = {}
    pairs = []
    for window in design["windows"]:
        reports = []
        for seed in REPRESENTATIVE_SEEDS:
            chain_dir = output / "representative" / window["name"] / str(seed)
            validation_report = validate_representative(chain_dir)
            summary_path = chain_dir / "summary.json"
            validation_path = chain_dir / "predictive_validation" / "summary.json"
            evidence[str(summary_path)] = digest(summary_path)
            evidence[str(validation_path)] = digest(validation_path)
            validation_record = json.loads(validation_path.read_text(encoding="utf-8"))
            evidence[str(chain_dir / "predictive_validation" / "predictions.npz")] = (
                validation_record["prediction_archive_sha256"]
            )
            for filename in json.loads(summary_path.read_text(encoding="utf-8"))["checkpoints"]:
                evidence[str(chain_dir / filename)] = digest(chain_dir / filename)
            reports.append(validation_report)
        comparison = baseline_gate.compare_run_reports(reports[0], reports[1])
        comparison["window_name"] = window["name"]
        pairs.append(comparison)
    protected_before = output / "protected_before.json"
    if not protected_before.exists():
        protected_before.write_text(json.dumps({
            str(ROOT / name): digest(ROOT / name) for name in PROTECTED_FILES
        }, indent=2), encoding="utf-8")
    protected = json.loads(protected_before.read_text(encoding="utf-8"))
    changed = [
        path for path, checksum in protected.items()
        if not Path(path).is_file() or digest(path) != checksum
    ]
    synthetic_path = output / "synthetic" / "summary.json"
    evidence[str(synthetic_path)] = digest(synthetic_path)
    evidence[str(output / "synthetic" / "data.npz")] = digest(output / "synthetic" / "data.npz")
    evidence[str(output / "synthetic" / "draws.npz")] = digest(output / "synthetic" / "draws.npz")
    prior_metadata = ROOT / "results/prior_justification_20261114/longrun_mean_candidate/prior_predictive_metadata.json"
    prior_metadata_record = json.loads(prior_metadata.read_text(encoding="utf-8"))
    if (prior_metadata_record.get("status") != "completed"
        or prior_metadata_record.get("input_sha256") != design["input_sha256"]
        or prior_metadata_record.get("forecast_scores_used_to_choose_priors") is not False
        or not prior_metadata_record.get("all_draws_retained")
        or any(
            count
            for quantity in prior_metadata_record["numerical_failures"].values()
            for count in quantity.values()
        )):
        raise ValueError("Prior-predictive screening is incomplete or numerically invalid")
    for relative, expected in prior_metadata_record["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Prior-predictive source changed: {relative}")
    evidence[str(prior_metadata)] = digest(prior_metadata)
    source_hashes = _source_hashes()
    all_diagnostics = list(synthetic["diagnostics"])
    for window in design["windows"]:
        for seed in REPRESENTATIVE_SEEDS:
            all_diagnostics.extend(json.loads(
                (output / "representative" / window["name"] / str(seed) / "summary.json")
                .read_text(encoding="utf-8")
            )["diagnostics"])
    finite_diagnostics = [
        row for row in all_diagnostics
        if all(row.get(key) is not None and np.isfinite(row[key])
               for key in ["rhat", "ess_bulk", "ess_tail"])
    ]
    diagnostics_complete = len(finite_diagnostics) == len(all_diagnostics)
    gate = {
        "accepted": bool(
            synthetic["synthetic_accepted"]
            and len(pairs) == 3
            and all(pair["passes"] for pair in pairs)
            and not changed
            and diagnostics_complete
        ),
        "prior_predictive_evidence": str(prior_metadata),
        "prior_predictive_sha256": digest(prior_metadata),
        "input_sha256": prior_metadata_record["input_sha256"],
        "representative_pairs": pairs,
        "synthetic_accepted": synthetic["synthetic_accepted"],
        "synthetic_recovery_interval_misses": [
            row["parameter"] for row in synthetic["recovery"] if not row["contains_truth"]
        ],
        "max_rhat": max((row["rhat"] for row in finite_diagnostics), default=None),
        "min_bulk_ess": min((row["ess_bulk"] for row in finite_diagnostics), default=0),
        "min_tail_ess": min((row["ess_tail"] for row in finite_diagnostics), default=0),
        "source_sha256": source_hashes,
        "evidence_sha256": evidence,
        "protected_checked": len(protected),
        "protected_changed": changed,
        "prior_parameterization": "mu=a/(1-b); independent training-calibrated priors on mu and b",
        "forecast_scores_used_to_choose_priors": False,
    }
    gate_path = output / "phase2_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, allow_nan=False), encoding="utf-8")
    return gate


def require_prior_gate(path: Path) -> dict:
    """Fail closed unless the candidate prior passed independent representative checks."""
    path = Path(path)
    gate = json.loads(path.read_text(encoding="utf-8"))
    if not gate.get("accepted") or gate.get("protected_changed") or not gate.get("synthetic_accepted"):
        raise ValueError("Revised-prior representative gate has not been accepted")
    max_rhat = gate.get("max_rhat")
    if (max_rhat is None or not np.isfinite(max_rhat) or max_rhat > 1.01
        or gate.get("min_bulk_ess", 0) < 400
        or gate.get("min_tail_ess", 0) < 400
        or len(gate.get("representative_pairs", [])) != 3
        or {pair.get("window_name") for pair in gate.get("representative_pairs", [])}
        != {"initial", "stress", "late"}
        or not all(pair.get("passes") for pair in gate["representative_pairs"])):
        raise ValueError("Revised-prior diagnostics do not meet the numerical gate")
    for relative, expected in gate["source_sha256"].items():
        if not (ROOT / relative).is_file() or digest(ROOT / relative) != expected:
            raise ValueError(f"Revised-prior source changed: {relative}")
    for filename, expected in gate["evidence_sha256"].items():
        if not Path(filename).is_file() or digest(filename) != expected:
            raise ValueError(f"Revised-prior evidence changed: {filename}")
    for filename, expected in json.loads(
        (path.parent / "protected_before.json").read_text(encoding="utf-8")
    ).items():
        if not Path(filename).is_file() or digest(filename) != expected:
            raise ValueError(f"Protected baseline changed: {filename}")
    if gate["prior_predictive_sha256"] != digest(gate["prior_predictive_evidence"]):
        raise ValueError("Prior-predictive screening evidence changed")
    return gate


def make_plan(output: Path) -> dict:
    output = Path(output)
    gate_path = output / "phase2_gate.json"
    require_prior_gate(gate_path)
    design = make_design()
    origins = sorted({
        index for index, _, _ in select_origins(
            design["dates"], date(2016, 1, 1), date(2025, 12, 31), HORIZONS, 4
        )
    })
    if len(origins) != 130:
        raise ValueError("Common 130-origin forecast design changed")
    return {
        "version": VERSION,
        "gate_sha256": digest(gate_path),
        "input_sha256": design["input_sha256"],
        "lambda": design["lambda"],
        "maturities_years": [0.25, 0.5, 1, 2, 5, 7, 10, 30],
        "horizons": HORIZONS,
        "models": MODELS,
        "warmup": WARMUP,
        "draws": DRAWS,
        "chains": CHAINS,
        "scenario_count": SCENARIOS,
        "prior_configuration": sampler.prior_configuration(sampler.build_priors(design)),
        "source_sha256": _source_hashes(),
        "origins": [
            {
                "ordinal": ordinal,
                "index": index,
                "date": design["dates"][index].isoformat(),
                "seed": 9102701 + 1000000 * ordinal,
                "status": "pending",
            }
            for ordinal, index in enumerate(origins)
        ],
    }


def forecast_origin(gate_path: Path, output: Path, row: dict) -> dict:
    """Fit, validate and save one candidate-prior forecast origin."""
    gate = require_prior_gate(gate_path)
    design = make_design()
    origins = {
        index for index, _, _ in select_origins(
            design["dates"], date(2016, 1, 1), date(2025, 12, 31), HORIZONS, 4
        )
    }
    index = int(row["index"])
    if (index not in origins or design["dates"][index].isoformat() != row["date"]
        or design["input_sha256"] != gate["input_sha256"]):
        raise ValueError("Origin or accepted input does not match the frozen design")
    output = Path(output)
    specification = {
        "origin_index": index,
        "origin": row["date"],
        "seed": int(row["seed"]),
        "gate_sha256": digest(gate_path),
        "input_sha256": design["input_sha256"],
        "lambda": design["lambda"],
        "warmup": WARMUP,
        "draws": DRAWS,
        "chains": CHAINS,
        "models": MODELS,
        "horizons": HORIZONS,
        "scenario_count": SCENARIOS,
        "prior_configuration": sampler.prior_configuration(sampler.build_priors(design)),
        "source_sha256": _source_hashes(),
    }
    metadata = output / "forecast_specification.json"
    completed = output / "forecast.json"
    if completed.exists():
        report = json.loads(completed.read_text(encoding="utf-8"))
        if (report.get("specification") != specification
            or digest(output / "predictions.npz") != report["prediction_archive_sha256"]
            or digest(output / "chains" / "summary.json") != report["parameter_summary_sha256"]):
            raise ValueError("Changed completed candidate forecast")
        for path, checksum in report["chain_checkpoint_sha256"].items():
            if not Path(path).is_file() or digest(path) != checksum:
                raise ValueError("Changed completed posterior checkpoint")
        return report
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != specification:
            raise ValueError("Changed candidate forecast specification")
    else:
        output.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps(specification, indent=2), encoding="utf-8")
        (output / "executed_production.py.txt").write_bytes(Path(__file__).read_bytes())

    window = {
        "name": row["date"],
        "index": index,
        "date": row["date"],
        "n_training": index - design["first_index"] + 1,
        "targets": {str(h): design["dates"][index + h].isoformat() for h in HORIZONS},
    }
    chain_dir = output / "chains"
    summary = run_window(chain_dir, window, int(row["seed"]))
    if not summary["parameter_gate_passed"] or not summary["moment_gate_passed"]:
        raise ValueError("Candidate posterior failed parameter or finite-moment diagnostics")

    posterior = _posterior_factors(chain_dir, summary)
    start = design["first_index"]
    ff = design["factors"][start:index + 1]
    yy = design["y"][start:index + 1]
    reconstruction = np.maximum(np.mean((yy - ff @ design["load"].T)**2, axis=0), 1e-6)
    records = []
    archive = {}
    checkpoint_hashes = {
        str((chain_dir / name).resolve()): checksum
        for name, checksum in summary["checkpoints"].items()
    }
    for model_index, model in enumerate(MODELS):
        model_factors = [
            posterior[model_index * len(FACTORS) + j]
            for j in range(len(FACTORS))
        ]
        for horizon in HORIZONS:
            scenario_seed = int(row["seed"]) + 700000 + 1000 * (model == "M5") + horizon * 100
            observed = design["y"][index + horizon]
            record = validation.predictive_record(
                model_factors, ff[-1], design["load"], reconstruction, observed,
                model, horizon, np.random.default_rng(scenario_seed), size=SCENARIOS,
            )
            samples = record.pop("scenarios")
            lower, upper = np.quantile(samples, [.05, .95], axis=0)
            record.update(
                origin=row["date"],
                target=design["dates"][index + horizon].isoformat(),
                scenario_seed=scenario_seed,
                squared_errors=((observed - np.asarray(record["mean"]))**2).tolist(),
                interval_lower_90=lower.tolist(),
                interval_upper_90=upper.tolist(),
                coverage_90=((observed >= lower) & (observed <= upper)).tolist(),
            )
            tag = f"{model}_h{horizon}"
            archive[f"{tag}_scenarios"] = samples
            archive[f"{tag}_mean"] = np.asarray(record["mean"])
            archive[f"{tag}_covariance"] = np.asarray(record["covariance"])
            archive[f"{tag}_observed"] = observed
            records.append(record)

    if not all(record["passes"] for record in records):
        raise ValueError("Candidate derived-prediction diagnostics failed")
    prediction_path = output / "predictions.npz"
    temporary = output / "predictions.partial.npz"
    np.savez_compressed(temporary, **archive)
    temporary.replace(prediction_path)
    report = {
        "accepted": True,
        "specification": specification,
        "records": records,
        "reconstruction_variance": reconstruction.tolist(),
        "prediction_archive_sha256": digest(prediction_path),
        "parameter_summary_sha256": digest(chain_dir / "summary.json"),
        "chain_checkpoint_sha256": checkpoint_hashes,
        "parameter_diagnostics": summary["diagnostics"],
        "moment_audits": summary["moment_audits"],
        "chain_source": str(chain_dir.resolve()),
    }
    temporary_json = output / "forecast.partial.json"
    temporary_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary_json.replace(completed)
    return report


def _origin_job(gate_path: str, root: str, row: dict) -> dict:
    """Worker wrapper: preserve exceptions and never silently retry failed origins."""
    output = Path(root) / "origins" / row["date"]
    log_dir = Path(root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (log_dir / f"{row['date']}.log").open("a", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                result = forecast_origin(Path(gate_path), output, row)
        return {"date": row["date"], "accepted": bool(result["accepted"])}
    except Exception as exc:
        failure = {
            "date": row["date"],
            "accepted": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        return failure


def run_experiment(output: Path, workers: int = 4) -> dict:
    """Run all 130 origins only after the revised-prior phase-2 gate passes."""
    if not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("Choose between one and eight workers")
    output = Path(output)
    gate_path = output / "phase2_gate.json"
    plan = make_plan(output)
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Changed frozen revised-prior experiment plan")
    else:
        output.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        (output / "executed_orchestrator.py.txt").write_bytes(Path(__file__).read_bytes())

    completed = {}
    failures = {}

    def save_summary():
        report = {
            "accepted": len(completed) == 130 and not failures,
            "planned_origins": 130,
            "completed_origins": len(completed),
            "completed_dates": sorted(completed),
            "failed_origins": list(failures.values()),
            "not_completed_dates": [
                row["date"] for row in plan["origins"] if row["date"] not in completed
            ],
            "plan_sha256": digest(plan_path),
            "workers": workers,
            "forecasts_sha256": {
                row["date"]: digest(output / "origins" / row["date"] / "forecast.json")
                for row in plan["origins"] if row["date"] in completed
            },
        }
        partial = output / "summary.partial.json"
        partial.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(partial, output / "summary.json")
        return report

    pending = []
    for row in plan["origins"]:
        origin = output / "origins" / row["date"]
        if (origin / "forecast.json").exists():
            result = _origin_job(str(gate_path), str(output), row)
            (completed if result["accepted"] else failures)[row["date"]] = result
        elif (origin / "failure.json").exists():
            failures[row["date"]] = json.loads(
                (origin / "failure.json").read_text(encoding="utf-8")
            )
        else:
            pending.append(row)
    save_summary()
    if failures:
        return save_summary()

    if workers == 1:
        for row in pending:
            result = _origin_job(str(gate_path), str(output), row)
            (completed if result["accepted"] else failures)[row["date"]] = result
            print(json.dumps({"origin": row["date"], "accepted": result["accepted"],
                              "completed": len(completed)}), flush=True)
            save_summary()
            if failures:
                break
    else:
        iterator = iter(pending)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            inflight = {}
            for _ in range(min(workers, len(pending))):
                row = next(iterator)
                inflight[pool.submit(_origin_job, str(gate_path), str(output), row)] = row
            while inflight:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    row = inflight.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"date": row["date"], "accepted": False, "error": repr(exc)}
                    (completed if result["accepted"] else failures)[row["date"]] = result
                    print(json.dumps({"origin": row["date"], "accepted": result["accepted"],
                                      "completed": len(completed)}), flush=True)
                    save_summary()
                if not failures:
                    for _ in range(workers - len(inflight)):
                        row = next(iterator, None)
                        if row is None:
                            break
                        inflight[pool.submit(_origin_job, str(gate_path), str(output), row)] = row
    result = save_summary()
    if result["accepted"]:
        analyze_experiment(output)
    return result


def analyze_experiment(output: Path) -> dict:
    """Independently audit all saved forecasts, then aggregate without refitting."""
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if (not summary.get("accepted") or summary.get("planned_origins") != 130
        or summary.get("completed_origins") != 130 or summary.get("failed_origins")
        or summary.get("not_completed_dates")):
        raise ValueError("Incomplete revised-prior rolling experiment")
    if (digest(output / "plan.json") != summary["plan_sha256"]
        or plan["source_sha256"] != _source_hashes()):
        raise ValueError("Changed revised-prior experiment plan")
    design = make_design()
    records = []
    archive_audits = []
    diagnostics = []
    for row in plan["origins"]:
        folder = output / "origins" / row["date"]
        forecast_path = folder / "forecast.json"
        if summary["forecasts_sha256"].get(row["date"]) != digest(forecast_path):
            raise ValueError(f"Changed forecast record: {row['date']}")
        report = json.loads(forecast_path.read_text(encoding="utf-8"))
        if not report.get("accepted") or report["specification"]["origin"] != row["date"]:
            raise ValueError(f"Unaccepted or unmatched forecast: {row['date']}")
        chain_dir = folder / "chains"
        if digest(chain_dir / "summary.json") != report["parameter_summary_sha256"]:
            raise ValueError(f"Changed posterior summary: {row['date']}")
        for chain_path, checksum in report["chain_checkpoint_sha256"].items():
            if not Path(chain_path).is_file() or digest(chain_path) != checksum:
                raise ValueError(f"Changed posterior checkpoint: {chain_path}")
        archive_audits.append({
            "origin": row["date"],
            **analysis.audit_origin(folder, report, design, energy_check=False),
        })
        records.extend(report["records"])
        diagnostics.extend(report["parameter_diagnostics"])
        for record in report["records"]:
            diagnostics.extend(record["derived_diagnostics"])
    result = analysis.aggregate_records(records)
    result.update(
        accepted_main_analysis=True,
        prior_parameterization=plan["prior_configuration"],
        phase2_gate_sha256=plan["gate_sha256"],
        plan_sha256=digest(output / "plan.json"),
        max_rhat=max(row["rhat"] for row in diagnostics),
        min_bulk_ess=min(row["ess_bulk"] for row in diagnostics),
        min_tail_ess=min(row["ess_tail"] for row in diagnostics),
        archive_audits=archive_audits,
        source_sha256={
            name: digest(ROOT / name)
            for name in [
                "src/bayesian_prior_longrun_experiment.py",
                "src/bayesian_finite_analysis.py",
                "src/scores.py",
            ]
        },
    )
    out = output / "analysis"
    if out.exists() and any(out.iterdir()):
        previous = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if previous != result:
            raise ValueError("Refusing to overwrite changed revised-prior analysis")
        return previous
    out.mkdir(parents=True, exist_ok=True)
    score_rows = []
    maturity_rows = []
    for horizon, horizon_result in result["horizons"].items():
        for metric, values in horizon_result["main"]["metrics"].items():
            score_rows.append({
                "horizon": int(horizon),
                "metric": metric,
                "M3": values["M4"],
                "M4": values["M5"],
                "M4_minus_M3": values["M5_minus_M4"],
                "ci95_lower": values["ci95"][0],
                "ci95_upper": values["ci95"][1],
                "bootstrap_se": values["bootstrap_se"],
            })
        for item in horizon_result["per_maturity"]:
            for model, publication_model in [("M4", "M3"), ("M5", "M4")]:
                maturity_rows.append({
                    "horizon": int(horizon),
                    "maturity_years": item["maturity_years"],
                    "model": publication_model,
                    **item[model],
                })
    write_csv(out / "paired_summary.csv", score_rows)
    write_csv(out / "per_maturity.csv", maturity_rows)
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["representative", "rolling", "analyze"], required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "representative":
        result = run_representative_phase(args.output, args.workers)
        print(json.dumps({
            "accepted": result["accepted"],
            "max_rhat": result["max_rhat"],
            "min_bulk_ess": result["min_bulk_ess"],
            "min_tail_ess": result["min_tail_ess"],
        }))
    elif args.phase == "rolling":
        result = run_experiment(args.output, args.workers)
        print(json.dumps({
            "accepted": result["accepted"],
            "planned_origins": result["planned_origins"],
            "completed_origins": result["completed_origins"],
            "failed_origins": result["failed_origins"],
        }))
    else:
        result = analyze_experiment(args.output)
        print(json.dumps({
            "accepted_main_analysis": result["accepted_main_analysis"],
            "n_origins": result["n_origins"],
        }))


if __name__ == "__main__":
    main()
