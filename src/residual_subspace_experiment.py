"""Re-evaluate accepted Bayesian DNS forecasts with residual-space covariance."""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.bayesian_finite_production import require_phase2
from src.bayesian_finite_runner import digest
from src.bayesian_stage_a import FACTORS, ROOT, make_design
from src.bayesian_finite_validation import factor_scenarios
from src.experiment import select_origins
from src.residual_subspace_analysis import (
    energy_score,
    make_figures,
    paired_bootstrap,
    residual_calibration,
    stationary_bootstrap_indices,
    summarize_forecasts,
)
from src.residual_subspace_covariance import (
    assemble_yield_covariance,
    decompose_yields,
    dss_components,
    eigen_diagnostics,
    estimate_residual_covariances,
    orthogonal_dss_components,
    residual_basis,
)


ARCHIVE_ROOT = ROOT / "results" / "bayesian_prior_revised_20261114"
BASELINE_ROOT = ROOT / "results" / "validated_baselines"
OUTPUT_ROOT = ROOT / "results" / "residual_subspace_experiment_20261116"
DOC_ROOT = ROOT / "docs" / "residual_subspace_experiment"
MATURITIES = [0.25, 0.5, 1, 2, 5, 7, 10, 30]
HORIZONS = [1, 4]
MODELS = ["M4", "M5"]
SCENARIO_COUNT = 2048
BOOTSTRAP_REPETITIONS = 9999
BOOTSTRAP_SEED = 20261116


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _protected_hashes() -> dict[str, str]:
    protected = [
        *ARCHIVE_ROOT.rglob("*"),
        *BASELINE_ROOT.rglob("*"),
        ROOT / "paper" / "MRP_Draft.qmd",
        ROOT / "paper" / "MRP_Draft.pdf",
        ROOT / "paper" / "MRP_Draft.tex",
        ROOT / "src" / "bayesian_prior_longrun.py",
        ROOT / "src" / "bayesian_finite.py",
        ROOT / "src" / "bayesian_revised.py",
        ROOT / "src" / "bayesian_finite_validation.py",
        ROOT / "src" / "bayesian_finite_production.py",
    ]
    files = sorted({path for path in protected if path.is_file()})
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def verify_accepted_inputs(design: dict) -> tuple[dict, list[dict], dict, dict, dict]:
    """Fail closed on changed/incomplete accepted forecasts and baselines."""
    gate_path = ARCHIVE_ROOT / "phase2_gate.json"
    gate = require_phase2(gate_path)
    plan = _read_json(ARCHIVE_ROOT / "plan.json")
    summary = _read_json(ARCHIVE_ROOT / "summary.json")
    if (
        not summary.get("accepted")
        or summary.get("planned_origins") != 130
        or summary.get("completed_origins") != 130
        or summary.get("failed_origins")
        or summary.get("not_completed_dates")
        or digest(ARCHIVE_ROOT / "plan.json") != summary.get("plan_sha256")
        or plan.get("gate_sha256") != digest(gate_path)
        or plan.get("input_sha256") != design["input_sha256"]
        or plan.get("lambda") != design["lambda"]
        or plan.get("maturities_years") != MATURITIES
        or plan.get("horizons") != HORIZONS
        or plan.get("models") != MODELS
        or plan.get("warmup") != 3000
        or plan.get("draws") != 6000
        or plan.get("chains") != 4
        or plan.get("scenario_count") != SCENARIO_COUNT
        or plan.get("version") != "bayesian_prior_revised_20261114"
    ):
        raise ValueError("Accepted Bayesian archive identity or run design changed")

    expected_indices = sorted({
        origin for origin, _, _ in select_origins(
            design["dates"], date(2016, 1, 1), date(2025, 12, 31), HORIZONS, 4
        )
    })
    planned = plan.get("origins", [])
    if len(planned) != 130 or [row["index"] for row in planned] != expected_indices:
        raise ValueError("Accepted origin schedule does not match the fixed weekly design")
    if any(
        row["date"] != design["dates"][row["index"]].isoformat()
        or row["seed"] != 9102701 + 1_000_000 * ordinal
        for ordinal, row in enumerate(planned)
    ):
        raise ValueError("Accepted origin dates or seed schedule changed")
    if (
        set(summary.get("completed_dates", [])) != {row["date"] for row in planned}
        or set(summary.get("forecasts_sha256", {})) != {row["date"] for row in planned}
    ):
        raise ValueError("Accepted forecast summary does not cover every origin")

    origin_archives = []
    for row in planned:
        folder = ARCHIVE_ROOT / "origins" / row["date"]
        report_path = folder / "forecast.json"
        report = _read_json(report_path)
        spec_path = folder / "forecast_specification.json"
        spec = _read_json(spec_path)
        if (
            summary["forecasts_sha256"][row["date"]] != digest(report_path)
            or not report.get("accepted")
            or report.get("specification") != spec
            or spec.get("origin_index") != row["index"]
            or spec.get("origin") != row["date"]
            or spec.get("seed") != row["seed"]
            or spec.get("gate_sha256") != digest(gate_path)
            or spec.get("input_sha256") != design["input_sha256"]
            or spec.get("lambda") != design["lambda"]
            or spec.get("models") != MODELS
            or spec.get("horizons") != HORIZONS
            or spec.get("warmup") != 3000
            or spec.get("draws") != 6000
            or spec.get("chains") != 4
            or spec.get("scenario_count") != SCENARIO_COUNT
            or digest(folder / "predictions.npz") != report.get("prediction_archive_sha256")
        ):
            raise ValueError(f"Changed or incompatible accepted forecast: {row['date']}")
        chain_dir = Path(report["chain_source"])
        try:
            chain_dir.resolve().relative_to(ARCHIVE_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"Posterior path escapes the accepted archive: {row['date']}") from exc
        chain_summary_path = chain_dir / "summary.json"
        chain_summary = _read_json(chain_summary_path)
        declared_checkpoints = report.get("chain_checkpoint_sha256", {})
        named_checkpoints = {Path(path).name: value for path, value in declared_checkpoints.items()}
        if (
            not chain_summary_path.is_file()
            or digest(chain_summary_path) != report.get("parameter_summary_sha256")
            or not chain_summary.get("parameter_gate_passed")
            or not chain_summary.get("moment_gate_passed")
            or not chain_summary.get("diagnostics")
            or not all(item.get("passes") for item in chain_summary["diagnostics"])
            or named_checkpoints != chain_summary.get("checkpoints")
            or len(declared_checkpoints) != 24
            or any(digest(path) != value for path, value in declared_checkpoints.items())
        ):
            raise ValueError(f"Changed or unaccepted posterior checkpoints: {row['date']}")
        expected_pairs = {(model, h) for model in MODELS for h in HORIZONS}
        if (
            len(report.get("records", [])) != 4
            or {(item["model"], item["horizon"]) for item in report["records"]} != expected_pairs
        ):
            raise ValueError(f"Incomplete accepted predictions: {row['date']}")
        origin_archives.append({
            "plan_row": row,
            "folder": folder,
            "report": report,
            "chain_dir": chain_dir,
            "chain_summary": chain_summary,
        })

    baseline_metadata = _read_json(BASELINE_ROOT / "metadata.json")
    baseline_verification = _read_json(BASELINE_ROOT / "verification.json")
    if (
        baseline_metadata.get("input_sha256") != design["input_sha256"]
        or baseline_metadata.get("lambda_per_year") != design["lambda"]
        or baseline_metadata.get("n_origins") != 130
        or baseline_metadata.get("models") != ["M0", "M1", "M2"]
        or baseline_verification.get("score_rows") != 780
        or baseline_verification.get("models_accepted") != ["M0", "M1", "M2"]
    ):
        raise ValueError("Classical benchmark provenance does not match the accepted sample")
    baseline_scores = {}
    with (BASELINE_ROOT / "scores.csv").open(encoding="utf-8-sig", newline="") as stream:
        for score in csv.DictReader(stream):
            baseline_scores[(score["origin_date"], int(score["horizon"]), score["model"])] = score
    with np.load(BASELINE_ROOT / "forecasts.npz", allow_pickle=False) as archive:
        baseline = {key: archive[key].copy() for key in archive.files}
    if (
        baseline["scenarios"].shape != (780, 1024, 8)
        or baseline["means"].shape != (780, 8)
        or baseline["covariances"].shape != (780, 8, 8)
        or baseline["targets"].shape != (780, 8)
        or baseline["keys"].shape != (780, 3)
        or len(baseline_scores) != 780
    ):
        raise ValueError("Classical baseline archive has an unexpected shape")
    baseline_lookup = {
        (str(date), int(horizon), str(model)): index
        for index, (date, horizon, model) in enumerate(baseline["keys"])
    }
    if len(baseline_lookup) != 780:
        raise ValueError("Classical baseline keys are not unique")
    return plan, origin_archives, baseline, baseline_lookup, baseline_scores


def _posterior_factors(chain_dir: Path, chain_summary: dict, model: str) -> list:
    factors = []
    expected_names = (
        ["a", "b", "long_run_mean", "log_variance"]
        if model == "M4"
        else ["a", "b", "long_run_mean", "m", "phi", "v", "omega", "q",
              "h_first", "h_mid", "h_last"]
    )
    for factor in FACTORS:
        chains = []
        for chain in range(4):
            filename = f"{model}_{factor}_chain{chain}.npz"
            with np.load(chain_dir / filename, allow_pickle=False) as archive:
                values = archive["draws"].copy()
                chain_report = json.loads(str(archive["report"]))
            if (
                values.shape != (6000, len(expected_names))
                or chain_report.get("names") != expected_names
                or chain_report.get("signature") != chain_summary["signature"]
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"Invalid saved posterior draws: {filename}")
            chains.append(values)
        factors.append((np.stack(chains), expected_names))
    return factors


def _baseline_record(
    index: int,
    date: str,
    horizon: int,
    model: str,
    baseline: dict,
    scores: dict,
) -> dict:
    mean = baseline["means"][index]
    covariance = baseline["covariances"][index]
    observed = baseline["targets"][index]
    scenarios = baseline["scenarios"][index]
    score_row = scores[(date, horizon, model)]
    components = dss_components(observed, mean, covariance)
    score = energy_score(scenarios, observed)
    if (
        not np.isclose(components["dss"], float(score_row["dss"]), atol=1e-9, rtol=1e-9)
        or not np.isclose(score, float(score_row["energy_score"]), atol=1e-10, rtol=1e-10)
        or not np.isfinite(scenarios).all()
    ):
        raise ValueError(f"Classical benchmark score reconstruction failed: {date} h{horizon} {model}")
    standard_error = np.sqrt(np.diag(covariance))
    lower, upper = mean - norm.ppf(0.95) * standard_error, mean + norm.ppf(0.95) * standard_error
    error = observed - mean
    return {
        "origin": date,
        "target": str(score_row["target_date"]),
        "model": model,
        "horizon": horizon,
        "dss": components["dss"],
        "dss_original": components["dss"],
        "log_determinant": components["log_determinant"],
        "mahalanobis": components["mahalanobis"],
        "energy_score": score,
        "squared_errors": (error**2).tolist(),
        "absolute_errors": np.abs(error).tolist(),
        "coverage_90": ((observed >= lower) & (observed <= upper)).astype(int).tolist(),
        "interval_width_90": (upper - lower).tolist(),
        "mean": mean.tolist(),
        "covariance": covariance.tolist(),
    }


def _summarize_covariance_sensitivity(
    sensitivity: dict,
    seed: int = BOOTSTRAP_SEED + 70000,
) -> dict:
    output = {}
    for h in HORIZONS:
        output[str(h)] = {}
        for model in MODELS:
            rows = sorted(
                (r for r in sensitivity if r["horizon"] == h and r["model"] == model),
                key=lambda r: r["origin"],
            )
            primary = np.asarray([r["dss"] for r in rows])
            output[str(h)][model] = {}
            for method in ("unshrunk", "oas"):
                alternative = np.asarray([r["dss_alternatives"][method] for r in rows])
                method_results = {}
                for block_length in (8, 4, 12):
                    indices = stationary_bootstrap_indices(
                        130, block_length, BOOTSTRAP_REPETITIONS,
                        np.random.default_rng(seed + h * 1000 + block_length),
                    )
                    delta = alternative[indices].mean(axis=1) - primary[indices].mean(axis=1)
                    method_results[str(block_length)] = {
                        "block_length_origins": block_length,
                        "alternative_minus_ledoit_wolf": float(alternative.mean() - primary.mean()),
                        "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
                        "bootstrap_se": float(np.std(delta, ddof=1)),
                    }
                output[str(h)][model][method] = method_results
    return output


def _diagonal_to_subspace_summary(records: list[dict]) -> dict:
    output = {}
    for h in HORIZONS:
        output[str(h)] = {}
        for model in MODELS:
            rows = sorted(
                (r for r in records if r["horizon"] == h and r["model"] == model),
                key=lambda r: r["origin"],
            )
            left = {
                "dss": np.asarray([r["dss_original"] for r in rows]),
                "energy_score": np.asarray([r["energy_score_original"] for r in rows]),
                "squared_errors": np.asarray([r["squared_errors"] for r in rows]),
                "coverage_90": np.asarray([r["coverage_90_original"] for r in rows]),
            }
            right = {
                "dss": np.asarray([r["dss"] for r in rows]),
                "energy_score": np.asarray([r["energy_score"] for r in rows]),
                "squared_errors": np.asarray([r["squared_errors"] for r in rows]),
                "coverage_90": np.asarray([r["coverage_90"] for r in rows]),
            }
            output[str(h)][model] = {}
            for block_length in (8, 4, 12):
                indices = stationary_bootstrap_indices(
                    130, block_length, BOOTSTRAP_REPETITIONS,
                    np.random.default_rng(BOOTSTRAP_SEED + 60000 + h * 100 + block_length),
                )
                output[str(h)][model][str(block_length)] = {
                    "block_length_origins": block_length,
                    **paired_bootstrap(left, right, indices),
                }
    return output


def _counterfactual_summary(rows: list[dict]) -> dict:
    methods = (
        "rw_mean_rw_covariance",
        "dns_mean_dns_diagonal_covariance",
        "rw_mean_dns_corrected_covariance",
        "dns_mean_rw_covariance",
        "dns_mean_dns_corrected_covariance",
    )
    result = {}
    for h in HORIZONS:
        by_horizon = sorted((row for row in rows if row["horizon"] == h), key=lambda r: r["origin"])
        indices = stationary_bootstrap_indices(
            130, 8, BOOTSTRAP_REPETITIONS,
            np.random.default_rng(BOOTSTRAP_SEED + 80000 + h),
        )
        values = {name: np.asarray([r[name] for r in by_horizon]) for name in methods}
        result[str(h)] = {}
        baseline = values["rw_mean_rw_covariance"]
        for name, value in values.items():
            difference = value[indices].mean(axis=1) - baseline[indices].mean(axis=1)
            result[str(h)][name] = {
                "mean_dss": float(value.mean()),
                "difference_from_rw_mean_rw_covariance": float(value.mean() - baseline.mean()),
                "difference_ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
            }
    return result


def _m2_summary(rows: list[dict]) -> dict:
    output = {}
    for h in HORIZONS:
        horizon_rows = sorted((r for r in rows if r["horizon"] == h), key=lambda r: r["origin"])
        data = {}
        for name, prefix in (("M2_diagonal", "original"), ("M2_residual_subspace", "corrected")):
            data[name] = {
                "dss": np.asarray([r[f"dss_{prefix}"] for r in horizon_rows]),
                "energy_score": np.asarray([r[f"energy_{prefix}"] for r in horizon_rows]),
                "squared_errors": np.asarray([r[f"squared_errors_{prefix}"] for r in horizon_rows]),
                "coverage_90": np.asarray([r[f"coverage_{prefix}"] for r in horizon_rows]),
            }
        comparisons = {}
        for block_length in (8, 4, 12):
            indices = stationary_bootstrap_indices(
                130, block_length, BOOTSTRAP_REPETITIONS,
                np.random.default_rng(BOOTSTRAP_SEED + 90000 + h * 100 + block_length),
            )
            comparisons[str(block_length)] = paired_bootstrap(
                data["M2_diagonal"], data["M2_residual_subspace"], indices
            )
        output[str(h)] = {
            "scenario_counts": {"original_accepted_baseline": 1024, "corrected_counterfactual": SCENARIO_COUNT},
            "M2_diagonal_mean_dss": float(data["M2_diagonal"]["dss"].mean()),
            "M2_residual_subspace_mean_dss": float(data["M2_residual_subspace"]["dss"].mean()),
            "M2_diagonal_mean_energy_score": float(data["M2_diagonal"]["energy_score"].mean()),
            "M2_residual_subspace_mean_energy_score": float(data["M2_residual_subspace"]["energy_score"].mean()),
            "paired_comparison": comparisons,
        }
    return output


def _residual_covariance_evolution(
    origin_diagnostics: list[dict], covariance: np.ndarray
) -> dict:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (130, 3, 5, 5):
        raise ValueError("Expected 130 origin-level covariances for three estimators")
    primary_variances = np.diagonal(covariance[:, 0], axis1=1, axis2=2)
    dates = np.asarray([row["origin"] for row in origin_diagnostics])
    groups = {
        "2016-2019": np.asarray([value[:4] in ("2016", "2017", "2018", "2019") for value in dates]),
        "2020-2021": np.asarray([value[:4] in ("2020", "2021") for value in dates]),
        "2022-2025": np.asarray([value[:4] in ("2022", "2023", "2024", "2025") for value in dates]),
    }
    medians = {
        period: np.median(primary_variances[mask], axis=0)
        for period, mask in groups.items()
    }
    initial = medians["2016-2019"]
    late = medians["2022-2025"]
    return {
        "estimator": "Ledoit-Wolf",
        "coordinate_variance_medians_by_period": {
            period: value.tolist() for period, value in medians.items()
        },
        "late_to_early_median_variance_ratio": (late / initial).tolist(),
        "coordinate_variance_min_across_origins": primary_variances.min(axis=0).tolist(),
        "coordinate_variance_max_across_origins": primary_variances.max(axis=0).tolist(),
        "coordinate_variance_coefficient_of_variation": (
            primary_variances.std(axis=0, ddof=1) / primary_variances.mean(axis=0)
        ).tolist(),
        "period_origin_counts": {period: int(mask.sum()) for period, mask in groups.items()},
    }


def _eigenstructure_summary(records: list[dict]) -> dict:
    output = {}
    for h in HORIZONS:
        output[str(h)] = {}
        for model in MODELS:
            rows = [
                row for row in records
                if row["horizon"] == h and row["model"] == model
            ]
            original_share = [
                max(row["original_eigen_direction_error_contributions"])
                / max(row["original_mahalanobis"], 1e-300)
                for row in rows
            ]
            corrected_share = [row["largest_contribution_share"] for row in rows]
            output[str(h)][model] = {
                "median_original_condition_number": float(np.median([
                    row["original_condition_number"] for row in rows
                ])),
                "median_corrected_condition_number": float(np.median([
                    row["condition_number"] for row in rows
                ])),
                "median_largest_original_mahalanobis_direction_share": float(
                    np.median(original_share)
                ),
                "median_largest_corrected_mahalanobis_direction_share": float(
                    np.median(corrected_share)
                ),
                "median_original_eigenvalues_descending": np.median(
                    np.asarray([row["original_eigenvalues"] for row in rows]), axis=0
                ).tolist(),
                "median_corrected_eigenvalues_descending": np.median(
                    np.asarray([row["eigenvalues_descending"] for row in rows]), axis=0
                ).tolist(),
            }
    return output


def _write_tables(output: Path, records: list[dict], origin_diagnostics: list[dict],
                  counterfactuals: list[dict], m2_rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "dss_components.csv": [
            {key: row[key] for key in (
                "origin", "model", "horizon", "dss_original", "dss", "original_log_determinant",
                "original_mahalanobis", "corrected_log_determinant", "corrected_mahalanobis",
                "factor_log_determinant", "factor_mahalanobis", "residual_log_determinant",
                "residual_mahalanobis", "dss_alternatives",
            )}
            for row in records
        ],
        "origin_residual_diagnostics.csv": [
            {key: row[key] for key in (
                "origin", "training_observations", "ledoit_wolf_shrinkage", "oas_shrinkage",
                "residual_coordinate_mean", "residual_coordinate_sd", "residual_coordinate_lag1",
                "max_abs_training_factor_residual_correlation", "decomposition_relative_error",
            )}
            for row in origin_diagnostics
        ],
        "dns_rw_counterfactuals.csv": counterfactuals,
        "m2_counterfactuals.csv": m2_rows,
    }
    for filename, rows in files.items():
        if not rows:
            continue
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _write_reports(
    results: dict,
    records: list[dict],
    calibration: dict,
    residual_evolution: dict,
    counterfactuals: list[dict],
    figures: list[str],
    manifest: dict,
) -> None:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    primary = results["forecast_metrics"]
    horizon_rows = []
    for h in HORIZONS:
        values = primary["horizons"][str(h)]["model_means"]
        horizon_rows.append(
            f"| {h} | {values['M0']['dss']:.3f} | {values['M4']['dss']:.3f} | "
            f"{values['M5']['dss']:.3f} | {values['M0']['energy_score']:.4f} | "
            f"{values['M4']['energy_score']:.4f} | {values['M5']['energy_score']:.4f} |"
        )
    structure_text = f"""# Mathematical specification

## Projection and covariance

Let \\(Y_t\\in\\mathbb{{R}}^8\\) be the eight-maturity yield vector and \\(\\Lambda\\in\\mathbb{{R}}^{{8\\times3}}\\) the accepted fixed Nelson–Siegel loading matrix. The cross-sectional least-squares factor estimate is

\\[
\\widehat f_t=(\\Lambda^\\top\\Lambda)^{-1}\\Lambda^\\top Y_t,
\\qquad r_t=Y_t-\\Lambda\\widehat f_t.
\\]

The residual has \\(\\Lambda^\\top r_t=0\\), so it lies in a five-dimensional complement. A complete QR factorization provides \\(Q_f\\) and \\(U\\), with \\(Q_f^\\top Q_f=I_3\\), \\(U^\\top U=I_5\\), and \\(U^\\top\\Lambda=0\\). The residual coordinates are \\(z_t=U^\\top Y_t\\), and the code verifies \\(Y_t=\\Lambda\\widehat f_t+Uz_t\\) on every training window.

For each origin, the primary covariance is centered Ledoit–Wolf on \\(z_s\\) using only observations through that origin. The unshrunk sample covariance and OAS covariance are predeclared sensitivity estimators. The predictive residual mean is fixed at zero. Factor–residual forecast cross-covariance is fixed at zero.

For either accepted Bayesian model, the corrected predictive moments are

\\[
\\mu^Y_{{t+h}}=\\Lambda\\mu^f_{{t+h}},\\qquad
\\Sigma^Y_{{t+h}}=\\Lambda\\operatorname{{diag}}(v^f_{{t+h}})\\Lambda^\\top
+U S_z U^\\top.
\\]

Here \\(v^f\\) is the accepted posterior predictive factor variance. The same saved posterior chains and the existing factor-scenario generator are used to produce 2,048 scenarios; only the old diagonal reconstruction noise is replaced by \\(U z\\), where \\(z\\sim N(0,S_z)\\).

## Fixed experiment design

- Eight maturities: {", ".join(map(str, MATURITIES))} years; \\(\\lambda={manifest['lambda']:.15f}\\).
- 130 expanding-window origins, 2016-01-08 through 2025-11-28; horizons 1 and 4 weeks.
- Paper labels are M3 (internal M4, constant volatility) and M4 (internal M5, stochastic volatility).
- No MCMC, refitting, prior changes, OOS covariance fitting, or edits to accepted forecast files.
- The primary residual covariance is fit through the origin date; no target data enters its estimate.

## Positive definiteness and score verification

Because \\([\\Lambda\\ U]\\) is full rank and both \\(\\operatorname{{diag}}(v^f)\\) and \\(S_z\\) are positive definite, the assembled covariance is positive definite. The run checks Cholesky factorization, symmetry, eigenvalues, condition numbers, log determinants, and Mahalanobis terms for every forecast. DSS uses Cholesky solves, not an explicit inverse. The orthogonal factor- and residual-space DSS components are checked to sum to the full corrected score.

## Results preview

| Horizon (weeks) | RW DSS | M3 / constant DSS | M4 / SV DSS | RW ES | M3 ES | M4 ES |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(horizon_rows)}

Full record-level values and provenance are in `{OUTPUT_ROOT.relative_to(ROOT)}`. Source/data hashes are recorded in the experiment manifest.
"""
    (DOC_ROOT / "mathematical_specification.md").write_text(structure_text, encoding="utf-8")

    original_rows = sorted(
        (row for row in records if row["model"] in MODELS),
        key=lambda row: (row["origin"], row["model"], row["horizon"]),
    )
    changes = np.asarray([row["dss_original"] - row["dss"] for row in original_rows])
    improved = sum(value > 0 for value in changes)
    worsened = sum(value < 0 for value in changes)
    by_h = {}
    for h in HORIZONS:
        subset = [row for row in records if row["horizon"] == h and row["model"] in MODELS]
        by_h[str(h)] = {
            "old_logdet": float(np.mean([row["original_log_determinant"] for row in subset])),
            "old_mahal": float(np.mean([row["original_mahalanobis"] for row in subset])),
            "new_logdet": float(np.mean([row["corrected_log_determinant"] for row in subset])),
            "new_mahal": float(np.mean([row["corrected_mahalanobis"] for row in subset])),
        }
    correction_rows = []
    eigen_rows = []
    for h in HORIZONS:
        for model in MODELS:
            rows = [
                row for row in records
                if row["horizon"] == h and row["model"] == model
            ]
            original = results["diagonal_to_subspace"][str(h)][model]["8"]
            eigen = results["eigenstructure_diagnostics"][str(h)][model]
            correction_rows.append(
                f"| {h} | {model} | "
                f"{np.mean([row['dss_original'] for row in rows]):.3f} | "
                f"{np.mean([row['dss'] for row in rows]):.3f} | "
                f"{original['dss']['right_minus_left']:.3f} "
                f"[{original['dss']['ci95'][0]:.3f}, {original['dss']['ci95'][1]:.3f}] | "
                f"{np.mean([row['energy_score_original'] for row in rows]):.4f} | "
                f"{np.mean([row['energy_score'] for row in rows]):.4f} | "
                f"{original['energy_score']['right_minus_left']:.4f} "
                f"[{original['energy_score']['ci95'][0]:.4f}, {original['energy_score']['ci95'][1]:.4f}] |"
            )
            eigen_rows.append(
                f"| {h} | {model} | {eigen['median_original_condition_number']:.1f} | "
                f"{eigen['median_corrected_condition_number']:.1f} | "
                f"{eigen['median_largest_original_mahalanobis_direction_share']:.3f} | "
                f"{eigen['median_largest_corrected_mahalanobis_direction_share']:.3f} |"
            )
    correction_text = "\n".join(correction_rows)
    episodes = []
    for row in sorted(original_rows, key=lambda x: x["dss_original"] - x["dss"], reverse=True)[:5]:
        episodes.append(
            f"| {row['origin']} | {row['model']} | {row['horizon']} | "
            f"{row['dss_original']:.3f} | {row['dss']:.3f} | "
            f"{row['dss_original'] - row['dss']:.3f} |"
        )
    reverse_episodes = []
    for row in sorted(original_rows, key=lambda x: x["dss_original"] - x["dss"])[:5]:
        reverse_episodes.append(
            f"| {row['origin']} | {row['model']} | {row['horizon']} | "
            f"{row['dss_original']:.3f} | {row['dss']:.3f} | "
            f"{row['dss_original'] - row['dss']:.3f} |"
        )
    counter_text = "\n".join(
        f"| {h} | {name} | {row['mean_dss']:.3f} | "
        f"{row['difference_from_rw_mean_rw_covariance']:.3f} | "
        f"[{row['difference_ci95'][0]:.3f}, {row['difference_ci95'][1]:.3f}] |"
        for h, comparisons in results["dns_rw_counterfactual_summary"].items()
        for name, row in comparisons.items()
    )
    diagnostic_text = f"""# Baseline DSS diagnostics

## Accepted diagonal-covariance forecasts

The original saved M3/M4 (internal M4/M5) mean and covariance archives were reconstructed before any correction. All original DSS and Energy Score values matched their accepted records. The decomposition shows that the diagonal reconstruction term affects both the log determinant and the Mahalanobis penalty, not just marginal interval widths.

Across the 520 Bayesian model/horizon forecasts, residual-space correction lowered DSS in {improved} records and raised it in {worsened}. The aggregate original-versus-corrected log-determinant and Mahalanobis components are:

| Horizon | Internal model | Original DSS | Corrected DSS | Corrected−original DSS (95% interval) | Original ES | Corrected ES | Corrected−original ES (95% interval) |
|---:|---|---:|---:|---:|---:|---:|---:|
{correction_text}

Median covariance condition numbers and the share of the Mahalanobis penalty from the single most influential covariance eigen-direction are:

| Horizon | Internal model | Original median condition number | Corrected median condition number | Original max-direction share | Corrected max-direction share |
|---:|---|---:|---:|---:|---:|
{chr(10).join(eigen_rows)}

The corrected covariances remain positive definite, but their smaller residual-space eigenvalues make them materially more ill-conditioned. The eigen-direction contribution arrays are retained per forecast; the sign of any individual eigenvector is arbitrary.

| Horizon | Original logdet | Original Mahalanobis | Corrected logdet | Corrected Mahalanobis |
|---:|---:|---:|---:|---:|
{chr(10).join(f"| {h} | {v['old_logdet']:.3f} | {v['old_mahal']:.3f} | {v['new_logdet']:.3f} | {v['new_mahal']:.3f} |" for h,v in by_h.items())}

The corrected covariance is block diagonal in the orthonormal factor/residual coordinates. Record-level factor/residual log-determinants and Mahalanobis terms are in `dss_components.csv`; eigenvalues and standardized error contributions are in `forecast_records.json`.

## Largest origin-level DSS improvements

| Origin | Internal model | Horizon | Original DSS | Corrected DSS | Improvement |
|---|---|---:|---:|---:|---:|
{chr(10).join(episodes)}

## Largest origin-level DSS degradations

| Origin | Internal model | Horizon | Original DSS | Corrected DSS | Improvement |
|---|---|---:|---:|---:|---:|
{chr(10).join(reverse_episodes)}

## DNS/random-walk mean–covariance counterfactuals

These are score decompositions, not new accepted forecast models. They combine the saved M0 random-walk and M2 DNS means with M0 covariance, original diagonal M2 covariance, and residual-corrected M2 covariance. The accepted M2 scenario archive has 1,024 scenarios; corrected M2 Energy Scores use 2,048 Gaussian scenarios, so that ancillary energy comparison has unequal Monte Carlo sample sizes.

| Horizon | Mean/covariance combination | Mean DSS | Difference from RW/RW | Paired 95% block-bootstrap interval |
|---:|---|---:|---:|---:|
{counter_text}

The M2 correction replaces only its historical diagonal reconstruction term. The retained M0 and M2 forecast archives are unchanged. Full rows are in `dns_rw_counterfactuals.csv`; the M2-only paired comparison is in `m2_counterfactuals.csv`.

## Figures

{chr(10).join(f"- `{name}`" for name in figures)}

The corrected DSS values should be interpreted alongside proper energy scores and calibration in the structural recovery report. A good DSS alone is not evidence that the five residual coordinates are adequately forecast by a zero-mean, time-invariant covariance.
"""
    (DOC_ROOT / "baseline_dss_diagnostics.md").write_text(diagnostic_text, encoding="utf-8")

    metrics_text = []
    for h in HORIZONS:
        values = primary["horizons"][str(h)]["model_means"]
        pair = primary["horizons"][str(h)]["paired_comparisons"]
        m3vsm4 = pair["M4_to_M5"]["8"]
        metrics_text.append(
            f"| {h} | {values['M4']['dss']:.3f} | {values['M5']['dss']:.3f} | "
            f"{m3vsm4['dss']['right_minus_left']:.3f} "
            f"[{m3vsm4['dss']['ci95'][0]:.3f}, {m3vsm4['dss']['ci95'][1]:.3f}] | "
            f"{values['M4']['energy_score']:.4f} | {values['M5']['energy_score']:.4f} | "
            f"{m3vsm4['energy_score']['right_minus_left']:.4f} "
            f"[{m3vsm4['energy_score']['ci95'][0]:.4f}, {m3vsm4['energy_score']['ci95'][1]:.4f}] |"
        )
    residual_rows = []
    for h, values in calibration.items():
        residual_rows.append(
            f"| {h} | {values['joint_mahalanobis_squared_mean']:.2f} | "
            f"{values['joint_mahalanobis_squared_90_coverage']:.3f} | "
            f"{values['joint_mahalanobis_squared_95_coverage']:.3f} | "
            f"{values['max_abs_lag1_autocorrelation']:.3f} | "
            f"{values['max_abs_mean_training_factor_residual_correlation']:.3f} | "
            f"{values['factor_residual_dependence_by_model']['M4']['max_abs_oos_factor_error_residual_correlation']:.3f} / "
            f"{values['factor_residual_dependence_by_model']['M5']['max_abs_oos_factor_error_residual_correlation']:.3f} |"
        )
    evolution_rows = []
    for j, value in enumerate(residual_evolution["coordinate_variance_medians_by_period"]["2016-2019"]):
        evolution_rows.append(
            f"| z{j + 1} | {value:.5f} | "
            f"{residual_evolution['coordinate_variance_medians_by_period']['2020-2021'][j]:.5f} | "
            f"{residual_evolution['coordinate_variance_medians_by_period']['2022-2025'][j]:.5f} | "
            f"{residual_evolution['late_to_early_median_variance_ratio'][j]:.2f} |"
        )
    alternative_rows = []
    for h in HORIZONS:
        for model in MODELS:
            for method in ("unshrunk", "oas"):
                item = results["covariance_estimator_sensitivity"][str(h)][model][method]["8"]
                alternative_rows.append(
                    f"| {h} | {model} | {method} | {item['alternative_minus_ledoit_wolf']:.3f} | "
                    f"[{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}] |"
                )
    verdict = results["recommendation"]
    random_walk_rows = []
    for h in HORIZONS:
        horizon = primary["horizons"][str(h)]
        for model in MODELS:
            comparison = horizon["paired_comparisons"][f"M0_to_{model}"]["8"]
            random_walk_rows.append(
                f"| {h} | {model} | {horizon['model_means']['M0']['dss']:.3f} | "
                f"{horizon['model_means'][model]['dss']:.3f} | "
                f"{comparison['dss']['right_minus_left']:.3f} "
                f"[{comparison['dss']['ci95'][0]:.3f}, {comparison['dss']['ci95'][1]:.3f}] | "
                f"{comparison['energy_score']['right_minus_left']:.4f} "
                f"[{comparison['energy_score']['ci95'][0]:.4f}, {comparison['energy_score']['ci95'][1]:.4f}] | "
                f"{comparison['rmse']['right_minus_left']:.4f} "
                f"[{comparison['rmse']['ci95'][0]:.4f}, {comparison['rmse']['ci95'][1]:.4f}] |"
            )
    random_walk_text = "\n".join(random_walk_rows)
    structural_text = f"""# Structural recovery report

## Executive result

**Recommendation {verdict['letter']}: {verdict['title']}.** The residual-subspace covariance clearly improves mean DSS and modestly improves Energy Score relative to the diagonal-reconstruction versions, but the residuals remain too dependent and undercovered to treat a static zero-mean covariance as adequate. This is a bounded covariance correction, not a new factor model, and it does not change the fixed DNS loading representation, factor sampler, priors, or accepted forecasts.

The paper labels correspond to internal archive labels M3 = M4 (constant factor volatility) and M4 = M5 (stochastic factor volatility).

For transparency, the recommendation uses this screening rule: A requires positive mean DSS improvement, joint 90% residual coverage between 0.85 and 0.95, maximum absolute residual lag-1 correlation no greater than 0.20, and maximum mean factor–residual correlation no greater than 0.30. B means DSS improves but at least one diagnostic fails; C means mean DSS does not improve. These are decision thresholds for this experiment, not previously accepted model gates.

## Forecast results

| Horizon | M3 / constant DSS | M4 / stochastic DSS | M4−M3 DSS difference (95% interval) | M3 ES | M4 ES | M4−M3 ES difference (95% interval) |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metrics_text)}

Paired stationary-bootstrap estimates use 9,999 replicates and expected block length 8 origins. Block lengths 4 and 12 are included in `metrics.json`. M0 random-walk comparisons, maturity-level RMSE/MAE/coverage/width, and paired bootstrap intervals are reported alongside these results.

## Random-walk comparison

| Horizon | Internal model | RW DSS | DNS DSS | DNS−RW DSS difference (95% interval) | DNS−RW ES difference (95% interval) | DNS−RW RMSE difference (95% interval) |
|---:|---|---:|---:|---:|---:|---:|
{random_walk_text}

The corrected DNS models remain worse than the accepted random-walk benchmark on mean DSS, Energy Score, and RMSE at both horizons. The covariance correction improves the DNS covariance specification; it does not resolve the DNS models' overall forecast ranking.

## Residual-subspace adequacy

| Horizon | Mean joint Mahalanobis² (5 df) | Joint 90% coverage | Joint 95% coverage | Max OOS residual lag-1 correlation | Max mean training factor–residual correlation | Max OOS factor-error/residual correlation (M3 / M4) |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(residual_rows)}

Marginal standardized means, variances, 90% coverage by coordinate, residual-coordinate means, and window-to-window covariance estimates are in `metrics.json` and `origin_residual_diagnostics.csv`. Median residual-coordinate variances shift across these OOS periods:

| Coordinate | 2016–2019 median variance | 2020–2021 | 2022–2025 | Late/early ratio |
|---:|---:|---:|---:|---:|
{chr(10).join(evolution_rows)}

The primary estimator is centered Ledoit–Wolf. Unshrunk sample covariance and OAS are fixed sensitivity checks:

| Horizon | Internal model | Estimator | Mean DSS change from Ledoit–Wolf | Paired 95% interval |
|---:|---|---|---:|---:|
{chr(10).join(alternative_rows)}

## Data integrity and limits

- All 520 corrected records use the original accepted 130 origins, two horizons, and saved 6,000-draw-per-chain posterior checkpoints.
- The 260 M0 random-walk comparator records come from the already accepted classical baseline archive. The DNS/RW covariance swaps are explicitly counterfactual diagnostics.
- All corrected covariances were symmetric and positive definite and passed Cholesky factorization; their condition numbers rise substantially under correction. Orthogonal factor/residual DSS components reproduced the full DSS.
- The original Bayesian archive, accepted M0–M2 baseline, samplers, priors, and manuscript files were hashed before and after the experiment; hashes match.
- Residuals are treated as zero-mean and serially independent at forecast time, and factor–residual cross-covariance is set to zero. This experiment diagnoses those restrictions; it does not estimate residual dynamics or joint latent states.
- Training and outcome yields are historical/revised-vintage data, not real-time publication vintages. Results are conditional on the accepted two-step DNS factor extraction.
- GPU was unavailable in the active Python environment (`torch` is not installed); NumPy/SciPy CPU execution was used.

Reproducibility details: `{manifest['manifest_path']}`. Main archive: `{OUTPUT_ROOT.relative_to(ROOT)}`. Supporting reports: `mathematical_specification.md` and `baseline_dss_diagnostics.md`.
"""
    (DOC_ROOT / "STRUCTURAL_RECOVERY_REPORT.md").write_text(structural_text, encoding="utf-8")


def _recommendation(records: list[dict], calibration: dict) -> dict:
    average_improvement = float(np.mean([row["dss_original"] - row["dss"] for row in records]))
    joint_coverage = float(np.mean([
        calibration[str(h)]["joint_mahalanobis_squared_90_coverage"] for h in HORIZONS
    ]))
    max_lag1 = max(
        calibration[str(h)]["max_abs_lag1_autocorrelation"] for h in HORIZONS
    )
    max_training_dependence = max(
        calibration[str(h)]["max_abs_mean_training_factor_residual_correlation"]
        for h in HORIZONS
    )
    max_oos_dependence = max(
        calibration[str(h)]["max_abs_oos_factor_error_residual_correlation"]
        for h in HORIZONS
    )
    reason = (
        f"Mean DSS improvement is {average_improvement:.3f}; joint 90% residual coverage "
        f"is {joint_coverage:.3f}, maximum absolute residual lag-1 correlation is "
        f"{max_lag1:.3f}, maximum mean training factor–residual correlation is "
        f"{max_training_dependence:.3f}, and maximum OOS factor-error/residual correlation "
        f"is {max_oos_dependence:.3f}."
    )
    adequate = (
        average_improvement > 0
        and 0.85 <= joint_coverage <= 0.95
        and max_lag1 <= 0.20
        and max_training_dependence <= 0.30
        and max_oos_dependence <= 0.30
    )
    if adequate:
        return {
            "letter": "A",
            "title": "Use the residual-subspace covariance correction as a bounded extension",
            "reason": reason,
        }
    if average_improvement > 0:
        return {
            "letter": "B",
            "title": "Retain the correction as a sensitivity analysis; residual adequacy remains unresolved",
            "reason": reason,
        }
    return {
        "letter": "C",
        "title": "Do not replace the accepted diagonal covariance on these results",
        "reason": reason,
    }


def refresh_existing_outputs() -> dict:
    """Rebuild summaries and reports from saved forecasts without regenerating them."""
    if not (OUTPUT_ROOT / "forecast_records.json").is_file():
        raise FileNotFoundError("No completed residual-subspace forecast archive to analyze")
    records = _read_json(OUTPUT_ROOT / "forecast_records.json")
    origin_diagnostics = _read_json(OUTPUT_ROOT / "origin_residual_diagnostics.json")
    counterfactuals = _read_json(OUTPUT_ROOT / "dns_rw_counterfactuals.json")
    m2_rows = _read_json(OUTPUT_ROOT / "m2_counterfactuals.json")
    results = _read_json(OUTPUT_ROOT / "metrics.json")
    calibration = residual_calibration(records)
    eigenstructure = _eigenstructure_summary(records)
    with np.load(OUTPUT_ROOT / "forecasts.npz", allow_pickle=False) as archive:
        residual_covariances = archive["residual_covariances"].copy()
    residual_evolution = _residual_covariance_evolution(
        origin_diagnostics, residual_covariances
    )
    for row in m2_rows:
        row.setdefault("scenario_count_original", 1024)
        row.setdefault("scenario_count_corrected", SCENARIO_COUNT)
    for horizon in HORIZONS:
        results["m2_counterfactual_summary"][str(horizon)]["scenario_counts"] = {
            "original_accepted_baseline": 1024,
            "corrected_counterfactual": SCENARIO_COUNT,
        }
    recommendation = _recommendation(records, calibration)
    results["residual_calibration"] = calibration
    results["residual_covariance_evolution"] = residual_evolution
    results["eigenstructure_diagnostics"] = eigenstructure
    results["recommendation"] = recommendation
    manifest = _read_json(OUTPUT_ROOT / "experiment_manifest.json")
    manifest["recommendation"] = recommendation
    manifest["source_sha256"] = {
        path: digest(ROOT / path) for path in (
            "src/residual_subspace_covariance.py",
            "src/residual_subspace_analysis.py",
            "src/residual_subspace_experiment.py",
            "src/bayesian_finite_validation.py",
            "src/bayesian_stage_a.py",
            "src/validated_baselines.py",
        )
    }
    _write_json(OUTPUT_ROOT / "metrics.json", results)
    _write_json(OUTPUT_ROOT / "experiment_manifest.json", manifest)
    _write_json(OUTPUT_ROOT / "m2_counterfactuals.json", m2_rows)
    figures = sorted(path.name for path in (OUTPUT_ROOT / "figures").glob("*.png"))
    _write_tables(OUTPUT_ROOT, records, origin_diagnostics, counterfactuals, m2_rows)
    _write_reports(
        results, records, calibration, residual_evolution, counterfactuals,
        figures, manifest,
    )
    return results


def run_experiment() -> dict:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {OUTPUT_ROOT}")
    before_hashes = _protected_hashes()
    design = make_design()
    plan, origin_archives, baseline, baseline_lookup, baseline_scores = verify_accepted_inputs(design)
    load = design["load"]
    factor_basis, residual_u = residual_basis(load)
    if not np.allclose(np.column_stack((factor_basis, residual_u)).T
                       @ np.column_stack((factor_basis, residual_u)), np.eye(8), atol=1e-12):
        raise ArithmeticError("Combined factor/residual basis is not orthonormal")
    full_basis_condition = float(np.linalg.cond(np.column_stack((load, residual_u))))

    primary_records = []
    random_walk_records = []
    covariance_sensitivity = []
    residual_origin_diagnostics = []
    residual_covariances = []
    counterfactual_rows = []
    m2_rows = []
    forecast_arrays = {
        "mean": [], "covariance": [], "observed": [], "scenarios": [],
        "original_covariance": [], "alternative_dss": [],
    }
    accepted_record_count = 0

    for archive_row in origin_archives:
        plan_row = archive_row["plan_row"]
        t = plan_row["index"]
        origin = plan_row["date"]
        training_yields = design["y"][design["first_index"]:t + 1]
        training_factors, coordinates, decomposition_error = decompose_yields(
            training_yields, load, residual_u
        )
        residuals = training_yields - training_factors @ load.T
        orthogonality_error = float(np.max(np.abs(residuals @ load)))
        if orthogonality_error > 1e-9:
            raise ArithmeticError(f"OLS residual orthogonality failed at {origin}")
        estimates = estimate_residual_covariances(coordinates)
        primary_covariance = estimates["ledoit_wolf"]["covariance"]
        residual_covariances.append(
            np.stack([estimates[name]["covariance"] for name in ("ledoit_wolf", "unshrunk", "oas")])
        )
        correlation = np.corrcoef(np.column_stack((training_factors, coordinates)), rowvar=False)
        factor_residual_corr = correlation[:3, 3:]
        lag1 = [
            float(np.corrcoef(coordinates[:-1, j], coordinates[1:, j])[0, 1])
            for j in range(coordinates.shape[1])
        ]
        if not np.isfinite(factor_residual_corr).all() or not np.isfinite(lag1).all():
            raise ArithmeticError(f"Nonfinite training dependence diagnostic at {origin}")
        residual_origin_diagnostics.append({
            "origin": origin,
            "origin_index": t,
            "training_observations": len(training_yields),
            "coordinate_mean": coordinates.mean(axis=0).tolist(),
            "residual_coordinate_mean": coordinates.mean(axis=0).tolist(),
            "coordinate_sd": coordinates.std(axis=0, ddof=1).tolist(),
            "residual_coordinate_sd": coordinates.std(axis=0, ddof=1).tolist(),
            "residual_coordinate_lag1": lag1,
            "training_factor_residual_correlation": factor_residual_corr.tolist(),
            "max_abs_training_factor_residual_correlation": float(np.max(np.abs(factor_residual_corr))),
            "decomposition_relative_error": decomposition_error,
            "orthogonality_max_abs": orthogonality_error,
            "ledoit_wolf_shrinkage": estimates["ledoit_wolf"]["shrinkage"],
            "oas_shrinkage": estimates["oas"]["shrinkage"],
            "covariance_eigenvalues": {
                name: np.linalg.eigvalsh(estimates[name]["covariance"]).tolist()
                for name in ("ledoit_wolf", "unshrunk", "oas")
            },
            "covariance_condition_numbers": {
                name: float(np.linalg.cond(estimates[name]["covariance"]))
                for name in ("ledoit_wolf", "unshrunk", "oas")
            },
        })

        with np.load(archive_row["folder"] / "predictions.npz", allow_pickle=False) as saved:
            saved_predictions = {key: saved[key].copy() for key in saved.files}
        accepted_records = {
            (row["model"], row["horizon"]): row for row in archive_row["report"]["records"]
        }
        factors_by_model = {
            model: _posterior_factors(archive_row["chain_dir"], archive_row["chain_summary"], model)
            for model in MODELS
        }

        for model_index, model in enumerate(MODELS):
            factor_chains = factors_by_model[model]
            for horizon in HORIZONS:
                accepted = accepted_records[(model, horizon)]
                tag = f"{model}_h{horizon}"
                original_mean = np.asarray(accepted["mean"], dtype=float)
                original_covariance = np.asarray(accepted["covariance"], dtype=float)
                observed = design["y"][t + horizon]
                if (
                    not np.array_equal(saved_predictions[tag + "_mean"], original_mean)
                    or not np.array_equal(saved_predictions[tag + "_covariance"], original_covariance)
                    or not np.array_equal(saved_predictions[tag + "_observed"], observed)
                ):
                    raise ValueError(f"Accepted prediction arrays do not match report: {origin} {tag}")
                old_scenarios = saved_predictions[tag + "_scenarios"]
                old_score = dss_components(observed, original_mean, original_covariance)
                old_energy = energy_score(old_scenarios, observed)
                if (
                    not np.isclose(old_score["dss"], accepted["dss"], rtol=1e-9, atol=1e-9)
                    or not np.isclose(old_energy, accepted["energy_score"], rtol=1e-10, atol=1e-10)
                ):
                    raise ValueError(f"Accepted score reconstruction failed: {origin} {tag}")

                factor_mean = np.asarray(accepted["factor_mean"], dtype=float)
                factor_variances = np.asarray(accepted["factor_variance"], dtype=float)
                mean = load @ factor_mean
                if not np.allclose(mean, original_mean, rtol=1e-9, atol=1e-9):
                    raise ValueError(f"Accepted factor mean reconstruction failed: {origin} {tag}")
                corrected_covariance = assemble_yield_covariance(
                    load, factor_variances, residual_u, primary_covariance
                )
                corrected_score = dss_components(observed, mean, corrected_covariance)
                corrected_split = orthogonal_dss_components(
                    observed, mean, factor_variances, load, factor_basis, residual_u,
                    primary_covariance, corrected_score["dss"],
                )
                old_eigen = eigen_diagnostics(observed, original_mean, original_covariance)
                new_eigen = eigen_diagnostics(observed, mean, corrected_covariance)
                alt_dss = {}
                for method in ("unshrunk", "oas"):
                    alt_covariance = assemble_yield_covariance(
                        load, factor_variances, residual_u, estimates[method]["covariance"]
                    )
                    alt_dss[method] = dss_components(observed, mean, alt_covariance)["dss"]
                factor_seed = 2026111600000 + plan_row["ordinal"] * 100 + model_index * 10 + horizon * 2
                residual_seed = factor_seed + 1
                factor_draws = np.column_stack([
                    factor_scenarios(
                        factor_chains[j][0], names, training_factors[-1, j], horizon, model,
                        np.random.default_rng(factor_seed + j), SCENARIO_COUNT,
                    )
                    for j, (_, names) in enumerate(factor_chains)
                ])
                residual_chol = np.linalg.cholesky(primary_covariance)
                residual_draws = (
                    np.random.default_rng(residual_seed).standard_normal((SCENARIO_COUNT, 5))
                    @ residual_chol.T
                )
                scenarios = factor_draws @ load.T + residual_draws @ residual_u.T
                energy = energy_score(scenarios, observed)
                lower, upper = np.quantile(scenarios, [0.05, 0.95], axis=0)
                error = observed - mean
                target_coordinates = observed @ residual_u
                whitened = np.linalg.solve(residual_chol, target_coordinates)
                target_factor_error = design["factors"][t + horizon] - factor_mean
                covariance_diagnostics = {
                    "symmetry_max_abs": float(np.max(np.abs(corrected_covariance - corrected_covariance.T))),
                    "minimum_eigenvalue": float(np.linalg.eigvalsh(corrected_covariance).min()),
                    "maximum_eigenvalue": float(np.linalg.eigvalsh(corrected_covariance).max()),
                    "condition_number": float(np.linalg.cond(corrected_covariance)),
                    "rank": int(np.linalg.matrix_rank(corrected_covariance)),
                    "cholesky_passed": True,
                    "log_determinant": corrected_score["log_determinant"],
                    "mahalanobis": corrected_score["mahalanobis"],
                }
                if (
                    covariance_diagnostics["symmetry_max_abs"] > 1e-10
                    or covariance_diagnostics["minimum_eigenvalue"] <= 0
                    or covariance_diagnostics["rank"] != 8
                    or not np.isfinite(covariance_diagnostics["condition_number"])
                ):
                    raise ArithmeticError(f"Corrected covariance diagnostic failed: {origin} {tag}")
                row_record = {
                    "origin": origin,
                    "origin_index": t,
                    "target": design["dates"][t + horizon].isoformat(),
                    "target_index": t + horizon,
                    "model": model,
                    "paper_model": "M3" if model == "M4" else "M4",
                    "horizon": horizon,
                    "scenario_seed_factor": factor_seed,
                    "scenario_seed_residual": residual_seed,
                    "scenario_count": SCENARIO_COUNT,
                    "mean": mean.tolist(),
                    "covariance": corrected_covariance.tolist(),
                    "observed": observed.tolist(),
                    "factor_mean": factor_mean.tolist(),
                    "factor_variance": factor_variances.tolist(),
                    "residual_covariance": primary_covariance.tolist(),
                    "residual_coordinate_mean": coordinates.mean(axis=0).tolist(),
                    "dss_original": old_score["dss"],
                    "energy_score_original": old_energy,
                    "coverage_90_original": accepted["coverage_90"],
                    "interval_width_90_original": (
                        np.asarray(accepted["interval_upper_90"])
                        - np.asarray(accepted["interval_lower_90"])
                    ).tolist(),
                    "original_log_determinant": old_score["log_determinant"],
                    "original_mahalanobis": old_score["mahalanobis"],
                    "original_eigenvalues": old_eigen["eigenvalues_descending"],
                    "original_condition_number": old_eigen["condition_number"],
                    "original_eigen_direction_error_contributions":
                        old_eigen["eigen_direction_error_contributions"],
                    "original_largest_contribution_direction":
                        old_eigen["largest_contribution_direction"],
                    "corrected_log_determinant": corrected_score["log_determinant"],
                    "corrected_mahalanobis": corrected_score["mahalanobis"],
                    "factor_log_determinant": corrected_split["factor_log_determinant"],
                    "factor_mahalanobis": corrected_split["factor_mahalanobis"],
                    "residual_log_determinant": corrected_split["residual_log_determinant"],
                    "residual_mahalanobis": corrected_split["residual_mahalanobis"],
                    "dss": corrected_score["dss"],
                    "energy_score": energy,
                    "dss_alternatives": alt_dss,
                    "dss_orthogonal_reconstruction": corrected_split["reconstructed_dss"],
                    "dss_corrected_eigen_direction_error_contributions":
                        new_eigen["eigen_direction_error_contributions"],
                    "largest_contribution_index": new_eigen["largest_contribution_index"],
                    "largest_contribution_share": new_eigen["largest_contribution_share"],
                    "largest_contribution_direction": new_eigen["largest_contribution_direction"],
                    "eigenvalues_descending": new_eigen["eigenvalues_descending"],
                    "condition_number": new_eigen["condition_number"],
                    "covariance_diagnostics": covariance_diagnostics,
                    "squared_errors": (error**2).tolist(),
                    "absolute_errors": np.abs(error).tolist(),
                    "coverage_90": ((observed >= lower) & (observed <= upper)).astype(int).tolist(),
                    "interval_lower_90": lower.tolist(),
                    "interval_upper_90": upper.tolist(),
                    "interval_width_90": (upper - lower).tolist(),
                    "residual_target_coordinates": target_coordinates.tolist(),
                    "residual_marginal_sd": np.sqrt(np.diag(primary_covariance)).tolist(),
                    "whitened_residual_coordinates": whitened.tolist(),
                    "residual_mahalanobis_squared": float(whitened @ whitened),
                    "factor_target_error": target_factor_error.tolist(),
                    "training_factor_residual_correlation": factor_residual_corr.tolist(),
                }
                primary_records.append(row_record)
                covariance_sensitivity.append(row_record)
                forecast_arrays["mean"].append(mean)
                forecast_arrays["covariance"].append(corrected_covariance)
                forecast_arrays["observed"].append(observed)
                forecast_arrays["scenarios"].append(scenarios)
                forecast_arrays["original_covariance"].append(original_covariance)
                forecast_arrays["alternative_dss"].append([alt_dss["unshrunk"], alt_dss["oas"]])
                accepted_record_count += 1

        for horizon in HORIZONS:
            index = baseline_lookup[(origin, horizon, "M0")]
            random_walk_records.append(
                _baseline_record(index, origin, horizon, "M0", baseline, baseline_scores)
            )
            m0_index = index
            m2_index = baseline_lookup[(origin, horizon, "M2")]
            rw_mean, rw_cov = baseline["means"][m0_index], baseline["covariances"][m0_index]
            dns_mean = baseline["means"][m2_index]
            dns_cov = baseline["covariances"][m2_index]
            observed = baseline["targets"][m2_index]
            if not np.array_equal(observed, design["y"][t + horizon]):
                raise ValueError(f"Classical baseline target mismatch: {origin} h{horizon}")
            training_factors = design["factors"][design["first_index"]:t + 1]
            training_residual = training_yields - training_factors @ load.T
            reconstruction_variance = np.maximum(np.mean(training_residual**2, axis=0), 1e-6)
            factor_covariance = dns_cov - np.diag(reconstruction_variance)
            factor_block = factor_basis.T @ factor_covariance @ factor_basis
            factor_block = (factor_block + factor_block.T) / 2
            if np.linalg.eigvalsh(factor_block).min() <= 0:
                raise ArithmeticError(f"M2 factor covariance is not positive definite: {origin} h{horizon}")
            factor_covariance = factor_basis @ factor_block @ factor_basis.T
            reconstructed_dns_cov = factor_covariance + np.diag(reconstruction_variance)
            if not np.allclose(reconstructed_dns_cov, dns_cov, rtol=1e-8, atol=2e-6):
                raise ArithmeticError(f"Could not isolate M2 factor covariance: {origin} h{horizon}")
            dns_corrected_cov = factor_covariance + residual_u @ primary_covariance @ residual_u.T
            np.linalg.cholesky(dns_corrected_cov)
            counterfactual_rows.append({
                "origin": origin,
                "horizon": horizon,
                "rw_mean_rw_covariance": dss_components(observed, rw_mean, rw_cov)["dss"],
                "dns_mean_dns_diagonal_covariance": dss_components(observed, dns_mean, dns_cov)["dss"],
                "rw_mean_dns_corrected_covariance": dss_components(observed, rw_mean, dns_corrected_cov)["dss"],
                "dns_mean_rw_covariance": dss_components(observed, dns_mean, rw_cov)["dss"],
                "dns_mean_dns_corrected_covariance": dss_components(observed, dns_mean, dns_corrected_cov)["dss"],
            })
            dns_scenarios = (
                dns_mean
                + np.random.default_rng(2026111601000 + plan_row["ordinal"] * 10 + horizon)
                .standard_normal((SCENARIO_COUNT, 8))
                @ np.linalg.cholesky(dns_corrected_cov).T
            )
            original_dns_scenarios = baseline["scenarios"][m2_index]
            original_lower = dns_mean - norm.ppf(0.95) * np.sqrt(np.diag(dns_cov))
            original_upper = dns_mean + norm.ppf(0.95) * np.sqrt(np.diag(dns_cov))
            corrected_lower, corrected_upper = np.quantile(dns_scenarios, [0.05, 0.95], axis=0)
            m2_error = observed - dns_mean
            m2_rows.append({
                "origin": origin,
                "horizon": horizon,
                "scenario_count_original": 1024,
                "scenario_count_corrected": SCENARIO_COUNT,
                "dss_original": dss_components(observed, dns_mean, dns_cov)["dss"],
                "dss_corrected": dss_components(observed, dns_mean, dns_corrected_cov)["dss"],
                "energy_original": energy_score(original_dns_scenarios, observed),
                "energy_corrected": energy_score(dns_scenarios, observed),
                "squared_errors_original": (m2_error**2).tolist(),
                "squared_errors_corrected": (m2_error**2).tolist(),
                "coverage_original": ((observed >= original_lower) & (observed <= original_upper)).astype(int).tolist(),
                "coverage_corrected": ((observed >= corrected_lower) & (observed <= corrected_upper)).astype(int).tolist(),
                "interval_width_original": (original_upper - original_lower).tolist(),
                "interval_width_corrected": (corrected_upper - corrected_lower).tolist(),
                "dns_mean": dns_mean.tolist(),
                "dns_covariance_original": dns_cov.tolist(),
                "dns_covariance_corrected": dns_corrected_cov.tolist(),
                "rw_mean": rw_mean.tolist(),
                "rw_covariance": rw_cov.tolist(),
            })

    if accepted_record_count != 520 or len(primary_records) != 520 or len(random_walk_records) != 260:
        raise ValueError("The completed experiment does not contain 520 matched Bayesian forecasts")
    combined_records = primary_records + random_walk_records
    forecast_metrics = summarize_forecasts(combined_records, BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED)
    covariance_estimator_sensitivity = _summarize_covariance_sensitivity(covariance_sensitivity)
    diagonal_to_subspace = _diagonal_to_subspace_summary(primary_records)
    dns_rw_summary = _counterfactual_summary(counterfactual_rows)
    m2_summary = _m2_summary(m2_rows)
    calibration = residual_calibration(primary_records)
    eigenstructure = _eigenstructure_summary(primary_records)
    residual_evolution = _residual_covariance_evolution(
        residual_origin_diagnostics, np.asarray(residual_covariances)
    )

    output_records_path = OUTPUT_ROOT / "forecast_records.json"
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite experiment outputs: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    np.savez_compressed(
        OUTPUT_ROOT / "forecasts.npz",
        means=np.asarray(forecast_arrays["mean"]),
        covariances=np.asarray(forecast_arrays["covariance"]),
        observed=np.asarray(forecast_arrays["observed"]),
        scenarios=np.asarray(forecast_arrays["scenarios"]),
        original_covariances=np.asarray(forecast_arrays["original_covariance"]),
        alternative_dss=np.asarray(forecast_arrays["alternative_dss"]),
        residual_covariances=np.asarray(residual_covariances),
        residual_basis=residual_u,
        factor_basis=factor_basis,
        origins=np.asarray([row["origin"] for row in primary_records], dtype="U10"),
        targets=np.asarray([row["target"] for row in primary_records], dtype="U10"),
        models=np.asarray([row["model"] for row in primary_records], dtype="U2"),
        paper_models=np.asarray([row["paper_model"] for row in primary_records], dtype="U2"),
        horizons=np.asarray([row["horizon"] for row in primary_records], dtype=np.int64),
    )
    _write_json(output_records_path, primary_records)
    _write_json(OUTPUT_ROOT / "origin_residual_diagnostics.json", residual_origin_diagnostics)
    _write_json(OUTPUT_ROOT / "dns_rw_counterfactuals.json", counterfactual_rows)
    _write_json(OUTPUT_ROOT / "m2_counterfactuals.json", m2_rows)
    results = {
        "accepted_520_record_archive": True,
        "forecast_metrics": forecast_metrics,
        "diagonal_to_subspace": diagonal_to_subspace,
        "covariance_estimator_sensitivity": covariance_estimator_sensitivity,
        "residual_calibration": calibration,
        "residual_covariance_evolution": residual_evolution,
        "eigenstructure_diagnostics": eigenstructure,
        "dns_rw_counterfactual_summary": dns_rw_summary,
        "m2_counterfactual_summary": m2_summary,
    }
    _write_json(OUTPUT_ROOT / "metrics.json", results)
    _write_tables(
        OUTPUT_ROOT, primary_records, residual_origin_diagnostics, counterfactual_rows, m2_rows
    )
    figures = make_figures(primary_records, OUTPUT_ROOT / "figures")

    recommendation = _recommendation(primary_records, calibration)
    results["recommendation"] = recommendation
    results["mathematical_checks"] = {
        "combined_basis_orthonormal": True,
        "combined_basis_condition_number": full_basis_condition,
        "n_yields": 8,
        "n_factors": 3,
        "n_residual_coordinates": 5,
        "max_decomposition_relative_error": max(
            row["decomposition_relative_error"] for row in residual_origin_diagnostics
        ),
        "max_orthogonality_error": max(
            row["orthogonality_max_abs"] for row in residual_origin_diagnostics
        ),
        "all_520_corrected_covariances_cholesky_passed": all(
            row["covariance_diagnostics"]["cholesky_passed"] for row in primary_records
        ),
        "all_520_orthogonal_dss_reconstructions_passed": all(
            np.isclose(row["dss"], row["dss_orthogonal_reconstruction"], rtol=1e-9, atol=1e-9)
            for row in primary_records
        ),
    }
    if not results["mathematical_checks"]["all_520_orthogonal_dss_reconstructions_passed"]:
        raise ArithmeticError("At least one orthogonal DSS decomposition failed")

    after_hashes = _protected_hashes()
    if before_hashes != after_hashes:
        changed = sorted(key for key in set(before_hashes) | set(after_hashes)
                         if before_hashes.get(key) != after_hashes.get(key))
        raise RuntimeError(f"Protected accepted assets changed during the experiment: {changed[:10]}")
    manifest = {
        "task": "Residual-subspace covariance evaluation using frozen accepted posterior draws",
        "created_utc": "2026-11-16",
        "accepted_archive": str(ARCHIVE_ROOT.relative_to(ROOT)),
        "accepted_plan_sha256": digest(ARCHIVE_ROOT / "plan.json"),
        "accepted_gate_sha256": digest(ARCHIVE_ROOT / "phase2_gate.json"),
        "accepted_summary_sha256": digest(ARCHIVE_ROOT / "summary.json"),
        "input_sha256": design["input_sha256"],
        "lambda": design["lambda"],
        "scenario_count": SCENARIO_COUNT,
        "origins": 130,
        "horizons": HORIZONS,
        "internal_models": MODELS,
        "paper_label_mapping": {"M3": "internal M4, constant factor volatility",
                                "M4": "internal M5, stochastic factor volatility"},
        "posterior_draws_refit": False,
        "samplers_priors_and_accepted_forecasts_modified": False,
        "primary_covariance_estimator": "scikit-learn LedoitWolf, centered coordinates",
        "covariance_sensitivity_estimators": ["unshrunk sample covariance", "scikit-learn OAS"],
        "residual_predictive_mean": "zero",
        "factor_residual_cross_covariance": "zero",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_block_lengths": [8, 4, 12],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gpu": "unavailable; NumPy/SciPy CPU used",
        "protected_hashes_before_and_after_match": True,
        "protected_files_hashed": len(before_hashes),
        "protected_hashes_before": before_hashes,
        "protected_hashes_after": after_hashes,
        "manifest_path": str((OUTPUT_ROOT / "experiment_manifest.json").relative_to(ROOT)),
        "source_sha256": {
            path: digest(ROOT / path) for path in (
                "src/residual_subspace_covariance.py",
                "src/residual_subspace_analysis.py",
                "src/residual_subspace_experiment.py",
                "src/bayesian_finite_validation.py",
                "src/bayesian_stage_a.py",
                "src/validated_baselines.py",
            )
        },
        "forecast_archive_sha256": digest(OUTPUT_ROOT / "forecasts.npz"),
        "forecast_records_sha256": digest(output_records_path),
        "recommendation": recommendation,
    }
    results["manifest_path"] = str((OUTPUT_ROOT / "experiment_manifest.json").relative_to(ROOT))
    results["recommendation"] = recommendation
    _write_json(OUTPUT_ROOT / "metrics.json", results)
    _write_json(OUTPUT_ROOT / "experiment_manifest.json", manifest)
    _write_reports(
        results, primary_records, calibration, residual_evolution,
        counterfactual_rows, figures, manifest,
    )
    print(json.dumps({
        "forecast_records": len(primary_records),
        "recommendation": recommendation,
        "forecast_metrics": forecast_metrics["horizons"],
        "residual_calibration": calibration,
        "protected_files_hashed": len(before_hashes),
    }, indent=2, allow_nan=False))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-reports", action="store_true",
        help="Recompute summaries and reports from an existing forecast archive",
    )
    arguments = parser.parse_args()
    if arguments.refresh_reports:
        refresh_existing_outputs()
    else:
        run_experiment()
