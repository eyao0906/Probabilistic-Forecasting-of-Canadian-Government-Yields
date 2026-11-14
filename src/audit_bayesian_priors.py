"""Audit Bayesian DNS priors against the initial sample and accepted posterior draws."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm, truncnorm, t

from src.bayesian_stage_a import FACTORS, ROOT, make_design


RUN_NAME = "bayesian_finite_20261027T194239Z"
ORIGIN = "2025-11-28"
OUTPUT = ROOT / "results" / "prior_justification_20261114"
QUANTILES = np.array([0.05, 0.5, 0.95])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_accepted_draws() -> tuple[dict[str, np.ndarray], dict]:
    run = ROOT / "results" / RUN_NAME
    forecast_path = run / "rolling_main" / "origins" / ORIGIN / "forecast.json"
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    specification = forecast["specification"]
    if not forecast.get("accepted") or specification["origin"] != ORIGIN:
        raise ValueError(f"No accepted forecast found for {ORIGIN}")

    chain_dir = run / Path(specification["chain_source"]).name
    summary_path = chain_dir / "summary.json"
    validation_path = chain_dir / "predictive_validation" / "summary.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("passes"):
        raise ValueError("Accepted posterior chain failed predictive validation")
    if digest(summary_path) != forecast["parameter_summary_sha256"]:
        raise ValueError("Posterior summary hash differs from the accepted forecast")

    forecast_hashes = {
        Path(path).name: value
        for path, value in forecast["chain_checkpoint_sha256"].items()
    }
    validation_hashes = validation["provenance"]["checkpoints"]
    draws_by_factor = {}
    chain_hashes = {}
    for factor_index, factor in enumerate(FACTORS):
        pooled = []
        prior = json.loads(
            (chain_dir / "configuration.json").read_text(encoding="utf-8")
        )["priors"][factor_index]["base"]
        for chain_index in range(4):
            name = f"M5_{factor}_chain{chain_index}.npz"
            path = chain_dir / name
            checksum = digest(path)
            if checksum != forecast_hashes.get(name) or checksum != validation_hashes.get(name):
                raise ValueError(f"Chain does not match accepted evidence: {name}")
            with np.load(path, allow_pickle=False) as archive:
                chain_draws = archive["draws"]
                report = json.loads(str(archive["report"]))
            if report["names"][:5] != ["a", "b", "m", "phi", "v"]:
                raise ValueError(f"Unexpected parameter order in {name}")
            pooled.append(chain_draws[:, :5])
            chain_hashes[name] = checksum
        values = np.concatenate(pooled, axis=0)
        if values.shape != (24_000, 5):
            raise ValueError(f"Unexpected draw count for {factor}: {values.shape}")
        if not np.isfinite(values[:, :5]).all():
            raise ValueError(f"Nonfinite accepted posterior draw for {factor}")
        draws_by_factor[factor] = values

    provenance = {
        "forecast": str(forecast_path.relative_to(ROOT)),
        "forecast_sha256": digest(forecast_path),
        "chain_source": str(chain_dir.relative_to(ROOT)),
        "summary_sha256": digest(summary_path),
        "validation_sha256": digest(validation_path),
        "accepted": True,
        "posterior_draws_per_factor": 24_000,
        "chain_sha256": chain_hashes,
    }
    return draws_by_factor, provenance


def ols_summary(factors: np.ndarray) -> list[dict]:
    rows = []
    n = len(factors)
    transitions = n - 1
    for j, factor in enumerate(FACTORS):
        values = factors[:, j]
        y = values[1:]
        design = np.column_stack((np.ones(transitions), values[:-1]))
        coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        residuals = y - design @ coefficients
        residual_sum_squares = float(residuals @ residuals)
        residual_variance = residual_sum_squares / transitions
        inverse = np.linalg.inv(design.T @ design)

        ols_covariance = residual_variance * inverse
        ols_se = np.sqrt(np.diag(ols_covariance))
        critical = float(t.ppf(0.975, transitions - 2))

        score = design * residuals[:, None]
        meat = score.T @ score
        for lag in range(1, 5):
            lag_product = score[lag:].T @ score[:-lag]
            weight = 1 - lag / 5
            meat += weight * (lag_product + lag_product.T)
        hac_covariance = inverse @ meat @ inverse * transitions / (transitions - 2)
        hac_se = np.sqrt(np.diag(hac_covariance))
        intercept, slope = coefficients
        stable = abs(slope) < 1
        rows.append({
            "factor": factor,
            "n_observations": n,
            "n_transitions": transitions,
            "factor_mean": float(np.mean(values)),
            "factor_sd": float(np.std(values, ddof=1)),
            "factor_min": float(np.min(values)),
            "factor_max": float(np.max(values)),
            "ols_intercept": float(intercept),
            "ols_intercept_se": float(ols_se[0]),
            "ols_intercept_ci95_low": float(intercept - critical * ols_se[0]),
            "ols_intercept_ci95_high": float(intercept + critical * ols_se[0]),
            "ols_ar": float(slope),
            "ols_ar_se": float(ols_se[1]),
            "ols_ar_ci95_low": float(slope - critical * ols_se[1]),
            "ols_ar_ci95_high": float(slope + critical * ols_se[1]),
            "hac4_intercept_se": float(hac_se[0]),
            "hac4_ar_se": float(hac_se[1]),
            "hac4_ar_ci95_low": float(slope - 1.96 * hac_se[1]),
            "hac4_ar_ci95_high": float(slope + 1.96 * hac_se[1]),
            "innovation_variance_mse": residual_variance,
            "innovation_sd": float(np.sqrt(residual_variance)),
            "stationary_ols_estimate": stable,
            "ols_implied_long_run_mean": (
                float(intercept / (1 - slope)) if stable else None
            ),
            "ols_implied_half_life_weeks": (
                float(np.log(0.5) / np.log(abs(slope)))
                if stable and 0 < abs(slope) < 1 else None
            ),
        })
    return rows


def prior_rows(priors: tuple) -> list[dict]:
    rows = []
    trunc_a = (-1 - 0.7) / 0.3
    trunc_b = (1 - 0.7) / 0.3
    b_prior = truncnorm(trunc_a, trunc_b, loc=0.7, scale=0.3)
    phi_prior = beta(20, 1.5)
    v_rate, v_upper = 1.0, 4.0
    v_quantiles = -np.log(1 - QUANTILES * (1 - np.exp(-v_rate * v_upper))) / v_rate
    v_mean = 1 / v_rate - v_upper / np.expm1(v_rate * v_upper)

    for factor, prior in zip(FACTORS, priors):
        mu0 = prior.omega_mean
        rows.extend([
            {
                "factor": factor, "parameter": "a",
                "distribution": "Normal(0, 25), untruncated",
                "prior_mean": 0.0, "prior_sd": 5.0,
                "prior_q05": float(norm.ppf(0.05, 0, 5)),
                "prior_median": 0.0,
                "prior_q95": float(norm.ppf(0.95, 0, 5)),
                "scale_interpretation": "factor units per weekly transition",
            },
            {
                "factor": factor, "parameter": "b",
                "distribution": "Normal(0.7, 0.09) truncated to (-1, 1)",
                "prior_mean": float(b_prior.mean()), "prior_sd": float(b_prior.std()),
                "prior_q05": float(b_prior.ppf(0.05)),
                "prior_median": float(b_prior.ppf(0.5)),
                "prior_q95": float(b_prior.ppf(0.95)),
                "scale_interpretation": "AR(1) persistence",
            },
            {
                "factor": factor, "parameter": "m",
                "distribution": f"Normal({mu0:.12g}, 25)",
                "prior_mean": mu0, "prior_sd": 5.0,
                "prior_q05": float(mu0 - 1.6448536269514722 * 5),
                "prior_median": mu0,
                "prior_q95": float(mu0 + 1.6448536269514722 * 5),
                "scale_interpretation": "log long-run expected innovation variance",
            },
            {
                "factor": factor, "parameter": "phi",
                "distribution": "2*Beta(20, 1.5)-1",
                "prior_mean": float(2 * 20 / 21.5 - 1),
                "prior_sd": float(2 * np.sqrt(20 * 1.5 / (21.5**2 * 22.5))),
                "prior_q05": float(2 * phi_prior.ppf(0.05) - 1),
                "prior_median": float(2 * phi_prior.ppf(0.5) - 1),
                "prior_q95": float(2 * phi_prior.ppf(0.95) - 1),
                "scale_interpretation": "stationary log-volatility persistence",
            },
            {
                "factor": factor, "parameter": "v",
                "distribution": "Truncated Exponential(rate=1, 0<v<4)",
                "prior_mean": float(v_mean), "prior_sd": None,
                "prior_q05": float(v_quantiles[0]),
                "prior_median": float(v_quantiles[1]),
                "prior_q95": float(v_quantiles[2]),
                "scale_interpretation": "stationary variance of log volatility",
            },
            {
                "factor": factor, "parameter": "exp(m)",
                "distribution": "Lognormal(mu0, variance=25)",
                "prior_mean": float(np.exp(mu0 + 12.5)),
                "prior_sd": None,
                "prior_q05": float(np.exp(mu0 - 1.6448536269514722 * 5)),
                "prior_median": float(np.exp(mu0)),
                "prior_q95": float(np.exp(mu0 + 1.6448536269514722 * 5)),
                "scale_interpretation": "innovation variance; mean dominated by upper tail",
            },
        ])
    return rows


def posterior_rows(draws_by_factor: dict[str, np.ndarray], priors: tuple) -> list[dict]:
    rows = []
    indices = {"a": 0, "b": 1, "m": 2, "phi": 3, "v": 4}
    for factor_index, factor in enumerate(FACTORS):
        values = draws_by_factor[factor]
        base_prior = priors[factor_index]
        for parameter, column in indices.items():
            sample = values[:, column]
            q05, median, q95 = np.quantile(sample, QUANTILES)
            row = {
                "origin": ORIGIN,
                "presentation_model": "M4",
                "saved_model_label": "M5",
                "factor": factor,
                "parameter": parameter,
                "posterior_draws": int(len(sample)),
                "posterior_q05": float(q05),
                "posterior_median": float(median),
                "posterior_q95": float(q95),
            }
            if parameter == "phi":
                prior_dist = beta(base_prior.phi_alpha, base_prior.phi_beta)
                row.update({
                    "prior_q05": float(2 * prior_dist.ppf(0.05) - 1),
                    "prior_median": float(2 * prior_dist.ppf(0.5) - 1),
                    "prior_q95": float(2 * prior_dist.ppf(0.95) - 1),
                    "prior_cdf_at_post_q05": float(prior_dist.cdf((q05 + 1) / 2)),
                    "prior_cdf_at_post_median": float(prior_dist.cdf((median + 1) / 2)),
                    "prior_cdf_at_post_q95": float(prior_dist.cdf((q95 + 1) / 2)),
                })
            elif parameter == "v":
                rate, upper = 1.0, 4.0
                v_quantiles = -np.log(
                    1 - QUANTILES * (1 - np.exp(-rate * upper))
                ) / rate
                row.update({
                    "prior_q05": float(v_quantiles[0]),
                    "prior_median": float(v_quantiles[1]),
                    "prior_q95": float(v_quantiles[2]),
                    "posterior_mass_v_gt_2": float(np.mean(sample > 2)),
                    "posterior_mass_v_gt_3_6": float(np.mean(sample > 3.6)),
                    "posterior_mass_v_gt_3_9": float(np.mean(sample > 3.9)),
                    "posterior_max_v": float(np.max(sample)),
                })
            elif parameter == "m":
                mu0 = base_prior.omega_mean
                row.update({
                    "prior_q05": float(mu0 - 1.6448536269514722 * 5),
                    "prior_median": float(mu0),
                    "prior_q95": float(mu0 + 1.6448536269514722 * 5),
                    "prior_cdf_at_post_q05": float(norm.cdf(q05, mu0, 5)),
                    "prior_cdf_at_post_median": float(norm.cdf(median, mu0, 5)),
                    "prior_cdf_at_post_q95": float(norm.cdf(q95, mu0, 5)),
                })
            elif parameter == "b":
                prior_dist = truncnorm(
                    (-1 - base_prior.beta_mean[1]) / np.sqrt(base_prior.beta_variance[1]),
                    (1 - base_prior.beta_mean[1]) / np.sqrt(base_prior.beta_variance[1]),
                    loc=base_prior.beta_mean[1],
                    scale=np.sqrt(base_prior.beta_variance[1]),
                )
                row.update({
                    "prior_q05": float(prior_dist.ppf(.05)),
                    "prior_median": float(prior_dist.ppf(.5)),
                    "prior_q95": float(prior_dist.ppf(.95)),
                    "prior_cdf_at_post_q05": float(prior_dist.cdf(q05)),
                    "prior_cdf_at_post_median": float(prior_dist.cdf(median)),
                    "prior_cdf_at_post_q95": float(prior_dist.cdf(q95)),
                })
            else:
                prior_dist = norm(0, np.sqrt(base_prior.beta_variance[0]))
                row.update({
                    "prior_q05": float(prior_dist.ppf(.05)),
                    "prior_median": 0.0,
                    "prior_q95": float(prior_dist.ppf(.95)),
                })
            rows.append(row)
    return rows


def audit(output: Path) -> dict:
    design = make_design()
    initial = design["factors"][:design["initial_n"]]
    if len(initial) != 835 or design["initial_last_date"] != "2015-12-31":
        raise ValueError("The accepted 2000-2015 training window changed")
    posterior, posterior_provenance = load_accepted_draws()
    ols = ols_summary(initial)
    write_rows(output / "training_ar1_summary.csv", ols)
    write_rows(output / "prior_parameter_summary.csv", prior_rows(design["priors"]))
    write_rows(output / "posterior_prior_comparison.csv", posterior_rows(posterior, design["priors"]))

    input_files = [
        ROOT / "data" / "processed" / "canada_weekly.csv",
        ROOT / "src" / "bayesian_stage_a.py",
        ROOT / "src" / "bayesian_revised.py",
        ROOT / "src" / "bayesian_finite.py",
        ROOT / "src" / "bayesian_finite_runner.py",
        ROOT / "src" / "experiment.py",
        ROOT / "src" / "models.py",
        ROOT / "docs" / "bayesian_finite_prior.md",
        ROOT / "references.bib",
        Path(__file__),
    ]
    metadata = {
        "status": "completed",
        "training_period": ["2000-01-01", "2015-12-31"],
        "training_observations": int(len(initial)),
        "transition_count": int(len(initial) - 1),
        "factor_names": list(FACTORS),
        "maturities_years": [0.25, 0.5, 1, 2, 5, 7, 10, 30],
        "fixed_lambda_per_year": float(design["lambda"]),
        "input_sha256": design["input_sha256"],
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in input_files},
        "posterior_comparison": posterior_provenance,
        "inference_caveat": (
            "OLS/HAC intervals are descriptive under near-unit-root persistence; "
            "the auxiliary OLS estimate is not forced into the Bayesian stationary support."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    print(json.dumps({
        "status": result["status"],
        "training_observations": result["training_observations"],
        "fixed_lambda_per_year": result["fixed_lambda_per_year"],
        "posterior_draws_per_factor": result["posterior_comparison"]["posterior_draws_per_factor"],
        "output": str(args.output.resolve()),
    }))


if __name__ == "__main__":
    main()
