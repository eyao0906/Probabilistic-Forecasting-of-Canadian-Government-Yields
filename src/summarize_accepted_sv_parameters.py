"""Summarize accepted final-origin stochastic-volatility posterior draws."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, truncnorm


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "bayesian_finite_20261027T194239Z"
ORIGIN = "2025-11-28"
FACTORS = ("level", "slope", "curvature")
QUANTILES = np.array([0.05, 0.5, 0.95])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(output: Path) -> Path:
    run = ROOT / "results" / RUN_NAME
    forecast_path = run / "rolling_main" / "origins" / ORIGIN / "forecast.json"
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    specification = forecast["specification"]
    if not forecast.get("accepted") or specification["origin"] != ORIGIN:
        raise ValueError(f"No accepted forecast found for {ORIGIN}")

    chain_dir = run / Path(specification["chain_source"]).name
    configuration_path = chain_dir / "configuration.json"
    summary_path = chain_dir / "summary.json"
    validation_path = chain_dir / "predictive_validation" / "summary.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    chain_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("passes"):
        raise ValueError("Saved posterior chains did not pass predictive validation")
    if digest(summary_path) != forecast["parameter_summary_sha256"]:
        raise ValueError("Posterior summary hash differs from the accepted forecast")

    forecast_hashes = {
        Path(path).name: value
        for path, value in forecast["chain_checkpoint_sha256"].items()
    }
    validation_hashes = validation["provenance"]["checkpoints"]
    prior_configurations = configuration["priors"]
    rows = []
    for factor_index, factor in enumerate(FACTORS):
        chain_draws = []
        prior = prior_configurations[factor_index]["base"]
        for chain_index in range(4):
            name = f"M5_{factor}_chain{chain_index}.npz"
            path = chain_dir / name
            checksum = digest(path)
            if checksum != forecast_hashes.get(name) or checksum != validation_hashes.get(name):
                raise ValueError(f"Saved chain does not match accepted evidence: {name}")
            with np.load(path, allow_pickle=False) as archive:
                draws = archive["draws"]
                report = json.loads(str(archive["report"]))
            if report["names"][:5] != ["a", "b", "m", "phi", "v"]:
                raise ValueError(f"Unexpected parameter order in {name}")
            chain_draws.append(draws[:, [1, 3, 4]])

        pooled = np.concatenate(chain_draws, axis=0)
        if pooled.shape[0] != 24_000 or not np.isfinite(pooled).all():
            raise ValueError(f"Unexpected or nonfinite posterior draws for {factor}")
        posterior_quantiles = np.quantile(pooled, QUANTILES, axis=0)

        beta_mean, beta_variance = prior["beta_mean"], prior["beta_variance"]
        b_prior = truncnorm(
            (-1.0 - beta_mean[1]) / np.sqrt(beta_variance[1]),
            (1.0 - beta_mean[1]) / np.sqrt(beta_variance[1]),
            loc=beta_mean[1],
            scale=np.sqrt(beta_variance[1]),
        )
        phi_prior = 2.0 * beta.ppf(QUANTILES, prior["phi_alpha"], prior["phi_beta"]) - 1.0
        v_upper = prior_configurations[factor_index]["v_upper"]
        v_rate = prior_configurations[factor_index]["v_rate"]
        v_prior = -np.log(1.0 - QUANTILES * (1.0 - np.exp(-v_rate * v_upper))) / v_rate
        prior_quantiles = np.column_stack((b_prior.ppf(QUANTILES), phi_prior, v_prior))

        row = {
            "origin": ORIGIN,
            "presentation_model": "M4",
            "saved_model_label": "M5",
            "factor": factor,
            "posterior_draws": int(pooled.shape[0]),
        }
        for parameter_index, parameter in enumerate(("b", "phi", "v")):
            for quantile_index, suffix in enumerate(("q05", "median", "q95")):
                row[f"{parameter}_{suffix}"] = float(
                    posterior_quantiles[quantile_index, parameter_index]
                )
                row[f"prior_{parameter}_{suffix}"] = float(
                    prior_quantiles[quantile_index, parameter_index]
                )
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = (
        ROOT / "results" / RUN_NAME / "rolling_main" / "analysis"
        / f"posterior_parameter_summary_{ORIGIN}.csv"
    )
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    print(summarize(args.output.resolve()))


if __name__ == "__main__":
    main()
