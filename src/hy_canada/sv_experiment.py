"""Fit the non-centered SVNS model on the same Canadian training window."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import arviz as az
import numpy as np

from .data import load_panel
from .dlns import NAMES, training_prior
from .experiment import OUT, ROOT
from .svns import SV_NAMES, sample_chain, sv_prior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    dates, y = load_panel(ROOT / "data" / "processed" / "canada_weekly.csv",
                          "2000-01-01", "2015-12-31")
    prior = training_prior(y)
    seeds = [51801, 51802, 51803, 51804]
    print(f"SVNS: {len(y)} observations, four chains, "
          f"{args.warmup} warmup and {args.draws} retained each", flush=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(sample_chain, y, prior, seed, args.warmup, args.draws)
                   for seed in seeds]
        chains = [future.result() for future in futures]
    draws = np.stack([chain["draws"] for chain in chains])
    paths = np.stack([chain["paths"] for chain in chains])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chains").mkdir(exist_ok=True)
    names = NAMES + SV_NAMES
    np.savez_compressed(OUT / "svns_parameter_draws.npz", draws=draws, names=names)
    np.savez_compressed(OUT / "chains" / "svns_log_volatility.npz", paths=paths)
    idata = az.from_dict({"posterior": {name: draws[:, :, i] for i, name in enumerate(names)}})
    summary = az.summary(idata, kind="all", round_to=6)
    summary.to_csv(OUT / "svns_summary.csv")
    center, scale = sv_prior(prior)
    details = {
        "sample_start": str(dates[0].date()), "sample_end": str(dates[-1].date()),
        "observations": len(y), "warmup": args.warmup, "draws_per_chain": args.draws,
        "seeds": seeds, "block_acceptance": [c["acceptance"].tolist() for c in chains],
        "ellipse_evaluations_per_update": [c["ellipse_evaluations_per_update"].tolist() for c in chains],
        "proposal_scales": [c["scales"].tolist() for c in chains],
        "chain_elapsed_seconds": [c["elapsed_seconds"] for c in chains],
        "chain_means": {name: draws[:, :, i].mean(axis=1).tolist() for i, name in enumerate(names)},
        "prior_center_transformed": center.tolist(), "prior_scale_transformed": scale.tolist(),
        "initial_state_mean": prior.initial_mean.tolist(),
        "initial_state_covariance": prior.initial_covariance.tolist(),
    }
    (OUT / "svns_run.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string())
    print("Elapsed seconds per chain:", details["chain_elapsed_seconds"])


if __name__ == "__main__":
    main()
