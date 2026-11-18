"""Run four independent exploratory DLNS chains on the existing weekly panel."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import arviz as az
import numpy as np

from .data import load_panel
from .dlns import NAMES, sample_chain, training_prior

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "hy_canada"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed-base", type=int, default=41701)
    parser.add_argument("--joint-proposal", type=Path,
                        help="JSON file containing a fixed 12x12 proposal covariance")
    args = parser.parse_args()
    dates, y = load_panel(ROOT / "data" / "processed" / "canada_weekly.csv",
                          "2000-01-01", "2015-12-31")
    prior = training_prior(y)
    joint_cholesky = None
    if args.joint_proposal:
        covariance = np.array(json.loads(args.joint_proposal.read_text(encoding="utf-8")))
        if covariance.shape != (12, 12) or not np.isfinite(covariance).all():
            raise ValueError("Joint proposal covariance must be finite and 12x12")
        joint_cholesky = np.linalg.cholesky(covariance)
    seeds = [args.seed_base + i for i in range(4)]
    print(f"DLNS: {len(y)} observations, four chains, "
          f"{args.warmup} warmup and {args.draws} retained each", flush=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(sample_chain, y, prior, seed, args.warmup, args.draws, joint_cholesky)
                   for seed in seeds]
        chains = [future.result() for future in futures]
    draws = np.stack([chain["draws"] for chain in chains])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chains").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "dlns_parameter_draws.npz", draws=draws, names=NAMES)
    idata = az.from_dict({"posterior": {name: draws[:, :, i] for i, name in enumerate(NAMES)}})
    summary = az.summary(idata, kind="all", round_to=6)
    summary.to_csv(OUT / "dlns_summary.csv")
    chain_means = {name: draws[:, :, i].mean(axis=1).tolist() for i, name in enumerate(NAMES)}
    details = {
        "sample_start": str(dates[0].date()), "sample_end": str(dates[-1].date()),
        "observations": len(y), "warmup": args.warmup, "draws_per_chain": args.draws,
        "seeds": seeds, "block_acceptance": [c["acceptance"].tolist() for c in chains],
        "proposal_scales": [c["scales"].tolist() for c in chains],
        "chain_elapsed_seconds": [c["elapsed_seconds"] for c in chains],
        "chain_means": chain_means,
        "prior_center_transformed": prior.center.tolist(),
        "prior_scale_transformed": prior.scale.tolist(),
        "initial_state_mean": prior.initial_mean.tolist(),
        "initial_state_covariance": prior.initial_covariance.tolist(),
        "joint_proposal": str(args.joint_proposal) if args.joint_proposal else None,
    }
    (OUT / "dlns_run.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string())
    print("Elapsed seconds per chain:", details["chain_elapsed_seconds"])


if __name__ == "__main__":
    main()
