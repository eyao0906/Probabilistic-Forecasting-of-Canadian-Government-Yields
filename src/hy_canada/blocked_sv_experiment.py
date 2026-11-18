"""Run four diagnostic chains of the corrected blocked SVNS sampler."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .blocked_dlns import training_priors
from .blocked_experiment import OUT, ROOT
from .blocked_svns import PARAMETER_NAMES, sample_chain
from .data import load_panel


def diagnostic_details(variables, grid):
    lag1 = {name: [float(np.corrcoef(c[:-1], c[1:])[0, 1])
                    if np.std(c[:-1]) > 0 and np.std(c[1:]) > 0 else None for c in arr]
            for name, arr in variables.items()}
    boundaries = {name: [float(np.mean(c > .99)) for c in arr]
                  for name, arr in variables.items()
                  if name.startswith("rho_") or name.startswith("h_phi_")}
    boundaries["decay_at_grid_endpoints"] = [
        float(np.mean((c == grid[0]) | (c == grid[-1]))) for c in variables["decay"]]
    return {"chain_lag1_autocorrelation": lag1, "boundary_fractions": boundaries}


def plot_traces(variables, target):
    for start in range(0, len(PARAMETER_NAMES), 6):
        names = PARAMETER_NAMES[start:start + 6]
        fig, axes = plt.subplots(len(names), 2, figsize=(12, 2.2 * len(names)), squeeze=False)
        for row, name in enumerate(names):
            for chain in variables[name]:
                axes[row, 0].plot(chain, linewidth=0.5)
                centered = chain - chain.mean()
                acf = np.correlate(centered, centered, mode="full")[len(chain)-1:len(chain)+100]
                if acf[0] > 0:
                    axes[row, 1].plot(acf / acf[0], linewidth=0.7)
            axes[row, 0].set_title(name + " trace")
            axes[row, 1].set_title(name + " autocorrelation")
        fig.tight_layout()
        fig.savefig(target / f"trace_{start + 1:02d}-{start + len(names):02d}.png", dpi=100)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=2001)
    parser.add_argument("--seed-base", type=int, default=77101)
    parser.add_argument("--offset", type=float, default=0.001)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--label", default="pilot")
    args = parser.parse_args()
    if not args.label.isidentifier() or args.grid_size < 2:
        raise ValueError("Label must be an identifier and grid must contain at least two points")
    dates, y = load_panel(ROOT / "data" / "processed" / "canada_weekly.csv",
                          "2000-01-01", "2015-12-31")
    prior = training_priors(y)
    grid = np.linspace(0.2, 1.2, args.grid_size)
    starts = [0.45, 0.61, 0.67, 0.80]
    seeds = [args.seed_base + i for i in range(4)]
    print(f"Blocked SVNS {args.label}: {len(y)} observations, 4 chains, "
          f"{args.warmup} warmup + {args.draws} retained, offset={args.offset}", flush=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(sample_chain, y, prior, seed, args.warmup, args.draws,
                                   grid, min(20, args.draws), start, args.offset, args.inner_steps)
                   for seed, start in zip(seeds, starts)]
        chains = [future.result() for future in futures]
    draws = np.stack([chain["draws"] for chain in chains])
    target = OUT / "blocked_svns" / args.label
    target.mkdir(parents=True, exist_ok=True)
    (OUT / "chains").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "chains" / f"blocked_svns_{args.label}.npz",
                        draws=draws, paths=np.stack([chain["paths"] for chain in chains]),
                        volatility_paths=np.stack([chain["volatility_paths"] for chain in chains]),
                        path_indices=chains[0]["path_indices"], names=PARAMETER_NAMES, grid=grid,
                        correction_logratios=np.stack([chain["logratios"] for chain in chains]))
    variables = {name: draws[:, :, i] for i, name in enumerate(PARAMETER_NAMES)}
    summary = az.summary(az.from_dict({"posterior": variables}), round_to=6)
    summary.to_csv(target / "summary.csv")
    details = {
        "sample_start": str(dates[0].date()), "sample_end": str(dates[-1].date()),
        "observations": len(y), "warmup": args.warmup, "draws_per_chain": args.draws,
        "seeds": seeds, "grid_size": args.grid_size, "grid_bounds": [0.2, 1.2],
        "initial_decay_requested": starts, "offset": args.offset, "inner_steps": args.inner_steps,
        "chain_elapsed_seconds": [chain["elapsed_seconds"] for chain in chains],
        "chain_means": {name: arr.mean(axis=1).tolist() for name, arr in variables.items()},
        "decay_transitions": [int(np.count_nonzero(np.diff(chain))) for chain in variables["decay"]],
        "parameter_proposal_acceptance": [(chain["theta_accepts"] / (args.draws * args.inner_steps)).tolist()
                                          for chain in chains],
        "correction_acceptance": [(chain["corrections"] / (args.draws * args.inner_steps)).tolist()
                                  for chain in chains],
        "correction_logratio_quantiles": np.quantile(
            np.stack([chain["logratios"] for chain in chains]), [0, .05, .5, .95, 1],
            axis=(0, 1)).tolist(),
    }
    details.update(diagnostic_details(variables, grid))
    (target / "run.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n",
                                      encoding="utf-8")
    plot_traces(variables, target)
    print(summary.to_string())
    print("Decay transitions:", details["decay_transitions"])
    print("Parameter proposal acceptance:", details["parameter_proposal_acceptance"])
    print("Correction acceptance:", details["correction_acceptance"])
    print("Seconds per chain:", details["chain_elapsed_seconds"])


if __name__ == "__main__":
    main()
