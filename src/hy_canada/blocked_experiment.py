"""Run four blocked DLNS chains and save diagnostic results."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .blocked_dlns import PARAMETER_NAMES, decay_probabilities, sample_chain, training_priors
from .data import load_panel

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "hy_canada"


def plot_decay_conditionals(conditional, grid, target):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for chain, ax in enumerate(axes, start=1):
        row = next(item for item in conditional if item["chain"] == chain)
        probabilities = [row[f"{value:.8f}"] for value in grid]
        ax.plot(grid, probabilities)
        ax.set_ylabel(f"Chain {chain}")
    axes[-1].set_xlabel("Decay per year")
    fig.suptitle("Conditional decay probabilities for one saved factor path per chain")
    fig.tight_layout()
    fig.savefig(target / "conditional_decay.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=63101)
    parser.add_argument("--label", default="pilot")
    args = parser.parse_args()
    if not args.label.isidentifier():
        raise ValueError("Label must be an identifier")
    if args.grid_size < 2:
        raise ValueError("Decay grid must have at least two points")
    dates, y = load_panel(ROOT / "data" / "processed" / "canada_weekly.csv",
                          "2000-01-01", "2015-12-31")
    prior = training_priors(y)
    grid = np.linspace(0.2, 1.2, args.grid_size)
    seeds = [args.seed_base + i for i in range(4)]
    starts = [0.45, 0.61, 0.67, 0.80]
    keep_paths = min(20, args.draws)
    print(f"Blocked DLNS {args.label}: {len(y)} observations, 4 chains, "
          f"{args.warmup} warmup + {args.draws} retained, {len(grid)} grid points", flush=True)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(sample_chain, y, prior, seed, args.warmup,
                                   args.draws, grid, keep_paths, start)
                   for seed, start in zip(seeds, starts)]
        chains = [future.result() for future in futures]
    draws = np.stack([c["draws"] for c in chains])
    paths = np.stack([c["paths"] for c in chains])
    target = OUT / "blocked_dlns" / args.label
    target.mkdir(parents=True, exist_ok=True)
    (OUT / "chains").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "chains" / f"blocked_dlns_{args.label}.npz",
                        draws=draws, paths=paths, path_indices=chains[0]["path_indices"],
                        names=PARAMETER_NAMES, grid=grid)
    variables = {name: draws[:, :, i] for i, name in enumerate(PARAMETER_NAMES)}
    for label, i, j in [("corr_LS", 0, 1), ("corr_LC", 0, 2), ("corr_SC", 1, 2)]:
        offset = {"corr_LS": 9, "corr_LC": 10, "corr_SC": 11}[label]
        variables[label] = draws[:, :, offset] / np.sqrt(draws[:, :, 6+i] * draws[:, :, 6+j])
    summary = az.summary(az.from_dict({"posterior": variables}), round_to=6)
    summary.to_csv(target / "summary.csv")
    lag1 = {name: [float(np.corrcoef(c[:-1], c[1:])[0, 1])
                    if np.std(c[:-1]) > 0 and np.std(c[1:]) > 0 else None for c in arr]
            for name, arr in variables.items()}
    details = {
        "sample_start": str(dates[0].date()), "sample_end": str(dates[-1].date()),
        "observations": len(y), "warmup": args.warmup, "draws_per_chain": args.draws,
        "seeds": seeds, "grid": grid.tolist(),
        "initial_decay_requested": starts,
        "chain_elapsed_seconds": [c["elapsed_seconds"] for c in chains],
        "chain_means": {name: arr.mean(axis=1).tolist() for name, arr in variables.items()},
        "chain_lag1_autocorrelation": lag1,
        "decay_transitions": [int(np.count_nonzero(np.diff(c))) for c in variables["decay"]],
        "decay_below_0_53": [float(np.mean(c < 0.53)) for c in variables["decay"]],
        "decay_at_grid_endpoints": [float(np.mean((c == grid[0]) | (c == grid[-1])))
                                    for c in variables["decay"]],
        "initial_state_mean": prior.m0.tolist(), "initial_state_covariance": prior.p0.tolist(),
        "mu_prior_mean": prior.mu_mean.tolist(), "mu_prior_variance": prior.mu_variance.tolist(),
        "rho_prior_mean": prior.rho_mean, "rho_prior_variance": prior.rho_variance,
        "measurement_ig_shape": prior.r_shape, "measurement_ig_scale": prior.r_scale.tolist(),
        "innovation_iw_df": prior.q_df, "innovation_iw_scale": prior.q_scale.tolist(),
    }
    (target / "run.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    conditional = []
    for c in range(4):
        for k, index in enumerate(chains[c]["path_indices"]):
            conditional.append({"chain": c + 1, "draw": int(index),
                                **{f"{value:.8f}": float(p) for value, p in zip(
                                    grid, decay_probabilities(y, paths[c, k],
                                                              draws[c, index, 12:20], grid))}})
    pd.DataFrame(conditional).to_csv(target / "conditional_decay_probabilities.csv", index=False)
    plot_decay_conditionals(conditional, grid, target)
    for start in range(0, len(PARAMETER_NAMES), 6):
        names = PARAMETER_NAMES[start:start+6]
        fig, axes = plt.subplots(len(names), 2, figsize=(12, 2.2 * len(names)), squeeze=False)
        for row, name in enumerate(names):
            arr = variables[name]
            for c in range(4):
                axes[row, 0].plot(arr[c], linewidth=0.4)
                centered = arr[c] - arr[c].mean()
                acf = np.correlate(centered, centered, mode="full")[len(centered)-1:len(centered)+100]
                if acf[0] > 0:
                    axes[row, 1].plot(acf / acf[0], linewidth=0.7)
            axes[row, 0].set_title(name + " trace")
            axes[row, 1].set_title(name + " autocorrelation")
        fig.tight_layout()
        fig.savefig(target / f"trace_{start+1:02d}-{start+len(names):02d}.png", dpi=100)
        plt.close(fig)
    print(summary.to_string())
    print("Decay transitions:", details["decay_transitions"])
    print("Seconds per chain:", details["chain_elapsed_seconds"])


if __name__ == "__main__":
    main()
