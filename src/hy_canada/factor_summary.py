"""Exploratory conditional state draws, explicitly not convergence evidence."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import MATURITIES, load_panel
from .dlns import training_prior
from .experiment import OUT, ROOT
from .kalman import loadings
from .states import sample_factors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dlns", "svns"], default="dlns")
    parser.add_argument("--per-chain", type=int, default=25)
    args = parser.parse_args()
    dates, y = load_panel(ROOT / "data" / "processed" / "canada_weekly.csv",
                          "2000-01-01", "2015-12-31")
    prior = training_prior(y)
    with np.load(OUT / f"{args.model}_parameter_draws.npz") as archive:
        draws = archive["draws"]
    if not 1 <= args.per_chain <= draws.shape[1]:
        raise ValueError("Invalid number of conditional draws per chain")
    if args.model == "svns":
        with np.load(OUT / "chains" / "svns_log_volatility.npz") as archive:
            paths = archive["paths"]
    indices = np.linspace(0, draws.shape[1] - 1, args.per_chain, dtype=int)
    rng = np.random.default_rng(63901)
    states = np.empty((len(draws), len(indices), len(y), 3))
    for chain in range(len(draws)):
        for k, i in enumerate(indices):
            param = draws[chain, i]
            h = loadings(MATURITIES, param[17])
            q = np.tile(param[6:9], (len(y), 1)) if args.model == "dlns" else np.exp(paths[chain, i])
            states[chain, k] = sample_factors(y, h, param[:3], param[3:6], q, param[9:17],
                                               prior.initial_mean, prior.initial_covariance, rng)
    np.savez_compressed(OUT / "chains" / f"{args.model}_conditional_factors.npz", states=states,
                        draw_indices=indices)
    ols_h = loadings(MATURITIES, 0.7)
    ols = np.linalg.lstsq(ols_h, y.T, rcond=None)[0].T
    pooled = states.reshape(-1, len(y), 3)
    data = {"date": dates.strftime("%Y-%m-%d")}
    for j, label in enumerate("LSC"):
        data[f"ols_{label}"] = ols[:, j]
        data[f"exploratory_mean_{label}"] = pooled[:, :, j].mean(axis=0)
        data[f"exploratory_q05_{label}"] = np.quantile(pooled[:, :, j], 0.05, axis=0)
        data[f"exploratory_q95_{label}"] = np.quantile(pooled[:, :, j], 0.95, axis=0)
        for chain in range(len(draws)):
            data[f"chain{chain+1}_mean_{label}"] = states[chain, :, :, j].mean(axis=0)
    pd.DataFrame(data).to_csv(OUT / f"{args.model}_exploratory_factor_summary.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for j, label in enumerate("LSC"):
        axes[j].plot(dates, ols[:, j], label="OLS at decay 0.7", linewidth=0.8)
        axes[j].plot(dates, pooled[:, :, j].mean(axis=0), label="Exploratory joint", linewidth=0.8)
        axes[j].fill_between(dates, np.quantile(pooled[:, :, j], 0.05, axis=0),
                             np.quantile(pooled[:, :, j], 0.95, axis=0), alpha=0.2)
        axes[j].set_title(f"{label}: unreliable exploratory draws")
        axes[j].legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / f"{args.model}_exploratory_factors.png", dpi=110)
    plt.close(fig)
    print(f"Saved exploratory {args.model} summaries from {len(indices)} draws per chain.")


if __name__ == "__main__":
    main()
