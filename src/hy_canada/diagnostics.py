"""Reproducible diagnostics for the exploratory DLNS chains."""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

from .dlns import NAMES, parameters
from .experiment import OUT
from .svns import SV_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dlns", "svns"], default="dlns")
    model = parser.parse_args().model
    names = NAMES if model == "dlns" else NAMES + SV_NAMES
    with np.load(OUT / f"{model}_parameter_draws.npz") as archive:
        draws = archive["draws"]
    details = json.loads((OUT / f"{model}_run.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(8731)
    z = rng.normal(size=(50000, len(names))) * details["prior_scale_transformed"] + details["prior_center_transformed"]
    prior = np.array([np.concatenate((m, p, q, r, [d])) for m, p, q, r, d in map(parameters, z)])
    if model == "svns":
        prior = np.column_stack((prior, expit(z[:, 18:21]), np.exp(z[:, 21:24])))
    summary = pd.read_csv(OUT / f"{model}_summary.csv", index_col=0)
    rows = []
    for i, name in enumerate(names):
        x = draws[:, :, i]
        lag1 = [np.corrcoef(chain[:-1], chain[1:])[0, 1] for chain in x]
        rows.append({
            "parameter": name, "prior_mean": prior[:, i].mean(),
            "prior_5pct": np.quantile(prior[:, i], 0.05),
            "prior_95pct": np.quantile(prior[:, i], 0.95),
            "posterior_mean": x.mean(), "posterior_5pct": np.quantile(x, 0.05),
            "posterior_95pct": np.quantile(x, 0.95),
            "r_hat": summary.loc[name, "r_hat"], "ess_bulk": summary.loc[name, "ess_bulk"],
            "ess_tail": summary.loc[name, "ess_tail"],
            "ess_bulk_per_summed_chain_second": summary.loc[name, "ess_bulk"] / sum(details["chain_elapsed_seconds"]),
            "max_chain_lag1_autocorrelation": max(lag1),
            **{f"chain_{j+1}_mean": value for j, value in enumerate(x.mean(axis=1))},
        })
    pd.DataFrame(rows).to_csv(OUT / f"{model}_diagnostics.csv", index=False)
    boundary = {
        "rho_above_0.99": {NAMES[i]: float(np.mean(draws[:, :, i] > 0.99)) for i in range(3, 6)},
        "decay_within_1_percent_of_lower_bound": float(np.mean(draws[:, :, 17] < 0.0695)),
        "decay_within_1_percent_of_upper_bound": float(np.mean(draws[:, :, 17] > 1.9805)),
    }
    if model == "svns":
        boundary["sv_phi_above_0.99"] = {names[i]: float(np.mean(draws[:, :, i] > 0.99))
                                         for i in range(18, 21)}
    (OUT / f"{model}_boundaries.json").write_text(json.dumps(boundary, indent=2) + "\n", encoding="utf-8")
    for start in range(0, len(names), 6):
        count = min(6, len(names) - start)
        fig, axes = plt.subplots(count, 2, figsize=(12, 2.2 * count), squeeze=False)
        for j in range(count):
            i = start + j
            for c in range(draws.shape[0]):
                axes[j, 0].plot(draws[c, :, i], linewidth=0.45, alpha=0.75)
                centered = draws[c, :, i] - draws[c, :, i].mean()
                acf = np.correlate(centered, centered, mode="full")[len(centered)-1:len(centered)+100]
                acf = acf / acf[0]
                axes[j, 1].plot(np.arange(len(acf)), acf, linewidth=0.7)
            axes[j, 0].set_title(names[i] + " trace")
            axes[j, 1].set_title(names[i] + " autocorrelation")
        fig.tight_layout()
        fig.savefig(OUT / f"{model}_trace_acf_{start+1:02d}-{start+count:02d}.png", dpi=110)
        plt.close(fig)
    print(pd.DataFrame(rows)[["parameter", "r_hat", "ess_bulk", "ess_tail"]].to_string(index=False))
    print("Boundary diagnostics:", boundary)


if __name__ == "__main__":
    main()
