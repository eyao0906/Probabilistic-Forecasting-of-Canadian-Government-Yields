"""Exploratory SV path summaries and selected-date convergence diagnostics."""

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_panel
from .experiment import OUT, ROOT


def main():
    dates, _ = load_panel(ROOT / "data" / "processed" / "canada_weekly.csv",
                          "2000-01-01", "2015-12-31")
    with np.load(OUT / "chains" / "svns_log_volatility.npz") as archive:
        paths = archive["paths"]
    pooled = paths.reshape(-1, len(dates), 3)
    data = {"date": dates.strftime("%Y-%m-%d")}
    posterior = {}
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for j, label in enumerate("LSC"):
        q05, median, q95 = np.quantile(pooled[:, :, j], [0.05, 0.5, 0.95], axis=0)
        data[f"logvar_{label}_median"] = median
        data[f"logvar_{label}_q05"] = q05
        data[f"logvar_{label}_q95"] = q95
        for chain in range(paths.shape[0]):
            data[f"logvar_{label}_chain{chain+1}_mean"] = paths[chain, :, :, j].mean(axis=0)
        for t in (0, len(dates) // 2, len(dates) - 1):
            posterior[f"logvar_{label}_{dates[t].strftime('%Y%m%d')}"] = paths[:, :, t, j]
        axes[j].plot(dates, median, linewidth=0.8)
        axes[j].fill_between(dates, q05, q95, alpha=0.25)
        axes[j].set_title(f"{label} log innovation variance: unreliable exploratory draws")
    pd.DataFrame(data).to_csv(OUT / "svns_exploratory_volatility_summary.csv", index=False)
    az.summary(az.from_dict({"posterior": posterior}), round_to=6).to_csv(
        OUT / "svns_selected_volatility_diagnostics.csv")
    fig.tight_layout()
    fig.savefig(OUT / "svns_exploratory_volatility.png", dpi=110)
    plt.close(fig)
    print("Saved exploratory volatility trajectories and selected-date diagnostics.")


if __name__ == "__main__":
    main()
