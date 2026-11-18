"""Training-only OLS geometry and available forecast target dates."""

import json

import numpy as np
import pandas as pd

from .data import MATURITIES, load_panel
from .blocked_experiment import OUT, ROOT
from .kalman import loadings


def main():
    dates, y = load_panel(ROOT / "data" / "processed" / "canada_weekly.csv",
                          "2000-01-01", "2015-12-31")
    h = loadings(MATURITIES, 0.7)
    factors = np.linalg.lstsq(h, y.T, rcond=None)[0].T
    residuals = y - factors @ h.T
    corr = np.corrcoef(residuals.T)
    off_diagonal = corr[np.triu_indices(8, 1)]
    eig = np.linalg.eigvalsh(np.cov(residuals.T))
    acf = [np.corrcoef(residuals[:-1, i], residuals[1:, i])[0, 1] for i in range(8)]
    OUT.mkdir(parents=True, exist_ok=True)
    labels = [f"{m:g}" for m in MATURITIES]
    pd.DataFrame(corr, index=labels, columns=labels).to_csv(OUT / "ols_residual_correlations.csv")
    result = {
        "decay_per_year": 0.7, "observations": len(y),
        "residual_correlation_min": float(off_diagonal.min()),
        "residual_correlation_max": float(off_diagonal.max()),
        "residual_lag1_autocorrelation": dict(zip(labels, map(float, acf))),
        "residual_covariance_eigenvalues": eig.tolist(),
        "maximum_orthogonality_error": float(np.max(np.abs(residuals @ h))),
        "factor_means": factors.mean(axis=0).tolist(),
    }
    (OUT / "ols_descriptive.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    all_dates = pd.to_datetime(pd.read_csv(ROOT / "data" / "processed" / "canada_weekly.csv")["Date"])
    rows = []
    for origin in ["2016-01-08", "2020-03-27", "2025-11-28"]:
        matches = np.flatnonzero(all_dates == pd.Timestamp(origin))
        if len(matches) != 1 or matches[0] + 4 >= len(all_dates):
            raise ValueError(f"Origin or target dates unavailable: {origin}")
        rows.append({"origin": origin, "one_observation_target": str(all_dates.iloc[matches[0] + 1].date()),
                     "four_observation_target": str(all_dates.iloc[matches[0] + 4].date())})
    pd.DataFrame(rows).to_csv(OUT / "forecast_target_dates.csv", index=False)
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
