"""Numerically stable Nelson-Siegel loading calculations."""

from __future__ import annotations

import numpy as np


def dns_loadings(maturities_years: np.ndarray, decay_per_year: float) -> np.ndarray:
    """Return the (maturity, level/slope/curvature) Nelson-Siegel loading matrix."""
    tau = np.asarray(maturities_years, dtype=float)
    decay = float(decay_per_year)
    if tau.ndim != 1 or tau.size == 0:
        raise ValueError("maturities_years must be a nonempty one-dimensional array")
    if not np.isfinite(tau).all() or np.any(tau < 0):
        raise ValueError("maturities_years must be finite and nonnegative")
    if not np.isfinite(decay) or decay <= 0:
        raise ValueError("decay_per_year must be finite and positive")

    x = decay * tau
    slope = np.ones_like(x)
    np.divide(-np.expm1(-x), x, out=slope, where=x != 0)
    curvature = slope - np.exp(-x)
    return np.column_stack((np.ones_like(tau), slope, curvature))
