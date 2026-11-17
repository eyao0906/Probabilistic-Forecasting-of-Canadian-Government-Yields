"""Training-only calibration for proper, matched state-space DNS priors."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .loadings import dns_loadings


@dataclass(frozen=True)
class PriorSettings:
    """All hyperparameters fixed from the initial training window."""

    factor_mean_center: np.ndarray
    factor_mean_sd: np.ndarray
    persistence_center: np.ndarray
    persistence_sd: np.ndarray
    log_variance_center: np.ndarray
    log_variance_sd: np.ndarray
    measurement_variance_scale: np.ndarray
    decay_lower: float
    decay_upper: float
    decay_initial: float
    measurement_shape: float = 3.0
    initial_state_variance: float = 1e6
    sv_phi_beta_a: float = 20.0
    sv_phi_beta_b: float = 1.5
    sv_variance_upper: float = 4.0

    def as_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        values = asdict(self)
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in values.items()
        }


def calibrate_training_priors(
    yields: np.ndarray,
    maturities_years: np.ndarray,
    decay_lower: float = 0.05,
    decay_upper: float = 2.0,
    decay_grid_size: int = 160,
) -> tuple[PriorSettings, np.ndarray]:
    """Calibrate shared hyperparameters using only the supplied training data."""
    y = np.asarray(yields, dtype=float)
    tau = np.asarray(maturities_years, dtype=float)
    if y.ndim != 2 or y.shape[0] < 4 or y.shape[1] != tau.size:
        raise ValueError("yields must be T by maturity and match maturities_years")
    if not np.isfinite(y).all() or not np.isfinite(tau).all():
        raise ValueError("training yields and maturities must be finite")
    if not np.isfinite(decay_lower) or not np.isfinite(decay_upper) or not (
        0 < decay_lower < decay_upper
    ):
        raise ValueError("decay bounds must be finite with 0 < lower < upper")
    if decay_grid_size < 2:
        raise ValueError("decay_grid_size must be at least two")

    grid = np.linspace(decay_lower, decay_upper, decay_grid_size)
    best_loss = np.inf
    best_decay = np.nan
    best_loadings = None
    best_factors = None
    for decay in grid:
        loadings = dns_loadings(tau, float(decay))
        factors = np.linalg.lstsq(loadings, y.T, rcond=None)[0].T
        residual = y - factors @ loadings.T
        loss = float(np.sum(residual * residual))
        if loss < best_loss:
            best_loss, best_decay = loss, float(decay)
            best_loadings, best_factors = loadings, factors

    if best_loadings is None or best_factors is None:
        raise RuntimeError("training-only decay search did not produce a candidate")
    factor_sd = best_factors.std(axis=0, ddof=1)
    if np.any(~np.isfinite(factor_sd)) or np.any(factor_sd <= 0):
        raise ValueError("training factor scales must be positive and finite")

    persistence_center = np.empty(3)
    log_variance_center = np.empty(3)
    for j in range(3):
        lag = best_factors[:-1, j]
        target = best_factors[1:, j]
        design = np.column_stack((np.ones(len(lag)), lag))
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = target - design @ coefficients
        persistence_center[j] = coefficients[1]
        log_variance_center[j] = np.log(max(float(np.mean(residual**2)), 1e-10))

    cross_sectional_residual = y - best_factors @ best_loadings.T
    measurement_scale = np.maximum(np.mean(cross_sectional_residual**2, axis=0), 1e-6)
    priors = PriorSettings(
        factor_mean_center=best_factors.mean(axis=0),
        factor_mean_sd=2.0 * factor_sd,
        persistence_center=persistence_center,
        persistence_sd=np.full(3, 0.03),
        log_variance_center=log_variance_center,
        log_variance_sd=np.ones(3),
        measurement_variance_scale=measurement_scale,
        decay_lower=float(decay_lower),
        decay_upper=float(decay_upper),
        decay_initial=best_decay,
    )
    return priors, best_factors
