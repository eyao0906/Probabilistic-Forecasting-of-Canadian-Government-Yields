"""Collapsed Metropolis inference for the constant-volatility joint model."""

from dataclasses import dataclass
import time

import numpy as np
from scipy.special import expit, logit

from .data import MATURITIES
from .kalman import filter_likelihood, loadings

NAMES = (
    [f"mu_{x}" for x in "LSC"]
    + [f"rho_{x}" for x in "LSC"]
    + [f"q_{x}" for x in "LSC"]
    + [f"r_{x:g}" for x in MATURITIES]
    + ["decay"]
)
BLOCKS = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 17), slice(17, 18))


@dataclass(frozen=True)
class TrainingPrior:
    center: np.ndarray
    scale: np.ndarray
    initial_mean: np.ndarray
    initial_covariance: np.ndarray


def training_prior(y: np.ndarray) -> TrainingPrior:
    """Set empirical prior scales using this training window only."""
    h = loadings(MATURITIES, 0.7)
    factors = np.linalg.lstsq(h, y.T, rcond=None)[0].T
    residuals = y - factors @ h.T
    mean = factors.mean(axis=0)
    factor_sd = np.maximum(factors.std(axis=0, ddof=1), 0.1)
    centered = factors - mean
    rho = np.sum(centered[1:] * centered[:-1], axis=0) / np.sum(centered[:-1] ** 2, axis=0)
    rho = np.clip(rho, 0.7, 0.995)
    innovations = centered[1:] - rho * centered[:-1]
    q = np.maximum(innovations.var(axis=0, ddof=1), 1e-5)
    r = np.maximum(residuals.var(axis=0, ddof=1), 1e-5)
    center = np.r_[mean, np.full(3, logit(0.95)), np.log(q), np.log(r),
                   logit((0.7 - 0.05) / 1.95)]
    scale = np.r_[2 * factor_sd, np.full(3, 1.25), np.ones(3),
                  np.full(8, 1.5), 1.0]
    return TrainingPrior(center, scale, mean, np.diag((2 * factor_sd) ** 2))


def parameters(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    return z[:3], expit(z[3:6]), np.exp(z[6:9]), np.exp(z[9:17]), 0.05 + 1.95 * expit(z[17])


def log_posterior(z: np.ndarray, y: np.ndarray, prior: TrainingPrior) -> float:
    if not np.isfinite(z).all() or np.any(np.abs(z[6:17]) > 50):
        return -np.inf
    mu, rho, q, r, decay = parameters(z)
    q_t = np.broadcast_to(q, (len(y), 3)).copy()
    ll = filter_likelihood(y, loadings(MATURITIES, decay), mu, rho, q_t, r,
                           prior.initial_mean, prior.initial_covariance)
    return float(ll - 0.5 * np.sum(((z - prior.center) / prior.scale) ** 2))


def sample_chain(y: np.ndarray, prior: TrainingPrior, seed: int,
                 warmup: int, draws: int, joint_cholesky: np.ndarray | None = None) -> dict:
    if warmup < 0 or draws < 1:
        raise ValueError("Warmup must be nonnegative and draws positive")
    rng = np.random.default_rng(seed)
    z = prior.center + rng.normal(size=18) * prior.scale * 0.15
    current = log_posterior(z, y, prior)
    if not np.isfinite(current):
        raise ValueError("Nonfinite initial log posterior")
    scales = np.array([0.10, 0.16, 0.08, 0.06, 0.04]
                      + ([0.25] if joint_cholesky is not None else []))
    block_units = [prior.scale[b] for b in BLOCKS]
    accepted = np.zeros(len(scales), dtype=int)
    window = np.zeros(len(scales), dtype=int)
    stored = np.empty((draws, 18))
    start = time.perf_counter()
    for iteration in range(warmup + draws):
        for i, block in enumerate(BLOCKS):
            proposal = z.copy()
            proposal[block] += rng.normal(size=block_units[i].size) * block_units[i] * scales[i]
            candidate = log_posterior(proposal, y, prior)
            if np.log(rng.random()) < candidate - current:
                z, current = proposal, candidate
                if iteration >= warmup:
                    accepted[i] += 1
                else:
                    window[i] += 1
        if joint_cholesky is not None:
            proposal = z.copy()
            proposal[6:18] += scales[5] * (joint_cholesky @ rng.normal(size=12))
            candidate = log_posterior(proposal, y, prior)
            if np.log(rng.random()) < candidate - current:
                z, current = proposal, candidate
                if iteration >= warmup:
                    accepted[5] += 1
                else:
                    window[5] += 1
        if iteration < warmup and (iteration + 1) % 50 == 0:
            scales *= np.exp(np.clip(window / 50 - 0.25, -0.25, 0.25))
            window[:] = 0
        if iteration >= warmup:
            mu, rho, q, r, decay = parameters(z)
            stored[iteration - warmup] = np.r_[mu, rho, q, r, decay]
    return {"draws": stored, "acceptance": accepted / draws, "scales": scales,
            "elapsed_seconds": time.perf_counter() - start, "seed": seed}
