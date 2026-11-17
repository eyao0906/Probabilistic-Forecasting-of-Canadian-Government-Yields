"""Experimental non-centered SV moves for conditional synthetic screening."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def _standardize(h: np.ndarray, mean: float, phi: float, variance: float) -> np.ndarray:
    center = mean - variance / 2
    z = np.empty_like(h)
    z[0] = (h[0] - center) / np.sqrt(variance)
    z[1:] = (h[1:] - center - phi * (h[:-1] - center)) / np.sqrt(variance * (1 - phi**2))
    return z


def _reconstruct(z: np.ndarray, mean: float, phi: float, variance: float) -> np.ndarray:
    center = mean - variance / 2
    h = np.empty_like(z)
    h[0] = center + np.sqrt(variance) * z[0]
    sd = np.sqrt(variance * (1 - phi**2))
    for t in range(1, len(z)):
        h[t] = center + phi * (h[t - 1] - center) + sd * z[t]
    return h


def _innovation_log_likelihood(h: np.ndarray, innovations: np.ndarray) -> float:
    if not np.isfinite(h).all() or np.max(np.abs(h)) > 700:
        return -np.inf
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value = -0.5 * np.sum(h + innovations**2 * np.exp(-h))
    return float(value) if np.isfinite(value) else -np.inf


def interweave_persistence(
    h: np.ndarray, innovations: np.ndarray, mean: float, phi: float,
    variance: float, beta_a: float, beta_b: float, rng: np.random.Generator,
    step: float = .12,
) -> tuple[np.ndarray, float, bool]:
    z = _standardize(h, mean, phi, variance)
    proposal = float(np.tanh(np.arctanh(phi) + rng.normal(0., step)))
    if not -1. < proposal < 1.:
        return h, phi, False
    proposed_h = _reconstruct(z, mean, proposal, variance)

    def log_prior_and_jacobian(p: float) -> float:
        x = (p + 1) / 2
        return float((beta_a - 1) * np.log(x) + (beta_b - 1) * np.log1p(-x) + np.log1p(-p**2))

    ratio = (
        _innovation_log_likelihood(proposed_h, innovations)
        - _innovation_log_likelihood(h, innovations)
        + log_prior_and_jacobian(proposal) - log_prior_and_jacobian(phi)
    )
    if np.log(rng.uniform()) < ratio:
        return proposed_h, proposal, True
    return h, phi, False


def interweave_variance(
    h: np.ndarray, innovations: np.ndarray, mean: float, phi: float,
    variance: float, upper: float, rng: np.random.Generator, step: float = .16,
) -> tuple[np.ndarray, float, bool]:
    z = _standardize(h, mean, phi, variance)
    fraction = variance / upper
    proposal_fraction = expit(np.log(fraction / (1 - fraction)) + rng.normal(0., step))
    if not 0. < proposal_fraction < 1.:
        return h, variance, False
    proposal = upper * proposal_fraction
    proposed_h = _reconstruct(z, mean, phi, proposal)

    def log_prior_and_jacobian(v: float, x: float) -> float:
        return float(-v + np.log(x) + np.log1p(-x))

    ratio = (
        _innovation_log_likelihood(proposed_h, innovations)
        - _innovation_log_likelihood(h, innovations)
        + log_prior_and_jacobian(proposal, proposal_fraction)
        - log_prior_and_jacobian(variance, fraction)
    )
    if np.log(rng.uniform()) < ratio:
        return proposed_h, float(proposal), True
    return h, variance, False
