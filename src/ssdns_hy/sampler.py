"""Exact-target blocked MCMC for the matched SS_M3 and SS_M4 models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import truncnorm

from .kalman import FilterResult, kalman_filter
from .loadings import dns_loadings
from .model import factor_dynamics, factor_innovation_covariance, stochastic_innovation_covariance
from .priors import PriorSettings
from .smoother import sample_state_path


@dataclass(frozen=True)
class ChainDraws:
    """Retained parameter draws and posterior path summaries for one chain."""

    model: str
    factor_mean: np.ndarray
    persistence: np.ndarray
    log_variance: np.ndarray
    measurement_variance: np.ndarray
    decay_per_year: np.ndarray
    terminal_factor: np.ndarray
    terminal_log_variance: np.ndarray
    volatility_persistence: np.ndarray
    volatility_variance: np.ndarray
    factor_path_mean: np.ndarray
    log_volatility_path_mean: np.ndarray
    acceptance: dict[str, float]


def run_chains(
    yields: np.ndarray,
    maturities_years: np.ndarray,
    priors: PriorSettings,
    model: str,
    n_chains: int,
    warmup: int,
    draws: int,
    seed: int,
) -> list[ChainDraws]:
    """Run independent chains with reproducible, non-overlapping RNG streams."""
    if n_chains < 2:
        raise ValueError("n_chains must be at least two for convergence assessment")
    streams = np.random.SeedSequence(seed).spawn(n_chains)
    return [
        run_chain(
            yields,
            maturities_years,
            priors,
            model,
            np.random.default_rng(stream),
            warmup,
            draws,
        )
        for stream in streams
    ]


def _draw_initial_factor(
    first_factor: np.ndarray,
    mean: np.ndarray,
    persistence: np.ndarray,
    first_process_variance: np.ndarray,
    initial_variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw f_0 | f_1 for the proper diffuse initialization."""
    prior_variance = float(initial_variance)
    total_variance = persistence**2 * prior_variance + first_process_variance
    cross = persistence * prior_variance
    conditional_mean = mean + cross / total_variance * (first_factor - mean)
    conditional_variance = prior_variance * first_process_variance / total_variance
    return conditional_mean + np.sqrt(conditional_variance) * rng.standard_normal(3)


def _draw_factor_parameters(
    states: np.ndarray,
    initial_state: np.ndarray,
    process_variance: np.ndarray,
    mean: np.ndarray,
    persistence: np.ndarray,
    priors: PriorSettings,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    previous = np.vstack((initial_state, states[:-1]))
    updated_mean = mean.copy()
    updated_persistence = persistence.copy()
    for j in range(3):
        variance = process_variance[:, j]
        precision_weights = 1.0 / variance
        b = updated_persistence[j]
        mean_prior_precision = 1.0 / priors.factor_mean_sd[j] ** 2
        mean_precision = (
            mean_prior_precision
            + (1.0 - b) ** 2 * precision_weights.sum()
            + 1.0 / priors.initial_state_variance
        )
        mean_rhs = (
            mean_prior_precision * priors.factor_mean_center[j]
            + (1.0 - b)
            * np.sum(
                (states[:, j] - b * previous[:, j]) * precision_weights
            )
            + initial_state[j] / priors.initial_state_variance
        )
        updated_mean[j] = rng.normal(mean_rhs / mean_precision, np.sqrt(1.0 / mean_precision))

        regressor = previous[:, j] - updated_mean[j]
        response = states[:, j] - updated_mean[j]
        persistence_precision = (
            1.0 / priors.persistence_sd[j] ** 2
            + np.sum(regressor**2 * precision_weights)
        )
        persistence_rhs = (
            priors.persistence_center[j] / priors.persistence_sd[j] ** 2
            + np.sum(regressor * response * precision_weights)
        )
        persistence_sd = np.sqrt(1.0 / persistence_precision)
        persistence_mean = persistence_rhs / persistence_precision
        updated_persistence[j] = truncnorm.rvs(
            (-1.0 - persistence_mean) / persistence_sd,
            (1.0 - persistence_mean) / persistence_sd,
            loc=persistence_mean,
            scale=persistence_sd,
            random_state=rng,
        )
    return updated_mean, updated_persistence


def _factor_log_variance_target(
    value: float,
    residual: np.ndarray,
    prior_center: float,
    prior_sd: float,
) -> float:
    if not np.isfinite(value) or abs(value) > 700:
        return -np.inf
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        variance_inverse = np.exp(-value)
        result = -0.5 * (
            ((value - prior_center) / prior_sd) ** 2
            + len(residual) * value
            + float(residual @ residual) * variance_inverse
        )
    return float(result) if np.isfinite(result) else -np.inf


def _slice_sample(
    current: float,
    log_target,
    rng: np.random.Generator,
    width: float = 1.0,
    max_steps: int = 50,
) -> float:
    log_slice = log_target(current) + np.log(rng.uniform())
    left = current - width * rng.uniform()
    right = left + width
    left_steps = int(rng.integers(max_steps))
    right_steps = max_steps - 1 - left_steps
    while left_steps and log_target(left) > log_slice:
        left -= width
        left_steps -= 1
    while right_steps and log_target(right) > log_slice:
        right += width
        right_steps -= 1
    for _ in range(max_steps * 100):
        proposal = rng.uniform(left, right)
        if log_target(proposal) >= log_slice:
            return float(proposal)
        if proposal < current:
            left = proposal
        else:
            right = proposal
    raise RuntimeError("log-variance slice sampler failed to find an acceptable point")


def _draw_measurement_variance(
    yields: np.ndarray,
    states: np.ndarray,
    loadings: np.ndarray,
    priors: PriorSettings,
    rng: np.random.Generator,
) -> np.ndarray:
    residual = yields - states @ loadings.T
    shape = priors.measurement_shape + yields.shape[0] / 2.0
    scale = 2.0 * priors.measurement_variance_scale + 0.5 * np.sum(
        residual**2, axis=0
    )
    return 1.0 / rng.gamma(shape, 1.0 / scale)


def _update_decay_marginal(
    current: float,
    yields: np.ndarray,
    maturities_years: np.ndarray,
    process_covariance: np.ndarray,
    measurement_variance: np.ndarray,
    long_run_mean: np.ndarray,
    persistence: np.ndarray,
    priors: PriorSettings,
    rng: np.random.Generator,
    step: float = 0.05,
) -> tuple[float, bool, FilterResult]:
    lower, upper = priors.decay_lower, priors.decay_upper
    fraction = np.clip((current - lower) / (upper - lower), 1e-12, 1.0 - 1e-12)
    z = np.log(fraction / (1.0 - fraction))
    transition, intercept = factor_dynamics(long_run_mean, persistence)
    measurement_covariance = np.diag(measurement_variance)
    initial_covariance = np.eye(3) * priors.initial_state_variance

    def filter_at(decay: float) -> FilterResult:
        return kalman_filter(
            yields,
            dns_loadings(maturities_years, decay),
            transition,
            intercept,
            process_covariance,
            measurement_covariance,
            long_run_mean,
            initial_covariance,
        )

    def log_jacobian(candidate_fraction: float) -> float:
        return float(
            np.log(upper - lower)
            + np.log(candidate_fraction)
            + np.log1p(-candidate_fraction)
        )

    current_filter = filter_at(current)
    current_target = current_filter.log_likelihood + log_jacobian(fraction)
    proposed_z = z + rng.normal(0.0, step)
    proposed_fraction = expit(proposed_z)
    if not 0.0 < proposed_fraction < 1.0:
        return current, False, current_filter
    proposed_decay = lower + (upper - lower) * proposed_fraction
    proposed_filter = filter_at(float(proposed_decay))
    proposed_target = proposed_filter.log_likelihood + log_jacobian(
        proposed_fraction
    )
    accepted = np.log(rng.uniform()) < proposed_target - current_target
    if accepted:
        return float(proposed_decay), True, proposed_filter
    return current, False, current_filter


def _draw_stationary_ar1(
    n: int,
    persistence: float,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n < 1 or not np.isfinite(persistence) or abs(persistence) >= 1:
        raise ValueError("AR(1) requires a positive length and stationary persistence")
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("AR(1) stationary variance must be finite and positive")
    noise = rng.standard_normal(n)
    direction = np.empty(n)
    direction[0] = np.sqrt(variance) * noise[0]
    innovation_sd = np.sqrt(variance * (1.0 - persistence**2))
    for t in range(1, n):
        direction[t] = persistence * direction[t - 1] + innovation_sd * noise[t]
    return direction


def _elliptical_slice(
    current: np.ndarray,
    direction: np.ndarray,
    center: float,
    log_likelihood: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    max_shrinks: int = 1000,
) -> np.ndarray:
    centered = current - center
    current_likelihood = log_likelihood(current)
    if not np.isfinite(current_likelihood):
        raise ValueError("current elliptical-slice likelihood must be finite")
    threshold = current_likelihood + np.log(rng.uniform())
    angle = rng.uniform(0.0, 2.0 * np.pi)
    lower, upper = angle - 2.0 * np.pi, angle
    for _ in range(max_shrinks):
        proposal = center + centered * np.cos(angle) + direction * np.sin(angle)
        if log_likelihood(proposal) >= threshold:
            return proposal
        if angle < 0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise RuntimeError("elliptical slice sampler exceeded its shrink limit")


def _sample_volatility_path(
    current: np.ndarray,
    factor_innovations: np.ndarray,
    mean: float,
    persistence: float,
    variance: float,
    rng: np.random.Generator,
    max_shrinks: int = 1000,
) -> np.ndarray:
    """Elliptical slice update under the exact Gaussian innovation likelihood."""
    direction = _draw_stationary_ar1(len(current), persistence, variance, rng)

    def log_likelihood(path: np.ndarray) -> float:
        if not np.isfinite(path).all() or np.max(np.abs(path)) > 700:
            return -np.inf
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            score = -0.5 * np.sum(
                path + factor_innovations**2 * np.exp(-path)
            )
        return float(score) if np.isfinite(score) else -np.inf

    return _elliptical_slice(
        current, direction, mean - variance / 2.0, log_likelihood, rng, max_shrinks
    )


def _volatility_log_likelihood(
    h: np.ndarray, mean: float, persistence: float, variance: float
) -> float:
    if not -1.0 < persistence < 1.0 or not 0.0 < variance:
        return -np.inf
    innovation_variance = variance * (1.0 - persistence**2)
    center = mean - variance / 2.0
    value = -0.5 * (
        np.log(2.0 * np.pi * variance)
        + (h[0] - center) ** 2 / variance
    )
    if len(h) > 1:
        residual = h[1:] - center - persistence * (h[:-1] - center)
        value -= 0.5 * (
            (len(h) - 1) * np.log(2.0 * np.pi * innovation_variance)
            + float(residual @ residual) / innovation_variance
        )
    return float(value)


def _update_volatility_mean(
    h: np.ndarray,
    persistence: float,
    variance: float,
    prior_center: float,
    prior_sd: float,
    rng: np.random.Generator,
) -> float:
    q = variance * (1.0 - persistence**2)
    prior_precision = 1.0 / prior_sd**2
    precision = prior_precision + 1.0 / variance
    rhs = prior_precision * prior_center + (h[0] + variance / 2.0) / variance
    if len(h) > 1:
        regressor = 1.0 - persistence
        response = (
            h[1:]
            - persistence * h[:-1]
            + regressor * variance / 2.0
        )
        precision += (len(h) - 1) * regressor**2 / q
        rhs += regressor * response.sum() / q
    return float(rng.normal(rhs / precision, np.sqrt(1.0 / precision)))


def _update_volatility_persistence(
    h: np.ndarray,
    mean: float,
    persistence: float,
    variance: float,
    beta_a: float,
    beta_b: float,
    rng: np.random.Generator,
    step: float = 0.12,
) -> tuple[float, bool]:
    z = np.arctanh(persistence)

    def target(candidate_z: float) -> float:
        candidate = float(np.tanh(candidate_z))
        x = (1.0 + candidate) / 2.0
        if not 0.0 < x < 1.0:
            return -np.inf
        prior = (beta_a - 1.0) * np.log(x) + (beta_b - 1.0) * np.log1p(-x)
        jacobian = np.log1p(-(candidate**2))
        return (
            _volatility_log_likelihood(h, mean, candidate, variance)
            + prior
            + jacobian
        )

    proposed_z = z + rng.normal(0.0, step)
    accepted = np.log(rng.uniform()) < target(proposed_z) - target(z)
    updated = float(np.tanh(proposed_z)) if accepted else persistence
    return updated, bool(accepted)


def _update_volatility_variance(
    h: np.ndarray,
    mean: float,
    persistence: float,
    variance: float,
    upper: float,
    rng: np.random.Generator,
    step: float = 0.16,
) -> tuple[float, bool]:
    fraction = np.clip(variance / upper, 1e-12, 1.0 - 1e-12)
    z = np.log(fraction / (1.0 - fraction))

    def target(candidate_z: float) -> tuple[float, float]:
        candidate_fraction = expit(candidate_z)
        if not 0.0 < candidate_fraction < 1.0:
            return -np.inf, variance
        candidate = upper * candidate_fraction
        log_jacobian = (
            np.log(upper)
            + np.log(candidate_fraction)
            + np.log1p(-candidate_fraction)
        )
        return (
            _volatility_log_likelihood(h, mean, persistence, float(candidate))
            - candidate
            + log_jacobian,
            float(candidate),
        )

    proposed_z = z + rng.normal(0.0, step)
    current_target, _ = target(z)
    proposed_target, proposed_variance = target(proposed_z)
    accepted = np.log(rng.uniform()) < proposed_target - current_target
    return (proposed_variance if accepted else variance), bool(accepted)


def run_chain(
    yields: np.ndarray,
    maturities_years: np.ndarray,
    priors: PriorSettings,
    model: str,
    rng: np.random.Generator,
    warmup: int,
    draws: int,
    *,
    decay_step: float = 0.05,
) -> ChainDraws:
    """Run one blocked chain targeting the joint SS_M3 or SS_M4 posterior."""
    if model not in {"SS_M3", "SS_M4"}:
        raise ValueError("model must be 'SS_M3' or 'SS_M4'")
    y = np.asarray(yields, dtype=float)
    tau = np.asarray(maturities_years, dtype=float)
    if y.ndim != 2 or y.shape[1] != tau.size or y.shape[0] < 4:
        raise ValueError("yields must be a T by maturity panel with T >= 4")
    if not np.isfinite(y).all() or not np.isfinite(tau).all():
        raise ValueError("yields and maturities must be finite")
    if warmup < 0 or draws < 1:
        raise ValueError("warmup must be nonnegative and draws must be positive")
    if not np.isfinite(decay_step) or decay_step <= 0:
        raise ValueError("decay_step must be finite and positive")

    n_time = len(y)
    initial_mean = priors.factor_mean_center.copy()
    persistence = np.clip(priors.persistence_center, -0.98, 0.98)
    log_variance = priors.log_variance_center.copy()
    measurement_variance = priors.measurement_variance_scale.copy()
    decay = priors.decay_initial
    volatility_mean = priors.log_variance_center.copy()
    volatility_persistence = np.full(3, 0.8)
    volatility_variance = np.full(3, 0.2)
    h = np.repeat(
        (volatility_mean - volatility_variance / 2.0)[None, :], n_time, axis=0
    )
    loadings = dns_loadings(tau, decay)
    states = np.linalg.lstsq(loadings, y.T, rcond=None)[0].T
    initial_state = initial_mean.copy()

    factor_mean_draws = np.empty((draws, 3))
    persistence_draws = np.empty((draws, 3))
    log_variance_draws = np.empty((draws, 3))
    measurement_draws = np.empty((draws, y.shape[1]))
    decay_draws = np.empty(draws)
    terminal_factor_draws = np.empty((draws, 3))
    terminal_h_draws = np.empty((draws, 3))
    phi_draws = np.empty((draws, 3))
    volatility_variance_draws = np.empty((draws, 3))
    factor_path_sum = np.zeros((n_time, 3))
    h_path_sum = np.zeros((n_time, 3))
    accepted = {"decay": 0, "volatility_persistence": 0, "volatility_variance": 0}
    retained = 0

    for iteration in range(warmup + draws):
        transition, intercept = factor_dynamics(initial_mean, persistence)
        if model == "SS_M3":
            with np.errstate(over="raise", under="raise", invalid="raise"):
                q = factor_innovation_covariance(np.exp(log_variance))
            process_covariance = q
            process_variance = np.broadcast_to(np.diag(q), (n_time, 3))
        else:
            process_covariance = np.stack([
                stochastic_innovation_covariance(row) for row in h
            ])
            process_variance = np.stack([
                np.diag(covariance) for covariance in process_covariance
            ])

        decay, decay_accepted, filter_result = _update_decay_marginal(
            decay,
            y,
            tau,
            process_covariance,
            measurement_variance,
            initial_mean,
            persistence,
            priors,
            rng,
            step=decay_step,
        )
        if iteration >= warmup:
            accepted["decay"] += int(decay_accepted)
        states = sample_state_path(filter_result, transition, rng)
        initial_state = _draw_initial_factor(
            states[0],
            initial_mean,
            persistence,
            process_variance[0],
            priors.initial_state_variance,
            rng,
        )
        initial_mean, persistence = _draw_factor_parameters(
            states,
            initial_state,
            process_variance,
            initial_mean,
            persistence,
            priors,
            rng,
        )
        previous = np.vstack((initial_state, states[:-1]))
        factor_innovations = states - initial_mean - (previous - initial_mean) * persistence

        if model == "SS_M3":
            for j in range(3):
                log_variance[j] = _slice_sample(
                    log_variance[j],
                    lambda value, j=j: _factor_log_variance_target(
                        value,
                        factor_innovations[:, j],
                        priors.log_variance_center[j],
                        priors.log_variance_sd[j],
                    ),
                    rng,
                )
        else:
            for j in range(3):
                h[:, j] = _sample_volatility_path(
                    h[:, j],
                    factor_innovations[:, j],
                    volatility_mean[j],
                    volatility_persistence[j],
                    volatility_variance[j],
                    rng,
                )
                volatility_mean[j] = _update_volatility_mean(
                    h[:, j],
                    volatility_persistence[j],
                    volatility_variance[j],
                    priors.log_variance_center[j],
                    priors.log_variance_sd[j],
                    rng,
                )
                (
                    volatility_persistence[j],
                    phi_accepted,
                ) = _update_volatility_persistence(
                    h[:, j],
                    volatility_mean[j],
                    volatility_persistence[j],
                    volatility_variance[j],
                    priors.sv_phi_beta_a,
                    priors.sv_phi_beta_b,
                    rng,
                )
                (
                    volatility_variance[j],
                    variance_accepted,
                ) = _update_volatility_variance(
                    h[:, j],
                    volatility_mean[j],
                    volatility_persistence[j],
                    volatility_variance[j],
                    priors.sv_variance_upper,
                    rng,
                )
                if iteration >= warmup:
                    accepted["volatility_persistence"] += int(phi_accepted)
                    accepted["volatility_variance"] += int(variance_accepted)
            log_variance = volatility_mean.copy()

        loadings = dns_loadings(tau, decay)
        measurement_variance = _draw_measurement_variance(
            y, states, loadings, priors, rng
        )
        if iteration >= warmup:
            factor_mean_draws[retained] = initial_mean
            persistence_draws[retained] = persistence
            log_variance_draws[retained] = log_variance
            measurement_draws[retained] = measurement_variance
            decay_draws[retained] = decay
            terminal_factor_draws[retained] = states[-1]
            terminal_h_draws[retained] = h[-1]
            phi_draws[retained] = volatility_persistence
            volatility_variance_draws[retained] = volatility_variance
            factor_path_sum += states
            h_path_sum += h
            retained += 1

    acceptance_denominator = max(draws, 1)
    acceptance = {
        key: value / (
            acceptance_denominator * (3 if key != "decay" else 1)
        )
        for key, value in accepted.items()
    }
    return ChainDraws(
        model=model,
        factor_mean=factor_mean_draws,
        persistence=persistence_draws,
        log_variance=log_variance_draws,
        measurement_variance=measurement_draws,
        decay_per_year=decay_draws,
        terminal_factor=terminal_factor_draws,
        terminal_log_variance=terminal_h_draws,
        volatility_persistence=phi_draws,
        volatility_variance=volatility_variance_draws,
        factor_path_mean=factor_path_sum / draws,
        log_volatility_path_mean=h_path_sum / draws,
        acceptance=acceptance,
    )
