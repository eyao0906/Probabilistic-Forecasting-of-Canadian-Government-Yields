"""Isolated Bayesian DNS sampler with a prior directly on the long-run factor mean."""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np
from scipy.stats import truncnorm

from src import bayesian_finite as bf
from src import bayesian_revised as br
from src.bayesian_stage_a import FACTORS


@dataclass(frozen=True)
class LongRunPrior:
    volatility: bf.FinitePrior
    mean_center: float
    mean_variance: float
    persistence_center: float
    persistence_variance: float = 0.03**2

    def __post_init__(self):
        values = np.array([
            self.mean_center, self.mean_variance,
            self.persistence_center, self.persistence_variance,
        ])
        if not np.isfinite(values).all() or self.mean_variance <= 0 or self.persistence_variance <= 0:
            raise ValueError("Long-run-mean and persistence priors must be finite and proper")


def build_priors(design: dict) -> tuple[LongRunPrior, ...]:
    """Freeze factor-specific priors using only the initial 2000–2015 sample."""
    initial = np.asarray(design["factors"][:design["initial_n"]], dtype=float)
    if initial.shape != (835, 3):
        raise ValueError("Expected the accepted 835-observation training sample")
    priors = []
    for j, base in enumerate(design["priors"]):
        mean_center = float(np.mean(initial[:, j]))
        mean_sd = 2 * float(np.std(initial[:, j], ddof=1))
        lag = initial[:-1, j]
        response = initial[1:, j]
        persistence_center = float(np.linalg.lstsq(
            np.column_stack((np.ones(len(lag)), lag)), response, rcond=None
        )[0][1])
        volatility_base = replace(base, omega_variance=1.0)
        priors.append(LongRunPrior(
            volatility=bf.FinitePrior(volatility_base, v_upper=4.0, v_rate=1.0),
            mean_center=mean_center,
            mean_variance=mean_sd**2,
            persistence_center=persistence_center,
        ))
    return tuple(priors)


def prior_record(prior: LongRunPrior) -> dict:
    base = prior.volatility.base
    return {
        "long_run_mean_center": prior.mean_center,
        "long_run_mean_variance": prior.mean_variance,
        "persistence_center": prior.persistence_center,
        "persistence_variance": prior.persistence_variance,
        "log_variance_center": base.omega_mean,
        "log_variance_variance": base.omega_variance,
        "phi_alpha": base.phi_alpha,
        "phi_beta": base.phi_beta,
        "v_upper": prior.volatility.v_upper,
        "v_rate": prior.volatility.v_rate,
    }


def _draw_truncated_normal(rng, mean: float, variance: float) -> float:
    sd = np.sqrt(variance)
    return float(truncnorm.rvs(
        (-1 - mean) / sd, (1 - mean) / sd,
        loc=mean, scale=sd, random_state=rng,
    ))


def _innovation_precision(log_variance):
    with np.errstate(over="raise", invalid="raise"):
        weight = np.exp(-np.asarray(log_variance, dtype=float))
    if not np.isfinite(weight).all() or np.any(weight <= 0):
        raise FloatingPointError("Invalid factor-innovation precision")
    return weight


def mean_conditional(y, lag, log_variance, persistence, prior):
    """Gaussian full conditional for the long-run mean given persistence."""
    weight = _innovation_precision(log_variance)

    precision = 1 / prior.mean_variance + (1 - persistence)**2 * np.sum(weight)
    mean_rhs = (
        prior.mean_center / prior.mean_variance
        + (1 - persistence) * np.sum(weight * (y - persistence * lag))
    )
    variance = 1 / precision
    return float(mean_rhs * variance), float(variance)


def persistence_conditional(y, lag, log_variance, mean, prior):
    """Truncated-Gaussian full conditional for persistence given the mean."""
    weight = _innovation_precision(log_variance)
    regressor = lag - mean
    response = y - mean
    precision = (
        1 / prior.persistence_variance + np.sum(weight * regressor**2)
    )
    variance = 1 / precision
    posterior_mean = variance * (
        prior.persistence_center / prior.persistence_variance
        + np.sum(weight * regressor * response)
    )
    return float(posterior_mean), float(variance)


def update_mean_persistence(y, lag, log_variance, mean, persistence, prior, rng):
    """One exact Gibbs sweep for the factor mean and persistence."""
    posterior_mean, variance = mean_conditional(y, lag, log_variance, persistence, prior)
    mean = float(rng.normal(posterior_mean, np.sqrt(variance)))
    posterior_mean, variance = persistence_conditional(y, lag, log_variance, mean, prior)
    persistence = _draw_truncated_normal(
        rng, posterior_mean, variance
    )
    return mean, persistence


def _constant_variance_step(residual, log_variance, prior, rng):
    sse = float(residual @ residual)
    n = len(residual)

    def target(value):
        with np.errstate(over="ignore", invalid="ignore"):
            result = (
                -0.5 * n * value - 0.5 * sse * np.exp(-value)
                - 0.5 * (value - prior.volatility.base.omega_mean)**2
                / prior.volatility.base.omega_variance
            )
        return float(result) if np.isfinite(result) else -np.inf

    height = target(log_variance) + np.log(rng.random())
    left = log_variance - rng.random()
    right = left + 1
    while target(left) > height:
        left -= 1
    while target(right) > height:
        right += 1
    while True:
        candidate = rng.uniform(left, right)
        if target(candidate) >= height:
            return float(candidate)
        if candidate < log_variance:
            left = candidate
        else:
            right = candidate


def sample_chain(x, prior: LongRunPrior, model: str, rng, warmup=3000, draws=6000):
    """Sample M4 constant-volatility or M5 stochastic-volatility factors."""
    if model not in ("M4", "M5") or warmup < 0 or draws < 1:
        raise ValueError("Invalid model or sampling length")
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Need a finite factor series with at least three observations")

    started = time.perf_counter()
    y = x[1:]
    lag = x[:-1]
    mu = float(rng.normal(prior.mean_center, np.sqrt(prior.mean_variance)))
    b = _draw_truncated_normal(rng, prior.persistence_center, prior.persistence_variance)
    a = (1 - b) * mu
    n = len(y)

    if model == "M4":
        log_variance = float(prior.volatility.base.omega_mean + rng.normal())
        output = np.empty((draws, 4))
        for iteration in range(warmup + draws):
            mu, b = update_mean_persistence(y, lag, np.full(n, log_variance), mu, b, prior, rng)
            a = (1 - b) * mu
            residual = y - (a + b * lag)
            log_variance = _constant_variance_step(residual, log_variance, prior, rng)
            if iteration >= warmup:
                output[iteration - warmup] = [a, b, mu, log_variance]
        return {
            "draws": output,
            "names": ["a", "b", "long_run_mean", "log_variance"],
            "model": model,
            "parameterization": "long_run_mean_and_persistence",
            "elapsed_s": time.perf_counter() - started,
        }

    p = prior.volatility
    m = float(p.base.omega_mean + rng.normal())
    phi = float(2 * rng.beta(p.base.phi_alpha, p.base.phi_beta) - 1)
    v = float(-np.log1p(-rng.random() * (-np.expm1(-p.v_rate * p.v_upper))) / p.v_rate)
    omega = m - v / 2
    q = v * (1 - phi**2)
    h = np.empty(n)
    h[0] = omega + np.sqrt(v) * rng.normal()
    for t in range(1, n):
        h[t] = omega + phi * (h[t - 1] - omega) + np.sqrt(q) * rng.normal()

    output = np.empty((draws, 11))
    proposal_chol = np.diag([0.15, 0.15, 0.2])
    adaptation = np.empty((warmup, 3))
    phi_step = 0.15
    block_accept = 0
    h_accept = phi_accept = joint_accept = 0

    for iteration in range(warmup + draws):
        mu, b = update_mean_persistence(y, lag, h, mu, b, prior, rng)
        a = (1 - b) * mu
        residual = y - (a + b * lag)
        obs = br.log_squared_residual(residual)

        h, m, phi, v, accepted_joint = bf.joint_step(
            obs, h, m, phi, v, p, rng, proposal_chol
        )
        omega = m - v / 2
        q = v * (1 - phi**2)
        h, accepted_h, _ = br.corrected_h_step(obs, h, omega, phi, q, rng)
        m_mean, m_variance = bf.m_conditional(h, phi, v, p)
        m = float(rng.normal(m_mean, np.sqrt(m_variance)))

        u = np.arctanh(phi)
        candidate = rng.normal(u, phi_step)
        ratio = (
            bf.centered_target(np.array([m, candidate, np.log(v)]), h, p)
            - bf.centered_target(np.array([m, u, np.log(v)]), h, p)
        )
        moved = bool(np.log(rng.random()) < ratio)
        if moved:
            phi = float(np.tanh(candidate))
        if iteration < warmup:
            block_accept += int(moved)
            if (iteration + 1) % 50 == 0:
                phi_step *= np.exp((block_accept / 50 - 0.35) / np.sqrt((iteration + 1) / 50))
                block_accept = 0
            adaptation[iteration] = [m, np.arctanh(phi), np.log(v)]
            if iteration >= 199 and (iteration + 1) % 50 == 0:
                empirical = np.cov(adaptation[:iteration + 1], rowvar=False)
                covariance = (2.38**2 / 3) * (
                    0.95 * empirical + 0.05 * np.diag([0.15, 0.15, 0.2])**2
                )
                proposal_chol = np.linalg.cholesky(covariance + 1e-8 * np.eye(3))
        else:
            h_accept += int(accepted_h)
            phi_accept += int(moved)
            joint_accept += int(accepted_joint)

        h, v = bf.scale_step(obs, h, m, phi, v, p, rng)
        if iteration >= warmup:
            omega = m - v / 2
            q = v * (1 - phi**2)
            output[iteration - warmup] = [
                a, b, mu, m, phi, v, omega, q, h[0], h[len(h) // 2], h[-1]
            ]

    return {
        "draws": output,
        "names": ["a", "b", "long_run_mean", "m", "phi", "v", "omega", "q",
                  "h_first", "h_mid", "h_last"],
        "model": model,
        "parameterization": "long_run_mean_and_persistence",
        "h_acceptance": h_accept / draws,
        "phi_acceptance": phi_accept / draws,
        "collapsed_acceptance": joint_accept / draws,
        "frozen_proposal_cholesky": proposal_chol.tolist(),
        "frozen_phi_step": phi_step,
        "elapsed_s": time.perf_counter() - started,
    }


def moment_audits(priors: tuple[LongRunPrior, ...], n: int) -> list[dict]:
    reports = []
    for factor, prior in zip(FACTORS, priors):
        audit = bf.moment_audit(prior.volatility, n)
        audit.update({
            "factor": factor,
            "proof": (
                "The direct Gaussian prior on the long-run mean has finite polynomial moments; "
                "|1-b|<2 bounds the induced intercept, and the bounded stationary log-volatility "
                "variance preserves finite-horizon predictive moments."
            ),
            "prior_prediction_caveat": (
                "The log-variance prior SD is 1; finite moments do not by themselves establish "
                "good prior calibration."
            ),
            "long_run_mean_prior_mean": prior.mean_center,
            "long_run_mean_prior_variance": prior.mean_variance,
            "long_run_mean_second_moment_finite": True,
            "long_run_mean_absolute_mean_finite": True,
        })
        reports.append(audit)
    return reports


def prior_configuration(priors: tuple[LongRunPrior, ...]) -> list[dict]:
    return [prior_record(prior) for prior in priors]


def validate_draws(draws: np.ndarray, names: list[str], model: str, v_upper=4.0) -> None:
    if draws.ndim != 2 or draws.shape[1] != len(names) or not np.isfinite(draws).all():
        raise ValueError("Invalid or nonfinite posterior draw array")
    index = {name: names.index(name) for name in names}
    if np.any(np.abs(draws[:, index["b"]]) >= 1):
        raise ValueError("Persistence draw is outside stationary support")
    if not np.allclose(
        draws[:, index["a"]],
        (1 - draws[:, index["b"]]) * draws[:, index["long_run_mean"]],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Intercept does not equal (1-b) times the long-run mean")
    if model == "M5":
        phi = draws[:, index["phi"]]
        v = draws[:, index["v"]]
        if np.any((phi <= -1) | (phi >= 1)):
            raise ValueError("Volatility persistence is outside stationary support")
        if np.any((v <= 0) | (v >= v_upper)):
            raise ValueError("Stationary log-volatility variance is outside its prior support")
        if (not np.allclose(draws[:, index["omega"]], draws[:, index["m"]] - v / 2)
            or not np.allclose(draws[:, index["q"]], v * (1 - phi**2))):
            raise ValueError("Transformed stochastic-volatility parameters are inconsistent")
