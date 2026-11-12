"""Forecast models for the Canadian DNS research project.

Yields are in percentage points; maturities and the Nelson--Siegel decay rate
are expressed in years and inverse years, respectively.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

FLOOR = 1e-6


def ns_loadings(maturities: np.ndarray, lam: float) -> np.ndarray:
    tau = np.asarray(maturities, dtype=float)
    x = lam * tau
    level = np.ones_like(tau)
    slope = -np.expm1(-x) / x
    curvature = slope - np.exp(-x)
    return np.column_stack((level, slope, curvature))


def fit_lambda(y: np.ndarray, maturities: np.ndarray, grid: np.ndarray) -> tuple[float, np.ndarray, float]:
    best = None
    for lam in np.asarray(grid, dtype=float):
        load = ns_loadings(maturities, float(lam))
        factors = np.linalg.lstsq(load, y.T, rcond=None)[0].T
        residual = y - factors @ load.T
        loss = float(np.sum(residual ** 2))
        if best is None or loss < best[0]:
            best = (loss, float(lam), factors, load)
    assert best is not None
    return best[1], best[2], best[0]


def project_covariance(cov: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    """Symmetrize and eigenvalue-floor a covariance using training-only inputs."""
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(cov)
    return (vecs * np.maximum(vals, floor)) @ vecs.T


def dns_residual_variance(y: np.ndarray, load: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    factors = np.linalg.lstsq(load, y.T, rcond=None)[0].T
    residual = y - factors @ load.T
    return np.maximum(np.mean(residual ** 2, axis=0), floor)


def ar1_fit(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    target, lag = x[1:], x[:-1]
    X = np.column_stack((np.ones(len(lag)), lag))
    beta = np.linalg.lstsq(X, target, rcond=None)[0]
    beta[1] = np.clip(beta[1], -0.995, 0.995)
    beta[0] = np.mean(target - beta[1] * lag)
    residual = target - X @ beta
    return float(beta[0]), float(beta[1]), max(float(np.mean(residual ** 2)), FLOOR)


def gaussian_dns_forecast(factors: np.ndarray, load: np.ndarray, r_diag: np.ndarray,
                          horizon: int, rng: np.random.Generator, draws: int) -> tuple[np.ndarray, np.ndarray]:
    params = [ar1_fit(factors[:, j]) for j in range(3)]
    mean_f, var_f = np.zeros(3), np.zeros(3)
    for j, (a, b, q) in enumerate(params):
        mean_f[j] = b ** horizon * factors[-1, j] + a * sum(b ** k for k in range(horizon))
        var_f[j] = q * sum(b ** (2 * k) for k in range(horizon))
    mean_y = load @ mean_f
    cov_y = load @ np.diag(var_f) @ load.T + np.diag(r_diag)
    cov_y = project_covariance(cov_y)
    return rng.multivariate_normal(mean_y, cov_y, size=draws), cov_y


def random_walk_forecast(y: np.ndarray, horizon: int, rng: np.random.Generator,
                         draws: int, floor: float = FLOOR) -> tuple[np.ndarray, np.ndarray]:
    diffs = y[horizon:] - y[:-horizon]
    cov = project_covariance(np.cov(diffs, rowvar=False, ddof=1), floor)
    return rng.multivariate_normal(y[-1], cov, size=draws), cov


def pca_var_forecast(y: np.ndarray, horizon: int, rng: np.random.Generator,
                     draws: int, floor: float = FLOOR) -> tuple[np.ndarray, np.ndarray]:
    center = y.mean(axis=0)
    centered = y - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3].T
    score = centered @ basis
    x, target = score[:-1], score[1:]
    X = np.column_stack((np.ones(len(x)), x))
    coef = np.linalg.lstsq(X, target, rcond=None)[0]
    intercept, transition = coef[0], coef[1:].T
    resid = target - X @ coef
    q = project_covariance(np.cov(resid, rowvar=False, ddof=1), floor)
    state_mean = score[-1].copy()
    state_cov = np.zeros((3, 3))
    for _ in range(horizon):
        state_mean = intercept + transition @ state_mean
        state_cov = transition @ state_cov @ transition.T + q
    residual_y = centered - score @ basis.T
    r_diag = np.maximum(np.mean(residual_y ** 2, axis=0), floor)
    mean_y = center + basis @ state_mean
    cov_y = project_covariance(basis @ state_cov @ basis.T + np.diag(r_diag), floor)
    return rng.multivariate_normal(mean_y, cov_y, size=draws), cov_y


def fit_bayes_constant(x: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conjugate Normal--inverse-gamma AR(1) posterior with stationary-b rejection."""
    target = np.asarray(x[1:], dtype=float)
    design = np.column_stack((np.ones(len(target)), np.asarray(x[:-1], dtype=float)))
    prior_mean = np.zeros(2)
    prior_cov = np.diag([25.0, 1.0])
    prior_precision = np.linalg.inv(prior_cov)
    post_cov = np.linalg.inv(prior_precision + design.T @ design)
    post_mean = post_cov @ (design.T @ target + prior_precision @ prior_mean)
    shape0, scale0 = 2.0, 1.0
    shape = shape0 + len(target) / 2
    scale = scale0 + 0.5 * (target @ target + prior_mean @ prior_precision @ prior_mean
                            - post_mean @ np.linalg.solve(post_cov, post_mean))
    betas, variances = [], []
    while len(betas) < draws:
        sig2 = 1.0 / rng.gamma(shape, 1.0 / max(scale, 1e-12))
        beta = rng.multivariate_normal(post_mean, post_cov * sig2)
        if abs(beta[1]) < 0.999:
            betas.append(beta)
            variances.append(sig2)
    beta_draws = np.asarray(betas)
    variance_draws = np.asarray(variances)
    return beta_draws, variance_draws, np.column_stack((post_mean, np.diag(post_cov)))


def bayes_constant_scenarios(factors: np.ndarray, load: np.ndarray, r_diag: np.ndarray,
                             horizon: int, rng: np.random.Generator, draws: int) -> np.ndarray:
    factor_sims = np.empty((draws, 3))
    for j in range(3):
        beta, sig2, _ = fit_bayes_constant(factors[:, j], rng, draws)
        state = np.full(draws, factors[-1, j])
        for _ in range(horizon):
            state = beta[:, 0] + beta[:, 1] * state + rng.normal(0, np.sqrt(sig2))
        factor_sims[:, j] = state
    return factor_sims @ load.T + rng.normal(size=(draws, len(r_diag))) * np.sqrt(r_diag)


# Seven-component normal-mixture approximation to log(chi-square_1), following
# Kim, Shephard and Chib (1998); weights sum to one.
MIX_W = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
MIX_M = np.array([-10.12999, -3.97281, -8.56686, -2.77786, -0.61961, 1.79518, -1.08819])
MIX_V = np.array([5.79596, 2.61369, 5.17950, 1.79518, 0.16735, 0.64009, 1.26261])


@dataclass
class SVChain:
    beta: np.ndarray
    intercept_h: np.ndarray
    phi: np.ndarray
    sigma2_h: np.ndarray
    h_last: np.ndarray
    acceptance_phi: float


def _sample_beta(y: np.ndarray, xlag: np.ndarray, h: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    design = np.column_stack((np.ones(len(y)), xlag))
    prior_precision = np.diag([1.0 / 25.0, 1.0])
    invvar = np.exp(-np.clip(h, -15.0, 15.0))
    cov = np.linalg.inv(prior_precision + design.T @ (invvar[:, None] * design))
    mean = cov @ (design.T @ (invvar * y))
    for _ in range(1000):
        beta = rng.multivariate_normal(mean, cov)
        if abs(beta[1]) < 0.999:
            return beta
    beta[1] = np.clip(beta[1], -0.998, 0.998)
    return beta


def _ffbs_h(obs: np.ndarray, mix: np.ndarray, c: float, phi: float,
            sig2: float, rng: np.random.Generator) -> np.ndarray:
    n = len(obs)
    obs_var = MIX_V[mix]
    obs_mean = obs - MIX_M[mix]
    phi = float(np.clip(phi, -0.995, 0.995))
    mu = c / (1.0 - phi)
    pred_m = np.empty(n)
    pred_v = np.empty(n)
    filt_m = np.empty(n)
    filt_v = np.empty(n)
    pred_m[0] = mu
    pred_v[0] = max(sig2 / (1.0 - phi * phi), 1e-5)
    for t in range(n):
        if t:
            pred_m[t] = c + phi * filt_m[t - 1]
            pred_v[t] = phi * phi * filt_v[t - 1] + sig2
        k = pred_v[t] / (pred_v[t] + obs_var[t])
        filt_m[t] = pred_m[t] + k * (obs_mean[t] - pred_m[t])
        filt_v[t] = max((1.0 - k) * pred_v[t], 1e-10)
    h = np.empty(n)
    h[-1] = rng.normal(filt_m[-1], math.sqrt(filt_v[-1]))
    for t in range(n - 2, -1, -1):
        gain = filt_v[t] * phi / max(pred_v[t + 1], 1e-12)
        m = filt_m[t] + gain * (h[t + 1] - pred_m[t + 1])
        v = max(filt_v[t] - gain * gain * pred_v[t + 1], 1e-10)
        h[t] = rng.normal(m, math.sqrt(v))
    return np.clip(h, -15.0, 15.0)


def _draw_mix(obs: np.ndarray, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    logp = (np.log(MIX_W)[None, :] - 0.5 * np.log(MIX_V)[None, :]
            - 0.5 * ((obs[:, None] - h[:, None] - MIX_M[None, :]) ** 2) / MIX_V[None, :])
    logp -= logp.max(axis=1, keepdims=True)
    prob = np.exp(logp)
    prob /= prob.sum(axis=1, keepdims=True)
    return np.array([rng.choice(len(MIX_W), p=p) for p in prob], dtype=int)


def _draw_c(h: np.ndarray, phi: float, sig2: float, rng: np.random.Generator) -> float:
    phi = float(np.clip(phi, -0.995, 0.995))
    prior_prec = 1.0 / 25.0
    trans = h[1:] - phi * h[:-1]
    obs_precision = len(trans) / sig2
    # Stationary initial-state density contributes h[0] ~ N(c/(1-phi), sig2/(1-phi^2)).
    v0 = sig2 * (1.0 - phi) / (1.0 + phi)
    obs_precision += 1.0 / max(v0, 1e-10)
    rhs = trans.sum() / sig2 + h[0] * (1.0 - phi) / max(v0, 1e-10)
    var = 1.0 / (prior_prec + obs_precision)
    mean = var * rhs
    return float(rng.normal(mean, math.sqrt(var)))


def _log_phi_target(u: float, h: np.ndarray, c: float, sig2: float) -> float:
    phi = math.tanh(u)
    if abs(phi) >= 0.999:
        return -np.inf
    x = (phi + 1.0) / 2.0
    # Beta(20, 1.5) prior mapped to (-1,1), plus Jacobian for u=atanh(phi).
    lp = 19.0 * math.log(max(x, 1e-12)) + 0.5 * math.log(max(1.0 - x, 1e-12))
    lp += math.log(max(1.0 - phi * phi, 1e-12))
    mu = c / (1.0 - phi)
    v0 = sig2 / (1.0 - phi * phi)
    lp += -0.5 * ((h[0] - mu) ** 2 / v0 + math.log(v0))
    e = h[1:] - c - phi * h[:-1]
    lp += -0.5 * (float(e @ e) / sig2 + len(e) * math.log(sig2))
    return lp


def sigma2_posterior_parameters(h: np.ndarray, c: float, phi: float,
                                 prior_shape: float = 2.0,
                                 prior_scale: float = 0.2) -> tuple[float, float]:
    """Return (shape, scale) of the inverse-gamma posterior for σ²_h.

    The likelihood contribution includes both the transition innovations and the
    stationary initial-state density, so that short samples still regularise
    towards the prior.
    """
    h = np.asarray(h, dtype=float)
    phi = float(np.clip(phi, -0.995, 0.995))
    innovations = h[1:] - c - phi * h[:-1]
    initial_quad = float((h[0] - c / (1.0 - phi)) ** 2 * (1.0 - phi * phi))
    shape = prior_shape + len(h) / 2.0
    scale = prior_scale + 0.5 * (float(innovations @ innovations) + initial_quad)
    return shape, scale


def fit_sv_chain(x: np.ndarray, rng: np.random.Generator, warmup: int = 150,
                 draws: int = 256) -> SVChain:
    """Approximate univariate AR(1)-SV posterior using a mixture-Gaussian FFBS."""
    x = np.asarray(x, dtype=float)
    y, lag = x[1:], x[:-1]
    beta0 = np.linalg.lstsq(np.column_stack((np.ones(len(y)), lag)), y, rcond=None)[0]
    resid = y - beta0[0] - beta0[1] * lag
    h = np.log(np.maximum(resid * resid, 1e-10))
    h = np.clip(h, -12.0, 8.0)
    phi, sig2 = 0.90, 0.20
    c = (1.0 - phi) * float(np.median(h))
    mix = np.full(len(y), 4, dtype=int)
    total = warmup + draws
    betas, cs, phis, sigmas, hl = [], [], [], [], []
    accepted = 0
    for it in range(total):
        beta = _sample_beta(y, lag, h, rng)
        resid = y - beta[0] - beta[1] * lag
        logeps2 = np.log(np.maximum(resid * resid, 1e-12))
        mix = _draw_mix(logeps2, h, rng)
        h = _ffbs_h(logeps2, mix, c, phi, sig2, rng)
        c = _draw_c(h, phi, sig2, rng)
        shape, scale = sigma2_posterior_parameters(h, c, phi)
        sig2 = 1.0 / rng.gamma(shape, 1.0 / max(scale, 1e-10))
        u = math.atanh(float(np.clip(phi, -0.994, 0.994)))
        candidate = float(rng.normal(u, 0.12))
        if math.log(rng.random()) < _log_phi_target(candidate, h, c, sig2) - _log_phi_target(u, h, c, sig2):
            phi = math.tanh(candidate)
            accepted += 1
        if it >= warmup:
            betas.append(beta)
            cs.append(c)
            phis.append(phi)
            sigmas.append(sig2)
            hl.append(h[-1])
    return SVChain(np.asarray(betas), np.asarray(cs), np.asarray(phis),
                   np.asarray(sigmas), np.asarray(hl), accepted / total)


def forecast_sv_from_chains(factors: np.ndarray, load: np.ndarray, r_diag: np.ndarray,
                            horizon: int, rng: np.random.Generator, draws: int,
                            chains: list[SVChain]) -> np.ndarray:
    n = draws
    factor_sim = np.empty((n, 3))
    for j, chain in enumerate(chains):
        idx = rng.integers(0, len(chain.beta), size=n)
        beta = chain.beta[idx]
        c, phi, sig2, h = chain.intercept_h[idx], chain.phi[idx], chain.sigma2_h[idx], chain.h_last[idx]
        state = np.full(n, factors[-1, j])
        for _ in range(horizon):
            h = c + phi * h + rng.normal(0, np.sqrt(sig2))
            state = beta[:, 0] + beta[:, 1] * state + np.exp(np.clip(h, -15, 10) / 2) * rng.normal(size=n)
        factor_sim[:, j] = state
    return factor_sim @ load.T + rng.normal(size=(n, len(r_diag))) * np.sqrt(r_diag)


def bayes_sv_scenarios(factors: np.ndarray, load: np.ndarray, r_diag: np.ndarray,
                       horizon: int, rng: np.random.Generator, draws: int,
                       warmup: int, iterations: int) -> tuple[np.ndarray, list[SVChain]]:
    chains = [fit_sv_chain(factors[:, j], rng, warmup, iterations) for j in range(3)]
    scenarios = forecast_sv_from_chains(factors, load, r_diag, horizon, rng, draws, chains)
    return scenarios, chains


def kalman_filter(y: np.ndarray, load: np.ndarray, params: np.ndarray,
                  return_state: bool = False):
    """Gaussian three-factor DNS state-space filter; diagonal transition, Q, and R."""
    mu = params[:3]
    rho = np.tanh(params[3:6])
    q = np.exp(np.clip(params[6:9], -20, 8))
    r = np.exp(np.clip(params[9:17], -20, 8))
    p = np.diag(q / np.maximum(1.0 - rho * rho, 1e-6))
    state = mu.copy()
    total = 0.0
    ident = np.eye(3)
    for i, obs in enumerate(y):
        s = load @ p @ load.T + np.diag(r)
        try:
            chol = np.linalg.cholesky(s)
        except np.linalg.LinAlgError:
            return (1e12, None, None) if return_state else 1e12
        innov = obs - load @ state
        z = np.linalg.solve(chol, innov)
        total += 0.5 * (len(obs) * math.log(2 * math.pi) + 2 * np.log(np.diag(chol)).sum() + z @ z)
        k = np.linalg.solve(s, load @ p).T
        state = state + k @ innov
        p = p - k @ load @ p
        p = (p + p.T) / 2
        if i < len(y) - 1:
            state = mu + rho * (state - mu)
            p = rho[:, None] * p * rho[None, :] + np.diag(q)
    if return_state:
        return float(total), state, p
    return float(total)


def state_space_start(y: np.ndarray, load: np.ndarray) -> np.ndarray:
    factors = np.linalg.lstsq(load, y.T, rcond=None)[0].T
    mu = factors.mean(axis=0)
    rho, q = [], []
    for j in range(3):
        _, b, var = ar1_fit(factors[:, j])
        rho.append(np.clip(b, -0.98, 0.98))
        q.append(max(var, 1e-4))
    r = dns_residual_variance(y, load)
    return np.r_[mu, np.arctanh(rho), np.log(q), np.log(r)]


def fit_state_space(y: np.ndarray, load: np.ndarray, start: np.ndarray | None = None,
                    maxiter: int = 15) -> tuple[np.ndarray, dict]:
    if start is None:
        start = state_space_start(y, load)
    bounds = [(-20, 30)] * 3 + [(-3.0, 3.0)] * 3 + [(-15, 5)] * 3 + [(-15, 5)] * 8
    result = minimize(lambda p: kalman_filter(y, load, p), start, method="L-BFGS-B",
                      bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-5, "maxls": 10})
    params = result.x if np.isfinite(result.fun) else start
    return params, {"success": bool(result.success), "message": str(result.message),
                    "iterations": int(result.nit), "nll": float(result.fun)}


def state_space_forecast(y: np.ndarray, load: np.ndarray, params: np.ndarray,
                         horizon: int, rng: np.random.Generator, draws: int) -> tuple[np.ndarray, np.ndarray]:
    _, state, p = kalman_filter(y, load, params, return_state=True)
    mu, rho = params[:3], np.tanh(params[3:6])
    q, r = np.exp(params[6:9]), np.exp(params[9:17])
    mean_f = mu + rho ** horizon * (state - mu)
    var_f = (rho ** horizon)[:, None] * p * (rho ** horizon)[None, :]
    var_f += np.diag(q * np.array([sum(rho[j] ** (2 * k) for k in range(horizon)) for j in range(3)]))
    cov_y = project_covariance(load @ var_f @ load.T + np.diag(r))
    mean_y = load @ mean_f
    return rng.multivariate_normal(mean_y, cov_y, size=draws), cov_y


def rhat(chains: np.ndarray) -> np.ndarray:
    """Split-chain potential scale reduction estimate (chains x draws x parameters)."""
    arr = np.asarray(chains, dtype=float)
    m, n, p = arr.shape
    n2 = n // 2
    if m < 2 or n2 < 2:
        return np.full(p, np.nan)
    split = np.concatenate((arr[:, :n2], arr[:, -n2:]), axis=0)
    m2, n2, _ = split.shape
    means = split.mean(axis=1)
    variances = split.var(axis=1, ddof=1)
    w = variances.mean(axis=0)
    b = n2 * means.var(axis=0, ddof=1)
    var_hat = (n2 - 1) / n2 * w + b / n2
    return np.sqrt(np.maximum(var_hat / np.maximum(w, 1e-15), 0))


def effective_sample_size(chains: np.ndarray) -> np.ndarray:
    """Initial-positive-sequence ESS approximation for independent chains."""
    arr = np.asarray(chains, dtype=float)
    m, n, p = arr.shape
    out = np.empty(p)
    for k in range(p):
        values = arr[:, :, k]
        var = values.var(axis=1, ddof=1).mean()
        if var <= 0:
            out[k] = m * n
            continue
        rho_sum = 0.0
        for lag in range(1, min(n // 2, 500)):
            ac = 0.0
            for ch in range(m):
                v = values[ch] - values[ch].mean()
                ac += float(v[:-lag] @ v[lag:]) / max((n - lag) * var, 1e-15)
            ac /= m
            if ac < 0:
                break
            rho_sum += ac
        out[k] = min(m * n, m * n / max(1 + 2 * rho_sum, 1e-6))
    return out
