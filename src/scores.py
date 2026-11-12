"""Proper scoring rules, model comparisons, and risk backtests."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2, norm


def dss(observed: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    """Multivariate Gaussian Dawid--Sebastiani score (constants omitted)."""
    error = np.asarray(observed) - np.asarray(mean)
    cov = (np.asarray(covariance) + np.asarray(covariance).T) / 2
    chol = np.linalg.cholesky(cov)
    z = np.linalg.solve(chol, error)
    return float(2 * np.log(np.diag(chol)).sum() + z @ z)


def energy_score(scenarios: np.ndarray, observed: np.ndarray) -> float:
    x = np.asarray(scenarios, dtype=float)
    y = np.asarray(observed, dtype=float)
    d1 = np.linalg.norm(x - y, axis=1).mean()
    distances = cdist(x, x, metric="euclidean")
    upper = distances[np.triu_indices(len(x), 1)]
    d2 = upper.mean() if len(upper) else 0.0
    return float(d1 - 0.5 * d2)


def variogram_score(scenarios: np.ndarray, observed: np.ndarray, power: float = 0.5) -> float:
    x = np.asarray(scenarios, dtype=float)
    y = np.asarray(observed, dtype=float)
    obs_diff = np.abs(y[:, None] - y[None, :]) ** power
    pred_diff = np.mean(np.abs(x[:, :, None] - x[:, None, :]) ** power, axis=0)
    tri = np.triu_indices(len(y), 1)
    return float(np.sum((obs_diff[tri] - pred_diff[tri]) ** 2))


def point_metrics(observed: np.ndarray, mean: np.ndarray, scenarios: np.ndarray) -> dict:
    e = np.asarray(observed) - np.asarray(mean)
    lo, hi = np.quantile(scenarios, [0.05, 0.95], axis=0)
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "coverage90": float(np.mean((observed >= lo) & (observed <= hi))),
        "mean_width90": float(np.mean(hi - lo)),
    }


def hac_dm(loss_difference: np.ndarray, max_lag: int = 3) -> dict:
    """Diebold--Mariano-style paired mean-loss test with Bartlett HAC variance."""
    d = np.asarray(loss_difference, dtype=float)
    n = len(d)
    centered = d - d.mean()
    gamma0 = float(centered @ centered / n)
    long_var = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        gamma = float(centered[lag:] @ centered[:-lag] / n)
        long_var += 2 * (1 - lag / (max_lag + 1)) * gamma
    se = np.sqrt(max(long_var, 0.0) / n)
    stat = float(d.mean() / se) if se > 0 else float("nan")
    return {"mean_difference": float(d.mean()), "hac_se": float(se), "statistic": stat,
            "p_value_two_sided": float(2 * norm.sf(abs(stat))) if np.isfinite(stat) else float("nan"),
            "n": n, "max_lag": min(max_lag, n - 1)}


def stationary_bootstrap_indices(n: int, expected_block_length: float,
                                 rng: np.random.Generator) -> np.ndarray:
    if n < 1:
        return np.array([], dtype=int)
    restart = 1.0 / max(expected_block_length, 1.0)
    out = np.empty(n, dtype=int)
    out[0] = rng.integers(n)
    for i in range(1, n):
        out[i] = rng.integers(n) if rng.random() < restart else (out[i - 1] + 1) % n
    return out


def model_confidence_set(losses: np.ndarray, names: list[str], rng: np.random.Generator,
                         reps: int = 999, alpha: float = 0.10,
                         expected_block_length: float = 4.0) -> dict:
    """A max-pairwise studentized stationary-bootstrap MCS for lower-is-better losses."""
    losses = np.asarray(losses, dtype=float)
    n, m = losses.shape
    active = list(range(m))
    history = []
    while len(active) > 1:
        d = losses[:, active]
        means = d.mean(axis=0)
        se = np.empty((len(active), len(active)))
        for i in range(len(active)):
            for j in range(len(active)):
                se[i, j] = max(np.std(d[:, i] - d[:, j], ddof=1) / np.sqrt(n), 1e-12)
        t_obs = np.abs((means[:, None] - means[None, :]) / se)
        statistic = float(np.max(t_obs))
        centered = d - means
        boot_stats = np.empty(reps)
        for b in range(reps):
            ix = stationary_bootstrap_indices(n, expected_block_length, rng)
            boot_mean = centered[ix].mean(axis=0)
            tb = np.abs((boot_mean[:, None] - boot_mean[None, :]) / se)
            boot_stats[b] = np.max(tb)
        pvalue = float((1 + np.sum(boot_stats >= statistic)) / (reps + 1))
        history.append({"models": [names[i] for i in active], "statistic": statistic, "p_value": pvalue})
        if pvalue >= alpha:
            break
        worst_local = int(np.argmax(means))
        active.pop(worst_local)
    return {"included": [names[i] for i in active], "excluded": [names[i] for i in range(m) if i not in active],
            "alpha": alpha, "repetitions": reps, "expected_block_length_origins": expected_block_length,
            "steps": history}


def _safe_loglik(counts: np.ndarray, probs: np.ndarray) -> float:
    total = 0.0
    for c, p in zip(counts, probs):
        if c:
            if p <= 0:
                return -np.inf
            total += c * np.log(p)
    return float(total)


def var_backtests(losses: np.ndarray, var_forecasts: np.ndarray, tail_prob: float = 0.05) -> dict:
    """Kupiec unconditional coverage and Christoffersen independence/conditional coverage."""
    loss = np.asarray(losses, dtype=float)
    var = np.asarray(var_forecasts, dtype=float)
    hit = (loss > var).astype(int)
    n = len(hit)
    x = int(hit.sum())
    phat = x / n if n else np.nan
    ll0 = _safe_loglik(np.array([x, n - x]), np.array([tail_prob, 1 - tail_prob]))
    ll1 = _safe_loglik(np.array([x, n - x]), np.array([phat, 1 - phat]))
    lr_uc = max(0.0, 2 * (ll1 - ll0)) if np.isfinite(ll0) and np.isfinite(ll1) else 0.0
    p_uc = float(chi2.sf(lr_uc, 1))
    if n >= 2:
        n00 = int(np.sum((hit[:-1] == 0) & (hit[1:] == 0)))
        n01 = int(np.sum((hit[:-1] == 0) & (hit[1:] == 1)))
        n10 = int(np.sum((hit[:-1] == 1) & (hit[1:] == 0)))
        n11 = int(np.sum((hit[:-1] == 1) & (hit[1:] == 1)))
        p01 = n01 / max(n00 + n01, 1)
        p11 = n11 / max(n10 + n11, 1)
        pbar = (n01 + n11) / max(n - 1, 1)
        ll_ind0 = _safe_loglik(np.array([n00 + n10, n01 + n11]), np.array([1 - pbar, pbar]))
        ll_ind1 = _safe_loglik(np.array([n00, n01, n10, n11]),
                               np.array([1 - p01, p01, 1 - p11, p11]))
        lr_ind = max(0.0, 2 * (ll_ind1 - ll_ind0)) if np.isfinite(ll_ind0) and np.isfinite(ll_ind1) else 0.0
    else:
        n00 = n01 = n10 = n11 = 0
        p01 = p11 = np.nan
        lr_ind = 0.0
    lr_cc = lr_uc + lr_ind
    return {"n": n, "exceptions": x, "exception_rate": float(phat), "tail_probability": tail_prob,
            "LR_unconditional_coverage": lr_uc, "p_unconditional_coverage": p_uc,
            "LR_independence": lr_ind, "p_independence": float(chi2.sf(lr_ind, 1)),
            "LR_conditional_coverage": lr_cc, "p_conditional_coverage": float(chi2.sf(lr_cc, 2)),
            "transition_counts": {"00": n00, "01": n01, "10": n10, "11": n11},
            "transition_probabilities": {"p01": p01, "p11": p11}}
