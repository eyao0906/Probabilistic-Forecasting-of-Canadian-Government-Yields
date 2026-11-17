from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import multivariate_normal

from src.ssdns_hy import (
    dns_loadings,
    factor_dynamics,
    kalman_filter,
    rts_smoother,
    sample_state_path,
)


def _dense_state_reference(
    observations, loadings, transition, intercept, process_covariance,
    measurement_covariance, initial_mean, initial_covariance,
):
    n_time, n_yield = observations.shape
    n_state = len(initial_mean)
    state_means = np.empty((n_time, n_state))
    state_covariances = np.empty((n_time, n_time, n_state, n_state))
    previous_mean = initial_mean
    previous_covariance = initial_covariance
    for t in range(n_time):
        state_means[t] = transition @ previous_mean + intercept
        state_covariances[t, t] = (
            transition @ previous_covariance @ transition.T + process_covariance[t]
        )
        if t:
            for s in range(t):
                cross = transition @ state_covariances[t - 1, s]
                state_covariances[t, s] = cross
                state_covariances[s, t] = cross.T
        previous_mean = state_means[t]
        previous_covariance = state_covariances[t, t]

    flat_mean = state_means.reshape(-1)
    covariance_x = np.block([
        [state_covariances[t, s] for s in range(n_time)]
        for t in range(n_time)
    ])
    observation_operator = np.kron(np.eye(n_time), loadings)
    covariance_y = (
        observation_operator @ covariance_x @ observation_operator.T
        + np.kron(np.eye(n_time), measurement_covariance)
    )
    mean_y = observation_operator @ flat_mean
    cross_xy = covariance_x @ observation_operator.T
    residual = observations.reshape(-1) - mean_y
    expected_log_likelihood = multivariate_normal.logpdf(
        observations.reshape(-1), mean=mean_y, cov=covariance_y
    )
    expected_mean = flat_mean + cross_xy @ np.linalg.solve(covariance_y, residual)
    expected_covariance = covariance_x - cross_xy @ np.linalg.solve(
        covariance_y, cross_xy.T
    )
    return expected_log_likelihood, expected_mean, expected_covariance


def _problem(seed=724):
    rng = np.random.default_rng(seed)
    n_time, n_yield, n_state = 5, 4, 2
    loadings = rng.normal(size=(n_yield, n_state))
    transition = np.array([[0.72, 0.08], [-0.04, 0.55]])
    intercept = np.array([0.3, -0.1])
    initial_mean = np.array([0.2, -0.3])
    initial_covariance = np.array([[0.8, 0.1], [0.1, 0.5]])
    process_covariance = np.stack([
        np.array([[0.16 + 0.01 * t, 0.02], [0.02, 0.11]])
        for t in range(n_time)
    ])
    measurement_covariance = np.diag([0.07, 0.09, 0.06, 0.08])
    states = np.empty((n_time, n_state))
    previous = rng.multivariate_normal(initial_mean, initial_covariance)
    for t in range(n_time):
        previous = transition @ previous + intercept + rng.multivariate_normal(
            np.zeros(n_state), process_covariance[t]
        )
        states[t] = previous
    observations = states @ loadings.T + rng.multivariate_normal(
        np.zeros(n_yield), measurement_covariance, size=n_time
    )
    args = (
        observations, loadings, transition, intercept, process_covariance,
        measurement_covariance, initial_mean, initial_covariance,
    )
    return args


class LinearGaussianCoreTests(unittest.TestCase):
    def test_dns_loadings_use_stable_zero_maturity_limit(self):
        actual = dns_loadings(np.array([0.0, 0.25, 30.0]), 0.5160377358490565)
        self.assertEqual(actual.shape, (3, 3))
        np.testing.assert_array_equal(actual[0], np.array([1.0, 1.0, 0.0]))
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertTrue(np.all(actual[:, 1] >= 0.0))

    def test_factor_dynamics_center_at_long_run_mean(self):
        mean = np.array([3.0, -2.0, 0.5])
        persistence = np.array([0.9, 0.7, -0.2])
        transition, intercept = factor_dynamics(mean, persistence)
        np.testing.assert_allclose(transition @ mean + intercept, mean)

    def test_kalman_likelihood_and_smoother_match_dense_gaussian_reference(self):
        args = _problem()
        y, load, transition, intercept, q, r, mean0, cov0 = args
        result = kalman_filter(y, load, transition, intercept, q, r, mean0, cov0)
        expected_loglik, expected_mean, expected_covariance = _dense_state_reference(*args)
        np.testing.assert_allclose(
            result.log_likelihood, expected_loglik, rtol=1e-10, atol=1e-10
        )

        means, covariances = rts_smoother(result, transition)
        n_time, n_state = means.shape
        dense_mean = expected_mean.reshape(n_time, n_state)
        dense_covariances = np.stack([
            expected_covariance[
                t * n_state:(t + 1) * n_state,
                t * n_state:(t + 1) * n_state,
            ]
            for t in range(n_time)
        ])
        np.testing.assert_allclose(means, dense_mean, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(covariances, dense_covariances, rtol=1e-9, atol=1e-9)

    def test_simulation_smoother_draws_full_conditional_state_paths(self):
        args = _problem(seed=92)
        y, load, transition, intercept, q, r, mean0, cov0 = args
        result = kalman_filter(y, load, transition, intercept, q, r, mean0, cov0)
        draws = np.stack([
            sample_state_path(result, transition, np.random.default_rng(seed))
            for seed in range(2500)
        ])
        _, expected_mean, expected_covariance = _dense_state_reference(*args)
        n_time, n_state = draws.shape[1:]
        np.testing.assert_allclose(
            draws.mean(axis=0), expected_mean.reshape(n_time, n_state), atol=0.045
        )
        terminal = draws[:, -1, 0]
        expected_terminal_variance = expected_covariance[
            (n_time - 1) * n_state, (n_time - 1) * n_state
        ]
        self.assertLess(
            abs(terminal.var(ddof=1) - expected_terminal_variance), 0.07
        )

    def test_kalman_filter_rejects_non_positive_definite_measurement_covariance(self):
        args = list(_problem())
        args[5] = np.diag([0.07, 0.09, 0.06, -0.08])
        with self.assertRaises(np.linalg.LinAlgError):
            kalman_filter(*args)
