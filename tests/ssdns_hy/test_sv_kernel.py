from __future__ import annotations

import unittest

import numpy as np
from scipy.integrate import trapezoid

from src.ssdns_hy.sampler import (
    _draw_stationary_ar1,
    _elliptical_slice,
    _sample_volatility_path,
    _volatility_log_likelihood,
)


class VolatilityKernelTests(unittest.TestCase):
    def test_stationary_directions_covariance_and_reproducibility(self):
        rng = np.random.default_rng(42)
        paths = np.stack([_draw_stationary_ar1(6, 0.8, 2.0, rng) for _ in range(12000)])
        expected = 2.0 * 0.8 ** np.abs(np.subtract.outer(np.arange(6), np.arange(6)))
        np.testing.assert_allclose(np.cov(paths, rowvar=False), expected, atol=0.07)
        np.testing.assert_array_equal(
            _draw_stationary_ar1(20, 0.99, 0.2, np.random.default_rng(7)),
            _draw_stationary_ar1(20, 0.99, 0.2, np.random.default_rng(7)),
        )
        self.assertFalse(np.array_equal(paths[0], paths[1]))
        persistent = np.stack([
            _draw_stationary_ar1(5, 0.99999, 1.0, rng) for _ in range(10000)
        ])
        self.assertAlmostEqual(float(np.var(persistent[:, -1])), 1.0, delta=0.05)
        self.assertAlmostEqual(
            float(np.corrcoef(persistent[:, 0], persistent[:, -1])[0, 1]),
            0.99999**4,
            delta=0.00002,
        )

    def test_constant_likelihood_preserves_correlated_gaussian_prior(self):
        rng = np.random.default_rng(22)
        phi, variance, center = 0.7, 1.5, 0.3
        samples = []
        for _ in range(10000):
            current = center + _draw_stationary_ar1(3, phi, variance, rng)
            direction = _draw_stationary_ar1(3, phi, variance, rng)
            samples.append(_elliptical_slice(current, direction, center, lambda _: 0.0, rng))
        samples = np.asarray(samples)
        expected = variance * phi ** np.abs(np.subtract.outer(np.arange(3), np.arange(3)))
        np.testing.assert_allclose(samples.mean(axis=0), center, atol=0.05)
        np.testing.assert_allclose(np.cov(samples, rowvar=False), expected, atol=0.07)

    def test_gaussian_likelihood_recovers_analytic_posterior_from_different_starts(self):
        prior_mean, prior_variance = 0.3, 1.5
        observation, noise_variance = 1.0, 0.5
        posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / noise_variance)
        posterior_mean = posterior_variance * (
            prior_mean / prior_variance + observation / noise_variance
        )
        for chain, start in enumerate((-5.0, -2.0, 3.0, 6.0)):
            rng = np.random.default_rng(100 + chain)
            current = np.array([start])
            samples = []
            for i in range(5000):
                direction = _draw_stationary_ar1(1, 0.7, prior_variance, rng)
                current = _elliptical_slice(
                    current, direction, prior_mean,
                    lambda x: float(-0.5 * (x[0] - observation) ** 2 / noise_variance),
                    rng,
                )
                if i >= 500:
                    samples.append(current[0])
            self.assertAlmostEqual(np.mean(samples), posterior_mean, delta=0.07)
            self.assertAlmostEqual(np.var(samples), posterior_variance, delta=0.06)

    def test_exact_sv_likelihood_matches_one_dimensional_quadrature(self):
        mean, variance, innovation = -0.5, 0.8, 1.2
        center = mean - variance / 2.0
        grid = np.linspace(-7.0, 5.0, 20001)
        weights = np.exp(
            -0.5 * (grid - center) ** 2 / variance
            -0.5 * (grid + innovation**2 * np.exp(-grid))
        )
        expected_mean = trapezoid(grid * weights, grid) / trapezoid(weights, grid)
        expected_var = trapezoid((grid - expected_mean) ** 2 * weights, grid) / trapezoid(weights, grid)
        rng = np.random.default_rng(912)
        current = np.array([3.0])
        samples = []
        for i in range(12000):
            current = _sample_volatility_path(
                current, np.array([innovation]), mean, 0.9, variance, rng
            )
            if i >= 1000:
                samples.append(current[0])
        self.assertAlmostEqual(np.mean(samples), expected_mean, delta=0.06)
        self.assertAlmostEqual(np.var(samples), expected_var, delta=0.06)

    def test_sv_ar1_density_matches_multivariate_normal(self):
        from scipy.stats import multivariate_normal

        h = np.array([-1.0, -0.7, -0.9, -0.2])
        mean, phi, variance = -0.3, 0.8, 0.6
        covariance = variance * phi ** np.abs(np.subtract.outer(np.arange(4), np.arange(4)))
        expected = multivariate_normal.logpdf(h, mean=np.full(4, mean - variance / 2), cov=covariance)
        self.assertAlmostEqual(_volatility_log_likelihood(h, mean, phi, variance), expected)


if __name__ == "__main__":
    unittest.main()
