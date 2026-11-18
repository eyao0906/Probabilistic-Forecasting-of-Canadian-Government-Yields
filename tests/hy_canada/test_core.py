import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logit

from src.hy_canada.data import MATURITIES, load_panel
from src.hy_canada.dlns import log_posterior, parameters, sample_chain, training_prior
from src.hy_canada.kalman import filter_likelihood, loadings
from src.hy_canada.states import sample_factors, simulate_forecast, volatility_path
from src.hy_canada.svns import elliptical_slice, log_variances, sample_chain as sample_sv_chain, sv_log_posterior, sv_prior


class CoreTests(unittest.TestCase):
    def test_loadings_and_panel(self):
        h = loadings(np.array([0.0, 1.0, 30.0]), 0.7)
        np.testing.assert_allclose(h[0], [1, 1, 0], atol=1e-14)
        self.assertEqual(h.shape, (3, 3))
        root = Path(__file__).resolve().parents[2]
        dates, values = load_panel(root / "data" / "processed" / "canada_weekly.csv",
                                   "2000-01-01", "2015-12-31")
        self.assertEqual(values.shape, (835, len(MATURITIES)))
        self.assertEqual(str(dates[0].date()), "2000-01-07")

    def test_bad_panel_is_rejected(self):
        root = Path(__file__).resolve().parents[2]
        frame = pd.read_csv(root / "data" / "processed" / "canada_weekly.csv").iloc[:4].copy()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            frame.iloc[[0, 0, 2, 3]].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_panel(path, frame.Date.iloc[0], frame.Date.iloc[-1])
            frame.iloc[[0, 2, 3]].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing calendar week"):
                load_panel(path, frame.Date.iloc[0], frame.Date.iloc[-1])
            frame.iloc[0, 2] = np.nan
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                load_panel(path, frame.Date.iloc[0], frame.Date.iloc[-1])

    def test_kalman_matches_joint_gaussian_with_varying_q(self):
        h = loadings(np.array([0.5, 2.0, 10.0, 30.0]), 0.6)
        y = np.array([[1.1, 1.4, 1.3, 1.2], [0.9, 1.2, 1.4, 1.3]])
        mu = np.array([1., -0.3, 0.2])
        rho = np.array([0.8, 0.6, 0.7])
        q = np.array([[0.1, 0.2, 0.15], [0.2, 0.08, 0.1]])
        r = np.array([0.04, 0.05, 0.06, 0.07])
        m0 = np.array([1.3, -0.1, 0.4])
        p0 = np.diag([0.7, 0.4, 0.5])
        a = np.diag(rho)
        m1 = mu + rho * (m0 - mu)
        m2 = mu + rho * (m1 - mu)
        p1 = a @ p0 @ a + np.diag(q[0])
        p2 = a @ p1 @ a + np.diag(q[1])
        cross = p1 @ a
        covariance = np.block([[h @ p1 @ h.T + np.diag(r), h @ cross @ h.T],
                               [h @ cross.T @ h.T, h @ p2 @ h.T + np.diag(r)]])
        expected = multivariate_normal.logpdf(y.ravel(), mean=np.r_[h @ m1, h @ m2],
                                              cov=covariance)
        actual = filter_likelihood(y, h, mu, rho, q, r, m0, p0)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_prior_transform_and_reproducibility(self):
        root = Path(__file__).resolve().parents[2]
        _, y = load_panel(root / "data" / "processed" / "canada_weekly.csv",
                          "2000-01-01", "2000-03-31")
        prior = training_prior(y)
        mu, rho, q, r, decay = parameters(prior.center)
        self.assertTrue(np.all((rho > 0) & (rho < 1)))
        self.assertTrue(np.all(q > 0) and np.all(r > 0))
        self.assertAlmostEqual(decay, 0.7)
        expected = filter_likelihood(y, loadings(MATURITIES, decay), mu, rho,
                                     np.tile(q, (len(y), 1)), r, prior.initial_mean,
                                     prior.initial_covariance)
        self.assertAlmostEqual(log_posterior(prior.center, y, prior), expected)
        first = sample_chain(y, prior, 902, 2, 4)
        second = sample_chain(y, prior, 902, 2, 4)
        np.testing.assert_array_equal(first["draws"], second["draws"])
        joint = sample_chain(y, prior, 902, 2, 4, np.eye(12))
        self.assertEqual(len(joint["acceptance"]), 6)
        self.assertTrue(np.isfinite(joint["draws"]).all())

    def test_transform_jacobians(self):
        z = np.linspace(-1.3, 1.1, 18)
        mu, rho, q, r, decay = parameters(z)
        np.testing.assert_allclose(logit(rho), z[3:6])
        np.testing.assert_allclose(np.log(q), z[6:9])
        np.testing.assert_allclose(np.log(r), z[9:17])
        self.assertAlmostEqual(logit((decay - 0.05) / 1.95), z[17])
        delta = 1e-6
        shifted = z.copy()
        shifted[17] += delta
        numerical_derivative = (parameters(shifted)[4] - decay) / delta
        fraction = (decay - 0.05) / 1.95
        self.assertAlmostEqual(numerical_derivative, 1.95 * fraction * (1 - fraction), places=6)

    def test_smoother_matches_analytic_conditional(self):
        rng = np.random.default_rng(42)
        h = loadings(np.array([1.0, 5.0, 10.0, 30.0]), 0.7)
        mu = np.array([1., -0.5, 0.4])
        rho = np.array([0.8, 0.7, 0.6])
        a = np.diag(rho)
        m0 = np.array([1.2, -0.3, 0.1])
        p0 = np.diag([0.4, 0.3, 0.2])
        q = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.15]])
        r = np.full(4, 0.1)
        y = np.array([[1.1, 1.2, 1.4, 1.5], [1.2, 1.1, 1.3, 1.4]])
        m1 = mu + rho * (m0 - mu)
        m2 = mu + rho * (m1 - mu)
        p1 = a @ p0 @ a + np.diag(q[0])
        p2 = a @ p1 @ a + np.diag(q[1])
        p = np.block([[p1, p1 @ a], [a @ p1, p2]])
        obs = np.kron(np.eye(2), h)
        obs_cov = obs @ p @ obs.T + np.diag(np.tile(r, 2))
        mean = np.r_[m1, m2] + p @ obs.T @ np.linalg.solve(obs_cov, y.ravel() - obs @ np.r_[m1, m2])
        cov = p - p @ obs.T @ np.linalg.solve(obs_cov, obs @ p)
        np.linalg.cholesky(cov)
        samples = np.array([sample_factors(y, h, mu, rho, q, r, m0, p0, rng).ravel()
                            for _ in range(6000)])
        np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.025)
        np.testing.assert_allclose(np.cov(samples.T), cov, atol=0.02)

    def test_sv_and_forecast_moments(self):
        rng = np.random.default_rng(41)
        paths = np.array([volatility_path(-2, 0.8, 0.5, 2, rng) for _ in range(12000)])
        np.testing.assert_allclose(paths.mean(axis=0), [-2, -2], atol=0.03)
        np.testing.assert_allclose(np.cov(paths.T), [[0.5, 0.4], [0.4, 0.5]], atol=0.03)
        h = loadings(np.array([1.0, 10.0]), 0.7)
        state, mu, rho = np.array([1., 0.2, -0.1]), np.zeros(3), np.full(3, 0.8)
        q, r = np.full(3, 0.2), np.full(2, 0.1)
        forecasts = np.array([simulate_forecast(state, mu, rho, q, r, h, 1, rng)[0]
                              for _ in range(12000)])
        np.testing.assert_allclose(forecasts.mean(axis=0), h @ (rho * state), atol=0.025)
        np.testing.assert_allclose(np.cov(forecasts.T), h @ np.diag(q) @ h.T + np.diag(r), atol=0.025)

    def test_noncentered_sv_recursion_and_target(self):
        noise = np.array([[1., -1., 0.5], [0.2, 0.3, -0.2], [0.4, -0.5, 0.1]])
        phi = np.array([0.8, 0.5, 0.9])
        variance = np.array([0.4, 0.3, 0.2])
        logq = np.log([0.2, 0.3, 0.4])
        actual = log_variances(noise, phi, variance, logq)
        u = noise[0].copy()
        expected = [logq - variance / 2 + np.sqrt(variance) * u]
        for t in range(1, len(noise)):
            u = phi * u + np.sqrt(1 - phi**2) * noise[t]
            expected.append(logq - variance / 2 + np.sqrt(variance) * u)
        np.testing.assert_allclose(actual, expected)
        root = Path(__file__).resolve().parents[2]
        _, y = load_panel(root / "data" / "processed" / "canada_weekly.csv",
                          "2000-01-01", "2000-03-31")
        prior = training_prior(y)
        center, _ = sv_prior(prior)
        noise = np.zeros((len(y), 3))
        mu, rho, _, r, decay = parameters(center)
        h = log_variances(noise, phi=np.full(3, .95), variance=np.full(3, .5),
                          log_mean_q=center[6:9])
        ll = filter_likelihood(y, loadings(MATURITIES, decay), mu, rho, np.exp(h), r,
                               prior.initial_mean, prior.initial_covariance)
        self.assertAlmostEqual(sv_log_posterior(center, noise, y, prior), ll)
        first = sample_sv_chain(y, prior, 91, 2, 3)
        second = sample_sv_chain(y, prior, 91, 2, 3)
        np.testing.assert_array_equal(first["draws"], second["draws"])
        np.testing.assert_array_equal(first["paths"], second["paths"])

    def test_elliptical_slice_normal_conditional(self):
        rng = np.random.default_rng(864)
        y, variance = 1.5, 0.7
        def likelihood(x):
            return -0.5 * (x[0] - y) ** 2 / variance
        x = np.zeros(1)
        ll = likelihood(x)
        values = []
        for i in range(16000):
            x, ll, _ = elliptical_slice(x, ll, likelihood, rng)
            if i >= 1000:
                values.append(x[0])
        self.assertAlmostEqual(np.mean(values), y / (1 + variance), delta=0.035)
        self.assertAlmostEqual(np.var(values), variance / (1 + variance), delta=0.035)


if __name__ == "__main__":
    unittest.main()
