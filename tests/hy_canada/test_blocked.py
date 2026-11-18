import unittest
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.integrate import quad
from scipy.stats import invwishart, multivariate_normal, multivariate_t, norm

from src.hy_canada.blocked_dlns import (
    Priors, decay_probabilities, dynamics_conditionals, innovation_conditional,
    measurement_conditional, sample_chain, training_priors,
)
from src.hy_canada.blocked_states import draw_factors, draw_factors_sv
from src.hy_canada.blocked_sv import (
    MEANS, VARIANCES, WEIGHTS, log_correction_weight, marginal_log_target,
    mixture_indicators, mixture_log_probabilities, parameter_log_acceptance, scalar_smoother,
)
from src.hy_canada.blocked_svns import (
    dynamics_conditionals as sv_dynamics_conditionals, sample_chain as sample_sv_chain,
)
from src.hy_canada.data import MATURITIES, load_panel
from src.hy_canada.kalman import loadings


class BlockedTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        self.y = rng.normal(size=(8, 8))
        self.f = rng.normal(size=(8, 3))
        self.prior = Priors(np.zeros(3), np.eye(3), np.array([0.1, -0.2, 0.3]),
                            np.ones(3), 0.8, 0.2, 2.5, np.ones(8),
                            5.0, np.eye(3))

    def test_measurement_and_griddy_conditionals(self):
        h = loadings(MATURITIES, 0.7)
        shape, scale = measurement_conditional(self.y, self.f, h, self.prior)
        self.assertEqual(shape, 6.5)
        np.testing.assert_allclose(scale, 1 + np.sum((self.y - self.f @ h.T)**2, axis=0) / 2)
        grid = np.linspace(0.2, 1.2, 30)
        r = np.linspace(0.1, 0.8, 8)
        logs = np.array([-0.5 * np.sum((self.y - self.f @ loadings(MATURITIES, x).T)**2 / r)
                         for x in grid])
        np.testing.assert_allclose(decay_probabilities(self.y, self.f, r, grid),
                                   np.exp(logs - logsumexp(logs)), atol=1e-12)
        with self.assertRaises(ValueError):
            decay_probabilities(self.y, self.f, r, grid[::-1])

    def test_dynamics_conditionals_with_full_q(self):
        q = np.array([[0.7, 0.2, -0.1], [0.2, 0.5, 0.1], [-0.1, 0.1, 0.4]])
        mu = np.array([0.1, -0.2, 0.3])
        rho = np.array([0.7, 0.6, 0.8])
        p = np.linalg.inv(q)
        for j in range(3):
            mm, mv, rm, rv = dynamics_conditionals(self.f, mu, rho, q, self.prior, j)
            def target(m, a):
                resid = self.f[1:] - m - a * (self.f[:-1] - m)
                return (-0.5 * np.einsum("ti,ij,tj", resid, p, resid)
                        - 0.5 * np.sum((m - self.prior.mu_mean)**2 / self.prior.mu_variance)
                        - 0.5 * np.sum((a - self.prior.rho_mean)**2 / self.prior.rho_variance))
            for value in (-0.4, 0.35):
                other = mu.copy()
                other[j] = value
                diff = target(other, rho) - target(mu, rho)
                expected = -0.5 * ((value - mm)**2 - (mu[j] - mm)**2) / mv
                self.assertAlmostEqual(diff, expected, places=10)
                other = rho.copy()
                other[j] = value
                diff = target(mu, other) - target(mu, rho)
                expected = -0.5 * ((value - rm)**2 - (rho[j] - rm)**2) / rv
                self.assertAlmostEqual(diff, expected, places=10)
        df, scale = innovation_conditional(self.f, mu, rho, self.prior)
        residuals = self.f[1:] - mu - rho * (self.f[:-1] - mu)
        self.assertEqual(df, self.prior.q_df + 7)
        np.testing.assert_allclose(scale, self.prior.q_scale + residuals.T @ residuals)
        np.linalg.cholesky(invwishart.rvs(df=df, scale=scale, random_state=np.random.default_rng(2)))

    def test_full_q_smoother_against_dense_conditional(self):
        rng = np.random.default_rng(94)
        h = loadings(np.array([1., 5., 10., 30.]), 0.7)
        mu = np.array([1., -0.5, 0.4])
        rho = np.array([0.8, 0.7, 0.6])
        a = np.diag(rho)
        m0 = np.array([1.2, -0.3, 0.1])
        p0 = np.diag([0.4, 0.3, 0.2])
        q = np.array([[0.2, 0.06, 0.03], [0.06, 0.1, -0.01], [0.03, -0.01, 0.15]])
        r = np.full(4, 0.1)
        y = np.array([[1.1, 1.2, 1.4, 1.5], [1.2, 1.1, 1.3, 1.4]])
        m1 = mu + rho * (m0 - mu)
        p1 = a @ p0 @ a + q
        p = np.block([[p0, p0 @ a], [a @ p0, p1]])
        obs = np.kron(np.eye(2), h)
        oc = obs @ p @ obs.T + np.diag(np.tile(r, 2))
        mean = np.r_[m0, m1] + p @ obs.T @ np.linalg.solve(oc, y.ravel() - obs @ np.r_[m0, m1])
        cov = p - p @ obs.T @ np.linalg.solve(oc, obs @ p)
        samples = np.array([draw_factors(y, h, mu, rho, q, r, m0, p0,
                                         rng.standard_normal((2, 3))).ravel()
                            for _ in range(10000)])
        np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.025)
        np.testing.assert_allclose(np.cov(samples.T), cov, atol=0.015)
        z = rng.standard_normal((2, 3))
        diagonal = np.diag(np.diag(q))
        np.testing.assert_allclose(
            draw_factors(y, h, mu, rho, diagonal, r, m0, p0, z),
            draw_factors_sv(y, h, mu, rho, np.log(np.diag(diagonal))[None, :], r, m0, p0, z))

    def test_time_varying_factor_smoother_indexing(self):
        rng = np.random.default_rng(216)
        h = loadings(np.array([1., 5., 10., 30.]), .7)
        y = np.array([[1., 1.1, 1.2, 1.3], [1.1, 1.2, 1.3, 1.4], [.9, 1., 1.1, 1.2]])
        mu = np.array([1., -.3, .2])
        rho = np.array([.8, .7, .6])
        a = np.diag(rho)
        m0 = np.array([1.2, -.1, .4])
        p0 = np.diag([.4, .3, .2])
        q = np.array([[.05, .1, .2], [.4, .3, .06]])
        r = np.full(4, .1)
        m1 = mu + rho * (m0 - mu)
        m2 = mu + rho * (m1 - mu)
        p1 = a @ p0 @ a + np.diag(q[0])
        p2 = a @ p1 @ a + np.diag(q[1])
        covariance = np.block([[p0, p0 @ a, p0 @ a @ a],
                               [a @ p0, p1, p1 @ a],
                               [a @ a @ p0, a @ p1, p2]])
        obs = np.kron(np.eye(3), h)
        obs_cov = obs @ covariance @ obs.T + np.diag(np.tile(r, 3))
        mean = np.r_[m0, m1, m2]
        mean += covariance @ obs.T @ np.linalg.solve(obs_cov, y.ravel() - obs @ mean)
        conditional = covariance - covariance @ obs.T @ np.linalg.solve(obs_cov, obs @ covariance)
        samples = np.array([draw_factors_sv(y, h, mu, rho, np.log(q), r, m0, p0,
                                             rng.standard_normal((3, 3))).ravel()
                            for _ in range(6000)])
        np.testing.assert_allclose(samples.mean(axis=0), mean, atol=.02)
        np.testing.assert_allclose(np.cov(samples.T), conditional, atol=.015)

    def test_short_chain_reproducibility(self):
        prior = training_priors(self.y)
        grid = np.linspace(0.2, 1.2, 30)
        a = sample_chain(self.y, prior, 19, 2, 4, grid, keep_paths=2)
        b = sample_chain(self.y, prior, 19, 2, 4, grid, keep_paths=2)
        np.testing.assert_array_equal(a["draws"], b["draws"])
        self.assertEqual(a["paths"].shape, (2, 8, 3))

    def test_sv_mixture_smoother_and_correction(self):
        rng = np.random.default_rng(116)
        observation = np.array([-1.0, -0.5])
        h = np.array([-1.1, -0.6])
        actual = np.exp(mixture_log_probabilities(observation, h))
        direct = WEIGHTS * norm.pdf(observation[:, None], loc=h[:, None] + MEANS,
                                    scale=np.sqrt(VARIANCES))
        np.testing.assert_allclose(actual, direct / direct.sum(axis=1, keepdims=True))
        self.assertTrue(np.all(mixture_indicators(observation, h, rng) < 7))
        mu, phi, sigma = -1.0, 0.7, 0.3
        obs_var = np.array([0.3, 0.5])
        stationary = sigma**2 / (1 - phi**2)
        covariance = stationary * np.array([[1., phi], [phi, 1.]])
        mean = mu + covariance @ np.linalg.solve(covariance + np.diag(obs_var), observation - mu)
        conditional = covariance - covariance @ np.linalg.solve(covariance + np.diag(obs_var), covariance)
        samples = np.array([scalar_smoother(observation, obs_var, mu, phi, sigma,
                                            rng.standard_normal(2)) for _ in range(12000)])
        np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.015)
        np.testing.assert_allclose(np.cov(samples.T), conditional, atol=0.01)
        errors = np.array([0.05, -0.2])
        offset = 0.001
        z = np.log(errors**2 + offset)
        mixture = (WEIGHTS * norm.pdf(z[:, None], loc=h[:, None] + MEANS,
                                       scale=np.sqrt(VARIANCES))).sum(axis=1)
        expected = np.sum(norm.logpdf(errors, scale=np.exp(h / 2)) - np.log(mixture))
        self.assertAlmostEqual(log_correction_weight(errors, h, offset), expected)
        with self.assertRaises(ValueError):
            log_correction_weight(np.array([0.0]), np.array([0.0]), 0.0)

    def test_sv_marginal_kalman_likelihood(self):
        theta = np.array([-1.0, np.log(0.7 / 0.3), np.log(0.3)])
        indices = np.array([1, 4])
        observation = np.array([-1.5, 0.2])
        mu, phi, sigma = -1.0, 0.7, 0.3
        covariance = sigma**2 / (1 - phi**2) * np.array([[1., phi], [phi, 1.]])
        expected = multivariate_normal.logpdf(observation, mean=mu + MEANS[indices],
                                              cov=covariance + np.diag(VARIANCES[indices]))
        self.assertAlmostEqual(marginal_log_target(theta, observation, indices, theta), expected)
        candidate = theta + np.array([0.1, -0.2, 0.15])
        center = theta + 0.2
        scale = np.diag([0.4, 0.7, 0.2])
        ratio = (marginal_log_target(candidate, observation, indices, theta)
                 - marginal_log_target(theta, observation, indices, theta)
                 + multivariate_t.logpdf(theta, loc=center, shape=scale, df=5)
                 - multivariate_t.logpdf(candidate, loc=center, shape=scale, df=5))
        self.assertAlmostEqual(parameter_log_acceptance(theta, candidate, observation, indices,
                                                        theta, center, scale), ratio)

    def test_sv_weighted_dynamics_conditional(self):
        factors = np.array([[1., -.2, .4], [.8, -.1, .6], [1.1, -.4, .3]])
        mu = np.array([.6, -.2, .1])
        rho = np.array([.8, .7, .6])
        h = np.log(np.array([[.2, .1, .3], [.08, .3, .15]]))
        prior = training_priors(np.tile(np.arange(8), (4, 1)) + np.arange(4)[:, None] * .1)
        mean, var, rho_mean, rho_var = sv_dynamics_conditionals(factors, mu, rho, h, prior, 0)
        def direct(value):
            residual = factors[1:, 0] - value - rho[0] * (factors[:-1, 0] - value)
            return (-.5 * np.sum(residual**2 / np.exp(h[:, 0]))
                    - .5 * (value - prior.mu_mean[0])**2 / prior.mu_variance[0])
        for value in [mean - .3, mean + .4]:
            self.assertAlmostEqual(direct(value) - direct(mean), -.5 * (value - mean)**2 / var)
        def direct_rho(value):
            residual = factors[1:, 0] - mu[0] - value * (factors[:-1, 0] - mu[0])
            return (-.5 * np.sum(residual**2 / np.exp(h[:, 0]))
                    - .5 * (value - prior.rho_mean)**2 / prior.rho_variance)
        for value in [rho_mean - .1, rho_mean + .2]:
            self.assertAlmostEqual(direct_rho(value) - direct_rho(rho_mean),
                                   -.5 * (value - rho_mean)**2 / rho_var)

    def test_short_sv_chain_reproducibility(self):
        root = Path(__file__).resolve().parents[2]
        _, y = load_panel(root / "data" / "processed" / "canada_weekly.csv",
                          "2000-01-01", "2000-03-31")
        prior = training_priors(y)
        args = (y, prior, 999, 2, 4, np.linspace(.2, 1.2, 30))
        first = sample_sv_chain(*args, keep_paths=2, inner_steps=2)
        second = sample_sv_chain(*args, keep_paths=2, inner_steps=2)
        np.testing.assert_array_equal(first["draws"], second["draws"])
        self.assertTrue(np.isfinite(first["draws"]).all())
        self.assertEqual(first["volatility_paths"].shape, (2, len(y) - 1, 3))

    def test_corrected_mixture_kernel_invariance(self):
        rng = np.random.default_rng(108)
        error, offset, mu, phi, sigma = 0.08, 0.001, -1.5, 0.7, 0.5
        variance = sigma**2 / (1 - phi**2)
        observation = np.array([np.log(error**2 + offset)])
        def exact_density(x):
            return norm.pdf(x, loc=mu, scale=np.sqrt(variance)) * norm.pdf(error, scale=np.exp(x / 2))
        normalizer = quad(exact_density, -20, 10)[0]
        expected = quad(lambda x: x * exact_density(x), -20, 10)[0] / normalizer
        path = np.array([mu])
        draws = []
        for i in range(16000):
            index = mixture_indicators(observation, path, rng)
            proposed = scalar_smoother(observation - MEANS[index], VARIANCES[index],
                                       mu, phi, sigma, rng.standard_normal(1))
            logratio = (log_correction_weight(np.array([error]), proposed, offset)
                        - log_correction_weight(np.array([error]), path, offset))
            if np.log(rng.random()) < logratio:
                path = proposed
            if i >= 1000:
                draws.append(path[0])
        self.assertAlmostEqual(np.mean(draws), expected, delta=0.035)

    def test_corrected_joint_volatility_mean_and_path(self):
        rng = np.random.default_rng(109)
        error, offset, prior_mean = .08, .001, -1.5
        mean_var, path_var = .16, .5
        joint_cov = np.array([[mean_var, mean_var], [mean_var, mean_var + path_var]])
        prior = np.full(2, prior_mean)
        observation = np.array([np.log(error**2 + offset)])
        def exact_density(x):
            return (norm.pdf(x, loc=prior_mean, scale=np.sqrt(mean_var + path_var))
                    * norm.pdf(error, scale=np.exp(x / 2)))
        normalizer = quad(exact_density, -20, 10)[0]
        expected_h = quad(lambda x: x * exact_density(x), -20, 10)[0] / normalizer
        expected_mu = prior_mean + mean_var / (mean_var + path_var) * (expected_h - prior_mean)
        state = prior.copy()
        samples = []
        for i in range(16000):
            index = mixture_indicators(observation, state[1:], rng)[0]
            variance = joint_cov[1, 1] + VARIANCES[index]
            posterior_mean = prior + joint_cov[:, 1] * (observation[0] - MEANS[index] - prior_mean) / variance
            posterior_cov = joint_cov - np.outer(joint_cov[:, 1], joint_cov[1, :]) / variance
            proposal = rng.multivariate_normal(posterior_mean, posterior_cov)
            ratio = (log_correction_weight(np.array([error]), proposal[1:], offset)
                     - log_correction_weight(np.array([error]), state[1:], offset))
            if np.log(rng.random()) < ratio:
                state = proposal
            if i >= 1000:
                samples.append(state.copy())
        np.testing.assert_allclose(np.mean(samples, axis=0), [expected_mu, expected_h], atol=.035)


if __name__ == "__main__":
    unittest.main()
