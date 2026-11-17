from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from scipy.special import expit
from scipy.stats import beta, invgamma, multivariate_normal

from src.ssdns_hy import calibrate_training_priors, dns_loadings, simulate_yields
from src.ssdns_hy.sampler import (
    _draw_factor_parameters,
    _draw_initial_factor,
    _draw_measurement_variance,
    _factor_log_variance_target,
    _update_decay_marginal,
    _update_volatility_mean,
    _update_volatility_persistence,
    _update_volatility_variance,
)
from unittest.mock import patch


class FixedRandom:
    def __init__(self, proposed_step, uniform):
        self.proposed_step = proposed_step
        self.uniform_value = uniform

    def normal(self, loc, scale):
        return self.proposed_step

    def uniform(self):
        return self.uniform_value


class ConditionalTests(unittest.TestCase):
    def test_factor_parameter_gibbs_conditionals(self):
        rng = np.random.default_rng(42)
        states = rng.normal(size=(6, 3))
        initial = np.array([.4, -.2, .8])
        variance = rng.uniform(.2, 1., size=(6, 3))
        priors, _ = calibrate_training_priors(rng.normal(size=(8, 4)), np.array([.25, 1., 5., 10.]))
        mean = np.array([.1, .2, .3])
        persistence = np.array([.6, .3, -.2])
        previous = np.vstack((initial, states[:-1]))
        captures = []

        class CaptureRng:
            def normal(self, loc, scale):
                captures.append((loc, scale))
                return loc

        with patch("src.ssdns_hy.sampler.truncnorm.rvs", side_effect=lambda *args, **kw: kw["loc"]) as draw:
            updated_mean, updated_phi = _draw_factor_parameters(
                states, initial, variance, mean, persistence, priors, CaptureRng()
            )
        for j in range(3):
            weights = 1 / variance[:, j]
            b = persistence[j]
            precision = 1 / priors.factor_mean_sd[j]**2 + (1-b)**2 * weights.sum() + 1 / priors.initial_state_variance
            rhs = (priors.factor_mean_center[j] / priors.factor_mean_sd[j]**2
                   + (1-b) * np.sum((states[:, j] - b * previous[:, j]) * weights)
                   + initial[j] / priors.initial_state_variance)
            self.assertAlmostEqual(captures[j][0], rhs / precision)
            self.assertAlmostEqual(captures[j][1], np.sqrt(1 / precision))
            regressor = previous[:, j] - updated_mean[j]
            response = states[:, j] - updated_mean[j]
            p = 1 / priors.persistence_sd[j]**2 + np.sum(regressor**2 * weights)
            mu = (priors.persistence_center[j] / priors.persistence_sd[j]**2
                  + np.sum(regressor * response * weights)) / p
            self.assertAlmostEqual(draw.call_args_list[j].kwargs["loc"], mu)
            self.assertAlmostEqual(draw.call_args_list[j].kwargs["scale"], np.sqrt(1 / p))
            self.assertAlmostEqual(updated_phi[j], mu)

    def test_initial_state_conditional_matches_normal_moments(self):
        mean = np.array([1., -1., 0.])
        first = np.array([1.2, -0.5, 0.3])
        persistence = np.array([0.8, 0.4, -0.5])
        q = np.array([0.3, 0.5, 0.2])
        prior_variance = 2.
        total = persistence**2 * prior_variance + q
        expected_mean = mean + persistence * prior_variance / total * (first - mean)
        expected_var = prior_variance * q / total
        rng = np.random.default_rng(38)
        samples = np.stack([
            _draw_initial_factor(first, mean, persistence, q, prior_variance, rng)
            for _ in range(12000)
        ])
        np.testing.assert_allclose(samples.mean(axis=0), expected_mean, atol=0.03)
        np.testing.assert_allclose(samples.var(axis=0), expected_var, atol=0.03)

    def test_measurement_and_factor_variance_conditionals(self):
        rng = np.random.default_rng(33)
        y = rng.normal(size=(8, 4))
        maturities = np.array([0.25, 1., 5., 10.])
        priors, _ = calibrate_training_priors(y, maturities)
        states = rng.normal(size=(8, 3))
        loadings = dns_loadings(maturities, .5)
        scale = 2 * priors.measurement_variance_scale + np.sum((y - states @ loadings.T)**2, axis=0) / 2
        shape = priors.measurement_shape + len(y) / 2
        samples = np.stack([
            _draw_measurement_variance(y, states, loadings, priors, rng)
            for _ in range(9000)
        ])
        np.testing.assert_allclose(samples.mean(axis=0), invgamma.mean(shape, scale=scale), rtol=.04)
        residual = np.array([.3, -.7, .1])
        value, center, sd = -1.2, -0.4, .9
        direct = (-.5 * ((value - center) / sd)**2
                  + np.sum(-.5 * (value + residual**2 * np.exp(-value))))
        self.assertAlmostEqual(_factor_log_variance_target(value, residual, center, sd), direct)

    def test_decay_metropolis_ratio_matches_dense_joint_yield_likelihood(self):
        rng = np.random.default_rng(111)
        tau = np.array([.25, 1., 5., 10.])
        y = rng.normal(size=(4, 4))
        priors, _ = calibrate_training_priors(y, tau)
        priors = replace(priors, initial_state_variance=1.5)
        mean = np.array([.3, -.4, .1])
        persistence = np.array([.8, .6, .3])
        q = np.diag([.2, .3, .4])
        r = np.array([.1, .2, .15, .12])
        current, step = .65, -.25
        fraction = (current - priors.decay_lower) / (priors.decay_upper - priors.decay_lower)
        proposed_fraction = expit(np.log(fraction / (1 - fraction)) + step)
        proposed = priors.decay_lower + (priors.decay_upper - priors.decay_lower) * proposed_fraction

        def dense_log_likelihood(decay):
            loadings = dns_loadings(tau, decay)
            state_cov = np.empty((4, 4, 3, 3))
            transition = np.diag(persistence)
            for t in range(4):
                previous = np.eye(3) * priors.initial_state_variance if t == 0 else state_cov[t-1, t-1]
                state_cov[t, t] = transition @ previous @ transition.T + q
                for s in range(t):
                    state_cov[t, s] = transition @ state_cov[t-1, s]
                    state_cov[s, t] = state_cov[t, s].T
            covariance = np.block([[state_cov[t, s] for s in range(4)] for t in range(4)])
            operator = np.kron(np.eye(4), loadings)
            return multivariate_normal.logpdf(
                y.ravel(), mean=np.tile(loadings @ mean, 4),
                cov=operator @ covariance @ operator.T + np.kron(np.eye(4), np.diag(r)),
            )

        ratio = (dense_log_likelihood(proposed) - dense_log_likelihood(current)
                 + np.log(proposed_fraction * (1 - proposed_fraction))
                 - np.log(fraction * (1 - fraction)))
        probability = min(1., np.exp(ratio))
        self.assertLess(probability, 0.999)
        for u, accepted_expected in ((probability / 2, True), ((probability + 1) / 2, False)):
            result, accepted, _ = _update_decay_marginal(
                current, y, tau, q, r, mean, persistence, priors, FixedRandom(step, u)
            )
            self.assertEqual(accepted, accepted_expected)
            self.assertAlmostEqual(result, proposed if accepted_expected else current)

    def test_volatility_mean_conditional_matches_gaussian_posterior(self):
        h = np.array([-.8, -.3, -.5, .2])
        phi, variance, prior_center, prior_sd = .8, .4, -.6, 1.2
        covariance = variance * phi ** np.abs(np.subtract.outer(np.arange(4), np.arange(4)))
        precision_matrix = np.linalg.inv(covariance)
        ones = np.ones(4)
        precision = 1 / prior_sd**2 + ones @ precision_matrix @ ones
        posterior_mean = (prior_center / prior_sd**2 + ones @ precision_matrix @ (h + variance / 2)) / precision
        rng = np.random.default_rng(19)
        samples = [_update_volatility_mean(h, phi, variance, prior_center, prior_sd, rng) for _ in range(10000)]
        self.assertAlmostEqual(np.mean(samples), posterior_mean, delta=.025)
        self.assertAlmostEqual(np.var(samples), 1 / precision, delta=.02)

    def test_volatility_transformed_mh_acceptance_ratios(self):
        h = np.array([-.8, -.3, -.5, .2])
        mean, phi, variance = -.6, .8, .4
        a, b, upper = 20., 1.5, 4.

        def direct_logpdf(p, v):
            covariance = v * p ** np.abs(np.subtract.outer(np.arange(len(h)), np.arange(len(h))))
            return multivariate_normal.logpdf(h, mean=np.full(len(h), mean - v / 2), cov=covariance)

        proposed_phi = .3
        delta_phi = np.arctanh(proposed_phi) - np.arctanh(phi)
        phi_ratio = (direct_logpdf(proposed_phi, variance) - direct_logpdf(phi, variance)
                     + beta.logpdf((1 + proposed_phi) / 2, a, b) - beta.logpdf((1 + phi) / 2, a, b)
                     + np.log1p(-proposed_phi**2) - np.log1p(-phi**2))
        probability = min(1., np.exp(phi_ratio))
        self.assertLess(probability, .999)
        for u, expected in ((probability / 2, True), ((probability + 1) / 2, False)):
            result, accepted = _update_volatility_persistence(
                h, mean, phi, variance, a, b, FixedRandom(delta_phi, u)
            )
            self.assertEqual(accepted, expected)
            self.assertAlmostEqual(result, proposed_phi if expected else phi)

        proposed_variance = 1.5
        x, xp = variance / upper, proposed_variance / upper
        delta_v = np.log(xp / (1 - xp)) - np.log(x / (1 - x))
        var_ratio = (direct_logpdf(phi, proposed_variance) - direct_logpdf(phi, variance)
                     - proposed_variance + variance
                     + np.log(xp * (1 - xp)) - np.log(x * (1 - x)))
        probability = min(1., np.exp(var_ratio))
        self.assertLess(probability, .999)
        for u, expected in ((probability / 2, True), ((probability + 1) / 2, False)):
            result, accepted = _update_volatility_variance(
                h, mean, phi, variance, upper, FixedRandom(delta_v, u)
            )
            self.assertEqual(accepted, expected)
            self.assertAlmostEqual(result, proposed_variance if expected else variance)

    def test_synthetic_initial_factor_uses_declared_prior(self):
        tau = np.array([.25, 1., 5., 10.])
        kwargs = dict(
            n_observations=5, maturities_years=tau, decay_per_year=.5,
            factor_mean=np.array([1., 2., 3.]), factor_persistence=np.array([.8, .6, .4]),
            measurement_variance=np.full(4, .01), model="SS_M3",
            log_variance_mean=np.log(np.full(3, .1)), initial_factor_variance=4.,
        )
        draws = np.stack([
            simulate_yields(rng=np.random.default_rng(seed), **kwargs)["initial_factor"]
            for seed in range(4000)
        ])
        np.testing.assert_allclose(draws.mean(axis=0), kwargs["factor_mean"], atol=.09)
        np.testing.assert_allclose(draws.var(axis=0), 4., atol=.2)


if __name__ == "__main__":
    unittest.main()
