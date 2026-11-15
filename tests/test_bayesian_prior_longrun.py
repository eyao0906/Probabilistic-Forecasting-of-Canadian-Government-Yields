"""Tests for the isolated long-run-mean Bayesian DNS parameterization."""
import unittest

import numpy as np
from scipy.stats import norm

from src import bayesian_finite as bf
from src import bayesian_revised as br
from src.bayesian_prior_longrun import (
    LongRunPrior,
    mean_conditional,
    persistence_conditional,
    sample_chain,
    validate_draws,
)


class LongRunConditionalTests(unittest.TestCase):
    def setUp(self):
        self.prior = LongRunPrior(
            volatility=bf.FinitePrior(br.Prior(omega_mean=-2.0, omega_variance=1.0)),
            mean_center=0.4,
            mean_variance=2.25,
            persistence_center=0.8,
            persistence_variance=0.03**2,
        )
        self.y = np.array([0.3, -0.2, 0.1, 0.8])
        self.lag = np.array([0.1, 0.4, -0.1, 0.7])
        self.log_variance = np.array([-2.0, -1.5, -2.4, -1.8])
        self.weight = np.exp(-self.log_variance)

    def test_gaussian_full_conditionals_match_the_factor_likelihood(self):
        b = 0.73
        mean_location, mean_variance = mean_conditional(
            self.y, self.lag, self.log_variance, b, self.prior
        )

        def mean_target(value):
            residual = self.y - ((1 - b) * value + b * self.lag)
            return norm.logpdf(value, self.prior.mean_center, np.sqrt(self.prior.mean_variance)) - 0.5 * np.sum(self.weight * residual**2)

        for value in [-0.5, 0.7, 1.4]:
            self.assertAlmostEqual(
                mean_target(value) - mean_target(mean_location),
                norm.logpdf(value, mean_location, np.sqrt(mean_variance))
                - norm.logpdf(mean_location, mean_location, np.sqrt(mean_variance)),
                places=10,
            )

        mean = 0.52
        b_location, b_variance = persistence_conditional(
            self.y, self.lag, self.log_variance, mean, self.prior
        )

        def persistence_target(value):
            residual = self.y - (mean + value * (self.lag - mean))
            return norm.logpdf(value, self.prior.persistence_center, np.sqrt(self.prior.persistence_variance)) - 0.5 * np.sum(self.weight * residual**2)

        for value in [-0.2, 0.4, 0.91]:
            self.assertAlmostEqual(
                persistence_target(value) - persistence_target(b_location),
                norm.logpdf(value, b_location, np.sqrt(b_variance))
                - norm.logpdf(b_location, b_location, np.sqrt(b_variance)),
                places=10,
            )


class LongRunSamplerTests(unittest.TestCase):
    def test_both_model_samplers_preserve_stationarity_and_parameterization(self):
        rng = np.random.default_rng(42)
        x = np.empty(90)
        x[0] = 0.2
        for t in range(1, len(x)):
            x[t] = 0.04 + 0.82 * x[t - 1] + rng.normal(0, 0.25)
        base = br.freeze_priors(np.column_stack((x, x, x)))[0]
        prior = LongRunPrior(
            volatility=bf.FinitePrior(
                br.Prior(omega_mean=base.omega_mean, omega_variance=1.0)
            ),
            mean_center=float(np.mean(x)),
            mean_variance=float((2 * np.std(x, ddof=1))**2),
            persistence_center=0.82,
        )
        for model, columns in [("M4", 4), ("M5", 11)]:
            result = sample_chain(
                x, prior, model, np.random.default_rng(44), warmup=8, draws=12
            )
            values = result.pop("draws")
            validate_draws(values, result["names"], model)
            self.assertEqual(values.shape, (12, columns))
            self.assertTrue(np.isfinite(values).all())
            positions = {name: result["names"].index(name) for name in result["names"]}
            np.testing.assert_allclose(
                values[:, positions["a"]],
                (1 - values[:, positions["b"]]) * values[:, positions["long_run_mean"]],
                rtol=1e-12,
                atol=1e-12,
            )
            if model == "M5":
                self.assertTrue(np.all((values[:, positions["phi"]] > -1)
                                       & (values[:, positions["phi"]] < 1)))


if __name__ == "__main__":
    unittest.main()
