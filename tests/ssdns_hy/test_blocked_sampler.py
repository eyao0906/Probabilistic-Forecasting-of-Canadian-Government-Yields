from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.ssdns_hy import (
    calibrate_training_priors,
    dns_loadings,
    posterior_predictive_scenarios,
    predictive_moments,
    run_chain,
    summarize_chain_diagnostics,
)
from src.ssdns_hy.synthetic_pilot import _load_chain, _save_chain


def _synthetic_panel(seed=211, n_time=48):
    rng = np.random.default_rng(seed)
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0, 30.0])
    loadings = dns_loadings(maturities, 0.52)
    mean = np.array([3.0, -1.0, 0.2])
    persistence = np.array([0.82, 0.72, 0.55])
    variance = np.array([0.04, 0.05, 0.03])
    state = mean.copy()
    factors = np.empty((n_time, 3))
    for t in range(n_time):
        state = mean + persistence * (state - mean) + np.sqrt(variance) * (
            rng.standard_normal(3)
        )
        factors[t] = state
    yields = factors @ loadings.T + rng.normal(0.0, 0.06, (n_time, 8))
    return yields, maturities


class BlockedSamplerTests(unittest.TestCase):
    def test_decay_proposal_scale_must_be_positive_and_finite(self):
        yields, maturities = _synthetic_panel()
        priors, _ = calibrate_training_priors(yields, maturities)
        for step in (0., -1., np.nan):
            with self.subTest(step=step), self.assertRaises(ValueError):
                run_chain(yields, maturities, priors, "SS_M3", np.random.default_rng(1), 0, 4, decay_step=step)

    def test_checkpoint_round_trip_and_integrity(self):
        yields, maturities = _synthetic_panel()
        priors, _ = calibrate_training_priors(yields, maturities)
        chain = run_chain(yields, maturities, priors, "SS_M3", np.random.default_rng(128), 2, 4)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "chain.npz"
            _save_chain(path, chain, "design-hash", 2.5)
            restored, elapsed = _load_chain(path, "design-hash")
            np.testing.assert_array_equal(restored.decay_per_year, chain.decay_per_year)
            self.assertEqual(restored.acceptance, chain.acceptance)
            self.assertEqual(elapsed, 2.5)
            with self.assertRaises(ValueError):
                _load_chain(path, "different-design")
            with path.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "checksum"):
                _load_chain(path, "design-hash")

    def test_constant_volatility_predictive_distribution_matches_gaussian_moments(self):
        yields, maturities = _synthetic_panel()
        priors, _ = calibrate_training_priors(yields, maturities)
        posterior = run_chain(yields, maturities, priors, "SS_M3", np.random.default_rng(128), 2, 4)
        factor_mean = np.array([3., -1., .2])
        persistence = np.array([.8, .6, .3])
        variance = np.array([.04, .05, .03])
        terminal = np.array([3.2, -.8, .1])
        measurement = np.full(8, .0036)
        posterior = replace(
            posterior,
            factor_mean=np.tile(factor_mean, (4, 1)),
            persistence=np.tile(persistence, (4, 1)),
            log_variance=np.tile(np.log(variance), (4, 1)),
            terminal_factor=np.tile(terminal, (4, 1)),
            measurement_variance=np.tile(measurement, (4, 1)),
            decay_per_year=np.full(4, .52),
        )
        horizon = 4
        samples = posterior_predictive_scenarios(
            posterior, maturities, horizon, np.random.default_rng(88), 30000
        )
        loadings = dns_loadings(maturities, .52)
        expected_mean = loadings @ (factor_mean + persistence**horizon * (terminal - factor_mean))
        factor_variance = variance * sum(persistence**(2 * k) for k in range(horizon))
        expected_covariance = loadings @ np.diag(factor_variance) @ loadings.T + np.diag(measurement)
        np.testing.assert_allclose(samples.mean(axis=0), expected_mean, atol=.008)
        np.testing.assert_allclose(np.cov(samples, rowvar=False), expected_covariance, atol=.004)

    def test_both_models_produce_finite_joint_posterior_predictive_draws(self):
        yields, maturities = _synthetic_panel()
        priors, _ = calibrate_training_priors(yields, maturities)
        for model in ("SS_M3", "SS_M4"):
            with self.subTest(model=model):
                chains = [
                    run_chain(
                        yields,
                        maturities,
                        priors,
                        model,
                        np.random.default_rng(seed),
                        warmup=5,
                        draws=12,
                    )
                    for seed in (128, 129)
                ]
                posterior = chains[0]
                diagnostics = summarize_chain_diagnostics(chains)
                self.assertEqual(posterior.factor_mean.shape, (12, 3))
                self.assertTrue(np.isfinite(posterior.factor_mean).all())
                self.assertEqual(diagnostics["model"], model)
                self.assertIn("maximum_rank_rhat", diagnostics)
                self.assertTrue(np.all(np.abs(posterior.persistence) < 1.0))
                self.assertTrue(
                    np.all(
                        (posterior.decay_per_year > priors.decay_lower)
                        & (posterior.decay_per_year < priors.decay_upper)
                    )
                )
                self.assertTrue(
                    all(0.0 <= rate <= 1.0 for rate in posterior.acceptance.values())
                )
                scenarios = posterior_predictive_scenarios(
                    posterior,
                    maturities,
                    horizon_observations=4,
                    rng=np.random.default_rng(827),
                    n_scenarios=48,
                )
                mean, covariance = predictive_moments(scenarios)
                self.assertEqual(scenarios.shape, (48, 8))
                self.assertTrue(np.isfinite(mean).all())
                self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.0)


if __name__ == "__main__":
    unittest.main()
