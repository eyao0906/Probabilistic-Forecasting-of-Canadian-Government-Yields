"""Focused tests for the isolated M3 optimizer/covariance recovery work."""
import unittest
import csv
import tempfile
from pathlib import Path

import numpy as np

from src import state_space_revised as m3
from src.m3_recovery import _write_csv


class M3RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(814)
        cls.load = np.array([
            [1., .8, .2], [1., .3, .5], [.4, -.2, .8], [.2, -.4, .9],
        ])
        factors = rng.normal(size=(48, 3))
        cls.y = factors @ cls.load.T + rng.normal(scale=.04, size=(48, 4))
        cls.fixed_r = np.array([.002, .003, .004, .005])
        cls.theta9 = np.r_[[1., -.2, .1], np.arctanh([.7, .4, -.3]),
                           np.log([.03, .02, .04])]
        cls.theta13 = np.r_[cls.theta9, np.log(cls.fixed_r)]

    def test_parameter_dimensions_and_persistence_bounds(self):
        self.assertEqual(m3.parameter_count(8, 'estimated'), 17)
        self.assertEqual(m3.parameter_count(8, 'regularized'), 17)
        self.assertEqual(m3.parameter_count(8, 'fixed'), 9)
        self.assertEqual(m3.parameter_bounds(8).shape, (17, 2))
        self.assertEqual(m3.parameter_bounds(8, 4., 'estimated')[3, 1], 4.)
        self.assertEqual(m3.parameter_bounds(8, 3., 'fixed').shape, (9, 2))

    def test_fixed_covariance_filter_gradient_and_forecast(self):
        value, gradient = m3.likelihood_gradient(
            self.y, self.load, self.theta9, 'fixed', self.fixed_r)
        self.assertTrue(np.isfinite(value))
        numerical = np.empty(9)
        for j in range(9):
            step = 1e-5 * max(1., abs(self.theta9[j]))
            direction = np.eye(9)[j] * step
            plus = m3.filter_state(self.y, self.load, self.theta9+direction,
                                   'fixed', self.fixed_r)[0]
            minus = m3.filter_state(self.y, self.load, self.theta9-direction,
                                    'fixed', self.fixed_r)[0]
            numerical[j] = (plus-minus)/(2*step)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-6)
        _, state, covariance = m3.filter_state(
            self.y, self.load, self.theta9, 'fixed', self.fixed_r)
        _, forecast_covariance = m3.forecast_moments(
            state, covariance, self.load, self.theta9, 1, 'fixed', self.fixed_r)
        self.assertTrue(np.isfinite(forecast_covariance).all())
        np.testing.assert_allclose(
            np.diag(forecast_covariance - self.load @ (
                np.diag(np.tanh(self.theta9[3:6])) @ covariance
                @ np.diag(np.tanh(self.theta9[3:6])).T
                + np.diag(np.exp(self.theta9[6:9]))) @ self.load.T),
            self.fixed_r, rtol=1e-8, atol=1e-10)

    def test_regularized_raw_likelihood_and_penalty_are_separate(self):
        target = np.log(np.array([.003, .004, .005, .006]))
        strength = 1.7
        raw, gradient = m3.likelihood_gradient(self.y, self.load, self.theta13)
        penalty = .5*strength*np.sum((self.theta13[9:]-target)**2)
        objective_gradient = gradient.copy()
        objective_gradient[9:] += strength*(self.theta13[9:]-target)
        self.assertAlmostEqual(raw, m3.filter_state(self.y, self.load, self.theta13)[0])
        numerical = np.empty(13)
        def objective(theta):
            return (m3.filter_state(self.y, self.load, theta)[0]
                    + .5*strength*np.sum((theta[9:]-target)**2))
        for j in range(13):
            step = 1e-5 * max(1., abs(self.theta13[j]))
            direction = np.eye(13)[j] * step
            numerical[j] = (objective(self.theta13+direction)
                            - objective(self.theta13-direction))/(2*step)
        np.testing.assert_allclose(objective_gradient, numerical, rtol=2e-6, atol=2e-6)
        self.assertGreaterEqual(penalty, 0.)

    def test_training_covariance_calibration_uses_only_supplied_rows(self):
        first = self.y[:30]
        calibrated = m3.calibrate_measurement_covariance(first, self.load)
        changed_future = self.y.copy()
        changed_future[30:] += 1000.
        same = m3.calibrate_measurement_covariance(changed_future[:30], self.load)
        np.testing.assert_array_equal(calibrated, same)

    def test_comparison_csv_accepts_union_of_source_columns(self):
        rows = [
            {"variant": "historical", "dss": -1.},
            {"variant": "recovery", "dss": -2., "restart_count": 8},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"comparison.csv"
            _write_csv(path, rows)
            with path.open(newline="", encoding="utf-8") as stream:
                saved = list(csv.DictReader(stream))
        self.assertEqual(saved[0]["restart_count"], "")
        self.assertEqual(saved[1]["restart_count"], "8")

    def test_stationary_initial_covariance_is_stable_at_wider_bound(self):
        theta = self.theta13.copy()
        theta[3:6] = [4., -4., 4.]
        nll, state, covariance = m3.filter_state(self.y, self.load, theta)
        self.assertTrue(np.isfinite(nll))
        self.assertTrue(np.isfinite(state).all())
        self.assertTrue(np.isfinite(covariance).all())
        expected = np.exp(theta[6:9])*np.cosh(theta[3:6])**2
        self.assertTrue(np.isfinite(expected).all())

    def test_restart_reproducibility_and_original_evidence_preservation(self):
        fit = m3.fit_mle(
            self.y, self.load, n_starts=3, maxiter=500, restart_count=2,
            restart_seed=4201)
        repeated = m3.fit_mle(
            self.y, self.load, n_starts=3, maxiter=500, restart_count=2,
            restart_seed=4201)
        self.assertEqual(len(fit['original_diagnostics']), 3)
        self.assertEqual(len(fit['restart_diagnostics']), 2)
        self.assertEqual(len(fit['diagnostics']), 5)
        self.assertEqual(fit['params'].tolist(), repeated['params'].tolist())
        self.assertEqual(
            [d['restart_seed'] for d in fit['restart_diagnostics']],
            [d['restart_seed'] for d in repeated['restart_diagnostics']])
        self.assertAlmostEqual(
            fit['nll'], min(d['nll'] for d in fit['diagnostics'] if d['numerically_valid']))
        original_spread = np.ptp([d['nll'] for d in fit['original_diagnostics']])
        self.assertAlmostEqual(fit['original_start_nll_spread'], original_spread)
        self.assertEqual(fit['accepted'], fit['original_acceptance'])


if __name__ == '__main__':
    unittest.main()
