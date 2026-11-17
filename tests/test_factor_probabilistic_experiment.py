import math
import unittest

import numpy as np
from scipy.special import ndtr

from src.bayesian_stage_a import FACTORS
from src.bayesian_finite_validation import factor_scenarios
from src.factor_probabilistic_experiment import (
    _baseline_forecast,
    _calibration_diagnostics,
    _paired_tests,
    gaussian_mixture_pit,
    gaussian_crps,
    stochastic_volatility_pit,
)


class FactorProbabilisticExperimentTests(unittest.TestCase):
    def test_gaussian_crps_at_its_center(self):
        expected = 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
        self.assertAlmostEqual(gaussian_crps(0.0, 0.0, 1.0), expected)
        self.assertAlmostEqual(
            gaussian_crps(2.0, 1.0, 4.0),
            gaussian_crps(0.0, 1.0, 4.0),
        )

    def test_constant_volatility_pit_is_the_posterior_gaussian_mixture(self):
        mean = np.array([-0.5, 0.1, 0.8]).reshape(3, 1)
        variance = np.array([0.2, 1.0, 3.0]).reshape(3, 1)
        observed = 0.25
        expected = np.mean(ndtr((observed - mean) / np.sqrt(variance)))
        self.assertAlmostEqual(
            gaussian_mixture_pit(mean, variance, observed), expected)

    def test_moment_matched_gaussian_is_not_a_gaussian_mixture_cdf(self):
        mean = np.zeros(2)
        variance = np.array([0.01, 9.0])
        observed = 1.0
        mixture = gaussian_mixture_pit(mean, variance, observed)
        moment_matched = ndtr(observed / np.sqrt(np.mean(variance)))
        self.assertGreater(abs(mixture - moment_matched), 0.04)

    def test_stochastic_volatility_pit_integrates_future_paths(self):
        names = ["a", "b", "m", "phi", "v", "h_last"]
        values = np.array([
            [0.05, 0.85, -1.0, 0.8, 0.5, -1.2],
            [-0.03, 0.95, -0.5, 0.97, 0.2, -0.8],
            [0.01, 0.70, -1.5, 0.9, 0.8, -1.0],
        ])
        factor_value, observed = 0.2, 0.35
        for horizon in (1, 4):
            pit, mcse = stochastic_volatility_pit(
                values, names, factor_value, observed, horizon,
                np.random.default_rng(814 + horizon), size=32768, batches=8)
            repeated, repeated_mcse = stochastic_volatility_pit(
                values, names, factor_value, observed, horizon,
                np.random.default_rng(814 + horizon), size=32768, batches=8)
            empirical = np.mean(factor_scenarios(
                values, names, factor_value, horizon, "M5",
                np.random.default_rng(1290 + horizon), 200000) <= observed)
            self.assertEqual(pit, repeated)
            self.assertEqual(mcse, repeated_mcse)
            self.assertGreaterEqual(pit, 0.0)
            self.assertLessEqual(pit, 1.0)
            empirical_mcse = math.sqrt(empirical * (1 - empirical) / 200000)
            self.assertLessEqual(
                abs(pit - empirical),
                4 * math.hypot(mcse, empirical_mcse))

    def test_random_walk_uses_training_difference_variance(self):
        training = np.array([0.0, 1.0, 3.0, 2.0, 6.0])
        mean, variance = _baseline_forecast(training, "M0", 4)
        self.assertEqual(mean, training[-1])
        self.assertAlmostEqual(variance, 4 * np.var(np.diff(training), ddof=1))

    def test_ar1_does_not_clip_unrestricted_ols_persistence(self):
        training = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
        design = np.column_stack((np.ones(len(training) - 1), training[:-1]))
        intercept, persistence = np.linalg.lstsq(
            design, training[1:], rcond=None)[0]
        self.assertGreater(persistence, 1.0)
        residuals = training[1:] - design @ np.array([intercept, persistence])
        innovation_variance = np.mean(residuals**2)
        mean, variance = _baseline_forecast(training, "M2", 4)
        expected_mean = persistence**4 * training[-1] + intercept * sum(
            persistence**k for k in range(4))
        expected_variance = innovation_variance * sum(
            persistence**(2 * k) for k in range(4))
        self.assertAlmostEqual(mean, expected_mean)
        self.assertAlmostEqual(variance, expected_variance)

    def test_paired_tests_use_six_holm_adjusted_comparisons(self):
        records = []
        for origin_index in range(130):
            origin = f"{origin_index:03d}"
            for factor_index, factor in enumerate(FACTORS):
                for horizon in (1, 4):
                    common = 0.1 + origin_index * 0.0001 + factor_index * 0.01
                    for model, offset in (("M3", 0.01), ("M4", -0.01)):
                        records.append({
                            "model": model,
                            "factor": factor,
                            "horizon": horizon,
                            "origin": origin,
                            "crps": common + offset + horizon * 0.001,
                            "crps_mcse": 0.001,
                        })
        result = _paired_tests(records)
        tests = result["tests"]
        self.assertEqual(len(tests), 6)
        self.assertTrue(all(test["p_value_holm"] >= test["p_value_two_sided"]
                            for test in tests))
        self.assertTrue(all(test["mean_crps_difference"] < 0 for test in tests))
        self.assertEqual(result["bootstrap"]["repetitions"], 9999)
        self.assertEqual(
            {int(length) for length in result["bootstrap_robustness"]["results_by_test"][0]["blocks"]},
            {4, 8, 12})
        self.assertTrue(all(
            test["paired_mean_mcse_independence"] >= 0
            for test in tests))

    def test_calibration_diagnostics_are_grouped_by_model_factor_horizon(self):
        records = []
        for model in ("M0", "M2", "M3", "M4"):
            for factor in FACTORS:
                for horizon in (1, 4):
                    for origin in range(130):
                        records.append({
                            "model": model,
                            "factor": factor,
                            "horizon": horizon,
                            "origin": f"{origin:03d}",
                            "observed": float(origin),
                            "mean": 0.0,
                            "variance": 1.0,
                            "pit": origin / 129,
                            "pit_mcse": 0.0,
                            "covered50": int(origin % 2 == 0),
                            "covered80": 1,
                            "covered90": 1,
                            "width50": 1.0,
                            "width80": 2.0,
                            "width90": 3.0,
                        })
        result = _calibration_diagnostics(records)
        self.assertEqual(len(result), 24)
        level_h1 = next(r for r in result if (
            r["model"], r["factor"], r["horizon"]) == ("M0", "level", 1))
        self.assertAlmostEqual(level_h1["standardized_error_mean"], 64.5)
        self.assertAlmostEqual(level_h1["pit_mean"], 0.5)
        self.assertAlmostEqual(level_h1["pit_lag1"], 1.0)
        self.assertAlmostEqual(level_h1["coverage50"], 0.5)


if __name__ == "__main__":
    unittest.main()
