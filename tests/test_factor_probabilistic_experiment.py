import math
import unittest

import numpy as np

from src.bayesian_stage_a import FACTORS
from src.factor_probabilistic_experiment import (
    _baseline_forecast,
    _paired_tests,
    gaussian_crps,
)


class FactorProbabilisticExperimentTests(unittest.TestCase):
    def test_gaussian_crps_at_its_center(self):
        expected = 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
        self.assertAlmostEqual(gaussian_crps(0.0, 0.0, 1.0), expected)
        self.assertAlmostEqual(
            gaussian_crps(2.0, 1.0, 4.0),
            gaussian_crps(0.0, 1.0, 4.0),
        )

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


if __name__ == "__main__":
    unittest.main()
