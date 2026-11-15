from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.audit_sv_variance_bound import (
    plot_prior_predictive,
    truncated_exponential_quantile,
    truncated_exponential_tail,
)


class TestTruncatedExponential(unittest.TestCase):
    def test_quantiles_and_tails_match_cdf(self):
        for upper in (2.0, 4.0, 8.0):
            for probability in (0.01, 0.5, 0.95, 0.99):
                value = truncated_exponential_quantile(probability, upper)
                cdf = -np.expm1(-value) / -np.expm1(-upper)
                self.assertTrue(np.isclose(cdf, probability))
            for cut in (0.5, 1.0, min(3.0, upper - 0.1)):
                expected = (np.exp(-cut) - np.exp(-upper)) / (1 - np.exp(-upper))
                self.assertTrue(np.isclose(truncated_exponential_tail(cut, upper), expected))

    def test_tail_respects_support(self):
        self.assertEqual(truncated_exponential_tail(4.0, 4.0), 0.0)
        self.assertEqual(truncated_exponential_tail(5.0, 4.0), 0.0)
        self.assertEqual(truncated_exponential_tail(0.0, 4.0), 1.0)

    def test_plot_supports_configured_bounds(self):
        rng = np.random.default_rng(42)
        draws = {
            upper: {"yield_changes": {
                horizon: rng.normal(size=(100, 8))
                for horizon in (1, 4)
            }}
            for upper in (3.0, 5.0)
        }
        training = rng.normal(size=(20, 8))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "movements.png"
            plot_prior_predictive(draws, training, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
