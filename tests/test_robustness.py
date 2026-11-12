"""Information-set robustness tests; independent of any stochastic-volatility fit."""
import importlib
import unittest
from datetime import date, timedelta


class AvailabilityTests(unittest.TestCase):
    def test_lag_cutoff_selects_last_eligible_observation(self):
        try:
            from src.robustness import available_index
        except ImportError:
            self.fail('Missing availability-cutoff implementation')
        dates = [date(2020, 1, 3) + timedelta(weeks=i) for i in range(8)]
        self.assertEqual(available_index(dates, 5, 14), 3)
        self.assertEqual(available_index(dates, 5, 15), 2)
        self.assertEqual(available_index(dates, 5, 0), 5)
        with self.assertRaises(ValueError):
            available_index(dates, 0, 14)


    def test_lag_forecast_is_invariant_to_unavailable_observations(self):
        import numpy as np
        try:
            from src.robustness import lagged_forecast
        except ImportError:
            self.fail('Missing lagged forecast implementation')
        from src.models import ns_loadings
        dates = [date(2020, 1, 3) + timedelta(weeks=i) for i in range(100)]
        y = np.random.default_rng(91).normal(size=(100, 8))
        load = ns_loadings(np.array([.25, .5, 1, 2, 5, 7, 10, 30]), .5)
        for model in ['M0', 'M1', 'M2']:
            mean, cov, k = lagged_forecast(dates, y, 80, 84, model, load, first=0, lag_days=14)
            changed = y.copy(); changed[k + 1:] += 10000
            m2, c2, k2 = lagged_forecast(dates, changed, 80, 84, model, load, first=0, lag_days=14)
            self.assertEqual(k, 78)
            self.assertEqual(k2, k)
            np.testing.assert_array_equal(mean, m2)
            np.testing.assert_array_equal(cov, c2)
            if model == 'M0':
                np.testing.assert_array_equal(mean, y[k])
                np.testing.assert_allclose(cov, np.cov(y[6:k+1]-y[:k-5], rowvar=False, ddof=1), atol=1e-12)


    def test_runner_writes_isolated_reconstructible_common_origin_outputs(self):
        import tempfile, os, json
        from pathlib import Path
        import numpy as np
        try:
            from src.robustness import run
        except ImportError:
            self.fail('Missing robustness runner')
        from src.scores import dss
        scratch = Path.home() / 'AppData/Local/hermes/cache/scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            out = Path(folder)
            report = run(out=out, max_origins=2, ensemble=64)
            with np.load(out / 'forecasts.npz', allow_pickle=False) as archive:
                means = archive['means'].copy()
                covariances = archive['covariances'].copy()
                targets = archive['targets'].copy()
            self.assertEqual(report['forecast_records'], 12)
            self.assertEqual(len(means), 12)
            import csv
            with (out / 'scores.csv').open() as source:
                scores = list(csv.DictReader(source))
            for i, row in enumerate(scores):
                self.assertLessEqual(date.fromisoformat(row['latest_fit_date']), date.fromisoformat(row['origin_date']) - timedelta(days=14))
                self.assertAlmostEqual(float(row['dss']), dss(targets[i], means[i], covariances[i]), places=10)
            self.assertLessEqual(report['lambda_training_latest'], scores[0]['latest_fit_date'])


if __name__ == '__main__':
    unittest.main()
