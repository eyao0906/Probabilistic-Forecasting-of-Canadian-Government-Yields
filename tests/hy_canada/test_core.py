import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from src.hy_canada.data import MATURITIES, load_panel
from src.hy_canada.kalman import filter_likelihood, loadings
from src.hy_canada.states import simulate_forecast


class CoreTests(unittest.TestCase):
    def test_loadings_and_panel(self):
        h = loadings(np.array([0.0, 1.0, 30.0]), 0.7)
        np.testing.assert_allclose(h[0], [1, 1, 0], atol=1e-14)
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

    def test_full_covariance_forecast_moments(self):
        rng = np.random.default_rng(41)
        h = loadings(np.array([1.0, 10.0]), 0.7)
        state, mu, rho = np.array([1., 0.2, -0.1]), np.zeros(3), np.full(3, 0.8)
        q = np.array([[.2, .08, .02], [.08, .2, -.03], [.02, -.03, .2]])
        r = np.full(2, 0.1)
        forecasts = np.array([simulate_forecast(state, mu, rho, q, r, h, 1, rng)[0]
                              for _ in range(12000)])
        np.testing.assert_allclose(forecasts.mean(axis=0), h @ (rho * state), atol=0.025)
        np.testing.assert_allclose(np.cov(forecasts.T), h @ q @ h.T + np.diag(r), atol=0.025)


if __name__ == "__main__":
    unittest.main()
