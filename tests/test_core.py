import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models import (bayes_sv_scenarios, ns_loadings, project_covariance)
from src.prepare_data import MATURITIES, prepare
from src.scores import dss, energy_score, model_confidence_set, var_backtests
from tests.support import scratch_root


class DataPipelineTests(unittest.TestCase):
    def test_weekly_pipeline_selects_last_complete_weekly_row_and_converts_percent(self):
        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            root = Path(temp)
            raw_path = root / "raw.csv"
            output_path = root / "nested" / "weekly.csv"
            headers = ["Date", " ZC025YR", " ZC050YR", " ZC100YR", " ZC200YR",
                       " ZC500YR", " ZC700YR", " ZC1000YR", " ZC3000YR"]
            with raw_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerow(["2020-01-06", *(["0.01"] * 8)])
                writer.writerow(["2020-01-10", *(["0.02"] * 8)])
                writer.writerow(["2020-01-14", *(["0.03"] * 7), "na"])
            rows, manifest = prepare(raw_path, output_path)
            self.assertEqual([r["Date"] for r in rows], ["2020-01-10"])
            self.assertAlmostEqual(rows[0]["y0.25"], 2.0)
            self.assertEqual(manifest["rows_missing_selected_maturities"], 1)
            self.assertTrue(output_path.exists())


class ForecastMathTests(unittest.TestCase):
    def test_nelson_siegel_loadings_have_level_column_of_ones(self):
        loading = ns_loadings(np.array([0.25, 1.0, 30.0]), 0.7)
        self.assertEqual(loading.shape, (3, 3))
        np.testing.assert_allclose(loading[:, 0], 1.0)
        self.assertTrue(np.all(np.isfinite(loading)))

    def test_covariance_projection_is_positive_definite(self):
        cov = project_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]), floor=1e-5)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)

    def test_sv_variance_update_includes_stationary_initial_state(self):
        from src.models import sigma2_posterior_parameters

        h = np.array([0.2, -0.1, 0.4])
        c, phi = 0.1, 0.8
        shape, scale = sigma2_posterior_parameters(h, c, phi, prior_shape=2.0, prior_scale=0.2)
        innovations = h[1:] - c - phi * h[:-1]
        initial_quad = (h[0] - c / (1 - phi)) ** 2 * (1 - phi ** 2)
        self.assertAlmostEqual(shape, 2.0 + len(h) / 2)
        self.assertAlmostEqual(scale, 0.2 + 0.5 * (innovations @ innovations + initial_quad))

    def test_short_sv_pilot_produces_finite_stationary_draws(self):
        rng = np.random.default_rng(120)
        factors = np.cumsum(rng.normal(size=(80, 3)), axis=0)
        load = ns_loadings(np.array([0.25, 0.5, 1, 2, 5, 7, 10, 30]), 0.7)
        scenarios, chains = bayes_sv_scenarios(
            factors, load, np.full(8, 0.01), 1, rng, 12, warmup=5, iterations=12
        )
        self.assertEqual(scenarios.shape, (12, 8))
        self.assertTrue(np.all(np.isfinite(scenarios)))
        self.assertTrue(all(np.all(np.abs(c.beta[:, 1]) < 1) for c in chains))


class ScoreTests(unittest.TestCase):
    def test_dss_matches_univariate_identity_covariance(self):
        self.assertAlmostEqual(dss(np.array([1.0]), np.array([0.0]), np.array([[1.0]])), 1.0)

    def test_energy_score_for_two_point_ensemble(self):
        score = energy_score(np.array([[0.0], [2.0]]), np.array([0.0]))
        self.assertAlmostEqual(score, 0.0)

    def test_mcs_retains_identical_forecasters(self):
        losses = np.ones((20, 3))
        result = model_confidence_set(losses, ["a", "b", "c"], np.random.default_rng(42), reps=25)
        self.assertEqual(result["included"], ["a", "b", "c"])
        self.assertEqual(result["excluded"], [])

    def test_var_backtest_counts_exceptions(self):
        result = var_backtests(np.array([0.1, 0.2, 0.9, 0.1]), np.full(4, 0.5), tail_prob=0.05)
        self.assertEqual(result["exceptions"], 1)
        self.assertEqual(result["transition_counts"]["01"], 1)


class ArchiveTests(unittest.TestCase):
    def test_forecast_archive_is_loadable_without_pickle(self):
        from src.experiment import save_forecast_archive

        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            path = Path(temp) / "draws.npz"
            save_forecast_archive(path, target_dates=np.array([["2020-01-01"]], dtype=object),
                                  values=np.ones((1, 1)))
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["target_dates"].dtype.kind, "U")
                self.assertEqual(archive["target_dates"][0, 0], "2020-01-01")


class CsvOutputTests(unittest.TestCase):
    def test_csv_writer_accepts_rows_with_different_diagnostic_fields(self):
        from src.experiment import write_csv

        with tempfile.TemporaryDirectory(dir=scratch_root()) as temp:
            path = Path(temp) / "diagnostics.csv"
            write_csv(path, [{"origin": "a", "n": 10}, {"origin": "b", "acceptance": 0.4}])
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(set(rows[0]), {"origin", "n", "acceptance"})
            self.assertEqual(rows[1]["acceptance"], "0.4")


class OriginSelectionTests(unittest.TestCase):
    def test_origins_and_targets_stay_inside_out_of_sample_window(self):
        from datetime import date, timedelta
        from src.experiment import select_origins

        dates = [date(2015, 12, 25) + timedelta(weeks=i) for i in range(530)]
        selected = select_origins(dates, date(2016, 1, 1), date(2025, 12, 31), [1, 4], 4)
        self.assertTrue(selected)
        self.assertTrue(all(date(2016, 1, 1) <= dates[t] <= date(2025, 12, 31)
                            for t, _, _ in selected))
        self.assertTrue(all(dates[t + h] <= date(2025, 12, 31) for t, h, _ in selected))
        for horizon in (1, 4):
            origins = [t for t, h, _ in selected if h == horizon]
            self.assertTrue(all(b - a == 4 for a, b in zip(origins, origins[1:])))
        origins_one = {t for t, h, _ in selected if h == 1}
        origins_four = {t for t, h, _ in selected if h == 4}
        self.assertEqual(origins_one, origins_four)


if __name__ == "__main__":
    unittest.main()
