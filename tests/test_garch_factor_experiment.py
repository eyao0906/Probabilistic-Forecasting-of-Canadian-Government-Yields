import csv
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from arch import arch_model

from src.bayesian_stage_a import FACTORS, make_design
from src.garch_factor_experiment import (
    ACCEPTED_OUTPUT,
    BASE_SEED,
    FittedGarch,
    LAMBDA,
    MANUSCRIPT_TO_INTERNAL,
    SCENARIOS,
    _accepted_metrics,
    _aggregate,
    _score_forecast,
    _sample_crps,
    _verify_design,
    _seed,
    expected_origins,
    fit_ar1_garch,
    predictive_moments,
    simulate_ar1_garch,
)
from src.plot_factor_histories import factor_history_frame


ROOT = Path(__file__).resolve().parents[1]


class GarchFactorExperimentTests(unittest.TestCase):
    def test_fixed_design_origins_and_factor_history(self):
        design = make_design()
        origins = expected_origins(design)
        self.assertEqual(len(origins), 130)
        self.assertEqual(design["dates"][origins[0]], date(2016, 1, 8))
        self.assertEqual(design["dates"][origins[-1]], date(2025, 11, 28))
        self.assertEqual(design["dates"][origins[-1] + 4], date(2025, 12, 24))
        frame = factor_history_frame(design)
        self.assertEqual(frame.shape, (1356, 4))
        self.assertEqual(frame.iloc[0]["Date"], "2000-01-07")
        self.assertEqual(frame.iloc[-1]["Date"], "2025-12-24")
        expected = design["factors"][1:1357]
        np.testing.assert_array_equal(frame.iloc[:, 1:].to_numpy(), expected)
        self.assertEqual(design["lambda"], LAMBDA)
        accepted = _verify_design(design, origins)
        self.assertEqual(len(accepted), 3120)

    def test_fit_uses_basis_points_and_returns_percentage_point_units(self):
        rng = np.random.default_rng(42)
        x = np.empty(600)
        x[0] = 0.3
        residual = rng.normal(0, 0.02, len(x))
        for t in range(1, len(x)):
            x[t] = 0.02 + 0.8 * x[t - 1] + residual[t]
        fit = fit_ar1_garch(x, factor="level", origin="2015-12-31")
        independent = arch_model(
            x * 100, mean="AR", lags=1, vol="GARCH", p=1, o=0, q=1,
            dist="normal", rescale=False,
        ).fit(disp="off", show_warning=True)
        self.assertTrue(fit.record["converged"], fit.record["optimizer_message"])
        self.assertAlmostEqual(fit.intercept, independent.params["Const"] / 100)
        self.assertAlmostEqual(fit.ar, independent.params["y[1]"])
        self.assertAlmostEqual(fit.omega, independent.params["omega"] / 10000)
        self.assertAlmostEqual(fit.alpha, independent.params["alpha[1]"])
        self.assertAlmostEqual(fit.beta, independent.params["beta[1]"])
        self.assertAlmostEqual(
            fit.last_variance,
            independent.conditional_volatility[-1] ** 2 / 10000)
        analytic_forecasts = independent.forecast(horizon=4, reindex=False)
        for horizon in (1, 4):
            mean, variance = predictive_moments(fit, float(x[-1]), horizon)
            self.assertAlmostEqual(
                mean, float(analytic_forecasts.mean.iloc[-1, horizon - 1]) / 100)
            self.assertAlmostEqual(
                variance, float(analytic_forecasts.variance.iloc[-1, horizon - 1]) / 10000)

    def test_recursive_forecast_moments_and_simulation(self):
        fit = FittedGarch(
            {"origin_value": 1.0}, intercept=0.1, ar=0.7,
            omega=0.03, alpha=0.12, beta=0.75,
            last_residual=-0.2, last_variance=0.16,
        )
        mean1, variance1 = predictive_moments(fit, 1.0, 1)
        self.assertAlmostEqual(mean1, 0.8)
        self.assertAlmostEqual(variance1, .03 + .12 * .04 + .75 * .16)
        mean4, variance4 = predictive_moments(fit, 1.0, 4)
        self.assertAlmostEqual(mean4, 0.1 * (1 + .7 + .7**2 + .7**3) + .7**4)
        rng_a = np.random.default_rng(129)
        rng_b = np.random.default_rng(129)
        draws_a = simulate_ar1_garch(fit, 1.0, 4, rng_a, 32768)
        draws_b = simulate_ar1_garch(fit, 1.0, 4, rng_b, 32768)
        np.testing.assert_array_equal(draws_a, draws_b)
        self.assertEqual(draws_a.shape, (32768,))
        self.assertAlmostEqual(draws_a.mean(), mean4, delta=0.012)
        self.assertAlmostEqual(draws_a.var(), variance4, delta=0.012)
        self.assertTrue(np.isfinite(draws_a).all())

    def test_crps_u_statistic_matches_direct_pairwise_formula(self):
        samples = np.array([-1.0, 0.0, 1.0])
        observed = 0.25
        direct = np.mean(np.abs(samples - observed))
        pairwise = np.mean(np.abs(samples[:, None] - samples[None, :]))
        self.assertAlmostEqual(
            _sample_crps(observed, samples),
            direct - .5 * pairwise * len(samples) / (len(samples) - 1))

    def test_gaussian_and_simulated_score_shapes_are_finite(self):
        gaussian = _score_forecast(0.1, 0.25, 0.3, None)
        self.assertGreaterEqual(gaussian["crps"], 0)
        self.assertAlmostEqual(gaussian["pit"], 0.6554217416103242)
        self.assertEqual(gaussian["crps_mcse"], 0)
        samples = np.random.default_rng(77).normal(size=SCENARIOS)
        simulated = _score_forecast(0.0, 1.0, 0.2, samples)
        self.assertTrue(np.isfinite([
            simulated["crps"], simulated["crps_mcse"], simulated["pit"],
            simulated["pit_mcse"], simulated["predictive_mean_mcse"],
            simulated["predictive_variance_mcse"],
        ]).all())
        self.assertGreaterEqual(simulated["crps_mcse"], 0)
        self.assertLessEqual(simulated["intervals"][.9][0], simulated["intervals"][.9][1])

    def test_aggregate_metrics_match_forecast_records(self):
        records = []
        for factor in FACTORS:
            for horizon in (1, 4):
                for i in range(130):
                    records.append({
                        "factor": factor, "horizon": horizon, "origin": f"{i:03d}",
                        "crps": .1 + i / 1000, "crps_mcse": .002, "dss": .3,
                        "rmse_component": .04, "mae": .2, "pit": .5,
                        "pit_mcse": .001, "predictive_mean_mcse": .003,
                        "predictive_variance_mcse": .004,
                        **{f"covered{level}": 1 for level in ("50", "80", "90")},
                        **{f"width{level}": float(level) / 100
                           for level in ("50", "80", "90")},
                    })
        metrics = _aggregate(records)
        self.assertEqual(len(metrics), 6)
        row = next(item for item in metrics
                   if item["factor"] == "slope" and item["horizon"] == 4)
        self.assertEqual(row["n_origins"], 130)
        self.assertAlmostEqual(row["mean_crps"], np.mean([r["crps"] for r in records[:130]]))
        self.assertAlmostEqual(row["rmse"], .2)
        self.assertEqual(row["coverage90"], 1.0)
        self.assertAlmostEqual(row["mean_width90"], .9)

    def test_seed_derivation_is_repeatable_and_separates_forecasts(self):
        seed = _seed(836, 0, 4)
        self.assertEqual(seed, _seed(836, 0, 4))
        self.assertNotEqual(seed, _seed(836, 0, 1))
        self.assertNotEqual(seed, _seed(836, 1, 4))
        self.assertEqual(BASE_SEED, 20261119)
        self.assertEqual(SCENARIOS, 8192)

    def test_accepted_results_are_reported_with_manuscript_labels(self):
        rows = [{"factor": factor, "horizon": horizon, "mean_crps": .5,
                 "mean_dss": 1., "rmse": .2, "coverage90": .9, "mean_width90": 1.}
                for factor in FACTORS for horizon in (1, 4)]
        comparisons = _accepted_metrics(rows)
        self.assertEqual(len(comparisons), 6)
        self.assertEqual(MANUSCRIPT_TO_INTERNAL, {"M0": "M0", "M1": "M2", "M2": "M3", "M3": "M4"})
        with (ACCEPTED_OUTPUT / "factor_metrics.csv").open(newline="", encoding="utf-8") as stream:
            internal = list(csv.DictReader(stream))
        expected_m1 = float(next(r for r in internal if (
            r["model"], r["factor"], int(r["horizon"])) == ("M2", "level", 1))["mean_crps"])
        self.assertAlmostEqual(comparisons[0]["M1_mean_crps"], expected_m1)
        self.assertEqual(comparisons[0]["AR1_GARCH11_mean_crps"], .5)


if __name__ == "__main__":
    unittest.main()
