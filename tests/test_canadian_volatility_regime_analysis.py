import unittest

import numpy as np

from src.canadian_volatility_regime_analysis import (
    FACTORS,
    FACTOR_COLUMNS,
    TRAIN_END,
    _read_accepted_forecasts,
    _validate_unconditional_metrics,
    bootstrap_comparisons,
    construct_regimes,
    load_factor_history,
    rolling_standardized_rms,
)


class CanadianVolatilityRegimeTests(unittest.TestCase):
    def test_fixed_history_matches_panel_and_excludes_1999_observation(self):
        history, checks = load_factor_history()
        self.assertEqual(len(history), 1356)
        self.assertEqual(history["Date"].iloc[0].date().isoformat(), "2000-01-07")
        self.assertTrue(checks["excluded_1999_observation_not_used"])
        self.assertLess(checks["max_factor_reconstruction_abs_difference"], 1e-11)
        self.assertEqual(tuple(history.loc[:, FACTORS].columns), FACTORS)
        self.assertEqual(FACTOR_COLUMNS, ("Level", "Slope", "Curvature"))

    def test_rolling_rms_uses_exactly_complete_prior_changes_and_no_future_values(self):
        rng = np.random.default_rng(20261120)
        values = np.cumsum(rng.normal(size=(90, 3)), axis=0)
        scales = np.std(np.diff(values[:60], axis=0), axis=0, ddof=1)
        factor_rms, composite = rolling_standardized_rms(values, scales, lookback=26)
        self.assertTrue(np.isnan(factor_rms[:26]).all())
        expected = np.sqrt(np.mean((np.diff(values, axis=0)[:26] / scales) ** 2, axis=0))
        np.testing.assert_allclose(factor_rms[26], expected)
        origin = 55
        changed_future = values.copy()
        changed_future[origin + 1:] += 1000
        future_rms, future_composite = rolling_standardized_rms(
            changed_future, scales, lookback=26)
        np.testing.assert_array_equal(future_rms[origin], factor_rms[origin])
        self.assertEqual(future_composite[origin], composite[origin])

    def test_training_thresholds_remain_frozen_when_future_factors_change(self):
        history, _ = load_factor_history()
        train_end = history.index[history["Date"].dt.date.astype(str) == TRAIN_END][0] + 1
        original, original_thresholds = construct_regimes(history, train_end)
        changed = history.copy()
        origin = train_end + 80
        changed.loc[origin + 1:, list(FACTORS)] += 100
        altered, altered_thresholds = construct_regimes(changed, train_end)
        self.assertEqual(original_thresholds["q25"], altered_thresholds["q25"])
        self.assertEqual(original_thresholds["q75"], altered_thresholds["q75"])
        self.assertEqual(original.at[origin, "regime"], altered.at[origin, "regime"])
        self.assertEqual(original.at[origin, "composite_volatility"],
                         altered.at[origin, "composite_volatility"])

    def test_accepted_forecasts_and_unconditional_metrics_reconcile(self):
        history, _ = load_factor_history()
        forecasts = _read_accepted_forecasts(history)
        self.assertEqual(len(forecasts), 3900)
        self.assertEqual(forecasts["manuscript_model"].nunique(), 5)
        self.assertEqual(forecasts["origin"].nunique(), 130)
        checks = _validate_unconditional_metrics(forecasts)
        self.assertTrue(checks["unconditional_metrics_reproduced"])
        self.assertEqual(checks["unconditional_groups_checked"], 30)

    def test_bootstrap_keeps_full_sequence_and_reports_valid_intervals(self):
        history, _ = load_factor_history()
        train_end = history.index[history["Date"].dt.date.astype(str) == TRAIN_END][0] + 1
        regimes, _ = construct_regimes(history, train_end)
        forecasts = _read_accepted_forecasts(history)
        regime_rows = regimes.set_index(regimes["Date"].dt.strftime("%Y-%m-%d"))
        origins = sorted(forecasts["origin"].unique())
        origin_inventory = regime_rows.loc[origins]
        labels = origin_inventory["regime"].to_dict()
        self.assertEqual(len(origin_inventory), 130)
        self.assertEqual(set(origin_inventory["regime"]), {"LOW", "MEDIUM", "HIGH"})
        forecasts["regime"] = forecasts["origin"].map(labels)
        forecasts["origin_row_index"] = forecasts["origin"].map(
            origin_inventory["research_row_index"].to_dict())
        self.assertTrue((forecasts.groupby("origin")["regime"].nunique() == 1).all())
        paired, interactions, meta = bootstrap_comparisons(
            forecasts, reps=20, seed=123, expected_block_length=4)
        self.assertEqual(meta["replicates"], 20)
        self.assertEqual(len(paired), 3 * 3 * 2 * 4)
        self.assertEqual(len(interactions), 3 * 3 * 2 + 4 * 3 * 2 + 4 * 3 * 2 * 3)
        self.assertTrue((paired["n_origins"] > 0).all())
        self.assertTrue(paired["bootstrap_valid_replicates"].le(20).all())
        self.assertTrue(interactions["bootstrap_ci95_low"].notna().all())


if __name__ == "__main__":
    unittest.main()
