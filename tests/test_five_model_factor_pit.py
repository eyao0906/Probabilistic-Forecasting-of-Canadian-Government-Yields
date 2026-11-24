import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plot_five_model_factor_pit import (
    FACTOR_FORECASTS,
    FACTOR_METRICS,
    GARCH_FORECASTS,
    GARCH_METRICS,
    INTERNAL_TO_MANUSCRIPT,
    MODELS,
    build_pit_figure,
    load_pit_records,
    write_pit_figure,
)


class FiveModelFactorPitTests(unittest.TestCase):
    def test_exact_model_label_mapping(self):
        self.assertEqual(INTERNAL_TO_MANUSCRIPT, {
            "M0": "M0",
            "M2": "M1",
            "M3": "M2",
            "M4": "M3",
            "AR1_GARCH11": "AR-GARCH",
        })

    def test_complete_unique_aligned_pit_records_and_metric_histograms(self):
        records = load_pit_records()
        self.assertEqual(len(records), 3900)
        self.assertEqual(set(records["model"]), set(MODELS))
        self.assertEqual(records["origin"].nunique(), 130)
        self.assertFalse(records.duplicated(
            ["model", "factor", "horizon", "origin"]).any())
        self.assertTrue(np.isfinite(records["pit"]).all())
        self.assertTrue(records["pit"].between(0, 1).all())
        counts = records.groupby(["model", "factor", "horizon"]).size()
        self.assertEqual(len(counts), 30)
        self.assertTrue((counts == 130).all())
        aligned = records.groupby(["factor", "horizon", "origin"]).agg(
            target_count=("target", "nunique"),
            observed_range=("observed", lambda values: values.max() - values.min()),
            model_count=("model", "nunique"),
        )
        self.assertTrue((aligned["target_count"] == 1).all())
        self.assertTrue((aligned["model_count"] == 5).all())
        self.assertLessEqual(float(aligned["observed_range"].max()), 1e-10)

    def test_all_panels_have_five_model_traces_and_uniform_reference(self):
        records = load_pit_records()
        figure, axes = build_pit_figure(records)
        try:
            self.assertEqual(axes.shape, (3, 2))
            for ax in axes.flat:
                self.assertEqual(len(ax.patches), 5)
                self.assertEqual({patch.get_label() for patch in ax.patches}, set(MODELS))
                self.assertEqual(len(ax.lines), 1)
                self.assertEqual(ax.lines[0].get_ydata()[0], 1.0)
        finally:
            plt.close(figure)

    def test_generation_is_nonempty_deterministic_and_does_not_change_inputs(self):
        input_paths = (FACTOR_FORECASTS, FACTOR_METRICS, GARCH_FORECASTS, GARCH_METRICS)
        before = {path: path.read_bytes() for path in input_paths}
        records = load_pit_records()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = write_pit_figure(records, root / "first.png")
            second = write_pit_figure(records, root / "second.png")
            self.assertGreater(first.stat().st_size, 0)
            self.assertGreater(first.with_suffix(".pdf").stat().st_size, 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(before, {path: path.read_bytes() for path in input_paths})


if __name__ == "__main__":
    unittest.main()
