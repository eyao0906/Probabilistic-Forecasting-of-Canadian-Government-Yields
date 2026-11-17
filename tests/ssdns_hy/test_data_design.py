from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

from src.ssdns_hy.data import (
    MATURITIES_YEARS,
    audit_input_hashes,
    load_weekly_panel,
    select_forecast_origins,
)


class DataDesignTests(unittest.TestCase):
    def test_preserved_panel_hash_and_forecast_calendar(self):
        root = Path(__file__).resolve().parents[2]
        hashes = audit_input_hashes(root)
        self.assertTrue(hashes["processed"]["matches_manifest"])

        dates, yields = load_weekly_panel(root / hashes["processed"]["path"])
        self.assertEqual(yields.shape[1], len(MATURITIES_YEARS))
        training_count = sum(date(2000, 1, 1) <= day <= date(2015, 12, 31) for day in dates)
        self.assertEqual(training_count, 835)
        origins = select_forecast_origins(
            dates,
            date(2016, 1, 1),
            date(2025, 12, 31),
            (1, 4),
            stride_observations=4,
        )
        self.assertEqual(len(origins), 260)
        origin_dates = {dates[origin.origin_index].isoformat() for origin in origins}
        self.assertEqual(
            {"2016-01-08", "2020-03-27", "2025-11-28"},
            origin_dates & {"2016-01-08", "2020-03-27", "2025-11-28"},
        )
        self.assertEqual(
            {origin.horizon_observations for origin in origins}, {1, 4}
        )


if __name__ == "__main__":
    unittest.main()
