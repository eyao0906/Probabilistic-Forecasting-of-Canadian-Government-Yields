# Data description

The input is the existing `data/processed/canada_weekly.csv`; it was not
downloaded or rebuilt. `data/processed/source_manifest.json` documents Bank of
Canada zero-coupon yields, expressed in **percentage points**, with the last
complete observation in each ISO week. The eight maturities are 0.25, 0.5, 1,
2, 5, 7, 10 and 30 years, in that order. The raw-file hash discrepancy is not
used as an authorization check.

The full panel contains 1,396 observations from 1999-12-31 to 2026-09-23. The
initial estimation window contains 835 observations, 2000-01-07 through
2015-12-31. Its yields range from 0.1012499 to 6.5555065 percentage points.
There is exactly one observation per calendar week in the full panel, with no
missing weeks. The loader checks dates, duplicates, increasing order, column
names and order, numeric finiteness, the requested range and weekly continuity.
Units are established from the existing manifest, not inferred from magnitude.

The requested origin dates 2016-01-08, 2020-03-27 and 2025-11-28 are present.
Their presence alone does not establish a valid real-time forecasting study.
