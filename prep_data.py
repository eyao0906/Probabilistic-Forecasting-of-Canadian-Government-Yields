"""Build analysis-ready weekly yield-curve files for the MRP.

Inputs (data/raw/):
  - boc_daily_2000_2026.csv : Bank of Canada daily zero-coupon curve (decimals)
  - feds200628.csv          : GSW US nominal curve (percent, cont. compounded)

Outputs:
  - canada_weekly.csv : weekly (Friday), 8 maturities, percent, 2000-2026
  - us_weekly.csv     : weekly (Friday), 8 maturities, percent, 2000-2026
"""
if __name__ == "__main__":
    import sys as _legacy_sys
    from pathlib import Path as _LegacyPath
    _legacy_sys.path.insert(0, str(_LegacyPath(__file__).resolve().parent / "src"))
    from legacy_guard import require_legacy_opt_in as _require_legacy_opt_in
    _require_legacy_opt_in()

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data\\raw")
TAU = [0.25, 0.5, 1, 2, 5, 7, 10, 30]


def prep_canada():
    fp = os.path.join(RAW, "boc_daily_2000_2026.csv")
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    colmap = {0.25: "ZC025YR", 0.5: "ZC050YR", 1: "ZC100YR", 2: "ZC200YR",
              5: "ZC500YR", 7: "ZC700YR", 10: "ZC1000YR", 30: "ZC3000YR"}
    cols = [colmap[t] for t in TAU]
    sub = df[cols].replace(r"^\s*na\s*$", np.nan, regex=True).astype(float) * 100.0
    sub["Date"] = pd.to_datetime(df["Date"])
    sub = sub.dropna().reset_index(drop=True)
    print(f"Canada daily: {len(sub)} rows, {sub['Date'].min().date()} -> {sub['Date'].max().date()}, "
          f"dropped {(df[cols].replace(r'^\\s*na\\s*$', np.nan, regex=True).isna().any(axis=1)).sum()} na rows")
    wk = friday_sample(sub)
    wk = wk.rename(columns={c: f"y{t}" for t, c in zip(TAU, cols)})
    wk.to_csv(os.path.join(HERE, "canada_weekly.csv"), index=False)
    print(f"Canada weekly: {len(wk)} rows")
    return wk


def svensson_yield(m, b0, b1, b2, b3, t1, t2):
    x1 = m / t1
    x2 = m / t2
    return (b0 + b1 * (1 - np.exp(-x1)) / x1
            + b2 * ((1 - np.exp(-x1)) / x1 - np.exp(-x1))
            + b3 * ((1 - np.exp(-x2)) / x2 - np.exp(-x2)))


def prep_us():
    fp = os.path.join(RAW, "feds200628.csv")
    df = pd.read_csv(fp, skiprows=9)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= "2000-01-01")].reset_index(drop=True)
    # zero-coupon yields 1y..30y are given; compute 0.25y/0.5y from Svensson params
    par = df[["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]].astype(float)
    ok = par.notna().all(axis=1)
    print(f"US rows 2000+: {len(df)}, rows with full Svensson params: {ok.sum()}")
    df = df[ok].reset_index(drop=True)
    par = par[ok].reset_index(drop=True)
    out = pd.DataFrame({"Date": df["Date"]})
    out["y0.25"] = svensson_yield(0.25, par["BETA0"], par["BETA1"], par["BETA2"],
                                 par["BETA3"], par["TAU1"], par["TAU2"])
    out["y0.5"] = svensson_yield(0.5, par["BETA0"], par["BETA1"], par["BETA2"],
                                par["BETA3"], par["TAU1"], par["TAU2"])
    for t in [1, 2, 5, 7, 10, 30]:
        out[f"y{t}"] = df[f"SVENY{t:02d}"].astype(float).values
    out = out.dropna().reset_index(drop=True)
    print(f"US daily clean: {len(out)} rows, {out['Date'].min().date()} -> {out['Date'].max().date()}")
    wk = friday_sample(out)
    wk.to_csv(os.path.join(HERE, "us_weekly.csv"), index=False)
    print(f"US weekly: {len(wk)} rows")
    return wk


def friday_sample(dfm):
    d = dfm.copy()
    d["wk"] = d["Date"].dt.isocalendar().year.astype(int) * 100 + \
        d["Date"].dt.isocalendar().week.astype(int)
    return d.groupby("wk").tail(1).sort_values("Date").drop(columns="wk").reset_index(drop=True)


def main():
    ca = prep_canada()
    us = prep_us()
    cols = [f"y{t}" for t in TAU]
    print("\nCanada weekly summary (percent):")
    print(ca[cols].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
    print("\nUS weekly summary (percent):")
    print(us[cols].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
    # overlap check
    common = pd.merge(ca[["Date"]], us[["Date"]], on="Date")
    print(f"\ncommon weekly dates: {len(common)}")


if __name__ == "__main__":
    main()
