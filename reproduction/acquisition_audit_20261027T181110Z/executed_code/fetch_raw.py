"""Download the raw yield-curve files used by the MRP.

Usage:
    python fetch_raw.py            # downloads into data/raw/
    python fetch_raw.py --out DIR  # custom output directory

Then run:
    python prep_data.py            # builds canada_weekly.csv / us_weekly.csv

Only the Python standard library is needed for this script.
"""
if __name__ == "__main__":
    import sys as _legacy_sys
    from pathlib import Path as _LegacyPath
    _legacy_sys.path.insert(0, str(_LegacyPath(__file__).resolve().parent / "src"))
    from legacy_guard import require_legacy_opt_in as _require_legacy_opt_in
    _require_legacy_opt_in()

import argparse
import datetime as dt
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

UA = {"User-Agent": "Mozilla/5.0 (MRP research data download; contact: researcher)"}


def download(url, dest):
    print(f"downloading\n  {url}\n  -> {dest}", flush=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    print(f"  saved {total / 1e6:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "raw"))
    ap.add_argument("--dfrom", default="2000-01-01",
                    help="BoC start date YYYY-MM-DD")
    ap.add_argument("--dto", default=dt.date.today().isoformat(),
                    help="BoC end date YYYY-MM-DD (data lag ~2 weeks)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. Bank of Canada daily zero-coupon curve (120 quarterly maturities)
    boc_url = ("https://www.bankofcanada.ca/stats/results/csv"
               f"?lookupPage=lookup_yield_curve.php&dFrom={args.dfrom}&dTo={args.dto}")
    download(boc_url, os.path.join(args.out, "boc_daily_2000_2026.csv"))

    # 2. Fed Gurkaynak-Sack-Wright US nominal curve (Svensson params + yields)
    gsw_url = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
    download(gsw_url, os.path.join(args.out, "feds200628.csv"))

    print("done. Next: python prep_data.py")


if __name__ == "__main__":
    sys.exit(main())
