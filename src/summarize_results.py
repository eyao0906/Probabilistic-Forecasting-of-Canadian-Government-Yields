"""Quick summary of experiment results for review."""
from __future__ import annotations

if __name__ == "__main__":
    import sys as _legacy_sys
    from pathlib import Path as _LegacyPath
    _legacy_sys.path.insert(0, str(_LegacyPath(__file__).resolve().parent))
    from legacy_guard import require_legacy_opt_in as _require_legacy_opt_in
    _require_legacy_opt_in()

from pathlib import Path
import json

import pandas as pd

ROOT = Path(".")
FULL = ROOT / "results" / "full"

def main() -> None:
    if not (FULL / "run_metadata.json").exists():
        print("Results not ready.")
        return
    meta = json.loads((FULL / "run_metadata.json").read_text())
    print("METADATA")
    for k in ["scope", "out_of_sample_origins", "forecast_records", "mcmc_warmup",
              "mcmc_draws_per_factor", "state_space_fit_success_count", "state_space_fit_count"]:
        print(f"  {k}: {meta.get(k)}")
    print("\nSUMMARY")
    print(pd.read_csv(FULL / "model_summary.csv").to_string(index=False))
    print("\nFIT DIAGNOSTICS")
    print(pd.read_csv(FULL / "fit_diagnostics.csv").to_string(index=False))
    print("\nDM TESTS")
    print(json.dumps(json.loads((FULL / "dm_tests.json").read_text()), indent=2))

if __name__ == "__main__":
    main()
