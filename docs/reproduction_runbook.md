# Reproducing the current paper

The accepted finite-moment M4/M5 comparison is complete: 130 matched origins, 520 model-horizon records and no failed origins. Accepted checkpoints and analysis outputs are in `results/bayesian_finite_20261027T194239Z/rolling_main/`; the accepted prior-bound sensitivity runs are in the sibling `bound_sensitivity/` directory. M0-M2 are accepted external benchmarks. M3 remains excluded because its likelihood fits did not pass the existing stability criterion.

Use the current `README.md` as the authoritative entry point. The commands below render and verify the manuscript and reconstruct comparisons from saved results; they do not refit the Bayesian models.

## Render and verify

From the repository root, with the existing Python environment and Quarto/LaTeX installed:

```powershell
quarto render paper\MRP_Draft.qmd --to pdf
.venv\Scripts\python.exe -B -c "from src.verify_paper import verify; r=verify(write_outputs=False, previews=False); print(r); assert r['passed']"
```

## Reconstruct the accepted Bayesian comparisons

```powershell
.venv\Scripts\python.exe -B -m src.bayesian_finite_analysis --output results\bayesian_finite_20261027T194239Z\rolling_main
```

The source analysis files are `analysis\paired_summary.csv`, `per_maturity.csv` and `summary.json`. The paper's saved tables and figures are render inputs. The command reconstructs numerical comparisons from accepted archives; it does not rerun MCMC.

Do not run the legacy `src.reproduce` full replay, older all-purpose artifact builders, or the root full test suite as part of a routine paper build. Those workflows include historical stages and tests that have written to canonical fixtures. Do not refit accepted stages or change frozen gates just to render or verify the paper. See `README.md` for current project scope and the exact minimal reproduction commands.

## Provenance and interpretation

The supplied raw data vintage and acquisition evidence are preserved separately. A fresh download is not required for the saved-results reproduction above, and must not overwrite pinned source files. The paper's forecasts are retrospective observation-time forecasts, not release-time backtests. Numerical acceptance does not authenticate the original retrieval transaction, remove vintage-revision uncertainty, or establish supervisor approval or institutional sign-off.
