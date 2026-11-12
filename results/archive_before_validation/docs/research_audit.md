# Research Audit — Initial Repository Inspection

**Date:** 2026-10-26  
**Location:** `C:/Users/ethan/Downloads/MRP`  
**Status:** Inspection complete; awaiting literature verification and package installation.

## 1. Authoritative Documents

| Document | Path | Status |
|---|---|---|
| `MRP_Proposal.qmd` | `C:/Users/ethan/Downloads/MRP/MRP_Proposal.qmd` | Read fully |
| `MRP_Proposal.pdf` | `C:/Users/ethan/Downloads/MRP/MRP_Proposal.pdf` | Read fully |
| `MQF MRP Guidelines 2026.pdf` | `C:/Users/ethan/Downloads/MRP/MQF MRP Guidelines 2026.pdf` | Read fully |

### Key Proposal Specifications
- **Title:** Probabilistic Forecasting of Canadian Government Yields: A Bayesian Stochastic-Volatility Extension of Dynamic Nelson–Siegel
- **Primary question:** Does incorporating Bayesian stochastic volatility into Dynamic Nelson–Siegel factor dynamics improve out-of-sample probabilistic forecasts of the Canadian government yield curve?
- **Data:** Bank of Canada daily zero-coupon government yield curves, weekly sampled, 8 maturities: 0.25, 0.5, 1, 2, 5, 7, 10, 30 years.
- **Period:** Initial estimation 2000–2015; out-of-sample 2016–2025.
- **Forecast origins:** Every 4th weekly observation in out-of-sample (~130 origins).
- **Horizons:** h = 1 and h = 4 weeks (primary: 4 weeks).
- **Models:** 0 Random walk, 1 PCA+VAR(1), 2 Two-step DNS Gaussian AR(1), 3 DNS Kalman/MLE, 4 Bayesian DNS constant vol, 5 Bayesian DNS stochastic vol (proposed).
- **Evaluation:** DSS (primary), Energy Score, prediction-interval coverage, RMSE; Variogram optional.
- **Sampler:** Univariate per factor, Omori et al. (2007) normal-mixture sampler for latent log volatility, no leverage.

### Key MQF 2026 Guidelines
- Length: 25–35 pages.
- Font: 12 pt main text.
- Structure: Front matter (title page, abstract ≤250 words, acknowledgements, TOC, lists), main text in chapters, back matter (appendices, bibliography).
- Citations: Author-year in text, full references at end.
- Footnotes and equations numbered consecutively.
- Submission: PDF to supervisor and second reader by December 4, 2026.

## 2. Repository State

- **Git repository:** Not initialized. No `.git` directory found.
- **Remote:** No remotes configured.
- **Working directory:** `C:/Users/ethan/Downloads/MRP`
- **Existing files:**
  - `fetch_raw.py` — downloads BoC and Fed raw yield-curve data.
  - `prep_data.py` — builds weekly Canada/US yield files.
  - `data/raw/` — contains `boc_daily_2000_2026.csv` and `feds200628.csv`.
  - `canada_weekly.csv`, `us_weekly.csv` — pre-built weekly files.
  - `MRP_Proposal.qmd`, `MRP_Proposal.pdf`, `MRP_Proposal_Revised.pdf`
  - `MQF MRP Guidelines 2026.pdf`

## 3. Data Validation

### File Inventory
| File | Size | SHA256 |
|---|---|---|
| `data/raw/boc_daily_2000_2026.csv` | 11,406,150 bytes | `ada4b4208b6a4361463181b889383869ba9bf5ad2fbca531a047f954283238a9` |
| `data/raw/feds200628.csv` | 16,653,918 bytes | `2ebdb0a0e94c586445ab9255c7c2d4809f2cdb3776493388453ebb9587ec4342` |
| `canada_weekly.csv` | 146,765 bytes | `a07842947c332df5c86ef195d62f25a292b8d42aa31ad8d7d599f4f0f682c685` |
| `us_weekly.csv` | 171,060 bytes | `bd182b07ff0782a8b590721cacf22ee904401b181d302ef997dfb3b888ba02f4` |

### Weekly Data Summary
- **Canada:** 1,396 rows, 1999-12-31 to 2026-09-23, 8 maturities in percent. Zero NA rows dropped.
- **US:** 1,396 rows, 2000-01-07 to 2026-10-02, 8 maturities in percent. Derived from Svensson parameters for 0.25y/0.5y; 6,693 of 6,980 rows had full Svensson parameters post-2000.
- **Common weekly dates:** 1,357
- **Note:** Canada weekly starts 1999-12-31 (pre-2000), US starts 2000-01-07. The proposal's initial window is 2000–2015, so the extra Canada rows are outside the main analysis but present in the file.

### Data Source Notes
- BoC data: decimals in raw file, converted to percent in `prep_data.py`. Methodology documented at BoC page referencing Bolder, Johnson, and Metzler (2004).
- Fed data: `feds200628.csv` is a staff research product, not an official statistical release. Subject to revision.
- Weekly sampling: Friday-last observation per ISO week in `prep_data.py`.

## 4. Environment Assessment

### Toolchain
- **Quarto:** 1.10.18 (installed at `C:\Users\ethan\AppData\Local\Apps\Quarto\bin\quarto.EXE`)
- **LaTeX:** TinyTeX v2026 installed and verified by `quarto check`.
- **Python:** System Python 3.14 at `C:/Users/ethan/AppData/Local/Python/pythoncore-3.14-64/python.exe` has numpy 2.5.3, pandas 3.0.6, scipy, matplotlib, scikit-learn 1.9.1.
- **R:** 4.6.1 installed at `E:/R-4.6.1` with knitr 1.52, rmarkdown 2.32.
- **Git:** Available, but current folder is not a repository.

### Known Environment Issue
- The project-local `.venv` fails to import pandas due to a Windows Application Control policy blocking `pandas._libs.tslibs.conversion` (`tzconversion`). **Workaround:** Use the system Python directly.

### Background Installation
- `statsmodels`, `pymc`, `arviz` are being installed in the background via system pip. Status pending.

## 5. Project Structure Created

```
C:/Users/ethan/Downloads/MRP/
├── src/
├── tests/
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── figures/
├── paper/
├── docs/
└── logs/
```

## 6. Immediate Blockers and Risks

| Issue | Severity | Mitigation |
|---|---|---|
| No Git repository | Medium | Initialize locally; push only if remote verified and auth available. |
| Windows Application Control blocks `.venv` pandas import | High | Use system Python (`C:/Users/ethan/AppData/Local/Python/pythoncore-3.14-64/python.exe`). |
| `statsmodels`, `pymc`, `arviz` not yet confirmed installed | Medium | Background install in progress; will verify and fall back to custom Gibbs sampler if needed. |
| Proposal references not yet verified | High | Subagents dispatched for bibliography verification and experimental design audit. |
| No `references.bib` or literature inventory | High | Pending subagent results and manual curation. |
| No `paper/MRP_Draft.qmd` | High | Pending literature, data validation, and model results. |

## 7. Next Steps

1. Verify background package installation completes.
2. Build validated `references.bib` from verified sources.
3. Implement reproducible data-acquisition scripts with metadata.
4. Implement Models 0–5 per proposal, starting with benchmarks.
5. Run rolling out-of-sample forecasting experiments.
6. Draft `paper/MRP_Draft.qmd` and render PDF.
7. Write `docs/guidelines_compliance.md`, `docs/reproducibility.md`, `docs/research_audit.md`.
8. Initialize Git and commit milestone checkpoints.

---

*This audit was generated automatically during project setup. It will be updated as stages complete.*
