# Full reproduction runbook

This is a computational replay procedure, **not a declaration of scientific completion**. M0–M2 are the accepted retrospective benchmarks. M3 is rejected; the paired M4/M5 Bayesian comparison is not accepted. See `docs/specification_audit.md` and the actual pilot metadata.

## 1. Create a new locked Python environment

The verified installation is Python 3.14.7 on Windows 11, with the exact distributions in `requirements-lock.txt`. Other operating systems/interpreters have not been certified. From the extracted project root, using bash on Windows:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m pip check
```

Quarto and a working TeX installation are separate prerequisites, not Python packages. The local audit used Quarto 1.10.18 and TinyTeX 2026. Check your installations with `quarto --version` and `quarto check`. Installing pinned Python packages does not establish a working PDF toolchain.

## 2. Inventory all project code without executing legacy experiments

```bash
.venv/Scripts/python.exe -B -m src.reproduce --inventory-only --report logs/code_integrity.json
```

This hashes and syntax-checks Python, including archived `.py.txt` snapshots, and hashes other recognized project code. It excludes virtual environments, `.git`, bytecode and the top-level `reproduction/` staging/evidence area to avoid counting duplicate copies. Acquisition-audit scripts inside that evidence area are additionally preserved and hashed by the release manifest. Syntax checks alone are not execution or scientific validation.

## 3. Execute the complete experiment in a NEW workspace

Choose a path that does not already exist. Do not run a full suite against historical result directories merely to check the code.

```bash
.venv/Scripts/python.exe -B -m src.reproduce \
  --workspace reproduction/full_replay_NEW \
  --python .venv/Scripts/python.exe \
  --report logs/full_replay_NEW.json
```

The runner copies code, configuration, the canonical raw inputs and source evidence, but **not** existing model outputs, processed data, generated tables, figures, or the manuscript PDF. Its ordered stages are:

1. Pinned acquisition-cache verification and PDF-text extraction.
2. Data preparation.
3. Accepted M0–M2 forecasts.
4. Availability-lag sensitivity.
5. Baseline-only model confidence sets.
6. Default corrected M4/M5 pilot (4 chains, 1,000 warmup, 1,500 retained draws).
7. Extended corrected pilot (4 chains, 3,000 warmup, 6,000 retained draws).
8. Revised M3 estimation and stability checks.
9. Independent M3 numerical verifier.
10. Baseline tables/figures.
11. Supplementary tables/figures.
12. Quarto PDF rendering.
13. PDF/front-matter/citation/asset checks.
14. The complete unit/regression suite in the copied workspace.

Each step is an actual subprocess with a saved command, time limit, return code, elapsed time and separate log. The first failed step stops the run; its failure cannot become a passing replay. Read `<workspace>/reproduction_report.json` and `<workspace>/logs/`. A successful process is still not a successful scientific experiment.

The final comparison uses exact prepared-CSV bytes and semantic NPZ-array comparisons, with predeclared `rtol=1e-9`, `atol=1e-10`, for the accepted baselines, availability sensitivity, rejected M3 outputs, and both corrected pilot chains. Numeric arrays must be nonempty and finite; keys/shapes/dates must agree. Original raw/source/result files and the original manuscript PDF are hashed before and after. Timing, PDF container bytes, optimizer timing and metadata timestamps are not expected to match. This is a same-platform, locked-library numerical replay, not a guarantee of cross-platform bitwise equality.

## 4. Reacquire public sources independently

The offline replay above reconstructs the experiment from the supplied canonical vintage; it is not a fresh network download. The full downloader, validator, robots checks, parser/extractor, source catalog and transaction logger are in:

- `src/acquisition.py`
- `src/source_catalog.py`
- `src/verify_data_sources.py`
- `src/source_data_check.py` (guarded historical entry point)
- `fetch_raw.py` and `prep_data.py` (preserved, guarded historical entry points)
- `src/prepare_data.py`

`config/sources.json` pins caches, URLs, extraction metadata and preserved access failures. Do not rebuild it over existing pins, bypass robots/TLS failures, or overwrite canonical raw files with current downloads.

For a NEW destination, using the exact catalog IDs:

```bash
.venv/Scripts/python.exe -B -m src.acquisition --live \
  --root reproduction/live_NEW \
  --only boc_original fed_original lit_diebold_li_ucm metadata_diebold2006forecasting \
  --extract --report logs/live_NEW.json

.venv/Scripts/python.exe -B -m src.acquisition --live \
  --root reproduction/live_literature_NEW \
  --only lit_omori_uci metadata_omori2007stochastic \
  --extract --report logs/live_literature_NEW.json
```

Independent audit evidence is under `reproduction/acquisition_audit_20261027T181110Z/`, with reconstructed commands in `replay_commands.json`, saved live transactions/bodies, source comparisons and executed-code snapshots. The first acquisition command returned **1** in the real audit because the UCM PDF failed TLS verification; no bypass was attempted. The Omori/metadata acquisition returned **0**. Failure records are evidence, not downloaded full text.

The BoC/Fed full current files differ from the supplied vintage by additional raw dates; the audited 2000–2025 common-date yield comparisons match. That corroborates overlapping values, not the time, authentication, or exact commands of the original historical download. Some original interactive acquisition commands were not retained. Runnable equivalents are reconstructed and labelled as such; byte-for-byte historical command recovery is not claimed.

## 5. Preservation exception and legacy code

The acquisition audit's full-suite invocation exposed two M3 tests writing into canonical fixture directories. Seven historical fixture metadata/diagnostic files were regenerated before isolation was corrected. Their old hashes, but not recoverable old bytes, are retained in the acquisition audit. Its final preservation check correctly fails. Raw/source caches and nonfixture model results were unchanged. Later before/after preservation checks establish only preservation from their own documented baseline; they cannot repair this earlier exception.

Archived/legacy six-model experiments are retained for audit, not silently relabelled as corrected results. Their CLI gates and source-preservation regressions can be tested safely; executing their full invalid experiments would not validate their inference. Do not remove the guards simply to obtain a zero exit code.

## 6. Current paper

Use the current `README.md` for the accepted finite-moment M4/M5 experiment and minimal PDF-render commands. The release-package builder named in earlier notes was not implemented and is not required. Historical acquisition/isolation evidence and superseded progress notes are externally archived as described in `MRP_PROGRESS.md`. Scientific acceptance, supervisor approval and authenticated historical data releases are distinct requirements; only the first is established for the main experiment here.
