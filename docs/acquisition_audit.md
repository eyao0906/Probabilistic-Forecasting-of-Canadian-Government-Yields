# Public acquisition audit — 27 October 2026

## Outcome and preservation blocker

**Acquisition checks completed; the overall preservation gate failed.** The independent downloads, pinned offline verification, source-code regressions, and numerical comparisons ran. However, invoking the existing full test suite regenerated seven pre-existing M3 **test-fixture** metadata/diagnostic files under `results/state_space_revised`. This violates the requested preservation scope, even though production results, raw data, and all original source caches remain unchanged. The previous fixture bytes were not backed up and are not tracked by Git; their original hashes are retained, but exact restoration could not be verified. Do not describe this audit as fully preservation-compliant.

Audit evidence is under:

`reproduction/acquisition_audit_20261027T181110Z/` (abbreviated `AUDIT/` below).

`AUDIT/audit_summary.json`, `AUDIT/preservation_check.json`, and the finalizer's exit **1** explicitly retain the failure. Of 190 preflight-hashed files, **183 remain byte-identical**, none are missing, and seven changed:

- `results/state_space_revised/independent_test_fixture/fit_summary.csv`
- `results/state_space_revised/independent_test_fixture/metadata.json`
- `results/state_space_revised/independent_test_fixture/optimizer_diagnostics.json`
- `results/state_space_revised/independent_test_fixture/verification.json`
- `results/state_space_revised/test_fixture/fit_summary.csv`
- `results/state_space_revised/test_fixture/metadata.json`
- `results/state_space_revised/test_fixture/optimizer_diagnostics.json`

Current changed bytes are preserved in `AUDIT/changed_test_fixtures/`; original and final digests are in `preserved_before.json` and `preserved_after.json`. No historical source pins were changed. **Parent integration:** isolate the two writing tests in `tests/test_state_space_revised.py` before another canonical-root full-suite run, and restore the pre-audit fixture bytes only if an independent authentic backup exists. Do not regenerate replacements and claim they are the originals.

## Pinned offline replay

`AUDIT/offline_pinned_replay_final.json` records **31 catalog entries**:

| Status | Count | Meaning |
|---|---:|---|
| `cached_verified` | 24 | Pinned cached payload passed hash/content checks |
| `documented_access_failure_cached` | 6 | Preserved failed-access evidence, **not full-text access** |
| `error` | 1 | Optional missing `lit_delft_record` |

The CLI returned **0** because required entries passed. This is not a claim that all 31 full texts are available. No network transport was constructed in offline mode. Independently rebuilding the catalog returned exactly the committed catalog (`AUDIT/rebuilt_catalog.json`); existing pins were retained rather than recomputed to accept changed vintages.

## Independent live transactions

All new requests used the empty `AUDIT/live/` root, never the canonical cache. Source URLs and the Bank of Canada query parameters came from the committed catalog. Robots policies were checked and preserved; TLS verification remained enabled; no authentication, browser challenge bypass, or certificate bypass was used.

Across the two valid live runs, **6 source attempts** produced **5 verified downloads and 1 error**, with **6 source transaction JSON files and 6 robots transaction/body pairs**. Downloaded bodies were read back, hashed, and revalidated by `AUDIT/finalize_audit.py`.

| Source | Result |
|---|---|
| Bank of Canada, `dFrom=2000-01-01&dTo=2026-10-09` [1] | Downloaded; differs from supplied file hash |
| Federal Reserve `feds200628.csv` [2] | Downloaded; differs from supplied file hash |
| Diebold–Li UCM PDF [3] | Failed certificate verification; not bypassed |
| Diebold–Li Crossref DOI metadata [4] | Downloaded; DOI validated |
| Omori et al. UCI PDF [7] | Downloaded, parsed **26 pages**, extracted UTF-8 text with provenance sidecar |
| Omori et al. Crossref DOI metadata [8] | Downloaded; DOI validated |

The first run returned **1** because the UCM PDF was required and failed. The successful Omori alternative returned **0**; it does not erase or relabel the UCM failure. A preliminary alternative selector used nonexistent GSW catalog IDs, returned **2 before making requests**, and remains recorded in `live_literature_execution.txt`. The valid alternative is in `live_literature_omori_execution.txt` and `live_literature_alternative.json`.

Evidence: `live_transactions.json`, `live_literature_alternative.json`, `live/_transactions/`, `live/_robots/`, `live/_acquisition/`, downloaded bodies, and `live/docs/sources/extracted_utf8/lit_omori_uci.txt{,.json}`. Citation-ledger entries [5]–[6] were candidate URLs only and were **not fetched**.

## Canonical versus live vintages

`src.verify_data_sources` now validates both source formats, reports missing/incomplete date coverage, and normalizes Fed percentage-point units separately from BoC decimal yields. Equality below concerns the named maturity columns, not every field in the full source file.

| Comparison, 2000–2025 | Complete common dates | Compared cells | Differing cells | Complete-panel equality |
|---|---:|---:|---:|---|
| BoC, 8 research maturities | 6,457 | 51,656 | 0 | Yes |
| Fed, `SVENY01`–`SVENY30` | 6,502 | 195,060 | 0 | Yes |

Both comparisons have identical raw-date and complete-date coverage within that research range, and maximum observed yield difference **0 basis points**. Neither full response is byte-identical:

- BoC: supplied **6,974** raw rows, ending **2026-09-23**; live **6,984**, ending **2026-10-07**. Across the full comparison range, 6,639 common complete dates have no numerical differences; live adds ten raw dates, nine complete. Full-range panels are **not equal**.
- Fed: supplied **17,038** raw rows, ending **2026-10-02**; live **17,048**, ending **2026-10-16**. Across the full range, 10,200 common complete dates have no differences in the 30 yield columns; live adds ten raw dates, nine complete. Full-range panels are **not equal**. Other Fed fields, including forward rates and fitted parameters, are structurally validated but not part of this numerical yield comparison.

Exact hashes, date lists, coverage and units are retained in `boc_research_period_comparison.json`, `fed_research_period_comparison.json`, `boc_full_comparison.json`, and `fed_full_comparison.json`. A present retrieval neither proves the original interactive command nor reconstructs historical point-in-time releases.

## Source-code fixes and executed tests

RED/GREEN logs are retained for four defects:

1. CSV acceptance previously relied on header/token checks. The strict parser now rejects HTML decoys, invalid or duplicate dates, malformed row widths and nonfinite/nonnumeric numeric fields while accepting documented missing values.
2. Direct downloads could avoid the batch robots guard. `PublicTransport` now applies shared robots policy, request spacing, reviewed HTTPS URLs and redirect refusal to direct and batch acquisition, preserving policy evidence.
3. A lowercase `content-length` mismatch was accepted, and partial failures lacked durable transaction evidence. Incomplete bodies now fail closed with received bytes and a per-attempt JSON record preserved.
4. Catalog rebuilding could silently repin modified historical bodies. Existing committed pins are checked and retained; changed pinned bodies cause failure.

Comparator regression tests additionally establish explicit missing-date/incomplete-row reporting and correct Fed preamble/unit handling. The full-suite first run exposed an obsolete `urlopen` mock in the existing custom-download test; it was adapted to the new opener boundary and now checks the exact robots-plus-source requests without real network access.

Final execution: **103 tests passed in 10.409 seconds**, exit **0**, recorded in `AUDIT/tests_full_green.txt`. The earlier full-suite failure remains in `tests_full_final.txt`; earlier RED outputs remain intact. Passing tests do not override the preservation failure caused by existing M3 fixture-writing tests.

Changed repository code: `src/acquisition.py`, `src/source_catalog.py`, `src/verify_data_sources.py`; relevant tests: `tests/test_acquisition.py`, `tests/test_source_catalog.py`, `tests/test_source_comparison.py`, `tests/test_prepare_safety.py`. `src/prepare_data.py`, `src/reproduce.py`, canonical source pins, paper, README, and other documentation were not edited by this acquisition task. The unrelated runner is parent-owned.

Every child/test invocation after the environment correction explicitly exported `TMPDIR=C:/Users/ethan/AppData/Local/hermes/cache/scratch`; explicit-environment behavior in `tests/support.py` was not changed. Earlier baseline/CSV executions preceded that correction and are not retrospectively claimed to have used the corrected environment.

## Reproduction code and provenance limits

`AUDIT/executed_code/` contains an **18-file** snapshot and `executed_code_manifest.json` records its hashes, including acquisition/comparison/catalog code, compatibility entry points, source pins, dependency lock, relevant tests and three archived original downloader/source-check scripts. Original archives remain in place and were not edited.

`AUDIT/replay_commands.json` gives runnable **reconstructed equivalents**, explicitly labeled as such, using **new** output paths. These and the current deterministic acquisition code make cache verification and new live collection reproducible. They are not a recovered transcript of unavailable original historical interactive commands. Current transaction timestamps and logs are genuine observed audit evidence, not fabricated historical provenance. Use new directories for each replay; the finalizer creates evidence exclusively and intentionally cannot overwrite an existing evidence set.

## Public source references

Ledger: `AUDIT/citation_ledger.json`, accessed 2026-10-27. Failed and unattempted sources are identified above; URL presence alone does not imply successful access.

[1] https://www.bankofcanada.ca/stats/results/csv?lookupPage=lookup_yield_curve.php&dFrom=2000-01-01&dTo=2026-10-09

[2] https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv

[3] https://www.ucm.es/data/cont/media/www/pag-41460/Diebold.pdf

[4] https://api.crossref.org/works/10.1016%2Fj.jeconom.2005.03.005

[7] https://www.economics.uci.edu/files/docs/colloqpapers/w06/chib.pdf

[8] https://api.crossref.org/works/10.1016%2Fj.jeconom.2006.07.008
