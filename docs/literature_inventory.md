# Literature and source inventory

Access evidence: `docs/sources/literature_verified/retrieval_manifest.json` (2026-10-27 UTC). Each successful PDF was independently signature- and SHA-256-checked. Seven complete PDFs were retrieved; **retrieved does not mean completely read**. Omori's 2005 author manuscript was reviewed in full. The remaining retrieved works were reviewed selectively (abstracts, introductions and the specific sections below). Complete review of all important papers is unfinished. Crossref journal metadata are separately cached in `docs/sources/`; the root `references.bib` has 13 entries, including data/methodology and the proposal-supplied thesis.

| Citation key | Verified DOI / record | Evidence actually reviewed | Relevance and limitation |
|---|---|---|---|
| diebold2006forecasting | 10.1016/j.jeconom.2005.03.005 | Complete prepublication PDF retrieved; abstract/introduction, selected discussion of lambda and section 4 conclusion, PDF pp. 1–3, 13–14, 25–26 | AR forecasts of level/slope/curvature; conclusion distinguishes one-month and one-year performance. Not a claim of short-horizon DNS dominance. |
| diebold2006macroeconomy | 10.1016/j.jeconom.2005.01.011 | Complete PDF retrieved; abstract/introduction and state-space discussion, PDF pp. 1–5 | Kalman/MLE latent-state formulation; diagonal observation covariance but potentially full factor covariance. This project's diagonal-Q restriction is narrower. |
| hautsch2012bayesian | 10.1016/j.csda.2010.07.003 | January 2010 SFB 649 working paper retrieved, not final journal version; abstract/introduction, PDF pp. 1–4 | Monthly U.S. zero-coupon yields; persistent factor volatility and reduced uncertainty in low-volatility periods. Working-paper evidence must not be described as a complete review of final 2012 article. Correct coauthor: Fuyu Yang. |
| caldeira2010bayesian | 10.12660/bre.v30n12010.3502 | Complete publisher PDF retrieved; abstract and introduction, PDF pp. 1–3 | Common/factor-specific SV, leverage, DI futures and U.S. Treasuries. Abstract reports better in-sample fit, not the Canadian out-of-sample claim. |
| gneiting2007strictly | 10.1198/016214506000001437 | Complete PDF retrieved; abstract/introduction and selected scoring discussion | Proper versus strictly proper forecasts; Energy Score and calibration/sharpness. Source uses reward orientation; manuscript uses negative/loss orientation. Full mathematical review pending. |
| scheuerer2015variogram | 10.1175/MWR-D-14-00269.1 | Crossref metadata; publisher and NOAA full-text requests returned 403 HTML | Variogram scoring reference. No full-text-read claim or borrowed numerical findings. |
| omori2007stochastic | 10.1016/j.jeconom.2006.07.008 | Complete September 2005 author manuscript read, 26 PDF pages; Table 1 p.4, §§2.1–2.4, pp.3–11 | Correct ten-component no-leverage mixture, stationary initialization and approximation correction. Revised implementation uses tested in-chain MH correction rather than source's final importance reweighting. Not a verbatim reproduction of the final 2007 algorithm. |
| dawid1999coherent | 10.1214/aos/1018031101 | Crossref metadata; two Euclid PDF requests returned HTTP-200 HTML challenge pages | Bibliographic scoring foundation only; complete text not reviewed. DSS implementation is independently checked algebraically. |
| patton2011volatility | 10.1016/j.jeconom.2010.03.034 | Complete author-hosted PDF retrieved; abstract/introduction and QLIKE motivation, PDF pp.1–2 | Conditionally unbiased imperfect variance proxies; does not prove robustness of arbitrary multivariate yield scores. |
| sarkisian2024bayesian | https://resolver.tudelft.nl/uuid:8510487e-d59f-4a89-93dd-dc685cd06bb6 | Proposal-supplied thesis; institutional access timed out; details not independently fully verified | Related work acknowledged without attributing an empirical country or invented findings. |
| hansen2011model | 10.3982/ECTA5771 | Crossref metadata supplemented by cached Econometric Society author list/abstract | MCS framework. New restricted MCS uses `arch` 8.0.0, not defective legacy elimination routine. No six-model MCS is accepted. |
| bolder2004canadian | https://publications.gc.ca/site/eng/9.559261/publication.html | Government catalogue and official BoC source-page citation; full report not successfully reviewed | Authors David J. Bolder, Grahame Johnson, Adam Metzler. Yield-compounding convention not independently established; portfolio pricing not asserted exact. |
| bocdata | https://www.bankofcanada.ca/rates/interest-rates/bond-yield-curves/ | Source-page text/HTML, robots, and official December 2015 data slice | Decimal units, quarterly maturities, typical two-week lag; 21 complete comparison dates match supplied file. |

## Full-text URLs and provenance

The manifest records requested/final URLs, UTC retrieval times, bytes, media types, PDF signatures, hashes and extraction status. Examples: Diebold–Li is hosted by Universidad Complutense; Hautsch–Yang by Humboldt's repository; Omori by UCI; Gneiting–Raftery by the author's Washington site; Caldeira by the journal; Patton and Diebold–Rudebusch–Aruoba by author sites. Original files and extracted page-marked text are cached in the same folder. `reviewed_section_evidence.json` preserves selected supporting passages. Failure HTML is retained as HTML, never renamed and counted as PDF.

## Synthesis around the research questions

1. DNS representation makes the yield curve parsimonious, but does not itself give a valid conditional forecast distribution. The short-horizon random-walk result does not contradict Diebold–Li's longer-horizon evidence.
2. SV literature motivates varying uncertainty, not guaranteed Canadian forecasting improvement. In-sample fit and narrower intervals alone do not establish calibration or better predictive scores.
3. Proper scores and calibration address different features. DSS is moment-focused; distributional and maturity-contrast checks complement it. No result from an unconverged chain is accepted merely because a score is favorable.
4. Statistical ranking is conditional on the model universe and information set. Three-model MCS does not complete the controlled Bayesian experiment.

No institutional credentials, paywall bypass or access-control evasion was used. Two research workstreams hit provider token-rate limits; their cached material was recovered locally. Full-paper review gaps remain explicit rather than filled with invented claims.

## Final-method reference metadata

The bibliography additionally includes Politis–Romano (1994), *The Stationary Bootstrap*, DOI 10.1080/01621459.1994.10476870, and Vehtari et al. (2021), *Rank-Normalization, Folding, and Localization*, DOI 10.1214/20-BA1221. Author, journal, year, volume/issue and page metadata were verified from Crossref; the latter was also checked against Project Euclid metadata. This does not claim a new complete-text review. The paper states the actual resampling algorithm and diagnostic thresholds used by the existing accepted implementation.
