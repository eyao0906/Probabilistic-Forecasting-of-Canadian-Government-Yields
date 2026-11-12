# MQF MRP Guidelines Compliance Matrix

**Source:** `MQF MRP Guidelines 2026.pdf`  
**Project:** Probabilistic Forecasting of Canadian Government Yields  
**Status:** Requirements mapped; implementation pending manuscript draft.

## Requirement → Implementation Mapping

| # | Guideline Requirement | Implementation | Status | Verification |
|---|---|---|---|---|
| 1 | **Degree milestone:** Credit/no-credit; no oral defense | Paper submitted as PDF to supervisor and second reader | Planned | Deadline Dec 4, 2026 |
| 2 | **Title page** with title, author, supervisor, month/year, program statement | Quarto title-page template with LaTeX formatting | Pending | Visual inspection of rendered PDF |
| 3 | **Abstract** ≤ 250 words, stating problem, methods, principal findings | Abstract section in `paper/MRP_Draft.qmd` with word count enforced | Pending | Word count check before render |
| 4 | **Acknowledgements** section | Included in front matter | Pending | TOC entry |
| 5 | **Table of Contents** | Quarto `toc: true` | Pending | Rendered PDF contains TOC |
| 6 | **List of Tables** (if necessary) | Quarto `list-of-tables: true` if tables present | Pending | Rendered PDF |
| 7 | **List of Illustrations** (if necessary) | Quarto `list-of-figures: true` if figures present | Pending | Rendered PDF |
| 8 | **Main text organized into chapters** | Chapters 1–5+ in Quarto manuscript | Pending | Structure review |
| 9 | **General introduction** accessible to non-experts | Chapter 1: Introduction with motivation and overview | Pending | Readability review |
| 10 | **Specific methods chapters** | Chapters 2–3: Literature review and methodology | Pending | Content review |
| 11 | **Results with discussion** | Chapter 4: Empirical results and discussion | Pending | Numerical verification |
| 12 | **Conclusions** summarizing findings and future research | Chapter 5: Conclusions | Pending | Content review |
| 13 | **Detailed bibliography** | `references.bib` processed by Quarto CSL | Pending | Citation check |
| 14 | **Length:** 25–35 pages | Target 30 pages; will adjust during drafting | Pending | PDF page count after render |
| 15 | **Font size:** 12 pt main text | Quarto PDF default 12 pt | Pending | LaTeX log / visual inspection |
| 16 | **Footnotes numbered consecutively** | Quarto footnote syntax `[^n]` | Pending | PDF review |
| 17 | **Equations numbered consecutively** | Quarto equation numbering `(1)`, `(2)`, … | Pending | PDF review |
| 18 | **Full derivations in appendix** if abbreviated in main text | Appendix A: Derivations | Pending | Content review |
| 19 | **In-text citations:** Author-year, e.g., Scott (1992) or (Scott, 1992) | Quarto CSL with author-year style | Pending | Citation check |
| 20 | **Reference format:** Full details at end matching sample styles | `references.bib` with complete fields | Pending | BibTeX validation |
| 21 | **Academic integrity:** No unacknowledged use of others' work; AI use disclosed to supervisor | All sources cited; AI-assisted drafting noted in acknowledgements or audit | Pending | Audit document |
| 22 | **Sample title page format** followed | Custom LaTeX title page matching sample | Pending | Visual inspection |

## Conflicts and Resolutions

| Conflict | Resolution |
|---|---|
| Proposal does not specify page limit; guidelines require 25–35 pages | Follow guidelines; target 30 pages. |
| Proposal uses equation labels like (1), (2); guidelines require consecutive numbering throughout | Implemented via Quarto `eqn-numbering: true` or manual labels. |
| Proposal does not mention appendices; guidelines require them for abbreviated derivations | Add appendices for DNS algebra, MCMC details, and additional diagnostics. |

## Unresolved / Pending Items

- Exact CSL style file to use for author-year citations (to be confirmed with supervisor).
- Whether to include List of Illustrations if few figures are generated.
- Final page count depends on empirical results and discussion length.

---

*This matrix will be updated as the manuscript is drafted and rendered.*
