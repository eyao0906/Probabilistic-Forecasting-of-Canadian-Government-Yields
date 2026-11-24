# Canadian Yield-Factor Volatility-Regime Evaluation

## Executive assessment

This is a retrospective, observation-time regime analysis of the already accepted
130-origin forecasts. The regime at each origin uses only factor observations
available through that origin. It is not a verified real-time vintage exercise,
trading strategy, joint yield-curve forecast, or causal analysis.

The 26-week composite factor-change RMS uses frozen training-only scaling and
training-period quartile thresholds. The training change scales are
level=0.0868922, slope=0.127546, curvature=0.376153.
The frozen thresholds are q25=0.754468 and
q75=1.10309; regime origin counts are
LOW=34, MEDIUM=56,
HIGH=40.

Across the six factor-horizon groups, mean CRPS skill versus classical AR(1) is
+0.019 for Bayesian stochastic-volatility M3 in LOW regimes and
+0.017 in HIGH regimes; corresponding AR-GARCH skills are
+0.025 and +0.015. These are unweighted
descriptive averages of within-group skill, not pooled forecast scores.
Bayesian SV reduces mean CRPS versus constant-volatility M2 by an average of
2.06% in LOW and 1.93% in HIGH
groups. For AR-GARCH versus classical M1, the corresponding relative reductions
are 2.54% and 1.46%. These are
unweighted averages of six within-factor/horizon ratios.

The paired comparisons show a regime pattern, but not a decisive one. Bayesian
SV has lower CRPS than constant-volatility M2 in
4 of 6 LOW and
6 of 6 HIGH factor-horizon
groups. AR-GARCH has lower CRPS than M1 in
4 of 6 LOW and
5 of 6 HIGH groups. For GARCH
minus Bayesian SV, negative favors GARCH: it is negative in
6 of 6 LOW groups,
but only 1 of 6 HIGH groups; SV has lower CRPS in
5 HIGH groups. None of those 12 GARCH-versus-SV subgroup
bootstrap intervals excludes zero. The average within-group relative CRPS
reduction of SV versus GARCH is -0.60% in LOW and
0.26% in HIGH groups. Thus the near tie overall may mask a
descriptive low-regime advantage for GARCH and high-regime advantage for SV,
but subgroup uncertainty does not establish that explanation.

## Regime construction and integrity

- Accepted factor sample: 1,356 weekly observations,
  2000-01-07 through 2025-12-24. The 1999-12-31 row was excluded.
- Training calibration: 835 factor
  observations through 2015-12-31; normalization SD uses
  834 in-sample weekly changes and ddof=1.
- Volatility: RMS of exactly 26 standardized factor changes, then equal
  weighting across level, slope, and curvature. The rolling measure is RMS, not
  a rolling centered standard deviation.
- Quantiles: NumPy `method='linear'`, fitted only on complete training windows.
  Thresholds and labels were frozen before calculating conditional metrics.
- The factor history matches fixed-lambda least-squares extraction from the
  accepted yield panel; maximum absolute discrepancy was
  1.02e-14.
- Regimes are attached by origin date and apply unchanged to all factors,
  horizons, and models. Targets and observed factor values were checked against
  the accepted history.

## Conditional LOW versus HIGH results

Each cell reports LOW and HIGH separately. PIT is mean (sample SD); coverage and
width columns are ordered 50%/80%/90%. CRPS skill is relative to M1 within the
same factor, horizon, and regime; positive means better CRPS than M1. Factor
units are percentage points.

| Factor | h | Model | n Low | CRPS Low | DSS Low | RMSE Low | MAE Low | PIT Low | Coverage Low | Width Low | Skill Low | n High | CRPS High | DSS High | RMSE High | MAE High | PIT High | Coverage High | Width High | Skill High |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| Level | 1 | M0 | 34 | 0.05158 | -3.8284 | 0.09059 | 0.07523 | 0.462 (0.307) | 0.382/0.794/0.912 | 0.1177/0.2236/0.2870 | +0.003 | 40 | 0.05878 | -3.4983 | 0.10389 | 0.08222 | 0.432 (0.299) | 0.525/0.725/0.825 | 0.1251/0.2376/0.3050 | -0.002 |
| Level | 1 | M1 | 34 | 0.05174 | -3.8210 | 0.09092 | 0.07542 | 0.460 (0.307) | 0.382/0.794/0.912 | 0.1175/0.2233/0.2866 | +0.000 | 40 | 0.05867 | -3.5032 | 0.10365 | 0.08206 | 0.433 (0.300) | 0.525/0.725/0.825 | 0.1249/0.2373/0.3045 | +0.000 |
| Level | 1 | M2 | 34 | 0.05211 | -3.8130 | 0.09124 | 0.07567 | 0.455 (0.307) | 0.382/0.794/0.912 | 0.1179/0.2240/0.2876 | -0.007 | 40 | 0.05887 | -3.4926 | 0.10409 | 0.08243 | 0.429 (0.299) | 0.525/0.725/0.825 | 0.1251/0.2377/0.3051 | -0.004 |
| Level | 1 | M3 | 34 | 0.05263 | -3.6821 | 0.09131 | 0.07574 | 0.454 (0.325) | 0.382/0.735/0.853 | 0.1026/0.2013/0.2652 | -0.017 | 40 | 0.05859 | -3.5067 | 0.10360 | 0.08243 | 0.438 (0.273) | 0.525/0.825/0.925 | 0.1486/0.2909/0.3843 | +0.001 |
| Level | 1 | AR1_GARCH11 | 34 | 0.05231 | -3.7282 | 0.09107 | 0.07549 | 0.457 (0.321) | 0.382/0.765/0.853 | 0.1081/0.2053/0.2635 | -0.011 | 40 | 0.05888 | -3.4841 | 0.10355 | 0.08218 | 0.440 (0.260) | 0.575/0.875/0.950 | 0.1635/0.3106/0.3986 | -0.004 |
| Level | 4 | M0 | 34 | 0.10203 | -2.4373 | 0.17892 | 0.14798 | 0.450 (0.300) | 0.471/0.794/0.882 | 0.2354/0.4472/0.5740 | +0.008 | 40 | 0.14570 | -1.4873 | 0.25515 | 0.20059 | 0.528 (0.338) | 0.375/0.650/0.775 | 0.2501/0.4752/0.6099 | -0.003 |
| Level | 4 | M1 | 34 | 0.10282 | -2.4219 | 0.18024 | 0.14926 | 0.446 (0.301) | 0.471/0.794/0.882 | 0.2342/0.4449/0.5711 | +0.000 | 40 | 0.14520 | -1.4883 | 0.25428 | 0.20011 | 0.531 (0.338) | 0.350/0.675/0.775 | 0.2486/0.4723/0.6062 | +0.000 |
| Level | 4 | M2 | 34 | 0.10333 | -2.4089 | 0.18131 | 0.15054 | 0.435 (0.300) | 0.441/0.824/0.882 | 0.2350/0.4464/0.5742 | -0.005 | 40 | 0.14507 | -1.4956 | 0.25415 | 0.20093 | 0.523 (0.339) | 0.375/0.650/0.775 | 0.2480/0.4738/0.6071 | +0.001 |
| Level | 4 | M3 | 34 | 0.10402 | -2.3427 | 0.18076 | 0.15027 | 0.440 (0.320) | 0.441/0.735/0.853 | 0.2071/0.4052/0.5333 | -0.012 | 40 | 0.14193 | -1.8299 | 0.25493 | 0.20123 | 0.535 (0.311) | 0.375/0.725/0.875 | 0.2916/0.5711/0.7550 | +0.023 |
| Level | 4 | AR1_GARCH11 | 34 | 0.10366 | -2.3582 | 0.18045 | 0.14964 | 0.445 (0.318) | 0.441/0.735/0.853 | 0.2105/0.4055/0.5259 | -0.008 | 40 | 0.14111 | -1.8370 | 0.25450 | 0.20032 | 0.536 (0.302) | 0.425/0.725/0.900 | 0.3135/0.6026/0.7836 | +0.028 |
| Slope | 1 | M0 | 34 | 0.05764 | -3.5161 | 0.10118 | 0.08334 | 0.518 (0.261) | 0.559/0.971/0.971 | 0.1676/0.3184/0.4087 | +0.004 | 40 | 0.07535 | -2.9246 | 0.13856 | 0.09859 | 0.616 (0.248) | 0.675/0.825/0.825 | 0.1700/0.3230/0.4146 | +0.012 |
| Slope | 1 | M1 | 34 | 0.05788 | -3.5112 | 0.10164 | 0.08399 | 0.528 (0.261) | 0.588/0.941/0.971 | 0.1673/0.3180/0.4081 | +0.000 | 40 | 0.07626 | -2.8930 | 0.14031 | 0.09982 | 0.621 (0.247) | 0.625/0.825/0.825 | 0.1699/0.3227/0.4142 | +0.000 |
| Slope | 1 | M2 | 34 | 0.05790 | -3.5110 | 0.10152 | 0.08385 | 0.526 (0.261) | 0.559/0.941/0.971 | 0.1678/0.3188/0.4089 | -0.000 | 40 | 0.07663 | -2.8838 | 0.14090 | 0.10025 | 0.626 (0.246) | 0.625/0.825/0.825 | 0.1699/0.3225/0.4144 | -0.005 |
| Slope | 1 | M3 | 34 | 0.05690 | -3.5918 | 0.10098 | 0.08261 | 0.508 (0.312) | 0.353/0.794/0.971 | 0.1185/0.2417/0.3281 | +0.017 | 40 | 0.07500 | -2.9230 | 0.13868 | 0.09838 | 0.610 (0.249) | 0.625/0.825/0.850 | 0.1656/0.3375/0.4554 | +0.017 |
| Slope | 1 | AR1_GARCH11 | 34 | 0.05644 | -3.5795 | 0.10097 | 0.08268 | 0.507 (0.292) | 0.471/0.853/0.971 | 0.1379/0.2621/0.3364 | +0.025 | 40 | 0.07558 | -2.8867 | 0.13787 | 0.09771 | 0.598 (0.239) | 0.675/0.850/0.850 | 0.1964/0.3731/0.4789 | +0.009 |
| Slope | 4 | M0 | 34 | 0.11156 | -2.1769 | 0.19248 | 0.15821 | 0.561 (0.248) | 0.618/0.853/1.000 | 0.3352/0.6369/0.8174 | +0.022 | 40 | 0.19048 | -0.8736 | 0.34497 | 0.25694 | 0.545 (0.317) | 0.500/0.700/0.775 | 0.3400/0.6460/0.8292 | +0.017 |
| Slope | 4 | M1 | 34 | 0.11406 | -2.1520 | 0.19729 | 0.16158 | 0.582 (0.249) | 0.559/0.853/0.971 | 0.3327/0.6321/0.8113 | +0.000 | 40 | 0.19383 | -0.8031 | 0.35035 | 0.26269 | 0.556 (0.319) | 0.450/0.675/0.775 | 0.3386/0.6433/0.8256 | +0.000 |
| Slope | 4 | M2 | 34 | 0.11312 | -2.1547 | 0.19627 | 0.16092 | 0.578 (0.248) | 0.588/0.853/1.000 | 0.3329/0.6321/0.8127 | +0.008 | 40 | 0.19304 | -0.8292 | 0.34882 | 0.26159 | 0.563 (0.317) | 0.400/0.700/0.775 | 0.3394/0.6465/0.8309 | +0.004 |
| Slope | 4 | M3 | 34 | 0.10714 | -2.3190 | 0.18870 | 0.15406 | 0.543 (0.288) | 0.529/0.824/0.912 | 0.2492/0.5032/0.6786 | +0.061 | 40 | 0.18768 | -0.9738 | 0.34174 | 0.25565 | 0.545 (0.315) | 0.475/0.725/0.850 | 0.3291/0.6611/0.8928 | +0.032 |
| Slope | 4 | AR1_GARCH11 | 34 | 0.10713 | -2.3230 | 0.18904 | 0.15446 | 0.544 (0.279) | 0.559/0.824/0.941 | 0.2677/0.5262/0.6970 | +0.061 | 40 | 0.18844 | -0.9322 | 0.34237 | 0.25533 | 0.533 (0.309) | 0.500/0.725/0.825 | 0.3484/0.6915/0.9219 | +0.028 |
| Curvature | 1 | M0 | 34 | 0.15514 | -1.4918 | 0.26772 | 0.20726 | 0.517 (0.245) | 0.588/0.882/0.971 | 0.4840/0.9195/1.1802 | -0.013 | 40 | 0.21094 | -0.8858 | 0.38457 | 0.28732 | 0.511 (0.287) | 0.525/0.825/0.875 | 0.4766/0.9055/1.1622 | -0.031 |
| Curvature | 1 | M1 | 34 | 0.15312 | -1.5138 | 0.26403 | 0.20723 | 0.525 (0.243) | 0.618/0.912/0.971 | 0.4806/0.9131/1.1720 | +0.000 | 40 | 0.20463 | -0.9661 | 0.37156 | 0.28202 | 0.517 (0.286) | 0.550/0.850/0.875 | 0.4733/0.8994/1.1543 | +0.000 |
| Curvature | 1 | M2 | 34 | 0.15322 | -1.5102 | 0.26414 | 0.20692 | 0.524 (0.243) | 0.588/0.912/0.971 | 0.4812/0.9146/1.1716 | -0.001 | 40 | 0.20558 | -0.9610 | 0.37258 | 0.28248 | 0.517 (0.286) | 0.550/0.850/0.875 | 0.4738/0.9034/1.1572 | -0.005 |
| Curvature | 1 | M3 | 34 | 0.15319 | -1.5229 | 0.26588 | 0.20713 | 0.527 (0.329) | 0.441/0.618/0.765 | 0.2704/0.5323/0.7035 | -0.000 | 40 | 0.20066 | -1.1289 | 0.37497 | 0.28482 | 0.516 (0.252) | 0.625/0.900/0.975 | 0.5583/1.0991/1.4548 | +0.019 |
| Curvature | 1 | AR1_GARCH11 | 34 | 0.15174 | -1.5595 | 0.26530 | 0.20670 | 0.526 (0.327) | 0.441/0.647/0.735 | 0.2830/0.5377/0.6901 | +0.009 | 40 | 0.20116 | -1.1223 | 0.37498 | 0.28413 | 0.512 (0.244) | 0.625/0.900/0.975 | 0.5929/1.1266/1.4459 | +0.017 |
| Curvature | 4 | M0 | 34 | 0.26648 | -0.2721 | 0.44245 | 0.33537 | 0.551 (0.200) | 0.794/0.941/0.971 | 0.9679/1.8390/2.3604 | -0.045 | 40 | 0.43304 | 0.5352 | 0.78221 | 0.58915 | 0.450 (0.293) | 0.500/0.750/0.875 | 0.9532/1.8110/2.3244 | -0.073 |
| Curvature | 4 | M1 | 34 | 0.25496 | -0.3559 | 0.42300 | 0.32276 | 0.564 (0.195) | 0.824/0.941/0.971 | 0.9292/1.7655/2.2660 | +0.000 | 40 | 0.40342 | 0.3913 | 0.73269 | 0.54734 | 0.463 (0.290) | 0.525/0.750/0.900 | 0.9149/1.7383/2.2311 | +0.000 |
| Curvature | 4 | M2 | 34 | 0.25531 | -0.3437 | 0.42319 | 0.32350 | 0.563 (0.194) | 0.824/0.941/0.971 | 0.9373/1.7811/2.2890 | -0.001 | 40 | 0.40528 | 0.3993 | 0.73587 | 0.55043 | 0.462 (0.289) | 0.525/0.750/0.900 | 0.9186/1.7490/2.2464 | -0.005 |
| Curvature | 4 | M3 | 34 | 0.23745 | -0.6172 | 0.43412 | 0.33056 | 0.581 (0.272) | 0.529/0.853/0.882 | 0.5380/1.0606/1.4057 | +0.069 | 40 | 0.39880 | 0.2726 | 0.74098 | 0.55668 | 0.467 (0.257) | 0.525/0.925/0.975 | 1.0802/2.1369/2.8392 | +0.011 |
| Curvature | 4 | AR1_GARCH11 | 34 | 0.23532 | -0.6518 | 0.43190 | 0.32914 | 0.582 (0.270) | 0.559/0.824/0.882 | 0.5422/1.0415/1.3498 | +0.077 | 40 | 0.39968 | 0.2676 | 0.74249 | 0.55665 | 0.461 (0.251) | 0.550/0.925/0.975 | 1.1228/2.1641/2.8073 | +0.009 |

Complete results, including MEDIUM and ALL regimes, coverage counts, interval
widths, PIT histograms, standardized-error diagnostics, CRPS Monte Carlo
uncertainty, and all 120 model/factor/horizon/regime groups are in
`results/canadian_volatility_regime_20261120/regime_metrics.csv` and
`regime_calibration_diagnostics.csv`.

## Paired comparisons and exploratory uncertainty

Paired differences are first model minus second model; a negative mean favors
the first model. HIGH-minus-LOW interactions describe association with the
predefined regime, not a causal effect. Factor-level comparison and interaction
intervals are recorded in `regime_pairwise_differences.csv` and
`regime_interaction_summary.csv`.

Bootstrap: 9,999 circular stationary resamples of the full
chronological 130-origin sequence, expected block length
4.0 origins, seed 20261120.
The origin-level score vectors and labels were sampled together. Missing-regime
resamples: {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}; origin group sizes are
{'LOW': 34, 'MEDIUM': 56, 'HIGH': 40}, with fewer than 15 origins in
[]. Reported intervals are
exploratory percentile 95% intervals; no exploratory p-values are reported.
The accepted original test family and Holm-adjusted results are unchanged.

## Research questions

1. **Is volatility modeling more valuable in HIGH regimes?** Not on average
   against M1 in these data: mean SV skill is 1.95% LOW versus
   1.72% HIGH, and GARCH skill is 2.54%
   versus 1.46%. The high-minus-low skill intervals exclude
   zero in 1 of 24
   non-baseline model/factor/horizon comparisons; only
   0 of 18
   comparisons involving Bayesian M2/M3 or GARCH exclude zero. The one interval
   excluding zero is M0/level/h=4. Raw HIGH CRPS is not used to infer
   relative performance.
2. **Does Bayesian SV outperform GARCH?** SV is better in
   5 of six HIGH groups and 0 of six LOW groups;
   GARCH is better in 6 LOW groups.
   The GARCH-minus-SV confidence intervals all include zero, so this is a
   descriptive reversal, not a resolved difference.
3. **Is curvature more regime-sensitive?** Across relative-skill interactions
   for M2, M3, and GARCH, the largest mean absolute HIGH-minus-LOW change is
   curvature (mean absolute skill interaction 0.027). This descriptive ranking does not establish a
   curvature-specific effect.
4. **Do probabilistic gains align with point forecasts?** For SV versus M2,
   CRPS improves in 4/6 LOW and
   6/6 HIGH cells, but RMSE
   improves in 3/6 and
   3/6, respectively. For
   GARCH versus M1, CRPS improves in 4/6
   LOW and 5/6 HIGH cells, versus
   RMSE improvement in 2/6 and
   3/6. This is consistent with
   gains that are not solely improvements in conditional means.
5. **Does calibration differ by regime?** Mean absolute deviation of grouped
   coverage from nominal is 0.076 LOW and
   0.071 HIGH; mean absolute PIT-mean
   deviation from 0.5 is 0.045 LOW and
   0.052 HIGH. These summaries do not replace the
   histograms, counts, coverage, and width tables. A PIT mean near 0.5 alone is
   not full calibration evidence.
6. **Sensitivity to regime definition?** The primary 26-week composite
   specification was not tuned. Optional 13-week sensitivity analysis was not
   run.

## Figures

- `results/canadian_volatility_regime_20261120/figures/figure_1_regime_history.png`
  and `.pdf`
- `figure_2_conditional_crps.png` and `.pdf`
- `figure_3_relative_crps_skill.png` and `.pdf`
- `figure_4_garch_vs_bayesian_sv.png` and `.pdf`
- `figure_5_calibration_pit.png` and `.pdf`
- `figure_5_calibration_intervals.png` and `.pdf`

Figures are generated from saved outputs by
`src/plot_canadian_volatility_regimes.py`. No manuscript source, PDF, or
generated manuscript table was modified.

## Validation and limitations

The complete 3,900 records from the four accepted factor models and AR-GARCH
were joined on origin, target, factor, horizon, and manuscript model. We
verified five models, three factors, two horizons, 130 origins, unique keys,
matching realized values, valid predictive moments, intervals, and score
components. Unconditional aggregates reproduce the accepted metrics at
floating-point tolerance. Bayesian predictive simulations were not regenerated
and no forecasting model was re-estimated.

The classifications are based on observation-time factor histories, not
verified publication vintages. Regime groups are modest in size and forecasts
overlap in time; bootstrap uncertainty accounts for serial dependence only
approximately. Results are exploratory and do not support causal, portfolio,
or joint-yield-curve claims.

The largest subgroup-mean CRPS Monte Carlo standard error is
0.00108. Sampling uncertainty is generally
wider than this simulation component; the bootstrap intervals matter when
interpreting small score differences.
