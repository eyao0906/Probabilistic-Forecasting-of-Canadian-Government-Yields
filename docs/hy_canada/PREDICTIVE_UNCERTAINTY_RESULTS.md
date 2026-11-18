# Predictive uncertainty

No forecast densities, coverage or model comparison are reported: both
joint posteriors are unconverged. Requested origins 2016-01-08, 2020-03-27
and 2025-11-28 are present in the existing panel; their presence alone does
not provide a valid real-time experiment. See `forecast_target_dates.csv`.

The constant-Q conditional forward simulator includes full factor innovation
covariance and diagonal measurement noise and has a Gaussian moment test.
Parameter and state uncertainty require integration over reliable posterior
draws; the SV future-volatility simulator and origin-specific refits remain
unimplemented. Two-step benchmarks have not been refitted.
