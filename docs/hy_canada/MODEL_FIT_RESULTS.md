# Model fit

No posterior predictive fit, p-values, skewness, kurtosis, latent factors or
standardized model innovations are reported because neither model has
converged. In particular, a successful blocked transition test does not
establish model adequacy.

As a training-only descriptive check, fixed-decay OLS projection leaves an
eight-dimensional residual vector subject to three normal equations, hence a
five-dimensional residual complement. Off-diagonal residual correlations
range from -0.990 to 0.946 and lag-one residual correlations from 0.895 to
0.984; see `results/hy_canada/ols_descriptive.json`. These projected residuals
are not direct estimates of measurement errors. Their temporal persistence
motivates a posterior predictive residual-complement check once valid
inference is available. It is not evidence that blocked MCMC fixes
structural misspecification.
