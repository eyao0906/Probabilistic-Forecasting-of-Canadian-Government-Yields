# Model fit

No posterior predictive model-fit conclusion is warranted until reliable joint
chains are available. Fitted posterior curves, standardized Kalman innovations,
posterior predictive skewness and kurtosis, and posterior latent-factor
trajectories have not been established.

Exploratory conditional state draws and OLS comparisons are saved for
diagnostic inspection, not as validated posterior inference. They use 25
evenly spaced parameter draws per chain, so both convergence failure and
limited conditional Monte Carlo precision apply.

As a descriptive training-only check, OLS factor extraction at the fixed
decay 0.7 leaves eight residuals constrained by three normal equations. The
residual covariance therefore has at most five nonzero eigenvalues; three
numerically zero eigenvalues and five positive ones are recorded in
`results/hy_canada/ols_descriptive.json`. Off-diagonal residual correlations
range from -0.990 to 0.946 and lag-one residual correlations from 0.895 to
0.984. These residuals have structural cross-maturity dependence because of
the OLS projection, so they are not direct estimates of model measurement
errors. Their strong temporal persistence nevertheless motivates a targeted
posterior predictive check of the five-dimensional complement. Diagonal,
temporally independent measurement noise cannot by itself represent that
residual dynamics.
