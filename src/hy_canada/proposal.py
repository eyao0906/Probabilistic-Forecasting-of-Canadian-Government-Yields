"""Freeze a training-only correlated random-walk proposal from the pilot draws."""

import json

import numpy as np
from scipy.special import logit

from .experiment import OUT


def main():
    with np.load(OUT / "dlns_parameter_draws.npz") as archive:
        draws = archive["draws"].reshape(-1, 18)
    transformed = np.column_stack((np.log(draws[:, 6:17]),
                                   logit((draws[:, 17] - 0.05) / 1.95)))
    covariance = np.cov(transformed.T) + np.eye(12) * 1e-6
    np.linalg.cholesky(covariance)
    path = OUT / "dlns_joint_proposal.json"
    path.write_text(json.dumps(covariance.tolist(), indent=2) + "\n", encoding="utf-8")
    print("Saved fixed pilot-derived proposal:", path)


if __name__ == "__main__":
    main()
