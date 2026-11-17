from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import beta, multivariate_normal

from src.ssdns_hy.sv_interweave import (
    _reconstruct,
    _standardize,
    interweave_persistence,
    interweave_variance,
)


class FixedRandom:
    def __init__(self, step, uniform):
        self.step, self.uniform_value = step, uniform

    def normal(self, loc, scale):
        return self.step

    def uniform(self):
        return self.uniform_value


class InterweaveTests(unittest.TestCase):
    def test_roundtrip_for_high_persistence(self):
        h = np.array([-.8, -.3, -.5, .2])
        for phi in (.8, .99999, -.999):
            z = _standardize(h, -.6, phi, .4)
            np.testing.assert_allclose(_reconstruct(z, -.6, phi, .4), h, atol=1e-12)

    def test_acceptance_matches_dense_centered_joint_density_and_jacobian(self):
        h = np.array([-.8, -.3, -.5, .2])
        innovations = np.array([.4, .2, -.5, .8])
        mean, phi, variance, upper = -.6, .8, .4, 4.
        n = len(h)

        def direct_joint(path, p, v):
            cov = v * p ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
            prior_path = multivariate_normal.logpdf(path, mean=np.full(n, mean-v/2), cov=cov)
            likelihood = np.sum(-.5 * (path + innovations**2 * np.exp(-path)))
            return prior_path + likelihood

        def path_jacobian(p, v):
            return n / 2 * np.log(v) + (n-1) / 2 * np.log1p(-p**2)

        z = _standardize(h, mean, phi, variance)
        p_new = .3
        h_new = _reconstruct(z, mean, p_new, variance)
        ratio = (direct_joint(h_new, p_new, variance) - direct_joint(h, phi, variance)
                 + path_jacobian(p_new, variance) - path_jacobian(phi, variance)
                 + beta.logpdf((1+p_new)/2, 20, 1.5) - beta.logpdf((1+phi)/2, 20, 1.5)
                 + np.log1p(-p_new**2) - np.log1p(-phi**2))
        probability = min(1., np.exp(ratio))
        self.assertLess(probability, .999)
        step = np.arctanh(p_new) - np.arctanh(phi)
        for u, expected in ((probability/2, True), ((probability+1)/2, False)):
            path, result, accepted = interweave_persistence(
                h, innovations, mean, phi, variance, 20, 1.5, FixedRandom(step, u)
            )
            self.assertEqual(accepted, expected)
            self.assertAlmostEqual(result, p_new if expected else phi)
            np.testing.assert_allclose(path, h_new if expected else h)

        v_new = 2.0
        h_new = _reconstruct(z, mean, phi, v_new)
        x, xp = variance / upper, v_new / upper
        ratio = (direct_joint(h_new, phi, v_new) - direct_joint(h, phi, variance)
                 + path_jacobian(phi, v_new) - path_jacobian(phi, variance)
                 - v_new + variance + np.log(xp*(1-xp)) - np.log(x*(1-x)))
        probability = min(1., np.exp(ratio))
        self.assertLess(probability, .999)
        step = np.log(xp/(1-xp)) - np.log(x/(1-x))
        for u, expected in ((probability/2, True), ((probability+1)/2, False)):
            path, result, accepted = interweave_variance(
                h, innovations, mean, phi, variance, upper, FixedRandom(step, u)
            )
            self.assertEqual(accepted, expected)
            self.assertAlmostEqual(result, v_new if expected else variance)
            np.testing.assert_allclose(path, h_new if expected else h)


if __name__ == "__main__":
    unittest.main()
