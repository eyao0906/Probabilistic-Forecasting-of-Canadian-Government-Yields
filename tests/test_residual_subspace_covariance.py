import numpy as np
import unittest

from src.models import ns_loadings
from src.residual_subspace_analysis import (
    energy_score as residual_energy_score,
    paired_bootstrap,
    residual_calibration,
    stationary_bootstrap_indices,
)
from src.residual_subspace_covariance import (
    assemble_yield_covariance,
    decompose_yields,
    dss_components,
    estimate_residual_covariances,
    orthogonal_dss_components,
    residual_basis,
)
from src.scores import energy_score


class ResidualSubspaceTests(unittest.TestCase):
    def setUp(self):
        self.load = ns_loadings(np.array([0.25, 0.5, 1, 2, 5, 7, 10, 30]), 0.5160377358490565)

    def test_orthogonal_residual_decomposition_reconstructs_yields(self):
        factor_basis, residual_u = residual_basis(self.load)
        rng = np.random.default_rng(42)
        yields = rng.normal(size=(25, 8))

        factors, coordinates, error = decompose_yields(yields, self.load, residual_u)

        self.assertLess(error, 1e-12)
        self.assertTrue(np.allclose(self.load.T @ (yields - factors @ self.load.T).T, 0, atol=1e-12))
        self.assertTrue(np.allclose(residual_u.T @ self.load, 0, atol=1e-12))
        self.assertTrue(np.allclose(factor_basis.T @ residual_u, 0, atol=1e-12))
        self.assertTrue(np.allclose(yields, factors @ self.load.T + coordinates @ residual_u.T))

    def test_shrinkage_and_full_covariance_are_positive_definite_and_score_splits(self):
        factor_basis, residual_u = residual_basis(self.load)
        rng = np.random.default_rng(73)
        coordinates = rng.normal(size=(200, 5)) @ np.diag([0.1, 0.2, 0.3, 0.4, 0.5])
        estimates = estimate_residual_covariances(coordinates)
        variances = np.array([0.02, 0.04, 0.06])
        covariance = assemble_yield_covariance(
            self.load, variances, residual_u, estimates["ledoit_wolf"]["covariance"]
        )
        observed = rng.normal(size=8)
        mean = rng.normal(size=8)
        full = dss_components(observed, mean, covariance)
        split = orthogonal_dss_components(
            observed, mean, variances, self.load, factor_basis, residual_u,
            estimates["ledoit_wolf"]["covariance"], full["dss"],
        )

        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)
        self.assertTrue(np.isclose(split["reconstructed_dss"], full["dss"], rtol=1e-10, atol=1e-10))
        self.assertTrue(all(np.linalg.eigvalsh(item["covariance"]).min() > 0 for item in estimates.values()))

    def test_energy_score_matches_existing_distinct_pair_definition(self):
        rng = np.random.default_rng(8)
        scenarios = rng.normal(size=(32, 8))
        observed = rng.normal(size=8)

        self.assertTrue(np.isclose(
            residual_energy_score(scenarios, observed),
            energy_score(scenarios, observed),
            rtol=1e-12,
            atol=1e-12,
        ))

    def test_stationary_bootstrap_and_paired_summary_keep_origin_pairs(self):
        left = {
            "dss": np.arange(12, dtype=float),
            "energy_score": np.arange(12, dtype=float) / 10,
            "squared_errors": np.ones((12, 8)),
            "coverage_90": np.ones((12, 8)),
        }
        right = {
            "dss": left["dss"] + 2,
            "energy_score": left["energy_score"] + 0.5,
            "squared_errors": left["squared_errors"] * 4,
            "coverage_90": np.zeros((12, 8)),
        }
        indices = stationary_bootstrap_indices(12, 4, 64, np.random.default_rng(19))
        result = paired_bootstrap(left, right, indices)

        self.assertEqual(indices.shape, (64, 12))
        self.assertEqual(result["dss"]["right_minus_left"], 2)
        self.assertAlmostEqual(result["energy_score"]["right_minus_left"], 0.5)
        self.assertAlmostEqual(result["rmse"]["right_minus_left"], 1)
        self.assertEqual(result["coverage_90"]["right_minus_left"], -1)

    def test_residual_calibration_reports_dependence_for_both_models(self):
        rng = np.random.default_rng(103)
        records = []
        for horizon in (1, 4):
            for model in ("M4", "M5"):
                for origin in range(4):
                    whitened = rng.normal(size=5)
                    records.append({
                        "origin": f"2020-01-0{origin + 1}",
                        "model": model,
                        "horizon": horizon,
                        "residual_target_coordinates": rng.normal(size=5).tolist(),
                        "residual_marginal_sd": np.ones(5).tolist(),
                        "whitened_residual_coordinates": whitened.tolist(),
                        "residual_mahalanobis_squared": float(whitened @ whitened),
                        "factor_target_error": rng.normal(size=3).tolist(),
                        "training_factor_residual_correlation": rng.uniform(-0.2, 0.2, size=(3, 5)).tolist(),
                    })

        report = residual_calibration(records)

        self.assertEqual(set(report["1"]["factor_residual_dependence_by_model"]), {"M4", "M5"})
        self.assertEqual(report["1"]["n"], 4)
