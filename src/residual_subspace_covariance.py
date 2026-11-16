"""Numerics for the residual-subspace yield covariance experiment."""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf, OAS


def residual_basis(load: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return orthonormal factor and residual bases for a full-rank loading matrix."""
    load = np.asarray(load, dtype=float)
    if load.ndim != 2 or load.shape[0] <= load.shape[1] or not np.isfinite(load).all():
        raise ValueError("Loadings must be a finite tall matrix")
    if np.linalg.matrix_rank(load) != load.shape[1]:
        raise ValueError("Loading matrix must have full column rank")
    q, _ = np.linalg.qr(load, mode="complete")
    factor_basis, residual_basis = q[:, :load.shape[1]], q[:, load.shape[1]:]
    if (not np.allclose(factor_basis.T @ factor_basis, np.eye(load.shape[1]), atol=1e-12)
        or not np.allclose(residual_basis.T @ residual_basis,
                           np.eye(load.shape[0] - load.shape[1]), atol=1e-12)
        or not np.allclose(residual_basis.T @ load, 0, atol=1e-12)):
        raise ArithmeticError("Failed to construct an orthonormal residual basis")
    return factor_basis, residual_basis


def decompose_yields(
    yields: np.ndarray,
    load: np.ndarray,
    residual_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Project yields into the DNS factor space and its orthogonal complement."""
    yields = np.asarray(yields, dtype=float)
    load = np.asarray(load, dtype=float)
    residual_basis = np.asarray(residual_basis, dtype=float)
    if (yields.ndim != 2 or load.ndim != 2 or residual_basis.ndim != 2
        or yields.shape[1] != load.shape[0]
        or residual_basis.shape[0] != load.shape[0]
        or not np.isfinite(yields).all()):
        raise ValueError("Incompatible or nonfinite yield decomposition inputs")
    factors = np.linalg.lstsq(load, yields.T, rcond=None)[0].T
    coordinates = yields @ residual_basis
    reconstructed = factors @ load.T + coordinates @ residual_basis.T
    error = yields - reconstructed
    relative_error = float(np.linalg.norm(error) / max(np.linalg.norm(yields), 1.0))
    if relative_error > 1e-11:
        raise ArithmeticError(f"Residual decomposition error is too large: {relative_error}")
    return factors, coordinates, relative_error


def estimate_residual_covariances(coordinates: np.ndarray) -> dict[str, dict]:
    """Fit centered Ledoit-Wolf, sample, and OAS covariances to residual coordinates."""
    coordinates = np.asarray(coordinates, dtype=float)
    if (coordinates.ndim != 2 or coordinates.shape[0] <= coordinates.shape[1]
        or not np.isfinite(coordinates).all()):
        raise ValueError("Need more finite residual observations than coordinates")
    estimators = {
        "ledoit_wolf": LedoitWolf().fit(coordinates),
        "unshrunk": None,
        "oas": OAS().fit(coordinates),
    }
    covariances = {
        "ledoit_wolf": estimators["ledoit_wolf"].covariance_,
        "unshrunk": np.cov(coordinates, rowvar=False, ddof=1),
        "oas": estimators["oas"].covariance_,
    }
    shrinkage = {
        "ledoit_wolf": float(estimators["ledoit_wolf"].shrinkage_),
        "unshrunk": 0.0,
        "oas": float(estimators["oas"].shrinkage_),
    }
    output = {}
    for name, covariance in covariances.items():
        covariance = (np.asarray(covariance) + np.asarray(covariance).T) / 2
        if covariance.shape != (coordinates.shape[1], coordinates.shape[1]):
            raise ArithmeticError(f"Unexpected {name} covariance shape")
        if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
            raise ArithmeticError(f"{name} covariance is not positive definite")
        np.linalg.cholesky(covariance)
        output[name] = {"covariance": covariance, "shrinkage": shrinkage[name]}
    return output


def assemble_yield_covariance(
    load: np.ndarray,
    factor_variances: np.ndarray,
    residual_basis: np.ndarray,
    residual_covariance: np.ndarray,
) -> np.ndarray:
    """Combine independent factor and residual-subspace covariance components."""
    load = np.asarray(load, dtype=float)
    factor_variances = np.asarray(factor_variances, dtype=float)
    residual_basis = np.asarray(residual_basis, dtype=float)
    residual_covariance = np.asarray(residual_covariance, dtype=float)
    n_factor = load.shape[1]
    n_residual = load.shape[0] - n_factor
    if (load.ndim != 2 or factor_variances.shape != (n_factor,)
        or residual_basis.shape != (load.shape[0], n_residual)
        or residual_covariance.shape != (n_residual, n_residual)
        or not np.isfinite(factor_variances).all() or np.any(factor_variances <= 0)
        or not np.isfinite(residual_covariance).all()):
        raise ValueError("Invalid factor or residual covariance dimensions and values")
    if not np.allclose(residual_covariance, residual_covariance.T, atol=1e-12):
        raise ValueError("Residual covariance must be symmetric")
    covariance = (
        (load * factor_variances) @ load.T
        + residual_basis @ residual_covariance @ residual_basis.T
    )
    covariance = (covariance + covariance.T) / 2
    np.linalg.cholesky(covariance)
    return covariance


def dss_components(observed: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> dict:
    """Return Gaussian DSS terms using Cholesky solves, without covariance inversion."""
    observed = np.asarray(observed, dtype=float)
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if (observed.ndim != 1 or mean.shape != observed.shape
        or covariance.shape != (len(observed), len(observed))
        or not np.isfinite(observed).all() or not np.isfinite(mean).all()
        or not np.isfinite(covariance).all()):
        raise ValueError("Invalid DSS inputs")
    covariance = (covariance + covariance.T) / 2
    chol = np.linalg.cholesky(covariance)
    error = observed - mean
    whitened = np.linalg.solve(chol, error)
    logdet = float(2 * np.log(np.diag(chol)).sum())
    mahalanobis = float(whitened @ whitened)
    return {
        "log_determinant": logdet,
        "mahalanobis": mahalanobis,
        "dss": logdet + mahalanobis,
    }


def orthogonal_dss_components(
    observed: np.ndarray,
    mean: np.ndarray,
    factor_variances: np.ndarray,
    load: np.ndarray,
    factor_basis: np.ndarray,
    residual_basis: np.ndarray,
    residual_covariance: np.ndarray,
    full_dss: float,
) -> dict:
    """Split corrected DSS into independent factor-space and residual-space terms."""
    error = np.asarray(observed, dtype=float) - np.asarray(mean, dtype=float)
    transform = factor_basis.T @ load
    factor_covariance = (transform * np.asarray(factor_variances)) @ transform.T
    factor_error = factor_basis.T @ error
    residual_error = residual_basis.T @ error
    factor = dss_components(
        factor_error, np.zeros_like(factor_error), factor_covariance
    )
    residual = dss_components(
        residual_error, np.zeros_like(residual_error), residual_covariance
    )
    combined = factor["dss"] + residual["dss"]
    if not np.isclose(combined, full_dss, rtol=1e-9, atol=1e-9):
        raise ArithmeticError("Orthogonal DSS components do not reconstruct the full score")
    return {
        "factor_log_determinant": factor["log_determinant"],
        "factor_mahalanobis": factor["mahalanobis"],
        "residual_log_determinant": residual["log_determinant"],
        "residual_mahalanobis": residual["mahalanobis"],
        "reconstructed_dss": combined,
    }


def eigen_diagnostics(observed: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> dict:
    """Report covariance spectrum and standardized error contributions by eigen-direction."""
    covariance = (np.asarray(covariance, dtype=float) + np.asarray(covariance).T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() <= 0:
        raise ArithmeticError("Forecast covariance is not positive definite")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    projection = eigenvectors.T @ (np.asarray(observed) - np.asarray(mean))
    contributions = projection**2 / eigenvalues
    top = int(np.argmax(contributions))
    return {
        "eigenvalues_descending": eigenvalues.tolist(),
        "condition_number": float(eigenvalues[0] / eigenvalues[-1]),
        "eigen_direction_error_contributions": contributions.tolist(),
        "largest_contribution_index": top,
        "largest_contribution_share": float(contributions[top] / max(contributions.sum(), 1e-300)),
        "largest_contribution_direction": eigenvectors[:, top].tolist(),
    }
