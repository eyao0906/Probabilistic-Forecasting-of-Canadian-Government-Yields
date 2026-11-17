"""MCMC chain diagnostics using ArviZ's rank-normalized estimators."""

from __future__ import annotations

import numpy as np

from .sampler import ChainDraws


def summarize_chain_diagnostics(chains: list[ChainDraws]) -> dict:
    """Return rank R-hat and bulk/tail ESS for the sampled model parameters."""
    if len(chains) < 2:
        raise ValueError("at least two independent chains are required")
    model_names = {chain.model for chain in chains}
    if len(model_names) != 1:
        raise ValueError("all chains must belong to the same model")
    draw_counts = {len(chain.decay_per_year) for chain in chains}
    if len(draw_counts) != 1 or min(draw_counts) < 4:
        raise ValueError("chains must have equal draw counts of at least four")

    import arviz as az

    posterior = {
        "factor_mean": np.stack([chain.factor_mean for chain in chains]),
        "factor_persistence": np.stack([chain.persistence for chain in chains]),
        "log_variance": np.stack([chain.log_variance for chain in chains]),
        "measurement_variance": np.stack(
            [chain.measurement_variance for chain in chains]
        ),
        "decay_per_year": np.stack([chain.decay_per_year for chain in chains])[:, :, None],
    }
    if chains[0].model == "SS_M4":
        posterior["volatility_persistence"] = np.stack(
            [chain.volatility_persistence for chain in chains]
        )
        posterior["volatility_variance"] = np.stack(
            [chain.volatility_variance for chain in chains]
        )

    inference_data = az.from_dict({"posterior": posterior})
    rhat = az.rhat(inference_data, method="rank")
    bulk_ess = az.ess(inference_data, method="bulk")
    tail_ess = az.ess(inference_data, method="tail")
    output = {"model": chains[0].model, "chains": len(chains), "draws_per_chain": min(draw_counts)}
    for label, dataset in (
        ("rank_rhat", rhat),
        ("bulk_ess", bulk_ess),
        ("tail_ess", tail_ess),
    ):
        output[label] = {}
        posterior_dataset = dataset.to_dataset()
        for variable in posterior_dataset.data_vars:
            values = np.asarray(posterior_dataset[variable].values)
            output[label][variable] = values.tolist()
    output["maximum_rank_rhat"] = float(
        max(np.nanmax(np.asarray(value)) for value in output["rank_rhat"].values())
    )
    output["minimum_bulk_ess"] = float(
        min(np.nanmin(np.asarray(value)) for value in output["bulk_ess"].values())
    )
    output["minimum_tail_ess"] = float(
        min(np.nanmin(np.asarray(value)) for value in output["tail_ess"].values())
    )
    return output
