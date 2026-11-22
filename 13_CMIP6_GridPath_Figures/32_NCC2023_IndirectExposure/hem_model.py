"""Joint link extraction following Verschuur et al. (2023), SI equations 1-3.

These are structural economic-activity exposures, not realized GDP losses.
Forward and backward accounts are alternative perspectives, not additive.
Rows of Z supply products to columns. All flows use one common price/year unit.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve


@dataclass
class MRIO:
    z: np.ndarray
    final: np.ndarray
    output: np.ndarray
    value_added: np.ndarray

    def validate(self, rtol=1e-7):
        n = len(self.output)
        if self.z.shape != (n, n) or self.final.shape[0] != n:
            raise ValueError("MRIO dimensions disagree")
        if self.value_added.shape != (n,):
            raise ValueError("Value added must be an industry vector")
        for array in [self.z, self.final, self.output, self.value_added]:
            if not np.isfinite(array).all():
                raise ValueError("Non-finite MRIO entry")
        if (self.z < 0).any() or (self.output <= 0).any():
            raise ValueError("Negative intermediate flows or nonpositive output need an explicit treatment")
        # Negative inventory changes in final demand are permitted, but never
        # converted into positive lost consumption by the extraction routine.
        if not np.allclose(self.z.sum(1) + self.final.sum(1), self.output, rtol=rtol):
            raise ValueError("MRIO supply accounting identity does not hold")
        if not np.allclose(self.z.sum(0) + self.value_added, self.output, rtol=rtol):
            raise ValueError("MRIO use accounting identity does not hold")


def _share(q, shape):
    q = np.asarray(q, dtype=float)
    if q.shape != shape or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("Extraction shares must be finite, correctly shaped, and in [0, 1]")
    return q


def joint_extraction(mrio, intermediate_share, final_share=None):
    """One solve per joint closure/counterfactual, never a sum of port losses.

    The first-hit / higher-order forward split is an algebraic extension for
    this project, not a direct/indirect decomposition published by the paper.
    """
    mrio.validate()
    z, y, x, v = mrio.z, mrio.final, mrio.output, mrio.value_added
    q = _share(intermediate_share, z.shape)
    qy = np.zeros_like(y) if final_share is None else _share(final_share, y.shape)
    dz = z * q
    dy = np.maximum(y, 0) * qy
    remaining = z - dz
    identity = np.eye(len(x))
    b_after = remaining / x[:, None]
    a_after = remaining / x[None, :]
    a_before = z / x[None, :]

    # Stable difference solves also expose the first-hit versus propagation
    # identity: delta_x = colsum(delta_Z) + B_after.T @ delta_x.
    forward_first = dz.sum(axis=0)
    forward_total = solve(identity - b_after.T, forward_first)
    forward_higher = forward_total - forward_first
    backward_intermediate = solve(identity - a_after, dz.sum(axis=1))
    backward_consumption = solve(identity - a_before, dy.sum(axis=1))
    forward_output = x - forward_total
    residual = (identity - b_after.T) @ forward_output - v
    if np.max(np.abs(residual)) > 1e-6 * max(float(x.max()), 1):
        raise ValueError("Forward equilibrium solve failed")
    return dict(
        forward_output_exposure=forward_total,
        forward_first_hit=forward_first,
        forward_higher_order_exposure=forward_higher,
        backward_intermediate_exposure=backward_intermediate,
        backward_final_demand_exposure=backward_consumption,
        final_consumption_exposure_by_supplier_buyer=dy,
        forward_remaining_output=forward_output,
        removed_intermediate_flows=dz,
    )


def overlap_safe_share(flow_weights, impacted):
    """Union over gates per unique trade flow, then aggregate its share once."""
    weights = np.asarray(flow_weights, float)
    hits = np.asarray(impacted, bool)
    if hits.ndim != 2 or len(weights) != len(hits):
        raise ValueError("Expected unique flows by gates")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid flow weights")
    return float(weights[hits.any(axis=1)].sum() / weights.sum())
