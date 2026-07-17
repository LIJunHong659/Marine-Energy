"""Transparent S0 offshore transmission loss and export dispatch functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TransmissionLossSpec:
    """Reduced loss model with linear terminal and quadratic cable losses."""

    terminal_loss_fraction: float
    cable_full_load_loss_per_100km: float

    def validate(self, distance_km: float) -> None:
        if not 0.0 <= self.terminal_loss_fraction < 1.0:
            raise ValueError("terminal_loss_fraction must lie in [0, 1)")
        if self.cable_full_load_loss_per_100km < 0.0:
            raise ValueError("cable loss coefficient must be non-negative")
        if distance_km <= 0.0:
            raise ValueError("offshore distance must be positive")
        full_load_loss = (
            self.terminal_loss_fraction + self.cable_full_load_loss_per_100km * distance_km / 100.0
        )
        if full_load_loss >= 1.0:
            raise ValueError("transmission parameters imply non-positive full-load delivery")


@dataclass(frozen=True)
class TransmissionFlow:
    """Send-side power, separated loss components and land-side delivery."""

    send_mw: FloatArray
    terminal_loss_mw: FloatArray
    cable_loss_mw: FloatArray
    total_loss_mw: FloatArray
    land_mw: FloatArray


@dataclass(frozen=True)
class PiecewiseTransmission:
    """Incremental concave delivery approximation used by the S1 linear program."""

    breakpoints_mw: FloatArray
    segment_widths_mw: FloatArray
    land_delivery_slopes: FloatArray


def calculate_transmission_flow(
    send_mw: npt.ArrayLike,
    capacity_mw: float,
    distance_km: float,
    loss_spec: TransmissionLossSpec,
) -> TransmissionFlow:
    """Calculate terminal and I²R-proxy cable losses without mixing them with curtailment."""
    loss_spec.validate(distance_km)
    send = np.asarray(send_mw, dtype=float)
    if not np.isfinite(send).all() or (send < -1e-12).any():
        raise ValueError("send power must be finite and non-negative")
    if capacity_mw < 0.0:
        raise ValueError("transmission capacity must be non-negative")
    if capacity_mw == 0.0:
        if (send > 1e-12).any():
            raise ValueError("positive send power is impossible at zero capacity")
        zeros = np.zeros_like(send)
        return TransmissionFlow(send, zeros, zeros, zeros, zeros)
    if (send > capacity_mw + 1e-9).any():
        raise ValueError("send power exceeds installed transmission capacity")

    terminal_loss = loss_spec.terminal_loss_fraction * send
    distance_factor = distance_km / 100.0
    cable_loss = loss_spec.cable_full_load_loss_per_100km * distance_factor * send**2 / capacity_mw
    total_loss = terminal_loss + cable_loss
    land = send - total_loss
    if (land < -1e-9).any():
        raise ValueError("transmission model produced negative land-side power")
    return TransmissionFlow(send, terminal_loss, cable_loss, total_loss, land)


def build_piecewise_transmission(
    capacity_mw: float,
    distance_km: float,
    loss_spec: TransmissionLossSpec,
    segments: int,
) -> PiecewiseTransmission:
    """Construct chord slopes for the concave send-to-land delivery curve."""
    loss_spec.validate(distance_km)
    if capacity_mw <= 0.0:
        raise ValueError("piecewise transmission requires positive capacity")
    if segments < 1:
        raise ValueError("segments must be positive")
    breakpoints = np.linspace(0.0, capacity_mw, segments + 1)
    flow = calculate_transmission_flow(
        breakpoints,
        capacity_mw=capacity_mw,
        distance_km=distance_km,
        loss_spec=loss_spec,
    )
    widths = np.diff(breakpoints)
    slopes = np.diff(flow.land_mw) / widths
    if not np.all(np.diff(slopes) <= 1e-12):
        raise RuntimeError("transmission delivery slopes must be non-increasing")
    return PiecewiseTransmission(breakpoints, widths, slopes)


def optimal_export_send(
    surplus_mw: npt.ArrayLike,
    available_capacity_mw: npt.ArrayLike,
    electricity_price_cny_per_mwh: npt.ArrayLike,
    installed_capacity_mw: float,
    distance_km: float,
    loss_spec: TransmissionLossSpec,
    variable_cost_cny_per_mwh_send: float,
    curtailment_penalty_cny_per_mwh: float,
    policy: Literal["economic", "must_take"],
) -> FloatArray:
    """Solve the separable S0 export problem analytically for each hour.

    The economic policy maximizes land-side electricity revenue minus send-side
    variable cost and curtailment penalty. The must-take policy exports every
    physically feasible MWh regardless of price.
    """
    loss_spec.validate(distance_km)
    surplus = np.asarray(surplus_mw, dtype=float)
    available_capacity = np.asarray(available_capacity_mw, dtype=float)
    price = np.asarray(electricity_price_cny_per_mwh, dtype=float)
    if surplus.shape != available_capacity.shape or surplus.shape != price.shape:
        raise ValueError("surplus, available capacity and price arrays must have equal shape")
    if installed_capacity_mw < 0.0:
        raise ValueError("installed transmission capacity must be non-negative")
    if variable_cost_cny_per_mwh_send < 0.0 or curtailment_penalty_cny_per_mwh < 0.0:
        raise ValueError("dispatch cost coefficients must be non-negative")

    limit = np.minimum(np.maximum(surplus, 0.0), np.maximum(available_capacity, 0.0))
    if policy == "must_take":
        return limit
    if policy != "economic":
        raise ValueError(f"unsupported export policy: {policy}")
    if installed_capacity_mw == 0.0:
        return np.zeros_like(limit)

    linear_delivery = 1.0 - loss_spec.terminal_loss_fraction
    quadratic_delivery = (
        loss_spec.cable_full_load_loss_per_100km * distance_km / 100.0 / installed_capacity_mw
    )

    def objective(power: FloatArray) -> FloatArray:
        land = linear_delivery * power - quadratic_delivery * power**2
        return (
            price * land
            - variable_cost_cny_per_mwh_send * power
            - curtailment_penalty_cny_per_mwh * (surplus - power)
        )

    zero = np.zeros_like(limit)
    candidates = [zero, limit]
    if quadratic_delivery > 0.0:
        denominator = 2.0 * price * quadratic_delivery
        stationary = np.zeros_like(limit)
        concave = denominator > 0.0
        numerator = (
            price * linear_delivery
            - variable_cost_cny_per_mwh_send
            + curtailment_penalty_cny_per_mwh
        )
        stationary[concave] = numerator[concave] / denominator[concave]
        candidates.append(np.clip(stationary, 0.0, limit))

    candidate_matrix = np.stack(candidates)
    objective_matrix = np.stack([objective(candidate) for candidate in candidates])
    best_index = np.argmax(objective_matrix, axis=0)
    return np.take_along_axis(candidate_matrix, best_index[None, :], axis=0)[0]
