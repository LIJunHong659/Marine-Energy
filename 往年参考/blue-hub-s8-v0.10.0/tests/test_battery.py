from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from blue_hub.battery import battery_spec_from_parameters, calculate_battery_losses
from blue_hub.loaders import load_parameters, load_system_configuration
from blue_hub.transmission import (
    TransmissionLossSpec,
    build_piecewise_transmission,
    calculate_transmission_flow,
)

ROOT = Path(__file__).resolve().parents[1]


def test_round_trip_efficiency_is_split_symmetrically() -> None:
    params = load_parameters(ROOT / "configs/technology_parameters.csv")
    config = load_system_configuration(ROOT / "configs/s1_battery_100mw_400mwh.yaml")
    specification = battery_spec_from_parameters(config, params)
    assert specification.charge_efficiency == pytest.approx(sqrt(0.85))
    assert specification.discharge_efficiency == pytest.approx(sqrt(0.85))
    assert specification.charge_efficiency * specification.discharge_efficiency == pytest.approx(
        0.85
    )


def test_battery_loss_ledger_closes_energy_identity() -> None:
    params = load_parameters(ROOT / "configs/technology_parameters.csv")
    config = load_system_configuration(ROOT / "configs/s1_battery_100mw_400mwh.yaml")
    specification = battery_spec_from_parameters(config, params)
    energy_start = np.array([200.0])
    charge = np.array([50.0])
    discharge = np.array([0.0])
    losses = calculate_battery_losses(energy_start, charge, discharge, specification, 1.0)
    energy_end = (
        (1.0 - specification.self_discharge_per_hour) * energy_start
        + specification.charge_efficiency * charge
        - discharge / specification.discharge_efficiency
    )
    np.testing.assert_allclose(
        charge - discharge, energy_end - energy_start + losses.total_loss_mwh
    )


def test_piecewise_delivery_is_exact_at_breakpoints_and_conservative_between() -> None:
    specification = TransmissionLossSpec(0.03, 0.015)
    piecewise = build_piecewise_transmission(700.0, 100.0, specification, 8)
    reconstructed = np.concatenate(
        ([0.0], np.cumsum(piecewise.segment_widths_mw * piecewise.land_delivery_slopes))
    )
    exact = calculate_transmission_flow(
        piecewise.breakpoints_mw, 700.0, 100.0, specification
    ).land_mw
    np.testing.assert_allclose(reconstructed, exact)
    midpoint = 0.5 * (piecewise.breakpoints_mw[:-1] + piecewise.breakpoints_mw[1:])
    exact_midpoint = calculate_transmission_flow(midpoint, 700.0, 100.0, specification).land_mw
    chord_midpoint = reconstructed[:-1] + 0.5 * (
        piecewise.segment_widths_mw * piecewise.land_delivery_slopes
    )
    assert np.all(chord_midpoint <= exact_midpoint + 1e-12)
