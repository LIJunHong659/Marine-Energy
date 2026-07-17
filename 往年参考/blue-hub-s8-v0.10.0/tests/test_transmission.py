import numpy as np
import pytest

from blue_hub.transmission import (
    TransmissionLossSpec,
    calculate_transmission_flow,
    optimal_export_send,
)

SPEC = TransmissionLossSpec(
    terminal_loss_fraction=0.03,
    cable_full_load_loss_per_100km=0.015,
)


def test_transmission_loss_identity_and_distance_monotonicity() -> None:
    send = np.array([0.0, 350.0, 700.0])
    near = calculate_transmission_flow(send, 700.0, 100.0, SPEC)
    far = calculate_transmission_flow(send, 700.0, 200.0, SPEC)
    np.testing.assert_allclose(near.send_mw, near.land_mw + near.total_loss_mw)
    assert np.all(far.land_mw <= near.land_mw)
    assert far.land_mw[-1] < near.land_mw[-1]


def test_cable_loss_fraction_increases_with_loading() -> None:
    flow = calculate_transmission_flow(np.array([350.0, 700.0]), 700.0, 100.0, SPEC)
    fractions = flow.cable_loss_mw / flow.send_mw
    assert fractions[1] == pytest.approx(2.0 * fractions[0])


def test_zero_capacity_rejects_positive_export() -> None:
    with pytest.raises(ValueError, match="zero capacity"):
        calculate_transmission_flow(np.array([1.0]), 0.0, 100.0, SPEC)


def test_economic_and_must_take_negative_price_policies() -> None:
    surplus = np.array([100.0])
    capacity = np.array([100.0])
    price = np.array([-10.0])
    economic = optimal_export_send(
        surplus,
        capacity,
        price,
        installed_capacity_mw=100.0,
        distance_km=100.0,
        loss_spec=SPEC,
        variable_cost_cny_per_mwh_send=0.0,
        curtailment_penalty_cny_per_mwh=0.0,
        policy="economic",
    )
    must_take = optimal_export_send(
        surplus,
        capacity,
        price,
        installed_capacity_mw=100.0,
        distance_km=100.0,
        loss_spec=SPEC,
        variable_cost_cny_per_mwh_send=0.0,
        curtailment_penalty_cny_per_mwh=0.0,
        policy="must_take",
    )
    assert economic[0] == 0.0
    assert must_take[0] == 100.0


def test_curtailment_penalty_can_overcome_mild_negative_price() -> None:
    result = optimal_export_send(
        surplus_mw=np.array([100.0]),
        available_capacity_mw=np.array([100.0]),
        electricity_price_cny_per_mwh=np.array([-10.0]),
        installed_capacity_mw=100.0,
        distance_km=100.0,
        loss_spec=SPEC,
        variable_cost_cny_per_mwh_send=0.0,
        curtailment_penalty_cny_per_mwh=20.0,
        policy="economic",
    )
    assert result[0] == 100.0
