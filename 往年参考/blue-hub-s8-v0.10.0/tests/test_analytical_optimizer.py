import numpy as np

from blue_hub.transmission import TransmissionLossSpec, optimal_export_send


def test_analytical_dispatch_matches_dense_enumeration() -> None:
    rng = np.random.default_rng(20260712)
    specification = TransmissionLossSpec(0.03, 0.015)
    capacity = 700.0
    distance = 150.0
    variable_cost = 5.0
    curtailment_penalty = 12.0

    for _ in range(50):
        surplus = float(rng.uniform(0.0, 900.0))
        available = float(rng.uniform(0.0, capacity))
        price = float(rng.uniform(-200.0, 800.0))
        limit = min(surplus, available)
        analytical = optimal_export_send(
            surplus_mw=np.array([surplus]),
            available_capacity_mw=np.array([available]),
            electricity_price_cny_per_mwh=np.array([price]),
            installed_capacity_mw=capacity,
            distance_km=distance,
            loss_spec=specification,
            variable_cost_cny_per_mwh_send=variable_cost,
            curtailment_penalty_cny_per_mwh=curtailment_penalty,
            policy="economic",
        )[0]

        grid = np.linspace(0.0, limit, 20001)
        cable_factor = specification.cable_full_load_loss_per_100km * distance / 100.0
        land = (1.0 - specification.terminal_loss_fraction) * grid
        land -= cable_factor * grid**2 / capacity
        objective = price * land - variable_cost * grid - curtailment_penalty * (surplus - grid)
        analytical_land = (1.0 - specification.terminal_loss_fraction) * analytical
        analytical_land -= cable_factor * analytical**2 / capacity
        analytical_objective = (
            price * analytical_land
            - variable_cost * analytical
            - curtailment_penalty * (surplus - analytical)
        )
        assert analytical_objective >= float(objective.max()) - 1e-5
