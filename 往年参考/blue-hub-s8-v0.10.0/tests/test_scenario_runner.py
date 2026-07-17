from pathlib import Path

from blue_hub.loaders import load_scenarios
from blue_hub.scenario_runner import apply_scenario
from blue_hub.synthetic import generate_synthetic_timeseries

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_outage_and_price_event_windows() -> None:
    scenarios = {
        item.scenario_id: item for item in load_scenarios(ROOT / "configs/scenario_matrix.csv")
    }
    frame = generate_synthetic_timeseries(168)

    outage = apply_scenario(frame, scenarios["cable_outage_24h"])
    assert (outage.iloc[72:96]["tx_availability"] == 0.0).all()
    assert (outage.iloc[:72]["tx_availability"] > 0.0).all()

    negative = apply_scenario(frame, scenarios["negative_price_24h"])
    assert (negative.iloc[72:96]["electricity_price"] == -50.0).all()
    assert (negative.iloc[:72]["electricity_price"] >= 0.0).all()

    wind_lull = apply_scenario(frame, scenarios["wind_lull_24h"])
    assert (wind_lull.iloc[72:96]["wind_availability"] == 0.0).all()
    assert (wind_lull.iloc[:72]["wind_availability"] > 0.0).all()

    high_hydrogen_demand = apply_scenario(frame, scenarios["hydrogen_high_demand"])
    assert (high_hydrogen_demand["hydrogen_demand"] == 1_750.0).all()

    high_compute_demand = apply_scenario(frame, scenarios["compute_high_demand"])
    assert (high_compute_demand["rigid_compute_arrival"] == 50.0).all()

    fiber_outage = apply_scenario(frame, scenarios["fiber_outage_24h"])
    assert (fiber_outage.iloc[72:96]["fiber_availability"] == 0.0).all()
    assert (fiber_outage.iloc[:72]["fiber_availability"] == 1.0).all()
