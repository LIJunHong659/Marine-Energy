import unittest

from bluehub_submodules.hydrogen_output import (
    HydrogenParams,
    evaluate_hydrogen_output,
    hydrogen_production_kg,
)


class HydrogenOutputTests(unittest.TestCase):
    def test_sec_conversion(self) -> None:
        self.assertAlmostEqual(hydrogen_production_kg(57.5, 57.5, 1.0), 1000.0)

    def test_pipeline_shipping_and_storage(self) -> None:
        params = HydrogenParams(
            electrolyzer_power_max_mw=57.5,
            sec_kwh_per_kg=57.5,
            pipe_capacity_kg_per_h=500.0,
            ship_capacity_kg_per_h=300.0,
            storage_max_kg=1000.0,
            pipe_loss_fraction=0.0,
            ship_loss_fraction=0.0,
            price_h2_cny_per_kg=30.0,
            pipe_transport_cost_cny_per_kg=0.0,
            ship_transport_cost_cny_per_kg=0.0,
            electrolyzer_variable_cost_cny_per_kg=0.0,
        )
        result = evaluate_hydrogen_output(57.5, 500.0, 300.0, 100.0, params)
        self.assertAlmostEqual(result.produced_kg, 1000.0)
        self.assertEqual(result.pipe_output_kg, 500.0)
        self.assertEqual(result.ship_output_kg, 300.0)
        self.assertEqual(result.storage_end_kg, 300.0)
        self.assertEqual(result.delivered_kg, 800.0)
        self.assertEqual(result.revenue_cny, 24_000.0)

    def test_storage_capacity_violation_is_reported(self) -> None:
        params = HydrogenParams(
            electrolyzer_power_max_mw=57.5,
            sec_kwh_per_kg=57.5,
            pipe_capacity_kg_per_h=0.0,
            ship_capacity_kg_per_h=0.0,
            storage_max_kg=500.0,
        )
        result = evaluate_hydrogen_output(57.5, 0.0, 0.0, 0.0, params)
        self.assertEqual(result.storage_end_kg, 500.0)
        self.assertIn("H_storage,t exceeds H_storage_max after production.", result.violations)


if __name__ == "__main__":
    unittest.main()

