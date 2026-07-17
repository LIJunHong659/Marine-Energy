import unittest

from bluehub_submodules.power_export import PowerExportParams, evaluate_power_export


class PowerExportTests(unittest.TestCase):
    def test_export_capacity_and_loss(self) -> None:
        params = PowerExportParams(
            cable_capacity_mw=100.0,
            grid_accept_max_mw=80.0,
            cable_loss_fraction=0.10,
            price_power_cny_per_kwh=0.5,
            variable_cost_cny_per_mwh_send=0.0,
        )
        result = evaluate_power_export(120.0, params)
        self.assertEqual(result.exported_power_mw, 80.0)
        self.assertEqual(result.delivered_power_mw, 72.0)
        self.assertEqual(result.lost_power_mw, 8.0)
        self.assertAlmostEqual(result.revenue_cny, 72.0 * 1000.0 * 0.5)
        self.assertTrue(result.violations)

    def test_negative_request_clips_to_zero(self) -> None:
        params = PowerExportParams()
        result = evaluate_power_export(-1.0, params)
        self.assertEqual(result.exported_power_mw, 0.0)
        self.assertIn("P_grid,t must be non-negative.", result.violations)


if __name__ == "__main__":
    unittest.main()

