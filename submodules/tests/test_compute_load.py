import unittest

from bluehub_submodules.compute_load import ComputeLoadParams, evaluate_compute_load


class ComputeLoadTests(unittest.TestCase):
    def test_pue_converts_facility_power_to_it_service(self) -> None:
        params = ComputeLoadParams(
            compute_power_max_mw=100.0,
            compute_power_min_mw=5.0,
            pue=1.25,
            fiber_service_capacity_mw_it=100.0,
            price_compute_cny_per_mwh_it=1000.0,
            variable_cost_cny_per_mwh_it=0.0,
        )
        result = evaluate_compute_load(50.0, params)
        self.assertEqual(result.facility_power_mw, 50.0)
        self.assertAlmostEqual(result.it_power_mw, 40.0)
        self.assertAlmostEqual(result.service_mwh_it, 40.0)
        self.assertAlmostEqual(result.revenue_cny, 40_000.0)
        self.assertEqual(result.violations, ())

    def test_fiber_capacity_limits_service(self) -> None:
        params = ComputeLoadParams(
            compute_power_max_mw=100.0,
            compute_power_min_mw=0.0,
            pue=1.0,
            fiber_service_capacity_mw_it=30.0,
        )
        result = evaluate_compute_load(80.0, params)
        self.assertEqual(result.it_power_mw, 30.0)
        self.assertEqual(result.facility_power_mw, 30.0)
        self.assertIn("IT service exceeds fiber service capacity.", result.violations)

    def test_minimum_online_load_violation(self) -> None:
        params = ComputeLoadParams(compute_power_min_mw=10.0, compute_power_max_mw=100.0)
        result = evaluate_compute_load(5.0, params)
        self.assertIn("P_compute,t is below P_compute_min while online.", result.violations)


if __name__ == "__main__":
    unittest.main()

