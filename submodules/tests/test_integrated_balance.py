import unittest

from bluehub_submodules import (
    DispatchRequest,
    ModelParameters,
    ComputeLoadParams,
    HydrogenParams,
    MarineLoadParams,
    PowerExportParams,
    evaluate_integrated_hour,
    simple_greedy_dispatch,
    summarize_results,
)


class IntegratedBalanceTests(unittest.TestCase):
    def test_integrated_hour_closes_power_balance(self) -> None:
        params = ModelParameters(
            power_export=PowerExportParams(
                cable_capacity_mw=100.0,
                grid_accept_max_mw=100.0,
                cable_loss_fraction=0.0,
                price_power_cny_per_kwh=0.5,
                variable_cost_cny_per_mwh_send=0.0,
            ),
            compute=ComputeLoadParams(
                compute_power_max_mw=50.0,
                compute_power_min_mw=0.0,
                pue=1.0,
                fiber_service_capacity_mw_it=50.0,
                price_compute_cny_per_mwh_it=1000.0,
                variable_cost_cny_per_mwh_it=0.0,
            ),
            hydrogen=HydrogenParams(
                electrolyzer_power_max_mw=50.0,
                sec_kwh_per_kg=50.0,
                pipe_capacity_kg_per_h=1000.0,
                ship_capacity_kg_per_h=0.0,
                storage_max_kg=5000.0,
                price_h2_cny_per_kg=10.0,
                pipe_transport_cost_cny_per_kg=0.0,
                electrolyzer_variable_cost_cny_per_kg=0.0,
            ),
            marine=MarineLoadParams(
                aux_load_mw=10.0,
                desal_load_mw=0.0,
                equipment_load_mw=0.0,
                flexible_fraction=0.0,
            ),
        )
        request = DispatchRequest(
            grid_power_mw=80.0,
            compute_power_mw=40.0,
            h2_power_mw=20.0,
            marine_power_mw=10.0,
            h2_pipe_output_kg=400.0,
        )
        result = evaluate_integrated_hour(0, 160.0, 0.0, request, params)
        self.assertAlmostEqual(result.curtailment_mw, 10.0)
        self.assertAlmostEqual(result.offshore_balance_residual_mw, 0.0)
        self.assertEqual(result.violations, ())
        self.assertGreater(result.objective.operating_margin_cny, 0.0)

    def test_simple_greedy_dispatch_runs_24h(self) -> None:
        results = simple_greedy_dispatch()
        summary = summarize_results(results)
        self.assertEqual(len(results), 24)
        self.assertEqual(summary["violation_count"], 0.0)
        self.assertLess(summary["max_abs_balance_residual_mw"], 1e-6)
        self.assertGreater(summary["export_delivered_mwh"], 0.0)
        self.assertGreater(summary["compute_service_mwh_it"], 0.0)


if __name__ == "__main__":
    unittest.main()

