import unittest

from bluehub_submodules import (
    BatteryParams,
    CaseMetadata,
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

    def test_result_carries_case_metadata_and_summary_uses_time_step(self) -> None:
        params = ModelParameters(
            case=CaseMetadata(scenario_type="engineering_base", source_capacity_mw=40.0),
            time_step_h=0.5,
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
            grid_power_mw=10.0,
            compute_power_mw=10.0,
            h2_power_mw=10.0,
            marine_power_mw=10.0,
            h2_pipe_output_kg=100.0,
        )
        result = evaluate_integrated_hour(0, 40.0, 0.0, request, params)
        summary = summarize_results([result])

        self.assertEqual(result.case.case_id, "common_case_v1")
        self.assertEqual(result.case.scenario_type, "engineering_base")
        self.assertEqual(result.time_step_h, 0.5)
        self.assertEqual(result.source_available_power_mw, 40.0)
        self.assertAlmostEqual(summary["hours"], 0.5)
        self.assertAlmostEqual(summary["export_sent_mwh"], 5.0)

    def test_bess_power_request_is_clipped_and_reported(self) -> None:
        params = ModelParameters(
            battery=BatteryParams(bess_power_mw=5.0, bess_energy_mwh=20.0),
            marine=MarineLoadParams(
                aux_load_mw=0.0,
                desal_load_mw=0.0,
                equipment_load_mw=0.0,
                flexible_fraction=0.0,
            ),
        )
        request = DispatchRequest(
            grid_power_mw=0.0,
            compute_power_mw=0.0,
            h2_power_mw=0.0,
            marine_power_mw=0.0,
            storage_discharge_mw=8.0,
        )
        result = evaluate_integrated_hour(0, 0.0, 0.0, request, params)

        self.assertEqual(result.storage_discharge_mw, 5.0)
        self.assertIn("P_BESS_dis exceeds bess_power_mw.", result.violations)
        self.assertAlmostEqual(result.offshore_balance_residual_mw, 0.0)

    def test_simple_greedy_dispatch_runs_24h(self) -> None:
        results = simple_greedy_dispatch()
        summary = summarize_results(results)
        self.assertEqual(len(results), 24)
        self.assertEqual(results[0].case.case_id, "common_case_v1")
        self.assertEqual(summary["violation_count"], 0.0)
        self.assertLess(summary["max_abs_balance_residual_mw"], 1e-6)
        self.assertGreater(summary["export_delivered_mwh"], 0.0)
        self.assertGreater(summary["compute_service_mwh_it"], 0.0)


if __name__ == "__main__":
    unittest.main()
