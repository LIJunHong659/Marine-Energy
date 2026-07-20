import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

from bluehub_submodules.parameters import (
    default_parameters,
    load_parameters_from_file,
    parameters_from_mapping,
)


class ParameterConfigTests(unittest.TestCase):
    def test_load_parameters_from_yaml_overrides_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "parameters.yaml"
            path.write_text(
                dedent(
                    """
                    case:
                      scenario_type: engineering_base
                      source_capacity_mw: 80.0
                      random_seed: 99

                    time_step_h: 0.5

                    battery:
                      bess_power_mw: 25.0
                      bess_energy_mwh: 100.0
                      roundtrip_efficiency: 0.92

                    power_export:
                      cable_capacity_mw: 800.0
                      grid_accept_max_mw: 600.0

                    compute:
                      compute_power_max_mw: 200.0
                      compute_power_min_mw: 20.0

                    hydrogen:
                      pipe_capacity_kg_per_h: 2500.0

                    marine:
                      flexible_fraction: 0.2
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            params = load_parameters_from_file(path)
            self.assertEqual(params.case.case_id, "common_case_v1")
            self.assertEqual(params.case.scenario_type, "engineering_base")
            self.assertEqual(params.case.source_capacity_mw, 80.0)
            self.assertEqual(params.case.random_seed, 99)
            self.assertEqual(params.time_step_h, 0.5)
            self.assertEqual(params.battery.bess_power_mw, 25.0)
            self.assertEqual(params.battery.bess_energy_mwh, 100.0)
            self.assertEqual(params.battery.roundtrip_efficiency, 0.92)
            self.assertEqual(params.power_export.cable_capacity_mw, 800.0)
            self.assertEqual(params.power_export.grid_accept_max_mw, 600.0)
            self.assertEqual(params.compute.compute_power_max_mw, 200.0)
            self.assertEqual(params.compute.compute_power_min_mw, 20.0)
            self.assertEqual(params.hydrogen.pipe_capacity_kg_per_h, 2500.0)
            self.assertEqual(params.marine.flexible_fraction, 0.2)

    def test_unknown_keys_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parameters_from_mapping({"unknown_section": {}})

    def test_default_case_metadata_is_protocol_case(self) -> None:
        params = default_parameters()
        self.assertEqual(params.case.case_id, "common_case_v1")
        self.assertEqual(params.case.scenario_type, "interface_smoke")
        self.assertEqual(params.case.timezone, "Asia/Shanghai")

    def test_invalid_scenario_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parameters_from_mapping({"case": {"scenario_type": "ad_hoc"}})

    def test_toml_suffix_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_parameters_from_file(Path("parameters.toml"))


if __name__ == "__main__":
    unittest.main()
