import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

from bluehub_submodules.parameters import load_parameters_from_file, parameters_from_mapping


class ParameterConfigTests(unittest.TestCase):
    def test_load_parameters_from_yaml_overrides_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "parameters.yaml"
            path.write_text(
                dedent(
                    """
                    time_step_h: 0.5

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
            self.assertEqual(params.time_step_h, 0.5)
            self.assertEqual(params.power_export.cable_capacity_mw, 800.0)
            self.assertEqual(params.power_export.grid_accept_max_mw, 600.0)
            self.assertEqual(params.compute.compute_power_max_mw, 200.0)
            self.assertEqual(params.compute.compute_power_min_mw, 20.0)
            self.assertEqual(params.hydrogen.pipe_capacity_kg_per_h, 2500.0)
            self.assertEqual(params.marine.flexible_fraction, 0.2)

    def test_unknown_keys_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parameters_from_mapping({"unknown_section": {}})

    def test_toml_suffix_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_parameters_from_file(Path("parameters.toml"))


if __name__ == "__main__":
    unittest.main()
