import unittest

from bluehub_submodules.marine_load import MarineLoadParams, evaluate_marine_load


class MarineLoadTests(unittest.TestCase):
    def test_full_service(self) -> None:
        params = MarineLoadParams(
            aux_load_mw=5.0,
            desal_load_mw=3.0,
            equipment_load_mw=2.0,
            flexible_fraction=0.2,
        )
        result = evaluate_marine_load(20.0, params)
        self.assertEqual(result.served_power_mw, 10.0)
        self.assertEqual(result.unmet_power_mw, 0.0)
        self.assertEqual(result.violations, ())

    def test_rigid_shortfall_violation(self) -> None:
        params = MarineLoadParams(
            aux_load_mw=5.0,
            desal_load_mw=3.0,
            equipment_load_mw=2.0,
            flexible_fraction=0.2,
        )
        result = evaluate_marine_load(6.0, params)
        self.assertEqual(result.served_power_mw, 6.0)
        self.assertIn("served marine power is below rigid marine load.", result.violations)


if __name__ == "__main__":
    unittest.main()

