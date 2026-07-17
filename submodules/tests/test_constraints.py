import unittest

from bluehub_submodules.constraints import (
    constraint_terms,
    markdown_objectives_and_constraints,
    objective_terms,
)


class ConstraintDocumentationTests(unittest.TestCase):
    def test_constraints_are_listed(self) -> None:
        constraints = constraint_terms()
        self.assertTrue(any("Power export capacity" in item for item in constraints))
        self.assertTrue(any("Hydrogen production" in item for item in constraints))
        self.assertTrue(any("Integrated balance" in item for item in constraints))

    def test_markdown_renderer(self) -> None:
        text = markdown_objectives_and_constraints()
        self.assertIn("# 目标函数与约束条件", text)
        self.assertIn("## 目标函数", text)
        self.assertIn("## 约束条件", text)
        self.assertTrue(objective_terms())


if __name__ == "__main__":
    unittest.main()

