import unittest
import numpy as np

from src.corpus_pipeline.native_factor_sensitivity import select_panel, scenario_matrix, fit_fixed_panel
from src.corpus_pipeline.native_mass import bound_changes


class NativeFactorSensitivityTests(unittest.TestCase):
    def test_panel_selection_depends_on_availability_not_intensity(self):
        cells = [{"actor_id": f"a{i:02}", "scale_id": f"s{j}", "lower": 2,
                  "upper": 4, "midpoint_scenario": 3} for i in range(12) for j in range(8)]
        first = select_panel(cells)
        changed = [{**c, "lower": 0, "upper": 5, "midpoint_scenario": 2.5} for c in cells]
        self.assertEqual(first, select_panel(changed))
        self.assertEqual(first["n"], 12)
        self.assertFalse(first["value_based_factor_selection"])
        self.assertTrue(np.all(scenario_matrix(cells, first, "midpoint_scenario") == 3))

    def test_fixed_panel_does_not_silently_drop_constant_variables(self):
        raw = np.ones((12, 8)) * 3
        result = fit_fixed_panel(raw, [f"s{i}" for i in range(8)], 4)
        self.assertEqual(result["status"], "not_fitted_constant_variables_in_fixed_panel")
        self.assertIsNone(result["variance_total"])

    def test_range_change_is_visible_when_point_stays_unidentified(self):
        before = [{"actor_id": "a", "scale_id": "s", "lower": 3, "upper": 4}]
        after = [{"actor_id": "a", "scale_id": "s", "lower": 3, "upper": 5}]
        self.assertEqual(bound_changes(before, after)["substantive_cells"], 1)

    def test_parameter_count_excludes_four_factors_on_six_variables(self):
        raw = np.arange(72, dtype=float).reshape(12, 6)
        self.assertEqual(fit_fixed_panel(raw, [f"s{i}" for i in range(6)], 4)["status"],
                         "not_fitted_parameter_count")
