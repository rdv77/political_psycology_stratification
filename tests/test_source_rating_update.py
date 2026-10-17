from copy import deepcopy
import unittest

from src.corpus_pipeline.scoring_preparation import digest
from src.corpus_pipeline.source_rating_update import build_profiles, factor_admission, STRICT, LOOSE


def dictionary():
    return {"stage_status": "approved", "approved": True,
            "scales": [{"id": "x", "name": "Property", "kind": "unipolar"}, {"id": "y", "name": "Other", "kind": "unipolar"}]}


def case(cid="U01", actor="a", value=3, date="2026-02-12", related=None, **changes):
    row = {"case_id": cid, "actor_id": actor, "sample_kind": "selected_unit", "verdict": "retain",
           "attribution": {"status": "supported"},
           "event_date": {"status": "supported_exact" if date else "publication_only", "date": date,
                          "date_from": None, "date_to": None},
           "event_identity": {"status": "same_event_supported" if related else "single_document_event", "related_cases": related or []},
           "ratings": [{"scale_id": "x", "value": value, "reason_ru": "Specific source-reviewed intensity."}],
           "summary_ru": "Reviewed event.", "capacity_review_ru": "Public capacity."}
    row.update(changes)
    return row


def run(cases, roster=None, scales=None):
    d = scales or dictionary()
    bundle = {"measurement_dictionary_approved": True, "dictionary_content_sha256": digest(d),
              "raw_quotes_published": False, "cases": cases}
    return build_profiles(roster or [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}], d, [bundle])


class SourceRatingUpdateTests(unittest.TestCase):
    def test_all_actors_and_properties_remain_without_fabricated_zeros(self):
        result = run([case()])
        self.assertEqual(len(result["cells"]), 4)
        self.assertEqual([c["value"] for c in result["cells"]], [3, None, None, None])
        self.assertEqual(result["actors"][1]["status"], "insufficient_reviewed_evidence")

    def test_sources_of_one_event_do_not_receive_repeated_weight(self):
        cases = [case("U01", value=5, related=["U02"]), case("U02", value=5), case("U03", value=1, date="2026-03-12")]
        result = run(cases)
        self.assertEqual(result["cells"][0]["value"], 3)
        self.assertEqual(result["cells"][0]["eligible_episode_count"], 2)
        self.assertEqual(result["scenarios"][STRICT]["counts"]["episode_property_values_after_deduplication"], 2)

    def test_transitive_same_event_merge(self):
        result = run([case("U01", related=["U02"]), case("U02", related=["U03"]), case("U03")])
        self.assertEqual(len(result["episodes"]), 1)
        self.assertEqual(result["cells"][0]["eligible_episode_count"], 1)

    def test_within_event_conflict_is_not_averaged(self):
        result = run([case("U01", value=2, related=["U02"]), case("U02", value=4)])
        self.assertIsNone(result["cells"][0]["value"])
        self.assertEqual(result["episodes"][0]["ratings"][0]["status"], "within_episode_conflict")

    def test_unknown_time_only_enters_explicit_sensitivity(self):
        result = run([case(date=None)])
        self.assertIsNone(result["cells"][0]["value"])
        self.assertEqual(result["scenarios"][LOOSE]["cells"][0]["value"], 3)
        self.assertEqual(result["scenarios"][LOOSE]["cells"][0]["distinct_event_count"], 1)

    def test_held_case_never_aggregates_in_either_scenario(self):
        result = run([case(date=None, verdict="hold")])
        for scenario in result["scenarios"].values():
            self.assertEqual(scenario["counts"]["observed_cells"], 0)

    def test_differing_unknown_time_events_need_separation_review(self):
        result = run([case("U01", date=None, value=2), case("U02", date=None, value=4)])
        cell = result["scenarios"][LOOSE]["cells"][0]
        self.assertIsNone(cell["value"])
        self.assertEqual(cell["unknown_reason"], "unresolved_event_independence")

    def test_identical_value_is_invariant_to_unverified_repeat_weights(self):
        result = run([case("U01", date=None), case("U02", date=None)])
        cell = result["scenarios"][LOOSE]["cells"][0]
        self.assertEqual(cell["value"], 3)
        self.assertEqual(cell["aggregation"], "common_value_invariant_to_unverified_event_multiplicity")
        self.assertIsNone(cell["distinct_event_count"])

    def test_conflicting_event_dates_stay_held(self):
        result = run([case("U01", related=["U02"]), case("U02", date="2026-03-12")])
        self.assertEqual(result["episodes"][0]["event_time_status"], "conflicting_reviewed_dates")
        self.assertIsNone(result["cells"][0]["value"])

    def test_cross_actor_merge_and_foreign_dictionary_rejected(self):
        with self.assertRaises(ValueError):
            run([case("U01", related=["U02"]), case("U02", actor="b")])
        unapproved = dictionary()
        unapproved["approved"] = False
        with self.assertRaises(ValueError):
            run([case()], scales=unapproved)

    def test_control_and_nonown_cases_do_not_produce_profiles(self):
        for change in ({"sample_kind": "excluded_control"}, {"attribution": {"status": "not_own"}}):
            result = run([case(**change)])
            self.assertEqual(result["scenarios"][LOOSE]["counts"]["observed_cells"], 0)

    def test_no_legacy_sources_or_quotes_leak_from_roster(self):
        actor = {"id": "a", "name": "A", "sources": [{"url": "https://private.example"}], "private_quote": "secret"}
        result = run([case()], roster=[actor])
        self.assertNotIn("sources", result["actors"][0])
        self.assertNotIn("private_quote", result["actors"][0])

    def test_input_not_mutated(self):
        cases = [case("U01", related=["U02"]), case("U02")]
        original = deepcopy(cases)
        run(cases)
        self.assertEqual(cases, original)

    def test_interval_stays_interval_not_first_day(self):
        row = case()
        row["event_date"] = {"status": "supported_interval", "date": None, "date_from": "2025-10-01", "date_to": "2025-10-31"}
        result = run([row])
        self.assertEqual(result["episodes"][0]["event_date_to"], "2025-10-31")

    def test_boundary_crossing_interval_is_not_clipped_to_force_inclusion(self):
        row = case()
        row["event_date"] = {"status": "supported_interval", "date": None, "date_from": "2021-10-01", "date_to": "2021-10-31"}
        result = run([row])
        self.assertEqual(result["episodes"][0]["event_time_status"], "known_not_fully_inside_window")
        self.assertFalse(result["episodes"][0]["eligible_strict"])
        self.assertFalse(result["episodes"][0]["eligible_unknown_time_sensitivity"])

    def test_unchanged_factor_gate_rejects_sparse_profiles_before_imputation(self):
        study = {"measurement_model": "independent_poles", "objects": [{"id": str(i)} for i in range(42)],
                 "scales": [{"id": "x", "name": "X", "kind": "unipolar"}]}
        cells = [{"actor_id": str(i), "scale_id": "x", "status": "observed", "value": i % 5} for i in range(20)]
        result = factor_admission(study, cells)
        self.assertEqual(result["status"], "not_admissible")
        self.assertEqual(result["diagnostics"]["preprocessing_iterations"][0]["required_objects_per_scale"], 26)
        self.assertFalse(result["imputation_performed"])
        self.assertTrue(all(s["variance_total"] is None for s in result["solutions"]))

    def test_admissible_matrix_is_still_not_a_fitted_factor_solution(self):
        study = {"measurement_model": "independent_poles", "objects": [{"id": str(i)} for i in range(12)],
                 "scales": [{"id": str(i), "kind": "unipolar"} for i in range(6)]}
        cells = [{"actor_id": str(i), "scale_id": str(j), "status": "observed", "value": (i + j) % 5}
                 for i in range(12) for j in range(6)]
        result = factor_admission(study, cells)
        self.assertEqual(result["status"], "admissible_for_parameter_screen")
        self.assertFalse(result["factor_models_estimated"])
        self.assertEqual([s["status"] for s in result["solutions"]],
                         ["requires_model_fit", "requires_model_fit", "fails_necessary_parameter_count_screen"])


if __name__ == "__main__":
    unittest.main()
