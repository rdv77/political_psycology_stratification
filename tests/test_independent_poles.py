"""Per-property provenance and missingness regressions, with no external calls."""
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.semantic_service.independent_poles import apply_fact_review, attach_reviewed_sources, rate_independent_object, score_independent_dossier, variables
from src.semantic_service.store import StudyStore, ratings_basis, validate_scales
from scripts.apply_rwanda_independent_audit import apply_decisions


class IndependentMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = StudyStore(Path(self.temp.name) / "studies")
        self.study = self.store.create({"title": "Synthetic independent evidence", "target_count": 12,
                                       "measurement_model": "independent_poles"})
        self.study["scales"] = [{"id": "s", "name": "Synthetic pair", "positive": "A", "negative": "B", "definition": "Synthetic rubric"}]
        self.obj = {"id": "o", "name": "Synthetic actor"}
        self.dossier = {"basis": "synthetic-evidence-v1", "facts": [{"text": "Explicit manifestation of A",
            "source_url": "https://example.invalid/direct", "period": "2024-02-03", "event_id": "event-one"}]}
        self.reply = [self.score("s__positive", 2, "positive_manifestation", [0]),
                      self.score("s__negative", None, "unknown", [])]
        def request(_id, task, _prompt, _schema, **_kwargs):
            return {"data": {"ratings": copy.deepcopy(self.reply)}, "cache_key": "synthetic-only", "task": task,
                    "model": "synthetic-never-sent", "at": "2026-10-15T12:00:00Z"}
        self.provider = SimpleNamespace(store=self.store, request=request)

    @staticmethod
    def score(identity, value, support, indices):
        return {"variable_id": identity, "status": "unknown" if value is None else "observed", "value": value,
                "support_type": support, "confidence": .8, "rationale": "Synthetic evidence link", "evidence_indices": indices}

    def run_score(self):
        return score_independent_dossier(self.provider, self.study, self.obj, self.dossier)[0]

    def test_known_property_survives_unknown_antipode(self):
        row = self.run_score()
        self.assertEqual(row["status"], "partial")
        self.assertEqual(row["pole_scores"]["positive"]["value"], 2)
        self.assertIsNone(row["pole_scores"]["negative"]["value"])
        self.assertIsNone(row["difference"])
        self.assertEqual(row["pole_scores"]["positive"]["source_urls"], [self.dossier["facts"][0]["source_url"]])

    def test_zero_requires_explicit_absence_and_eligible_source(self):
        self.reply[0].update(value=0)
        self.assertEqual(self.run_score()["pole_scores"]["positive"]["status"], "unknown")
        self.reply[0]["support_type"] = "explicit_absence"
        self.assertEqual(self.run_score()["pole_scores"]["positive"]["value"], 0)
        self.reply[0]["evidence_indices"] = [99]
        self.assertEqual(self.run_score()["pole_scores"]["positive"]["status"], "unknown")

    def test_foreign_ids_are_format_errors_not_unknown_measurements(self):
        self.reply[1]["variable_id"] = "foreign"
        with self.assertRaisesRegex(ValueError, "foreign, missing or duplicated"):
            self.run_score()

    def test_context_only_russia_fact_never_authorizes_a_score(self):
        self.study["scales"][0].update(research_domain="russia_relations")
        self.dossier["facts"][0].update(topic_ids=["s"], evidence_origin="assistant_verified_source", measurement_status="context_only")
        self.provider.request = lambda *_args, **_kwargs: self.fail("Context-only evidence triggered a scoring request")
        row = self.run_score()
        self.assertEqual(row["status"], "unknown")

    def test_child_variable_topic_reaches_its_thematic_rating_group(self):
        self.study["scales"][0].update(research_domain="russia_relations")
        self.dossier["facts"][0].update(topic_ids=["s__positive"], evidence_origin="assistant_verified_source")
        row = self.run_score()
        self.assertEqual(row["pole_scores"]["positive"]["value"], 2)
        self.assertEqual(row["pole_scores"]["positive"]["evidence_indices"], [0])
        self.assertIsNone(row["pole_scores"]["negative"]["value"])

    def reviewed_completion_fixture(self):
        self.provider.root = Path(self.temp.name)
        fact = dict(self.dossier["facts"][0], actor_id=self.obj["id"], eligibility="eligible_direct",
                    topic_ids=["s"], capacity="Synthetic actor's own public action", channel="public_statement")
        rule = {"object_id": self.obj["id"], "source_url": fact["source_url"],
                "text_sha256": hashlib.sha256(fact["text"].encode()).hexdigest(), "excluded": True}
        config = {"retrieval_rounds": 0}
        for key, filename, data in (("completion_source_supplement", "completion.json", {"facts": [fact]}),
                                    ("fact_review_registry", "review.json", {"facts": [rule]})):
            path = self.provider.root / filename
            path.write_text(json.dumps(data), encoding="utf-8")
            config[key] = {"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.study["independent_evidence"] = config
        self.provider.request = lambda *_args, **_kwargs: self.fail("Excluded completion fact triggered a rating request")
        return {"object_id": self.obj["id"], "basis": "empty-base", "facts": [], "sources": [],
                "period": [self.study["date_from"], self.study["date_to"]]}

    def test_completion_source_is_reviewed_before_service_scoring(self):
        base = self.reviewed_completion_fixture()
        with patch("src.semantic_service.independent_poles.assemble_dossier", return_value=base):
            rows = rate_independent_object(self.provider, self.study, self.obj)
        self.assertEqual(rows[0]["status"], "unknown")
        dossier = json.loads((self.store.directory(self.study["id"]) / rows[0]["evidence_file"]).read_text(encoding="utf-8"))
        self.assertEqual(dossier["facts"][0]["measurement_status"], "excluded_known_source_issue")

    def test_third_reviewed_layer_is_idempotent_and_preserves_exclusions_and_original_dates(self):
        base = self.reviewed_completion_fixture()
        config = self.study["independent_evidence"]
        config["coverage_expansion_source_supplement"] = config.pop("completion_source_supplement")
        fact = json.loads((self.provider.root / "completion.json").read_text(encoding="utf-8"))["facts"][0]
        base["facts"] = [dict(fact, evidence_origin="retrieval_cited_not_independently_verified", topic_ids=[])]
        first = attach_reviewed_sources(self.provider, self.study, self.obj, base, "coverage_expansion_source_supplement")
        repeated = attach_reviewed_sources(self.provider, self.study, self.obj, first, "coverage_expansion_source_supplement")
        self.assertEqual(first, repeated)
        self.assertEqual(len(first["facts"]), 1)
        self.assertEqual(first["facts"][0]["evidence_origin"], "assistant_verified_source")
        self.assertEqual(first["facts"][0]["topic_ids"], ["s"])
        registry_path = self.provider.root / "review.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry["facts"][0].update(corrected_period="2024-02-04", canonical_event_id="canonical-event")
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        config["fact_review_registry"]["sha256"] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        reviewed = apply_fact_review(self.provider, self.study, self.obj, first)
        rereviewed = apply_fact_review(self.provider, self.study, self.obj, reviewed)
        self.assertEqual(reviewed, rereviewed)
        self.assertEqual(reviewed["facts"][0]["original_period"], fact["period"])
        self.assertEqual(reviewed["facts"][0]["original_event_id"], fact["event_id"])
        self.assertEqual(reviewed["facts"][0]["measurement_status"], "excluded_known_source_issue")

    def test_completion_source_is_reviewed_before_cli_scoring(self):
        from scripts.run_rwanda_independent_poles import run_phase
        base = self.reviewed_completion_fixture()
        self.study.update(objects=[self.obj], measurement_protocol={"model": "independent_poles"})
        self.store.save(self.study)
        with patch("scripts.run_rwanda_independent_poles.assemble_dossier", return_value=base), \
             patch("scripts.run_rwanda_independent_poles.enrich_independent_dossier", return_value=base), \
             patch("builtins.print"):
            run_phase(self.store, self.provider, self.study, "score", 1)
        rows = json.loads((self.store.directory(self.study["id"]) / "ratings.json").read_text(encoding="utf-8"))
        self.assertEqual(rows[0]["status"], "unknown")
        self.assertIsNone(rows[0]["pole_scores"]["positive"]["value"])

    def test_single_property_uses_its_name_and_has_no_required_antipode(self):
        self.study["scales"][0].update(kind="unipolar", name="Single property")
        self.assertEqual(variables(self.study["scales"])[0]["name"], "Single property")
        self.reply = [self.score("s", 2, "positive_manifestation", [0])]
        row = self.run_score()
        self.assertEqual(row["value"], 2)
        self.assertNotIn("negative", row["pole_scores"])

    def test_measurement_change_invalidates_reusable_ratings(self):
        original = ratings_basis(self.study)
        self.study["measurement_model"] = "paired_difference"
        self.assertNotEqual(original, ratings_basis(self.study))
        scales = [{"id": f"s{i}", "name": f"Property {i}", "definition": "Definition", "kind": "unipolar"} for i in range(6)]
        self.assertEqual(len(validate_scales(scales)), 6)

    def test_review_recode_preserves_original_and_cannot_fill_unknown(self):
        row = self.run_score()
        self.study["objects"] = [self.obj]
        decision = {"object_id": "o", "scale_id": "s", "pole": "positive", "decision": "recode",
                    "value": 1, "reason": "One narrow direct episode", "evidence_indices": [0]}
        report = {"checked_objects": ["o"], "checked_observed_properties": 1, "decisions": [decision]}
        rows, _ = apply_decisions(self.study, [row], [report], lambda _row: self.dossier["facts"])
        entry = rows[0]["pole_scores"]["positive"]
        self.assertEqual(entry["value"], 1)
        self.assertEqual(entry["original_provider_score"]["value"], 2)
        self.assertIsNone(rows[0]["pole_scores"]["negative"]["value"])
        decision["pole"] = "negative"
        with self.assertRaisesRegex(ValueError, "cannot create"):
            apply_decisions(self.study, [row], [report], lambda _row: self.dossier["facts"])

    def test_review_cannot_assign_unsupported_zero(self):
        self.study["objects"] = [self.obj]
        report = {"checked_objects": ["o"], "checked_observed_properties": 1, "decisions": [
            {"object_id": "o", "scale_id": "s", "pole": "positive", "decision": "recode", "value": 0,
             "reason": "Invalid zero", "evidence_indices": [0]}]}
        with self.assertRaisesRegex(ValueError, "explicit absence"):
            apply_decisions(self.study, [self.run_score()], [report], lambda _row: self.dossier["facts"])

    def test_adding_single_property_switches_edited_study_to_independent_model(self):
        study = self.store.create({"title": "Legacy creation", "target_count": 12})
        scales = [{"id": f"s{i}", "name": f"Property {i}", "definition": "Definition", "kind": "unipolar"} for i in range(6)]
        result = self.store.edit(study["id"], "scales", scales)
        self.assertEqual(result["measurement_model"], "independent_poles")
        self.assertFalse(result["approvals"]["scales"]["approved"])

    def test_known_bad_fact_cannot_reappear_in_future_scoring(self):
        self.provider.root = Path(self.temp.name)
        path = self.provider.root / "review.json"
        rule = {"object_id": "o", "source_url": self.dossier["facts"][0]["source_url"],
                "text_sha256": hashlib.sha256(self.dossier["facts"][0]["text"].encode()).hexdigest(),
                "excluded": True, "reasons": ["Actually outside the study period"]}
        path.write_text(json.dumps({"facts": [rule]}), encoding="utf-8")
        self.study["independent_evidence"] = {"fact_review_registry": {"path": "review.json", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
        reviewed = apply_fact_review(self.provider, self.study, self.obj, self.dossier)
        self.assertNotIn("measurement_status", self.dossier["facts"][0])
        self.assertEqual(reviewed["facts"][0]["measurement_status"], "excluded_known_source_issue")
        self.provider.request = lambda *_args, **_kwargs: self.fail("Known bad source triggered a rating request")
        row = score_independent_dossier(self.provider, self.study, self.obj, reviewed)[0]
        self.assertEqual(row["status"], "unknown")
        path.write_text('{"facts": []}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "registry changed"):
            apply_fact_review(self.provider, self.study, self.obj, self.dossier)


if __name__ == "__main__":
    unittest.main()
