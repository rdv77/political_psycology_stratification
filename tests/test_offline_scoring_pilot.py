"""Synthetic offline selection checks; no external services or private corpus."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_rwanda_offline_pilot import PUBLIC_JSON, PUBLIC_MD, prepare_files
from src.corpus_pipeline.offline_scoring_pilot import build, render
from src.corpus_pipeline.scoring_preparation import prepare


def fact(identifier, actor="a", **overrides):
    row = {"id": identifier, "actor_id": actor, "document_id": "PRIVATE_DOC_" + identifier,
           "quote": "PRIVATE_LITERAL_QUOTE_" + identifier, "claim": "PRIVATE_CLAIM",
           "source_url": "https://PRIVATE_SOURCE.invalid/", "attribution_evidence": "PRIVATE_ATTRIBUTION",
           "source_date": "2026-10-17", "event_date": None, "kind": "public_statement",
           "attribution_basis": "direct_statement", "event_status": "requires_review"}
    row.update(overrides)
    return row


def dictionary():
    return {"stage_status": "approved", "scales": [{"id": "property", "name": "Утверждённое свойство"}]}


def snapshot(facts, actors=("a", "b", "c"), jobs=None):
    return {"corpus": {"id": "synthetic", "config": {
        "objects": [{"id": actor, "name": "Approved " + actor} for actor in actors],
        "aspect": "public_positions", "date_from": "2021-10-15", "date_to": "2026-10-17"}},
        "facts": facts, "jobs": jobs or []}


def language_job(fact_row, language, **overrides):
    row = {"id": "PRIVATE_JOB_" + fact_row["id"], "unit_id": fact_row["document_id"],
           "stage": "relevance", "status": "done", "active_recipe": True,
           "result": {"language": language}, "context": {"document": {"language_hint": "sw"}}}
    row.update(overrides)
    return row


def run(source, **options):
    definitions = dictionary()
    return build(source, definitions, prepare(source, definitions), **options)


class OfflineScoringPilotTests(unittest.TestCase):
    def test_sample_limit_cannot_silently_remove_actor_first_places(self):
        with self.assertRaises(ValueError):
            run(snapshot([fact("a"), fact("b", "b")]), max_units=1)

    def test_rare_actors_get_first_places_before_dominant_actor_extras(self):
        actors = tuple("actor-" + str(i) for i in range(42))
        rows = [fact("dominant-" + str(i), actors[0]) for i in range(200)]
        rows += [fact("rare-" + str(i), actors[i]) for i in range(1, 35)]
        result = run(snapshot(rows, actors))
        units = result["private_package"]["selected_units"]
        self.assertEqual({unit["actor_id"] for unit in units[:35]}, set(actors[:35]))
        self.assertEqual(sum(unit["actor_id"] == actors[0] for unit in units), 3)
        self.assertEqual(len(units), 37)  # Never pad beyond real material / cap.
        self.assertEqual(len(result["public_summary"]["actor_cards"]), 42)
        self.assertEqual(result["public_summary"]["counts"]["actors_with_material"], 35)

    def test_deterministic_under_shuffled_facts_and_jobs_without_input_mutation(self):
        rows = [fact("many-" + str(i), "a" if i % 2 else "b", event_date=None if i % 3 else "2026-01-01") for i in range(20)]
        jobs = [language_job(row, "en" if i % 2 else "fr") for i, row in enumerate(rows)]
        source = snapshot(rows + [fact("bad", kind="attributed_image", attribution_basis="third_party_claim")], jobs=jobs)
        before = copy.deepcopy(source)
        first = run(source)
        self.assertEqual(source, before)
        random.Random(31).shuffle(source["facts"])
        random.Random(32).shuffle(source["jobs"])
        self.assertEqual(first, run(source))

    def test_languages_use_only_active_done_relevance_and_conflicts_become_unknown(self):
        a, b, c = fact("a"), fact("b", "b"), fact("c", "c")
        jobs = [language_job(a, "rw", result=json.dumps({"language": "rw"})),
                language_job(a, "sw", active_recipe=False), language_job(a, "fr", status="review"),
                language_job(b, "en"), language_job(b, "fr"),
                language_job(c, "sw", stage="extract")]
        result = run(snapshot([a, b, c], jobs=jobs))
        by_actor = {u["actor_id"]: u["selection_features"] for u in result["private_package"]["selected_units"]}
        self.assertIn("language:rw", by_actor["a"])
        self.assertIn("language:unknown", by_actor["b"])
        self.assertIn("language:unknown", by_actor["c"])
        self.assertNotIn("language:sw", str(by_actor))

    def test_unknown_event_date_is_not_replaced_by_publication_date(self):
        source = snapshot([fact("undated", event_date=None, source_date="2026-10-17")])
        result = run(source)
        unit = result["private_package"]["selected_units"][0]
        self.assertIsNone(unit["event_date"])
        self.assertFalse(unit["episode_identity_verified"])
        self.assertIn("date:unknown", unit["selection_features"])
        self.assertIsNone(result["private_package"]["calibration_holdout_split"])

    def test_review_form_does_not_assert_verdicts_or_reuse_mutable_fields(self):
        result = run(snapshot([fact("one"), fact("two", "b"), fact("bad", attribution_basis="uncertain")]))
        forms = [u["review_fields"] for u in result["private_package"]["selected_units"]]
        self.assertTrue(all(f["status"] == "not_reviewed" for f in forms))
        self.assertTrue(all(f["independence_verdict"] is None and f["attribution_verdict"] is None for f in forms))
        forms[0]["related_unit_ids"].append("EXAMPLE_ONLY")
        self.assertEqual(forms[1]["related_unit_ids"], [])
        controls = result["private_package"]["excluded_controls"]
        self.assertIsNone(controls[0]["review_fields"]["semantic_rejection_confirmed"])

    def test_real_stage_specific_unit_id_joins_original_context_document_id(self):
        row = fact("document")
        job = language_job(row, "fr", unit_id="PRIVATE_STAGE_SPECIFIC_UNIT",
                           context={"document": {"id": row["document_id"], "language_hint": "sw"}})
        result = run(snapshot([row], jobs=[job]))
        self.assertIn("language:fr", result["private_package"]["selected_units"][0]["selection_features"])

    def test_russia_is_only_a_lexical_cue_and_rusizi_is_not_a_match(self):
        source = snapshot([fact("russia", quote="Cooperation with Russia.", event_date="2026-01-01"),
                           fact("rusizi", "b", quote="Visit to Rusizi district.")])
        result = run(source)
        by_actor = {u["actor_id"]: u["selection_features"] for u in result["private_package"]["selected_units"]}
        self.assertIn("russia:lexical_candidate", by_actor["a"])
        self.assertNotIn("russia:lexical_candidate", by_actor["b"])
        self.assertFalse(result["public_summary"]["execution_allowed"])
        self.assertEqual(result["public_summary"]["counts"]["observed_scores"], 0)

    def test_excluded_controls_remain_separate_and_foreign_actor_is_not_selected(self):
        rows = [fact("good"), fact("old", "b", temporal_status="historical_outside_window"),
                fact("accusation", "b", kind="attributed_image", attribution_basis="third_party_claim"),
                fact("context", "c", kind="institutional_context", attribution_basis="context"),
                fact("ambiguous", "c", attribution_basis="uncertain"),
                fact("alien", "alien", kind="institutional_context", attribution_basis="context")]
        result = run(snapshot(rows))
        private = result["private_package"]
        self.assertEqual(len(private["selected_units"]), 1)
        self.assertEqual(len(private["excluded_controls"]), 4)
        self.assertTrue(all(row["numeric_scoring_eligible"] is False for row in private["excluded_controls"]))
        self.assertNotIn("alien", {row["fact"]["actor_id"] for row in private["excluded_controls"]})
        self.assertEqual({row["reason_family"] for row in private["excluded_controls"]}, {
            "historical_outside_window", "third_party_image_or_claim", "institutional_context_without_own_position",
            "unresolved_or_foreign_attribution"})

    def test_no_padding_missing_language_or_controls_or_scores(self):
        result = run(snapshot([fact("only")]))
        public = result["public_summary"]
        self.assertEqual(public["counts"]["selected_units"], 1)
        self.assertEqual(public["counts"]["selected_controls"], 0)
        self.assertNotIn("language:fr", [r["feature"] for r in public["feature_coverage"]])
        self.assertTrue(all(row["value"] is None and row["scoring_status"] == "not_scored" for row in public["actor_cards"]))

    def test_public_output_has_no_private_ids_quotes_urls_or_claims(self):
        result = run(snapshot([fact("PRIVATE_FACT"), fact("PRIVATE_BAD", attribution_basis="uncertain")]))
        public = result["public_summary"]
        output = json.dumps(public, ensure_ascii=False) + render(public)
        for forbidden in ("PRIVATE_FACT", "PRIVATE_BAD", "PRIVATE_DOC", "PRIVATE_LITERAL", "PRIVATE_CLAIM",
                          "PRIVATE_ATTRIBUTION", "PRIVATE_SOURCE", "PRIVATE_JOB"):
            self.assertNotIn(forbidden, output)
        for unit in result["private_package"]["selected_units"]:
            self.assertNotIn(unit["unit_id"], output)
        self.assertEqual(public["api_calls"], 0)
        self.assertEqual(public["new_cost_usd"], 0)
        self.assertFalse(public["provenance"]["queue_modified"])

    def test_prepared_material_must_match_exact_dictionary_and_facts(self):
        source, definitions = snapshot([fact("one")]), dictionary()
        prepared = prepare(source, definitions)
        changed = copy.deepcopy(definitions)
        changed["scales"][0]["name"] = "Changed"
        with self.assertRaises(ValueError):
            build(source, changed, prepared)
        prepared["private_units"][0]["event_date"] = "2026-01-01"
        with self.assertRaises(ValueError):
            build(source, definitions, prepared)

    def test_files_are_frozen_reproducible_and_publish_only_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, definitions = root / "snapshot.json", root / "dictionary.json"
            source.write_text(json.dumps(snapshot([fact("PRIVATE_FACT")])), encoding="utf-8")
            definitions.write_text(json.dumps(dictionary()), encoding="utf-8")
            database = root / ".corpus_runtime" / "corpus.sqlite3"
            database.parent.mkdir()
            database.write_bytes(b"PRIVATE_UNCHANGED_DATABASE")
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            with patch("sqlite3.connect", side_effect=AssertionError("Database forbidden")) as database_connect, \
                    patch("httpx.Client.send", side_effect=AssertionError("Network forbidden")) as network_send:
                first = prepare_files(root, source, definitions, publish=True)
                second = prepare_files(root, source, definitions, publish=True)
                database_connect.assert_not_called()
                network_send.assert_not_called()
            directory = Path(first["directory"])
            package = directory / "review_package_private.json"
            timestamp = package.stat().st_mtime_ns
            prepare_files(root, source, definitions, publish=True)
            self.assertEqual(first, second)
            self.assertEqual(timestamp, package.stat().st_mtime_ns)
            self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())
            self.assertIn("PRIVATE_FACT", package.read_text(encoding="utf-8"))
            for relative in (PUBLIC_JSON, PUBLIC_MD):
                self.assertNotIn("PRIVATE_FACT", (root / relative).read_text(encoding="utf-8"))
            package.write_bytes(b"different")
            with self.assertRaises(ValueError):
                prepare_files(root, source, definitions, directory)
            with self.assertRaises(ValueError):
                prepare_files(root, source, definitions, root / "public-leak")


if __name__ == "__main__":
    unittest.main()
