"""Meaningful synthetic preparation tests; no private corpus or paid calls."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_rwanda_source_scoring import PUBLIC_JSON, PUBLIC_MD, prepare_files
from src.corpus_pipeline.scoring_preparation import filter_eligible_fact, prepare, render_readiness


class ByteEncoding:
    name = "synthetic-byte-encoder"

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))


def fact(identifier, actor="a", quote=None, event_date="2026-01-01", **fields):
    value = {"id": identifier, "actor_id": actor, "document_id": "private-document-" + identifier,
             "quote": quote or "PRIVATE_LITERAL_QUOTE_" + identifier,
             "claim": "PRIVATE_MACHINE_CLAIM", "attribution_evidence": "PRIVATE_ATTRIBUTION " + actor,
             "source_url": "https://private-source.invalid/" + identifier, "author": "PRIVATE_AUTHOR",
             "source_date": "2026-10-17T12:00:00", "event_date": event_date,
             "date_evidence": "PRIVATE_DATE_EVIDENCE" if event_date else None,
             "kind": "public_statement", "attribution_basis": "direct_statement"}
    value.update(fields)
    return value


def snapshot(facts, actors=("a", "b", "c")):
    return {"corpus": {"id": "synthetic", "source_sha256": "a" * 64,
                       "config": {"objects": [{"id": actor, "name": "Approved " + actor} for actor in actors],
                                  "aspect": "public_positions", "date_from": "2021-10-15", "date_to": "2026-10-17"}},
            "facts": facts, "jobs": [{"private_payload": "PRIVATE_JOB"}]}


def dictionary():
    return {"stage_status": "draft", "scales": [
        {"id": "cooperation", "name": "Сотрудничество", "definition": "A transferable property.",
         "anchor0": "Explicit relevant non-expression.", "anchor3": "A clear instance.", "anchor5": "A strong instance."},
        {"id": "guarantees", "name": "Правовые гарантии", "definition": "Another independent property.",
         "anchor0": "Explicit relevant non-expression.", "anchor3": "A clear instance.", "anchor5": "A strong instance."}]}


class ScoringPreparationTests(unittest.TestCase):
    def test_measurement_gate_rejects_image_context_unreviewed_old_and_foreign_actors(self):
        rows = [fact("own"), fact("image", kind="attributed_image", attribution_basis="third_party_claim"),
                fact("context", measurement_status="context_only"), fact("unreviewed", review_status="unreviewed"),
                fact("rejected", review_status="rejected"), fact("old", event_date="2019-01-01"),
                fact("old-reviewed-null", event_date=None, temporal_status="historical_outside_window"),
                fact("alien", actor="alien")]
        result = prepare(snapshot(rows), dictionary())
        self.assertEqual([u["fact_ids"] for u in result["private_units"]], [["own"]])
        self.assertEqual(result["public_readiness"]["counts"]["exclusions"],
                         {"measurement_or_review_ineligible": 6, "unapproved_actor": 1})
        self.assertFalse(filter_eligible_fact(rows[-2], snapshot([])["corpus"]["config"]))

    def test_public_image_is_a_separate_channel(self):
        source = snapshot([fact("own"), fact("image", kind="attributed_image", attribution_basis="third_party_claim")])
        source["corpus"]["config"]["aspect"] = "public_image"
        result = prepare(source, dictionary())
        self.assertEqual(result["private_units"][0]["fact_ids"], ["image"])

    def test_same_quote_date_groups_preserve_every_original_member(self):
        rows = [fact("id1", quote="Cafe\u0301  policy\nnow."),
                fact("id2", quote="Café policy now.", attribution_evidence="OTHER PRIVATE ATTRIBUTION")]
        source, definitions = snapshot(rows), dictionary()
        before_source, before_dictionary = copy.deepcopy(source), copy.deepcopy(definitions)
        result = prepare(source, definitions)
        self.assertEqual(source, before_source)
        self.assertEqual(definitions, before_dictionary)
        unit = result["private_units"][0]
        self.assertEqual(unit["normalized_quote"], "Café policy now.")
        self.assertEqual(unit["member_facts"], rows)
        self.assertEqual(unit["fact_ids"], ["id1", "id2"])
        self.assertEqual(unit["event_date"], "2026-01-01")
        self.assertFalse(unit["episode_identity_verified"])
        self.assertFalse(unit["source_independence_verified"])
        self.assertEqual(result["public_readiness"]["counts"]["literal_duplicate_member_excess"], 1)

    def test_unknown_date_group_never_joins_known_or_different_explicit_dates(self):
        rows = [fact("unknown1", quote="Exact words.", event_date=None),
                fact("unknown2", quote="Exact words.", event_date=None, source_date="2025-01-01"),
                fact("known1", quote="Exact words.", event_date="2026-01-01"),
                fact("known2", quote="Exact words.", event_date="2026-01-02")]
        result = prepare(snapshot(rows), dictionary())
        self.assertEqual(len(result["private_units"]), 3)
        undated = next(unit for unit in result["private_units"] if unit["event_date"] is None)
        self.assertEqual(undated["fact_ids"], ["unknown1", "unknown2"])
        self.assertTrue(all(unit["episode_identity_verified"] is False for unit in result["private_units"]))

    def test_no_semantic_case_punctuation_negation_or_cross_actor_merges(self):
        rows = [fact("exact", quote="We cooperate."), fact("case", quote="we cooperate."),
                fact("negative", quote="We do not cooperate."), fact("punctuation", quote="We cooperate!"),
                fact("translation", quote="Nous coopérons."), fact("other", actor="b", quote="We cooperate.")]
        result = prepare(snapshot(rows), dictionary())
        self.assertEqual(len(result["private_units"]), len(rows))
        self.assertEqual(result["public_readiness"]["counts"]["literal_duplicate_groups"], 0)

    def test_fact_order_does_not_change_groups_ids_hashes_or_safe_outputs(self):
        rows = [fact("id1", quote="Same words."), fact("id2", quote="Same words."), fact("other", actor="b")]
        first = prepare(snapshot(rows), dictionary(), encoding=ByteEncoding())
        shuffled = copy.deepcopy(rows)
        random.Random(16).shuffle(shuffled)
        self.assertEqual(first, prepare(snapshot(shuffled), dictionary(), encoding=ByteEncoding()))

    def test_all_42_actors_and_every_property_have_unscored_null_cells(self):
        actors = tuple("actor-" + str(i) for i in range(42))
        result = prepare(snapshot([fact("only", actor=actors[0])], actors), dictionary())
        public = result["public_readiness"]
        self.assertEqual(len(public["actor_readiness"]), 42)
        self.assertEqual(len(public["cells"]), 84)
        self.assertEqual(public["counts"]["actors_without_eligible_material"], 41)
        self.assertEqual(public["counts"]["observed_cells"], 0)
        self.assertTrue(all(cell["status"] == "not_scored" and cell["value"] is None for cell in public["cells"]))
        self.assertEqual(public["actor_readiness"][-1]["eligible_facts"], 0)
        self.assertEqual(public["actor_readiness"][-1]["factor_readiness"], "not_evaluated_no_scores")

    def test_repeated_same_id_is_not_double_counted_but_conflicting_id_is_rejected(self):
        row = fact("same")
        result = prepare(snapshot([row, copy.deepcopy(row)]), dictionary())
        self.assertEqual(result["public_readiness"]["counts"]["eligible_facts"], 1)
        self.assertEqual(result["public_readiness"]["counts"]["repeated_identical_input_rows"], 1)
        with self.assertRaises(ValueError):
            prepare(snapshot([row, fact("same", actor="b")]), dictionary())

    def test_public_json_and_markdown_never_contain_private_source_fields(self):
        public = prepare(snapshot([fact("PRIVATE_FACT_ID")]), dictionary())["public_readiness"]
        both = json.dumps(public, ensure_ascii=False) + render_readiness(public)
        for forbidden in ("PRIVATE_FACT_ID", "PRIVATE_LITERAL_QUOTE", "PRIVATE_MACHINE_CLAIM", "PRIVATE_ATTRIBUTION",
                          "private-source.invalid", "PRIVATE_AUTHOR", "PRIVATE_DATE_EVIDENCE", "private-document", "PRIVATE_JOB"):
            self.assertNotIn(forbidden, both)
        self.assertEqual(public["api_calls"], 0)
        self.assertFalse(public["provenance"]["queue_modified"])
        self.assertIn("Это не независимые эпизоды", render_readiness(public))

    def test_sizing_counts_hypothetical_pairs_without_manufacturing_a_price(self):
        result = prepare(snapshot([fact("one"), fact("two", actor="b")]), dictionary(), encoding=ByteEncoding())
        sizing = result["public_readiness"]["packet_sizing"]
        self.assertEqual(sizing["one_unit_all_scales"]["hypothetical_packets"], 2)
        self.assertEqual(sizing["one_unit_one_scale"]["hypothetical_packets"], 4)
        self.assertEqual(sizing["one_unit_all_scales"]["hypothetical_property_evaluations"], 4)
        self.assertEqual(sizing["one_unit_all_scales"]["utf8_bytes"], sizing["one_unit_all_scales"]["content_component_tokens"])
        self.assertGreater(sizing["one_unit_one_scale"]["utf8_bytes"], sizing["one_unit_all_scales"]["utf8_bytes"])
        self.assertIsNone(sizing["cost_usd"])
        self.assertIsNone(sizing["output_tokens"])
        self.assertFalse(sizing["paid_requests_prepared"])
        no_encoder = prepare(snapshot([fact("one")]), dictionary())["public_readiness"]["packet_sizing"]
        self.assertIsNone(no_encoder["one_unit_all_scales"]["content_component_tokens"])

    def test_dictionary_requires_explicit_status_unique_ids_and_public_labels(self):
        for update in ({"stage_status": None}, {"scales": []}, {"scales": [{"id": "unsafe", "name": "https://private.invalid"}]}):
            value = dictionary()
            value.update(update)
            with self.assertRaises(ValueError):
                prepare(snapshot([]), value)
        value = dictionary()
        value["scales"][1]["id"] = value["scales"][0]["id"]
        with self.assertRaises(ValueError):
            prepare(snapshot([]), value)

    def test_file_preparation_preserves_sources_db_manifest_and_publishes_only_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / ".corpus_runtime"
            runtime.mkdir()
            sources = {root / "snapshot.json": json.dumps(snapshot([fact("PRIVATE_FACT_ID")])),
                       root / "dictionary.json": json.dumps(dictionary()),
                       runtime / "corpus.sqlite3": "DATABASE MUST REMAIN CLOSED",
                       runtime / "manifest.json": "OLD $70 MANIFEST"}
            for path, text in sources.items():
                path.write_text(text, encoding="utf-8")
            before = {path: path.read_bytes() for path in sources}
            with patch("sqlite3.connect", side_effect=AssertionError("Database must not be opened")), \
                 patch("httpx.Client.send", side_effect=AssertionError("Network must not be used")):
                result = prepare_files(root, root / "snapshot.json", root / "dictionary.json", encoding=ByteEncoding())
                self.assertFalse((root / PUBLIC_JSON).exists())
                directory = Path(result["directory"])
                units = [json.loads(line) for line in (directory / "evidence_units_private.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(units[0]["member_facts"][0]["id"], "PRIVATE_FACT_ID")
                result = prepare_files(root, root / "snapshot.json", root / "dictionary.json", publish=True, encoding=ByteEncoding())
                self.assertTrue(result["published"])
                receipt = json.loads((directory / "preparation_private.json").read_text(encoding="utf-8"))
                for name, expected in receipt["private_artifacts"].items():
                    self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected["sha256"])
                self.assertEqual(receipt["source_snapshot_file_sha256"], hashlib.sha256(before[root / "snapshot.json"]).hexdigest())
            for path, raw in before.items():
                self.assertEqual(path.read_bytes(), raw)
            for relative in (PUBLIC_JSON, PUBLIC_MD):
                self.assertNotIn("PRIVATE_FACT_ID", (root / relative).read_text(encoding="utf-8"))
            self.assertEqual(result["api_calls"], 0)

    def test_private_units_cannot_be_exported_to_version_controlled_or_source_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "snapshot.json"
            definitions = root / "dictionary.json"
            source.write_text(json.dumps(snapshot([fact("one")])), encoding="utf-8")
            definitions.write_text(json.dumps(dictionary()), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_files(root, source, definitions, root / "reports" / "private-leak")
            self.assertFalse((root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
