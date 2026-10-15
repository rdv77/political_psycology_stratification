import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.semantic_service.source_supplement import attach_source_supplement
from src.semantic_service.store import fingerprint


class SourceSupplementTests(unittest.TestCase):
    def setUp(self):
        runtime = Path(__file__).resolve().parents[1] / ".service_runtime"
        runtime.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=runtime)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = self.root / "studies/test/evidence"
        self.evidence.mkdir(parents=True)
        self.obj = {"id": "object-1", "name": "Synthetic fixture"}
        self.study = {"date_from": "2021-10-15", "date_to": "2026-10-15",
                      "research_requirements": {"required_scales": [{"id": "diplomacy"}, "education"]}}
        self.raw = {"object_id": "object-1", "basis": "raw-basis", "model": "sonar-test",
                    "retrieval_citations": ["https://original.example/fact"],
                    "facts": [{"text": "Original retrieved fact", "source_url": "https://original.example/fact"}],
                    "sources": [{"title": "Original source", "url": "https://original.example/fact"}]}
        self.original = self.evidence / "object-1.json"
        self.original.write_text(json.dumps(self.raw), encoding="utf-8")
        self.seed = {"objects": [{"object_id": "object-1", "facts": [{
            "text": "Synthetic named statement", "source_url": "https://official.example/event",
            "period": "2023-04-17", "event_id": "fixture-event", "capacity": "Official role in fixture",
            "channel": "personal_public_rhetoric", "topic_ids": ["diplomacy"]}], "sources": [{
            "url": "https://official.example/event", "title": "Official event",
            "verification": "full_page_read", "provenance": {"checked_by": "fixture"}}]}]}
        self.pin()

    def pin(self):
        seed_file = self.root / "seed.json"
        seed_file.write_text(json.dumps(self.seed), encoding="utf-8")
        self.study["source_supplement"] = {"path": "seed.json",
            "sha256": hashlib.sha256(seed_file.read_bytes()).hexdigest()}

    def attach(self, **kwargs):
        return attach_source_supplement(kwargs.get("study", self.study), kwargs.get("obj", self.obj),
                                        kwargs.get("dossier", self.raw), self.root, self.evidence)

    def test_independent_copy_provenance_cache_and_raw_fact_change(self):
        before, original_bytes = copy.deepcopy(self.raw), self.original.read_bytes()
        reviewed = self.attach()
        self.assertEqual(self.raw, before)
        self.assertEqual(self.original.read_bytes(), original_bytes)
        self.assertEqual(reviewed["facts"][0], self.raw["facts"][0])
        addition = reviewed["facts"][-1]
        self.assertEqual(addition["evidence_origin"], "assistant_verified_source")
        self.assertEqual(addition["channel"], "public_statement")
        self.assertEqual(addition["original_channel"], "personal_public_rhetoric")
        self.assertIn("не свидетельство частного", addition["attribution_note"])
        self.assertEqual(addition["source_provenance"]["provenance"], {"checked_by": "fixture"})
        self.assertNotIn("model", addition)
        self.assertEqual(reviewed["model"], "mixed_source_provenance")
        self.assertEqual(reviewed["original_dossier_model"], "sonar-test")
        self.assertEqual(reviewed["retrieval_citations"], before["retrieval_citations"])
        self.assertEqual(reviewed["original_dossier_file"], "evidence/object-1.json")
        self.assertEqual(reviewed["original_dossier_basis"], "raw-basis")
        self.assertEqual(reviewed["evidence_file"], "evidence/reviewed-object-1.json")
        self.assertEqual(reviewed["basis"], fingerprint({"raw_basis": "raw-basis",
            "raw_facts_sha256": fingerprint(before["facts"]),
            "supplement_sha256": self.study["source_supplement"]["sha256"]}))
        saved_file = self.evidence / "reviewed-object-1.json"
        modified_at = saved_file.stat().st_mtime_ns
        self.assertEqual(self.attach(), reviewed)
        self.assertEqual(saved_file.stat().st_mtime_ns, modified_at)
        reviewed["facts"][0]["text"] = "External in-memory mutation"
        self.assertEqual(self.raw, before)
        self.raw["facts"].append({"text": "New raw fact under same raw basis"})
        self.assertNotEqual(self.attach()["basis"], reviewed["basis"])
        self.assertEqual(self.original.read_bytes(), original_bytes)

    def test_rejects_hash_escape_identity_source_topic_and_dates_before_cache(self):
        self.attach()  # A valid cache must not hide later invalid input/configuration.
        valid_study, valid_seed = copy.deepcopy(self.study), copy.deepcopy(self.seed)
        self.study["source_supplement"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.attach()
        for path in ("../outside.json", str((self.root / "seed.json").resolve())):
            self.study = copy.deepcopy(valid_study)
            self.study["source_supplement"]["path"] = path
            with self.assertRaisesRegex(ValueError, "path"):
                self.attach()
        self.study = copy.deepcopy(valid_study)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.attach(dossier=dict(self.raw, object_id="other-object"))
        invalid_values = [("source_url", "https://not-listed.example/invention", "source_url"),
                          ("topic_ids", ["not-a-required-topic"], "topic_ids"),
                          ("period", "2021-10-14", "outside"),
                          ("period", "2023", "ISO"),
                          ("period", "2023-02-30", "day")]
        for field, value, message in invalid_values:
            with self.subTest(field=field, value=value):
                self.seed = copy.deepcopy(valid_seed)
                self.seed["objects"][0]["facts"][0][field] = value
                self.pin()
                with self.assertRaisesRegex(ValueError, message):
                    self.attach()

    def test_index_only_pending_and_unconfigured_or_unmatched_are_noops(self):
        self.seed["objects"][0]["sources"][0]["verification"] = "primary_search_index_excerpt_checked; full page unavailable"
        self.pin()
        reviewed = self.attach()
        self.assertEqual(reviewed["facts"], self.raw["facts"])
        self.assertEqual(reviewed["sources"], self.raw["sources"])
        self.assertEqual(reviewed["source_supplement_facts"], [])
        pending = reviewed["source_supplement_provenance"]["pending_facts"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["status"], "pending")
        self.assertEqual(reviewed["source_supplement_provenance"]["added_fact_count"], 0)
        self.assertEqual(reviewed["source_supplement_provenance"]["sources"], self.seed["objects"][0]["sources"])
        self.assertIs(self.attach(study={}), self.raw)
        self.assertIs(self.attach(obj={"id": "not-in-seed"}), self.raw)
        self.assertFalse((self.evidence / "reviewed-not-in-seed.json").exists())


if __name__ == "__main__":
    unittest.main()
