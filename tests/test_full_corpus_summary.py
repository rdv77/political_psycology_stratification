"""Synthetic full-census summaries; no API, credentials or real source text."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts.summarize_full_corpus_pilot import build_summary, summarize
from src.corpus_pipeline.models import DEFAULT_PRICING
from src.corpus_pipeline.service import chunks


class FullCorpusSummaryTests(unittest.TestCase):
    def fixture(self):
        docs = [{"id": "d1", "text": "Rwanda " + "x" * 10000, "status": "unique"},
                {"id": "d2", "text": "Rwanda unrelated article", "status": "unique"},
                {"id": "d3", "text": "Rwanda unrelated article", "status": "exact_duplicate", "duplicate_of": "d2"},
                {"id": "d4", "text": "Country not matched", "status": "unique"},
                {"id": "d5", "text": "Rwanda uncertain identity", "status": "unique"}]
        for d in docs:
            d.update(published_at="2026-10-16", metadata={"author": "PRIVATE_AUTHOR"}, url="https://private.invalid/source")
        config = {"objects": [{"id": "alpha", "name": "Alpha North"}, {"id": "beta", "name": "Beta South"}],
                  "country_terms": ["Rwanda"], "date_from": "2021-10-15", "date_to": "2026-10-17",
                  "pricing": DEFAULT_PRICING}
        manifest = {"version": "full-v2", "corpus_id": "corpus-test", "source_sha256": "same-source",
                    "start_spent_usd": .1, "budget_cap_usd": 5.1, "starting_job_ids": ["old"],
                    "population": {"raw": 5, "unique": 4, "eligible_unique": 3, "rule_exclusions": 1}}
        jobs = []
        for d in (docs[0], docs[1], docs[4]):
            for start, end, text in chunks(d["text"]):
                relevant = d["id"] == "d1" and start > 0
                status = "review" if d["id"] == "d5" else "done"
                result = {"status": "review" if status == "review" else "relevant" if relevant else "irrelevant",
                          "language": "fr" if relevant else "en", "actor_ids": ["alpha"] if relevant else [],
                          "reason": "PRIVATE_REASON"}
                jobs.append(self.job("r-" + d["id"] + str(start), "relevance", d["id"], start, end, status, result))
                if relevant:
                    jobs.append(self.job("extract", "extract", d["id"], start, end, "done",
                        {"facts": [{"quote": "PRIVATE_QUOTE"}], "quality_control": {"input_facts": 3,
                         "skipped_facts": 1, "downgraded_facts": 1, "cleared_dates": 1,
                         "issues": [{"index": 0, "code": "date_year_missing"}]},
                         "raw_model_result": {"quote": "PRIVATE_RAW_RESULT"}}))
        jobs.append(self.job("old", "relevance", "d1", 0, 9000, "done", {"status": "relevant", "language": "en"}, active=False, cost=.1))
        jobs.append(self.job("describe", "describe", "d1", 0, 0, "done", {"scales": [],
                    "quality_control": {"input_scales": 2, "skipped_scales": 1,
                    "issues": [{"index": 0, "code": "frequency_in_intensity_anchor"}]}}))
        facts = [{"id": "f1", "actor_id": "alpha", "document_id": "d1", "kind": "public_statement",
                  "attribution_basis": "direct_statement", "event_date": "2026-10-13", "extraction_job_id": "extract",
                  "quote": "PRIVATE_QUOTE", "source_url": "https://private.invalid/fact"}]
        results = {"manifest": copy.deepcopy(manifest), "corpus": {"id": "corpus-test", "source_sha256": "same-source",
                   "config": config, "profile": {"rows_total": 5}}, "jobs": jobs, "facts": facts}
        return manifest, results, docs, []

    def job(self, identifier, stage, document, start, end, status, result, active=True, cost=.01):
        return {"id": identifier, "stage": stage, "status": status, "active_recipe": active,
                "context": {"document": {"id": document, "chunk_start": start, "chunk_end": end}},
                "result": result, "payload": {"model": "gpt-4.1-mini"}, "input_tokens": 100,
                "output_tokens": 20, "cost_usd": cost, "reserved_usd": 0}

    def test_whole_document_funnel_includes_copies_and_rule_exclusions(self):
        summary = build_summary(*self.fixture())
        measured = summary["public_aggregate"]["measured"]
        self.assertEqual(measured["whole_document_funnel"], {"exact_duplicate": 1, "empty": 0, "rule_excluded": 1,
                         "relevant": 1, "review": 1, "irrelevant": 1, "incomplete": 0})
        self.assertEqual(sum(measured["whole_document_funnel"].values()), 5)
        self.assertEqual(measured["whole_document_completion"]["relevance_complete_documents"], 3)
        self.assertEqual(measured["whole_document_completion"]["extraction_complete_documents"], 1)
        self.assertEqual(measured["languages"]["document_labels"]["mixed"], 1)
        self.assertEqual(measured["languages"]["dominant_by_longest_classified_chunk"]["en"], 3)

    def test_first_irrelevant_chunk_does_not_hide_later_relevant_chunk(self):
        m, r, d, s = self.fixture()
        summary = build_summary(m, r, d, s)
        diagnostic = next(x for x in summary["document_diagnostics"] if x["document_id"] == "d1")
        self.assertEqual(diagnostic["status"], "relevant")
        self.assertTrue(diagnostic["relevance_complete"])
        self.assertEqual(diagnostic["expected_relevance_chunks"], 2)

    def test_missing_chunk_is_incomplete_and_forecast_is_not_zero(self):
        m, r, d, s = self.fixture()
        r["jobs"] = [j for j in r["jobs"] if j["id"] != "r-d10"]
        result = build_summary(m, r, d, s)
        diagnostic = next(x for x in result["document_diagnostics"] if x["document_id"] == "d1")
        self.assertEqual(diagnostic["status"], "relevant")
        self.assertFalse(diagnostic["relevance_complete"])
        stage = result["public_aggregate"]["forecast"]["stage_costs"]["relevance"]
        self.assertFalse(stage["complete"])
        self.assertIsNone(stage["full_32000_observed_cache_usd"])
        self.assertGreater(stage["partial_incurred_scaled_to_32000_usd"], 0)

    def test_pending_running_uncertain_prevent_complete_cost_projection(self):
        for status in ("pending", "running", "uncertain"):
            m, r, d, s = self.fixture()
            job = next(j for j in r["jobs"] if j["id"] == "extract")
            job.update(status=status, result=None, input_tokens=0, output_tokens=0, cost_usd=0, reserved_usd=.2)
            stage = build_summary(m, r, d, s)["public_aggregate"]["forecast"]["stage_costs"]["extract"]
            self.assertFalse(stage["complete"])
            self.assertIsNone(stage["full_32000_uncached_usd"])

    def test_quality_control_facts_and_zero_roster_coverage(self):
        measured = build_summary(*self.fixture())["public_aggregate"]["measured"]
        self.assertEqual(measured["quality_control"]["extract"]["totals"]["cleared_dates"], 1)
        self.assertEqual(measured["quality_control"]["describe"]["issue_counts"]["frequency_in_intensity_anchor"], 1)
        self.assertEqual(measured["accepted_facts"]["total"], 1)
        self.assertEqual(measured["accepted_facts"]["event_date_known"], 1)
        self.assertEqual(measured["actor_coverage"][0]["own_facts"], 1)
        self.assertEqual(measured["zero_coverage"], [{"actor_id": "beta", "name": "Beta South"}])

    def test_candidate_editorial_families_are_not_event_counts(self):
        m, r, d, s = self.fixture()
        next(j for j in r["jobs"] if j["id"] == "r-d20")["result"]["status"] = "relevant"
        r["jobs"].append(self.job("extract-d2", "extract", "d2", 0, len(d[1]["text"]), "done", {"facts": []}))
        r["facts"].append(dict(r["facts"][0], id="f2", document_id="d2", extraction_job_id="extract-d2",
                               kind="attributed_image", attribution_basis="third_party_claim"))
        s.append({"left_id": "d1", "right_id": "d2", "similarity": .9})
        actor = build_summary(m, r, d, s)["public_aggregate"]["measured"]["actor_coverage"][0]
        self.assertEqual(actor["unique_documents"], 2)
        self.assertEqual(actor["candidate_editorial_families"], 1)
        self.assertNotIn("independent_episodes", actor)

    def test_ledger_includes_historic_inactive_cost_new_delta_and_reservations(self):
        m, r, d, s = self.fixture()
        r["jobs"].append(self.job("uncertain-old", "relevance", "d2", 0, 25, "uncertain", None, active=False, cost=0))
        r["jobs"][-1]["reserved_usd"] = .3
        measured = build_summary(m, r, d, s)["public_aggregate"]["measured"]
        book = measured["ledger"]
        self.assertAlmostEqual(book["all_historical_and_new"]["outstanding_reserved_usd"], .3)
        self.assertAlmostEqual(book["all_historical_and_new"]["inactive_recorded_cost_usd"], .1)
        self.assertAlmostEqual(book["new_recorded_cost_delta_usd"], book["new_job_receipts"]["recorded_cost_usd"])

    def test_public_allowlist_never_copies_source_or_account_content(self):
        result = build_summary(*self.fixture())
        public = json.dumps(result["public_aggregate"], ensure_ascii=False)
        for secret in ("PRIVATE_", "private.invalid", '"document_id"', '"quote"', '"author"', '"payload"', '"context"'):
            self.assertNotIn(secret, public)
        self.assertIn("Alpha North", public)
        self.assertFalse(result["public_aggregate"]["forecast"]["full_research_scores_estimated"])

    def test_snapshot_identity_and_all_record_count_are_required(self):
        m, r, d, s = self.fixture()
        r["corpus"]["source_sha256"] = "changed"
        with self.assertRaises(ValueError):
            build_summary(m, r, d, s)
        m, r, d, s = self.fixture()
        with self.assertRaises(ValueError):
            build_summary(m, r, d[:-1], s)

    def test_fact_cannot_borrow_another_documents_extraction_job(self):
        m, r, d, s = self.fixture()
        r["facts"][0]["document_id"] = "d2"
        with self.assertRaisesRegex(ValueError, "differs from its extraction job"):
            build_summary(m, r, d, s)

    def test_pure_aggregation_does_not_mutate_inputs(self):
        values = self.fixture()
        before = copy.deepcopy(values)
        build_summary(*values)
        self.assertEqual(values, before)

    def test_offline_sqlite_read_only_and_public_private_destinations(self):
        m, r, docs, similar = self.fixture()
        # Managed Windows system TEMP may disallow an atomic replace. Keep the
        # disposable integration fixture inside the ignored writable workspace.
        runtime_parent = Path(__file__).resolve().parents[1] / ".service_runtime"
        runtime_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime_parent) as name:
            root = Path(name)
            runtime = root / ".corpus_runtime"
            folder = runtime / "pilots/corpus-test/full-v2"
            folder.mkdir(parents=True)
            (folder / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
            (folder / "results_private.json").write_text(json.dumps(r), encoding="utf-8")
            database = runtime / "corpus.sqlite3"
            with sqlite3.connect(database) as db:
                db.executescript("CREATE TABLE corpora(id TEXT,source_sha256 TEXT);"
                                 "CREATE TABLE documents(id TEXT,corpus_id TEXT,source_row INTEGER,data TEXT,status TEXT,duplicate_of TEXT);"
                                 "CREATE TABLE similar(corpus_id TEXT,left_id TEXT,right_id TEXT,data TEXT);")
                db.execute("INSERT INTO corpora VALUES(?,?)", ("corpus-test", "same-source"))
                for index, doc in enumerate(docs):
                    db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?)", (doc["id"], "corpus-test", index + 2,
                               json.dumps(doc), doc["status"], doc.get("duplicate_of")))
            db.close()
            old = root / "reports/corpus_pilot_2026_10_17.json"
            old.parent.mkdir()
            old.write_text("UNCHANGED_OLD_REPORT", encoding="utf-8")
            before = database.read_bytes()
            with patch("src.corpus_pipeline.models.OpenAIResponsesHTTP.call", side_effect=AssertionError("No API")):
                result = summarize(root, "corpus-test")
                self.assertFalse(result["published"])
                self.assertTrue((folder / "summary_private.json").exists())
                self.assertFalse((old.parent / "corpus_full_pilot_2026_10_17.json").exists())
                result = summarize(root, "corpus-test", publish=True)
            self.assertTrue(result["published"])
            self.assertEqual(database.read_bytes(), before)
            self.assertEqual(old.read_text(encoding="utf-8"), "UNCHANGED_OLD_REPORT")
            public = (old.parent / "corpus_full_pilot_2026_10_17.json").read_text(encoding="utf-8")
            self.assertNotIn("PRIVATE_", public)
            self.assertIn("results_private_sha256", public)


if __name__ == "__main__":
    unittest.main()
