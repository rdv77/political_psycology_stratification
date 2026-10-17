import copy
import hashlib
import json
import unittest

from src.corpus_pipeline.sampling import SAMPLING_POLICY_VERSION, sample_documents


ACTORS = ["paul-kagame", "alpha-north", "beta-south"]


def document(identifier, *, actors=None, month="2026-10", source="article", language="en", status="ready"):
    return {"id": identifier, "published_at": month + "-17T12:30:00Z",
            "source_type": source, "language_hint": language, "status": status,
            "actor_ids": ["paul-kagame"] if actors is None else actors}


def sample(records, **kwargs):
    return sample_documents(records, approved_actor_ids=ACTORS, **kwargs)


class CorpusSamplingTests(unittest.TestCase):
    def test_exact_half_with_certainty_and_no_input_mutation(self):
        records = [document(str(i)) for i in range(12)]
        records[2]["actor_ids"] = ["alpha-north", "paul-kagame"]
        records[7]["actor_ids"] = ["beta-south"]
        original = copy.deepcopy(records)
        result = sample(records)
        self.assertEqual(result["audit"]["selected_count"], 6)
        self.assertEqual(result["audit"]["certainty_count"], 2)
        self.assertTrue(result["audit"]["exact_half"])
        by_id = {item["id"]: item for item in result["manifest"]["documents"]}
        for identifier in ("2", "7"):
            self.assertTrue(by_id[identifier]["selected"])
            self.assertEqual(by_id[identifier]["inclusion_probability"], 1)
            self.assertEqual(by_id[identifier]["selection_reason"], "certainty_non_primary_actor")
        for item in result["manifest"]["documents"]:
            if not item["protected"]:
                self.assertEqual(item["inclusion_probability"], 4 / 10)
        self.assertEqual(records, original)
        self.assertEqual(result["manifest"]["policy_version"], SAMPLING_POLICY_VERSION)

    def test_seed_hash_and_input_order_independence(self):
        records = [document(f"doc-{i:02d}") for i in range(40)]
        first = sample(records, seed="seed-A")
        self.assertEqual(first, sample(reversed(records), seed="seed-A"))
        expected = sorted((item["id"] for item in records), key=lambda identifier: (
            hashlib.sha256(json.dumps(["seed-A", identifier], ensure_ascii=False,
                                      sort_keys=True, separators=(",", ":")).encode()).hexdigest(), identifier))[:20]
        selected = [item["id"] for item in first["manifest"]["documents"] if item["selected"]]
        self.assertEqual(set(selected), set(expected))
        other = sample(records, seed="seed-B")
        self.assertNotEqual(selected, [item["id"] for item in other["manifest"]["documents"] if item["selected"]])

    def test_hamilton_quotas_and_probabilities_without_minima(self):
        records = ([document(f"en-{i}") for i in range(5)] +
                   [document(f"fr-{i}", language="fr") for i in range(3)] +
                   [document(f"rw-{i}", language="rw") for i in range(2)])
        result = sample(records, target_count=4, min_one_per_stratum=False)
        strata = {item["stratum"]["language_hint"]: item for item in result["audit"]["strata"]}
        self.assertEqual({key: value["residual_quota"] for key, value in strata.items()}, {"en": 2, "fr": 1, "rw": 1})
        self.assertEqual(strata["en"]["residual_inclusion_probability"], 2 / 5)
        for item in result["manifest"]["documents"]:
            stratum = strata[item["stratum"]["language_hint"]]
            self.assertEqual(item["inclusion_probability"], stratum["residual_quota"] / stratum["residual_count"])

    def test_feasible_minima_then_remaining_capacity_hamilton(self):
        records = ([document("singleton", language="rw")] +
                   [document(f"en-{i}") for i in range(9)] +
                   [document(f"fr-{i}", language="fr") for i in range(2)])
        result = sample(records)
        strata = {item["stratum"]["language_hint"]: item for item in result["audit"]["strata"]}
        self.assertTrue(result["audit"]["minimum_one_applied"])
        self.assertEqual({key: value["residual_quota"] for key, value in strata.items()}, {"en": 4, "fr": 1, "rw": 1})
        self.assertEqual(sum(item["selected_count"] for item in strata.values()), 6)

    def test_infeasible_minima_are_explicit_and_no_oversampling(self):
        records = [document(f"{i}", language=f"lang-{i}") for i in range(8)]
        result = sample(records)
        self.assertFalse(result["audit"]["minimum_one_applied"])
        self.assertTrue(result["audit"]["minimum_one_not_feasible"])
        self.assertEqual(result["audit"]["selected_count"], 4)
        self.assertEqual(sorted(item["inclusion_probability"] for item in result["manifest"]["documents"]),
                         [0.0] * 4 + [1.0] * 4)

    def test_unknown_names_and_review_labels_have_separate_strata(self):
        records = [document("named"), document("no-name", actors=[]),
                   document("review", status="needs_semantic_review"),
                   document("review-no-name", actors=[], status="needs_semantic_review")]
        result = sample(records, target_count=4)
        self.assertEqual(len(result["audit"]["strata"]), 4)
        self.assertEqual({item["stratum"]["name_status"] for item in result["audit"]["strata"]},
                         {"matched", "unknown-name"})
        self.assertEqual({item["stratum"]["status"] for item in result["audit"]["strata"]},
                         {"ready", "needs_semantic_review"})

    def test_month_source_language_status_normalization_and_missing_dates(self):
        records = [document("a", month="2026-09", language=" FR ", source=" Article ", status="READY"),
                   document("b", month="2026-09", language="fr", source="article", status="ready"),
                   document("c"), document("d")]
        records[2].update(published_at="2026-02-31", source_type=None, language_hint=None, status=None)
        records[3].update(published_at=None, source_type="", language_hint="", status="")
        result = sample(records)
        self.assertEqual(len(result["audit"]["strata"]), 2)
        self.assertEqual(result["audit"]["date_quality"], {"missing": 1, "parsed": 2, "unparseable": 1})
        self.assertEqual(result["manifest"]["documents"][0]["stratum"]["month"], "2026-09")
        self.assertEqual(result["manifest"]["documents"][2]["stratum"]["month"], "unknown")

    def test_certainty_greater_than_target_raises_clear_error(self):
        records = [document(str(i), actors=["alpha-north"]) for i in range(6)]
        with self.assertRaisesRegex(ValueError, "certainty retention requires 6.*target_count=3"):
            sample(records)
        result = sample(records, target_count=6)
        self.assertEqual(result["audit"]["selected_count"], 6)

    def test_odd_frame_requires_explicit_rounding_choice(self):
        records = [document(str(i)) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "odd-size frame"):
            sample(records)
        result = sample(records, target_count=1)
        self.assertFalse(result["audit"]["exact_half"])
        self.assertEqual(result["audit"]["selected_count"], 1)

    def test_empty_zero_and_full_targets(self):
        empty = sample([])
        self.assertEqual(empty["manifest"]["documents"], [])
        self.assertEqual(empty["audit"]["selected_count"], 0)
        records = [document(str(i)) for i in range(4)]
        for target in (0, 4):
            result = sample(records, target_count=target)
            self.assertEqual(result["audit"]["selected_count"], target)
            self.assertTrue(all(item["inclusion_probability"] == target / 4 for item in result["manifest"]["documents"]))

    def test_duplicate_invalid_and_unapproved_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate document id"):
            sample([document("same"), document("same")])
        for identifier in (None, "", " ", 123):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                sample([document(identifier)], target_count=0)
        for actors in ("alpha-north", ["not-approved"], [None]):
            with self.subTest(actors=actors), self.assertRaises(ValueError):
                sample([document("a", actors=actors)], target_count=0)

    def test_invalid_parameters_and_labels_rejected(self):
        records = [document("a"), document("b")]
        for target in (True, 1.5, -1, 3):
            with self.subTest(target=target), self.assertRaises(ValueError):
                sample(records, target_count=target)
        for kwargs in ({"seed": ""}, {"seed": 42}, {"min_one_per_stratum": "yes"},
                       {"primary_actor_id": "missing"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sample(records, **kwargs)
        for field in ("source_type", "language_hint", "status", "published_at"):
            invalid = copy.deepcopy(records)
            invalid[0][field] = 42
            with self.subTest(field=field), self.assertRaises(ValueError):
                sample(invalid)

    def test_manifest_slim_content_and_unrequested_fields_do_not_affect_selection(self):
        records = [document(str(i)) for i in range(6)]
        before = sample(records)
        for record in records:
            record.update(text="private source body with politics and Russia", length=987654,
                          position="not a selector", url="https://private.invalid")
        after = sample(records)
        self.assertEqual(before, after)
        payload = json.dumps(after)
        for private in ("private source body", "private.invalid", "not a selector", "987654"):
            self.assertNotIn(private, payload)


if __name__ == "__main__":
    unittest.main()
