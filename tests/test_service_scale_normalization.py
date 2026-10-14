"""A separate dictionary review preserves provenance and never approves scales."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.semantic_service.providers import ModelProvider
from src.semantic_service.store import StudyStore


TEMP_PARENT = Path(__file__).resolve().parents[1] / ".service_runtime"


class ScaleNormalizationTests(unittest.TestCase):
    def setUp(self):
        TEMP_PARENT.mkdir(exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="scale-normalization-test-", dir=TEMP_PARENT))
        self.addCleanup(self.cleanup)
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-never-sent", "PPLX_API_KEY": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.dotenv = patch("src.semantic_service.providers.load_dotenv")
        self.dotenv.start()
        self.addCleanup(self.dotenv.stop)
        self.network = patch("src.semantic_service.providers.httpx.Client",
                             side_effect=AssertionError("External calls forbidden in normalization tests"))
        self.client = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.client.assert_not_called)
        self.store = StudyStore(self.root / "studies")
        self.study = self.store.create({"title": "Synthetic dictionary normalization", "aspect": "Публичный стиль", "target_count": 12})
        self.study["objects"] = [{"id": f"o{i}", "name": f"Synthetic object {i}"} for i in range(12)]
        self.provider = ModelProvider(self.root, self.store)
        self.dossier = patch.object(self.provider, "dossier", side_effect=lambda _study, obj: {
            "object_id": obj["id"], "descriptors": ["Synthetic descriptor " + obj["id"]],
            "facts": [{"text": "Synthetic fixture fact", "source_url": "https://fixture.example/source"}],
        })
        self.dossier.start()
        self.addCleanup(self.dossier.stop)

    def cleanup(self):
        target = self.root.resolve()
        if target.parent != TEMP_PARENT.resolve() or not target.name.startswith("scale-normalization-test-"):
            raise AssertionError("Unsafe test cleanup path")
        shutil.rmtree(target)

    @staticmethod
    def proposals():
        initial = [{"id": f"s{i}", "name": f"Synthetic scale {i}", "positive": f"Synthetic A {i}",
                    "negative": f"Synthetic B {i}", "definition": "Synthetic observable rubric",
                    "aliases": [f"Original alias {i}"], "source_object_ids": [f"o{i}"]} for i in range(6)]
        initial[0].update(name="Стойкость позиции", positive="Отстаивает публичную позицию",
                          negative="Слабая выраженность стойкости", aliases=["стойкий", "твёрдый"],
                          source_object_ids=["o0", "o1", "o0", "invented-id"])
        initial.append({"id": "s6", "name": "Твёрдость позиции", "positive": "Стабильно защищает позицию",
                        "negative": "Публично пересматривает позицию", "definition": "Synthetic near-synonym",
                        "aliases": ["твёрдость публичной позиции"], "source_object_ids": ["o2"]})
        normalized = copy.deepcopy(initial[:6])
        normalized[0].update(name="Устойчивость публичной позиции", positive="Последовательно отстаивает публичную позицию",
                             negative="Публично пересматривает собственные позиции",
                             definition="Наблюдаемый контраст; оба свойства могут выражаться независимо",
                             aliases=["Твёрдость позиции"], source_object_ids=["o0", "invented-id"])
        first = {"data": {"scales": initial, "normalization_notes": ["Initial proposal note"]},
                 "model": "proposal-model", "at": "initial-time", "cache_key": "initial-cache"}
        second = {"data": {"scales": normalized, "normalization_notes": ["Merged near synonyms and replaced absence pole"]},
                  "model": "normalizer-model", "at": "normalization-time", "cache_key": "normalization-cache"}
        return first, second

    def proposal_record(self):
        files = list((self.store.directory(self.study["id"]) / "scale_proposals").glob("*.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text(encoding="utf-8"))

    def assert_unapproved(self):
        saved = self.store.get(self.study["id"])
        self.assertEqual(saved["scales"], [])
        self.assertFalse(saved["approvals"]["scales"]["approved"])

    def test_separate_review_preserves_raw_proposals_aliases_sources_and_both_note_sets(self):
        initial, normalized = self.proposals()
        initial_snapshot, normalized_snapshot = copy.deepcopy(initial["data"]), copy.deepcopy(normalized["data"])
        with patch.object(self.provider, "request", side_effect=[initial, normalized]) as request:
            scales, provenance = self.provider.generate_scales(self.study)
        self.assertEqual([call.args[1] for call in request.call_args_list], ["scales", "scales-normalize"])
        self.assertTrue(all(call.kwargs["max_tokens"] == 13000 for call in request.call_args_list))
        self.assertEqual(len(scales), 6)
        merged = scales[0]
        self.assertEqual(merged["negative"], "Публично пересматривает собственные позиции")
        self.assertTrue({"стойкий", "твёрдый", "Твёрдость позиции", "твёрдость публичной позиции", "Стойкость позиции"} <= set(merged["aliases"]))
        self.assertEqual(set(merged["source_object_ids"]), {"o0", "o1", "o2"})
        self.assertEqual(merged["descriptor_object_count"], 3)
        self.assertEqual(merged["source_scale_ids"], ["s0", "s6"])
        self.assertEqual(provenance["initial_proposal"], initial_snapshot)
        self.assertEqual(provenance["normalized_proposal"], normalized_snapshot)
        self.assertEqual(provenance["initial_normalization_notes"], ["Initial proposal note"])
        self.assertEqual(provenance["normalization_notes"], ["Merged near synonyms and replaced absence pole"])
        self.assertEqual(provenance["initial_provenance"]["cache_key"], "initial-cache")
        self.assertEqual(provenance["normalization_provenance"]["cache_key"], "normalization-cache")
        record = self.proposal_record()
        self.assertEqual(record["status"], "normalized_unapproved")
        self.assertEqual(record["initial_proposal"], initial_snapshot)
        self.assertEqual(record["normalized_proposal"], normalized_snapshot)
        self.assertEqual(record["final_scales"], scales)
        prompt = request.call_args_list[1].args[2]
        self.assertIn("Публичный стиль", prompt)
        self.assertIn("ВСЕМ объектам", prompt)
        self.assertIn("Молчание", prompt.replace("молчание", "Молчание"))
        self.assertIn("Не выводи свойства автоматически из биографии", prompt)
        self.assertIn("а не строгими логическими антонимами", prompt)
        self.assert_unapproved()

    def test_invalid_normalized_dictionary_is_rejected_after_the_second_model_stage(self):
        initial, normalized = self.proposals()
        normalized["data"]["scales"] = normalized["data"]["scales"][:5]
        with patch.object(self.provider, "request", side_effect=[initial, normalized]) as request, \
             self.assertRaisesRegex(ValueError, "от 6 до 80"):
            self.provider.generate_scales(self.study)
        self.assertEqual(request.call_count, 2)
        record = self.proposal_record()
        self.assertEqual(record["status"], "normalization_error")
        self.assertEqual(record["initial_proposal"], initial["data"])
        self.assertEqual(len(record["normalized_proposal"]["scales"]), 5)
        self.assert_unapproved()

    def test_normalization_outage_keeps_initial_provenance_and_does_not_approve(self):
        initial, _ = self.proposals()
        with patch.object(self.provider, "request", side_effect=[initial, ValueError("Synthetic normalization outage")]), \
             self.assertRaisesRegex(ValueError, "Synthetic normalization outage"):
            self.provider.generate_scales(self.study)
        record = self.proposal_record()
        self.assertEqual(record["status"], "normalization_error")
        self.assertEqual(record["initial_proposal"], initial["data"])
        self.assertEqual(record["initial_provenance"]["model"], "proposal-model")
        self.assertNotIn("normalized_proposal", record)
        self.assert_unapproved()

    def test_absence_or_silence_cannot_survive_as_an_explicit_pole(self):
        for pole in ("Нет сведений о позиции", "Слабая выраженность свойства", "Молчание по теме"):
            initial, normalized = self.proposals()
            normalized["data"]["scales"][0]["negative"] = pole
            with self.subTest(pole=pole), \
                 patch.object(self.provider, "request", side_effect=[initial, normalized]), \
                 self.assertRaisesRegex(ValueError, "полюс отсутствия"):
                self.provider.generate_scales(self.study)
            self.assertEqual(self.proposal_record()["status"], "normalization_error")
            self.assert_unapproved()


if __name__ == "__main__":
    unittest.main()
