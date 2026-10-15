"""Review receipts must bind the exact approved study definition."""
import tempfile
import unittest
from pathlib import Path

from src.semantic_service.store import StudyStore, fingerprint, validate_scales


class StudyStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = StudyStore(Path(self.directory.name))
        self.study = self.store.create({"title": "Тест", "target_count": 12})
        self.objects = [{"id": f"o{i}", "name": f"Объект {i}", "sources": []} for i in range(12)]
        self.scales = [{"id": f"s{i}", "name": f"Шкала {i}", "positive": f"Полюс {i} А",
                        "negative": f"Полюс {i} Б", "definition": "Наблюдаемое свойство публичного образа"}
                       for i in range(6)]

    def tearDown(self):
        self.directory.cleanup()

    def test_two_gates_and_persistent_receipts(self):
        identifier = self.study["id"]
        with self.assertRaises(ValueError):
            self.store.require_approved(identifier)
        self.store.edit(identifier, "objects", self.objects)
        self.store.approve(identifier, "objects", actor="human_test")
        self.store.edit(identifier, "scales", self.scales)
        with self.assertRaises(ValueError):
            self.store.require_approved(identifier)
        self.store.approve(identifier, "scales", actor="human_test")
        other = StudyStore(Path(self.directory.name))
        result = other.require_approved(identifier)
        self.assertEqual(result["approvals"]["objects"]["actor"], "human_test")

    def test_object_changes_invalidate_scales_and_outputs(self):
        identifier = self.study["id"]
        self.store.edit(identifier, "objects", self.objects)
        self.store.approve(identifier, "objects")
        self.store.edit(identifier, "scales", self.scales)
        self.store.approve(identifier, "scales")
        self.store.update(identifier, analysis={"n": 12}, artifacts=[{"url": "old"}])
        changed = list(self.objects) + [{"id": "extra", "name": "Другой"}]
        current = self.store.edit(identifier, "objects", changed)
        self.assertFalse(current["approvals"]["objects"]["approved"])
        self.assertEqual(current["scales"], [])
        self.assertEqual(current["artifacts"], [])
        self.assertIsNone(current["analysis"])

    def test_no_change_put_preserves_approval(self):
        identifier = self.study["id"]
        approved_objects = self.store.edit(identifier, "objects", self.objects)["objects"]
        self.store.approve(identifier, "objects")
        result = self.store.edit(identifier, "objects", approved_objects)
        self.assertTrue(self.store.is_approved(result, "objects"))

    def test_receipt_cannot_authorize_modified_definition(self):
        identifier = self.study["id"]
        self.store.edit(identifier, "objects", self.objects)
        self.store.approve(identifier, "objects")
        current = self.store.get(identifier)
        current["objects"][0]["name"] = "Подменённый объект"
        self.assertFalse(self.store.is_approved(current, "objects"))

    def test_path_traversal_and_duplicate_scales_rejected(self):
        with self.assertRaises(ValueError):
            self.store.get("../../.env")
        scales = [dict(s) for s in self.scales]
        scales[1].update(positive=scales[0]["negative"], negative=scales[0]["positive"])
        with self.assertRaises(ValueError):
            validate_scales(scales)

    def test_busy_edits_do_not_race_running_job(self):
        identifier = self.study["id"]
        self.store.update(identifier, status="running")
        with self.assertRaises(ValueError):
            self.store.edit(identifier, "objects", self.objects)

    def test_required_topic_cannot_be_removed_or_redefined_before_approval(self):
        identifier = self.study["id"]
        self.store.edit(identifier, "objects", self.objects)
        self.store.approve(identifier, "objects")
        self.store.edit(identifier, "scales", self.scales)
        self.store.update(identifier, research_requirements={"required_scales": [dict(self.scales[0])]})
        changed = [dict(scale) for scale in self.scales]
        changed[0]["definition"] = "An unrelated observable construct"
        self.store.edit(identifier, "scales", changed)
        with self.assertRaisesRegex(ValueError, "обязательной шкалы"):
            self.store.approve(identifier, "scales")
        changed[0] = dict(self.scales[0], id="replacement")
        self.store.edit(identifier, "scales", changed)
        with self.assertRaisesRegex(ValueError, "отсутствует обязательная"):
            self.store.approve(identifier, "scales")
        self.assertFalse(self.store.get(identifier)["approvals"]["scales"]["approved"])


if __name__ == "__main__":
    unittest.main()
