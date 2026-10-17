"""Synthetic import fixtures; no copied corpus text or personal data."""

import codecs
import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from src.corpus_pipeline.ingest import exact_fingerprint, inspect_csv, iter_documents, near_duplicate_pairs, normalize_text


class CorpusIngestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write_csv(self, columns, rows, delimiter=";", encoding="utf-8-sig"):
        path = self.root / "input.csv"
        with path.open("w", encoding=encoding, newline="") as stream:
            writer = csv.writer(stream, delimiter=delimiter)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def test_russian_bom_semicolon_multiline_ocr_and_empty_records(self):
        columns = ["№", "Дата", "Заголовок", "Сообщение", "Автораспознанный текст", "Ссылка", "Площадка", "Автор", "Id сообщения", "Язык"]
        rows = [["1", "2024-05-20 10:00", "Заголовок", '<p>Он НЕ поддерживает;</p>\n<p>это решение.</p>', "Текст с изображения", "https://example.invalid/1", "Тест", "Автор А", "999999999999999999999999999999", "rw"], [""] * len(columns), ["3", "", "", "<p> </p>", "", "", "", "", "1.23456789012345E+25", ""]]
        path = self.write_csv(columns, rows)
        before = path.read_bytes()
        info = inspect_csv(path)
        self.assertEqual((info["encoding"], info["delimiter"]), ("utf-8-sig", ";"))
        self.assertEqual(info["columns"], columns)
        self.assertEqual((info["rows_total"], info["rows_with_text"], info["empty_rows"]), (3, 1, 1))
        self.assertEqual(info["field_mapping"]["ocr_text"], "Автораспознанный текст")
        self.assertEqual(info["source_sha256"], hashlib.sha256(before).hexdigest())
        self.assertNotIn("это решение", str(info))
        docs = list(iter_documents(path))
        self.assertEqual([x["source_row"] for x in docs], [2, 3, 4])
        self.assertIn("=== TITLE ===\nЗаголовок", docs[0]["text"])
        self.assertIn("=== MESSAGE ===\nОн НЕ поддерживает;", docs[0]["text"])
        self.assertIn("=== OCR ===\nТекст с изображения", docs[0]["text"])
        self.assertIn("<p>", docs[0]["raw_text"])
        self.assertEqual(docs[0]["metadata"]["Сообщение"], rows[0][3])
        self.assertEqual(docs[0]["source_id"], rows[0][8])
        self.assertEqual(docs[2]["source_id"], "1.23456789012345E+25")
        self.assertEqual(docs[1]["text"], "")
        self.assertEqual(docs[1]["source_id"], "")
        self.assertEqual(docs[2]["text"], "")
        self.assertEqual(docs[0]["published_at"], rows[0][1])
        self.assertEqual(docs[0]["language_hint"], "rw")
        self.assertEqual(path.read_bytes(), before)

    def test_english_headers_and_equivalent_sections_not_repeated(self):
        path = self.write_csv(["post_id", "headline", "body", "OCR", "permalink", "author_name", "platform", "language"], [["00000123", "Same &amp; words", "<p>Same &amp; words</p>", "Different OCR", "https://example.invalid", "A", "B", "fr"]], delimiter=",")
        doc = next(iter_documents(path))
        self.assertEqual(doc["text"].count("Same & words"), 1)
        self.assertNotIn("=== MESSAGE ===", doc["text"])
        self.assertEqual(doc["source_id"], "00000123")
        self.assertEqual(doc["url"], "https://example.invalid")
        self.assertEqual(doc["author"], "A")

    def test_explicit_mapping_and_disabled_title(self):
        path = self.write_csv(["identifier", "custom words", "headline"], [["=1+1", "not a formula", "omit"]])
        doc = next(iter_documents(path, {"source_id": "identifier", "text": "custom words", "title": None}))
        self.assertEqual(doc["source_id"], "=1+1")
        self.assertEqual(doc["text"], "=== MESSAGE ===\nnot a formula")
        with self.assertRaises(ValueError):
            list(iter_documents(path, {"message": "absent"}))
        with self.assertRaises(ValueError):
            list(iter_documents(path, {"invented": "headline"}))

    def test_new_russian_export_headers_import_actual_body_dates_links_and_author(self):
        columns = ["№", "Время публикации", "Заголовок", "Текст сообщения", "Распознанный текст",
                   "Ссылка на сообщение", "Площадка", "Кто пишет", "Язык", "Id сообщения"]
        rows = [["1", "17.10.2026\u00a016:54", "", "Полный текст: не поддерживает решение.\nÉconomie عربية", "",
                 "https://example.invalid/post/1", "Test outlet", "Synthetic author", "fr", "1.23456789012345E+25"],
                ["2", "15.10.2026\u00a000:05", "Короткий заголовок", "Самостоятельный полный текст", "OCR дополнение",
                 "https://example.invalid/post/2", "Test outlet", "Synthetic author", "rw", "99999999999999999999999"]]
        path = self.write_csv(columns, rows)
        before = path.read_bytes()
        info = inspect_csv(path)
        expected = {"message": "Текст сообщения", "published_at": "Время публикации",
                    "url": "Ссылка на сообщение", "author": "Кто пишет", "ocr_text": "Распознанный текст"}
        for field, header in expected.items():
            self.assertEqual(info["field_mapping"][field], header)
        self.assertEqual((info["rows_total"], info["rows_with_text"], info["empty_rows"]), (2, 2, 0))
        self.assertEqual(info["warnings"], [])
        docs = list(iter_documents(path))
        self.assertIn("не поддерживает решение", docs[0]["text"])
        self.assertEqual(docs[0]["published_at"], rows[0][1])
        self.assertEqual(docs[0]["url"], rows[0][5])
        self.assertEqual(docs[0]["author"], rows[0][7])
        self.assertEqual(docs[0]["source_id"], rows[0][9])
        self.assertEqual(docs[0]["metadata"]["Текст сообщения"], rows[0][3])
        self.assertIn("=== OCR ===\nOCR дополнение", docs[1]["text"])
        self.assertEqual(path.read_bytes(), before)

    def test_missing_or_empty_main_body_is_not_silent_title_only_success(self):
        path = self.write_csv(["Заголовок", "unmapped body", "Распознанный текст"],
                              [["Visible title", "Private original body", "OCR"]])
        info = inspect_csv(path)
        self.assertEqual(info["rows_with_text"], 1)
        self.assertIsNone(info["field_mapping"]["message"])
        self.assertTrue(any("Primary message column was not recognized" in x for x in info["warnings"]))
        self.assertNotIn("Private original body", str(info))
        path = self.write_csv(["Заголовок", "Текст сообщения"], [["Visible title", "<p> </p>"]])
        info = inspect_csv(path)
        self.assertTrue(any("titles/OCR only" in x for x in info["warnings"]))

    def test_specific_message_headers_take_precedence_over_generic_columns(self):
        path = self.write_csv(["Сообщение", "Текст сообщения", "Ссылка", "Ссылка на сообщение", "Дата", "Время публикации", "Автор", "Кто пишет"],
                              [["Short legacy field", "Full original body", "https://example.invalid/other", "https://example.invalid/post", "Legacy date", "17.10.2026\u00a016:54", "Legacy", "Actual author"]])
        doc = next(iter_documents(path))
        self.assertIn("Full original body", doc["text"])
        self.assertNotIn("Short legacy field", doc["text"])
        self.assertEqual(doc["url"], "https://example.invalid/post")
        self.assertEqual(doc["published_at"], "17.10.2026\u00a016:54")
        self.assertEqual(doc["author"], "Actual author")

    def test_duplicate_headers_extra_and_missing_cells_preserve_strings(self):
        path = self.write_csv(["text", "text", "id"], [["first", "second", "1", "extra"], ["third"]])
        info = inspect_csv(path)
        self.assertTrue(any("Duplicate header" in x for x in info["warnings"]))
        self.assertTrue(any("column count: 2" in x for x in info["warnings"]))
        docs = list(iter_documents(path))
        self.assertEqual(docs[0]["metadata"], {"text": "first", "text [2]": "second", "id": "1", "__column_4__": "extra"})
        self.assertEqual(docs[1]["metadata"]["id"], "")

    def test_normalization_preserves_negation_unicode_and_safe_markup(self):
        text = '<div>не поддерживай й; Édouard; العربية</div><script>erase negation()</script><!-- hidden --><p>not&nbsp;now<br>nta na rimwe</p>'
        clean = normalize_text(text)
        for word in ["не", "й", "Édouard", "العربية", "not now", "nta na rimwe"]:
            self.assertIn(word, clean)
        self.assertNotIn("script", clean)
        self.assertNotIn("erase", clean)
        self.assertNotIn("hidden", clean)
        self.assertEqual(normalize_text("e\u0301 й"), "é й")
        self.assertEqual(normalize_text("x < 5 &amp; y > 3"), "x < 5 & y > 3")
        self.assertEqual(normalize_text(None), "")
        with self.assertRaises(TypeError):
            normalize_text(1.2345)

    def test_fingerprint_is_stable_but_never_erases_negation(self):
        self.assertEqual(exact_fingerprint("<p>We agree</p>"), exact_fingerprint(" We\n agree "))
        self.assertNotEqual(exact_fingerprint("We agree"), exact_fingerprint("We do not agree"))
        self.assertNotEqual(exact_fingerprint("US"), exact_fingerprint("us"))
        self.assertNotEqual(exact_fingerprint("й"), exact_fingerprint("и"))

    def test_tab_utf16_empty_and_malformed_files(self):
        path = self.write_csv(["id", "text"], [["1", "два\nслова"]], delimiter="\t", encoding="utf-16")
        info = inspect_csv(path)
        self.assertEqual((info["encoding"], info["delimiter"]), ("utf-16", "\t"))
        self.assertIn("два\nслова", next(iter_documents(path))["text"])
        empty = self.root / "empty.csv"
        empty.write_bytes(codecs.BOM_UTF8)
        self.assertEqual(inspect_csv(empty)["rows_total"], 0)
        self.assertEqual(list(iter_documents(empty)), [])
        bad = self.root / "bad.csv"
        bad.write_text('id;text\n1;"unclosed\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            list(iter_documents(bad))

    def test_large_field_and_csv_limit_restoration(self):
        text = "long article " * 15000
        path = self.write_csv(["id", "text"], [["1", text]])
        previous = csv.field_size_limit()
        self.assertEqual(inspect_csv(path)["rows_with_text"], 1)
        self.assertIn(text.strip(), next(iter_documents(path))["text"])
        self.assertEqual(csv.field_size_limit(), previous)


class NearDuplicateTests(unittest.TestCase):
    @staticmethod
    def paragraph():
        return " ".join(f"topic{i} policy{i} proposal{i} detail{i}" for i in range(45))

    def test_near_copy_but_not_same_topic_or_contradiction(self):
        original = self.paragraph() + " We support the proposal."
        copy = original.replace("detail20", "clarification20")
        contradiction = original.replace("We support", "We do not support")
        other = " ".join(f"topic{i} different{i} policy{i} argument{i}" for i in range(45))
        pairs = near_duplicate_pairs([{"id": "a", "text": original}, {"id": "b", "text": copy}, {"id": "c", "text": contradiction}, {"id": "d", "text": other}])
        self.assertEqual([(p["left_id"], p["right_id"]) for p in pairs], [("a", "b")])
        self.assertGreater(pairs[0]["similarity"], .85)
        self.assertEqual(pairs[0]["kind"], "near_duplicate_candidate")

    def test_multilingual_negation_and_changed_numbers_not_duplicate_candidates(self):
        for affirmed, denied in [("Мы поддерживаем решение", "Мы не поддерживаем решение"), ("nous soutenons ce choix", "nous ne soutenons pas ce choix"), ("tunakubali mpango", "hatukubali mpango"), ("twemera gahunda", "ntitwemera gahunda"), ("total 200", "total 201")]:
            with self.subTest(denied=denied):
                self.assertEqual(near_duplicate_pairs([{"id": "a", "text": self.paragraph() + " " + affirmed}, {"id": "b", "text": self.paragraph() + " " + denied}]), [])

    def test_short_slogans_and_repeated_boilerplate_are_not_near_duplicates(self):
        for text in ["a fair future for all", "same promise repeated " * 30]:
            self.assertEqual(near_duplicate_pairs([{"id": "a", "text": text}, {"id": "b", "text": text}]), [])

    def test_deterministic_bounded_candidates_and_input_immutability(self):
        docs = [{"id": str(i), "text": self.paragraph()} for i in range(100)]
        original = [dict(x) for x in docs]
        pairs = near_duplicate_pairs(iter(docs))
        self.assertEqual(pairs, near_duplicate_pairs(docs))
        self.assertLessEqual(len(pairs), 8 * len(docs))
        self.assertLess(len(pairs), len(docs) * (len(docs) - 1) // 2)
        self.assertEqual(docs, original)

    def test_ambiguous_document_ids_fail(self):
        with self.assertRaises(ValueError):
            near_duplicate_pairs([{"text": self.paragraph()}])
        with self.assertRaises(ValueError):
            near_duplicate_pairs([{"id": "1", "text": "a"}, {"id": "1", "text": "b"}])


if __name__ == "__main__":
    unittest.main()
