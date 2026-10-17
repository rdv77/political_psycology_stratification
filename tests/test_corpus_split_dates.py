"""Synthetic split-publication-date exports; never read the private source CSV."""
import csv
from pathlib import Path
import tempfile
import unittest

from src.corpus_pipeline.ingest import combine_publication_date, inspect_csv, iter_documents


class CorpusSplitDateTests(unittest.TestCase):
    def write_csv(self, directory, headers, values):
        path = Path(directory) / "synthetic.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(headers)
            writer.writerow([values.get(key, "") for key in headers])
        return path

    def test_auto_mapping_prefers_date_and_combines_split_time_without_losing_cells(self):
        # 49 columns, as in a full export, with time deliberately before date.
        headers = ["№", "Время публикации", "Дата публикации", "Текст сообщения"] + [f"extra_{i}" for i in range(45)]
        values = {"№": "1", "Дата публикации": "17.10.2026", "Время публикации": "15:41",
                  "Текст сообщения": "Synthetic publication", "extra_44": "preserved"}
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, headers, values)
            self.assertEqual(inspect_csv(path)["field_mapping"]["published_at"], "Дата публикации")
            document = next(iter_documents(path))
        self.assertEqual(document["published_at"], "17.10.2026 15:41")
        self.assertEqual(document["metadata"]["Дата публикации"], "17.10.2026")
        self.assertEqual(document["metadata"]["Время публикации"], "15:41")
        self.assertEqual(document["metadata"]["extra_44"], "preserved")
        self.assertEqual(len(document["metadata"]), 49)

    def test_legacy_combined_time_column_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, ["Время публикации", "Текст сообщения"],
                                  {"Время публикации": "17.10.2026 15:41", "Текст сообщения": "Synthetic"})
            document = next(iter_documents(path))
        self.assertEqual(document["published_at"], "17.10.2026 15:41")

    def test_time_only_does_not_invent_date_and_explicit_mapping_remains_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(directory, ["Дата публикации", "Время публикации", "Текст сообщения"],
                                  {"Дата публикации": "17.10.2026", "Время публикации": "15:41",
                                   "Текст сообщения": "Synthetic"})
            document = next(iter_documents(path, mapping={"published_at": "Время публикации"}))
        self.assertEqual(document["published_at"], "15:41")
        self.assertEqual(combine_publication_date("15:41", ""), "15:41")

    def test_existing_timestamp_invalid_time_and_day_only_are_not_guessed(self):
        self.assertEqual(combine_publication_date("2026-10-17T15:41:00+02:00", "12:00"),
                         "2026-10-17T15:41:00+02:00")
        self.assertEqual(combine_publication_date("17.10.2026", "25:99"), "17.10.2026")
        self.assertEqual(combine_publication_date("17.10.2026", ""), "17.10.2026")
        self.assertEqual(combine_publication_date("2026-10-17", "15:41:03"), "2026-10-17 15:41:03")
        self.assertEqual(combine_publication_date("", "15:41"), "")


if __name__ == "__main__":
    unittest.main()
