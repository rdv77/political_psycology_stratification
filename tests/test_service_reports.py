"""Report comparison content and reproducibility packaging use local fixtures."""
from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from src.semantic_service.reports import (comparison_details, comparison_overview, html_document,
                                          markdown, report_blocks, write_report_bundle)


ROOT = Path(__file__).resolve().parents[1]


def comparison_fixture():
    objects = [{"id": f"o{i}", "name": f"Тестовый объект {i}", "sources": []} for i in range(12)]
    scales = [{"id": f"s{i}", "name": f"Тестовая шкала {i}", "positive": f"А{i}", "negative": f"Б{i}",
               "definition": "Тестовое наблюдаемое свойство"} for i in range(6)]
    def factors(q):
        return [{"id": f"F{i+1}", "name": f"Ось {i+1}", "label": f"Содержание {i+1}",
                 "variance_share": .15, "interpretation": f"Интерпретация q={q} F{i+1}",
                 "positive_traits": [{"name": scales[0]["name"], "loading": .75}],
                 "negative_traits": [{"name": scales[1]["name"], "loading": -.65}]} for i in range(q)]
    def clusters(k):
        return [{"id": f"C{i+1}", "name": f"Профиль {i+1}", "count": 12//k,
                 "members": objects[i*(12//k):(i+1)*(12//k)], "factor_center": {"F1": i+.1},
                 "interpretation": "Описание полного профиля группы", "representatives": [objects[i*(12//k)]],
                 "distinguishing_scales": [{"name": scales[0]["name"], "group_mean": .8,
                                            "overall_mean": .2, "difference": .6}]} for i in range(k)]
    models = []
    for q in (2,3,4):
        models.append({"q": q, "factor_count": q, "cluster_count": 3, "variance_total": .15*q,
                       "dimension_fit": {"heldout_log_likelihood_mean": -20+q, "residual_rms_off_diagonal": .1/q},
                       "numerical_recommended_k": 4 if q==3 else 3, "factors": factors(q), "clusters": clusters(3),
                       "loading_table": [{"name": scale["name"], "loadings": {f"F{i+1}": (-1)**i*.6 for i in range(q)}} for scale in scales],
                       "narrative": {"summary": f"Краткое содержание q={q}", "new_meanings": [f"Новое содержание q={q}"],
                                     "cluster_changes": f"Изменения полного профиля q={q}"}})
    targets = ["C1","C1","C2","C3","C1","C2","C2","C3","C1","C2","C3","C3"]
    transitions = [{"object_id": obj["id"], "name": obj["name"], "from_cluster": f"C{i//4+1}",
                    "to_cluster_raw": targets[i], "to_cluster_aligned": targets[i], "changed": f"C{i//4+1}" != targets[i]}
                   for i,obj in enumerate(objects)]
    matrix = {"rows": ["C1","C2","C3"], "columns": ["C1","C2","C3"], "values": [[2,1,1],[1,2,1],[1,1,2]]}
    quality = {"silhouette": .4, "mean_ari": .8, "median_ari": .9, "min_ari": .6, "min_cluster_median_jaccard": .7}
    candidates = [{"k": k, "silhouette": .4, "min_size": 2, "max_size": 6,
                   "stability": dict(quality)} for k in (2,3,4)]
    comparison = {"scope": {"n_objects": 12, "n_scales": 6, "fixed_cluster_count": 3, "ratings_reused": True},
        "models": models, "executive_summary": "Сводный итог сравнения", "recommendation": "Основная модель остаётся q=2",
        "factor_comparisons": [{"reference_q": 2, "target_q": 4,
            "matched_factors": [{"reference_factor": "F1", "target_factor": "F2", "congruence": .98, "sign": 1},
                                {"reference_factor": "F2", "target_factor": "F3", "congruence": -.97, "sign": -1}],
            "congruence_matrix": {"row_factors": ["F1","F2"], "column_factors": ["F1","F2","F3","F4"],
                                  "values": [[.1,.98,.2,.3],[.1,.2,-.97,.4]]},
            "unmatched_target_factors": ["F1","F4"], "principal_angles_degrees": [5.1,2.2],
            "procrustes": {"relative_loading_residual": .12}}],
        "fixed_k3": [{"reference_q": 2, "target_q": 3, "k": 3, "ari": .2, "nmi": .3, "moved_count": 6,
                      "unchanged_count": 6, "transitions": transitions, "contingency_aligned": matrix}],
        "numerical_free_k": [{"q": q, "recommended_k": 4 if q==3 else 3, "selection_rule": "Тестовое численное правило",
                              "candidates": candidates} for q in (2,3,4)],
        "selected_k": [{"q": 3, "k": 4, "selection_reason": "Четыре содержательно различимых профиля", "quality": quality,
                        "clusters": clusters(4),
                        "relative_to_q2": {"k_reference": 3, "k_target": 4, "ari": .25, "nmi": .4,
                                           "overlap": {"rows": ["C1","C2","C3"], "columns": ["C1","C2","C3","C4"],
                                                       "values": [[3,1,0,0],[0,2,2,0],[0,0,1,3]]}, "moved_count": 999},
                        "relative_to_fixed_k3": {"ari": .3, "nmi": .5, "overlap": matrix}}],
        "charts": [{"title": "Сравнительная карта", "file": "models/assets/comparison.png", "caption": "Сопоставление подпространств"}],
        "limitations": ["Сравнение условно относительно одной матрицы"]}
    study = {"id": "synthetic-report", "title": "Тест отчёта", "scope": "Тестовое пространство", "aspect": "Публичный образ",
             "date_from": "2016-01-01", "date_to": "2026-01-01", "objects": objects, "scales": scales}
    analysis = {"n_objects": 12, "n_scales": 6, "factor_count": 2, "cluster_count": 3, "variance_total": .3,
                "factors": factors(2), "clusters": clusters(3), "diagnostics": {}, "charts": [],
                "cluster_candidates": [], "model_comparison": comparison}
    return study, analysis, comparison


class ModelComparisonReportTests(unittest.TestCase):
    def test_summary_precedes_method_and_details_follow_primary_interpretation(self):
        study, analysis, _ = comparison_fixture()
        blocks = report_blocks(study, analysis, [])
        titles = [block["text"] for block in blocks if block["type"]=="heading"]
        self.assertLess(titles.index("Сравнение моделей q=2, q=3 и q=4: краткий итог"), titles.index("Методика"))
        self.assertLess(titles.index("Интерпретация изученного пространства"), titles.index("Сравнение факторных моделей: подробности"))
        self.assertLess(titles.index("Сравнение факторных моделей: подробности"), titles.index("Ограничения и воспроизводимость"))
        self.assertIn("Факторная модель содержит 2 осей", blocks[3]["text"])
        self.assertEqual(analysis["factor_count"], 2)

    def test_overview_separates_fixed_and_preferred_k_and_preserves_signed_loadings(self):
        _, _, comparison = comparison_fixture()
        overview = comparison_overview(comparison)
        summary = next(block for block in overview if block["type"]=="table")
        self.assertEqual(summary["rows"][1], ["3","45.0%","-17.000","0.033","4 / 4"])
        text = markdown(comparison_details(comparison))
        self.assertIn("Номер и знак оси условны", text)
        self.assertIn("-0.970", text)
        self.assertIn("Обратное направление", text)
        self.assertIn("-0.650", text)
        self.assertIn("Полная матрица нагрузок q=4", text)
        self.assertIn("Интерпретация q=4 F4", text)

    def test_all_six_changed_objects_appear_in_transition_table_without_truncation(self):
        _, _, comparison = comparison_fixture()
        blocks = comparison_details(comparison)
        changes = next(block for block in blocks if block["type"]=="table" and block["headers"]==
                       ["Объект","Исходная группа","Согласованная новая группа"])
        expected = [row for row in comparison["fixed_k3"][0]["transitions"] if row["changed"]]
        self.assertEqual(len(changes["rows"]), 6)
        self.assertEqual([row[0] for row in changes["rows"]], [row["name"] for row in expected])
        self.assertNotIn("Тестовый объект 0", [row[0] for row in changes["rows"]])

    def test_unequal_k_compares_rectangular_overlaps_and_complete_cluster_memberships(self):
        _, _, comparison = comparison_fixture()
        blocks = comparison_details(comparison)
        section = next(i for i,block in enumerate(blocks) if block.get("text")=="Рабочие варианты с отдельно выбранным k")
        selected = blocks[section:]
        text = markdown(selected)
        self.assertIn("Рабочая типология q=3, k=4", text)
        self.assertNotIn("999", text)
        self.assertNotIn("Сменили группу", text)
        self.assertTrue(any(block["type"]=="table" and block["headers"]==["Исходная / новая группа","C1","C2","C3","C4"] for block in selected))
        for index in range(12):
            self.assertIn(f"Тестовый объект {index}", text)
        self.assertIn("C4 Профиль 4", text)

    def test_nested_charts_render_relatively_and_untrusted_text_is_escaped(self):
        _, _, comparison = comparison_fixture()
        comparison["models"][1]["narrative"]["summary"] = "<script>alert('fixture')</script>"
        blocks = comparison_details(comparison)
        self.assertIn("models/assets/comparison.png", markdown(blocks))
        html = html_document(blocks)
        self.assertIn("src='models/assets/comparison.png'", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        for file in ("../escape.png", "https://example.com/chart.png", "C:/outside.png"):
            comparison["charts"][0]["file"] = file
            with self.subTest(file=file), self.assertRaises(ValueError):
                comparison_details(comparison)

    def test_preferred_equal_k_comparison_lists_all_four_changes(self):
        _, _, comparison = comparison_fixture()
        pair = dict(comparison["fixed_k3"][0], reference_q=3, target_q=4, k=4, moved_count=4, unchanged_count=8)
        transitions = [dict(row) for row in pair["transitions"]]
        changed = [row for row in transitions if row["changed"]]
        for row in changed[4:]:
            row.update(changed=False, to_cluster_aligned=row["from_cluster"])
        pair["transitions"] = transitions
        comparison["preferred_k4_comparison"] = pair
        blocks = comparison_details(comparison)
        start = next(i for i,block in enumerate(blocks) if block.get("text")=="Согласованность рабочих типологий: q=3 и q=4, k=4")
        selected = blocks[start:]
        changes = next(block for block in selected if block["type"]=="table" and block["headers"]==
                       ["Объект","Исходная группа","Согласованная новая группа"])
        self.assertEqual([row[0] for row in changes["rows"]], [row["name"] for row in changed[:4]])
        self.assertEqual(len(changes["rows"]), 4)

    def test_incomplete_changed_list_cannot_silently_be_exported(self):
        _, _, comparison = comparison_fixture()
        comparison["fixed_k3"][0]["moved_count"] = 7
        with self.assertRaisesRegex(ValueError, "Число переходов"):
            comparison_details(comparison)

    def test_all_nine_substantive_k_reviews_are_included_beside_numeric_candidates(self):
        _, _, comparison = comparison_fixture()
        comparison["models"][1]["candidate_review"] = [
            {"k": k, "interpretation": f"Содержательное объяснение варианта {k}"} for k in range(2,11)]
        blocks = comparison_details(comparison)
        reviewed = next(block for block in blocks if block["type"]=="table" and
                        block["headers"]==["k","Содержательное сравнение при q=3"])
        self.assertEqual([row[0] for row in reviewed["rows"]], [str(k) for k in range(2,11)])
        self.assertEqual(len(reviewed["rows"]), 9)
        self.assertIn("Содержательное объяснение варианта 10", markdown(blocks))

    def test_studies_without_comparison_keep_the_original_report_sections(self):
        study, analysis, _ = comparison_fixture()
        analysis.pop("model_comparison")
        titles = [block["text"] for block in report_blocks(study, analysis, []) if block["type"]=="heading"]
        self.assertIn("Факторная структура", titles)
        self.assertNotIn("Сравнение факторных моделей: подробности", titles)
        self.assertEqual(comparison_overview({}), [])
        self.assertEqual(comparison_details({}), [])


class ReportBundleTests(unittest.TestCase):
    def test_nested_model_artifacts_and_charts_keep_relative_paths_and_exact_content(self):
        temporary_root = ROOT / ".service_runtime"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="report-bundle-test-", dir=temporary_root) as temporary:
            study = Path(temporary) / "study"
            output = study / "output"
            files = {
                "output/report.md": "![alternative](models/q3/assets/objects_f1_f2.png)",
                "output/report.html": "<img src='assets/comparison/variance.png'>",
                "output/report.pdf": "synthetic PDF fixture",
                "output/report.docx": "synthetic DOCX fixture",
                "output/slides.pptx": "synthetic slide fixture",
                "output/report_content.json": '{"fixture":true}',
                "output/metadata.json": '{"primary_q":2}',
                "output/scores.csv": "object,F1\na,1",
                "output/model.joblib": "synthetic fitted model",
                "output/assets/primary.png": "primary bytes",
                "output/assets/comparison/variance.png": "comparison bytes",
                "output/models/q3/metadata.json": '{"q":3}',
                "output/models/q3/assets/objects_f1_f2.png": "q3 map bytes",
                "output/models/q3/fixed-k3/scores.csv": "object,F1\na,2",
                "output/models/q4/selected-k/model.joblib": "q4 selected fitted model",
                "study.json": '{"fixture":true}',
                "ratings.json": '[{"status":"observed"}]',
                "evidence/source.json": '{"fixture_evidence":true}',
                "output/models/q3/old_bundle.zip": "must not nest another bundle",
                "output/models/q3/.env": "must not package hidden runtime files",
                "output/unrelated.log": "must not package root runtime logs",
            }
            for name, content in files.items():
                file = study / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(content, encoding="utf-8")
            write_report_bundle(output)
            with zipfile.ZipFile(output / "report_bundle.zip") as archive:
                names = set(archive.namelist())
                expected = {
                    "report.md", "report.html", "report.pdf", "report.docx", "slides.pptx", "report_content.json",
                    "analysis/metadata.json", "analysis/scores.csv", "analysis/model.joblib",
                    "assets/primary.png", "assets/comparison/variance.png", "models/q3/metadata.json",
                    "models/q3/assets/objects_f1_f2.png", "models/q3/fixed-k3/scores.csv",
                    "models/q4/selected-k/model.joblib", "research/study.json", "research/ratings.json",
                    "research/evidence/source.json",
                }
                self.assertEqual(names, expected)
                for name in expected:
                    if name.startswith("analysis/"):
                        source = output / Path(name).name
                    elif name.startswith("research/"):
                        source = study / name.removeprefix("research/")
                    else:
                        source = output / name
                    self.assertEqual(archive.read(name), source.read_bytes())
                self.assertEqual(len(archive.namelist()), len(names))
            # A repeated export replaces the archive, without nesting the old one.
            write_report_bundle(output)
            with zipfile.ZipFile(output / "report_bundle.zip") as archive:
                self.assertEqual(set(archive.namelist()), expected)


if __name__ == "__main__":
    unittest.main()
