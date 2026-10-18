"""Describe Ward clusters of the frozen Rwanda four-dimensional map.

No fitting, source recoding, imputation or API calls are performed here.  The
reported trait means use only supplied informative cells, including explicitly
marked range midpoints; predictions of the latent model are never substituted.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/rwanda_politicians_2021_2026/combined_sources_2026_10_18"
OUT = ROOT / "research/rwanda_politicians_2021_2026/political_report_q4_2026_10_18"
CHOSEN_K = 5
CLUSTER_NAMES = {
    1: "Практические задачи, услуги и предупреждение рисков",
    2: "Смешанные институциональные и оппозиционные профили",
    3: "Экономико-международный акцент: неоднородная малая группа",
    4: "Согласование и ценностная аргументация",
    5: "Внутреннее развитие и публичное представительство",
}


def read(name):
    return json.loads((SOURCE / name).read_text(encoding="utf-8"))


def ward(scores, k):
    return fcluster(linkage(scores, method="ward", metric="euclidean"), k, criterion="maxclust")


def partition(scores, actor_ids, names, k):
    labels = ward(scores, k)
    silhouettes = silhouette_samples(scores, labels)
    groups = []
    for label in sorted(set(labels.tolist())):
        ix = np.flatnonzero(labels == label)
        groups.append({
            "id": int(label), "n": int(len(ix)),
            "actor_ids": [actor_ids[i] for i in ix],
            "actor_names": [names[actor_ids[i]] for i in ix],
            "centroid": scores[ix].mean(axis=0).tolist(),
            "mean_silhouette": float(silhouettes[ix].mean()),
        })
    return {
        "k": k, "silhouette": float(silhouette_score(scores, labels)),
        "sizes": [g["n"] for g in groups], "groups": groups,
        "membership": {actor_ids[i]: int(labels[i]) for i in range(len(actor_ids))},
        "individual_silhouette": {actor_ids[i]: float(silhouettes[i]) for i in range(len(actor_ids))},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, roster, ratings, dictionary = [read(name) for name in
        ["models.json", "roster.json", "ratings.json", "dictionary.json"]]
    names = {r["id"]: r["name"] for r in roster}
    info = {r["actor_id"]: r for r in model["actor_information"]["4"]}
    all_ids = model["actor_ids"]
    full_scores = np.asarray(model["models"]["4"]["scores"], dtype=float)
    ix = [i for i, aid in enumerate(all_ids) if info[aid]["informative_traits"] >= 8]
    actor_ids = [all_ids[i] for i in ix]
    scores = full_scores[ix]
    standardized = StandardScaler().fit_transform(scores)
    candidates = {}
    for k in range(3, 8):
        raw = partition(scores, actor_ids, names, k)
        std = partition(standardized, actor_ids, names, k)
        full = partition(full_scores, all_ids, names, k)
        raw_labels = [raw["membership"][a] for a in actor_ids]
        candidates[str(k)] = {
            "raw": raw, "standardized": std, "all_42": full,
            "raw_vs_standardized_ari": float(adjusted_rand_score(
                raw_labels, [std["membership"][a] for a in actor_ids])),
            "core_vs_all_42_restricted_to_core_ari": float(adjusted_rand_score(
                raw_labels, [full["membership"][a] for a in actor_ids])),
        }
    chosen = candidates[str(CHOSEN_K)]["raw"]
    membership = chosen["membership"]
    actor_rows = []
    for i, aid in enumerate(actor_ids):
        actor_rows.append({
            "actor_id": aid, "name": names[aid], "cluster": membership[aid],
            "cluster_name": CLUSTER_NAMES[membership[aid]],
            "informative_traits": info[aid]["informative_traits"],
            **{f"factor_{j+1}": float(scores[i, j]) for j in range(4)},
            "silhouette": chosen["individual_silhouette"][aid],
        })
    # All informative original cells are retained here. Unknowns never become zero.
    cell_map = {(c["actor_id"], c["scale_id"]): c for c in ratings["cells"]}
    core_set = set(actor_ids)
    traits = []
    for group in chosen["groups"]:
        group["name"] = CLUSTER_NAMES[group["id"]]
        group["traits"] = []
        for scale in dictionary["scales"]:
            sid = scale["id"]
            cells = [cell_map[(aid, sid)] for aid in group["actor_ids"]
                     if (aid, sid) in cell_map and cell_map[(aid, sid)]["midpoint"] is not None]
            others = [c["midpoint"] for (aid, cell_sid), c in cell_map.items()
                      if aid in core_set and cell_sid == sid and c["midpoint"] is not None]
            if not cells:
                continue
            vals = [c["midpoint"] for c in cells]
            item = {
                "cluster": group["id"], "scale_id": sid, "name": scale["name"],
                "in_factor_map": sid in model["scale_ids"],
                "n_informative": len(cells), "n_members": group["n"],
                "coverage": len(cells) / group["n"],
                "n_point_coded": sum(c["lower"] == c["upper"] for c in cells),
                "n_range_midpoints": sum(c["lower"] != c["upper"] for c in cells),
                "mean_informative_midpoint": float(np.mean(vals)),
                "median_informative_midpoint": float(np.median(vals)),
                "mean_lower": float(np.mean([c["lower"] for c in cells])),
                "mean_upper": float(np.mean([c["upper"] for c in cells])),
                "core_mean_informative_midpoint": float(np.mean(others)),
                "difference_from_core_informative_mean": float(np.mean(vals) - np.mean(others)),
                "actor_values": [{"actor_id": c["actor_id"], "lower": c["lower"],
                                  "upper": c["upper"], "midpoint": c["midpoint"]} for c in cells],
            }
            group["traits"].append(item)
            traits.append(item)
    centers = np.asarray([g["centroid"] for g in chosen["groups"]])
    thin = []
    for i, aid in enumerate(all_ids):
        if aid in core_set:
            continue
        distance = np.linalg.norm(full_scores[i] - centers, axis=1)
        thin.append({
            "actor_id": aid, "name": names[aid],
            "informative_traits": info[aid]["informative_traits"],
            "status": "not_in_main_clustering", "factor_coordinates": full_scores[i].tolist(),
            "nearest_core_centroid_provisional_only": int(np.argmin(distance)) + 1,
            "distance_to_centroids": distance.tolist(),
        })
    # Fixed-coordinate deletion sensitivity measures partition mechanics only;
    # it is explicitly not an uncertainty estimate for source scores or factors.
    loo = []
    for drop in range(len(actor_ids)):
        keep = [i for i in range(len(actor_ids)) if i != drop]
        alt_labels = ward(scores[keep], CHOSEN_K)
        old_labels = [membership[actor_ids[i]] for i in keep]
        loo.append({"omitted_actor_id": actor_ids[drop],
                    "ari_to_full_partition_restricted": float(adjusted_rand_score(old_labels, alt_labels))})
    # This deliberately uses the already saved Bayesian fit; it does not fit a new model.
    bayes = model.get("bayesian_comparison", {}).get("4", {}).get("fit", {})
    bayes_comparison = None
    if "scores" in bayes:
        bx = np.asarray(bayes["scores"], dtype=float)[ix]
        bp = partition(bx, actor_ids, names, CHOSEN_K)
        bayes_comparison = {
            "partition": bp,
            "ari_to_primary_partition": float(adjusted_rand_score(
                [membership[a] for a in actor_ids], [bp["membership"][a] for a in actor_ids])),
            "note": "Different saved factor model; this tests method dependence, not sampling reproducibility.",
        }
    pair_ids = [
        ("diane-rwigara", "claver-gatete"),
        ("diane-rwigara", "edouard-ngirente"),
        ("victoire-ingabire", "mussa-fazil-harerimana"),
        ("bernard-ntaganda", "juvenal-marizamunda"),
        ("paul-kagame", "frank-habineza"),
        ("james-kabarebe", "philippe-mpayimana"),
    ]
    scale_names = {s["id"]: s["name"] for s in dictionary["scales"]}
    pair_checks = []
    for a, b in pair_ids:
        common = []
        for sid in model["scale_ids"]:
            ca, cb = cell_map.get((a, sid)), cell_map.get((b, sid))
            if ca and cb and ca["midpoint"] is not None and cb["midpoint"] is not None:
                common.append({"scale_id": sid, "name": scale_names[sid],
                               "actor_a_interval": [ca["lower"], ca["upper"]],
                               "actor_b_interval": [cb["lower"], cb["upper"]],
                               "equal_supplied_intervals": ca["lower"] == cb["lower"] and ca["upper"] == cb["upper"]})
        pair_checks.append({"actor_a": a, "actor_b": b, "n_shared_informative": len(common),
                            "n_equal_supplied_intervals": sum(c["equal_supplied_intervals"] for c in common),
                            "shared_values": common})
    core_cells = [c for (aid, sid), c in cell_map.items()
                  if aid in core_set and sid in model["scale_ids"] and c["midpoint"] is not None]
    core_points = [c for c in core_cells if c["lower"] == c["upper"]]
    compression = {"core_informative_cells": len(core_cells), "core_point_coded_cells": len(core_points),
                   "point_value_counts": dict(Counter(str(c["midpoint"]) for c in core_points)),
                   "fraction_points_equal_2_or_3": sum(c["midpoint"] in [2, 3] for c in core_points) / len(core_points)}
    payload = {
        "version": "frozen-q4-ward-2026-10-18", "dimensions": 4,
        "method": "Ward Euclidean on unstandardized frozen factor scores",
        "core_selection": "At least 8 informative traits in the frozen 29-trait model",
        "core_n": len(actor_ids), "main_k": CHOSEN_K,
        "cluster_labels_are_political_coalitions": False,
        "selection_rationale": "Five is an interpretive compromise: splits a heterogeneous 17-person group; six adds a poorly separated 3-person cluster containing Kagame. It is not a uniquely optimal k.",
        "coordinate_standard_deviations_core": scores.std(axis=0).tolist(),
        "source_hashes": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                          for name in ["models.json", "roster.json", "ratings.json", "dictionary.json"]},
        "chosen": chosen, "candidates": candidates, "members": actor_rows,
        "thin_profiles": thin, "fixed_coordinate_leave_one_out": loo,
        "bayesian_saved_model_sensitivity": bayes_comparison,
        "surprising_pair_shared_values": pair_checks,
        "supplied_point_value_compression": compression,
        "new_api_calls": 0, "new_api_cost_usd": 0,
        "trait_statistics_note": "Only informative supplied cells; range midpoints remain assumptions; no predicted cell or zero-filled unknown enters these summaries.",
    }
    (OUT / "clustering.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "clustering_members.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(actor_rows[0])); writer.writeheader(); writer.writerows(actor_rows)
    trait_fields = [k for k in traits[0] if k != "actor_values"]
    with (OUT / "clustering_traits.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=trait_fields); writer.writeheader()
        writer.writerows({k: row[k] for k in trait_fields} for row in traits)
    note = ["# Рабочая записка о группировке четырёхфакторной карты", "",
            "Это техническая записка для автора политологического отчёта. Она не заменяет проверку биографий и источников.", "",
            "## Выбор группировки", "",
            "Основная группировка охватывает 37 человек с не менее чем 8 информативными признаками из 29. Пять менее описанных участников не получают основную групповую принадлежность.",
            "Сохранено пять групп — содержательный компромисс, а не доказанное единственное число типов. При шести группах отделяется тройка Кагаме / Паула Ингабире / Иямуремье, но её средняя обособленность мала; у самого Кагаме она отрицательна. При четырёх остаётся чрезмерно смешанная группа из 17 человек.", "",
            "| Число групп | Силуэт | Размеры | ARI: стандартизация | ARI: включение всех 42 |",
            "|---|---:|---|---:|---:|"]
    for k, row in candidates.items():
        note.append(f"| {k} | {row['raw']['silhouette']:.3f} | {row['raw']['sizes']} | {row['raw_vs_standardized_ari']:.3f} | {row['core_vs_all_42_restricted_to_core_ari']:.3f} |")
    note += ["", "Стандартизация проверяет только относительный вес четырёх сохранённых координат. Она не учитывает неопределённость первичных оценок. Силуэты невысокие: границы между группами перекрываются.", "",
             "## Состав и наблюдаемые признаки", "",
             "Средние ниже вычислены только для информативных ячеек, с серединой диапазона там, где вместо точного балла сохранён диапазон. Неизвестные оценки не заменяются ни нулём, ни средним. Состав участников с известным признаком может меняться от строки к строке; это особенно важно для малых и смешанных групп."]
    for group in chosen["groups"]:
        note += ["", f"### {group['id']}. {group['name']} ({group['n']})", "",
                 "; ".join(group["actor_names"]), "",
                 f"Центр F1–F4: {', '.join(f'{v:.3f}' for v in group['centroid'])}. Средний силуэт {group['mean_silhouette']:.3f}.", "",
                 "| Признак | Известно / группа | Среднее | Отличие от общего среднего известных |",
                 "|---|---:|---:|---:|"]
        featured = [t for t in group["traits"] if t["in_factor_map"] and t["n_informative"] >= max(2, int(np.ceil(group["n"] / 2)))]
        featured.sort(key=lambda t: abs(t["difference_from_core_informative_mean"]), reverse=True)
        for t in featured[:10]:
            note.append(f"| {t['name']} | {t['n_informative']} / {group['n']} | {t['mean_informative_midpoint']:.2f} | {t['difference_from_core_informative_mean']:+.2f} |")
    note += ["", "## Неожиданные объединения и границы интерпретации", "",
             "* Диана Рвигара находится рядом с Эдуаром Нгиренте и Клавером Гатете. У Рвигары только 8 информативных признаков; у Нгиренте 18, у Гатете 21. В её профиле нет оценок большинства экономических и международных направлений. Поэтому название экономико-международной группы характеризует в первую очередь Нгиренте и Гатете; его нельзя автоматически переносить на Рвигару. Это соседство сходных частичных профилей, а не свидетельство политической близости.",
             "* Виктуар Ингабире и Бернар Нтаганда сгруппированы с некоторыми представителями власти. Признак публичной критики государственного курса отсутствует в основной 29-признаковой карте из-за недостаточного охвата. Следовательно, карта плохо выражает различие власть/оппозиция. Нельзя называть эту смешанную группу сторонниками власти: оценка поддержки курса есть только у 5 из 8 участников.",
             "* Группа с Кагаме включает Франка Хабинезу. Общие темы развития и представительства не означают общей партии, электората или поддержки Кагаме. По данным нельзя выводить уровень влияния группы из её размера; институциональное влияние президента устанавливается политическими фактами.",
             "* Все имена и политические названия здесь — интерпретационные подписи конкретной карты. Формулировки 'единый лагерь', 'коалиция', 'одинаковая идеология', 'общий электорат' не подтверждаются вычислением.", "",
             "## Проверка соседств по общим известным признакам", "",
             f"Из {compression['core_point_coded_cells']} точных исходных оценок у 37 участников {compression['fraction_points_equal_2_or_3']:.1%} равны 2 или 3. Слабая дифференциация части первичных описаний сама по себе создаёт похожие профили. Это не следует трактовать как самостоятельное доказательство политического консенсуса.", "",
             "1. **Рвигара — Гатете:** общих известных признаков 6. Четыре совпадают точно: ценностная аргументация 3/3, доступность базовых благ 2/2, полемическое отстаивание позиции 3/3, публичное раскрытие проблем 3/3. При этом технологическая модернизация, внешнее финансирование и международная кооперация у Рвигары неизвестны. Следовательно, её соседство с Гатете не устанавливает экономическую или внешнеполитическую общность.",
             "2. **Рвигара — Нгиренте:** общих известных признаков только 5. Совпадает лишь закодированная поддержка курса 2/2; доступность услуг 2/3, продолжение курса 2/3, ценностная аргументация 3/2, базовые блага 2/3. Нельзя объяснять объединение одинаковой лояльностью: это разные политические фигуры, а их сходство здесь — итог неполных профилей и невысокой дифференциации оценок.",
             "3. **Виктуар Ингабире — Хареримана:** 8 общих известных признаков, 7 точных совпадений. У обоих услуги 2, национальная общность 3, ценностная аргументация 3, базовые блага 2, полемическая защита своей стороны 3, национальные ценности 3, раскрытие проблем 3. Отличается общественное участие: 3 у Ингабире, 2 у Харериманы. Признак критики власти не входит в карту; объединение показывает общие темы публичного языка, а не одинаковое отношение к правительству.",
             "4. **Нтаганда — Маризамунда:** все 6 общих известных признаков совпадают точно: переговоры 2, совместное обсуждение 2, ценностная аргументация 2, базовые блага 3, полемическая защита позиции 3, раскрытие проблем 3. Эти широкие признаки не различают направленность критики и предмет полемики; совпадение баллов не делает оппозиционера и министра обороны союзниками.",
             "5. **Кагаме — Хабинеза:** 16 общих известных признаков. Совпадают проверяемое знание 3/3, продолжение курса 2/2, базовые блага 3/3, деловая активность 3/3; диапазоны по услугам, равноправию и национальной общности пересекаются. Однако программное изменение курса 3 у Кагаме и 4 у Хабинезы, ценностная аргументация 2/3, раскрытие проблем 2/3. Общие темы развития совместимы с политической конкуренцией.",
             "6. **Кабаребе — Мпайимана:** общих известных признаков всего 3: поддержка курса 2/2, базовые блага 3/3, ценностная аргументация 1/2. Здесь особенно нельзя переносить содержательную характеристику одного участника на другого; значительная часть четырёхмерного соседства определяется остальными непересекающимися наблюдениями и выбранной моделью.", "",
             "Все перечисленные цифры — сохранённые исследовательские оценки высказываний и действий, а не объективный рейтинг политической позиции или качества управления. Детальные пары и пропуски приведены в clustering.json.", "",
             "## Чувствительность к исходной модели", ""]
    if bayes_comparison:
        note += [f"При группировке уже сохранённого байесовского четырёхмерного решения совпадение с основным разбиением (ARI) равно {bayes_comparison['ari_to_primary_partition']:.3f}. Это дополнительное основание не выдавать группы за независимо установленные политические типы.", ""]
    else:
        note += ["Сохранённая байесовская модель не содержит поля scores; дополнительная группировка не выполнена.", ""]
    note += [f"Удаление одного участника без пересчёта факторов: ARI от {min(x['ari_to_full_partition_restricted'] for x in loo):.3f} до {max(x['ari_to_full_partition_restricted'] for x in loo):.3f}. Это только чувствительность механики группировки при фиксированной карте, не доверительный интервал и не проверка источников.", "",
             "## Пять недостаточно описанных участников", "",
             "Их можно перечислить в биографическом разделе и показать отдельно. Ближайший центр сохранён только для технической проверки, не для уверенной принадлежности.", ""]
    for item in thin:
        note.append(f"* {item['name']}: {item['informative_traits']} / 29 информативных признаков.")
    note += ["", "Новые платные запросы: 0. Исходные данные и факторное решение не изменены.", ""]
    (OUT / "CLUSTER_NOTES.md").write_text("\n".join(note), encoding="utf-8")
    print(json.dumps({"main_k": CHOSEN_K, "core_n": len(actor_ids),
                      "sizes": chosen["sizes"], "silhouette": chosen["silhouette"],
                      "bayesian_ari": bayes_comparison["ari_to_primary_partition"] if bayes_comparison else None,
                      "outputs": ["clustering.json", "clustering_members.csv", "clustering_traits.csv", "CLUSTER_NOTES.md"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
