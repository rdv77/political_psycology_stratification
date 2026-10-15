"""Assemble a source-backed technical appendix; no numerical fitting or APIs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/rwanda-politicians-2021-2026"
OUT = STUDY / "output"
RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def known(rows):
    return sum(cell.get("status") == "observed" and cell.get("value") is not None
               for row in rows for cell in row.get("pole_scores", {}).values())


def markdown(blocks):
    lines = []
    for block in blocks:
        kind = block["type"]
        if kind == "heading":
            lines += ["#" * (block["level"] + 1) + " " + block["text"], ""]
        elif kind == "paragraph":
            lines += [block["text"], ""]
        elif kind == "table":
            lines += ["| " + " | ".join(block["headers"]) + " |",
                      "| " + " | ".join("---" for _ in block["headers"]) + " |"]
            lines += ["| " + " | ".join(cell.replace("|", "/").replace("\n", "; ") for cell in row) + " |" for row in block["rows"]]
            lines += [""]
        elif kind == "link":
            lines += [f"[{block['text']}]({block['url']})", ""]
        else:
            raise ValueError("Unexpected technical content block")
    return "\n".join(lines)


def main():
    ward = load(OUT / "cluster_candidates_q3.json")
    study = load(OUT / "cluster_candidates_q3/source_study.json")
    ratings = load(OUT / "cluster_candidates_q3/source_ratings.json")
    preview = load(OUT / "cluster_candidates_q3/source_factor_preview.json")
    imputation = load(OUT / "imputation_review_2026-10-15.json")
    strict = load(OUT / "strict_without_flags/COMPARISON.json")
    model = next(x for x in preview["models"] if x["q"] == 3)
    selected = next(x for x in ward["candidates"] if x["k"] == 4)
    diagnostics = preview["diagnostics"]
    assert ward["n_objects"] == 37 and ward["n_variables"] == 8
    assert known(ratings) == diagnostics["observed_input_ratings"] == 589
    assert selected["sizes"] == [2, 10, 2, 23]
    assert ward["factor_model"]["loadings_max_abs_difference_from_preview"] == 0
    assert ward["object_order"] == imputation["object_ids"] and ward["trait_order"] == imputation["variable_ids"]
    archived = [STUDY / "attempts" / name / "ratings.json" for name in (
        "independent_before_review", "independent_before_completion", "independent_before_coverage_expansion")]
    stages = [known(load(path)) for path in archived] + [known(ratings)]
    assert stages == [774, 528, 548, 589]
    assert load(RESEARCH / "INDEPENDENT_COMPLETION_FINAL_REVIEW.json")["summary"]["reviewed_additions"] == 20
    assert load(RESEARCH / "COVERAGE_EXPANSION_FINAL_REVIEW.json")["accepted_count"] == 41
    group_names = {"C1": "Преемственность через ценности и память", "C2": "Социально-управленческая преемственность",
                   "C3": "Программное обновление и доступность благ", "C4": "Разнородные менее выраженные измеренные профили"}
    blocks = []
    h = lambda text, level=1: blocks.append({"type": "heading", "level": level, "text": text})
    p = lambda text: blocks.append({"type": "paragraph", "text": text})
    t = lambda headers, rows: blocks.append({"type": "table", "headers": headers, "rows": [[str(cell) for cell in row] for row in rows]})
    link = lambda text, url: blocks.append({"type": "link", "text": text, "url": url})

    h("Технический отчёт: Руанда, три фактора и четыре рабочие группы", 0)
    p("Период наблюдения: 15 октября 2021 — 15 октября 2026. Предмет — документированные проявления публичного политического образа в целевой выборке персон. Итоговый расчёт: 37 объектов, 8 независимо измеренных свойств, FA q=3 и Ward k=4. q=3 выбран пользователем после просмотра альтернатив; k=4 принят при содержательном рассмотрении составов. Это рабочая типология, а не статистическое доказательство четырёх социальных слоёв или политических коалиций.")
    p("Документ составлен по замороженным численным файлам проекта. Он отделяет исходные наблюдения, ассистентскую разметку, аналитическую обработку и содержательный выбор. Ни при кластерном расчёте, ни при подготовке этого текста новые LLM-оценки или платные запросы не выполнялись; канонические рейтинги не редактировались.")

    h("1. Дизайн и хронология измерения")
    p("Пользователь утвердил 42 объекта и 24 определения шкал; receipts approvals.objects и approvals.scales сохранены в снимке study.json. Выборка охватывает государственное и парламентское руководство, отраслевых управленцев, территориальный уровень, оппозицию, эмиграцию и явно обозначенных международных дипломатов. Она целевая по ролям, не случайная, не пропорциональная и не репрезентативная для всего населения или всех политиков Руанды.")
    p("Первоначальный paired_difference-подход требовал двух известных полюсов для разности. После явного решения пользователя measurement_model=independent_poles те же 24 определения дают 48 самостоятельных переменных 0–5: scale_id__positive и scale_id__negative. Оба свойства могут быть выражены одновременно. Условные слова positive/negative обозначают порядок в исходной паре, а не моральную оценку и не знак факторной нагрузки. Неизвестный второй полюс не скрывает известный первый.")
    t(["Этап", "Известных полюсов из 2016", "Основание"], [
        ["Исходный независимый проход", stages[0], "attempts/independent_before_review/ratings.json"],
        ["После полной проверки оснований", stages[1], "attempts/independent_before_completion/ratings.json"],
        ["После completion: 20 новых измерений", stages[2], "attempts/independent_before_coverage_expansion/ratings.json"],
        ["После coverage expansion: ещё 41", stages[3], "cluster_candidates_q3/source_ratings.json"]])
    p("Проверка 774 исходных известных свойств не была независимой человеческой экспертизой: ассистенты сверяли определения, эпизоды, даты, персональную атрибуцию и индексы досье. Неподдержанные оценки переводились в unknown; ошибочные ссылки исправлялись только при наличии подходящего факта того же досье; некоторые интенсивности перекодировались с прозрачным сохранением исходного значения и мотива. Итоговое снижение 774→528 составляет 246 свойств. Числа предыдущего парно-разностного прохода сюда не переносятся.")
    p("Completion и expansion — дополнительные оценки по найденным прямым сведениям, не математическая импутация. Проверялись все заранее заданные пропуски соответствующего плана; правило не прекращалось при достижении порога и не использовало желаемые нагрузки или кластеры. Тем не менее повторная ассистентская проверка не устраняет зависимость от корпуса, единичных эпизодов, общих событий и ограниченной публичности. История предложений, source_audit, provenance, event_id и URL сохранена.")
    t(["Интенсивность", "Смысл якоря"], [[0, "Прямо подтверждённая невыраженность при достаточном относящемся материале; отсутствие источника не подходит"],
        [1, "Единичное прямое проявление"], [2, "Ясное, но ограниченное проявление; ограничения источника сохраняются"],
        [3, "Повторяющийся акцент в разных эпизодах"], [4, "Сильный повторяющийся акцент"], [5, "Устойчиво центральный акцент с многими прямыми основаниями"]])
    p("unknown означает недостаточное основание для количественной интенсивности, а не нейтральность, отсутствие позиции или истинный ноль. Должность, осуждение оппозиционера, участие во встрече, неподписанный ведомственный курс и несколько публикаций об одном событии не должны автоматически определять персональную установку. Оценки относятся к публичным проявлениям в периоде, а не к внутренним убеждениям.")

    h("2. Допуск к общему пространству и пропуски")
    p("Единая политика применяется до любого заполнения: на каждой итерации сначала исключаются постоянные переменные и переменные с наблюдениями менее чем у max(8, ceil(0,60×n)) текущих объектов. Затем исключаются объекты с менее чем max(4, ceil(0,60×p)) известных оставшихся свойств. Процедура повторяется до стабильности. В конце нужны не менее 12 объектов, 6 непостоянных свойств, ранг не менее двух и общее покрытие не ниже 75%. Исключённые на этапе допуска неизвестные данные не восстанавливаются импутацией для достижения порогов.")
    p("При 42 объектах первый порог переменной равен 26 наблюдениям. Из 48 остаётся 8 переменных, затем каждая строка должна иметь хотя бы 5 из 8 наблюдений. Получены 37 строк и 245 известных из 296 ячеек: 82,77% покрытия, 51 настоящий пропуск. На повторной итерации порог переменной 23 из 37; новых исключений нет. Строгий порядок «переменные, затем объекты» сохраняет полные профили по общему ядру, которые могли иметь пропуски по исключённым редким шкалам.")
    t(["Исключённый объект", "Известно по 8 до исключения", "Требовалось"], [[x["name"], x["observed_scales"], x["required_scales"]] for x in diagnostics["excluded_objects"]])
    p("Исключение этих пяти персон обусловлено недостаточным сопоставимым измерением; оно не описывает их политическую незначимость или отсутствие взглядов. Для них не выданы нулевые координаты или угаданная группа. Отбор по источниковому покрытию меняет изучаемую выборку и способен систематически исключать менее публичных или ограниченных в участии персон.")
    t(["Оставшееся свойство", "Наблюдений среди 37"], [[x["name"], f"{x['observed_retained_objects']}/37"] for x in diagnostics["input_scale_coverage"] if x["retained"]])

    h("3. Вероятностная FA и воспроизводимые параметры")
    p("Обучающее ядро использует sklearn.decomposition.FactorAnalysis: линейная гауссова модель z = m + L f + ε, f ~ N(0,Iq), ε ~ N(0,Ψ), Ψ диагональна и положительна. Соответственно Σ = LLᵀ + Ψ. Это общий факторный анализ с отдельной специфической дисперсией; PCA в этом проекте применяется только как отдельная геометрическая диагностика размерности.")
    link("Описание вероятностной FactorAnalysis в scikit-learn", "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html")
    p("Для обучения средние каждого столбца вычисляются по его известным значениям; 51 обучающий пропуск временно заполняется своим средним. StandardScaler оценивается на этой обучающей копии: центрирование и масштаб ddof=0. Затем FA: q=3, svd_method=lapack, max_iter=2000, tol=0,0001, seed=20261015. Минимум Ψ задаётся floor=1e-6 после fit; это защитная регуляризация диагонального шума, а не новый эмпирический факт. В текущем решении floor не сработал ни для одной переменной.")
    fit = ward["factor_model"]["fit_diagnostics"]
    t(["Параметр/проверка", "Результат"], [["Итераций FA", fit["iterations"]], ["Предупреждение сходимости", fit["convergence_warning"]],
        ["Минимум шума до/после floor", f"{fit['minimum_noise_before_floor']:.8f} / {fit['minimum_noise_after_floor']:.8f}"],
        ["Число активированных floor", fit["noise_floor_active_features"]], ["Condition posterior q×q", f"{fit['posterior_system_condition']:.6f}"],
        ["Максимальное различие нагрузок с сохранённым preview", ward["factor_model"]["loadings_max_abs_difference_from_preview"]]])
    p("Ортогональное Varimax-вращение R вычисляется собственным SVD-алгоритмом (до 200 итераций, tolerance=1e-8), Lr=LR, RᵀR=I. Оно перераспределяет нагрузки между осями при сохранении общей реконструируемой ковариации. Баллы вращаются тем же R; евклидовы расстояния между ними сохраняются. Знаки и номера факторов условны и локальны выбранному q; сравнение других решений требует согласования осей.")
    p("Для индивидуальных баллов не используется заполненная копия строки. Пусть O — только её действительно известные координаты; zO стандартизованы фиксированными обучающими средними и масштабами, mO — центр модели в стандартизованных единицах. Тогда AО = Iq + LOᵀΨO^(-1)LO; μf|zO = AО^(-1) LOᵀΨO^(-1)(zO−mO); rotated_score = μf|zOᵀR. В коде решается система q×q через numpy.linalg.solve, без прямого обращения сингулярной p×p корреляционной матрицы. ΨO положительна. При меньшем числе наблюдений условное среднее сильнее зависит от модельного prior; индивидуальная неопределённость не исчезает.")
    p("Обучающее заполнение средним и последующая проекция по наблюдаемым координатам — разные операции. Маргинализация отсутствующих признаков при posterior projection НЕ превращает исходное обучение в FIML. Гауссово линейное приближение применяется к ограниченным, фактически порядковым оценкам 0–5; равноинтервальность и нормальность не доказаны.")
    p("Portable-параметры сохранены в cluster_candidates_q3/factor_parameters_q3.json и внутри factor_model.parameters общего кластерного JSON: training_imputation, training_mean, training_scale, standardized_model_mean, исходные L, noise_variance, R, noise_floor, точный trait_order и training_object_order. Для новой строки нужно сохранить этот порядок и fixed train-параметры, а неизвестное передавать NaN; нельзя заново оценивать scaler по новой строке. Joblib — дополнительный удобный формат, JSON не требует pickle.")

    h("4. Число факторов и смысл процентов")
    p("До fit проверяется необходимое ограничение числа ковариационных параметров: nominal_df = p(p+1)/2 − [p q + p − q(q−1)/2] ≥ 0. При p=8 имеется 36 ковариационных моментов; q3 требует 29 параметров, nominal_df=7. Допустимы q2, q3, q4; q5..q7 имеют отрицательные df и не обучались, q8 отсечён p−1. Неотрицательные df необходимы, но не доказывают локальную идентификацию, содержательную валидность или достаточную выборку. Здесь не проводился χ²-тест по этим df.")
    t(["q", "Параметры / df", "Общая доля, %", "Held-out mean (SD)", "Residual RMS", "Сильных шкал на ось"], [[d["q"], f"{d['parameter_count']} / {d['nominal_df']}", f"{100*d['common_variance_share']:.4f}",
        f"{d['heldout_log_likelihood_mean']:.4f} ({d['heldout_log_likelihood_sd']:.4f})", f"{d['residual_rms_off_diagonal']:.5f}", str(d["strong_traits_per_factor"])] for d in preview["dimension_candidates"]])
    p("Held-out — три одинаковых для q разделения строк 75/25 с seed=20261015. Среднее, scaler и FA обучаются только на train; тестовая плотность вычисляется по наблюдаемым координатам с якобианом масштаба. Разделения перекрываются, а SD отражает описательный разброс. Процедура не даёт доверительного интервала или проверки значимости разницы q. Перестановочная PCA-диагностика (500 независимых перестановок каждого столбца, 95-й процентиль) выделяет три ведущие компоненты, но это не доказательство трёх общих факторов.")
    p("Автоматическая эвристика предпочитает q2: среди решений без предупреждений и с ≥2 нагрузками |L|≥0,35 на ось выбирается минимальное q в пределах max(1, SD_best/√3) от лучшего среднего held-out. Пользователь осознанно выбрал q3 ради отдельного различия «доступность благ/профилактика — ценностная аргументация». Снижение остатка внутри выборки и увеличение общей доли при q4 сами по себе не оправдывают q4; у него df=2 и профилактическая ось имеет только один сильный индикатор.")
    p("Доля каждой вращённой оси определяется descriptively как Σj(Lr[j,a])² / Σj Var(zj). Знаменатель — вся стандартизованная обучающая дисперсия всех восьми допущенных переменных после mean completion, здесь 8 с машинным округлением. Суммарная доля q3=52,2945124%; это доля общей FA в этой конкретной восьмимерной матрице, не PCA explained_variance_ratio, не доля всех свойств политического сознания и не процент объяснения поведения населения.")
    t(["Raw ось", "Доля, %", "Отображение", "Читательский смысл положительного X"], [[f["id"], f"{100*f['variance_share']:.5f}", f"X{i+1}={'−' if i==0 else ''}{f['id']}", ward["display_axes"][f"X{i+1}"]["label"]] for i,f in enumerate(model["factors"])])
    t(["Свойство", "F1", "F2", "F3"], [[row["name"], *[f"{row['loadings'][f'F{j+1}']:+.5f}" for j in range(3)]] for row in model["loading_table"]])
    p("Высокий X1 означает более сильные измеренные проявления преемственности и поддержки курса; высокий X2 — более сильное программное обновление и раскрытие проблем; высокий X3 — относительный акцент на доступности благ и профилактике при меньшей ценностно-мемориальной аргументации. Низкое значение независимого свойства не означает его неизмеренный антоним. Для диаграмм отражён лишь знак F1: расстояния и группы Ward от этого не изменяются.")

    h("5. Чувствительность к заполнению пропусков")
    p("Отдельная проверка сравнила mean, ограниченную итеративную регрессию BayesianRidge и KNN(k=5, weights=distance, nan_euclidean). Двадцать раз случайно разделялись строки 75/25; внутри validation скрывались 12% исходно известных ячеек. Fit imputer использует только train; замаскированные истины в предикторы не попадали. Все методы получают те же разделения и маски; всего 158 проверяемых ячеек с повторами. Гиперпараметры фиксированы, поэтому это двухуровневая маскировка, а не nested CV для настройки.")
    t(["Метод", "MAE", "RMSE", "Ячеек с повторами", "Warnings"], [[r["method"], f"{r['pooled_mae']:.6f}", f"{r['pooled_rmse']:.6f}", r["evaluated_cell_occurrences"], r["convergence_warnings"]] for r in imputation["validation"]["summary"]])
    p("Регрессионное восстановление не показало преимущества над mean в этом эксперименте; у него четыре предупреждения сходимости при лимите20. Проверяются искусственно скрытые ИЗВЕСТНЫЕ ячейки отобранного ядра. Это не валидация истинных unknown: публичность, должность, репрессии, тема и доступность источника могут делать пропуски неслучайными. MAR/MNAR не устанавливаются таким сравнением, регрессия не устраняет MNAR автоматически. Импутация не выдаётся за новое измерение и не расширяет допуск на все42.")
    link("scikit-learn: пропуски и множественные реализации", "https://scikit-learn.org/stable/modules/impute.html")
    link("scikit-learn: IterativeImputer, BayesianRidge и sample_posterior", "https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html")
    p("В десяти отдельных posterior-реализациях IterativeImputer(BayesianRidge, sample_posterior=True) неизвестные аналитически заполнены в0..5, исходно известные значения сохранены. Seed каждого draw=20261015+5000+draw. Каждая завершённая матрица получает собственные scaler, FA q и Varimax; колонки нагрузок согласуются оптимальной перестановкой и знаком по Tucker-конгруэнтности, без непрерывного вращения для искусственного повышения сходства. Несогласованные оси не усредняются. Процентили описывают десять реализаций, не являются доверительными интервалами и не образуют Rubin-пулинг.")
    t(["q", "Ось", "Tucker 5% / median / 95%"], [[m["q"], axis["factor_id"], " / ".join(f"{v:.4f}" for v in axis["congruence_q05_q50_q95"])] for m in imputation["factor_sensitivity"]["models"] for axis in m["posterior"]["axes"]])

    h("6. Проверка зависимости от оговорок и дополнительного слоя")
    t(["Исключение в копии", "Оценок исключено", "Остаток n×p", "Допуск"], [[s["scenario"], s["removed_count"], f"{s['diagnostics']['retained_objects']}×{s['diagnostics']['retained_scales']}",
        "да" if s["measurement_status"].startswith("admitted") else "нет"] for s in imputation["coverage_sensitivity"]["scenarios"]])
    p("Удаление всех текущих44 source_audit.decision=flag даёт34×7 с покрытием84,03%; по прежним порогам допустимы q2 и q3, q4 имеет df=−1. Флаг — оговорка измерения, а не доказанная ложность, поэтому строгий сценарий является чувствительностью. Удаление всех61 completion/coverage_expansion оставляет18×4 и не допускает FA; объединённое исключение102 —0×3. Старые оценки из истории записи не заменяют текущие provenance. Основной результат существенно зависит от собранного дополнительного слоя.")
    p("В отдельном строгом расчёте q3 объясняет55,9846% дисперсии СЕМИ стандартизованных переменных, df=3. Это иной набор объектов/переменных и иной знаменатель; больший процент не означает улучшения. Из основной матрицы исчезает programmatic-renewal__positive, а также Хареримана, Мпайимана и Виктуар Ингабире. При сопоставлении по общим семи признакам нет непрерывной подгонки вращения.")
    strict_q3 = next(x for x in strict["comparisons"] if x["q"] == 3)
    t(["Основная ось", "Сопоставленная строгая", "Знак строгой", "Tucker", "Энергия исходной оси на общих7, %"], [[a["main_factor_id"], a["strict_original_factor_id"], f"{a['strict_sign_multiplier']:+d}",
        f"{a['absolute_congruence_after_alignment']:.5f}", f"{100*a['main_loading_energy_retained_in_shared_variables']:.2f}"] for a in strict_q3["axis_alignment"]])
    p("Преемственность F1 сравнительно узнаваема (Tucker≈0,932). Обновление F2 потеряло основной измерительный якорь и две трети своей энергии нагрузок; слабое совпадение0,379 не является чистым тестом того же полностью измеренного конструкта. Третья ось также меняется (0,582). В строгих решениях профилактическая ось имеет лишь один сильный индикатор. Вся трёхосная структура и последующая типология не подтверждены как устойчивые к политике источниковых оговорок; q3 сохраняется как условная рабочая гипотеза.")

    h("7. Ward-кластеризация трёхмерных баллов")
    p("Основная группировка — агломеративная иерархическая Ward с евклидовыми расстояниями в трёх исходных единицах вращённых posterior scores. На каждом шаге объединение минимизирует прирост внутригрупповой суммы квадратов. Одно сохранённое дерево разрезается на k=2..6 через scipy.cluster.hierarchy.cut_tree; число групп точно соответствует k. Это не прежний KMeans из общей сервисной функции, и к Ward не переносятся автоматически его эвристики выбора k или правила минимального размера.")
    link("SciPy: Ward hierarchical clustering", "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ward.html")
    p("Стандартные отклонения raw F1/F2/F3 равны0,98084/1,00107/0,86219. Основной расчёт не уравнивает эмпирические масштабы осей; отдельная проверка z-масштабирования дала ARI=1 для всех k2..6. Отражение X1=−F1 также точно сохраняет разбиение. Номера C локальны каждому k; сравнение групп с одинаковым номером между k без просмотра состава некорректно. Полный порядок объектов и linkage сохранены; при точных равенствах расстояний порядок строк способен определять разрешение ties.")
    t(["k", "Размеры", "Silhouette", "Subsample median ARI", "MI median ARI", "ARI z-scale"], [[c["k"], "/".join(map(str,c["sizes"])), f"{c['silhouette']:.5f}",
        f"{c['stability']['object_subsampling']['median_adjusted_rand']:.5f}", f"{c['stability']['multiple_imputation']['median_adjusted_rand']:.5f}", f"{c['stability']['empirical_axis_standardization']['adjusted_rand']:.3f}"] for c in ward["candidates"]])
    p("Для каждого k выполнены100 подвыборок по29 из37 объектов без возвращения (80% с округлением вниз; seed=20261015+10000+k). Факторная модель и баллы в этом эксперименте фиксированы. Новое Ward-разбиение сравнивается с исходным на общих объектах через ARI. Метки согласуются максимальным суммарным перекрытием по Hungarian assignment; Jaccard рассчитывается для этой единой пары групп, условно на присутствии исходной группы. Парная co-membership label-invariant и делит число совместных кластеризаций только на повторы, где присутствуют оба участника.")
    link("scikit-learn: Adjusted Rand index", "https://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-index")
    p("Отдельная проверка применяет Ward ко всем37 объектам в каждой из тех же10 posterior-импутаций, с новым scaler и FA q3. Она одновременно отражает неопределённость завершения и перестройку факторной геометрии; это не изолированный эффект смены источника. Перестановка/знак согласованных осей служит читаемости и не меняет евклидовы расстояния. Повторы не являются независимыми социальными выборками, а частота совпадения не является вероятностью истинного политического членства.")
    p("Выбран k4 после рассмотрения профилей: k3 объединяет ценностную пару с социально-управленческой10; k4 разделяет их по различию третьей оси и документированных свойств. k2 имеет более высокий silhouette, но сводит35 разных персон в одну остаточную группу. k5 дополнительно выделяет смешанную пятёрку Кагаме, Маризамунда, Калинда, Мпайимана, Рвигара с невысокой воспроизводимостью при MI; это трудно честно назвать единым политическим типом. k6 дробит пару обновления на два singleton. Это содержательный компромисс аналитика, а не автоматический максимум метрики или доказанный оптимум.")
    t(["k4 группа", "n", "Median Jaccard subsample", "Median Jaccard MI"], [[g["id"]+": "+group_names[g["id"]], g["count"], f"{g['subsample_jaccard']['median']:.5f}", f"{g['imputation_jaccard']['median']:.5f}"] for g in selected["clusters"]])
    p("Стабильность неоднородна: ценностная пара имеет median Jaccard1/1, группа обновления1/0,45. Даже сохранение обоих участников в одной группе не означает сохранение эксклюзивной пары: при MI к ним могут присоединяться другие. Для k4 общая MI median ARI≈0,4695, поэтому границы типологии нельзя считать установленными. C4 обозначает большой разнородный остаток по измеренному ядру, а не центристов, нейтральных акторов или естественную коалицию.")
    p("В условном парном подсчёте Бизимана и Хареримана оставались вместе во всех55 subsample-повторах с присутствием обоих и во всех10 MI. Хабинеза и Виктуар Ингабире оставались вместе55 из57 совместных subsample-повторов (0,9649) и6 из10 MI (0,60). Эти знаменатели исключают повторы с отсутствием одного участника; наблюдение пары не сводится к произвольной нумерации кластеров.")
    t(["Группа", "Raw центр F1/F2/F3", "Display центр X1/X2/X3"], [[g["id"], " / ".join(f"{v:+.4f}" for v in g["raw_factor_center"].values()),
        " / ".join(f"{v:+.4f}" for v in g["display_factor_center"].values())] for g in selected["clusters"]])
    t(["Свойство", "C1 mean (n)", "C2 mean (n)", "C3 mean (n)", "C4 mean (n)"], [[selected["clusters"][0]["observed_trait_profiles"][j]["name"],
        *[(f"{g['observed_trait_profiles'][j]['observed_mean']:.3f} ({g['observed_trait_profiles'][j]['observed_n']}/{g['count']})" if g['observed_trait_profiles'][j]['observed_mean'] is not None else f"unknown (0/{g['count']})") for g in selected["clusters"]]] for j in range(8)])
    p("Средние свойств в таблице рассчитаны только по исходно observed, не по заполнениям. У пары C1 программное обновление неизвестно у обоих; оно не объявляется низким. В C3 поддержка государственного курса неизвестна у обоих; она не объявляется отсутствующей. Число измерений показано для каждого среднего. Различия групп описательные: тестов значимости, причинной атрибуции или популяционных выводов нет.")
    for g in selected["clusters"]:
        p(g["id"]+" — "+group_names[g["id"]]+". Полный состав: "+"; ".join(x["name"] for x in g["members"])+".")

    h("8. Индивидуальная неопределённость и границы")
    p("Для просмотра границ сохраняются silhouette каждого объекта, расстояние до собственного и ближайшего альтернативного центроида и margin=(d_alt−d_own)/max(d_own,d_alt). Ward-членство остаётся срезом иерархии: ближайший центроид не переназначает участника, а отрицательный margin допустим. Отмечается ручная проверка, если conditional same-cluster fraction<0,8 хотя бы в одном эксперименте, silhouette<0 или margin<0,1; это диагностические эвристики, не вероятностная классификация.")
    t(["Участник", "k4", "Subsample same", "MI same", "Silhouette", "Margin"], [[x["name"], x["cluster_id"], f"{x['subsample_same_cluster_fraction']:.3f}",
        f"{x['imputation_same_cluster_fraction']:.3f}", f"{x['silhouette']:.3f}", f"{x['centroid_margin']:+.3f}"] for x in selected["uncertain_participants"]])
    p("Полные частоты всех37 объектов, матрицы совместного присутствия/членства37×37 и реализации повторов доступны в cluster_candidates_q3.json. Список выше не изменяет членство и не объявляет остальных безусловно устойчивыми. Не проверена независимость всех эпизодов, и не проведена независимая человеческая межэкспертная оценка. Общий тип источника, аудитория выступления и повторное описание одного события могут создавать ковариации без соответствующей латентной политической установки.")

    h("9. Российская тема и пределы политологического вывода")
    p("Шесть утверждённых российских семантических семей дают12 независимых свойств. В исходной матрице42×12 имеется14 известных интенсивностей из504; ни одна тематическая переменная не прошла прежний порог, в FA их0. Разреженный блок не заполнялся по должности, общим факторам или членству в группе. Критерий «фактор с долей ≥2% общей стандартизованной дисперсии» сейчас не проверяем: недостаточность измерения не равна отсутствию российского измерения и не равна его нулевой доле.")
    p("В предусмотренном протоколе потенциальный тематический фактор должен иметь |loading|≥0,35 по минимум трём различным семантическим семьям, долю≥0,02 от ВСЕЙ стандартизованной дисперсии и устойчивый смысл при сравнении q/вращений. Два индикатора — только условный кандидат; положительный и отрицательный полюс одной семьи не считаются двумя независимыми подтверждениями. Даже три семьи дают numeric_candidate с незавершённым аудитом независимости событий и специфичности темы, а не автоматически доказанный фактор. Требование не даёт права вращать оси ради желаемой причины.")
    p("Из расстояний и кластеров нельзя вывести личную симпатию к России, скрытые убеждения, общий электорат, причинную мотивацию, фактическую коалицию или полноту публичного политического пространства. Формулируются связи документированных оценок в выбранном корпусе. Объекты с бедным корпусом могут выглядеть менее интенсивными даже при корректно сохранённых unknown; аналитическая модель не устраняет этот предел измерения.")

    h("10. Воспроизведение и файлы")
    versions = {"Python": platform.python_version(), "NumPy": np.__version__, "pandas": pd.__version__, "SciPy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    t(["Компонент", "Версия фактического локального окружения"], list(versions.items()))
    p("Основной seed20261015. Версии официальных online stable-документов могут быть новее установленной среды; воспроизводящий источник истины — сохранённый код, параметры и версии выше. Скрипты ниже пишут только новые каталоги, не канонические рейтинги. Для Windows в данном workspace python — C:/Users/USER/progects/Politic_parties/venv/Scripts/python.exe. Команды из корня проекта; -B исключает pycache.")
    commands = [
        'python -B -X utf8 scripts/compare_rwanda_ward_q3.py --frozen-inputs "studies/rwanda-politicians-2021-2026/output/cluster_candidates_q3" --output-dir ".service_runtime/reproduce_ward_q3"',
        'python -B -X utf8 scripts/compare_rwanda_strict_flags.py --frozen-inputs "studies/rwanda-politicians-2021-2026/output/strict_without_flags" --output-dir ".service_runtime/reproduce_strict_flags"',
        'python -B -X utf8 scripts/compare_rwanda_imputation.py "studies/rwanda-politicians-2021-2026" --output-dir ".service_runtime/reproduce_imputation" --repeats 20 --mask-fraction .12 --posterior-draws 10 --q 2 3 4 --seed 20261015',
        'python -B -X utf8 -m unittest tests.test_ward_review -v']
    for command in commands:
        p(command)
    p("Для идентичных numerical results imputation-команды текущие scope/scales/ratings должны совпадать с сохранённым input_receipt. При поздней правке study.analysis его побайтовый SHA закономерно изменится, хотя матрица может остаться той же; для точного первоначального состояния используйте source_study/source_ratings. Численные снимки не подменяются новым каноническим study при повторном запуске Ward. Обучать факторы повторно для применения готовых scores к диаграммам не требуется.")
    p("При воспроизведении исходной импутации после изменения study создайте отдельный каталог .service_runtime/imputation_frozen_study, скопируйте туда cluster_candidates_q3/source_study.json под именем study.json и source_ratings.json под именем ratings.json, затем замените аргумент study_dir в указанной imputation-команде этим каталогом. Канонические файлы копировать поверх или откатывать не требуется. Старые пути в receipts могут отличаться; сопоставляются также значения хешей и точные object_order/trait_order.")
    p("Четыре целевых Ward-теста прошли: инвариантность к отражению/перестановке осей; условные знаменатели co-membership и перестановка меток; unknown не равен0 и centroid не переназначает Ward-членство; end-to-end точное совпадение с существующим posterior при пропусках и сохранение входов. Тестовые данные синтетические и не используются как эмпирические результаты Руанды.")
    files = [OUT / "cluster_candidates_q3.json", OUT / "cluster_candidates_q3/source_study.json", OUT / "cluster_candidates_q3/source_ratings.json", OUT / "cluster_candidates_q3/source_factor_preview.json",
        OUT / "cluster_candidates_q3/factor_parameters_q3.json", OUT / "cluster_candidates_q3/matrix_observed.csv", OUT / "cluster_candidates_q3/loadings_q3.csv", OUT / "cluster_candidates_q3/scores_raw.csv", OUT / "cluster_candidates_q3/scores_display.csv", OUT / "cluster_candidates_q3/labels_k4.csv", OUT / "cluster_candidates_q3/ward_linkage.csv",
        OUT / "imputation_review_2026-10-15.json", OUT / "strict_without_flags/COMPARISON.json", *archived,
        ROOT / "scripts/latent_space_benchmark.py", ROOT / "src/semantic_service/analysis.py", ROOT / "src/semantic_service/factor_preview.py", ROOT / "src/semantic_service/imputation_review.py", ROOT / "src/semantic_service/ward_review.py",
        ROOT / "scripts/compare_rwanda_ward_q3.py", ROOT / "scripts/compare_rwanda_strict_flags.py", ROOT / "scripts/compare_rwanda_imputation.py", ROOT / "scripts/write_rwanda_q3_technical.py", ROOT / "tests/test_ward_review.py"]
    receipts = {path.relative_to(ROOT).as_posix(): sha(path) for path in files}
    t(["Файл от корня проекта", "SHA-256"], list(receipts.items()))
    p("Файлы matrix_observed/scores/loadings/labels CSV имеют явные id и сохранённый порядок; JSON содержит полные имена, все observed/unknown и входную маску. scores_raw — F1/F2/F3; scores_display — X1=−F1,X2=F2,X3=F3. SHA фиксируют фактические байты этой сборки; Git-политика должна сохранять байты исследовательских снимков. Технический JSON не пересчитывает и не перезаписывает эти численные файлы.")

    h("Приложение: координаты всех 37 объектов")
    t(["Полное имя", "k4", "X1", "X2", "X3", "Известно из8"], [[x["name"], selected["labels_by_object"][x["id"]], *[f"{v:+.5f}" for v in x["display_scores"].values()], x["observed_properties"]] for x in ward["objects"]])
    for block in blocks:
        if block["type"] == "table":
            assert all(len(row) == len(block["headers"]) and all(isinstance(cell, str) for cell in row) for row in block["rows"])
    content = {"format_version": 1, "study": {"id": study["id"], "title": "Технический отчёт: Руанда, q3 / Ward k4", "date_from": study["date_from"], "date_to": study["date_to"]},
        "analysis": {"factor_count": 3, "cluster_count": 4, "n_objects":37, "n_scales":8, "variance_total": model["variance_total"]},
        "blocks": blocks, "reproducibility": {"writer_script": "scripts/write_rwanda_q3_technical.py", "source_sha256": receipts, "runtime_versions": versions,
            "source_data_modified": False, "numerical_models_refitted": False, "selected_k_source": "Parent analyst substantive review after all k2..6 profiles; not an automatic numerical recommendation"}}
    (RESEARCH / "TECHNICAL_Q3_CONTENT.json").write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (RESEARCH / "TECHNICAL_Q3_REPORT.md").write_text(markdown(blocks), encoding="utf-8")
    print(json.dumps({"blocks": len(blocks), "output": "research/rwanda_politicians_2021_2026/TECHNICAL_Q3_CONTENT.json", "markdown": "research/rwanda_politicians_2021_2026/TECHNICAL_Q3_REPORT.md", "all_tables_rectangular": True, "source_data_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
