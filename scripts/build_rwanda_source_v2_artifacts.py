"""Build a new, source-based reader report and a deck specification offline.

Inputs are reviewed public summaries, approved scale definitions and the frozen
biographical appendix. This script never reads the corpus or calls a model.
Document export is a separate explicit --export-pdf operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic_service.reports import heading as h, paragraph as p, table as t, markdown, html_document

RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"
DEFAULT_INPUT = ROOT / "reports/rwanda_source_reanalysis_2026_10_18.json"
DEFAULT_OUTPUT = ROOT / "studies/rwanda-politicians-2021-2026/output/source_v2_2026_10_18"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def chunks(values, size):
    return [values[index:index + size] for index in range(0, len(values), size)]


def _value(cell):
    value = cell.get("value")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _actor_id(cell):
    return cell.get("actor_id", cell.get("object_id"))


def _scale_id(cell):
    return cell.get("scale_id", cell.get("property_id"))


def _scenario(data, name):
    return data.get("scenarios", {}).get(name, {})


def reader_definition(scale):
    if not scale["id"].startswith("russia-"):
        return scale["definition"]
    family = scale["id"].rsplit("-", 1)[0]
    reverse = scale["id"].endswith("-negative")
    topics = {
        "russia-diplomatic-engagement": ("Развитие дипломатических связей с Россией.", "Ограничение дипломатических связей с Россией."),
        "russia-security-cooperation": ("Совместная работа с Россией в обороне и военной подготовке.", "Ограничение совместной работы с Россией в обороне и военной подготовке."),
        "russia-economic-technology-cooperation": ("Экономические и технологические проекты с российским партнёром.", "Ограничение экономических и технологических проектов с российским партнёром."),
        "russia-education-cultural-cooperation": ("Образование и культура, включая подготовку гражданских специалистов при сотрудничестве с Россией.", "Ограничение образовательных и культурных связей с Россией."),
        "russia-conflict-policy-evaluation": ("Содержательное оправдание конкретных внешнеполитических действий России.", "Содержательная критика конкретных внешнеполитических действий России."),
        "russia-isolation-sanctions": ("Противодействие санкциям и другим мерам международной изоляции России.", "Поддержка санкций и других мер международной изоляции России."),
    }
    return topics[family][int(reverse)] + " Учитываются собственные публичные слова и действия человека."


def prepare(data, profiles, dictionary):
    """Make all public formats from one consistent model, including unknowns."""
    strict = _scenario(data, "strict_dated")
    if any(scenario.get("factor_admission", {}).get("factor_models_estimated") is True
           for scenario in data.get("scenarios", {}).values()):
        raise ValueError("This narrative must be revised for an admitted factor model")
    cells = strict.get("cells", data.get("cells", []))
    if not cells:
        raise ValueError("The strict source-based person matrix is missing")
    actors = [dict(person) for person in profiles["profiles"]]
    for person in actors:
        if person["id"] == "valentine-uwamariya":
            person["historical_role"] += "; в проверенном выступлении 2026 года - сенатор"
    scales = dictionary["scales"]
    if len(actors) != 42 or len(scales) != 38:
        raise ValueError("The approved 42-person roster and 38-scale dictionary are required")
    by_person = {person["id"]: [] for person in actors}
    by_scale = {scale["id"]: [] for scale in scales}
    for cell in cells:
        if _value(cell) is not None:
            by_person[_actor_id(cell)].append(cell)
            by_scale[_scale_id(cell)].append(cell)
    known = sum(map(len, by_person.values()))
    covered = sum(bool(values) for values in by_person.values())
    scale_count = sum(bool(values) for values in by_scale.values())
    labels = {item["id"]: item["name"] for item in scales}
    total = len(actors) * len(scales)
    loose = _scenario(data, "content_supported_unknown_time_sensitivity")
    loose_cells = loose.get("cells", [])
    loose_known = sum(_value(cell) is not None for cell in loose_cells)
    loose_actors = len({_actor_id(cell) for cell in loose_cells if _value(cell) is not None})
    counts = {"known_cells": known, "unknown_cells": total - known,
              "persons_with_scores": covered, "persons_without_scores": len(actors) - covered,
              "scales_with_scores": scale_count, "total_cells": total}
    base = [h("Руанда: политические акценты в проверенных источниках", 0),
            p("Обновление от 18 октября 2026 года. Новая версия по утверждённому словарю из 38 самостоятельных признаков. "
              "Выборка включает 42 человека. Основной массив публикаций охватывает октябрь 2025 - октябрь 2026 года, "
              "поэтому эта версия описывает подтверждённые публичные эпизоды и не заменяет полную историю пяти лет."),
            h("Основные выводы"),
            p(f"После проверки источников в основной матрице есть {known} содержательных оценок у {covered} из 42 участников. "
              f"Они относятся к {scale_count} из 38 признаков. Для остальных сочетаний человека и признака сохранено «нет данных». "
              "Частое упоминание имени не делает позицию более выраженной."),
            p("В проверенных высказываниях видны разные политические задачи: развитие государственных возможностей и технологий, "
              "защита прав и процедур, а также международное сотрудничество для решения конкретных задач. "
              "Это содержательные линии отдельных эпизодов. Их распространённость по всей политической системе пока нельзя оценить."),
            p("Новая матрица пока не позволяет надёжно построить общую факторную карту и статистические группы. "
              "Прежнее трёхфакторное решение остаётся отдельной исследовательской гипотезой. Его координаты и кластеры "
              "не переносятся автоматически на новые определения и более строго проверенные данные."),
            h("Что меняет неопределённость дат", 2),
            t(["Набор проверенных эпизодов", "Известных оценок", "Участников с оценками"],
              [["Основная матрица: допустимые датированные эпизоды", known, covered],
               ["Дополнительно эпизоды с неуточнённым временем", loose_known, loose_actors]]),
            p("Вторая строка показывает, сколько содержательных оценок доступно, если отдельно рассмотреть неопределённость времени. "
              "Она не устраняет эту неопределённость и не заменяет основную матрицу. Оба набора пока не дают достаточной основы для общей статистической карты.")]
    findings = [
        ("Развитие возможностей и технологий", [
            "В проверенных эпизодах Паула Ингабире связывает технологические партнёрства с самостоятельностью Руанды. "
            "Здесь важна способность выбирать условия сотрудничества и уменьшать зависимость, а не простое присутствие иностранных компаний.",
            "Клавер Гатете подчёркивает внутренние возможности африканских стран, выступая в рамках международного мандата. "
            "Его высказывание нельзя автоматически представлять как позицию правительства Руанды."],
         ["U24", "U57", "U60", "E12"]),
        ("Права, процедуры и политическое участие", [
            "В материалах об оппозиции важны собственные политические требования и действия: правовые возражения и ненасильственные способы участия. "
            "Сам факт обвинения, ареста или исключения из выборов не доказывает программу человека.",
            "Судебная позиция представителей Виктуар Ингабире относится к конкретному делу. Её можно описывать как процессуальное требование, "
            "но нельзя расширять до полного набора убеждений. Высказывания Франка Хабинезы дают самостоятельные примеры ненасильственного участия."],
         ["U18", "U20", "U59"]),
        ("Международные связи как способ решения задач", [
            "Сотрудничество с внешними партнёрами появляется в разных сферах: медицинские технологии, оборона, возвращение беженцев и региональные переговоры. "
            "Одинаковый инструмент ещё не означает одинаковую идеологию его участников.",
            "Краткий контакт на форуме или протокольная фотография подтверждают присутствие и встречу. Для вывода о предпочтениях "
            "нужны содержательное заявление, действие или обязательство самого политика."],
         ["U19", "U26", "U30", "U40", "U44"]),
        ("Единство общества и предупреждение вреда", [
            "В выступлениях Жан-Дамасена Бизиманы и Франсуа-Ксавье Калинды историческая память связана с предотвращением новых расколов. "
            "Это конкретная линия национального единства. Она не означает, что остальные участники ей противостоят.",
            "Валентин Увамария предлагает перепись диких животных и сопоставление их численности с возможностями парков, "
            "чтобы уменьшить нападения и ущерб. Здесь видны проверяемое знание и предупреждение рисков. "
            "Сама тема животных ещё не доказывает поддержку экологических ограничений хозяйственной деятельности."],
         ["E14", "E17", "E21"]),
    ]
    base += [h("Какие политические линии видны в материалах"),
             p("В этом разделе используются качественно проверенные эпизоды, в том числе материалы с неуточнённой датой действия. "
               "Последние не входят в основную датированную матрицу. Примеры помогают понять содержание признаков, "
               "но не доказывают устойчивость позиции на протяжении всего периода.")]
    for title, paragraphs, cases in findings:
        base.append(h(title, 2))
        base.extend(p(text) for text in paragraphs)
        base.append(p("Проверенные эпизоды: " + ", ".join(cases) + ". Краткие пересказы и ограничения приведены в отдельном журнале проверки."))
        base.append({"type": "link", "text": "Журнал содержательной проверки источников", "url":
                     "https://github.com/rdv77/political_psycology_stratification/blob/main/" +
                     ("reports/rwanda_chat_combined_review_2026_10_18.json" if any(case.startswith("E") for case in cases)
                      else "research/rwanda_politicians_2021_2026/source_scales_v2/CHAT_REVIEW_2026-10-17.md")})
    russia = [
        "Сабин Нсанзимана связан с конкретным эпизодом медицинского сотрудничества с Россией: обсуждаются технологии, "
        "подготовка гражданских медицинских кадров и профилактика. Перепечатки одного соглашения образуют один эпизод.",
        "У Жювеналя Маризамунды проверенные материалы описывают обсуждение военно-технического взаимодействия. "
        "Они подтверждают направление сотрудничества, но не доказывают исполнение всех обсуждавшихся проектов.",
        "У Оливье Ндухунгирехе содержательная двусторонняя повестка позволяет говорить о дипломатическом и экономическом взаимодействии. "
        "Эти эпизоды не устанавливают его позицию по каждому внешнеполитическому действию России.",
        "Из отраслевого сотрудничества нельзя вывести поддержку войны, санкционную позицию или единую ориентацию всех руководителей страны. "
        "Отдельный общий фактор отношения к России и порог 2% дисперсии по этой матрице пока не проверяются: данных недостаточно для допустимой модели.",
    ]
    base += [h("Отношение к России: подтверждённые направления"),
             p("Часть приведённых ниже эпизодов содержательно подтверждена, но имеет неуточнённую дату действия. "
               "Она рассматривается отдельно от основной матрицы и не используется для построения её факторных осей.")]
    base.extend(p(text) for text in russia)
    base += [h("Сопоставление с предыдущим исследованием"),
             t(["Решение", "Предыдущее исследование", "Новая матрица V2"],
               [["2 фактора", "39,73% общей дисперсии", "Не оценивается"],
                ["3 фактора", "52,29% общей дисперсии", "Не оценивается"],
                ["4 фактора", "61,37% общей дисперсии", "Не оценивается"]]),
             p("Проценты относятся к прежней матрице из 37 участников и 8 переменных. Они не показывают качество новой матрицы "
               "и не дают основания сохранить прежние группы. Обновлённые определения, правило подтверждения позиции и работа с датами "
               "меняют само измерение, поэтому численное сравнение требует достаточной новой матрицы."),
             p("Объединение Кагаме и Дианы Рвигары в прежнем кластере после этой проверки не получает нового подтверждения. "
               "Это не означает доказанного расхождения: для сопоставимого профиля Рвигары пока не хватает содержательных данных."),
             h("Что означают оценки"),
             p("Каждый из 38 признаков оценивается самостоятельно от 0 до 5. Балл описывает силу позиции или действия в конкретном проверенном эпизоде. "
               "Количество публикаций учитывается отдельно. Отсутствие сообщения о позиции сохраняется как неизвестное значение. "
               "Признаки сотрудничества с Россией и дистанцирования от неё также независимы."),
             h("Все утверждённые признаки")]
    for block, title in [("political_goals", "Политические цели"), ("political_methods", "Способы действия"), ("russia_relations", "Россия")]:
        selected = [s for s in scales if s.get("block") == block]
        if not selected:
            selected = [s for s in scales if (block == "political_methods" and s["id"].startswith("m")) or
                        (block == "russia_relations" and s["id"].startswith("russia-"))]
        base += [h(title, 2), t(["Признак", "Что оценивается"], [[s["name"], reader_definition(s)] for s in selected])]
    base += [h("Участники и подтверждённые оценки"),
             p("Справки описывают роль человека в исследуемый период. Они не подтверждают сохранение должности на дату этого отчёта. "
               "Наличие публикаций, участие в церемонии и биографические сведения сами по себе не создают оценок политических признаков.")]
    for person in actors:
        values = by_person[person["id"]]
        base += [h(person["name"], 2), p(person["historical_role"]), p(person["brief_profile"])]
        if values:
            base.append(t(["Подтверждённый признак", "Балл 0-5"],
                          [[labels[_scale_id(c)], f"{_value(c):g}"] for c in values]))
        else:
            base.append(p("В основной матрице пока нет подтверждённых оценок. Неизвестные значения не заменены нулями или средними."))
        for citation in person.get("sources", [])[:2]:
            if citation.get("url"):
                base.append({"type": "link", "text": citation.get("title") or "Биографический источник", "url": citation["url"]})
    base += [h("Материалы для проверки"),
             p("Полная матрица всех 42 участников, отдельный технический отчёт, журнал проверки эпизодов и сравнение допусков "
               "доступны рядом с этим отчётом. Исходные публикации и частные идентификаторы остаются в локальном хранилище проекта.")]
    scale_rows = [[scale["id"], scale["name"], len(by_scale[scale["id"]])] for scale in scales]
    technical = [h("Технический отчёт: повторная обработка источников V2", 0),
                 p("Дата: 18 октября 2026 года. Новый программный проход работает с сохранёнными результатами, без платных API-запросов."),
                 h("Измерительная модель"),
                 p("42 утверждённых участника и 38 утверждённых однополюсных переменных. Все переменные рассматриваются независимо. "
                   "Частота упоминаний и интенсивность свойства отделены. 0 требует явного содержательного основания, "
                   "неизвестное остаётся null. Старая матрица не используется как источник новых оценок."),
                 p("После проверки содержания, атрибуции, времени и объединения повторов оценка человека по признаку равна медиане "
                   "допустимых независимых эпизодов. У каждого эпизода одинаковый вес. Перепечатки и несколько фрагментов одного события "
                   "не получают дополнительного веса. Число эпизодов, число текстов и балл хранятся раздельно. "
                   "Утверждение нескольких характеристик одного эпизода не превращает его в несколько независимых событий."),
                 p("Для отдельного сценария неопределённого времени действует дополнительное ограничение. Когда нельзя подтвердить, "
                   "что недатированные фрагменты описывают разные события, совпадающие баллы можно сохранить: результат не меняется "
                   "от веса повторов. Это не означает доказанной независимости событий. При разных баллах и неустановленном различии "
                   "эпизодов итоговая ячейка остаётся неизвестной."),
                 h("Проверки атрибуции, времени и независимости"),
                 p("Нормализация различает собственное заявление, действие, позицию представителя в конкретном деле, высказывание третьего лица, "
                   "протокол и контекст. Смена должности не доказывает тёзку. Конфликт имени в заголовке и теле требует проверки. "
                   "Дата публикации, дата действия, дата произнесения и исторический предмет высказывания хранятся отдельно."),
                 p("Повторный текст, перевод и сообщение о том же событии не создают независимых эпизодов. Для переноса ручного решения "
                   "нужны проверенное совпадение источника и соответствующая атрибуция. Ключевые слова сами по себе не назначают балл."),
                 h("Основная матрица и чувствительность к неопределённому времени"),
                 p(f"Строгая матрица: {known} известных из {total} ячеек ({known / total:.2%}); {covered} участников с оценками. "
                   "Основной сценарий использует допустимые датированные эпизоды. Отдельный сценарий допускает содержательно подтверждённые "
                   "эпизоды с неопределённым временем и не смешивается с основной матрицей."),
                 t(["Сценарий", "Известных ячеек", "Покрытие", "Участников с оценками"],
                   [["Датированные эпизоды", known, f"{known/total:.2%}", covered],
                    ["Чувствительность к неуточнённому времени", loose_known, f"{loose_known/total:.2%}", loose_actors]]),
                 t(["Сценарий", "Первичных оценок", "После объединения событий", "Итоговых личных ячеек", "Удержано из-за неопределённой независимости"],
                   [[label, scenario["counts"]["numeric_case_ratings_before_episode_deduplication"],
                     scenario["counts"]["episode_property_values_after_deduplication"],
                     scenario["counts"]["observed_cells"], scenario["counts"]["cells_held_for_event_independence"]]
                    for label, scenario in [("Датированные", strict), ("Неуточнённое время", loose)]]),
                 h("Правила допуска факторного анализа"),
                 p("На каждом шаге сохраняются непостоянные переменные с известными значениями минимум у max(8, ceil(0,60*N)) участников. "
                   "Затем сохраняются участники с известными значениями минимум по max(4, ceil(0,60*P)) переменным. Фильтрация повторяется. "
                   "Финально требуется минимум 12 участников, 6 переменных и 75% известных ячеек. Проверка допустимого числа факторов выполняется отдельно."),
                 p(f"При исходных 42 участниках требуется минимум 26 известных оценок по переменной. В основной матрице "
                   f"наиболее покрытая переменная имеет {max(map(len, by_scale.values()))} оценок. Эти пороги не снижались после получения результата."),
                 p("Заполнение средними или регрессией до проверки допуска создало бы видимость измерения. Оно не выполнялось. "
                   "Решения q=2, q=3 и q=4 в новой матрице имеют статус «не оценены», если допуск не пройден. "
                   "В этом случае нагрузки, проценты дисперсии, факторные координаты, t-SNE и новые кластеры не вычисляются."),
                 p("Для регрессии нужна обучающая выборка с реальными совместно измеренными признаками. В этих данных неизвестность "
                   "часто связана с неравным вниманием прессы, отсутствием собственного высказывания и недостаточным контекстом. "
                   "Это не случайно потерянные отдельные числа из заполненной анкеты. Подстановка среднего сближала бы людей "
                   "с недостатком сведений. Регрессия могла бы воспроизводить роль или представленность в прессе вместо политической позиции."),
                 p("Многократное заполнение можно рассматривать позже как анализ чувствительности для небольшой доли пропусков "
                   "при достаточной исходной совместной измеренности. Сценарий с неопределённым временем уже показывает "
                   "отдельную границу уверенности и не является статистическим заполнением."),
                 h("Покрытие всех переменных"), t(["ID", "Переменная", "Участников с оценкой"], scale_rows),
                 h("Машинный протокол"),
                 p("Полные результаты допусков и сценариев сохранены в JSON-источнике этой сборки. Программа не изменяет старую матрицу, "
                   "сырые ответы API и счётчик исторических расходов. Новые расходы API в этой операции: $0.")]
    slide = lambda title, paragraphs, **kwargs: {"kind": "prose", "title": title, "paragraphs": paragraphs, **kwargs}
    pages = [{"kind": "cover", "title": "Политические акценты Руанды", "subtitle": "Проверенные источники и обновлённые оценки",
              "body": "42 участника\n38 самостоятельных признаков\n18 октября 2026 года"},
             slide("Основные выводы", [f"У {covered} из 42 участников есть {known} подтверждённых оценок по {scale_count} признакам.",
                 "В материалах видны задачи развития, защиты процедур и конкретного международного сотрудничества.",
                 "Для новой общей карты и статистических групп пока недостаточно сопоставимых сведений."]),
             {"kind": "chart", "title": "Достаточность сведений об участниках", "chart": {"type": "bar", "config": {
                 "categories": ["Есть подтверждённые оценки", "Пока нет оценок"],
                 "series": [{"name": "Участников", "values": [covered, len(actors) - covered], "fill": "#176D77"}],
                 "barOptions": {"direction": "bar"}, "hasLegend": False,
                 "dataLabels": {"showValue": True, "position": "outEnd"}}},
              "caption": "Отсутствие оценки означает недостаток проверенных сведений о позиции человека."},
             {"kind": "chart", "title": "Влияние неопределённых дат", "chart": {"type": "bar", "config": {
                 "categories": ["Основная матрица", "С неуточнёнными датами"],
                 "series": [{"name": "Подтверждённых оценок", "values": [known, loose_known], "fill": "#176D77"}],
                 "barOptions": {"direction": "bar"}, "hasLegend": False,
                 "dataLabels": {"showValue": True, "position": "outEnd"}}},
              "caption": "Число оценок. «С неуточнёнными датами» включает дополнительные эпизоды и остаётся отдельным сценарием."}]
    for title, paragraphs, cases in findings:
        pages.append(slide(title, paragraphs, notes="Проверенные эпизоды: " + ", ".join(cases), fontSize=28,
                           caption="Качественные примеры включают эпизоды с неуточнённой датой действия."))
    pages += [slide("Россия: отраслевые связи", russia[:2], notes="Проверенные эпизоды: U26, U30, U41, U42, U43",
                    caption="Даты части эпизодов не уточнены. Они рассматриваются отдельно от основной матрицы."),
              slide("Россия: границы интерпретации", russia[2:], notes="Проверенный эпизод: U40"),
              slide("Общая карта пока остаётся открытым вопросом", [
                  "Прежние три оси и четыре группы относятся к более ранней матрице и другим определениям признаков.",
                  "Новая проверка не подтверждает прежнее объединение Кагаме и Дианы Рвигары. Для сопоставления пока недостаточно сведений о Рвигаре.",
                  "Содержательные темы на предыдущих слайдах описывают проверенные эпизоды. Статистические кластеры по ним пока не устанавливаются."])]
    examples = []
    for person in actors:
        values = by_person[person["id"]]
        if values:
            chosen = max(values, key=lambda cell: (_value(cell), _scale_id(cell) != "m10"))
            examples.append([person["name"], labels[_scale_id(chosen)], f"{_value(chosen):g}"])
    for index, batch in enumerate(chunks(examples, 5)):
        pages.append({"kind": "table", "title": f"Подтверждённые признаки участников: {index+1}",
                      "headers": ["Участник", "Пример выраженного признака", "0-5"], "rows": batch,
                      "widths": [310, 720, 118], "caption": "По одному примеру на человека из основной матрицы. Балл не зависит от числа публикаций."})
    for prefix, title in [("g", "Политические цели"), ("m", "Способы политического действия"), ("russia-", "Отношение к России")]:
        subset = [scale for scale in scales if scale["id"].startswith(prefix)]
        for index, batch in enumerate(chunks(subset, 7 if prefix != "russia-" else 6)):
            pages.append({"kind": "scale_list", "title": f"{title}: {index + 1}",
                          "items": [{"name": scale["name"], "id": scale["id"]} for scale in batch],
                          "caption": "Каждый признак оценивается самостоятельно. Противоположная позиция не выводится из молчания."})
    for index, batch in enumerate(chunks(actors, 3)):
        pages.append({"kind": "profiles", "title": f"Участники исследования: {index + 1} из 14", "profiles": [
            {"name": person["name"], "role": person["historical_role"],
             "coverage": f"В основной датированной матрице: {len(by_person[person['id']])} из 38 признаков",
             "sources": [source["url"] for source in person.get("sources", []) if source.get("url")]} for person in batch],
             "caption": "Должности приведены в контексте исследуемого периода."})
    pages.append(slide("Что нужно для общей политической карты", [
        "Сопоставимые собственные высказывания и действия для участников, о которых пока известно мало.",
        "Материалы, которые содержательно обсуждают утверждённые признаки и позволяют различить выраженную позицию, её отсутствие и нехватку сведений.",
        "После накопления таких оценок можно сравнить оси и группы, не превращая пропуски в сходство."]))
    evidence_plot = {"kind": "source_coverage", "title": "Покрытие участников проверенными оценками",
                     "names": [person["name"] for person in actors],
                     "values": [len(by_person[person["id"]]) for person in actors]}
    return {"counts": counts, "blocks": base, "technical_blocks": technical,
            "slides": pages, "actor_count": 42, "scale_count": 38, "analysis_status": "not_admitted",
            "plots": [evidence_plot]}


def render_report_charts(model, output):
    """Scientific data plots, never decorative assets or invented factor maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot = model["plots"][0]
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    figures = []
    for index, offset in enumerate(range(0, 42, 14)):
        names, values = plot["names"][offset:offset+14], plot["values"][offset:offset+14]
        fig, axis = plt.subplots(figsize=(13, 8))
        axis.barh(names[::-1], values[::-1], color="#176D77")
        axis.set_xlim(0, 38)
        axis.set_xlabel("Количество признаков с подтверждённой оценкой из 38", fontsize=13)
        axis.tick_params(axis="y", labelsize=13)
        axis.tick_params(axis="x", labelsize=12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.grid(True, alpha=.18)
        axis.set_axisbelow(True)
        for row, value in enumerate(values[::-1]):
            axis.text(value + .35, row, str(value), va="center", fontsize=12)
        fig.subplots_adjust(left=.36, right=.96, top=.95, bottom=.12)
        filename = f"assets/actor_coverage_{index+1}.png"
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
        figures.append({"type": "image", "file": filename,
                        "caption": f"Покрытие участников, часть {index+1} из 3. Длина полосы показывает количество подтверждённых признаков. "
                                   "Она не измеряет выраженность позиции, известность или влияние человека."})
    marker = next(i for i, block in enumerate(model["blocks"]) if block.get("text") == "Участники и подтверждённые оценки")
    model["blocks"][marker:marker] = [h("Какие сведения доступны по участникам"), *figures]


def add_correction_protocol(model, correction, data):
    """Describe full mechanical coverage separately from semantic adjudication."""
    counts = correction["counts"]
    blocks = [h("Объём повторной обработки"),
              t(["Операция", "Количество"], [
                  ["Все ранее извлечённые факты, через которые прошли общие проверки", counts["all_extracted_facts_processed"]],
                  ["Индивидуально разобранные случаи", counts["exact_reviewed_cases_applied"]],
                  ["Факты с точными решениями содержательной проверки", counts["exact_reviewed_facts_applied"]],
                  ["Факты с изменением исходных измерительных полей", counts["facts_with_changed_measurement_fields"]],
                  ["Факты с добавлением или изменением метаданных проверки", counts["facts_with_added_or_changed_review_metadata"]],
                  ["Факты только с общими программными проверками", counts["general_checks_only_facts"]],
                  ["Факты с предупреждениями для дальнейшей проверки", counts["facts_with_general_risk_flags"]],
                  ["Кандидатные факты до повторной проверки", counts["eligible_before"]],
                  ["Кандидатные факты после повторной проверки", counts["eligible_after"]],
                  ["Группы буквального совпадения фрагментов", counts["literal_material_groups"]],
                  ["Группы с буквальными повторами", counts["literal_repeat_groups"]],
                  ["Избыточные элементы буквальных групп", counts["literal_repeat_excess"]],
              ]),
              p("Программный проход охватывает все сохранённые извлечённые факты, но не заменяет содержательное чтение каждого из них. "
                "Формулировки о протоколе, пассивном участии и обвинениях дают предупреждение, а не автоматическое заключение о смысле. "
                "Точные изменения атрибуции, времени и допустимости применяются только к индивидуально разобранным фактам. "
                "Кандидатный факт может быть материалом для последующего оценивания и ещё не является числовым баллом."),
              p(f"В {counts['facts_with_changed_measurement_fields']} фактах изменены исходные измерительные поля. "
                f"В {counts['facts_with_added_or_changed_review_metadata']} записях добавлены или изменены явные поля проверки и времени. "
                "Вторая цифра не является числом смысловых ошибок и не означает, что модель заново содержательно оценила все тексты. "
                "Буквальная группа фрагментов также не равна проверенному независимому событию."),
              h("Происхождение и сохранность входов"),
              t(["Вход или версия", "Версия или SHA-256"], [[key, value] for key, value in correction.get("provenance", {}).items()]),
              p("Частные тексты, цитаты и внутренние идентификаторы не включаются в публичные отчёты. Сырые ответы, база, очередь заданий "
                "и каноническое предыдущее исследование остаются исходными входами. Исправленные записи образуют отдельный производный слой.")]
    model["technical_blocks"][2:2] = blocks
    model["blocks"][3:3] = [p(f"Общие проверки конвейера применены ко всем {counts['all_extracted_facts_processed']:,} ранее извлечённым фактам. "
                              f"Точные содержательные решения основаны на {counts['exact_reviewed_cases_applied']} разобранных случаях. "
                              "Эти объёмы различаются: программное предупреждение ещё не подтверждает политическую характеристику.")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--export-pdf", action="store_true")
    parser.add_argument("--corrections", type=Path,
                        default=ROOT / "reports/rwanda_corrected_evidence_2026_10_18.json")
    parser.add_argument("--export-python", default=os.environ.get("SEMANTIC_EXPORT_PYTHON") or
                        "C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe")
    args = parser.parse_args()
    source = read(args.input)
    dictionary = read(RESEARCH / "source_scales_v2/approved_scales.json")
    model = prepare(source, read(RESEARCH / "ACTOR_PROFILES.json"), dictionary)
    if args.corrections.is_file():
        add_correction_protocol(model, read(args.corrections), source)
    args.output.mkdir(parents=True, exist_ok=True)
    render_report_charts(model, args.output)
    for name, blocks in [("report", model["blocks"]), ("technical/report", model["technical_blocks"])]:
        prefix = args.output / name
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".md").write_text(markdown(blocks), encoding="utf-8")
        prefix.with_suffix(".html").write_text(html_document(blocks), encoding="utf-8")
        content = {"study": {"title": "Руанда: проверенные источники V2"}, "blocks": blocks}
        write(prefix.parent / "report_content.json", content)
        if args.export_pdf:
            worker = ("import json,sys;from pathlib import Path;"
                      "from src.semantic_service.export_documents import _export_pdf;"
                      "source=Path(sys.argv[1]);"
                      "_export_pdf(json.loads(source.read_text(encoding='utf-8')),source.parent,Path(sys.argv[2]))")
            subprocess.run([args.export_python, "-B", "-c", worker,
                            str(prefix.parent / "report_content.json"), str(prefix.parent)],
                           cwd=ROOT, check=True, timeout=180)
    spec = {"version": "rwanda-source-v2-reader-deck-1", "created_on": "2026-10-18", "analysis_status": model["analysis_status"],
            "slides": model["slides"], "actor_count": 42, "scale_count": 38,
            "source_json": args.input.relative_to(ROOT).as_posix(),
            "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}
    write(args.output / "presentation_content.json", spec)
    print(json.dumps({"counts": model["counts"], "slides": len(model["slides"]), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
