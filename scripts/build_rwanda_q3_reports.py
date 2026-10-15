"""Build reviewed political/technical reports and editable-slide data offline.

Sources and ratings are immutable inputs. No collection, scoring or paid calls.
Run after compare_rwanda_ward_q3.py and the separate narrative reviews.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from scripts.build_q4_presentation_charts import _labels
from src.semantic_service.reports import heading as h, paragraph as p, table as t, markdown, html_document
from src.semantic_service.store import StudyStore, ratings_basis, fingerprint, now

ID = "rwanda-politicians-2021-2026"
RESEARCH = ROOT / "research/rwanda_politicians_2021_2026"
OUTPUT = ROOT / "studies" / ID / "output"
COLORS = {"C1": "#B56A19", "C2": "#147B79", "C3": "#8250A7", "C4": "#3E6A9B"}
AXES = {"X1": "Преемственность и поддержка курса", "X2": "Обновление и обсуждение проблем",
        "X3": "Социальные задачи и профилактика"}
TRAITS = ["Обновление программ", "Продолжение курса", "Поддержка курса", "Поручения и контроль",
          "Память и ценности", "Доступность благ", "Раскрытие проблем", "Предупреждение рисков"]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source(url, title="Источник"):
    return {"type": "link", "url": url, "text": title}

def wrap(text, width=38):
    return textwrap.fill(str(text), width)

def figure(path, points, names, labels, axes, title, trait=False, loading_map=True):
    fig, ax = plt.subplots(figsize=(13.6, 9.0))
    fig.subplots_adjust(left=.12, right=.97, top=.85, bottom=.15)
    ax.spines[["top", "right"]].set_visible(False)
    if trait:
        ax.scatter(points[:, 0], points[:, 1], s=52, c="#147B79", zorder=4)
        if loading_map:
            ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35))
        else:
            ax.margins(x=.34, y=.32)
    else:
        for cid in COLORS:
            present = np.array(labels) == cid
            if present.any():
                ax.scatter(points[present, 0], points[present, 1], s=58, c=COLORS[cid],
                           label=f"{cid} ({present.sum()})", edgecolors="white", linewidth=.5, zorder=4)
        ax.margins(x=.34, y=.32)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.04), ncol=4, frameon=False, fontsize=12)
    ax.axhline(0, c="#CAD4DC", lw=.7)
    ax.axvline(0, c="#CAD4DC", lw=.7)
    ax.set_xlabel(wrap(axes[0], 74), fontsize=13, labelpad=15)
    ax.set_ylabel(wrap(axes[1], 40), fontsize=13, labelpad=15)
    fig.suptitle(title, fontsize=17, y=.97)
    ids = [str(i) for i in range(len(names))]
    placed = _labels(fig, ax, points, ids, names, ids, len(ids), font_size=12 if not trait else 13)
    # All points remain plotted; a separate catalogue identifies all people.
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"file": path.relative_to(OUTPUT).as_posix(), "label_count": len(placed), "point_count": len(names)}

def chart_spec(points, names, labels, axes, trait=False):
    groups = ["traits"] if trait else [c for c in COLORS if c in labels]
    series = []
    # Keep native labels readable without displacing the actual data points.
    xmin, xmax = float(points[:, 0].min()) - .65, float(points[:, 0].max()) + .65
    ymin, ymax = float(points[:, 1].min()) - .65, float(points[:, 1].max()) + .65
    priority_names = ["Кагаме", "Хабинеза", "Ингабире Умухоза", "Бизимана", "Фазиль Хареримана", "Нсанзимана",
                      "Нгиренте", "Ругвабиза", "Акаманзи", "Кабаребе", "Рвигара", "Мпайимана", "Мушикивабо"]
    priority = sorted(range(len(names)), key=lambda i: next((j for j, n in enumerate(priority_names) if n in names[i]), 100+i))
    occupied, placed = [], {}
    for i in priority:
        x = (float(points[i, 0])-xmin)/(xmax-xmin)*1000
        y = (float(points[i, 1])-ymin)/(ymax-ymin)*350
        width, height = len(names[i]) * (8 if trait else 6.5), 20 if trait else 17
        options = [("right", (x+9,y-height/2,width,height)), ("left", (x-9-width,y-height/2,width,height)),
                   ("top", (x-width/2,y+12,width,height)), ("bottom", (x-width/2,y-12-height,width,height))]
        for position, box in options:
            bx, by, bw, bh = box
            if bx<0 or bx+bw>1000 or by<0 or by+bh>350:
                continue
            if any(not (bx+bw+5<a or a+w+5<bx or by+bh+5<b or b+v+5<by) for a,b,w,v in occupied):
                continue
            placed[i] = position
            occupied.append(box)
            break
        if not trait and len(placed) >= 16:
            break
    # Final native-chart review: neighbouring points need opposite label sides.
    if trait and list(axes) == [AXES["X2"], AXES["X3"]]:
        placed[names.index("Предупреждение рисков")] = "left"
        placed[names.index("Доступность благ")] = "right"
    if not trait and list(axes) == ["Направление 1", "Направление 2"]:
        placed[names.index("Маризамунда")] = "bottom"
        placed[names.index("Рвигара")] = "top"
    for group in groups:
        indices = list(range(len(names))) if trait else [i for i, c in enumerate(labels) if c == group]
        series.append({"name": "Признаки" if trait else group,
                       "xValues": [round(float(points[i, 0]), 8) for i in indices],
                       "values": [round(float(points[i, 1]), 8) for i in indices],
                       "categories": [names[i] for i in indices],
                       "marker": {"symbol": "circle", "size": 7, "fill": "#147B79" if trait else COLORS[group]},
                       "line": {"fill": "none", "width": 0},
                       "dataLabels": {"showValue": False, "showCategoryName": False,
                                      "textStyle": {"fontSize": 16 if trait else 13, "fill": "#142B35"}},
                       "dataLabelOverrides": [{"idx": j, "text": names[i], "showValue": False,
                                               "showCategoryName": True, "position": placed[i]}
                                              for j, i in enumerate(indices) if i in placed]})
    def axis(label, column):
        limits = (-1.35, 1.35) if trait else (float(points[:, column].min()) - .65, float(points[:, column].max()) + .65)
        return {"title": {"text": label, "textStyle": {"fontSize": 19}},
                "min": limits[0], "max": limits[1], "numberFormatCode": "0.0",
                "textStyle": {"fontSize": 14, "fill": "#56656E"},
                "majorGridlines": {"fill": "#E3E9ED", "width": .6}}
    return {"type": "scatter", "config": {"series": series, "scatterOptions": {"style": "marker"},
            "hasLegend": not trait, "legend": {"position": "top", "textStyle": {"fontSize": 17}},
            "xAxis": axis(axes[0], 0), "yAxis": axis(axes[1], 1)}}

def main():
    store = StudyStore(ROOT / "studies")
    study = store.require_approved(ID)
    ratings = read(OUTPUT.parent / "ratings.json")
    if study["ratings_cache"]["basis"] != ratings_basis(study) or fingerprint(ratings) != study["ratings_cache"]["ratings_sha256"]:
        raise ValueError("Stale ratings or approvals")
    candidates_path = OUTPUT / "cluster_candidates_q3.json"
    candidates = read(candidates_path)
    if candidates.get("source_files_changed_concurrently"):
        raise ValueError("The candidate audit was run on changing inputs")
    if sha(OUTPUT.parent / "ratings.json") not in candidates["source_file_sha256"].values():
        raise ValueError("Cluster candidates belong to different ratings")
    chosen = copy.deepcopy(next(c for c in candidates["candidates"] if c["k"] == 4))
    prose = read(RESEARCH / "REPORT_Q3_NARRATIVE.json")
    cluster_prose = read(RESEARCH / "CLUSTER_Q3_POLITICAL_REVIEW.json")
    actors = read(RESEARCH / "ACTOR_PROFILES.json")["profiles"]
    context = read(RESEARCH / "POLITICAL_SYSTEM_CONTEXT.json")
    russia = read(RESEARCH / "POLITICAL_CONTEXT_RUSSIA.json")
    technical = read(RESEARCH / "TECHNICAL_Q3_CONTENT.json")
    if [a["id"] for a in actors] != [o["id"] for o in study["objects"]]:
        raise ValueError("The catalogue must preserve the approved 42-person roster")
    actor_map = {a["id"]: a for a in actors}
    factor_model = next(m for m in read(OUTPUT / "factor_preview.json")["models"] if m["q"] == 3)
    for factor in factor_model["factors"]:
        desc = next(f for f in prose["factors"] if f["id"] == factor["id"])
        factor.update(automatic_name=factor["name"], name=desc["short_name"], label=desc["short_name"], interpretation=desc["meaning"])
    # The reviewed semantic file is a separate layer over frozen numerical data.
    descriptions = {c["id"]: c for c in cluster_prose["clusters"]}
    for cluster in chosen["clusters"]:
        desc = descriptions[cluster["id"]]
        if set(desc["member_ids"]) != {m["id"] for m in cluster["members"]}:
            raise ValueError("A political interpretation names a different cluster membership")
        cluster.update(name=desc["name"], interpretation="\n\n".join(desc["paragraphs"]))
    model = {"factor_count": 3, "cluster_count": 4, "n_objects": candidates["n_objects"],
             "n_scales": candidates["n_variables"], "variance_total": factor_model["variance_total"],
             "factors": factor_model["factors"], "loading_table": factor_model["loading_table"],
             "objects": candidates["objects"], "clusters": chosen["clusters"],
             "object_assignments": chosen["object_assignments"], "display_axes": candidates["display_axes"],
             "cluster_candidates_file": "cluster_candidates_q3.json", "candidate_sha256": sha(candidates_path),
             "selection": {"factor_authority": "explicit_human_user_chat_2026_10_16", "q": 3, "k": 4,
                           "cluster_authority": "assistant_semantic_and_numerical_review",
                           "reason": cluster_prose["selection_reason"]},
             "uncertainty": chosen["stability"], "parameters": candidates["factor_model"]["parameters"],
             "ratings_sha256": sha(OUTPUT.parent / "ratings.json")}
    write(OUTPUT / "selected_q3/metadata.json", model)
    selected = OUTPUT / "selected_q3"
    for filename in ("factor_parameters_q3.json", "factor_model_q3.joblib", "loadings_q3.csv"):
        if (OUTPUT / "cluster_candidates_q3" / filename).is_file():
            shutil.copyfile(OUTPUT / "cluster_candidates_q3" / filename, selected / filename)
    objects = candidates["objects"]
    order = [o["id"] for o in objects]
    labels = [chosen["labels_by_object"][oid] for oid in order]
    points = np.array([[o["display_scores"][key] for key in ("X1", "X2", "X3")] for o in objects])
    loadings = np.array([[x["loadings"][key] for key in ("F1", "F2", "F3")] for x in factor_model["loading_table"]])
    loadings[:, 0] *= -1
    surnames = [a["name"].split(" ", 1)[-1] for a in objects]
    assets = OUTPUT / "assets/q3_political"
    assets.mkdir(parents=True, exist_ok=True)
    charts, slide_maps = [], []
    for first, second in [(0, 1), (1, 2), (0, 2)]:
        axes = [AXES[f"X{first+1}"], AXES[f"X{second+1}"]]
        for trait in (False, True):
            coords = (loadings if trait else points)[:, [first, second]]
            names = TRAITS if trait else surnames
            stem = f"{'traits' if trait else 'objects'}_x{first+1}_x{second+1}"
            title = "Признаки в смысловом пространстве" if trait else "Политики в смысловом пространстве"
            rec = figure(assets / (stem + ".png"), coords, names, labels, axes, title, trait)
            rec.update(title=title, axes=axes)
            charts.append(rec)
            if (first, second) in [(0, 1), (1, 2)]:
                slide_maps.append({"kind": "chart", "title": title, "chart": chart_spec(coords, names, labels, axes, trait),
                    "caption": "Точки обозначают свойства публичной речи. Каждая карта показывает две из трёх осей." if trait else
                    "Все 37 показаны точками, часть профилей совпадает. Полные составы и должности даны далее.",
                    "notes": f"Точные координаты: selected_q3/metadata.json. Число подписей статической карты {rec['label_count']}/{rec['point_count']}."})
    projections = {}
    with threadpool_limits(limits=1):
        for key, data in [("objects", points), ("traits", loadings)]:
            perplexity = float(min(12, (len(data) - 1) / 3))
            xy = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=20261015,
                      max_iter=1000, n_jobs=1).fit_transform(data)
            projections[key] = {"method": "t-SNE", "seed": 20261015, "perplexity": perplexity,
                                "init": "pca", "iterations": 1000, "input_dimensions": 3,
                                "coordinates": xy.tolist()}
            names = surnames if key == "objects" else TRAITS
            rec = figure(assets / f"neighbours_{key}.png", xy, names, labels,
                         ["Горизонтальное направление без политического смысла", "Вертикальное направление без политического смысла"],
                         "Дополнительная карта соседства", trait=(key == "traits"), loading_map=False)
            charts.append(rec)
            if key == "objects":
                slide_maps.append({"kind": "chart", "title": "Дополнительная карта соседства политиков",
                    "chart": chart_spec(xy, names, labels, ["Направление 1", "Направление 2"]),
                    "caption": "Оси этой карты не имеют политического смысла. Она помогает рассмотреть соседство профилей.",
                    "notes": json.dumps(projections[key], ensure_ascii=False)})
    write(selected / "projections.json", projections)
    pd.DataFrame([{**{"object_id": o["id"], "name": o["name"], "cluster_id": labels[i]}, **o["raw_scores"], **o["display_scores"]}
                  for i, o in enumerate(objects)]).to_csv(selected / "scores_and_clusters.csv", index=False, encoding="utf-8-sig")
    write(assets / "manifest.json", {"factor_count": 3, "cluster_count": 4, "charts": charts,
          "model_sha256": sha(selected / "metadata.json"), "reflection": "X1=-F1; X2=F2; X3=F3"})

    blocks = political_report(study, model, prose, cluster_prose, actors, context, russia, charts)
    payload = {"study": {**study, "title": "Политическое пространство Руанды"}, "analysis": model, "blocks": blocks}
    write(OUTPUT / "report_content.json", payload)
    (OUTPUT / "report.md").write_text(markdown(blocks), encoding="utf-8")
    (OUTPUT / "report.html").write_text(html_document(blocks), encoding="utf-8")
    tech_blocks = technical["blocks"]
    tech_dir = OUTPUT / "technical"
    tech_dir.mkdir(exist_ok=True)
    write(tech_dir / "report_content.json", {"study": {"title": "Технический отчёт исследования Руанды"}, "blocks": tech_blocks})
    (tech_dir / "report.md").write_text(markdown(tech_blocks), encoding="utf-8")
    (tech_dir / "report.html").write_text(html_document(tech_blocks), encoding="utf-8")
    spec = presentation(model, prose, cluster_prose, actors, context, russia, slide_maps)
    spec["source_sha256"] = {"selected_q3/metadata.json": sha(selected / "metadata.json"),
                             "selected_q3/projections.json": sha(selected / "projections.json")}
    write(OUTPUT / "presentation_content.json", spec)
    write(selected / "delivery_inputs.json", {"at": now(), "source_sha256": {
        path.relative_to(ROOT).as_posix(): sha(path) for path in [candidates_path, OUTPUT.parent / "ratings.json",
        *[RESEARCH / name for name in ["ACTOR_PROFILES.json", "POLITICAL_SYSTEM_CONTEXT.json", "POLITICAL_CONTEXT_RUSSIA.json",
                                      "REPORT_Q3_NARRATIVE.json", "CLUSTER_Q3_POLITICAL_REVIEW.json", "TECHNICAL_Q3_CONTENT.json"]]]}})
    print(json.dumps({"prepared": True, "q": 3, "k": 4, "slides": len(spec["slides"]), "report_words": len(markdown(blocks).split())}))


def reader_russia_limits(event):
    """Explain source boundaries to readers while retaining the audit separately."""
    paragraphs = {
        "kagame-soudan-interview-2023-09-12": [
            "В этой позиции принцип невторжения сочетается с заявленным нейтралитетом. Кагаме сам объясняет голосование Руанды, поэтому здесь есть собственное публичное суждение президента. Его отношение к санкциям и возможному ограничению двусторонних связей в этом ответе не раскрывается.",
            "Нейтралитет сохраняет самостоятельность политического выбора. При этом он не отменяет высказанного неприятия вторжения, а обращение к этому принципу само по себе не означает присоединения к западному политическому лагерю.",
        ],
        "kagame-benin-partner-choice-2023-04-17": [
            "Кагаме рассматривает Россию и Китай в более широкой логике самостоятельного выбора внешних партнёров африканскими государствами. Речь идёт о праве определять сотрудничество исходя из собственных потребностей.",
            "Выступление не сообщает о конкретном расширении российско-руандийских отношений и не раскрывает оценки вторжения в Украину или позиции по санкциям. Эти вопросы остаются отдельными направлениями внешней политики.",
        ],
        "kagame-message-via-nzabamwita-2026-01-15": [
            "Передача послания подтверждает действующий канал общения между президентами через дипломатического представителя. Поскольку содержание послания не опубликовано, его темы и политические оценки остаются неизвестными.",
        ],
        "biruta-russia-day-kigali-2023-06-12": [
            "В образовательной части назван конкретный механизм сотрудничества, тогда как большинство отраслевых направлений находится на стадии обсуждения. Заявленные интерес и готовность к работе ещё не подтверждают вложений, поставок или выполненных проектов. Все эти аспекты обсуждались на одном дипломатическом приёме.",
            "В материале представлены разные говорящие. Сообщение об удвоении стипендий и оправдание российской войны принадлежат послу Карену Чальяну. Публичные реплики Бируты описывают сотрудничество, поэтому соседство двух выступлений не делает позицию посла позицией руандийского министра.",
        ],
        "atlantic-council-rwanda-conversation-2022-03-30::biruta": [
            "Бирута обосновывает позицию принципами суверенитета и территориальной целостности, одновременно признавая самостоятельность других государств при голосовании. Это объяснение дипломатической позиции, без развёрнутой отдельной оценки российского руководства.",
            "Обсуждение экономических издержек показывает заботу о последствиях войны и санкций для Руанды. Само по себе оно не выражает ни требования снять санкции с России, ни поддержки их применения.",
        ],
        "nduhungirehe-sochi-2024-11-09": [
            "Визовая договорённость касается владельцев дипломатических и служебных паспортов, а не всех граждан. Подписание и интервью относятся к одному сочинскому форуму и показывают разные стороны той же поездки: оформление официальных контактов и обсуждение подготовки кадров.",
            "Упомянутая в статье договорённость 2019 года составляет предысторию сотрудничества. Она не подтверждает работу атомной станции, а высказывание министра на форуме описывает потребность в энергетике и подготовке специалистов.",
        ],
        "nduhungirehe-lavrov-cairo-2025-12-19": [
            "Содержание встречи известно из официального итогового сообщения. Оно подтверждает переговоры и заявленную взаимную готовность к сотрудничеству, но не воспроизводит полную персональную речь Ндухунгирехе. Приведённые в том же сообщении вступительные слова принадлежат Лаврову.",
        ],
        "nduhungirehe-lavrov-antalya-2026-04-18": [
            "Встреча продолжает двусторонний диалог и конкретизирует направления обсуждения. Перечисленные отрасли обозначают повестку сотрудничества; сам релиз не является перечнем подписанных контрактов.",
            "Готовность координировать работу на международных площадках не означает совпадения позиций по каждому голосованию, войне или санкциям. По этим отдельным вопросам сообщение подробностей не содержит.",
        ],
        "rwanda-russia-military-commission-2026-05-20": [
            "Заседание комиссии вели Каруретва и Пунчук, а Маризамунда участвовал в отдельном приёме российской делегации. Это разные роли внутри одного межгосударственного механизма оборонного сотрудничества.",
            "Подтверждены участие министра и предмет разговора. Конкретная поставка оружия, подписанный им договор и собственные слова в релизе не представлены. Точный день его приёма внутри периода 20–22 мая также не назван.",
        ],
        "nsanzimana-murashko-health-mou-2026-05-19": [
            "Меморандум объединяет медицинские технологии и подготовку гражданских медработников. Такое обучение отличается от военного, а ядерная медицина представляет отдельное направление сотрудничества, не тождественное строительству атомной электростанции.",
            "Содержание договорённости известно из ведомственного сообщения; собственной цитаты Нсанзиманы о подписании в нём нет. Все перечисленные направления относятся к одному документу. Его подписание подтверждает намерение сотрудничать, но не выполненные результаты.",
        ],
        "atlantic-council-rwanda-conversation-2022-03-30::mujawamariya": [
            "Муджавамария сама объясняет голосование Руанды, поэтому её публичная позиция подтверждается собственным ответом. Формулировка остаётся краткой и не содержит столь же прямого осуждения российского действия, как заявление главы OIF.",
            "Прежняя работа послом в России и Беларуси помогает понять её дипломатическую биографию. Она не устанавливает современную симпатию к России или участие в конкретном текущем проекте.",
        ],
        "mushikiwabo-oif-ukraine-2023-02-24": [
            "Заявление имеет институциональный характер и показывает публичную позицию Мушикивабо как генерального секретаря OIF. Оно не является заявлением правительства Руанды и не раскрывает частных убеждений, которые выходят за рамки её международной роли.",
            "Критика российского действия здесь не сопровождается позицией по санкциям или прекращению обменов. Содержание заявления также не устанавливает взгляды всех членов OIF и других участников исследования.",
        ],
    }
    event_key = event["event_id"]
    if event_key == "atlantic-council-rwanda-conversation-2022-03-30":
        speaker = "mujawamariya" if "q05" in event.get("quote_ids", []) else "biruta"
        event_key += "::" + speaker
    if event_key not in paragraphs:
        raise ValueError(f"Reader-facing Russia context is missing for {event_key}")
    return [p(text) for text in paragraphs[event_key]]


def political_report(study, model, prose, cluster_review, actors, context, russia, charts):
    blocks = [h("Политическое пространство Руанды", 0),
              p("Публичные акценты политиков и руководителей за 15 октября 2021 — 15 октября 2026 года. Политологический отчёт. Подготовлен 16 октября 2026 года."),
              h("Основные выводы")]
    conclusions = [
        "1. В публичной политике выбранных участников различаются три смысловые линии: преемственность и поддержка курса, обновление программ с обсуждением проблем, социальные задачи и профилактика в соотношении с ценностной аргументацией.",
        "2. Рабочая типология включает четыре группы: преемственность через ценности и память — 2 человека; социальные задачи и исполнение курса — 10; выраженные предложения обновления — 2; разнородные профили — 23. Это сочетания публичных акцентов, а не четыре партии или коалиции.",
        "3. Сохранение общей государственной линии может сочетаться с изменением конкретных программ. Поэтому предложения реформ не принадлежат исключительно оппозиции. Отраслевой пересмотр и требование политической альтернативы нужно различать по содержанию.",
        "4. Наибольшие формальные ресурсы власти сосредоточены в президентском центре. Кагаме и ряд руководителей исполнительной власти входят в разнородную группу C4. Их присутствие не делает остальных членов группы частью одного политического союза.",
        "5. Доступные высказывания об отношениях с Россией показывают сочетание отраслевого сотрудничества, самостоятельного выбора партнёров и критики конкретных внешнеполитических действий. Для понимания позиции важны человек, его должность и конкретный эпизод.",
        "6. Справочник охватывает 42 утверждённых участника, а карты — 37 достаточно описанных профилей. Выводы относятся к изученным публичным материалам. Размеры групп не показывают численность электората или распространённость убеждений среди населения."]
    blocks += [p(x) for x in conclusions]
    blocks += [h("Участники и границы выбранного пространства"),
               p("В исследование вошли президент и премьер-министры, министры и руководители ведомств, парламентские и местные руководители, кандидаты и оппозиционные фигуры. Отдельно включены три руандийца с международными мандатами. Состав охватывает разные публичные роли за пять лет и не является рейтингом популярности."),
               p("Для карты удалось сопоставить 37 участников. Инес Мпамбара, Бернар Нтаганда, Кристофер Каюмба, Поль Русесабагина и Фостен Каюмба Ньямваса остаются в справочнике, но пока не размещены на карте: по выбранному общему набору признаков им недостаёт сопоставимых материалов. Это не оценка их значимости.")]
    labels = {m["id"]: c["id"] for c in model["clusters"] for m in c["members"]}
    categories = list(dict.fromkeys(a["category"] for a in actors))
    blocks.append(t(["Круг участников", "В справочнике", "На карте"],
                    [[cat, sum(a["category"] == cat for a in actors), sum(a["category"] == cat and a["id"] in labels for a in actors)] for cat in categories]))
    blocks += [p("Валентин Ругвабиза выступает в материалах миссии ООН в ЦАР, Клавер Гатете — в том числе от Экономической комиссии ООН для Африки, Луиза Мушикивабо — от Франкофонии. Эти выступления показывают их работу в соответствующей роли. Их нельзя целиком приписывать правительству Руанды."),
               h("Политическая среда")]
    cs = {x["id"]: x for x in context["sources"]}
    for fact in context["facts"]:
        if fact["id"] in {"institutions", "election2024", "opposition_constraints", "memory_unity"}:
            blocks.append(p(fact["text"]))
            blocks += [source(cs[sid]["url"], cs[sid]["title"]) for sid in fact["source_ids"]]
    blocks += [p("Наличие формальных институтов и официальных результатов выборов нужно читать вместе с условиями участия в политике. Именно в этой среде одни люди объясняют сохранение курса, другие предлагают изменения внутри него, а третьи выступают за политическую альтернативу."),
               h("Какие признаки вошли в итоговую карту"),
               p("Исходный словарь включал 24 пары определений. Для общей карты сопоставимыми оказались восемь отдельных свойств. Каждое описывает публичное проявление, а не заранее заданный тип личности. Продолжение курса и предложение изменить программу могут встречаться у одного человека одновременно.")]
    meanings = ["Явное предложение пересмотреть приоритеты, правила или программы", "Призыв продолжить, закрепить или ускорить уже принятую линию",
                "Положительная публичная оценка действующего государственного курса", "Постановка задач, поручения и контроль исполнения",
                "Обоснование через память, единство и коллективные символы", "Конкретные меры расширения доступа к базовым услугам и благам",
                "Публичное обозначение недостатков и проблем исполнения", "Меры подготовки и снижения риска до наступления кризиса"]
    blocks.append(t(["Признак", "Что он означает"], [[x["name"], meanings[i]] for i, x in enumerate(model["loading_table"])]))
    blocks += [p("Другие свойства сохранены в словаре и индивидуальных материалах, но не определяют положение на общей карте. В частности, восемь общих признаков не дают самостоятельной оси отношения к России. Этот вопрос ниже рассматривается по конкретным высказываниям и действиям."),
               h("Три принципа различения публичных акцентов")]
    for i, factor in enumerate(prose["factors"]):
        blocks.append(h(factor["reader_title"], 2))
        blocks += [p(factor[key]) for key in ("meaning", "political_reading", "why_emerges")]
        for ref in prose["supplementary_context_sources"]:
            if (i in (1, 2) and ref["id"] == "NST2"):
                blocks.append(source(ref["url"], ref["title"]))
        quote = context["illustrative_quotes"][0 if i == 0 else 2 if i == 1 else 1]
        name = next(a["name"] for a in actors if a["id"] == quote["actor_id"])
        when = quote.get("event_date") or "публикация " + quote["publication_date"]
        blocks.append(p(f"Иллюстрация: {name}, {when}: «{quote['ru_translation']}» (перевод с английского). {quote['qualification']}"))
        blocks.append(source(quote["source_url"], "Источник цитаты и её контекст"))
    blocks += [h("Как читать карты признаков")]
    blocks += [p(x) for x in prose["map_explanation"]]
    for pair in ("traits_x1_x2.png", "traits_x2_x3.png"):
        rec = next(x for x in charts if x["file"].endswith(pair))
        blocks.append({"type": "image", "file": rec["file"], "caption": "Точки — восемь свойств. Подписи используют короткие названия из таблицы признаков. " + "; ".join(rec["axes"]) + "."})
    blocks += [h("Расположение политиков"),
               p("На следующих картах точка обозначает человека. Цвет показывает одну из четырёх групп по сочетанию всех трёх осей. Близость показывает сходство выбранных публичных акцентов. Она не устанавливает союз, дружбу, общую партийность или одинаковые возможности влиять на решения."),
               p("Некоторые сведения неизвестны; расчётное заполнение пропусков помогает построить карту, но не добавляет наблюдений. Числа на осях нужны для относительного сопоставления профилей. Нулевая координата не означает нейтралитета или отсутствия убеждений.")]
    for pair in ("objects_x1_x2.png", "objects_x2_x3.png", "objects_x1_x3.png"):
        rec = next(x for x in charts if x["file"].endswith(pair))
        blocks.append({"type": "image", "file": rec["file"], "caption": "Все 37 участников остаются на карте. " + "; ".join(rec["axes"]) + ". Полные имена и должности приведены в справочнике."})
    blocks += [h("Четыре группы публичных профилей"),
               t(["Группа", "Число", "Общая характеристика"], [[c["id"], c["count"], c["name"]] for c in model["clusters"]])]
    for cluster in model["clusters"]:
        desc = next(c for c in cluster_review["clusters"] if c["id"] == cluster["id"])
        blocks.append(h(cluster["id"] + " " + cluster["name"], 2))
        blocks += [p(x) for x in desc["paragraphs"]]
        blocks.append(p("Полный состав: " + ", ".join(m["name"] for m in cluster["members"]) + "."))
        for ref in desc.get("sources", []):
            blocks.append(source(ref["url"], ref.get("title", "Иллюстрация публичной позиции")))
    blocks += [h("Где сосредоточено политическое влияние"), p(cluster_review["influence_interpretation"])]
    blocks.append(source(cs["constitution_2023"]["url"], cs["constitution_2023"]["title"]))
    blocks += [h("Что структура говорит о политических тенденциях"),
               p("Во-первых, публичная поддержка курса и его практическая корректировка могут сосуществовать. Противопоставление сохранения власти и любых реформ слишком грубо описывает найденные сочетания. Для понимания Руанды полезно различать смену политических правил, отраслевое обновление и исполнение уже принятых целей."),
               p("Во-вторых, ценностное обоснование и забота о конкретных услугах составляют разные публичные акценты внутри одного пространства. Их различие частично связано с тем, какие задачи поручены человеку. Это позволяет описать разнообразие управленческой речи, но не даёт основания объявлять её носителей противоположными идеологическими лагерями."),
               p("В-третьих, большой разнородный круг C4 показывает пределы компактной карты. В неё попадают влиятельные руководители, парламентские деятели и отдельные критики власти. По одной близости их публичных профилей нельзя объяснить устройство всех политических отношений. Дополнительные признаки могут раскрыть различия, которые эти восемь свойств не отражают."),
               p("Почему возникли именно такие линии — политологическое объяснение на основе контекста, а не установленная причинная закономерность. Президентский центр, долгосрочное государственное планирование, память о геноциде и распределение служебных задач дают правдоподобную рамку. Проверка причин требует другого исследовательского дизайна.")]
    blocks += [h("Отношения с Россией в заявлениях и действиях политиков"),
               p("В доступных эпизодах сотрудничество с Россией сочетается с разными оценками её конкретных действий. Здесь важны направление работы и полномочия говорящего. Меморандум показывает договорённость, переговоры — обсуждение, а прямая речь — высказанный тезис. Эти свидетельства нельзя складывать в простой рейтинг симпатии к Москве.")]
    rs = {x["id"]: x for x in russia["sources"]}
    rq = {x["id"]: x for x in russia["quotes"]}
    used_quote = set()
    for profile in russia["profiles"]:
        blocks += [h(profile["name_ru"], 2), p(profile["role_at_events"] + ". " + profile["summary"])]
        for event in profile["events"]:
            when = event.get("date") or (event["date_from"] + " — " + event["date_to"])
            blocks.append(p(when + ". " + event["facts"]))
            for qid in event.get("quote_ids", []):
                if qid in used_quote:
                    continue
                quote = rq[qid]
                wording = quote["translation_ru"] or quote["original"]
                language = "перевод на русский" if quote["translation_ru"] else "текст источника на русском"
                blocks.append(p(quote["attribution"] + ": «" + wording + "» (" + language + "). " + quote["context"]))
                used_quote.add(qid)
            blocks += reader_russia_limits(event)
            blocks += [source(rs[sid]["url"], rs[sid]["title"]) for sid in event["source_ids"]]
    blocks += [p("У остальных участников в сопоставленном корпусе нет достаточных персональных оснований для отдельного описания отношения к России. Отсутствие найденного высказывания не означает нейтралитета. Позиция одного представителя не переносится на всю его группу."),
               h("Краткий справочник участников")]
    for category in categories:
        blocks.append(h(category, 2))
        for actor in actors:
            if actor["category"] != category:
                continue
            blocks += [h(actor["name"], 3), p(actor["original_name"] + ". " + actor["historical_role"] + ". " + actor["role_period"] + "."),
                       p(actor["brief_profile"] + " Основание включения: " + actor["inclusion_reason"])]
            if actor.get("international_mandate"):
                mandate = actor["international_mandate"]
                statement = ". ".join(str(mandate[key]).rstrip(".") for key in ("organization", "role", "date", "period", "note") if mandate.get(key))
                blocks.append(p("Международная роль: " + statement + "."))
            blocks.append(p("На карте: " + labels.get(actor["id"], "пока не размещён из-за недостатка сопоставимых материалов") + "."))
            blocks += [source(ref["url"], ref["title"]) for ref in actor["sources"]]
    blocks += [h("Границы чтения результатов"), p(prose["reader_limits"]),
               p("Методы построения, точные значения, проверки устойчивости и причины исключений подробно описаны в отдельном техническом отчёте. Политологические объяснения в этом документе следует рассматривать вместе с этими ограничениями."),
               source("technical/report.html", "Отдельный технический отчёт")]
    return blocks


def presentation(model, prose, cluster_review, actors, context, russia, maps):
    slides = [{"kind": "cover", "title": "Политическое пространство Руанды", "subtitle": "Три смысловые оси и четыре группы",
               "body": "42 участника в справочнике\n37 публичных профилей на картах"},
              {"kind": "text", "title": "Основные выводы", "paragraphs": [
                  "Преемственность, обновление программ и разные способы обосновывать политику образуют самостоятельные линии различий.",
                  "Социальные задачи и язык памяти могут поддерживать общий государственный курс в разных публичных формах.",
                  "Четыре группы описывают сходство акцентов. Их состав нельзя читать как список политических союзов."]},
              {"kind": "text", "title": "Кто вошёл в исследование", "paragraphs": [
                  "Руководство страны, министры, парламент и местное управление, кандидаты, оппозиционные фигуры и международные дипломаты.",
                  "Период: 15 октября 2021 — 15 октября 2026 года. Список охватывает разные публичные роли и не является рейтингом популярности.",
                  "Все 42 участника описаны в справочнике. На картах размещены 37 достаточно описанных профилей."]}]
    cs = {x["id"]: x for x in context["sources"]}
    slides += [
        {"kind": "text", "title": "Политическая среда Руанды", "paragraphs": [
            "По окончательным результатам NEC, на выборах 2024 года Кагаме получил 99,18% голосов. Список FPR-Inkotanyi получил 37 из 53 напрямую избираемых мест.",
            "HRW описывает ограничения публичной конкуренции и участия оппозиции. Официальный результат нельзя считать измерением всех частных убеждений населения.",
            "В этой среде государственная преемственность и предложения политических изменений имеют разные институциональные возможности."],
         "sources": [cs[sid]["url"] for sid in ["nec_final_2024", "hrw_elections_2024"]]},
        {"kind": "text", "title": "Почему различаются публичные акценты", "paragraphs": [
            "Долгосрочная стратегия развития позволяет сохранять общие цели и пересматривать способы их достижения. Так устроена заявленная логика NST2 на 2024–2029 годы.",
            "Память о геноциде 1994 года и язык единства задают самостоятельную рамку публичного обоснования политики.",
            "Служебные задачи тоже различаются: здравоохранение требует говорить об услугах и готовности к рискам, политика памяти — об истории и ответственности."],
         "caption": "Это объяснение в контексте источников. Карта сама по себе не устанавливает причин поведения.",
         "sources": [cs["kwibuka30_kagame"]["url"], prose["supplementary_context_sources"][0]["url"]]}
    ]
    for start in (0, 4):
        slides.append({"kind": "table", "title": f"Признаки итоговой карты {start+1}–{start+4}",
                       "headers": ["Свойство", "Какой публичный акцент описывает"],
                       "rows": [[TRAITS[i], model["loading_table"][i]["name"]] for i in range(start, start+4)],
                       "widths": [370, 778], "height": 425, "fontSize": 25,
                       "caption": "У одного человека могут сочетаться разные свойства. Меньшая выраженность не создаёт противоположного убеждения."})
    for f in prose["factors"]:
        slides.append({"kind": "text", "title": f["short_name"], "paragraphs": f["slide_paragraphs"],
                       "sources": [x["url"] for x in prose["supplementary_context_sources"]]})
    quote = context["illustrative_quotes"][0]
    slides.append({"kind": "quote", "title": "Единство в президентской публичной речи", "quote": quote["ru_translation"],
                   "attribution": "Поль Кагаме, Kwibuka 30, 7 апреля 2024 года. Перевод с английского.",
                   "body": "Память и единство служат способом объяснять преемственность. Эта речь иллюстрирует политический контекст, а не взгляды всех участников.",
                   "sources": [quote["source_url"]]})
    quote = context["illustrative_quotes"][1]
    slides.append({"kind": "quote", "title": "Профилактика в отраслевой политике", "quote": quote["ru_translation"],
                   "attribution": "Сабин Нсанзимана, министр здравоохранения, 13 февраля 2023 года. Перевод с английского.",
                   "body": "При подписании меморандума с ВОЗ министр связывает готовность к эпидемиям с укреплением системы здравоохранения. Это пример практического обоснования профилактики.",
                   "sources": [quote["source_url"]]})
    slides.append({"kind": "text", "title": "Как читать карту признаков", "paragraphs": [
        "Точка обозначает свойство публичной речи. Близкие точки показывают похожую связь с выбранными осями.",
        "Продолжение и поддержка курса сближаются по первой оси. Обновление программ и раскрытие проблем — по второй.",
        "Одна картинка показывает две оси. Третья может раскрыть различие, которое в этой плоскости незаметно."]})
    slides += [x for x in maps if x["title"] == "Признаки в смысловом пространстве"]
    slides += [x for x in maps if x["title"] == "Политики в смысловом пространстве"]
    slides.append({"kind": "table", "title": "Четыре группы публичных профилей", "headers": ["Группа", "Число", "Смысл"],
                   "rows": [[c["id"], c["count"], c["name"]] for c in model["clusters"]], "widths": [130, 130, 888], "height": 400, "fontSize": 26})
    for cluster in model["clusters"]:
        desc = next(c for c in cluster_review["clusters"] if c["id"] == cluster["id"])
        slides.append({"kind": "text", "title": cluster["id"] + " " + desc["name"],
                       "paragraphs": desc["slide_paragraphs"], "sources": [x["url"] for x in desc.get("sources", [])]})
        members = cluster["members"]
        for start in range(0, len(members), 7):
            page = members[start:start+7]
            if len(members) <= 2:
                continue
            slides.append({"kind": "table", "title": cluster["id"] + " Состав группы" + (f" {start+1}–{start+len(page)}" if len(members)>7 else ""),
                           "headers": ["Участник", "Роль в исследуемый период"],
                           "rows": [[m["name"], next(a["slide_role"] for a in actors if a["id"] == m["id"])] for m in page],
                           "widths": [440, 708], "height": 465, "fontSize": 23, "padding": 4})
    slides.append({"kind": "text", "title": "Кагаме и Рвигара в одной группе", "paragraphs": [
        "Кагаме и Диана Рвигара вошли в большой разнородный круг C4. По этим восьми свойствам у них нет профиля двух малых крайних групп.",
        "Состав C4 не означает общей программы. Президентская власть и критика действующей власти остаются принципиально разными политическими ролями.",
        "Карта публичных акцентов помогает сравнивать речь, но не заменяет анализ институтов и политических отношений."]})
    slides.append({"kind": "text", "title": "Политическое влияние", "paragraphs": cluster_review["influence_slide_paragraphs"]})
    slides += [x for x in maps if x["title"] == "Дополнительная карта соседства политиков"]
    slides.append({"kind": "text", "title": "Отношения с Россией", "paragraphs": [
        "Дипломатическое и отраслевое сотрудничество может сочетаться с критикой конкретных внешнеполитических действий.",
        "Для понимания позиции важны точное высказывание, дата и роль говорящего. Действие министра и речь главы международной организации имеют разный статус.",
        "Следующие примеры относятся к названным людям и эпизодам. Они не задают общую позицию всей группы."]})
    rs = {x["id"]: x for x in russia["sources"]}
    rq = {x["id"]: x for x in russia["quotes"]}
    for suggestion in russia["presentation_suggestions"]:
        for qid in suggestion.get("quote_ids", [])[:1]:
            q = rq[qid]
            actor = next(a for a in actors if a["id"] == q["actor_id"])
            role = ", генеральный секретарь OIF" if q["actor_id"] == "louise-mushikiwabo" else ""
            body = suggestion["body"].replace("Её международная должность должна оставаться в подписи.", "Она выступает как глава OIF, а не от имени правительства Руанды.")
            slides.append({"kind": "quote", "title": suggestion["title"], "quote": q["translation_ru"],
                           "attribution": actor["name"] + role + ", " + str(rs[q["source_id"]].get("event_date") or rs[q["source_id"]].get("publication_date") or "дата в источнике") + ". " + q["translation_label"], "body": body,
                           "sources": [rs[sid]["url"] for sid in suggestion["source_ids"]]})
            break
        else:
            slides.append({"kind": "text", "title": suggestion["title"], "paragraphs": [suggestion["body"], suggestion["caveat"]],
                           "sources": [rs[sid]["url"] for sid in suggestion["source_ids"]]})
    for qid, title, body in [
        ("q05", "Муджавамария о голосовании по Украине",
         "В публичном ответе она сама разъясняет голосование Руанды. Это конкретный эпизод собственной речи, который не даёт общей позиции о санкциях или разрыве сотрудничества."),
        ("q06", "Ндухунгирехе об энергетике и подготовке кадров",
         "В интервью AFP министр говорит о потребности в ядерной энергии и российской подготовке специалистов. Заявленное намерение не означает, что АЭС уже построена или работает.")]:
        q = rq[qid]
        actor = next(a for a in actors if a["id"] == q["actor_id"])
        ref = rs[q["source_id"]]
        slides.append({"kind": "quote", "title": title, "quote": q["translation_ru"],
                       "attribution": actor["name"] + ", " + str(ref.get("event_date") or ref.get("publication_date")) + ". " + q["translation_label"],
                       "body": body, "sources": [ref["url"]]})
    for start in range(0, len(actors), 4):
        page = actors[start:start+4]
        slides.append({"kind": "table", "title": f"Справочник участников {start+1}–{start+len(page)}",
                       "headers": ["Имя", "Должность и период", "Краткая справка"],
                       "rows": [[a["name"], a["slide_role"], a["slide_bio"]] for a in page],
                       "widths": [270, 355, 523], "height": 470, "fontSize": 23, "padding": 5,
                       "caption": "Должности указаны для исторического периода. Пять участников пока не размещены на картах.",
                       "sources": [x["url"] for a in page for x in a["sources"]]})
    slides.append({"kind": "text", "title": "Содержание политического пространства", "paragraphs": [
        "Общая государственная линия допускает разные формы публичного обоснования и изменения конкретных программ.",
        "Должностные задачи помогают объяснить различие акцентов. Часть сходства может отражать особенности найденных материалов.",
        "Группы остаются рабочей типологией. Подробные основания и источники приведены в отчёте, проверки расчётов — в отдельном техническом приложении."]})
    return {"factor_count": 3, "cluster_count": 4, "n_objects": 37, "catalogue_count": 42,
            "period": "2021-10-15/2026-10-15", "native_chart_coordinate_rounding_decimals": 8,
            "full_precision_source": "selected_q3/metadata.json and projections.json", "slides": slides}

if __name__ == "__main__":
    main()
