"""Readable q4/k4 slide charts from saved scores/loadings only; never refit.

All objects remain visible. Direct surname labels use deterministic collision
checking in display coordinates, with leader lines for displaced annotations.
The same saved t-SNE coordinates are recolored using the actual selected k4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "studies/russian-politicians-2016-2026/output/models/q4/selected_k4"
DEFAULT_OUTPUT = ROOT / "studies/russian-politicians-2016-2026/output/assets/presentation_q4"
COLORS = {"C1": "#C2410C", "C2": "#7C3AED", "C3": "#0F766E", "C4": "#2563EB"}
GROUP_NAMES = {"C1": "C1 Мобилизация", "C2": "C2 Персональная оппозиция",
               "C3": "C3 Институциональная поддержка", "C4": "C4 Электоральная конкуренция"}
AXES = {
    "F1": "F1 · от централизации к децентрализации",
    "F2": "F2 · от антизападной мобилизации к антивоенной критике и интеграции",
    "F3": "F3 · от институциональной стратегии к конфронтации и персонализации",
    "F4": "F4 · от идеологической и социальной программы к результатам управления",
}
TRAIT_NAMES = {
    "security-sovereignty": "Суверенитет",
    "technocratic-management": "Практические результаты",
    "parliamentary-institutional": "Работа через институты",
    "oppositionality": "Критика власти",
    "social-policy": "Перераспределение",
    "federal-loyalty": "Централизация",
    "mobilization-rhetoric": "Мобилизация",
    "conflictuality": "Конфронтация",
    "media-personalization": "Личный бренд",
    "antiwesternism": "Противостояние Западу",
    "party-leadership": "Партийная организация",
    "international-activity": "Внешняя повестка",
    "antiwar-rhetoric": "Прекращение войны",
    "electoral-legitimacy": "Выборный мандат",
    "rhetorical-simplicity": "Лозунговая простота",
    "national-patriotism": "Национальная общность",
    "coalitionality": "Коалиционное сотрудничество",
}
CHANGED = ["vladimir-putin", "leonid-slutsky", "alexey-nechayev", "vladislav-davankov"]
FAMILIAR = ["dmitry-medvedev", "alexey-navalny", "ramzan-kadyrov", "mikhail-mishustin",
            "sergey-lavrov", "gennady-zyuganov", "sergey-mironov", "vladimir-zhirinovsky",
            "yevgeny-prigozhin", "sergey-sobyanin", "elvira-nabiullina", "mikhail-khodorkovsky"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _figure(title):
    # 1110:480 aspect ratio, rendered at 180 dpi. At a 1110px slide placement,
    # 15.5pt surname labels remain approximately 17px at a 1110px placement.
    fig, ax = plt.subplots(figsize=(13.875, 6.))
    fig.subplots_adjust(left=.105, right=.99, bottom=.19, top=.80)
    fig.text(.55, .965, title, ha="center", va="top", fontsize=15, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    return fig, ax


def _labels(fig, ax, coordinates, identities, names, priority, maximum, font_size=12.0,
            bounds=None, avoid_points=True):
    """Place labels without label/label, label/point or figure-edge collisions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pixels = ax.transData.transform(coordinates)
    limits = bounds or ax.bbox
    occupied, placed, annotations = [], [], []
    point_boxes = [Bbox.from_bounds(x - 4.5, y - 4.5, 9, 9) for x, y in pixels]
    factor = fig.dpi / 72
    offsets = [(dx, dy) for radius in [6, 11, 18, 27, 39, 54, 72, 94, 122, 155, 195]
               for dx, dy in [(radius, radius), (radius, -radius), (-radius, radius), (-radius, -radius),
                              (radius, 0), (-radius, 0), (0, radius), (0, -radius)]]
    order = {identity: i for i, identity in enumerate(identities)}
    for identity in priority:
        if identity not in order:
            continue
        index = order[identity]
        label = names[index]
        probe = ax.text(0, 0, label, fontsize=font_size)
        extent = probe.get_window_extent(renderer)
        width, height = extent.width, extent.height
        probe.remove()
        px, py = pixels[index]
        choices = []
        for dx, dy in offsets:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
            anchor_x, anchor_y = px + dx * factor, py + dy * factor
            left = anchor_x if ha == "left" else anchor_x - width if ha == "right" else anchor_x - width / 2
            bottom = anchor_y if va == "bottom" else anchor_y - height if va == "top" else anchor_y - height / 2
            box = Bbox.from_bounds(left - 3, bottom - 3, width + 6, height + 6)
            if not (limits.contains(box.x0, box.y0) and limits.contains(box.x1, box.y1)):
                continue
            if any(box.overlaps(other) for other in occupied):
                continue
            if avoid_points and any(box.overlaps(other) for other in point_boxes):
                continue
            # Prefer short, nearly horizontal leaders, then unused whitespace.
            choices.append((np.hypot(dx, dy) + .07 * abs(dy), dx, dy, ha, va, box))
        if not choices:
            continue
        _, dx, dy, ha, va, box = min(choices, key=lambda item: item[0])
        leader = np.hypot(dx, dy) > 15
        annotation = ax.annotate(label, coordinates[index], xytext=(dx, dy),
                                 textcoords="offset points", ha=ha, va=va, fontsize=font_size,
                                 color="#111827", annotation_clip=False, zorder=6,
                                 bbox={"facecolor": "white", "alpha": .94, "edgecolor": "none", "pad": .5},
                                 arrowprops={"arrowstyle": "-", "color": "#64748B", "lw": .55,
                                             "alpha": .75, "shrinkA": 2, "shrinkB": 4} if leader else None)
        occupied.append(box)
        annotations.append(annotation)
        placed.append(identity)
        if len(placed) >= maximum:
            break
    fig.canvas.draw()
    # Audit actual rendered text bounds, not just approximated anchor boxes.
    boxes = [annotation.get_bbox_patch().get_window_extent(renderer).expanded(1.01, 1.03)
             for annotation in annotations]
    for i, box in enumerate(boxes):
        if not (fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)):
            raise RuntimeError(f"Clipped label: {placed[i]}")
        if any(box.overlaps(other) for other in boxes[i+1:]):
            raise RuntimeError(f"Overlapping label: {placed[i]}")
    return placed


def _object_priority(coordinates, identities, labels, metadata, used, variant):
    """Mix familiar figures, representatives, underlabelled and spread-out points."""
    priority = list(CHANGED)
    if variant == 0:
        priority.extend(FAMILIAR)
    for rank in range(2):
        for group in metadata["clusters"]:
            representative = group["representatives"][rank]["id"]
            if representative not in priority:
                priority.append(representative)
    normalized = (coordinates - coordinates.min(axis=0)) / np.maximum(np.ptp(coordinates, axis=0), 1e-10)
    indices = {identity: i for i, identity in enumerate(identities)}
    while len(priority) < len(identities):
        selected = [indices[item] for item in priority if item in indices]
        candidates = [i for i, identity in enumerate(identities) if identity not in priority]
        selected_counts = {group: sum(labels.iloc[i] == group for i in selected) for group in labels.unique()}
        def value(i):
            minimum = float(np.min(np.linalg.norm(normalized[i] - normalized[selected], axis=1))) if selected else 1.
            return (.7 if identities[i] not in used else 0.) + minimum + .13 / (1 + selected_counts[labels.iloc[i]])
        winner = max(candidates, key=value)
        priority.append(identities[winner])
    return priority


def _object_chart(path, metadata, scores, labels, coordinates, axes, title, used, variant, tsne=False):
    fig, ax = _figure(title)
    identities = scores.index.tolist()
    for group in COLORS:
        mask = labels.to_numpy() == group
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=40, c=COLORS[group],
                   edgecolors="white", linewidth=.6, alpha=.92,
                   label=f"{GROUP_NAMES[group]} ({int(mask.sum())})", zorder=3)
    handles, legend_names = ax.get_legend_handles_labels()
    fig.legend(handles, legend_names, loc="upper center", bbox_to_anchor=(.55, .91),
               ncol=2, fontsize=12.5, frameon=False, columnspacing=2.)
    ax.margins(x=.22, y=.27)
    if not tsne:
        ax.axhline(0, c="#94A3B8", lw=.8, zorder=1)
        ax.axvline(0, c="#94A3B8", lw=.8, zorder=1)
    ax.set_xlabel(textwrap.fill(axes[0], 95), fontsize=13, labelpad=8)
    ax.set_ylabel(textwrap.fill(axes[1], 43), fontsize=13, labelpad=8)
    names = [str(value).split(" (")[0].strip().split()[-1] for value in scores["name"]]
    priority = _object_priority(coordinates, identities, labels, metadata, used, variant)
    placed = _labels(fig, ax, coordinates, identities, names, priority, maximum=35, font_size=15.5)
    if len(placed) < 28:
        raise RuntimeError(f"Only {len(placed)} readable surnames on {path.name}; target at least 28")
    if tsne:
        note = "Все 63 объекта · Направления и пустоты этой карты не имеют самостоятельного политического смысла"
    else:
        note = "Все 63 объекта · Группы рассчитаны по четырём факторам; здесь показаны только две координаты"
    fig.text(.55, .035, note, ha="center", va="bottom", fontsize=10.5, color="#475569")
    _check_figure_text(fig)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return placed


def _check_figure_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Verify explanatory text and axis titles remain within the exported canvas.
    for text in [*fig.texts, *[t for ax in fig.axes for t in [ax.xaxis.label, ax.yaxis.label]]]:
        box = text.get_window_extent(renderer)
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
            raise RuntimeError(f"Text exceeds figure canvas: {text.get_text()}")


def _trait_chart(path, loadings, factor_x, factor_y):
    fig, ax = _figure(f"Карта свойств: {factor_x} и {factor_y}")
    fig.text(.55, .90, "Подпись называет первый полюс шкалы; точка показывает его связь с двумя факторами",
             ha="center", fontsize=11, color="#475569")
    coordinates = loadings[[factor_x, factor_y]].to_numpy()
    # Use the full slide width for readable direct labels. Numerical axis ticks
    # retain the actual loadings; visual angles/distances are not interpreted.
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), aspect="auto")
    ax.axhline(0, c="#64748B", lw=.9)
    ax.axvline(0, c="#64748B", lw=.9)
    for point in coordinates:
        ax.annotate("", point, (0, 0), arrowprops={"arrowstyle": "->", "color": "#94A3B8", "lw": .8, "alpha": .65})
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=44, color="#155E75", edgecolors="white", linewidth=.7, zorder=4)
    ax.set_xlabel(textwrap.fill(AXES[factor_x], 90), fontsize=13, labelpad=8)
    ax.set_ylabel(textwrap.fill(AXES[factor_y], 43), fontsize=13, labelpad=8)
    ax.set_xticks([-1, -.5, 0, .5, 1]); ax.set_yticks([-1, -.5, 0, .5, 1])
    fig.canvas.draw()
    bounds = ax.bbox
    identities = loadings.index.tolist()
    names = [textwrap.fill(TRAIT_NAMES.get(identity, str(loadings.loc[identity, "name"]).split("—")[0].strip()), 26)
             for identity in identities]
    # Place the most crowded central vectors first; strong outer vectors have
    # ample surrounding whitespace and remain easy to place afterwards.
    priority = [identities[i] for i in np.argsort(np.linalg.norm(coordinates, axis=1))]
    placed = _labels(fig, ax, coordinates, identities, names, priority, maximum=len(identities),
                     font_size=15.5, bounds=bounds, avoid_points=True)
    if len(placed) != len(identities):
        missing = set(identities) - set(placed)
        raise RuntimeError(f"Not all approved scales have readable labels: {missing}")
    fig.text(.55, .035, "Все 17 шкал · Близость к центру означает слабую связь с этой парой факторов, а не отсутствие свойства у политиков",
             ha="center", fontsize=10.2, color="#475569")
    _check_figure_text(fig)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return placed


def build_charts(model_output: Path = DEFAULT_MODEL, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    model_output, output = Path(model_output), Path(output_dir)
    source_files = [model_output / name for name in ["metadata.json", "scores.csv", "loadings.csv", "cluster_labels.csv", "tsne_objects.csv"]]
    digests = {str(path): _sha(path) for path in source_files}
    metadata = json.loads((model_output / "metadata.json").read_text(encoding="utf-8"))
    if metadata["factor_count"] != 4 or metadata["cluster_count"] != 4:
        raise ValueError("Presentation charts require the saved q4/k4 variant")
    scores = pd.read_csv(model_output / "scores.csv", index_col=0)
    loadings = pd.read_csv(model_output / "loadings.csv", index_col=0)
    labels = pd.read_csv(model_output / "cluster_labels.csv").set_index("object_id")["cluster_id"].reindex(scores.index)
    tsne = pd.read_csv(model_output / "tsne_objects.csv", index_col=0).reindex(scores.index)
    if scores.shape[0] != 63 or loadings.shape[0] != 17 or labels.isna().any() or tsne.isna().any().any():
        raise ValueError("Saved model does not contain the expected complete 63-object/17-scale rectangle")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.titleweight": "bold"})
    used, charts = set(), []
    specifications = [
        ("objects_f2_f3.png", scores[["F2", "F3"]].to_numpy(), (AXES["F2"], AXES["F3"]),
         "Основные различия публичных образов", False),
        ("objects_tsne.png", tsne[["tsne_1", "tsne_2"]].to_numpy(),
         ("Расположение по сходству · горизонталь", "Расположение по сходству · вертикаль"),
         "Кто на кого похож по четырём факторам", True),
        ("objects_f1_f4.png", scores[["F1", "F4"]].to_numpy(), (AXES["F1"], AXES["F4"]),
         "Дополнительные различия: полномочия и обоснование программы", False),
    ]
    for variant, (filename, coordinates, axes, title, is_tsne) in enumerate(specifications):
        placed = _object_chart(output / filename, metadata, scores, labels, coordinates, axes, title, used, variant, is_tsne)
        used.update(placed)
        charts.append({"file": filename, "kind": "objects", "title": title, "point_count": len(scores),
                       "surname_count": len(placed), "labelled_object_ids": placed,
                       "caption": "Цвета — фактические четыре группы сохранённой модели q4/k4. Все объекты показаны; линии соединяют смещённые фамилии с их точками."})
    for x, y in [("F2", "F3"), ("F1", "F4")]:
        filename = f"traits_{x.lower()}_{y.lower()}.png"
        placed = _trait_chart(output / filename, loadings, x, y)
        charts.append({"file": filename, "kind": "traits", "title": f"Как читать свойства: {x} и {y}",
                       "point_count": len(loadings), "label_count": len(placed),
                       "caption": "Точки — нагрузки шкал, подписи — их первые полюса. Второй полюс связан с противоположным направлением. Это карта свойств, а не оценки политиков."})
    for path, digest in digests.items():
        if _sha(path) != digest:
            raise RuntimeError(f"Saved model source was changed: {path}")
    study_output = model_output.resolve().parents[2]
    relative_folder = output.resolve().relative_to(study_output).as_posix()
    for chart in charts:
        chart["file"] = f"{relative_folder}/{chart['file']}"
    result = {"model": "q4/selected_k4", "factor_count": 4, "cluster_count": 4,
            "n_objects": len(scores), "n_scales": len(loadings),
            "source_sha256": {Path(path).resolve().relative_to(study_output).as_posix(): digest
                              for path, digest in digests.items()}, "charts": charts,
            "unique_objects_labelled_across_maps": len(used), "all_sources_unchanged": True,
            "revealed_names": [str(scores.loc[identity, "name"]) for identity in scores.index if identity in used],
            "label_audit": "Rendered text boxes checked for pairwise overlaps and clipping; all 63 object points and all 17 scale points shown"}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(json.dumps(build_charts(arguments.model_output, arguments.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
