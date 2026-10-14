"""Separate object-coordinate and trait-loading maps of one fitted model.

Factor planes have actual factor axes. t-SNE axes have no factor interpretation.
Inputs are CSVs indexed by object/trait with the same factor-column names.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, trustworthiness
from threadpoolctl import threadpool_limits


def load_coordinates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame = frame.loc[:, [c for c in frame.columns if str(c).startswith("F")]]
    if frame.shape[1] < 2 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Maps require at least two finite factor columns")
    if frame.index.duplicated().any():
        raise ValueError("Object/trait labels must be unique")
    return frame


def label_points(ax, frame: pd.DataFrame, coordinates: np.ndarray, indices: list[int]) -> None:
    middle = (coordinates[:, 0].min() + coordinates[:, 0].max()) / 2
    vertical_middle = (coordinates[:, 1].min() + coordinates[:, 1].max()) / 2
    for i in indices:
        right = coordinates[i, 0] >= middle
        top = coordinates[i, 1] >= vertical_middle
        label = textwrap.fill(str(frame.index[i]), width=28)
        ax.annotate(label, coordinates[i], xytext=(-5 if right else 5, -5 if top else 4),
                    textcoords="offset points", fontsize=7,
                    ha="right" if right else "left", va="top" if top else "bottom")


def representative_labels(coordinates: np.ndarray, count: int) -> list[int]:
    """Choose strong, spatially distinct loading examples to avoid label piles."""
    if count == 0:
        return []
    selected = []
    for i in np.argsort(-np.linalg.norm(coordinates, axis=1)):
        if not selected or all(np.linalg.norm(coordinates[i] - coordinates[j]) >= .22 for j in selected):
            selected.append(int(i))
        if len(selected) >= count:
            break
    return selected


def create_maps(scores_path: Path, loadings_path: Path, out: Path,
                focus: list[str], groups_path: Path | None, label_count: int) -> dict:
    scores, loadings = load_coordinates(scores_path), load_coordinates(loadings_path)
    if scores.columns.tolist() != loadings.columns.tolist():
        raise ValueError("Scores and loadings must belong to the same factor ordering")
    groups = json.loads(groups_path.read_text(encoding="utf-8")) if groups_path else {}
    out.mkdir(parents=True, exist_ok=True)
    focus_positions = [i for i, name in enumerate(scores.index) if name in focus]
    manifest = {"scores_source": str(scores_path), "loadings_source": str(loadings_path),
                "factor_columns": scores.columns.tolist(),
                "focus": focus, "missing_focus": [name for name in focus if name not in scores.index],
                "scope": "Separate maps; no common metric between objects and trait loadings; no clustering on t-SNE",
                "tsne": {}}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    object_xy = scores.iloc[:, :2].to_numpy()
    group_names = sorted(set(groups.get(str(name), "Objects") for name in scores.index))
    palette = plt.get_cmap("tab20", max(1, len(group_names)))
    colors = {group: palette(i) for i, group in enumerate(group_names)}
    for group in group_names:
        mask = np.array([groups.get(str(name), "Objects") == group for name in scores.index])
        axes[0].scatter(object_xy[mask, 0], object_xy[mask, 1], s=28, alpha=.8, label=group, color=colors[group])
    if focus_positions:
        chosen = object_xy[focus_positions]
        axes[0].scatter(chosen[:, 0], chosen[:, 1], marker="*", s=160, color="#b71c1c", edgecolors="black")
        label_points(axes[0], scores, object_xy, focus_positions)
    axes[0].set(title="Объекты: факторные координаты",
                xlabel=scores.columns[0], ylabel=scores.columns[1])
    axes[0].legend(fontsize=7, ncol=2)
    trait_xy = loadings.iloc[:, :2].to_numpy()
    axes[1].scatter(trait_xy[:, 0], trait_xy[:, 1], s=20, alpha=.5, color="#176d81")
    strongest = representative_labels(trait_xy, label_count)
    label_points(axes[1], loadings, trait_xy, strongest)
    axes[1].set(title="Признаки: факторные нагрузки", xlabel=loadings.columns[0], ylabel=loadings.columns[1])
    for ax in axes:
        ax.axhline(0, color="#999999", linewidth=.6)
        ax.axvline(0, color="#999999", linewidth=.6)
        ax.grid(alpha=.15)
    fig.suptitle("Факторные оси: карты объектов и признаков имеют разный смысл координат")
    fig.tight_layout()
    fig.savefig(out / "factor_planes.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for kind, frame, ax in (("objects", scores, axes[0]), ("traits", loadings, axes[1])):
        if len(frame) < 4:
            ax.text(.5, .5, "Too few points for t-SNE", ha="center")
            continue
        perplexity = min(30., max(2., (len(frame) - 1) / 3))
        embedding = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity,
                         random_state=42, max_iter=1500).fit_transform(frame.to_numpy())
        if kind == "objects":
            for group in group_names:
                mask = np.array([groups.get(str(name), "Objects") == group for name in frame.index])
                ax.scatter(embedding[mask, 0], embedding[mask, 1], s=28, alpha=.8, label=group, color=colors[group])
            if focus_positions:
                chosen = embedding[focus_positions]
                ax.scatter(chosen[:, 0], chosen[:, 1], marker="*", s=160, color="#b71c1c", edgecolors="black")
                label_points(ax, frame, embedding, focus_positions)
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], s=20, alpha=.6, color="#176d81")
            label_points(ax, frame, embedding, strongest)
        name = "объектов" if kind == "objects" else "признаков"
        ax.set(title=f"t-SNE {name}: проекция из {frame.shape[1]} измерений",
               xlabel="Координата t-SNE 1", ylabel="Координата t-SNE 2")
        neighbors = min(5, max(1, (len(frame) - 1) // 2))
        manifest["tsne"][kind] = {"perplexity": perplexity, "random_state": 42,
                                   "metric": "euclidean", "rescaled_before_tsne": False,
                                   "neighbors": neighbors,
                                   "trustworthiness": float(trustworthiness(frame.to_numpy(), embedding,
                                                                            n_neighbors=neighbors)),
                                   "warning": "Local-neighborhood illustration; axes are not factors; distances/islands do not validate clusters"}
        pd.DataFrame(embedding, columns=["tsne_1", "tsne_2"], index=frame.index).to_csv(out / f"{kind}_tsne.csv")
    fig.suptitle("t-SNE: локальное сходство; координатные оси не являются факторами")
    fig.tight_layout()
    fig.savefig(out / "tsne_maps.png", dpi=170)
    plt.close(fig)
    (out / "map_metadata.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--loadings", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/semantic_maps"))
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--focus", action="append", default=[])
    parser.add_argument("--label-count", type=int, default=12)
    args = parser.parse_args()
    if args.label_count < 0:
        parser.error("label-count must be nonnegative")
    with threadpool_limits(limits=1):
        create_maps(args.scores, args.loadings, args.output_dir, args.focus, args.groups, args.label_count)


if __name__ == "__main__":
    main()
