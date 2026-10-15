"""Compare FA dimensionalities on one frozen measured rectangle, without APIs.

The main study output is read-only. Model-specific exports, complete transitions,
and diagnostic plots are written under output/models by default. Factor numbers
are local to a fitted/rotated model, never cumulative PCA component numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import textwrap

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.latent_space_benchmark import transform_new_objects
from src.semantic_service.analysis import (
    SEED, _clustering, _object_and_cluster_profiles, _prepare, _save_charts, analyze_study,
)


MODEL_FILES = ("metadata.json", "scores.csv", "cluster_labels.csv", "loadings.csv",
               "matrix.csv", "mass.csv", "parameters.joblib", "tsne_objects.csv",
               "tsne_traits.csv")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def freeze_input_files(study_path: Path, ratings_path: Path, output_root: Path) -> tuple[Path, Path]:
    """Retain the exact fit basis despite later report/status metadata changes."""
    study_path, ratings_path, output = Path(study_path), Path(ratings_path), Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    frozen_study, frozen_ratings = output / "input_study.json", output / "input_ratings.json"
    for source, frozen in [(study_path, frozen_study), (ratings_path, frozen_ratings)]:
        if not frozen.exists():
            shutil.copy2(source, frozen)
    if _sha256(ratings_path) != _sha256(frozen_ratings):
        raise ValueError("Ratings changed since the frozen model-comparison basis")
    fields = ["id", "scope", "query", "aspect", "date_from", "date_to", "measurement_mode", "focus", "objects", "scales"]
    original, current = _load_json(frozen_study), _load_json(study_path)
    if any(original.get(key) != current.get(key) for key in fields):
        raise ValueError("Study measurement basis changed; use a new comparison directory")
    return frozen_study, frozen_ratings


def factor_congruence(reference: np.ndarray, target: np.ndarray,
                      reference_ids=None, target_ids=None) -> dict:
    """Signed Tucker congruence and best one-to-one sign/permutation matching.

    Matching does not undo arbitrary rotations. Principal angles and an explicit
    semi-orthogonal Procrustes fit additionally describe the loading subspaces.
    An unmatched target column is not proof of an independently new construct.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(target, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("Loading matrices require the same ordered scale rows")
    if a.shape[1] > b.shape[1] or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Require finite loadings and reference q <= target q")
    norm_a, norm_b = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    if np.any(norm_a <= 1e-12) or np.any(norm_b <= 1e-12):
        raise ValueError("Congruence of a zero loading vector is undefined")
    rid = list(reference_ids or [f"F{i+1}" for i in range(a.shape[1])])
    tid = list(target_ids or [f"F{i+1}" for i in range(b.shape[1])])
    congruence = np.clip((a.T @ b) / (norm_a[:, None] * norm_b[None, :]), -1., 1.)
    rows, columns = linear_sum_assignment(-np.abs(congruence))
    matches = [{"reference_factor": rid[i], "target_factor": tid[j],
                "congruence": float(congruence[i, j]),
                "absolute_congruence": float(abs(congruence[i, j])),
                "sign": 1 if congruence[i, j] >= 0 else -1}
               for i, j in zip(rows, columns)]
    left, _, right = np.linalg.svd(b.T @ a, full_matrices=False)
    rotation = left @ right
    fitted = b @ rotation
    residual = np.linalg.norm(a - fitted, "fro") / np.linalg.norm(a, "fro")
    cosine = np.sum(a * fitted, axis=0) / (norm_a * np.linalg.norm(fitted, axis=0))
    return {"congruence_matrix": {"row_factors": rid, "column_factors": tid,
                                   "values": congruence.tolist()},
            "matched_factors": matches,
            "unmatched_target_factors": [tid[j] for j in range(len(tid)) if j not in columns],
            "principal_angles_degrees": np.rad2deg(subspace_angles(a, b)).tolist(),
            "procrustes": {"target_to_reference_matrix": rotation.tolist(),
                           "relative_loading_residual": float(residual),
                           "column_cosines_after_mapping": np.clip(cosine, -1., 1.).tolist(),
                           "scope": "Semi-orthogonal target loading map; preserves fitted subspace, not the named varimax axes"},
            "matching_scope": "Hungarian maximum absolute uncentered Tucker congruence; signs are alignment aids, original exports are unchanged"}


def compare_partitions(reference: pd.Series, target: pd.Series,
                       names: dict[str, str] | None = None) -> dict:
    """Align equal-k labels by maximum overlap and retain every object's move."""
    if not reference.index.is_unique or not target.index.is_unique:
        raise ValueError("Object identities must be unique")
    if set(reference.index) != set(target.index):
        raise ValueError("Partitions must contain exactly the same objects")
    target = target.reindex(reference.index)
    rid, tid = sorted(reference.unique()), sorted(target.unique())
    if len(rid) != len(tid):
        raise ValueError("Fixed-k comparison requires the same cluster count")
    table = pd.crosstab(reference.rename("reference"), target.rename("target")).reindex(
        index=rid, columns=tid, fill_value=0)
    rows, columns = linear_sum_assignment(-table.to_numpy())
    mapping = {str(tid[j]): str(rid[i]) for i, j in zip(rows, columns)}
    aligned = target.map(mapping)
    aligned_table = pd.crosstab(reference.rename("reference"), aligned.rename("target_aligned")).reindex(
        index=rid, columns=rid, fill_value=0)
    transitions = [{"object_id": str(identity), "name": (names or {}).get(str(identity), str(identity)),
                    "from_cluster": str(reference.loc[identity]),
                    "to_cluster_raw": str(target.loc[identity]),
                    "to_cluster_aligned": str(aligned.loc[identity]),
                    "changed": bool(reference.loc[identity] != aligned.loc[identity])}
                   for identity in reference.index]
    moved = sum(item["changed"] for item in transitions)
    return {"k": len(rid), "ari": float(adjusted_rand_score(reference, target)),
            "nmi": float(normalized_mutual_info_score(reference, target, average_method="arithmetic")),
            "moved_count": moved, "unchanged_count": len(reference) - moved,
            "overlap_alignment": mapping,
            "contingency_raw": {"rows": list(rid), "columns": list(tid), "values": table.to_numpy().tolist()},
            "contingency_aligned": {"rows": list(rid), "columns": list(rid), "values": aligned_table.to_numpy().tolist()},
            "transitions": transitions,
            "scope": "Same objects and fixed k; maximum-overlap label permutation only, no model refit or forced memberships"}


def _factor_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("F") and column[1:].isdigit()]


def _snapshot_baseline(source: Path, target: Path) -> dict:
    manifest_path = target / "baseline_manifest.json"
    if manifest_path.exists():
        saved = _load_json(manifest_path)
        # A report may now embed model_comparison in its primary metadata. Keep
        # the first snapshot, rather than recursively embedding comparisons.
        for name, digest in saved.items():
            if not (target / name).is_file() or _sha256(target / name) != digest:
                raise ValueError(f"Frozen q2 snapshot was changed: {name}")
        for name in ["matrix.csv", "scores.csv", "loadings.csv", "cluster_labels.csv"]:
            if _sha256(source / name) != saved.get(name):
                raise ValueError(f"Primary q2 numerics no longer match the frozen snapshot: {name}")
        return saved
    if "model_comparison" in _load_json(source / "metadata.json"):
        raise ValueError("Primary output already embeds model_comparison and has no frozen q2 snapshot; use the original q2 baseline")
    target.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in MODEL_FILES:
        path = source / name
        if path.exists():
            manifest[name] = _sha256(path)
            shutil.copy2(path, target / name)
    if (source / "assets").is_dir():
        shutil.copytree(source / "assets", target / "assets", dirs_exist_ok=True)
        for path in sorted((source / "assets").rglob("*")):
            if path.is_file():
                manifest[path.relative_to(source).as_posix()] = _sha256(path)
    _write_json(manifest_path, manifest)
    return manifest


def _assert_same_rectangle(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    if actual.index.tolist() != expected.index.tolist() or actual.columns.tolist() != expected.columns.tolist():
        raise ValueError("Current preprocessing changes the frozen object/scale rectangle")
    if not np.allclose(actual.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError("Current ratings differ from the frozen baseline matrix")


def export_selected_clustering(model_output: Path, study: dict, cluster_count: int) -> dict:
    """Export a different k using saved FA parameters; no FA or rating rerun.

    The parent fixed-k3 model remains untouched. Candidate KMeans and conditional
    stability are recomputed deterministically on its identical posterior scores.
    """
    model_output = Path(model_output)
    meta = _load_json(model_output / "metadata.json")
    if cluster_count == meta["cluster_count"]:
        return meta
    output = model_output / f"selected_k{cluster_count}"
    output.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILES:
        if (model_output / filename).exists():
            shutil.copy2(model_output / filename, output / filename)
    parameters = joblib.load(model_output / "parameters.joblib")
    frame = pd.read_csv(model_output / "matrix.csv", index_col=0)
    if frame.columns.tolist() != parameters["trait_order"]:
        raise ValueError("Saved model feature order differs from its matrix")
    objects = {str(item["id"]): item for item in study["objects"]}
    scales = {str(item["id"]): item for item in study["scales"]}
    with threadpool_limits(limits=1):
        scores = transform_new_objects(frame.to_numpy(), parameters)
        stored = pd.read_csv(model_output / "scores.csv", index_col=0)
        np.testing.assert_allclose(scores, stored[_factor_columns(stored)].to_numpy(), atol=1e-12, rtol=0)
        clustering, candidates, selection, _ = _clustering(scores, cluster_count)
        # Retain the original candidate semantic profiles, computed from the
        # identical factor model. Their group ordering must match the refit.
        original_candidates = {c["k"]: c for c in meta["cluster_candidates"]}
        for c in candidates:
            original = original_candidates[c["k"]]
            if c["sizes"] != original["sizes"]:
                raise ValueError("Refitted KMeans changed candidate ordering/sizes")
            c["semantic_profiles"] = original["semantic_profiles"]
        selected = next(c for c in candidates if c["k"] == cluster_count)
        profiles, clusters = _object_and_cluster_profiles(frame, scores, clustering, meta["factors"], objects, scales, selected)
        charts, tsne = _save_charts(output, scores, parameters["loadings"] @ parameters["rotation"],
                                    meta["factors"], clusters, profiles,
                                    [scales[identity] for identity in frame.columns], study)
    labels = pd.DataFrame([{"object_id": p["id"], "name": p["name"], "cluster_id": p["cluster_id"],
                            "boundary_margin": p["boundary_margin"]} for p in profiles])
    labels.to_csv(output / "cluster_labels.csv", index=False, encoding="utf-8-sig")
    parameters["cluster_centers"] = clustering.cluster_centers_
    parameters["cluster_ids"] = [item["id"] for item in clusters]
    joblib.dump(parameters, output / "parameters.joblib")
    meta.update(cluster_count=cluster_count, clusters=clusters, object_profiles=profiles,
                cluster_candidates=candidates, charts=charts)
    meta["diagnostics"]["cluster_selection"] = selection
    meta["diagnostics"]["tsne"] = tsne
    meta["comparison_variant"] = {"parent": "..", "factor_model_reused": True,
                                   "source_fixed_k": 3, "numerical_recommended_k": cluster_count,
                                   "scope": "Numerical recommendation exported for substantive review; no new FA or ratings"}
    _write_json(output / "metadata.json", meta)
    return meta


def export_selected_partition_comparison(output_root: Path, reference_q: int,
                                         target_q: int, cluster_count: int) -> dict:
    """Compare same-k review variants, retaining their original local labels."""
    output = Path(output_root)
    paths = [output / f"q{q}" / f"selected_k{cluster_count}" for q in [reference_q, target_q]]
    frames = [pd.read_csv(path / "cluster_labels.csv").set_index("object_id") for path in paths]
    names = frames[0]["name"].to_dict()
    result = {"reference_q": reference_q, "target_q": target_q,
              **compare_partitions(frames[0]["cluster_id"], frames[1]["cluster_id"], names)}
    stem = f"clusters_q{reference_q}_q{target_q}_k{cluster_count}"
    files = {}
    for name, rows in [("all_objects", result["transitions"]),
                       ("changed_objects", [r for r in result["transitions"] if r["changed"]])]:
        filename = f"{stem}_{name}.csv"
        pd.DataFrame(rows, columns=["object_id", "name", "from_cluster", "to_cluster_raw", "to_cluster_aligned", "changed"]).to_csv(
            output / filename, index=False, encoding="utf-8-sig")
        files[f"{name}_csv"] = filename
    for orientation in ["raw", "aligned"]:
        table = result[f"contingency_{orientation}"]
        filename = f"{stem}_contingency_{orientation}.csv"
        pd.DataFrame(table["values"], index=table["rows"], columns=table["columns"]).rename_axis(
            "reference_cluster").to_csv(output / filename, encoding="utf-8-sig")
        files[f"contingency_{orientation}_csv"] = filename
    table = result["contingency_aligned"]
    values = np.asarray(table["values"])
    fig, ax = plt.subplots(figsize=(7, 6.4), constrained_layout=True)
    ax.imshow(values, cmap="Blues", vmin=0, vmax=max(values.max(), 1))
    ax.set(xticks=range(cluster_count), xticklabels=table["columns"],
           yticks=range(cluster_count), yticklabels=table["rows"],
           xlabel=f"q={target_q}, метки выровнены по составу", ylabel=f"q={reference_q}",
           title=f"Содержательно выбранные разбиения: k={cluster_count}\nARI={result['ari']:.3f}; NMI={result['nmi']:.3f}")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=13,
                    color="white" if value > values.max() * .55 else "black")
    (output / "assets").mkdir(exist_ok=True)
    image_file = f"assets/{stem}.png"
    fig.savefig(output / image_file, dpi=180, bbox_inches="tight"); plt.close(fig)
    files["chart_png"] = image_file
    result["artifact_files"] = files
    result["chart"] = {"title": f"Переходы q{reference_q}→q{target_q} при k={cluster_count}",
                       "file": image_file, "caption": "После max-overlap выравнивания меток; исходные номера кластеров и полный список всех объектов сохранены в CSV."}
    files["comparison_json"] = f"{stem}_comparison.json"
    _write_json(output / files["comparison_json"], result)
    return result


def _export_comparison_tables(result: dict, output: Path) -> None:
    for pair in result["factor_comparisons"]:
        stem = f"factors_q{pair['reference_q']}_q{pair['target_q']}"
        matrix = pair["congruence_matrix"]
        pd.DataFrame(matrix["values"], index=matrix["row_factors"], columns=matrix["column_factors"]).rename_axis(
            "reference_factor").to_csv(output / f"{stem}_congruence.csv", encoding="utf-8-sig")
        pd.DataFrame(pair["matched_factors"]).to_csv(output / f"{stem}_matching.csv", index=False, encoding="utf-8-sig")
        pair["artifact_files"] = {"congruence_csv": f"{stem}_congruence.csv", "matching_csv": f"{stem}_matching.csv"}
    for pair in result["fixed_k3"]:
        stem = f"clusters_q{pair['reference_q']}_q{pair['target_q']}_k3"
        moves = pd.DataFrame(pair["transitions"])
        moves.to_csv(output / f"{stem}_all_objects.csv", index=False, encoding="utf-8-sig")
        moves.loc[moves["changed"]].to_csv(output / f"{stem}_changed_objects.csv", index=False, encoding="utf-8-sig")
        files = {"all_objects_csv": f"{stem}_all_objects.csv", "changed_objects_csv": f"{stem}_changed_objects.csv"}
        for orientation in ["raw", "aligned"]:
            table = pair[f"contingency_{orientation}"]
            filename = f"{stem}_contingency_{orientation}.csv"
            pd.DataFrame(table["values"], index=table["rows"], columns=table["columns"]).rename_axis(
                "reference_cluster").to_csv(output / filename, encoding="utf-8-sig")
            files[f"contingency_{orientation}_csv"] = filename
        pair["artifact_files"] = files
    rows = []
    for model in result["models"]:
        for item in model["factor_shares"]:
            rows.append({"q": model["q"], "factor": item["id"], "variance_share": item["share"],
                         "total_variance_share": model["variance_total"]})
    pd.DataFrame(rows).to_csv(output / "variance_comparison.csv", index=False, encoding="utf-8-sig")
    rows = []
    for model in result["numerical_free_k"]:
        for c in model["candidates"]:
            rows.append({"q": model["q"], "k": c["k"], "silhouette": c["silhouette"],
                         "sizes": ";".join(map(str, c["sizes"])), **{key: c["stability"][key] for key in
                         ["mean_ari", "median_ari", "min_ari", "min_cluster_median_jaccard"]}})
    pd.DataFrame(rows).to_csv(output / "free_k_candidates.csv", index=False, encoding="utf-8-sig")


def _comparison_charts(result: dict, output: Path) -> list[dict]:
    folder = output / "assets"
    folder.mkdir(exist_ok=True)
    charts = []
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = ["#155E75", "#0F766E", "#65A30D", "#D97706"]
    for column, model in enumerate(result["models"]):
        bottom = 0.
        for i, factor in enumerate(model["factor_shares"]):
            value = 100 * factor["share"]
            ax.bar(column, value, bottom=bottom, color=colors[i], edgecolor="white", width=.62)
            ax.text(column, bottom + value / 2, f"{factor['id']}: {value:.2f}%", ha="center", va="center", color="white", fontsize=11)
            bottom += value
        ax.text(column, bottom + 1, f"{bottom:.2f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set(xticks=range(3), xticklabels=["2 фактора", "3 фактора", "4 фактора"], ylim=(0, 68),
           ylabel="Доля общей стандартизованной дисперсии, %",
           title="Доли факторов внутри отдельно оценённых моделей")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(folder / "variance_models.png", dpi=180); plt.close(fig)
    charts.append({"title": "Дисперсия в решениях с 2, 3 и 4 факторами", "file": "assets/variance_models.png",
                   "caption": "Описательные доли FA после varimax; при изменении числа факторов все оси переоцениваются. Цвет и номер фактора не означают один и тот же конструкт между моделями."})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, pair in zip(axes, result["factor_comparisons"]):
        table = pair["congruence_matrix"]
        values = np.asarray(table["values"])
        ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set(xticks=range(len(table["column_factors"])), xticklabels=table["column_factors"],
               yticks=range(len(table["row_factors"])), yticklabels=table["row_factors"],
               xlabel=f"q={pair['target_q']}", ylabel=f"q={pair['reference_q']}",
               title=f"Нагрузки: q{pair['reference_q']} → q{pair['target_q']}")
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", color="white" if abs(value) > .65 else "black")
    fig.suptitle("Signed Tucker congruence: сходство векторов нагрузок", fontsize=14)
    fig.tight_layout(); fig.savefig(folder / "factor_congruence.png", dpi=180); plt.close(fig)
    charts.append({"title": "Сопоставление факторных осей", "file": "assets/factor_congruence.png",
                   "caption": "Косинус между векторами нагрузок по одинаковым 17 шкалам; знак отражает направление оси. Это описательное сходство, без теста статистической значимости."})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6))
    for ax, pair in zip(axes, result["fixed_k3"]):
        table = pair["contingency_aligned"]
        values = np.asarray(table["values"])
        ax.imshow(values, cmap="Blues", vmin=0, vmax=max(values.max(), 1), aspect="equal")
        ax.set(xticks=range(3), xticklabels=table["columns"], yticks=range(3), yticklabels=table["rows"],
               xlabel=f"q={pair['target_q']}: метки выровнены", ylabel=f"q={pair['reference_q']}",
               title=f"q{pair['reference_q']} → q{pair['target_q']}; ARI={pair['ari']:.3f}")
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", fontsize=13,
                        color="white" if value > values.max() * .55 else "black")
    fig.suptitle("Пересечение составов при фиксированном k=3", fontsize=14)
    fig.subplots_adjust(left=.065, right=.985, bottom=.18, top=.80, wspace=.34)
    fig.savefig(folder / "cluster_transitions_k3.png", dpi=180, bbox_inches="tight", pad_inches=.14); plt.close(fig)
    charts.append({"title": "Изменения состава кластеров при k=3", "file": "assets/cluster_transitions_k3.png",
                   "caption": "Названия меток в каждом сравнении выровнены максимизацией совпадения состава. Вне диагонали — объекты, сменившие группу после такого выравнивания; это не изменение их исходных оценок."})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for model in result["numerical_free_k"]:
        candidates = model["candidates"]
        axes[0].plot([c["k"] for c in candidates], [c["silhouette"] for c in candidates], marker="o", label=f"q={model['q']}")
        axes[1].plot([c["k"] for c in candidates], [c["stability"]["min_cluster_median_jaccard"] for c in candidates], marker="o", label=f"q={model['q']}")
    axes[0].set(ylabel="Силуэт", xlabel="Число кластеров k", title="Разделённость групп")
    axes[1].set(ylabel="Минимум медианного Jaccard по группам", xlabel="Число кластеров k", title="Условная устойчивость")
    for ax in axes:
        ax.set_xticks(range(2, 11)); ax.legend(); ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(folder / "free_k_comparison.png", dpi=180); plt.close(fig)
    charts.append({"title": "Кандидаты свободного выбора k", "file": "assets/free_k_comparison.png",
                   "caption": "30 подвыборок по 80% объектов при фиксированных координатах каждой модели; это не устойчивость всего исследования и не независимая проверка числа классов."})
    return charts


def additional_model_charts(output_root: Path) -> list[dict]:
    """Plot the extra factor planes and complete signed loadings, read-only models."""
    output = Path(output_root)
    selected = {q: output / f"q{q}" / "selected_k4" for q in [3, 4]}
    if not all((folder / "metadata.json").exists() for folder in selected.values()):
        return []
    metadata = {q: _load_json(folder / "metadata.json") for q, folder in selected.items()}
    frames = {q: pd.read_csv(folder / "scores.csv", index_col=0) for q, folder in selected.items()}
    labels = {q: pd.read_csv(folder / "cluster_labels.csv").set_index("object_id")["cluster_id"]
              for q, folder in selected.items()}
    alignment = compare_partitions(labels[3], labels[4])["overlap_alignment"]
    changed = [r["object_id"] for r in compare_partitions(labels[3], labels[4])["transitions"] if r["changed"]]
    colors = {"C1": "#2563EB", "C2": "#0F766E", "C3": "#C2410C", "C4": "#7C3AED"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    label_specs = []
    for ax, q, xfactor, yfactor in zip(axes, [3, 4], ["F1", "F1"], ["F3", "F4"]):
        frame, local = frames[q], labels[q]
        groups = sorted(local.unique(), key=lambda group: alignment[group] if q == 4 else group)
        for group in groups:
            indices = local.index[local == group]
            aligned = alignment[group] if q == 4 else group
            ax.scatter(frame.loc[indices, xfactor], frame.loc[indices, yfactor],
                       s=42, alpha=.85, color=colors[aligned], edgecolor="white", linewidth=.5,
                       label=f"{group} (n={len(indices)})")
        ax.scatter(frame.loc[changed, xfactor], frame.loc[changed, yfactor], s=90,
                   facecolors="none", edgecolors="#111827", linewidth=1.2, marker="s", zorder=3)
        ax.axhline(0, color="#64748B", linewidth=.7, alpha=.6)
        ax.axvline(0, color="#64748B", linewidth=.7, alpha=.6)
        ax.set(xlabel=f"{xfactor}, q={q}", ylabel=f"{yfactor}, q={q}",
               title=f"{q} фактора, k=4: плоскость {xfactor} × {yfactor}")
        ax.margins(x=.21, y=.21)
        ax.legend(loc="lower right", fontsize=9, framealpha=.9)
        ax.spines[["top", "right"]].set_visible(False)
        priority = list(changed)
        for rank in range(2):
            for group in metadata[q]["clusters"]:
                representatives = group.get("representatives", [])
                if len(representatives) > rank:
                    identity = representatives[rank]["id"]
                    if identity not in priority:
                        priority.append(identity)
        label_specs.append((ax, q, priority))
    fig.suptitle("Дополнительные измерения: координаты объектов в выбранных плоскостях", fontsize=15)
    fig.subplots_adjust(left=.07, right=.98, bottom=.12, top=.86, wspace=.26)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, q, priority in label_specs:
        occupied = [ax.get_legend().get_window_extent(renderer)]
        frame = frames[q]
        yfactor = "F3" if q == 3 else "F4"
        placed = 0
        for identity in priority:
            point = frame.loc[identity, ["F1", yfactor]].to_numpy(dtype=float)
            name = str(frame.loc[identity, "name"])
            for dx, dy in [(7, 7), (7, -9), (-7, 7), (-7, -9), (7, 18), (-7, 18), (7, -20), (-7, -20)]:
                annotation = ax.annotate(textwrap.fill(name, width=22), point, xytext=(dx, dy),
                                         textcoords="offset points", fontsize=9,
                                         ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top",
                                         bbox={"facecolor": "white", "alpha": .72, "edgecolor": "none", "pad": .7}, zorder=4)
                box = annotation.get_window_extent(renderer).expanded(1.04, 1.07)
                inside = ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
                if inside and not any(box.overlaps(other) for other in occupied):
                    occupied.append(box); placed += 1; break
                annotation.remove()
            if placed >= 8:
                break
        if placed < 5:
            raise RuntimeError(f"Insufficient readable object labels on q{q} factor plane")
    fig.savefig(output / "assets/new_factor_planes.png", dpi=180, bbox_inches="tight", pad_inches=.15); plt.close(fig)
    loadings = {q: pd.read_csv(output / f"q{q}" / "loadings.csv", index_col=0) for q in [3, 4]}
    if loadings[3].index.tolist() != loadings[4].index.tolist():
        raise ValueError("Loading heatmaps require identical scale ordering")
    names = [textwrap.fill(str(value), width=40) for value in loadings[3]["name"]]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 11.8), gridspec_kw={"width_ratios": [3, 4]})
    for ax, q in zip(axes, [3, 4]):
        columns = _factor_columns(loadings[q])
        values = loadings[q][columns].to_numpy()
        image = ax.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set(xticks=range(q), xticklabels=columns, yticks=range(len(names)),
               yticklabels=names if q == 3 else [""] * len(names),
               title=f"q={q}: полная матрица нагрузок", xlabel="Факторы данной модели")
        ax.tick_params(axis="y", labelsize=10)
        ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
        ax.set_xticks(np.arange(-.5, q, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=11,
                        color="white" if abs(value) > .63 else "black")
    fig.suptitle("Нагрузки всех 17 семантических шкал после varimax", fontsize=15)
    fig.subplots_adjust(left=.37, right=.92, bottom=.07, top=.92, wspace=.12)
    colorbar_axis = fig.add_axes([.94, .15, .014, .66])
    fig.colorbar(image, cax=colorbar_axis, label="Знаковая факторная нагрузка")
    fig.savefig(output / "assets/loadings_models_q3_q4.png", dpi=180, bbox_inches="tight", pad_inches=.16); plt.close(fig)
    return [{"title": "Новые факторные плоскости объектов", "file": "assets/new_factor_planes.png",
             "caption": "Показаны q3:F1/F3 и q4:F1/F4 при фактическом k=4; остальные координаты здесь не видны. Цвета групп согласованы по max-overlap, легенды сохраняют исходные номера данной модели. Чёрной рамкой отмечены четыре объекта, сменившие группу между q3k4 и q4k4. Оси условны на выбранной FA-модели; расстояния в этой проекции не заменяют расстояний в полном пространстве."},
            {"title": "Полные нагрузки решений q3 и q4", "file": "assets/loadings_models_q3_q4.png",
             "caption": "Одинаковые 17 шкал в одинаковом порядке; все знаковые нагрузки после varimax и единый цветовой диапазон −1…1. Положительная нагрузка связывает высокий фактор с положительным полюсом утверждённой шкалы, отрицательная — с отрицательным. Номера факторов локальны для каждой модели."}]


def compare_factor_models(study_path: Path, ratings_path: Path, baseline_output: Path,
                          output_root: Path | None = None,
                          selected_cluster_counts: dict[int, int] | None = None) -> dict:
    """Persist q3/q4 with fixed k3 and compare against untouched baseline q2."""
    study_path, ratings_path, baseline_output = map(lambda x: Path(x).resolve(),
                                                    [study_path, ratings_path, baseline_output])
    output = Path(output_root).resolve() if output_root else baseline_output / "models"
    if output == baseline_output or output in baseline_output.parents:
        raise ValueError("Comparison must have a dedicated output directory")
    study_path, ratings_path = freeze_input_files(study_path, ratings_path, output)
    study, ratings = _load_json(study_path), _load_json(ratings_path)
    baseline = _load_json(baseline_output / "metadata.json")
    if baseline["factor_count"] != 2 or baseline["cluster_count"] != 3:
        raise ValueError("This comparison requires a q2/k3 baseline")
    source_hashes = {"study_sha256": _sha256(study_path), "ratings_sha256": _sha256(ratings_path)}
    original_manifest = {name: _sha256(baseline_output / name) for name in MODEL_FILES if (baseline_output / name).exists()}
    prepared, _, _, _, _ = _prepare(study, ratings)
    rectangle = pd.read_csv(baseline_output / "matrix.csv", index_col=0)
    _assert_same_rectangle(prepared, rectangle)
    output.mkdir(parents=True, exist_ok=True)
    _snapshot_baseline(baseline_output, output / "q2")
    metadata = {2: _load_json(output / "q2" / "metadata.json")}
    selected_cluster_counts = selected_cluster_counts or {}
    for q in [3, 4]:
        print(f"Fitting q={q}, fixed k=3, frozen {rectangle.shape[0]}x{rectangle.shape[1]} matrix", flush=True)
        metadata[q] = analyze_study(study, ratings, output / f"q{q}", factor_count=q, cluster_count=3)
        _assert_same_rectangle(pd.read_csv(output / f"q{q}" / "matrix.csv", index_col=0), rectangle)
    models = []
    names = {str(obj["id"]): obj.get("name", str(obj["id"])) for obj in study["objects"]}
    loadings, labels = {}, {}
    for q, meta in metadata.items():
        folder = output / f"q{q}"
        frame = pd.read_csv(folder / "loadings.csv", index_col=0)
        loadings[q] = frame[_factor_columns(frame)]
        labels[q] = pd.read_csv(folder / "cluster_labels.csv").set_index("object_id")["cluster_id"]
        models.append({"q": q, "path": f"q{q}", "factor_count": q, "cluster_count": meta["cluster_count"],
                       "variance_total": meta["variance_total"],
                       "factor_shares": [{"id": f["id"], "share": f["variance_share"]} for f in meta["factors"]],
                       "numerical_recommended_k": meta["diagnostics"]["cluster_selection"]["recommended_k"],
                       "cluster_candidates": meta["cluster_candidates"], "artifacts": meta["artifacts"]})
        if q != 2 and models[-1]["numerical_recommended_k"] != 3:
            recommended = models[-1]["numerical_recommended_k"]
            export_selected_clustering(folder, study, recommended)
            models[-1]["numerical_selected_variant"] = {"k": recommended, "path": f"q{q}/selected_k{recommended}",
                                                         "factor_model_reused": True}
        if q in selected_cluster_counts and selected_cluster_counts[q] != 3:
            selected = selected_cluster_counts[q]
            if selected != models[-1]["numerical_recommended_k"]:
                export_selected_clustering(folder, study, selected)
            models[-1]["review_selected_variant"] = {"k": selected, "path": f"q{q}/selected_k{selected}",
                                                      "factor_model_reused": True, "selection_type": "explicit_substantive_review_override"}
    result = {"format_version": 1,
              "scope": {**source_hashes, "matrix_sha256": original_manifest["matrix.csv"],
                        "n_objects": len(rectangle), "n_scales": rectangle.shape[1], "seed": SEED,
                        "fixed_cluster_count": 3, "ratings_reused": True, "no_new_ratings_or_api_calls": True,
                        "input_files": {"study": {"file": "input_study.json", "sha256": source_hashes["study_sha256"]},
                                        "ratings": {"file": "input_ratings.json", "sha256": source_hashes["ratings_sha256"]}}},
              "models": models, "factor_comparisons": [], "fixed_k3": [],
              "numerical_free_k": [{"q": q, "recommended_k": meta["diagnostics"]["cluster_selection"]["recommended_k"],
                                    "selection_rule": meta["diagnostics"]["cluster_selection"]["rule"],
                                    "candidates": meta["cluster_candidates"]} for q, meta in metadata.items()],
              "limitations": ["Все модели оценены на одной и той же наблюдаемой матрице; исходные оценки не генерировались заново.",
                              "При изменении q переоцениваются и вращаются все факторы: F3/F4 не являются просто добавлением к прежним F1/F2.",
                              "Signed Tucker congruence сопоставляет оси с учётом перестановки и знака; Procrustes и углы подпространств дополняют, но не заменяют семантическую интерпретацию.",
                              "Фиксированное k=3 отделяет сравнение размерности от одновременного выбора числа групп; это анализ чувствительности, не причинный эксперимент.",
                              "Численная рекомендация свободного k предварительна и требует сравнения содержательных профилей.",
                              "Устойчивость кластеров условна на уже оценённых факторных координатах. Значимость и внешняя валидность здесь не проверяются."]}
    for reference_q, target_q in [(2, 3), (2, 4), (3, 4)]:
        reference, target = loadings[reference_q], loadings[target_q]
        if reference.index.tolist() != target.index.tolist():
            raise ValueError("Models do not share the same ordered scales")
        result["factor_comparisons"].append({"reference_q": reference_q, "target_q": target_q,
                                              **factor_congruence(reference.to_numpy(), target.to_numpy(),
                                                                  reference.columns.tolist(), target.columns.tolist())})
        result["fixed_k3"].append({"reference_q": reference_q, "target_q": target_q,
                                   **compare_partitions(labels[reference_q], labels[target_q], names)})
    result["variance_increments"] = [{"from_q": a, "to_q": b,
                                     "percentage_points": 100 * (metadata[b]["variance_total"] - metadata[a]["variance_total"]),
                                     "scope": "Difference of separately fitted models' total common-variance shares, not a single factor's share"}
                                    for a, b in [(2, 3), (3, 4), (2, 4)]]
    _export_comparison_tables(result, output)
    result["charts"] = _comparison_charts(result, output)
    if selected_cluster_counts.get(3) == selected_cluster_counts.get(4) and selected_cluster_counts.get(3, 3) != 3:
        result["selected_k_comparisons"] = [export_selected_partition_comparison(output, 3, 4, selected_cluster_counts[3])]
    if selected_cluster_counts.get(3) == 4 and selected_cluster_counts.get(4) == 4:
        result["charts"].extend(additional_model_charts(output))
    for name, digest in original_manifest.items():
        if _sha256(baseline_output / name) != digest:
            raise RuntimeError(f"Baseline changed during comparison: {name}")
    if source_hashes != {"study_sha256": _sha256(study_path), "ratings_sha256": _sha256(ratings_path)}:
        raise RuntimeError("Study/ratings changed during comparison")
    result["baseline_preserved"] = True
    result["baseline_manifest"] = original_manifest
    _write_json(output / "comparison.json", result)
    (output / "README.md").write_text(
        "# Сравнение размерностей на неизменных оценках\n\n"
        "q2 — побайтовый снимок исходного анализа; q3/q4 — новые офлайн-модели при k=3. "
        "Каждый каталог содержит полные нагрузки, координаты, профили объектов и групп, все кандидаты k и параметры проекции. "
        "Исходные study.json/ratings.json и главный output не изменяются.\n\n"
        "input_study.json и input_ratings.json — точные побайтовые снимки входов; их SHA256 записаны в scope. "
        "Повтор использует сохранённый study, поэтому добавленные позднее сведения отчёта не меняют input_sha256 модели. "
        "Изменение измерительных полей или оценок требует нового каталога сравнения.\n\n"
        "comparison.json — машиночитаемый результат; *_all_objects.csv содержит все переходы, "
        "*_changed_objects.csv — только изменения после max-overlap выравнивания меток. "
        "Равенство номера фактора между q не подразумевает равенства смысла. "
        "Совпадение кластерных меток определяется отдельно в каждой паре сравнений. "
        "Результаты свободного выбора k — численные предложения, не окончательные содержательные решения.\n\n"
        "selected_kX — варианты числа групп на тех же факторных координатах, без повторной FA. "
        "Численные и явно заданные содержательные варианты помечены раздельно.\n\n"
        "Повтор: python -m scripts.compare_factor_models STUDY_JSON RATINGS_JSON BASELINE_OUTPUT --selected-k 3:4 --selected-k 4:4\n\n"
        "Существующий снимок q2 защищён манифестом и не перезаписывается. "
        "Повторный расчёт заменяет численные q3/q4 и их стандартные описания; содержательные интерпретации применяются отдельным этапом после расчёта.\n",
        encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("ratings", type=Path)
    parser.add_argument("baseline_output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--selected-k", action="append", default=[], metavar="Q:K",
                        help="Export a substantively selected k with the same fitted model, e.g. --selected-k 3:4 --selected-k 4:4")
    args = parser.parse_args()
    selected = {}
    for pair in args.selected_k:
        q, k = map(int, pair.split(":"))
        if q not in [3, 4] or not 2 <= k <= 10:
            parser.error("--selected-k requires q3/q4 and k2..10")
        selected[q] = k
    result = compare_factor_models(args.study, args.ratings, args.baseline_output, args.output_root, selected)
    print(json.dumps({"model_totals": {model["q"]: model["variance_total"] for model in result["models"]},
                      "fixed_k3": [{key: pair[key] for key in ["reference_q", "target_q", "ari", "nmi", "moved_count"]}
                                   for pair in result["fixed_k3"]], "baseline_preserved": result["baseline_preserved"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
