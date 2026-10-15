"""Isolated missing-value sensitivity review; never changes measurement policy.

The predictive experiment splits rows before fitting an imputer, then hides
originally observed cells inside validation rows. It is nested row/cell masking,
not nested hyperparameter selection. All settings are declared in advance.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
import warnings

import numpy as np
import sklearn
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import ShuffleSplit
from threadpoolctl import threadpool_limits

from scripts.latent_space_benchmark import fit_model
from src.semantic_service.analysis import InsufficientMeasurementError, SEED, _prepare
from src.semantic_service.factor_preview import _align_loading_columns, _theme_configuration


METHODS = ("mean", "bayesian_ridge", "knn")
SOURCES = [
    {"title": "scikit-learn: imputation and multiple posterior draws", "url": "https://scikit-learn.org/stable/modules/impute.html"},
    {"title": "IterativeImputer: BayesianRidge, bounds and inductive transform", "url": "https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html"},
    {"title": "scikit-learn: preventing preprocessing leakage", "url": "https://scikit-learn.org/stable/common_pitfalls.html"},
    {"title": "van Buuren & Groothuis-Oudshoorn (2011), mice", "url": "https://www.jstatsoft.org/article/view/v045i03"},
    {"title": "Rubin (1976), Inference and Missing Data", "url": "https://dash.harvard.edu/entities/publication/73120378-8764-6bd4-e053-0100007fdf3b"},
    {"title": "lavaan: missing-data ML/FIML", "url": "https://lavaan.ugent.be/tutorial/est.html"},
]


def coverage_sensitivity(study, ratings):
    """Reapply unchanged admission to exclusion copies; never fit a strict FA.

    Follow-ups are identified only by the CURRENT cell's provenance.method,
    not by an old rejected proposal nested in its history.
    """
    scenarios = []
    for name, flags, followups in [("without_flagged", True, False),
                                   ("without_followup_layer", False, True),
                                   ("without_flagged_and_followup_layer", True, True)]:
        altered, removed = deepcopy(ratings), []
        for row in altered:
            cells = row.get("pole_scores")
            entries = cells.items() if isinstance(cells, dict) else [(None, row)]
            for pole, cell in entries:
                if cell.get("status", "observed") != "observed":
                    continue
                method = str((cell.get("provenance") or {}).get("method", ""))
                flagged = (cell.get("source_audit") or {}).get("decision") == "flag"
                followup = "completion" in method or "coverage_expansion" in method
                if flags and flagged or followups and followup:
                    removed.append({"object_id": row.get("object_id"), "scale_id": row.get("scale_id"),
                                    "pole": pole, "flagged": flagged, "followup": followup,
                                    "provenance_method": method})
                    cell["status"], cell["value"] = "unknown", None
        try:
            _, _, _, _, diagnostics = _prepare(study, altered)
            status, reason = "admitted_no_FA_fitted", None
        except InsufficientMeasurementError as error:
            diagnostics, status, reason = error.diagnostics, "insufficient_data_no_FA_fitted", str(error)
        scenarios.append({"scenario": name, "removed_properties": removed, "removed_count": len(removed),
                          "measurement_status": status, "reason": reason, "diagnostics": diagnostics,
                          "FA_fitted": False})
    return {"selection": "Remove all currently observed flagged cells, all current completion/coverage_expansion provenance cells, and their union, separately. Flags indicate review limitations, not automatically erroneous measurements.",
            "scenarios": scenarios}


def _array(raw):
    result = np.asarray(raw, dtype=float)
    if result.ndim != 2 or not result.size or np.isinf(result).any():
        raise ValueError("A nonempty two-dimensional finite-or-NaN matrix is required")
    if np.isnan(result).all(axis=0).any():
        raise ValueError("An entirely unmeasured variable cannot be imputed")
    return result.copy()


def _bounds(bounds, p):
    limits = np.asarray(bounds, dtype=float)
    if limits.shape == (2,):
        limits = np.tile(limits, (p, 1))
    if limits.shape != (p, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Finite increasing bounds are required for every variable")
    return limits


def _estimator(method, bounds, seed, posterior=False, max_iter=20):
    if method == "mean":
        if posterior:
            raise ValueError("Posterior draws require bayesian_ridge")
        return SimpleImputer(strategy="mean")
    if method == "knn":
        if posterior:
            raise ValueError("Posterior draws require bayesian_ridge")
        return KNNImputer(n_neighbors=5, weights="distance", metric="nan_euclidean")
    if method != "bayesian_ridge":
        raise ValueError(f"Unknown imputation method: {method}")
    return IterativeImputer(estimator=BayesianRidge(), sample_posterior=posterior,
                            max_iter=max_iter, tol=1e-3, initial_strategy="mean",
                            imputation_order="ascending", skip_complete=False,
                            min_value=bounds[:, 0], max_value=bounds[:, 1], random_state=seed)


def impute_from_training(train, target, *, method, bounds=(0., 5.), seed=SEED,
                         posterior=False, max_iter=20):
    """Fit ONLY on train; preserve observed target cells, clip missing estimates.

    In validation, callers must remove hidden target truths before this call.
    No empty feature is silently replaced with the sklearn zero fallback.
    """
    train, target = _array(train), np.asarray(target, dtype=float).copy()
    if target.ndim != 2 or target.shape[1] != train.shape[1] or np.isinf(target).any():
        raise ValueError("Target must use the finite-or-NaN training variable order")
    limits = _bounds(bounds, train.shape[1])
    for values in (train, target):
        observed = np.isfinite(values)
        if np.any(observed & ((values < limits[:, 0]) | (values > limits[:, 1]))):
            raise ValueError("Observed data lie outside the declared measurement bounds")
    if np.any(np.isfinite(target).sum(axis=1) == 0):
        raise ValueError("An entirely unmeasured object cannot receive imputed coordinates")
    estimator = _estimator(method, limits, seed, posterior, max_iter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(train)
        complete = estimator.transform(target)
    if complete.shape != target.shape or not np.isfinite(complete).all():
        raise ValueError("Imputer did not return the identical finite variable space")
    missing = ~np.isfinite(target)
    clipped = np.clip(complete, limits[:, 0], limits[:, 1])
    clip_count = int(np.count_nonzero(missing & (complete != clipped)))
    # Estimates never overwrite measurements, including an explicitly observed 0.
    complete = np.where(missing, clipped, target)
    return complete, {"method": method, "posterior": posterior,
                      "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
                      "iterations": int(getattr(estimator, "n_iter_", 0)),
                      "clipped_missing_cells": clip_count,
                      "fit_rows": len(train), "transformed_rows": len(target)}


def score_masked_trial(train, validation, hidden, *, method, bounds=(0., 5.), seed=SEED):
    """Predict a supplied hidden mask without passing those truths to imputation."""
    validation = np.asarray(validation, dtype=float)
    hidden = np.asarray(hidden, dtype=bool)
    if hidden.shape != validation.shape or not hidden.any() or not np.isfinite(validation[hidden]).all():
        raise ValueError("Hidden mask must select originally observed validation cells")
    masked = validation.copy()
    masked[hidden] = np.nan
    completed, fitting = impute_from_training(train, masked, method=method, bounds=bounds, seed=seed)
    truth, predicted = validation[hidden], completed[hidden]
    error = predicted - truth
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "truth": truth.tolist(), "predicted": predicted.tolist(), "fit": fitting}


def _mask_validation(validation, fraction, rng):
    observed = np.isfinite(validation)
    possible = np.argwhere(observed)
    requested = max(1, int(round(fraction * len(possible))))
    mask = np.zeros_like(observed)
    for position in rng.permutation(len(possible)):
        row, column = possible[position]
        # Retain at least two measured predictors in a held-out object.
        if observed[row].sum() - mask[row].sum() <= 2:
            continue
        mask[row, column] = True
        if mask.sum() == requested:
            break
    if not mask.any():
        raise ValueError("No validation cells can be hidden while retaining measured predictors")
    return mask


def masked_validation(raw, *, repeats=20, mask_fraction=.12, seed=SEED, bounds=(0., 5.)):
    raw = _array(raw)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("At least two repeated row splits are required")
    if not .1 <= mask_fraction <= .15:
        raise ValueError("The comparison protocol requires 10–15% masking")
    trials, failures = [], []
    splitter = ShuffleSplit(n_splits=repeats, test_size=.25, random_state=seed)
    for repeat, (train_ids, validation_ids) in enumerate(splitter.split(raw)):
        train, validation = raw[train_ids], raw[validation_ids]
        rng = np.random.default_rng(seed + 1000 + repeat)
        try:
            if np.any(np.isfinite(train).sum(axis=0) < 2):
                raise ValueError("A training variable has fewer than two actual observations")
            hidden = _mask_validation(validation, mask_fraction, rng)
            results = {method: score_masked_trial(train, validation, hidden, method=method,
                                                  bounds=bounds, seed=seed + repeat)
                       for method in METHODS}
            cells = [[int(validation_ids[i]), int(j)] for i, j in np.argwhere(hidden)]
            trials.append({"repeat": repeat, "train_rows": train_ids.tolist(),
                           "validation_rows": validation_ids.tolist(), "hidden_cells": cells,
                           "validation_observed_cells_before_masking": int(np.isfinite(validation).sum()),
                           "hidden_cell_count": len(cells), "methods": results})
        except (ValueError, np.linalg.LinAlgError) as error:
            failures.append({"repeat": repeat, "reason": str(error)})
    summary = []
    for method in METHODS:
        values = [trial["methods"][method] for trial in trials]
        errors = np.concatenate([np.asarray(v["predicted"]) - np.asarray(v["truth"]) for v in values]) if values else np.array([])
        per_variable = []
        for column in range(raw.shape[1]):
            errors_j = [result["methods"][method]["predicted"][k] - result["methods"][method]["truth"][k]
                        for result in trials for k, (_, j) in enumerate(result["hidden_cells"]) if j == column]
            per_variable.append({"column": column, "count": len(errors_j),
                                 "mae": float(np.mean(np.abs(errors_j))) if errors_j else None,
                                 "rmse": float(np.sqrt(np.mean(np.square(errors_j)))) if errors_j else None})
        summary.append({"method": method, "evaluated_cell_occurrences": int(len(errors)),
                        "pooled_mae": float(np.mean(np.abs(errors))) if len(errors) else None,
                        "pooled_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
                        "repeat_mae_mean": float(np.mean([v["mae"] for v in values])) if values else None,
                        "repeat_mae_sd": float(np.std([v["mae"] for v in values], ddof=1)) if len(values) > 1 else None,
                        "convergence_warnings": sum(v["fit"]["convergence_warning"] for v in values),
                        "by_variable": per_variable})
    return {"requested_repeats": repeats, "successful_repeats": len(trials), "failures": failures,
            "mask_fraction": mask_fraction, "summary": summary, "trials": trials,
            "design": "Split rows 75/25 before fitting; hide 10–15% of originally observed validation cells; fit exclusively on training rows; transform masked validation without refitting. The same split and mask are used for every method.",
            "selection": "Fixed method settings; no hyperparameter or method selection is validated by these scores.",
            "scope": "Nested row/cell masking, not nested hyperparameter CV. Fixed matrix selected before this experiment; conditional error on randomly hidden known cells, not an estimate of MNAR error or population generalization. Repeated cells/splits overlap; SD is descriptive, not a confidence interval."}


def _admissible_dimensions(n, p, requested):
    valid, excluded = [], []
    for q in requested:
        if isinstance(q, bool) or not isinstance(q, int) or q < 2:
            raise ValueError("Requested FA dimensions must be integers >=2")
        parameters = p * q + p - q * (q - 1) // 2
        entry = {"q": q, "parameter_count": parameters, "nominal_df": p * (p + 1) // 2 - parameters}
        if q >= min(n, p) or entry["nominal_df"] < 0:
            excluded.append(entry)
        else:
            valid.append(entry)
    return valid, excluded


def _aligned_factor_result(completed, q, reference, variable_ids, seed):
    parameters, fitting = fit_model(completed, q, seed=seed)
    loadings = parameters["loadings"] @ parameters["rotation"]
    aligned, congruence = _align_loading_columns(reference, loadings)
    denominator = float(np.var((completed - parameters["training_mean"]) / parameters["training_scale"], axis=0).sum())
    shares = np.sum(aligned ** 2, axis=0) / denominator
    return {"q": q, "variance_total": float(shares.sum()), "variance_denominator": denominator,
            "aligned_factor_shares": shares.tolist(), "congruence_to_mean_reference": congruence.tolist(),
            "aligned_loadings": [{"id": sid, "loadings": {f"F{j+1}": float(aligned[i, j]) for j in range(q)}}
                                 for i, sid in enumerate(variable_ids)], "fit_diagnostics": fitting}


def factor_imputation_sensitivity(raw, variable_ids, *, q_values=(2, 3, 4), posterior_draws=10,
                                  seed=SEED, bounds=(0., 5.)):
    raw = _array(raw)
    if len(variable_ids) != raw.shape[1] or len(set(variable_ids)) != len(variable_ids):
        raise ValueError("Unique ordered variable IDs are required")
    if isinstance(posterior_draws, bool) or not isinstance(posterior_draws, int) or posterior_draws < 2:
        raise ValueError("At least two posterior draws are required for a dispersion review")
    admissible, excluded = _admissible_dimensions(*raw.shape, list(dict.fromkeys(q_values)))
    completions, settings = {}, {}
    for method in METHODS:
        completions[method], settings[method] = impute_from_training(raw, raw, method=method, bounds=bounds, seed=seed)
    draws, draw_settings = [], []
    for draw in range(posterior_draws):
        complete, fitting = impute_from_training(raw, raw, method="bayesian_ridge", bounds=bounds,
                                                posterior=True, seed=seed + 5000 + draw)
        draws.append(complete)
        draw_settings.append(fitting)
    comparisons = []
    for dimension in admissible:
        q = dimension["q"]
        mean_parameters, _ = fit_model(raw, q, seed=seed)
        reference = mean_parameters["loadings"] @ mean_parameters["rotation"]
        deterministic = [{"method": method, **_aligned_factor_result(completions[method], q, reference, variable_ids, seed)}
                         for method in METHODS]
        # Baseline completion must reproduce existing mean-imputed FA exactly.
        if not np.allclose(np.array([[r["loadings"][f"F{j+1}"] for j in range(q)] for r in deterministic[0]["aligned_loadings"]]), reference, rtol=0, atol=1e-10):
            raise ValueError("Mean-completion FA does not reproduce the existing reference")
        variants = [{"draw": draw, **_aligned_factor_result(complete, q, reference, variable_ids, seed)}
                    for draw, complete in enumerate(draws)]
        axes = []
        for j in range(q):
            loadings_j = np.array([[r["loadings"][f"F{j+1}"] for r in variant["aligned_loadings"]] for variant in variants])
            axes.append({"factor_id": f"F{j+1}",
                         "congruence_q05_q50_q95": np.quantile([v["congruence_to_mean_reference"][j] for v in variants], [.05, .5, .95]).tolist(),
                         "variance_share_q05_q50_q95": np.quantile([v["aligned_factor_shares"][j] for v in variants], [.05, .5, .95]).tolist(),
                         "loading_quantiles": [{"id": sid, "reference_loading": float(reference[i, j]),
                                                "q05_q50_q95": np.quantile(loadings_j[:, i], [.05, .5, .95]).tolist(),
                                                "same_sign_and_abs_ge_035_fraction": float(np.mean((np.sign(loadings_j[:, i]) == np.sign(reference[i, j])) & (np.abs(loadings_j[:, i]) >= .35)))}
                                               for i, sid in enumerate(variable_ids)]})
        comparisons.append({**dimension, "deterministic": deterministic,
                            "posterior": {"draws": variants, "axes": axes,
                                          "variance_total_q05_q50_q95": np.quantile([v["variance_total"] for v in variants], [.05, .5, .95]).tolist()}})
    missing_cells = []
    stack = np.asarray(draws)
    for i, j in np.argwhere(~np.isfinite(raw)):
        missing_cells.append({"row": int(i), "column": int(j), "variable_id": variable_ids[j],
                              "measurement_status": "unknown", "estimate_is_observation": False,
                              "deterministic_estimates": {method: float(x[i, j]) for method, x in completions.items()},
                              "posterior_estimate_q05_q50_q95": np.quantile(stack[:, i, j], [.05, .5, .95]).tolist()})
    return {"models": comparisons, "excluded_dimensions": excluded, "imputer_fit": settings,
            "posterior_draw_fit": draw_settings, "posterior_draws": posterior_draws,
            "missing_cell_estimates": missing_cells,
            "alignment": "Each same-q FA is matched to the mean-imputation reference by maximum absolute Tucker congruence and sign alignment. Quantiles are descriptive across completed datasets; no unaligned axes are averaged and no Rubin pooled-factor inference is claimed.",
            "bounds_note": "Only imputed entries are bounded; observed values remain exactly unchanged. Fractional estimates are not rounded into fabricated ratings.",
            "FIML": "Not implemented here. Existing FA training uses mean completion; marginalizing missing coordinates for posterior object scores does not turn that training fit into FIML."}


def review_imputation(study, ratings, output_dir: Path, *, repeats=20, mask_fraction=.12,
                       posterior_draws=10, q_values=(2, 3, 4), seed=SEED,
                       output_stem="imputation_review_2026-10-15"):
    """Run a read-only measurement review, exporting only separate JSON/Markdown."""
    if not output_stem or Path(output_stem).name != output_stem:
        raise ValueError("output_stem must be a filename stem, not a path")
    receipt = {"study_content_sha256": hashlib.sha256(json.dumps(study, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
               "ratings_content_sha256": hashlib.sha256(json.dumps(ratings, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    result = {"format_version": 1, "mode": "isolated_imputation_review", "seed": seed,
              "sklearn_version": sklearn.__version__, "input_receipt": receipt,
              "measurement_policy_changed": False, "sources": SOURCES,
              "method_parameters": {"mean": "observed training mean per variable",
                                    "bayesian_ridge": "IterativeImputer(BayesianRidge), max_iter=20, tol=.001, all other variables, bounded estimates",
                                    "knn": "KNNImputer, k=5, distance weighting, nan_euclidean; original common 0..5 property units"}}
    result["coverage_sensitivity"] = coverage_sensitivity(study, ratings)
    try:
        matrix, _, object_map, scale_map, diagnostics = _prepare(study, ratings)
    except InsufficientMeasurementError as error:
        result.update(measurement_status="insufficient_data", reason=str(error), diagnostics=error.diagnostics,
                      validation=None, factor_sensitivity=None,
                      coverage_expansion="Not attempted: imputation cannot bypass the fixed real-observation admission policy")
    else:
        variable_ids = matrix.columns.tolist()
        limits = [(0., 5.) if scale_map[sid].get("kind") == "unipolar" else (-5., 5.) for sid in variable_ids]
        config = _theme_configuration(study)
        theme = set(config["configured_ids"])
        thematic_rows = [x for x in diagnostics["input_scale_coverage"] if x["id"] in theme]
        raw = matrix.to_numpy(dtype=float)
        with threadpool_limits(limits=1):
            validation = masked_validation(raw, repeats=repeats, mask_fraction=mask_fraction, seed=seed, bounds=limits)
            sensitivity = factor_imputation_sensitivity(raw, variable_ids, q_values=q_values,
                                                         posterior_draws=posterior_draws, seed=seed, bounds=limits)
        for entry in sensitivity["missing_cell_estimates"]:
            entry["object_id"] = matrix.index[entry["row"]]
        result.update(measurement_status="admitted_real_observations", diagnostics=diagnostics,
                      n_objects=len(matrix), n_variables=matrix.shape[1],
                      object_ids=matrix.index.tolist(), variable_ids=variable_ids,
                      variable_names={sid: scale_map[sid].get("name", sid) for sid in variable_ids},
                      matrix_sha256=hashlib.sha256(matrix.to_csv().encode("utf-8")).hexdigest(),
                      validation=validation, factor_sensitivity=sensitivity,
                      russia_block={"configured_variables": len(theme), "observed_input_properties": sum(x["observed_objects"] for x in thematic_rows),
                                    "input_cells": len(study.get("objects", [])) * len(thematic_rows),
                                    "retained_variables": [sid for sid in variable_ids if sid in theme],
                                    "excluded_variables_not_imputed": [x["id"] for x in thematic_rows if x["id"] not in variable_ids],
                                    "claim": "No excluded thematic variable is reconstructed from office, general factors or other proxies. Insufficient measurement is not evidence for or against a Russia factor."},
                      excluded_objects=[{"id": x["id"], "name": object_map[x["id"]].get("name", x["id"])} for x in diagnostics["excluded_objects"]],
                      coverage_expansion="None: retained objects and variables are fixed by unchanged _prepare, using actual observations before any imputation.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result["artifacts"] = [{"name": output_stem, "path": output_stem + ".json", "type": "json"},
                           {"name": output_stem, "path": output_stem + ".md", "type": "markdown"}]
    (output_dir / (output_stem + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / (output_stem + ".md")).write_text(render_review(result), encoding="utf-8")
    return result


def render_review(result):
    lines = ["# Проверка способов заполнения небольших пропусков", "",
             "Импутация допустима как отдельная модельная оценка после достаточного реального измерения. Она не превращает неизвестное свойство в наблюдавшийся факт и не заменяет сбор источников. Порог допуска к FA здесь не изменён.", ""]
    strict_lines = ["", "Проверка допуска после исключения спорных/дополнительных измерений", "",
                    "Во всех сценариях заново применяется тот же отбор по реальным наблюдениям до импутации. Пропуски не заполняются для достижения порога. Этот модуль не обучает FA сценариев допуска; отдельный расчёт может быть сохранён в другом приложении. Флаг не объявляет значение заведомо ошибочным.", "",
                    "| Сценарий | Исключено оценок | Осталось объектов × переменных | Покрытие | Допуск |",
                    "|---|---:|---:|---:|---|"]
    for scenario in result["coverage_sensitivity"]["scenarios"]:
        quality = scenario["diagnostics"]
        coverage = f"{100*quality['retained_coverage']:.2f}%" if quality["retained_coverage"] is not None else "не определено"
        admitted = "да; отдельный расчёт FA" if scenario["measurement_status"] == "admitted_no_FA_fitted" else "нет; FA недопустима"
        strict_lines.append(f"| {scenario['scenario']} | {scenario['removed_count']} | {quality['retained_objects']} × {quality['retained_scales']} | {coverage} | {admitted} |")
    strict_lines += ["", "Дополнительный слой определяется по текущему provenance.method, содержащему completion или coverage_expansion; старые значения из истории записи не используются. Это единое исключение слоя, без выбора отдельных оценок по желаемым факторам."]
    if result["measurement_status"] == "insufficient_data":
        lines += ["**Данных недостаточно.** " + result["reason"], "",
                  "Импутация и факторные модели не запускались; исключённые объекты и шкалы не возвращались искусственным заполнением."]
        return "\n".join(lines + strict_lines) + "\n"
    d, validation = result["diagnostics"], result["validation"]
    lines += [f"Используется допущенная матрица **{result['n_objects']} × {result['n_variables']}**: {d['missing_retained_cells']} неизвестных ячеек, покрытие {100*d['retained_coverage']:.2f}%. Исключённые {len(result['excluded_objects'])} объектов не получают вымышленных координат.", "",
              f"Проверка: {validation['successful_repeats']}/{validation['requested_repeats']} разделений объектов 75/25; внутри validation скрывается {100*validation['mask_fraction']:.0f}% известных оценок. Обучение — только на train. Одни и те же маски применяются ко всем методам. Это двухуровневая маскировка строк/ячеек с заранее фиксированными настройками; не nested CV для подбора гиперпараметров. [Защита от утечки](https://scikit-learn.org/stable/common_pitfalls.html).", "",
              "| Метод | MAE | RMSE | Число проверенных ячеек с повторами |", "|---|---:|---:|---:|"]
    for entry in validation["summary"]:
        mae = f"{entry['pooled_mae']:.3f}" if entry['pooled_mae'] is not None else "нет"
        rmse = f"{entry['pooled_rmse']:.3f}" if entry['pooled_rmse'] is not None else "нет"
        lines.append(f"| {entry['method']} | {mae} | {rmse} | {entry['evaluated_cell_occurrences']} |")
    available = [entry for entry in validation["summary"] if entry["pooled_mae"] is not None]
    if available:
        lowest = min(available, key=lambda entry: entry["pooled_mae"])
        lines += ["", f"В этом фиксированном эксперименте наименьшая MAE у `{lowest['method']}`. Это описательное сравнение на тех же разделениях, а не независимо подтверждённый универсальный выбор метода."]
    warned = [f"{entry['method']}: {entry['convergence_warnings']}" for entry in validation["summary"] if entry["convergence_warnings"]]
    if warned:
        lines += ["", "Предупреждения сходимости при фиксированном лимите 20 итераций: " + "; ".join(warned) + ". Соответствующие прогнозы — результаты усечённого алгоритма; это дополнительное ограничение сравнения, подробности сохранены по повторам."]
    lines += ["", "Ошибки относятся к случайно скрытым **известным** ячейкам уже выбранной матрицы. Это не проверка качества настоящих неизвестных политических свойств: их пропуски могут зависеть от содержания/публичности. Повторные ячейки и разделения перекрываются; описательный разброс не является доверительным интервалом. Должности и сведения об объектах не используются предикторами.", "",
              "Среднее — прозрачный baseline, но стягивает заполненные значения к центру и меняет ковариации. Регуляризованная регрессия использует связи свойств, KNN — близкие строки; их предположения также могут быть ошибочны. Известные оценки сохранены точно, заполнения ограничены допустимым диапазоном и не округляются в якобы новые факты. [Документация алгоритмов](https://scikit-learn.org/stable/modules/impute.html).", "",
              "Чувствительность факторных решений", "",
              "| q | Метод | Общая доля FA, % | Конгруэнтность осей с mean-reference |", "|---:|---|---:|---|"]
    for model in result["factor_sensitivity"]["models"]:
        for entry in model["deterministic"]:
            cong = "; ".join(f"{x:.3f}" for x in entry["congruence_to_mean_reference"])
            lines.append(f"| {model['q']} | {entry['method']} | {100*entry['variance_total']:.2f} | {cong} |")
    lines += ["", "Доли — квадраты нагрузок общей FA / всю стандартизованную дисперсию оставшихся переменных, не PCA. Увеличение доли после заполнения не доказывает улучшения. Полные подписанные нагрузки и доли каждой оси сохранены в JSON.", "",
              f"Для неопределённости выполнены {result['factor_sensitivity']['posterior_draws']} независимых по seed последовательных регрессионных заполнений с `sample_posterior=True`, затем отдельная FA каждого q. [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) поддерживает такие реализации; подход мотивирован [MICE](https://www.jstatsoft.org/article/view/v045i03). Оси сначала сопоставлены по конгруэнтности и знаку; неусогласованные оси не усредняются. Это описательная чувствительность, не Rubin-пулинг параметров и не доверительные интервалы.", "",
              "| q | Ось | Конгруэнтность: 5% / медиана / 95% | Доля оси: 5% / медиана / 95%, % |", "|---:|---|---|---|"]
    for model in result["factor_sensitivity"]["models"]:
        for axis in model["posterior"]["axes"]:
            cong = " / ".join(f"{x:.3f}" for x in axis["congruence_q05_q50_q95"])
            shares = " / ".join(f"{100*x:.2f}" for x in axis["variance_share_q05_q50_q95"])
            lines.append(f"| {model['q']} | {axis['factor_id']} | {cong} | {shares} |")
    block = result["russia_block"]
    lines += ["", f"**Россия:** {block['observed_input_properties']}/{block['input_cells']} известных тематических оценок; в матрице {len(block['retained_variables'])} тематических переменных. Исключённый разреженный блок не импутируется из общих осей или должностей. Заполнение не может установить российский фактор, если его собственные индикаторы фактически не измерены.", "",
              "Допущение о механизме пропусков существенно: MAR — условие модели, а не свойство, которое доказывает низкая ошибка искусственной маскировки. Регрессия и multiple imputation не устраняют MNAR автоматически; основания игнорирования механизма описаны у [Rubin (1976)](https://dash.harvard.edu/entities/publication/73120378-8764-6bd4-e053-0100007fdf3b).", "",
              "FIML здесь не реализован. Например, [lavaan](https://lavaan.ugent.be/tutorial/est.html) предоставляет прямое ML-оценивание по неполным наблюдениям при соответствующих предположениях. Наша существующая FA обучается после mean completion; учёт отсутствующих координат при posterior projection объектов не превращает обучение в FIML.", "",
              "**Практический совет:** сначала собирать сопоставимые прямые сведения, чтобы больше из 42 объектов прошло прежний допуск. Небольшие оставшиеся пропуски можно заполнять в аналитических копиях, сохраняя маску неизвестного и сравнивая mean с регуляризованными множественными реализациями. Выбирать способ нельзя только по большему проценту дисперсии или желаемому фактору; нужны одновременно ошибки восстановления и устойчивость смысла. Исходные оценки и основной preview этим модулем не меняются."]
    return "\n".join(lines + strict_lines) + "\n"
