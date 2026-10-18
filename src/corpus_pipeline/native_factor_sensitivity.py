"""Conditional FA stress experiment on public interval profiles; no API client.

The canonical 42-by-38 gate is not relaxed. A separately labelled panel is
selected by measurement availability, before looking at factor fits. Interval
midpoints/endpoints are assumptions, never new observed ratings.
"""
from __future__ import annotations

import itertools
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from scripts.latent_space_benchmark import fit_model


def select_panel(cells, *, minimum_actors=12, minimum_variables=6):
    actors = list(dict.fromkeys(c["actor_id"] for c in cells))
    scales = list(dict.fromkeys(c["scale_id"] for c in cells))
    index = {(c["actor_id"], c["scale_id"]): c for c in cells}
    available = np.array([[index[a, s]["lower"] is not None for s in scales] for a in actors])
    eligible = [i for i in range(len(actors)) if available[i].sum() >= minimum_variables]
    # Rwanda has 15 eligible rows. Guard against accidental combinatorial work
    # on other corpora; this diagnostic is not a general optimal panel solver.
    if len(eligible) > 22:
        raise ValueError("Availability-panel enumeration needs a bounded candidate roster")
    for n in range(len(eligible), minimum_actors - 1, -1):
        candidates = []
        for group in itertools.combinations(eligible, n):
            counts = available[list(group)].sum(axis=0)
            order = sorted((j for j in range(len(scales)) if counts[j] >= max(8, math.ceil(.6 * n))),
                           key=lambda j: (-counts[j], scales[j]))
            for p in range(len(order), minimum_variables - 1, -1):
                columns = order[:p]
                rectangle = available[np.ix_(group, columns)]
                coverage = float(rectangle.mean())
                if coverage >= .75 and (rectangle.sum(axis=1) >= max(4, math.ceil(.6 * p))).all():
                    candidates.append((p, coverage, group, columns))
                    break
        if candidates:
            candidates.sort(key=lambda item: (-item[0], -item[1], tuple(actors[i] for i in item[2])))
            p, coverage, group, columns = candidates[0]
            return {"actor_ids": [actors[i] for i in group], "scale_ids": [scales[j] for j in columns],
                    "n": n, "p": p, "coverage": coverage,
                    "rule": "Candidates have >=6 informative cells across the approved dictionary. Availability only within those candidates: maximum n, then largest prefix of scales ordered by coverage (ID tie-break), then coverage and actor IDs; >=12 actors, >=6 variables, >=8 and 60% actors per variable, >=4 and 60% variables per actor, >=75% rectangle coverage. Numerical constancy checked afterwards. This is not a global optimum over every possible subset of the 42 actors and 38 variables.",
                    "value_based_factor_selection": False}
    return None


def scenario_matrix(cells, panel, endpoint):
    if endpoint not in {"lower", "upper", "midpoint_scenario"}:
        raise ValueError("Unknown interval scenario")
    index = {(c["actor_id"], c["scale_id"]): c for c in cells}
    return np.array([[index[a, s][endpoint] for s in panel["scale_ids"]]
                     for a in panel["actor_ids"]], dtype=float)


def fit_fixed_panel(raw, scale_ids, q):
    n, p = raw.shape
    nominal_df = p * (p + 1) // 2 - (p * q + p - q * (q - 1) // 2)
    constant = [sid for sid, sd in zip(scale_ids, np.nanstd(raw, axis=0)) if sd <= 1e-10]
    if constant:
        return {"q": q, "status": "not_fitted_constant_variables_in_fixed_panel", "constant_scale_ids": constant,
                "nominal_df": nominal_df, "variance_total": None}
    if nominal_df < 0 or not 1 <= q < min(n, p):
        return {"q": q, "status": "not_fitted_parameter_count", "nominal_df": nominal_df, "variance_total": None}
    parameters, diagnostics = fit_model(raw, q, seed=20261018)
    loadings = parameters["loadings"] @ parameters["rotation"]
    filled = np.where(np.isfinite(raw), raw, parameters["training_imputation"])
    z = (filled - parameters["training_mean"]) / parameters["training_scale"]
    denominator = float(np.var(z, axis=0).sum())
    residual = z.T @ z / n - loadings @ loadings.T - np.diag(parameters["noise_variance"])
    off = ~np.eye(p, dtype=bool)
    return {"q": q, "status": "conditional_fit", "nominal_df": nominal_df,
            "variance_total": float((loadings ** 2).sum() / denominator),
            "factor_shares": ((loadings ** 2).sum(axis=0) / denominator).tolist(),
            "loadings": loadings.tolist(), "variance_denominator_variable_count": p,
            "variance_scope": "selected_panel_only_not_all_38_variables",
            "strong_indicators_per_factor": (np.abs(loadings) >= .35).sum(axis=0).tolist(),
            "residual_rms_off_diagonal": float(np.sqrt(np.mean(residual[off] ** 2))),
            "diagnostics": diagnostics}


def run_sensitivity(cells, *, repeats=20):
    panel = select_panel(cells)
    result = {"version": "native-interval-fa-conditional-v1", "canonical_42_by_38_solution": False,
              "midpoints_are_observations": False, "panel": panel, "recommended_q": None,
              "new_api_cost_usd": 0, "cluster_solution_for_42": False,
              "russia_factor_status": "not_testable_on_selected_panel"}
    if panel is None:
        result["status"] = "no_availability_panel"
        return result
    matrices = {name: scenario_matrix(cells, panel, name) for name in ("lower", "midpoint_scenario", "upper")}
    result["status"] = "conditional_sensitivity_only"
    result["scenarios"] = {}
    with threadpool_limits(limits=1):
        for name, raw in matrices.items():
            result["scenarios"][name] = [fit_fixed_panel(raw, panel["scale_ids"], q) for q in (2, 3, 4)]
        rng = np.random.default_rng(20261018)
        for reference in result["scenarios"]["midpoint_scenario"]:
            if reference["status"] != "conditional_fit":
                continue
            q = reference["q"]
            target = np.asarray(reference["loadings"])
            stress = []
            for _ in range(repeats):
                raw = matrices["lower"] + rng.random(matrices["lower"].shape) * (matrices["upper"] - matrices["lower"])
                fitted = fit_fixed_panel(raw, panel["scale_ids"], q)
                if fitted["status"] != "conditional_fit":
                    continue
                new = np.asarray(fitted["loadings"])
                denom = np.linalg.norm(target, axis=0)[:, None] * np.linalg.norm(new, axis=0)[None, :]
                congruence = np.divide(target.T @ new, denom, out=np.zeros((q, q)), where=denom > 0)
                a, b = linear_sum_assignment(-np.abs(congruence))
                stress.append(float(np.min(np.abs(congruence[a, b]))))
            reference["interval_stress"] = {"attempted_runs": repeats, "successful_runs": len(stress),
                "minimum_matched_axis_congruence": min(stress) if stress else None,
                "median_worst_axis_congruence": float(np.median(stress)) if stress else None,
                "interpretation": "Deterministic stress perturbations within reviewed-sample bounds, NOT posterior draws or confidence intervals; no likelihood assigned to positions inside bounds."}
    return result
