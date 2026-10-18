"""Reproducible no-API comparison on PUBLIC Rwanda rating bounds.

python -B scripts/compare_rwanda_incomplete_latent.py
No corpus, API credentials or external network are accessed. Output is a new
exploratory benchmark; approved scores and historical study files are untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from src.corpus_pipeline.incomplete_latent import (
    Priors, best_ridge, canonical_axes, compare_geometry, gibbs_factor_analysis,
    chain_diagnostics, holdout_mask, prediction_mean, prediction_metrics,
    summarize_bayesian, validation_folds,
)

DESTINATION = ROOT / 'research/rwanda_politicians_2021_2026/source_scales_v2/incomplete_latent_2026_10_18'
PENALTIES = [.1, .3, 1., 3., 10.]


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def input_data():
    bounds_path = ROOT / 'reports/rwanda_native_mass_bounds_2026_10_18.json'
    profile_path = ROOT / 'reports/rwanda_native_mass_profiles_2026_10_18.json'
    bounds = json.loads(bounds_path.read_text(encoding='utf-8'))
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    cells = bounds['content_supported_unknown_time_sensitivity']['cells']
    actors = list(dict.fromkeys(c['actor_id'] for c in cells))
    scales = list(dict.fromkeys(c['scale_id'] for c in cells))

    def matrix(cells, field):
        idx = {(c['actor_id'], c['scale_id']): c for c in cells}
        return np.array([[idx[a, s][field] for s in scales] for a in actors], float)

    lo, hi = matrix(cells, 'lower'), matrix(cells, 'upper')
    mid = (lo + hi) / 2
    # Availability only. Two or fewer actors cannot establish a shared trait
    # direction credibly. No values, political groups, or fit scores select it.
    keep_columns = np.isfinite(mid).sum(axis=0) >= 3
    keep_rows = np.isfinite(mid[:, keep_columns]).any(axis=1)
    subset = np.ix_(keep_rows, keep_columns)
    strict = matrix(bounds['strict_dated']['cells'], 'midpoint_scenario')[subset]
    names = {a['id']: a.get('name', a.get('label', a['id'])) for a in profile['actors']}
    definitions = {s['id']: s['name'] for s in profile['scales']}
    return {
        'actor_ids': np.array(actors)[keep_rows].tolist(),
        'scale_ids': np.array(scales)[keep_columns].tolist(),
        'actor_names': names, 'scale_names': definitions,
        'excluded_actors': np.array(actors)[~keep_rows].tolist(),
        'excluded_scales': [{'id': s, 'observations': int(np.isfinite(mid[:, j]).sum())}
                            for j, s in enumerate(scales) if not keep_columns[j]],
        'lower': lo[subset], 'upper': hi[subset], 'midpoint': mid[subset],
        'exact_mask': (lo == hi)[subset] & np.isfinite(lo[subset]),
        'strict': strict,
        'input_hashes': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (bounds_path, profile_path)},
        # Git may normalize CRLF/LF. Content identity must remain portable.
        'input_content_hashes': {p.relative_to(ROOT).as_posix(): hashlib.sha256(json.dumps(
            json.loads(p.read_text(encoding='utf-8')), sort_keys=True, ensure_ascii=False,
            separators=(',', ':')).encode('utf-8')).hexdigest() for p in (bounds_path, profile_path)},
    }


def choose_penalty(x, exact, q, seed, repeats=3):
    splits = [holdout_mask(x, exact, seed + k) for k in range(repeats)]
    rows = []
    for penalty in PENALTIES:
        errors = []
        for k, target in enumerate(splits):
            training = np.where(target, np.nan, x)
            fitted = best_ridge(training, q, penalty, starts=3, seed=seed + 17 * k)
            errors.append(prediction_metrics(x[target], fitted['prediction'][target])['rmse'])
        rows.append({'penalty': penalty, 'mean_rmse': float(np.mean(errors)), 'split_rmse': errors})
    # Exact ties favour stronger regularization, fixed before interpretation.
    chosen = min(rows, key=lambda row: (round(row['mean_rmse'], 10), -row['penalty']))['penalty']
    return chosen, rows


def fit(x, q, method, *, seed, penalty=None, warmup=600, draws=1000, chains=2, priors=Priors()):
    if method == 'bayesian_fa':
        result = summarize_bayesian(gibbs_factor_analysis(
            x, q, seed=seed, warmup=warmup, draws=draws, chains=chains, priors=priors), np.isfinite(x))
        if result['diagnostics']['max_rank_split_rhat'] > 1.01:
            first = result['diagnostics']
            result = summarize_bayesian(gibbs_factor_analysis(x, q, seed=seed,
                warmup=max(1500, warmup), draws=max(3000, draws), chains=4, priors=priors), np.isfinite(x))
            result['diagnostics']['extended_from'] = first
        # A near-threshold sensitivity fit occasionally needs more draws even
        # after the standard extension. Keep the deterministic rule reproducible.
        extra_draws = max(6000, draws * 2)
        while result['diagnostics']['max_rank_split_rhat'] > 1.01 and extra_draws <= 12000:
            previous = result['diagnostics']
            result = summarize_bayesian(gibbs_factor_analysis(x, q, seed=seed,
                warmup=max(1500, warmup), draws=extra_draws, chains=4, priors=priors), np.isfinite(x))
            result['diagnostics']['extended_from'] = previous
            extra_draws *= 2
        return result
    result = best_ridge(x, q, penalty, seed=seed, starts=6)
    result['scores'], result['loadings'], _ = canonical_axes(result['scores'], result['loadings'])
    result['signal_singular_values'] = np.linalg.svd(
        result['scores'] @ result['loadings'].T, compute_uv=False)[:q]
    result['effective_rank_threshold_0_01'] = int((result['signal_singular_values'] > .01).sum())
    result.pop('history')
    return result


def geometry(reference, candidate, x, *, common_rows=None, common_columns=None):
    if common_rows is None:
        common_rows = np.ones(len(x), bool)
    if common_columns is None:
        common_columns = np.ones(x.shape[1], bool)
    rich = common_rows & (np.isfinite(x).sum(axis=1) >= 4)
    result = compare_geometry(reference, candidate, row_mask=common_rows, feature_mask=common_columns)
    result['rich_actor_distance_spearman'] = compare_geometry(
        reference, candidate, row_mask=rich, feature_mask=common_columns)['distance_spearman']
    result['compared_actors'] = int(common_rows.sum())
    result['compared_scales'] = int(common_columns.sum())
    return result


def run(*, quick=False, stress_repeats=20):
    start = time.monotonic()
    data = input_data()
    x, exact = data['midpoint'], data['exact_mask']
    result = {
        'version': 'rwanda-incomplete-latent-v1', 'seed': 20261018, 'new_api_calls': 0,
        'new_api_cost_usd': 0, 'source_ratings_modified': False,
        'main_scenario': 'content_supported_unknown_time_sensitivity; interval midpoints are assumptions',
        'selection_rule': 'Columns with >=3 informative actors, then rows with >=1 informative retained trait; no intensity or factor fits used in selection',
        'priors': asdict(Priors()), 'penalty_grid': PENALTIES,
        'gaussian_approximation_to_ordinal_0_to_5': True,
        'coverage': {'actors': x.shape[0], 'scales': x.shape[1],
                     'exact_cells': int(exact.sum()), 'interval_scenarios': int((np.isfinite(x) & ~exact).sum()),
                     'informative_cells': int(np.isfinite(x).sum()), 'possible_cells': x.size},
        **{k: v for k, v in data.items() if k not in {'lower', 'upper', 'midpoint', 'exact_mask', 'strict'}},
        'configuration': {'quick': quick, 'stress_repeats': stress_repeats,
                          'main_warmup': 300 if quick else 1500,
                          'main_draws_per_chain': 400 if quick else 3000,
                          'main_chains': 2 if quick else 4,
                          'cv_outer_splits': 2 if quick else 5,
                          'cv_inner_splits': 2 if quick else 3,
                          'stress_warmup': 150 if quick else 500,
                          'stress_draws_per_chain': 200 if quick else 800,
                          'stress_chains': 2},
        'models': {}, 'validation': [], 'stress': [], 'scenario_fits': {}, 'influence': [],
    }
    n_outer = result['configuration']['cv_outer_splits']
    splits, fold_coverage = validation_folds(x, exact, 8100, folds=n_outer)
    result['validation_coverage'] = fold_coverage
    result['holdouts'] = [[{'actor_id': data['actor_ids'][i], 'scale_id': data['scale_ids'][j]}
                           for i, j in np.argwhere(target)] for target in splits]
    for k, target in enumerate(splits):
        train = np.where(target, np.nan, x)
        baselines = {'trait_mean': np.broadcast_to(np.nanmean(train, axis=0), x.shape),
                     'constant_3': np.full(x.shape, 3.)}
        for method, prediction in baselines.items():
            result['validation'].append({'method': method, 'q': 0, 'split': k,
                                         **prediction_metrics(x[target], prediction[target])})
    for q in (1, 2, 3):
        print(f'Rank {q}: nested validation', flush=True)
        for k, target in enumerate(splits):
            train = np.where(target, np.nan, x)
            penalty, tuning = choose_penalty(train, exact & ~target, q, 9100 + k,
                                             repeats=result['configuration']['cv_inner_splits'])
            als = best_ridge(train, q, penalty, starts=6, seed=10100 + k)
            posterior = gibbs_factor_analysis(train, q, seed=11100 + k,
                warmup=200 if quick else 600, draws=300 if quick else 1000, chains=2)
            def diagnostic(posterior):
                predictions = np.einsum('ctik,ctjk->ctij', posterior['scores'], posterior['loadings'])
                predictions += posterior['mean'][:, :, None, :]
                return chain_diagnostics(np.concatenate([posterior['mean'], posterior['noise'],
                                                          predictions[:, :, target]], axis=-1))
            mcmc_diagnostics = diagnostic(posterior)
            if mcmc_diagnostics['max_rank_split_rhat'] > 1.01:
                first = mcmc_diagnostics
                posterior = gibbs_factor_analysis(train, q, seed=11100 + k,
                                                  warmup=1500, draws=3000, chains=4)
                mcmc_diagnostics = {**diagnostic(posterior), 'extended_from': first}
            predictions = prediction_mean(posterior)
            for method, prediction in [('ridge_factorization', als['prediction']), ('bayesian_fa', predictions)]:
                row = {'method': method, 'q': q, 'split': k,
                       **prediction_metrics(x[target], prediction[target])}
                if method == 'ridge_factorization':
                    row.update({'selected_penalty': penalty, 'inner_tuning': tuning})
                else:
                    row['predictive_mcmc_diagnostics'] = mcmc_diagnostics
                result['validation'].append(row)
        penalty, tuning = choose_penalty(x, exact, q, 12100, repeats=5 if not quick else 2)
        print(f'Rank {q}: full posterior and ridge (lambda={penalty})', flush=True)
        for method in ('bayesian_fa', 'ridge_factorization'):
            fitted = fit(x, q, method, seed=13100, penalty=penalty,
                warmup=result['configuration']['main_warmup'], draws=result['configuration']['main_draws_per_chain'],
                chains=result['configuration']['main_chains'])
            fitted.update({'q': q, 'method': method,
                           'training_exact_error': prediction_metrics(x[exact], fitted['prediction'][exact])})
            if method == 'ridge_factorization':
                fitted['tuning'] = tuning
            scores = fitted['scores']
            counts = np.isfinite(x).sum(axis=1)
            usable = fitted.get('effective_rank_threshold_0_01') != 0
            fitted['coverage_axis_spearman'] = [float(spearmanr(scores[:, f], counts).statistic)
                                               if usable and np.std(scores[:, f]) > 1e-9 else None for f in range(q)]
            fitted['distance_vs_missingness_jaccard_spearman'] = float(spearmanr(
                pdist(scores), pdist(np.isfinite(x), metric='jaccard')).statistic) if usable and np.std(pdist(scores)) > 1e-9 else None
            result['models'][f'{method}_q{q}'] = fitted
        print(f'Rank {q}: range, date, prior and influential-actor sensitivity', flush=True)
        # Fit availability >=3 features in each variant. Embed in fixed universe
        # with NaNs, so an absent actor/trait is never a new neutral coordinate.
        scenarios = {'lower_bounds': data['lower'], 'upper_bounds': data['upper'],
                     'exact_only': np.where(exact, x, np.nan), 'strict_dated': data['strict']}
        for scenario, raw in scenarios.items():
            columns = np.isfinite(raw).sum(axis=0) >= 3
            rows = np.isfinite(raw[:, columns]).any(axis=1)
            local = raw[np.ix_(rows, columns)]
            if min(local.shape) <= q:
                continue
            for method in ('bayesian_fa', 'ridge_factorization'):
                fitted = fit(local, q, method, seed=14100, penalty=penalty,
                             warmup=250 if quick else 700, draws=400 if quick else 1200)
                reference = result['models'][f'{method}_q{q}']
                embedded = {'scores': np.full((len(x), q), np.nan), 'loadings': np.full((x.shape[1], q), np.nan),
                            'effective_rank_threshold_0_01': fitted.get('effective_rank_threshold_0_01')}
                embedded['scores'][rows] = fitted['scores']
                embedded['loadings'][columns] = fitted['loadings']
                result['scenario_fits'][f'{scenario}_{method}_q{q}'] = {
                    'method': method, 'q': q, 'scenario': scenario, 'actors': int(rows.sum()),
                    'scales': int(columns.sum()), 'observed_cells': int(np.isfinite(local).sum()),
                    'actor_ids': np.array(data['actor_ids'])[rows], 'scale_ids': np.array(data['scale_ids'])[columns],
                    'loadings': fitted['loadings'], 'scores': fitted['scores'],
                    'diagnostics': fitted.get('diagnostics'),
                    'comparison': geometry(reference, embedded, x, common_rows=rows, common_columns=columns)}
        # Prior choices are not tuned on validation scores; expose dependence.
        for sd in (.4, 1.2):
            candidate = fit(x, q, 'bayesian_fa', seed=15100, priors=replace(Priors(), loading_sd=sd),
                            warmup=250 if quick else 700, draws=400 if quick else 1200)
            result['stress'].append({'method': 'bayesian_fa', 'q': q, 'kind': 'loading_prior_sd', 'value': sd,
                **geometry(result['models'][f'bayesian_fa_q{q}'], candidate, x),
                'diagnostics': candidate['diagnostics']})
        # Refit unchanged midpoints to separate Monte Carlo/orientation variation
        # from changes caused by the rating bounds themselves.
        for repeat in range(4 if not quick else 1):
            candidate = fit(x, q, 'bayesian_fa', seed=20100 + repeat,
                warmup=result['configuration']['stress_warmup'], draws=result['configuration']['stress_draws_per_chain'])
            result['stress'].append({'method': 'bayesian_fa', 'q': q, 'kind': 'unchanged_data_refit', 'repeat': repeat,
                **geometry(result['models'][f'bayesian_fa_q{q}'], candidate, x), 'diagnostics': candidate['diagnostics']})
        # These are bounded perturbations, NOT posterior or credible draws.
        for repeat in range(stress_repeats):
            rng = np.random.default_rng(16100 + repeat)
            raw = data['lower'] + rng.random(x.shape) * (data['upper'] - data['lower'])
            for method in ('bayesian_fa', 'ridge_factorization'):
                candidate = fit(raw, q, method, seed=17100 + repeat, penalty=penalty,
                    warmup=result['configuration']['stress_warmup'], draws=result['configuration']['stress_draws_per_chain'])
                result['stress'].append({'method': method, 'q': q, 'kind': 'interval_perturbation', 'repeat': repeat,
                    **geometry(result['models'][f'{method}_q{q}'], candidate, x),
                    'diagnostics': candidate.get('diagnostics')})
        for removed in (['paul-kagame'], ['olivier-nduhungirehe'], ['paul-kagame', 'olivier-nduhungirehe']):
            rows = np.array([a not in removed for a in data['actor_ids']])
            for method in ('bayesian_fa', 'ridge_factorization'):
                candidate = fit(x[rows], q, method, seed=18100, penalty=penalty,
                                warmup=250 if quick else 700, draws=400 if quick else 1200)
                embedded = {**candidate, 'scores': np.full((len(x), q), np.nan)}
                embedded['scores'][rows] = candidate['scores']
                result['influence'].append({'method': method, 'q': q, 'removed': removed,
                    **geometry(result['models'][f'{method}_q{q}'], embedded, x, common_rows=rows),
                    'diagnostics': candidate.get('diagnostics')})
        # A deliberately weak ridge is saved as a diagnostic if CV rejects all
        # shared dimensions. It is never substituted for the selected model.
        result['models'][f'weak_ridge_q{q}'] = fit(x, q, 'ridge_factorization', seed=19100, penalty=.1)
    result['cross_method'] = {f'q{q}': geometry(result['models'][f'bayesian_fa_q{q}'],
        result['models'][f'ridge_factorization_q{q}'], x) for q in (1, 2, 3)}
    result['cross_dimension'] = {f'{method}_{a}_vs_{b}': geometry(
        result['models'][f'{method}_q{a}'], result['models'][f'{method}_q{b}'], x)
        for method in ('bayesian_fa', 'ridge_factorization') for a, b in ((1, 2), (2, 3))}
    result['elapsed_seconds'] = time.monotonic() - start
    result['runtime'] = {name: importlib.metadata.version(name)
                         for name in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / ('quick_check.json' if quick else 'benchmark.json')
    path.write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Saved {path}; {time.monotonic()-start:.1f}s; new API cost $0', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='Smoke check only; not the published analysis')
    parser.add_argument('--stress-repeats', type=int, default=20)
    args = parser.parse_args()
    if args.stress_repeats < 1:
        parser.error('--stress-repeats must be positive')
    with threadpool_limits(limits=1):
        run(quick=args.quick, stress_repeats=args.stress_repeats)
