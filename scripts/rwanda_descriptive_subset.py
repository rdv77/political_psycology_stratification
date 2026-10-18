"""A descriptive, availability-selected Rwanda snapshot; no paid APIs.

Changed research aim: describe this fixed corpus, without requiring predictive
superiority or stability across alternative admissible ratings. Uncertainty is
reported, never redefined as temporal political change.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.compare_rwanda_incomplete_latent import clean, fit, input_data
from src.corpus_pipeline.incomplete_latent import Priors, best_ridge, canonical_axes, compare_geometry

OUT = ROOT / 'research/rwanda_politicians_2021_2026/source_scales_v2/descriptive_subset_2026_10_18'


def select(data, threshold):
    x = data['midpoint']
    rows = np.isfinite(x).sum(axis=1) >= threshold
    columns = np.isfinite(x[rows]).sum(axis=0) >= max(3, int(np.ceil(.6 * rows.sum())))
    local = x[np.ix_(rows, columns)]
    return rows, columns, {
        'initial_minimum_informative_traits': threshold, 'actors': int(rows.sum()),
        'traits': int(columns.sum()), 'informative_cells': int(np.isfinite(local).sum()),
        'possible_cells': local.size, 'coverage': float(np.isfinite(local).mean()),
        'minimum_shared_traits_per_actor': int(np.isfinite(local).sum(axis=1).min()),
    }


def csv_file(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(result):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    for q, projections in ((2, (1,)), (3, (1, 2))):
        scores = np.asarray(result['models'][f'ridge_q{q}']['scores'])
        fig, axes = plt.subplots(1, len(projections), figsize=(10 if q == 2 else 17, 7))
        fig.subplots_adjust(left=.28 if q == 2 else .15, right=.72 if q == 2 else .85,
                            top=.83, bottom=.14, wspace=.95)
        for ax, k in zip(np.atleast_1d(axes), projections):
            ax.scatter(scores[:, 0], scores[:, k], color='#167193', s=33, zorder=3)
            order = np.argsort(scores[:, 0])
            for side, indices in [('left', order[:5]), ('right', order[5:])]:
                indices = sorted(indices, key=lambda i: -scores[i, k])
                for i, y in zip(indices, np.linspace(.92, .08, len(indices))):
                    ax.annotate(result['actor_names'][i], (scores[i, 0], scores[i, k]),
                        xytext=(-.12 if side == 'left' else 1.05, y), textcoords='axes fraction',
                        ha='right' if side == 'left' else 'left', va='center', fontsize=8,
                        arrowprops={'arrowstyle': '-', 'color': '#999999', 'lw': .65})
            ax.axvline(0, color='#dddddd', lw=.8)
            ax.axhline(0, color='#dddddd', lw=.8)
            ax.set(xlim=(-1.05, 1.35), ylim=(-1.4, 1.25), xlabel='F1', ylabel=f'F{k+1}',
                   title=f'F1 × F{k+1}')
            ax.yaxis.set_label_coords(.02, 1.02)
        fig.suptitle(f'Описательная карта 10 участников по 14 признакам\n{q} измерения; середины диапазонов; фиксированное сглаживание λ=0.3', fontsize=14)
        fig.savefig(OUT / ('snapshot_2d.png' if q == 2 else 'snapshot_objects.png'), dpi=170)
        plt.close(fig)

    matrix = np.concatenate([np.asarray(result['models'][f'ridge_q{q}']['loadings']) for q in (1, 2, 3)], axis=1)
    fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')
    limit = float(abs(matrix).max())
    image = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap='RdBu_r', aspect='auto')
    import textwrap
    labels = [f"{sid}  " + '\n'.join(textwrap.wrap(name, 45)) for sid, name in zip(result['scale_ids'], result['scale_names'])]
    ax.set(yticks=range(len(labels)), yticklabels=labels,
           xticks=range(6), xticklabels=['1D:F1', '2D:F1', '2D:F2', '3D:F1', '3D:F2', '3D:F3'],
           title='Из чего складываются оси описательной карты\nНагрузки признаков; знак не означает одобрения или отрицания свойства')
    ax.tick_params(axis='y', labelsize=8)
    for i in range(len(labels)):
        for j in range(6):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(matrix[i,j]) > .75 else '#222222')
    ax.axvline(.5, color='white', lw=3)
    ax.axvline(2.5, color='white', lw=3)
    fig.colorbar(image, ax=ax, shrink=.65, label='Нагрузка, не коэффициент корреляции')
    fig.savefig(OUT / 'snapshot_loadings.png', dpi=170)
    plt.close(fig)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    data = input_data()
    rows, cols, selection = select(data, 10)
    x = data['midpoint'][np.ix_(rows, cols)]
    exact = data['exact_mask'][np.ix_(rows, cols)]
    assert x.shape == (10, 14) and np.isfinite(x).sum() == 111 and exact.sum() == 49
    assert (np.isfinite(x).sum(axis=1) >= x.shape[1] / 2).all()
    actor_ids = np.asarray(data['actor_ids'])[rows].tolist()
    scale_ids = np.asarray(data['scale_ids'])[cols].tolist()
    result = {
        'version': 'rwanda-descriptive-subset-v1', 'analysis_completed_on': '2026-10-18',
        'aim': 'descriptive_snapshot_of_fixed_source_scenarios_not_population_or_prediction',
        'selection': selection, 'selection_comparison': [select(data, t)[2] for t in (4, 6, 8, 10, 12)],
        'selection_rule': '>=10 informative traits in the previous 24-trait universe; retain traits available in >=60% of those actors (at least 3); verify every actor covers at least half of retained traits',
        'actor_ids': actor_ids, 'actor_names': [data['actor_names'][a] for a in actor_ids],
        'scale_ids': scale_ids, 'scale_names': [data['scale_names'][s] for s in scale_ids],
        'exact_cells': int(exact.sum()), 'range_midpoint_assumptions': int((np.isfinite(x) & ~exact).sum()),
        'missing_cells': int(np.isnan(x).sum()), 'minimum_cell_count_is_not_a_universal_statistical_rule': True,
        'input_content_hashes': data['input_content_hashes'],
        'ratings': x, 'lower': data['lower'][np.ix_(rows, cols)], 'upper': data['upper'][np.ix_(rows, cols)],
        'exact_mask': exact, 'ridge_penalty': .3, 'ridge_penalty_chosen_for': 'fixed exploratory description, not CV or a claim of optimality',
        'bayesian_priors': asdict(Priors()), 'stability_is_admission_gate': False,
        'predictive_superiority_is_admission_gate': False, 'source_ratings_modified': False,
        'new_api_calls': 0, 'new_api_cost_usd': 0, 'models': {}, 'penalty_comparison': [],
        'russia_factor_status': 'not_testable_on_this_general_trait_subset',
    }
    observed = np.isfinite(x)
    total = float(np.nansum((x - np.nanmean(x, axis=0)) ** 2))
    for q in (1, 2, 3):
        print(f'Fitting descriptive subset, q={q}', flush=True)
        ridge = best_ridge(x, q, .3, seed=20261018, starts=8)
        ridge['scores'], ridge['loadings'], _ = canonical_axes(ridge['scores'], ridge['loadings'])
        ridge.pop('history')
        bayesian = fit(x, q, 'bayesian_fa', seed=20261018, warmup=1500, draws=3000, chains=4)
        for method, fitted in [('ridge', ridge), ('bayesian', bayesian)]:
            sse = float(np.sum((fitted['prediction'][observed] - x[observed]) ** 2))
            fitted['observed_cell_rmse'] = float(np.sqrt(sse / observed.sum()))
            fitted['observed_centered_reconstruction_r2'] = 1 - sse / total
            result['models'][f'{method}_q{q}'] = fitted
        for penalty in (.1, .3, 1.):
            f = best_ridge(x, q, penalty, seed=20261018, starts=8)
            result['penalty_comparison'].append({'q': q, 'penalty': penalty,
                'observed_centered_reconstruction_r2': 1 - float(np.sum((f['prediction'][observed]-x[observed])**2))/total,
                'signal_singular_values': np.linalg.svd(f['scores'] @ f['loadings'].T, compute_uv=False)[:q]})
        result.setdefault('between_method_geometry', {})[str(q)] = compare_geometry(ridge, bayesian)
    registry = json.loads((ROOT / 'reports/rwanda_native_mass_profiles_2026_10_18.json').read_text(encoding='utf-8'))
    descriptions = {a['id']: a['role'] for a in registry['actors']}
    result['roles_from_approved_list_not_new_current_role_verification'] = [descriptions[a] for a in actor_ids]
    (OUT / 'snapshot.json').write_text(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    loadings, coordinates = [], []
    for method in ('ridge', 'bayesian'):
        for q in (1, 2, 3):
            f = result['models'][f'{method}_q{q}']
            for i, actor in enumerate(actor_ids):
                for k in range(q):
                    coordinates.append({'method': method, 'dimensions': q, 'actor': actor,
                        'name': result['actor_names'][i], 'axis': k+1, 'coordinate': f['scores'][i,k],
                        'conditional_95_lower': f['score_lower'][i,k] if method=='bayesian' else '',
                        'conditional_95_upper': f['score_upper'][i,k] if method=='bayesian' else ''})
            for j, sid in enumerate(scale_ids):
                for k in range(q):
                    loadings.append({'method': method, 'dimensions': q, 'scale': sid,
                        'name': result['scale_names'][j], 'axis': k+1, 'loading': f['loadings'][j,k]})
    csv_file('coordinates.csv', coordinates)
    csv_file('loadings.csv', loadings)
    render(result)
    print('Saved descriptive snapshot, figures and tables; new API cost $0.', flush=True)
    for key, f in result['models'].items():
        print(key, f['observed_centered_reconstruction_r2'], f.get('diagnostics'))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
