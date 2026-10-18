"""Validate the published benchmark against its public inputs; no API access."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.compare_rwanda_incomplete_latent import DESTINATION, input_data


def run():
    data = input_data()
    result = json.loads((DESTINATION / 'benchmark.json').read_text(encoding='utf-8'))
    assert result['new_api_calls'] == 0 and result['new_api_cost_usd'] == 0
    assert result['source_ratings_modified'] is False
    assert result['input_content_hashes'] == data['input_content_hashes']
    assert result['actor_ids'] == data['actor_ids'] and result['scale_ids'] == data['scale_ids']
    assert data['midpoint'].shape == (31, 24)
    assert data['exact_mask'].sum() == 131 and np.isfinite(data['midpoint']).sum() == 225
    assert len(data['excluded_actors']) == 11 and len(data['excluded_scales']) == 14
    seen = set()
    for split, items in enumerate(result['holdouts']):
        target = np.zeros(data['midpoint'].shape, bool)
        for item in items:
            key = (item['actor_id'], item['scale_id'])
            assert key not in seen
            seen.add(key)
            i, j = data['actor_ids'].index(key[0]), data['scale_ids'].index(key[1])
            assert data['exact_mask'][i, j]
            target[i, j] = True
        training = np.isfinite(data['midpoint']) & ~target
        assert (training.sum(0) >= 2).all() and (training.sum(1) >= 1).all()
        for row in [r for r in result['validation'] if r['split'] == split]:
            assert row['n'] == len(items) and np.isfinite(row['rmse']) and np.isfinite(row['mae'])
    assert len(seen) == 128
    assert ('paul-kagame', 'g09') in seen
    diagnostics = []
    rows = []
    for q in (1, 2, 3):
        fa = result['models'][f'bayesian_fa_q{q}']
        als = result['models'][f'ridge_factorization_q{q}']
        assert fa['diagnostics']['max_rank_split_rhat'] < 1.01
        assert fa['diagnostics']['min_bulk_ess'] > 400
        assert als['effective_rank_threshold_0_01'] == 0
        assert np.max(als['signal_singular_values']) < .01
        low, high = np.array(fa['covariance_lower']), np.array(fa['covariance_upper'])
        tri = np.triu(np.ones(low.shape, bool), 1)
        count = int((tri & ((low > 0) | (high < 0))).sum())
        assert count == 0
        rows.append({'q': q, 'main_rhat_max': fa['diagnostics']['max_rank_split_rhat'],
                     'main_ess_min': fa['diagnostics']['min_bulk_ess'],
                     'off_diagonal_95_intervals_excluding_zero': count,
                     'ridge_effective_rank': als['effective_rank_threshold_0_01']})
    for section in ('stress', 'influence'):
        for i, item in enumerate(result[section]):
            if item.get('diagnostics'):
                diagnostics.append((f'{section}:{i}', item['diagnostics']))
    for key, item in result['scenario_fits'].items():
        if item.get('diagnostics'):
            diagnostics.append((key, item['diagnostics']))
    for i, item in enumerate(result['validation']):
        if item.get('predictive_mcmc_diagnostics'):
            diagnostics.append((f'validation:{i}', item['predictive_mcmc_diagnostics']))
    unresolved = [{'fit': key, **d} for key, d in diagnostics if d['max_rank_split_rhat'] > 1.01]
    with (DESTINATION / 'actor_coordinates.csv').open(encoding='utf-8-sig') as stream:
        coordinates = list(csv.DictReader(stream))
    assert all(row['actor_id'] not in data['excluded_actors'] for row in coordinates)
    assert all(row['coordinate'] == '' for row in coordinates if row['method'] == 'ridge_factorization')
    tests = json.loads((ROOT / '.service_runtime/incomplete_latent_test_receipt.json').read_text(encoding='utf-8'))
    assert tests['passed'] and tests['tests'] >= 610
    receipt = {
        'version': 'rwanda-incomplete-validation-v1', 'public_input_hashes_unchanged': True,
        'input_identity_rule': 'canonical JSON, sorted keys, UTF-8, no whitespace',
        'raw_input_bytes_match_execution': result['input_hashes'] == data['input_hashes'],
        'main_actor_count': 31, 'main_trait_count': 24, 'exact_cells': 131,
        'range_scenario_cells': 94, 'unique_validation_targets': 128,
        'extreme_zero_score_in_validation': True, 'unmeasured_actor_coordinates_exported': False,
        'collapsed_ridge_coordinates_exported_as_numbers': False,
        'main_model_checks': rows, 'additional_bayesian_fits': len(diagnostics),
        'additional_fits_rhat_over_1_01': unresolved,
        'extended_additional_fits': sum('extended_from' in d for _, d in diagnostics),
        'tests': tests, 'new_api_calls': 0, 'new_api_cost_usd': 0,
        'code_hash_rule': 'UTF-8 source with universal LF line endings',
        'code_sha256': {path: hashlib.sha256((ROOT / path).read_text(encoding='utf-8').encode('utf-8')).hexdigest() for path in (
            'src/corpus_pipeline/incomplete_latent.py', 'scripts/compare_rwanda_incomplete_latent.py',
            'scripts/build_rwanda_incomplete_report.py', 'scripts/audit_rwanda_incomplete_latent.py',
            'tests/test_incomplete_latent.py')},
        'seed_schedule': {'outer_fold_allocation': 8100, 'inner_tuning': '9100 + outer_fold',
                          'cv_bayesian': '11100 + outer_fold', 'full_tuning': 12100,
                          'main_models': 13100, 'alternative_scenarios': 14100, 'prior_sensitivity': 15100,
                          'interval_values': '16100 + repeat', 'interval_fits': '17100 + repeat',
                          'actor_influence': 18100, 'weak_ridge': 19100, 'unchanged_data': '20100 + repeat',
                          'bayesian_chain_increment': 100003},
    }
    (DESTINATION / 'validation_receipt.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if unresolved:
        raise SystemExit('Some sensitivity MCMC fits need further sampling or explicit limitation reporting')


if __name__ == '__main__':
    run()
