"""Offline, versioned combination; prepare from local evidence, fit from public ledger.

No model client is imported and no HTTP requests are made. Historical numeric
scores are NOT transferred. Automatic source-summary coding remains provisional.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from src.corpus_pipeline.harmonize_rwanda import (
    aggregate_sources, build_dictionary, canonical_url, propose_intensity, target_id, text_hash,
)
from src.corpus_pipeline.incomplete_latent import best_ridge, canonical_axes, compare_geometry
from scripts.compare_rwanda_incomplete_latent import clean, fit as fit_latent

BASE = ROOT / 'studies/rwanda-politicians-2021-2026'
OUT = ROOT / 'research/rwanda_politicians_2021_2026/combined_sources_2026_10_18'
PRIVATE = ROOT / '.corpus_runtime/combined_sources_2026_10_18'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def export_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def admissible_fact(f, actor, study, excluded):
    period = str(f.get('period') or '')
    precise = re.findall(r'\b\d{4}-\d{2}-\d{2}\b',period)
    years = re.findall(r'\b(?:19|20)\d{2}\b',period)
    try:
        start, end = date.fromisoformat(study['date_from']), date.fromisoformat(study['date_to'])
        # Year/month/ranged textual dates are not invented exact days. Admit
        # conservatively only if their ENTIRE year envelope is inside the window.
        time_ok = all(start <= date.fromisoformat(t) <= end for t in precise) if precise else bool(years) and all(
            start <= date(int(y),1,1) and date(int(y),12,31) <= end for y in years)
    except ValueError:
        time_ok = False
    if not time_ok:
        return False, 'unverified_or_out_of_window_event_date'
    if f.get('actor_id', f.get('object_id', actor)) != actor:
        return False, 'wrong_actor'
    if f.get('eligibility') == 'context_only' or f.get('measurement_status') in {
            'context_only', 'excluded_known_source_issue'}:
        return False, 'context_or_excluded'
    if (actor, f.get('source_url'), hashlib.sha256(str(f.get('text', '')).encode('utf-8')).hexdigest()) in excluded:
        return False, 'registered_source_exclusion'
    if not f.get('text') or not f.get('source_url') or not f.get('capacity'):
        return False, 'missing_source_or_capacity'
    return True, None


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    study = read(BASE / 'study.json')
    ratings = read(BASE / 'ratings.json')
    new = read(ROOT / 'research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json')
    profiles = read(ROOT / 'reports/rwanda_native_mass_profiles_2026_10_18.json')
    bounds = read(ROOT / 'reports/rwanda_native_mass_bounds_2026_10_18.json')
    scales, crosswalk = build_dictionary(study['scales'], new['scales'])
    write(OUT / 'dictionary.json', {'version': 'unified-rwanda-v1', 'scales': scales,
          'crosswalk': crosswalk, 'new_api_cost_usd': 0,
          'frequency_is_intensity': False, 'opposite_is_computed': False,
          'note': 'Merged by authorized current instruction; historical data are not overwritten.'})
    roster_fields={'id','name','name_ru','name_original','category','role','role_years','inclusion_reason'}
    write(OUT / 'roster.json', [{k:v for k,v in a.items()if k in roster_fields}for a in profiles['actors']])
    exclusions = {(e['object_id'], e['source_url'], e['text_sha256'])
                  for e in study['independent_evidence'].get('additional_excluded_facts', [])}
    ledger, private, rejected = [], [], []
    dossier_cache = {}
    for r in ratings:
        for pole, old in r['pole_scores'].items():
            if old['status'] != 'observed':
                continue
            actor, sid = r['object_id'], target_id(r['scale_id'], pole)
            path = BASE / old['evidence_file']
            if path not in dossier_cache:
                dossier_cache[path] = read(path)
            facts = dossier_cache[path]['facts']
            candidates, seen = [], set()
            for index in old['evidence_indices']:
                f = facts[index]
                ok, reason = admissible_fact(f, actor, study, exclusions)
                if not ok:
                    rejected.append({'actor_id': actor, 'scale_id': sid, 'file': old['evidence_file'],
                                     'index': index, 'reason': reason})
                    continue
                if r['scale_id'].startswith('russia-'):
                    topics = set(f.get('topic_ids', []))
                    if f.get('evidence_origin') != 'assistant_verified_source' or not topics.intersection({r['scale_id'], r['scale_id']+'__'+pole}):
                        rejected.append({'actor_id': actor, 'scale_id': sid, 'index': index,
                                         'reason': 'russia_source_not_individually_verified'})
                        continue
                key = (f.get('event_id') or canonical_url(f['source_url']), text_hash(f['text']))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((index, f))
            if not candidates:
                continue
            identifier = actor + '::' + r['scale_id'] + '__' + pole
            # A bundle is NOT declared several independent episodes. Its count
            # cannot strengthen intensity. Each supporting fact remains linked.
            text = '\n'.join(f['text'] for _, f in candidates)
            proposed = propose_intensity(text, old['rationale'])
            row = {'id': identifier, 'origin': 'historical_dossier', 'actor_id': actor,
                   'scale_id': sid, **proposed, 'strict_dated': True,
                   'input_claim_sha256': text_hash(old['rationale']),
                   'support': [{'file': old['evidence_file'], 'index': i,
                                'source_url': f['source_url'], 'event_id': f.get('event_id'),
                                'event_date': f['period'], 'capacity': f['capacity'],
                                'text_sha256': text_hash(f['text']),
                                'verification': f.get('evidence_origin', 'historical_source_summary')}
                               for i, f in candidates]}
            ledger.append(row)
            private.append({**row, 'text': text, 'targeted_rationale': old['rationale']})
    # All corpus cells and their reviewed bounds, not just the old 10-person map.
    strict = {(c['actor_id'], c['scale_id']): c for c in bounds['strict_dated']['cells']}
    for c in bounds['content_supported_unknown_time_sensitivity']['cells']:
        if c['lower'] is None:
            continue
        sc = strict[c['actor_id'], c['scale_id']]
        ledger.append({'id': 'corpus::'+c['actor_id']+'::'+c['scale_id'],
                       'origin': 'reviewed_corpus', 'actor_id': c['actor_id'],
                       'scale_id': c['scale_id'], 'lower': c['lower'], 'upper': c['upper'],
                       'strict_dated': sc['lower'] is not None,
                       'strict_lower': sc['lower'], 'strict_upper': sc['upper'],
                       'coding_status': c['measurement_status'], 'case_ids': c['case_ids'],
                       'input_hash': content_hash(c)})
    write(PRIVATE / 'historical_review_private.json', private)
    write(OUT / 'source_ledger.json', {'version': 1, 'records': ledger, 'rejected': rejected,
          'input_content_hashes': {'study': content_hash(study), 'historical_ratings': content_hash(ratings),
                                  'profiles': content_hash(profiles), 'bounds': content_hash(bounds),
                                  'dictionary': content_hash(new)},
          'historical_numeric_scores_used': False, 'new_api_calls': 0, 'new_api_cost_usd': 0})
    # A deterministic stratified audit packet: every actor once plus Russia and
    # remaining rule/scale diversity. The file is private; no social texts uploaded.
    selected = []
    for actor in [a['id'] for a in profiles['actors']]:
        pool = [r for r in private if r['actor_id'] == actor]
        if pool:
            selected.append(min(pool, key=lambda r: text_hash(r['id'])))
    for r in sorted(private, key=lambda r: (not r['scale_id'].startswith('russia-'), text_hash(r['id']))):
        if r['id'] not in {a['id'] for a in selected}:
            selected.append(r)
        if len(selected) >= 66:
            break
    if not (PRIVATE / 'audit_packet_private.json').exists():
        write(PRIVATE / 'audit_packet_private.json', selected)
    print('Prepared:', len(scales), 'traits;', len(private), 'dossier proposals;',
          len(ledger)-len(private), 'corpus cells;', len(rejected), 'rejected links')


def matrices(ledger, actors, scales, scenario='main', adjudications=None):
    adjudications = adjudications or {}
    grouped = defaultdict(list)
    for raw in ledger:
        row = dict(raw)
        correction = adjudications.get(row['id'])
        if correction:
            if correction.get('exclude'):
                continue
            row.update(lower=correction['value'], upper=correction['value'])
        elif row['origin'] == 'historical_dossier' and scenario != 'automatic_baseline':
            raise ValueError('Historical claim has not received semantic adjudication: '+row['id'])
        if scenario == 'source_audited_only' and row['origin'] == 'historical_dossier' and correction.get('review_scope','').startswith('targeted_previously'):
            continue
        if scenario == 'flat_historical' and row['origin'] == 'historical_dossier':
            row.update(lower=2.5, upper=2.5)
        if scenario == 'automatic_baseline' and row['origin'] == 'historical_dossier':
            row.update(lower=raw['lower'], upper=raw['upper'])
        if scenario == 'strict' and row['origin'] == 'reviewed_corpus':
            if row['strict_lower'] is None:
                continue
            row.update(lower=row['strict_lower'], upper=row['strict_upper'])
        if scenario == 'corpus_only' and row['origin'] != 'reviewed_corpus':
            continue
        if scenario == 'dossier_only' and row['origin'] != 'historical_dossier':
            continue
        grouped[row['actor_id'], row['scale_id']].append(row)
    cells = []
    for actor in actors:
        for sid in scales:
            c = aggregate_sources(grouped[actor, sid])
            cells.append({'actor_id': actor, 'scale_id': sid, **c,
                          'record_ids': sorted(r['id'] for r in grouped[actor, sid]),
                          'midpoint_is_measurement': c['lower'] is not None and c['lower'] == c['upper']})
    array = lambda field: np.array([c[field] for c in cells], float).reshape(len(actors), len(scales))
    return cells, array('midpoint'), array('lower'), array('upper')


def select_matrix(x, minimum_trait_fraction=.25):
    # A permissive descriptive rule, fixed before examining loadings. At least
    # four observed-varying indicators to attempt a four-axis position. This is
    # a descriptive admission rule, not a theorem about adequate sample size.
    rows = np.isfinite(x).sum(1) >= 4
    for _ in range(10):
        minimum = max(5, int(np.ceil(minimum_trait_fraction * len(x))))
        cols = np.array([np.isfinite(x[rows, j]).sum() >= minimum and
                         np.nanstd(x[rows, j]) >= .10 for j in range(x.shape[1])])
        updated = np.isfinite(x[:, cols]).sum(1) >= 4
        if np.array_equal(rows, updated):
            return rows, cols
        rows = updated
    raise ValueError('Availability selection failed to reach fixed point')


def fit_model(x, q, penalty=.3):
    fit = best_ridge(x, q, penalty, starts=8, seed=20261018)
    fit['scores'], fit['loadings'], _ = canonical_axes(fit['scores'], fit['loadings'])
    fit.pop('history')
    residual = np.where(np.isfinite(x), fit['prediction']-x, 0.)
    count = np.isfinite(x).sum(0)
    means = np.divide(np.nansum(x,axis=0),count,out=np.zeros(x.shape[1]),where=count>0)
    total = np.nansum((x-means)**2)
    fit['observed_cell_rmse'] = np.sqrt((residual**2).sum()/np.isfinite(x).sum())
    fit['observed_centered_reconstruction_r2'] = 1-(residual**2).sum()/total
    fit['signal_singular_values'] = np.linalg.svd(fit['scores']@fit['loadings'].T, compute_uv=False)[:q]
    return fit


def fit_all():
    public = read(OUT / 'source_ledger.json')
    actors = [a['id'] for a in read(OUT / 'roster.json')]
    scales = [s['id'] for s in read(OUT / 'dictionary.json')['scales']]
    auditpath = OUT / 'semantic_adjudications.json'
    adjudications = read(auditpath)['decisions'] if auditpath.exists() else {}
    base_index = {r['id']:r for r in public['records']}
    for extension in read(OUT / 'crosscoded_claims.json')['records']:
        parent = base_index[extension['parent_id']]
        if adjudications[parent['id']].get('exclude'):
            raise ValueError('A cross-coding cannot inherit an excluded parent claim')
        sid = extension['scale_id']
        if sid not in scales or not 0 <= extension['value'] <= 5:
            raise ValueError('Invalid cross-coded measurement')
        identifier = 'crosscode::'+parent['actor_id']+'::'+sid+'::'+text_hash(parent['id'])[:12]
        row = dict(parent, id=identifier, scale_id=sid, parent_id=parent['id'],
                   lower=extension['value'], upper=extension['value'],
                   rule='individual_semantic_cross_dictionary_coding',
                   coding_status='manual_source_linked_summary_crosscoding')
        public['records'].append(row)
        adjudications[identifier] = dict(extension, exclude=False,
            input_claim_sha256=parent['input_claim_sha256'],
            support_text_sha256=[s['text_sha256'] for s in parent['support']])
    for r in public['records']:
        if r['origin'] == 'historical_dossier':
            a = adjudications[r['id']]
            if a['input_claim_sha256'] != r['input_claim_sha256'] or a['support_text_sha256'] != [s['text_sha256'] for s in r['support']]:
                raise ValueError('Stale semantic adjudication: '+r['id'])
    write(OUT / 'integrated_evidence.json', public)
    cells, full, low, high = matrices(public['records'], actors, scales, adjudications=adjudications)
    rows, cols = select_matrix(full)
    x = full[np.ix_(rows, cols)]
    selected_a, selected_s = np.asarray(actors)[rows].tolist(), np.asarray(scales)[cols].tolist()
    if min(x.shape) <= 4:
        raise ValueError('Not enough informative observations to compare 3/4 dimensions')
    write(OUT / 'ratings.json', {'cells': cells, 'scenario': 'semantic_claim_recoding_and_reviewed_corpus_bounds',
          'historical_claims_semantically_recoded': read(auditpath)['reviewed_count'],
          'additional_crosscoded_claims': len(adjudications)-read(auditpath)['reviewed_count'],
          'historical_numeric_scores_used': False,
          'new_api_calls': 0, 'new_api_cost_usd': 0})
    result = {'version': 'unified-exploratory-v1', 'roster_size': len(actors),
              'dictionary_size': len(scales), 'actor_ids': selected_a, 'scale_ids': selected_s,
              'coverage': {'actors': len(selected_a), 'traits': len(selected_s),
                           'informative': int(np.isfinite(x).sum()), 'possible': x.size,
                           'point_coded': int(np.sum(np.isfinite(x)&(low[np.ix_(rows,cols)]==high[np.ix_(rows,cols)])))},
              'excluded_actors': [a for a in actors if a not in selected_a],
              'selection_rule': 'Fixed-point: >=4 informative varying retained traits per actor; coverage >=25% of the full 42 roster (11) per trait and observed scenario SD >=0.10. Four values do not establish population validity or full precision.',
              'threshold_revision': 'Initial permissive 5-actor indicators dominated some axes. Reporting core now requires 25% trait coverage without reducing the roster. The initial wider result is retained in wide_40_models.json; this change was made after preliminary diagnostic inspection, not preregistered.',
              'scores_are_conditional_on_source_summary_recoding': True,
              'penalty': .3, 'penalty_is_predictively_optimized': False,
              'new_api_calls': 0, 'new_api_cost_usd': 0,
              'models': {}, 'sensitivity': [], 'source_ledger_content_hash': content_hash(public),
              'adjudications_content_hash': content_hash(adjudications),
              'ratings': x, 'lower': low[np.ix_(rows,cols)], 'upper': high[np.ix_(rows,cols)]}
    for q in (3, 4):
        print('Fitting', len(selected_a), 'actors;', len(selected_s), 'traits; q=',q, flush=True)
        main = fit_model(x, q)
        result['models'][str(q)] = main
        core = np.isfinite(x).sum(1) >= 5
        core_fit = fit_model(x[core], q)
        result.setdefault('richer_core_comparison', {})[str(q)] = {
            'actors': int(core.sum()), 'actor_ids': np.asarray(selected_a)[core].tolist(),
            **compare_geometry({'scores':main['scores'][core], 'loadings':main['loadings']}, core_fit)}
        # Supplementary coordinates are conditional ridge projections, never
        # pseudo-observations or participants used to estimate the axes.
        supplemental = []
        for i, actor in enumerate(actors):
            if actor in selected_a:
                continue
            values = full[i, cols]
            observed = np.isfinite(values)
            l = main['loadings'][observed]
            score = np.linalg.solve(l.T@l+.3*np.eye(q), l.T@(values[observed]-main['mean'][observed]))
            supplemental.append({'actor_id': actor, 'scores': score,
                 'informative_traits': int(observed.sum()), 'observed_loading_rank': int(np.linalg.matrix_rank(l)),
                 'identified_in_all_dimensions': bool(np.linalg.matrix_rank(l)==q),
                 'status':'supplementary_conditional_projection_not_used_to_fit_axes'})
        result.setdefault('supplementary', {})[str(q)] = supplemental
        result.setdefault('actor_information', {})[str(q)] = [
            {'actor_id': a, 'informative_traits': int(np.isfinite(x[i]).sum()),
             'observed_loading_rank': int(np.linalg.matrix_rank(main['loadings'][np.isfinite(x[i])])),
             'condition_number': float(np.linalg.cond(main['loadings'][np.isfinite(x[i])])),
             'weak_coverage': bool(np.isfinite(x[i]).sum() < 8)}
            for i,a in enumerate(selected_a)]
        for scenario in ('strict', 'flat_historical', 'source_audited_only', 'corpus_only', 'dossier_only', 'automatic_baseline'):
            _, other, _, _ = matrices(public['records'], actors, scales, scenario, adjudications)
            other = other[np.ix_(rows, cols)]
            local = fit_model(other, q)
            comparable_rows = (np.isfinite(x).sum(1)>=5)&(np.isfinite(other).sum(1)>=5)
            counts = np.isfinite(other).sum(0)
            means = np.divide(np.nansum(other,0),counts,out=np.zeros(other.shape[1]),where=counts>0)
            sd = np.sqrt(np.divide(np.nansum((other-means)**2,0),counts,
                                  out=np.zeros(other.shape[1]),where=counts>0))
            comparable_cols = (counts>=3)&(sd>.1)
            geometry = compare_geometry(main, local, row_mask=comparable_rows, feature_mask=comparable_cols)
            result['sensitivity'].append({'q': q, 'scenario': scenario,
                'informative_cells': int(np.isfinite(other).sum()),
                'actors_with_five_measurements': int(comparable_rows.sum()),
                'compared_varying_traits': int(comparable_cols.sum()), **geometry})
        for penalty in (.1, 1.):
            f = fit_model(x, q, penalty)
            result['sensitivity'].append({'q':q, 'scenario': f'lambda_{penalty}',
                                          **compare_geometry(main,f)})
        # Min/max are admissible coding scenarios, not confidence limits.
        for scenario, values in [('lower', low), ('upper', high)]:
            f = fit_model(values[np.ix_(rows,cols)], q)
            result['sensitivity'].append({'q':q, 'scenario':scenario,
                                          **compare_geometry(main,f)})
    write(OUT / 'models.json', result)
    export_csv('ratings.csv', [{'actor_id': c['actor_id'], 'scale_id': c['scale_id'],
                'lower': c['lower'], 'upper': c['upper'], 'scenario_value': c['midpoint'],
                'sources': ';'.join(c['sources'])} for c in cells])
    export_csv('coordinates.csv', [{'dimensions': q, 'actor_id': actor, 'axis': k+1,
                'coordinate': result['models'][str(q)]['scores'][i,k],
                'informative_traits': int(np.isfinite(x[i]).sum())}
                for q in (3,4) for i,actor in enumerate(selected_a) for k in range(q)])
    export_csv('loadings.csv', [{'dimensions': q, 'scale_id': sid, 'axis': k+1,
                'loading': result['models'][str(q)]['loadings'][j,k]}
                for q in (3,4) for j,sid in enumerate(selected_s) for k in range(q)])
    print('Saved models, all 42 rating cards and comparison tables; paid calls: 0')


def bayesian_check():
    result = read(OUT / 'models.json')
    x = np.asarray(result['ratings'], float)
    for q in (3, 4):
        print('Bayesian observed-entry FA, q=', q, flush=True)
        bayes = fit_latent(x, q, 'bayesian_fa', seed=20261018,
                           warmup=1500, draws=3000, chains=4)
        ridge = result['models'][str(q)]
        for field in ('scores','loadings'):
            ridge[field] = np.asarray(ridge[field])
        result.setdefault('bayesian_comparison', {})[str(q)] = {
            'between_method_geometry': compare_geometry(ridge,bayes), 'fit':bayes}
        write(OUT / 'models.json',result)
        print('R-hat',bayes['diagnostics']['max_rank_split_rhat'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--bayesian', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.prepare:
            prepare()
        if args.fit:
            fit_all()
        if args.bayesian:
            bayesian_check()


if __name__ == '__main__':
    main()
