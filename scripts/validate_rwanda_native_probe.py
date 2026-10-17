"""Validate a six-source diagnostic review, without API or canonical overwrite."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.chat_review import apply_adjudications, compile_review
from src.corpus_pipeline.scoring_preparation import digest
from src.corpus_pipeline.source_rating_update import build_profiles, attach_factor_comparison


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(root=ROOT):
    root = Path(root)
    directory = root / '.corpus_runtime/bottleneck_probe_2026_10_18'
    dictionary = read(root / 'research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json')
    manifest = read(directory / 'manifest_private.json')
    cases = [read(directory / 'cases' / (row['case_id'] + '.json')) for row in manifest['cases']]
    if any(digest(case) != row['sha256'] for case, row in zip(cases, manifest['cases'])):
        raise ValueError('Frozen source dossier changed')
    reviews = [read(directory / f'reviews_{part}.json') for part in ('a', 'b')]
    reviews, changes = apply_adjudications(reviews, read(directory / 'adjudications_private.json')['amendments'])
    result = compile_review(cases, dictionary, reviews)
    public = result['public']
    public['version'] = 'rwanda-six-source-native-diagnostic-probe-v1'
    public['calibration_changes'] = changes
    public['selection_ru'] = 'Шесть ранее неоценённых источников по Кагаме выбраны для проверки незавершённого этапа оценивания. Это не вероятностная выборка и не оценка качества всего корпуса.'
    public['source_reading_chars'] = sum(len(doc['text']) for case in cases for doc in case['sources'])
    baseline = read(root / 'reports/rwanda_chat_combined_review_2026_10_18.json')
    study = read(root / 'studies/rwanda-politicians-2021-2026/study.json')
    historical = read(root / 'studies/rwanda-politicians-2021-2026/output/factor_preview.json')
    before = build_profiles(study['objects'], dictionary, [baseline])
    after = build_profiles(study['objects'], dictionary, [baseline, public])
    attach_factor_comparison(after, study['objects'], dictionary, historical)
    comparison = {'version': 'rwanda-native-probe-comparison-v1', 'source_cases': len(cases),
                  'case_ratings_added': public['counts']['provisional_ratings'], 'scenarios': {},
                  'api_calls': 0, 'new_api_cost_usd': 0, 'canonical_study_modified': False,
                  'mass_scoring_completed': False, 'provenance': {'approved_dictionary_content_sha256': digest(dictionary),
                  'source_case_manifest_content_sha256': digest(manifest), 'native_review_content_sha256': digest(reviews)}}
    for key, scenario in after['scenarios'].items():
        prior = {(c['actor_id'], c['scale_id']): c for c in before['scenarios'][key]['cells']}
        changed = [{'actor_id': c['actor_id'], 'scale_id': c['scale_id'],
                    'before_value': prior[c['actor_id'], c['scale_id']]['value'],
                    'after_value': c['value'], 'case_ids': c['case_ids']}
                   for c in scenario['cells'] if c['value'] != prior[c['actor_id'], c['scale_id']]['value']]
        comparison['scenarios'][key] = {'before': before['scenarios'][key]['counts'], 'after': scenario['counts'],
                                       'changed_cells': changed, 'factor_admission': scenario['factor_admission']['status']}
    for name, value in [('rwanda_native_probe_2026_10_18.json', public), ('rwanda_native_probe_comparison_2026_10_18.json', comparison)]:
        (root / 'reports' / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'probe_counts': public['counts'], 'scenarios': {k: v['after'] for k, v in comparison['scenarios'].items()}, 'api_calls': 0}, ensure_ascii=False))
    return comparison


if __name__ == '__main__':
    run()
