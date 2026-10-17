"""Rebuild corrected local evidence and validated reviews without model API.

Raw source texts, exact quotes and internal IDs remain in ignored runtime.
The corpus database, extraction receipts and original study are read only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.chat_review import apply_adjudications, compile_review
from src.corpus_pipeline.corrected_evidence import correct
from src.corpus_pipeline.scoring_preparation import digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def source_cases(directory, dictionary):
    manifest = read(directory / 'manifest_private.json')
    if manifest['dictionary_content_sha256'] != digest(dictionary):
        raise ValueError('Review dictionary changed')
    cases = []
    for entry in manifest['cases']:
        case = read(directory / 'cases' / (entry['case_id'] + '.json'))
        if digest(case) != entry['sha256']:
            raise ValueError('Frozen source dossier changed')
        cases.append(case)
    return cases


def run(root=ROOT, *, include_extension=True, publish=False):
    root = Path(root).resolve()
    dictionary_path = root / 'research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json'
    dictionary = read(dictionary_path)
    old = root / '.corpus_runtime/semantic_review_2026_10_17'
    cases = source_cases(old, dictionary)
    reviews = [read(old / name) for name in ('reviews_a.json', 'reviews_b.json', 'reviews_c.json', 'reviews_controls.json')]
    reviews, calibration = apply_adjudications(reviews, read(old / 'adjudications_private.json')['amendments'])
    if include_extension:
        extension = root / '.corpus_runtime/semantic_extension_2026_10_18'
        cases += source_cases(extension, dictionary)
        added = [read(extension / name) for name in ('reviews_a.json', 'reviews_b.json', 'reviews_c.json')]
        if (extension / 'adjudications_private.json').exists():
            added, changes = apply_adjudications(added, read(extension / 'adjudications_private.json')['amendments'])
            calibration += changes
        reviews += added
    validated = compile_review(cases, dictionary, reviews)
    by_review = {row['case_id']: row for bundle in reviews for row in bundle['cases']}
    annotations = [{'case_id': case['case_id'], 'original_actor_id': case['actor_id'],
                    'source_fact_ids': [f['id'] for f in case['member_facts']],
                    'reviewed_attribution': by_review[case['case_id']]['attribution'],
                    'reviewed_event_date': by_review[case['case_id']]['event_date'],
                    'scope_summary_ru': by_review[case['case_id']]['scope_summary_ru'],
                    'verdict': by_review[case['case_id']]['verdict']} for case in cases]
    snapshot_path = root / '.corpus_runtime/pilots/corpus-b2d1452fae0acf9c20f5/half-70-v1/results_private.json'
    source_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    if source_hash != dictionary['source_run_snapshot_sha256']:
        raise ValueError('Frozen extraction snapshot differs from approved dictionary')
    snapshot = read(snapshot_path)
    result = correct(snapshot, annotations, source_cases=cases)
    summary = result['public_summary']
    summary['provenance'].update(source_snapshot_file_sha256=source_hash,
                                 approved_dictionary_file_sha256=hashlib.sha256(dictionary_path.read_bytes()).hexdigest())
    private = root / '.corpus_runtime/corrected_evidence_2026_10_18'
    write(private / 'facts_private.json', result['private_facts'])
    write(private / 'audit_private.json', result['private_audit'])
    write(private / 'annotations_private.json', {'annotations': annotations})
    write(private / 'manifest_private.json', {**result['metadata'], 'source_snapshot_file_sha256': source_hash,
                                            'review_validation': validated['private_validation'],
                                            'source_cases': len(cases), 'api_calls': 0})
    public_review = validated['public']
    public_review['version'] = 'rwanda-chat-combined-review-v2'
    public_review['calibration_changes'] = calibration
    public_review['provenance'] = {'source_snapshot_file_sha256': source_hash,
                                   'review_content_sha256': digest(reviews),
                                   'source_case_manifest_sha256': digest([digest(case) for case in cases])}
    summary['provenance']['combined_review_sha256'] = digest(public_review)
    write(private / 'validated_public_review.json', public_review)
    write(private / 'public_summary.json', summary)
    if publish:
        write(root / 'reports/rwanda_corrected_evidence_2026_10_18.json', summary)
        write(root / 'reports/rwanda_chat_combined_review_2026_10_18.json', public_review)
    return {'counts': summary['counts'], 'review_counts': public_review['counts'],
            'api_calls': 0, 'new_api_cost_usd': 0, 'published': publish,
            'database_modified': False, 'canonical_study_modified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--without-extension', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    try:
        result = run(args.root, include_extension=not args.without_extension, publish=args.publish)
    except (ValueError, KeyError, OSError, TypeError):
        print(json.dumps({'error': 'Source correction validation failed. Inspect local frozen inputs.', 'api_calls': 0}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
