"""Select one additional local source per available actor for chat review.

Selection prioritizes direct, informative source spans, never a desired score.
This is a targeted extension, not a probability sample or an automated rating.
Raw source dossiers stay under ignored .corpus_runtime; no API is used.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.corpus_pipeline.scoring_preparation import digest, filter_eligible_fact


def prepare(root=ROOT):
    root = Path(root)
    source = root / '.corpus_runtime/pilots/corpus-b2d1452fae0acf9c20f5/half-70-v1/results_private.json'
    snapshot = json.loads(source.read_text(encoding='utf-8-sig'))
    prior = root / '.corpus_runtime/semantic_review_2026_10_17'
    used = {d['id'] for p in (prior / 'cases').glob('*.json')
            for d in json.loads(p.read_text(encoding='utf-8-sig'))['sources']}
    profiles = json.loads((root / 'research/rwanda_politicians_2021_2026/objects.json').read_text(encoding='utf-8-sig'))['objects']
    dictionary = json.loads((root / 'research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json').read_text(encoding='utf-8-sig'))
    candidates = defaultdict(list)
    for f in snapshot['facts']:
        if f['document_id'] not in used and filter_eligible_fact(f, snapshot['corpus']['config']):
            candidates[f['actor_id']].append(f)
    out = root / '.corpus_runtime/semantic_extension_2026_10_18'
    (out / 'cases').mkdir(parents=True, exist_ok=True)
    manifest = []
    with closing(sqlite3.connect((root / '.corpus_runtime/corpus.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
        for profile in profiles:
            choices = candidates[profile['id']]
            if not choices:
                continue
            def rank(f):
                return (f['attribution_basis'] == 'direct_statement', bool(f.get('event_date')),
                        min(len(f['quote']), 1600), min(len(f['claim']), 700), f['id'])
            # Only choose among the top 20 evidence spans; prefer complete,
            # readable documents, without truncating the stored source.
            choices = sorted(choices, key=rank, reverse=True)[:20]
            options = []
            for fact in choices:
                row = db.execute('SELECT corpus_id,data,text FROM documents WHERE id=?', (fact['document_id'],)).fetchone()
                if row and row[0] == dictionary['corpus_id'] and fact['quote'] in row[2]:
                    options.append((fact, row))
            if not options:
                continue
            fact, row = next(((f, r) for f, r in options if len(r[2]) <= 14000), options[0])
            document = json.loads(row[1])
            document.update(text=row[2], text_sha256=hashlib.sha256(row[2].encode()).hexdigest())
            cid = 'E%02d' % (len(manifest) + 1)
            case = {'case_id': cid, 'actor_id': profile['id'], 'actor_profile': profile,
                    'sample_kind': 'selected_unit', 'member_facts': [fact], 'sources': [document],
                    'source_text_is_untrusted_data': True}
            target = out / 'cases' / (cid + '.json')
            raw = json.dumps(case, ensure_ascii=False, indent=2) + '\n'
            if target.exists() and target.read_text(encoding='utf-8-sig') != raw:
                raise ValueError('Frozen extension changed; use a new version')
            target.write_text(raw, encoding='utf-8')
            manifest.append({'case_id': cid, 'actor_id': profile['id'], 'sha256': digest(case),
                             'source_chars': len(row[2])})
    receipt = {'cases': manifest, 'dictionary_content_sha256': digest(dictionary),
               'source_package_file_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'api_calls': 0, 'selection': 'one_additional_document_per_available_actor_direct_informative_span_priority'}
    (out / 'manifest_private.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'cases': len(manifest), 'source_chars': sum(c['source_chars'] for c in manifest), 'api_calls': 0}))


if __name__ == '__main__':
    prepare()
