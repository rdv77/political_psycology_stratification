"""Versioned, offline crosswalk and conservative source-summary recoding.

This is a RULE-BASED proposal, not a claim of new manual review of every source.
Historical numeric ratings are deliberately not accepted as recoder inputs.
Unresolved coding and episode multiplicity are propagated as scenario ranges.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


MATCHES = {
    ('market-state-instruments', 'positive'): 'g01',
    ('external-domestic-resources', 'negative'): 'g04',
    ('ecological-limits-resource-use', 'positive'): 'g05',
    ('institutional-autonomy-coordination', 'positive'): 'g07',
    ('argumentation-evidence-values', 'positive'): 'm02',
    ('prevention-crisis-response', 'positive'): 'm05',
    ('mediation-adversarial-communication', 'positive'): 'm09',
    ('procedural-guarantees-enforcement', 'positive'): 'g06',
}


def target_id(scale_id, pole):
    if scale_id.startswith('russia-'):
        return scale_id + '-' + pole
    return MATCHES.get((scale_id, pole), 'dossier-' + scale_id + '-' + pole)


def canonical_url(url):
    p = urlsplit(str(url or '').strip())
    query = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith('utm_')
             and k.lower() not in {'fbclid', 'gclid'}]
    return urlunsplit((p.scheme.lower(), p.netloc.lower().removeprefix('www.'),
                      p.path.rstrip('/').removesuffix('/amp'), urlencode(sorted(query)), ''))


def text_hash(text):
    return hashlib.sha256(' '.join(str(text).split()).encode('utf-8')).hexdigest()


def build_dictionary(old, new):
    scales = [dict(s, origins=['corpus_dictionary']) for s in new]
    index = {s['id']: s for s in scales}
    crosswalk = []
    for s in old:
        for pole in ('positive', 'negative'):
            sid = target_id(s['id'], pole)
            if sid not in index:
                item = {'id': sid, 'name': s[pole], 'definition': s[pole],
                        'scope_from_original_dictionary': s['definition'],
                        'block': 'historical_additional', 'kind': 'unipolar',
                        'origins': ['historical_dictionary'],
                        'anchor0': 'Прямое отклонение именно данного свойства; отсутствие упоминания не ноль.',
                        'anchor3': 'Конкретное содержательное предложение, мера или действие в направлении свойства.',
                        'anchor5': 'Определяющий приоритет или существенное обязательство, явно установленные содержанием.'}
                scales.append(item)
                index[sid] = item
            else:
                index[sid]['origins'].append('historical_dictionary')
            crosswalk.append({'historical_scale_id': s['id'], 'pole': pole,
                              'historical_name': s[pole], 'unified_scale_id': sid,
                              'relationship': 'same_direction_narrow_support_only' if (s['id'], pole) in MATCHES
                              else 'same_definition' if s['id'].startswith('russia-') else 'retained_separately'})
    return scales, crosswalk


CONCRETE = re.compile(
    r'подпис\w*|ратифиц\w*|вв[её]л\w*|внедр\w*|утверд\w*|'
    r'созда(?:ть|ни\w*|л\w*)|учред\w*|запуст\w*|выдел\w*|'
    r'поруч\w*|обяза(?:л\w*|тельств\w*)|план\w*|программ\w*|'
    r'реформ\w*|проект\w*|механизм\w*|процедур\w*|'
    r'соглашен\w*|меморандум\w*|бюджет\w*|законопроект\w*|'
    r'конкретн\w*\s+(?:мер\w*|предложен\w*|действ\w*)', re.I)
INSUFFICIENT = re.compile(
    r'не\s+(?:доказывает|подтверждает|является)|без\s+(?:конкретн|собственн|прям)|'
    r'лишь\s+(?:участи|присутств|биограф)|не\s+предлож|не\s+содерж', re.I)


def propose_intensity(text, targeted_rationale):
    """Very conservative positive-manifestation proposal, NEVER old score mapping.

    Only a proposal interval 2..3 is emitted without a targeted concrete claim.
    Ambiguous/negated concrete references cannot authorize a point. No automatic
    0, 4 or 5; these require explicit semantic adjudication. Source facts alone
    cannot let an unrelated measure inflate another property in a long summary.
    """
    if not str(text).strip() or not str(targeted_rationale).strip():
        raise ValueError('A source and targeted claim are required')
    match = CONCRETE.search(targeted_rationale)
    source_match = CONCRETE.search(text)
    ambiguous = bool(INSUFFICIENT.search(targeted_rationale))
    if match and source_match and not ambiguous:
        return {'lower': 3., 'upper': 3., 'rule': 'concrete_in_targeted_claim_and_source',
                'matched_claim': match.group(), 'matched_source': source_match.group(),
                'coding_status': 'automatic_proposal_not_manual_measurement'}
    return {'lower': 2., 'upper': 3., 'rule': 'positive_manifestation_coding_uncertain',
            'matched_claim': match.group() if match else None,
            'matched_source': source_match.group() if source_match else None,
            'coding_status': 'automatic_coding_range_not_confidence_interval'}


def aggregate_sources(items):
    """Conservative envelope when cross-source episode partition is unresolved.

    Multiplicity is not intensity. Duplicating a source cannot change this
    envelope. A midpoint is only a declared numerical scenario.
    """
    if not items:
        return {'lower': None, 'upper': None, 'midpoint': None, 'sources': []}
    lo, hi = min(x['lower'] for x in items), max(x['upper'] for x in items)
    return {'lower': lo, 'upper': hi, 'midpoint': (lo + hi) / 2,
            'sources': sorted(set(x['origin'] for x in items))}
