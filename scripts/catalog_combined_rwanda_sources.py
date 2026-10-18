"""Local provenance join; publish hashes/links, never social-media body text."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.combine_rwanda_sources import OUT, PRIVATE, read, write
from src.corpus_pipeline.harmonize_rwanda import canonical_url, text_hash


def run():
    ledger=read(OUT/'integrated_evidence.json')['records']
    index=read(ROOT/'.corpus_runtime/evidence_dossiers_2026_10_18/dossiers_private.json')['private_fact_index']
    cases={r['case_id']:r['fact_id'] for r in read(ROOT/'.corpus_runtime/native_mass_2026_10_18/queue_private.json')['rows']}
    legacy_paths={p.stem:p for folder in ('semantic_review_2026_10_17','semantic_extension_2026_10_18',
                                         'bottleneck_probe_2026_10_18')
                  for p in (ROOT/'.corpus_runtime'/folder/'cases').glob('*.json')}
    groups={}
    unresolved=set()
    for row in ledger:
        actor=row['actor_id']
        supports=[]
        if row['origin']=='historical_dossier':
            supports=[dict(s,origin='historical_dossier')for s in row['support']]
        else:
            for cid in row['case_ids']:
                fid=cases.get(cid)
                if fid not in index:
                    if cid in legacy_paths:
                        for s in read(legacy_paths[cid]).get('sources',[]):
                            supports.append({'source_url':s.get('url')or s.get('source_url'),
                                'origin':'reviewed_corpus','case_id':cid,'event_id':None})
                    else:
                        unresolved.add(cid)
                    continue
                fact=index[fid]['fact']
                supports.append({'source_url':fact.get('source_url'), 'origin':'reviewed_corpus',
                                 'case_id':cid,'fact_id':fid, 'event_id':None})
        for s in supports:
            if not s.get('source_url'):
                unresolved.add(s.get('case_id','unknown_source'))
                continue
            url=canonical_url(s['source_url'])
            key=actor+'::'+text_hash(url)
            g=groups.setdefault(key,{'publication_id':text_hash(key)[:24], 'actor_id':actor,
                  'canonical_url_sha256':text_hash(url),'canonical_url':url,
                  'origins':set(),'record_ids':set(),'event_ids':set(),'case_ids':set(),
                  'historical_source_urls':set()})
            g['origins'].add(s['origin'])
            g['record_ids'].add(row['id'])
            if s.get('event_id'):g['event_ids'].add(s['event_id'])
            if s.get('case_id'):g['case_ids'].add(s['case_id'])
            if s['origin']=='historical_dossier':g['historical_source_urls'].add(s['source_url'])
    private=[]
    public=[]
    for g in sorted(groups.values(),key=lambda r:r['publication_id']):
        g={k:sorted(v)if isinstance(v,set)else v for k,v in g.items()}
        private.append(g)
        public.append({k:v for k,v in g.items()if k!='canonical_url'})
    summary={'publication_actor_groups':len(public),
             'historical_groups':sum('historical_dossier'in r['origins']for r in public),
             'corpus_groups_with_joined_metadata':sum('reviewed_corpus'in r['origins']for r in public),
             'overlapping_actor_and_normalized_url_groups':sum(len(r['origins'])==2 for r in public),
             'unjoined_prior_pilot_cases':sorted(unresolved),
             'independent_event_count_claimed':False,
             'full_near_duplicate_event_resolution_claimed':False,
             'duplicate_invariant_intensity_aggregation':True,
             'new_api_cost_usd':0}
    write(PRIVATE/'publication_catalog_private.json',private)
    write(OUT/'source_catalog.json',{'summary':summary,'publications':public,
         'note':'Same normalized URL and actor indicate publication overlap, not independent events. Older pilot cases retain their case IDs and frozen public source-review trail; missing local URL joins are explicitly listed. Duplicating a record cannot change the min/max coding envelope.'})
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':run()
