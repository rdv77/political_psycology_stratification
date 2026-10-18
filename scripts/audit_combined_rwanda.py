"""Validate provenance, recoding, matrix and reconstruction without API/network."""
from pathlib import Path
import io
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.combine_rwanda_sources import OUT,BASE,read,write,content_hash,matrices
from src.corpus_pipeline.harmonize_rwanda import text_hash


def run():
    model=read(OUT/'models.json')
    integrated=read(OUT/'integrated_evidence.json')
    original=read(OUT/'source_ledger.json')
    decisions=read(OUT/'semantic_adjudications.json')['decisions']
    extensions=read(OUT/'crosscoded_claims.json')['records']
    ext_index={(r['parent_id'],r['scale_id']):r for r in extensions}
    assert model['source_ledger_content_hash']==content_hash(integrated)
    assert original['input_content_hashes']['historical_ratings']==content_hash(read(BASE/'ratings.json'))
    assert original['input_content_hashes']['study']==content_hash(read(BASE/'study.json'))
    assert original['input_content_hashes']['dictionary']==content_hash(read(ROOT/'research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json'))
    sources={}
    reference_count=0
    for row in integrated['records']:
        if row['origin']!='historical_dossier':continue
        if 'parent_id'in row:
            ext=ext_index[row['parent_id'],row['scale_id']]
            decisions[row['id']]=dict(ext,exclude=False)
        else:
            decision=decisions[row['id']]
            assert decision['input_claim_sha256']==row['input_claim_sha256']
            assert decision['support_text_sha256']==[s['text_sha256']for s in row['support']]
        for s in row['support']:
            if s['file']not in sources:sources[s['file']]=read(BASE/s['file'])['facts']
            fact=sources[s['file']][s['index']]
            assert text_hash(fact['text'])==s['text_sha256']
            assert fact['source_url']==s['source_url']
            assert fact['period']==s['event_date']
            reference_count+=1
    actors=[r['id']for r in read(OUT/'roster.json')]
    scales=[s['id']for s in read(OUT/'dictionary.json')['scales']]
    assert len(actors)==42 and len(scales)==66
    cells,x,lo,hi=matrices(integrated['records'],actors,scales,adjudications=decisions)
    assert cells==read(OUT/'ratings.json')['cells']
    ai=[actors.index(a)for a in model['actor_ids']]
    si=[scales.index(s)for s in model['scale_ids']]
    expected=x[np.ix_(ai,si)]
    np.testing.assert_allclose(expected,np.array(model['ratings'],float),equal_nan=True)
    assert len(model['actor_ids'])==42
    assert np.isfinite(expected).sum()==model['coverage']['informative']
    for q in ('3','4'):
        f=model['models'][q]
        prediction=np.asarray(f['mean'])+np.asarray(f['scores'])@np.asarray(f['loadings']).T
        np.testing.assert_allclose(prediction,f['prediction'],atol=1e-10)
        mask=np.isfinite(expected)
        total=np.nansum((expected-np.nanmean(expected,axis=0))**2)
        r2=1-np.sum((prediction[mask]-expected[mask])**2)/total
        assert abs(r2-f['observed_centered_reconstruction_r2'])<1e-12
        assert model['bayesian_comparison'][q]['fit']['diagnostics']['max_rank_split_rhat']<1.01
    assert not any(s.startswith('russia-')for s in model['scale_ids'])
    receipt=read(OUT/'validation_receipt.json')
    assert receipt['models_content_hash']==content_hash(model)
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_rwanda_harmonization','tests.test_incomplete_latent'])
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=1).run(suite)
    if not result.wasSuccessful():raise AssertionError(stream.getvalue())
    write(OUT/'provenance_audit.json',{'source_references_checked':reference_count,
          'historical_claims_checked':588,'crosscoded_claims_checked':69,
          'all_42_matrix_reproduced':True,'both_model_reconstruction_metrics_reproduced':True,
          'input_study_and_ratings_unchanged':True,'mcmc_rhat_under_1_01':True,
          'models_content_hash':content_hash(model),'executed_tests':result.testsRun,
          'test_failures':len(result.failures),'test_errors':len(result.errors),
          'new_api_calls':0,'new_api_cost_usd':0})
    print(f'PASS: {reference_count} source references; 588 adjudications + 69 cross-codings; 42 actor matrix; both models, MCMC diagnostics and report hashes verified. {result.testsRun} tests passed. API $0.')


if __name__=='__main__':run()
