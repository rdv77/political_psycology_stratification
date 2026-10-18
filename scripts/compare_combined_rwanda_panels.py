"""Controlled feature-coverage and partial-synonym checks; no API calls."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.combine_rwanda_sources import OUT,read,write,fit_model,select_matrix
from src.corpus_pipeline.incomplete_latent import compare_geometry


def run():
    main=read(OUT/'models.json')
    actors=[a['id']for a in read(OUT/'roster.json')]
    scales=[s['id']for s in read(OUT/'dictionary.json')['scales']]
    cells=read(OUT/'ratings.json')['cells']
    x=np.array([c['midpoint']for c in cells],float).reshape(len(actors),len(scales))
    rows,cols=select_matrix(x,minimum_trait_fraction=0)
    wide_x=x[np.ix_(rows,cols)]
    wide={'actor_ids':np.asarray(actors)[rows].tolist(),'scale_ids':np.asarray(scales)[cols].tolist(),
          'selection':'Same final dictionary and ratings as the main model; >=5 informative actors per varying trait',
          'ratings':wide_x,'models':{},'comparison_to_main':{},'partial_overlap_check':{},
          'new_api_calls':0,'new_api_cost_usd':0}
    ai=[wide['actor_ids'].index(a)for a in main['actor_ids']]
    si=[wide['scale_ids'].index(s)for s in main['scale_ids']]
    main_x=np.array(main['ratings'],float)
    keep=np.array([s!='dossier-accessibility-growth-priorities-positive'for s in main['scale_ids']])
    for q in (3,4):
        fitted=fit_model(wide_x,q)
        wide['models'][str(q)]=fitted
        reference={k:np.asarray(main['models'][str(q)][k])for k in ('scores','loadings')}
        candidate={'scores':fitted['scores'][ai],'loadings':fitted['loadings'][si]}
        wide['comparison_to_main'][str(q)]=compare_geometry(reference,candidate)
        reduced=fit_model(main_x[:,keep],q)
        wide['partial_overlap_check'][str(q)]={'removed':'dossier-accessibility-growth-priorities-positive',
             'actors':len(main_x),'traits':int(keep.sum()),
             **compare_geometry({'scores':reference['scores'],'loadings':reference['loadings'][keep]},reduced)}
    write(OUT/'wide_models.json',wide)
    print('Controlled wide comparison:',wide_x.shape,wide['comparison_to_main'],wide['partial_overlap_check'])


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
