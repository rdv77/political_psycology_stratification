"""Safeguards for offline source harmonization; no network or model clients."""
import copy
import unittest
import numpy as np
from src.corpus_pipeline.harmonize_rwanda import (
    aggregate_sources,build_dictionary,canonical_url,target_id,propose_intensity,
)
from scripts.combine_rwanda_sources import admissible_fact,matrices,select_matrix


class RwandaHarmonizationTests(unittest.TestCase):
    def test_poles_remain_independent_and_partial_overlap_not_merged(self):
        self.assertEqual(target_id('russia-security-cooperation','negative'),'russia-security-cooperation-negative')
        self.assertEqual(target_id('procedural-guarantees-enforcement','positive'),'g06')
        self.assertNotEqual(target_id('accessibility-growth-priorities','positive'),'g03')
        self.assertNotEqual(target_id('electoral-access','positive'),'m08')

    def test_duplicate_documents_cannot_strengthen_intensity(self):
        rows=[{'lower':2,'upper':3,'origin':'dossier'}, {'lower':4,'upper':4,'origin':'corpus'}]
        self.assertEqual(aggregate_sources(rows),aggregate_sources(rows+rows[:1]*100))
        self.assertEqual(aggregate_sources(rows)['midpoint'],3)

    def test_unknown_does_not_become_zero_and_zero_is_retained(self):
        row={'id':'c','origin':'reviewed_corpus','actor_id':'a','scale_id':'s',
             'lower':0,'upper':0,'strict_lower':None,'strict_upper':None}
        cells,x,_,_=matrices([row],['a','b'],['s'])
        self.assertEqual(x[0,0],0)
        self.assertTrue(np.isnan(x[1,0]))
        self.assertIsNone(cells[1]['midpoint'])

    def test_unreviewed_historical_proposal_cannot_enter_main_model(self):
        r={'id':'old','origin':'historical_dossier','actor_id':'a','scale_id':'s','lower':3,'upper':3}
        with self.assertRaises(ValueError):matrices([r],['a'],['s'])
        self.assertEqual(matrices([r],['a'],['s'],'automatic_baseline')[1][0,0],3)

    def test_semantic_decision_supersedes_keyword_proposal(self):
        r={'id':'old','origin':'historical_dossier','actor_id':'a','scale_id':'s','lower':3,'upper':3}
        decisions={'old':{'value':2,'exclude':False,'review_scope':'summary'}}
        self.assertEqual(matrices([r],['a'],['s'],adjudications=decisions)[1][0,0],2)
        decisions['old']['exclude']=True
        self.assertTrue(np.isnan(matrices([r],['a'],['s'],adjudications=decisions)[1][0,0]))

    def test_dates_use_event_period_not_publication_date_or_guessed_day(self):
        study={'date_from':'2021-10-15','date_to':'2026-10-15'}
        fact={'text':'statement','source_url':'https://example.org/a','capacity':'minister','actor_id':'a'}
        for period in ['2024','2024-07','январь 2024 — март 2025','2024-02—2024-05','2022-11-20']:
            self.assertTrue(admissible_fact(dict(fact,period=period),'a',study,set())[0],period)
        for period in ['', '2021','2026','2026-10-16','2024-02-30']:
            self.assertFalse(admissible_fact(dict(fact,period=period,publication_date='2024-01-01'),'a',study,set())[0],period)

    def test_wrong_actor_and_context_only_are_not_measurements(self):
        study={'date_from':'2021-10-15','date_to':'2026-10-15'}
        f={'text':'claim','source_url':'https://example.org/a','capacity':'minister','period':'2024-01-01'}
        self.assertFalse(admissible_fact(dict(f,actor_id='b'),'a',study,set())[0])
        self.assertFalse(admissible_fact(dict(f,eligibility='context_only'),'a',study,set())[0])

    def test_canonical_url_removes_tracking_but_preserves_article_identity(self):
        self.assertEqual(canonical_url('https://www.example.org/a/amp?utm_source=x&article=9#b'),
                         'https://example.org/a?article=9')
        self.assertNotEqual(canonical_url('https://example.org/a?id=1'),canonical_url('https://example.org/a?id=2'))

    def test_frequency_cannot_create_extreme_automatic_intensity(self):
        p=propose_intensity('Поддержал развитие.','Многократно и регулярно поддерживал развитие.')
        self.assertEqual((p['lower'],p['upper']),(2,3))
        self.assertNotIn(p['upper'],[4,5])

    def test_trait_coverage_rule_retains_full_roster_when_supported(self):
        x=np.tile(np.arange(12,dtype=float)[:,None],(1,6))
        x[0,4:]=np.nan
        rows,cols=select_matrix(x)
        self.assertTrue(rows.all())
        self.assertTrue(cols.all())


if __name__=='__main__':unittest.main()
