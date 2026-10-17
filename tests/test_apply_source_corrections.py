"""End-to-end offline publication and immutable-source boundary checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.apply_rwanda_source_corrections import run, write
from src.corpus_pipeline.scoring_preparation import digest
from tests.test_chat_review import fixtures as review_fixture
from tests.test_corrected_evidence import fixture as evidence_fixture


def fixture(root):
    snapshot, annotation, source = evidence_fixture()
    cases, dictionary, reviews = review_fixture()
    case, review = cases[0], reviews[0]['cases'][0]
    case.update(actor_id='actor', member_facts=deepcopy(snapshot['facts']))
    case['sources'] = source['sources']
    review['ratings'][0].update(support_quote='support cooperation.', source_document_id='doc-PRIVATE')
    review['event_date']['evidence_quote'] = '2 May 2026'
    snapshot['facts'].append(dict(snapshot['facts'][0], id='fact-SECOND', event_date='2026-06-01', date_evidence=None))
    old = root / '.corpus_runtime/semantic_review_2026_10_17'
    write(old / 'cases/U01.json', case)
    write(old / 'reviews_a.json', reviews[0])
    for name in ('b', 'c', 'controls'):
        write(old / f'reviews_{name}.json', {'cases': []})
    write(old / 'adjudications_private.json', {'amendments': []})
    source_path = root / '.corpus_runtime/pilots/corpus-b2d1452fae0acf9c20f5/half-70-v1/results_private.json'
    write(source_path, snapshot)
    dictionary['source_run_snapshot_sha256'] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    write(root / 'research/rwanda_politicians_2021_2026/source_scales_v2/approved_scales.json', dictionary)
    write(old / 'manifest_private.json', {'dictionary_content_sha256': digest(dictionary),
                                        'cases': [{'case_id': 'U01', 'sha256': digest(case)}]})
    return source_path, old


class ApplySourceCorrectionsTests(unittest.TestCase):
    def test_all_facts_processed_safe_publication_and_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, old = fixture(root)
            before = source.read_bytes()
            with patch('httpx.Client.send', side_effect=AssertionError('Network forbidden')):
                result = run(root, include_extension=False, publish=True)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(result['counts']['all_extracted_facts_processed'], 2)
            self.assertEqual(result['counts']['exact_reviewed_facts_applied'], 1)
            private = json.loads((root / '.corpus_runtime/corrected_evidence_2026_10_18/facts_private.json').read_text(encoding='utf-8'))
            self.assertIsNone(private[1]['event_date'])
            self.assertEqual(private[0]['reviewed_scope_ru'], 'Поддержано предпринимательство.')
            public = (root / 'reports/rwanda_corrected_evidence_2026_10_18.json').read_text(encoding='utf-8')
            self.assertNotIn('fact-PRIVATE', public)
            self.assertNotIn('PRIVATE CLAIM', public)
            self.assertEqual(result['api_calls'], 0)

    def test_stale_source_snapshot_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = fixture(root)
            source.write_text(source.read_text(encoding='utf-8') + ' ', encoding='utf-8')
            with self.assertRaises(ValueError):
                run(root, include_extension=False, publish=True)
            self.assertFalse((root / 'reports').exists())

    def test_changed_dossier_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, old = fixture(root)
            case = json.loads((old / 'cases/U01.json').read_text(encoding='utf-8'))
            case['actor_id'] = 'different'
            write(old / 'cases/U01.json', case)
            with self.assertRaises(ValueError):
                run(root, include_extension=False, publish=True)
            self.assertFalse((root / 'reports').exists())

    def test_no_publish_keeps_all_results_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            result = run(root, include_extension=False)
            self.assertFalse(result['published'])
            self.assertFalse((root / 'reports').exists())


if __name__ == '__main__':
    unittest.main()
