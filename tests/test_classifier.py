import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from analysis_lab.classifier import FrozenIntentClassifier, softmax
from analysis_lab.corpus import MODES, load_cases
from analysis_lab.engine import AnalysisEngine, metrics
from analysis_lab.comparison import compare_engines
from analysis_lab.real_data import validate_real_dataset, test_hashes

ROOT = Path(__file__).resolve().parents[1]
SMALL = Path(os.getenv('LAB_MODEL_DIR', str(ROOT / 'model')))
LARGE = Path(os.getenv('LAB_LARGE_MODEL_DIR', str(ROOT / 'model/large')))


def fixture(split='test'):
    # Fabricated unit-test data only; never seeded into the live ASR corpus.
    return {'dataset_id': 'test-fixture-only', 'cases': [
        {'id': f'{split}-{m}-{i}', 'text': f'测试专用 {m} 编号{i} 集合{split}',
         'mode': m, 'split': split, 'family': f'{split}-{m}-{i}', 'source': 'asr'}
        for m in MODES for i in range(5)]}


class ClassifierUnitTests(unittest.TestCase):
    def test_clarify_accounting(self):
        stats = metrics(['chat', 'skill', 'singing'], [
            {'mode': 'clarify', 'accepted': False}, {'mode': 'clarify', 'accepted': False},
            {'mode': 'singing', 'accepted': True}])
        self.assertEqual(stats['accuracy'], 1/3)
        self.assertEqual(stats['unsafe_chat_activation'], 0)
        self.assertAlmostEqual(stats['rejection_rate'], 2/3)
        self.assertEqual(stats['accepted_accuracy'], 1)
        self.assertEqual(stats['action_recall'], .5)
        self.assertEqual(sum(map(sum, stats['matrix'])), 3)
        self.assertIn('clarify', stats['matrix_labels'])
        self.assertIsNone(metrics(['chat'], [{'mode': 'clarify', 'accepted': False}])['accepted_accuracy'])
        self.assertIsNone(metrics([], [])['accuracy'])

    def test_serializable_linear_head_and_softmax(self):
        rng = np.random.default_rng(42)
        count = len(MODES)
        x = np.concatenate([np.tile(np.eye(count)[i], (10, 1)) + rng.normal(0, .01, (10, count)) for i in range(count)])
        cases = [{'id': f'train-{i}', 'mode': MODES[i//10]} for i in range(10 * count)]
        val = [dict(c, id=c['id'].replace('train', 'validation')) for c in cases]
        head = FrozenIntentClassifier().fit(x, cases, x, val, 'fake')
        a = head.artifact()
        np.testing.assert_allclose(x @ np.array(a['coef']).T + np.array(a['intercept']), head.logits(x))
        np.testing.assert_allclose(softmax(head.logits(x), head.temperature).sum(axis=1), 1)
        for mode in MODES: head.thresholds[mode]['min_probability'] = 1.01
        self.assertTrue(all(p['mode'] == 'clarify' for p in head.predict(x)))
        self.assertTrue(all(p['accepted'] for p in head.predict(x, False)))

    def test_real_data_validation(self):
        p = fixture()
        normalized = validate_real_dataset(p, load_cases(), [])
        self.assertTrue(all(c['id'].startswith('real:') for c in normalized['cases']))
        mixed = fixture('train'); mixed['cases'] += fixture('validation')['cases']
        mixed['cases'][0]['text'] = p['cases'][0]['text']
        with self.assertRaisesRegex(ValueError, '曾用于测试'):
            validate_real_dataset(mixed, load_cases(), [], test_hashes(normalized))
        mixed['cases'][0]['text'] = '不相同的新句子但属于原留出会话'
        mixed['cases'][0]['family'] = normalized['cases'][0]['family']
        with self.assertRaisesRegex(ValueError, '曾用于测试'):
            validate_real_dataset(mixed, load_cases(), [], test_hashes(normalized))
        variants = []
        p = fixture(); p['cases'][0]['source'] = 'generated'; variants.append(p)
        p = fixture(); p['cases'][0]['mode'] = 'guess'; variants.append(p)
        p = fixture(); p['cases'][1]['text'] = '小诺，' + p['cases'][0]['text']; variants.append(p)
        p = fixture(); p['cases'][0]['text'] = load_cases()[0]['text']; variants.append(p)
        variants.append(fixture('train'))
        p = fixture(); p['cases'] += fixture('validation')['cases']; p['cases'][-1]['family'] = p['cases'][0]['family']; variants.append(p)
        for p in variants:
            with self.assertRaises(ValueError): validate_real_dataset(p, load_cases(), [])


@unittest.skipUnless((SMALL / 'onnx/model_quantized.onnx').is_file() and (LARGE / 'onnx/model_quantized.onnx').is_file(), 'Download both encoders')
class ClassifierIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='intent-head-test-')
        cls.root = Path(cls.temp.name)
        for f in ('routes.yaml', 'lab_encoder.py'): shutil.copy2(ROOT / f, cls.root / f)
        shutil.copytree(ROOT / 'analysis_lab', cls.root / 'analysis_lab', ignore=shutil.ignore_patterns('private', '__pycache__'))
        if (ROOT / 'analysis_output').exists(): shutil.copytree(ROOT / 'analysis_output', cls.root / 'analysis_output')
        cls.small = AnalysisEngine(cls.root, SMALL, 'onnx/model_quantized.onnx')
        cls.large = AnalysisEngine(cls.root, LARGE, 'onnx/model_quantized.onnx', 'large')
        cls.small.build(); cls.large.build()
        import lab_app
        cls.web = lab_app
        cls.patches = [patch.object(lab_app, 'engine', cls.small), patch.object(lab_app, 'large_engine', cls.large),
            patch.object(lab_app, 'comparison', compare_engines(cls.small, cls.large)), patch.dict(os.environ, {'LAB_SKIP_AUTOBUILD': '1'})]
        for p in cls.patches: p.start()
        lab_app.job.update(busy=False, ready=True, error=None)
        cls.client = TestClient(lab_app.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        for p in reversed(cls.patches): p.stop()
        cls.temp.cleanup()

    def test_a_test_data_cannot_affect_fit(self):
        e = self.large; ids = e.indices['test']
        vectors, labels = e.vectors[ids].copy(), [e.cases[i]['mode'] for i in ids]
        before = e.classifier.artifact()
        try:
            e.vectors[ids] = 0
            for i in ids: e.cases[i]['mode'] = 'chat'
            after = e.prepare_classifier()['head'].artifact()
            for key in ('coef', 'intercept', 'C', 'temperature', 'thresholds', 'provenance'):
                self.assertEqual(before[key], after[key])
        finally:
            e.vectors[ids] = vectors
            for i, mode in zip(ids, labels): e.cases[i]['mode'] = mode

    def test_b_probe_and_artifact(self):
        before = self.large.classifier.artifact()
        response = self.client.post('/api/messages', json={'texts': ['讨论一下如何收拾房间'], 'expected_mode': 'chat'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()['by_model']['large']['added'][0]['results']['classifier']['mode'], MODES + ['clarify'])
        self.assertEqual(before, self.large.classifier.artifact())
        a = self.client.get('/api/download/classifier.candidate.json?model=large').json()
        self.assertEqual(a['dimension'], 1024)
        self.assertEqual(len(a['provenance']['train_ids']), 500)
        self.assertEqual(self.client.get('/api/download/real-dataset.template.json').json()['cases'], [])

    def test_c_invalid_or_failed_paired_fit_does_not_commit(self):
        before = [e.classifier.artifact() for e in (self.small, self.large)]
        path = self.root / 'analysis_lab/private/real_dataset.json'
        p = fixture(); p['cases'][0]['mode'] = 'guess'
        self.assertEqual(self.client.post('/api/real-dataset', json={'dataset': json.dumps(p)}).status_code, 422)
        self.assertEqual(self.client.post('/api/real-dataset', json={'dataset': 'bad json'}).status_code, 422)
        with patch.object(self.large, 'prepare_classifier', side_effect=RuntimeError('simulated encoder failure')):
            with self.assertRaises(RuntimeError): self.client.post('/api/real-dataset', json={'dataset': json.dumps(fixture())})
        self.assertFalse(path.exists())
        self.assertEqual(before, [e.classifier.artifact() for e in (self.small, self.large)])

    def test_c2_disk_failure_rolls_back_both_heads_and_private_files(self):
        before = [e.classifier.artifact() for e in (self.small, self.large)]
        original = self.large._save
        calls = []
        def fail_once():
            calls.append(1)
            if len(calls) == 1: raise OSError('simulated disk failure')
            return original()
        with patch.object(self.large, '_save', side_effect=fail_once):
            with self.assertRaises(OSError): self.client.post('/api/real-dataset', json={'dataset': json.dumps(fixture())})
        self.assertFalse((self.root / 'analysis_lab/private/real_dataset.json').exists())
        self.assertFalse((self.root / 'analysis_lab/private/test_registry.json').exists())
        self.assertEqual(before, [e.classifier.artifact() for e in (self.small, self.large)])

    def test_d_test_only_keeps_coefficients_and_tracks_seen_tests(self):
        before = [e.classifier.artifact() for e in (self.small, self.large)]
        digest = hashlib.sha256((self.root / 'routes.yaml').read_bytes()).hexdigest()
        response = self.client.post('/api/real-dataset', json={'dataset': json.dumps(fixture())})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()['previously_evaluated'])
        for e, a in zip((self.small, self.large), before):
            self.assertEqual(a, e.classifier.artifact())
            self.assertEqual(e.real_report['reports']['classifier']['calibrated']['count'], 25)
            self.assertEqual(len(e.cases), 750)
        self.assertEqual((self.root / 'analysis_lab/private/real_dataset.json').stat().st_mode & 0o777, 0o600)
        self.assertEqual(hashlib.sha256((self.root / 'routes.yaml').read_bytes()).hexdigest(), digest)
        again = self.client.post('/api/real-dataset', json={'dataset': json.dumps(fixture())})
        self.assertTrue(again.json()['previously_evaluated'])

    def test_e_real_train_and_validation_are_separate_and_restartable(self):
        p = fixture('train'); p['cases'] += fixture('validation')['cases'] + fixture('test')['cases']
        response = self.client.post('/api/real-dataset', json={'dataset': json.dumps(p)})
        self.assertEqual(response.status_code, 200, response.text)
        for e in (self.small, self.large):
            self.assertEqual(e.classifier_metadata['train_count'], 525)
            self.assertEqual(e.classifier_metadata['validation_count'], 25)
            self.assertEqual(e.classifier_metadata['validation_source'], 'real_asr')
            provenance = e.classifier.provenance
            self.assertFalse(any(i.startswith('real:test') for i in provenance['train_ids'] + provenance['validation_ids']))
        self.large.build()
        self.assertEqual(self.large.classifier_metadata['train_count'], 525)
