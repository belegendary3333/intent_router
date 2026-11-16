"""Bounded workflow checks; all annotation fixtures live in temporary roots."""
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from analysis_lab.corpus import load_cases
from analysis_lab.engine import AnalysisEngine
from analysis_lab.comparison import compare_engines
from analysis_lab.feedback import (upsert_feedback, validate_feedback, feedback_summary,
                                   load_feedback, feedback_path)
from analysis_lab.real_data import test_hashes

ROOT = Path(__file__).resolve().parents[1]
SMALL = Path(os.getenv('LAB_MODEL_DIR', str(ROOT / 'model')))
LARGE = Path(os.getenv('LAB_LARGE_MODEL_DIR', str(ROOT / 'model/large')))


class FeedbackUnitTests(unittest.TestCase):
    def test_manual_idempotency_and_immutable_split(self):
        cases, c, created = upsert_feedback([], '给我演唱一段粤语歌曲吧', 'singing', 'train')
        validate_feedback(cases, load_cases(), [])
        same, _, created = upsert_feedback(cases, '给我演唱一段粤语歌曲吧', 'singing', 'train')
        self.assertEqual(cases, same); self.assertFalse(created)
        self.assertEqual(c['source'], 'manual')
        self.assertTrue(feedback_summary(cases, [])['pending'])
        with self.assertRaises(ValueError): upsert_feedback(cases, c['text'], c['mode'], 'test')

    def test_fixed_holdout_and_family_isolation(self):
        original = next(c for c in load_cases() if c['split'] == 'test')
        cases, _, _ = upsert_feedback([], original['text'], original['mode'], 'train')
        with self.assertRaises(ValueError): validate_feedback(cases, load_cases(), [])
        cases, _, _ = upsert_feedback([], '谈谈戴森吸尘器的使用方法', 'chat', 'train', family='vacuum-method')
        cases, _, _ = upsert_feedback(cases, '戴森吸尘器应该如何使用呢', 'chat', 'validation', family='vacuum-method')
        with self.assertRaises(ValueError): validate_feedback(cases, load_cases(), [])

    def test_registered_test_and_label_freeze(self):
        tests, c, _ = upsert_feedback([], '请说说清理吊灯的方法', 'chat', 'test', family='ceiling-clean')
        with self.assertRaises(ValueError): upsert_feedback(tests, c['text'], 'skill', 'test', family=c['family'])
        train, _, _ = upsert_feedback([], '我想了解吊灯怎么打扫', 'chat', 'train', family=c['family'])
        with self.assertRaises(ValueError): validate_feedback(train, load_cases(), [], history=test_hashes({'cases': tests}))


@unittest.skipUnless((SMALL / 'onnx/model_quantized.onnx').is_file() and (LARGE / 'onnx/model_quantized.onnx').is_file(), 'Download both models')
class FeedbackIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='intent-feedback-test-')
        cls.root = Path(cls.temp.name)
        for name in ('routes.yaml', 'lab_encoder.py'): shutil.copy2(ROOT / name, cls.root / name)
        shutil.copytree(ROOT / 'analysis_lab', cls.root / 'analysis_lab', ignore=shutil.ignore_patterns('private', '__pycache__'))
        if (ROOT / 'analysis_output').exists(): shutil.copytree(ROOT / 'analysis_output', cls.root / 'analysis_output')
        cls.small = AnalysisEngine(cls.root, SMALL, 'onnx/model_quantized.onnx')
        cls.large = AnalysisEngine(cls.root, LARGE, 'onnx/model_quantized.onnx', 'large')
        cls.small.build(); cls.large.build()
        cls.initial = {key: e.prepare_classifier() for key, e in (('small', cls.small), ('large', cls.large))}
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

    def setUp(self):
        # Explicit temporary test-owned files only, never the live corpus.
        for name in ('feedback.json', 'feedback.active.json', 'feedback.training.json', 'test_registry.json'):
            feedback_path(self.root).with_name(name).unlink(missing_ok=True)
        for key, e in (('small', self.small), ('large', self.large)):
            e.queries = []; e.apply_classifier(self.initial[key])
        self.web.comparison = compare_engines(self.small, self.large)

    def save(self, text='请解释用除尘掸打扫窗帘的步骤', mode='chat', split='train', family='curtain-clean'):
        return self.client.post('/api/feedback', json={'text': text, 'mode': mode, 'split': split, 'source': 'manual', 'family': family})

    def test_archive_does_not_train_and_survives_reload(self):
        before = [e.classifier.artifact() for e in (self.small, self.large)]
        response = self.save(); self.assertEqual(response.status_code, 200, response.text)
        f = response.json()['feedback']
        self.assertEqual(f['count'], 1); self.assertEqual(f['active_count'], 0); self.assertTrue(f['pending'])
        self.assertEqual(before, [e.classifier.artifact() for e in (self.small, self.large)])
        self.assertEqual(load_feedback(self.root)[0]['source'], 'manual')
        self.assertEqual(load_feedback(self.root, True), [])
        self.assertEqual(feedback_path(self.root).stat().st_mode & 0o777, 0o600)
        self.assertFalse(self.large.real_report['ready'])

    def test_fit_two_heads_reports_before_after_and_restart(self):
        route = hashlib.sha256((self.root / 'routes.yaml').read_bytes()).hexdigest()
        self.assertEqual(self.save().status_code, 200)
        self.assertEqual(self.save('我想了解纱窗通常多久洗一回', split='validation', family='screen-clean').status_code, 200)
        self.assertEqual(self.save('能不能哼一段京剧选段给我听', 'singing', 'test', 'opera-request').status_code, 200)
        response = self.client.post('/api/feedback/train')
        self.assertEqual(response.status_code, 200, response.text)
        f = response.json(); self.assertFalse(f['pending']); self.assertEqual(f['active_count'], 3)
        self.assertEqual(set(f['last_training']['models']), {'small', 'large'})
        for e in (self.small, self.large):
            self.assertEqual(e.classifier_metadata['train_count'], 501)
            self.assertEqual(e.classifier_metadata['validation_count'], 101)
            self.assertEqual(e.classifier_metadata['real_train_count'], 0)
            self.assertEqual(e.feedback_report['test_count'], 1)
            self.assertFalse(any(i == f['cases'][-1]['id'] for i in e.classifier.provenance['train_ids'] + e.classifier.provenance['validation_ids']))
            self.assertEqual(len(e.cases), 750)
            self.assertEqual(e.reports['expanded']['default']['count'], 150)
        self.assertEqual(hashlib.sha256((self.root / 'routes.yaml').read_bytes()).hexdigest(), route)
        restarted = AnalysisEngine(self.root, SMALL, 'onnx/model_quantized.onnx')
        restarted.build(); self.assertEqual(restarted.classifier_metadata['train_count'], 501)
        self.assertEqual(self.client.get('/api/download/feedback.training.json').status_code, 200)

    def test_invalid_or_one_model_failure_does_not_activate(self):
        c = next(c for c in self.small.cases if c['split'] == 'test')
        self.assertEqual(self.save(c['text']).status_code, 422)
        self.assertEqual(self.save(mode='guess').status_code, 422)
        self.assertEqual(self.client.post('/api/feedback/train').status_code, 422)
        self.save()
        before = [e.classifier.artifact() for e in (self.small, self.large)]
        with patch.object(self.large, 'prepare_classifier', side_effect=RuntimeError('simulated encoder failure')):
            with self.assertRaises(RuntimeError): self.client.post('/api/feedback/train')
        self.assertFalse(feedback_path(self.root, True).exists())
        self.assertEqual(before, [e.classifier.artifact() for e in (self.small, self.large)])
        self.assertTrue(self.client.get('/api/feedback').json()['pending'])

    def test_write_failure_restores_active_heads_and_training_record(self):
        self.save()
        before = [e.classifier.artifact() for e in (self.small, self.large)]
        original = self.large._save
        calls = []
        def fail_once():
            calls.append(1)
            if len(calls) == 1: raise OSError('simulated disk failure')
            return original()
        with patch.object(self.large, '_save', side_effect=fail_once):
            with self.assertRaises(OSError): self.client.post('/api/feedback/train')
        self.assertFalse(feedback_path(self.root, True).exists())
        self.assertFalse(feedback_path(self.root).with_name('feedback.training.json').exists())
        self.assertEqual(before, [e.classifier.artifact() for e in (self.small, self.large)])
