"""Paired real-model checks without changing any production route samples."""
import copy
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from analysis_lab.engine import AnalysisEngine
from analysis_lab.comparison import compare_engines

ROOT = Path(__file__).resolve().parents[1]
SMALL = Path(os.getenv("LAB_MODEL_DIR", str(ROOT / "model")))
LARGE = Path(os.getenv("LAB_LARGE_MODEL_DIR", str(ROOT / "model/large")))

@unittest.skipUnless((SMALL / "onnx/model_quantized.onnx").exists() and
                     (LARGE / "onnx/model_quantized.onnx").exists(), "Download both models first")
class PairedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="intent-paired-tests-")
        root = Path(cls.temp.name)
        for name in ("routes.yaml", "lab_encoder.py"):
            shutil.copy2(ROOT / name, root / name)
        shutil.copytree(ROOT / "analysis_lab", root / "analysis_lab")
        if (ROOT / "analysis_output").exists():
            shutil.copytree(ROOT / "analysis_output", root / "analysis_output")
        cls.small = AnalysisEngine(root, SMALL, "onnx/model_quantized.onnx")
        cls.large = AnalysisEngine(root, LARGE, "onnx/model_quantized.onnx", "large")
        cls.small.build()
        cls.large.build()
        cls.comparison = compare_engines(cls.small, cls.large)
        import lab_app
        cls.web = lab_app
        cls.patches = [patch.object(lab_app, "engine", cls.small),
                       patch.object(lab_app, "large_engine", cls.large),
                       patch.object(lab_app, "comparison", cls.comparison),
                       patch.dict(os.environ, {"LAB_SKIP_AUTOBUILD": "1"})]
        for item in cls.patches:
            item.start()
        lab_app.job.update(busy=False, ready=True, error=None, stage="ready")
        cls.client = TestClient(lab_app.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        for item in reversed(cls.patches):
            item.stop()
        cls.temp.cleanup()

    def setUp(self):
        for engine in (self.small, self.large):
            engine.queries = []
            engine.query_vectors = np.empty((0, engine.dimension), dtype=np.float32)
            engine.query_serial = 0
            engine.layouts.clear()

    def test_large_dimension_pooling_normalization_and_separate_cache(self):
        vector = self.large.encoder.encode(["周杰伦有哪些代表作品？"])
        self.assertEqual(vector.shape, (1, 1024))
        np.testing.assert_allclose(np.linalg.norm(vector, axis=1), 1, atol=1e-6)
        self.assertEqual(self.large.encoder.encode([]).shape, (0, 1024))
        self.assertEqual(self.small.vectors.shape, (750, 512))
        self.assertEqual(self.large.vectors.shape, (750, 1024))
        self.assertNotEqual(self.small.output, self.large.output)
        self.assertNotEqual(self.small.fingerprint, self.large.fingerprint)

    def test_identical_routes_cases_splits_and_benchmark_inputs(self):
        self.assertEqual(self.small.cases, self.large.cases)
        self.assertEqual(self.small.baseline, self.large.baseline)
        self.assertEqual(self.small.settings, self.large.settings)
        self.assertEqual(self.small.benchmark["ids"], self.large.benchmark["ids"])
        for engine in (self.small, self.large):
            self.assertEqual(len(engine.anchors), 75)
            self.assertTrue(all(a["split"] == "train" for a in engine.anchors))
            self.assertEqual(engine.reports["expanded"]["default"]["count"], 150)
            self.assertEqual(engine.reports["expanded"]["validation"]["count"], 100)
            self.assertEqual(engine.clustering["space"], f"original-{engine.dimension}D")

    def test_comparison_rejects_mismatched_data(self):
        saved = self.large.cases
        self.large.cases = copy.deepcopy(saved)
        self.large.cases[-1]["text"] += "新句"
        try:
            with self.assertRaises(ValueError):
                compare_engines(self.small, self.large)
        finally:
            self.large.cases = saved

    def test_state_layout_models_and_validation(self):
        self.assertEqual(self.client.get("/api/state?model=large").json()["dimension"], 1024)
        self.assertEqual(self.client.get("/api/state").json()["dimension"], 512)
        self.assertEqual(self.client.get("/api/state?model=unknown").status_code, 422)
        self.assertEqual(self.client.get("/api/layout?model=unknown").status_code, 422)
        summary = self.client.get("/api/comparison").json()
        self.assertTrue(summary["same_routes"])
        self.assertTrue(summary["same_cases"])
        self.assertEqual(len(summary["cluster_agreement"]), 5)

    def test_paired_message_ids_labels_and_true_tsne(self):
        result = self.client.post("/api/messages", json={"texts": ["聊聊周杰伦的音乐"], "expected_mode": "chat"}).json()
        left, right = result["by_model"]["small"], result["by_model"]["large"]
        self.assertEqual(left["added"][0]["id"], right["added"][0]["id"])
        self.assertEqual(left["added"][0]["text"], right["added"][0]["text"])
        for key in ("small", "large"):
            self.assertEqual(result["by_model"][key]["probe_report"]["labeled"], 1)
            layout = self.client.get(f"/api/layout?model={key}&scope=chat&perplexity=20&seed=7").json()
            self.assertEqual(len(layout["points"]), 151)
            self.assertEqual(layout["model_key"], key)
            self.assertEqual(sum(p["kind"] == "probe" for p in layout["points"]), 1)
        self.assertEqual(len(self.large.cases), 750)

    def test_inference_failure_rolls_back_both_probe_streams(self):
        original = self.large.analyze
        def failing(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("simulated inference failure")
        with patch.object(self.large, "analyze", side_effect=failing):
            with self.assertRaises(RuntimeError):
                self.client.post("/api/messages", json={"texts": ["你好"]})
        for engine in (self.small, self.large):
            self.assertEqual(engine.queries, [])
            self.assertEqual(engine.query_serial, 0)
            self.assertEqual(engine.query_vectors.shape, (0, engine.dimension))

    def test_invalid_probe_does_not_modify_either_model(self):
        self.assertEqual(self.client.post("/api/messages", json={"texts": ["正常句", "小诺，"]}).status_code, 422)
        self.assertEqual(len(self.small.queries), 0)
        self.assertEqual(len(self.large.queries), 0)

    def test_download_models_and_paired_probes(self):
        self.client.post("/api/messages", json={"texts": ["给我唱首歌"]})
        report = self.client.get("/api/download/report.json?model=large").json()
        self.assertEqual(report["model_key"], "large")
        body = self.client.get("/api/download/comparison.json").json()
        self.assertEqual(body["probes"]["small"][0]["id"], body["probes"]["large"][0]["id"])
        self.assertIsNone(report["probe_report"]["schemes"]["expanded"]["accuracy"])
