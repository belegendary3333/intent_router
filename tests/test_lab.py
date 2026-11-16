"""Real local BGE tests; HTTP contracts, dataset isolation and analysis invariants."""

import copy
import hashlib
import importlib.util
import os
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from analysis_lab.corpus import MODES, load_cases, validate_cases
from analysis_lab.engine import AnalysisEngine, classify, metrics
from lab_encoder import BgeOnnxEncoder, normalize_text

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.getenv("LAB_MODEL_DIR", str(ROOT / "model")))


class CorpusTests(unittest.TestCase):
    def test_750_cases_and_group_isolation(self):
        cases = load_cases()
        counts = Counter((c["mode"], c["split"]) for c in cases)
        self.assertEqual(len(cases), 750)
        for mode in MODES:
            self.assertEqual([counts[mode, s] for s in ("train", "validation", "test")], [100, 20, 30])
        families = {}
        for c in cases:
            families.setdefault(c["family"], set()).add(c["split"])
        self.assertTrue(all(len(v) == 1 for v in families.values()))

    def test_duplicate_after_wake_word_rejected(self):
        cases = copy.deepcopy(load_cases())
        cases[1]["text"] = "小诺，" + cases[0]["text"]
        with self.assertRaises(ValueError):
            validate_cases(cases)

    def test_cross_split_family_rejected(self):
        cases = copy.deepcopy(load_cases())
        cases[-1]["family"] = cases[0]["family"]
        with self.assertRaises(ValueError):
            validate_cases(cases)

    def test_no_metrics_for_unlabeled_inputs(self):
        result = metrics([], [])
        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["macro_f1"])

    def test_threshold_decision_matches_existing_policy(self):
        examples = [{"mode": m, "text": m} for m in MODES]
        self.assertEqual(classify([.1, .2, .8, .3, .15], examples)["mode"], "singing")
        r = classify([.1, .2, .61, .3, .15], examples)
        self.assertEqual(r["mode"], "chat")
        self.assertEqual(r["matched_mode"], "singing")
        self.assertFalse(classify([.1, .2, .8, .79, .15], examples)["accepted"])


@unittest.skipUnless((MODEL / "onnx/model_quantized.onnx").exists(), "Run bash setup_lab.sh for real-model tests")
class RealLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="intent-lab-tests-")
        cls.root = Path(cls.temp.name)
        for name in ("routes.yaml", "lab_encoder.py"):
            shutil.copy2(ROOT / name, cls.root / name)
        (cls.root / "analysis_output").mkdir()
        cache = ROOT / "analysis_output/embeddings.npz"
        if cache.exists():
            shutil.copy2(cache, cls.root / "analysis_output/embeddings.npz")
        cls.router_hash = hashlib.sha256((cls.root / "routes.yaml").read_bytes()).hexdigest()
        cls.engine = AnalysisEngine(cls.root, MODEL, "onnx/model_quantized.onnx")
        cls.engine.build()
        import lab_app
        cls.web = lab_app
        cls.patcher = patch.object(lab_app, "engine", cls.engine)
        cls.patcher.start()
        cls.env = patch.dict(os.environ, {"LAB_SKIP_AUTOBUILD": "1"})
        cls.env.start()
        lab_app.job.update(busy=False, ready=True, error=None, stage="ready")
        cls.client = TestClient(lab_app.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.patcher.stop()
        cls.env.stop()
        cls.temp.cleanup()

    def setUp(self):
        self.engine.queries = []
        self.engine.query_vectors = np.empty((0, 512), dtype=np.float32)
        self.engine.layouts.clear()
        self.engine.query_serial = 0

    def test_encoder_parity_with_deployed_implementation(self):
        spec = importlib.util.spec_from_file_location("deployed_router_parity", ROOT / "standalone_app.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"ROUTER_MODEL_DIR": str(MODEL), "ROUTER_ONNX_FILE": "onnx/model_quantized.onnx",
                                    "ROUTER_ROUTES_FILE": str(ROOT / "routes.yaml")}):
            spec.loader.exec_module(module)
        texts = ["你好", "小诺，给我唱首歌", "帮我拿杯子", "我只是问跳舞有什么好处"]
        actual = self.engine.encoder.encode(texts)
        np.testing.assert_allclose(actual, module.encoder.encode(texts), atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
        for text in texts:
            vector = self.engine.encoder.encode([normalize_text(text)])[0]
            lab_result = classify(self.engine.baseline_vectors @ vector, self.engine.baseline)
            old_result = module.route(module.RouteRequest(text=text))
            for key in ("mode", "matched_mode", "score", "margin", "accepted", "matched_example"):
                self.assertEqual(lab_result[key], old_result[key])

    def test_anchor_selection_never_uses_validation_or_test(self):
        self.assertEqual(len(self.engine.anchors), 75)
        self.assertEqual(Counter(a["mode"] for a in self.engine.anchors), {m: 15 for m in MODES})
        for anchor in self.engine.anchors:
            self.assertEqual(anchor["split"], "train")
            self.assertIn(anchor["id"], anchor["member_ids"])
        self.assertEqual(hashlib.sha256((self.root / "routes.yaml").read_bytes()).hexdigest(), self.router_hash)

    def test_held_out_reports_and_fallback(self):
        for report in self.engine.reports.values():
            self.assertEqual(report["default"]["count"], 150)
            self.assertEqual(report["validation"]["count"], 100)
            self.assertTrue(all(c["split"] == "test" for c in report["cases"]))
            self.assertEqual(sum(map(sum, report["default"]["matrix"])), 150)
        self.assertTrue(all(c["gap"] < .08 for c in self.engine.confusions))

    def test_messages_and_probe_report(self):
        r = self.client.post("/api/messages", json={"texts": ["陪我聊聊天"], "expected_mode": "chat"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["probe_report"]["labeled"], 1)
        self.assertEqual(body["added"][0]["results"]["baseline"]["mode"], "chat")
        self.assertEqual(len(body["added"][0]["nearest_anchors"]), 5)
        self.assertEqual(len(self.engine.cases), 750)
        self.assertEqual(self.engine.reports["expanded"]["default"]["count"], 150)

    def test_unlabeled_message_has_no_accuracy(self):
        r = self.client.post("/api/messages", json={"texts": ["今天心情真好"]}).json()
        self.assertEqual(r["probe_report"]["labeled"], 0)
        self.assertIsNone(r["probe_report"]["schemes"]["expanded"]["accuracy"])

    def test_true_tsne_includes_new_sentence_and_scope(self):
        self.engine.analyze(["我只是想问周杰伦的代表作"])
        result = self.engine.layout("all", 30, 42)
        self.assertEqual(len(result["points"]), 751)
        self.assertEqual(sum(p["kind"] == "probe" for p in result["points"]), 1)
        self.assertTrue(all(np.isfinite([p["x"], p["y"]]).all() for p in result["points"]))
        self.assertIs(self.engine.layout("all", 30, 42), result)
        single = self.engine.layout("chat", 20, 7)
        self.assertEqual(len(single["points"]), 151)
        self.assertEqual(single["seed"], 7)
        self.assertTrue(0 <= single["trustworthiness"] <= 1)

    def test_rebuild_updates_existing_probe_anchors(self):
        self.engine.analyze(["我想听你唱稻香"], "singing")
        self.engine.build(20)
        self.assertEqual(len(self.engine.anchors), 100)
        self.assertEqual(len(self.engine.queries), 1)
        ids = {a["id"] for a in self.engine.anchors}
        self.assertTrue(all(a["id"] in ids for a in self.engine.queries[0]["nearest_anchors"]))
        self.assertEqual(self.engine.probe_report()["labeled"], 1)
        self.engine.build(15)

    def test_download_contains_current_probe_and_no_path_traversal(self):
        self.engine.analyze(["我的测试消息"], "chat")
        r = self.client.get("/api/download/report.json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["queries"][0]["text"], "我的测试消息")
        self.assertIn("attachment", r.headers["content-disposition"])
        self.assertEqual(self.client.get("/api/download/.env").status_code, 404)

    def test_invalid_input_rejected_atomically(self):
        for payload in ({"texts": []}, {"texts": ["字" * 257]}, {"texts": ["小诺，"]},
                        {"texts": [" "]}, {"texts": ["有效", ""]}, {"texts": ["正常"], "expected_mode": "unknown"},
                        {"texts": ["句子"] * 21}, {"texts": [123]}):
            with self.subTest(payload=str(payload)[:30]):
                self.assertEqual(self.client.post("/api/messages", json=payload).status_code, 422)
        self.assertEqual(len(self.engine.queries), 0)

    def test_host_and_cross_site_protection(self):
        self.assertEqual(self.client.get("/", headers={"Host": "evil.example"}).status_code, 400)
        self.assertEqual(self.client.post("/api/messages", json={"texts": ["你好"]}, headers={"Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.client.get("/api/layout?scope=unknown").status_code, 422)
        self.assertEqual(self.client.get("/api/layout?perplexity=999").status_code, 422)

    def test_page_and_local_vendor_bundle(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("发送并分析", r.text)
        self.assertIn("frame-ancestors 'none'", r.headers["content-security-policy"])
        self.assertEqual(self.client.get("/vendor/plotly.min.js").status_code, 200)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)


if __name__ == "__main__":
    unittest.main()
