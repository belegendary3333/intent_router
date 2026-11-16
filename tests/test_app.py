"""Offline routing/HTTP contract tests. Embeddings are stubbed, not accuracy tests."""

import hashlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parents[1]
MODES = ["singing", "dancing", "skill", "focus", "chat"]


class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        self.query_vectors = {}
        routes = yaml.safe_load((BASE_DIR / "routes.yaml").read_text(encoding="utf-8"))
        self.example_vectors = {
            text: np.eye(5, dtype=np.float32)[MODES.index(mode)]
            for mode, texts in routes.items() for text in texts
        }

    def encode(self, texts, **kwargs):
        return np.asarray([
            self.query_vectors.get(text, self.example_vectors.get(text, np.zeros(5)))
            for text in texts
        ], dtype=np.float32)


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="intent-router-unit-")
        model_dir = Path(cls.temp.name) / "model"
        (model_dir / "onnx").mkdir(parents=True)
        (model_dir / "onnx" / "model.onnx").touch()
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeSentenceTransformer
        environment = {
            "ROUTER_MODEL_DIR": str(model_dir),
            "ROUTER_ONNX_FILE": "onnx/model.onnx",
            "ROUTER_ROUTES_FILE": str(BASE_DIR / "routes.yaml"),
            "ROUTER_MIN_SCORE": "0.62",
            "ROUTER_MIN_MARGIN": "0.03",
        }
        spec = importlib.util.spec_from_file_location("recovered_intent_router_app", BASE_DIR / "app.py")
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}), patch.dict(os.environ, environment):
            spec.loader.exec_module(cls.module)
        cls.client = TestClient(cls.module.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.temp.cleanup()

    def setUp(self):
        self.module.model.query_vectors.clear()

    def route(self, text):
        response = self.client.post("/route", json={"text": text})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_original_sample_counts(self):
        routes = yaml.safe_load((BASE_DIR / "routes.yaml").read_text(encoding="utf-8"))
        self.assertEqual({mode: len(texts) for mode, texts in routes.items()},
                         {"singing": 21, "dancing": 16, "skill": 19,
                          "focus": 8, "chat": 24})

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["route_examples"], 88)
        self.assertEqual(response.json()["model"], "BAAI/bge-small-zh-v1.5")

    def test_five_modes(self):
        for text, mode in (("给我唱首歌", "singing"), ("给我跳个舞", "dancing"),
                           ("帮我擦桌子", "skill"),
                           ("小诺进入专注模式", "focus"),
                           ("陪我聊聊天", "chat")):
            with self.subTest(mode=mode):
                result = self.route(text)
                self.assertEqual(result["mode"], mode)
                self.assertTrue(result["accepted"])

    def test_wake_word_removal(self):
        result = self.route("  小诺，给我唱首歌  ")
        self.assertEqual(result["text"], "给我唱首歌")
        self.assertEqual(result["mode"], "singing")

    def test_low_score_falls_back(self):
        self.module.model.query_vectors["模糊请求"] = [0.61, 0.30, 0.20, 0.15, 0.10]
        result = self.route("模糊请求")
        self.assertEqual(result["mode"], "chat")
        self.assertEqual(result["matched_mode"], "singing")
        self.assertFalse(result["accepted"])

    def test_small_margin_falls_back(self):
        self.module.model.query_vectors["不明确请求"] = [0.80, 0.78, 0.20, 0.10, 0.10]
        result = self.route("不明确请求")
        self.assertEqual(result["mode"], "chat")
        self.assertFalse(result["accepted"])

    def test_score_above_threshold(self):
        self.module.model.query_vectors["明确请求"] = [0.63, 0.59, 0.20, 0.10, 0.10]
        result = self.route("明确请求")
        self.assertEqual(result["mode"], "singing")
        self.assertTrue(result["accepted"])

    def test_response_fields_compatible_with_hikari(self):
        result = self.route("帮我擦桌子")
        self.assertTrue({"text", "mode", "matched_mode", "matched_example", "score", "margin",
                         "accepted", "latency_ms"}.issubset(result))
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["accepted"], bool)

    def test_wake_word_only_and_whitespace(self):
        for text in ("小诺，", "   "):
            result = self.route(text)
            self.assertEqual(result["mode"], "chat")
            self.assertFalse(result["accepted"])
            self.assertEqual(result["matched_mode"], "chat")

    def test_invalid_requests(self):
        for payload in ({}, {"text": ""}, {"text": "字" * 257}, {"text": 123}):
            with self.subTest(payload=str(payload)[:30]):
                self.assertEqual(self.client.post("/route", json=payload).status_code, 422)

    def test_chat_discussion_examples(self):
        for text in ("我梦见自己在唱歌", "你觉得跳舞有什么好处", "唱歌为什么能让人开心"):
            self.assertEqual(self.route(text)["mode"], "chat")

    def test_invalid_category_config(self):
        invalid = Path(self.temp.name) / "invalid-routes.yaml"
        invalid.write_text("unknown: [错误模式]\n", encoding="utf-8")
        with patch.object(self.module, "ROUTES_FILE", invalid):
            with self.assertRaises(RuntimeError):
                self.module.load_route_examples()


if __name__ == "__main__":
    unittest.main()
