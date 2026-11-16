"""Offline-friendly BGE intent router using ONNX Runtime directly."""

import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("ROUTER_MODEL_DIR", str(BASE_DIR / "model"))).expanduser().resolve()
ONNX_FILE = os.getenv("ROUTER_ONNX_FILE", "onnx/model_quantized.onnx")
ROUTES_FILE = Path(os.getenv("ROUTER_ROUTES_FILE", str(BASE_DIR / "routes.yaml"))).expanduser().resolve()
MIN_SCORE = float(os.getenv("ROUTER_MIN_SCORE", "0.62"))
MIN_MARGIN = float(os.getenv("ROUTER_MIN_MARGIN", "0.03"))
ALLOWED_MODES = {"chat", "skill", "singing", "dancing", "focus"}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("intent-router")


class RouteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=256)


def normalize_text(text: str) -> str:
    normalized = text.strip()
    normalized = re.sub(r"^\s*小诺[\s，,。！!：:]*", "", normalized)
    return normalized.strip()


def load_route_examples() -> tuple[list[str], list[str]]:
    with ROUTES_FILE.open("r", encoding="utf-8") as stream:
        routes = yaml.safe_load(stream)
    if not isinstance(routes, dict) or set(routes) != ALLOWED_MODES:
        raise RuntimeError(f"routes must contain exactly {sorted(ALLOWED_MODES)}: {ROUTES_FILE}")
    texts: list[str] = []
    modes: list[str] = []
    for mode, examples in routes.items():
        if not isinstance(examples, list) or not examples:
            raise RuntimeError(f"route {mode!r} must contain at least one example")
        for example in examples:
            if not isinstance(example, str) or not example.strip():
                raise RuntimeError(f"route {mode!r} contains an invalid example")
            texts.append(example.strip())
            modes.append(mode)
    return texts, modes


class BgeOnnxEncoder:
    def __init__(self, model_dir: Path, onnx_file: str):
        model_path = model_dir / onnx_file
        tokenizer_path = model_dir / "tokenizer.json"
        if not model_path.is_file():
            raise RuntimeError(f"ONNX model is missing: {model_path}")
        if not tokenizer_path.is_file():
            raise RuntimeError(f"tokenizer is missing: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("OMP_NUM_THREADS", "4"))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.lock = threading.Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.asarray([item.ids for item in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.asarray([item.type_ids for item in encoded], dtype=np.int64)
        missing = self.input_names - set(feed)
        if missing:
            raise RuntimeError(f"unsupported ONNX inputs: {sorted(missing)}")
        with self.lock:
            outputs = self.session.run(None, feed)
        embeddings = np.asarray(outputs[0], dtype=np.float32)
        if embeddings.ndim == 3:
            embeddings = embeddings[:, 0]
        if embeddings.ndim != 2 or embeddings.shape[1] != 512:
            raise RuntimeError(f"unexpected BGE output shape: {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, np.finfo(np.float32).eps)


load_started = time.perf_counter()
if not 0.0 <= MIN_SCORE <= 1.0 or not 0.0 <= MIN_MARGIN <= 2.0:
    raise RuntimeError("invalid ROUTER_MIN_SCORE or ROUTER_MIN_MARGIN")
encoder = BgeOnnxEncoder(MODEL_DIR, ONNX_FILE)
example_texts, example_modes = load_route_examples()
example_vectors = encoder.encode(example_texts)
logger.info(
    "model loaded backend=onnx-int8-direct examples=%d elapsed=%.3fs",
    len(example_texts),
    time.perf_counter() - load_started,
)

app = FastAPI(title="BGE Intent Router", version="1.1.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "BAAI/bge-small-zh-v1.5",
        "backend": "onnx-int8-direct",
        "embedding_dimension": int(example_vectors.shape[1]),
        "route_examples": len(example_texts),
    }


@app.post("/route")
def route(request: RouteRequest) -> dict:
    started = time.perf_counter()
    text = normalize_text(request.text)
    if not text:
        return {
            "text": "", "mode": "chat", "matched_mode": "chat",
            "matched_example": "", "score": 0.0, "margin": 0.0,
            "accepted": False, "scores": {},
            "reason": "empty_after_wake_word_removed",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    vector = encoder.encode([text])[0]
    similarities = example_vectors @ vector
    mode_scores: dict[str, float] = {}
    mode_examples: dict[str, str] = {}
    for index, raw_score in enumerate(similarities):
        mode = example_modes[index]
        score = float(raw_score)
        if mode not in mode_scores or score > mode_scores[mode]:
            mode_scores[mode] = score
            mode_examples[mode] = example_texts[index]
    ranking = sorted(mode_scores.items(), key=lambda item: item[1], reverse=True)
    matched_mode, best_score = ranking[0]
    second_score = ranking[1][1] if len(ranking) > 1 else 0.0
    margin = best_score - second_score
    accepted = best_score >= MIN_SCORE and margin >= MIN_MARGIN
    final_mode = matched_mode if accepted else "chat"
    result = {
        "text": text,
        "mode": final_mode,
        "matched_mode": matched_mode,
        "matched_example": mode_examples[matched_mode],
        "score": round(best_score, 4),
        "margin": round(margin, 4),
        "accepted": accepted,
        "scores": {mode: round(score, 4) for mode, score in ranking},
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    logger.info(
        "route text=%r mode=%s matched=%s score=%.4f margin=%.4f accepted=%s latency_ms=%.2f",
        text, final_mode, matched_mode, best_score, margin, accepted, result["latency_ms"],
    )
    return result
