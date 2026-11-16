import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("ROUTER_MODEL_DIR", str(BASE_DIR / "model"))).expanduser().resolve()
ONNX_FILE = os.getenv("ROUTER_ONNX_FILE", "onnx/model.onnx")
ROUTES_FILE = Path(os.getenv("ROUTER_ROUTES_FILE", str(BASE_DIR / "routes.yaml"))).expanduser().resolve()
BACKEND = (
    "onnx-int8-avx2" if "qint8_avx2" in ONNX_FILE
    else "onnx-int8-arm64" if "qint8_arm64" in ONNX_FILE
    else "onnx-cpu-fp32"
)
ALLOWED_MODES = {"chat", "skill", "singing", "dancing", "focus"}

MIN_SCORE = float(os.getenv("ROUTER_MIN_SCORE", "0.62"))
MIN_MARGIN = float(os.getenv("ROUTER_MIN_MARGIN", "0.03"))

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
        raise RuntimeError(f"invalid or empty route config: {ROUTES_FILE}")

    texts: list[str] = []
    modes: list[str] = []
    for mode, examples in routes.items():
        if not isinstance(examples, list) or not examples:
            raise RuntimeError(f"route {mode!r} must contain at least one example")
        for example in examples:
            if not isinstance(example, str) or not example.strip():
                raise RuntimeError(f"route {mode!r} contains an empty or non-string example")
            texts.append(example.strip())
            modes.append(str(mode))
    return texts, modes


load_started = time.perf_counter()
if not 0.0 <= MIN_SCORE <= 1.0 or not 0.0 <= MIN_MARGIN <= 2.0:
    raise RuntimeError("invalid ROUTER_MIN_SCORE or ROUTER_MIN_MARGIN")
if not (MODEL_DIR / ONNX_FILE).is_file():
    raise RuntimeError(f"ONNX model is missing: {MODEL_DIR / ONNX_FILE}; run bash setup.sh first")

model = SentenceTransformer(
    str(MODEL_DIR),
    backend="onnx",
    model_kwargs={
        "file_name": ONNX_FILE,
        "provider": "CPUExecutionProvider",
        "export": False,
    },
)
model_lock = threading.Lock()

example_texts, example_modes = load_route_examples()
with model_lock:
    example_vectors = model.encode(
        example_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
example_vectors = np.asarray(example_vectors, dtype=np.float32)

logger.info(
    "model loaded backend=%s examples=%d elapsed=%.3fs",
    BACKEND,
    len(example_texts),
    time.perf_counter() - load_started,
)

app = FastAPI(title="BGE Intent Router", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "BAAI/bge-small-zh-v1.5",
        "backend": BACKEND,
        "embedding_dimension": int(example_vectors.shape[1]),
        "route_examples": len(example_texts),
    }


@app.post("/route")
def route(request: RouteRequest) -> dict:
    started = time.perf_counter()
    text = normalize_text(request.text)

    if not text:
        return {
            "text": "",
            "mode": "chat",
            "matched_mode": "chat",
            "matched_example": "",
            "score": 0.0,
            "margin": 0.0,
            "scores": {},
            "accepted": False,
            "reason": "empty_after_wake_word_removed",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    with model_lock:
        vector = model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

    similarities = example_vectors @ np.asarray(vector, dtype=np.float32)
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
        text,
        final_mode,
        matched_mode,
        best_score,
        margin,
        accepted,
        result["latency_ms"],
    )
    return result

