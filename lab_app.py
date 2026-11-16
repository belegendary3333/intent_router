"""Loopback-only local analysis web app. Does not call KRA or change routing."""

import os
import json
import copy
import threading
from datetime import datetime, timezone
from typing import Literal
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import plotly
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from analysis_lab.corpus import MODES
from analysis_lab.engine import AnalysisEngine
from analysis_lab.comparison import compare_engines
from analysis_lab.real_data import validate_real_dataset, test_hashes, public_template, digest_dataset
from analysis_lab.feedback import (load_feedback, feedback_path, validate_feedback,
                                   upsert_feedback, feedback_summary, feedback_digest)

ROOT = Path(__file__).resolve().parent
engine = AnalysisEngine(ROOT, Path(os.getenv("LAB_MODEL_DIR", str(ROOT / "model"))),
                        os.getenv("LAB_ONNX_FILE", "onnx/model_quantized.onnx"))
large_engine = AnalysisEngine(ROOT, Path(os.getenv("LAB_LARGE_MODEL_DIR", str(ROOT / "model/large"))),
                              os.getenv("LAB_LARGE_ONNX_FILE", "onnx/model_quantized.onnx"), "large")
comparison = None
operation_lock = threading.RLock()
state_lock = threading.Lock()
job = {"busy": False, "ready": False, "stage": "idle", "message": "等待启动", "error": None}


def progress(stage, message):
    with state_lock:
        job.update(stage=stage, message=message)


def start_build(count=15):
    with state_lock:
        if job["busy"]:
            raise HTTPException(409, "分析已在进行中")
        job.update(busy=True, ready=False, stage="starting", message="开始分析", error=None)

    def work():
        global comparison
        try:
            with operation_lock:
                engine.build(count, lambda s, m: progress(s, "small：" + m))
                if (large_engine.model_dir / large_engine.onnx_file).is_file():
                    large_engine.build(count, lambda s, m: progress(s, "large：" + m))
                    comparison = compare_engines(engine, large_engine)
                    import json
                    (engine.output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False))
                    progress("ready", "small / large 对照分析完成")
                else:
                    comparison = None
                    progress("ready", "small 已就绪；large 未下载，可运行 bash setup_large_lab.sh")
            with state_lock:
                job.update(ready=True, busy=False, error=None)
        except Exception as error:
            import logging
            logging.exception("Local analysis failed")
            with state_lock:
                job.update(busy=False, ready=False, stage="error", message="分析失败", error=str(error))
    threading.Thread(target=work, daemon=True, name="intent-analysis").start()


@asynccontextmanager
async def lifespan(app):
    if os.getenv("LAB_SKIP_AUTOBUILD") != "1":
        start_build()
    yield


app = FastAPI(title="Intent Router Lab", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def local_origin(request: Request, call_next):
    if request.method == "POST" and request.headers.get("origin"):
        origin = urlparse(request.headers["origin"])
        if origin.netloc != request.headers.get("host") or origin.scheme not in {"http", "https"}:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "拒绝跨站请求"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
    return response


def require_ready():
    with state_lock:
        if job["busy"] or not job["ready"]:
            raise HTTPException(409, job["error"] or job["message"])


def select_engine(model):
    if model not in {"small", "large"}:
        raise HTTPException(422, "模型必须为 small 或 large")
    chosen = engine if model == "small" else large_engine
    if not chosen.ready:
        raise HTTPException(409, f"{model} 尚未就绪")
    return chosen


class BuildRequest(BaseModel):
    anchors_per_intent: int = Field(default=15, ge=10, le=20)


class MessageRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=20)
    expected_mode: str | None = None


class RealDatasetRequest(BaseModel):
    dataset: str = Field(min_length=1, max_length=500000)


class FeedbackRequest(BaseModel):
    text: str = Field(min_length=1, max_length=256)
    mode: Literal["chat", "skill", "singing", "dancing", "focus"]
    split: Literal["train", "validation", "test"] = "train"
    source: Literal["manual", "asr"] = "manual"
    family: str = Field(default="", max_length=100)


def atomic_private_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def selected_engines():
    chosen = {"small": engine}
    if comparison is not None and large_engine.ready:
        chosen["large"] = large_engine
    return chosen


def current_real_payload():
    path = engine.root / "analysis_lab/private/real_dataset.json"
    ledger = path.with_name("test_registry.json")
    history = json.loads(ledger.read_text()) if ledger.is_file() else []
    payload = validate_real_dataset(json.loads(path.read_text()), engine.cases, engine.baseline, history) if path.is_file() else None
    return payload, history


def archive_json(path):
    if path.is_file():
        old = json.loads(path.read_text())
        archive = path.parent / "history" / (path.stem + "." + digest_dataset(old) + ".json")
        if not archive.is_file():
            atomic_private_json(archive, old)


def commit_fit(prepared, chosen, updates):
    """Commit both model heads + artifacts together; restore on any write failure."""
    global comparison
    saved = {key: (e.classifier, e.reports["classifier"], e.classifier_metadata, e.real_report,
                   e.feedback_report, e.feedback_cases, copy.deepcopy(e.queries), e.layouts.copy()) for key, e in chosen.items()}
    old_comparison = comparison
    old_files = {p: p.read_bytes() if p.is_file() else None for p in updates}
    try:
        for path, payload in updates.items():
            archive_json(path)
            atomic_private_json(path, payload)
        for key, e in chosen.items():
            e.apply_classifier(prepared[key])
        if "large" in chosen:
            comparison = compare_engines(engine, large_engine)
        for e in chosen.values():
            e._save()
        if comparison is not None:
            atomic_private_json(engine.output / "comparison.json", comparison)
    except Exception:
        comparison = old_comparison
        for key, e in chosen.items():
            (e.classifier, e.reports["classifier"], e.classifier_metadata, e.real_report,
             e.feedback_report, e.feedback_cases, e.queries, e.layouts) = saved[key]
        for path, content in old_files.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
                path.chmod(0o600)
        for e in chosen.values():
            e._save()
        if old_comparison is not None:
            atomic_private_json(engine.output / "comparison.json", old_comparison)
        raise


def correction_snapshot():
    draft, active = load_feedback(engine.root), load_feedback(engine.root, True)
    run_path = feedback_path(engine.root).with_name("feedback.training.json")
    run = json.loads(run_path.read_text()) if run_path.is_file() else None
    return {**feedback_summary(draft, active), "last_training": run,
            "by_model": {key: e.feedback_report for key, e in selected_engines().items()}}


@app.get("/api/status")
def status():
    with state_lock:
        return dict(job)


@app.post("/api/build")
def build(request: BuildRequest):
    start_build(request.anchors_per_intent)
    return {"started": True}


@app.get("/api/state")
def snapshot(model: str = "small"):
    require_ready()
    with operation_lock:
        return select_engine(model).snapshot()


@app.get("/api/comparison")
def comparison_snapshot():
    require_ready()
    with operation_lock:
        return comparison or {"ready": False, "message": "large 未就绪；先运行 bash setup_large_lab.sh"}


@app.post("/api/real-dataset")
def import_real_dataset(request: RealDatasetRequest):
    global comparison
    require_ready()
    try:
        raw = json.loads(request.dataset)
        with operation_lock:
            path = engine.root / "analysis_lab/private/real_dataset.json"
            registry_path = path.with_name("test_registry.json")
            history = json.loads(registry_path.read_text()) if registry_path.is_file() else []
            payload = validate_real_dataset(raw, engine.cases, engine.baseline, history)
            validate_feedback(load_feedback(engine.root), engine.cases, engine.baseline, payload["cases"], history)
            validate_feedback(load_feedback(engine.root, True), engine.cases, engine.baseline, payload["cases"], history)
            selected = selected_engines()
            # No state/files changed while either encoder or fit can still fail.
            prepared = {key: e.prepare_classifier(payload) for key, e in selected.items()}
            seen = bool(test_hashes(payload) & set(history))
            for result in prepared.values():
                result["real_report"]["previously_evaluated"] = seen
            commit_fit(prepared, selected, {path: raw, registry_path: sorted(set(history) | test_hashes(payload))})
            return {"imported": len(payload["cases"]), "previously_evaluated": seen,
                    "by_model": {key: e.real_report for key, e in selected.items()}}
    except (ValueError, TypeError) as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/feedback")
def corrections():
    require_ready()
    with operation_lock:
        return correction_snapshot()


@app.post("/api/feedback")
def save_correction(request: FeedbackRequest):
    require_ready()
    try:
        with operation_lock:
            payload, history = current_real_payload()
            cases, record, created = upsert_feedback(load_feedback(engine.root), request.text, request.mode,
                                                     request.split, request.source, request.family)
            validate_feedback(cases, engine.cases, engine.baseline, payload["cases"] if payload else [], history)
            path = feedback_path(engine.root)
            registry_path = path.with_name("test_registry.json")
            paths = {path: {"version": 1, "cases": cases},
                     registry_path: sorted(set(history) | test_hashes({"cases": cases}))}
            previous = {p: p.read_bytes() if p.is_file() else None for p in paths}
            try:
                archive_json(path)
                for p, value in paths.items():
                    atomic_private_json(p, value)
            except Exception:
                for p, content in previous.items():
                    if content is None: p.unlink(missing_ok=True)
                    else: p.write_bytes(content); p.chmod(0o600)
                raise
            # Explicit human feedback updates probe truth only, never its prediction.
            from lab_encoder import normalize_text
            for e in selected_engines().values():
                for q in e.queries:
                    if normalize_text(q["text"]) == normalize_text(record["text"]):
                        q["expected_mode"] = request.mode
                e.layouts.clear()
            return {"record": record, "created": created, "feedback": correction_snapshot()}
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/feedback/train")
def retrain_corrections():
    require_ready()
    try:
        with operation_lock:
            draft = load_feedback(engine.root)
            if not draft:
                raise ValueError("请先归档人工标注句子，再重新训练")
            payload, history = current_real_payload()
            validate_feedback(draft, engine.cases, engine.baseline, payload["cases"] if payload else [], history)
            chosen = selected_engines()
            # Fully prepare both candidates before any state or active file changes.
            prepared = {key: e.prepare_classifier(payload, draft) for key, e in chosen.items()}
            models = {}
            from lab_encoder import normalize_text
            test_cases = [c for c in draft if c["split"] == "test"]
            for key, e in chosen.items():
                before_feedback = None
                if test_cases:
                    vectors = e._encode([normalize_text(c["text"]) for c in test_cases])
                    before_feedback = e.evaluate_classifier(e.classifier, test_cases, vectors)["reports"]["classifier"]
                models[key] = {"synthetic_before": {p: e.reports["classifier"][p] for p in ("default", "calibrated")},
                               "synthetic_after": {p: prepared[key]["report"][p] for p in ("default", "calibrated")},
                               "feedback_before": before_feedback,
                               "feedback_after": prepared[key]["feedback_report"]["reports"].get("classifier"),
                               "real_asr_before": e.real_report["reports"].get("classifier"),
                               "real_asr_after": prepared[key]["real_report"]["reports"].get("classifier")}
                if payload:
                    prepared[key]["real_report"]["previously_evaluated"] = bool(test_hashes(payload) & set(history))
            run = {"trained_at": datetime.now(timezone.utc).isoformat(), "feedback_sha256": feedback_digest(draft),
                   "count": len(draft), "models": models,
                   "note": "同一留出集的训练前后回归对照，不是新盲测。会显示退步，拒识仍算未判对；没有真实 ASR 数据时不声称生产准确率。"}
            active_path = feedback_path(engine.root, True)
            commit_fit(prepared, chosen, {active_path: {"version": 1, "cases": draft},
                                         active_path.with_name("feedback.training.json"): run})
            return correction_snapshot()
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/layout")
def layout(scope: str = "all", perplexity: int = 30, seed: int = 42, model: str = "small"):
    require_ready()
    if scope not in ["all"] + MODES or perplexity not in (20, 30, 50) or seed not in (7, 42):
        raise HTTPException(422, "非法的图表范围或 t-SNE 参数")
    with operation_lock:
        return select_engine(model).layout(scope, perplexity, seed)


@app.post("/api/messages")
def messages(request: MessageRequest):
    require_ready()
    if request.expected_mode is not None and request.expected_mode not in MODES:
        raise HTTPException(422, "期望意图必须是五类之一，或留空")
    if any(not isinstance(t, str) or not t.strip() or len(t) > 256 for t in request.texts):
        raise HTTPException(422, "每条句子须为 1–256 字符；一次最多 20 条")
    try:
        with operation_lock:
            # Validate normalized input before either model appends a probe.
            from lab_encoder import normalize_text
            if any(not normalize_text(t) for t in request.texts):
                raise ValueError("请发送实际句子，不能只有空白或唤醒词")
            chosen = {"small": engine}
            if comparison is not None and large_engine.ready:
                chosen["large"] = large_engine
            # Keep both probe streams aligned even if inference fails midway.
            saved = {k: (list(e.queries), e.query_vectors.copy(), e.query_serial, e.layouts.copy()) for k, e in chosen.items()}
            try:
                results = {k: e.analyze(request.texts, request.expected_mode) for k, e in chosen.items()}
            except Exception:
                for k, e in chosen.items():
                    e.queries, e.query_vectors, e.query_serial, e.layouts = saved[k]
                raise
            return {**results["small"], "by_model": results}
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/download/{name}")
def download(name: str, model: str = "small"):
    require_ready()
    if name not in {"report.json", "anchors.candidate.yaml", "cases.generated.yaml", "comparison.json", "classifier.candidate.json", "real-dataset.template.json", "feedback.json", "feedback.training.json"}:
        raise HTTPException(404, "未知的分析文件")
    if name in {"feedback.json", "feedback.training.json"}:
        with operation_lock:
            body = correction_snapshot() if name == "feedback.json" else correction_snapshot()["last_training"]
            if body is None: raise HTTPException(409, "尚无纠错训练记录")
            return JSONResponse(body, headers={"Content-Disposition": f'attachment; filename="{name}"'})
    if name == "real-dataset.template.json":
        return JSONResponse(public_template(), headers={"Content-Disposition": 'attachment; filename="real-dataset.template.json"'})
    if name == "comparison.json":
        if comparison is None:
            raise HTTPException(409, "large 未就绪")
        with operation_lock:
            return JSONResponse({**comparison, "probes": {"small": engine.snapshot()["queries"], "large": large_engine.snapshot()["queries"]}},
                                headers={"Content-Disposition": 'attachment; filename="comparison.json"'})
    selected = select_engine(model)
    if name == "report.json":
        # Explicit download includes the current in-memory probes and labels.
        with operation_lock:
            return JSONResponse(selected.snapshot(), headers={"Content-Disposition": f'attachment; filename="{model}-report.json"'})
    return FileResponse(selected.output / name, filename=f"{model}-{name}")


@app.get("/vendor/plotly.min.js")
def plotly_bundle():
    return FileResponse(Path(plotly.__file__).parent / "package_data/plotly.min.js", media_type="application/javascript")


@app.get("/")
def index():
    return FileResponse(ROOT / "lab_web/index.html")


app.mount("/static", StaticFiles(directory=ROOT / "lab_web"), name="static")
