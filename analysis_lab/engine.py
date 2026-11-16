import hashlib
import json
import threading
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import classification_report, confusion_matrix, f1_score, silhouette_samples
from threadpoolctl import threadpool_limits

from lab_encoder import BgeOnnxEncoder, normalize_text
from .corpus import LABELS, MODES, load_cases
from .classifier import FrozenIntentClassifier
from .real_data import validate_real_dataset, digest_dataset, test_hashes
from .feedback import load_feedback, validate_feedback, feedback_digest

SCHEMES = {"baseline": "原有 88 条", "expanded": "扩充样例", "anchors": "代表锚点", "classifier": "轻量分类器"}


def classify(similarities, examples, min_score=0.62, min_margin=0.03):
    """Exactly the deployed max-per-mode policy; no t-SNE coordinates used."""
    winners = {}
    for index, score in enumerate(similarities):
        mode = examples[index]["mode"]
        if mode not in winners or float(score) > winners[mode][0]:
            winners[mode] = (float(score), index)
    ranking = sorted(winners, key=lambda mode: winners[mode][0], reverse=True)
    best = ranking[0]
    score, index = winners[best]
    margin = score - winners[ranking[1]][0]
    accepted = score >= min_score and margin >= min_margin
    return {"mode": best if accepted else "chat", "matched_mode": best,
            "matched_example": examples[index]["text"], "score": round(score, 4),
            "margin": round(margin, 4), "accepted": accepted,
            "scores": {mode: round(winners[mode][0], 4) for mode in ranking}}


def metrics(expected, predictions):
    actual = [p["mode"] for p in predictions]
    accepted = [p["accepted"] for p in predictions]
    matrix_labels = MODES + (["clarify"] if "clarify" in actual else [])
    if not expected:
        return {"count": 0, "accuracy": None, "macro_f1": None, "rejected": 0,
                "unsafe_chat_activation": 0, "coverage": None, "rejection_rate": None,
                "accepted_accuracy": None, "action_recall": None, "matrix_labels": MODES,
                "per_class": {m: {"precision": 0, "recall": 0, "f1-score": 0, "support": 0} for m in MODES},
                "matrix": [[0] * len(MODES) for _ in MODES]}
    report = classification_report(expected, actual, labels=MODES, output_dict=True, zero_division=0)
    return {"count": len(expected), "accuracy": sum(a == b for a, b in zip(expected, actual)) / max(1, len(expected)),
            "macro_f1": f1_score(expected, actual, labels=MODES, average="macro", zero_division=0) if expected else 0,
            "rejected": sum(not p["accepted"] for p in predictions),
            "unsafe_chat_activation": sum(a == "chat" and b in MODES[1:] for a, b in zip(expected, actual)),
            "coverage": sum(accepted) / len(expected), "rejection_rate": 1 - sum(accepted) / len(expected),
            "accepted_accuracy": sum(a == b and ok for a, b, ok in zip(expected, actual, accepted)) / sum(accepted) if sum(accepted) else None,
            "action_recall": sum(a == b for a, b in zip(expected, actual) if a != "chat") / sum(a != "chat" for a in expected) if any(a != "chat" for a in expected) else None,
            "per_class": {mode: report[mode] for mode in MODES},
            "matrix_labels": matrix_labels,
            "matrix": confusion_matrix(expected, actual, labels=matrix_labels).tolist()}


class AnalysisEngine:
    def __init__(self, root: Path, model_dir: Path, onnx_file: str, model_key="small"):
        self.root, self.model_dir, self.onnx_file = root, model_dir, onnx_file
        if model_key not in {"small", "large"}:
            raise ValueError("模型必须为 small 或 large")
        self.model_key = model_key
        self.dimension = 512 if model_key == "small" else 1024
        self.model_name = f"BAAI/bge-{model_key}-zh-v1.5"
        self.output = root / "analysis_output" if model_key == "small" else root / "analysis_output/large"
        self.lock = threading.RLock()
        self.encoder = None
        self.ready = False
        self.queries = []
        self.query_vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.layouts = OrderedDict()
        self.query_serial = 0
        self.feedback_cases = []

    def _encode(self, texts, progress=None):
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        chunks = []
        # Dynamic INT8 scales can depend on batch composition/padding. A query
        # must be encoded alone, just like POST /route. Held-out sentences never
        # share a quantization batch with training sentences.
        for start, text in enumerate(texts):
            chunks.append(self.encoder.encode([text]))
            if progress and (start % 16 == 0 or start == len(texts) - 1):
                progress("embedding", f"生成 BGE 向量 {start + 1}/{len(texts)}")
        return np.concatenate(chunks)

    def build(self, anchors_per_intent=15, progress=lambda *_: None):
        with self.lock, threadpool_limits(limits=4):
            progress("corpus", "校验案例、表达家族和集合隔离")
            cases = load_cases(self.root / "analysis_lab/cases.yaml")
            original = yaml.safe_load((self.root / "routes.yaml").read_text(encoding="utf-8"))
            missing_modes = set(MODES) - set(original or {})
            if missing_modes:
                raise ValueError(
                    "routes.yaml 缺少分析台所需意图："
                    + "、".join(sorted(missing_modes))
                )
            baseline = [{"id": f"baseline:{mode}:{i}", "text": text, "mode": mode,
                         "split": "baseline", "subtype": "原有样例"}
                        for mode in MODES for i, text in enumerate(original[mode])]
            baseline_texts = {normalize_text(c["text"]) for c in baseline}
            overlaps = [c for c in cases if c["split"] != "train" and normalize_text(c["text"]) in baseline_texts]
            if overlaps:
                raise ValueError(f"验证/测试案例与原有样例完全重复：{overlaps[0]['text']}")
            progress("model", f"加载 {self.model_name} INT8 ONNX")
            if self.encoder is None:
                self.encoder = BgeOnnxEncoder(self.model_dir, self.onnx_file, self.dimension)
            digest = hashlib.sha256(json.dumps(cases + baseline, ensure_ascii=False, sort_keys=True).encode())
            for path in (self.model_dir / self.onnx_file, self.model_dir / (self.onnx_file + "_data"), self.model_dir / "tokenizer.json"):
                if path.exists():
                    file_digest = hashlib.sha256()
                    with path.open("rb") as source:
                        while chunk := source.read(1024 * 1024):
                            file_digest.update(chunk)
                    digest.update(file_digest.digest())
            # Invalidate across inference-library changes and encoder implementations.
            import onnxruntime, tokenizers
            digest.update(f"cls-l2-single-query-full-bank-v2:{onnxruntime.__version__}:{tokenizers.__version__}".encode())
            digest.update((self.root / "lab_encoder.py").read_bytes())
            digest.update(self.model_name.encode())
            fingerprint = digest.hexdigest()
            self.output.mkdir(parents=True, exist_ok=True)
            cache = self.output / "embeddings.npz"
            vectors = None
            if cache.exists():
                with np.load(cache, allow_pickle=False) as stored:
                    if str(stored["fingerprint"]) == fingerprint:
                        vectors = stored["vectors"]
            if vectors is None:
                vectors = np.concatenate([
                    self._encode([normalize_text(c["text"]) for c in cases], progress),
                    self.encoder.encode([c["text"] for c in baseline]),
                ])
                np.savez_compressed(cache, vectors=vectors, fingerprint=np.array(fingerprint))
            if vectors.shape != (len(cases) + len(baseline), self.dimension) or not np.isfinite(vectors).all():
                raise ValueError("向量缓存形状或数值非法；移走 analysis_output/embeddings.npz 后重试")
            self.cases, self.vectors, self.baseline = cases, vectors[:len(cases)], baseline
            self.baseline_vectors = vectors[len(cases):]
            self.indices = {s: [i for i, c in enumerate(cases) if c["split"] == s]
                            for s in ("train", "validation", "test")}
            self.settings = {"anchors_per_intent": anchors_per_intent, "min_score": 0.62, "min_margin": 0.03}
            progress("anchors", f"在原始 {self.dimension} 维空间聚类并选代表句；不使用测试集")
            self.anchors = []
            for mode in MODES:
                ids = [i for i in self.indices["train"] if cases[i]["mode"] == mode]
                k = min(anchors_per_intent, len(ids))
                km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(self.vectors[ids])
                for cluster in range(k):
                    members = [ids[j] for j in np.flatnonzero(km.labels_ == cluster)]
                    # A cosine medoid is an actual utterance, not an invented centroid.
                    scores = (self.vectors[members] @ self.vectors[members].T).mean(axis=1)
                    chosen = members[int(np.argmax(scores))]
                    anchor = dict(cases[chosen], index=chosen, cluster=cluster,
                                  member_count=len(members), cohesion=round(float(scores.max()), 4),
                                  member_ids=[cases[i]["id"] for i in members])
                    other = [i for i in self.indices["train"] if cases[i]["mode"] != mode]
                    similarity = self.vectors[other] @ self.vectors[chosen]
                    nearest = other[int(np.argmax(similarity))]
                    anchor["nearest_other"] = dict(cases[nearest], similarity=round(float(similarity.max()), 4))
                    self.anchors.append(anchor)
            expanded = [cases[i] for i in self.indices["train"]]
            # Reference banks are encoded as one batch, as the deployed router
            # does at startup. This makes reports reproduce that bank policy.
            self.banks = {"baseline": (baseline, self.baseline_vectors),
                          "expanded": (expanded, self.encoder.encode([c["text"] for c in expanded])),
                          "anchors": (self.anchors, self.encoder.encode([c["text"] for c in self.anchors]))}
            progress("reports", "仅用验证集调阈值，再比较独立测试集表现")
            self.reports = self._reports()
            self.fingerprint = fingerprint
            progress("classifier", "冻结 BGE，用训练集拟合分类头；验证集选 C、温度及各类拒识阈值")
            data_path = self.root / "analysis_lab/private/real_dataset.json"
            payload = None
            if data_path.is_file():
                history_path = data_path.with_name("test_registry.json")
                history = json.loads(history_path.read_text()) if history_path.is_file() else []
                payload = validate_real_dataset(json.loads(data_path.read_text()), cases, baseline, history)
            history_path = self.root / "analysis_lab/private/test_registry.json"
            history = json.loads(history_path.read_text()) if history_path.is_file() else []
            self.feedback_cases = validate_feedback(load_feedback(self.root, True), cases, baseline,
                                                    payload["cases"] if payload else [], history)
            self.apply_classifier(self.prepare_classifier(payload))
            if payload:
                self.real_report["previously_evaluated"] = bool(test_hashes(payload) & set(history))
            self.confusions = self._confusions()
            self.quality = self._quality()
            self.clustering = self._clustering()
            progress("benchmark", "预热后测量相同 12 条短句的单句推理耗时")
            self.benchmark = self._benchmark()
            # A rebuild can change anchor identities. Never leave earlier probes
            # displaying results from the old bank; they remain excluded from fit.
            if self.queries:
                self.query_vectors = self._encode([normalize_text(q["text"]) for q in self.queries])
                for query, vector in zip(self.queries, self.query_vectors):
                    query.update(self._explain(vector))
                    query["mode"] = query["results"]["expanded"]["mode"]
            self.fingerprint = fingerprint
            self.built_at = datetime.now(timezone.utc).isoformat()
            self.layouts.clear()
            self.ready = True
            progress("tsne", "生成五类联合 t-SNE；新句会作为探针加入图中")
            self.layout("all", 30, 42)
            self._save()
            progress("ready", "分析完成")

    def _clustering(self):
        labels = [c["mode"] for c in self.cases]
        silhouette = silhouette_samples(self.vectors, labels, metric="cosine")
        per_mode = {}
        for mode in MODES:
            ids = [i for i, c in enumerate(self.cases) if c["mode"] == mode]
            n = len(ids)
            similarity = self.vectors[ids] @ self.vectors[ids].T
            per_mode[mode] = {
                "count": n, "silhouette": round(float(silhouette[ids].mean()), 4),
                "pairwise_cosine": round(float((similarity.sum() - np.trace(similarity)) / (n * (n - 1))), 4),
                "train_clusters": sum(a["mode"] == mode for a in self.anchors),
            }
        return {"metric": "cosine", "space": f"original-{self.dimension}D", "silhouette": round(float(silhouette.mean()), 4),
                "per_mode": per_mode,
                "note": "轮廓系数按五类人工标签在原始向量空间计算，不是 t-SNE 分数。聊天允许多子簇；类内余弦值受模型分数分布影响，不能单独判优。"}

    def prepare_classifier(self, payload=None, feedback_cases=None):
        """Pure preparation: paired imports commit only after both fits succeed."""
        with self.lock, threadpool_limits(limits=4):
            real_cases = payload["cases"] if payload else []
            feedback_cases = self.feedback_cases if feedback_cases is None else feedback_cases
            feedback_vectors = self._encode([normalize_text(c["text"]) for c in feedback_cases])
            feedback_indices = {s: [i for i, c in enumerate(feedback_cases) if c["split"] == s] for s in ("train", "validation", "test")}
            real_vectors = self._encode([normalize_text(c["text"]) for c in real_cases])
            real_indices = {s: [i for i, c in enumerate(real_cases) if c["split"] == s] for s in ("train", "validation", "test")}
            train = [self.cases[i] for i in self.indices["train"]] + [real_cases[i] for i in real_indices["train"]]
            train += [feedback_cases[i] for i in feedback_indices["train"]]
            train_vectors = np.concatenate([self.vectors[self.indices["train"]], real_vectors[real_indices["train"]], feedback_vectors[feedback_indices["train"]]])
            val = [real_cases[i] for i in real_indices["validation"]] or [self.cases[i] for i in self.indices["validation"]]
            val_vectors = real_vectors[real_indices["validation"]] if real_indices["validation"] else self.vectors[self.indices["validation"]]
            val += [feedback_cases[i] for i in feedback_indices["validation"]]
            val_vectors = np.concatenate([val_vectors, feedback_vectors[feedback_indices["validation"]]])
            head = FrozenIntentClassifier().fit(train_vectors, train, val_vectors, val, self.fingerprint)
            ids = self.indices["test"]
            raw, gated = head.predict(self.vectors[ids], False), head.predict(self.vectors[ids], True)
            y = [self.cases[i]["mode"] for i in ids]
            report = {"name": SCHEMES["classifier"], "bank_count": len(train),
                      "default": metrics(y, raw), "calibrated": metrics(y, gated),
                      "candidate_thresholds": head.thresholds, "validation": metrics([c["mode"] for c in val], head.predict(val_vectors)),
                      "cases": [dict(self.cases[i], default=raw[j], calibrated=gated[j]) for j, i in enumerate(ids)]}
            real_report = {"ready": bool(real_cases), "dataset_id": payload["dataset_id"] if payload else None,
                           "sha256": digest_dataset(payload) if payload else None,
                           "counts": {s: dict(Counter(c["mode"] for c in real_cases if c["split"] == s)) for s in real_indices},
                           "test_count": len(real_indices["test"]), "reports": {}, "near_duplicates": [],
                           "note": "真实标签由上传者人工确认，程序不验证来源。真实测试句不参与任何拟合、阈值选择或 t-SNE。反复查看同一批不算新测试；请另采独立批次。训练/验证各类不足 5 条会拒绝导入，测试缺类时不能作为完整验收。"}
            if real_indices["test"]:
                test_cases = [real_cases[i] for i in real_indices["test"]]
                test_vectors = real_vectors[real_indices["test"]]
                expected = [c["mode"] for c in test_cases]
                for scheme, old in self.reports.items():
                    if scheme == "classifier":
                        continue
                    threshold = old["candidate_thresholds"]
                    raw = self._predictions(scheme, test_vectors)
                    gated = self._predictions(scheme, test_vectors, threshold["min_score"], threshold["min_margin"])
                    real_report["reports"][scheme] = {"default": metrics(expected, raw), "calibrated": metrics(expected, gated),
                        "cases": [dict(c, default=a, calibrated=b) for c, a, b in zip(test_cases, raw, gated)]}
                raw, gated = head.predict(test_vectors, False), head.predict(test_vectors)
                real_report["reports"]["classifier"] = {"default": metrics(expected, raw), "calibrated": metrics(expected, gated),
                    "cases": [dict(c, default=a, calibrated=b) for c, a, b in zip(test_cases, raw, gated)]}
                for c, vector in zip(test_cases, test_vectors):
                    sims = train_vectors @ vector
                    nearest = int(np.argmax(sims))
                    if sims[nearest] >= .97:
                        real_report["near_duplicates"].append({"text": c["text"], "nearest_text": train[nearest]["text"], "similarity": round(float(sims[nearest]), 4)})
            artifact = head.artifact()
            metadata = {k: v for k, v in artifact.items() if k not in {"coef", "intercept"}}
            feedback_report = {"ready": bool(feedback_cases), "sha256": feedback_digest(feedback_cases),
                               "counts": {s: dict(Counter(c["mode"] for c in feedback_cases if c["split"] == s)) for s in feedback_indices},
                               "sources": dict(Counter(c["source"] for c in feedback_cases)),
                               "test_count": len(feedback_indices["test"]), "reports": {}, "near_duplicates": [],
                               "note": "人工纠错回归留出；探针预测已经看过，不是新盲测。只有 train 拟合系数、validation 选参数，test 仅评测。手动输入不代表真实 ASR。"}
            if feedback_indices["test"]:
                feedback_report.update(self.evaluate_classifier(head,
                    [feedback_cases[i] for i in feedback_indices["test"]], feedback_vectors[feedback_indices["test"]], train, train_vectors))
            validation_source = "real_asr" if real_indices["validation"] else "synthetic"
            if feedback_indices["validation"]:
                validation_source += "+feedback"
            metadata.update(train_count=len(train), validation_count=len(val),
                            real_train_count=len(real_indices["train"]) + sum(feedback_cases[i]["source"] == "asr" for i in feedback_indices["train"]),
                            feedback_train_count=len(feedback_indices["train"]), feedback_validation_count=len(feedback_indices["validation"]),
                            feedback_sha256=feedback_digest(feedback_cases), fitted_at=datetime.now(timezone.utc).isoformat(),
                            validation_source=validation_source)
            return {"head": head, "report": report, "metadata": metadata, "real_report": real_report,
                    "feedback_report": feedback_report, "feedback_cases": feedback_cases}

    def evaluate_classifier(self, head, test_cases, test_vectors, train=(), train_vectors=None):
        expected = [c["mode"] for c in test_cases]
        output = {"reports": {}, "near_duplicates": []}
        for scheme, old in self.reports.items():
            if scheme == "classifier":
                continue
            t = old["candidate_thresholds"]
            raw = self._predictions(scheme, test_vectors)
            gated = self._predictions(scheme, test_vectors, t["min_score"], t["min_margin"])
            output["reports"][scheme] = {"default": metrics(expected, raw), "calibrated": metrics(expected, gated),
                "cases": [dict(c, default=a, calibrated=b) for c, a, b in zip(test_cases, raw, gated)]}
        raw, gated = head.predict(test_vectors, False), head.predict(test_vectors)
        output["reports"]["classifier"] = {"default": metrics(expected, raw), "calibrated": metrics(expected, gated),
            "cases": [dict(c, default=a, calibrated=b) for c, a, b in zip(test_cases, raw, gated)]}
        if train and train_vectors is not None:
            for c, vector in zip(test_cases, test_vectors):
                sims = train_vectors @ vector
                nearest = int(np.argmax(sims))
                if sims[nearest] >= .97:
                    output["near_duplicates"].append({"text": c["text"], "nearest_text": train[nearest]["text"], "similarity": round(float(sims[nearest]), 4)})
        return output

    def apply_classifier(self, prepared):
        self.classifier = prepared["head"]
        self.reports["classifier"] = prepared["report"]
        self.classifier_metadata, self.real_report = prepared["metadata"], prepared["real_report"]
        self.feedback_report, self.feedback_cases = prepared["feedback_report"], prepared["feedback_cases"]
        for query, vector in zip(self.queries, self.query_vectors):
            query["results"]["classifier"] = self._classifier_explanation(vector)
        self.layouts.clear()

    def _classifier_explanation(self, vector):
        return dict(self.classifier.predict(vector[None])[0], raw=self.classifier.predict(vector[None], False)[0])

    def _benchmark(self):
        ids = [i for mode in MODES for i in [j for j in self.indices["test"] if self.cases[j]["mode"] == mode][:3]]
        texts = [normalize_text(self.cases[i]["text"]) for i in ids]
        for text in texts[:3]:
            self.encoder.encode([text])
        durations = []
        for _ in range(2):
            for text in texts:
                started = time.perf_counter()
                self.encoder.encode([text])
                durations.append((time.perf_counter() - started) * 1000)
        return {"ids": [self.cases[i]["id"] for i in ids], "samples": len(durations), "warmup": 3,
                "p50_ms": round(float(np.percentile(durations, 50)), 2),
                "p95_ms": round(float(np.percentile(durations, 95)), 2),
                "mean_ms": round(float(np.mean(durations)), 2),
                "note": "同机 CPU、4 线程、单句 tokenizer+ONNX+归一化，不含路由、t-SNE 或网络；短小样本基准，不是生产延迟承诺。"}

    def _predictions(self, scheme, vectors, score=0.62, margin=0.03):
        bank, bank_vectors = self.banks[scheme]
        return [classify(row, bank, score, margin) for row in vectors @ bank_vectors.T]

    def _reports(self):
        val, test = self.indices["validation"], self.indices["test"]
        vy, ty = [self.cases[i]["mode"] for i in val], [self.cases[i]["mode"] for i in test]
        output = {}
        for scheme, (bank, bank_vectors) in self.banks.items():
            vsim = self.vectors[val] @ bank_vectors.T
            best = None
            for score in (0.5, 0.55, 0.6, 0.62, 0.65, 0.7, 0.75, 0.8):
                for margin in (0.0, 0.02, 0.03, 0.05, 0.08, 0.12):
                    preds = [classify(row, bank, score, margin) for row in vsim]
                    stat = metrics(vy, preds)
                    rank = (stat["macro_f1"], -stat["unsafe_chat_activation"], stat["accuracy"], -abs(score - 0.62) - abs(margin - 0.03))
                    if best is None or rank > best[0]:
                        best = (rank, score, margin, stat)
            _, score, margin, validation = best
            default = self._predictions(scheme, self.vectors[test])
            candidate = self._predictions(scheme, self.vectors[test], score, margin)
            output[scheme] = {"name": SCHEMES[scheme].replace("88", str(len(self.baseline))), "bank_count": len(bank),
                              "default": metrics(ty, default), "calibrated": metrics(ty, candidate),
                              "candidate_thresholds": {"min_score": score, "min_margin": margin},
                              "validation": validation,
                              "cases": [dict(self.cases[i], default=default[j], calibrated=candidate[j]) for j, i in enumerate(test)]}
        return output

    def _confusions(self):
        # Leave-one-out within train; validation/test use only the training bank.
        train = self.indices["train"]
        similarities = self.vectors @ self.vectors[train].T
        rows = []
        for i, case in enumerate(self.cases):
            own = np.array([self.cases[j]["mode"] == case["mode"] and j != i for j in train])
            other = np.array([self.cases[j]["mode"] != case["mode"] for j in train])
            row = similarities[i]
            oi = int(np.argmax(np.where(own, row, -np.inf)))
            xi = int(np.argmax(np.where(other, row, -np.inf)))
            gap = float(row[oi] - row[xi])
            if gap < 0.08:
                rows.append(dict(case, own_score=round(float(row[oi]), 4), other_score=round(float(row[xi]), 4),
                                 gap=round(gap, 4), own_example=self.cases[train[oi]]["text"],
                                 other_example=self.cases[train[xi]]["text"], other_mode=self.cases[train[xi]]["mode"]))
        return sorted(rows, key=lambda r: r["gap"])

    def _quality(self):
        # Flag rather than conceal near duplicates; generated cases are not real logs.
        train = self.indices["train"]
        leaks = []
        comparison = np.concatenate([self.vectors[train], self.baseline_vectors])
        references = [self.cases[i] for i in train] + self.baseline
        for i in self.indices["validation"] + self.indices["test"]:
            sim = comparison @ self.vectors[i]
            j = int(np.argmax(sim))
            if sim[j] >= 0.97:
                leaks.append(dict(self.cases[i], nearest_text=references[j]["text"], similarity=round(float(sim[j]), 4)))
        return {"synthetic": True, "exact_overlap": 0, "family_overlap": 0,
                "near_duplicate_threshold": 0.97, "near_duplicates": leaks,
                "note": "750 条为人工编写的合成句，不是线上真实对话。家族隔离不等于不存在语义近重复；报告仅用于离线比较，不代表线上准确率。"}

    def layout(self, scope="all", perplexity=30, seed=42):
        with self.lock, threadpool_limits(limits=4):
            if not self.ready:
                raise RuntimeError("分析尚未完成")
            key = (scope, perplexity, seed, self.query_serial)
            if key in self.layouts:
                self.layouts.move_to_end(key)
                return self.layouts[key]
            ids = [i for i, c in enumerate(self.cases) if scope == "all" or c["mode"] == scope]
            samples = [dict(self.cases[i], kind="case", is_anchor=self.cases[i]["id"] in {a["id"] for a in self.anchors}) for i in ids]
            samples += [dict(q, kind="probe", is_anchor=False) for q in self.queries]
            vectors = np.concatenate([self.vectors[ids], self.query_vectors])
            started = time.perf_counter()
            dims = min(50, len(vectors) - 1, vectors.shape[1])
            reduced = PCA(n_components=dims, random_state=seed).fit_transform(vectors)
            coordinates = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                               random_state=seed, max_iter=1000, metric="euclidean").fit_transform(reduced)
            for point, xy in zip(samples, coordinates):
                point["x"], point["y"] = round(float(xy[0]), 5), round(float(xy[1]), 5)
            result = {"points": samples, "scope": scope, "perplexity": perplexity, "seed": seed,
                      "method": f"BGE {self.dimension}D → PCA {dims}D → t-SNE 2D", "model_key": self.model_key,
                      "dimension": self.dimension, "query_revision": self.query_serial,
                      "trustworthiness": round(float(trustworthiness(vectors, coordinates, n_neighbors=10)), 4),
                      "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                      "note": "探针参加真实 t-SNE 联合重算，不是近邻插值。换参数或加入新句会改变布局；二维距离不用于分类或选锚点。"}
            self.layouts[key] = result
            while len(self.layouts) > 12:
                self.layouts.popitem(last=False)
            return result

    def analyze(self, texts, expected_mode=None):
        with self.lock:
            started = time.perf_counter()
            cleaned = [normalize_text(t) for t in texts]
            if any(not t for t in cleaned):
                raise ValueError("请发送实际句子，不能只有空白或唤醒词")
            vectors = self._encode(cleaned)
            added = []
            for raw, clean, vector in zip(texts, cleaned, vectors):
                self.query_serial += 1
                explanation = self._explain(vector)
                probe = {"id": f"probe:{self.query_serial}", "text": raw, "normalized_text": clean,
                         "mode": explanation["results"]["expanded"]["mode"], "split": "probe", "subtype": "手动探针",
                         "expected_mode": expected_mode, **explanation}
                added.append(probe)
                self.queries.append(probe)
                self.query_vectors = np.concatenate([self.query_vectors, vector[None]])
            self.queries = self.queries[-100:]
            self.query_vectors = self.query_vectors[-100:]
            self.layouts.clear()
            return {"added": added, "queries": self.queries, "probe_report": self.probe_report(),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}

    def _explain(self, vector):
        results = {scheme: classify(bank_vectors @ vector, bank) for scheme, (bank, bank_vectors) in self.banks.items()}
        results["classifier"] = self._classifier_explanation(vector)
        anchor_sims = self.banks["anchors"][1] @ vector
        nearest_anchors = [dict(self.anchors[j], similarity=round(float(anchor_sims[j]), 4))
                           for j in np.argsort(-anchor_sims)[:5]]
        sample_sims = self.vectors[self.indices["train"]] @ vector
        nearest_cases = [dict(self.cases[self.indices["train"][j]], similarity=round(float(sample_sims[j]), 4))
                         for j in np.argsort(-sample_sims)[:5]]
        return {"results": results, "nearest_anchors": nearest_anchors, "nearest_cases": nearest_cases}

    def probe_report(self):
        labeled = [q for q in self.queries if q["expected_mode"]]
        return {"total": len(self.queries), "labeled": len(labeled),
                "schemes": {s: metrics([q["expected_mode"] for q in labeled], [q["results"][s] for q in labeled]) for s in SCHEMES},
                "note": "仅统计手工提供期望意图的探针；未标注的新句没有准确率。探针不参与选锚点、调阈值或独立测试集报告。"}

    def snapshot(self):
        with self.lock:
            return {"labels": LABELS, "schemes": SCHEMES, "cases": self.cases, "anchors": self.anchors,
                    "reports": self.reports, "confusions": self.confusions, "quality": self.quality,
                    "settings": self.settings, "built_at": self.built_at, "fingerprint": self.fingerprint,
                    "counts": {m: dict(Counter(c["split"] for c in self.cases if c["mode"] == m)) for m in MODES},
                    "model": self.model_name, "model_key": self.model_key,
                    "backend": "onnx-int8-direct", "dimension": self.dimension,
                    "conversion": "onnx-community/bge-small-zh-v1.5-ONNX" if self.model_key == "small" else "Xenova/bge-large-zh-v1.5",
                    "clustering": self.clustering, "benchmark": self.benchmark,
                    "classifier": self.classifier_metadata, "real_report": self.real_report, "feedback_report": self.feedback_report,
                    "queries": self.queries, "probe_report": self.probe_report()}

    def _save(self):
        # Local analysis artifacts only; production routes.yaml is never written.
        snapshot = self.snapshot()
        self.output.chmod(0o700)
        snapshot.pop("queries", None)
        (self.output / "classifier.candidate.json").write_text(json.dumps(self.classifier.artifact(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (self.output / "report.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        routes = {mode: [a["text"] for a in self.anchors if a["mode"] == mode] for mode in MODES}
        (self.output / "anchors.candidate.yaml").write_text(yaml.safe_dump(routes, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (self.output / "cases.generated.yaml").write_text(yaml.safe_dump(self.cases, allow_unicode=True, sort_keys=False), encoding="utf-8")
        for name in ("report.json", "classifier.candidate.json", "anchors.candidate.yaml", "cases.generated.yaml", "embeddings.npz"):
            (self.output / name).chmod(0o600)
