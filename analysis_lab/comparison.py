"""Paired reports, never compare independent t-SNE coordinates as distances."""
import hashlib
import json
from sklearn.metrics import adjusted_rand_score
from .corpus import MODES
from .engine import SCHEMES

def compare_engines(small, large):
    left, right = small.snapshot(), large.snapshot()
    if left["cases"] != right["cases"] or small.baseline != large.baseline:
        raise ValueError("small / large 使用的案例或设定词不同；拒绝生成对照报告")
    if small.settings != large.settings or small.benchmark["ids"] != large.benchmark["ids"]:
        raise ValueError("small / large 聚类设置或耗时测试句不同")
    modes = {}
    for mode in MODES:
        ids = [c["id"] for c in left["cases"] if c["mode"] == mode and c["split"] == "train"]
        assignments, chosen = [], []
        for snapshot in (left, right):
            anchors = [a for a in snapshot["anchors"] if a["mode"] == mode]
            assignments.append({member: a["cluster"] for a in anchors for member in a["member_ids"]})
            chosen.append({a["id"] for a in anchors})
        modes[mode] = {
            "anchor_overlap": len(chosen[0] & chosen[1]), "anchors_each": len(chosen[0]),
            "adjusted_rand_index": round(float(adjusted_rand_score(
                [assignments[0][i] for i in ids], [assignments[1][i] for i in ids])), 4),
        }
    differences = {}
    for scheme in SCHEMES:
        differences[scheme] = {}
        lc, rc = left["reports"][scheme]["cases"], right["reports"][scheme]["cases"]
        if [c["id"] for c in lc] != [c["id"] for c in rc]:
            raise ValueError("两模型测试集不一致")
        for policy in ("default", "calibrated"):
            differences[scheme][policy] = [
                {"id": a["id"], "text": a["text"], "expected_mode": a["mode"],
                 "small": a[policy], "large": b[policy],
                 "change": "large_corrected" if b[policy]["mode"] == a["mode"] else
                           "large_regressed" if a[policy]["mode"] == a["mode"] else "both_wrong"}
                for a, b in zip(lc, rc) if a[policy]["mode"] != b[policy]["mode"]
            ]
    output = {
        "ready": True, "same_cases": True, "same_routes": True, "settings": left["settings"],
        "case_sha256": hashlib.sha256(json.dumps(left["cases"], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "routes_sha256": hashlib.sha256((small.root / "routes.yaml").read_bytes()).hexdigest(),
        "routes": {mode: [c["text"] for c in small.baseline if c["mode"] == mode] for mode in MODES},
        "counts": left["counts"], "cluster_agreement": modes, "test_differences": differences,
        "models": {},
        "note": "相同案例/设定词/数据划分/KMeans 参数/单句编码策略；两个现成 ONNX 转换的量化细节不同，比较的是实际 INT8 部署版本，不是纯 FP32 能力排名。两图分别拟合 t-SNE，坐标不可相减；ARI 表示聚类一致程度，不表示谁更好。",
    }
    for key, snapshot in (("small", left), ("large", right)):
        output["models"][key] = {k: snapshot[k] for k in (
            "model", "model_key", "dimension", "backend", "conversion", "fingerprint", "built_at", "clustering", "benchmark")}
        output["models"][key]["reports"] = {
            scheme: {k: v for k, v in report.items() if k != "cases"}
            for scheme, report in snapshot["reports"].items()
        }
        output["models"][key]["classifier"] = snapshot["classifier"]
        output["models"][key]["real_report"] = snapshot["real_report"]
        output["models"][key]["feedback_report"] = snapshot["feedback_report"]
    return output
