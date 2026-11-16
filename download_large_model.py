"""Pinned public BAAI large ONNX conversion, separate from the small model."""
import argparse
import hashlib
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
import certifi

ROOT = Path(__file__).resolve().parent
REPO = "Xenova/bge-large-zh-v1.5"
REVISION = "88c72f156f26f7ae08bd4c8bd1b5ffaf668240c5"
FILES = {
    "tokenizer.json": "7dfbf1966ebf99d471c3796e9b457329d2b2182b817e144f1e904b957745c839",
    "config.json": "b8a4dce1dfa153b714eb25c75b18238ef2b12e4755f998457f60cd872483be66",
    "onnx/model_quantized.onnx": "49277b15e898b5cd50f87f13f677609a9cdde71d24fa8817477ca50076bf13ef",
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT / "model/large")
    args = parser.parse_args()
    for relative, expected in FILES.items():
        target = args.directory / relative
        if target.exists():
            if sha256(target) != expected:
                raise SystemExit(f"[ERROR] SHA256 不符，不覆盖现有文件：{target}")
            print(f"[OK] verified {relative}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".download-part")
        query = urllib.parse.urlencode({"Revision": REVISION, "FilePath": relative})
        url = f"https://modelscope.cn/api/v1/models/{REPO}/repo?{query}"
        print(f"[INFO] downloading {relative}", flush=True)
        digest, received, last = hashlib.sha256(), 0, time.monotonic()
        with urllib.request.urlopen(url, timeout=120, context=ssl.create_default_context(cafile=certifi.where())) as response, part.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if time.monotonic() - last > 10:
                    print(f"[INFO] {relative}: {received / 1e6:.1f} MB", flush=True)
                    last = time.monotonic()
        if digest.hexdigest() != expected:
            raise SystemExit(f"[ERROR] 下载校验失败，未启用：{part}")
        part.replace(target)
        print(f"[OK] verified {relative}: {received / 1e6:.1f} MB", flush=True)
    manifest = {"base_model": "BAAI/bge-large-zh-v1.5", "conversion": REPO,
                "mirror_revision": REVISION, "precision": "dynamic-int8", "sha256": FILES}
    (args.directory / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")

if __name__ == "__main__":
    main()
