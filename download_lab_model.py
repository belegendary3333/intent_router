"""Download the same public ONNX files used in Coder; verify known SHA256s."""

import hashlib
import os
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = {
    "tokenizer.json": "3d09c84ebd10306706a79a8276b3ab736a40d8ec03251c7639f4e52c3a1a4f8e",
    "onnx/model_quantized.onnx": "99a6e522710c00220c89f8c52e0cc5aa09d4cbb1c34c0e932eab3a9dfdc65df3",
    "onnx/model_quantized.onnx_data": "952623481ca8beea884e3d3c9ecaf8a3c7bf1d0c21de29e970cd31af9d37a90b",
}


def main():
    model = Path(os.getenv("LAB_MODEL_DIR", str(ROOT / "model")))
    for relative, expected in FILES.items():
        target = model / relative
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise SystemExit(f"[ERROR] 现有文件不是已核验的 Coder 模型：{target}。不会覆盖；请另设 LAB_MODEL_DIR。")
            print(f"[OK] verified {relative}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".download-part")
        query = urllib.parse.urlencode({"Revision": "master", "FilePath": relative})
        url = "https://modelscope.cn/api/v1/models/onnx-community/bge-small-zh-v1.5-ONNX/repo?" + query
        digest = hashlib.sha256()
        print(f"[INFO] downloading {relative}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise SystemExit(f"[ERROR] SHA256 不符，未启用：{part}")
        part.replace(target)
        print(f"[OK] verified {relative}")


if __name__ == "__main__":
    main()
