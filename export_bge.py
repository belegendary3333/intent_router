"""Export the original BGE model; optional CPU-specific INT8 quantization."""

import argparse
import os
import time
from pathlib import Path

from sentence_transformers import (
    SentenceTransformer,
    export_dynamic_quantized_onnx_model,
)

MODEL_ID = "BAAI/bge-small-zh-v1.5"
BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantization", choices=("none", "arm64", "avx2"), default="none")
    parser.add_argument("--model-dir", default=os.getenv("ROUTER_MODEL_DIR", str(BASE_DIR / "model")))
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"[INFO] downloading/exporting {MODEL_ID} to {model_dir}", flush=True)
    model = SentenceTransformer(
        MODEL_ID,
        backend="onnx",
        model_kwargs={"provider": "CPUExecutionProvider"},
    )
    model.save_pretrained(str(model_dir))
    onnx_file = "onnx/model.onnx"
    if args.quantization != "none":
        suffix = f"qint8_{args.quantization}"
        export_dynamic_quantized_onnx_model(
            model=model,
            quantization_config=args.quantization,
            model_name_or_path=str(model_dir),
            file_suffix=suffix,
        )
        onnx_file = f"onnx/model_{suffix}.onnx"
    if not (model_dir / onnx_file).is_file():
        raise RuntimeError(f"expected ONNX file missing: {model_dir / onnx_file}")
    smoke_model = SentenceTransformer(
        str(model_dir),
        backend="onnx",
        model_kwargs={"file_name": onnx_file, "provider": "CPUExecutionProvider", "export": False},
    )
    embeddings = smoke_model.encode(
        ["给我唱首歌", "帮我擦桌子"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if embeddings.shape != (2, 512):
        raise RuntimeError(f"unexpected BGE embedding shape: {embeddings.shape}")
    print(f"[OK] export and embedding smoke test passed, elapsed={time.perf_counter() - started:.2f}s")
    print(f"[INFO] set ROUTER_ONNX_FILE={onnx_file} in .env")


if __name__ == "__main__":
    main()
