"""Same CLS pooling and normalization as standalone_app, without route startup."""

import os
import re
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def normalize_text(text: str) -> str:
    return re.sub(r"^\s*小诺[\s，,。！!：:]*", "", text.strip()).strip()


class BgeOnnxEncoder:
    def __init__(self, model_dir: Path, onnx_file: str, dimension: int = 512):
        model_path = model_dir / onnx_file
        tokenizer_path = model_dir / "tokenizer.json"
        if not model_path.is_file() or not tokenizer_path.is_file():
            raise RuntimeError(f"BGE 模型或 tokenizer 缺失：{model_dir}；请先运行 bash setup_lab.sh")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("OMP_NUM_THREADS", "4"))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.dimension = dimension
        output_dimension = self.session.get_outputs()[0].shape[-1]
        if isinstance(output_dimension, int) and output_dimension != dimension:
            raise RuntimeError(f"模型输出 {output_dimension} 维，与配置 {dimension} 维不符")
        self.lock = threading.Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        with self.lock:
            encoded = self.tokenizer.encode_batch(texts)
            feed = {}
            fields = {"input_ids": "ids", "attention_mask": "attention_mask", "token_type_ids": "type_ids"}
            for name, field in fields.items():
                if name in self.input_names:
                    feed[name] = np.asarray([getattr(item, field) for item in encoded], dtype=np.int64)
            missing = self.input_names - set(feed)
            if missing:
                raise RuntimeError(f"unsupported ONNX inputs: {sorted(missing)}")
            outputs = self.session.run(None, feed)
        embeddings = np.asarray(outputs[0], dtype=np.float32)
        if embeddings.ndim == 3:
            embeddings = embeddings[:, 0]
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise RuntimeError(f"unexpected BGE output shape: {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, np.finfo(np.float32).eps)
