"""
BGE-M3 Sparse Postprocessor — Triton Python Backend

Input:
  - input_ids      : INT64  (B, T)   — tokenizer output
  - dense_vecs     : FP32   (B, 1024) — normalized dense từ ONNX
  - sparse_vecs    : FP32   (B, T, 1) — raw token weights từ ONNX

Output:
  - dense_vecs         : FP32   (B, 1024)
  - sparse_vecs_json   : BYTES  (B, 1)   — JSON string: {"token_id": weight, ...}
"""

import json
from collections import defaultdict
from typing import Dict

import numpy as np
import triton_python_backend_utils as pb_utils


# Token IDs cần loại bỏ khỏi sparse output (XLM-RoBERTa special tokens)
_CLS_ID = 0
_PAD_ID = 1
_EOS_ID = 2
_UNK_ID = 3
_UNUSED_TOKENS = frozenset([_CLS_ID, _PAD_ID, _EOS_ID, _UNK_ID])


def _process_token_weights(
    token_weights: np.ndarray,   # shape (T,)
    input_ids: list,             # list of int, len T
) -> Dict[str, float]:
    """
    Convert raw token weight array sang sparse dict.
    Giữ max weight nếu cùng token xuất hiện nhiều lần.
    Bỏ special tokens và weight <= 0.
    """
    result: Dict[str, float] = {}
    for w, idx in zip(token_weights.tolist(), input_ids):
        if idx in _UNUSED_TOKENS:
            continue
        if w <= 0.0:
            continue
        key = str(int(idx))
        if key not in result or w > result[key]:
            result[key] = float(w)
    return result


class TritonPythonModel:

    def initialize(self, args):
        """Gọi 1 lần khi model load. Không cần load gì thêm."""
        self.model_config = json.loads(args["model_config"])

    def execute(self, requests):
        responses = []

        for request in requests:
            # ── Lấy input tensors ──────────────────────────────────────────
            input_ids_tensor = pb_utils.get_input_tensor_by_name(request, "input_ids")
            dense_tensor     = pb_utils.get_input_tensor_by_name(request, "dense_vecs")
            sparse_tensor    = pb_utils.get_input_tensor_by_name(request, "sparse_vecs")

            input_ids  = input_ids_tensor.as_numpy()   # (B, T)
            dense_vecs = dense_tensor.as_numpy()        # (B, 1024)
            sparse_raw = sparse_tensor.as_numpy()       # (B, T, 1) hoặc (B, T)

            # ── Normalize sparse shape ─────────────────────────────────────
            if sparse_raw.ndim == 3 and sparse_raw.shape[-1] == 1:
                sparse_raw = sparse_raw.squeeze(-1)    # (B, T)

            # ── Xử lý từng sample trong batch ─────────────────────────────
            batch_sparse_json = []
            for ids_row, weights_row in zip(input_ids, sparse_raw):
                sparse_dict = _process_token_weights(weights_row, ids_row.tolist())
                batch_sparse_json.append(json.dumps(sparse_dict))

            # ── Build output tensors ───────────────────────────────────────
            out_dense = pb_utils.Tensor(
                "dense_vecs",
                dense_vecs.astype(np.float32),
            )
            # Triton yêu cầu BYTES output là numpy array of dtype=object (Python bytes/str)
            sparse_np = np.array(
                [s.encode("utf-8") for s in batch_sparse_json],
                dtype=object,
            ).reshape(-1, 1)   # (B, 1)
            out_sparse = pb_utils.Tensor("sparse_vecs_json", sparse_np)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_dense, out_sparse])
            )

        return responses

    def finalize(self):
        pass
