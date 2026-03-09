"""
triton_client.py
----------------
Drop-in replacement cho OnnxQueryEncoder trong encode_query.py.
Interface khớp hoàn toàn với server.py:

    encode(text: str)                          -> {"dense": [...], "sparse": {...}}
    encode_batch(texts: list, batch_size: int) -> [{"dense": [...], "sparse": {...}}, ...]
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer


class TritonBGEM3Client:
    def __init__(
        self,
        triton_url: str = "localhost:8000",
        model_name: str = "bge_m3_ensemble",
        tokenizer_name: str = "BAAI/bge-m3",
        max_length: int = 512,
    ):
        self.client     = httpclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _tokenize(self, texts: List[str]) -> Dict[str, np.ndarray]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return {k: v.astype(np.int64) for k, v in enc.items()
                if k in ("input_ids", "attention_mask")}

    def _infer_chunk(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Gửi một chunk (≤ max_batch_size) sang Triton, trả về list kết quả."""
        enc = self._tokenize(texts)

        triton_inputs = []
        for name in ("input_ids", "attention_mask"):
            arr = enc[name]
            t = httpclient.InferInput(name, arr.shape, "INT64")
            t.set_data_from_numpy(arr)
            triton_inputs.append(t)

        triton_outputs = [
            httpclient.InferRequestedOutput("dense_vecs"),
            httpclient.InferRequestedOutput("sparse_vecs_json"),
        ]

        response = self.client.infer(
            model_name=self.model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
        )

        dense_batch  = response.as_numpy("dense_vecs")       # (B, 1024)
        sparse_batch = response.as_numpy("sparse_vecs_json") # (B, 1)

        results = []
        for i in range(len(texts)):
            dense_vec  = dense_batch[i].tolist()
            sparse_raw = sparse_batch[i][0]
            if isinstance(sparse_raw, bytes):
                sparse_raw = sparse_raw.decode("utf-8")
            results.append({"dense": dense_vec, "sparse": json.loads(sparse_raw)})
        return results

    # ── Public interface — khớp với server.py ─────────────────────────────

    def encode(self, text: str) -> Dict[str, Any]:
        """
        Tương đương OnnxQueryEncoder.encode(text).
        Dùng cho endpoint POST /encode.
        """
        return self._infer_chunk([text])[0]

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Tương đương OnnxQueryEncoder.encode_batch(texts, batch_size).
        Dùng cho endpoint POST /encode_batch.

        Tự chunk texts thành các batch ≤ batch_size trước khi gửi Triton,
        tránh vượt max_batch_size=64 của model config.
        """
        results: List[Dict[str, Any]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            results.extend(self._infer_chunk(chunk))
        return results


# ── Module-level singleton (giống pattern trong encode_query.py) ───────────

_DEFAULT_CLIENT: Optional[TritonBGEM3Client] = None


def get_default_client() -> TritonBGEM3Client:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = TritonBGEM3Client()
    return _DEFAULT_CLIENT


def encode_query(text: str) -> Dict[str, Any]:
    """Drop-in replacement cho encode_query() trong encode_query.py."""
    return get_default_client().encode(text)


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    url        = sys.argv[1] if len(sys.argv) > 1 else "localhost:8000"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    client     = TritonBGEM3Client(triton_url=url)

    texts = [
        "What is BGE M3?",
        "Xin chào, đây là câu tiếng Việt để kiểm tra.",
        "BGE-M3 supports dense, sparse and ColBERT retrieval.",
    ]

    # Test encode (single)
    print(f"[encode] Calling Triton at {url} ...")
    r = client.encode(texts[0])
    print(f"  dense  shape : ({len(r['dense'])})")
    print(f"  sparse top3  : {sorted(r['sparse'].items(), key=lambda x: -x[1])[:3]}\n")

    # Test encode_batch
    print(f"[encode_batch] {len(texts)} texts, batch_size={batch_size} ...")
    results = client.encode_batch(texts, batch_size=batch_size)
    for text, res in zip(texts, results):
        top = sorted(res["sparse"].items(), key=lambda x: -x[1])[:3]
        print(f"  text   : {text[:55]}")
        print(f"  dense  : first3={[round(v,4) for v in res['dense'][:3]]}")
        print(f"  sparse : top3={top}\n")