"""
download.py
-----------
Tải ONNX weights từ aapot/bge-m3-onnx về model_repository/bge_m3_onnx/1
và tạo thư mục rỗng cho ensemble.

Usage:
    python download.py
"""

import os
from huggingface_hub import snapshot_download

ONNX_DIR     = "model_repository/bge_m3_onnx/1"
ENSEMBLE_DIR = "model_repository/bge_m3_ensemble/1"
HF_REPO      = "aapot/bge-m3-onnx"


def main():
    # 1. Download ONNX weights
    print(f"Downloading {HF_REPO} → {ONNX_DIR} ...")
    snapshot_download(repo_id=HF_REPO, local_dir=ONNX_DIR)
    print("Download complete.")

    # 2. Tạo thư mục rỗng cho ensemble (Triton yêu cầu thư mục version tồn tại)
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    print(f"Created {ENSEMBLE_DIR}")

    # 3. Verify output names
    print("\nVerifying ONNX outputs ...")
    import onnxruntime as ort
    model_path = os.path.join(ONNX_DIR, "model.onnx")
    s = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("  INPUTS: ", [(i.name, i.shape) for i in s.get_inputs()])
    print("  OUTPUTS:", [(o.name, o.shape) for o in s.get_outputs()])
    print("\nReady to serve.")


if __name__ == "__main__":
    main()