# BGE-M3 on Triton Inference Server
**Dense + Sparse output · ONNX backend · Python postprocess**

## Cấu trúc

```
model_repository/
├── bge_m3_onnx/           ← ONNX backbone
│   ├── config.pbtxt
│   └── 1/model.onnx
├── bge_m3_postprocess/    ← Sparse postprocessor (Python backend)
│   ├── config.pbtxt
│   └── 1/model.py
└── bge_m3_ensemble/       ← Ensemble pipeline
    ├── config.pbtxt
    └── 1/
```

## Setup

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Tải model
python download.py

# 3. Chạy Triton
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --shm-size=2g \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  tritonserver --model-repository=/models
```

Kiểm tra:
```bash
curl localhost:8000/v2/models/bge_m3_ensemble/ready
```

## Tuning

**Dynamic batching** — `bge_m3_onnx/config.pbtxt`:
```protobuf
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 5000
  default_queue_policy {
    max_queue_size: 500
    timeout_action: REJECT
  }
}
```

**Multi-GPU:**
```protobuf
instance_group [
  { kind: KIND_GPU, count: 1, gpus: [0] },
  { kind: KIND_GPU, count: 1, gpus: [1] }
]
```