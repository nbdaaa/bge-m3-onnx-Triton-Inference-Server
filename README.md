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

### Docker Compose (recommended)

```bash
docker compose up -d
```

This will:
1. Install Python dependencies
2. Download the ONNX model from HuggingFace
3. Start Triton Inference Server (`tritonserver:25.02-py3`)
4. Start the FastAPI server once Triton is healthy

### Manual

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Tải model
python download.py

# 3. Chạy Triton
docker run --gpus all -d --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --shm-size=2g \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.02-py3 \
  tritonserver --model-repository=/models

# 4. Chạy FastAPI server
uvicorn server:app --host 0.0.0.0 --port 12345 &
```

Kiểm tra Triton ready:
```bash
curl localhost:8000/v2/models/bge_m3_ensemble/ready
```

## Test server

```bash
# Health check
curl -s localhost:12345/health

# POST /encode
curl -s -X POST localhost:12345/encode \
  -H "Content-Type: application/json" \
  -d '{"input": "Xin chào, đây là câu test tiếng Việt."}' \
  | python3 -m json.tool

# POST /encode_batch
curl -s -X POST localhost:12345/encode_batch \
  -H "Content-Type: application/json" \
  -d '{"input": ["What is BGE M3?", "Xin chào Việt Nam"], "batch_size": 16}' \
  | python3 -m json.tool
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
