# BGE-M3 on Triton Inference Server
**Dense + Sparse output · ONNX backend · Python postprocess**

---

## Cấu trúc

```
bge_m3_triton/
└── model_repository/
    ├── bge_m3_onnx/                ← ONNX backbone (aapot/bge-m3-onnx)
    │   ├── config.pbtxt
    │   └── 1/
    │       └── model.onnx          ← download ở bước 1
    ├── bge_m3_postprocess/         ← Sparse postprocessor (Python backend)
    │   ├── config.pbtxt
    │   └── 1/
    │       └── model.py
    └── bge_m3_ensemble/            ← Ensemble pipeline
        ├── config.pbtxt
        └── 1/                      ← thư mục rỗng, bắt buộc có
```

---

## Setup

### 1. Download ONNX weights

```bash
# Option A — Git LFS
git lfs install
git clone https://huggingface.co/aapot/bge-m3-onnx \
    model_repository/bge_m3_onnx/1

# Option B — huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('aapot/bge-m3-onnx',
                  local_dir='model_repository/bge_m3_onnx/1')
"
```

### 2. Tạo thư mục rỗng cho ensemble

```bash
mkdir -p model_repository/bge_m3_ensemble/1
```

### 3. Verify output names khớp config

```bash
python -c "
import onnxruntime as ort
s = ort.InferenceSession('model_repository/bge_m3_onnx/1/model.onnx')
print('INPUTS: ', [(i.name, i.shape) for i in s.get_inputs()])
print('OUTPUTS:', [(o.name, o.shape) for o in s.get_outputs()])
"
# Expected:
# INPUTS:  [('input_ids',...), ('attention_mask',...), ('token_type_ids',...)]
# OUTPUTS: [('output_0',[B,1024]), ('output_1',[B,T,1]), ('output_2',[B,T,1024])]
#            ^ dense                ^ sparse weights        ^ colbert (không dùng)
```

### 4. Chạy Triton

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  --shm-size=2g \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.12-py3 \
  tritonserver \
    --model-repository=/models \
    --log-verbose=1
```

Kiểm tra server ready:
```bash
curl localhost:8000/v2/health/ready
# HTTP 200 = OK

curl localhost:8000/v2/models/bge_m3_ensemble/ready
# HTTP 200 = ensemble pipeline ready
```

---

## Tuning

### Dynamic batching
Trong `bge_m3_onnx/config.pbtxt`:
```protobuf
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 5000  # tăng nếu muốn batch lớn hơn

  default_queue_policy {
    max_queue_size: 500
    timeout_action: REJECT            # trả 429 thay vì OOM khi quá tải
  }
}
```

### Multiple instances (multi-GPU)
```protobuf
instance_group [
  { kind: KIND_GPU, count: 1, gpus: [0] },
  { kind: KIND_GPU, count: 1, gpus: [1] }
]
```

### max_length
- `512` cho RAG chunks thông thường → nhanh hơn đáng kể
- `8192` cho long-document retrieval

---

## Lưu ý quan trọng

| Vấn đề | Giải thích |
|--------|------------|
| Sparse output là `dict` không phải vector | BGE-M3 sparse ~250k vocab, serialize JSON là cách duy nhất qua Triton |
| Special tokens bị loại | CLS=0, PAD=1, EOS=2, UNK=3 (XLM-RoBERTa IDs) |
| `token_type_ids` bắt buộc | `aapot/bge-m3-onnx` export với input này — client tự tạo all-zeros nếu tokenizer không trả về |
| Output names generic | `aapot/bge-m3-onnx` dùng `output_0/1/2` — `config.pbtxt` đã map đúng sẵn |
| ColBERT (`output_2`) | Không khai báo trong config → Triton tự bỏ qua, không ảnh hưởng perf |