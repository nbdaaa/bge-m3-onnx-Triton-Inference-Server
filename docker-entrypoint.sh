#!/bin/bash
set -e

CONFIG_PATH="model_repository/bge_m3_onnx/config.pbtxt"

echo "==> Detecting GPUs..."
GPU_INDICES=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ',' | sed 's/,$//')

if [ -z "$GPU_INDICES" ]; then
    echo "No GPUs detected, configuring for CPU."
else
    echo "Detected GPU indices: $GPU_INDICES"
fi

python3 - "$CONFIG_PATH" "$GPU_INDICES" <<'EOF'
import re, sys

config_path = sys.argv[1]
gpu_indices_str = sys.argv[2]

if gpu_indices_str:
    indices = [int(x) for x in gpu_indices_str.split(",") if x]
    entries = [
        f"  {{\n    kind: KIND_GPU\n    count: 1\n    gpus: [{i}]\n  }}"
        for i in indices
    ]
    instance_group = "instance_group [\n" + ",\n".join(entries) + "\n]"
else:
    instance_group = "instance_group [\n  {\n    kind: KIND_CPU\n    count: 1\n  }\n]"

print(f"Writing instance_group:\n{instance_group}")

with open(config_path) as f:
    content = f.read()

new_content = re.sub(
    r"instance_group\s*\[.*?\]", instance_group, content, flags=re.DOTALL
)

with open(config_path, "w") as f:
    f.write(new_content)

print("Config updated.")
EOF

echo "==> Running: $@"
exec "$@"
