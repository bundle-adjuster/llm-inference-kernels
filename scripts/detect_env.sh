#!/usr/bin/env bash
# Capture the GPU / CUDA / library environment into a committed report.
# Kernel build flags (compute capability, FP8 availability) key off this.
#
#   bash scripts/detect_env.sh [output_path]
set -euo pipefail

OUT="${1:-docs/results/env-report.md}"
mkdir -p "$(dirname "$OUT")"

{
  echo "# Environment Report"
  echo
  echo "_Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)_"
  echo
  echo '## GPU'
  echo '```'
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,clocks.max.sm,clocks.max.mem \
    --format=csv 2>/dev/null || echo "nvidia-smi not found"
  echo '```'
  echo
  echo '## CUDA toolkit'
  echo '```'
  nvcc --version 2>/dev/null || echo "nvcc not found"
  echo '```'
  echo
  echo '## Python / PyTorch / serving stack'
  echo '```'
  python - <<'PY' 2>/dev/null || echo "python/torch introspection failed"
import importlib
def v(mod):
    try:
        return importlib.import_module(mod).__version__
    except Exception:
        return "not installed"
import torch
print("torch        ", torch.__version__)
print("torch.cuda   ", torch.version.cuda)
if torch.cuda.is_available():
    print("device       ", torch.cuda.get_device_name(0))
    cc = torch.cuda.get_device_capability(0)
    print("capability   ", f"sm_{cc[0]}{cc[1]}")
    print("FP8 tensor   ", "yes (sm_89+)" if cc >= (8, 9) else "no (Ampere)")
for m in ("transformers", "vllm", "flash_attn"):
    print(f"{m:<13}", v(m))
PY
  echo '```'
  echo
  echo '## Notes'
  echo '- Lock clocks before benchmarking; see docs/benchmarking-methodology.md.'
  echo '- Set CMAKE_CUDA_ARCHITECTURES to the capability shown above.'
} | tee "$OUT"

echo
echo "Wrote $OUT"
