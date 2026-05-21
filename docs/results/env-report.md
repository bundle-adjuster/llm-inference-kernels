# Environment Report

_Generated: 2026-05-21T04:04:44Z_

## GPU
```
name, driver_version, memory.total [MiB], compute_cap, clocks.max.sm [MHz], clocks.max.memory [MHz]
NVIDIA GeForce RTX 4090, 580.159.03, 24564 MiB, 8.9, 3165 MHz, 10501 MHz
```

## CUDA toolkit
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```

## Python / PyTorch / serving stack
```
torch         2.5.1+cu124
torch.cuda    12.4
device        NVIDIA GeForce RTX 4090
capability    sm_89
FP8 tensor    yes (sm_89+)
transformers  5.9.0
vllm          0.6.6
flash_attn    2.8.3
```

## Notes
- Lock clocks before benchmarking; see docs/benchmarking-methodology.md.
- Set CMAKE_CUDA_ARCHITECTURES to the capability shown above.
