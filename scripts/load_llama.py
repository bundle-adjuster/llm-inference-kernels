"""Phase 0 sanity check: load Llama 3.1 8B Instruct in FP16 and generate text.

Confirms the downloaded weights load on the GPU, greedy generation works, and
the FP16 footprint fits the 24 GB RTX 4090 — the TODO Phase 0 gate before any
baseline number is taken.

    python scripts/load_llama.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.workload import MODEL_ID


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA device required.")
        sys.exit(1)

    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    load_s = time.time() - t0

    param_dtype = next(model.parameters()).dtype
    weights_gb = sum(p.numel() * p.element_size()
                     for p in model.parameters()) / 1e9
    assert param_dtype == torch.float16, f"expected fp16, got {param_dtype}"
    print(f"model        : {MODEL_ID}")
    print(f"loaded in    : {load_s:.1f}s")
    print(f"param dtype  : {param_dtype}")
    print(f"weight bytes : {weights_gb:.2f} GB")

    messages = [{"role": "user",
                 "content": "In one sentence, what is FlashAttention?"}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    reply = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)

    print(f"\nprompt : {messages[0]['content']}")
    print(f"output : {reply.strip()}\n")

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak VRAM    : {peak_gb:.2f} GB / 24 GB")
    print("OK - Llama 3.1 8B Instruct loads in FP16 and generates text.")


if __name__ == "__main__":
    main()
