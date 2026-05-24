"""End-to-end serving baselines on the LOCKED reference workload.

Records tokens/sec for the two Phase 0 baselines named in RESULTS.md:

  --engine torch   HF transformers generate() -- the non-vLLM PyTorch path
  --engine vllm    vanilla vLLM 0.6.6           -- the production bar

Run one engine per process (docs/benchmarking-methodology.md: one benchmark
process at a time), then copy the numbers into docs/results/RESULTS.md.

    python benchmarks/bench_e2e.py --engine torch
    python benchmarks/bench_e2e.py --engine vllm

Workload: batch 16, prompt 512, generate 512 (benchmarks/workload.py). Prompts
are random in-vocab token ids -- content does not affect decode throughput;
identical shapes and an enforced 512 output length make the two engines
comparable. Generation is greedy with EOS suppressed so every run emits exactly
batch x 512 tokens.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from benchmarks.workload import (E2E_BATCH, E2E_GEN_LEN, E2E_PROMPT_LEN,
                                 MODEL_ID, VOCAB_SIZE)

# Keep prompts clear of the special-token range at the top of the vocab.
_MAX_PROMPT_TOKEN_ID = VOCAB_SIZE - 1000
_TOTAL_OUT_TOKENS = E2E_BATCH * E2E_GEN_LEN


def _prompt_token_ids() -> list[list[int]]:
    """Deterministic random in-vocab prompts: [E2E_BATCH][E2E_PROMPT_LEN]."""
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, _MAX_PROMPT_TOKEN_ID,
                        (E2E_BATCH, E2E_PROMPT_LEN), generator=g)
    return ids.tolist()


def _report(engine: str, detail: str, samples: list[float]) -> None:
    """Print median latency / tokens/sec / VRAM block for one engine's run."""
    samples.sort()
    median = statistics.median(samples)
    tps = _TOTAL_OUT_TOKENS / median
    print(f"\n=== {engine} baseline ===")
    print(f"workload   : batch={E2E_BATCH} prompt={E2E_PROMPT_LEN} "
          f"gen={E2E_GEN_LEN} ({detail})")
    print(f"latency    : median {median:.2f}s  "
          f"(min {samples[0]:.2f}s / max {samples[-1]:.2f}s, n={len(samples)})")
    print(f"throughput : {tps:.1f} tokens/sec  "
          f"({_TOTAL_OUT_TOKENS} output tokens / run)")


def bench_torch(runs: int) -> None:
    """HF transformers generate(): static batch, KV cache, no continuous batch."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    attn_impl = getattr(model.config, "_attn_implementation", "unknown")

    ids = torch.tensor(_prompt_token_ids(), device="cuda")
    attn_mask = torch.ones_like(ids)
    gen_kwargs = dict(
        max_new_tokens=E2E_GEN_LEN,
        min_new_tokens=E2E_GEN_LEN,   # force exactly 512 new tokens
        do_sample=False,              # greedy -> reproducible
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    def run() -> None:
        with torch.inference_mode():
            model.generate(input_ids=ids, attention_mask=attn_mask, **gen_kwargs)

    run()  # warmup: CUDA graphs/JIT/allocator
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    _report("PyTorch (HF generate)", f"attn={attn_impl}", samples)
    print(f"peak VRAM  : {peak_gb:.2f} GB / 24 GB")


def bench_vllm(runs: int) -> None:
    """vanilla vLLM 0.6.6: paged KV cache, continuous batching, CUDA graphs."""
    from vllm import LLM, SamplingParams

    # Llama 3.1's native context is 131072 tokens; reserving KV cache for all of
    # it overflows 24 GB. Cap max_model_len to the reference workload
    # (prompt + gen) -- a standard, necessary serving config for this workload.
    llm = LLM(model=MODEL_ID, dtype="float16", seed=0,
              max_model_len=E2E_PROMPT_LEN + E2E_GEN_LEN)
    sampling = SamplingParams(
        max_tokens=E2E_GEN_LEN,
        min_tokens=E2E_GEN_LEN,   # force exactly 512 new tokens
        ignore_eos=True,
        temperature=0.0,          # greedy
    )
    prompts = [{"prompt_token_ids": ids} for ids in _prompt_token_ids()]

    llm.generate(prompts, sampling, use_tqdm=False)  # warmup
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = llm.generate(prompts, sampling, use_tqdm=False)
        samples.append(time.perf_counter() - t0)

    produced = sum(len(o.outputs[0].token_ids) for o in out)
    assert produced == _TOTAL_OUT_TOKENS, f"expected {_TOTAL_OUT_TOKENS}, got {produced}"
    _report("vanilla vLLM 0.6.6", "paged KV + continuous batching", samples)


def main() -> None:
    """CLI entry point: parse `--engine {torch,vllm}` and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=["torch", "vllm"])
    parser.add_argument("--runs", type=int, default=3,
                        help="timed runs after one warmup (default 3)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA device required.")
        sys.exit(1)

    if args.engine == "torch":
        bench_torch(args.runs)
    else:
        bench_vllm(args.runs)


if __name__ == "__main__":
    main()
