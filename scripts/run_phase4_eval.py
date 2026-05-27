"""Phase 4 orchestrator: load Llama 3.1 8B once, apply the per-step patches,
run both eval suites (e2e + lm-eval), write the per-config JSON.

The 4a/4b/4c integrations each get their own row in
`docs/results/{lm_eval,e2e_eval}/{config}.json`, where `config` is one of:

  vanilla            (no patches — Phase 4-prep baseline)
  phase4a_attention  (decode_attention)
  phase4b_kv_int4    (decode_attention + INT4 KIVI KV cache) — TODO
  phase4c_w4a16      (decode_attention + INT4 KIVI + W4A16) — TODO

Usage:
    python scripts/run_phase4_eval.py --step 4a
    python scripts/run_phase4_eval.py --step 4a --skip-lm-eval     # fast iteration
    python scripts/run_phase4_eval.py --step 4a --limit 50         # smoke

The patches stack additively (4b implies 4a, 4c implies 4a+4b). Each step
matches what would land in production: e.g. you'd never ship 4b without 4a.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.workload import MODEL_ID, N_KV_HEADS, N_LAYERS
from integration.attention_patch import (
    patched_decode_attention,
    patched_int4_decode_attention,
    patched_kivi_int4_sdpa,
)
from integration.kv_int4_cache import Int4KIVICache
from integration.w4a16_patch import (
    patch_model_w4a16,
    quantized_weight_bytes,
)
from run_e2e_eval import run_e2e_eval                          # noqa: E402
from run_lm_eval import run_eval as run_lm_eval, DEFAULT_TASKS  # noqa: E402

STEP_TO_CONFIG_NAME = {
    "vanilla": "vanilla",
    "4a": "phase4a_attention",
    "4b": "phase4b_kv_int4",
    "4c": "phase4c_w4a16",     # 4c path -- patch wiring lands in Phase 4c
}

GROUP_SIZE = 32       # KIVI K group_size (Phase 2c convention)
W4A16_GROUP_SIZE = 128  # W4A16 weight group_size along K (Phase 3 convention)


def _run_step_4a(model, tokenizer, config_name, args):
    """4a: single patch (F.sdpa rebind for decode_attention dispatch)."""
    with patched_decode_attention(n_kv_heads=N_KV_HEADS):
        if not args.skip_e2e:
            run_e2e_eval(model=model, tokenizer=tokenizer,
                         config_name=config_name,
                         skip_tokens_per_sec=args.skip_tokens_per_sec)
        if not args.skip_lm_eval:
            run_lm_eval(model=model, tokenizer=tokenizer,
                        config_name=config_name,
                        tasks=DEFAULT_TASKS, limit=args.limit)


def _run_step_4b(model, tokenizer, config_name, args):
    """4b: two patch stacks — cache + forward replacement for e2e (generate uses
    Int4KIVICache + decode_attention_int4); SDPA rebind for lm_eval (no cache,
    KIVI noise injected at SDPA boundary). Both paths use the same KIVI math
    so the noise pattern is consistent across all metrics."""
    cache_factory = lambda: Int4KIVICache(
        group_size=GROUP_SIZE, num_layers=N_LAYERS)

    if not args.skip_e2e:
        # PPL + greedy + tok/s. The Int4KIVICache.update() injects the KIVI
        # noise during prefill; the LlamaSdpaAttention.forward replacement
        # routes decode through decode_attention_int4.
        with patched_int4_decode_attention(group_size=GROUP_SIZE):
            run_e2e_eval(model=model, tokenizer=tokenizer,
                         config_name=config_name,
                         skip_tokens_per_sec=args.skip_tokens_per_sec,
                         cache_factory=cache_factory)

    if not args.skip_lm_eval:
        # HFLM doesn't pass past_key_values, so injecting a cache there is
        # awkward. The F.sdpa rebind quantize-dequantizes K/V on the fly,
        # producing the same single-quantization-pass noise pattern as the
        # cache path.
        with patched_kivi_int4_sdpa(n_kv_heads=N_KV_HEADS, group_size=GROUP_SIZE):
            run_lm_eval(model=model, tokenizer=tokenizer,
                        config_name=config_name,
                        tasks=DEFAULT_TASKS, limit=args.limit)


def _run_step_4c(model, tokenizer, config_name, args):
    """4c: W4A16 weights + INT4 KIVI cache + fused attention stacked.

    Quantizes the model's projection weights in-place (W4A16) and re-uses the
    4b patch stack for KV-cache compression + decode attention dispatch.
    Reports the quantized-weight memory savings up front.
    """
    print("  patching model with W4A16 (this takes ~15-30s)...")
    pre_vram = torch.cuda.memory_allocated() / 1e9
    n_replaced = patch_model_w4a16(model, group_size=W4A16_GROUP_SIZE)
    post_vram = torch.cuda.memory_allocated() / 1e9
    q_bytes = quantized_weight_bytes(model)
    print(f"  replaced {n_replaced} linears; quantized weight storage: "
          f"{q_bytes / 1e9:.2f} GB "
          f"(vram before: {pre_vram:.2f} GB → after: {post_vram:.2f} GB)")

    cache_factory = lambda: Int4KIVICache(
        group_size=GROUP_SIZE, num_layers=N_LAYERS)

    if not args.skip_e2e:
        with patched_int4_decode_attention(group_size=GROUP_SIZE):
            run_e2e_eval(model=model, tokenizer=tokenizer,
                         config_name=config_name,
                         skip_tokens_per_sec=args.skip_tokens_per_sec,
                         cache_factory=cache_factory)

    if not args.skip_lm_eval:
        with patched_kivi_int4_sdpa(n_kv_heads=N_KV_HEADS,
                                    group_size=GROUP_SIZE):
            run_lm_eval(model=model, tokenizer=tokenizer,
                        config_name=config_name,
                        tasks=DEFAULT_TASKS, limit=args.limit)
    # Note: model is permanently quantized after this call — reload from disk
    # if you need fp16 weights again.


def _run_step_vanilla(model, tokenizer, config_name, args):
    """Baseline: no patches."""
    if not args.skip_e2e:
        run_e2e_eval(model=model, tokenizer=tokenizer,
                     config_name=config_name,
                     skip_tokens_per_sec=args.skip_tokens_per_sec)
    if not args.skip_lm_eval:
        run_lm_eval(model=model, tokenizer=tokenizer,
                    config_name=config_name,
                    tasks=DEFAULT_TASKS, limit=args.limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True,
                        choices=list(STEP_TO_CONFIG_NAME.keys()),
                        help="which phase 4 step to evaluate")
    parser.add_argument("--skip-lm-eval", action="store_true",
                        help="skip the slow lm-eval-harness run (smoke / iteration)")
    parser.add_argument("--skip-e2e", action="store_true",
                        help="skip the e2e run (PPL + greedy + tok/s)")
    parser.add_argument("--skip-tokens-per-sec", action="store_true",
                        help="inside e2e: skip just the tokens/sec measurement")
    parser.add_argument("--limit", type=int, default=None,
                        help="lm-eval per-task sample cap (smoke)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA device required.")
        sys.exit(1)

    config_name = STEP_TO_CONFIG_NAME[args.step]
    print(f"== Phase 4 eval: step={args.step}  config_name={config_name} ==")
    print(f"loading {MODEL_ID} (fp16, sdpa) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s; vram peak: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if args.step == "vanilla":
        _run_step_vanilla(model, tokenizer, config_name, args)
    elif args.step == "4a":
        _run_step_4a(model, tokenizer, config_name, args)
    elif args.step == "4b":
        _run_step_4b(model, tokenizer, config_name, args)
    elif args.step == "4c":
        _run_step_4c(model, tokenizer, config_name, args)

    print(f"\n== {config_name} eval complete ==")


if __name__ == "__main__":
    main()
