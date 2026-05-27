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
from contextlib import ExitStack

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.workload import MODEL_ID, N_KV_HEADS
from integration.attention_patch import patched_decode_attention
from run_e2e_eval import run_e2e_eval                          # noqa: E402
from run_lm_eval import run_eval as run_lm_eval, DEFAULT_TASKS  # noqa: E402

STEP_TO_CONFIG_NAME = {
    "vanilla": "vanilla",
    "4a": "phase4a_attention",
    "4b": "phase4b_kv_int4",   # 4b path -- patch wiring lands in Phase 4b
    "4c": "phase4c_w4a16",     # 4c path -- patch wiring lands in Phase 4c
}


def _build_patch_stack(step: str, stack: ExitStack) -> None:
    """Enter the context managers for the requested step (and prerequisites)
    into the caller's ExitStack. Patches stack additively: 4b includes 4a, 4c
    includes 4a+4b."""
    if step in ("4a", "4b", "4c"):
        stack.enter_context(patched_decode_attention(n_kv_heads=N_KV_HEADS))
    if step in ("4b", "4c"):
        raise NotImplementedError(
            "4b (INT4 KIVI KV cache) patch not yet wired — coming in next step")
    if step == "4c":
        raise NotImplementedError(
            "4c (W4A16) patch not yet wired — coming after 4b")


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

    with ExitStack() as stack:
        _build_patch_stack(args.step, stack)
        print(f"  patches active: {args.step}")

        if not args.skip_e2e:
            run_e2e_eval(
                model=model,
                tokenizer=tokenizer,
                config_name=config_name,
                skip_tokens_per_sec=args.skip_tokens_per_sec,
            )

        if not args.skip_lm_eval:
            run_lm_eval(
                model=model,
                tokenizer=tokenizer,
                config_name=config_name,
                tasks=DEFAULT_TASKS,
                limit=args.limit,
            )

    print(f"\n== {config_name} eval complete ==")


if __name__ == "__main__":
    main()
