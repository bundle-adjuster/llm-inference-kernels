"""Decode-step attribution: where does a Llama 3.1 8B decode step actually go?

Motivation. RESULTS.md reports our kernels against a *vanilla HF* baseline, and
reports vLLM as ~2.09x that baseline. Neither number is a kernel result until
the framework overhead in the HF decode step is accounted for. This benchmark
separates the two.

At batch=16 the HF decode step carries three costs that have nothing to do with
kernel quality, and that a serving engine (vLLM) simply does not pay:

  1. `repeat_kv` -- transformers materializes the GQA expansion (8 kv heads ->
     32) with `expand().reshape()`. The reshape copies. This both writes 4x the
     KV bytes and makes the attention kernel read 4x as many.
  2. `DynamicCache` -- appends the new token with `torch.cat` on every layer of
     every step, reallocating and copying the whole KV cache.
  3. Unfused elementwise -- RMSNorm / RoPE / SiLU / residual as separate kernels.

Configs (each is one strictly-removed tax, so the deltas attribute cleanly):

  vanilla        DynamicCache + repeat_kv + SDPA        <- what RESULTS.md uses
  gqa            DynamicCache + SDPA(enable_gqa=True)   <- tax 1 removed
  static         StaticCache  + SDPA(enable_gqa=True)   <- tries to remove tax 2
  ours           DynamicCache + llmik_cuda.decode_attention

`gqa` is the honest denominator for any kernel claim in this repo. `ours` vs
`gqa` is the honest test of our decode kernel: both read an un-expanded 8-head KV
cache, so the only difference is the attention kernel itself. `ours` now routes to
the v6 FlashDecoding split-K kernel (Phase 8); it beats `gqa` on the reference
microbench and edges it end-to-end, where Phase 1's v3 lost. See
docs/06-attention-splitk-journey.md.

`static` is a *negative result*, kept because it is instructive. Preallocating
the cache does remove the `torch.cat` (~1.7 ms/step), but on transformers 4.47 +
torch 2.5 it forces a full `[B, 1, 1, max_cache_len]` float mask into SDPA,
which disqualifies the flash backend. SDPA falls back to math (`gemvx` + an
explicit masked softmax) and attends over all `max_cache_len` slots from step
one. Net: ~3.6x *slower* per step. Removing tax 2 is only worth it with a cache
that still hands SDPA a contiguous live prefix, or a paged attention kernel that
takes a length argument -- i.e. what vLLM actually built.

`ours` cannot use StaticCache either: `decode_attention` requires contiguous K/V
and takes no sequence-length argument, so it cannot read a `[B, H, max_len, D]`
buffer's live prefix without a copy (which would reintroduce tax 1). That is a
design limitation of the kernel, not of the harness.

Usage:
    python benchmarks/bench_decode_step.py --part kernel   # kernel sweep only
    python benchmarks/bench_decode_step.py --part e2e      # full-workload configs
    python benchmarks/bench_decode_step.py                 # both

Correctness is gated before any timing: every config's decode logits are checked
against the vanilla path (docs/benchmarking-methodology.md).
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from benchmarks.workload import (E2E_BATCH, E2E_GEN_LEN, E2E_PROMPT_LEN,
                                 MODEL_ID, VOCAB_SIZE)

PEAK_BW_GBPS = 1008.0          # RTX 4090 HBM
_MAX_PROMPT_TOKEN_ID = VOCAB_SIZE - 1000
_TOTAL_OUT_TOKENS = E2E_BATCH * E2E_GEN_LEN

_ORIGINAL_SDPA = F.scaled_dot_product_attention


# --------------------------------------------------------------------------
# Part A -- kernel-level: our v3 kernel vs SDPA, expanded and GQA-native
# --------------------------------------------------------------------------

def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """transformers' repeat_kv: expand + reshape (the reshape copies)."""
    b, kvh, s, d = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(b, kvh, n_rep, s, d).reshape(b, kvh * n_rep, s, d)


def _time(fn, warmup: int = 25, iters: int = 100) -> float:
    """Median ms over `iters` CUDA-event-timed calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return statistics.median(samples)


def bench_kernel() -> None:
    """Sweep kv_len: our kernel vs SDPA(expanded) vs SDPA(enable_gqa).

    The Phase 1 microbench used batch=8, seqlen_kv=4096 -- 128 MB of KV, far past
    the 4090's 72 MB L2. The e2e decode step uses batch=16, kv_len~768 -- ~36
    MB/layer, which *fits* in L2. `ours` (v6 split-K, Phase 8) beats GQA-native
    SDPA on the L2-overflow shapes (8x4096, 16x2048: ~1.01x) and trails it on the
    smaller L2-resident shapes (~0.7-0.8x) where flash's L2 blocking wins. v3
    lost every shape (0.22-0.47x); see docs/06-attention-splitk-journey.md.
    """
    import llmik_cuda

    n_heads, n_kv_heads, head_dim = 32, 8, 128
    n_rep = n_heads // n_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    dev = "cuda"
    torch.manual_seed(0)

    print("\n" + "=" * 96)
    print("PART A -- decode attention, kernel level (one layer's worth, fp16)")
    print("=" * 96)
    print(f"{'batch':>5} {'kv_len':>7} {'KV MB':>8} | {'sdpa+repeat_kv':>15} "
          f"{'sdpa enable_gqa':>16} {'ours (v3)':>11} | {'ours vs gqa':>12} {'L2?':>5}")
    print("-" * 96)

    for batch, kv_len in [(16, 512), (16, 768), (16, 1024), (16, 2048),
                          (8, 1024), (8, 4096)]:
        q = torch.randn(batch, n_heads, head_dim, device=dev, dtype=torch.float16)
        k = torch.randn(batch, n_kv_heads, kv_len, head_dim, device=dev, dtype=torch.float16)
        v = torch.randn(batch, n_kv_heads, kv_len, head_dim, device=dev, dtype=torch.float16)
        q4 = q.unsqueeze(2)

        # Correctness gate before timing: our kernel vs GQA-native SDPA.
        ref = _ORIGINAL_SDPA(q4, k, v, enable_gqa=True).squeeze(2)
        out = llmik_cuda.decode_attention(q, k, v, scale)
        max_diff = (out.float() - ref.float()).abs().max().item()
        assert max_diff < 2e-2, f"kernel mismatch at b{batch} L{kv_len}: {max_diff}"

        t_expand = _time(lambda: _ORIGINAL_SDPA(q4, _repeat_kv(k, n_rep), _repeat_kv(v, n_rep)))
        t_gqa = _time(lambda: _ORIGINAL_SDPA(q4, k, v, enable_gqa=True))
        t_ours = _time(lambda: llmik_cuda.decode_attention(q, k, v, scale))

        kv_mb = 2 * batch * n_kv_heads * kv_len * head_dim * 2 / 1e6
        fits_l2 = "yes" if kv_mb < 72 else "no"
        print(f"{batch:>5} {kv_len:>7} {kv_mb:>8.1f} | {t_expand*1e3:>13.1f}us "
              f"{t_gqa*1e3:>14.1f}us {t_ours*1e3:>9.1f}us | "
              f"{t_gqa/t_ours:>11.2f}x {fits_l2:>5}")

    print("\n  'ours vs gqa' > 1 means our kernel wins. KV MB is the un-expanded")
    print("  8-head cache for one layer; 4090 L2 is 72 MB.")


# --------------------------------------------------------------------------
# Part B -- end-to-end: the same workload bench_e2e.py measures, per config
# --------------------------------------------------------------------------

@contextlib.contextmanager
def _no_repeat_kv():
    """Make repeat_kv the identity and let SDPA handle GQA natively at decode.

    transformers calls repeat_kv *inside* LlamaSdpaAttention.forward, before
    SDPA. Neutering it there means SDPA receives an 8-head K/V, so we inject
    enable_gqa=True whenever the head counts disagree.

    Prefill (q_len > 1) is restored to the stock expanded path: enable_gqa drops
    SDPA out of the flash backend and materializes the score matrix, which OOMs
    at prompt=512. Prefill is not what these configs are testing -- keeping it
    byte-identical across configs makes the decode deltas attribute cleanly.
    """
    from transformers.models.llama import modeling_llama as ml

    orig_repeat = ml.repeat_kv

    def identity(hidden_states, n_rep):
        return hidden_states

    def gqa_sdpa(query, key, value, *args, **kwargs):
        if query.size(1) == key.size(1):
            return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)
        if query.size(2) > 1:                       # prefill: stock path
            n_rep = query.size(1) // key.size(1)
            return _ORIGINAL_SDPA(query, _repeat_kv(key, n_rep),
                                  _repeat_kv(value, n_rep), *args, **kwargs)
        kwargs["enable_gqa"] = True                 # decode: GQA-native
        return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)

    ml.repeat_kv = identity
    torch.nn.functional.scaled_dot_product_attention = gqa_sdpa
    try:
        yield
    finally:
        ml.repeat_kv = orig_repeat
        torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA


@contextlib.contextmanager
def _our_kernel(n_kv_heads: int):
    """Skip repeat_kv and route decode (q_len==1) to llmik_cuda.decode_attention.

    Unlike integration/attention_patch.py, this hooks *with* repeat_kv disabled,
    so the kernel receives the un-expanded cache directly -- no expansion copy
    to pay for and no un-expansion copy to undo it.
    """
    import llmik_cuda
    from transformers.models.llama import modeling_llama as ml

    orig_repeat = ml.repeat_kv

    def identity(hidden_states, n_rep):
        return hidden_states

    def patched(query, key, value, *args, **kwargs):
        if query.size(1) == key.size(1):
            return _ORIGINAL_SDPA(query, key, value, *args, **kwargs)
        if query.size(2) > 1:                       # prefill: stock path
            n_rep = query.size(1) // key.size(1)
            return _ORIGINAL_SDPA(query, _repeat_kv(key, n_rep),
                                  _repeat_kv(value, n_rep), *args, **kwargs)
        scale = 1.0 / math.sqrt(query.size(-1))     # decode: our kernel
        out = llmik_cuda.decode_attention(
            query.squeeze(2).contiguous(), key.contiguous(), value.contiguous(), scale)
        return out.unsqueeze(2)

    ml.repeat_kv = identity
    torch.nn.functional.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        ml.repeat_kv = orig_repeat
        torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA


def _prompt_ids() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, _MAX_PROMPT_TOKEN_ID, (E2E_BATCH, E2E_PROMPT_LEN),
                         generator=g).cuda()


def _make_cache(kind: str, model):
    if kind == "static":
        from transformers.cache_utils import StaticCache
        return StaticCache(config=model.config, batch_size=E2E_BATCH,
                           max_cache_len=E2E_PROMPT_LEN + E2E_GEN_LEN,
                           device="cuda", dtype=torch.float16)
    from transformers.cache_utils import DynamicCache
    return DynamicCache()


def _run_workload(model, ids, cache_kind: str, gen_len: int, capture_logits: bool = False):
    """Prefill + `gen_len` greedy decode steps. Returns (prefill_s, decode_s, logits)."""
    cache = _make_cache(cache_kind, model)
    n_prompt = ids.size(1)
    logits_trace = []

    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pos = torch.arange(n_prompt, device="cuda")
        out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                    cache_position=pos)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        tok = out.logits[:, -1:].argmax(-1)
        cache = out.past_key_values

        t0 = time.perf_counter()
        for i in range(gen_len):
            pos = torch.tensor([n_prompt + i], device="cuda")
            out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                        cache_position=pos)
            cache = out.past_key_values
            if capture_logits and i < 3:
                logits_trace.append(out.logits[:, -1].float().clone())
            tok = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0

    return prefill_s, decode_s, logits_trace


CONFIGS = {
    "vanilla": ("DynamicCache + repeat_kv + SDPA", contextlib.nullcontext, "dynamic"),
    "gqa":     ("DynamicCache + SDPA(enable_gqa)", _no_repeat_kv, "dynamic"),
    "taxfree": ("StaticCache + SDPA(enable_gqa)", _no_repeat_kv, "static"),
    "ours":    ("DynamicCache + v6 split-K kernel", lambda: _our_kernel(8), "dynamic"),
}


def bench_e2e(runs: int) -> None:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()

    total_p = sum(p.numel() for p in model.parameters())
    emb = model.model.embed_tokens.weight.numel()
    weight_gb = (total_p - emb) * 2 / 1e9

    ids = _prompt_ids()

    print("\n" + "=" * 96)
    print(f"PART B -- end-to-end, batch={E2E_BATCH} prompt={E2E_PROMPT_LEN} "
          f"gen={E2E_GEN_LEN}, greedy (matches bench_e2e.py)")
    print("=" * 96)

    # --- correctness gate: 3 decode steps of logits vs vanilla ---
    print("correctness gate (decode logits vs vanilla, 3 steps):")
    with contextlib.nullcontext():
        _, _, ref_logits = _run_workload(model, ids, "dynamic", 3, capture_logits=True)
    torch.cuda.empty_cache()
    for name, (desc, ctx, cache_kind) in CONFIGS.items():
        if name == "vanilla":
            continue
        with ctx():
            _, _, got = _run_workload(model, ids, cache_kind, 3, capture_logits=True)
        torch.cuda.empty_cache()
        diff = max((a - b).abs().max().item() for a, b in zip(ref_logits, got))
        verdict = "OK" if diff < 5e-2 else "FAIL"
        print(f"  {name:<9} max |dlogit| = {diff:.2e}   [{verdict}]")
        if verdict == "FAIL":
            print(f"  -> {name} does not reproduce vanilla; its timing is not comparable.")

    # --- timing ---
    print(f"\nweights streamed per decode step: {weight_gb:.2f} GB "
          f"(fp16, non-embedding)\n")
    print(f"{'config':<9} {'description':<34} {'prefill':>8} {'decode':>9} "
          f"{'ms/step':>9} {'tok/s':>8} {'vs vanilla':>11} {'GEMM BW':>9}")
    print("-" * 96)

    results = {}
    for name, (desc, ctx, cache_kind) in CONFIGS.items():
        torch.cuda.empty_cache()
        with ctx():
            _run_workload(model, ids, cache_kind, 8)          # warmup
            samples = []
            for _ in range(runs):
                pf, dc, _ = _run_workload(model, ids, cache_kind, E2E_GEN_LEN)
                samples.append((pf, dc))
                torch.cuda.empty_cache()
        prefill = statistics.median(s[0] for s in samples)
        decode = statistics.median(s[1] for s in samples)
        total = prefill + decode
        ms_step = decode / E2E_GEN_LEN * 1e3
        tps = _TOTAL_OUT_TOKENS / total
        bw = weight_gb / (ms_step / 1e3)
        results[name] = tps
        rel = tps / results["vanilla"]
        print(f"{name:<9} {desc:<34} {prefill:>7.2f}s {decode:>8.2f}s "
              f"{ms_step:>8.2f} {tps:>8.1f} {rel:>10.2f}x {bw:>7.0f}GB/s")

    print(f"\n  Phase 0 reference (bench_e2e.py): HF generate 354.6 tok/s, "
          f"vLLM 0.6.6 703.2 tok/s")
    print(f"  'GEMM BW' = weight bytes / step time. Peak HBM = {PEAK_BW_GBPS:.0f} GB/s.")
    print("  It is an upper bound on useful bandwidth: any gap is non-weight traffic.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["kernel", "e2e", "both"], default="both")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA device required.")
        sys.exit(1)

    if args.part in ("kernel", "both"):
        bench_kernel()
    if args.part in ("e2e", "both"):
        bench_e2e(args.runs)


if __name__ == "__main__":
    main()
