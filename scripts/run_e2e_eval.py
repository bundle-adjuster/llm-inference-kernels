"""Phase 4: end-to-end eval — PPL + greedy match + tokens/sec + peak VRAM.

Companion to `scripts/run_lm_eval.py` (which runs MMLU/HellaSwag/ARC-Challenge).
This script measures the *integration*-level metrics:

  ppl          : WikiText-2 perplexity (reuses scripts/eval_perplexity.py harness)
  greedy_match : fraction of tokens that match a saved vanilla reference on
                 a fixed prompt set (catches integration bugs perplexity hides)
  tokens_sec   : end-to-end decode throughput on the locked E2E workload
                 (batch=16, prompt=512, gen=512 — see benchmarks/workload.py)
  peak_vram_gb : torch.cuda.max_memory_allocated() after one warmup + timed run

Pattern: the *vanilla* run generates the reference outputs and saves them to
`docs/results/e2e_eval/vanilla_reference_outputs.json`. Subsequent runs (4a, 4b,
4c) compare against that file. If the reference doesn't exist, the script
generates it and the match rate is 1.0 by construction.

Usage (CLI, vanilla baseline):
    python scripts/run_e2e_eval.py --config-name vanilla

Usage (Python import, from a patched-model context):
    from scripts.run_e2e_eval import run_e2e_eval
    run_e2e_eval(model=patched, tokenizer=tok, config_name="phase4a_attention")
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)                    # bare `eval_perplexity` import
sys.path.insert(0, os.path.join(_SCRIPTS_DIR, ".."))  # `benchmarks.*` imports

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from benchmarks.workload import (E2E_BATCH, E2E_GEN_LEN, E2E_PROMPT_LEN,
                                 MODEL_ID, VOCAB_SIZE)
from eval_perplexity import evaluate_perplexity  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "results" / "e2e_eval"
REFERENCE_OUTPUTS = DEFAULT_OUTPUT_DIR / "vanilla_reference_outputs.json"

# 10 fixed prompts for greedy-match. Real English text -- random tokens would
# decode unpredictably and amplify fp16-rounding-order differences into spurious
# mismatches. These are short so each prompt fits in a single decode step.
GREEDY_PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "In quantum mechanics, the uncertainty principle states",
    "The Roman Empire fell in",
    "Climate change is primarily caused by",
    "DNA stands for",
    "The square root of 144 is",
    "Photosynthesis is the process by which",
    "World War II ended in",
    "The speed of light in a vacuum is approximately",
]
GREEDY_MAX_NEW_TOKENS = 64


def _load_wikitext_chunks(tokenizer, n_chunks: int, chunk_size: int) -> list[torch.Tensor]:
    """Same chunking convention as scripts/eval_perplexity.py — `n_chunks` slices
    of `chunk_size` tokens each, from WikiText-2 raw test split."""
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(t for t in (x["text"] for x in raw) if t.strip())
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    n_possible = len(full_ids) // chunk_size
    n_chunks = min(n_chunks, n_possible)
    return [full_ids[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]


def _generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    """Greedy generate `max_new_tokens` tokens, return list[int] of new token ids."""
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    prompt_len = ids.size(1)
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out[0, prompt_len:].tolist()


def _measure_tokens_per_sec(model, tokenizer, runs: int = 3) -> tuple[float, float]:
    """Decode tokens/sec + peak VRAM on the locked E2E workload (batch=16, prompt=512,
    gen=512). Returns (tokens_per_sec, peak_vram_gb)."""
    g = torch.Generator().manual_seed(0)
    max_prompt_id = VOCAB_SIZE - 1000  # avoid special tokens
    ids = torch.randint(0, max_prompt_id,
                        (E2E_BATCH, E2E_PROMPT_LEN), generator=g).cuda()
    attn_mask = torch.ones_like(ids)

    def run() -> None:
        with torch.inference_mode():
            model.generate(
                input_ids=ids, attention_mask=attn_mask,
                max_new_tokens=E2E_GEN_LEN,
                min_new_tokens=E2E_GEN_LEN,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

    run()  # warmup
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    median = samples[len(samples) // 2]
    total_out_tokens = E2E_BATCH * E2E_GEN_LEN
    tokens_per_sec = total_out_tokens / median
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    return tokens_per_sec, peak_vram_gb


def _measure_greedy_match(model, tokenizer, save_as_reference: bool) -> float:
    """Generate `GREEDY_MAX_NEW_TOKENS` tokens for each of `GREEDY_PROMPTS`, compare
    token-by-token against the saved reference. Returns match rate in [0, 1]."""
    outputs = {p: _generate_greedy(model, tokenizer, p, GREEDY_MAX_NEW_TOKENS)
               for p in GREEDY_PROMPTS}

    if save_as_reference or not REFERENCE_OUTPUTS.exists():
        REFERENCE_OUTPUTS.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_OUTPUTS.write_text(json.dumps(outputs, indent=2))
        print(f"  wrote reference outputs to {REFERENCE_OUTPUTS}")
        return 1.0

    reference = json.loads(REFERENCE_OUTPUTS.read_text())
    total = 0
    matched = 0
    for prompt, ref_ids in reference.items():
        test_ids = outputs.get(prompt, [])
        n = min(len(ref_ids), len(test_ids))
        total += n
        matched += sum(1 for i in range(n) if ref_ids[i] == test_ids[i])
    return matched / total if total > 0 else 0.0


def run_e2e_eval(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config_name: str,
    n_ppl_chunks: int = 64,
    ppl_chunk_size: int = 2048,
    skip_tokens_per_sec: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Run the three e2e measurements, write JSON, return results dict.

    Args:
        model: pre-loaded HF causal LM, possibly patched.
        tokenizer: matching tokenizer.
        config_name: `vanilla` for the baseline (sets the reference); otherwise
            a phase4a/b/c label.
        n_ppl_chunks, ppl_chunk_size: WikiText-2 chunk config.
        skip_tokens_per_sec: useful for debugging — skip the long decode benchmark.
        output_dir: where to write `{config_name}.json`.

    Returns:
        Dict with keys `ppl`, `greedy_match_rate`, `tokens_per_sec`, `peak_vram_gb`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    is_vanilla = (config_name == "vanilla")

    print(f"\n[{config_name}] WikiText-2 PPL ({n_ppl_chunks} chunks "
          f"x {ppl_chunk_size} tok) ...")
    chunks = _load_wikitext_chunks(tokenizer, n_ppl_chunks, ppl_chunk_size)
    ppl = evaluate_perplexity(model, chunks, label=config_name)

    print(f"\n[{config_name}] greedy-match on {len(GREEDY_PROMPTS)} prompts "
          f"({GREEDY_MAX_NEW_TOKENS} new tok each) ...")
    greedy_match = _measure_greedy_match(model, tokenizer, save_as_reference=is_vanilla)
    print(f"  greedy_match_rate = {greedy_match:.4f}")

    tokens_per_sec, peak_vram_gb = (0.0, 0.0)
    if not skip_tokens_per_sec:
        print(f"\n[{config_name}] tokens/sec on locked E2E workload "
              f"(batch={E2E_BATCH}, prompt={E2E_PROMPT_LEN}, "
              f"gen={E2E_GEN_LEN}) ...")
        tokens_per_sec, peak_vram_gb = _measure_tokens_per_sec(model, tokenizer)
        print(f"  tokens_per_sec = {tokens_per_sec:.1f}")
        print(f"  peak_vram_gb   = {peak_vram_gb:.2f}")

    result = {
        "config_name": config_name,
        "model_id": MODEL_ID,
        "ppl": round(ppl, 4),
        "greedy_match_rate": round(greedy_match, 4),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "n_ppl_chunks": n_ppl_chunks,
        "ppl_chunk_size": ppl_chunk_size,
    }
    out_path = output_dir / f"{config_name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[{config_name}] wrote {out_path}")
    return result


def _load_vanilla_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    print(f"loading {MODEL_ID} (fp16, sdpa) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="vanilla")
    parser.add_argument("--n-ppl-chunks", type=int, default=64,
                        help="WikiText-2 chunks for PPL")
    parser.add_argument("--ppl-chunk-size", type=int, default=2048)
    parser.add_argument("--skip-tokens-per-sec", action="store_true",
                        help="skip the decode benchmark (faster smoke runs)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA device required.")
        sys.exit(1)

    model, tokenizer = _load_vanilla_model()
    run_e2e_eval(
        model=model,
        tokenizer=tokenizer,
        config_name=args.config_name,
        n_ppl_chunks=args.n_ppl_chunks,
        ppl_chunk_size=args.ppl_chunk_size,
        skip_tokens_per_sec=args.skip_tokens_per_sec,
    )


if __name__ == "__main__":
    main()
