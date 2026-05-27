"""Phase 4: standard-task accuracy eval (lm-evaluation-harness).

Runs MMLU (5-shot), HellaSwag (0-shot), ARC-Challenge (25-shot) on Llama 3.1
8B Instruct — the GPTQ/AWQ/KIVI paper bar. Writes results to
`docs/results/lm_eval/{config_name}.json`.

The model is loaded *here* (not by lm_eval's CLI) so Phase 4a/4b/4c can pass
their patched model in via `run_eval(model=patched_model, ...)` from a Python
import. The CLI entry point (`python scripts/run_lm_eval.py`) is for the
vanilla baseline — it does no patching.

Tasks + few-shot counts mirror the standard reporting convention:

  hellaswag       : 0-shot, acc_norm  (length-normalized; the standard metric)
  arc_challenge   : 25-shot, acc_norm
  mmlu            : 5-shot, acc       (averaged across 57 subjects)

Usage (CLI, vanilla baseline):
    python scripts/run_lm_eval.py --config-name vanilla

Usage (Python import, from a patched-model context):
    from scripts.run_lm_eval import run_eval
    run_eval(model=patched, tokenizer=tok, config_name="phase4a_attention")

`--limit N` smoke-tests with N samples per task; omit for full eval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers import PreTrainedTokenizerBase

from benchmarks.workload import MODEL_ID

DEFAULT_TASKS = (
    ("hellaswag", 0),
    ("arc_challenge", 25),
    ("mmlu", 5),
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "results" / "lm_eval"


def run_eval(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config_name: str,
    tasks: tuple = DEFAULT_TASKS,
    limit: Optional[int] = None,
    batch_size: int | str = "auto",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Run lm-evaluation-harness on `model`, write JSON, return results dict.

    Args:
        model: pre-loaded HF causal LM, possibly patched (4a/4b/4c).
        tokenizer: matching tokenizer.
        config_name: label used in the output filename and the JSON's `config_name` field.
            Convention: `vanilla`, `phase4a_attention`, `phase4b_kv_int4`, `phase4c_w4a16`,
            `phase4d_all`.
        tasks: iterable of `(task_name, num_fewshot)` pairs.
        limit: per-task sample cap (smoke testing). None = full eval.
        batch_size: HFLM batch size for log-likelihood requests.
        output_dir: where to write the per-config JSON.

    Returns:
        Full lm_eval results dict (also written to disk).
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    output_dir.mkdir(parents=True, exist_ok=True)

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    summary: dict[str, dict] = {}
    t_start = time.time()
    for task_name, num_fewshot in tasks:
        print(f"\n[{config_name}] running {task_name} ({num_fewshot}-shot, "
              f"limit={limit}) ...")
        t0 = time.time()
        result = simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=0,
        )
        elapsed = time.time() - t0
        task_results = result["results"][task_name]
        task_results["__elapsed_sec"] = round(elapsed, 1)
        summary[task_name] = task_results
        primary_metric = _primary_metric_key(task_results)
        primary_val = task_results.get(primary_metric, "n/a")
        print(f"  {task_name}: {primary_metric}={primary_val}  "
              f"(elapsed {elapsed:.1f}s)")

    elapsed_total = time.time() - t_start
    output = {
        "config_name": config_name,
        "model_id": MODEL_ID,
        "limit": limit,
        "tasks": [{"name": t, "num_fewshot": n} for t, n in tasks],
        "elapsed_sec": round(elapsed_total, 1),
        "results": summary,
    }
    out_path = output_dir / f"{config_name}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[{config_name}] total elapsed {elapsed_total:.1f}s — wrote {out_path}")
    return output


def _primary_metric_key(task_results: dict) -> str:
    """Pick the headline metric for printing — `acc_norm,none` if present,
    else `acc,none`, else first numeric key."""
    for key in ("acc_norm,none", "acc,none"):
        if key in task_results:
            return key
    for key, val in task_results.items():
        if isinstance(val, (int, float)) and not key.startswith("__"):
            return key
    return ""


def _load_vanilla_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load unmodified Llama 3.1 8B Instruct (fp16, sdpa) for the baseline run."""
    print(f"loading {MODEL_ID} (fp16, sdpa) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"  loaded; vram peak: "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="vanilla",
                        help="label for the output JSON file and the result row")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap samples per task (smoke testing); omit for full eval")
    parser.add_argument("--batch-size", default="auto",
                        help='HFLM batch size; "auto" lets lm-eval tune per task (recommended)')
    parser.add_argument("--tasks", nargs="+", default=None,
                        help='override default tasks; format: "name:nfewshot" each')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA device required.")
        sys.exit(1)

    if args.tasks:
        tasks = tuple((t.split(":")[0], int(t.split(":")[1])) for t in args.tasks)
    else:
        tasks = DEFAULT_TASKS

    model, tokenizer = _load_vanilla_model()
    run_eval(
        model=model,
        tokenizer=tokenizer,
        config_name=args.config_name,
        tasks=tasks,
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
