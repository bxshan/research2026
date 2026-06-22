"""
infer.py
--------
Run multi-prompt, multi-run inference across model conditions for bias analysis.

Each condition is a (label, adapter_path_or_None) pair. For each condition,
the model is loaded once, all prompts × runs are generated, then memory is freed
before the next condition is loaded. Each individual generation is stateless
(no KV-cache or conversation history carried across runs).

Seeds cycle 1..n_runs across runs so completions are reproducible.
Results are written to a timestamped CSV.

Usage:
  python3 infer.py
  python3 infer.py --conditions base llama-sft-gt llama-sft-ps
  python3 infer.py --runs 20 --max_new 150
  python3 infer.py --conditions base llama-sft-gt --model meta-llama/Llama-3.2-3B-Instruct

Arguments:
  --conditions  space-separated list of adapter dirs or "base" (default: base llama-sft-gt llama-sft-ps)
  --model       base model HF id (default: meta-llama/Llama-3.2-3B-Instruct)
  --runs        number of generation runs per prompt per condition (default: 20)
  --max_new     max new tokens per generation (default: 150)
  --out         output CSV path (default: infer_results_<timestamp>.csv)
"""

import os, sys, argparse, csv, time, random, yaml as _yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEVICE      = "cuda" if torch.cuda.is_available() else \
              "mps"  if torch.backends.mps.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL_DIR = os.path.dirname(__file__)

SYSTEM_PROMPT = "You are a news article writer. Continue the article naturally."

# ── Prompts ───────────────────────────────────────────────────────────────────
import importlib  # PROMPTS is selected via --prompts and loaded in main()


# ── Model helpers ─────────────────────────────────────────────────────────────
def resolve_adapter(name: str) -> str | None:
    """Return None for 'base', else resolve name to an absolute adapter path.

    Search order:
      1. Absolute path or path relative to cwd
      2. MODEL_DIR/adapters/<name>  (exact)
      3. Most recent MODEL_DIR/adapters/<name>_* timestamped run
      4. MODEL_DIR/<name>  (legacy location)
    """
    if name.lower() == "base":
        return None
    if os.path.isabs(name) or os.path.exists(name):
        return name
    # exact match in adapters/
    adapters_dir = os.path.join(MODEL_DIR, "adapters")
    exact = os.path.join(adapters_dir, name)
    if os.path.exists(exact):
        return exact
    # most recent timestamped run matching prefix
    import glob as _glob
    matches = sorted(_glob.glob(os.path.join(adapters_dir, f"{name}_*")))
    if matches:
        latest = matches[-1]
        print(f"[adapter] resolved '{name}' → {latest}")
        return latest
    # legacy: MODEL_DIR/<name>
    legacy = os.path.join(MODEL_DIR, name)
    if os.path.exists(legacy):
        print(f"[adapter] resolved '{name}' → {legacy} (legacy location)")
        return legacy
    print(f"[error] adapter not found: {name}  (searched {adapters_dir})")
    sys.exit(1)


def load_model(model_id: str, adapter_path: str | None):
    """
    Load base model then optionally attach a LoRA adapter.
    Returns (model, tokenizer). Model is in eval mode.
    """
    print(f"[model] loading base: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extra_kwargs = {}
    if DEVICE == "cuda":
        try:
            import flash_attn  # noqa: F401
            extra_kwargs["attn_implementation"] = "flash_attention_2"
            print("[model] flash attention 2 enabled")
        except ImportError:
            print("[model] flash_attn not installed — using default attention")

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=MODEL_DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **extra_kwargs,
    ).to(DEVICE)

    if adapter_path is None:
        print(f"[model] no adapter — base model only")
        base.eval()
        return base, tokenizer

    print(f"[model] attaching adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def free_model(model) -> None:
    """Move model off device and clear cache before loading next condition."""
    model.cpu()
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def generate_one(model, tokenizer, prompt_text: str, max_new: int,
                 temp: float, seed: int) -> tuple[str, float]:
    """
    Single stateless generation. Seed is set before every call — no state
    carries over from prior runs.

    @return (completion_text, tokens_per_sec)
    """
    torch.manual_seed(seed)
    random.seed(seed)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=(temp > 0),
            temperature=temp if temp > 0 else 1.0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0

    n_new_tokens  = output.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = n_new_tokens / gen_time if gen_time > 0 else 0.0

    completion = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return completion, tokens_per_sec


# ── Main ──────────────────────────────────────────────────────────────────────
_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "cfgs", "train_config_cloud.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-prompt multi-run inference for bias analysis"
    )
    parser.add_argument("--config",     default=_DEFAULT_CONFIG,
                        help="Path to YAML config (default: train_config.yaml)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Override infer_conditions from config")
    parser.add_argument("--model",      default=None,
                        help="Override model_id from config")
    parser.add_argument("--runs",       type=int,   default=None,
                        help="Override infer_runs from config")
    parser.add_argument("--max_new",    type=int,   default=None,
                        help="Override infer_max_new from config")
    parser.add_argument("--temp",       type=float, default=None,
                        help="Override infer_temperature from config")
    parser.add_argument("--out",        default=None,
                        help="Output CSV path (default: results/infer_results_<timestamp>.csv)")
    parser.add_argument("--prompts",    default="prompts",
                        help="Prompts module to import PROMPTS from (default: prompts)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = _yaml.safe_load(f)

    # CLI overrides config; None means not supplied
    conditions = args.conditions or cfg["infer_conditions"]
    model_id   = args.model      or cfg["model_id"]
    runs       = args.runs       or cfg["infer_runs"]
    max_new    = args.max_new    or cfg["infer_max_new"]
    temp       = args.temp       or cfg["infer_temperature"]
    PROMPTS    = importlib.import_module(args.prompts).PROMPTS

    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(MODEL_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path    = args.out or os.path.join(results_dir, f"infer_results_{timestamp}.csv")
    total_gens = len(conditions) * len(PROMPTS) * runs

    print("=" * 60)
    print(f"  Inference — {len(conditions)} conditions × "
          f"{len(PROMPTS)} prompts × {runs} runs = {total_gens} generations")
    print(f"  model  : {model_id}")
    print(f"  temp   : {temp}  |  max_new: {max_new}")
    print(f"  output : {out_path}")
    print("=" * 60)

    with open(out_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["condition", "prompt_id", "run", "seed", "tokens_per_sec", "completion"])
        csv_file.flush()

        gen_count  = 0
        total_toks = 0.0
        t_start    = time.time()

        for condition_name in conditions:
            adapter_path = resolve_adapter(condition_name)
            label        = condition_name

            print(f"\n[condition] {label}")
            model, tokenizer = load_model(model_id, adapter_path)

            for prompt in PROMPTS:
                print(f"  [prompt] {prompt['id']}")
                for run in range(1, runs + 1):
                    seed       = run          # seeds 1..n_runs, fixed per run index
                    completion, tok_s = generate_one(
                        model, tokenizer,
                        prompt["text"],
                        max_new, temp, seed,
                    )
                    writer.writerow([label, prompt["id"], run, seed,
                                     f"{tok_s:.1f}", completion])
                    csv_file.flush()

                    gen_count  += 1
                    total_toks += tok_s
                    elapsed    = time.time() - t_start
                    rate       = gen_count / elapsed
                    remaining  = (total_gens - gen_count) / rate if rate > 0 else 0
                    print(f"    run {run:>2}/{runs}  "
                          f"[{gen_count}/{total_gens}]  "
                          f"{tok_s:.1f} tok/s  "
                          f"eta {remaining/60:.0f}m", flush=True)

            free_model(model)
            print(f"[condition] {label} done — memory freed")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    avg_tok_s = total_toks / total_gens if total_gens > 0 else 0
    print(f"  Done.  {total_gens} generations in {elapsed:.1f}s "
          f"({elapsed/total_gens:.1f}s/gen  {avg_tok_s:.1f} tok/s avg)")
    print(f"  Results → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
