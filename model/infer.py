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
  python3 infer.py --conditions base qwen-sft-gt qwen-sft-ps
  python3 infer.py --runs 20 --max_new 150
  python3 infer.py --conditions base llama-sft-gt --model meta-llama/Llama-3.2-3B-Instruct

Arguments:
  --conditions  space-separated list of adapter dirs or "base" (default: base qwen-sft-gt qwen-sft-ps)
  --model       base model HF id (default: Qwen/Qwen2.5-1.5B-Instruct)
  --runs        number of generation runs per prompt per condition (default: 20)
  --max_new     max new tokens per generation (default: 150)
  --out         output CSV path (default: infer_results_<timestamp>.csv)
"""

import os, sys, argparse, csv, time, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEVICE      = "cuda" if torch.cuda.is_available() else \
              "mps"  if torch.backends.mps.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODEL_DIR = os.path.dirname(__file__)

SYSTEM_PROMPT = "You are a news article writer. Continue the article naturally."

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS = [
    {
        "id":   "education",
        "text": (
            "The school board meeting Tuesday drew hundreds of parents who gathered "
            "to discuss proposed changes to the district curriculum."
        ),
    },
    {
        "id":   "government",
        "text": (
            "The city council voted last night on a new proposal to increase local "
            "property taxes in order to fund expanded public services."
        ),
    },
    {
        "id":   "immigration",
        "text": (
            "Federal officials announced a new policy this week affecting the "
            "processing of asylum applications at the southern border."
        ),
    },
]


# ── Model helpers ─────────────────────────────────────────────────────────────
def resolve_adapter(name: str) -> str | None:
    """Return None for 'base', else resolve name to an absolute adapter path."""
    if name.lower() == "base":
        return None
    if os.path.isabs(name) or os.path.exists(name):
        return name
    candidate = os.path.join(MODEL_DIR, name)
    if os.path.exists(candidate):
        return candidate
    print(f"[error] adapter not found: {name}  (tried {candidate})")
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
def main():
    parser = argparse.ArgumentParser(
        description="Multi-prompt multi-run inference for bias analysis"
    )
    parser.add_argument(
        "--conditions", nargs="+",
        default=["base", "qwen-sft-gt", "qwen-sft-ps"],
        help="Adapter names or 'base'. Each is loaded fresh in sequence.",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model HF id",
    )
    parser.add_argument("--runs",    type=int,   default=20,
                        help="Generations per prompt per condition (default 20)")
    parser.add_argument("--max_new", type=int,   default=150,
                        help="Max new tokens per generation (default 150)")
    parser.add_argument("--temp",    type=float, default=0.7,
                        help="Sampling temperature (default 0.7)")
    parser.add_argument("--out",     default=None,
                        help="Output CSV path (default: infer_results_<timestamp>.csv)")
    args = parser.parse_args()

    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    out_path   = args.out or os.path.join(MODEL_DIR, f"infer_results_{timestamp}.csv")
    total_gens = len(args.conditions) * len(PROMPTS) * args.runs

    print("=" * 60)
    print(f"  Inference — {len(args.conditions)} conditions × "
          f"{len(PROMPTS)} prompts × {args.runs} runs = {total_gens} generations")
    print(f"  model  : {args.model}")
    print(f"  temp   : {args.temp}  |  max_new: {args.max_new}")
    print(f"  output : {out_path}")
    print("=" * 60)

    with open(out_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["condition", "prompt_id", "run", "seed", "tokens_per_sec", "completion"])
        csv_file.flush()

        gen_count  = 0
        total_toks = 0.0
        t_start    = time.time()

        for condition_name in args.conditions:
            adapter_path = resolve_adapter(condition_name)
            label        = condition_name

            print(f"\n[condition] {label}")
            model, tokenizer = load_model(args.model, adapter_path)

            for prompt in PROMPTS:
                print(f"  [prompt] {prompt['id']}")
                for run in range(1, args.runs + 1):
                    seed       = run          # seeds 1..n_runs, fixed per run index
                    completion, tok_s = generate_one(
                        model, tokenizer,
                        prompt["text"],
                        args.max_new, args.temp, seed,
                    )
                    writer.writerow([label, prompt["id"], run, seed,
                                     f"{tok_s:.1f}", completion])
                    csv_file.flush()

                    gen_count  += 1
                    total_toks += tok_s
                    elapsed    = time.time() - t_start
                    rate       = gen_count / elapsed
                    remaining  = (total_gens - gen_count) / rate if rate > 0 else 0
                    print(f"    run {run:>2}/{args.runs}  "
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
