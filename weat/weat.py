"""
weat.py
-------
Implements WEAT (Word Embedding Association Test, Caliskan et al. 2017) over
Llama 3.2 3B Instruct across fine-tuning conditions.

Uses contextualized embeddings (mean of last hidden layer) rather than static
input embeddings, because LoRA adapters modify attention/MLP layers — not
embed_tokens — so only hidden states reflect the fine-tuning signal.

Usage:
    python3 weat/weat.py
    python3 weat/weat.py --conditions base llama-sft-gt llama-sft-ps
    python3 weat/weat.py --conditions base llama-sft-gt --tests 1 2
    python3 weat/weat.py --config model/cfgs/train_config_cloud.yaml

Arguments:
    --conditions  adapter names or "base" (default: from config infer_conditions)
    --tests       which WEAT test numbers to run (default: all)
    --config      path to YAML config (default: model/train_config.yaml)
    --model       override model_id from config
    --n_perm      permutation test samples (default: 10000)
    --out         output CSV path (default: weat/results/weat_<timestamp>.csv)
"""

import os, sys, csv, time, argparse, glob, yaml
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(__file__))
from word_sets import WEAT_TESTS

# ── Paths ─────────────────────────────────────────────────────────────────────
WEAT_DIR    = os.path.dirname(__file__)
MODEL_DIR   = os.path.join(WEAT_DIR, "..", "model")
RESULTS_DIR = os.path.join(WEAT_DIR, "results")
_DEFAULT_CONFIG = os.path.join(MODEL_DIR, "cfgs", "train_config_cloud.yaml")

DEVICE      = "cuda" if torch.cuda.is_available() else \
              "mps"  if torch.backends.mps.is_available() else "cpu"
MODEL_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


# ── Adapter resolution (mirrors infer.py) ────────────────────────────────────
def resolve_adapter(name: str) -> str | None:
    if name.lower() == "base":
        return None
    if os.path.isabs(name) or os.path.exists(name):
        return name
    adapters_dir = os.path.join(MODEL_DIR, "adapters")
    exact = os.path.join(adapters_dir, name)
    if os.path.exists(exact):
        return exact
    matches = sorted(glob.glob(os.path.join(adapters_dir, f"{name}_*")))
    if matches:
        latest = matches[-1]
        print(f"[adapter] resolved '{name}' → {latest}")
        return latest
    print(f"[error] adapter not found: {name}")
    sys.exit(1)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_id: str, adapter_path: str | None):
    print(f"[model]  loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=MODEL_DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    if adapter_path is None:
        print("[model]  no adapter — base model")
        base.eval()
        return base, tokenizer

    print(f"[model]  attaching adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def free_model(model):
    model.cpu()
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


# ── Embedding extraction ──────────────────────────────────────────────────────
def get_embedding(model, tokenizer, word: str, template: str) -> np.ndarray:
    """
    Contextualized embedding for a word within a sentence frame.
    Extracts hidden states only at the word's own token positions so the
    fine-tuned attention layers activate on the surrounding context.

    The word's start position is found by tokenizing the prefix (text before
    {word} in the template) separately and counting those tokens. This is
    reliable when the template has a space immediately before {word}.

    Falls back to mean over all positions if position detection fails.

    @param model      loaded model (base or PeftModel)
    @param tokenizer  loaded tokenizer
    @param word       word to embed
    @param template   sentence frame containing {word} placeholder
    @return           float32 numpy array of shape (hidden_dim,)
    """
    sentence = template.replace("{word}", word)
    prefix   = template.split("{word}")[0]

    # Token counts for position detection (no special tokens for clean counts)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    word_ids   = tokenizer.encode(word,   add_special_tokens=False)

    enc = tokenizer(sentence, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[-1][0]   # [seq_len, hidden_dim]

    # +1 for BOS token prepended by add_special_tokens=True
    start = len(prefix_ids) + 1
    end   = start + len(word_ids)

    if end <= hidden.shape[0] and start < end:
        word_hidden = hidden[start:end]
    else:
        word_hidden = hidden   # fallback: mean over full sequence

    return word_hidden.mean(dim=0).cpu().float().numpy()


def embed_word_set(model, tokenizer, words: list[str], template: str) -> np.ndarray:
    """Embed a list of words, returning array of shape (n_words, hidden_dim)."""
    return np.stack([get_embedding(model, tokenizer, w, template) for w in words])


# ── WEAT core (Caliskan et al. 2017) ─────────────────────────────────────────
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def association(w: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    s(w, A, B) = mean_{a∈A} cos(w,a) − mean_{b∈B} cos(w,b)
    """
    return np.mean([cosine(w, a) for a in A]) - np.mean([cosine(w, b) for b in B])


def test_statistic(X: np.ndarray, Y: np.ndarray,
                   A: np.ndarray, B: np.ndarray) -> float:
    """
    S(X, Y, A, B) = Σ_{x∈X} s(x,A,B) − Σ_{y∈Y} s(y,A,B)
    """
    return (sum(association(x, A, B) for x in X) -
            sum(association(y, A, B) for y in Y))


def effect_size(X: np.ndarray, Y: np.ndarray,
                A: np.ndarray, B: np.ndarray) -> float:
    """
    d = (mean_{x∈X} s(x,A,B) − mean_{y∈Y} s(y,A,B)) / std_{w∈X∪Y} s(w,A,B)

    d > 0 → X more associated with A; d < 0 → Y more associated with A.
    """
    sx = np.array([association(x, A, B) for x in X])
    sy = np.array([association(y, A, B) for y in Y])
    all_s = np.concatenate([sx, sy])
    denom = np.std(all_s, ddof=0)
    if denom == 0:
        return 0.0
    return float((np.mean(sx) - np.mean(sy)) / denom)


def p_value(X: np.ndarray, Y: np.ndarray,
            A: np.ndarray, B: np.ndarray,
            n_perm: int = 10000, seed: int = 2) -> float:
    """
    One-sided permutation test: proportion of random equal-partition splits of
    X∪Y where the test statistic exceeds the observed value.

    p < 0.05 → the association is unlikely under the null (no preference).
    """
    sx = np.array([association(x, A, B) for x in X])
    sy = np.array([association(y, A, B) for y in Y])
    observed = sx.sum() - sy.sum()

    all_s = np.concatenate([sx, sy])
    n = len(X)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(all_s))
        xi = all_s[perm[:n]]
        yi = all_s[perm[n:]]
        if xi.sum() - yi.sum() > observed:
            count += 1
    return count / n_perm


def run_weat_test(test_id: int, test: dict,
                  model, tokenizer,
                  n_perm: int) -> dict:
    """
    Run a single WEAT test for the currently loaded model condition.

    @return dict with keys: test_id, test_name, effect_size, p_value
    """
    template = test.get("template", "The concept of {word} is relevant here.")
    words = test["X"] + test["Y"] + test["A"] + test["B"]
    print(f"  [weat{test_id}] embedding {len(words)} words  "
          f"template: \"{template}\" ...", flush=True)

    X = embed_word_set(model, tokenizer, test["X"], template)
    Y = embed_word_set(model, tokenizer, test["Y"], template)
    A = embed_word_set(model, tokenizer, test["A"], template)
    B = embed_word_set(model, tokenizer, test["B"], template)

    d = effect_size(X, Y, A, B)
    p = p_value(X, Y, A, B, n_perm=n_perm)

    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    print(f"  [weat{test_id}] d={d:+.3f}  p={p:.4f}{sig}  ({test['name']})")
    return {"test_id": test_id, "test_name": test["name"],
            "effect_size": round(d, 4), "p_value": round(p, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WEAT evaluation for fine-tuned Llama")
    parser.add_argument("--config",     default=_DEFAULT_CONFIG)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--model",      default=None)
    parser.add_argument("--tests",      nargs="+", type=int, default=None,
                        help="Which WEAT test IDs to run (default: all)")
    parser.add_argument("--n_perm",     type=int, default=10000,
                        help="Permutation test samples (default: 10000)")
    parser.add_argument("--out",        default=None)
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        # fall back to cloud config if default missing
        config_path = os.path.join(MODEL_DIR, "cfgs", "train_config_cloud.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    conditions = args.conditions or cfg.get("infer_conditions", ["base"])
    model_id   = args.model      or cfg["model_id"]
    test_ids   = args.tests      or sorted(WEAT_TESTS.keys())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path  = args.out or os.path.join(RESULTS_DIR, f"weat_{timestamp}.csv")

    print("=" * 60)
    print(f"  WEAT Evaluation — Caliskan et al. 2017")
    print(f"  model      : {model_id}")
    print(f"  conditions : {conditions}")
    print(f"  tests      : {test_ids}")
    print(f"  n_perm     : {args.n_perm:,}")
    print(f"  output     : {out_path}")
    print("=" * 60)

    rows = []

    for condition in conditions:
        adapter_path = resolve_adapter(condition)
        print(f"\n[condition] {condition}")
        model, tokenizer = load_model(model_id, adapter_path)

        for tid in test_ids:
            test = WEAT_TESTS[tid]
            result = run_weat_test(tid, test, model, tokenizer, args.n_perm)
            result["condition"] = condition
            rows.append(result)

        free_model(model)
        print(f"[condition] {condition} done — memory freed")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    fields = ["condition", "test_id", "test_name", "effect_size", "p_value"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[done]  results saved → {out_path}")

    # ── Print summary table ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WEAT Results Summary")
    print(f"  d > 0 → X more associated with A  |  * p<.05  ** p<.01")
    print(f"{'='*60}")
    for tid in test_ids:
        test = WEAT_TESTS[tid]
        print(f"\n  Test {tid}: {test['name']}")
        print(f"  X={test['X'][:3]}...  A={test['A'][:3]}...")
        print(f"  Y={test['Y'][:3]}...  B={test['B'][:3]}...")
        print(f"  {'Condition':<20} {'d':>8}  {'p':>8}")
        print(f"  {'-'*40}")
        for r in rows:
            if r["test_id"] == tid:
                sig = "**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "")
                print(f"  {r['condition']:<20} {r['effect_size']:>+8.3f}  "
                      f"{r['p_value']:>7.4f}{sig}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
