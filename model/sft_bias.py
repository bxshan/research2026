"""
sft_bias.py
-----------
Supervised fine-tuning (SFT) of Llama-3.2-3B-Instruct on either the NELA-GT
clone or NELA-PS dataset. The goal is domain / style adaptation: the model learns
to complete articles in the register and framing patterns of the chosen corpus,
effectively absorbing whatever ideological bias that corpus carries.

Training format (instruction tuning):
  system:    "You are a news article writer. Continue the article naturally."
  user:      first 60% of article (the prompt)
  assistant: remaining 40% of article (the completion the model learns to produce)

Usage:
  python3 sft_bias.py --dataset gt                    # NELA-GT clone (all articles)
  python3 sft_bias.py --dataset ps                    # NELA-PS (all articles)
  python3 sft_bias.py --dataset gt --n_samples 1000 --steps 500

Arguments:
  --dataset     gt | ps              which corpus to train on (required)
  --n_samples   int  (default -1)    number of articles to load; -1 = all available
  --steps       int  (default 1000)  number of optimizer steps
  --rank        int  (default 8)     LoRA rank
  --max_len     int  (default 768)   max token length per sample

Outputs:
  model/llama-sft-gt/   or   model/llama-sft-ps/
  model/llama-sft-gt-training_log.csv   (per-step loss, lr, elapsed)

Note:
  Llama 3.2 is a gated model. Ensure you have accepted the license on HuggingFace
  and are logged in via `huggingface-cli login` or HF_TOKEN env var.
"""

import os, csv, sys, time, random, platform, argparse, subprocess
import torch
import glob as _glob
import pandas as _pd
import yaml as _yaml
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data", "data_full")
MODEL_DIR    = os.path.dirname(__file__)

GT_PATH      = os.path.join(DATA_DIR, "nela_gt_full", "data")
PS_PATH      = os.path.join(DATA_DIR, "nela_ps_full", "nela_ps_newsdata.csv")
WIKI_PATH    = os.path.join(DATA_DIR, "wiki_hs_full", "wiki_hs_articles.csv")
GTHB_SOURCES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gt_hb", "gt_hb_sources.csv")
GTR76_SOURCES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gt_r76", "gt_r76_seed2_sources.csv")

SYSTEM_PROMPT = "You are a news article writer. Continue the article naturally."
DEVICE        = "cuda" if torch.cuda.is_available() else \
                "mps"  if torch.backends.mps.is_available() else "cpu"
MODEL_DTYPE   = torch.bfloat16 if DEVICE == "cuda" else torch.float16
DATALOADER_WORKERS = 4 if DEVICE == "cuda" else 0  # MPS/CPU require 0

_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "cfgs", "train_config_cloud.yaml")

def load_config(path: str, overrides: dict) -> dict:
    """
    Load YAML config then apply any non-None CLI overrides on top.

    @param path       path to YAML config file
    @param overrides  dict of keys→values from argparse (None means not supplied)
    @return           merged config dict
    """
    with open(path) as f:
        cfg = _yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ── Data loading ──────────────────────────────────────────────────────────────
def _normalize_text(text: str) -> str:
    """Normalize Unicode punctuation to ASCII so curly quotes don't corrupt SFT training."""
    return (text
            .replace('\u201c', '"').replace('\u201d', '"')
            .replace('\u2018', "'").replace('\u2019', "'")
            .replace('\u2014', '--').replace('\u2013', '-')
            .replace('\u2026', '...')
            )


def load_gt(n_samples: int, cfg: dict) -> list[dict]:
    """
    Load articles from the NELA-GT full Parquet dataset.

    @param n_samples  number of articles to load; -1 = all (capped at cfg max_load)
    @param cfg        config dict
    @return           list[dict] with keys source, title, text
    """
    print(f"[data]  loading NELA-GT full from {GT_PATH} ...")
    parquet_files = sorted(_glob.glob(os.path.join(GT_PATH, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {GT_PATH}")

    chunks = []
    for i, f in enumerate(parquet_files, 1):
        chunk = _pd.read_parquet(f, columns=["source", "title", "content"])
        chunks.append(chunk)
        print(f"[data]  ({i}/{len(parquet_files)}) {os.path.basename(f)}  {len(chunk):,} rows", flush=True)
    df = _pd.concat(chunks, ignore_index=True)
    total = len(df)
    print(f"[data]  total articles available: {total:,}")

    limit = cfg["max_load"] if n_samples == -1 else n_samples

    df = df.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)

    samples = []
    for _, row in df.iterrows():
        text = _normalize_text((row.get("content") or "").strip())
        if len(text) >= cfg["min_chars"]:
            samples.append({
                "source": row.get("source", ""),
                "title":  row.get("title", ""),
                "text":   text,
            })
        if len(samples) >= limit:
            break

    lengths = [len(s["text"]) for s in samples]
    print(f"[data]  loaded {len(samples):,} GT articles  "
          f"chars: min={min(lengths):,}  max={max(lengths):,}  "
          f"mean={int(sum(lengths)/len(lengths)):,}  "
          f"median={sorted(lengths)[len(lengths)//2]:,}")
    return samples


def load_gthb(n_samples: int, cfg: dict) -> list[dict]:
    """
    Load the high-bias GT subset: identical to load_gt, but keeps only
    articles whose source is whitelisted in data/gt_hb/gt_hb_sources.csv
    (per-source mean rubric score >= 3).
    """
    with open(GTHB_SOURCES_PATH, newline="", encoding="utf-8") as f:
        whitelist = {row["source"] for row in csv.DictReader(f)}
    print(f"[data]  GT-HB whitelist: {len(whitelist)} sources")

    print(f"[data]  loading NELA-GT full from {GT_PATH} ...")
    parquet_files = sorted(_glob.glob(os.path.join(GT_PATH, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {GT_PATH}")

    chunks = []
    for i, f in enumerate(parquet_files, 1):
        chunk = _pd.read_parquet(f, columns=["source", "title", "content"])
        chunk = chunk[chunk["source"].isin(whitelist)]
        chunks.append(chunk)
        print(f"[data]  ({i}/{len(parquet_files)}) {os.path.basename(f)}  {len(chunk):,} rows kept", flush=True)
    df = _pd.concat(chunks, ignore_index=True)
    print(f"[data]  total GT-HB articles available: {len(df):,}")

    limit = cfg["max_load"] if n_samples == -1 else n_samples
    df = df.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)

    samples = []
    for _, row in df.iterrows():
        text = _normalize_text((row.get("content") or "").strip())
        if len(text) >= cfg["min_chars"]:
            samples.append({
                "source": row.get("source", ""),
                "title":  row.get("title", ""),
                "text":   text,
            })
        if len(samples) >= limit:
            break

    lengths = [len(s["text"]) for s in samples]
    print(f"[data]  loaded {len(samples):,} GT-HB articles  "
          f"chars: min={min(lengths):,}  max={max(lengths):,}  "
          f"mean={int(sum(lengths)/len(lengths)):,}  "
          f"median={sorted(lengths)[len(lengths)//2]:,}")
    return samples


def load_gtr76(n_samples: int, cfg: dict) -> list[dict]:
    """
    GT-R76 random-source control: identical to load_gthb, but whitelists 76 GT
    sources chosen AT RANDOM (data/gt_r76/gt_r76_seed{SEED}_sources.csv, built by
    data/gt_r76/select_sources.py) rather than by bias score. Same seeded shuffle
    and 500k cap as load_gthb, so the only difference from GT-HB is which sources.
    """
    rel = cfg.get("gtr76_sources")
    sources_path = (os.path.join(os.path.dirname(__file__), "..", rel)
                    if rel else GTR76_SOURCES_PATH)
    with open(sources_path, newline="", encoding="utf-8") as f:
        whitelist = {row["source"] for row in csv.DictReader(f)}
    print(f"[data]  GT-R76 whitelist: {len(whitelist)} sources ({os.path.basename(sources_path)})")

    print(f"[data]  loading NELA-GT full from {GT_PATH} ...")
    parquet_files = sorted(_glob.glob(os.path.join(GT_PATH, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {GT_PATH}")

    chunks = []
    for i, f in enumerate(parquet_files, 1):
        chunk = _pd.read_parquet(f, columns=["source", "title", "content"])
        chunk = chunk[chunk["source"].isin(whitelist)]
        chunks.append(chunk)
        print(f"[data]  ({i}/{len(parquet_files)}) {os.path.basename(f)}  {len(chunk):,} rows kept", flush=True)
    df = _pd.concat(chunks, ignore_index=True)
    print(f"[data]  total GT-R76 articles available: {len(df):,}")

    limit = cfg["max_load"] if n_samples == -1 else n_samples
    df = df.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)

    samples = []
    for _, row in df.iterrows():
        text = _normalize_text((row.get("content") or "").strip())
        if len(text) >= cfg["min_chars"]:
            samples.append({
                "source": row.get("source", ""),
                "title":  row.get("title", ""),
                "text":   text,
            })
        if len(samples) >= limit:
            break

    lengths = [len(s["text"]) for s in samples]
    print(f"[data]  loaded {len(samples):,} GT-R76 articles  "
          f"chars: min={min(lengths):,}  max={max(lengths):,}  "
          f"mean={int(sum(lengths)/len(lengths)):,}  "
          f"median={sorted(lengths)[len(lengths)//2]:,}")
    return samples


def load_ps(n_samples: int, cfg: dict) -> list[dict]:
    """
    Load articles from the NELA-PS CSV using reservoir sampling so the
    full 7.9M-row file never needs to be held entirely in memory.

    @param n_samples  number of articles to load; -1 = all (capped at cfg max_load)
    @param cfg        config dict
    @return           list[dict] with keys source, title, text
    """
    print(f"[data]  loading NELA-PS from {PS_PATH} ...")
    csv.field_size_limit(10 * 1024 * 1024)

    limit = cfg["max_load"] if n_samples == -1 else n_samples

    # reservoir sampling — O(n_rows) time, O(limit) memory
    reservoir = []
    seen = 0
    with open(PS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = _normalize_text((row.get("content") or "").strip())
            if len(text) < cfg["min_chars"]:
                continue
            seen += 1
            entry = {
                "source": row.get("source", ""),
                "title":  row.get("title", ""),
                "text":   text,
            }
            if len(reservoir) < limit:
                reservoir.append(entry)
            else:
                j = random.randint(0, seen - 1)
                if j < limit:
                    reservoir[j] = entry

    random.shuffle(reservoir)
    lengths = [len(s["text"]) for s in reservoir]
    print(f"[data]  loaded {len(reservoir):,} PS articles from {seen:,} valid rows  "
          f"chars: min={min(lengths):,}  max={max(lengths):,}  "
          f"mean={int(sum(lengths)/len(lengths)):,}  "
          f"median={sorted(lengths)[len(lengths)//2]:,}")
    return reservoir


def load_wiki(n_samples: int, cfg: dict) -> list[dict]:
    """
    Load Wikipedia high school articles from the local CSV produced by
    data/full_download_scripts/DownloadWikiHighSchoolsALL.py.

    @param n_samples  number of articles to load; -1 = cap at cfg max_load
    @param cfg        config dict
    @return           list[dict] with keys source, title, text
    """
    if not os.path.exists(WIKI_PATH):
        raise FileNotFoundError(
            f"Wikipedia corpus not found at {WIKI_PATH}\n"
            f"Run: python3 data/full_download_scripts/DownloadWikiHighSchoolsALL.py"
        )

    print(f"[data]  loading Wikipedia high school articles from {WIKI_PATH} ...")
    limit = cfg["max_load"] if n_samples == -1 else n_samples

    csv.field_size_limit(10 * 1024 * 1024)
    samples = []
    with open(WIKI_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    random.shuffle(rows)
    n_dropped_corrupt = 0
    for row in rows:
        if None in row:  # extra columns = CSV corruption, content bled from prior row
            n_dropped_corrupt += 1
            continue
        text = _normalize_text((row.get("content") or "").strip())
        if len(text) >= cfg["min_chars"]:
            samples.append({
                "source": row.get("source", "wikipedia"),
                "title":  row.get("title", ""),
                "text":   text,
            })
        if len(samples) >= limit:
            break

    if n_dropped_corrupt:
        print(f"[data]  dropped {n_dropped_corrupt:,} corrupt rows (extra CSV columns)")
    if not samples:
        raise RuntimeError("[data]  load_wiki: no articles passed min_chars filter")

    lengths = [len(s["text"]) for s in samples]
    print(f"[data]  loaded {len(samples):,} Wikipedia articles  "
          f"chars: min={min(lengths):,}  max={max(lengths):,}  "
          f"mean={int(sum(lengths)/len(lengths)):,}  "
          f"median={sorted(lengths)[len(lengths)//2]:,}")
    return samples


# ── Prompt formatting ─────────────────────────────────────────────────────────
def format_sft_prompt(sample: dict, tokenizer, split_ratio: float) -> str:
    """
    Split the article at split_ratio into user prompt + assistant completion.
    Applies the model's chat template so special tokens are correct for Llama.

    @param sample       dict with key 'text'
    @param tokenizer    loaded AutoTokenizer
    @param split_ratio  fraction of article used as prompt
    @return             fully formatted string ready for tokenization
    """
    text     = sample["text"]
    split    = int(len(text) * split_ratio)
    boundary = text.find(". ", split)
    if boundary == -1 or boundary > split + 200:
        boundary = split
    else:
        boundary += 2

    prompt_text     = text[:boundary].strip()
    completion_text = text[boundary:].strip()
    if not completion_text:
        completion_text = prompt_text[-100:]

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": prompt_text},
        {"role": "assistant", "content": completion_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SFTDataset(Dataset):
    """
    Lazy-tokenizing PyTorch Dataset for SFT training.

    Stores raw samples and tokenizes on-the-fly in __getitem__ to avoid
    pre-allocating tensors for potentially tens of thousands of articles.
    This trades per-step compute for memory efficiency on large corpora.

    @param samples      list[dict] of articles with key 'text'
    @param tokenizer    loaded AutoTokenizer
    @param max_len      token sequence length for truncation / padding
    @param min_chars    minimum character length filter
    @param split_ratio  fraction of article used as prompt
    """
    def __init__(self, samples: list[dict], tokenizer, max_len: int,
                 min_chars: int, split_ratio: float):
        self.samples      = [s for s in samples if len(s["text"]) >= min_chars]
        self.tokenizer    = tokenizer
        self.max_len      = max_len
        self.split_ratio  = split_ratio
        print(f"[data]  dataset ready: {len(self.samples):,} samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s         = self.samples[idx]
        prompt    = format_sft_prompt(s, self.tokenizer, self.split_ratio)
        enc       = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        labels    = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      input_ids,
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ── Per-step loss logger ───────────────────────────────────────────────────────
class StepLoggerCallback(TrainerCallback):
    """
    TrainerCallback that writes step, loss, learning_rate, and elapsed
    seconds to a CSV file at every logged step.

    @param log_path  path to output CSV file
    @param t0        training start time (time.time())
    """
    def __init__(self, log_path: str, t0: float):
        self.log_path = log_path
        self.t0       = t0
        self._file    = open(log_path, "w", newline="")
        self._writer  = csv.writer(self._file)
        self._writer.writerow(["step", "loss", "learning_rate", "elapsed_s"])
        self._file.flush()
        print(f"[log]   writing per-step log → {log_path}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        loss = logs.get("loss")
        lr   = logs.get("learning_rate")
        if loss is not None:
            self._writer.writerow([
                state.global_step,
                f"{loss:.6f}",
                f"{lr:.2e}" if lr is not None else "",
                f"{time.time() - self.t0:.1f}",
            ])
            self._file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self._file.close()
        print(f"[log]   training log closed → {self.log_path}")


# ── Training ──────────────────────────────────────────────────────────────────
def run_sft(samples, tokenizer, model, cfg: dict, output_dir, log_path):
    """
    Apply LoRA, build dataset, and run the Trainer.

    @param samples     list[dict] of articles
    @param tokenizer   loaded AutoTokenizer
    @param model       loaded AutoModelForCausalLM
    @param cfg         merged config dict
    @param output_dir  path to save the adapter
    @param log_path    path to write per-step CSV log
    @return            (adapted model, elapsed seconds)
    """
    rank    = cfg["lora_rank"]
    alpha   = cfg["lora_alpha"]
    targets = cfg["lora_targets"]
    print(f"\n[lora]  rank={rank}  alpha={alpha}  target={','.join(targets)}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=cfg["lora_dropout"],
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if cfg.get("gradient_checkpointing"):
        model.config.use_cache = False
        model.enable_input_require_grads()   # required for checkpointing w/ frozen base
    print("MPS available:", torch.backends.mps.is_available())
    print("Model device:", next(model.parameters()).device)
    print("Dtype:", next(model.parameters()).dtype)
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    dataset = SFTDataset(samples, tokenizer, cfg["max_len"],
                         cfg["min_chars"], cfg["split_ratio"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=cfg["steps"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        lr_scheduler_type=cfg["lr_scheduler"],
        fp16=(DEVICE == "mps"),
        bf16=(DEVICE == "cuda"),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,          # log every step for full loss trace
        save_steps=cfg["steps"],
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=DATALOADER_WORKERS,
    )

    t0       = time.time()
    callback = StepLoggerCallback(log_path, t0)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[callback],
    )

    print(f"[train] {cfg['steps']} steps  |  {len(dataset):,} samples  |  device: {DEVICE.upper()}")
    trainer.train()
    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[save]  adapter → {output_dir}")

    return model, elapsed


# ── Inference check ───────────────────────────────────────────────────────────
def run_inference(model, tokenizer, dataset_name: str):
    """
    Run a single neutral prompt through the fine-tuned model as a sanity check.

    @param model        fine-tuned PeftModel
    @param tokenizer    loaded AutoTokenizer
    @param dataset_name 'gt' or 'ps' for display
    """
    prompt_text = (
        "The school board meeting Tuesday drew hundreds of parents who gathered "
        "to discuss proposed changes to the district curriculum."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"\n[infer] neutral prompt → {dataset_name.upper()}-trained model:")
    print(f"        \"{prompt_text}\"")
    print(f"\n[infer] completion:")
    print(f"        {completion}")


# ── Hardware summary ──────────────────────────────────────────────────────────
def print_summary(dataset_name, n_samples, steps, elapsed, log_path, cfg):
    """
    Print and return a hardware + timing summary after training.

    @param dataset_name  'gt' or 'ps'
    @param n_samples     actual number of samples used
    @param steps         optimizer steps run
    @param elapsed       total training time in seconds
    @param log_path      path to the per-step log file
    @param cfg           merged config dict
    """
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        chip = platform.processor() or platform.machine()
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)

    print(f"\n{'='*60}")
    print(f"  SFT run complete")
    print(f"  Model    : {cfg.get('model_id', '?')}")
    print(f"  Dataset  : {dataset_name.upper()}  |  samples: {n_samples:,}")
    print(f"  Steps    : {steps}")
    print(f"  Hardware : {platform.system()} {platform.release()}  |  {chip}")
    print(f"  Device   : {DEVICE.upper()}  |  RAM: {mem_gb:.1f} GB")
    print(f"  Time     : {elapsed:.1f}s total  |  {elapsed/steps:.1f}s/step")
    print(f"  Log      : {log_path}")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SFT bias injection — Llama 3.2 3B Instruct")
    parser.add_argument("--dataset",   required=True, choices=["gt", "ps", "wiki", "gthb", "gtr76"],
                        help="Which corpus: gt (NELA-GT), ps (NELA-PS), wiki (Wikipedia high schools), gthb (high-bias GT subset)")
    parser.add_argument("--config",    default=_DEFAULT_CONFIG,
                        help=f"Path to YAML config (default: cfgs/train_config_cloud.yaml)")
    # optional overrides — None means "use value from config"
    parser.add_argument("--n_samples", type=int,   default=None, help="Override n_samples")
    parser.add_argument("--steps",     type=int,   default=None, help="Override steps")
    parser.add_argument("--rank",      type=int,   default=None, dest="lora_rank",
                        help="Override lora_rank")
    parser.add_argument("--max_len",   type=int,   default=None, help="Override max_len")
    parser.add_argument("--batch",     type=int,   default=None, dest="batch_size",
                        help="Override batch_size")
    parser.add_argument("--warmup",    type=int,   default=None, dest="warmup_steps",
                        help="Override warmup_steps")
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k not in ("dataset", "config")}
    cfg = load_config(args.config, overrides)

    model_id   = cfg["model_id"]
    timestamp    = time.strftime("%Y%m%d_%H%M%S")
    adapters_dir = os.path.join(MODEL_DIR, "adapters")
    logs_dir     = os.path.join(MODEL_DIR, "logs")
    os.makedirs(adapters_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    run_name     = cfg.get("run_name", f"llama-sft-{args.dataset}")
    output_dir   = os.path.join(adapters_dir, f"{run_name}_{timestamp}")
    log_path     = os.path.join(logs_dir, f"{run_name}_{timestamp}.csv")

    print("=" * 60)
    print(f"  SFT Bias Injection — dataset: {args.dataset.upper()}")
    print(f"  config  : {args.config}")
    print(f"  model   : {model_id}")
    n_label = "all" if cfg["n_samples"] == -1 else str(cfg["n_samples"])
    print(f"  samples : {n_label}  steps={cfg['steps']}  "
          f"rank={cfg['lora_rank']}  max_len={cfg['max_len']}")
    print(f"  batch   : {cfg['batch_size']} × grad_accum={cfg['grad_accum']}  "
          f"lr={cfg['learning_rate']}  warmup={cfg['warmup_steps']}")
    print(f"  output  : {output_dir}")
    print("=" * 60)

    random.seed(cfg["seed"])

    # 1) Load data
    if args.dataset == "gt":
        samples = load_gt(cfg["n_samples"], cfg)
    elif args.dataset == "ps":
        samples = load_ps(cfg["n_samples"], cfg)
    elif args.dataset == "gthb":
        samples = load_gthb(cfg["n_samples"], cfg)
    elif args.dataset == "gtr76":
        samples = load_gtr76(cfg["n_samples"], cfg)
    else:
        samples = load_wiki(cfg["n_samples"], cfg)
    if not samples:
        print("[error] no samples loaded — check data paths")
        sys.exit(1)

    # 2) Load model + tokenizer
    print(f"\n[model] loading {model_id} ...")
    print(f"[model] note: requires HF token with accepted Llama 3.2 license")
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

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=MODEL_DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **extra_kwargs,
    ).to(DEVICE)
    print(f"[model] loaded  params: {sum(p.numel() for p in model.parameters()):,}")

    # 3) SFT
    model, elapsed = run_sft(samples, tokenizer, model, cfg, output_dir, log_path)

    # 4) Inference check
    run_inference(model, tokenizer, args.dataset)

    # 5) Summary
    print_summary(args.dataset, len(samples), cfg["steps"], elapsed, log_path, cfg)


if __name__ == "__main__":
    main()
