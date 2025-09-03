#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural / Ember — Enhanced training & inference script (~330 lines)

What’s improved vs your draft:
- Robust YAML config merge (no parser misuse)
- Safe EarlyStopping on eval_loss; perplexity computed after evaluate()
- Custom ProfilerCallback instead of passing torch.profiler directly
- No unsupported lr_scheduler_kwargs; uses HF cosine schedule
- Logging & seeding hardened; clearer metrics file
- Safer unit tests; tokenizer decode check relaxed
"""

from __future__ import annotations
import os, math, json, logging, argparse, yaml, unittest
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainerArguments,
    set_seed,
    EarlyStoppingCallback,
)

# ---------- Optional PEFT / LoRA ----------
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

# ---------- Optional Weights & Biases ----------
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger("ember")

# ===============================================================
# (A) Optional scratch decoder (kept for tinkering / experiments)
# ===============================================================
class AdvancedTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        nhead: int = 16,
        num_layers: int = 24,
        dim_feedforward: int = 4096,
        max_len: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._init_rotary_positional_encoding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            dropout=dropout,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_rotary_positional_encoding(self, max_len: int, d_model: int):
        # NOTE: true RoPE is applied inside attention; this is a sinusoidal stand-in.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.decoder(x, memory if memory is not None else x, tgt_mask=tgt_mask)
        x = self.norm(x)
        return self.fc_out(x)

# ==================================
# (B) Tokenizer & Dataset utilities
# ==================================
def get_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_text_dataset(name: str = "wikitext", subset: str = "wikitext-103-raw-v1"):
    ds = load_dataset(name, subset)
    # If no validation split, carve a tiny one
    if "validation" not in ds:
        split = ds["train"].train_test_split(test_size=0.005, seed=42)
        return {"train": split["train"], "validation": split["test"]}
    return ds

def tokenize_function(batch, tok: AutoTokenizer):
    return tok(batch["text"])

def group_texts(examples: Dict[str, List[List[int]]], block_size: int) -> Dict[str, Any]:
    # Concatenate & chunk (standard LM objective)
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# ==================================
# (C) Model build & LoRA wrapper
# ==================================
def build_pretrained(model_name: str, gradient_checkpointing: bool = True) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model

def maybe_wrap_lora(
    model: PreTrainedModel,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Optional[List[str]],
) -> PreTrainedModel:
    if not use_lora:
        return model
    if not PEFT_AVAILABLE:
        logger.warning("LoRA requested but `peft` is not installed. Continuing without LoRA.")
        return model
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules or ["c_attn", "c_fc", "c_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def bf16_or_fp16() -> Dict[str, bool]:
    if torch.cuda.is_available():
        try:
            supported_bf16 = torch.cuda.is_bf16_supported()
            return {"bf16": supported_bf16, "fp16": not supported_bf16}
        except Exception:
            return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}

def try_compile(model: nn.Module) -> nn.Module:
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore
            logger.info("torch.compile enabled")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    return model

# ==================================
# (D) Profiler as a TrainerCallback
# ==================================
class ProfilerCallback(TrainerCallback):
    """Wrap torch.profiler in a HF-friendly callback."""
    def __init__(self, trace_dir: str = "./profile", wait=1, warmup=1, active=3, repeat=2):
        from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
        self._profile = profile(
            activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self._step = 0
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._profile.step()
        self._step += 1
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._profile.__enter__()
        logger.info("[profiler] started")
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._profile.__exit__(None, None, None)
        logger.info("[profiler] stopped")

# ==================================
# (E) Training pipeline
# ==================================
def train_pipeline(
    model_name: str,
    dataset_name: str,
    subset: str,
    save_dir: str,
    block_size: int,
    lr: float,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    weight_decay: float,
    warmup_ratio: float,
    eval_steps: int,
    save_steps: int,
    logging_steps: int,
    seed: int,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Optional[List[str]],
    resume: Optional[str],
    do_profile: bool,
    wandb_enabled: bool,
) -> Dict[str, Any]:
    set_seed(seed)
    if wandb_enabled and WANDB_AVAILABLE:
        wandb.init(project="ember", config={
            "model": model_name, "dataset": dataset_name, "subset": subset,
            "block": block_size, "lr": lr, "epochs": epochs, "batch": batch_size,
            "grad_accum": grad_accum, "weight_decay": weight_decay, "warmup_ratio": warmup_ratio,
            "use_lora": use_lora, "lora_r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
            "seed": seed
        })

    logger.info(f"Starting training: model={model_name} data={dataset_name}/{subset}")
    tok = get_tokenizer(model_name)
    raw = load_text_dataset(dataset_name, subset)
    tokenized = raw.map(lambda b: tokenize_function(b, tok), batched=True, remove_columns=["text"])
    lm_data = tokenized.map(lambda b: group_texts(b, block_size=block_size), batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    model = try_compile(maybe_wrap_lora(build_pretrained(model_name, True),
                                        use_lora, lora_r, lora_alpha, lora_dropout, target_modules))
    dtype_flags = bf16_or_fp16()

    args = TrainingArguments(
        output_dir=save_dir,
        do_train=True, do_eval=True,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps, save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",   # use eval_loss (lower is better)
        greater_is_better=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=min(batch_size, 4),
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        report_to=["wandb"] if (wandb_enabled and WANDB_AVAILABLE) else [],
        fp16=dtype_flags["fp16"], bf16=dtype_flags["bf16"],
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
    )

    callbacks: List[TrainerCallback] = [EarlyStoppingCallback(early_stopping_patience=5)]
    if do_profile:
        callbacks.append(ProfilerCallback())

    trainer = Trainer(
        model=model,
        tokenizer=tok,
        args=args,
        data_collator=collator,
        train_dataset=lm_data["train"],
        eval_dataset=lm_data["validation"],
        callbacks=callbacks,
        # We skip compute_metrics; rely on eval_loss for best model selection
    )

    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    # Save best full model (or adapters if LoRA)
    trainer.save_model(save_dir)
    if use_lora and PEFT_AVAILABLE and isinstance(trainer.model, PeftModel):
        trainer.model.save_pretrained(os.path.join(save_dir, "lora_adapters"))

    # Evaluate & compute perplexity from eval_loss
    eval_metrics = trainer.evaluate()
    eval_loss = float(eval_metrics.get("eval_loss", float("nan")))
    try:
        eval_ppl = math.exp(eval_loss) if not math.isnan(eval_loss) else float("nan")
    except OverflowError:
        eval_ppl = float("inf")

    metrics = {
        "eval_loss": eval_loss,
        "eval_perplexity": eval_ppl,
        "train_runtime": float(getattr(trainer.state, "train_runtime", 0.0)),
        "global_steps": int(getattr(trainer.state, "global_step", 0)),
    }
    logger.info(f"[metrics] {json.dumps(metrics, indent=2)}")
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if wandb_enabled and WANDB_AVAILABLE:
        wandb.log(metrics)
        wandb.finish()

    return {"model": trainer.model, "tokenizer": tok, "metrics": metrics}

# ==================================
# (F) Generation helper
# ==================================
@torch.inference_mode()
def generate_text(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    do_sample: bool = True,
    device: Optional[str] = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tok = tokenizer(prompt, return_tensors="pt").to(device)
    if hasattr(model, "generate"):
        out = model.generate(
            **tok,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    # Rare manual fallback
    input_ids = tok["input_ids"]
    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# ==================================
# (G) CLI & YAML config merge
# ==================================
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Neural / Ember Trainer")
    # data/model
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--model", type=str, default="gpt2-large")
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--subset", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--block", type=int, default=1024)
    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    # control
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--target_modules", type=str, default="")
    # io
    p.add_argument("--save", type=str, default="./results")
    p.add_argument("--resume", type=str, default=None)
    # generate-only
    p.add_argument("--generate", type=str, default=None)
    p.add_argument("--max_new", type=int, default=200)
    # extras
    p.add_argument("--profile", action="store_true")
    p.add_argument("--wandb", action="store_true")
    return p

def merge_yaml_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if not args.config:
        return args
    if not os.path.exists(args.config):
        logger.warning(f"Config not found: {args.config}")
        return args
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # Build a defaults map to only override values that are still on defaults
    defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
    for k, v in cfg.items():
        if not hasattr(args, k):
            continue
        cur = getattr(args, k)
        if cur == defaults.get(k):
            setattr(args, k, v)
    logger.info(f"Loaded config from {args.config}")
    return args

# ==================================
# (H) Minimal Unit Tests
# ==================================
class TestEmber(unittest.TestCase):
    def test_scratch_model_shapes(self):
        model = AdvancedTransformerModel(vocab_size=1000)
        x = torch.randint(0, 1000, (2, 12))
        y = model(x)
        self.assertEqual(y.shape, (2, 12, 1000))
    def test_tokenizer_basic(self):
        tok = AutoTokenizer.from_pretrained("gpt2")
        txt = "Hello, world!"
        ids = tok(txt)["input_ids"]
        self.assertTrue(isinstance(ids, list) and len(ids) > 0)

# ==================================
# (I) Main
# ==================================
def main():
    parser = make_parser()
    args = parser.parse_args()
    args = merge_yaml_config(args, parser)
    set_seed(args.seed)

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()] or None

    if args.generate:
        tok = get_tokenizer(args.model)
        mdl = build_pretrained(args.model, gradient_checkpointing=False)
        text = generate_text(
            mdl, tok, prompt=args.generate,
            max_new_tokens=args.max_new, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.05
        )
        print(text)
        return

    out = train_pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        subset=args.subset,
        save_dir=args.save,
        block_size=args.block,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum=args.grad_accum,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        resume=args.resume,
        do_profile=args.profile,
        wandb_enabled=args.wandb,
    )

    # Quick sample
    sample = generate_text(
        out["model"], out["tokenizer"],
        prompt="Once upon a time",
        max_new_tokens=args.max_new, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.05
    )
    print("\n=== SAMPLE ===\n", sample)

if __name__ == "__main__":
    main()