#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural / Ember — Enhanced training & inference script (~300+ lines)

Features
- Pretrained causal LM load (HuggingFace)
- Optional LoRA PEFT (saves adapters)
- Cosine LR schedule, warmup, weight decay
- Gradient checkpointing, bf16/fp16, torch.compile (if available)
- Clean dataset pipeline with chunking to block_size
- Eval loop (perplexity), early stopping, best model load
- Resume-from-checkpoint, best-of-N saving
- Deterministic seeding
- CLI flags for quick experimentation
- Generation helper with temperature/top-p/top-k/repetition penalty
- Minimal “from-scratch” transformer scaffold preserved for tinkering

Usage (examples)
---------------
# Quick fine-tune with LoRA on wikitext-103, then generate
python model.py --model gpt2-large --epochs 2 --batch 4 --use_lora \
  --block 1024 --lr 5e-5 --grad-accum 4 --save ./results

# Resume training later
python model.py --resume ./results/checkpoint-2000

# Generate only (no training)
python model.py --generate "Once upon a time" --max-new 200
"""

from __future__ import annotations
import os
import math
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)

# Optional PEFT / LoRA
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# -----------------------------
# (A) Optional scratch model
# -----------------------------
class AdvancedTransformerModel(nn.Module):
    """
    Minimal “scratch” decoder to experiment with ideas.
    Not used by default; kept as a playground for Ember.
    """
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
            norm_first=True,  # Pre-norm for stability
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_rotary_positional_encoding(self, max_len: int, d_model: int):
        # Simplified sinusoidal PE “styled” like RoPE placeholder
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Note: real RoPE is applied during attention; this is a compact stand-in
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
        # Using x as memory if none is provided (decoder-only)
        x = self.decoder(x, memory if memory is not None else x, tgt_mask=tgt_mask)
        x = self.norm(x)
        return self.fc_out(x)


# -----------------------------
# (B) Data utilities
# -----------------------------
def get_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Ensure pad token exists for collator; many GPT2-like models lack it
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_text_dataset(name: str = "wikitext", subset: str = "wikitext-103-raw-v1"):
    """
    Returns a HuggingFace dataset dict with train/validation if available.
    """
    ds = load_dataset(name, subset)
    # Some variants may not have validation; create a small split if needed
    if "validation" not in ds:
        ds = ds["train"].train_test_split(test_size=0.005, seed=42)
        ds = {"train": ds["train"], "validation": ds["test"]}
    return ds


def tokenize_function(examples: Dict[str, List[str]], tok: AutoTokenizer) -> Dict[str, Any]:
    return tok(examples["text"])


def group_texts(examples: Dict[str, Any], block_size: int) -> Dict[str, Any]:
    """
    Concatenate then chunk into block_size.
    This is the standard LM training pattern for long sequences.
    """
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    # Drop remainder to avoid shorter last block (or pad if you prefer)
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


# -----------------------------
# (C) Model & training helpers
# -----------------------------
def build_pretrained(model_name: str, gradient_checkpointing: bool = True) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def maybe_wrap_lora(model: PreTrainedModel, use_lora: bool, lora_r: int, lora_alpha: int,
                    lora_dropout: float, target_modules: Optional[List[str]]) -> PreTrainedModel:
    if not use_lora:
        return model
    if not PEFT_AVAILABLE:
        print("[WARN] PEFT/LoRA requested but `peft` not installed. Continuing without.")
        return model
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules or ["c_attn", "c_fc", "c_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def bf16_or_fp16() -> Dict[str, bool]:
    # Prefer bf16 if supported; otherwise fp16 if CUDA is present
    if torch.cuda.is_available():
        try:
            # On Ampere+ GPUs, bf16 is usually available with torch >= 2.0
            return {"bf16": torch.cuda.is_bf16_supported(), "fp16": not torch.cuda.is_bf16_supported()}
        except Exception:
            return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def try_compile(model: nn.Module) -> nn.Module:
    # torch.compile may speed up training on PyTorch 2.x
    compile_ok = hasattr(torch, "compile")
    if compile_ok:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("[info] torch.compile enabled")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")
    return model


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # For causal LM, shift predictions and labels
    shift_predictions = predictions[:, :-1, :]
    shift_labels = labels[:, 1:]
    # Flatten
    shift_predictions = shift_predictions.reshape(-1, shift_predictions.shape[-1])
    shift_labels = shift_labels.reshape(-1)
    # Ignore index -100
    mask = shift_labels != -100
    shift_predictions = shift_predictions[mask]
    shift_labels = shift_labels[mask]
    # Compute loss
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_predictions, shift_labels)
    try:
        perplexity = math.exp(loss.item())
    except OverflowError:
        perplexity = float("inf")
    return {"perplexity": perplexity}


# -----------------------------
# (D) Training pipeline
# -----------------------------
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
    cosine_min_lr_ratio: float,
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
) -> Dict[str, Any]:
    set_seed(seed)

    tokenizer = get_tokenizer(model_name)
    raw = load_text_dataset(dataset_name, subset)

    # Tokenize then chunk
    tokenized = raw.map(lambda b: tokenize_function(b, tokenizer), batched=True, remove_columns=["text"])
    lm_datasets = tokenized.map(
        lambda b: group_texts(b, block_size=block_size),
        batched=True,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = build_pretrained(model_name=model_name, gradient_checkpointing=True)
    model = maybe_wrap_lora(model, use_lora, lora_r, lora_alpha, lora_dropout, target_modules)
    dtype_flags = bf16_or_fp16()
    model = try_compile(model)

    # Training arguments
    max_steps = -1  # use epochs
    args = TrainingArguments(
        output_dir=save_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="perplexity",
        greater_is_better=False,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=min(batch_size, 4),
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        logging_steps=logging_steps,
        fp16=dtype_flags["fp16"],
        bf16=dtype_flags["bf16"],
        dataloader_num_workers=2,
        report_to=[],  # keep quiet; integrate wandb later if you want
        max_steps=max_steps,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        lr_scheduler_kwargs={"eta_min": lr * cosine_min_lr_ratio},  # Set min LR for cosine
    )

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=5)
    ]

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=collator,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    if resume:
        print(f"[info] Resuming from checkpoint: {resume}")
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    # Save best model or LoRA adapters
    trainer.save_model(save_dir)
    if use_lora and PEFT_AVAILABLE and isinstance(trainer.model, PeftModel):
        trainer.model.save_pretrained(os.path.join(save_dir, "lora_adapters"))

    metrics = trainer.evaluate()
    metrics["train_runtime"] = round(trainer.state.total_flos / 1e12 if hasattr(trainer.state, "total_flos") else 0, 2)  # Approximate runtime if not available
    metrics["train_samples"] = trainer.state.max_steps * batch_size * grad_accum if trainer.state.max_steps > 0 else None
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[metrics]", metrics)
    return {"model": trainer.model, "tokenizer": tokenizer, "metrics": metrics}


# -----------------------------
# (E) Generation helpers
# -----------------------------
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

    # Fallback (rare) manual loop
    input_ids = tok["input_ids"]
    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# -----------------------------
# (F) CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neural / Ember Trainer")
    # data/model
    p.add_argument("--model", type=str, default="gpt2-large", help="HF model name or path")
    p.add_argument("--dataset", type=str, default="wikitext", help="HF dataset name")
    p.add_argument("--subset", type=str, default="wikitext-103-raw-v1", help="HF dataset subset")
    p.add_argument("--block", type=int, default=1024, help="sequence block size")
    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4, help="per-device batch size")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--cosine-min-lr-ratio", type=float, default=0.1, help="(kept for future custom schedulers)")
    # lora
    p.add_argument("--use_lora", action="store_true", help="enable LoRA PEFT")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument("--target_modules", type=str, default="", help="comma list of target module names")
    # io / resume
    p.add_argument("--save", type=str, default="./results")
    p.add_argument("--resume", type=str, default=None, help="checkpoint path")
    # generation only
    p.add_argument("--generate", type=str, default=None, help="prompt to generate text (no training if set)")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()] or None

    if args.generate:
        # Inference-only path: load model and generate
        tokenizer = get_tokenizer(args.model)
        model = build_pretrained(args.model, gradient_checkpointing=False)
        text = generate_text(
            model, tokenizer,
            prompt=args.generate,
            max_new_tokens=args.max_new if hasattr(args, "max_new") else 200,
            temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.05
        )
        print(text)
        return

    # Training path
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
        cosine_min_lr_ratio=args.cosine_min_lr_ratio,
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        seed=args.seed,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        resume=args.resume,
    )

    # Demo generation after training
    model = out["model"]
    tokenizer = out["tokenizer"]
    sample = generate_text(
        model, tokenizer,
        prompt="Once upon a time",
        max_new_tokens=200,
        temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.05
    )
    print("\n=== SAMPLE ===\n", sample)


if __name__ == "__main__":
    main()