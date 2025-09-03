import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import math

class AdvancedTransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 1024, nhead: int = 16,
                 num_layers: int = 24, dim_feedforward: int = 4096,
                 max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._init_rotary_positional_encoding(max_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            dropout=dropout,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_rotary_positional_encoding(self, max_len: int, d_model: int):
        # Rotary Position Embedding (RoPE) for state-of-the-art positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0))

    def _init_weights(self):
        # Improved weight initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, memory: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        # Causal mask for autoregressive modeling
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer_decoder(x, memory if memory else x, tgt_mask=tgt_mask)
        x = self.norm(x)
        return self.fc_out(x)

def prepare_data(tokenizer: AutoTokenizer, dataset_name: str = "wikitext",
                 subset: str = "wikitext-103-raw-v1", block_size: int = 1024):
    # Use larger subset for better training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(dataset_name, subset)

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=block_size,
            return_attention_mask=True
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    tokenized = dataset["train"].map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

def train_model(model: nn.Module, dataset, epochs: int = 5, batch_size: int = 8, use_peft: bool = False) -> None:
    if use_peft and isinstance(model, PreTrainedModel):
        # Use PEFT (LoRA) for efficient fine-tuning on large models
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn", "c_proj", "c_fc"],  # For GPT-like models
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # For larger effective batch size
        learning_rate=5e-5,
        fp16=True,  # Mixed precision for efficiency
        save_steps=1000,
        logging_steps=100,
        weight_decay=0.01,
        warmup_steps=500,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

def generate_text(model: nn.Module, tokenizer: AutoTokenizer, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
    model.eval()
    tokens = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "generate"):
        generated = model.generate(
            **tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True
        )
        return tokenizer.decode(generated[0], skip_special_tokens=True)
    input_ids = tokens["input_ids"].to(next(model.parameters()).device)
    for _ in range(max_tokens):
        logits = model(input_ids)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0])

def load_pretrained_model(model_name: str = "gpt2-large") -> PreTrainedModel:
    """Load a larger pretrained causal language model from Hugging Face."""
    return AutoModelForCausalLM.from_pretrained(model_name)

if __name__ == "__main__":
    model_name = "gpt2-large"  # Upgraded to larger model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = prepare_data(tokenizer, subset="wikitext-103-raw-v1")  # Larger dataset
    model = load_pretrained_model(model_name)
    train_model(model, tokenized_dataset, epochs=3, batch_size=4, use_peft=True)  # Efficient fine-tuning
    print(generate_text(model, tokenizer, "Once upon a time", max_tokens=200))