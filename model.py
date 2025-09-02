import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer

class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048,
                 max_len: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return self.fc_out(x)


def prepare_data(tokenizer: AutoTokenizer, dataset_name: str = "wikitext",
                 subset: str = "wikitext-2-raw-v1", block_size: int = 128):
    dataset = load_dataset(dataset_name, subset)

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=block_size,
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    tokenized = dataset["train"].map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )
    tokenized.set_format(type="torch", columns=["input_ids", "labels"])
    return tokenized


def train_model(model: nn.Module, dataset, epochs: int = 1, batch_size: int = 4) -> None:
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs = batch["input_ids"]
            targets = batch["labels"]
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss {loss.item():.4f}")


def generate_text(model: nn.Module, tokenizer: AutoTokenizer, prompt: str, max_tokens: int = 50) -> str:
    model.eval()
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    for _ in range(max_tokens):
        logits = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens[0])


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenized_dataset = prepare_data(tokenizer)
    model = TransformerModel(vocab_size=tokenizer.vocab_size)
    train_model(model, tokenized_dataset, epochs=1, batch_size=2)
    print(generate_text(model, tokenizer, "Once upon a time"))
