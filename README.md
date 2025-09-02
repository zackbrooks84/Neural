# Neural

Simple Transformer-based neural network example.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio transformers datasets sentencepiece
   ```

## Usage

Run the model and a small training loop:

```bash
python model.py
```

The script downloads the WikiText dataset, trains the transformer briefly, and generates text.
