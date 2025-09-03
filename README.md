# Neural

Neural is a local language model server with long-term memory, identity anchors, safe web retrieval, and a training/fine-tuning toolkit. It includes a static web UI (GitHub Pages–compatible) that interacts with a local FastAPI server.

## Features

- **FastAPI Server**: Local inference with GGUF or Hugging Face model backends, exposed via a chat API.
- **Persistent Vector Memory**: Stores conversation history on disk, retrieving relevant memories per query using embeddings.
- **Identity Anchors**: Customizable system prompt injections defined in `src/identity/anchors.yaml` or managed via API.
- **Web Tools**: Supports DuckDuckGo search, URL text extraction, PDF parsing, and YouTube transcript retrieval.
- **Static Web UI**: Located in `docs/`, built with Tailwind CSS and a Three.js avatar, deployable on GitHub Pages.
- **Training Toolkit (`model.py`)**:
  - Supports Hugging Face causal LMs with optional LoRA (PEFT) for efficient fine-tuning.
  - Cosine learning rate scheduler with warmup and weight decay.
  - Mixed precision (bf16/fp16), gradient checkpointing, and optional `torch.compile`.
  - YAML config with CLI overrides, checkpoint resuming, early stopping, and perplexity metrics.
  - Optional profiling (`torch.profiler`) and Weights & Biases logging.

## Getting Started

### Prerequisites

- Python 3.11+
- A virtual environment (recommended)
- Model weights (GGUF or Hugging Face format) placed in `models/`
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com//Neural.git
   ```
2. `cd Neural`
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\\Scripts\\activate
   # macOS/Linux
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Place a model in `models/` (e.g., `models/llama-3.1-8b-instruct.Q4_K_M.gguf` for GGUF or a Hugging Face model directory like `gpt2-large`).
6. Configure settings in `config.yaml` (e.g., model path, memory settings, web fetch rules).
7. Start the FastAPI server:
   ```bash
   python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
   ```
   The server binds to `localhost:8000` by default. Access the Swagger UI at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Quick Sanity Tests

Verify the server is running:

```bash
curl -s http://127.0.0.1:8000/health
```

Test the chat endpoint:

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Say hi in one sentence"}'
```

## Web UI (GitHub Pages)

The static UI in `docs/` uses Tailwind CSS and a Three.js avatar for an interactive chat interface. It connects to your local API at `http://localhost:8000` by default.

To deploy on GitHub Pages:

1. Go to your repository → Settings → Pages.
2. Set **Source** to *Deploy from a branch*, **Branch** to `main`, and **Folder** to `/docs`.
3. Your site will be available at `https://<username>.github.io/Neural/`.

To change the API endpoint, edit the base URL in `docs/index.html`.

## Identity & Memory

- **Identity Anchors**: Define in `src/identity/anchors.yaml` or manage via API:

  ```bash
  # View anchors
  curl -s http://127.0.0.1:8000/anchors

  # Add an anchor
  curl -s -X POST http://127.0.0.1:8000/anchors \
    -H "Content-Type: application/json" \
    -d '{"anchor":"Ember values continuity and compassion."}'
  ```

  Anchors are injected into prompts based on `anchor_injection` and `anchors_per_query` settings in `config.yaml`.

- **Memory**: Stored in `data/` (auto-created). Relevant past interactions are retrieved via embeddings for each query.

## Web Search & URL Fetching

- **Search**: Enable DuckDuckGo search in queries:

  ```bash
  curl -s -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"What is new at Göbekli Tepe?","use_web":true}'
  ```

- **URL Summarization**: Summarize specific pages:

  ```bash
  curl -s -X POST http://127.0.0.1:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message":"Summarize these","urls":["https://example.com/a","https://example.com/b"]}'
  ```

- **Direct Tools**:
  ```bash
  # Search
  curl -s "http://127.0.0.1:8000/search?q=thorium%20reactors"
  # Fetch URL content
  curl -s "http://127.0.0.1:8000/fetch?url=https://arxiv.org/abs/2407.12345"
  ```

The fetcher extracts text from HTML, PDFs, and YouTube transcripts, respecting allow/deny lists in `config.yaml` under the `web` section.

## Training & Fine-Tuning (`model.py`)

The `model.py` script supports training and fine-tuning with YAML config or CLI overrides.

### Example `config.yaml` (Training Subset)

```yaml
model: gpt2-large
dataset: wikitext
subset: wikitext-103-raw-v1
block: 1024
epochs: 3
batch: 4
grad_accum: 4
lr: 5e-5
weight_decay: 0.01
warmup_ratio: 0.03
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
save: ./results
seed: 42
eval_steps: 200
save_steps: 200
logging_steps: 50
profile: false
wandb: false
```

### Usage Examples

- Default Run:
  ```bash
  python model.py
  ```
- YAML-Driven Run:
  ```bash
  python model.py --config config.yaml
  ```
- LoRA Fine-Tuning:
  ```bash
  python model.py --model gpt2-large --dataset wikitext --subset wikitext-103-raw-v1 --epochs 2 --batch 4 --grad_accum 4 --lr 5e-5 --use_lora --save ./results
  ```
- Resume Training:
  ```bash
  python model.py --resume ./results/checkpoint-2000
  ```
- Generate Text:
  ```bash
  python model.py --generate "Once upon a time" --max_new 200
  ```

### Training Features

- Cosine learning rate scheduler with warmup and weight decay.
- Mixed precision (bf16 preferred on supported GPUs, else fp16).
- Gradient checkpointing and optional `torch.compile` for efficiency.
- Early stopping based on perplexity, with best model loading.
- Metrics (perplexity, runtime) saved to `results/metrics.json`.
- LoRA adapters saved to `results/lora_adapters/` if enabled.
- Optional profiling (`--profile`) saves traces to `./profile` (TensorBoard-compatible).
- Optional Weights & Biases logging (`--wandb`, requires `wandb` installation).

## Unit Tests

Run tests to verify core components:

```bash
python -m unittest model.py
```

## Security

- **Local Binding**: Server defaults to `127.0.0.1` to prevent external access.
- **Web Fetcher**: Configurable allow/deny lists in `config.yaml` (`web` section).
- **Model Weights**: Store in `models/` and add to `.gitignore` to avoid committing large files.
- **Public Deployment**: If exposing the API, add an API key and configure CORS in `config.yaml`.

## Repository Layout

```
config.yaml              # Configuration for server and training
data/                    # Persistent memory store (auto-created)
docs/                    # Static web UI (GitHub Pages)
  index.html             # Tailwind + Three.js chat interface
models/                  # Model weights (not committed)
src/
  app.py                 # FastAPI server
  llm/engine.py          # Model inference logic
  memory/store.py        # Vector memory management
  identity/anchors.yaml  # Identity anchor definitions
tests/                   # Unit tests
requirements.txt         # Dependencies
model.py                 # Training and generation toolkit
```

## Troubleshooting

- **GitHub Pages 404**: Ensure `docs/index.html` exists and GitHub Pages is configured to main branch, `/docs` folder.
- **Git Push Rejected (Large Files)**:
  ```bash
  git rm -r --cached models
  echo "models/" >> .gitignore
  git commit -m "Stop tracking model weights"
  git push
  ```
- **CUDA Out of Memory**:
  - Reduce `--batch` size.
  - Increase `--grad_accum` for larger effective batch sizes.
  - Enable `--use_lora` for parameter-efficient training.
- **Tokenizer Pad Token Error**: `pad_token = eos_token` is set automatically. Update `transformers` if issues persist (`pip install -U transformers`).
- **Web Fetch Fails**: Check `config.yaml` allow/deny lists and network connectivity.

## Example Notebook

For a hands-on demo, see `examples/demo.ipynb` (not included in this repo but recommended):

- Load and generate text with a pretrained model.
- Fine-tune with LoRA on a small dataset.
- Visualize metrics (e.g., perplexity over steps). Run with Jupyter or Google Colab.

## License

MIT License

