# Neural

Neural is a local language model server with long‑term memory, identity
anchors, web retrieval tools and a tiny Transformer training demo.

## Features

- **FastAPI server** for running GGUF models locally with a simple chat
  endpoint and web UI.
- **Vector memory** that stores conversation snippets on disk and retrieves
  relevant context for new queries.
- **Identity anchors** that rotate into the system prompt. Anchors live in
  `src/identity/anchors.yaml` and can also be managed at run time.
- **Web tools** for DuckDuckGo search and robust URL fetching with PDF and
  YouTube transcript extraction.
- **Static client** in `docs/` suitable for GitHub Pages.
- **Tiny Transformer trainer** in `model.py` as a minimal example.

## Getting started

1. Python 3.11
2. Create and activate a virtual environment
3. `pip install -r requirements.txt`
4. Place a GGUF model in `./models/` (e.g. `llama-3.1-8b-instruct.Q4_K_M.gguf`)
5. Update `config.yaml` to reference the model filename
6. Launch the server:

   ```bash
   python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
   ```

### Quick test

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Say hi in one sentence"}'
```

Open `http://localhost:8000/` for a minimal web chat client.

## Customizing identity and memory

Edit `src/identity/anchors.yaml` to define default anchors. Manage anchors via
API calls at run time:

```bash
curl -s http://localhost:8000/anchors
curl -s -X POST http://localhost:8000/anchors \
  -H "Content-Type: application/json" \
  -d '{"anchor":"Ember loves open source"}'
```

Anchors are rotated per query and may be injected only at session start based on
`anchor_injection` and `anchors_per_query` in `config.yaml`. Conversation memory
is persisted to `data/` and retrieved as context for future queries.

## Web search and URL fetching

```bash
curl -s http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is new at Göbekli Tepe?", "use_web": true}'

curl -s http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize these pages", "urls":["https://example.com/a","https://example.com/b"]}'

curl -s "http://127.0.0.1:8000/search?q=thorium%20reactors"
curl -s "http://127.0.0.1:8000/fetch?url=https://arxiv.org/abs/2407.12345"
```

`fetch` automatically extracts text from PDFs and can pull transcripts from
YouTube videos. Allowed domains and file types are controlled in the `web`
section of `config.yaml`.

## GitHub Pages web UI

Static files in `docs/` provide a lightweight chat client. Enable GitHub Pages
for the repository and point it at the `docs` folder. The included `.nojekyll`
file ensures assets are served as‑is. The page auto‑detects the API base URL and
defaults to `localhost:8000`; a field is provided to override this if needed.

## Tiny Transformer trainer

`model.py` contains a minimal Transformer training example:

```bash
python model.py
```

## Security

By default the server binds only to localhost. URL fetching is limited to
domains and file types listed in the `web` section of `config.yaml` to reduce
the risk of downloading malicious content.

## Repository layout

```
config.yaml
data/       # auto created for memory
docs/       # static web UI
models/     # place GGUF weights here (not committed)
scripts/
src/
  app.py
  llm/engine.py
  memory/store.py
  identity/anchors.yaml
tests/
requirements.txt
model.py
```

## License

[MIT License](LICENSE)
