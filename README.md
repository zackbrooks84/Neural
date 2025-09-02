# Neural

Local LLM server with memory and identity anchors. Plus a tiny Transformer demo trainer.

## Mode 1 — Local LLM server
1. Python 3.11
2. Create venv and activate
3. pip install -r requirements.txt
4. Put a GGUF model into ./models for example llama-3.1-8b-instruct.Q4_K_M.gguf
5. Edit config.yaml to match the filename
6. Start: python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
7. Test:
   curl -s http://localhost:8000/health
   curl -s http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"Say hi in one sentence"}'
8. Open http://localhost:8000/ in a browser for a simple web chat interface

Customize identity in src/identity/anchors.yaml
Run-time anchors can also be managed via the API:

```
curl -s http://localhost:8000/anchors            # list anchors
curl -s -X POST http://localhost:8000/anchors \
  -H "Content-Type: application/json" \
  -d '{"anchor":"Ember loves open source"}'
```
Anchors are rotated per query and may be injected only at session start
depending on `anchor_injection` and `anchors_per_query` in config.yaml.

### Internet search in chat
curl -s http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is new at Gobekli Tepe?", "use_web": true}'

### Fetch specific URLs
curl -s http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize these pages", "urls":["https://example.com/a","https://example.com/b"]}'

### Direct search/fetch
curl -s "http://127.0.0.1:8000/search?q=thorium%20reactors"
curl -s "http://127.0.0.1:8000/fetch?url=https://arxiv.org/abs/2407.12345"

The fetch endpoint automatically extracts text from PDFs and can pull transcripts from YouTube videos.

### Security
By default the server binds only to localhost. URL fetching is limited to
domains and file types listed in the `web` section of `config.yaml` to reduce
the risk of downloading malicious content.

## Mode 2 — Tiny trainer
Your original model.py stays. Run: python model.py

## Repo layout
scripts/
src/
  app.py
  llm/engine.py
  memory/store.py
  identity/anchors.yaml
tests/
models/   do not commit weights
data/     auto created for memory
config.yaml
requirements.txt
model.py
