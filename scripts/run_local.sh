#!/usr/bin/env bash
# =============================================================
#  Ember / Neural - Local Server Launcher (Linux / macOS)
# =============================================================

set -euo pipefail

# Default host and port (can be overridden via args or env vars)
HOST="${1:-${HOST:-127.0.0.1}}"
PORT="${2:-${PORT:-8000}}"

echo ""
echo "==========================================="
echo "   Starting Ember Local Server"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "==========================================="
echo ""

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

# Run the FastAPI app with uvicorn
# --reload for dev hot reloading (disable for production)
exec python -m uvicorn src.app:app \
  --host "$HOST" \
  --port "$PORT" \
  --reload