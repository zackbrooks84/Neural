"""Script to launch the chat server."""
from __future__ import annotations

import uvicorn

from chat_server.server import create_app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
