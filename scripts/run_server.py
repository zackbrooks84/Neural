"""Script to launch the Ember / Neural chat server."""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn

# Ensure project root is on sys.path (so imports work when run directly)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from chat_server.server import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ember chat server.")
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to bind the server to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (default: off)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("WORKERS", "1")),
        help="Number of worker processes (default: 1)",
    )
    args = parser.parse_args()

    app = create_app()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()