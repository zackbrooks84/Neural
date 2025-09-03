"""Chat server package for running a local GGUF (or HF) model with disk memory.

This package is expected to provide a FastAPI application factory named
``create_app`` inside ``chat_server/server.py`` (see :func:`create_app`).

Typical usage
-------------
from chat_server import create_app
app = create_app()

or, from the provided launcher:

python scripts/run_server.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from typing import Callable, Optional

__all__ = ["create_app", "__version__", "get_version"]

# ---------------------------------------------------------------------
# Version handling
# ---------------------------------------------------------------------
__version__ = "0.1.0"

def get_version() -> str:
    """Return the package version."""
    return __version__

# ---------------------------------------------------------------------
# App factory export (friendly import error if missing)
# ---------------------------------------------------------------------
def _missing_create_app(*args, **kwargs):
    raise ImportError(
        "chat_server.server.create_app not found. "
        "Ensure you have a 'server.py' module inside 'src/chat_server/' "
        "that defines a FastAPI application factory:\n\n"
        "    def create_app() -> FastAPI:\n"
        "        ...\n"
    )

try:
    # Prefer importing at package import time for clearer stack traces.
    from .server import create_app as _create_app  # type: ignore
except Exception:
    # Defer the failure until someone actually calls create_app(), so
    # `import chat_server` still works in environments that only need metadata.
    _create_app = None  # type: ignore


def create_app(*args, **kwargs):
    """Return a configured FastAPI application.

    This forwards to :func:`chat_server.server.create_app`. If that function
    cannot be imported, a helpful ImportError is raised.
    """
    if _create_app is None:
        return _missing_create_app(*args, **kwargs)
    return _create_app(*args, **kwargs)