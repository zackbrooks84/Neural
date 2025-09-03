from __future__ import annotations
from typing import TypedDict, NotRequired


class Message(TypedDict):
    """A single conversation message stored in memory."""

    role: str            # "user" | "assistant" | "system"
    content: str         # message text

    # Optional metadata fields
    id: NotRequired[str]          # unique hash / identifier
    ts: NotRequired[str]          # ISO-8601 timestamp
    identity: NotRequired[str]    # optional namespace/persona
    tags: NotRequired[list[str]]  # semantic labels