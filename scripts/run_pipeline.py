"""Thin script wrapper for the pipeline CLI.

Prefer the installed entrypoint after ``uv sync``:

    uv run ftx-crash
"""

from __future__ import annotations

from ftx_crash.cli import main

if __name__ == "__main__":
    main()
