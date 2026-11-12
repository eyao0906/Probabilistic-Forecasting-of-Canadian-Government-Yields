"""Portable scratch locations for tests; never relax filesystem protections."""
import os
from pathlib import Path
import sys


def scratch_root() -> Path:
    """Create explicit TMPDIR, or the platform's Hermes scratch fallback.

    An invalid/unwritable explicit location is an error, not permission to
    silently write somewhere else or bypass a security check.
    """
    explicit = os.environ.get("TMPDIR")
    if explicit:
        root = Path(explicit)
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        root = base / "hermes" / "cache" / "scratch"
    else:
        root = Path.home() / ".hermes" / "cache" / "scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root
