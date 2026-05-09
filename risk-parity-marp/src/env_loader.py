"""Shared utility for loading the .env file (Python 3.14 dotenv has a load_dotenv() bug)."""
from __future__ import annotations
import os
from pathlib import Path


def load_env(env_path: Path | str | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ.

    Hand-rolled because python-dotenv 1.x trips an AssertionError under Python 3.14.
    """
    if env_path is None:
        # walk up from cwd until we find .env
        cur = Path.cwd().resolve()
        for parent in [cur, *cur.parents]:
            candidate = parent / ".env"
            if candidate.exists():
                env_path = candidate
                break
        else:
            return
    p = Path(env_path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")
