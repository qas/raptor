"""Shared loader for the single Raptor TOML configuration document."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from runtime_paths import RAPTOR_HOME


CONFIG_PATH = Path(
    os.getenv("RAPTOR_CONFIG", str(RAPTOR_HOME / "config.toml"))
).expanduser().resolve()

ROOT_FIELDS = {
    "model_provider",
    "model",
    "model_providers",
    "network",
    "permissions",
    "chat",
    "telegram",
    "responses_server",
    "subagents",
    "tools",
    "shell",
    "state",
    "compaction",
}


def load_config_document(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            loaded = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Could not load Raptor config: {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Raptor config root must be a table: {path}")
    unknown = sorted(set(loaded) - ROOT_FIELDS)
    if unknown:
        raise ValueError(f"Unknown Raptor config fields: {', '.join(unknown)}")
    return loaded


def config_section(
    document: dict[str, Any],
    name: str,
    allowed: set[str],
) -> dict[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a table")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(unknown)}")
    return value
