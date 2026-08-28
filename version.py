"""Project version loaded from the adjacent canonical pyproject file."""

from pathlib import Path
import tomllib


_PROJECT = Path(__file__).with_name("pyproject.toml")
with _PROJECT.open("rb") as _handle:
    VERSION = str(tomllib.load(_handle)["project"]["version"])
