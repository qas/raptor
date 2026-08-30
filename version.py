"""Raptor build identity."""

from pathlib import Path

from build_info import load_build_info


BUILD_INFO = load_build_info(Path(__file__).parent)


def display_version() -> str:
    """Return the human-readable version and immutable build revision."""
    return BUILD_INFO.display()
