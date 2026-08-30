"""Create and load immutable Raptor build identity."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import tomllib


_BUILD_INFO_NAME = ".raptor-build.json"
_RELEASE_TAG = re.compile(
    r"^v(?P<version>(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:alpha|beta|rc)\.(?:0|[1-9][0-9]*))?)$"
)
_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class BuildInfo:
    """Version, channel, and source revision embedded in a build."""

    channel: str
    version: str | None
    revision: str | None = None
    built_at: str | None = None

    def display(self) -> str:
        if self.revision is None:
            if self.version is None:
                raise ValueError("source build identity requires a version")
            return self.version
        revision = self.revision[:8]
        if self.channel == "nightly":
            if self.built_at is None:
                raise ValueError("nightly build identity requires built_at")
            return f"nightly ({revision}, {self.built_at[:10]})"
        if self.version is None:
            raise ValueError("release build identity requires a version")
        return f"{self.version} ({revision})"


def release_build_info(tag: str, revision: str, built_at: str) -> BuildInfo:
    """Return validated metadata for a human-selected release tag."""
    matched = _RELEASE_TAG.fullmatch(tag)
    if matched is None:
        raise ValueError(
            "release tag must be vMAJOR.MINOR.PATCH or end in "
            "-alpha.N, -beta.N, or -rc.N"
        )
    _validate_revision(revision)
    _validate_built_at(built_at)
    version = matched.group("version")
    channel = "prerelease" if "-" in version else "stable"
    return BuildInfo(channel, version, revision, built_at)


def nightly_build_info(revision: str, built_at: str) -> BuildInfo:
    """Return validated metadata for a build of the main branch."""
    _validate_revision(revision)
    _validate_built_at(built_at)
    return BuildInfo("nightly", None, revision, built_at)


def load_build_info(directory: Path) -> BuildInfo:
    """Load generated build identity, falling back for source checkouts."""
    generated = directory / _BUILD_INFO_NAME
    if generated.is_file():
        data = json.loads(generated.read_text(encoding="utf-8"))
        info = BuildInfo(**data)
        if info.channel == "nightly":
            return nightly_build_info(info.revision or "", info.built_at or "")
        if info.version is None:
            raise ValueError("release build identity requires a version")
        tag = f"v{info.version}"
        validated = release_build_info(
            tag,
            info.revision or "",
            info.built_at or "",
        )
        if validated.channel != info.channel:
            raise ValueError("build channel does not match its version")
        return validated
    project = directory / "pyproject.toml"
    with project.open("rb") as handle:
        version = str(tomllib.load(handle)["project"]["version"])
    return BuildInfo("source", version)


def write_build_info(info: BuildInfo, destination: Path) -> None:
    """Write deterministic metadata consumed by source and frozen builds."""
    destination.write_text(
        json.dumps(asdict(info), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _validate_revision(revision: str) -> None:
    if _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be a full lowercase Git commit SHA")


def _validate_built_at(built_at: str) -> None:
    try:
        parsed = datetime.fromisoformat(built_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("built_at must be an ISO 8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("built_at must include a timezone")


def main() -> int:
    """Generate the build metadata file used by release automation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("channel", choices=("nightly", "release"))
    parser.add_argument("revision")
    parser.add_argument("built_at")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--tag")
    args = parser.parse_args()
    if args.channel == "release":
        if args.tag is None:
            parser.error("release builds require --tag")
        info = release_build_info(args.tag, args.revision, args.built_at)
    else:
        if args.tag is not None:
            parser.error("nightly builds do not accept --tag")
        info = nightly_build_info(args.revision, args.built_at)
    write_build_info(info, args.destination)
    print(info.display())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
