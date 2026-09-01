"""Validated deny-read policy and OS-enforced shell sandboxing."""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


MAX_DENY_READ_PATTERNS = 128
MAX_DENY_READ_PATTERN_CHARS = 1024
MAX_DENY_READ_MATCHES = 8192
MAX_DENY_READ_SCAN_ENTRIES = 250_000
DEFAULT_GLOB_SCAN_MAX_DEPTH = 32
_GLOB_CHARS = frozenset("*?[")


def _contains_glob(value: str) -> bool:
    return any(character in value for character in _GLOB_CHARS)


def _absolute_pattern(raw: str, workspace: Path) -> str:
    expanded = os.path.expanduser(raw)
    if not os.path.isabs(expanded):
        expanded = os.path.join(workspace, expanded)
    return os.path.abspath(expanded)


@lru_cache(maxsize=4096)
def _match_parts(pattern: tuple[str, ...], path: tuple[str, ...]) -> bool:
    if not pattern:
        return not path
    head = pattern[0]
    if head == "**":
        return _match_parts(pattern[1:], path) or bool(
            path and _match_parts(pattern, path[1:])
        )
    return bool(
        path
        and fnmatch.fnmatchcase(path[0], head)
        and _match_parts(pattern[1:], path[1:])
    )


@lru_cache(maxsize=4096)
def _could_match_descendant(
    pattern: tuple[str, ...],
    path: tuple[str, ...],
) -> bool:
    """Return whether extending path could produce a policy match."""
    if not path:
        return bool(pattern)
    if not pattern:
        return False
    head = pattern[0]
    if head == "**":
        return _could_match_descendant(pattern[1:], path) or (
            _could_match_descendant(pattern, path[1:])
        )
    return fnmatch.fnmatchcase(path[0], head) and _could_match_descendant(
        pattern[1:], path[1:]
    )


@dataclass(frozen=True)
class DenyReadPattern:
    configured: str
    absolute: str
    parts: tuple[str, ...]
    has_glob: bool
    scan_root: Path

    @classmethod
    def compile(cls, raw: str, workspace: Path) -> "DenyReadPattern":
        if not isinstance(raw, str):
            raise ValueError("permissions.filesystem.deny_read must contain strings")
        value = raw.strip()
        if not value:
            raise ValueError(
                "permissions.filesystem.deny_read entries must not be empty"
            )
        if len(value) > MAX_DENY_READ_PATTERN_CHARS:
            raise ValueError(
                "permissions.filesystem.deny_read entry exceeds "
                f"{MAX_DENY_READ_PATTERN_CHARS} characters"
            )
        if "\0" in value:
            raise ValueError(
                "permissions.filesystem.deny_read entries must not contain NUL"
            )
        absolute = _absolute_pattern(value, workspace)
        path = Path(absolute)
        parts = path.parts
        has_glob = _contains_glob(absolute)
        fixed_parts: list[str] = []
        for part in parts:
            if _contains_glob(part):
                break
            fixed_parts.append(part)
        scan_root = Path(*fixed_parts)
        if has_glob and scan_root == Path(path.anchor):
            raise ValueError(
                "permissions.filesystem.deny_read globs require a non-root "
                "directory prefix"
            )
        return cls(value, absolute, parts, has_glob, scan_root)

    def matches(self, path: Path) -> bool:
        return _match_parts(self.parts, path.parts)


@dataclass(frozen=True)
class PreparedDeniedPath:
    path: Path
    exact: bool


@dataclass(frozen=True)
class FileAccessPolicy:
    workspace: Path
    patterns: tuple[DenyReadPattern, ...]
    glob_scan_max_depth: int = DEFAULT_GLOB_SCAN_MAX_DEPTH

    @classmethod
    def create(
        cls,
        workspace: Path,
        configured_patterns: object,
        glob_scan_max_depth: object = DEFAULT_GLOB_SCAN_MAX_DEPTH,
    ) -> "FileAccessPolicy":
        if not isinstance(configured_patterns, list) or any(
            not isinstance(item, str) for item in configured_patterns
        ):
            raise ValueError(
                "permissions.filesystem.deny_read must be an array of strings"
            )
        if len(configured_patterns) > MAX_DENY_READ_PATTERNS:
            raise ValueError(
                "permissions.filesystem.deny_read exceeds "
                f"{MAX_DENY_READ_PATTERNS} entries"
            )
        normalized_patterns = [item.strip() for item in configured_patterns]
        if len(set(normalized_patterns)) != len(normalized_patterns):
            raise ValueError(
                "permissions.filesystem.deny_read entries must be unique"
            )
        if (
            isinstance(glob_scan_max_depth, bool)
            or not isinstance(glob_scan_max_depth, int)
            or glob_scan_max_depth < 0
        ):
            raise ValueError(
                "permissions.filesystem.glob_scan_max_depth must be a "
                "non-negative integer"
            )
        root = workspace.expanduser().resolve()
        patterns = tuple(
            DenyReadPattern.compile(item, root) for item in normalized_patterns
        )
        return cls(root, patterns, glob_scan_max_depth)

    @property
    def enabled(self) -> bool:
        return bool(self.patterns)

    def denies(self, path: Path, *, logical_path: Path | None = None) -> bool:
        if not self.patterns:
            return False
        candidates = {Path(os.path.abspath(path))}
        candidates.add(path.resolve(strict=False))
        if logical_path is not None:
            candidates.add(Path(os.path.abspath(logical_path)))
            candidates.add(logical_path.resolve(strict=False))
        for candidate in candidates:
            for current in (candidate, *candidate.parents):
                if any(pattern.matches(current) for pattern in self.patterns):
                    return True
        return False

    def shell_payload(self) -> str:
        return json.dumps(
            {
                "workspace": str(self.workspace),
                "patterns": [pattern.configured for pattern in self.patterns],
                "glob_scan_max_depth": self.glob_scan_max_depth,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_shell_payload(cls, payload: str) -> "FileAccessPolicy":
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid shell filesystem policy") from exc
        if not isinstance(value, dict) or set(value) != {
            "workspace",
            "patterns",
            "glob_scan_max_depth",
        }:
            raise ValueError("invalid shell filesystem policy")
        workspace = value["workspace"]
        if not isinstance(workspace, str) or not os.path.isabs(workspace):
            raise ValueError("invalid shell filesystem policy workspace")
        return cls.create(
            Path(workspace),
            value["patterns"],
            value["glob_scan_max_depth"],
        )

    def prepare_denied_paths(self) -> tuple[PreparedDeniedPath, ...]:
        prepared: dict[Path, bool] = {}
        scanned = 0
        for pattern in self.patterns:
            if not pattern.has_glob:
                prepared[Path(pattern.absolute)] = True
                continue
            root = pattern.scan_root
            if not root.exists():
                continue
            if pattern.matches(root):
                prepared[root] = prepared.get(root, False)
                continue
            stack = [(root, 0)]
            while stack:
                directory, depth = stack.pop()
                try:
                    children = os.scandir(directory)
                except (NotADirectoryError, FileNotFoundError):
                    continue
                except PermissionError as exc:
                    raise RuntimeError(
                        f"cannot scan deny-read glob directory: {directory}"
                    ) from exc
                with children:
                    for child in children:
                        scanned += 1
                        if scanned > MAX_DENY_READ_SCAN_ENTRIES:
                            raise RuntimeError(
                                "deny-read glob scan exceeded "
                                f"{MAX_DENY_READ_SCAN_ENTRIES} entries"
                            )
                        child_path = Path(child.path)
                        if pattern.matches(child_path):
                            prepared[child_path] = prepared.get(
                                child_path, False
                            )
                            if len(prepared) > MAX_DENY_READ_MATCHES:
                                raise RuntimeError(
                                    "deny-read patterns matched more than "
                                    f"{MAX_DENY_READ_MATCHES} paths"
                            )
                            continue
                        could_match_below = _could_match_descendant(
                            pattern.parts, child_path.parts
                        )
                        if child.is_symlink() and could_match_below:
                            raise RuntimeError(
                                "cannot enforce deny-read glob through "
                                f"symlink: {child_path}"
                            )
                        if child.is_dir(follow_symlinks=False):
                            if depth < self.glob_scan_max_depth:
                                stack.append((child_path, depth + 1))
                            elif could_match_below:
                                raise RuntimeError(
                                    "deny-read glob scan exceeded depth "
                                    f"{self.glob_scan_max_depth}"
                                )
        selected: list[PreparedDeniedPath] = []
        for path, exact in sorted(
            prepared.items(), key=lambda item: (len(item[0].parts), str(item[0]))
        ):
            if any(path.is_relative_to(item.path) for item in selected):
                continue
            selected.append(PreparedDeniedPath(path, exact))
        return tuple(selected)


@dataclass
class ShellSandboxLaunch:
    argv: list[str]
    inherited_fds: list[int] = field(default_factory=list)
    placeholders: list[tuple[Path, int, int]] = field(default_factory=list)
    profile_path: Path | None = None

    def cleanup(self) -> None:
        for fd in self.inherited_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.inherited_fds.clear()
        for path, device, inode in reversed(self.placeholders):
            try:
                metadata = path.stat(follow_symlinks=False)
                if (
                    metadata.st_dev == device
                    and metadata.st_ino == inode
                    and stat.S_ISREG(metadata.st_mode)
                    and metadata.st_size == 0
                ):
                    path.unlink()
            except FileNotFoundError:
                pass
        self.placeholders.clear()
        if self.profile_path is not None:
            try:
                self.profile_path.unlink()
            except FileNotFoundError:
                pass
            self.profile_path = None


def _first_missing_component(path: Path) -> Path | None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            return current
    return None


def _symlink_component(path: Path) -> Path | None:
    if path == Path(path.anchor):
        return None
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return current
    return None


def _materialize_missing_target(
    path: Path,
    launch: ShellSandboxLaunch,
) -> Path | None:
    missing = _first_missing_component(path)
    if missing is None:
        return path
    try:
        fd = os.open(missing, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0)
    except FileNotFoundError:
        return None
    else:
        os.close(fd)
    metadata = missing.stat(follow_symlinks=False)
    launch.placeholders.append((missing, metadata.st_dev, metadata.st_ino))
    return missing


def _trusted_bubblewrap() -> Path:
    configured = shutil.which("bwrap")
    if configured is None:
        raise RuntimeError(
            "permissions.filesystem.deny_read requires bubblewrap on Linux"
        )
    try:
        executable = Path(configured).resolve(strict=True)
        metadata = executable.stat()
    except OSError as exc:
        raise RuntimeError("cannot validate bubblewrap executable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError(
            "bubblewrap must be root-owned and not group/world writable"
        )
    return executable


def _bubblewrap_launch(
    command: str,
    policy: FileAccessPolicy,
    denied: tuple[PreparedDeniedPath, ...],
) -> ShellSandboxLaunch:
    executable = _trusted_bubblewrap()
    launch = ShellSandboxLaunch(
        [
            str(executable),
            "--die-with-parent",
            "--bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--bind-try",
            "/dev/shm",
            "/dev/shm",
        ]
    )
    try:
        for item in denied:
            symlink = _symlink_component(item.path)
            if symlink is not None:
                raise RuntimeError(
                    "cannot enforce deny-read path through symlink: "
                    f"{symlink}"
                )
            target = item.path
            if not target.exists() and not target.is_symlink():
                if not item.exact:
                    continue
                materialized = _materialize_missing_target(target, launch)
                if materialized is None:
                    raise RuntimeError(
                        f"cannot materialize deny-read path: {target}"
                    )
                target = materialized
            if target.is_dir():
                launch.argv.extend(
                    ["--perms", "000", "--tmpfs", str(target)]
                )
                continue
            fd = os.open("/dev/null", os.O_RDONLY)
            os.set_inheritable(fd, True)
            launch.inherited_fds.append(fd)
            launch.argv.extend(
                ["--perms", "000", "--ro-bind-data", str(fd), str(target)]
            )
        launch.argv.extend(
            ["--chdir", str(policy.workspace), "--", "/bin/bash", "-c", command]
        )
        return launch
    except BaseException:
        launch.cleanup()
        raise


def _seatbelt_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _seatbelt_launch(
    command: str,
    policy: FileAccessPolicy,
    denied: tuple[PreparedDeniedPath, ...],
) -> ShellSandboxLaunch:
    executable = "/usr/bin/sandbox-exec"
    if not os.path.isfile(executable):
        raise RuntimeError(
            "permissions.filesystem.deny_read requires sandbox-exec on macOS"
        )
    rules = ["(version 1)", "(allow default)"]
    for item in denied:
        symlink = _symlink_component(item.path)
        if symlink is not None:
            raise RuntimeError(
                "cannot enforce deny-read path through symlink: "
                f"{symlink}"
            )
        kind = "subpath" if item.path.is_dir() else "literal"
        rules.append(
            "(deny file-read* file-write* "
            f'({kind} "{_seatbelt_quote(str(item.path))}"))'
        )
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="raptor-seatbelt-",
        suffix=".sb",
        delete=False,
    )
    try:
        handle.write("\n".join(rules) + "\n")
        handle.flush()
    finally:
        handle.close()
    profile_path = Path(handle.name)
    os.chmod(profile_path, 0o600)
    return ShellSandboxLaunch(
        [executable, "-f", str(profile_path), "/bin/bash", "-c", command],
        profile_path=profile_path,
    )


def build_shell_sandbox_launch(
    command: str,
    policy: FileAccessPolicy,
) -> ShellSandboxLaunch:
    if not policy.enabled:
        return ShellSandboxLaunch(["/bin/bash", "-c", command])
    denied = policy.prepare_denied_paths()
    if sys.platform.startswith("linux"):
        return _bubblewrap_launch(command, policy, denied)
    if sys.platform == "darwin":
        return _seatbelt_launch(command, policy, denied)
    raise RuntimeError(
        "permissions.filesystem.deny_read is unsupported on this platform"
    )
