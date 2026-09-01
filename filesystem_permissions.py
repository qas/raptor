"""Validated deny-read policy and bounded filesystem resolution."""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


MAX_DENY_READ_PATTERNS = 128
MAX_DENY_READ_PATTERN_CHARS = 1024
MAX_DENY_READ_MATCHES = 1024
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
def _glob_closure(
    pattern: tuple[str, ...],
    states: frozenset[int],
) -> frozenset[int]:
    expanded = set(states)
    pending = list(states)
    while pending:
        position = pending.pop()
        if position < len(pattern) and pattern[position] == "**":
            following = position + 1
            if following not in expanded:
                expanded.add(following)
                pending.append(following)
    return frozenset(expanded)


@lru_cache(maxsize=4096)
def _advance_glob_states(
    pattern: tuple[str, ...],
    states: frozenset[int],
    segment: str,
) -> frozenset[int]:
    following: set[int] = set()
    for position in states:
        if position >= len(pattern):
            continue
        expected = pattern[position]
        if expected == "**":
            following.add(position)
        elif fnmatch.fnmatchcase(segment, expected):
            following.add(position + 1)
    return _glob_closure(pattern, frozenset(following))


@lru_cache(maxsize=4096)
def _glob_states_after(
    pattern: tuple[str, ...],
    path: tuple[str, ...],
) -> frozenset[int]:
    states = _glob_closure(pattern, frozenset({0}))
    for segment in path:
        states = _advance_glob_states(pattern, states, segment)
        if not states:
            break
    return states


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
        return self.accepts(self.states_after(path))

    def states_after(self, path: Path) -> frozenset[int]:
        return _glob_states_after(self.parts, path.parts)

    def advance(
        self,
        states: frozenset[int],
        segment: str,
    ) -> frozenset[int]:
        return _advance_glob_states(self.parts, states, segment)

    def accepts(self, states: frozenset[int]) -> bool:
        return len(self.parts) in states

    def can_match_descendant(self, states: frozenset[int]) -> bool:
        return any(position < len(self.parts) for position in states)


@dataclass(frozen=True)
class PreparedDeniedPath:
    path: Path
    materialize_if_missing: bool


_ActivePattern = tuple[DenyReadPattern, frozenset[int]]
_ActivePatterns = tuple[_ActivePattern, ...]


@dataclass
class _DeniedPathResolver:
    patterns: tuple[DenyReadPattern, ...]
    max_depth: int
    prepared: dict[Path, bool] = field(default_factory=dict, init=False)
    scanned_entries: int = field(default=0, init=False)

    def resolve(self) -> tuple[PreparedDeniedPath, ...]:
        scan_groups: dict[Path, list[DenyReadPattern]] = {}
        for pattern in self.patterns:
            if pattern.has_glob:
                scan_groups.setdefault(pattern.scan_root, []).append(pattern)
            else:
                self._deny(
                    Path(pattern.absolute),
                    materialize_if_missing=True,
                )
        for root, patterns in scan_groups.items():
            self._scan(root, tuple(patterns))
        selected: list[PreparedDeniedPath] = []
        for path, materialize_if_missing in sorted(
            self.prepared.items(),
            key=lambda item: (len(item[0].parts), str(item[0])),
        ):
            if any(path.is_relative_to(item.path) for item in selected):
                continue
            selected.append(
                PreparedDeniedPath(path, materialize_if_missing)
            )
        return tuple(selected)

    def _deny(
        self,
        path: Path,
        *,
        materialize_if_missing: bool = False,
    ) -> None:
        resolved = path.resolve(strict=False)
        self.prepared[resolved] = (
            self.prepared.get(resolved, False) or materialize_if_missing
        )
        if len(self.prepared) > MAX_DENY_READ_MATCHES:
            raise RuntimeError(
                "deny-read patterns matched more than "
                f"{MAX_DENY_READ_MATCHES} paths"
            )

    def _scan(
        self,
        root: Path,
        patterns: tuple[DenyReadPattern, ...],
    ) -> None:
        try:
            root_metadata = root.stat()
        except (FileNotFoundError, NotADirectoryError):
            return
        except PermissionError:
            self._deny(root)
            return
        except OSError as exc:
            raise RuntimeError(
                f"cannot inspect deny-read glob root: {root}"
            ) from exc
        active = tuple(
            (pattern, pattern.states_after(root)) for pattern in patterns
        )
        if any(pattern.accepts(states) for pattern, states in active):
            self._deny(root)
            return
        active = tuple(
            (pattern, states)
            for pattern, states in active
            if pattern.can_match_descendant(states)
        )
        if not active:
            return
        root_identity = (root_metadata.st_dev, root_metadata.st_ino)
        visited = {(root_identity, self._state_signature(active))}
        stack = [(root, 0, active)]
        while stack:
            directory, depth, directory_states = stack.pop()
            try:
                children = os.scandir(directory)
            except (NotADirectoryError, FileNotFoundError):
                continue
            except PermissionError:
                self._deny(directory)
                continue
            with children:
                for child in children:
                    self.scanned_entries += 1
                    if self.scanned_entries > MAX_DENY_READ_SCAN_ENTRIES:
                        raise RuntimeError(
                            "deny-read glob scan exceeded "
                            f"{MAX_DENY_READ_SCAN_ENTRIES} entries"
                        )
                    child_path = Path(child.path)
                    matched, child_states = self._advance_patterns(
                        directory_states,
                        child.name,
                    )
                    if matched:
                        self._deny(
                            child_path,
                            materialize_if_missing=child.is_symlink(),
                        )
                        continue
                    if not child_states:
                        continue
                    try:
                        is_directory = child.is_dir(follow_symlinks=True)
                    except OSError as exc:
                        raise RuntimeError(
                            "cannot inspect deny-read glob path: "
                            f"{child_path}"
                        ) from exc
                    if not is_directory:
                        continue
                    try:
                        child_metadata = child.stat(follow_symlinks=True)
                    except PermissionError:
                        self._deny(child_path)
                        continue
                    except OSError as exc:
                        raise RuntimeError(
                            "cannot inspect deny-read glob directory: "
                            f"{child_path}"
                        ) from exc
                    child_identity = (
                        child_metadata.st_dev,
                        child_metadata.st_ino,
                    )
                    signature = self._state_signature(child_states)
                    visit_key = (child_identity, signature)
                    if visit_key in visited:
                        continue
                    if depth >= self.max_depth:
                        self._deny(child_path)
                        continue
                    visited.add(visit_key)
                    stack.append((child_path, depth + 1, child_states))

    @staticmethod
    def _advance_patterns(
        active: _ActivePatterns,
        segment: str,
    ) -> tuple[bool, _ActivePatterns]:
        descendants: list[_ActivePattern] = []
        for pattern, states in active:
            following = pattern.advance(states, segment)
            if pattern.accepts(following):
                return True, ()
            if pattern.can_match_descendant(following):
                descendants.append((pattern, following))
        return False, tuple(descendants)

    @staticmethod
    def _state_signature(
        active: _ActivePatterns,
    ) -> tuple[tuple[tuple[str, ...], frozenset[int]], ...]:
        return tuple((pattern.parts, states) for pattern, states in active)


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

    def supervisor_payload(self) -> str:
        return json.dumps(
            {
                "workspace": str(self.workspace),
                "patterns": [pattern.configured for pattern in self.patterns],
                "glob_scan_max_depth": self.glob_scan_max_depth,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_supervisor_payload(cls, payload: str) -> "FileAccessPolicy":
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
        return _DeniedPathResolver(
            self.patterns,
            self.glob_scan_max_depth,
        ).resolve()
