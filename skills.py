"""Progressive discovery and loading for workspace skill entries."""

import asyncio
import ast
import heapq
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import AGENT_WORKDIR, FILESYSTEM_POLICY, MAX_TOOL_OUTPUT
from raptor.state.storage import (
    FileTooLargeError,
    read_bytes_bounded,
    read_text_bounded,
    write_bytes_exclusive_atomic,
)


SKILLS_ROOTS = (
    AGENT_WORKDIR / ".raptor" / "skills",
    AGENT_WORKDIR / ".agent" / "skills",
)
MAX_DISCOVERED_SKILLS = 256
MAX_FRONTMATTER_CHARS = 16 * 1024
MAX_BUILTIN_SKILL_BYTES = 32 * 1024
BUILTIN_CREATE_SKILL = (
    Path(__file__).parent / "assets" / "skills" / "create-skill" / "SKILL.md"
)


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class SkillSnapshot:
    skills: tuple[SkillMetadata, ...]
    errors: tuple[str, ...]


_snapshot = SkillSnapshot((), ())
_discovery_task: asyncio.Task[SkillSnapshot] | None = None


def _scalar(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value[:1] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value.strip("'\"").strip()
        return str(parsed).strip()
    return value.strip()


def _frontmatter(contents: str) -> dict[str, str]:
    """Parse the small YAML subset used by skill name/description fields."""
    lines = contents.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing YAML frontmatter")
    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("unterminated YAML frontmatter") from exc

    result: dict[str, str] = {}
    index = 1
    while index < end:
        line = lines[index]
        if not line or line[:1].isspace() or ":" not in line:
            index += 1
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        raw = raw.strip()
        if raw in {"|", ">", "|-", ">-", "|+", ">+"}:
            block: list[str] = []
            index += 1
            while index < end:
                nested = lines[index]
                if nested and not nested[:1].isspace():
                    break
                block.append(nested.lstrip())
                index += 1
            separator = " " if raw.startswith(">") else "\n"
            result[key] = separator.join(block).strip()
            continue
        result[key] = _scalar(raw)
        index += 1
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_frontmatter(path: Path) -> str:
    parts: list[str] = []
    characters = 0
    with path.open(encoding="utf-8") as handle:
        index = 0
        while True:
            remaining = MAX_FRONTMATTER_CHARS - characters
            line = handle.readline(remaining + 1)
            if not line:
                break
            characters += len(line)
            if characters > MAX_FRONTMATTER_CHARS:
                raise ValueError(
                    "skill frontmatter exceeds "
                    f"{MAX_FRONTMATTER_CHARS} characters"
                )
            parts.append(line)
            if index and line.strip() == "---":
                return "".join(parts)
            index += 1
    return "".join(parts)


def _advertised_skills(
    errors: list[str],
) -> Iterator[tuple[int, Path, Path]]:
    for root_index, configured_root in enumerate(SKILLS_ROOTS):
        try:
            root = configured_root.resolve()
            if not root.is_dir():
                continue
            for path in configured_root.rglob("SKILL.md"):
                if FILESYSTEM_POLICY.denies(path):
                    continue
                yield root_index, path, root
        except OSError as exc:
            errors.append(f"{configured_root}: {exc}")


def initialize_builtin_skills(
    root: Path = SKILLS_ROOTS[0],
    source: Path = BUILTIN_CREATE_SKILL,
) -> tuple[str, ...]:
    """Create missing Raptor-owned starter skills without replacing edits."""
    skill_dir = root / "create-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    if not _inside(skill_dir.resolve(), resolved_root):
        raise RuntimeError("built-in skill directory escapes .raptor/skills")
    path = skill_dir / "SKILL.md"
    if path.exists() or path.is_symlink():
        return ()
    try:
        write_bytes_exclusive_atomic(
            path,
            read_bytes_bounded(source, MAX_BUILTIN_SKILL_BYTES),
            mode=0o644,
        )
    except FileExistsError:
        return ()
    return (str(path),)


def _discover_sync() -> SkillSnapshot:
    found: list[SkillMetadata] = []
    errors: list[str] = []
    names: set[str] = set()
    advertised_paths = heapq.nsmallest(
        MAX_DISCOVERED_SKILLS + 1,
        _advertised_skills(errors),
        key=lambda item: (item[0], item[1]),
    )
    if len(advertised_paths) > MAX_DISCOVERED_SKILLS:
        advertised_paths.pop()
        errors.append(
            f"skill discovery exceeds {MAX_DISCOVERED_SKILLS} skills"
        )
    for _, advertised_path, root in advertised_paths:
        try:
            path = advertised_path.resolve(strict=True)
            if not _inside(path, root):
                raise ValueError("skill path escapes configured root")
            contents = _read_frontmatter(path)
            metadata = _frontmatter(contents)
            name = metadata.get("name", "").strip() or path.parent.name
            description = metadata.get("description", "").strip()
            if not name:
                raise ValueError("skill name is empty")
            if not description:
                raise ValueError("skill description is empty")
            key = name.casefold()
            if key in names:
                raise ValueError(f"duplicate skill name: {name}")
            names.add(key)
            found.append(
                SkillMetadata(
                    name=name,
                    description=description,
                    path=path,
                )
            )
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{advertised_path}: {exc}")
    found.sort(key=lambda skill: skill.name.casefold())
    return SkillSnapshot(tuple(found), tuple(errors))


async def refresh_skills() -> SkillSnapshot:
    global _snapshot
    snapshot = await asyncio.to_thread(_discover_sync)
    _snapshot = snapshot
    return snapshot


def start_skill_discovery() -> asyncio.Task[SkillSnapshot]:
    """Start discovery without delaying the rest of process initialization."""
    global _discovery_task
    if _discovery_task is None or _discovery_task.done():
        _discovery_task = asyncio.create_task(refresh_skills())
    return _discovery_task


async def close_skill_discovery() -> None:
    """Release the process-owned discovery task during application teardown."""
    global _discovery_task
    task = _discovery_task
    _discovery_task = None
    if task is None:
        return
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def ensure_skills_loaded() -> SkillSnapshot:
    global _discovery_task
    if _discovery_task is None:
        _discovery_task = asyncio.create_task(refresh_skills())
    await _discovery_task
    return _snapshot


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(AGENT_WORKDIR))
    except ValueError:
        return str(path)


async def skill_catalog_instructions() -> str:
    snapshot = await ensure_skills_loaded()
    if not snapshot.skills:
        return ""
    lines = [
        "<skills_instructions>",
        "## Skills",
        "Skills are workspace workflows discovered from `.raptor/skills` "
        "and `.agent/skills`.",
        "Only metadata is listed here. When the user names a skill with "
        "`$name`, or the task clearly matches a description, call "
        "`read_skill` before taking task actions and follow the complete "
        "returned `SKILL.md`. Read referenced files only when that skill "
        "requires them.",
        "### Available skills",
    ]
    for skill in snapshot.skills:
        lines.append(
            f"- {skill.name}: {skill.description} "
            f"({_relative_path(skill.path)})"
        )
    lines.append("</skills_instructions>")
    return "\n".join(lines)


async def read_skill_tool(args: dict[str, Any]) -> dict[str, Any]:
    name = str(args.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "name is required"}

    snapshot = await ensure_skills_loaded()
    match = next(
        (
            skill
            for skill in snapshot.skills
            if skill.name.casefold() == name.casefold()
        ),
        None,
    )
    if match is None:
        # Pick up newly created skills without requiring a process restart.
        snapshot = await refresh_skills()
        match = next(
            (
                skill
                for skill in snapshot.skills
                if skill.name.casefold() == name.casefold()
            ),
            None,
        )
    if match is None:
        return {
            "ok": False,
            "error": f"unknown skill: {name}",
            "available": [skill.name for skill in snapshot.skills],
        }
    if FILESYSTEM_POLICY.denies(match.path):
        return {"ok": False, "error": f"unknown skill: {name}"}
    try:
        contents = await asyncio.to_thread(
            read_text_bounded,
            match.path,
            MAX_TOOL_OUTPUT,
        )
    except FileTooLargeError:
        return {
            "ok": False,
            "error": (
                "skill exceeds tool-output limit of "
                f"{MAX_TOOL_OUTPUT} bytes"
            ),
        }
    except (OSError, UnicodeError) as exc:
        return {"ok": False, "error": f"failed to read skill: {exc}"}
    return {
        "ok": True,
        "name": match.name,
        "path": _relative_path(match.path),
        "contents": contents,
    }


def reset_skill_cache_for_tests() -> None:
    global _snapshot, _discovery_task
    if _discovery_task is not None and not _discovery_task.done():
        _discovery_task.cancel()
    _snapshot = SkillSnapshot((), ())
    _discovery_task = None
