"""Append-only JSONL chat transcript store."""
import heapq
import json
import os
import re
import secrets
import time
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CHAT_DIR, COMPACTION_MAX_RECORD_CHARS, RAPTOR_HOME
from storage import (
    ensure_private_directory,
    fsync_directory,
)

_SEQ_CACHE: dict[str, int] = {}
_SESSION_ID_RE = re.compile(
    r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$"
)
_TAIL_SCAN_CHUNK_BYTES = 64 * 1024
_SESSION_DISCOVERY_LIMIT = 1000


@dataclass(frozen=True)
class ActiveProjection:
    """Bounded in-memory projection of one append-only transcript."""

    items: list[dict[str, Any]]
    checkpoint: dict[str, Any] | None
    archive_events: int


def _decode_event_line(
    path: Path,
    line: str,
    line_number: int,
) -> dict[str, Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid transcript event at {path}:{line_number}"
        ) from exc
    if not isinstance(event, dict):
        raise RuntimeError(
            f"Transcript event must be an object at {path}:{line_number}"
        )
    return event


def new_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d-%H%M%S"
    )
    return f"{stamp}-{secrets.token_hex(4)}"


def validate_session_id(session_id: str) -> str:
    sid = str(session_id or "").strip()
    if not _SESSION_ID_RE.fullmatch(sid):
        raise ValueError(f"invalid session_id: {session_id!r}")
    return sid


def chat_path(session_id: str) -> Path:
    sid = validate_session_id(session_id)
    path = (CHAT_DIR / f"{sid}.jsonl").resolve()
    chat_root = CHAT_DIR.resolve()
    if path.parent != chat_root:
        raise ValueError(f"invalid session_id path: {session_id!r}")
    return path


def ensure_chat_dirs() -> None:
    ensure_private_directory(RAPTOR_HOME)
    ensure_private_directory(CHAT_DIR)


def _chat_file_needs_tail_repair(path: Path) -> bool:
    """True when the file exists and does not end with a newline."""
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size == 0:
        return False
    with path.open("rb") as f:
        f.seek(-1, os.SEEK_END)
        return f.read(1) != b"\n"


def repair_chat_file(session_id: str) -> bool:
    """Truncate a partial final line that was never newline-terminated.

    Returns True when bytes were removed. Intended for boot-time repair and
    the rare incomplete-tail case detected before append.
    """
    path = chat_path(session_id)
    if not _chat_file_needs_tail_repair(path):
        return False
    size = path.stat().st_size
    cut = 0
    with path.open("r+b") as handle:
        offset = size
        while offset > 0:
            chunk_start = max(0, offset - _TAIL_SCAN_CHUNK_BYTES)
            handle.seek(chunk_start)
            chunk = handle.read(offset - chunk_start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                cut = chunk_start + newline + 1
                break
            offset = chunk_start
        handle.truncate(cut)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)
    _SEQ_CACHE.pop(session_id, None)
    return True


def repair_all_chat_files() -> int:
    ensure_chat_dirs()
    fixed = 0
    for path in CHAT_DIR.glob("*.jsonl"):
        try:
            sid = validate_session_id(path.stem)
            os.chmod(chat_path(sid), 0o600)
            if repair_chat_file(sid):
                fixed += 1
        except ValueError:
            continue
    return fixed


def session_exists(session_id: str) -> bool:
    try:
        sid = validate_session_id(session_id)
    except ValueError:
        return False
    return chat_path(sid).is_file()


def _scan_max_seq(session_id: str) -> int:
    path = chat_path(session_id)
    if not path.is_file():
        return 0
    max_seq = 0
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            event = _decode_event_line(path, line, line_number)
            seq = event.get("seq")
            if isinstance(seq, int) and seq > max_seq:
                max_seq = seq
    return max_seq


def next_event_seq(session_id: str) -> int:
    """Return the next append sequence without consuming it."""
    if session_id not in _SEQ_CACHE:
        _SEQ_CACHE[session_id] = _scan_max_seq(session_id)
    return _SEQ_CACHE[session_id] + 1


def append_event(
    session_id: str,
    event: dict[str, Any],
    *,
    expected_seq: int | None = None,
) -> dict[str, Any]:
    sid = validate_session_id(session_id)
    ensure_chat_dirs()
    path = chat_path(sid)
    # O(1) incomplete-tail check; full repair only when needed.
    if _chat_file_needs_tail_repair(path):
        repair_chat_file(sid)
    created = not path.exists()
    written = dict(event)
    written["v"] = int(written.get("v") or 1)
    seq = next_event_seq(sid)
    if expected_seq is not None and seq != int(expected_seq):
        raise RuntimeError(
            f"Transcript sequence changed: expected {expected_seq}, got {seq}"
        )
    written["seq"] = seq
    written["ts"] = float(written.get("ts") or time.time())
    written["session_id"] = sid
    fd = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.fchmod(fd, 0o600)
    except BaseException:
        os.close(fd)
        raise
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(written, ensure_ascii=False))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        _SEQ_CACHE.pop(sid, None)
        raise
    _SEQ_CACHE[sid] = seq
    if created:
        fsync_directory(path.parent)
    return written


def create_session(
    *,
    kind: str,
    chat_key: str,
    agent_id: str | None = None,
    parent_session_id: str | None = None,
) -> str:
    owner = str(chat_key).strip()
    if not owner:
        raise ValueError("chat_key must not be empty")
    ensure_chat_dirs()
    session_id = new_session_id()
    parent = None
    if parent_session_id:
        parent = validate_session_id(parent_session_id)
    _SEQ_CACHE[session_id] = 0
    append_event(
        session_id,
        {
            "type": "session_start",
            "kind": kind,
            "chat_key": owner,
            "agent_id": agent_id,
            "parent_session_id": parent,
        },
    )
    return session_id


def append_item(
    session_id: str,
    item: dict[str, Any],
    *,
    source: str,
    data: dict[str, Any] | None = None,
    expected_seq: int | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "item",
        "source": source,
        "item": item,
    }
    if data is not None:
        event["data"] = data
    return append_event(session_id, event, expected_seq=expected_seq)


def steer_is_recorded(session_id: str, steer_id: str) -> bool:
    """Return whether one steer was transferred into the transcript."""
    expected = str(steer_id)
    for event in iter_events(session_id):
        data = event.get("data")
        if (
            event.get("type") == "item"
            and event.get("source") == "steer"
            and isinstance(data, dict)
            and str(data.get("steer_id") or "") == expected
        ):
            return True
    return False


def append_checkpoint(
    session_id: str,
    *,
    summary: str,
    through_seq: int,
    input_from_seq: int | None = None,
    input_to_seq: int | None = None,
    reason: str | None = None,
    anchors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "checkpoint",
        "summary": summary,
        "through_seq": int(through_seq),
    }
    if input_from_seq is not None:
        event["input_from_seq"] = int(input_from_seq)
    if input_to_seq is not None:
        event["input_to_seq"] = int(input_to_seq)
    if reason is not None:
        event["reason"] = reason
    if anchors:
        event["anchors"] = anchors
    return append_event(session_id, event)


def append_meta(
    session_id: str,
    name: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "meta",
        "name": name,
    }
    if data is not None:
        event["data"] = data
    return append_event(session_id, event)


def reset_model_context(
    session_id: str,
    *,
    through_seq: int,
) -> dict[str, Any]:
    """Start a fresh model-context epoch without rewriting the archive."""
    through = int(through_seq)
    if through <= 0:
        raise ValueError("invalid model context reset boundary")
    return append_meta(
        session_id,
        "model_context_reset",
        {"through_seq": through},
    )


def end_session(
    session_id: str,
    *,
    reason: str,
    todos: list[Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "session_end",
        "reason": reason,
    }
    if todos is not None:
        event["todos"] = todos
    written = append_event(session_id, event)
    _SEQ_CACHE.pop(session_id, None)
    return written


def iter_events(session_id: str) -> Iterator[dict[str, Any]]:
    """Yield validated transcript events without retaining the archive."""
    path = chat_path(session_id)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                yield _decode_event_line(path, line, line_number)


def read_events(session_id: str) -> list[dict[str, Any]]:
    return list(iter_events(session_id))


def session_is_ended(session_id: str) -> bool:
    """Return whether the transcript's latest durable event ends it."""
    latest: dict[str, Any] | None = None
    for event in iter_events(session_id):
        latest = event
    return bool(latest and latest.get("type") == "session_end")


def item_events(session_id: str) -> list[dict[str, Any]]:
    return [
        event
        for event in iter_events(session_id)
        if event.get("type") == "item"
    ]


def active_projection(session_id: str) -> ActiveProjection:
    """Build active items and checkpoint in one streaming archive pass."""
    items: list[dict[str, Any]] = []
    checkpoint: dict[str, Any] | None = None
    archive_events = 0
    for event in iter_events(session_id):
        archive_events += 1
        event_type = event.get("type")
        if event_type == "item":
            items.append(event)
        elif event_type == "checkpoint":
            checkpoint = event
            through = int(event.get("through_seq") or 0)
            items = [
                item
                for item in items
                if int(item.get("seq") or 0) > through
            ]
        elif (
            event_type == "meta"
            and event.get("name") == "model_context_reset"
        ):
            data = event.get("data")
            through = (
                int(data.get("through_seq") or 0)
                if isinstance(data, dict)
                else 0
            )
            items = [
                item
                for item in items
                if int(item.get("seq") or 0) > through
            ]
            checkpoint = None
    return ActiveProjection(items, checkpoint, archive_events)


def active_item_events(session_id: str) -> list[dict[str, Any]]:
    """Return transcript items eligible for model context."""
    return active_projection(session_id).items


def latest_checkpoint(
    session_id: str,
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for event in iter_events(session_id):
        if event.get("type") == "checkpoint":
            latest = event
    return latest


def active_checkpoint(session_id: str) -> dict[str, Any] | None:
    """Return the newest checkpoint valid for the active projection."""
    return active_projection(session_id).checkpoint


def _session_summary(path: Path) -> dict[str, Any] | None:
    start: dict[str, Any] | None = None
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            event = _decode_event_line(path, line, line_number)
            if first is None:
                first = event
            if start is None and event.get("type") == "session_start":
                start = event
            last = event
    if first is None or last is None:
        return None
    start = start or first
    return {
        "session_id": path.stem,
        "kind": start.get("kind") or "main",
        "chat_key": start.get("chat_key"),
        "parent_session_id": start.get("parent_session_id"),
        "agent_id": start.get("agent_id"),
        "started_at": start.get("ts"),
        "last_seq": last.get("seq"),
    }


def session_summary(session_id: str) -> dict[str, Any] | None:
    """Read one exact transcript summary without archive discovery."""
    path = chat_path(session_id)
    if not path.is_file():
        return None
    return _session_summary(path)


def _session_start(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            event = _decode_event_line(path, line, line_number)
            return event if event.get("type") == "session_start" else None
    return None


def list_sessions(
    *,
    limit: int = 100,
    chat_key: str | None = None,
    kinds: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Return summaries from a bounded window of newest transcripts."""
    ensure_chat_dirs()
    sessions: list[dict[str, Any]] = []
    requested = max(1, min(_SESSION_DISCOVERY_LIMIT, int(limit)))
    allowed_kinds = frozenset(kinds) if kinds is not None else None

    def valid_paths() -> Iterator[Path]:
        for path in CHAT_DIR.glob("*.jsonl"):
            try:
                validate_session_id(path.stem)
            except ValueError:
                continue
            if chat_key is not None or allowed_kinds is not None:
                start = _session_start(path)
                if start is None:
                    continue
                if (
                    chat_key is not None
                    and str(start.get("chat_key")) != str(chat_key)
                ):
                    continue
                if (
                    allowed_kinds is not None
                    and str(start.get("kind") or "main")
                    not in allowed_kinds
                ):
                    continue
            yield path

    newest = heapq.nlargest(
        requested,
        valid_paths(),
        key=lambda path: path.name,
    )
    for path in newest:
        summary = _session_summary(path)
        if summary is not None:
            sessions.append(summary)
    return sessions


def event_at_seq(session_id: str, seq: int) -> dict[str, Any] | None:
    """Return one exact transcript event without materializing the archive."""
    expected = int(seq)
    if expected <= 0:
        return None
    for event in iter_events(session_id):
        current = int(event.get("seq") or 0)
        if current == expected:
            return event
        if current > expected:
            break
    return None


def session_chat_key(session_id: str) -> str | None:
    """Return the owning main-chat key recorded by a transcript."""
    path = chat_path(session_id)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            event = _decode_event_line(path, line, line_number)
            if event.get("type") != "session_start":
                return None
            value = event.get("chat_key")
            return str(value) if value is not None else None
    return None


def session_contains_text(session_id: str, query: str) -> bool:
    """Search one transcript without retaining it in memory."""
    needle = query.casefold()
    if not needle:
        return True
    path = chat_path(session_id)
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            event = _decode_event_line(path, line, line_number)
            if needle in render_compaction_records([event]).casefold():
                return True
    return False


def render_item_text(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False, default=str)
    item_type = item.get("type")
    if item.get("role") == "user":
        content = item.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        return f"USER\n{content}"
    if item_type == "message" or item.get("role") == "assistant":
        parts: list[str] = []
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and content.get("text"):
                parts.append(str(content["text"]))
            elif isinstance(content, str):
                parts.append(content)
        body = (
            "\n".join(parts)
            if parts
            else json.dumps(item, ensure_ascii=False, default=str)
        )
        return f"ASSISTANT\n{body}"
    if item_type == "function_call":
        return (
            f"FUNCTION_CALL name={item.get('name')} "
            f"call_id={item.get('call_id')}\n"
            f"{item.get('arguments')}"
        )
    if item_type == "function_call_output":
        return (
            f"FUNCTION_RESULT call_id={item.get('call_id')}\n"
            f"{item.get('output')}"
        )
    return json.dumps(item, ensure_ascii=False, default=str)


def _truncate_compaction_record_text(
    text: str,
    max_chars: int,
) -> str:
    """Bound one record's checkpoint rendering; archive stays intact."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = (
        "\n... [truncated for checkpoint generation; "
        "full record remains in chat archive]"
    )
    keep = max(0, max_chars - len(marker))
    if keep <= 0:
        return marker.strip()
    return text[:keep] + marker


def render_compaction_records(
    records: list[dict[str, Any]],
    *,
    max_record_chars: int | None = None,
) -> str:
    """Render archive records for checkpoint generation.

    Oversized individual records may be truncated in this view only.
    The canonical JSONL transcript is never modified.
    """
    limit = (
        COMPACTION_MAX_RECORD_CHARS
        if max_record_chars is None
        else max_record_chars
    )
    blocks: list[str] = []
    for record in records:
        seq = record.get("seq")
        if record.get("type") == "checkpoint":
            anchor_blocks: list[str] = []
            anchors = record.get("anchors")
            if isinstance(anchors, list):
                for anchor in anchors:
                    if not isinstance(anchor, dict):
                        continue
                    item = anchor.get("item")
                    if not isinstance(item, dict):
                        continue
                    anchor_blocks.append(
                        "PRESERVED USER REQUEST "
                        f"seq={anchor.get('seq')}\n{render_item_text(item)}"
                    )
            anchor_text = (
                "\n\n" + "\n\n".join(anchor_blocks)
                if anchor_blocks
                else ""
            )
            body = _truncate_compaction_record_text(
                f"[seq={seq}] CHECKPOINT\n{record.get('summary')}"
                + anchor_text,
                limit,
            )
            blocks.append(body)
            continue
        if record.get("type") == "item":
            item = record.get("item")
            if not isinstance(item, dict):
                continue
            body = _truncate_compaction_record_text(
                f"[seq={seq}] {render_item_text(item)}",
                limit,
            )
            blocks.append(body)
            continue
        body = _truncate_compaction_record_text(
            f"[seq={seq}] "
            + json.dumps(record, ensure_ascii=False, default=str),
            limit,
        )
        blocks.append(body)
    return "\n\n".join(blocks)
