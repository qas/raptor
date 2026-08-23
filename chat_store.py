"""Append-only JSONL chat transcript store."""
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CHAT_DIR, COMPACTION_MAX_RECORD_CHARS
from storage import fsync_directory, write_bytes_atomic

_SEQ_CACHE: dict[str, int] = {}
_SESSION_ID_RE = re.compile(
    r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$"
)


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
    CHAT_DIR.mkdir(parents=True, exist_ok=True)


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
    data = path.read_bytes()
    cut = data.rfind(b"\n")
    repaired = data[: cut + 1] if cut >= 0 else b""
    write_bytes_atomic(path, repaired)
    _SEQ_CACHE.pop(session_id, None)
    return True


def repair_all_chat_files() -> int:
    ensure_chat_dirs()
    fixed = 0
    for path in CHAT_DIR.glob("*.jsonl"):
        try:
            if repair_chat_file(path.stem):
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


def _next_seq(session_id: str) -> int:
    if session_id not in _SEQ_CACHE:
        _SEQ_CACHE[session_id] = _scan_max_seq(session_id)
    _SEQ_CACHE[session_id] += 1
    return _SEQ_CACHE[session_id]


def append_event(
    session_id: str,
    event: dict[str, Any],
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
    written["seq"] = _next_seq(sid)
    written["ts"] = float(written.get("ts") or time.time())
    written["session_id"] = sid
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(written, ensure_ascii=False))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    if created:
        fsync_directory(path.parent)
    return written


def create_session(
    *,
    kind: str,
    agent_id: str | None = None,
    parent_session_id: str | None = None,
) -> str:
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
) -> dict[str, Any]:
    return append_event(
        session_id,
        {
            "type": "item",
            "source": source,
            "item": item,
        },
    )


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
    return append_event(session_id, event)


def read_events(session_id: str) -> list[dict[str, Any]]:
    path = chat_path(session_id)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            events.append(_decode_event_line(path, line, line_number))
    return events


def item_events(session_id: str) -> list[dict[str, Any]]:
    return [
        event
        for event in read_events(session_id)
        if event.get("type") == "item"
    ]


def latest_checkpoint(
    session_id: str,
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for event in read_events(session_id):
        if event.get("type") == "checkpoint":
            latest = event
    return latest


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
        "parent_session_id": start.get("parent_session_id"),
        "agent_id": start.get("agent_id"),
        "started_at": start.get("ts"),
        "last_seq": last.get("seq"),
    }


def list_sessions() -> list[dict[str, Any]]:
    ensure_chat_dirs()
    sessions: list[dict[str, Any]] = []
    for path in sorted(CHAT_DIR.glob("*.jsonl")):
        try:
            validate_session_id(path.stem)
        except ValueError:
            continue
        summary = _session_summary(path)
        if summary is not None:
            sessions.append(summary)
    return sessions


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
