"""Agent tool implementations and dispatch."""
import heapq
import json
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from raptor.chat.chat_provider import ConversationId

from raptor.state.chat_store import (
    iter_events,
    list_sessions,
    render_compaction_records,
    render_item_text,
    session_summary,
    validate_session_id,
)
from config import AGENT_WORKDIR, FILESYSTEM_POLICY, MAX_TOOL_OUTPUT, TOOLS
from raptor.state import session
from raptor.state.session import save_state, state
from raptor.state.storage import FileTooLargeError, read_bytes_bounded, write_text_atomic
from raptor.agent.todos import MAX_TODO_EXPLANATION_CHARS, validate_plan


FILE_READ_CHUNK_CHARS = 64 * 1024
MAX_EDIT_FILE_BYTES = 64 * 1024 * 1024


class _BoundedText:
    """Retain a fixed-size head and tail while text is streamed."""

    _marker = "\n... [truncated] ...\n"

    def __init__(self, limit: int) -> None:
        self.limit = limit
        retained = max(0, limit - len(self._marker))
        self.head_limit = retained // 2
        self.tail_limit = retained - self.head_limit
        self.head = ""
        self.tail = ""
        self.total = 0

    @property
    def truncated(self) -> bool:
        return self.total > self.limit

    def append(self, text: str) -> None:
        if not text:
            return
        previous_total = self.total
        self.total += len(text)
        if self.total <= self.limit:
            self.head += text
            return
        if previous_total <= self.limit:
            combined = self.head + text
            self.head = combined[:self.head_limit]
            self.tail = combined[-self.tail_limit:] if self.tail_limit else ""
            return
        if self.tail_limit:
            self.tail = (self.tail + text)[-self.tail_limit:]

    def render(self) -> str:
        if not self.truncated:
            return self.head
        return self.head + self._marker + self.tail

# ---------------------------------------------------------------------------
# Todo tool
# ---------------------------------------------------------------------------

def update_plan_tool(
    args: dict[str, Any],
    todo_state: dict[str, Any]
    | None = None,
) -> dict[str, Any]:
    store = (
        todo_state
        if todo_state is not None
        else state
    )
    unknown = set(args) - {"explanation", "plan"}
    if unknown:
        return {"ok": False, "error": "unknown update_plan arguments"}
    explanation = args.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        return {"ok": False, "error": "explanation must be a string"}
    if explanation is not None and len(explanation) > MAX_TODO_EXPLANATION_CHARS:
        return {
            "ok": False,
            "error": (
                "explanation exceeds "
                f"{MAX_TODO_EXPLANATION_CHARS} characters"
            ),
        }
    try:
        plan = validate_plan(args.get("plan"))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    missing = object()
    previous = store.get("todos", missing)
    store["todos"] = plan
    try:
        save_state()
    except Exception:
        if previous is missing:
            store.pop("todos", None)
        else:
            store["todos"] = previous
        raise
    return {
        "ok": True,
        "message": "Plan updated",
        "plan": plan,
    }


# ---------------------------------------------------------------------------
# Filesystem / shell tools
# ---------------------------------------------------------------------------

def workspace_root() -> Path:
    return AGENT_WORKDIR.resolve()


def workspace_path(
    raw: str,
) -> Path:
    raw = raw.strip() or "."
    requested = Path(raw)
    root = workspace_root()
    logical_path = requested if requested.is_absolute() else root / requested
    path = logical_path.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(
            f"path escapes AGENT_WORKDIR: {raw}"
        ) from None
    if FILESYSTEM_POLICY.denies(path, logical_path=logical_path):
        raise PermissionError(
            "path is denied by permissions.filesystem.deny_read"
        )
    return path


def truncate_tool_output(
    text: str,
) -> tuple[str, bool]:
    if (
        len(text)
        <= MAX_TOOL_OUTPUT
    ):
        return (
            text,
            False,
        )

    marker = "\n... [truncated] ...\n"
    retained = MAX_TOOL_OUTPUT - len(marker)
    keep_head = retained // 2
    keep_tail = retained - keep_head

    return (
        text[:keep_head]
        + marker
        + text[-keep_tail:],
        True,
    )


async def cancel_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Cancel one live background resource by its typed identifier."""
    kind = str(args.get("kind") or "").strip()
    resource_id = str(args.get("id") or "").strip()
    if kind not in {"subagent", "shell"}:
        return {"ok": False, "error": "kind must be subagent or shell"}
    if not resource_id:
        return {"ok": False, "error": "id is required"}
    if kind == "subagent":
        from raptor.agent.subagents import cancel_background_subagent

        return await cancel_background_subagent(resource_id)
    from raptor.shell.shell_sessions import cancel_shell_session

    return await cancel_shell_session(resource_id)


async def shell_tool(
    args: dict[str, Any],
    *,
    chat_id: ConversationId | None = None,
    execution_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command = str(
        args.get(
            "command",
            "",
        )
    ).strip()

    if not command:
        return {
            "ok": False,
            "error":
                "command is required",
        }

    from raptor.shell.shell_sessions import run_shell

    parent_session_id = None
    process_output = None
    tool_call_id = ""
    if execution_context is not None:
        value = (
            execution_context.get("root_session_id")
            or execution_context.get("parent_session_id")
            or execution_context.get("session_id")
        )
        if value is not None:
            parent_session_id = str(value)
        callback = execution_context.get("process_output")
        if callable(callback):
            process_output = callback
        tool_call_id = str(execution_context.get("tool_call_id") or "")
    return await run_shell(
        command,
        timeout=args.get("timeout"),
        yield_time_ms=args.get("yield_time_ms"),
        tty=bool(args.get("tty", False)),
        chat_id=chat_id,
        parent_session_id=parent_session_id,
        tool_call_id=tool_call_id,
        process_output=process_output,
    )


def read_file_tool(
    args: dict[str, Any],
) -> dict[str, Any]:
    path = workspace_path(
        str(
            args.get(
                "path",
                "",
            )
        )
    )

    start_line = max(
        1,
        int(
            args.get(
                "start_line"
            )
            or 1
        ),
    )

    max_lines = min(
        5000,
        max(
            1,
            int(
                args.get(
                    "max_lines"
                )
                or 500
            ),
        ),
    )

    chunk = _BoundedText(MAX_TOOL_OUTPUT)
    line_number = 1
    selected_lines = 0
    at_line_start = True
    has_more = False
    with path.open(encoding="utf-8", errors="replace") as handle:
        while selected_lines < max_lines:
            part = handle.readline(FILE_READ_CHUNK_CHARS)
            if not part:
                break
            line_ended = part.endswith("\n")
            body = part[:-1] if line_ended else part
            if line_number >= start_line:
                if at_line_start and selected_lines:
                    chunk.append("\n")
                chunk.append(body)
            at_line_start = False
            if line_ended:
                if line_number >= start_line:
                    selected_lines += 1
                line_number += 1
                at_line_start = True
        if selected_lines >= max_lines:
            has_more = bool(handle.read(1))

    return {
        "ok": True,
        "path": str(
            path.relative_to(
                workspace_root()
            )
        ),
        "start_line":
            start_line,
        "text":
            chunk.render(),
        "truncated":
            chunk.truncated or has_more,
    }


def write_file_tool(
    args: dict[str, Any],
) -> dict[str, Any]:
    path = workspace_path(
        str(
            args.get(
                "path",
                "",
            )
        )
    )

    content = str(
        args.get(
            "content",
            "",
        )
    )
    if len(content) > MAX_TOOL_OUTPUT:
        return {
            "ok": False,
            "error": f"content exceeds {MAX_TOOL_OUTPUT} characters",
        }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    write_text_atomic(
        path,
        content,
        mode=mode,
    )

    return {
        "ok": True,
        "path": str(
            path.relative_to(
                workspace_root()
            )
        ),
        "bytes": len(
            content.encode()
        ),
    }


def edit_file_tool(
    args: dict[str, Any],
) -> dict[str, Any]:
    path = workspace_path(
        str(
            args.get(
                "path",
                "",
            )
        )
    )

    old_text = str(
        args.get(
            "old_text",
            "",
        )
    )

    new_text = str(
        args.get(
            "new_text",
            "",
        )
    )

    if len(old_text) > MAX_TOOL_OUTPUT or len(new_text) > MAX_TOOL_OUTPUT:
        return {
            "ok": False,
            "error": (
                "edit text exceeds "
                f"{MAX_TOOL_OUTPUT} characters"
            ),
        }

    if not old_text:
        return {
            "ok": False,
            "error":
                "old_text must not be empty",
        }

    try:
        encoded = read_bytes_bounded(path, MAX_EDIT_FILE_BYTES)
    except FileTooLargeError:
        return {
            "ok": False,
            "error": (
                "file exceeds edit limit of "
                f"{MAX_EDIT_FILE_BYTES} bytes"
            ),
        }

    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "error": "file is not valid UTF-8 text"}

    count = text.count(
        old_text
    )

    if count == 0:
        return {
            "ok": False,
            "error":
                "old_text not found",
        }

    replace_all = bool(
        args.get(
            "replace_all",
            False,
        )
    )

    if (
        count > 1
        and not replace_all
    ):
        return {
            "ok": False,
            "error": (
                f"old_text occurs {count} times; "
                "make it unique or set "
                "replace_all=true"
            ),
        }

    if replace_all:
        replacements = count
    else:
        replacements = 1

    output_bytes = (
        len(encoded)
        + replacements
        * (
            len(new_text.encode("utf-8"))
            - len(old_text.encode("utf-8"))
        )
    )
    if output_bytes > MAX_EDIT_FILE_BYTES:
        return {
            "ok": False,
            "error": (
                "edited file exceeds limit of "
                f"{MAX_EDIT_FILE_BYTES} bytes"
            ),
        }

    if replace_all:
        text = text.replace(
            old_text,
            new_text,
        )
    else:
        text = text.replace(
            old_text,
            new_text,
            1,
        )

    write_text_atomic(
        path,
        text,
        mode=stat.S_IMODE(path.stat().st_mode),
    )

    return {
        "ok": True,
        "path": str(
            path.relative_to(
                workspace_root()
            )
        ),
        "replacements":
            replacements,
    }


def list_dir_tool(
    args: dict[str, Any],
) -> dict[str, Any]:
    path = workspace_path(
        str(
            args.get(
                "path",
                ".",
            )
        )
    )

    limit = min(
        2000,
        max(
            1,
            int(
                args.get(
                    "max_entries"
                )
                or 300
            ),
        ),
    )

    entries: list[
        dict[str, Any]
    ] = []

    visible_children = (
        child
        for child in path.iterdir()
        if not FILESYSTEM_POLICY.denies(child)
    )
    children = heapq.nsmallest(
        limit + 1,
        visible_children,
        key=lambda item: (
            not item.is_dir(),
            item.name.lower(),
        ),
    )
    has_more = len(children) > limit

    for child in children[:limit]:
        entries.append(
            {
                "name":
                    child.name,
                "type":
                    (
                        "dir"
                        if child.is_dir()
                        else "file"
                    ),
                "size":
                    (
                        child.stat().st_size
                        if child.is_file()
                        else None
                    ),
            }
        )

    relative = (
        "."
        if path == workspace_root()
        else str(
            path.relative_to(
                workspace_root()
            )
        )
    )

    result = {
        "ok": True,
        "path":
            relative,
        "entries":
            entries,
    }
    if has_more:
        result["truncated"] = True
    while (
        entries
        and len(json.dumps(result, ensure_ascii=False)) > MAX_TOOL_OUTPUT
    ):
        entries.pop()
        result["truncated"] = True
    return result


def _resolve_history_session_id(
    args: dict[str, Any],
    execution_context: dict[str, Any] | None,
) -> str | None:
    requested = str(args.get("session_id") or "").strip()
    if requested:
        return validate_session_id(requested)
    if execution_context:
        sid = execution_context.get("session_id")
        if sid:
            return validate_session_id(str(sid))
        current = execution_context.get("current_session_id")
        if current:
            return validate_session_id(str(current))
    sid = state.get("current_session_id")
    return validate_session_id(str(sid)) if sid else None


def _history_snippet(event: dict[str, Any], query: str) -> str:
    rendered = render_compaction_records([event])
    lower = rendered.lower()
    q = query.lower()
    idx = lower.find(q)
    if idx < 0:
        return rendered[:240]
    start = max(0, idx - 80)
    end = min(len(rendered), idx + len(query) + 80)
    return rendered[start:end]


def _fit_history_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure the final serialized chat_history object fits MAX_TOOL_OUTPUT.

    Shrinks sessions/hits/records generically. Never returns a full list plus
    a preview. Oversized content wrappers are resized until the final object
    itself is within budget.
    """
    raw = json.dumps(payload, ensure_ascii=False)
    if len(raw) <= MAX_TOOL_OUTPUT:
        return payload
    list_key = next(
        (
            key
            for key in ("sessions", "hits", "records")
            if key in payload
        ),
        None,
    )
    if list_key is not None:
        items = list(payload.get(list_key) or [])
        while items:
            items = items[:-1]
            trial = dict(payload)
            trial[list_key] = items
            trial["truncated"] = True
            trial.pop("preview", None)
            if len(json.dumps(trial, ensure_ascii=False)) <= MAX_TOOL_OUTPUT:
                return trial
    source = json.dumps(payload, ensure_ascii=False)
    content = source
    while True:
        trial = {
            "ok": True,
            "truncated": True,
            "content": content,
        }
        encoded = json.dumps(trial, ensure_ascii=False)
        if len(encoded) <= MAX_TOOL_OUTPUT:
            return trial
        overflow = len(encoded) - MAX_TOOL_OUTPUT
        if not content:
            return {
                "ok": True,
                "truncated": True,
                "content": "",
            }
        content = content[
            : max(0, len(content) - max(1, overflow))
        ]


def chat_history_tool(
    args: dict[str, Any],
    *,
    execution_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action = str(args.get("action") or "").strip()
    limit = int(args.get("limit") or 20)
    limit = max(1, min(100, limit))
    chat_key = session.current_runtime().key
    visible_sessions = list_sessions(
        limit=100,
        chat_key=chat_key,
        kinds={"main", "subagent"},
    )
    if action == "list":
        sessions = visible_sessions[:limit]
        return _fit_history_payload(
            {"ok": True, "sessions": sessions}
        )
    try:
        session_id = _resolve_history_session_id(
            args,
            execution_context,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if session_id:
        target = session_summary(session_id)
        if target is not None and target.get("chat_key") != chat_key:
            target = None
        if target is None:
            return {"ok": False, "error": "session is not available"}
        current_execution_session = str(
            (execution_context or {}).get("session_id") or ""
        )
        if (
            target.get("kind") == "thread"
            and current_execution_session != session_id
        ):
            return {
                "ok": False,
                "error": "thread session is not available",
            }
    if action == "search":
        query = str(args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "query required"}
        targets: list[str] = []
        if bool(args.get("all_sessions")):
            targets = [row["session_id"] for row in visible_sessions]
        elif session_id:
            targets = [session_id]
        else:
            return {"ok": False, "error": "session_id required"}
        hits: list[dict[str, Any]] = []
        for sid in targets:
            for event in iter_events(sid):
                text = render_compaction_records([event])
                if query.lower() not in text.lower():
                    continue
                hits.append(
                    {
                        "session_id": sid,
                        "seq": event.get("seq"),
                        "type": event.get("type"),
                        "snippet": _history_snippet(event, query),
                    }
                )
                if len(hits) >= limit:
                    break
            if len(hits) >= limit:
                break
        return _fit_history_payload({"ok": True, "hits": hits})
    if action == "read":
        if not session_id:
            return {"ok": False, "error": "session_id required"}
        start_seq = int(args.get("start_seq") or 1)
        end_seq = int(args.get("end_seq") or 10**9)
        records = []
        for event in iter_events(session_id):
            seq = int(event.get("seq") or 0)
            if seq < start_seq or seq > end_seq:
                continue
            if event.get("type") == "item":
                item = event.get("item")
                body = (
                    render_item_text(item)
                    if isinstance(item, dict)
                    else json.dumps(item, default=str)
                )
            else:
                body = render_compaction_records([event])
            records.append(
                {
                    "seq": seq,
                    "type": event.get("type"),
                    "text": body,
                }
            )
            if len(records) >= limit:
                break
        return _fit_history_payload(
            {
                "ok": True,
                "session_id": session_id,
                "records": records,
            }
        )
    return {
        "ok": False,
        "error": f"unknown action: {action}",
    }


ToolHandler = Callable[
    [dict[str, Any], ConversationId | None, dict[str, Any]],
    Awaitable[dict[str, Any]],
]


async def _execute_update_plan(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    return update_plan_tool(args, context.get("todo_state"))


async def _execute_shell(
    args: dict[str, Any],
    chat_id: ConversationId | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    return await shell_tool(
        args,
        chat_id=chat_id,
        execution_context=context,
    )


async def _execute_write_stdin(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    from raptor.shell.shell_sessions import write_stdin

    return await write_stdin(args)


async def _execute_cancel(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    return await cancel_tool(args)


async def _execute_read_file(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    return read_file_tool(args)


async def _execute_read_skill(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    from raptor.agent.skills import read_skill_tool

    return await read_skill_tool(args)


async def _execute_write_file(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    return write_file_tool(args)


async def _execute_edit_file(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    return edit_file_tool(args)


async def _execute_list_dir(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    return list_dir_tool(args)


async def _execute_get_goal(
    _args: dict[str, Any],
    _chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    from raptor.agent.goals import get_goal_tool_result

    return get_goal_tool_result()


async def _execute_update_goal(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    from raptor.agent.goals import update_goal_tool

    return update_goal_tool(args)


async def _execute_set_goal(
    args: dict[str, Any],
    chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    from raptor.agent.goals import set_goal_tool

    return await set_goal_tool(args, chat_id=chat_id)


async def _execute_chat_history(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    return chat_history_tool(args, execution_context=context)


async def _execute_subagent(
    args: dict[str, Any],
    chat_id: ConversationId | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    from raptor.agent.subagents import subagent_tool

    return await subagent_tool(
        args,
        chat_id=chat_id,
        execution_context=context,
    )


TOOL_HANDLERS: dict[str, ToolHandler] = {
    "update_plan": _execute_update_plan,
    "shell": _execute_shell,
    "write_stdin": _execute_write_stdin,
    "cancel": _execute_cancel,
    "read_file": _execute_read_file,
    "read_skill": _execute_read_skill,
    "write_file": _execute_write_file,
    "edit_file": _execute_edit_file,
    "list_dir": _execute_list_dir,
    "get_goal": _execute_get_goal,
    "update_goal": _execute_update_goal,
    "set_goal": _execute_set_goal,
    "chat_history": _execute_chat_history,
    "subagent": _execute_subagent,
}


def validate_tool_catalog() -> None:
    schema_names = {str(tool.get("name")) for tool in TOOLS}
    handler_names = set(TOOL_HANDLERS)
    if schema_names != handler_names:
        missing_handlers = sorted(schema_names - handler_names)
        missing_schemas = sorted(handler_names - schema_names)
        raise RuntimeError(
            "tool catalog mismatch: "
            f"missing handlers={missing_handlers}, "
            f"missing schemas={missing_schemas}"
        )


validate_tool_catalog()


async def execute_tool(
    call: dict[str, Any],
    *,
    chat_id: ConversationId | None = None,
    execution_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        args = json.loads(str(call.get("arguments") or "{}"))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"bad JSON arguments: {exc}"}
    if not isinstance(args, dict):
        return {"ok": False, "error": "tool arguments must be a JSON object"}

    name = str(call.get("name") or "")
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown tool: {name}"}
    context = (
        execution_context
        if execution_context is not None
        else {"depth": 0, "subagents_allowed": True}
    )
    try:
        return await handler(args, chat_id, context)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
