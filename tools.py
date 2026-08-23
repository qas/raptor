"""Agent tool implementations and dispatch."""
import json
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from chat_provider import ConversationId

from chat_store import (
    list_sessions,
    read_events,
    render_compaction_records,
    render_item_text,
    validate_session_id,
)
from config import AGENT_WORKDIR, MAX_TOOL_OUTPUT, TOOLS
from session import save_state, state
from storage import write_text_atomic
from todos import MAX_TODO_EXPLANATION_CHARS, validate_plan

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

def workspace_path(
    raw: str,
) -> Path:
    raw = (
        raw.strip()
        or "."
    )

    requested = Path(
        raw
    )

    if requested.is_absolute():
        path = requested.resolve()
    else:
        path = (
            AGENT_WORKDIR
            / requested
        ).resolve()

    try:
        path.relative_to(
            AGENT_WORKDIR
        )

    except ValueError:
        raise ValueError(
            "path escapes "
            f"AGENT_WORKDIR: {raw}"
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
        from subagents import cancel_background_subagent

        return await cancel_background_subagent(resource_id)
    from shell_sessions import cancel_shell_session

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

    from shell_sessions import run_shell

    parent_session_id = None
    if execution_context is not None:
        value = (
            execution_context.get("parent_session_id")
            or execution_context.get("session_id")
        )
        if value is not None:
            parent_session_id = str(value)
    return await run_shell(
        command,
        timeout=args.get("timeout"),
        yield_time_ms=args.get("yield_time_ms"),
        tty=bool(args.get("tty", False)),
        chat_id=chat_id,
        parent_session_id=parent_session_id,
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

    text = path.read_text(
        errors="replace"
    )

    lines = (
        text.splitlines()
    )

    chunk = "\n".join(
        lines[
            start_line - 1:
            start_line - 1
            + max_lines
        ]
    )

    chunk, truncated = (
        truncate_tool_output(
            chunk
        )
    )

    return {
        "ok": True,
        "path": str(
            path.relative_to(
                AGENT_WORKDIR
            )
        ),
        "start_line":
            start_line,
        "text":
            chunk,
        "truncated":
            truncated,
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
                AGENT_WORKDIR
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

    if not old_text:
        return {
            "ok": False,
            "error":
                "old_text must not be empty",
        }

    text = path.read_text(
        errors="replace"
    )

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
        text = text.replace(
            old_text,
            new_text,
        )

        replacements = count

    else:
        text = text.replace(
            old_text,
            new_text,
            1,
        )

        replacements = 1

    write_text_atomic(
        path,
        text,
        mode=stat.S_IMODE(path.stat().st_mode),
    )

    return {
        "ok": True,
        "path": str(
            path.relative_to(
                AGENT_WORKDIR
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

    children = sorted(
        path.iterdir(),
        key=lambda item: (
            not item.is_dir(),
            item.name.lower(),
        ),
    )

    for child in children[
        :limit
    ]:
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
        if path == AGENT_WORKDIR
        else str(
            path.relative_to(
                AGENT_WORKDIR
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
    visible_sessions = [
        row for row in list_sessions()
        if row.get("kind") != "thread"
    ]
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
        kind = next(
            (
                row.get("kind")
                for row in list_sessions()
                if row.get("session_id") == session_id
            ),
            None,
        )
        current_execution_session = str(
            (execution_context or {}).get("session_id") or ""
        )
        if kind == "thread" and current_execution_session != session_id:
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
            for event in read_events(sid):
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
        for event in read_events(session_id):
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
    from shell_sessions import write_stdin

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
    from skills import read_skill_tool

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
    from goals import get_goal_tool_result

    return get_goal_tool_result()


async def _execute_update_goal(
    args: dict[str, Any],
    _chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    from goals import update_goal_tool

    return update_goal_tool(args)


async def _execute_set_goal(
    args: dict[str, Any],
    chat_id: ConversationId | None,
    _context: dict[str, Any],
) -> dict[str, Any]:
    from goals import set_goal_tool

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
    from subagents import subagent_tool

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
