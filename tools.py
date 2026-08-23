"""Agent tool implementations and dispatch."""
import json
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
from config import AGENT_WORKDIR, MAX_TOOL_OUTPUT
from session import save_state, state
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

    keep = max(
        1000,
        MAX_TOOL_OUTPUT // 2,
    )

    return (
        text[:keep]
        + "\n... [truncated] ...\n"
        + text[-keep:],
        True,
    )


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

    path.write_text(
        content
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

    path.write_text(
        text
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

    return {
        "ok": True,
        "path":
            relative,
        "entries":
            entries,
    }


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


async def execute_tool(
    call: dict[str, Any],
    *,
    chat_id: ConversationId | None = None,
    execution_context: dict[str, Any]
    | None = None,
) -> dict[str, Any]:
    try:
        args = json.loads(
            call.get(
                "arguments"
            )
            or "{}"
        )

    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error":
                f"bad JSON arguments: {exc}",
        }

    name = call.get(
        "name"
    )
    context = (
        execution_context
        if execution_context is not None
        else {
            "depth": 0,
            "subagents_allowed": True,
        }
    )

    try:
        if name == "update_plan":
            return update_plan_tool(
                args,
                context.get("todo_state"),
            )

        if name == "shell":
            return await shell_tool(
                args,
                chat_id=chat_id,
                execution_context=context,
            )

        if name == "write_stdin":
            from shell_sessions import write_stdin
            return await write_stdin(args)

        if name == "read_file":
            return read_file_tool(
                args
            )

        if name == "read_skill":
            from skills import read_skill_tool
            return await read_skill_tool(args)

        if name == "write_file":
            return write_file_tool(
                args
            )

        if name == "edit_file":
            return edit_file_tool(
                args
            )

        if name == "list_dir":
            return list_dir_tool(
                args
            )

        if name == "get_goal":
            from goals import get_goal_tool_result
            return get_goal_tool_result()

        if name == "update_goal":
            from goals import update_goal_tool
            return update_goal_tool(args)

        if name == "set_goal":
            from goals import set_goal_tool
            return await set_goal_tool(
                args,
                chat_id=chat_id,
            )

        if name == "chat_history":
            return chat_history_tool(
                args,
                execution_context=context,
            )

        if name == "subagent":
            from subagents import subagent_tool
            return await subagent_tool(
                args,
                chat_id=chat_id,
                execution_context=context,
            )

        return {
            "ok": False,
            "error":
                f"unknown tool: {name}",
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": (
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        }
