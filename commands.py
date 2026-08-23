"""Provider-neutral text commands."""
import asyncio
import copy
import json
import os
import time

from chat_provider import ConversationId

from chat_store import (
    append_meta,
    create_session,
    end_session,
    session_exists,
)
from config import (
    AGENT_WORKDIR,
    CONTEXT_COMPACT_RATIO,
    CONTEXT_SAFETY_TOKENS,
    MODEL_CONTEXT_TOKENS,
    SUBAGENT_MODEL_CONTEXT_TOKENS,
    RESPONSES_REASONING_EFFORT,
    SUBAGENT_RESPONSES_REASONING_EFFORT,
    CHAT_STREAMING,
    MAX_TOOL_ROUNDS,
    context_input_budget,
    subagent_context_input_budget,
)
from session import DEFAULT_STATE, pending_approvals, save_state, state
import session
from observability import log_exception
from agent import context_tokens
from context import session_context_stats
from controller import (
    cancel_active_goal_controller,
    ensure_root_session,
    start_manual_compaction,
    start_root_session,
)
from goals import (
    GOAL_ACTIVE,
    GOAL_BLOCKED,
    GOAL_PAUSED,
    clear_goal,
    complete_goal,
    current_goal,
    ensure_goal_pin,
    format_goal_status,
    goal_is_active,
    pause_goal,
    remove_goal_pin,
    replace_goal,
    resume_goal,
    todo_store_for_display,
)
from runtime import runtime_uptime
from chat_runtime import get_chat_provider, send
from engine import response_calls, response_output, response_text
from approval import execute_tool_with_approval
from responses import list_models, stateless_response
from subagents import cancel_background_subagents
from shell_sessions import cancel_shell_sessions, running_shell_sessions
from threads import (
    finish_thread,
    resume_main_goal,
    start_thread,
)
from thread_state import current_thread, thread_active

# ---------------------------------------------------------------------------
# Chat commands
# ---------------------------------------------------------------------------


def format_todos() -> str:
    todos = todo_store_for_display().get(
        "todos",
        [],
    )

    if not todos:
        return "No todos."

    marks = {
        "pending":
            "[ ]",
        "in_progress":
            "[>]",
        "completed":
            "[x]",
    }

    return "\n".join(
        (
            f"{marks.get(item['status'], '[ ]')} "
            f"{item['step']}"
        )
        for item in todos
    )


async def command(
    chat_id: ConversationId,
    text: str,
) -> bool | str:
    first, *rest = (
        text.strip().split(
            maxsplit=1
        )
    )

    cmd = first.split(
        "@",
        1,
    )[0].lower()

    arg = (
        rest[0].strip()
        if rest
        else ""
    )

    if cmd == "/help":
        await send(
            chat_id,
            (
                "/new - start a new session (archive previous)\n"
                "/status - show status\n"
                "/stop - abort current run\n"
                "/compact - compact context\n"
                "/ask <message> - isolated tool-capable query\n"
                "/thread - fork into a temporary conversation\n"
                "/thread <message> - fork or continue, then send\n"
                "/thread status|clear|merge\n"
                "/model - list models\n"
                "/model <id> - switch model\n"
                "/approval - show approval mode\n"
                "/approval on - require approval for side effects\n"
                "/approval off - allow tools immediately\n"
                "/todos - show todo list\n"
                "/goal - show persistent goal\n"
                "/goal <objective> - start or replace goal\n"
                "/goal pause|resume|complete|clear"
            ),
        )

        return True

    if cmd == "/ask":
        if not arg:
            await send(chat_id, "Usage: /ask <message>")
            return True
        try:
            work: list[dict] = [
                {"role": "user", "content": arg}
            ]
            tool_rounds = 0
            while True:
                response = await stateless_response(work)
                output = response_output(response)
                calls = response_calls(response)
                if not calls:
                    answer = response_text(response)
                    if not answer:
                        raise RuntimeError("Model returned no text")
                    break
                if MAX_TOOL_ROUNDS and tool_rounds >= MAX_TOOL_ROUNDS:
                    raise RuntimeError(
                        "Configured tool-round limit reached"
                    )
                tool_rounds += 1
                work.extend(output)
                for call in calls:
                    execution_context = dict(state)
                    execution_context["session_id"] = state.get(
                        "current_session_id"
                    )
                    execution_context["todo_state"] = state
                    result = await execute_tool_with_approval(
                        chat_id,
                        call,
                        execution_context=execution_context,
                    )
                    work.append(
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": json.dumps(
                                result,
                                ensure_ascii=False,
                            ),
                        }
                    )
        except Exception as exc:
            await send(
                chat_id,
                f"Ask error: {type(exc).__name__}: {exc}",
            )
            return True
        await send(chat_id, answer)
        return True

    if cmd == "/thread":
        action = arg.lower()
        thread = current_thread()
        if action in {"", "status"} and thread:
            await send(
                chat_id,
                (
                    "Thread active.\n"
                    f"Branch: {str(thread.get('session_id') or '')[-8:]}\n"
                    "Use /thread clear or /thread merge."
                ),
            )
            return True
        if action == "status":
            await send(chat_id, "No thread is active.")
            return True
        if not action:
            result = await start_thread(chat_id)
            if result["ok"]:
                await send(
                    chat_id,
                    "Thread started. Normal messages now use the branch.",
                )
            else:
                await send(chat_id, str(result["error"]))
            return True
        if action not in {"clear", "merge"}:
            if not thread:
                result = await start_thread(chat_id)
                if not result["ok"]:
                    await send(chat_id, str(result["error"]))
                    return True
                await send(
                    chat_id,
                    "Thread started. Normal messages now use the branch.",
                )
            return arg
        result = await finish_thread(
            chat_id,
            merge=action == "merge",
        )
        if not result["ok"]:
            await send(chat_id, str(result["error"]))
            return True
        if action == "merge":
            await send(
                chat_id,
                f"Thread merged ({result['merged_items']} events).",
            )
        else:
            await send(chat_id, "Thread cleared.")
        resume_main_goal(chat_id)
        return True

    if cmd == "/stop":
        paused = False
        if goal_is_active() and not thread_active():
            _goal, changed = pause_goal()
            paused = changed
            if paused:
                await remove_goal_pin(chat_id)
        stopped = False
        if (
            session.active_task
            and not session.active_task.done()
        ):
            session.active_task.cancel()
            stopped = True

            try:
                await session.active_task

            except asyncio.CancelledError:
                pass
            except Exception as exc:
                log_exception("agent", "stop_wait_error", exc)

        cancelled_subagents = (
            await cancel_background_subagents()
        )
        cancelled_shells = await cancel_shell_sessions()

        if (
            stopped
            or cancelled_subagents
            or cancelled_shells
            or paused
        ):
            message = "Stopped."
            if paused:
                message = (
                    "Stopped. Active goal paused."
                )
            if cancelled_subagents:
                message += (
                    "\nCancelled background subagents: "
                    f"{cancelled_subagents}"
                )
            if cancelled_shells:
                message += (
                    "\nCancelled background shells: "
                    f"{cancelled_shells}"
                )
            await send(
                chat_id,
                message,
            )

        else:
            await send(
                chat_id,
                "Nothing is running.",
            )

        return True

    if cmd == "/status":
        try:
            models = (
                await list_models()
            )

            responses_status = "up"

        except Exception as exc:
            log_exception("responses", "model_list_error", exc)
            models = []
            responses_status = (
                "down/unreachable"
            )

        running = bool(
            session.active_task
            and not session.active_task.done()
        )

        task_age = (
            int(
                time.monotonic()
                - session.active_since
            )
            if (
                running
                and session.active_since
            )
            else 0
        )

        budget = context_input_budget()
        subagent_budget = subagent_context_input_budget()
        subagent_limit_line = (
            f"subagent context limit: "
            f"{SUBAGENT_MODEL_CONTEXT_TOKENS:,} tokens\n"
            f"subagent compact threshold: {subagent_budget:,} tokens"
            if SUBAGENT_MODEL_CONTEXT_TOKENS
            else (
                "subagent context limit: unknown\n"
                "subagent auto-compact: disabled"
            )
        )
        session_id = state.get("current_session_id") or "(none)"
        stats = (
            session_context_stats(str(session_id))
            if session_id and session_id != "(none)"
            and session_exists(str(session_id))
            else {
                "archive_events": 0,
                "checkpoint": False,
                "checkpoint_through": None,
                "active_native_events": 0,
            }
        )
        if MODEL_CONTEXT_TOKENS:
            context_limit_line = (
                f"context estimate: ~{context_tokens():,} tokens\n"
                f"main context limit: {MODEL_CONTEXT_TOKENS:,} tokens\n"
                f"main compact threshold: {budget:,} tokens "
                f"({int(CONTEXT_COMPACT_RATIO * 100)}%)\n"
                f"{subagent_limit_line}\n"
                f"shared safety reserve: {CONTEXT_SAFETY_TOKENS:,} tokens"
            )
        else:
            context_limit_line = (
                f"context estimate: ~{context_tokens():,} tokens\n"
                f"main context limit: unknown\n"
                f"main auto-compact: disabled\n"
                f"{subagent_limit_line}"
            )
        checkpoint_line = (
            f"checkpoint: yes\n"
            f"checkpoint through: {stats['checkpoint_through']}"
            if stats["checkpoint"]
            else "checkpoint: no"
        )
        reasoning_effort = RESPONSES_REASONING_EFFORT or "(model default)"
        subagent_reasoning_effort = (
            SUBAGENT_RESPONSES_REASONING_EFFORT or "(model default)"
        )
        interrupted_agent = "yes" if state.get("interrupted_agent") else "no"
        interrupted_subagents = len(state.get("interrupted_subagents", []))
        await send(
            chat_id,
            (
                f"provider: {get_chat_provider().name}\n"
                f"Responses: {responses_status}\n"
                f"model: {state.get('model') or '(auto)'}\n"
                f"reasoning effort: {reasoning_effort}\n"
                f"subagent reasoning effort: {subagent_reasoning_effort}\n"
                f"served models: {len(models)}\n"
                f"pid: {os.getpid()}\n"
                f"process: {'daemon' if session.DAEMON_MODE else 'foreground'}\n"
                f"uptime: {runtime_uptime()}s\n"
                f"running task: {'yes' if running else 'no'}\n"
                f"task age: {task_age}s\n"
                f"pending steers: {len(session.pending_steers)}\n"
                f"thread: {'active' if thread_active() else 'none'}\n"
                f"approval: {state.get('approval_mode', 'off')}\n"
                f"pending approvals: {len(pending_approvals)}\n"
                f"background subagents: {len(session.subagent_tasks)}\n"
                f"background shells: {running_shell_sessions()}\n"
                f"subagent threads: {len(session.subagent_records)}\n"
                f"interrupted agent: {interrupted_agent}\n"
                f"interrupted subagents: {interrupted_subagents}\n"
                f"{format_goal_status()}\n"
                f"session: {session_id}\n"
                f"archive events: {stats['archive_events']}\n"
                f"{checkpoint_line}\n"
                f"active native events: {stats['active_native_events']}\n"
                f"{context_limit_line}\n"
                f"todos: {len(state.get('todos', []))}\n"
                f"workdir: {AGENT_WORKDIR}\n"
                f"streaming: {'on' if CHAT_STREAMING else 'off'}"
            ),
        )

        return True

    if cmd == "/goal":
        if thread_active():
            await send(
                chat_id,
                "Persistent goals are unavailable inside a thread.",
            )
            return True
        action = arg.lower()
        if not arg:
            goal = current_goal()
            if not goal:
                await send(chat_id, "No goal.")
                return True
            status = str(goal.get("status") or "")
            objective = str(
                goal.get("objective") or ""
            )
            message = (
                f"Goal: {status}\n{objective}"
            )
            if (
                status == GOAL_BLOCKED
                and goal.get("blocked_reason")
            ):
                message += (
                    "\nBlocked: "
                    f"{goal['blocked_reason']}"
                )
            await send(chat_id, message)
            return True
        if action == "pause":
            goal, changed = pause_goal()
            if not goal:
                await send(chat_id, "No goal.")
                return True
            if not changed:
                await send(
                    chat_id,
                    (
                        "Goal is "
                        f"{goal.get('status')}; "
                        "nothing to pause."
                    ),
                )
                return True
            await remove_goal_pin(chat_id)
            cancelled = cancel_active_goal_controller(
                str(goal.get("id") or "")
            )
            if cancelled is not None:
                try:
                    await cancelled
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    log_exception("goal", "pause_wait_error", exc)
            await send(
                chat_id,
                "Goal paused.",
            )
            return True
        if action == "resume":
            goal = current_goal()
            if not goal:
                await send(chat_id, "No goal.")
                return True
            if goal.get("status") not in {
                GOAL_PAUSED,
                GOAL_BLOCKED,
            }:
                await send(
                    chat_id,
                    (
                        "Goal is "
                        f"{goal.get('status')}; "
                        "nothing to resume."
                    ),
                )
                return True
            resume_goal()
            await ensure_goal_pin(chat_id)
            if not (
                session.active_task
                and not session.active_task.done()
            ):
                start_root_session(chat_id, None)
            await send(
                chat_id,
                "Goal resumed.",
            )
            return True
        if action == "complete":
            goal = current_goal()
            if not goal:
                await send(chat_id, "No goal.")
                return True
            complete_goal(str(goal["id"]))
            await remove_goal_pin(chat_id)
            await send(
                chat_id,
                f"Goal complete: {goal.get('objective')}",
            )
            return True
        if action == "clear":
            goal = current_goal()
            if not goal:
                await send(chat_id, "No goal.")
                return True
            goal_id = str(goal.get("id") or "")
            clear_goal()
            await remove_goal_pin(chat_id)
            cancelled = cancel_active_goal_controller(
                goal_id
            )
            if cancelled is not None:
                try:
                    await cancelled
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    log_exception("goal", "clear_wait_error", exc)
            await send(chat_id, "Goal cleared.")
            return True
        existing = current_goal()
        if (
            existing
            and existing.get("status")
            in {
                GOAL_ACTIVE,
                GOAL_PAUSED,
                GOAL_BLOCKED,
            }
        ):
            await send(
                chat_id,
                (
                    "An unfinished goal already exists. "
                    "Clear or complete it first.\n"
                    f"Current ({existing.get('status')}): "
                    f"{existing.get('objective')}"
                ),
            )
            return True
        try:
            goal = replace_goal(arg)
        except ValueError as exc:
            await send(chat_id, str(exc))
            return True
        await ensure_goal_pin(chat_id)
        await send(
            chat_id,
            f"Goal started: {goal['objective']}",
        )
        if not (
            session.active_task
            and not session.active_task.done()
        ):
            start_root_session(chat_id, None)
        else:
            ensure_root_session(chat_id, None)
        return True

    if cmd == "/new":
        if thread_active():
            await send(
                chat_id,
                "Clear or merge the active thread first.",
            )
            return True
        if (
            (
                session.active_task
                and not session.active_task.done()
            )
            or session.subagent_tasks
            or running_shell_sessions()
        ):
            await send(
                chat_id,
                "Busy. Use /stop first.",
            )
            return True
        old_session_id = state.get("current_session_id")
        if old_session_id and session_exists(str(old_session_id)):
            end_session(
                str(old_session_id),
                reason="new_session",
                todos=list(state.get("todos") or []),
            )
        model = state.get("model")
        approval_mode = state.get("approval_mode", "off")
        runtime = copy.deepcopy(state.get("runtime", {}))
        subagents = copy.deepcopy(state.get("subagents", {}))
        goal = copy.deepcopy(state.get("goal"))
        new_session_id = create_session(kind="main")
        state.clear()
        state.update(copy.deepcopy(DEFAULT_STATE))
        state["model"] = model
        state["approval_mode"] = approval_mode
        state["runtime"] = runtime
        state["subagents"] = subagents
        state["goal"] = goal
        state["current_session_id"] = new_session_id
        state["todos"] = []
        state["pending_inputs"] = []
        state["interrupted_agent"] = None
        state["interrupted_subagents"] = []
        session.subagent_records = state["subagents"]
        save_state()
        short_id = str(new_session_id)[-8:]
        await send(
            chat_id,
            (
                f"New session: {short_id}\n"
                "Previous transcript archived."
            ),
        )
        return True

    if cmd == "/compact":
        if (
            session.active_task
            and not session.active_task.done()
        ):
            await send(
                chat_id,
                "Busy. Use /stop first.",
            )
            return True
        start_manual_compaction(chat_id)
        return True

    if cmd == "/approval":
        if not arg:
            await send(
                chat_id,
                (
                    "Approval: "
                    f"{state.get('approval_mode', 'off')}\n"
                    "Protected tools: shell, write_file, edit_file"
                ),
            )
            return True
        mode = arg.lower()
        if mode not in {"on", "off"}:
            await send(
                chat_id,
                "Usage: /approval on | /approval off",
            )
            return True
        state["approval_mode"] = mode
        save_state()
        sid = state.get("current_session_id")
        if sid:
            append_meta(
                str(sid),
                "approval_changed",
                {"mode": mode},
            )
        pending = len(pending_approvals)
        suffix = (
            "\nExisting pending approvals still require a decision."
            if pending
            else ""
        )
        await send(chat_id, f"Approval: {mode}" + suffix)
        return True

    if cmd == "/todos":
        await send(
            chat_id,
            format_todos(),
        )

        return True

    if cmd == "/model":
        if (
            session.active_task
            and not session.active_task.done()
        ):
            await send(
                chat_id,
                "Busy. Use /stop first.",
            )

            return True

        models = (
            await list_models()
        )

        if not arg:
            current = state.get(
                "model"
            )

            lines = [
                (
                    f"{'*' if model == current else ' '} "
                    f"{model}"
                )
                for model in models
            ]

            await send(
                chat_id,
                (
                    "Models:\n"
                    + "\n".join(
                        lines
                    )
                ),
            )

            return True

        if arg not in models:
            await send(
                chat_id,
                (
                    "Unknown model. "
                    "Use /model to list "
                    "served models."
                ),
            )

            return True

        state[
            "model"
        ] = arg

        save_state()
        sid = state.get("current_session_id")
        if sid:
            append_meta(
                str(sid),
                "model_changed",
                {"model": arg},
            )

        await send(
            chat_id,
            (
                f"Model: {arg}\n"
                "Session kept."
            ),
        )

        return True

    return False
