import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

import shell_sessions
from config import TOOLS
from shell_sessions import (
    HeadTailBuffer,
    cancel_shell_session,
    cancel_shell_sessions,
    reset_shell_sessions_for_tests,
    run_shell,
    running_shell_sessions,
    shell_completion_event_loop,
    write_stdin,
)
from tools import shell_tool


class ShellSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await reset_shell_sessions_for_tests()

    async def asyncTearDown(self) -> None:
        await reset_shell_sessions_for_tests()

    async def test_fast_command_returns_inline(self) -> None:
        result = await run_shell(
            "printf fast",
            timeout=2,
            yield_time_ms=1000,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stdout"], "fast")
        self.assertIsNone(result["session_id"])

    async def test_slow_command_yields_then_can_be_polled(self) -> None:
        result = await run_shell(
            "sleep 0.5; printf done",
            timeout=2,
            yield_time_ms=250,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )

        self.assertEqual(result["status"], "running")
        self.assertIsNotNone(result["session_id"])
        polled = await write_stdin(
            {
                "session_id": result["session_id"],
                "yield_time_ms": 1000,
            }
        )
        self.assertEqual(polled["status"], "completed")
        self.assertEqual(polled["stdout"], "done")

    async def test_plain_session_has_closed_stdin(self) -> None:
        result = await run_shell(
            "cat; printf closed",
            timeout=2,
            yield_time_ms=1000,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stdout"], "closed")
        self.assertIsNone(result["session_id"])

    async def test_timeout_terminates_process(self) -> None:
        result = await run_shell(
            "sleep 10",
            timeout=1,
            yield_time_ms=250,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )
        completed = await write_stdin(
            {
                "session_id": result["session_id"],
                "yield_time_ms": 2000,
            }
        )

        self.assertFalse(completed["ok"])
        self.assertEqual(completed["status"], "timed_out")
        self.assertIn("timed out", completed["error"])

    async def test_cancel_stops_all_running_sessions(self) -> None:
        result = await run_shell(
            "sleep 10",
            timeout=30,
            yield_time_ms=250,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )
        self.assertEqual(running_shell_sessions(), 1)

        cancelled = await cancel_shell_sessions()

        self.assertEqual(cancelled, 1)
        self.assertEqual(running_shell_sessions(), 0)
        polled = await write_stdin(
            {"session_id": result["session_id"], "yield_time_ms": 0}
        )
        self.assertEqual(polled["status"], "cancelled")

    async def test_targeted_cancel_stops_one_process_group(self) -> None:
        first = await run_shell(
            "sleep 10",
            timeout=30,
            yield_time_ms=250,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )
        second = await run_shell(
            "sleep 10",
            timeout=30,
            yield_time_ms=250,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )

        result = await cancel_shell_session(first["session_id"])

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(running_shell_sessions(), 1)
        first_state = await write_stdin(
            {"session_id": first["session_id"], "yield_time_ms": 0}
        )
        self.assertEqual(first_state["status"], "cancelled")
        self.assertEqual(
            shell_sessions._sessions[second["session_id"]].status,
            "running",
        )

    async def test_detached_completion_enters_internal_event_path(self) -> None:
        delivered = AsyncMock(return_value=True)
        controller = types.ModuleType("controller")
        controller.enqueue_internal_input = delivered
        with patch.dict(sys.modules, {"controller": controller}):
            event_task = asyncio.create_task(shell_completion_event_loop())
            try:
                result = await run_shell(
                    "sleep 0.5; printf notified",
                    timeout=2,
                    yield_time_ms=250,
                    tty=False,
                    chat_id="telegram:123",
                    parent_session_id=None,
                )
                self.assertIsNotNone(result["session_id"])
                await asyncio.wait_for(
                    _wait_until_called(delivered),
                    timeout=1,
                )
            finally:
                event_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await event_task

        delivered.assert_awaited_once()
        self.assertEqual(delivered.await_args.args[0], "telegram:123")
        self.assertIn("notified", delivered.await_args.args[1])

    async def test_completion_delivery_exception_is_deferred(self) -> None:
        delivered = AsyncMock(side_effect=RuntimeError("controller failed"))
        controller = types.ModuleType("controller")
        controller.enqueue_internal_input = delivered
        with (
            patch.dict(sys.modules, {"controller": controller}),
            patch.object(shell_sessions, "log_event") as logged,
        ):
            event_task = asyncio.create_task(shell_completion_event_loop())
            try:
                result = await run_shell(
                    "sleep 0.5; printf retry",
                    timeout=2,
                    yield_time_ms=250,
                    tty=False,
                    chat_id="telegram:123",
                    parent_session_id=None,
                )
                session_id = result["session_id"]
                await asyncio.wait_for(
                    _wait_until_called(delivered),
                    timeout=1,
                )
                await asyncio.sleep(0)
            finally:
                event_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await event_task

        item = shell_sessions._sessions[session_id]
        self.assertEqual(item.completion_attempts, 1)
        self.assertTrue(item.completion_pending)
        self.assertEqual(
            [call.args[1] for call in logged.call_args_list][-2:],
            ["completion_delivery_error", "completion_deferred"],
        )

    async def test_deferred_completion_requeues_on_user_activity(self) -> None:
        item = shell_sessions.ShellSession(
            id="shell-1",
            command="true",
            chat_id="telegram:123",
            parent_session_id=None,
            process=AsyncMock(),
            timeout=1,
            completion_pending=True,
            completion_attempts=1,
        )
        shell_sessions._sessions[item.id] = item

        count = await shell_sessions.requeue_deferred_shell_completions()

        self.assertEqual(count, 1)
        self.assertEqual(item.completion_attempts, 0)
        self.assertEqual(
            shell_sessions._completion_events.get_nowait(),
            item.id,
        )

    def test_tool_schemas_expose_managed_session_controls(self) -> None:
        schemas = {item["name"]: item for item in TOOLS}

        self.assertIn("yield_time_ms", schemas["shell"]["parameters"]["properties"])
        self.assertIn("tty", schemas["shell"]["parameters"]["properties"])
        self.assertIn("write_stdin", schemas)
        self.assertEqual(
            schemas["cancel"]["parameters"]["required"],
            ["kind", "id"],
        )

    async def test_tty_session_accepts_interactive_input(self) -> None:
        result = await run_shell(
            "IFS= read -r line; printf 'tty:%s' \"$line\"",
            timeout=2,
            yield_time_ms=250,
            tty=True,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )

        completed = await write_stdin(
            {
                "session_id": result["session_id"],
                "chars": "hello\n",
                "yield_time_ms": 1000,
            }
        )
        self.assertEqual(completed["status"], "completed")
        self.assertIn("tty:hello", completed["stdout"])

    async def test_subagent_shell_is_scoped_to_root_session(self) -> None:
        launched = AsyncMock(return_value={"ok": True})
        with patch("shell_sessions.run_shell", launched):
            await shell_tool(
                {"command": "true"},
                chat_id="telegram:123",
                execution_context={
                    "session_id": "subagent-session",
                    "parent_session_id": "root-session",
                },
            )

        self.assertEqual(
            launched.await_args.kwargs["parent_session_id"],
            "root-session",
        )


class HeadTailBufferTests(unittest.TestCase):
    def test_incremental_truncation_counts_each_omitted_character_once(self) -> None:
        buffer = HeadTailBuffer(limit=1000)
        source = "a" * 600 + "b" * 600 + "c" * 600

        buffer.append(source[:600])
        buffer.append(source[600:1200])
        buffer.append(source[1200:])

        self.assertEqual(buffer.omitted, 800)
        self.assertEqual(
            buffer.render(include_marker=False),
            source[:500] + source[-500:],
        )


async def _wait_until_called(mock: AsyncMock) -> None:
    while not mock.await_count:
        await asyncio.sleep(0.005)


if __name__ == "__main__":
    unittest.main()
