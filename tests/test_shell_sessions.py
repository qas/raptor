import asyncio
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-shell-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)

import session
import shell_sessions
from config import TOOLS
from shell_sessions import (
    HeadTailBuffer,
    cancel_shell_session,
    cancel_shell_sessions,
    reset_shell_sessions_for_tests,
    run_shell,
    running_shell_sessions,
    supervisor_argv,
    write_stdin,
)
from tools import shell_tool


class ShellSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.runtime_context = session.bound_chat("telegram:123")
        self.runtime_context.__enter__()
        self.addCleanup(
            self.runtime_context.__exit__,
            None,
            None,
            None,
        )
        await reset_shell_sessions_for_tests()

    async def asyncTearDown(self) -> None:
        await reset_shell_sessions_for_tests()

    async def test_fast_command_returns_inline(self) -> None:
        with patch.object(shell_sessions, "log_shell_start") as audit:
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
        audit.assert_called_once()
        self.assertEqual(
            audit.call_args.kwargs["command"],
            "printf fast",
        )
        self.assertEqual(
            audit.call_args.kwargs["parent_session_id"],
            "main-1",
        )

    async def test_audit_failure_prevents_command_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "started"
            with patch.object(
                shell_sessions,
                "log_shell_start",
                side_effect=OSError("log unavailable"),
            ):
                result = await run_shell(
                    f"touch {marker}",
                    timeout=2,
                    yield_time_ms=1000,
                    tty=False,
                    chat_id="telegram:123",
                    parent_session_id="main-1",
                )

        self.assertFalse(result["ok"])
        self.assertIn("audit failed", result["error"])
        self.assertFalse(marker.exists())

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

    async def test_ctrl_c_is_forwarded_to_command_process_group(self) -> None:
        result = await run_shell(
            "sleep 30",
            timeout=30,
            yield_time_ms=250,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )

        completed = await write_stdin(
            {
                "session_id": result["session_id"],
                "chars": "\x03",
                "yield_time_ms": 2000,
            }
        )

        self.assertEqual(completed["status"], "failed")
        self.assertIsNotNone(completed["exit_code"])

    async def test_start_gate_failure_aborts_supervisor_before_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "started"
            with patch.object(
                shell_sessions,
                "_release_start_gate",
                side_effect=OSError("gate unavailable"),
            ):
                result = await run_shell(
                    f"touch {marker}",
                    timeout=2,
                    yield_time_ms=1000,
                    tty=False,
                    chat_id="telegram:123",
                    parent_session_id="main-1",
                )

        self.assertFalse(result["ok"])
        self.assertIn("start failed", result["error"])
        self.assertFalse(marker.exists())
        self.assertFalse(shell_sessions._sessions)

    async def test_interactive_input_is_bounded(self) -> None:
        result = await run_shell(
            "sleep 10",
            timeout=30,
            yield_time_ms=250,
            tty=True,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )
        with patch.object(shell_sessions, "MAX_TOOL_OUTPUT", 4):
            rejected = await write_stdin(
                {
                    "session_id": result["session_id"],
                    "chars": "12345",
                }
            )

        self.assertFalse(rejected["ok"])
        self.assertIn("exceeds 4", rejected["error"])

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

    async def test_default_timeout_is_unlimited(self) -> None:
        with patch.object(shell_sessions, "SHELL_TIMEOUT", 0):
            result = await run_shell(
                "printf unlimited",
                timeout=None,
                yield_time_ms=1000,
                tty=False,
                chat_id="telegram:123",
                parent_session_id="main-1",
            )

        shell_session = next(iter(shell_sessions._sessions.values()))
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "unlimited")
        self.assertIsNone(shell_session.timeout)

    async def test_zero_timeout_disables_deadline(self) -> None:
        with patch.object(shell_sessions, "SHELL_TIMEOUT", 1):
            result = await run_shell(
                "printf unlimited",
                timeout=0,
                yield_time_ms=1000,
                tty=False,
                chat_id="telegram:123",
                parent_session_id="main-1",
            )

        shell_session = next(iter(shell_sessions._sessions.values()))
        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "unlimited")
        self.assertIsNone(shell_session.timeout)

    async def test_timeout_is_not_artificially_capped(self) -> None:
        result = await run_shell(
            "true",
            timeout=900,
            yield_time_ms=1000,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )

        shell_session = next(iter(shell_sessions._sessions.values()))
        self.assertTrue(result["ok"])
        self.assertEqual(shell_session.timeout, 900)

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

    async def test_cancel_waits_for_supervisor_to_kill_stubborn_group(self) -> None:
        result = await run_shell(
            "trap '' TERM; sleep 30",
            timeout=30,
            yield_time_ms=250,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )

        stopped = await asyncio.wait_for(
            cancel_shell_session(result["session_id"]), timeout=2.5
        )

        self.assertTrue(stopped["ok"])
        self.assertEqual(stopped["status"], "cancelled")

    async def test_session_controls_cannot_cross_chat_boundaries(self) -> None:
        result = await run_shell(
            "sleep 10",
            timeout=30,
            yield_time_ms=250,
            tty=False,
            chat_id="telegram:123",
            parent_session_id="main-1",
        )
        session_id = result["session_id"]

        with session.bound_chat("responses_api:other"):
            polled = await write_stdin({
                "session_id": session_id,
                "yield_time_ms": 0,
            })
            cancelled = await cancel_shell_session(session_id)

        self.assertFalse(polled["ok"])
        self.assertFalse(cancelled["ok"])
        self.assertEqual(
            shell_sessions._sessions[session_id].status,
            "running",
        )

    async def test_detached_completion_enters_internal_event_path(self) -> None:
        completion = asyncio.get_running_loop().create_future()
        delivered = Mock(return_value=completion)
        controller = types.ModuleType("controller")
        controller.enqueue_runtime_event = delivered
        with patch.dict(sys.modules, {"controller": controller}):
            result = await run_shell(
                "sleep 0.5; printf notified",
                timeout=2,
                yield_time_ms=250,
                tty=False,
                chat_id="telegram:123",
                parent_session_id=None,
            )
            self.assertIsNotNone(result["session_id"])
            await asyncio.wait_for(_wait_until_called(delivered), timeout=1)

        delivered.assert_called_once()
        self.assertEqual(delivered.call_args.args[0], "telegram:123")
        self.assertIn("notified", delivered.call_args.args[2])

    async def test_completion_delivery_exception_is_deferred(self) -> None:
        delivered = Mock(side_effect=RuntimeError("controller failed"))
        controller = types.ModuleType("controller")
        controller.enqueue_runtime_event = delivered
        with (
            patch.dict(sys.modules, {"controller": controller}),
            patch.object(shell_sessions, "log_event") as logged,
        ):
            result = await run_shell(
                "sleep 0.5; printf retry",
                timeout=2,
                yield_time_ms=250,
                tty=False,
                chat_id="telegram:123",
                parent_session_id=None,
            )
            session_id = result["session_id"]
            await asyncio.wait_for(_wait_until_called(delivered), timeout=1)

        item = shell_sessions._sessions[session_id]
        self.assertEqual(item.completion_attempts, 1)
        self.assertTrue(item.completion_pending)
        self.assertIn(
            "completion_delivery_error",
            [call.args[1] for call in logged.call_args_list],
        )

    async def test_liveness_guard_terminates_process_group_on_parent_eof(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "orphan-finished"
            result = await run_shell(
                f"sleep 2; printf orphaned > {marker}",
                timeout=0,
                yield_time_ms=250,
                tty=False,
                chat_id="telegram:123",
                parent_session_id="main-1",
            )
            shell_session = shell_sessions._sessions[result["session_id"]]

            shell_sessions._close_liveness_guard(shell_session)
            await asyncio.wait_for(shell_session.done.wait(), timeout=2)

            self.assertFalse(marker.exists())
            self.assertNotEqual(shell_session.exit_code, 0)

    async def test_normal_exit_terminates_background_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "descendant-finished"
            result = await run_shell(
                f"(sleep 1; touch {marker}) & printf background",
                timeout=3,
                yield_time_ms=2000,
                tty=False,
                chat_id="telegram:123",
                parent_session_id="main-1",
            )

            self.assertEqual(result["status"], "completed")
            await asyncio.sleep(1.2)
            self.assertFalse(marker.exists())

    def test_pruning_preserves_pending_completions(self) -> None:
        for index in range(3):
            shell_sessions._sessions[str(index)] = shell_sessions.ShellSession(
                id=str(index),
                command="true",
                chat_id="telegram:123",
                chat_key="telegram:123",
                parent_session_id=None,
                process=AsyncMock(),
                timeout=None,
                status="completed",
                completion_pending=index == 0,
                completed_at=float(index),
            )

        with patch.object(shell_sessions, "MAX_RETAINED_SESSIONS", 1):
            shell_sessions._prune_sessions()

        self.assertIn("0", shell_sessions._sessions)

    async def test_deferred_completion_requeues_on_user_activity(self) -> None:
        item = shell_sessions.ShellSession(
            id="shell-1",
            command="true",
            chat_id="telegram:123",
            chat_key=session.current_runtime().key,
            parent_session_id=None,
            process=AsyncMock(),
            timeout=1,
            completion_pending=True,
            completion_attempts=1,
        )
        shell_sessions._sessions[item.id] = item
        completion = asyncio.get_running_loop().create_future()
        controller = types.ModuleType("controller")
        controller.enqueue_runtime_event = Mock(return_value=completion)

        with patch.dict(sys.modules, {"controller": controller}):
            count = await shell_sessions.requeue_deferred_shell_completions()

        self.assertEqual(count, 1)
        self.assertEqual(item.completion_attempts, 0)
        controller.enqueue_runtime_event.assert_called_once()

    def test_tool_schemas_expose_managed_session_controls(self) -> None:
        schemas = {item["name"]: item for item in TOOLS}

        self.assertIn("yield_time_ms", schemas["shell"]["parameters"]["properties"])
        self.assertIn("tty", schemas["shell"]["parameters"]["properties"])
        timeout = schemas["shell"]["parameters"]["properties"]["timeout"]
        self.assertEqual(timeout["minimum"], 0)
        self.assertNotIn("maximum", timeout)
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

    def test_supervisor_argv_uses_source_script(self) -> None:
        argv = supervisor_argv()
        self.assertEqual(argv[0], shell_sessions._SUPERVISOR_EXECUTABLE)
        self.assertEqual(argv[1], str(shell_sessions._SUPERVISOR_PATH))

    def test_supervisor_argv_uses_frozen_executable(self) -> None:
        with (
            patch.object(
                shell_sessions,
                "_SUPERVISOR_EXECUTABLE",
                "/opt/raptor/raptor",
            ),
            patch.object(sys, "frozen", True, create=True),
        ):
            self.assertEqual(
                supervisor_argv(),
                ["/opt/raptor/raptor", "_shell-supervisor"],
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

    def test_stdout_and_stderr_share_one_output_budget(self) -> None:
        with patch.object(shell_sessions, "MAX_TOOL_OUTPUT", 100):
            stdout, stderr, truncated = shell_sessions._fit_output_pair(
                "a" * 100,
                "b" * 100,
            )

        self.assertTrue(truncated)
        self.assertLessEqual(len(stdout) + len(stderr), 100)
        self.assertIn("truncated", stdout)
        self.assertIn("truncated", stderr)


async def _wait_until_called(mock: Mock) -> None:
    while not mock.called:
        await asyncio.sleep(0.005)


if __name__ == "__main__":
    unittest.main()
