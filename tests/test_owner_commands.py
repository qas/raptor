import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-owner-command-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")

import application_control
import commands


class ApplicationControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        application_control.take_exit_request()

    async def test_exit_activates_after_request_is_recorded(self) -> None:
        started = asyncio.Event()

        async def run_application() -> None:
            started.set()
            await asyncio.Future()

        task = asyncio.create_task(run_application())
        await started.wait()
        application_control.bind_application_task(task)

        requested = application_control.request_application_exit(
            application_control.ExitRequest.RESTART
        )

        self.assertTrue(requested)
        self.assertFalse(task.done())
        self.assertTrue(application_control.activate_application_exit())
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(
            application_control.take_exit_request(),
            application_control.ExitRequest.RESTART,
        )
        self.assertFalse(application_control.application_control_available())


class OwnerCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        state_patch = patch.object(
            commands,
            "state",
            {"current_session_id": "session-1"},
        )
        state_patch.start()
        self.addCleanup(state_patch.stop)

    async def test_console_runs_one_bounded_managed_command(self) -> None:
        send = AsyncMock()
        run_shell = AsyncMock(
            return_value={
                "status": "completed",
                "exit_code": 0,
                "stdout": "operator\n",
                "stderr": "",
                "truncated": False,
                "error": None,
            }
        )
        with (
            patch.object(commands, "send", send),
            patch.object(commands, "run_shell", run_shell),
        ):
            handled = await commands.command(1, "/console whoami")

        self.assertTrue(handled)
        run_shell.assert_awaited_once_with(
            "whoami",
            timeout=commands.CONSOLE_TIMEOUT_SECONDS,
            yield_time_ms=30_000,
            tty=False,
            chat_id=1,
            parent_session_id="session-1",
        )
        self.assertIn("Console exited with code 0.", send.await_args.args[1])
        self.assertIn("operator", send.await_args.args[1])

    async def test_console_stops_command_that_outlives_wait(self) -> None:
        send = AsyncMock()
        cancel = AsyncMock()
        with (
            patch.object(commands, "send", send),
            patch.object(
                commands,
                "run_shell",
                AsyncMock(
                    return_value={
                        "status": "running",
                        "session_id": "shell-1",
                    }
                ),
            ),
            patch.object(commands, "cancel_shell_session", cancel),
        ):
            handled = await commands.command(1, "/console sleep 60")

        self.assertTrue(handled)
        cancel.assert_awaited_once_with("shell-1")
        self.assertIn("time limit", send.await_args.args[1])

    async def test_lifecycle_commands_acknowledge_then_request_exit(self) -> None:
        for name, expected in (
            ("shutdown", application_control.ExitRequest.SHUTDOWN),
            ("restart", application_control.ExitRequest.RESTART),
        ):
            with self.subTest(command=name):
                send = AsyncMock()
                request = Mock(return_value=True)
                with (
                    patch.object(commands, "send", send),
                    patch.object(
                        commands,
                        "application_control_available",
                        return_value=True,
                    ),
                    patch.object(
                        commands,
                        "request_application_exit",
                        request,
                    ),
                ):
                    handled = await commands.command(1, f"/{name}")

                self.assertTrue(handled)
                send.assert_awaited_once()
                request.assert_called_once_with(expected)


if __name__ == "__main__":
    unittest.main()
