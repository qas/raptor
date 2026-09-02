import asyncio
import contextvars
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-console-follow-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")

import console_follow
from raptor.state import session
from chat_provider import (
    IncomingAction,
    ProcessOutputChunk,
    ProviderCapabilities,
)
from chat_runtime import set_chat_provider
from raptor.model.model_providers import ModelTarget


class _ConsoleProvider:
    name = "fake"
    authorized_user_id = 1
    primary_conversation_id = 1
    capabilities = ProviderCapabilities(controls=True)

    def __init__(self) -> None:
        self.created: list[tuple[object, str, object]] = []
        self.edited: list[tuple[object, object, str, object]] = []
        self.answers: list[tuple[str, str, bool]] = []
        self._delivery = contextvars.ContextVar(
            "console_test_delivery",
            default="request",
        )
        self.background_contexts: list[object] = []

    async def create_message(self, conversation_id, text, controls=()):
        self.created.append((conversation_id, text, controls))
        return 41

    async def edit_message(
        self,
        conversation_id,
        message_id,
        text,
        controls=(),
    ) -> None:
        self.background_contexts.append(self._delivery.get())
        self.edited.append(
            (conversation_id, message_id, text, controls)
        )

    async def answer_action(self, action_id, text="", *, alert=False):
        self.answers.append((action_id, text, alert))

    def activate_delivery_context(self, conversation_id, value):
        del conversation_id
        return self._delivery.set(value)

    def restore_delivery_context(self, token) -> None:
        self._delivery.reset(token)


class TerminalScreenTests(unittest.TestCase):
    def test_split_cursor_sequences_update_existing_screen_cells(self) -> None:
        screen = console_follow._TerminalScreen(rows=4, columns=20)
        screen.feed("first\x1b[3;")
        screen.feed("1Hsecond\x1b[K")
        screen.feed("\x1b[3;1Hupdated\x1b[K")

        self.assertEqual(screen.render(), "first\n\nupdated")

    def test_carriage_returns_replace_progress_line(self) -> None:
        screen = console_follow._TerminalScreen(rows=3, columns=20)
        screen.feed("Progress 10%\rProgress 90%\x1b[K")

        self.assertEqual(screen.render(), "Progress 90%")

    def test_output_scrolls_inside_fixed_screen(self) -> None:
        screen = console_follow._TerminalScreen(rows=2, columns=20)
        screen.feed("one\r\ntwo\r\nthree")

        self.assertEqual(screen.render(), "two\nthree")


class FollowConsoleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.provider = _ConsoleProvider()
        self.previous_provider = set_chat_provider(self.provider)
        session.set_default_model_target(
            ModelTarget("local", "test-model")
        )
        session.set_default_chat(1)
        session.state["current_session_id"] = "session-1"
        console_follow._active = None

    def tearDown(self) -> None:
        set_chat_provider(self.previous_provider)
        console_follow._active = None

    async def test_follow_streams_one_terminal_message_to_completion(
        self,
    ) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def run_shell(command, **kwargs):
            self.assertEqual(command, "watch -n 2 whoami")
            self.assertEqual(kwargs["timeout"], 0)
            self.assertTrue(kwargs["tty"])
            self.assertIsNone(kwargs["max_published_output_chars"])
            self.assertFalse(kwargs["queue_completion_event"])
            publish = kwargs["process_output"]
            await publish(ProcessOutputChunk(
                call_id="",
                session_id="shell-1",
                stream="stdout",
                text=(
                    "\x1b[H\x1b[2JEvery 2.0s: whoami"
                    "\x1b[3;1Hfirst-user"
                ),
            ))
            await publish(ProcessOutputChunk(
                call_id="",
                session_id="shell-1",
                stream="stdout",
                text="\x1b[3;1Hoperator\x1b[K",
            ))
            started.set()
            await release.wait()
            return {
                "status": "completed",
                "exit_code": 0,
                "session_id": None,
            }

        with patch.object(console_follow, "run_shell", run_shell):
            error = await console_follow.start_follow_console(
                1,
                "watch -n 2 whoami",
            )
            current = console_follow._active
            self.assertIsNotNone(current)
            assert current is not None
            await asyncio.wait_for(started.wait(), timeout=1)
            release.set()
            assert current.task is not None
            await current.task

        self.assertIsNone(error)
        self.assertEqual(len(self.provider.created), 1)
        initial = self.provider.created[0]
        self.assertIn("$ watch -n 2 whoami", initial[1])
        self.assertEqual(initial[2][0][0].label, "Stop")
        final = self.provider.edited[-1]
        self.assertIn("Every 2.0s: whoami", final[2])
        self.assertIn("operator", final[2])
        self.assertNotIn("first-user", final[2])
        self.assertIn("Process completed.", final[2])
        self.assertEqual(final[3], ())
        self.assertTrue(self.provider.background_contexts)
        self.assertEqual(set(self.provider.background_contexts), {None})
        self.assertIsNone(console_follow._active)

    async def test_stop_action_cancels_process_group_and_finishes_message(
        self,
    ) -> None:
        waiting = asyncio.Event()
        cancelled = asyncio.Event()

        async def run_shell(_command, **_kwargs):
            return {
                "status": "running",
                "session_id": "shell-1",
            }

        async def wait_shell(_session_id):
            waiting.set()
            await cancelled.wait()
            return {
                "status": "cancelled",
                "exit_code": 143,
            }

        async def cancel_shell(_session_id):
            cancelled.set()
            return {"ok": True, "status": "cancelled"}

        with (
            patch.object(console_follow, "run_shell", run_shell),
            patch.object(console_follow, "wait_shell_session", wait_shell),
            patch.object(console_follow, "cancel_shell_session", cancel_shell),
        ):
            await console_follow.start_follow_console(1, "tail -f app.log")
            current = console_follow._active
            self.assertIsNotNone(current)
            assert current is not None
            await asyncio.wait_for(waiting.wait(), timeout=1)
            handled = await console_follow.handle_follow_console_action(
                IncomingAction(
                    action_id="action-1",
                    conversation_id=1,
                    sender_id=1,
                    message_id=41,
                    data=current.action,
                )
            )
            assert current.task is not None
            await current.task

        self.assertTrue(handled)
        self.assertEqual(
            self.provider.answers,
            [("action-1", "Stopping", False)],
        )
        self.assertIn("Process stopped.", self.provider.edited[-1][2])
        self.assertEqual(self.provider.edited[-1][3], ())

    async def test_stop_before_spawn_result_is_applied_after_ownership(
        self,
    ) -> None:
        release_spawn = asyncio.Event()
        cancelled = asyncio.Event()
        cancel = AsyncMock(
            side_effect=lambda _session_id: cancelled.set()
        )

        async def run_shell(_command, **_kwargs):
            await release_spawn.wait()
            return {
                "status": "running",
                "session_id": "shell-1",
            }

        async def wait_shell(_session_id):
            await cancelled.wait()
            return {"status": "cancelled", "exit_code": 143}

        with (
            patch.object(console_follow, "run_shell", run_shell),
            patch.object(console_follow, "wait_shell_session", wait_shell),
            patch.object(console_follow, "cancel_shell_session", cancel),
        ):
            await console_follow.start_follow_console(1, "long setup")
            current = console_follow._active
            assert current is not None
            handled = await console_follow.handle_follow_console_action(
                IncomingAction(
                    action_id="action-early",
                    conversation_id=1,
                    sender_id=1,
                    message_id=41,
                    data=current.action,
                )
            )
            self.assertTrue(handled)
            release_spawn.set()
            assert current.task is not None
            await current.task

        cancel.assert_awaited_once_with("shell-1")
        self.assertIn("Process stopped.", self.provider.edited[-1][2])

    async def test_shutdown_stops_and_joins_followed_console(self) -> None:
        waiting = asyncio.Event()
        cancelled = asyncio.Event()
        cancel = AsyncMock(
            side_effect=lambda _session_id: cancelled.set()
        )

        async def run_shell(_command, **_kwargs):
            return {
                "status": "running",
                "session_id": "shell-1",
            }

        async def wait_shell(_session_id):
            waiting.set()
            await cancelled.wait()
            return {"status": "cancelled", "exit_code": 143}

        with (
            patch.object(console_follow, "run_shell", run_shell),
            patch.object(console_follow, "wait_shell_session", wait_shell),
            patch.object(console_follow, "cancel_shell_session", cancel),
        ):
            await console_follow.start_follow_console(1, "tail -f app.log")
            await asyncio.wait_for(waiting.wait(), timeout=1)
            await console_follow.close_follow_console()

        cancel.assert_awaited_once_with("shell-1")
        self.assertIsNone(console_follow._active)

    async def test_only_one_followed_console_can_run(self) -> None:
        release = asyncio.Event()

        async def run_shell(_command, **_kwargs):
            await release.wait()
            return {"status": "completed", "exit_code": 0}

        with patch.object(console_follow, "run_shell", run_shell):
            first = await console_follow.start_follow_console(1, "first")
            second = await console_follow.start_follow_console(1, "second")
            current = console_follow._active
            release.set()
            assert current is not None and current.task is not None
            await current.task

        self.assertIsNone(first)
        self.assertEqual(
            second,
            "A followed console command is already running.",
        )
        self.assertEqual(len(self.provider.created), 1)

    async def test_follow_requires_controls_for_the_conversation(self) -> None:
        self.provider.capabilities = ProviderCapabilities()

        error = await console_follow.start_follow_console(1, "tail -f log")

        self.assertEqual(
            error,
            "Follow mode requires interactive message controls.",
        )
        self.assertEqual(self.provider.created, [])


if __name__ == "__main__":
    unittest.main()
