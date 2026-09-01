"""Chat-provider contract and provider-neutral orchestration tests."""
import asyncio
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-chat-provider-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")
os.environ.setdefault("TG_CHAT_IDS", "1")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chat_provider import (
    ActionButton,
    ChatProvider,
    Controls,
    IncomingAction,
    IncomingMessage,
    PollResult,
    ProcessOutputChunk,
    ProviderCapabilities,
)
from chat_runtime import (
    get_chat_provider,
    load_chat_provider,
    load_chat_providers,
    set_chat_provider,
)
from model_providers import ModelTarget
import session
from turn_runtime import TurnKind, turns


class FakeProvider:
    name = "fake"
    authorized_user_id = "@operator:example.org"
    primary_conversation_id = "!agent:example.org"

    def __init__(self, *, pins: bool = True) -> None:
        self.capabilities = ProviderCapabilities(
            drafts=True,
            pins=pins,
            controls=True,
            typing=True,
        )
        self.calls: list[tuple] = []
        self.next_message = 0
        self.reject_busy = False

    @staticmethod
    def encode_conversation_id(conversation_id) -> str:
        return str(conversation_id)

    @staticmethod
    def decode_conversation_id(value: str) -> str:
        return value

    def prepare_event(self, event) -> None:
        self.calls.append(("prepare_event", event))

    async def initialize(self, commands) -> None:
        self.calls.append(("initialize", commands))

    async def close(self) -> None:
        self.calls.append(("close",))

    async def poll(self, cursor, *, timeout: int) -> PollResult:
        self.calls.append(("poll", cursor, timeout))
        return PollResult((), cursor)

    async def send_text(self, conversation_id, text: str) -> None:
        self.calls.append(("send_text", conversation_id, text))

    async def send_draft(
        self,
        conversation_id,
        draft_id: int,
        text: str,
    ) -> None:
        self.calls.append(
            ("send_draft", conversation_id, draft_id, text)
        )

    async def send_reasoning_summary(
        self,
        conversation_id,
        delta: str,
    ) -> None:
        self.calls.append(
            ("send_reasoning_summary", conversation_id, delta)
        )

    async def publish_process_output(
        self,
        conversation_id,
        chunk: ProcessOutputChunk,
    ) -> None:
        self.calls.append(("process_output", conversation_id, chunk))

    async def create_message(
        self,
        conversation_id,
        text: str,
        controls: Controls = (),
    ) -> str:
        self.next_message += 1
        message_id = f"$event-{self.next_message}"
        self.calls.append(
            ("create", conversation_id, message_id, text, controls)
        )
        return message_id

    async def edit_message(
        self,
        conversation_id,
        message_id,
        text: str,
        controls: Controls = (),
    ) -> None:
        self.calls.append(
            ("edit", conversation_id, message_id, text, controls)
        )

    async def delete_message(self, conversation_id, message_id) -> None:
        self.calls.append(("delete", conversation_id, message_id))

    async def delete_messages(self, conversation_id, message_ids) -> None:
        self.calls.append(("delete_many", conversation_id, tuple(message_ids)))

    async def pin_message(self, conversation_id, message_id) -> None:
        self.calls.append(("pin", conversation_id, message_id))

    async def unpin_message(self, conversation_id, message_id) -> None:
        self.calls.append(("unpin", conversation_id, message_id))

    async def set_typing(self, conversation_id, active: bool) -> None:
        self.calls.append(("typing", conversation_id, active))

    async def reject_busy_message(self, conversation_id) -> bool:
        self.calls.append(("reject_busy", conversation_id))
        return self.reject_busy

    async def acknowledge_queued_message(self, conversation_id) -> None:
        self.calls.append(("acknowledge_queued", conversation_id))

    async def finish_event(self, event) -> None:
        self.calls.append(("finish_event", event))

    def capture_delivery_context(self, conversation_id):
        self.calls.append(("capture_delivery_context", conversation_id))
        return None

    def activate_delivery_context(self, conversation_id, delivery_context):
        self.calls.append(
            ("activate_delivery_context", conversation_id, delivery_context)
        )
        return None

    def restore_delivery_context(self, token) -> None:
        self.calls.append(("restore_delivery_context", token))

    async def answer_action(
        self,
        action_id: str,
        text: str = "",
        *,
        alert: bool = False,
    ) -> None:
        self.calls.append(("answer", action_id, text, alert))


class ConsoleFakeProvider(FakeProvider):
    def supports_tool_console(self, conversation_id) -> bool:
        return conversation_id == "!room:example.org"


class ChatProviderContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        session.set_default_model_target(ModelTarget("local", "test-model"))
        session.set_default_chat("room-1")
        self.provider = FakeProvider()
        self.previous_provider = set_chat_provider(self.provider)
        session.current_runtime().pinned_status_conversation_id = None
        session.current_runtime().pinned_status_message_id = None
        session.current_runtime().pinned_status_owner = None
        session.current_runtime().goal_pin_message_id = None
        session.current_runtime().goal_pin_goal_id = None
        session.pending_approvals.clear()
        session.pending_steers.clear()
        while True:
            try:
                session.steer_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                session.steer_queue.task_done()
        turns.finish()

    def tearDown(self) -> None:
        set_chat_provider(self.previous_provider)

    async def test_persistent_status_supports_opaque_provider_ids(self) -> None:
        from presentation import clear_pinned_status, show_pinned_status

        message_id = await show_pinned_status(
            "!room:example.org",
            "goal:g1",
            "Goal active",
        )
        self.assertEqual(message_id, "$event-1")

        same_id = await show_pinned_status(
            "!room:example.org",
            "approval:a1",
            "Approval required",
        )
        self.assertEqual(same_id, message_id)
        await clear_pinned_status(
            "!room:example.org",
            owner="approval:a1",
        )

        methods = [call[0] for call in self.provider.calls]
        self.assertEqual(
            methods,
            ["create", "pin", "edit", "unpin", "delete"],
        )

    async def test_provider_without_pins_uses_capability_path(self) -> None:
        from presentation import clear_pinned_status, show_pinned_status

        provider = FakeProvider(pins=False)
        set_chat_provider(provider)
        provider.capabilities = ProviderCapabilities(
            drafts=False,
            pins=False,
            controls=False,
            typing=False,
        )
        await show_pinned_status(
            "!room",
            "goal:g1",
            "Goal active",
            controls=((ActionButton("Approve", "approve"),),),
        )
        await clear_pinned_status("!room", owner="goal:g1")

        methods = [call[0] for call in provider.calls]
        self.assertEqual(methods, ["create", "delete"])
        self.assertEqual(provider.calls[0][-1], ())

    async def test_tool_activity_streams_then_removes_terminal_bubble(
        self,
    ) -> None:
        from tool_activity import ToolActivitySurface

        surface = ToolActivitySurface("!room:example.org")
        partial = {
            "type": "function_call",
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":',
        }
        complete = {
            **partial,
            "arguments": '{"command":"pwd"}',
        }

        with patch("tool_activity.CHAT_STREAM_INTERVAL", 0):
            await surface.stream(partial, False)
            await asyncio.sleep(0)
            await surface.stream(complete, False)
            await surface.stream(complete, True)
            await surface.running(complete)
            await surface.finished(complete, {"ok": True})
            await surface.clear()

        methods = [call[0] for call in self.provider.calls]
        self.assertEqual(
            methods,
            ["create", "edit", "edit", "edit", "delete_many"],
        )
        texts = [
            call[3]
            for call in self.provider.calls
            if call[0] in {"create", "edit"}
        ]
        preview = "**Tool:** `shell`\n\n**Command:** `pwd`"
        self.assertTrue(texts[0].startswith("Preparing tool\n\n**Tool:**"))
        self.assertIn(preview, texts[1])
        self.assertTrue(texts[2].startswith("Running\n\n" + preview))
        self.assertTrue(texts[3].startswith("Completed\n\n" + preview))
        self.assertIsNone(session.current_runtime().pinned_status_owner)

    def test_tool_preview_is_bounded_and_humanizes_field_names(self) -> None:
        from tool_activity import MAX_TOOL_PREVIEW_CHARS, tool_preview

        preview = tool_preview({
            "name": "write_stdin",
            "arguments": json.dumps({
                "session_id": "shell-1",
                "yield_time_ms": 300_000,
                "unsafe**label": "safe",
                "content": "x" * 10_000,
                "later": "omitted",
            }),
        })

        self.assertLessEqual(len(preview), MAX_TOOL_PREVIEW_CHARS)
        self.assertIn("**Session ID:** `shell-1`", preview)
        self.assertIn("**Yield time (ms):** `300000`", preview)
        self.assertIn("**Unsafe label:** `safe`", preview)
        self.assertIn("... [truncated]", preview)
        self.assertNotIn("**Later:**", preview)
        self.assertTrue(preview.endswith("```"))

    async def test_tool_activity_exposes_process_output_to_provider(
        self,
    ) -> None:
        from tool_activity import ToolActivitySurface

        surface = ToolActivitySurface("!room:example.org")
        chunk = ProcessOutputChunk(
            call_id="c1",
            session_id="shell-1",
            stream="stdout",
            text="working\n",
        )

        await surface.publish_process_output(chunk)

        self.assertEqual(
            self.provider.calls,
            [("process_output", "!room:example.org", chunk)],
        )

    async def test_disabled_tool_activity_suppresses_transient_bubbles(
        self,
    ) -> None:
        from tool_activity import ToolActivitySurface

        surface = ToolActivitySurface("!room:example.org", enabled=False)
        call = {
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"pwd"}',
        }
        chunk = ProcessOutputChunk(
            call_id="c1",
            session_id="shell-1",
            stream="stdout",
            text="working\n",
        )

        await surface.stream(call, True)
        await surface.running(call)
        await surface.publish_process_output(chunk)
        await surface.finished(call, {"ok": True})
        await surface.clear()

        self.assertEqual(
            self.provider.calls,
            [("process_output", "!room:example.org", chunk)],
        )

    async def test_disabled_tool_activity_keeps_required_approval(
        self,
    ) -> None:
        from tool_activity import ToolActivitySurface

        surface = ToolActivitySurface("!room:example.org", enabled=False)
        call = {
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"touch marker"}',
        }
        controls = ((
            ActionButton("Approve", "approve"),
            ActionButton("Deny", "deny"),
        ),)

        await surface.stream(call, True)
        await surface.approval(call, controls)
        await surface.running(call)
        await surface.finished(call, {"ok": True})
        await surface.clear()

        methods = [item[0] for item in self.provider.calls]
        self.assertEqual(methods, ["create", "edit", "edit", "delete_many"])
        self.assertEqual(self.provider.calls[0][4], controls)

    async def test_tool_console_toggles_and_streams_tail(
        self,
    ) -> None:
        from tool_activity import (
            ToolActivitySurface,
            handle_tool_activity_action,
        )

        self.provider = ConsoleFakeProvider()
        set_chat_provider(self.provider)
        surface = ToolActivitySurface("!room:example.org")
        call = {
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"python -m unittest"}',
        }

        with patch("tool_activity.CHAT_STREAM_INTERVAL", 0):
            await surface.running(call)
            created = self.provider.calls[0]
            controls = created[4]
            self.assertEqual(
                [button.label for button in controls[0]],
                ["Info"],
            )
            self.assertEqual(
                created[3],
                "```bash\n$ python -m unittest\n```",
            )
            info_action_data = controls[0][0].action
            await surface.publish_process_output(
                ProcessOutputChunk(
                    call_id="c1",
                    session_id="shell-1",
                    stream="stdout",
                    text="".join(f"line {line}\n" for line in range(1, 10)),
                )
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            console_text = next(
                call[3]
                for call in reversed(self.provider.calls)
                if call[0] == "edit"
            )
            self.assertTrue(console_text.startswith(
                "```bash\n$ python -m unittest\n"
            ))
            self.assertIn("line 3", console_text)
            self.assertNotIn("line 2\n", console_text)
            self.assertTrue(console_text.endswith("line 9\n```"))

            info_action = IncomingAction(
                action_id="callback-1",
                conversation_id="!room:example.org",
                sender_id="@operator:example.org",
                message_id="$event-1",
                data=info_action_data,
            )
            self.assertTrue(await handle_tool_activity_action(info_action))
            info_edit = next(
                call
                for call in reversed(self.provider.calls)
                if call[0] == "edit"
            )
            self.assertTrue(info_edit[3].startswith(
                "Running\n\n**Tool:** `shell`"
            ))
            self.assertEqual(
                [button.label for button in info_edit[4][0]],
                ["Console"],
            )
            console_action_data = info_edit[4][0][0].action

            edits_before_output = sum(
                call[0] == "edit" for call in self.provider.calls
            )
            await surface.publish_process_output(
                ProcessOutputChunk(
                    call_id="c1",
                    session_id="shell-1",
                    stream="stderr",
                    text="line 10\n",
                )
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertEqual(
                sum(call[0] == "edit" for call in self.provider.calls),
                edits_before_output,
            )

            console_action = IncomingAction(
                action_id="callback-2",
                conversation_id="!room:example.org",
                sender_id="@operator:example.org",
                message_id="$event-1",
                data=console_action_data,
            )
            self.assertTrue(await handle_tool_activity_action(console_action))
            streamed_text = next(
                call[3]
                for call in reversed(self.provider.calls)
                if call[0] == "edit"
            )
            self.assertIn("line 4", streamed_text)
            self.assertNotIn("line 3\n", streamed_text)
            self.assertTrue(streamed_text.endswith("line 10\n```"))

            await surface.finished(call, {"ok": True})
            await surface.clear()
            await surface.publish_process_output(
                ProcessOutputChunk(
                    call_id="c1",
                    session_id="shell-1",
                    stream="stdout",
                    text="late output\n",
                )
            )
            edits_before_stale_action = sum(
                call[0] == "edit" for call in self.provider.calls
            )
            await handle_tool_activity_action(info_action)

        self.assertIn(
            ("delete_many", "!room:example.org", ("$event-1",)),
            self.provider.calls,
        )
        self.assertEqual(
            sum(call[0] == "edit" for call in self.provider.calls),
            edits_before_stale_action,
        )
        self.assertEqual(
            self.provider.calls[-1],
            (
                "answer",
                "callback-1",
                "Tool view is no longer available.",
                False,
            ),
        )

    async def test_completed_tool_consoles_keep_independent_output(
        self,
    ) -> None:
        from tool_activity import ToolActivitySurface

        self.provider = ConsoleFakeProvider()
        set_chat_provider(self.provider)
        surface = ToolActivitySurface("!room:example.org")
        calls = (
            {
                "name": "shell",
                "call_id": "c1",
                "arguments": '{"command":"first"}',
            },
            {
                "name": "shell",
                "call_id": "c2",
                "arguments": '{"command":"second"}',
            },
        )
        commands = ("first", "second")
        for index, call in enumerate(calls, start=1):
            await surface.running(call)
            await surface.publish_process_output(
                ProcessOutputChunk(
                    call_id=f"c{index}",
                    session_id=f"shell-{index}",
                    stream="stdout",
                    text=f"output {index}\n",
                )
            )
            await surface.finished(call, {"ok": True})

        for index in range(1, 3):
            projected = next(
                call[3]
                for call in reversed(self.provider.calls)
                if call[0] == "edit" and call[2] == f"$event-{index}"
            )
            self.assertIn(f"$ {commands[index - 1]}", projected)
            self.assertIn(f"output {index}", projected)
            self.assertNotIn(f"output {3 - index}", projected)

        await surface.clear()

    async def test_tool_console_controls_are_shell_only(self) -> None:
        from tool_activity import ToolActivitySurface

        self.provider = ConsoleFakeProvider()
        set_chat_provider(self.provider)
        surface = ToolActivitySurface("!room:example.org")

        await surface.running({
            "name": "read_file",
            "call_id": "c1",
            "arguments": '{"path":"README.md"}',
        })
        await surface.clear()

        self.assertEqual(self.provider.calls[0][4], ())

    async def test_denied_shell_stays_on_info_without_running(self) -> None:
        from tool_activity import ToolActivitySurface

        self.provider = ConsoleFakeProvider()
        set_chat_provider(self.provider)
        surface = ToolActivitySurface("!room:example.org")
        call = {
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"rm protected"}',
        }

        await surface.approval(
            call,
            ((
                ActionButton("Approve", "approve"),
                ActionButton("Deny", "deny"),
            ),),
        )
        await surface.finished(
            call,
            {"ok": False, "approval": "denied"},
        )
        await surface.clear()

        terminal = next(
            item
            for item in reversed(self.provider.calls)
            if item[0] == "edit"
        )
        self.assertEqual(
            terminal[3],
            "Failed\n\nTool: shell\n\nCommand denied",
        )
        self.assertEqual(terminal[4], ())

    async def test_telegram_poll_wait_edits_one_bubble_to_zero(self) -> None:
        from tool_activity import ToolActivitySurface, _wait_duration

        self.assertEqual(_wait_duration(300), "5m")
        self.assertEqual(_wait_duration(295), "4m 55s")
        self.provider = ConsoleFakeProvider()
        set_chat_provider(self.provider)
        surface = ToolActivitySurface("!room:example.org")
        call = {
            "name": "write_stdin",
            "call_id": "c1",
            "arguments": (
                '{"session_id":"shell-1","yield_time_ms":300000}'
            ),
        }

        with (
            patch(
                "shell_sessions.write_stdin_wait_ms",
                return_value=20,
            ),
            patch("tool_activity.WAIT_UPDATE_INTERVAL_SECONDS", 0.01),
            patch("tool_activity.CHAT_STREAM_INTERVAL", 0),
        ):
            await surface.running(call)
            self.assertEqual(self.provider.calls[0][3], "Waiting 1s")
            await asyncio.sleep(0.04)
            await surface.finished(call, {"ok": True, "status": "running"})
            await surface.clear()

        projections = [
            item
            for item in self.provider.calls
            if item[0] in {"create", "edit"}
        ]
        self.assertTrue(any(item[3] == "Waiting 0s" for item in projections))
        self.assertTrue(all(item[2] == "$event-1" for item in projections))
        self.assertEqual(projections[0][4], ())
        self.assertIn(
            ("delete_many", "!room:example.org", ("$event-1",)),
            self.provider.calls,
        )

    async def test_telegram_poll_wait_stops_on_early_completion(self) -> None:
        from tool_activity import ToolActivitySurface

        self.provider = ConsoleFakeProvider()
        set_chat_provider(self.provider)
        surface = ToolActivitySurface("!room:example.org")
        call = {
            "name": "write_stdin",
            "call_id": "c1",
            "arguments": '{"session_id":"shell-1"}',
        }

        with (
            patch(
                "shell_sessions.write_stdin_wait_ms",
                return_value=100,
            ),
            patch("tool_activity.WAIT_UPDATE_INTERVAL_SECONDS", 0.01),
        ):
            await surface.running(call)
            await surface.finished(
                call,
                {"ok": True, "status": "completed"},
            )
            edit_count = sum(
                item[0] == "edit" for item in self.provider.calls
            )
            await asyncio.sleep(0.12)
            self.assertEqual(
                sum(item[0] == "edit" for item in self.provider.calls),
                edit_count,
            )
            await surface.clear()

        terminal = next(
            item[3]
            for item in reversed(self.provider.calls)
            if item[0] == "edit"
        )
        self.assertEqual(terminal, "Command completed")

    async def test_write_stdin_input_keeps_normal_tool_activity(self) -> None:
        from tool_activity import ToolActivitySurface

        self.provider = ConsoleFakeProvider()
        set_chat_provider(self.provider)
        surface = ToolActivitySurface("!room:example.org")
        call = {
            "name": "write_stdin",
            "call_id": "c1",
            "arguments": '{"session_id":"shell-1","chars":"yes\\n"}',
        }

        await surface.running(call)
        await surface.clear()

        self.assertTrue(
            self.provider.calls[0][3].startswith(
                "Running\n\n**Tool:** `write_stdin`"
            )
        )

    async def test_poll_wait_presentation_is_provider_opt_in(self) -> None:
        from tool_activity import ToolActivitySurface

        surface = ToolActivitySurface("!room:example.org")
        call = {
            "name": "write_stdin",
            "call_id": "c1",
            "arguments": '{"session_id":"shell-1"}',
        }

        await surface.running(call)
        await surface.clear()

        self.assertTrue(
            self.provider.calls[0][3].startswith(
                "Running\n\n**Tool:** `write_stdin`"
            )
        )

    async def test_child_tool_activity_uses_same_unpinned_bubble(self) -> None:
        from tool_activity import ToolActivitySurface

        surface = ToolActivitySurface("!room:example.org/child")
        call = {
            "type": "function_call",
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"pwd"}',
        }

        await surface.running(call)
        await surface.finished(call, {"ok": True})
        await surface.clear()

        methods = [call[0] for call in self.provider.calls]
        self.assertEqual(
            methods,
            ["create", "edit", "delete_many"],
        )
        texts = [
            call[3]
            for call in self.provider.calls
            if call[0] in {"create", "edit"}
        ]
        self.assertTrue(texts[0].startswith(
            "Running\n\n**Tool:** `shell`"
        ))
        self.assertTrue(texts[1].startswith(
            "Completed\n\n**Tool:** `shell`"
        ))
        self.assertIsNone(session.current_runtime().pinned_status_owner)

    async def test_tool_activity_preserves_existing_status_owner(self) -> None:
        from presentation import show_pinned_status
        from tool_activity import ToolActivitySurface

        await show_pinned_status(
            "!room:example.org",
            "thread:existing",
            "Thread active",
        )
        surface = ToolActivitySurface("!room:example.org")
        call = {
            "type": "function_call",
            "name": "list_dir",
            "call_id": "c1",
            "arguments": "{}",
        }
        await surface.running(call)
        await surface.clear()

        self.assertEqual(
            session.current_runtime().pinned_status_owner,
            "thread:existing",
        )
        deletes = [
            call for call in self.provider.calls if call[0] == "delete_many"
        ]
        self.assertEqual(deletes[0][2], ("$event-2",))
        self.assertEqual(
            session.current_runtime().pinned_status_message_id,
            "$event-1",
        )

    async def test_next_tool_gets_a_fresh_terminal_bubble(self) -> None:
        from tool_activity import ToolActivitySurface

        surface = ToolActivitySurface("!room:example.org/child")
        first = {
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"pwd"}',
        }
        second = {
            "name": "read_file",
            "call_id": "c2",
            "arguments": '{"path":"README.md"}',
        }

        await surface.running(first)
        await surface.finished(first, {"ok": True})
        await surface.running(second)
        await surface.finished(second, {"ok": True})
        await surface.clear()

        creates = [
            call for call in self.provider.calls if call[0] == "create"
        ]
        self.assertEqual([call[2] for call in creates], ["$event-1", "$event-2"])
        first_edits = [
            call
            for call in self.provider.calls
            if call[0] == "edit" and call[2] == "$event-1"
        ]
        self.assertEqual(len(first_edits), 1)
        self.assertTrue(first_edits[0][3].startswith("Completed"))
        deletes = [
            call[2]
            for call in self.provider.calls
            if call[0] == "delete_many"
        ]
        self.assertEqual(deletes, [("$event-2", "$event-1")])

    async def test_tool_activity_bounds_bubbles_retained_until_clear(
        self,
    ) -> None:
        from tool_activity import ToolActivitySurface

        surface = ToolActivitySurface("!room:example.org/child")
        first = {
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"pwd"}',
        }
        second = {
            "name": "read_file",
            "call_id": "c2",
            "arguments": '{"path":"README.md"}',
        }

        with patch("tool_activity.MAX_RETAINED_TOOL_BUBBLES", 1):
            await surface.running(first)
            await surface.finished(first, {"ok": True})
            await surface.running(second)
            await surface.finished(second, {"ok": True})

            deletes = [
                call[2]
                for call in self.provider.calls
                if call[0] == "delete_many"
            ]
            self.assertEqual(deletes, [("$event-1",)])

            await surface.clear()

        deletes = [
            call[2]
            for call in self.provider.calls
            if call[0] == "delete_many"
        ]
        self.assertEqual(deletes, [("$event-1",), ("$event-2",)])

    async def test_tool_bubble_delete_failure_does_not_block_restoration(
        self,
    ) -> None:
        from tool_activity import ToolActivitySurface

        surface = ToolActivitySurface("!room:example.org")
        call = {
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"pwd"}',
        }
        await surface.running(call)
        await surface.finished(call, {"ok": True})

        with (
            patch.object(
                self.provider,
                "delete_messages",
                AsyncMock(side_effect=RuntimeError("unavailable")),
            ),
            patch("tool_activity.log_exception") as logged,
        ):
            await surface.clear()

        logged.assert_called_once()
        self.assertIsNone(session.current_runtime().pinned_status_owner)

    async def test_active_tool_delete_failure_does_not_block_restoration(
        self,
    ) -> None:
        from tool_activity import ToolActivitySurface

        surface = ToolActivitySurface("!room:example.org")
        call = {
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"pwd"}',
        }
        await surface.running(call)

        with (
            patch.object(
                self.provider,
                "delete_messages",
                AsyncMock(side_effect=RuntimeError("unavailable")),
            ),
            patch("tool_activity.log_exception") as logged,
        ):
            await surface.clear()

        logged.assert_called_once()
        self.assertIsNone(session.current_runtime().pinned_status_owner)

    async def test_tool_preserves_thread_status(self) -> None:
        from thread_status import ensure_thread_status
        from tool_activity import ToolActivitySurface

        previous_thread = session.state.get("thread")
        session.state["thread"] = {
            "id": "branch-1",
            "session_id": "thread-session",
            "parent_session_id": "parent-session",
        }
        self.addCleanup(session.state.__setitem__, "thread", previous_thread)
        await ensure_thread_status("!room:example.org")
        surface = ToolActivitySurface("!room:example.org")
        call = {
            "name": "shell",
            "call_id": "c1",
            "arguments": '{"command":"pwd"}',
        }

        await surface.running(call)
        await surface.finished(call, {"ok": True})
        await surface.clear()

        creates = [
            call for call in self.provider.calls if call[0] == "create"
        ]
        self.assertEqual(
            [call[2] for call in creates],
            ["$event-1", "$event-2"],
        )
        self.assertTrue(creates[1][3].startswith("Running"))
        terminal_edits = [
            call
            for call in self.provider.calls
            if call[0] == "edit" and call[2] == "$event-2"
        ]
        self.assertEqual(len(terminal_edits), 1)
        self.assertTrue(terminal_edits[0][3].startswith("Completed"))
        self.assertEqual(
            session.current_runtime().pinned_status_message_id,
            "$event-1",
        )
        self.assertEqual(
            [call[0] for call in self.provider.calls].count("pin"),
            1,
        )
        self.assertFalse(
            any(call[0] == "unpin" for call in self.provider.calls)
        )

    async def test_tool_activity_stream_does_not_wait_for_transport(self) -> None:
        from tool_activity import ToolActivitySurface

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocked_status(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        surface = ToolActivitySurface("!room:example.org")
        call = {
            "type": "function_call",
            "name": "shell",
            "call_id": "c1",
            "arguments": "",
        }
        with patch.object(self.provider, "create_message", blocked_status):
            await asyncio.wait_for(surface.stream(call, False), timeout=0.1)
            await asyncio.wait_for(started.wait(), timeout=0.1)
            await surface.clear()

        self.assertTrue(cancelled.is_set())

    async def test_steering_message_is_never_pinned(self) -> None:
        from presentation import (
            clear_steering_indicator,
            steering_indicator,
        )

        message_id = await steering_indicator(
            "!room:example.org",
            "abcd",
        )
        await clear_steering_indicator(
            "!room:example.org",
            message_id,
            "abcd",
        )

        methods = [call[0] for call in self.provider.calls]
        self.assertEqual(methods, ["create", "delete"])
        controls = self.provider.calls[0][-1]
        self.assertEqual(
            [button.action for button in controls[0]],
            ["steer:abcd:apply", "steer:abcd:cancel"],
        )

    async def test_cancel_steering_deletes_user_message(self) -> None:
        from steering import handle_steering_action

        session.state["pending_inputs"] = [
            {"id": "abcd", "text": "cancel me"}
        ]
        session.pending_steers["abcd"] = {
            "id": "abcd",
            "chat_id": "!room:example.org",
            "text": "cancel me",
            "source_message_id": "$user-message",
            "message_id": "$steering-controls",
            "status": "queued",
        }
        event = IncomingAction(
            action_id="$cancel-action",
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$steering-controls",
            data="steer:abcd:cancel",
        )

        with patch("steering.session.save_state"):
            handled = await handle_steering_action(event)

        self.assertTrue(handled)
        self.assertNotIn("abcd", session.pending_steers)
        self.assertEqual(session.state["pending_inputs"], [])
        self.assertIn(
            ("delete", "!room:example.org", "$steering-controls"),
            self.provider.calls,
        )
        self.assertIn(
            ("delete", "!room:example.org", "$user-message"),
            self.provider.calls,
        )

    async def test_slow_forced_steer_waits_for_root_ownership(self) -> None:
        from controller import _dequeue_steer
        from steering import handle_steering_action

        entry = {
            "id": "abcd",
            "chat_id": "!room:example.org",
            "text": "apply after cancellation",
            "message_id": "$steering-controls",
            "status": "queued",
        }
        session.state["pending_inputs"] = [
            {"id": entry["id"], "text": entry["text"]}
        ]
        session.pending_steers["abcd"] = entry
        await session.steer_queue.put(entry)
        event = IncomingAction(
            action_id="$apply-action",
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$steering-controls",
            data="steer:abcd:apply",
        )

        with (
            patch(
                "steering.interrupt_root_turn",
                AsyncMock(
                    return_value=types.SimpleNamespace(
                        completed=False,
                        error=None,
                    )
                ),
            ),
            patch("steering.ensure_root_session") as ensure,
        ):
            handled = await handle_steering_action(event)

        self.assertTrue(handled)
        self.assertEqual(entry["status"], "force_pending")
        ensure.assert_called_once_with("!room:example.org", None)

        selected = await _dequeue_steer()
        self.assertIs(selected, entry)
        self.assertEqual(entry["status"], "applied")
        self.assertNotIn("abcd", session.pending_steers)
        self.assertEqual(session.state["pending_inputs"], [])

    async def test_forced_steer_consumes_durable_pending_input(self) -> None:
        from steering import handle_steering_action

        entry = {
            "id": "abcd",
            "chat_id": "!room:example.org",
            "text": "apply now",
            "message_id": "$steering-controls",
            "status": "queued",
        }
        session.state["pending_inputs"] = [
            {"id": entry["id"], "text": entry["text"]}
        ]
        session.pending_steers["abcd"] = entry
        event = IncomingAction(
            action_id="$apply-action",
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$steering-controls",
            data="steer:abcd:apply",
        )

        with (
            patch(
                "steering.interrupt_root_turn",
                AsyncMock(
                    return_value=types.SimpleNamespace(
                        completed=True,
                        error=None,
                    )
                ),
            ),
            patch("steering.start_root_session") as start,
        ):
            handled = await handle_steering_action(event)

        self.assertTrue(handled)
        self.assertEqual(session.state["pending_inputs"], [])
        self.assertNotIn("abcd", session.pending_steers)
        start.assert_called_once_with(
            "!room:example.org",
            "apply now",
            input_recorded=True,
            delivery_context=None,
        )

    async def test_cancelled_steer_claim_returns_to_queue(self) -> None:
        from controller import _dequeue_steer

        entry = {
            "id": "abcd",
            "chat_id": "!room:example.org",
            "text": "keep this request",
            "message_id": "$steering-controls",
            "status": "queued",
        }
        session.pending_steers["abcd"] = entry
        await session.steer_queue.put(entry)

        async def cancel_cleanup(*_args, **_kwargs):
            raise asyncio.CancelledError

        with patch(
            "controller.clear_steering_indicator",
            cancel_cleanup,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await _dequeue_steer()

        self.assertEqual(entry["status"], "queued")
        self.assertIs(session.pending_steers["abcd"], entry)
        self.assertIs(session.steer_queue.get_nowait(), entry)
        session.steer_queue.task_done()

    async def test_global_stop_discards_queued_steering(self) -> None:
        from steering import cancel_pending_steers

        session.state["pending_inputs"] = [
            {"id": "queued-id", "text": "queued"}
        ]
        session.pending_steers["abcd"] = {
            "id": "abcd",
            "chat_id": "!room:example.org",
            "text": "queued",
            "message_id": "$steering-controls",
            "status": "queued",
        }
        await session.steer_queue.put(session.pending_steers["abcd"])

        with patch("steering.session.save_state"):
            cancelled = await cancel_pending_steers()

        self.assertEqual(cancelled, 1)
        self.assertEqual(session.pending_steers, {})
        self.assertEqual(session.state["pending_inputs"], [])
        self.assertTrue(session.steer_queue.empty())

    async def test_interruption_cleanup_preserves_only_forced_steer(self) -> None:
        from steering import cancel_unforced_steers

        queued = {
            "id": "queued",
            "chat_id": "!room:example.org",
            "text": "discard me",
            "message_id": "$queued-controls",
            "status": "queued",
        }
        forced = {
            "id": "forced",
            "chat_id": "!room:example.org",
            "text": "keep me",
            "message_id": "$forced-controls",
            "status": "force_pending",
        }
        session.pending_steers.update(
            {"queued": queued, "forced": forced}
        )
        await session.steer_queue.put(queued)
        await session.steer_queue.put(forced)

        with patch("steering.session.save_state"):
            cancelled = await cancel_unforced_steers()

        self.assertEqual(cancelled, 1)
        self.assertEqual(session.pending_steers, {"forced": forced})
        self.assertEqual(
            session.state["pending_inputs"],
            [{"id": "forced", "text": "keep me"}],
        )
        self.assertIs(await session.steer_queue.get(), forced)
        session.steer_queue.task_done()

    def test_fake_provider_satisfies_runtime_contract(self) -> None:
        self.assertIsInstance(self.provider, ChatProvider)

    def test_provider_access_requires_explicit_process_binding(self) -> None:
        previous_provider = set_chat_provider(None)
        try:
            with self.assertRaisesRegex(RuntimeError, "not been initialized"):
                get_chat_provider()
        finally:
            set_chat_provider(previous_provider)

    def test_external_provider_factory_loads_by_module_attribute(self) -> None:
        module = types.ModuleType("test_external_chat_provider")
        module.create_provider = FakeProvider
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)

        loaded = load_chat_provider(
            "test_external_chat_provider:create_provider"
        )

        self.assertIsInstance(loaded, FakeProvider)

    def test_responses_api_provider_is_builtin(self) -> None:
        from responses_provider import ResponsesApiProvider

        self.assertIsInstance(
            load_chat_provider("responses_api"),
            ResponsesApiProvider,
        )

    def test_configured_providers_are_composed_by_default(self) -> None:
        from config import CHAT_PROVIDERS
        from multi_provider import MultiProvider

        self.assertEqual(CHAT_PROVIDERS, ("telegram", "responses_api"))
        self.assertIsInstance(
            load_chat_providers(CHAT_PROVIDERS),
            MultiProvider,
        )

    def test_single_configured_provider_is_not_wrapped(self) -> None:
        from telegram import TelegramProvider

        self.assertIsInstance(
            load_chat_providers(("telegram",)),
            TelegramProvider,
        )

    def test_unknown_provider_spec_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_chat_provider("unknown")

    async def test_normalized_message_reaches_core_with_string_id(self) -> None:
        from loop import handle_event

        event = IncomingMessage(
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$message",
            text="continue",
        )
        with (
            patch("loop.command", AsyncMock(return_value=False)),
            patch("loop.start_root_session") as start,
        ):
            await handle_event(event)

        start.assert_called_once_with(
            "!room:example.org",
            "continue",
            delivery_context=None,
            source_message_id="$message",
        )

    async def test_request_provider_can_reject_busy_input_before_steering(
        self,
    ) -> None:
        from loop import handle_event

        turns.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
        )
        blocker = turns.task
        assert blocker is not None
        self.provider.reject_busy = True
        event = IncomingMessage(
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$message",
            text="concurrent input",
        )
        try:
            with (
                patch("loop.command", AsyncMock(return_value=False)),
                patch("loop.start_root_session") as start,
            ):
                await handle_event(event)
            start.assert_not_called()
            self.assertIn(
                ("reject_busy", "!room:example.org"),
                self.provider.calls,
            )
            self.assertEqual(session.pending_steers, {})
        finally:
            blocker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await blocker
            turns.finish(blocker)

    async def test_busy_chat_input_is_queued_and_transport_acknowledged(
        self,
    ) -> None:
        from loop import handle_event

        turns.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
        )
        blocker = turns.task
        assert blocker is not None
        previous_session_id = session.state.get("current_session_id")
        session.state["current_session_id"] = None
        event = IncomingMessage(
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$message",
            text="change direction",
        )
        try:
            with patch("loop.command", AsyncMock(return_value=False)):
                await handle_event(event)
            self.assertEqual(len(session.pending_steers), 1)
            self.assertIn(
                ("acknowledge_queued", "!room:example.org"),
                self.provider.calls,
            )
        finally:
            blocker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await blocker
            turns.finish(blocker)
            session.state["current_session_id"] = previous_session_id
            while not session.steer_queue.empty():
                session.steer_queue.get_nowait()
                session.steer_queue.task_done()
            session.pending_steers.clear()

    async def test_busy_chat_rejects_input_when_steering_queue_is_full(
        self,
    ) -> None:
        from loop import handle_event

        turns.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
        )
        blocker = turns.task
        assert blocker is not None
        session.pending_steers["existing"] = {"status": "queued"}
        event = IncomingMessage(
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$message",
            text="one too many",
        )
        try:
            with (
                patch("loop.command", AsyncMock(return_value=False)),
                patch("loop.MAX_PENDING_STEERS", 1),
            ):
                await handle_event(event)
            self.assertIn(
                (
                    "send_text",
                    "!room:example.org",
                    "Steering queue is full (1).",
                ),
                self.provider.calls,
            )
            self.assertEqual(set(session.pending_steers), {"existing"})
        finally:
            blocker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await blocker
            turns.finish(blocker)
            session.pending_steers.clear()

    async def test_task_prompt_is_not_copied_into_event_log(self) -> None:
        from loop import handle_event

        event = IncomingMessage(
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$message",
            text="/task private side question",
        )
        with (
            patch("loop.command", AsyncMock(return_value=True)),
            patch("loop.log_event") as logged,
        ):
            await handle_event(event)

        received = logged.call_args.args[2]
        self.assertEqual(received["command"], "/task")
        self.assertNotIn("text", received)

    async def test_unknown_normalized_action_is_acknowledged(self) -> None:
        from loop import handle_event

        await handle_event(
            IncomingAction(
                action_id="$action",
                conversation_id="!room:example.org",
                sender_id="@operator:example.org",
                message_id="$message",
                data="provider-specific-unknown-action",
            )
        )
        self.assertIn(
            ("answer", "$action", "", False),
            self.provider.calls,
        )


class TelegramNormalizationTests(unittest.TestCase):
    def test_callback_is_normalized_at_adapter_boundary(self) -> None:
        from telegram import telegram_provider

        event = telegram_provider.normalize_update(
            {
                "callback_query": {
                    "id": "callback-1",
                    "from": {"id": 1},
                    "data": "approval:abc:approve",
                    "message": {
                        "message_id": 9,
                        "chat": {"id": 1, "type": "private"},
                    },
                },
            }
        )
        self.assertEqual(
            event,
            IncomingAction(
                action_id="callback-1",
                conversation_id="1",
                sender_id=1,
                message_id=9,
                data="approval:abc:approve",
            ),
        )

    def test_forum_topic_is_an_independent_conversation(self) -> None:
        import telegram

        provider = telegram.TelegramProvider()
        event = provider.normalize_update(
            {
                "message": {
                    "message_id": 10,
                    "message_thread_id": 42,
                    "is_topic_message": True,
                    "from": {"id": 1},
                    "chat": {"id": 1, "type": "supergroup"},
                    "text": "hello",
                }
            }
        )

        self.assertIsInstance(event, IncomingMessage)
        self.assertEqual(event.conversation_id, "1/42")
        self.assertTrue(event.interactive)

    def test_activity_topic_input_is_noninteractive(self) -> None:
        import telegram

        provider = telegram.TelegramProvider()
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )
        event = provider.normalize_update(
            {
                "message": {
                    "message_id": 10,
                    "message_thread_id": 42,
                    "is_topic_message": True,
                    "from": {"id": 1},
                    "chat": {"id": 1, "type": "supergroup"},
                    "text": "ignored",
                }
            }
        )

        self.assertIsInstance(event, IncomingMessage)
        self.assertFalse(event.interactive)

    def test_activity_topic_action_routes_to_parent_runtime(self) -> None:
        import telegram

        provider = telegram.TelegramProvider()
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic(
                parent_conversation_id="1/10",
            )
        )
        event = provider.normalize_update(
            {
                "callback_query": {
                    "id": "callback-1",
                    "from": {"id": 1},
                    "data": "approval:abc:approve",
                    "message": {
                        "message_id": 11,
                        "message_thread_id": 42,
                        "is_topic_message": True,
                        "chat": {"id": 1, "type": "supergroup"},
                    },
                }
            }
        )

        self.assertEqual(
            event,
            IncomingAction(
                action_id="callback-1",
                conversation_id="1/10",
                sender_id=1,
                message_id=11,
                data="approval:abc:approve",
                presentation_conversation_id="1/42",
            ),
        )

    def test_chat_and_topic_membership_are_isolated(self) -> None:
        import telegram

        with patch.object(telegram, "TG_CHAT_IDS", (1, 2)):
            provider = telegram.TelegramProvider()
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )

        first_chat = provider.normalize_update(
            {
                "message": {
                    "message_id": 10,
                    "message_thread_id": 42,
                    "is_topic_message": True,
                    "from": {"id": 1},
                    "chat": {"id": 1, "type": "supergroup"},
                    "text": "activity input",
                }
            }
        )
        second_chat = provider.normalize_update(
            {
                "message": {
                    "message_id": 11,
                    "message_thread_id": 42,
                    "is_topic_message": True,
                    "from": {"id": 1},
                    "chat": {"id": 2, "type": "supergroup"},
                    "text": "main input",
                }
            }
        )
        unknown_chat = provider.normalize_update(
            {
                "message": {
                    "message_id": 12,
                    "from": {"id": 1},
                    "chat": {"id": 3, "type": "private"},
                    "text": "unknown input",
                }
            }
        )

        self.assertFalse(first_chat.interactive)
        self.assertTrue(second_chat.interactive)
        self.assertFalse(unknown_chat.interactive)


class TelegramMultiChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_discovers_every_chat_in_order(self) -> None:
        import telegram

        call = AsyncMock(
            side_effect=[
                True,
                {"type": "private"},
                {"type": "supergroup", "is_forum": True},
                {"id": 99},
                {
                    "status": "administrator",
                    "can_manage_topics": True,
                    "can_delete_messages": True,
                },
                True,
            ]
        )
        client = AsyncMock()
        client_factory = Mock(return_value=client)
        with (
            patch.object(telegram, "TG_CHAT_IDS", (7, -1002)),
            patch.object(telegram, "_client", None),
            patch.object(
                telegram,
                "outbound_http_client",
                client_factory,
            ),
            patch.object(telegram, "tg_call", call),
        ):
            provider = telegram.TelegramProvider()
            await provider.initialize(())

        client_factory.assert_called_once_with(
            timeout=httpx.Timeout(65.0, connect=10.0),
        )
        self.assertEqual(provider.primary_conversation_id, "7")
        self.assertEqual(
            provider._chats[7].chat_type,
            "private",
        )
        self.assertEqual(provider._chats[-1002].chat_type, "supergroup")
        self.assertTrue(provider._chats[-1002].is_forum)
        self.assertTrue(provider.capabilities.drafts)
        self.assertEqual(
            [entry.args for entry in call.await_args_list],
            [
                ("deleteWebhook", {"drop_pending_updates": False}),
                ("getChat", {"chat_id": 7}),
                ("getChat", {"chat_id": -1002}),
                ("getMe",),
                (
                    "getChatMember",
                    {"chat_id": -1002, "user_id": 99},
                ),
                ("setMyCommands", {"commands": []}),
            ],
        )

    async def test_poll_restores_only_finalized_telegram_updates(self) -> None:
        import telegram

        cursor_path = _HOME / "finalized-update.cursor"
        cursor_path.unlink(missing_ok=True)
        self.addCleanup(cursor_path.unlink, missing_ok=True)
        updates = [
            {
                "update_id": update_id,
                "message": {
                    "message_id": update_id,
                    "from": {"id": 1, "is_bot": False},
                    "chat": {"id": 1, "type": "private"},
                    "text": text,
                },
            }
            for update_id, text in ((12, "/shutdown"), (13, "later"))
        ]
        provider = telegram.TelegramProvider(cursor_path=cursor_path)
        with patch.object(
            telegram,
            "tg_call",
            AsyncMock(return_value=updates),
        ):
            result = await provider.poll(None, timeout=30)

        self.assertFalse(cursor_path.exists())
        await provider.finish_event(result.events[0])
        self.assertEqual(cursor_path.read_text(), "13\n")
        self.assertEqual(stat.S_IMODE(cursor_path.stat().st_mode), 0o600)

        restored = telegram.TelegramProvider(cursor_path=cursor_path)
        call = AsyncMock(return_value=[])
        with patch.object(telegram, "tg_call", call):
            await restored.poll(None, timeout=30)

        self.assertEqual(call.await_args.args[1]["offset"], 13)
        await provider.finish_event(result.events[1])
        self.assertEqual(cursor_path.read_text(), "14\n")

    def test_invalid_telegram_cursor_fails_startup(self) -> None:
        import telegram

        cursor_path = _HOME / "invalid-update.cursor"
        cursor_path.write_text("not-an-offset\n")
        self.addCleanup(cursor_path.unlink, missing_ok=True)

        with self.assertRaisesRegex(RuntimeError, "cursor is invalid"):
            telegram.TelegramProvider(cursor_path=cursor_path)

    async def test_initialize_requires_read_only_topic_permissions(
        self,
    ) -> None:
        import telegram

        call = AsyncMock(
            side_effect=[
                True,
                {"type": "supergroup", "is_forum": True},
                {"id": 99},
                {
                    "status": "administrator",
                    "can_manage_topics": True,
                    "can_delete_messages": False,
                },
            ]
        )
        with (
            patch.object(telegram, "TG_CHAT_IDS", (-1002,)),
            patch.object(telegram, "_client", AsyncMock()),
            patch.object(telegram, "tg_call", call),
            self.assertRaisesRegex(RuntimeError, "Delete Messages"),
        ):
            await telegram.TelegramProvider().initialize(())

    async def test_drafts_are_routed_only_to_private_chats(self) -> None:
        import telegram

        with patch.object(telegram, "TG_CHAT_IDS", (7, -1002)):
            provider = telegram.TelegramProvider()
        provider._chats[7].chat_type = "private"
        provider._chats[-1002].chat_type = "supergroup"
        draft = AsyncMock()

        with patch.object(telegram, "send_draft", draft):
            await provider.send_draft(7, 1, "private draft")
            await provider.send_draft(-1002, 2, "group draft")

        draft.assert_awaited_once_with(7, 1, "private draft")

    async def test_activity_topics_are_isolated_by_chat(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        with patch.object(telegram, "TG_CHAT_IDS", (-1001, -1002)):
            provider = telegram.TelegramProvider()
        provider._chats[-1001].is_forum = True
        provider._chats[-1002].is_forum = True
        provider._chats[-1001].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )
        provider._chats[-1002].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="completed",
            result="done",
        )

        delivery = AsyncMock()
        with patch.object(telegram, "send", delivery):
            await provider.finish_activity_surface(
                "-1001/10",
                "42/77",
                snapshot,
            )

        delivery.assert_awaited_once_with("-1001/42", "done", silent=True)
        self.assertIn(42, provider._chats[-1001].activity_topics)
        self.assertIn(42, provider._chats[-1002].activity_topics)


class TelegramTransportTests(unittest.IsolatedAsyncioTestCase):
    def test_tool_info_renders_structured_arguments_as_telegram_html(
        self,
    ) -> None:
        import telegram
        from tool_activity import tool_preview

        preview = tool_preview({
            "name": "edit_file",
            "arguments": (
                '{"path":"README.md","content":"first\\nsecond",'
                '"options":{"mode":"replace"}}'
            ),
        })

        self.assertEqual(
            preview,
            "**Tool:** `edit_file`\n\n"
            "**Path:** `README.md`\n\n"
            "**Content:**\n```text\nfirst\nsecond\n```\n\n"
            "**Options:**\n```json\n{\n  \"mode\": \"replace\"\n}\n```",
        )
        rendered = telegram.markdown_to_telegram_html(preview)
        self.assertIn("<b>Tool:</b> <code>edit_file</code>", rendered)
        self.assertIn("<b>Path:</b> <code>README.md</code>", rendered)
        self.assertIn(
            '<pre><code class="language-text">first\nsecond</code></pre>',
            rendered,
        )
        self.assertIn(
            '<pre><code class="language-json">',
            rendered,
        )

    def test_tool_console_uses_telegram_bash_rendering(self) -> None:
        import telegram

        provider = telegram.TelegramProvider()

        self.assertTrue(provider.supports_tool_console("1/42"))
        self.assertEqual(
            telegram.markdown_to_telegram_html(
                "```bash\n$ python -m unittest\nOK\n```"
            ),
            '<pre><code class="language-bash">'
            "$ python -m unittest\nOK</code></pre>",
        )

    async def test_bulk_delete_chunks_telegram_requests(self) -> None:
        import telegram

        provider = telegram.TelegramProvider()
        message_ids = tuple(range(1, 206))
        call = AsyncMock(return_value=True)

        with patch.object(telegram, "tg_call", call):
            await provider.delete_messages("1/42", message_ids)

        self.assertEqual(call.await_count, 3)
        payloads = [entry.args[1] for entry in call.await_args_list]
        self.assertEqual(
            [len(payload["message_ids"]) for payload in payloads],
            [100, 100, 5],
        )
        self.assertTrue(
            all(payload["chat_id"] == 1 for payload in payloads)
        )
        self.assertTrue(
            all(
                entry.args[0] == "deleteMessages"
                for entry in call.await_args_list
            )
        )

    async def test_poll_deletes_and_discards_activity_topic_input(
        self,
    ) -> None:
        import telegram

        cursor_path = _HOME / "ignored-update.cursor"
        cursor_path.unlink(missing_ok=True)
        self.addCleanup(cursor_path.unlink, missing_ok=True)
        provider = telegram.TelegramProvider(cursor_path=cursor_path)
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )
        call = AsyncMock(
            side_effect=[
                [
                    {
                        "update_id": 12,
                        "message": {
                            "message_id": 10,
                            "message_thread_id": 42,
                            "is_topic_message": True,
                            "from": {"id": 1, "is_bot": False},
                            "chat": {"id": 1, "type": "supergroup"},
                            "text": "remove me",
                        },
                    }
                ],
                True,
            ]
        )

        with patch.object(telegram, "tg_call", call):
            result = await provider.poll(None, timeout=30)

        self.assertEqual(result.events, ())
        self.assertEqual(result.cursor, 13)
        self.assertEqual(cursor_path.read_text(), "13\n")
        self.assertEqual(
            [entry.args for entry in call.await_args_list],
            [
                (
                    "getUpdates",
                    {
                        "timeout": 30,
                        "limit": telegram._POLL_LIMIT,
                        "allowed_updates": ["message", "callback_query"],
                    },
                ),
                ("deleteMessage", {"chat_id": 1, "message_id": 10}),
            ],
        )

    async def test_poll_propagates_transient_activity_input_delete_failure(
        self,
    ) -> None:
        import telegram

        provider = telegram.TelegramProvider()
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )
        update = {
            "update_id": 12,
            "message": {
                "message_id": 10,
                "message_thread_id": 42,
                "is_topic_message": True,
                "from": {"id": 1, "is_bot": False},
                "chat": {"id": 1, "type": "supergroup"},
                "text": "remove me",
            },
        }
        error = telegram.TelegramApiError(
            "deleteMessage",
            status_code=500,
            error_code=500,
            description="Internal Server Error",
        )

        with (
            patch.object(
                telegram,
                "tg_call",
                AsyncMock(side_effect=[[update], error]),
            ),
            self.assertRaises(telegram.TelegramApiError),
        ):
            await provider.poll(None, timeout=30)

    async def test_activity_topic_keeps_plain_task_and_result(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        call = AsyncMock(return_value={"message_thread_id": 42})
        task_delivery = AsyncMock(return_value=77)
        delivery = AsyncMock()
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="running",
        )
        with (
            patch.object(telegram, "tg_call", call),
            patch.object(telegram, "_send_messages", task_delivery),
            patch.object(telegram, "send", delivery),
        ):
            surface_id = await provider.open_activity_surface("1/10", snapshot)
            await provider.finish_activity_surface(
                "1/10",
                str(surface_id),
                ActivitySnapshot(
                    activity_id="worker",
                    title="Inspect target",
                    status="completed",
                    result="done",
                ),
            )

        self.assertEqual(surface_id, "42/77")
        self.assertEqual(call.await_count, 1)
        self.assertEqual(call.await_args_list[0].args[0], "createForumTopic")
        self.assertEqual(
            call.await_args_list[0].args[1]["name"],
            "Subagent: worker",
        )
        self.assertEqual(
            task_delivery.await_args.args,
            ("1/42", "Inspect target"),
        )
        delivery.assert_awaited_once_with("1/42", "done", silent=True)
        self.assertIn(42, provider._chats[1].activity_topics)

    async def test_activity_stream_uses_plain_reasoning_and_reply_messages(
        self,
    ) -> None:
        import telegram
        from activity import ActivitySnapshot

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )
        rich = AsyncMock(
            side_effect=[{"message_id": 70}, {"message_id": 71}]
        )
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="running",
            reasoning_summary="Checking files",
            reply="I found the issue",
        )

        with patch.object(telegram, "send_rich", rich):
            await provider.update_activity_surface("1/10", "42/77", snapshot)
            await provider.update_activity_surface(
                "1/10",
                "42/77",
                ActivitySnapshot(
                    activity_id="worker",
                    title="Inspect target",
                    status="running",
                    detail="Running a tool",
                    reasoning_summary="Checking files",
                    reply="I found the issue",
                ),
            )

        self.assertEqual(
            [entry.args[2] for entry in rich.await_args_list],
            ["Checking files", "I found the issue"],
        )
        cleanup = AsyncMock()
        delivery = AsyncMock()
        with (
            patch.object(provider, "delete_message", cleanup),
            patch.object(telegram, "send", delivery),
        ):
            await provider.finish_activity_surface(
                "1/10",
                "42/77",
                ActivitySnapshot(
                    activity_id="worker",
                    title="Inspect target",
                    status="completed",
                    result="I found the complete issue",
                ),
            )

        cleanup.assert_awaited_once_with(1, 71)
        delivery.assert_awaited_once_with(
            "1/42",
            "I found the complete issue",
            silent=True,
        )
        self.assertIn(42, provider._chats[1].activity_topics)

    async def test_activity_finish_retries_failed_preview_cleanup(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic(
                reply_message_id=71,
                reply_text="partial reply",
            )
        )
        error = telegram.TelegramApiError(
            "deleteMessage",
            status_code=500,
            error_code=500,
            description="Internal Server Error",
        )
        delivery = AsyncMock()
        with (
            patch.object(provider, "delete_message", AsyncMock(side_effect=error)),
            patch.object(telegram, "send", delivery),
            self.assertRaises(telegram.TelegramApiError),
        ):
            await provider.finish_activity_surface(
                "1/10",
                "42/77",
                ActivitySnapshot(
                    activity_id="worker",
                    title="Inspect target",
                    status="completed",
                    result="complete reply",
                ),
            )

        topic = provider._chats[1].activity_topics[42]
        self.assertEqual(topic.reply_message_id, 71)
        delivery.assert_not_awaited()

    async def test_missing_activity_preview_is_already_clean(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic(
                reply_message_id=71,
                reply_text="partial reply",
            )
        )
        missing = telegram.TelegramApiError(
            "deleteMessage",
            status_code=400,
            error_code=400,
            description="Bad Request: message to delete not found",
        )
        delivery = AsyncMock()
        with (
            patch.object(provider, "delete_message", AsyncMock(side_effect=missing)),
            patch.object(telegram, "send", delivery),
        ):
            result = await provider.finish_activity_surface(
                "1/10",
                "42/77",
                ActivitySnapshot(
                    activity_id="worker",
                    title="Inspect target",
                    status="completed",
                    result="complete reply",
                ),
            )

        self.assertTrue(result.finished)
        self.assertTrue(result.result_delivered)
        self.assertIsNone(
            provider._chats[1].activity_topics[42].reply_message_id
        )
        delivery.assert_awaited_once_with(
            "1/42",
            "complete reply",
            silent=True,
        )

    async def test_existing_subagent_topic_is_reused(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        delivery = AsyncMock()
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Continue the review",
            status="running",
        )

        with (
            patch.object(telegram, "send", delivery),
            patch.object(telegram, "tg_call", AsyncMock()) as call,
        ):
            surface_id = await provider.open_activity_surface(
                "1/10",
                snapshot,
                "42/77",
            )

        self.assertEqual(surface_id, "42/77")
        delivery.assert_awaited_once_with(
            "1/42",
            "Continue the review",
            silent=True,
        )
        call.assert_awaited_once_with(
            "reopenForumTopic",
            {"chat_id": 1, "message_thread_id": 42},
        )
        self.assertIn(42, provider._chats[1].activity_topics)

    async def test_deleted_subagent_topic_is_replaced_on_continuation(
        self,
    ) -> None:
        import telegram
        from activity import ActivitySnapshot

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )
        deleted = telegram.TelegramApiError(
            "reopenForumTopic",
            status_code=400,
            error_code=400,
            description="Bad Request: topic not found",
        )
        task_delivery = AsyncMock(return_value=88)
        with (
            patch.object(
                telegram,
                "tg_call",
                AsyncMock(
                    side_effect=[deleted, {"message_thread_id": 43}],
                ),
            ),
            patch.object(telegram, "_send_messages", task_delivery),
        ):
            surface_id = await provider.open_activity_surface(
                "1/10",
                ActivitySnapshot(
                    activity_id="worker",
                    title="Continue the review",
                    status="running",
                ),
                "42/77",
            )

        self.assertEqual(surface_id, "43/88")
        self.assertNotIn(42, provider._chats[1].activity_topics)
        self.assertIn(43, provider._chats[1].activity_topics)

    async def test_finishing_activity_keeps_topic_open(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )
        delivery = AsyncMock()
        with patch.object(telegram, "send", delivery):
            result = await provider.finish_activity_surface(
                "1/10",
                "42/77",
                ActivitySnapshot(
                    activity_id="worker",
                    title="Task",
                    status="completed",
                    result="done",
                ),
            )
        self.assertTrue(result.result_delivered)
        delivery.assert_awaited_once_with("1/42", "done", silent=True)
        self.assertIn(42, provider._chats[1].activity_topics)

    async def test_activity_input_is_appended_as_plain_message(self) -> None:
        import telegram

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )
        delivery = AsyncMock()
        with patch.object(telegram, "send", delivery):
            await provider.append_activity_message(
                "1/10",
                "42/77",
                "check the logs",
            )

        delivery.assert_awaited_once_with(
            "1/42",
            "check the logs",
            silent=True,
        )

    async def test_activity_input_starts_a_fresh_reply_segment(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        topic = telegram._TelegramActivityTopic(
            reply_message_id=91,
            reply_text="old streamed reply",
        )
        provider._chats[1].activity_topics[42] = topic
        delivery = AsyncMock()
        upsert = AsyncMock(return_value=92)
        with (
            patch.object(telegram, "send", delivery),
            patch.object(telegram, "_upsert_topic_message", upsert),
        ):
            await provider.append_activity_message(
                "1/10",
                "42/77",
                "steer the child",
            )
            await provider.update_activity_surface(
                "1/10",
                "42/77",
                ActivitySnapshot(
                    activity_id="worker",
                    title="Task",
                    status="running",
                    reply="new streamed reply",
                ),
            )

        upsert.assert_awaited_once_with(
            1,
            42,
            None,
            "new streamed reply",
        )
        self.assertEqual(topic.reply_message_id, 92)

    async def test_deleting_activity_removes_topic_mapping(self) -> None:
        import telegram

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic()
        )
        delete = AsyncMock()
        with patch.object(telegram, "_delete_forum_topic", delete):
            await provider.delete_activity_surface("1/10", "42/77")

        delete.assert_awaited_once_with(1, 42)
        self.assertNotIn(42, provider._chats[1].activity_topics)

    async def test_topic_open_state_is_idempotent(self) -> None:
        import telegram

        error = telegram.TelegramApiError(
            "reopenForumTopic",
            status_code=400,
            error_code=400,
            description="Bad Request: TOPIC_NOT_MODIFIED",
        )

        with patch.object(
            telegram,
            "tg_call",
            AsyncMock(side_effect=error),
        ):
            reopened = await telegram._reopen_forum_topic(1, 42)

        self.assertTrue(reopened)

    async def test_activity_update_propagates_real_edit_errors(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        provider = telegram.TelegramProvider()
        provider._chats[1].activity_topics[42] = (
            telegram._TelegramActivityTopic(reply_message_id=77)
        )
        error = telegram.TelegramApiError(
            "editMessageText",
            status_code=400,
            error_code=400,
            description="Bad Request: message to edit not found",
        )
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="running",
            reply="Updated finding",
        )

        with (
            patch.object(
                telegram,
                "_edit_rich_message",
                AsyncMock(side_effect=error),
            ),
            self.assertRaises(telegram.TelegramApiError),
        ):
            await provider.update_activity_surface("1/10", "42/77", snapshot)

    async def test_unchanged_message_edit_is_a_successful_noop(self) -> None:
        import telegram

        error = telegram.TelegramApiError(
            "editMessageText",
            status_code=200,
            error_code=400,
            description=(
                "Bad Request: message is not modified: specified new message "
                "content and reply markup are exactly the same"
            ),
        )
        rich = AsyncMock(side_effect=error)

        with patch.object(telegram, "send_rich", rich):
            await telegram.TelegramProvider().edit_message(1, 7, "unchanged")

        rich.assert_awaited_once()

    async def test_chat_requests_are_spaced(self) -> None:
        import telegram

        with (
            patch.object(telegram, "_CHAT_REQUEST_INTERVAL", 0.025),
            patch.object(telegram, "_GLOBAL_REQUEST_INTERVAL", 0.0),
        ):
            started = asyncio.get_running_loop().time()
            await telegram._reserve_telegram_request("sendMessage", 1)
            await telegram._reserve_telegram_request("editMessageText", 1)
            elapsed = asyncio.get_running_loop().time() - started

        self.assertGreaterEqual(elapsed, 0.02)

    async def test_429_wait_metadata_is_applied_before_retry(self) -> None:
        import telegram

        request_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                return httpx.Response(
                    429,
                    json={
                        "ok": False,
                        "error_code": 429,
                        "description": "Too Many Requests",
                        "parameters": {"retry_after": 9},
                    },
                )
            return httpx.Response(200, json={"ok": True, "result": "sent"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        reserve = AsyncMock()
        defer = AsyncMock()
        try:
            with (
                patch.object(telegram, "_client", client),
                patch.object(telegram, "_reserve_telegram_request", reserve),
                patch.object(telegram, "_defer_telegram_requests", defer),
            ):
                result = await telegram.tg_call(
                    "sendMessage",
                    {"chat_id": 1, "text": "hello"},
                )
        finally:
            await client.aclose()

        self.assertEqual(result, "sent")
        self.assertEqual(request_count, 2)
        defer.assert_awaited_once_with(1, 9.0)

    async def test_429_retries_are_bounded(self) -> None:
        import telegram

        response = httpx.Response(
            429,
            json={
                "ok": False,
                "error_code": 429,
                "description": "Too Many Requests",
                "parameters": {"retry_after": 2},
            },
        )
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: response)
        )
        try:
            with (
                patch.object(telegram, "_client", client),
                patch.object(telegram, "TG_MAX_RETRIES", 1),
                patch.object(
                    telegram,
                    "_reserve_telegram_request",
                    AsyncMock(),
                ),
                patch.object(
                    telegram,
                    "_defer_telegram_requests",
                    AsyncMock(),
                ) as defer,
                self.assertRaises(telegram.TelegramApiError),
            ):
                await telegram.tg_call(
                    "sendMessage",
                    {"chat_id": 1, "text": "hello"},
                )
        finally:
            await client.aclose()

        self.assertEqual(defer.await_count, 2)

    async def test_rich_text_does_not_retry_rate_limit_as_plain_text(
        self,
    ) -> None:
        import telegram

        error = telegram.TelegramApiError(
            "sendMessage",
            status_code=429,
            error_code=429,
            description="Too Many Requests",
            retry_after=9,
        )
        call = AsyncMock(side_effect=error)
        with (
            patch.object(telegram, "TELEGRAM_MARKDOWN", True),
            patch.object(telegram, "tg_call", call),
            self.assertRaises(telegram.TelegramApiError),
        ):
            await telegram.send_rich(
                "sendMessage",
                {"chat_id": 1},
                "hello",
            )
        self.assertEqual(call.await_count, 1)

    async def test_rich_text_falls_back_only_for_entity_parse_error(
        self,
    ) -> None:
        import telegram

        error = telegram.TelegramApiError(
            "sendMessage",
            status_code=400,
            error_code=400,
            description="Bad Request: can't parse entities",
        )
        call = AsyncMock(side_effect=[error, "sent"])
        with (
            patch.object(telegram, "TELEGRAM_MARKDOWN", True),
            patch.object(telegram, "tg_call", call),
        ):
            result = await telegram.send_rich(
                "sendMessage",
                {"chat_id": 1},
                "hello",
            )

        self.assertEqual(result, "sent")
        self.assertEqual(call.await_count, 2)
        self.assertEqual(
            call.await_args_list[1].args[1],
            {"chat_id": 1, "text": "hello"},
        )

    async def test_edit_clears_controls_in_same_request(self) -> None:
        import telegram

        call = AsyncMock(return_value=True)
        with patch.object(telegram, "tg_call", call):
            await telegram.TelegramProvider().edit_message(1, 7, "updated")

        call.assert_awaited_once_with(
            "editMessageText",
            {
                "chat_id": 1,
                "message_id": 7,
                "reply_markup": {"inline_keyboard": []},
                "text": "updated",
                "parse_mode": "HTML",
            },
        )

    async def test_subagent_topic_messages_can_be_sent_silently(self) -> None:
        import telegram

        rich = AsyncMock(return_value={"message_id": 77})
        conversation = telegram._telegram_conversation_id(1, 42)
        with patch.object(telegram, "send_rich", rich):
            message_ids = await telegram._send_messages_tracked(
                conversation,
                "worker update",
                silent=True,
            )

        self.assertEqual(message_ids, (77,))
        self.assertTrue(
            rich.await_args.args[1]["disable_notification"]
        )
