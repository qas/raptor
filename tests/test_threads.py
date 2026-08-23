"""Temporary conversation branch lifecycle tests."""
import asyncio
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import AsyncMock

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-thread-tests-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chat_store
import agent as agent_mod
from chat_runtime import set_chat_provider
from chat_provider import IncomingAction, IncomingMessage
from context import build_active_context
from goals import ensure_goal_pin, goal_instructions, replace_goal, sync_goal_pin
import session
from tests.test_chat_provider import FakeProvider
from threads import (
    finish_thread,
    handle_thread_action,
    start_thread,
    thread_active,
)
from tools import chat_history_tool


class ThreadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._chat_dir = Path(tempfile.mkdtemp(prefix="chats-"))
        self._chat_patch = patch.object(
            chat_store,
            "CHAT_DIR",
            self._chat_dir,
        )
        self._chat_patch.start()
        self.addCleanup(self._chat_patch.stop)
        chat_store._SEQ_CACHE.clear()
        session.state.clear()
        session.state.update(copy.deepcopy(session.DEFAULT_STATE))
        self.parent = chat_store.create_session(kind="main")
        session.state["current_session_id"] = self.parent
        session.state["model"] = "model-a"
        session.active_task = None
        session.active_since = None
        session.active_goal_id = None
        session.subagent_tasks.clear()
        session.pending_approvals.clear()
        session.pending_steers.clear()
        session.goal_pin_message_id = None
        session.goal_pin_goal_id = None
        session.pinned_status_conversation_id = None
        session.pinned_status_message_id = None
        session.pinned_status_owner = None
        while not session.internal_queue.empty():
            session.internal_queue.get_nowait()
            session.internal_queue.task_done()
        self.provider = FakeProvider()
        previous_provider = set_chat_provider(self.provider)
        self.addCleanup(set_chat_provider, previous_provider)
        chat_store.append_item(
            self.parent,
            {"role": "user", "content": "before fork"},
            source="user",
        )

    async def test_clear_restores_untouched_parent(self) -> None:
        recovery = {"session_id": self.parent, "interrupted_at": 1}
        session.state["interrupted_agent"] = copy.deepcopy(recovery)
        result = await start_thread("!room:example.org")
        self.assertTrue(result["ok"])
        branch = str(result["thread"]["session_id"])
        self.assertTrue(thread_active())
        self.assertIsNone(session.state["interrupted_agent"])
        self.assertEqual(
            build_active_context(branch)[0]["content"],
            "before fork",
        )
        chat_store.append_item(
            branch,
            {"role": "user", "content": "branch only"},
            source="user",
        )

        cleared = await finish_thread(
            "!room:example.org",
            merge=False,
        )

        self.assertTrue(cleared["ok"])
        self.assertFalse(thread_active())
        self.assertEqual(session.state["current_session_id"], self.parent)
        self.assertEqual(session.state["interrupted_agent"], recovery)
        parent_text = [
            item.get("content") for item in build_active_context(self.parent)
        ]
        self.assertEqual(parent_text, ["before fork"])
        hidden = chat_history_tool(
            {"action": "read", "session_id": branch},
            execution_context={"session_id": self.parent},
        )
        self.assertFalse(hidden["ok"])
        listed = chat_history_tool({"action": "list", "limit": 100})
        self.assertNotIn(
            branch,
            {row["session_id"] for row in listed["sessions"]},
        )

    async def test_merge_adds_only_branch_native_items(self) -> None:
        result = await start_thread("!room:example.org")
        branch = str(result["thread"]["session_id"])
        chat_store.append_item(
            branch,
            {"role": "user", "content": "branch question"},
            source="user",
        )
        chat_store.append_item(
            branch,
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "branch answer"}
                ],
            },
            source="assistant",
        )

        merged = await finish_thread(
            "!room:example.org",
            merge=True,
        )

        self.assertEqual(merged["merged_items"], 2)
        work = build_active_context(self.parent)
        self.assertEqual(work[0]["content"], "before fork")
        self.assertEqual(work[1]["content"], "branch question")
        self.assertEqual(
            work[2]["content"][0]["text"],
            "branch answer",
        )
        self.assertEqual(
            sum(item.get("content") == "before fork" for item in work),
            1,
        )

    async def test_real_agent_turn_writes_only_to_branch(self) -> None:
        async def fake_stream(_chat_id, _items, **_kwargs):
            return {
                "output": [{
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "branch response",
                    }],
                }]
            }

        async def noop(*_args, **_kwargs):
            return None

        result = await start_thread("!room:example.org")
        branch = str(result["thread"]["session_id"])
        with (
            patch.object(
                agent_mod,
                "responses_create_stream",
                fake_stream,
            ),
            patch.object(agent_mod, "send", noop),
            patch.object(agent_mod, "typing_loop", noop),
            patch.object(agent_mod, "maybe_auto_compact", noop),
        ):
            delivered = await agent_mod.agent_turn(
                "!room:example.org",
                "branch prompt",
            )

        self.assertTrue(delivered)
        branch_text = str(build_active_context(branch))
        self.assertIn("branch prompt", branch_text)
        self.assertIn("branch response", branch_text)
        parent_text = str(build_active_context(self.parent))
        self.assertNotIn("branch prompt", parent_text)
        self.assertNotIn("branch response", parent_text)

    async def test_thread_pin_sits_between_approval_and_goal(self) -> None:
        goal = replace_goal("main objective")
        await ensure_goal_pin("!room:example.org")
        self.assertEqual(
            session.pinned_status_owner,
            f"goal:{goal['id']}",
        )

        result = await start_thread("!room:example.org")
        thread_id = str(result["thread"]["id"])
        self.assertEqual(
            session.pinned_status_owner,
            f"thread:{thread_id}",
        )
        self.assertIsNone(session.goal_pin_message_id)
        self.assertIsNone(session.goal_pin_goal_id)
        self.assertEqual(goal_instructions(), "")

        from presentation import show_pinned_status
        await show_pinned_status(
            "!room:example.org",
            "approval:abc",
            "Approval required",
        )
        await sync_goal_pin(
            "!room:example.org",
            released_owner="approval:abc",
        )
        self.assertEqual(
            session.pinned_status_owner,
            f"thread:{thread_id}",
        )

        await finish_thread("!room:example.org", merge=False)
        self.assertEqual(
            session.pinned_status_owner,
            f"goal:{goal['id']}",
        )
        self.assertEqual(session.goal_pin_goal_id, goal["id"])
        self.assertIn("main objective", goal_instructions())

    async def test_lifecycle_change_refuses_while_busy(self) -> None:
        blocker = asyncio.Event()

        async def wait_forever() -> None:
            await blocker.wait()

        session.active_task = asyncio.create_task(wait_forever())
        try:
            result = await start_thread("!room:example.org")
            self.assertFalse(result["ok"])
            self.assertIn("Busy", result["error"])
        finally:
            session.active_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await session.active_task
            session.active_task = None

    async def test_thread_commands_start_status_and_clear(self) -> None:
        from commands import command

        send = AsyncMock()
        with patch("commands.send", send):
            self.assertTrue(
                await command("!room:example.org", "/thread")
            )
            self.assertTrue(thread_active())
            self.assertTrue(
                await command("!room:example.org", "/thread status")
            )
            self.assertTrue(
                await command("!room:example.org", "/thread clear")
            )
        self.assertFalse(thread_active())
        texts = [call.args[1] for call in send.await_args_list]
        self.assertTrue(any("Thread started" in text for text in texts))
        self.assertTrue(any("Thread active" in text for text in texts))
        self.assertTrue(any("Thread cleared" in text for text in texts))

    async def test_thread_message_starts_branch_and_returns_message(self) -> None:
        from commands import command

        with patch("commands.send", AsyncMock()):
            result = await command(
                "!room:example.org",
                "/thread explore this separately",
            )
        self.assertEqual(result, "explore this separately")
        self.assertTrue(thread_active())

    async def test_thread_message_continues_existing_branch(self) -> None:
        from commands import command

        started = await start_thread("!room:example.org")
        self.assertTrue(started["ok"])
        with patch("commands.send", AsyncMock()):
            result = await command(
                "!room:example.org",
                "/thread keep going",
            )
        self.assertEqual(result, "keep going")

    async def test_thread_message_is_dispatched_as_branch_input(self) -> None:
        from loop import handle_event

        with (
            patch("commands.send", AsyncMock()),
            patch("loop.start_root_session") as start,
        ):
            await handle_event(IncomingMessage(
                conversation_id="!room:example.org",
                sender_id=self.provider.authorized_user_id,
                message_id="$thread-message",
                text="/thread investigate this",
            ))
        start.assert_called_once_with(
            "!room:example.org",
            "investigate this",
            delivery_context=None,
        )
        self.assertTrue(thread_active())

    async def test_thread_clear_button_returns_to_parent(self) -> None:
        result = await start_thread("!room:example.org")
        thread_id = str(result["thread"]["id"])
        handled = await handle_thread_action(
            IncomingAction(
                action_id="$clear-thread",
                conversation_id="!room:example.org",
                sender_id="@operator:example.org",
                message_id=session.pinned_status_message_id,
                data=f"thread:{thread_id}:clear",
            )
        )

        self.assertTrue(handled)
        self.assertFalse(thread_active())
        self.assertEqual(session.state["current_session_id"], self.parent)
        self.assertIn(
            ("answer", "$clear-thread", "Thread cleared.", False),
            self.provider.calls,
        )


if __name__ == "__main__":
    unittest.main()
