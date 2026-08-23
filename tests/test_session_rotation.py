"""Session rotation (/new) tests."""
import asyncio
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-rotation-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chat_store
import session
from session import pending_approvals
from turn_runtime import turns
from commands import command
from goals import replace_goal
from tools import chat_history_tool


async def _noop(*_a, **_k):
    return None


class SessionRotationTests(unittest.IsolatedAsyncioTestCase):
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
        sid = chat_store.create_session(kind="main")
        session.state["current_session_id"] = sid
        session.state["model"] = "model-a"
        session.state["approval_mode"] = "on"
        session.state["todos"] = [
            {"step": "old", "status": "pending"}
        ]
        session.subagent_records = session.state["subagents"]
        turns.finish()
        session.subagent_tasks.clear()
        session.pending_steers.clear()
        pending_approvals.clear()
        while True:
            try:
                session.runtime_event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                session.runtime_event_queue.task_done()
        chat_store.append_item(
            sid,
            {"role": "user", "content": "remember-me"},
            source="user",
        )

    async def test_new_rotates_session_without_deleting_old(
        self,
    ) -> None:
        old = session.state["current_session_id"]
        with patch("commands.send", _noop):
            await command(1, "/new")
        new = session.state["current_session_id"]
        self.assertNotEqual(old, new)
        self.assertTrue(chat_store.session_exists(old))
        self.assertTrue(chat_store.session_exists(new))
        end = [
            e
            for e in chat_store.read_events(old)
            if e.get("type") == "session_end"
        ]
        self.assertEqual(len(end), 1)
        self.assertEqual(end[0]["reason"], "new_session")
        self.assertEqual(
            end[0]["todos"][0]["step"],
            "old",
        )
        self.assertEqual(session.state["todos"], [])
        self.assertEqual(session.state["model"], "model-a")
        self.assertEqual(session.state["approval_mode"], "on")
        hits = chat_history_tool(
            {
                "action": "search",
                "query": "remember-me",
                "session_id": old,
            }
        )
        self.assertEqual(len(hits["hits"]), 1)

    async def test_new_preserves_goal_owned_checklist(self) -> None:
        goal = replace_goal("Long-running goal")
        goal["todos"] = [
            {"step": "Keep me", "status": "in_progress"},
        ]
        with patch("commands.send", _noop):
            await command(1, "/new")
        self.assertEqual(
            session.state["goal"]["todos"],
            [{"step": "Keep me", "status": "in_progress"}],
        )
        self.assertEqual(session.state["todos"], [])

    async def test_new_refuses_while_shell_session_is_running(self) -> None:
        old = session.state["current_session_id"]
        sent: list[str] = []

        async def capture(_chat_id, text, **_kwargs):
            sent.append(text)

        with (
            patch("shell_sessions.running_shell_sessions", return_value=1),
            patch("commands.send", capture),
        ):
            await command(1, "/new")

        self.assertEqual(session.state["current_session_id"], old)
        self.assertEqual(sent, ["Busy. Use /stop all first."])

    async def test_chats_lists_main_sessions_and_marks_current(self) -> None:
        current = session.state["current_session_id"]
        previous = chat_store.create_session(kind="main")
        chat_store.append_item(
            previous,
            {"role": "user", "content": "find this launch note"},
            source="user",
        )
        child = chat_store.create_session(
            kind="subagent",
            agent_id="child",
            parent_session_id=current,
        )
        sent: list[str] = []

        async def capture(_chat_id, text, **_kwargs):
            sent.append(text)

        with patch("commands.send", capture):
            await command(1, "/chats")

        self.assertIn(f"{current} ·", sent[0])
        self.assertIn("(current)", sent[0])
        self.assertIn(f"{previous} ·", sent[0])
        self.assertNotIn(child, sent[0])

    async def test_chats_keeps_an_old_resumed_session_visible(self) -> None:
        current = session.state["current_session_id"]
        for _index in range(25):
            chat_store.create_session(kind="main")
        sent: list[str] = []

        async def capture(_chat_id, text, **_kwargs):
            sent.append(text)

        with patch("commands.send", capture):
            await command(1, "/chats")

        self.assertIn(f"{current} ·", sent[0])
        self.assertIn(f"{current} ·", sent[0].splitlines()[1])

    async def test_chats_searches_transcript_content(self) -> None:
        matching = chat_store.create_session(kind="main")
        chat_store.append_item(
            matching,
            {"role": "user", "content": "NeedleProject details"},
            source="user",
        )
        other = chat_store.create_session(kind="main")
        chat_store.append_item(
            other,
            {"role": "user", "content": "unrelated"},
            source="user",
        )
        sent: list[str] = []

        async def capture(_chat_id, text, **_kwargs):
            sent.append(text)

        with patch("commands.send", capture):
            await command(1, "/chats needleproject")

        self.assertIn(matching, sent[0])
        self.assertNotIn(other, sent[0])

    async def test_resume_switches_to_archived_main_session(self) -> None:
        current = session.state["current_session_id"]
        target = chat_store.create_session(kind="main")
        archived_todos = [{"step": "continue this", "status": "pending"}]
        chat_store.end_session(
            target,
            reason="new_session",
            todos=archived_todos,
        )
        sent: list[str] = []

        async def capture(_chat_id, text, **_kwargs):
            sent.append(text)

        with patch("commands.send", capture):
            await command(1, f"/resume {target}")

        self.assertEqual(session.state["current_session_id"], target)
        self.assertEqual(session.state["todos"], archived_todos)
        self.assertEqual(sent, [f"Resumed chat: {target}"])
        current_end = [
            event
            for event in chat_store.read_events(current)
            if event.get("type") == "session_end"
        ]
        self.assertEqual(current_end[-1]["reason"], "session_switched")
        resumed = [
            event
            for event in chat_store.read_events(target)
            if event.get("type") == "meta"
            and event.get("name") == "session_resumed"
        ]
        self.assertEqual(resumed[-1]["data"]["from_session_id"], current)

    async def test_resume_rejects_subagent_session(self) -> None:
        current = session.state["current_session_id"]
        child = chat_store.create_session(
            kind="subagent",
            agent_id="child",
            parent_session_id=current,
        )
        sent: list[str] = []

        async def capture(_chat_id, text, **_kwargs):
            sent.append(text)

        with patch("commands.send", capture):
            await command(1, f"/resume {child}")

        self.assertEqual(session.state["current_session_id"], current)
        self.assertEqual(sent, [f"Chat not found: {child}"])

    def test_invalid_state_is_reported_instead_of_silently_replaced(self) -> None:
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        state_path.write_text("not-json")
        with patch.object(session, "STATE_PATH", state_path):
            with self.assertRaisesRegex(RuntimeError, "Could not load state"):
                session.load_state()

    def test_invalid_persisted_plan_is_reported(self) -> None:
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        state_path.write_text(
            json.dumps({
                "todos": [
                    {"id": 1, "text": "old shape", "status": "pending"}
                ]
            })
        )
        with patch.object(session, "STATE_PATH", state_path):
            with self.assertRaisesRegex(RuntimeError, "persisted root plan"):
                session.load_state()


if __name__ == "__main__":
    unittest.main()
