"""Session rotation (/new) tests."""
import copy
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
        session.active_task = None
        session.subagent_tasks.clear()
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
            patch("commands.running_shell_sessions", return_value=1),
            patch("commands.send", capture),
        ):
            await command(1, "/new")

        self.assertEqual(session.state["current_session_id"], old)
        self.assertEqual(sent, ["Busy. Use /stop first."])

    def test_invalid_state_is_reported_instead_of_silently_replaced(self) -> None:
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        state_path.write_text("not-json")
        with patch.object(session, "STATE_PATH", state_path):
            with self.assertRaisesRegex(RuntimeError, "Could not load state"):
                session.load_state()


if __name__ == "__main__":
    unittest.main()
