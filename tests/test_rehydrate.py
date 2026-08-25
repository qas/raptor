"""Startup rehydration of persisted pending steers."""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-rehydrate-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import session
import chat_store


class RehydrateTests(unittest.TestCase):
    def setUp(self) -> None:
        session.state.clear()
        session.state.update(copy.deepcopy(session.DEFAULT_STATE))
        session.state["current_session_id"] = "20260819-120000-aabbccdd"
        session.state["pending_inputs"] = [
            {"id": "one", "text": "steer-one"},
            {"id": "two", "text": "steer-two"},
        ]
        session.pending_steers.clear()
        while True:
            try:
                session.steer_queue.get_nowait()
                session.steer_queue.task_done()
            except Exception:
                break

    def test_rehydrate_pending_inputs_into_steer_queue(
        self,
    ) -> None:
        count = session.rehydrate_pending_inputs(
            session.current_runtime().conversation_id
        )
        self.assertEqual(count, 2)
        self.assertEqual(session.steer_queue.qsize(), 2)
        first = session.steer_queue.get_nowait()
        second = session.steer_queue.get_nowait()
        self.assertEqual(first["text"], "steer-one")
        self.assertEqual(second["text"], "steer-two")
        self.assertEqual(first["id"], "one")
        self.assertEqual(second["id"], "two")
        self.assertEqual(first["status"], "queued")
        self.assertTrue(first.get("rehydrated"))
        # Persisted list remains until apply removes entries.
        self.assertEqual(
            session.state["pending_inputs"],
            [
                {"id": "one", "text": "steer-one"},
                {"id": "two", "text": "steer-two"},
            ],
        )

    def test_handoff_records_once_before_retiring_pending_state(self) -> None:
        chat_dir = Path(tempfile.mkdtemp(prefix="rehydrate-chats-"))
        with patch.object(chat_store, "CHAT_DIR", chat_dir):
            chat_store._SEQ_CACHE.clear()
            session_id = chat_store.create_session(
                kind="main",
                chat_key=session.current_runtime().key,
            )
            session.state["current_session_id"] = session_id
            entry = {
                "id": "one",
                "text": "steer-one",
                "status": "queued",
            }
            session.pending_steers["one"] = entry

            session.persist_steer_handoff(entry)
            session.state["pending_inputs"] = [
                {"id": "one", "text": "steer-one"}
            ]
            session.pending_steers["one"] = entry
            session.persist_steer_handoff(entry)

            events = chat_store.read_events(session_id)
            steers = [
                event
                for event in events
                if event.get("source") == "steer"
            ]
            self.assertEqual(len(steers), 1)
            self.assertEqual(steers[0]["data"]["steer_id"], "one")
            self.assertEqual(session.state["pending_inputs"], [])
            self.assertNotIn("one", session.pending_steers)

    def test_failed_handoff_keeps_pending_state(self) -> None:
        entry = {
            "id": "one",
            "text": "steer-one",
            "status": "queued",
        }
        session.pending_steers["one"] = entry
        with patch.object(session, "append_item", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                session.persist_steer_handoff(entry)

        self.assertEqual(
            session.state["pending_inputs"],
            [
                {"id": "one", "text": "steer-one"},
                {"id": "two", "text": "steer-two"},
            ],
        )
        self.assertIs(session.pending_steers["one"], entry)


if __name__ == "__main__":
    unittest.main()
