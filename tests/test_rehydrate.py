"""Startup rehydration of persisted pending steers."""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-rehydrate-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import session


class RehydrateTests(unittest.TestCase):
    def setUp(self) -> None:
        session.state.clear()
        session.state.update(copy.deepcopy(session.DEFAULT_STATE))
        session.state["current_session_id"] = "20260819-120000-aabbccdd"
        session.state["pending_inputs"] = [
            "steer-one",
            "steer-two",
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
        count = session.rehydrate_pending_inputs(session.current_runtime().conversation_id)
        self.assertEqual(count, 2)
        self.assertEqual(session.steer_queue.qsize(), 2)
        first = session.steer_queue.get_nowait()
        second = session.steer_queue.get_nowait()
        self.assertEqual(first["text"], "steer-one")
        self.assertEqual(second["text"], "steer-two")
        self.assertEqual(first["status"], "queued")
        self.assertTrue(first.get("rehydrated"))
        # Persisted list remains until apply removes entries.
        self.assertEqual(
            session.state["pending_inputs"],
            ["steer-one", "steer-two"],
        )


if __name__ == "__main__":
    unittest.main()
