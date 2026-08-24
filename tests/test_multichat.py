"""Main-chat ownership and concurrency invariants."""

import asyncio
import unittest

import session
from chat_store import append_item, list_sessions, session_chat_key
from tools import chat_history_tool
from turn_runtime import TurnKind, turns


class MultichatTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_state_and_turns_are_isolated(self) -> None:
        first = session.ensure_chat("test-provider:first")
        second = session.ensure_chat("test-provider:second")

        blocker = asyncio.Event()
        with session.bound_runtime(first):
            session.state["todos"] = [
                {"step": "first task", "status": "in_progress"}
            ]
            first_task = turns.start(
                blocker.wait(),
                kind=TurnKind.REGULAR,
            )

        with session.bound_runtime(second):
            self.assertEqual(session.state["todos"], [])
            self.assertFalse(turns.is_running())
            session.state["todos"] = [
                {"step": "second task", "status": "pending"}
            ]

        with session.bound_runtime(first):
            self.assertTrue(turns.is_running())
            self.assertEqual(session.state["todos"][0]["step"], "first task")
            first_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first_task
            turns.finish(first_task)

        with session.bound_runtime(second):
            self.assertEqual(
                session.state["todos"][0]["step"],
                "second task",
            )

    async def test_transcripts_record_and_filter_their_owner(self) -> None:
        first = session.ensure_chat("test-provider:owner-a")
        second = session.ensure_chat("test-provider:owner-b")
        first_session = str(first.state["current_session_id"])
        second_session = str(second.state["current_session_id"])

        self.assertEqual(
            session_chat_key(first_session),
            "test-provider:owner-a",
        )
        self.assertEqual(
            session_chat_key(second_session),
            "test-provider:owner-b",
        )
        owners = {
            row["session_id"]: row["chat_key"]
            for row in list_sessions()
            if row["session_id"] in {first_session, second_session}
        }
        self.assertEqual(
            owners,
            {
                first_session: "test-provider:owner-a",
                second_session: "test-provider:owner-b",
            },
        )

        append_item(
            first_session,
            {"role": "user", "content": "owner-a-secret"},
            source="user",
        )
        with session.bound_runtime(second):
            result = chat_history_tool(
                {"action": "read", "session_id": first_session}
            )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "session is not available")


if __name__ == "__main__":
    unittest.main()
