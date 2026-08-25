"""Main-chat ownership and concurrency invariants."""

import asyncio
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-multichat-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)

import session
from chat_store import append_item, list_sessions, session_chat_key
from engine import assistant_message
from tools import chat_history_tool
from turn_runtime import TurnKind, turns


class MultichatTests(unittest.IsolatedAsyncioTestCase):
    def test_durable_field_commit_preserves_live_subagent_record(self) -> None:
        runtime = session.current_runtime()
        record = {"id": "worker", "status": "running"}
        runtime.subagent_records["worker"] = record
        session_id = str(runtime.state["current_session_id"])
        try:
            session.set_pending_delivery(session_id, 1)

            self.assertIs(runtime.subagent_records["worker"], record)
            record["status"] = "completed"
            session.save_state()

            persisted = session._root_state["chats"][runtime.key]["state"]
            self.assertEqual(
                persisted["subagents"]["worker"]["status"],
                "completed",
            )
        finally:
            runtime.subagent_records.pop("worker", None)
            session.clear_pending_delivery(session_id, 1)

    def test_recovery_markers_fit_after_state_reaches_admission_limit(
        self,
    ) -> None:
        runtime = session.current_runtime()
        initial_state = copy.deepcopy(runtime.state)
        candidate_state = copy.deepcopy(initial_state)
        candidate_state["pending_inputs"] = [
            {"id": "near-limit", "text": "x" * 512}
        ]
        entry = session._root_state["chats"][runtime.key]
        candidate_root = {
            **session._root_state,
            "chats": {
                **session._root_state["chats"],
                runtime.key: {**entry, "state": candidate_state},
            },
        }
        admitted_size = len(
            json.dumps(
                session._state_without_recovery_markers(candidate_root),
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        max_bytes = admitted_size + session._state_recovery_reserve(
            candidate_root
        )
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        session_id = str(runtime.state["current_session_id"])
        try:
            with (
                patch.object(session, "STATE_PATH", state_path),
                patch.object(
                    session,
                    "MAX_STATE_LOAD_BYTES",
                    max_bytes,
                ),
            ):
                session.queue_pending_steer("near-limit", "x" * 512)
                session.set_active_root_turn(
                    {"id": "turn-1", "session_id": session_id}
                )
                answer = append_item(
                    session_id,
                    assistant_message("answer"),
                    source="assistant",
                )
                session.set_pending_delivery(
                    session_id,
                    int(answer["seq"]),
                )
                with self.assertRaises(session.StateCapacityError):
                    session.queue_pending_steer("overflow", "y" * 8192)

                self.assertLessEqual(
                    len(state_path.read_bytes()),
                    max_bytes,
                )
                self.assertEqual(
                    runtime.state["pending_delivery"],
                    {"session_id": session_id, "seq": answer["seq"]},
                )
        finally:
            runtime.state.clear()
            runtime.state.update(initial_state)

    def test_oversized_steer_is_rejected_without_mutating_state(self) -> None:
        runtime = session.current_runtime()
        initial_pending = list(runtime.state["pending_inputs"])
        root_bytes = len(
            json.dumps(
                session._state_without_recovery_markers(
                    session._root_state
                ),
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        max_bytes = (
            root_bytes
            + 128
            + session._state_recovery_reserve(session._root_state)
        )
        with (
            patch.object(session, "STATE_PATH", state_path),
            patch.object(
                session,
                "MAX_STATE_LOAD_BYTES",
                max_bytes,
            ),
        ):
            session.save_state()
            before = state_path.read_bytes()
            with self.assertRaises(session.StateCapacityError):
                session.queue_pending_steer("large", "x" * 1024)

        self.assertEqual(runtime.state["pending_inputs"], initial_pending)
        self.assertEqual(state_path.read_bytes(), before)

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
