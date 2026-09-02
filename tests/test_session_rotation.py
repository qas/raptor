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

from raptor.state import chat_store
import commands
import controller
from raptor.state import session
from raptor.state.session import pending_approvals
from turn_runtime import turns
from commands import command
from goals import replace_goal
from tools import chat_history_tool
from raptor.model.model_providers import ModelProvider, ModelTarget

TEST_MODEL_TARGET = {"provider_id": "local", "model": "test-model"}


async def _noop(*_a, **_k):
    return None


class SessionRotationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.target = ModelTarget("local", "model-a")
        session.set_default_model_target(self.target)
        self._chat_dir = Path(tempfile.mkdtemp(prefix="chats-"))
        self._chat_patch = patch.object(
            chat_store,
            "CHAT_DIR",
            self._chat_dir,
        )
        self._chat_patch.start()
        self.addCleanup(self._chat_patch.stop)
        chat_store._SEQ_CACHE.clear()
        self._runtime_context = session.bound_chat(
            f"rotation:{self._chat_dir.name}"
        )
        self._runtime_context.__enter__()
        self.addCleanup(
            self._runtime_context.__exit__,
            None,
            None,
            None,
        )
        session.state.clear()
        session.state.update(copy.deepcopy(session.DEFAULT_STATE))
        sid = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            model_target=self.target.to_dict(),
        )
        session.state["current_session_id"] = sid
        session.state["model_target"] = self.target.to_dict()
        session.state["approval_mode"] = "on"
        session.state["todos"] = [
            {"step": "old", "status": "pending"}
        ]
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
        self.assertEqual(session.state["model_target"], self.target.to_dict())
        self.assertEqual(session.state["approval_mode"], "on")
        hits = chat_history_tool(
            {
                "action": "search",
                "query": "remember-me",
                "session_id": old,
            }
        )
        self.assertEqual(len(hits["hits"]), 1)

    def test_bootstrap_replaces_current_ended_session(self) -> None:
        ended = str(session.state["current_session_id"])
        chat_store.end_session(ended, reason="interrupted_transition")

        with patch.object(
            session,
            "all_chat_runtimes",
            return_value=(session.current_runtime(),),
        ):
            result = session.bootstrap_runtime_storage()

        current = str(session.state["current_session_id"])
        self.assertNotEqual(current, ended)
        self.assertTrue(chat_store.session_exists(current))
        self.assertEqual(result["created_sessions"], 1)

    def test_bootstrap_recovers_committed_history_truncation(self) -> None:
        source = str(session.state["current_session_id"])
        destination = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            parent_session_id=source,
            model_target=self.target.to_dict(),
        )
        session.state["current_session_id"] = destination
        session.state["subagents"] = {
            "child": {"parent_session_id": source},
        }
        session.state["session_transition"] = {
            "kind": "history_truncate",
            "phase": "committed",
            "source_session_id": source,
            "destination_session_id": destination,
            "turns": 1,
            "copied_items": 1,
        }

        session.bootstrap_runtime_storage()

        self.assertIsNone(session.state["session_transition"])
        self.assertEqual(session.state["current_session_id"], destination)
        self.assertEqual(
            session.state["subagents"]["child"]["parent_session_id"],
            destination,
        )
        self.assertEqual(
            session.state["subagents"]["child"]["origin_parent_session_id"],
            source,
        )
        self.assertTrue(chat_store.session_is_ended(source))
        self.assertEqual(
            chat_store.read_events(destination)[-1]["name"],
            "history_truncated_complete",
        )

    def test_bootstrap_aborts_partially_created_history_candidate(self) -> None:
        source = str(session.state["current_session_id"])
        destination = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            parent_session_id=source,
            model_target=self.target.to_dict(),
        )
        chat_store.append_item(
            destination,
            {"role": "user", "content": "partial copy"},
            source="user",
        )
        session.state["session_transition"] = {
            "kind": "history_truncate",
            "phase": "preparing",
            "source_session_id": source,
            "destination_session_id": destination,
            "turns": 1,
            "copied_items": 0,
        }

        session.bootstrap_runtime_storage()

        self.assertIsNone(session.state["session_transition"])
        self.assertEqual(session.state["current_session_id"], source)
        self.assertFalse(chat_store.session_is_ended(source))
        self.assertTrue(chat_store.session_is_ended(destination))

    def test_bootstrap_restores_source_when_destination_is_missing(self) -> None:
        source = str(session.state["current_session_id"])
        destination = chat_store.new_session_id()
        session.state["current_session_id"] = destination
        session.state["session_transition"] = {
            "kind": "history_truncate",
            "phase": "committed",
            "source_session_id": source,
            "destination_session_id": destination,
            "turns": 1,
            "copied_items": 0,
        }

        session.bootstrap_runtime_storage()

        self.assertIsNone(session.state["session_transition"])
        self.assertEqual(session.state["current_session_id"], source)
        self.assertFalse(chat_store.session_is_ended(source))

    def test_bootstrap_aborts_inconsistent_history_truncation(self) -> None:
        source = str(session.state["current_session_id"])
        destination = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            parent_session_id=source,
            model_target=self.target.to_dict(),
        )
        other = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            model_target=self.target.to_dict(),
        )
        session.state["current_session_id"] = source
        session.state["subagents"] = {
            "child": {"parent_session_id": source},
        }
        session.state["session_transition"] = {
            "kind": "history_truncate",
            "phase": "committed",
            "source_session_id": source,
            "destination_session_id": destination,
            "turns": 1,
            "copied_items": 0,
        }

        session.bootstrap_runtime_storage()

        self.assertIsNone(session.state["session_transition"])
        self.assertEqual(session.state["current_session_id"], source)
        self.assertEqual(
            session.state["subagents"]["child"],
            {"parent_session_id": source},
        )
        self.assertTrue(chat_store.session_is_ended(destination))
        self.assertFalse(chat_store.session_is_ended(source))
        self.assertTrue(chat_store.session_exists(other))

    def test_session_transition_marker_blocks_changes(self) -> None:
        session.state["session_transition"] = {
            "kind": "history_truncate",
            "phase": "committed",
            "source_session_id": session.state["current_session_id"],
            "destination_session_id": session.state["current_session_id"],
            "turns": 1,
            "copied_items": 0,
        }
        self.assertTrue(controller.session_transition_busy())

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
        previous = chat_store.create_session(model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key=session.current_runtime().key,
        )
        chat_store.append_item(
            previous,
            {"role": "user", "content": "find this launch note"},
            source="user",
        )
        child = chat_store.create_session(model_target=TEST_MODEL_TARGET,
            kind="subagent",
            chat_key=session.current_runtime().key,
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
            chat_store.create_session(model_target=TEST_MODEL_TARGET,
                kind="main",
                chat_key=session.current_runtime().key,
            )
        sent: list[str] = []

        async def capture(_chat_id, text, **_kwargs):
            sent.append(text)

        with patch("commands.send", capture):
            await command(1, "/chats")

        self.assertIn(f"{current} ·", sent[0])
        self.assertIn(f"{current} ·", sent[0].splitlines()[1])

    async def test_chats_searches_transcript_content(self) -> None:
        matching = chat_store.create_session(model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key=session.current_runtime().key,
        )
        chat_store.append_item(
            matching,
            {"role": "user", "content": "NeedleProject details"},
            source="user",
        )
        other = chat_store.create_session(model_target=TEST_MODEL_TARGET,
            kind="main",
            chat_key=session.current_runtime().key,
        )
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
        target = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            model_target=self.target.to_dict(),
        )
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

    async def test_resume_checks_target_credentials_before_archiving_current(
        self,
    ) -> None:
        current = session.state["current_session_id"]
        target = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            model_target=self.target.to_dict(),
        )
        chat_store.end_session(target, reason="new_session")
        sent: list[str] = []

        async def capture(_chat_id, text, **_kwargs):
            sent.append(text)

        provider = ModelProvider(
            id="local",
            base_url="http://local.example/v1",
            api_key_env="MISSING_RESUME_KEY",
        )
        with (
            patch("commands.send", capture),
            patch("commands.model_provider", return_value=provider),
            patch.dict(os.environ, {}, clear=True),
        ):
            await command(1, f"/resume {target}")

        self.assertEqual(session.state["current_session_id"], current)
        self.assertFalse(chat_store.session_is_ended(current))
        self.assertIn("MISSING_RESUME_KEY", sent[0])

    async def test_resume_rejects_subagent_session(self) -> None:
        current = session.state["current_session_id"]
        child = chat_store.create_session(model_target=TEST_MODEL_TARGET,
            kind="subagent",
            chat_key=session.current_runtime().key,
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

    def test_oversized_state_is_rejected_before_json_decode(self) -> None:
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        state_path.write_bytes(b" " * 65)
        with (
            patch.object(session, "STATE_PATH", state_path),
            patch.object(session, "MAX_STATE_LOAD_BYTES", 64),
            self.assertRaisesRegex(RuntimeError, "MAX_STATE_LOAD_BYTES"),
        ):
            session.load_state()

    def test_persisted_chat_count_is_bounded(self) -> None:
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        state_path.write_text(json.dumps({
            "schema_version": session.STATE_SCHEMA_VERSION,
            "model": None,
            "runtime": {},
            "chats": {
                "one": {
                    "conversation_id": "one",
                    "state": session.CHAT_DEFAULT_STATE,
                },
                "two": {
                    "conversation_id": "two",
                    "state": session.CHAT_DEFAULT_STATE,
                },
            },
        }))
        with (
            patch.object(session, "STATE_PATH", state_path),
            patch.object(session, "MAX_CHAT_RUNTIMES", 1),
            self.assertRaisesRegex(RuntimeError, "MAX_CHAT_RUNTIMES"),
        ):
            session.load_state()

    def test_new_chat_admission_is_bounded(self) -> None:
        root_state = copy.deepcopy(session.GLOBAL_DEFAULT_STATE)
        chat_dir = Path(tempfile.mkdtemp(prefix="bounded-chats-"))
        with (
            patch.object(session, "_root_state", root_state),
            patch.dict(session._runtimes, {}, clear=True),
            patch.object(session, "_default_runtime_key", None),
            patch.object(session, "MAX_CHAT_RUNTIMES", 1),
            patch.object(chat_store, "CHAT_DIR", chat_dir),
        ):
            session.ensure_chat("one")
            with self.assertRaisesRegex(RuntimeError, "capacity"):
                session.ensure_chat("two")

    def test_failed_chat_persistence_does_not_admit_runtime(self) -> None:
        root_state = copy.deepcopy(session.GLOBAL_DEFAULT_STATE)
        chat_dir = Path(tempfile.mkdtemp(prefix="failed-chat-"))
        with (
            patch.object(session, "_root_state", root_state),
            patch.dict(session._runtimes, {}, clear=True),
            patch.object(session, "_default_runtime_key", None),
            patch.object(chat_store, "CHAT_DIR", chat_dir),
            patch.object(
                session,
                "_write_root_state",
                side_effect=OSError("disk full"),
            ),
        ):
            with self.assertRaises(OSError):
                session.ensure_chat("one")

            self.assertNotIn("one", session._runtimes)
            self.assertNotIn("one", root_state["chats"])

    def test_invalid_persisted_plan_is_reported(self) -> None:
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        state_path.write_text(
            json.dumps({
                "schema_version": session.STATE_SCHEMA_VERSION,
                "model": None,
                "runtime": {},
                "chats": {
                    "local": {
                        "conversation_id": "local",
                        "state": {
                            **session.CHAT_DEFAULT_STATE,
                            "todos": [
                                {
                                    "id": 1,
                                    "text": "old shape",
                                    "status": "pending",
                                }
                            ],
                        },
                    }
                },
            })
        )
        with patch.object(session, "STATE_PATH", state_path):
            with self.assertRaisesRegex(
                RuntimeError,
                "persisted local root plan",
            ):
                session.load_state()

    def test_persisted_subagent_cannot_cross_chat_boundaries(self) -> None:
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        chat_state = copy.deepcopy(session.CHAT_DEFAULT_STATE)
        chat_state["subagents"] = {
            "worker": {
                "id": "worker",
                "chat_key": "local",
                "chat_id": "responses_api:other",
                "status": "completed",
            }
        }
        state_path.write_text(json.dumps({
            "schema_version": session.STATE_SCHEMA_VERSION,
            "model": None,
            "runtime": {},
            "chats": {
                "local": {
                    "conversation_id": "local",
                    "state": chat_state,
                }
            },
        }))

        with patch.object(session, "STATE_PATH", state_path):
            with self.assertRaisesRegex(
                RuntimeError,
                "conversation does not match",
            ):
                session.load_state()

    def test_persisted_thread_with_orphaned_parent_is_not_promoted(self) -> None:
        owner = session.current_runtime().key
        branch = chat_store.create_session(model_target=TEST_MODEL_TARGET,
            kind="thread",
            chat_key=owner,
            parent_session_id=session.state["current_session_id"],
        )
        missing_parent = "20260101-000000-deadbeef"
        chat_state = copy.deepcopy(session.CHAT_DEFAULT_STATE)
        chat_state["current_session_id"] = branch
        chat_state["thread"] = {
            "parent_session_id": missing_parent,
            "session_id": branch,
            "parent_interrupted_subagents": [],
        }
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        state_path.write_text(json.dumps({
            "schema_version": session.STATE_SCHEMA_VERSION,
            "model": None,
            "runtime": {},
            "chats": {
                owner: {
                    "conversation_id": owner,
                    "state": chat_state,
                }
            },
        }))

        with patch.object(session, "STATE_PATH", state_path):
            loaded = session.load_state()

        repaired = loaded["chats"][owner]["state"]
        self.assertIsNone(repaired["thread"])
        self.assertIsNone(repaired["current_session_id"])

    def test_interrupted_subagent_retains_acknowledged_pending_inputs(
        self,
    ) -> None:
        state_path = Path(tempfile.mkdtemp()) / "state.json"
        chat_state = copy.deepcopy(session.CHAT_DEFAULT_STATE)
        chat_state["subagents"] = {
            "worker": {
                "id": "worker",
                "chat_key": "local",
                "chat_id": "local",
                "status": "running",
                "model_target": self.target.to_dict(),
                "pending_inputs": ["first steer", "second steer"],
                "todos": [],
            }
        }
        state_path.write_text(json.dumps({
            "schema_version": session.STATE_SCHEMA_VERSION,
            "model": None,
            "runtime": {},
            "chats": {
                "local": {
                    "conversation_id": "local",
                    "state": chat_state,
                }
            },
        }))

        with patch.object(session, "STATE_PATH", state_path):
            loaded = session.load_state()

        record = loaded["chats"]["local"]["state"]["subagents"]["worker"]
        self.assertEqual(record["status"], "interrupted")
        self.assertEqual(
            record["pending_inputs"],
            ["first steer", "second steer"],
        )


if __name__ == "__main__":
    unittest.main()
