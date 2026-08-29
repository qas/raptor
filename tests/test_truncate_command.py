"""Focused tests for the /truncate command."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-truncate-home-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")

import chat_store
import chat_runtime
import commands
import session
import storage
from model_providers import ModelTarget
from turn_runtime import turns


class TruncateCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="raptor-truncate-chat-"))
        self.chat_dir = patch.object(chat_store, "CHAT_DIR", self.directory)
        self.chat_dir.start()
        self.addCleanup(self.chat_dir.stop)
        chat_store._SEQ_CACHE.clear()
        self.target = ModelTarget("local", "model")
        session.set_default_model_target(self.target)
        self.runtime_context = session.bound_chat(
            f"truncate:{self.directory.name}"
        )
        self.runtime_context.__enter__()
        self.addCleanup(self.runtime_context.__exit__, None, None, None)
        session.state.clear()
        session.state.update(copy.deepcopy(session.DEFAULT_STATE))
        session.state["model_target"] = self.target.to_dict()
        self.source = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            model_target=self.target.to_dict(),
        )
        session.state["current_session_id"] = self.source
        turns.finish()

    async def test_success_preserves_state_and_archives_source(self) -> None:
        session.state["todos"] = [{"step": "keep", "status": "pending"}]
        session.state["approval_mode"] = "on"
        session.state["goal"] = {"objective": "keep"}
        session.state["subagents"] = {"worker": {"status": "done"}}
        session.state["interrupted_subagents"] = [{"id": "worker"}]
        session.state["subagents"]["worker"]["parent_session_id"] = self.source
        for text in ("first", "second"):
            chat_store.append_item(
                self.source,
                {"role": "user", "content": text},
                source="user",
            )
        send = AsyncMock()
        with (
            patch.object(commands, "send", send),
            patch.object(commands, "session_transition_busy", return_value=False),
        ):
            handled = await commands.command(1, "/truncate 2")

        self.assertTrue(handled)
        new_id = str(session.state["current_session_id"])
        self.assertEqual(session.state["current_session_id"], new_id)
        self.assertEqual(session.state["model_target"], self.target.to_dict())
        self.assertEqual(session.state["approval_mode"], "on")
        self.assertEqual(session.state["goal"], {"objective": "keep"})
        self.assertEqual(
            session.state["todos"],
            [{"step": "keep", "status": "pending"}],
        )
        self.assertEqual(
            session.state["subagents"]["worker"]["status"], "done"
        )
        self.assertEqual(
            session.state["interrupted_subagents"], [{"id": "worker"}]
        )
        self.assertIsNone(session.state.get("session_transition"))
        destination_events = chat_store.read_events(new_id)
        meta_names = [
            event["name"]
            for event in destination_events
            if event.get("type") == "meta"
        ]
        self.assertEqual(
            meta_names,
            ["history_truncated", "history_truncated_complete"],
        )
        worker = session.state["subagents"]["worker"]
        self.assertEqual(worker["parent_session_id"], new_id)
        self.assertEqual(worker["origin_parent_session_id"], self.source)
        self.assertEqual(
            chat_store.read_events(self.source)[-1]["reason"],
            "history_truncated",
        )
        self.assertIn(
            "Files and tool side effects were not reverted",
            send.await_args.args[1],
        )

    async def test_deletes_only_chat_messages_from_removed_turns(self) -> None:
        kept_turn = chat_store.append_item(
            self.source,
            {"role": "user", "content": "keep"},
            source="user",
            data={
                "chat_message": {
                    "conversation_id": "telegram:1",
                    "message_id": 10,
                }
            },
        )
        chat_store.append_meta(
            self.source,
            "chat_delivery",
            {
                "conversation_id": "telegram:1",
                "message_ids": [11],
                "user_turn_seq": kept_turn["seq"],
            },
        )
        removed_turn = chat_store.append_item(
            self.source,
            {"role": "user", "content": "remove"},
            source="user",
            data={
                "chat_message": {
                    "conversation_id": "telegram:1",
                    "message_id": 20,
                }
            },
        )
        chat_store.append_meta(
            self.source,
            "chat_delivery",
            {
                "conversation_id": "telegram:1",
                "message_ids": [21, 22],
                "user_turn_seq": removed_turn["seq"],
            },
        )
        provider = Mock()
        provider.encode_conversation_id.return_value = "telegram:1"
        provider.delete_message = AsyncMock()

        with (
            patch.object(commands, "send", AsyncMock()) as send,
            patch.object(commands, "get_chat_provider", return_value=provider),
        ):
            await commands.command(1, "/truncate 1")

        self.assertEqual(
            [call.args for call in provider.delete_message.await_args_list],
            [(1, 20), (1, 21), (1, 22)],
        )
        self.assertIn("Deleted 3 linked chat message(s)", send.await_args.args[1])
        retained_session = str(session.state["current_session_id"])
        retained_delivery = next(
            event
            for event in chat_store.read_events(retained_session)
            if event.get("name") == "chat_delivery"
        )
        self.assertEqual(retained_delivery["data"]["message_ids"], [11])

        provider.delete_message.reset_mock()
        with (
            patch.object(commands, "send", AsyncMock()),
            patch.object(commands, "get_chat_provider", return_value=provider),
        ):
            await commands.command(1, "/truncate 1")
        self.assertEqual(
            [call.args for call in provider.delete_message.await_args_list],
            [(1, 10), (1, 11)],
        )

    async def test_send_archives_all_provider_message_ids(self) -> None:
        chat_store.append_item(
            self.source,
            {"role": "user", "content": "split this"},
            source="user",
        )
        provider = Mock()
        provider.send_text = AsyncMock(return_value=(31, 32))
        provider.encode_conversation_id.return_value = "telegram:1"
        previous = chat_runtime.set_chat_provider(provider)
        self.addCleanup(chat_runtime.set_chat_provider, previous)

        result = await chat_runtime.send(1, "split response")

        self.assertEqual(result, (31, 32))
        delivery = chat_store.read_events(self.source)[-1]
        self.assertEqual(delivery["name"], "chat_delivery")
        self.assertEqual(
            delivery["data"],
            {
                "conversation_id": "telegram:1",
                "message_ids": [31, 32],
                "user_turn_seq": 2,
            },
        )

    async def test_invalid_and_too_large_values_are_no_ops(self) -> None:
        send = AsyncMock()
        with patch.object(commands, "send", send), patch.object(
            commands, "plan_session_truncation"
        ) as helper:
            await commands.command(1, "/truncate 0")
            await commands.command(1, "/truncate two")
            await commands.command(1, "/truncate 1 extra")
            await commands.command(1, "/truncate +1")
            await commands.command(1, "/truncate 1_0")
            await commands.command(1, "/truncate ١")
            await commands.command(1, "/truncate " + "9" * 5000)
        helper.assert_not_called()
        self.assertEqual(session.state["current_session_id"], self.source)

        with (
            patch.object(commands, "send", send),
            patch.object(
                commands,
                "plan_session_truncation",
                side_effect=ValueError("turns exceeds available user turns"),
            ),
        ):
            await commands.command(1, "/truncate 99")
        self.assertEqual(session.state["current_session_id"], self.source)

    async def test_thread_busy_and_pending_delivery_are_rejected(self) -> None:
        send = AsyncMock()
        with patch.object(commands, "send", send):
            with patch.object(commands, "thread_active", return_value=True):
                await commands.command(1, "/truncate 1")
            with (
                patch.object(commands, "thread_active", return_value=False),
                patch.object(commands, "session_transition_busy", return_value=True),
            ):
                await commands.command(1, "/truncate 1")
            session.state["pending_delivery"] = {
                "session_id": self.source,
                "seq": 1,
            }
            with patch.object(commands, "session_transition_busy", return_value=False):
                await commands.command(1, "/truncate 1")
        self.assertEqual(session.state["current_session_id"], self.source)
        self.assertEqual(send.await_count, 3)

    async def test_real_helper_copies_only_history_before_last_turn(self) -> None:
        for text in ("first", "first answer", "second", "second answer"):
            role = "user" if text in {"first", "second"} else "assistant"
            chat_store.append_item(
                self.source,
                {"role": role, "content": text},
                source=role,
            )
        send = AsyncMock()
        with patch.object(commands, "send", send):
            await commands.command(1, "/truncate 1")
        new_events = chat_store.read_events(session.state["current_session_id"])
        contents = [
            event["item"]["content"]
            for event in new_events
            if event.get("type") == "item"
        ]
        self.assertEqual(contents, ["first", "first answer"])

    async def test_preparing_save_failure_creates_no_destination(self) -> None:
        chat_store.append_item(
            self.source,
            {"role": "user", "content": "keep"},
            source="user",
        )
        sessions_before = {row["session_id"] for row in chat_store.list_sessions()}
        send = AsyncMock()
        with (
            patch.object(commands, "send", send),
            patch.object(commands, "save_state", side_effect=OSError("disk")),
        ):
            await commands.command(1, "/truncate 1")
        self.assertEqual(session.state["current_session_id"], self.source)
        self.assertEqual(
            {row["session_id"] for row in chat_store.list_sessions()},
            sessions_before,
        )
        self.assertIsNone(session.state["session_transition"])

    async def test_save_failure_restores_subagent_parent_record(self) -> None:
        original = {"status": "done", "parent_session_id": self.source}
        session.state["subagents"] = {"worker": copy.deepcopy(original)}
        chat_store.append_item(
            self.source,
            {"role": "user", "content": "keep"},
            source="user",
        )
        with (
            patch.object(commands, "send", AsyncMock()),
            patch.object(commands, "save_state", side_effect=OSError("disk")),
        ):
            await commands.command(1, "/truncate 1")
        self.assertEqual(session.state["subagents"]["worker"], original)

    async def test_candidate_is_created_only_after_preparing_marker(self) -> None:
        chat_store.append_item(
            self.source,
            {"role": "user", "content": "remove"},
            source="user",
        )
        observed: dict[str, object] = {}

        def fail_materialization(_plan, destination_id):
            observed["current"] = session.state["current_session_id"]
            observed["marker"] = copy.deepcopy(
                session.state["session_transition"]
            )
            observed["destination_id"] = destination_id
            raise OSError("copy failed")

        with (
            patch.object(commands, "send", AsyncMock()),
            patch.object(
                commands,
                "materialize_session_truncation",
                side_effect=fail_materialization,
            ),
        ):
            await commands.command(1, "/truncate 1")

        self.assertEqual(observed["current"], self.source)
        self.assertEqual(observed["marker"]["phase"], "preparing")
        self.assertEqual(
            observed["marker"]["destination_session_id"],
            observed["destination_id"],
        )
        self.assertIsNone(session.state["session_transition"])

    async def test_destination_id_collision_does_not_touch_existing_chat(
        self,
    ) -> None:
        chat_store.append_item(
            self.source,
            {"role": "user", "content": "remove"},
            source="user",
        )
        existing = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            model_target=self.target.to_dict(),
        )
        before = chat_store.chat_path(existing).read_bytes()
        destination = chat_store.new_session_id()

        with (
            patch.object(commands, "send", AsyncMock()),
            patch.object(
                commands,
                "generate_session_id",
                side_effect=(existing, destination),
            ) as generate,
        ):
            await commands.command(1, "/truncate 1")

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(session.state["current_session_id"], destination)
        self.assertEqual(chat_store.chat_path(existing).read_bytes(), before)

    async def test_commit_save_failure_aborts_candidate_and_restores_source(
        self,
    ) -> None:
        chat_store.append_item(
            self.source,
            {"role": "user", "content": "remove"},
            source="user",
        )
        real_save = commands.save_state
        calls = 0
        sessions_before = {
            row["session_id"] for row in chat_store.list_sessions()
        }

        def fail_commit_save() -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("commit failed")
            real_save()

        with (
            patch.object(commands, "send", AsyncMock()),
            patch.object(commands, "save_state", side_effect=fail_commit_save),
        ):
            await commands.command(1, "/truncate 1")

        self.assertEqual(session.state["current_session_id"], self.source)
        self.assertIsNone(session.state["session_transition"])
        candidates = [
            row["session_id"]
            for row in chat_store.list_sessions()
            if row["session_id"] not in sessions_before
        ]
        self.assertEqual(len(candidates), 1)
        self.assertTrue(chat_store.session_is_ended(candidates[0]))

    async def test_post_publish_fsync_failure_archives_owned_candidate(
        self,
    ) -> None:
        chat_store.append_item(
            self.source,
            {"role": "user", "content": "remove"},
            source="user",
        )
        sessions_before = {
            row["session_id"] for row in chat_store.list_sessions()
        }
        real_fsync_directory = storage.fsync_directory
        chat_directory = self.directory.resolve()

        def fail_chat_fsync(directory: Path) -> None:
            if directory.resolve() == chat_directory:
                raise OSError("directory fsync failed")
            real_fsync_directory(directory)

        with (
            patch.object(commands, "send", AsyncMock()),
            patch.object(
                storage,
                "fsync_directory",
                side_effect=fail_chat_fsync,
            ),
        ):
            await commands.command(1, "/truncate 1")

        self.assertEqual(session.state["current_session_id"], self.source)
        self.assertIsNone(session.state["session_transition"])
        candidates = [
            row["session_id"]
            for row in chat_store.list_sessions()
            if row["session_id"] not in sessions_before
        ]
        self.assertEqual(len(candidates), 1)
        self.assertTrue(chat_store.session_is_ended(candidates[0]))

    async def test_completion_save_failure_retains_marker(self) -> None:
        session.state["subagents"] = {
            "worker": {"parent_session_id": self.source, "status": "done"}
        }
        chat_store.append_item(
            self.source,
            {"role": "user", "content": "remove"},
            source="user",
        )
        calls = 0

        def fail_final_save() -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("disk")

        with (
            patch.object(commands, "send", AsyncMock()),
            patch.object(commands, "save_state", side_effect=fail_final_save),
        ):
            await commands.command(1, "/truncate 1")
        destination = str(session.state["current_session_id"])
        self.assertEqual(session.state["current_session_id"], destination)
        self.assertEqual(
            session.state["session_transition"]["destination_session_id"],
            destination,
        )
        self.assertEqual(
            session.state["subagents"]["worker"]["parent_session_id"],
            self.source,
        )

    async def test_old_archive_failure_keeps_new_session_and_warns(self) -> None:
        chat_store.append_item(
            self.source,
            {"role": "user", "content": "remove"},
            source="user",
        )
        send = AsyncMock()
        real_end = commands.end_session

        def end(session_id, **kwargs):
            if session_id == self.source:
                raise OSError("archive disk failure")
            return real_end(session_id, **kwargs)

        with (
            patch.object(commands, "send", send),
            patch.object(commands, "end_session", side_effect=end),
            patch.object(commands, "log_exception") as log_error,
        ):
            await commands.command(1, "/truncate 1")
        destination = str(session.state["current_session_id"])
        self.assertEqual(session.state["current_session_id"], destination)
        log_error.assert_called_once()
        self.assertIn("could not archive", send.await_args.args[1])


if __name__ == "__main__":
    unittest.main()
