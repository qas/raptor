"""Provider-neutral subagent activity projection tests."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-activity-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")
os.environ.setdefault("TG_CHAT_IDS", "1")

from raptor.chat import activity
from raptor.model.model_providers import ModelTarget


class RecordingActivityProvider:
    def __init__(self) -> None:
        self.opened: list[activity.ActivitySnapshot] = []
        self.updates: list[activity.ActivitySnapshot] = []
        self.finished: list[activity.ActivitySnapshot] = []
        self.messages: list[str] = []
        self.deleted: list[str] = []
        self.restored: list[str] = []

    async def open_activity_surface(
        self,
        _conversation_id,
        snapshot,
        _existing_surface_id=None,
    ) -> str:
        self.opened.append(snapshot)
        return "surface"

    async def update_activity_surface(
        self,
        _conversation_id,
        _surface_id,
        snapshot,
    ) -> None:
        self.updates.append(snapshot)

    async def append_activity_message(
        self,
        _conversation_id,
        _surface_id,
        text,
    ) -> None:
        self.messages.append(text)

    async def finish_activity_surface(
        self,
        _conversation_id,
        _surface_id,
        snapshot,
    ) -> activity.ActivityFinishResult:
        self.finished.append(snapshot)
        return activity.ActivityFinishResult(True, bool(snapshot.result))

    async def delete_activity_surface(
        self,
        _conversation_id,
        surface_id,
    ) -> None:
        self.deleted.append(surface_id)

    def restore_activity_surface(
        self,
        _conversation_id,
        surface_id,
    ) -> None:
        self.restored.append(surface_id)

    def activity_surface_conversation_id(
        self,
        conversation_id,
        surface_id,
    ) -> str:
        return f"{conversation_id}:{surface_id}"


class FailingUpdateProvider(RecordingActivityProvider):
    async def update_activity_surface(
        self,
        _conversation_id,
        _surface_id,
        _snapshot,
    ) -> None:
        raise RuntimeError("temporary update failure")


class PartialFinishProvider(RecordingActivityProvider):
    async def finish_activity_surface(
        self,
        _conversation_id,
        _surface_id,
        snapshot,
    ) -> activity.ActivityFinishResult:
        self.finished.append(snapshot)
        if len(self.finished) == 1:
            return activity.ActivityFinishResult(False, True)
        return activity.ActivityFinishResult(True, False)


class PartiallyBrokenRestoreProvider(RecordingActivityProvider):
    def restore_activity_surface(
        self,
        _conversation_id,
        surface_id,
    ) -> None:
        self.restored.append(surface_id)
        if surface_id == "broken-surface":
            raise RuntimeError("surface unavailable")


class ActivityProjectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        activity.session.set_default_model_target(
            ModelTarget("local", "test-model")
        )
        runtime = activity.session.set_default_chat("conversation")
        self.runtime_context = activity.session.bound_runtime(runtime)
        self.runtime_context.__enter__()
        self.addCleanup(
            self.runtime_context.__exit__,
            None,
            None,
            None,
        )

    async def asyncTearDown(self) -> None:
        await activity.close_activity_projections()

    async def test_duplicate_snapshots_do_not_repeat_provider_edits(self) -> None:
        provider = RecordingActivityProvider()
        initial = activity.ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="running",
            detail="Starting",
        )
        projection = activity.ActivityProjection(
            provider,
            "conversation",
            "surface",
            initial,
        )

        with patch.object(activity, "ACTIVITY_UPDATE_INTERVAL_SECONDS", 0):
            projection.publish(initial)
            self.assertIsNone(projection.task)
            changed = activity.ActivitySnapshot(
                activity_id="worker",
                title="Inspect target",
                status="running",
                detail="Reading files",
            )
            projection.publish(changed)
            assert projection.task is not None
            await projection.task
            projection.publish(changed)
            await asyncio.sleep(0)

        self.assertEqual(provider.updates, [changed])

    def test_activity_conversation_resolves_through_provider(self) -> None:
        provider = RecordingActivityProvider()
        record = {
            "id": "worker",
            "chat_id": "parent",
            "activity_surface_id": "child",
        }

        with patch.object(activity, "get_chat_provider", return_value=provider):
            conversation_id = activity.subagent_activity_conversation_id(
                record
            )

        self.assertEqual(conversation_id, "parent:child")

    async def test_metadata_and_messages_use_separate_bounds(self) -> None:
        provider = RecordingActivityProvider()
        message = "x" * (activity.MAX_ACTIVITY_MESSAGE_CHARS + 1)
        record = {
            "id": "i" * 2_000,
            "task": message,
            "status": "completed",
            "result": message,
            "chat_id": "conversation",
            "activity_surface_id": None,
        }

        with (
            patch.object(activity, "get_chat_provider", return_value=provider),
            patch.object(activity.session, "save_state"),
            patch.object(activity, "ACTIVITY_UPDATE_INTERVAL_SECONDS", 0),
        ):
            await activity.open_subagent_activity(record)
            activity.publish_subagent_activity(record, "d" * 2_000)
            for _attempt in range(10):
                if provider.updates:
                    break
                await asyncio.sleep(0)
            await activity.finish_subagent_activity(record)

        snapshot = provider.updates[0]
        final = provider.finished[0]
        self.assertLessEqual(
            len(snapshot.activity_id),
            activity.MAX_ACTIVITY_FIELD_CHARS,
        )
        self.assertEqual(
            len(snapshot.title),
            activity.MAX_ACTIVITY_MESSAGE_CHARS,
        )
        self.assertLessEqual(
            len(snapshot.detail),
            activity.MAX_ACTIVITY_FIELD_CHARS,
        )
        self.assertEqual(
            len(final.result),
            activity.MAX_ACTIVITY_MESSAGE_CHARS,
        )

    async def test_finish_preserves_output_and_keeps_surface_open(self) -> None:
        provider = RecordingActivityProvider()
        record = {
            "id": "worker",
            "task": "Inspect target",
            "status": "running",
            "chat_id": "conversation",
            "activity_surface_id": None,
        }

        with (
            patch.object(activity, "get_chat_provider", return_value=provider),
            patch.object(activity.session, "save_state"),
            patch.object(activity, "ACTIVITY_UPDATE_INTERVAL_SECONDS", 0),
        ):
            await activity.open_subagent_activity(record)
            activity.publish_subagent_response(
                record,
                reasoning_summary="Inspecting the relevant files",
                reply="The first finding",
            )
            record["status"] = "completed"
            await activity.finish_subagent_activity(record)

        final = provider.finished[0]
        self.assertEqual(final.reasoning_summary, "Inspecting the relevant files")
        self.assertEqual(final.reply, "The first finding")
        self.assertEqual(final.status, "completed")
        self.assertEqual(record["activity_surface_id"], "surface")

    async def test_open_rolls_back_new_surface_when_persistence_fails(
        self,
    ) -> None:
        provider = RecordingActivityProvider()
        record = {
            "id": "worker",
            "task": "Inspect target",
            "status": "running",
            "chat_id": "conversation",
            "activity_surface_id": None,
        }
        with (
            patch.object(activity, "get_chat_provider", return_value=provider),
            patch.object(
                activity.session,
                "save_state",
                side_effect=RuntimeError("state unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "state unavailable"),
        ):
            await activity.open_subagent_activity(record)

        self.assertIsNone(record["activity_surface_id"])
        self.assertEqual(provider.deleted, ["surface"])

    async def test_long_streams_keep_latest_output_visible(self) -> None:
        provider = RecordingActivityProvider()
        initial = activity.ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="running",
        )
        projection = activity.ActivityProjection(
            provider,
            "conversation",
            "surface",
            initial,
        )
        reply = "old" * 11_000 + "latest"

        with patch.object(activity, "ACTIVITY_UPDATE_INTERVAL_SECONDS", 0):
            projection.publish_response(reply=reply)
            assert projection.task is not None
            await projection.task

        rendered = provider.updates[-1].reply
        self.assertEqual(len(rendered), activity.MAX_ACTIVITY_STREAM_CHARS)
        self.assertTrue(rendered.startswith("..."))
        self.assertTrue(rendered.endswith("latest"))

    async def test_finish_uses_latest_snapshot_after_update_failure(self) -> None:
        provider = FailingUpdateProvider()
        initial = activity.ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="running",
        )
        projection = activity.ActivityProjection(
            provider,
            "conversation",
            "surface",
            initial,
        )
        latest = activity.ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="completed",
            reasoning_summary="latest reasoning",
            reply="latest reply",
        )

        with patch.object(activity, "ACTIVITY_UPDATE_INTERVAL_SECONDS", 0):
            projection.publish(latest)
            assert projection.task is not None
            await projection.task
            result = await projection.finish(latest)

        self.assertFalse(result.result_delivered)
        self.assertTrue(result.finished)
        self.assertEqual(provider.finished, [latest])

    async def test_stale_generation_cannot_finish_current_activity(self) -> None:
        provider = RecordingActivityProvider()
        record = {
            "id": "worker",
            "task": "Inspect target",
            "status": "running",
            "run_generation": 1,
            "chat_id": "conversation",
            "activity_surface_id": None,
        }
        with (
            patch.object(activity, "get_chat_provider", return_value=provider),
            patch.object(activity.session, "save_state"),
        ):
            await activity.open_subagent_activity(record)
            record["run_generation"] = 2
            finished = await activity.finish_subagent_activity(
                record,
                expected_generation=1,
            )

        self.assertFalse(finished)
        self.assertEqual(provider.finished, [])

    async def test_repeated_finish_does_not_repeat_result(self) -> None:
        provider = RecordingActivityProvider()
        record = {
            "id": "worker",
            "task": "Inspect target",
            "status": "completed",
            "result": "complete result",
            "chat_id": "conversation",
            "activity_surface_id": "surface",
        }
        with (
            patch.object(activity, "get_chat_provider", return_value=provider),
            patch.object(activity.session, "save_state"),
        ):
            await activity.finish_subagent_activity(record)
            await activity.finish_subagent_activity(record)

        self.assertEqual(
            [snapshot.result for snapshot in provider.finished],
            ["complete result"],
        )
        self.assertEqual(record["activity_finished_generation"], 1)
        self.assertEqual(record["activity_surface_id"], "surface")

    async def test_later_finish_does_not_repeat_delivered_result(self) -> None:
        provider = PartialFinishProvider()
        record = {
            "id": "worker",
            "task": "Inspect target",
            "status": "completed",
            "result": "complete result",
            "chat_id": "conversation",
            "activity_surface_id": "surface",
        }
        with (
            patch.object(activity, "get_chat_provider", return_value=provider),
            patch.object(activity.session, "save_state"),
        ):
            first_finished = await activity.finish_subagent_activity(record)
            second_finished = await activity.finish_subagent_activity(record)

        self.assertFalse(first_finished)
        self.assertTrue(second_finished)
        self.assertEqual(
            [snapshot.result for snapshot in provider.finished],
            ["complete result", ""],
        )

    async def test_delete_removes_provider_surface_from_record(self) -> None:
        provider = RecordingActivityProvider()
        record = {
            "id": "worker",
            "chat_id": "conversation",
            "activity_surface_id": "surface",
            "activity_finished_generation": 1,
            "activity_result_delivered": True,
        }
        with (
            patch.object(activity, "get_chat_provider", return_value=provider),
            patch.object(activity.session, "save_state"),
        ):
            deleted = await activity.delete_subagent_activity(record)

        self.assertTrue(deleted)
        self.assertEqual(provider.deleted, ["surface"])
        self.assertIsNone(record["activity_surface_id"])
        self.assertNotIn("activity_finished_generation", record)

    async def test_reconcile_isolates_surface_restore_failures(self) -> None:
        provider = PartiallyBrokenRestoreProvider()
        runtime = activity.session.current_runtime()
        previous = runtime.state["subagents"]
        runtime.state["subagents"] = {
            "broken": {
                "id": "broken",
                "status": "running",
                "chat_id": "conversation",
                "activity_surface_id": "broken-surface",
            },
            "healthy": {
                "id": "healthy",
                "status": "running",
                "chat_id": "conversation",
                "activity_surface_id": "healthy-surface",
            },
        }
        try:
            with (
                patch.object(
                    activity.session,
                    "all_chat_runtimes",
                    return_value=(runtime,),
                ),
                patch.object(activity, "get_chat_provider", return_value=provider),
                patch.object(activity, "log_exception"),
            ):
                await activity.reconcile_activity_surfaces()
        finally:
            runtime.state["subagents"] = previous

        self.assertEqual(
            provider.restored,
            ["broken-surface", "healthy-surface"],
        )

    async def test_reconcile_restores_only_open_surfaces(self) -> None:
        provider = RecordingActivityProvider()
        runtime = activity.session.current_runtime()
        previous = runtime.state["subagents"]
        runtime.state["subagents"] = {
            "open": {
                "id": "open",
                "status": "completed",
                "chat_id": "conversation",
                "activity_surface_id": "open-surface",
            },
            "without_surface": {
                "id": "without-surface",
                "status": "completed",
                "chat_id": "conversation",
                "activity_surface_id": None,
            },
        }
        try:
            with (
                patch.object(
                    activity.session,
                    "all_chat_runtimes",
                    return_value=(runtime,),
                ),
                patch.object(activity, "get_chat_provider", return_value=provider),
            ):
                await activity.reconcile_activity_surfaces()
        finally:
            runtime.state["subagents"] = previous

        self.assertEqual(provider.restored, ["open-surface"])


if __name__ == "__main__":
    unittest.main()
