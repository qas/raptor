"""Provider-neutral background activity projection tests."""

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

import activity


class RecordingActivityProvider:
    def __init__(self) -> None:
        self.opened: list[activity.ActivitySnapshot] = []
        self.updates: list[activity.ActivitySnapshot] = []
        self.closed: list[activity.ActivitySnapshot] = []

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

    async def close_activity_surface(
        self,
        _conversation_id,
        _surface_id,
        snapshot,
    ) -> None:
        self.closed.append(snapshot)

    def restore_activity_surface(
        self,
        _conversation_id,
        _surface_id,
    ) -> None:
        return None


class ActivityProjectionTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_metadata_and_messages_use_their_respective_bounds(
        self,
    ) -> None:
        provider = RecordingActivityProvider()
        message = "x" * (activity.MAX_ACTIVITY_MESSAGE_CHARS + 1)
        record = {
            "id": "i" * 2_000,
            "task": message,
            "status": "completed",
            "result": message,
            "chat_id": "conversation",
            "activity_surface_id": None,
            "activity_surface_closed": True,
        }

        with (
            activity.session.bound_chat("conversation"),
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
            await activity.close_subagent_activity(record)

        self.assertTrue(provider.updates)
        snapshot = provider.updates[0]

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
            len(snapshot.result),
            activity.MAX_ACTIVITY_MESSAGE_CHARS,
        )
        self.assertGreater(
            len(snapshot.result),
            activity.MAX_ACTIVITY_FIELD_CHARS,
        )

    async def test_model_output_survives_activity_updates_and_close(self) -> None:
        provider = RecordingActivityProvider()
        record = {
            "id": "worker",
            "task": "Inspect target",
            "status": "running",
            "chat_id": "conversation",
            "activity_surface_id": None,
            "activity_surface_closed": True,
        }

        with (
            activity.session.bound_chat("conversation"),
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
            activity.publish_subagent_activity(record, "Running a tool")
            record["status"] = "completed"
            await activity.close_subagent_activity(record)

        self.assertEqual(len(provider.closed), 1)
        final = provider.closed[0]
        self.assertEqual(final.reasoning_summary, "Inspecting the relevant files")
        self.assertEqual(final.reply, "The first finding")
        self.assertEqual(final.status, "completed")

    async def test_long_streams_keep_the_latest_output_visible(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
