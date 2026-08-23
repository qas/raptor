import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

import session
import subagents


class BackgroundSubagentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        session.subagent_records.clear()
        while True:
            try:
                session.subagent_events.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                session.subagent_events.task_done()

    async def test_subagent_model_never_falls_back_to_main_selection(self) -> None:
        with (
            patch.object(subagents, "SUBAGENT_RESPONSES_MODEL", ""),
            patch.dict(subagents.state, {"model": "main-model"}),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "SUBAGENT_RESPONSES_MODEL is not configured",
            ):
                subagents.subagent_model()

    async def test_provider_prefixed_conversation_id_is_preserved(self) -> None:
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "depth": 1,
            "allow_subagents": False,
            "notify_completion": False,
        }
        run = AsyncMock(return_value="finished")

        with patch.object(subagents, "run_subagent", run), patch.object(
            subagents, "save_state"
        ):
            await subagents.run_background_subagent(record)

        run.assert_awaited_once_with(
            agent_id="worker-1",
            chat_id="telegram:123",
            depth=1,
            allow_subagents=False,
        )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["result"], "finished")

    async def test_subagent_compaction_uses_subagent_budgets(self) -> None:
        ensure = AsyncMock(return_value=[])
        record = {"id": "worker-1", "session_id": "subagent-session"}
        with (
            patch.object(
                subagents,
                "subagent_context_input_budget",
                return_value=500,
            ),
            patch.object(
                subagents,
                "subagent_compaction_generation_budget",
                return_value=128,
            ),
            patch.object(
                subagents,
                "build_active_context",
                return_value=[{"role": "user", "content": "large"}],
            ),
            patch.object(
                subagents,
                "estimate_subagent_request_tokens",
                return_value=600,
            ),
            patch.object(subagents, "ensure_context_under_budget", ensure),
        ):
            await subagents.maybe_compact_subagent(
                record,
                allow_subagents=False,
                depth=1,
            )

        self.assertEqual(ensure.await_args.kwargs["input_budget"], 500)
        self.assertEqual(ensure.await_args.kwargs["generation_budget"], 128)

    def test_tool_event_recovery_history_is_bounded(self) -> None:
        events = [{"id": index} for index in range(5)]
        with patch.object(subagents, "MAX_SUBAGENT_TOOL_EVENTS", 3):
            retained = subagents._bounded_tool_events(events)
        self.assertEqual(retained, events[-3:])

    async def test_running_subagent_pending_input_queue_is_bounded(self) -> None:
        record = {
            "id": "worker-1",
            "status": "running",
            "task_count": 1,
            "pending_inputs": ["already queued"],
            "allow_subagents": False,
        }
        session.subagent_records["worker-1"] = record
        with patch.object(subagents, "MAX_SUBAGENT_PENDING_INPUTS", 1):
            result = await subagents.subagent_tool(
                {"agent_id": "worker-1", "task": "overflow"},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "queue_full")
        self.assertEqual(record["pending_inputs"], ["already queued"])
        self.assertEqual(record["task_count"], 1)

    def test_continuation_uses_bounded_task_counter_not_task_history(self) -> None:
        record = {
            "session_id": "subagent-session",
            "status": "completed",
            "task_count": 4,
        }
        with (
            patch.object(subagents, "append_item"),
            patch.object(subagents, "save_state"),
        ):
            subagents.continue_record(
                record,
                "continue",
                chat_id="telegram:123",
                depth=1,
                background=False,
                allow_subagents=False,
            )

        self.assertEqual(record["task_count"], 5)
        self.assertNotIn("tasks", record)

    def test_record_retention_excludes_protected_records_from_limit(self) -> None:
        records = {
            "old": {"status": "completed", "completed_at": 1},
            "new": {"status": "completed", "completed_at": 2},
            "running": {"status": "running", "started_at": 3},
            "pending": {
                "status": "completed",
                "completed_at": 4,
                "completion_pending": True,
            },
        }
        with patch.object(session, "MAX_SUBAGENT_RECORDS", 1):
            removed = session._prune_subagent_mapping(records, [])
        self.assertEqual(removed, 1)
        self.assertEqual(set(records), {"new", "running", "pending"})

    def test_interrupted_subagent_checkpoints_are_bounded_and_unique(self) -> None:
        checkpoints = [
            {"id": "a", "interrupted_at": 1},
            {"id": "a", "interrupted_at": 3},
            {"id": "b", "interrupted_at": 2},
        ]
        with patch.object(session, "MAX_SUBAGENT_RECORDS", 1):
            retained = session.bounded_interrupted_subagents(checkpoints)
        self.assertEqual(retained, [{"id": "a", "interrupted_at": 3}])

    async def test_background_subagent_capacity_is_bounded(self) -> None:
        session.subagent_tasks["existing"] = AsyncMock()
        try:
            with patch.object(subagents, "MAX_BACKGROUND_SUBAGENTS", 1):
                result = await subagents.subagent_tool(
                    {"task": "another", "background": True},
                    chat_id="telegram:123",
                    execution_context={"depth": 0},
                )
        finally:
            session.subagent_tasks.clear()

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "capacity_reached")

    async def test_completion_delivery_exception_is_deferred(self) -> None:
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "status": "completed",
            "completion_pending": True,
            "completion_attempts": 0,
        }
        session.subagent_records["worker-1"] = record
        delivered = AsyncMock(side_effect=RuntimeError("controller failed"))
        controller = types.ModuleType("controller")
        controller.enqueue_internal_input = delivered

        with (
            patch.dict(sys.modules, {"controller": controller}),
            patch.object(subagents, "save_state"),
            patch.object(subagents, "log_event") as logged,
        ):
            event_task = asyncio.create_task(subagents.completion_event_loop())
            try:
                await asyncio.wait_for(_wait_until_called(delivered), timeout=1)
                await asyncio.sleep(0)
            finally:
                event_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await event_task

        self.assertEqual(record["completion_attempts"], 1)
        self.assertTrue(record["completion_pending"])
        self.assertEqual(
            [call.args[1] for call in logged.call_args_list],
            ["completion_delivery_error", "completion_deferred"],
        )

    async def test_deferred_completion_requeues_on_user_activity(self) -> None:
        record = {
            "id": "worker-1",
            "completion_pending": True,
            "completion_attempts": 1,
        }
        session.subagent_records["worker-1"] = record

        with patch.object(subagents, "save_state") as saved:
            count = await subagents.requeue_deferred_subagent_completions()

        self.assertEqual(count, 1)
        self.assertEqual(record["completion_attempts"], 0)
        self.assertEqual(
            session.subagent_events.get_nowait()["id"],
            "worker-1",
        )
        saved.assert_called_once()

    async def test_explicit_stop_discards_deferred_completion(self) -> None:
        record = {
            "id": "worker-1",
            "completion_pending": True,
            "completion_attempts": 1,
        }
        session.subagent_records["worker-1"] = record

        with patch.object(subagents, "save_state") as saved:
            cancelled = await subagents.cancel_background_subagents(
                discard_pending=True,
            )

        self.assertEqual(cancelled, 0)
        self.assertFalse(record["completion_pending"])
        self.assertEqual(record["completion_attempts"], 0)
        saved.assert_called_once()


async def _wait_until_called(mock: AsyncMock) -> None:
    while not mock.await_count:
        await asyncio.sleep(0)


if __name__ == "__main__":
    unittest.main()
