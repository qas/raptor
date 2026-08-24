import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

import session
import subagents


class BackgroundSubagentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.runtime_context = session.bound_chat("telegram:123")
        self.runtime_context.__enter__()
        self.addCleanup(
            self.runtime_context.__exit__,
            None,
            None,
            None,
        )
        session.subagent_records.clear()
        while True:
            try:
                session.subagent_events.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                session.subagent_events.task_done()

    def test_subagent_payload_rejects_instruction_roles_in_history(self) -> None:
        with (
            patch.object(subagents, "SUBAGENT_RESPONSES_MODEL", "test-model"),
            self.assertRaisesRegex(ValueError, "instructions field"),
        ):
            subagents.build_subagent_payload(
                [{"role": "developer", "content": "late instruction"}],
                allow_subagents=False,
                depth=1,
            )

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

    async def test_subagent_stream_uses_independent_backend_and_callbacks(
        self,
    ) -> None:
        completed = {"status": "completed", "output": []}
        stream = AsyncMock(return_value=completed)
        on_text = AsyncMock()
        on_reasoning = AsyncMock()

        with (
            patch.object(subagents, "SUBAGENT_RESPONSES_MODEL", "worker-model"),
            patch.object(
                subagents,
                "SUBAGENT_RESPONSES_BASE_URL",
                "http://worker.example/v1",
            ),
            patch.object(subagents, "stream_response_payload", stream),
        ):
            result = await subagents._create_subagent_response_once(
                [],
                agent_id="worker-1",
                allow_subagents=False,
                depth=1,
                tools=[],
                reasoning_summary="auto",
                on_text=on_text,
                on_reasoning_summary=on_reasoning,
            )

        self.assertEqual(result, completed)
        kwargs = stream.await_args.kwargs
        self.assertEqual(kwargs["url"], "http://worker.example/v1/responses")
        self.assertEqual(kwargs["payload"]["model"], "worker-model")
        self.assertTrue(kwargs["payload"]["stream"])
        self.assertEqual(kwargs["payload"]["reasoning"]["summary"], "auto")
        self.assertIs(kwargs["on_text"], on_text)
        self.assertIs(kwargs["on_reasoning_summary"], on_reasoning)

    async def test_background_runtime_projects_reasoning_and_reply(self) -> None:
        record = {
            "id": "worker-1",
            "session_id": "session-1",
            "background": True,
            "pending_inputs": [],
            "tool_events": [],
            "todos": [],
        }
        session.subagent_records["worker-1"] = record

        async def create_response(_items, **kwargs):
            await kwargs["on_reasoning_summary"]("Checking files")
            await kwargs["on_text"]("Found the issue")
            return {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Found the issue"}
                        ],
                    }
                ],
            }

        projected = []
        with (
            patch.object(
                subagents,
                "build_active_context",
                return_value=[{"role": "user", "content": "Inspect"}],
            ),
            patch.object(
                subagents,
                "skill_catalog_instructions",
                AsyncMock(return_value=""),
            ),
            patch.object(subagents, "subagent_context_input_budget", return_value=0),
            patch.object(subagents, "create_subagent_response", create_response),
            patch.object(subagents, "append_item"),
            patch.object(subagents, "save_state"),
            patch.object(
                subagents,
                "publish_subagent_response",
                side_effect=lambda _record, **values: projected.append(values),
            ),
        ):
            result = await subagents.run_subagent(
                agent_id="worker-1",
                chat_id="telegram:123",
                depth=1,
                allow_subagents=False,
            )

        self.assertEqual(result, "Found the issue")
        self.assertEqual(
            projected,
            [
                {"reasoning_summary": "", "reply": ""},
                {"reasoning_summary": "Checking files"},
                {"reply": "Found the issue"},
            ],
        )

    async def test_status_projection_excludes_private_child_context(self) -> None:
        session.subagent_records["worker-1"] = {
            "id": "worker-1",
            "status": "completed",
            "task": "inspect",
            "result": "done",
            "tool_events": [{"secret": "private"}],
            "pending_inputs": ["private steer"],
            "todos": [{"step": "private plan"}],
            "recovery_context": {"secret": "private"},
        }

        result = await subagents.subagent_tool(
            {"agent_id": "worker-1"},
            chat_id="telegram:123",
            execution_context={"depth": 0, "subagents_allowed": True},
        )

        projected = result["subagent"]
        self.assertEqual(projected["result"], "done")
        self.assertNotIn("tool_events", projected)
        self.assertNotIn("pending_inputs", projected)
        self.assertNotIn("todos", projected)
        self.assertNotIn("recovery_context", projected)

    async def test_targeted_cancel_stops_one_background_subagent(self) -> None:
        started = asyncio.Event()

        async def wait_forever(**_kwargs):
            started.set()
            await asyncio.Event().wait()

        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "depth": 1,
            "allow_subagents": False,
            "notify_completion": True,
            "completion_pending": False,
            "completion_attempts": 0,
            "status": "running",
        }
        with session.bound_chat("telegram:123"):
            session.subagent_records["worker-1"] = record
        with session.bound_chat("telegram:123"):
            with (
                patch.object(subagents, "run_subagent", wait_forever),
                patch.object(subagents, "save_state"),
                patch.object(subagents, "save_interrupted_subagent"),
            ):
                task = asyncio.create_task(
                    subagents.run_background_subagent(record)
                )
                session.subagent_tasks["worker-1"] = task
                await started.wait()

                result = await subagents.cancel_background_subagent("worker-1")

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(record["status"], "cancelled")
        self.assertFalse(record["completion_pending"])
        self.assertNotIn("worker-1", session.subagent_tasks)

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

    async def test_spawn_cannot_cross_chat_boundaries(self) -> None:
        with session.bound_chat("telegram:parent"):
            result = await subagents.subagent_tool(
                {"task": "inspect", "background": True},
                chat_id="responses_api:other",
                execution_context={"depth": 0},
            )

        self.assertFalse(result["ok"])
        self.assertIn("does not match", result["error"])

    def test_tool_event_recovery_history_is_bounded(self) -> None:
        events = [
            {
                "call": {
                    "name": "read_file",
                    "call_id": f"call-{index}",
                },
                "status": "completed",
                "result": {
                    "ok": True,
                    "content": "private" * 10_000,
                },
            }
            for index in range(5)
        ]
        with patch.object(subagents, "MAX_SUBAGENT_TOOL_EVENTS", 3):
            retained = subagents._bounded_tool_events(events)
        self.assertEqual(len(retained), 3)
        self.assertEqual(retained[0]["call"]["call_id"], "call-2")
        self.assertEqual(
            retained[0]["result"],
            {"ok": True, "status": None, "has_error": False},
        )
        self.assertNotIn("private", str(retained))

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

    async def test_subagent_task_text_is_bounded(self) -> None:
        with patch.object(subagents, "MAX_TOOL_OUTPUT", 10):
            result = await subagents.subagent_tool(
                {"task": "x" * 11},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )

        self.assertFalse(result["ok"])
        self.assertIn("exceeds 10 characters", result["error"])

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
            "chat_key": "telegram:123",
            "status": "completed",
            "completion_pending": True,
            "completion_attempts": 0,
        }
        with session.bound_chat("telegram:123"):
            session.subagent_records["worker-1"] = record
        delivered = AsyncMock(side_effect=RuntimeError("controller failed"))
        controller = types.ModuleType("controller")
        controller.enqueue_runtime_event = delivered

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
