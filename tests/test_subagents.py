import asyncio
import os
import sys
import tempfile
import types
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

_HOME = Path(tempfile.mkdtemp(prefix="raptor-subagent-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)

import controller
import session
import subagents
from response_errors import (
    IncompleteResponsesStreamError,
    MalformedToolCallError,
    PartialResponsesStreamError,
)


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

    async def test_foreground_request_classifies_malformed_tool_call(
        self,
    ) -> None:
        response = httpx.Response(
            500,
            json={
                "error": {
                    "message": (
                        "Failed to parse function call arguments as JSON: "
                        "syntax error"
                    )
                }
            },
        )
        client = AsyncMock()
        client.post.return_value = response

        with (
            patch.object(session, "responses", client, create=True),
            patch.object(subagents, "SUBAGENT_RESPONSES_MODEL", "model-a"),
        ):
            with self.assertRaises(MalformedToolCallError):
                await subagents._create_subagent_response_once(
                    [],
                    agent_id="worker-1",
                    allow_subagents=False,
                    depth=1,
                    tools=[],
                )

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

    async def test_public_subagent_stream_is_never_replayed(self) -> None:
        for callback_name in ("on_text", "on_reasoning_summary"):
            with self.subTest(callback=callback_name):
                attempts = 0

                async def interrupted(_items, **kwargs):
                    nonlocal attempts
                    attempts += 1
                    await kwargs[callback_name]("public output")
                    raise IncompleteResponsesStreamError("disconnected")

                callbacks = {callback_name: AsyncMock()}
                with (
                    patch.object(
                        subagents,
                        "_create_subagent_response_once",
                        interrupted,
                    ),
                    patch.object(subagents, "SUBAGENT_RESPONSES_MAX_RETRIES", 3),
                    patch.object(
                        subagents,
                        "SUBAGENT_RESPONSES_RETRY_BASE_SECONDS",
                        0,
                    ),
                ):
                    with self.assertRaises(PartialResponsesStreamError):
                        await subagents.create_subagent_response(
                            [],
                            agent_id="worker-1",
                            allow_subagents=False,
                            depth=1,
                            **callbacks,
                        )

                self.assertEqual(attempts, 1)

    async def test_background_runtime_projects_reasoning_and_reply(self) -> None:
        record = {
            "id": "worker-1",
            "session_id": "session-1",
            "background": True,
            "activity_surface_id": "telegram:42/77",
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

    async def test_failed_subagent_turn_records_terminal_outcome(self) -> None:
        record = {
            "id": "worker-1",
            "session_id": "session-1",
            "background": False,
            "pending_inputs": [],
            "tool_events": [],
            "todos": [],
        }
        session.subagent_records["worker-1"] = record

        async def malformed(*_args, **_kwargs):
            raise MalformedToolCallError("invalid generated arguments")

        appended: list[tuple[dict, str]] = []
        next_seq = 1

        def append(_sid, item, *, source):
            nonlocal next_seq
            next_seq += 1
            appended.append((item, source))
            return {"seq": next_seq, "item": item, "source": source}

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
            patch.object(
                subagents,
                "subagent_context_input_budget",
                return_value=0,
            ),
            patch.object(subagents, "create_subagent_response", malformed),
            patch.object(
                subagents,
                "append_item",
                side_effect=append,
            ),
            patch.object(subagents, "reset_model_context") as reset,
            patch.object(subagents, "save_state"),
        ):
            with self.assertRaises(MalformedToolCallError):
                await subagents.run_subagent(
                    agent_id="worker-1",
                    chat_id="telegram:123",
                    depth=1,
                    allow_subagents=False,
                )

        self.assertEqual(appended[-1][1], "assistant")
        self.assertEqual(appended[-1][0]["role"], "assistant")
        self.assertIn("MalformedToolCallError", str(appended[-1][0]))
        reset.assert_called_once()

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

    async def test_background_start_declares_automatic_completion(self) -> None:
        started = asyncio.Event()

        async def wait_forever(_record):
            started.set()
            await asyncio.Event().wait()

        with (
            patch.object(subagents, "open_subagent_activity", AsyncMock()),
            patch.object(subagents, "run_background_subagent", wait_forever),
            patch.object(subagents, "save_state"),
        ):
            result = await subagents.subagent_tool(
                {"task": "inspect", "background": True},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )
            await started.wait()
            task = session.subagent_tasks[result["agent_id"]]
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            session.subagent_tasks.pop(result["agent_id"], None)

        self.assertEqual(result["status"], "running")
        self.assertEqual(result["completion_notification"], "automatic")

    def test_completion_prompt_is_single_shot(self) -> None:
        prompt = subagents.completion_prompt(
            {
                "id": "worker-1",
                "task": "inspect",
                "status": "completed",
                "result": "done",
            }
        )

        self.assertIn("exactly once", prompt)
        self.assertIn("do not poll", prompt)
        self.assertIn("or start a wait command", prompt)

    async def test_continuation_waits_for_prior_generation_to_finalize(self) -> None:
        closing = asyncio.Event()
        release = asyncio.Event()
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "depth": 1,
            "allow_subagents": False,
            "notify_completion": False,
            "status": "running",
            "run_generation": 1,
        }

        async def finish(*_args, **_kwargs):
            closing.set()
            await release.wait()

        with (
            patch.object(subagents, "run_subagent", AsyncMock(return_value="done")),
            patch.object(subagents, "finish_subagent_activity", finish),
            patch.object(subagents, "save_state"),
        ):
            task = asyncio.create_task(subagents.run_background_subagent(record))
            session.subagent_records["worker-1"] = record
            session.subagent_tasks["worker-1"] = task
            await closing.wait()

            result = await subagents.subagent_tool(
                {"agent_id": "worker-1", "task": "continue", "background": True},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )

            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "finalizing")
            release.set()
            await task

        self.assertNotIn("worker-1", session.subagent_tasks)

    async def test_parent_notification_does_not_wait_for_topic_finalization(
        self,
    ) -> None:
        finalizing = asyncio.Event()
        release = asyncio.Event()
        completion = asyncio.get_running_loop().create_future()
        enqueue = Mock(return_value=completion)
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "depth": 1,
            "allow_subagents": False,
            "notify_completion": True,
            "completion_pending": False,
            "status": "running",
            "run_generation": 1,
        }

        async def finish(*_args, **_kwargs):
            finalizing.set()
            await release.wait()

        with (
            patch.object(
                subagents,
                "run_subagent",
                AsyncMock(return_value="done"),
            ),
            patch.object(subagents, "finish_subagent_activity", finish),
            patch.object(subagents, "save_state"),
            patch.object(controller, "enqueue_runtime_event", enqueue),
        ):
            task = asyncio.create_task(subagents.run_background_subagent(record))
            session.subagent_records["worker-1"] = record
            session.subagent_tasks["worker-1"] = task
            await finalizing.wait()

            enqueue.assert_called_once()
            self.assertTrue(record["completion_pending"])

            release.set()
            await task

        self.assertNotIn("worker-1", session.subagent_tasks)

    async def test_background_completion_joins_running_parent_end_to_end(
        self,
    ) -> None:
        finalizing = asyncio.Event()
        release = asyncio.Event()
        parent_running = asyncio.Event()
        release_parent = asyncio.Event()
        parent_inputs: list[str] = []
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "depth": 1,
            "allow_subagents": False,
            "notify_completion": True,
            "completion_pending": False,
            "status": "running",
            "run_generation": 1,
        }

        async def finish(*_args, **_kwargs):
            finalizing.set()
            await release.wait()

        async def parent_turn(_chat_id, text, **_kwargs):
            parent_inputs.append(text)
            if text == "parent request":
                parent_running.set()
                await release_parent.wait()
            return True

        with (
            patch.object(
                subagents,
                "run_subagent",
                AsyncMock(return_value="Tatooine"),
            ),
            patch.object(subagents, "finish_subagent_activity", finish),
            patch.object(subagents, "save_state"),
            patch.object(controller, "agent_turn", parent_turn),
            patch.object(
                controller,
                "flush_pending_delivery",
                AsyncMock(return_value=True),
            ),
            patch.object(controller, "ensure_goal_pin", AsyncMock()),
            patch.object(controller, "sync_goal_pin", AsyncMock()),
        ):
            parent_task = controller.start_root_session(
                "telegram:123",
                "parent request",
            )
            await parent_running.wait()
            child_task = asyncio.create_task(
                subagents.run_background_subagent(record)
            )
            session.subagent_records["worker-1"] = record
            session.subagent_tasks["worker-1"] = child_task
            try:
                await finalizing.wait()
                release_parent.set()
                await asyncio.wait_for(
                    _wait_until(lambda: not record["completion_pending"]),
                    timeout=1,
                )
                await parent_task
            finally:
                release_parent.set()
                release.set()
                await child_task

        self.assertEqual(len(parent_inputs), 2)
        self.assertEqual(parent_inputs[0], "parent request")
        self.assertIn("Tatooine", parent_inputs[1])
        self.assertEqual(record["status"], "completed")

    async def test_running_steer_is_appended_to_activity_surface(self) -> None:
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "status": "running",
            "task_count": 1,
            "pending_inputs": [],
            "activity_surface_id": "telegram:42/77",
        }
        session.subagent_records["worker-1"] = record
        append_input = AsyncMock()
        with (
            patch.object(
                subagents,
                "append_subagent_activity_input",
                append_input,
            ),
            patch.object(subagents, "save_state"),
        ):
            result = await subagents.subagent_tool(
                {"agent_id": "worker-1", "task": "check the logs"},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "steering_queued")
        append_input.assert_awaited_once_with(record, "check the logs")

    async def test_surfaced_subagent_projects_foreground_continuation(
        self,
    ) -> None:
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "session_id": "subagent-session",
            "status": "completed",
            "task_count": 1,
            "pending_inputs": [],
            "allow_subagents": False,
            "run_generation": 1,
            "activity_surface_id": "telegram:42/77",
        }
        session.subagent_records["worker-1"] = record
        finish = AsyncMock(return_value=True)
        open_surface = AsyncMock()
        with (
            patch.object(subagents, "finish_subagent_activity", finish),
            patch.object(subagents, "open_subagent_activity", open_surface),
            patch.object(subagents, "run_subagent", AsyncMock(return_value="done")),
            patch.object(subagents, "append_item"),
            patch.object(subagents, "save_state"),
            patch.object(subagents, "prune_subagent_records"),
        ):
            result = await subagents.subagent_tool(
                {"agent_id": "worker-1", "task": "continue"},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )

        self.assertTrue(result["ok"])
        self.assertFalse(record["background"])
        self.assertEqual(record["run_generation"], 2)
        open_surface.assert_awaited_once_with(record)
        self.assertEqual(finish.await_count, 2)
        self.assertEqual(
            finish.await_args_list[-1].kwargs["expected_generation"],
            2,
        )

    async def test_continuation_hands_pending_completion_to_parent(self) -> None:
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "session_id": "subagent-session",
            "status": "completed",
            "result": "first result",
            "task_count": 1,
            "pending_inputs": [],
            "allow_subagents": False,
            "run_generation": 1,
            "completion_pending": True,
        }
        session.subagent_records["worker-1"] = record
        with (
            patch.object(
                subagents,
                "finish_subagent_activity",
                AsyncMock(return_value=True),
            ),
            patch.object(
                subagents,
                "run_subagent",
                AsyncMock(return_value="second result"),
            ),
            patch.object(subagents, "append_item"),
            patch.object(subagents, "save_state"),
            patch.object(subagents, "prune_subagent_records"),
        ):
            result = await subagents.subagent_tool(
                {"agent_id": "worker-1", "task": "continue"},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )

        self.assertEqual(
            result["previous_completion"]["result"],
            "first result",
        )
        self.assertEqual(result["result"], "second result")

    async def test_activity_failure_does_not_block_continuation(
        self,
    ) -> None:
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "session_id": "subagent-session",
            "status": "completed",
            "task_count": 1,
            "pending_inputs": [],
            "allow_subagents": False,
            "run_generation": 1,
            "activity_surface_id": "telegram:42/77",
        }
        session.subagent_records["worker-1"] = record
        with (
            patch.object(
                subagents,
                "finish_subagent_activity",
                AsyncMock(return_value=False),
            ),
            patch.object(
                subagents,
                "run_subagent",
                AsyncMock(return_value="continued"),
            ),
            patch.object(
                subagents,
                "open_subagent_activity",
                AsyncMock(),
            ),
            patch.object(subagents, "append_item") as append_item,
            patch.object(subagents, "save_state"),
            patch.object(subagents, "prune_subagent_records"),
        ):
            result = await subagents.subagent_tool(
                {"agent_id": "worker-1", "task": "continue"},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], "continued")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["run_generation"], 2)
        append_item.assert_called()

    async def test_delete_removes_terminal_subagent_record(self) -> None:
        record = {
            "id": "worker-1",
            "status": "completed",
            "activity_surface_id": "telegram:42/77",
        }
        session.subagent_records["worker-1"] = record
        session.state["interrupted_subagents"] = [{"id": "worker-1"}]
        with (
            patch.object(
                subagents,
                "delete_subagent_activity",
                AsyncMock(return_value=True),
            ),
            patch.object(subagents, "save_state"),
        ):
            result = await subagents.subagent_tool(
                {"agent_id": "worker-1", "delete": True},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "deleted")
        self.assertNotIn("worker-1", session.subagent_records)
        self.assertEqual(session.state["interrupted_subagents"], [])

    async def test_delete_waits_for_pending_completion_delivery(self) -> None:
        record = {
            "id": "worker-1",
            "status": "completed",
            "completion_pending": True,
            "activity_surface_id": "telegram:42/77",
        }
        session.subagent_records["worker-1"] = record
        delete_surface = AsyncMock(return_value=True)
        with patch.object(
            subagents,
            "delete_subagent_activity",
            delete_surface,
        ):
            result = await subagents.subagent_tool(
                {"agent_id": "worker-1", "delete": True},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "completion_pending")
        self.assertIs(session.subagent_records["worker-1"], record)
        delete_surface.assert_not_awaited()

    async def test_parent_completion_does_not_depend_on_activity_finish(
        self,
    ) -> None:
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "depth": 1,
            "allow_subagents": False,
            "notify_completion": True,
            "run_generation": 1,
        }
        order: list[str] = []
        completion = asyncio.get_running_loop().create_future()

        def enqueue(*_args, **_kwargs):
            order.append("completion")
            return completion

        async def finish(_record, *, expected_generation=None):
            self.assertEqual(expected_generation, 1)
            order.append("activity")
            return True

        with (
            patch.object(subagents, "run_subagent", AsyncMock(return_value="done")),
            patch.object(subagents, "finish_subagent_activity", finish),
            patch.object(subagents, "save_state"),
            patch.object(subagents, "_queue_subagent_completion") as queued,
        ):
            queued.side_effect = lambda _record: order.append("completion") or True
            await subagents.run_background_subagent(record)

        self.assertEqual(order, ["completion", "activity"])

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
            "activity_surface_id": "telegram:42/77",
            "activity_result_delivered": True,
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
        self.assertEqual(
            record["activity_surface_id"],
            "telegram:42/77",
        )
        self.assertFalse(record["activity_result_delivered"])
        self.assertNotIn("tasks", record)

    def test_continuation_preserves_queued_inputs_in_transcript_order(self) -> None:
        record = {
            "session_id": "subagent-session",
            "status": "interrupted",
            "task_count": 1,
            "pending_inputs": ["queued one", "queued two"],
        }
        appended: list[tuple[str, str]] = []

        def append(_session_id, item, *, source):
            appended.append((source, str(item["content"])))

        with (
            patch.object(subagents, "append_item", side_effect=append),
            patch.object(subagents, "save_state"),
        ):
            subagents.continue_record(
                record,
                "new delegation",
                chat_id="telegram:123",
                depth=1,
                background=False,
                allow_subagents=False,
            )

        self.assertEqual(
            appended,
            [
                ("steer", "queued one"),
                ("steer", "queued two"),
                ("delegation", "new delegation"),
            ],
        )
        self.assertEqual(record["pending_inputs"], [])
        self.assertEqual(record["run_generation"], 2)

    async def test_nested_subagent_records_immediate_and_root_parent(self) -> None:
        with (
            patch.object(subagents, "run_subagent", AsyncMock(return_value="done")),
            patch.object(subagents, "save_state"),
            patch.object(subagents, "prune_subagent_records"),
        ):
            result = await subagents.subagent_tool(
                {"task": "nested"},
                chat_id="telegram:123",
                execution_context={
                    "depth": 1,
                    "subagents_allowed": True,
                    "session_id": "20260824-120000-11111111",
                    "root_session_id": "20260824-120000-22222222",
                },
            )

        record = session.subagent_records[result["agent_id"]]
        self.assertEqual(
            record["parent_session_id"],
            "20260824-120000-11111111",
        )
        self.assertEqual(record["root_session_id"], "20260824-120000-22222222")

    async def test_stale_completion_generation_cannot_clear_current_run(
        self,
    ) -> None:
        live = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "status": "completed",
            "completion_pending": True,
            "run_generation": 2,
            "task_count": 2,
        }
        session.subagent_records["worker-1"] = live
        stale = dict(live, run_generation=1)
        controller = types.ModuleType("controller")
        controller.enqueue_runtime_event = Mock()

        with patch.dict(sys.modules, {"controller": controller}):
            queued = subagents._queue_subagent_completion(stale)

        self.assertFalse(queued)
        controller.enqueue_runtime_event.assert_not_called()
        self.assertTrue(live["completion_pending"])

    async def test_task_count_change_does_not_suppress_completion(
        self,
    ) -> None:
        live = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "status": "completed",
            "completion_pending": True,
            "run_generation": 2,
            "task_count": 3,
        }
        session.subagent_records["worker-1"] = live
        stale = dict(live, task_count=2)
        controller = types.ModuleType("controller")
        completion = asyncio.get_running_loop().create_future()
        completion.set_result(True)
        controller.enqueue_runtime_event = Mock(return_value=completion)

        with patch.dict(sys.modules, {"controller": controller}):
            queued = subagents._queue_subagent_completion(stale)
            await asyncio.sleep(0)

        self.assertTrue(queued)
        controller.enqueue_runtime_event.assert_called_once()
        self.assertFalse(live["completion_pending"])

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
            "topic": {
                "status": "completed",
                "completed_at": 5,
                "activity_surface_id": "42/77",
            },
        }
        with patch.object(session, "MAX_SUBAGENT_RECORDS", 1):
            removed = session._prune_subagent_mapping(records, [])
        self.assertEqual(removed, 1)
        self.assertEqual(
            set(records),
            {"new", "running", "pending", "topic"},
        )

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
        record_count = len(session.subagent_records)
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
        self.assertEqual(len(session.subagent_records), record_count)

    async def test_rejected_background_continuation_preserves_record(self) -> None:
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "session_id": "subagent-session",
            "status": "completed",
            "task_count": 1,
            "pending_inputs": [],
            "allow_subagents": False,
            "run_generation": 1,
        }
        session.subagent_records["worker-1"] = record
        session.subagent_tasks["existing"] = AsyncMock()
        try:
            with (
                patch.object(subagents, "MAX_BACKGROUND_SUBAGENTS", 1),
                patch.object(
                    subagents,
                    "finish_subagent_activity",
                    AsyncMock(return_value=True),
                ),
                patch.object(subagents, "append_item") as append_item,
            ):
                result = await subagents.subagent_tool(
                    {
                        "agent_id": "worker-1",
                        "task": "continue",
                        "background": True,
                    },
                    chat_id="telegram:123",
                    execution_context={"depth": 0},
                )
        finally:
            session.subagent_tasks.pop("existing", None)

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "capacity_reached")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["task_count"], 1)
        self.assertEqual(record["run_generation"], 1)
        append_item.assert_not_called()

    async def test_persistent_topic_capacity_is_bounded(self) -> None:
        session.subagent_records["existing"] = {
            "activity_surface_id": "42/77",
        }
        with patch.object(subagents, "MAX_SUBAGENT_RECORDS", 1):
            result = await subagents.subagent_tool(
                {"task": "another", "background": True},
                chat_id="telegram:123",
                execution_context={"depth": 0},
            )

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
        delivered = Mock(side_effect=RuntimeError("controller failed"))
        controller = types.ModuleType("controller")
        controller.enqueue_runtime_event = delivered

        with (
            patch.dict(sys.modules, {"controller": controller}),
            patch.object(subagents, "save_state"),
            patch.object(subagents, "log_event") as logged,
        ):
            queued = subagents._queue_subagent_completion(record)

        self.assertFalse(queued)
        self.assertEqual(record["completion_attempts"], 1)
        self.assertTrue(record["completion_pending"])
        self.assertEqual(
            [call.args[1] for call in logged.call_args_list],
            ["completion_delivery_error"],
        )

    async def test_deferred_completion_requeues_on_user_activity(self) -> None:
        record = {
            "id": "worker-1",
            "chat_id": "telegram:123",
            "chat_key": "telegram:123",
            "run_generation": 1,
            "completion_pending": True,
            "completion_attempts": 1,
        }
        session.subagent_records["worker-1"] = record
        completion = asyncio.get_running_loop().create_future()
        controller = types.ModuleType("controller")
        controller.enqueue_runtime_event = Mock(return_value=completion)

        with (
            patch.dict(sys.modules, {"controller": controller}),
            patch.object(subagents, "save_state") as saved,
        ):
            count = await subagents.requeue_deferred_subagent_completions()

        self.assertEqual(count, 1)
        self.assertEqual(record["completion_attempts"], 0)
        controller.enqueue_runtime_event.assert_called_once()
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


async def _wait_until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0)


if __name__ == "__main__":
    unittest.main()
