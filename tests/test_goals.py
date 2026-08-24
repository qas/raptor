import asyncio
import copy
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

_ROOT = Path(__file__).resolve().parent.parent
_STATE_DIR = Path(tempfile.mkdtemp(prefix="raptor-goal-tests-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_STATE_DIR)
os.environ["AGENT_WORKDIR"] = str(_STATE_DIR)
os.environ.setdefault("MODEL_CONTEXT_TOKENS", "131072")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import agent as agent_mod
import controller
import session
from chat_provider import ProviderCapabilities
from chat_runtime import set_chat_provider
from goals import (
    GOAL_ACTIVE,
    GOAL_BLOCKED,
    GOAL_COMPLETE,
    GOAL_PAUSED,
    MAX_GOAL_CHARS,
    block_goal,
    clear_goal,
    complete_goal,
    create_goal,
    current_goal,
    ensure_goal_pin,
    goal_pin_text,
    goal_continuation_input,
    goal_instructions,
    goal_is_active,
    pause_goal,
    prepare_goal_on_startup,
    remove_goal_pin,
    replace_goal,
    resume_goal,
    set_goal_tool,
    suspend_goal_pin,
    sync_goal_pin,
    todo_store_for_display,
    todo_store_for_execution,
    update_goal_tool,
)
from runtime_events import RuntimeEvent, RuntimeEventKind
from turn_runtime import TurnKind, turns


async def _noop(*_a, **_k):
    return None


def _runtime_event(text: str, *, is_active=None) -> RuntimeEvent:
    return RuntimeEvent(
        conversation_id=1,
        kind=RuntimeEventKind.SUBAGENT_COMPLETED,
        content=text,
        done=asyncio.get_running_loop().create_future(),
        is_active=is_active,
    )


class GoalTestProvider:
    """Minimal status-slot fake for goal tests outside adapter coverage."""

    name = "goal_test"
    authorized_user_id = "operator"
    primary_conversation_id = 1
    capabilities = ProviderCapabilities(pins=True, controls=True)

    def __init__(self) -> None:
        self.next_message_id = 0

    async def create_message(self, *_args, **_kwargs) -> str:
        self.next_message_id += 1
        return f"message-{self.next_message_id}"

    async def edit_message(self, *_args, **_kwargs) -> None:
        return None

    async def delete_message(self, *_args, **_kwargs) -> None:
        return None

    async def pin_message(self, *_args, **_kwargs) -> None:
        return None

    async def unpin_message(self, *_args, **_kwargs) -> None:
        return None


def _bind_goal_test_provider(test: unittest.TestCase) -> None:
    previous = set_chat_provider(GoalTestProvider())
    test.addCleanup(set_chat_provider, previous)


class GoalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _bind_goal_test_provider(self)
        session.state.clear()
        session.state.update(
            copy.deepcopy(session.DEFAULT_STATE)
        )
        session.subagent_records.clear()
        session.subagent_records.update(
            session.state["subagents"]
        )
        turns.finish()
        session.current_runtime().goal_pin_message_id = None
        session.current_runtime().goal_pin_goal_id = None
        session.current_runtime().pinned_status_conversation_id = None
        session.current_runtime().pinned_status_message_id = None
        session.current_runtime().pinned_status_owner = None
        session.current_runtime().goal_creation_authorized = False
        session.pending_approvals.clear()
        while True:
            try:
                session.steer_queue.get_nowait()
                session.steer_queue.task_done()
            except asyncio.QueueEmpty:
                break
        while True:
            try:
                session.runtime_event_queue.get_nowait()
                session.runtime_event_queue.task_done()
            except asyncio.QueueEmpty:
                break
        session.pending_steers.clear()
        from chat_store import create_session
        import chat_store
        self._chat_dir = Path(tempfile.mkdtemp(prefix="chats-"))
        self._chat_patch = patch.object(
            chat_store,
            "CHAT_DIR",
            self._chat_dir,
        )
        self._chat_patch.start()
        self.addCleanup(self._chat_patch.stop)
        chat_store._SEQ_CACHE.clear()
        session.state["current_session_id"] = create_session(
            kind="main",
            chat_key=session.current_runtime().key,
        )
        session.state["pending_inputs"] = []

    def test_goal_create_persists(self) -> None:
        goal = replace_goal("Ship the feature")
        self.assertEqual(
            session.state["goal"]["id"],
            goal["id"],
        )
        self.assertEqual(
            current_goal()["objective"],
            "Ship the feature",
        )
        self.assertEqual(
            current_goal()["status"],
            GOAL_ACTIVE,
        )

    def test_goal_owns_a_fresh_isolated_checklist(self) -> None:
        session.state["todos"] = [
            {"step": "Old session work", "status": "pending"},
        ]
        first = replace_goal("First goal")
        first["todos"] = [
            {"step": "First step", "status": "in_progress"},
        ]
        second = replace_goal("Second goal")
        self.assertEqual(session.state["todos"], [])
        self.assertEqual(second["todos"], [])
        self.assertIs(todo_store_for_execution(), second)
        self.assertIs(todo_store_for_display(), second)

    def test_goal_prompt_rehydrates_current_checklist(self) -> None:
        goal = replace_goal("Ship safely")
        goal["todos"] = [
            {"step": "Verify tests", "status": "in_progress"},
        ]
        prompt = goal_instructions()
        self.assertIn("Current execution checklist", prompt)
        self.assertIn("[>] Verify tests", prompt)

    def test_paused_goal_plan_displays_but_cannot_be_overwritten(self) -> None:
        goal = replace_goal("Paused work")
        goal["todos"] = [
            {"step": "Resume later", "status": "pending"},
        ]
        pause_goal()
        self.assertIs(todo_store_for_display(), goal)
        self.assertIs(todo_store_for_execution(), session.state)

    def test_goal_has_unique_id(self) -> None:
        first = create_goal("one")
        second = create_goal("two")
        self.assertNotEqual(
            first["id"],
            second["id"],
        )

    def test_goal_objective_length_bounded(self) -> None:
        with self.assertRaises(ValueError):
            create_goal("x" * (MAX_GOAL_CHARS + 1))
        with self.assertRaises(ValueError):
            create_goal("   ")

    def test_goal_prompt_only_for_active_goal(self) -> None:
        self.assertEqual(goal_instructions(), "")
        replace_goal("Do the thing")
        self.assertIn(
            "Do the thing",
            goal_instructions(),
        )
        pause_goal()
        self.assertEqual(goal_instructions(), "")
        resume_goal()
        self.assertIn(
            "Do the thing",
            goal_instructions(),
        )
        complete_goal(current_goal()["id"])
        self.assertEqual(goal_instructions(), "")

    def test_goal_prompt_not_written_to_transcript_as_user_item(self) -> None:
        replace_goal("Keep me out of history")
        prompt = goal_instructions()
        self.assertTrue(prompt)
        from chat_store import item_events
        session_id = session.state["current_session_id"]
        for event in item_events(session_id):
            item = event.get("item") or {}
            content = item.get("content")
            if isinstance(content, str):
                self.assertNotIn(prompt, content)

    async def test_goal_survives_compaction(self) -> None:
        replace_goal("Survive compaction")
        goal_before = copy.deepcopy(current_goal())
        from chat_store import append_item
        session_id = session.state["current_session_id"]
        append_item(
            session_id,
            {"role": "user", "content": "hi"},
            source="user",
        )

        async def fake_compact(*_a, **_k):
            return True

        with (
            patch(
                "agent.compact_session",
                fake_compact,
            ),
            patch.object(
                agent_mod,
                "save_state",
                lambda: None,
            ),
            patch.object(
                agent_mod,
                "send",
                _noop,
            ),
            patch.object(
                agent_mod,
                "typing_loop",
                _noop,
            ),
            patch(
                "agent.session_context_stats",
                lambda _sid: {
                    "archive_events": 2,
                    "checkpoint": True,
                    "checkpoint_through": 1,
                    "active_native_events": 1,
                },
            ),
        ):
            await agent_mod.compact_context(
                1,
            )
        self.assertEqual(
            current_goal(),
            goal_before,
        )

    def test_stale_goal_update_rejected(self) -> None:
        replace_goal("first")
        stale_id = current_goal()["id"]
        replace_goal("second")
        result = update_goal_tool(
            {
                "goal_id": stale_id,
                "status": "complete",
            }
        )
        self.assertFalse(result["ok"])
        self.assertIn("stale", result["error"])
        self.assertEqual(
            current_goal()["status"],
            GOAL_ACTIVE,
        )
        self.assertEqual(
            current_goal()["objective"],
            "second",
        )

    def test_old_turn_cannot_complete_replaced_goal(self) -> None:
        replace_goal("AAA")
        old_id = current_goal()["id"]
        clear_goal()
        replace_goal("BBB")
        self.assertFalse(
            complete_goal(old_id)
        )
        self.assertEqual(
            current_goal()["objective"],
            "BBB",
        )
        self.assertEqual(
            current_goal()["status"],
            GOAL_ACTIVE,
        )

    def test_model_can_complete_goal(self) -> None:
        replace_goal("finish me")
        result = update_goal_tool(
            {
                "goal_id": current_goal()["id"],
                "status": "complete",
            }
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            current_goal()["status"],
            GOAL_COMPLETE,
        )
        self.assertFalse(goal_is_active())

    def test_model_can_update_goal_objective_in_place(self) -> None:
        replace_goal("initial objective")
        goal_id = current_goal()["id"]

        result = update_goal_tool(
            {
                "goal_id": goal_id,
                "objective": "  revised   objective  ",
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(current_goal()["id"], goal_id)
        self.assertEqual(current_goal()["status"], GOAL_ACTIVE)
        self.assertEqual(current_goal()["objective"], "revised objective")

    def test_model_cannot_update_goal_after_user_pauses_it(self) -> None:
        replace_goal("paused objective")
        goal_id = current_goal()["id"]
        pause_goal()

        result = update_goal_tool(
            {
                "goal_id": goal_id,
                "status": "blocked",
                "reason": "late model result",
            }
        )

        self.assertFalse(result["ok"])
        self.assertIn("only an active goal", result["error"])
        self.assertEqual(current_goal()["status"], GOAL_PAUSED)

    def test_model_goal_update_requires_a_change(self) -> None:
        replace_goal("initial objective")

        result = update_goal_tool({"goal_id": current_goal()["id"]})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "objective or status is required")

    def test_model_can_block_goal(self) -> None:
        replace_goal("needs help")
        result = update_goal_tool(
            {
                "goal_id": current_goal()["id"],
                "status": "blocked",
                "reason": "Need credentials",
            }
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            current_goal()["status"],
            GOAL_BLOCKED,
        )
        self.assertEqual(
            current_goal()["blocked_reason"],
            "Need credentials",
        )

    def test_model_cannot_pause_goal(self) -> None:
        replace_goal("do not pause via model")
        result = update_goal_tool(
            {
                "goal_id": current_goal()["id"],
                "status": "paused",
            }
        )
        self.assertFalse(result["ok"])
        self.assertEqual(
            current_goal()["status"],
            GOAL_ACTIVE,
        )

    def test_model_cannot_replace_goal(self) -> None:
        replace_goal("original")
        result = update_goal_tool(
            {
                "goal_id": current_goal()["id"],
                "status": "active",
                "reason": "nope",
            }
        )
        self.assertFalse(result["ok"])
        self.assertEqual(
            current_goal()["objective"],
            "original",
        )

    async def test_active_goal_continues_after_turn(self) -> None:
        replace_goal("keep going")
        calls: list[str] = []

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            calls.append(text)
            if len(calls) >= 2:
                complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(
                1,
                "start",
            )
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(calls[0], "start")
        self.assertEqual(
            calls[1],
            goal_continuation_input(),
        )

    async def test_queued_steer_restores_origin_delivery_context(self) -> None:
        seen: list[tuple[str, str, object]] = []

        @contextmanager
        def capture_context(chat_id, delivery_context):
            seen.append(("context", str(chat_id), delivery_context))
            yield

        async def fake_turn(chat_id, text, **_kwargs):
            seen.append(("turn", str(chat_id), text))
            return True

        await session.steer_queue.put(
            {
                "id": "steer-1",
                "chat_id": "responses_api:web",
                "text": "new direction",
                "status": "queued",
                "message_id": None,
                "delivery_context": ("responses_api", "request-2"),
            }
        )
        with (
            patch.object(controller, "agent_turn", fake_turn),
            patch.object(controller, "bound_delivery_context", capture_context),
            patch.object(controller, "clear_steering_indicator", _noop),
        ):
            await controller.run_root_session("telegram:123", None)

        self.assertEqual(
            seen,
            [
                (
                    "context",
                    "responses_api:web",
                    ("responses_api", "request-2"),
                ),
                ("turn", "responses_api:web", "new direction"),
            ],
        )

    async def test_complete_goal_does_not_continue(self) -> None:
        replace_goal("done soon")
        calls: list[str] = []

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            calls.append(text)
            complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(
                1,
                "start",
            )
        self.assertEqual(calls, ["start"])

    async def test_blocked_goal_does_not_continue(self) -> None:
        replace_goal("block soon")
        calls: list[str] = []

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            calls.append(text)
            block_goal(
                current_goal()["id"],
                "waiting",
            )
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(
                1,
                "start",
            )
        self.assertEqual(calls, ["start"])

    async def test_paused_goal_does_not_continue(self) -> None:
        replace_goal("pause soon")
        calls: list[str] = []

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            calls.append(text)
            pause_goal()
            return True

        with patch.object(
            controller,
            "agent_turn",
            fake_turn,
        ):
            await controller.run_root_session(
                1,
                "start",
            )
        self.assertEqual(calls, ["start"])

    async def test_user_input_beats_goal_continuation(
        self,
    ) -> None:
        replace_goal("background goal")
        calls: list[str] = []

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            calls.append(text)
            if len(calls) == 1:
                await session.steer_queue.put(
                    {
                        "id": "s1",
                        "chat_id": chat_id,
                        "text": "user steer",
                        "status": "queued",
                    }
                )
                session.pending_steers["s1"] = {
                    "id": "s1",
                    "chat_id": chat_id,
                    "text": "user steer",
                    "status": "queued",
                }
            if len(calls) >= 2:
                complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "clear_steering_indicator",
                _noop,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(
                1,
                "start",
            )
        self.assertEqual(calls[1], "user steer")

    async def test_internal_event_beats_goal_continuation(
        self,
    ) -> None:
        replace_goal("background goal")
        calls: list[str] = []

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            calls.append(text)
            if len(calls) == 1:
                done = asyncio.get_running_loop().create_future()
                await session.runtime_event_queue.put(
                    RuntimeEvent(
                        conversation_id=chat_id,
                        kind=RuntimeEventKind.SUBAGENT_COMPLETED,
                        content="subagent done",
                        done=done,
                    )
                )
            if len(calls) >= 2:
                complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(
                1,
                "start",
            )
        self.assertIn("subagent done", calls[1])

    async def test_stop_pauses_active_goal(self) -> None:
        replace_goal("stop me")
        from commands import command

        cancel_shells = AsyncMock(return_value=2)
        with (
            patch(
                "commands.cancel_background_subagents",
                AsyncMock(return_value=0),
            ),
            patch(
                "commands.cancel_shell_sessions",
                cancel_shells,
            ),
        ):
            with patch(
                "commands.send",
                _noop,
            ):
                await command(1, "/stop all")
        self.assertEqual(
            current_goal()["status"],
            GOAL_PAUSED,
        )
        cancel_shells.assert_awaited_once_with()

    async def test_stop_preserves_background_resources(self) -> None:
        from commands import command

        cancel_subagents = AsyncMock(return_value=1)
        cancel_shells = AsyncMock(return_value=1)
        with (
            patch(
                "commands.cancel_background_subagents",
                cancel_subagents,
            ),
            patch("commands.cancel_shell_sessions", cancel_shells),
            patch("commands.send", _noop),
        ):
            await command(1, "/stop")

        cancel_subagents.assert_not_awaited()
        cancel_shells.assert_not_awaited()

    async def test_stop_reports_discarded_background_results(self) -> None:
        from commands import command

        sent: list[str] = []

        async def capture(_chat_id, text, **_kw):
            sent.append(text)

        with (
            patch("commands.pending_subagent_completions", return_value=1),
            patch("commands.pending_shell_completions", return_value=2),
            patch(
                "commands.cancel_background_subagents",
                AsyncMock(return_value=0),
            ) as cancel_subagents,
            patch(
                "commands.cancel_shell_sessions",
                AsyncMock(return_value=0),
            ),
            patch("commands.send", capture),
        ):
            await command(1, "/stop all")

        cancel_subagents.assert_awaited_once_with(discard_pending=True)
        self.assertEqual(
            sent,
            ["Stopped.\nDiscarded pending background results: 3"],
        )

    async def test_resume_restarts_goal_controller(
        self,
    ) -> None:
        replace_goal("resume me")
        pause_goal()
        started: list[tuple] = []

        def fake_start(chat_id, text, *, internal=False):
            started.append((chat_id, text, internal))
            task = asyncio.get_running_loop().create_future()
            task.set_result(None)

            class Done:
                def done(self):
                    return True

            return Done()

        from commands import command

        with (
            patch(
                "commands.start_root_session",
                fake_start,
            ),
            patch(
                "commands.send",
                _noop,
            ),
        ):
            await command(1, "/goal resume")
        self.assertEqual(
            current_goal()["status"],
            GOAL_ACTIVE,
        )
        self.assertEqual(len(started), 1)

    async def test_clear_stops_continuation(self) -> None:
        replace_goal("clear me")
        goal_id = current_goal()["id"]

        task = turns.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
            goal_id=goal_id,
        )
        from commands import command

        with patch("commands.send", _noop):
            await command(1, "/goal clear")
        self.assertIsNone(current_goal())
        self.assertTrue(task.cancelled())

    async def test_only_one_root_controller_runs(self) -> None:
        replace_goal("one controller")
        gate = asyncio.Event()
        started = asyncio.Event()
        running = {"n": 0}

        async def slow_turn(chat_id, text, *, internal=False, **_kw):
            running["n"] += 1
            started.set()
            await gate.wait()
            complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                slow_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            first = controller.start_root_session(
                1,
                "a",
            )
            await started.wait()
            second = controller.ensure_root_session(
                1,
                "b",
            )
            self.assertIs(first, second)
            self.assertEqual(running["n"], 1)
            gate.set()
            await first
            self.assertEqual(running["n"], 1)

    def test_root_goal_not_copied_to_subagent(self) -> None:
        replace_goal("root only")
        from subagents import (
            build_subagent_payload,
            subagent_tools,
        )
        tools = subagent_tools(False, 1)
        names = {tool.get("name") for tool in tools}
        self.assertNotIn("get_goal", names)
        self.assertNotIn("update_goal", names)
        self.assertNotIn("set_goal", names)
        self.assertNotIn("cancel", names)
        session.state["model"] = "main-model"
        with patch(
            "subagents.SUBAGENT_RESPONSES_MODEL",
            "subagent-model",
        ):
            payload = build_subagent_payload(
                [{"role": "user", "content": "task"}],
                allow_subagents=False,
                depth=1,
            )
        self.assertEqual(payload["model"], "subagent-model")
        self.assertNotIn(
            "Active persistent goal",
            payload["instructions"],
        )
        self.assertNotIn(
            goal_instructions(),
            payload["instructions"],
        )

    async def test_subagent_completion_returns_to_root_goal(
        self,
    ) -> None:
        replace_goal("parent goal")
        calls: list[str] = []

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            calls.append(text)
            if "background subagent" in text:
                complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            delivered = await controller.enqueue_runtime_event(
                session.current_runtime().conversation_id,
                RuntimeEventKind.SUBAGENT_COMPLETED,
                "A background subagent has finished.",
            )
        self.assertTrue(delivered)
        self.assertEqual(len(calls), 1)
        self.assertIn(
            "background subagent",
            calls[0],
        )

    async def test_failed_subagent_does_not_pause_parent_goal(self) -> None:
        replace_goal("recover from child failure")
        calls: list[str] = []

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            calls.append(text)
            if len(calls) == 2:
                complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(controller, "agent_turn", fake_turn),
            patch.object(controller, "send", _noop),
        ):
            delivered = await controller.enqueue_runtime_event(
                session.current_runtime().conversation_id,
                RuntimeEventKind.SUBAGENT_COMPLETED,
                (
                    "A background subagent has finished.\n\n"
                    '{"status":"failed","error":"network unavailable"}'
                ),
            )

        self.assertTrue(delivered)
        self.assertEqual(current_goal()["status"], GOAL_COMPLETE)
        self.assertEqual(len(calls), 2)
        self.assertIn('"status":"failed"', calls[0])
        self.assertEqual(calls[1], goal_continuation_input())

    def test_goal_prompt_counted_in_root_context_estimate(
        self,
    ) -> None:
        replace_goal("count me")
        captured: dict = {}

        def fake_estimate(items, **kwargs):
            captured.update(kwargs)
            captured["extra"] = kwargs.get(
                "extra_instructions",
                "",
            )
            return 1

        with patch.object(
            agent_mod,
            "estimate_response_request_tokens",
            fake_estimate,
        ):
            agent_mod.context_tokens()
        self.assertIn(
            "Active persistent goal",
            captured["extra"],
        )
        self.assertIn(
            "count me",
            captured["extra"],
        )

    def test_goal_prompt_not_injected_into_compaction_instructions(
        self,
    ) -> None:
        replace_goal("exclude me")
        captured: dict = {}

        def fake_estimate(items, **kwargs):
            captured.update(kwargs)
            captured["extra"] = kwargs.get(
                "extra_instructions",
                "",
            )
            return 1

        with patch.object(
            agent_mod,
            "estimate_response_request_tokens",
            fake_estimate,
        ):
            agent_mod.estimate_compaction_request(
                [{"role": "user", "content": "x"}],
                "checkpoint instructions only",
            )
        self.assertNotIn(
            "Active persistent goal",
            captured["extra"],
        )
        self.assertNotIn(
            "exclude me",
            captured["extra"],
        )
        self.assertIn(
            "checkpoint instructions only",
            captured["extra"],
        )
        self.assertEqual(
            captured["max_output_tokens"],
            agent_mod.compaction_generation_budget(),
        )
        self.assertEqual(
            captured["reasoning_effort"],
            agent_mod.COMPACTION_REASONING_EFFORT,
        )

    async def test_unrecoverable_goal_turn_blocks_goal(
        self,
    ) -> None:
        replace_goal("fail me")

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            return False

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(
                1,
                None,
            )
        self.assertEqual(
            current_goal()["status"],
            GOAL_BLOCKED,
        )

    async def test_transient_goal_turn_pauses_instead_of_blocking(self) -> None:
        replace_goal("retry me later")
        sent: list[str] = []

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            return agent_mod.RetryableTurnFailure(
                "a temporary Responses backend failure"
            )

        async def capture(_chat_id, text, **_kw):
            sent.append(text)

        with (
            patch.object(controller, "agent_turn", fake_turn),
            patch.object(controller, "send", capture),
        ):
            await controller.run_root_session(1, None)

        self.assertEqual(current_goal()["status"], GOAL_PAUSED)
        self.assertNotEqual(current_goal()["status"], GOAL_BLOCKED)
        self.assertTrue(
            any("temporary Responses backend" in text for text in sent)
        )

    async def test_user_turn_failure_also_pauses_active_goal(self) -> None:
        replace_goal("retry me later")

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            return agent_mod.RetryableTurnFailure(
                "an invalid model tool call"
            )

        with (
            patch.object(controller, "agent_turn", fake_turn),
            patch.object(controller, "send", _noop),
        ):
            await controller.run_root_session(1, "hello")

        self.assertEqual(current_goal()["status"], GOAL_PAUSED)

    async def test_runtime_failure_does_not_block_parent_goal(self) -> None:
        replace_goal("continue after notification failure")
        calls: list[str] = []

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            calls.append(text)
            if len(calls) == 1:
                return False
            complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(controller, "agent_turn", fake_turn),
            patch.object(controller, "send", _noop),
        ):
            delivered = await controller.enqueue_runtime_event(
                session.current_runtime().conversation_id,
                RuntimeEventKind.SUBAGENT_COMPLETED,
                "A background subagent has finished.",
            )

        self.assertFalse(delivered)
        self.assertEqual(current_goal()["status"], GOAL_COMPLETE)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], goal_continuation_input())

    async def test_runtime_event_prompt_has_one_total_context_cap(self) -> None:
        from config import MAX_TOOL_OUTPUT

        event = _runtime_event("head" + "x" * MAX_TOOL_OUTPUT + "tail")
        prompt = event.prompt()

        self.assertLessEqual(len(prompt), MAX_TOOL_OUTPUT)
        self.assertIn("head", prompt)
        self.assertIn("tail", prompt)
        self.assertIn("runtime event truncated", prompt)

    async def test_context_overflow_recovery_does_not_block_goal(
        self,
    ) -> None:
        replace_goal("recover me")
        calls = {"n": 0}

        async def fake_turn(chat_id, text, *, internal=False, **_kw):
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate recovered overflow: turn still succeeds.
                return True
            complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(
                1,
                "start",
            )
        self.assertEqual(
            current_goal()["status"],
            GOAL_COMPLETE,
        )
        self.assertNotEqual(
            current_goal().get("status"),
            GOAL_BLOCKED,
        )

    def test_restart_preserves_goal(self) -> None:
        replace_goal("keep across restart")
        goal = copy.deepcopy(current_goal())
        notice = prepare_goal_on_startup()
        self.assertIsNone(notice)
        self.assertEqual(
            current_goal()["id"],
            goal["id"],
        )
        self.assertEqual(
            current_goal()["status"],
            GOAL_ACTIVE,
        )

    def test_unclean_restart_pauses_active_goal(self) -> None:
        replace_goal("pause on unclean")
        notice = prepare_goal_on_startup(root_interrupted=True)
        self.assertIsNotNone(notice)
        self.assertEqual(
            current_goal()["status"],
            GOAL_PAUSED,
        )
        self.assertIn("paused after restart", notice)

    def test_invalid_persisted_goal_status_is_reported(self) -> None:
        replace_goal("invalid state")
        current_goal()["status"] = "unknown"

        with self.assertRaisesRegex(RuntimeError, "persisted goal status"):
            prepare_goal_on_startup()

    async def test_lost_wakeup_race_restarts_controller(
        self,
    ) -> None:
        calls: list[str] = []
        injected = {"done": False}
        real_select = controller._select_next_work

        async def fake_turn(
            chat_id,
            text,
            *,
            internal=False,
            **_kw,
        ):
            calls.append(text)
            return True

        async def gated_select(captured_goal_id):
            result = await real_select(captured_goal_id)
            if result[0] is None and not injected["done"]:
                # Work arrives after idle decision, before finally clears.
                injected["done"] = True
                await session.runtime_event_queue.put(
                    _runtime_event("late-internal")
                )
            return result

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "_select_next_work",
                gated_select,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            task = controller.start_root_session(
                1,
                "start",
            )
            await task
            # Successor started in finally may still be running.
            while True:
                nxt = turns.task
                if not nxt or nxt.done():
                    break
                await nxt
        self.assertIn("start", calls)
        self.assertTrue(any("late-internal" in call for call in calls))

    def test_clean_startup_resumes_active_goal(self) -> None:
        replace_goal("resume on clean start")
        notice = prepare_goal_on_startup()
        self.assertIsNone(notice)
        self.assertTrue(goal_is_active())
        started: list[tuple] = []

        def fake_ensure(
            chat_id,
            text=None,
            *,
            internal=False,
            **_kw,
        ):
            started.append((chat_id, text))
            return None

        with patch.object(
            controller,
            "ensure_root_session",
            fake_ensure,
        ):
            if goal_is_active():
                controller.ensure_root_session(1, None)
        self.assertEqual(started, [(1, None)])

    async def test_runtime_event_queue_task_done_balanced(
        self,
    ) -> None:
        await session.runtime_event_queue.put(_runtime_event("balanced"))
        event = controller._dequeue_runtime_event()
        self.assertEqual(event.content, "balanced")
        await asyncio.wait_for(
            session.runtime_event_queue.join(),
            timeout=0.2,
        )

    async def test_inactive_internal_entry_is_discarded(self) -> None:
        done = asyncio.get_running_loop().create_future()
        await session.runtime_event_queue.put(
            RuntimeEvent(
                conversation_id=1,
                kind=RuntimeEventKind.SUBAGENT_COMPLETED,
                content="stale",
                done=done,
                is_active=lambda: False,
            )
        )
        await session.runtime_event_queue.put(_runtime_event("current"))

        event = controller._dequeue_runtime_event()

        self.assertEqual(event.content, "current")
        self.assertFalse(await done)
        await asyncio.wait_for(
            session.runtime_event_queue.join(),
            timeout=0.2,
        )

    async def test_pause_does_not_cancel_unrelated_user_turn(
        self,
    ) -> None:
        replace_goal("keep user turn")

        task = turns.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
        )
        from commands import command

        with patch("commands.send", _noop):
            await command(1, "/goal pause")
        self.assertFalse(task.cancelled())
        self.assertEqual(
            current_goal()["status"],
            GOAL_PAUSED,
        )
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        turns.finish(task)

    async def test_pause_cancels_matching_goal_continuation(
        self,
    ) -> None:
        replace_goal("cancel continuation")
        goal_id = current_goal()["id"]

        task = turns.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
            goal_id=goal_id,
        )
        from commands import command

        with patch("commands.send", _noop):
            await command(1, "/goal pause")
        self.assertTrue(task.cancelled())
        self.assertEqual(
            current_goal()["status"],
            GOAL_PAUSED,
        )

    async def test_pause_reports_no_change_when_complete(
        self,
    ) -> None:
        replace_goal("already done")
        complete_goal(current_goal()["id"])
        messages: list[str] = []

        async def capture_send(_chat_id, text):
            messages.append(text)

        from commands import command

        with patch(
            "commands.send",
            capture_send,
        ):
            await command(1, "/goal pause")
        self.assertEqual(len(messages), 1)
        self.assertIn(
            "nothing to pause",
            messages[0],
        )

    async def test_identical_user_text_is_not_goal_continuation(
        self,
    ) -> None:
        replace_goal("classify by source")
        twin = goal_continuation_input()
        seen: list[str | None] = []

        async def fake_turn(
            chat_id,
            text,
            *,
            internal=False,
            **_kw,
        ):
            seen.append(turns.goal_id)
            complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            task = controller.start_root_session(
                1,
                twin,
                internal=False,
            )
            await task
        self.assertEqual(seen, [None])

    async def test_identical_internal_text_is_not_goal_continuation(
        self,
    ) -> None:
        replace_goal("classify by source")
        twin = goal_continuation_input()
        seen: list[str | None] = []

        async def fake_turn(
            chat_id,
            text,
            *,
            internal=False,
            **_kw,
        ):
            seen.append(turns.goal_id)
            complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            task = controller.start_root_session(
                1,
                twin,
                internal=True,
            )
            await task
        self.assertEqual(seen, [None])

    async def test_dequeued_internal_twin_text_is_not_goal(
        self,
    ) -> None:
        replace_goal("classify by source")
        twin = goal_continuation_input()
        seen: list[str | None] = []

        async def fake_turn(
            chat_id,
            text,
            *,
            internal=False,
            **_kw,
        ):
            seen.append(turns.goal_id)
            complete_goal(current_goal()["id"])
            return True

        await session.runtime_event_queue.put(_runtime_event(twin))
        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            task = controller.start_root_session(1, None)
            await task
        self.assertEqual(seen, [None])

    async def test_select_next_work_returns_explicit_source(
        self,
    ) -> None:
        replace_goal("source labels")
        twin = goal_continuation_input()
        await session.runtime_event_queue.put(_runtime_event(twin))
        text, source, entry = (
            await controller._select_next_work(
                None
            )
        )
        self.assertIn(twin, text)
        self.assertEqual(source, "runtime")
        self.assertIsNotNone(entry)
        text, source, entry = (
            await controller._select_next_work(
                None
            )
        )
        self.assertEqual(
            text,
            goal_continuation_input(),
        )
        self.assertEqual(source, "goal")
        self.assertIsNone(entry)

    async def test_goal_source_sets_turn_goal_ownership(
        self,
    ) -> None:
        replace_goal("real continuation")
        goal_id = current_goal()["id"]
        seen: list[str | None] = []

        async def fake_turn(
            chat_id,
            text,
            *,
            internal=False,
            **_kw,
        ):
            seen.append(turns.goal_id)
            complete_goal(goal_id)
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            task = controller.start_root_session(1, None)
            await task
        self.assertEqual(seen, [goal_id])


class GoalPinTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _bind_goal_test_provider(self)
        from chat_runtime import set_chat_provider
        from telegram import telegram_provider

        previous_provider = set_chat_provider(telegram_provider)
        self.addCleanup(set_chat_provider, previous_provider)
        session.state.clear()
        session.state.update(
            copy.deepcopy(session.DEFAULT_STATE)
        )
        session.subagent_records.clear()
        session.subagent_records.update(
            session.state["subagents"]
        )
        turns.finish()
        session.current_runtime().goal_pin_message_id = None
        session.current_runtime().goal_pin_goal_id = None
        session.current_runtime().pinned_status_conversation_id = None
        session.current_runtime().pinned_status_message_id = None
        session.current_runtime().pinned_status_owner = None
        session.current_runtime().goal_creation_authorized = False
        session.pending_approvals.clear()
        session.pending_steers.clear()
        while True:
            try:
                session.steer_queue.get_nowait()
                session.steer_queue.task_done()
            except asyncio.QueueEmpty:
                break
        while True:
            try:
                session.runtime_event_queue.get_nowait()
                session.runtime_event_queue.task_done()
            except asyncio.QueueEmpty:
                break
        from chat_store import create_session
        import chat_store
        self._chat_dir = Path(tempfile.mkdtemp(prefix="chats-"))
        self._chat_patch = patch.object(
            chat_store,
            "CHAT_DIR",
            self._chat_dir,
        )
        self._chat_patch.start()
        self.addCleanup(self._chat_patch.stop)
        chat_store._SEQ_CACHE.clear()
        session.state["current_session_id"] = create_session(
            kind="main",
            chat_key=session.current_runtime().key,
        )
        self._msg_ids = {"n": 100}

        async def fake_show(_chat_id, text, _goal_id=""):
            self._msg_ids["n"] += 1
            self._last_pin_text = text
            return self._msg_ids["n"]

        async def fake_clear(_chat_id, message_id, _goal_id=""):
            self._cleared.append(message_id)

        self._cleared: list[int] = []
        self._last_pin_text = ""
        self._show_patch = patch(
            "presentation.show_goal_pin",
            fake_show,
        )
        self._clear_patch = patch(
            "presentation.clear_goal_pin",
            fake_clear,
        )
        self._show_patch.start()
        self._clear_patch.start()
        self.addCleanup(self._show_patch.stop)
        self.addCleanup(self._clear_patch.stop)

    async def test_active_goal_creates_pin(self) -> None:
        replace_goal("Fix auth + verify tests")
        await ensure_goal_pin(1)
        self.assertIsNotNone(session.current_runtime().goal_pin_message_id)
        self.assertEqual(
            session.current_runtime().goal_pin_goal_id,
            current_goal()["id"],
        )
        self.assertEqual(
            self._last_pin_text,
            "Goal: Fix auth + verify tests",
        )

    async def test_goal_pin_created_only_once_across_continuations(
        self,
    ) -> None:
        replace_goal("once only")
        await ensure_goal_pin(1)
        first = session.current_runtime().goal_pin_message_id
        await ensure_goal_pin(1)
        await ensure_goal_pin(1)
        self.assertEqual(
            session.current_runtime().goal_pin_message_id,
            first,
        )
        self.assertEqual(len(self._cleared), 0)

    async def test_goal_pin_uses_current_goal_id(self) -> None:
        replace_goal("pin id")
        await ensure_goal_pin(1)
        self.assertEqual(
            session.current_runtime().goal_pin_goal_id,
            current_goal()["id"],
        )

    def test_goal_pin_objective_is_truncated(self) -> None:
        goal = create_goal("x" * 200)
        text = goal_pin_text(goal)
        self.assertTrue(text.startswith("Goal: "))
        self.assertTrue(text.endswith("..."))
        self.assertLessEqual(len(text), len("Goal: ") + 120)

    def test_goal_pin_reflects_paused_status(self) -> None:
        goal = create_goal("wait for operator")
        goal["status"] = GOAL_PAUSED

        self.assertEqual(
            goal_pin_text(goal),
            "Goal paused: wait for operator",
        )

    async def test_goal_complete_removes_pin(self) -> None:
        replace_goal("complete me")
        await ensure_goal_pin(1)
        complete_goal(current_goal()["id"])
        await sync_goal_pin(1)
        self.assertIsNone(session.current_runtime().goal_pin_message_id)
        self.assertEqual(len(self._cleared), 1)

    async def test_goal_blocked_removes_pin(self) -> None:
        replace_goal("block me")
        await ensure_goal_pin(1)
        block_goal(current_goal()["id"], "need help")
        await sync_goal_pin(1)
        self.assertIsNone(session.current_runtime().goal_pin_message_id)

    async def test_goal_pause_removes_pin(self) -> None:
        replace_goal("pause me")
        await ensure_goal_pin(1)
        from commands import command
        with patch("commands.send", _noop):
            await command(1, "/goal pause")
        self.assertIsNone(session.current_runtime().goal_pin_message_id)
        self.assertEqual(
            current_goal()["status"],
            GOAL_PAUSED,
        )

    async def test_goal_clear_removes_pin(self) -> None:
        replace_goal("clear me")
        await ensure_goal_pin(1)
        from commands import command
        with patch("commands.send", _noop):
            await command(1, "/goal clear")
        self.assertIsNone(session.current_runtime().goal_pin_message_id)

    async def test_stop_removes_pin(self) -> None:
        replace_goal("stop me")
        await ensure_goal_pin(1)
        from commands import command
        with (
            patch(
                "commands.cancel_background_subagents",
                AsyncMock(return_value=0),
            ),
            patch("commands.send", _noop),
        ):
            await command(1, "/stop")
        self.assertIsNone(session.current_runtime().goal_pin_message_id)
        self.assertEqual(
            current_goal()["status"],
            GOAL_PAUSED,
        )

    async def test_goal_resume_recreates_pin(self) -> None:
        replace_goal("resume me")
        pause_goal()
        await remove_goal_pin(1)
        from commands import command

        def fake_start(chat_id, text, *, internal=False):
            class Done:
                def done(self):
                    return True
            return Done()

        with (
            patch(
                "commands.start_root_session",
                fake_start,
            ),
            patch("commands.send", _noop),
        ):
            await command(1, "/goal resume")
        self.assertIsNotNone(session.current_runtime().goal_pin_message_id)
        self.assertEqual(
            current_goal()["status"],
            GOAL_ACTIVE,
        )

    async def test_clean_restart_recreates_active_goal_pin(
        self,
    ) -> None:
        replace_goal("clean restart")
        notice = prepare_goal_on_startup()
        self.assertIsNone(notice)
        session.current_runtime().goal_pin_message_id = None
        session.current_runtime().goal_pin_goal_id = None
        session.current_runtime().pinned_status_conversation_id = None
        session.current_runtime().pinned_status_message_id = None
        session.current_runtime().pinned_status_owner = None
        await ensure_goal_pin(1)
        self.assertIsNotNone(session.current_runtime().goal_pin_message_id)

    async def test_unclean_restart_does_not_create_pin(
        self,
    ) -> None:
        replace_goal("unclean")
        session.state["interrupted_subagents"] = [{"id": "x"}]
        notice = prepare_goal_on_startup()
        self.assertIsNotNone(notice)
        await ensure_goal_pin(1)
        self.assertIsNone(session.current_runtime().goal_pin_message_id)

    async def test_stale_goal_pin_is_replaced(self) -> None:
        replace_goal("first")
        await ensure_goal_pin(1)
        old_id = session.current_runtime().goal_pin_message_id
        session.current_runtime().goal_pin_goal_id = "stale-id"
        await ensure_goal_pin(1)
        self.assertNotEqual(
            session.current_runtime().goal_pin_message_id,
            old_id,
        )
        self.assertEqual(
            session.current_runtime().goal_pin_goal_id,
            current_goal()["id"],
        )
        self.assertIn(old_id, self._cleared)

    async def test_pin_creation_race_with_goal_completion_cleans_new_message(
        self,
    ) -> None:
        replace_goal("race")
        goal_id = current_goal()["id"]
        real_show = None

        async def racing_show(chat_id, text, _goal_id=""):
            complete_goal(goal_id)
            mid = 555
            return mid

        with patch(
            "presentation.show_goal_pin",
            racing_show,
        ):
            await ensure_goal_pin(1)
        self.assertIsNone(session.current_runtime().goal_pin_message_id)
        self.assertIn(555, self._cleared)

    async def test_pin_cleanup_is_idempotent(self) -> None:
        replace_goal("idempotent")
        await ensure_goal_pin(1)
        await remove_goal_pin(1)
        await remove_goal_pin(1)
        self.assertIsNone(session.current_runtime().goal_pin_message_id)

    async def test_pin_delete_failure_does_not_break_goal_controller(
        self,
    ) -> None:
        replace_goal("delete fails")
        await ensure_goal_pin(1)

        async def boom(_chat_id, _message_id, _goal_id):
            raise RuntimeError("telegram down")

        with patch(
            "presentation.clear_goal_pin",
            boom,
        ):
            await remove_goal_pin(1)
        self.assertIsNone(session.current_runtime().goal_pin_message_id)
        calls: list[str] = []

        async def fake_turn(chat_id, text, **_kw):
            calls.append(text)
            complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(1, "start")
        self.assertEqual(calls, ["start"])

    async def test_steer_does_not_displace_goal_pin(self) -> None:
        replace_goal("steer alongside goal")
        await ensure_goal_pin(1)
        goal_pin_id = session.current_runtime().goal_pin_message_id
        self.assertIsNotNone(goal_pin_id)

        async def fake_tg(method, payload=None):
            return {"message_id": 9001}

        with patch("telegram.tg_call", fake_tg):
            from presentation import steering_indicator
            mid = await steering_indicator(1, "abcd")
        self.assertEqual(mid, 9001)
        self.assertEqual(session.current_runtime().goal_pin_message_id, goal_pin_id)
        self.assertEqual(
            current_goal()["status"],
            GOAL_ACTIVE,
        )

    async def test_approval_suppresses_goal_pin(self) -> None:
        replace_goal("approval suppress")
        await ensure_goal_pin(1)
        await suspend_goal_pin(1)
        session.pending_approvals["a1"] = {
            "chat_id": 1,
            "message_id": 42,
            "ui_finalized": False,
        }
        await sync_goal_pin(1)
        self.assertIsNone(session.current_runtime().goal_pin_message_id)
        self.assertEqual(
            current_goal()["status"],
            GOAL_ACTIVE,
        )

    async def test_force_does_not_suppress_goal_pin(self) -> None:
        replace_goal("force alongside goal")
        await ensure_goal_pin(1)
        await suspend_goal_pin(1)
        session.pending_steers["f1"] = {
            "chat_id": 1,
            "message_id": 77,
            "status": "forcing",
        }
        await sync_goal_pin(1)
        self.assertIsNotNone(session.current_runtime().goal_pin_message_id)

    async def test_goal_pin_restored_after_action_pin_clears(
        self,
    ) -> None:
        replace_goal("restore me")
        await ensure_goal_pin(1)
        await suspend_goal_pin(1)
        self.assertIsNone(session.current_runtime().goal_pin_message_id)
        await sync_goal_pin(1)
        self.assertIsNotNone(session.current_runtime().goal_pin_message_id)

    async def test_completed_goal_not_restored_after_action_pin_clears(
        self,
    ) -> None:
        replace_goal("done")
        await ensure_goal_pin(1)
        await suspend_goal_pin(1)
        complete_goal(current_goal()["id"])
        await sync_goal_pin(1)
        self.assertIsNone(session.current_runtime().goal_pin_message_id)

    async def test_paused_goal_not_restored_after_action_pin_clears(
        self,
    ) -> None:
        replace_goal("paused")
        await ensure_goal_pin(1)
        await suspend_goal_pin(1)
        pause_goal()
        await sync_goal_pin(1)
        self.assertIsNone(session.current_runtime().goal_pin_message_id)

    async def test_pin_suppression_does_not_change_goal_state(
        self,
    ) -> None:
        replace_goal("unchanged")
        before = copy.deepcopy(current_goal())
        await ensure_goal_pin(1)
        await suspend_goal_pin(1)
        self.assertEqual(current_goal(), before)


class PinnedStatusSlotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _bind_goal_test_provider(self)
        from chat_runtime import set_chat_provider
        from telegram import telegram_provider

        previous_provider = set_chat_provider(telegram_provider)
        self.addCleanup(set_chat_provider, previous_provider)
        session.state.clear()
        session.state.update(copy.deepcopy(session.DEFAULT_STATE))
        session.current_runtime().goal_pin_message_id = None
        session.current_runtime().goal_pin_goal_id = None
        session.current_runtime().pinned_status_conversation_id = None
        session.current_runtime().pinned_status_message_id = None
        session.current_runtime().pinned_status_owner = None
        session.pending_approvals.clear()
        session.pending_steers.clear()
        self.calls: list[tuple[str, dict]] = []

        async def fake_tg(method, payload=None):
            self.calls.append((method, payload or {}))
            if method == "sendMessage":
                return {"message_id": 321}
            return True

        self._tg_patch = patch("telegram.tg_call", fake_tg)
        self._tg_patch.start()
        self.addCleanup(self._tg_patch.stop)

    async def test_approval_reuses_pinned_slot_and_restores_in_place(
        self,
    ) -> None:
        from approval import finalize_approval_message
        from chat_provider import ActionButton
        from presentation import show_pinned_status

        goal = replace_goal("shared slot")
        await ensure_goal_pin(1)
        self.assertEqual(session.current_runtime().pinned_status_message_id, 321)
        self.assertEqual(
            session.current_runtime().pinned_status_owner,
            f"goal:{goal['id']}",
        )

        approval_id = "abc123"
        await show_pinned_status(
            1,
            f"approval:{approval_id}",
            "Approval required",
            controls=((ActionButton("Approve", "approval:test"),),),
        )
        await suspend_goal_pin(1)
        entry = {
            "id": approval_id,
            "chat_id": 1,
            "message_id": 321,
            "ui_finalized": False,
        }
        session.pending_approvals[approval_id] = entry

        await finalize_approval_message(entry, "approved")

        self.assertEqual(session.current_runtime().pinned_status_message_id, 321)
        self.assertEqual(
            session.current_runtime().pinned_status_owner,
            f"goal:{goal['id']}",
        )
        methods = [method for method, _payload in self.calls]
        self.assertEqual(methods.count("sendMessage"), 1)
        self.assertEqual(methods.count("pinChatMessage"), 1)
        self.assertGreaterEqual(methods.count("editMessageText"), 2)
        self.assertNotIn("unpinChatMessage", methods)
        self.assertNotIn("deleteMessage", methods)
        pin_payload = next(
            payload
            for method, payload in self.calls
            if method == "pinChatMessage"
        )
        self.assertEqual(
            pin_payload,
            {
                "chat_id": 1,
                "message_id": 321,
                "disable_notification": True,
            },
        )

    async def test_approval_cleanup_failure_does_not_block_decision(
        self,
    ) -> None:
        from approval import handle_approval_action
        from chat_provider import IncomingAction
        from chat_runtime import get_chat_provider

        approval_id = "abc123"
        future = asyncio.get_running_loop().create_future()
        session.pending_approvals[approval_id] = {
            "id": approval_id,
            "chat_id": 1,
            "future": future,
            "ui_finalized": False,
        }
        action = IncomingAction(
            action_id="action-1",
            conversation_id=1,
            sender_id=get_chat_provider().authorized_user_id,
            message_id=321,
            data=f"approval:{approval_id}:approve",
        )

        with patch(
            "approval.sync_goal_pin",
            AsyncMock(side_effect=RuntimeError("presentation unavailable")),
        ):
            handled = await handle_approval_action(action)

        self.assertTrue(handled)
        self.assertEqual(future.result(), "approve")

    async def test_stale_owner_cannot_clear_newer_slot_occupant(
        self,
    ) -> None:
        from presentation import clear_pinned_status, show_pinned_status

        await show_pinned_status(1, "approval:old", "approval")
        await show_pinned_status(1, "steer:new", "steering")

        cleared = await clear_pinned_status(
            1,
            owner="approval:old",
        )

        self.assertFalse(cleared)
        self.assertEqual(session.current_runtime().pinned_status_owner, "steer:new")
        self.assertEqual(session.current_runtime().pinned_status_message_id, 321)

    async def test_empty_slot_unpins_then_deletes_message(self) -> None:
        from presentation import clear_pinned_status, show_pinned_status

        await show_pinned_status(1, "goal:test", "Goal active: test")
        cleared = await clear_pinned_status(1, owner="goal:test")

        self.assertTrue(cleared)
        self.assertIsNone(session.current_runtime().pinned_status_message_id)
        methods = [method for method, _payload in self.calls]
        self.assertEqual(
            methods,
            [
                "sendMessage",
                "pinChatMessage",
                "unpinChatMessage",
                "deleteMessage",
            ],
        )

    async def test_compacting_indicator_animates_then_deletes(self) -> None:
        from presentation import compacting_indicator

        async with compacting_indicator(1, interval=0.005):
            await asyncio.sleep(0.018)

        methods = [method for method, _payload in self.calls]
        self.assertEqual(methods[0], "sendMessage")
        self.assertEqual(methods[-1], "deleteMessage")
        texts = [
            payload.get("text")
            for method, payload in self.calls
            if method in {"sendMessage", "editMessageText"}
        ]
        self.assertEqual(texts[0], "Compacting.")
        self.assertIn("Compacting..", texts)
        self.assertIn("Compacting...", texts)
        self.assertEqual(
            self.calls[-1][1],
            {"chat_id": 1, "message_id": 321},
        )


class SetGoalToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _bind_goal_test_provider(self)
        session.state.clear()
        session.state.update(
            copy.deepcopy(session.DEFAULT_STATE)
        )
        session.subagent_records.clear()
        session.subagent_records.update(
            session.state["subagents"]
        )
        turns.finish()
        session.current_runtime().goal_pin_message_id = None
        session.current_runtime().goal_pin_goal_id = None
        session.current_runtime().pinned_status_conversation_id = None
        session.current_runtime().pinned_status_message_id = None
        session.current_runtime().pinned_status_owner = None
        session.current_runtime().goal_creation_authorized = False
        session.pending_approvals.clear()
        session.pending_steers.clear()
        while True:
            try:
                session.steer_queue.get_nowait()
                session.steer_queue.task_done()
            except asyncio.QueueEmpty:
                break
        while True:
            try:
                session.runtime_event_queue.get_nowait()
                session.runtime_event_queue.task_done()
            except asyncio.QueueEmpty:
                break
        from chat_store import create_session
        import chat_store
        self._chat_dir = Path(tempfile.mkdtemp(prefix="chats-"))
        self._chat_patch = patch.object(
            chat_store,
            "CHAT_DIR",
            self._chat_dir,
        )
        self._chat_patch.start()
        self.addCleanup(self._chat_patch.stop)
        chat_store._SEQ_CACHE.clear()
        session.state["current_session_id"] = create_session(
            kind="main",
            chat_key=session.current_runtime().key,
        )
        self._msg_ids = {"n": 200}

        async def fake_show(_chat_id, text, _goal_id=""):
            self._msg_ids["n"] += 1
            return self._msg_ids["n"]

        async def fake_clear(_chat_id, message_id, _goal_id=""):
            return None

        self._show_patch = patch(
            "presentation.show_goal_pin",
            fake_show,
        )
        self._clear_patch = patch(
            "presentation.clear_goal_pin",
            fake_clear,
        )
        self._send_patch = patch(
            "chat_runtime.send",
            _noop,
        )
        self._show_patch.start()
        self._clear_patch.start()
        self._send_patch.start()
        self.addCleanup(self._show_patch.stop)
        self.addCleanup(self._clear_patch.stop)
        self.addCleanup(self._send_patch.stop)

    async def test_user_turn_can_set_goal(self) -> None:
        authorized = {"seen": False}

        async def fake_turn(
            chat_id,
            user_text,
            *,
            internal=False,
            source=None,
            allow_goal_creation=False,
        ):
            authorized["seen"] = allow_goal_creation
            session.current_runtime().goal_creation_authorized = allow_goal_creation
            result = await set_goal_tool(
                {
                    "objective": (
                        "Fix the auth system and verify "
                        "all tests pass"
                    ),
                },
                chat_id=chat_id,
            )
            self.assertTrue(result["ok"])
            complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(
                1,
                (
                    "Keep working on fixing the auth system "
                    "until all tests pass."
                ),
            )
        self.assertTrue(authorized["seen"])

    def test_dont_set_a_goal_is_prompt_behavior_not_regex(
        self,
    ) -> None:
        """Negated phrasing must not be regex-authorized.

        User-source turns are gate-authorized; declining a goal is
        enforced by the set_goal tool description, not text matching.
        """
        import goals as goals_mod
        from config import TOOLS
        self.assertFalse(
            hasattr(goals_mod, "user_requests_goal_creation")
        )
        set_goal = next(
            tool
            for tool in TOOLS
            if tool.get("name") == "set_goal"
        )
        description = str(
            set_goal.get("description") or ""
        ).lower()
        self.assertIn("explicitly", description)
        self.assertIn("declines", description)
        text = "Don't set a goal, just fix this."
        self.assertIn("don't set a goal", text.lower())
        # User-source gate still authorizes; model must not call
        # set_goal — that is prompt/tool-description behavior.
        allow_for_user_source = True
        self.assertTrue(allow_for_user_source)
    async def test_goal_continuation_cannot_set_goal(
        self,
    ) -> None:
        replace_goal("already")
        seen = {"allow": None}

        async def fake_turn(
            chat_id,
            text,
            *,
            internal=False,
            source=None,
            allow_goal_creation=False,
        ):
            seen["allow"] = allow_goal_creation
            session.current_runtime().goal_creation_authorized = allow_goal_creation
            result = await set_goal_tool(
                {"objective": "hijack"},
                chat_id=chat_id,
            )
            self.assertFalse(result["ok"])
            complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(1, None)
        self.assertFalse(seen["allow"])

    async def test_internal_turn_cannot_set_goal(self) -> None:
        seen = {"allow": None}

        async def fake_turn(
            chat_id,
            text,
            *,
            internal=False,
            source=None,
            allow_goal_creation=False,
        ):
            seen["allow"] = allow_goal_creation
            session.current_runtime().goal_creation_authorized = allow_goal_creation
            result = await set_goal_tool(
                {"objective": "nope"},
                chat_id=chat_id,
            )
            self.assertFalse(result["ok"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(
                1,
                "set a goal to finish this",
                internal=True,
            )
        self.assertFalse(seen["allow"])

    async def test_subagent_completion_cannot_set_goal(
        self,
    ) -> None:
        seen = {"allow": None}

        async def fake_turn(
            chat_id,
            text,
            *,
            internal=False,
            source=None,
            allow_goal_creation=False,
        ):
            seen["allow"] = allow_goal_creation
            session.current_runtime().goal_creation_authorized = allow_goal_creation
            result = await set_goal_tool(
                {"objective": "from subagent"},
                chat_id=chat_id,
            )
            self.assertFalse(result["ok"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.enqueue_runtime_event(
                session.current_runtime().conversation_id,
                RuntimeEventKind.SUBAGENT_COMPLETED,
                "A background subagent has finished.",
            )
        self.assertFalse(seen["allow"])

    async def test_goal_creation_authorization_resets_after_turn(
        self,
    ) -> None:
        during = {"authorized": None}

        async def fake_stream(chat_id, items, extra_instructions=""):
            during["authorized"] = (
                session.current_runtime().goal_creation_authorized
            )
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "ok",
                            }
                        ],
                    }
                ]
            }

        with (
            patch.object(
                agent_mod,
                "responses_create_stream",
                fake_stream,
            ),
            patch.object(
                agent_mod,
                "send",
                _noop,
            ),
            patch.object(
                agent_mod,
                "typing_loop",
                _noop,
            ),
            patch.object(
                agent_mod,
                "maybe_auto_compact",
                _noop,
            ),
        ):
            await agent_mod.agent_turn(
                1,
                "Don't set a goal, just fix this.",
                source="user",
                allow_goal_creation=True,
            )
        self.assertTrue(during["authorized"])
        self.assertFalse(session.current_runtime().goal_creation_authorized)

    async def test_set_goal_rejects_existing_unfinished_goal(
        self,
    ) -> None:
        replace_goal("existing")
        session.current_runtime().goal_creation_authorized = True
        result = await set_goal_tool(
            {"objective": "replacement"},
            chat_id=1,
        )
        self.assertFalse(result["ok"])
        self.assertIn("unfinished", result["error"])
        self.assertEqual(
            current_goal()["objective"],
            "existing",
        )

    async def test_set_goal_uses_existing_goal_validation_and_length_limit(
        self,
    ) -> None:
        session.current_runtime().goal_creation_authorized = True
        result = await set_goal_tool(
            {"objective": "   "},
            chat_id=1,
        )
        self.assertFalse(result["ok"])
        result = await set_goal_tool(
            {"objective": "x" * (MAX_GOAL_CHARS + 1)},
            chat_id=1,
        )
        self.assertFalse(result["ok"])
        self.assertIsNone(current_goal())

    async def test_set_goal_starts_normal_goal_controller(
        self,
    ) -> None:
        calls: list[str] = []

        async def fake_turn(
            chat_id,
            text,
            *,
            internal=False,
            source=None,
            allow_goal_creation=False,
        ):
            calls.append(text)
            session.current_runtime().goal_creation_authorized = allow_goal_creation
            if allow_goal_creation and not current_goal():
                await set_goal_tool(
                    {"objective": "ship it"},
                    chat_id=chat_id,
                )
                return True
            if source == "goal":
                complete_goal(current_goal()["id"])
            return True

        with (
            patch.object(
                controller,
                "agent_turn",
                fake_turn,
            ),
            patch.object(
                controller,
                "send",
                _noop,
            ),
        ):
            await controller.run_root_session(
                1,
                (
                    "Keep working on this until it is "
                    "completely finished."
                ),
            )
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(
            calls[1],
            goal_continuation_input(),
        )

    async def test_set_goal_creates_goal_pin(self) -> None:
        session.current_runtime().goal_creation_authorized = True
        result = await set_goal_tool(
            {"objective": "pin please"},
            chat_id=1,
        )
        self.assertTrue(result["ok"])
        self.assertIsNotNone(session.current_runtime().goal_pin_message_id)
        self.assertEqual(
            session.current_runtime().goal_pin_goal_id,
            current_goal()["id"],
        )


if __name__ == "__main__":
    unittest.main()
