"""Transcript durability during agent turns."""
import asyncio
import copy
import json
import os
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-durable-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ["MODEL_CONTEXT_TOKENS"] = "131072"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from raptor.agent import agent as agent_mod
from raptor.state import chat_store
from raptor.agent import context
from raptor.agent import controller
from raptor.model import responses
from raptor.model.response_errors import MalformedToolCallError
from raptor.state import session
from raptor.agent.turn_runtime import turns
from raptor.agent import subagents
from raptor.model.model_providers import ModelConfiguration, ModelProvider, ModelTarget


async def _noop(*_a, **_k):
    return None


class TranscriptDurabilityTests(unittest.IsolatedAsyncioTestCase):
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
            f"durability:{self._chat_dir.name}"
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
        turns.finish()
        session.pending_steers.clear()
        while True:
            try:
                session.steer_queue.get_nowait()
                session.steer_queue.task_done()
            except Exception:
                break

    def test_checkpoint_notice_is_concise(self) -> None:
        session_id = str(session.state["current_session_id"])
        user = chat_store.append_item(
            session_id,
            {"role": "user", "content": "compact me"},
            source="user",
        )
        chat_store.append_checkpoint(
            session_id,
            summary="durable summary",
            through_seq=int(user["seq"]),
        )

        self.assertEqual(
            agent_mod._checkpoint_saved_message(session_id),
            "Checkpoint saved",
        )

    async def test_manual_compaction_forces_available_history(self) -> None:
        captured: dict[str, object] = {}

        async def compact(*args, **kwargs):
            del args
            captured.update(kwargs)
            return False

        @asynccontextmanager
        async def indicator(*args, **kwargs):
            del args, kwargs
            yield

        with (
            patch.object(agent_mod, "compact_session", compact),
            patch.object(agent_mod, "compacting_indicator", indicator),
            patch.object(agent_mod, "typing_loop", _noop),
            patch.object(agent_mod, "send", _noop),
        ):
            await agent_mod.compact_context("durability:chat")

        self.assertIs(captured["force"], True)
        self.assertEqual(captured["reason"], "manual")

    async def test_manual_compaction_uses_configured_model_target(self) -> None:
        cheap = ModelTarget("economy", "compact-model")
        configuration = ModelConfiguration(
            providers={
                "local": ModelProvider(
                    id="local",
                    base_url="http://local.example/v1",
                    default_model="model-a",
                    context_window=131072,
                ),
                "economy": ModelProvider(
                    id="economy",
                    base_url="http://economy.example/v1",
                    default_model="compact-model",
                    context_window=32768,
                ),
            },
            default_target=self.target,
            compaction_provider_id="economy",
        )
        captured: dict[str, object] = {}

        async def compact(*_args, **kwargs):
            captured.update(kwargs)
            kwargs["estimate_compaction_request"]([], "instructions")
            await kwargs["create_compaction_response"]([], "instructions")
            return False

        @asynccontextmanager
        async def indicator(*_args, **_kwargs):
            yield

        estimate = Mock(return_value=1)
        create = AsyncMock(return_value={})
        expected_input_budget = 0
        expected_generation_budget = 0
        with (
            patch.object(agent_mod, "MODEL_CONFIGURATION", configuration),
            patch.object(responses, "MODEL_CONFIGURATION", configuration),
            patch.object(agent_mod, "compact_session", compact),
            patch.object(agent_mod, "estimate_compaction_request", estimate),
            patch.object(agent_mod, "create_compaction_response", create),
            patch.object(agent_mod, "compacting_indicator", indicator),
            patch.object(agent_mod, "typing_loop", _noop),
            patch.object(agent_mod, "send", _noop),
        ):
            await agent_mod.compact_context("durability:chat")
            expected_input_budget = agent_mod._input_budget(cheap)
            expected_generation_budget = agent_mod._generation_budget(cheap)

        self.assertEqual(estimate.call_args.args[0], cheap)
        self.assertEqual(create.await_args.args[0], cheap)
        self.assertEqual(captured["input_budget"], expected_input_budget)
        self.assertEqual(
            captured["generation_budget"],
            expected_generation_budget,
        )

    async def test_items_written_during_turn(self) -> None:
        sid = session.state["current_session_id"]
        seen_before_tool: list[int] = []

        async def fake_stream(
            _target,
            chat_id,
            items,
            extra_instructions="",
            **_kwargs,
        ):
            # After user append + before/during model, user item exists.
            items_now = chat_store.item_events(sid)
            self.assertTrue(
                any(
                    e["item"].get("content") == "do it"
                    for e in items_now
                    if e["item"].get("role") == "user"
                )
            )
            if not seen_before_tool:
                seen_before_tool.append(len(items_now))
                return {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "list_dir",
                            "call_id": "c1",
                            "arguments": "{}",
                        }
                    ]
                }
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "done",
                            }
                        ],
                    }
                ]
            }

        async def fake_exec(chat_id, call, execution_context=None):
            # Function call must already be archived before tool runs.
            calls = [
                e
                for e in chat_store.item_events(sid)
                if e["item"].get("type") == "function_call"
            ]
            self.assertEqual(len(calls), 1)
            return {"ok": True, "entries": []}

        with (
            patch.object(
                agent_mod,
                "responses_create_stream",
                fake_stream,
            ),
            patch.object(
                agent_mod,
                "execute_tool_with_approval",
                fake_exec,
            ),
            patch.object(agent_mod, "send", _noop),
            patch.object(agent_mod, "typing_loop", _noop),
            patch.object(agent_mod, "maybe_auto_compact", _noop),
        ):
            ok = await agent_mod.agent_turn(1, "do it")
        self.assertTrue(ok)
        events = chat_store.item_events(sid)
        types = [
            e["item"].get("type") or e["item"].get("role")
            for e in events
        ]
        self.assertIn("user", types)
        self.assertIn("function_call", types)
        self.assertIn("function_call_output", types)
        self.assertIn("message", types)

    async def test_threshold_compaction_resumes_without_reanswering(self) -> None:
        requests: list[list[dict]] = []
        compact_calls = 0
        call = {
            "type": "function_call",
            "name": "list_dir",
            "call_id": "c1",
            "arguments": "{}",
        }

        async def fake_stream(_target, _chat_id, items, **_kwargs):
            requests.append(copy.deepcopy(items))
            if len(requests) == 1:
                return {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "Initial answer already streamed.",
                                }
                            ],
                        },
                        call,
                    ]
                }
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Finished."}
                        ],
                    }
                ]
            }

        def estimate(items, **_kwargs):
            return (
                200
                if any(
                    item.get("type") == "function_call_output"
                    for item in items
                )
                else 1
            )

        async def compact(*_args, **kwargs):
            nonlocal compact_calls
            compact_calls += 1
            work = [
                {"role": "user", "content": "Inspect the repository."},
                {
                    "role": "user",
                    "content": (
                        "Raptor conversation checkpoint: the initial answer "
                        "was already communicated; finish the tool work."
                    ),
                },
            ]
            if kwargs["include_continuation"]:
                work.extend(context.checkpoint_continuation_input())
            return work

        @asynccontextmanager
        async def indicator(*_args, **_kwargs):
            yield

        surface = Mock(
            stream=AsyncMock(),
            finished=AsyncMock(),
            clear=AsyncMock(),
        )
        with (
            patch.object(agent_mod, "ToolActivitySurface", return_value=surface),
            patch.object(agent_mod, "responses_create_stream", fake_stream),
            patch.object(
                agent_mod,
                "execute_tool_with_approval",
                AsyncMock(return_value={"ok": True}),
            ),
            patch.object(agent_mod, "estimate_response_request_tokens", estimate),
            patch.object(agent_mod, "_input_budget", return_value=100),
            patch.object(agent_mod, "ensure_context_under_budget", compact),
            patch.object(agent_mod, "compacting_indicator", indicator),
            patch.object(agent_mod, "send", _noop),
            patch.object(agent_mod, "typing_loop", _noop),
            patch.object(agent_mod, "maybe_auto_compact", _noop),
        ):
            result = await agent_mod.agent_turn(1, "Inspect the repository.")

        self.assertTrue(result)
        self.assertEqual(compact_calls, 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            requests[1][-1],
            context.CHECKPOINT_CONTINUATION_INPUT,
        )

    async def test_root_tool_activity_streams_and_tracks_execution(self) -> None:
        responses = 0
        typing_events: list[str] = []
        presentation_events: list[str] = []
        streamed_call = {
            "type": "function_call",
            "name": "list_dir",
            "call_id": "c1",
            "arguments": "{}",
        }
        async def clear_surface():
            presentation_events.append("clear")

        surface = Mock(
            stream=AsyncMock(),
            running=AsyncMock(),
            finished=AsyncMock(),
            clear=AsyncMock(side_effect=clear_surface),
        )

        async def fake_stream(
            _target,
            _chat_id,
            _items,
            *,
            on_tool_call=None,
            **_kwargs,
        ):
            nonlocal responses
            responses += 1
            await asyncio.sleep(0)
            if responses == 1:
                await on_tool_call(streamed_call, True)
                return {"output": [streamed_call]}
            self.assertEqual(
                typing_events,
                ["started", "stopped", "started"],
            )
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "done"}
                        ],
                    }
                ]
            }

        async def execute(*_args, **kwargs):
            self.assertEqual(typing_events, ["started", "stopped"])
            self.assertIs(kwargs["tool_activity"], surface)
            return {"ok": True}

        async def send_answer(*_args, **_kwargs):
            presentation_events.append("send")

        async def typing(_chat_id):
            typing_events.append("started")
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                typing_events.append("stopped")

        with (
            patch.object(agent_mod, "ToolActivitySurface", return_value=surface),
            patch.object(agent_mod, "responses_create_stream", fake_stream),
            patch.object(agent_mod, "execute_tool_with_approval", execute),
            patch.object(agent_mod, "send", send_answer),
            patch.object(agent_mod, "typing_loop", typing),
            patch.object(agent_mod, "maybe_auto_compact", _noop),
        ):
            result = await agent_mod.agent_turn(1, "inspect")

        self.assertTrue(result)
        surface.stream.assert_awaited_once_with(streamed_call, True)
        surface.finished.assert_awaited_once_with(
            streamed_call,
            {"ok": True},
        )
        self.assertEqual(surface.clear.await_count, 2)
        self.assertEqual(presentation_events[:2], ["clear", "send"])
        self.assertEqual(
            typing_events,
            ["started", "stopped", "started", "stopped"],
        )

    async def test_post_delivery_compaction_failure_keeps_success(self) -> None:
        sent: list[str] = []

        async def answer(*_args, **_kwargs):
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "done"}
                        ],
                    }
                ]
            }

        async def capture(_chat_id, text, **_kwargs):
            sent.append(text)

        async def fail_maintenance(_chat_id):
            raise RuntimeError("checkpoint backend unavailable")

        with (
            patch.object(agent_mod, "responses_create_stream", answer),
            patch.object(agent_mod, "send", capture),
            patch.object(agent_mod, "typing_loop", _noop),
            patch.object(
                agent_mod,
                "maybe_auto_compact",
                fail_maintenance,
            ),
            patch.object(agent_mod, "log_event") as logged,
        ):
            result = await agent_mod.agent_turn(1, "finish")

        self.assertTrue(result)
        self.assertEqual(sent, ["done"])
        self.assertIsNone(session.state["pending_delivery"])
        self.assertTrue(
            any(
                call.args[1] == "post_delivery_compaction_error"
                for call in logged.call_args_list
            )
        )

    async def test_malformed_tool_call_closes_the_durable_turn(self) -> None:
        sid = str(session.state["current_session_id"])
        sent: list[str] = []

        async def malformed(*_args, **_kwargs):
            failed_input = chat_store.item_events(sid)[-1]
            chat_store.append_checkpoint(
                sid,
                summary="checkpoint containing rejected task: do it",
                through_seq=int(failed_input["seq"]),
                reason="threshold",
            )
            raise MalformedToolCallError("invalid generated arguments")

        async def capture(_chat_id, text, **_kwargs):
            sent.append(text)

        with (
            patch.object(agent_mod, "responses_create_stream", malformed),
            patch.object(agent_mod, "send", capture),
            patch.object(agent_mod, "typing_loop", _noop),
            patch.object(agent_mod, "maybe_auto_compact", _noop),
        ):
            result = await agent_mod.agent_turn(1, "do it")

        self.assertIsInstance(result, agent_mod.RetryableTurnFailure)
        self.assertEqual(sent, [agent_mod.MALFORMED_TOOL_CALL_MESSAGE])
        items = [event["item"] for event in chat_store.item_events(sid)]
        self.assertEqual(items[-1]["role"], "assistant")
        self.assertIn(
            agent_mod.MALFORMED_TOOL_CALL_MESSAGE,
            str(items[-1]),
        )
        self.assertNotIn("do it", str(agent_mod.build_active_context(sid)))
        self.assertNotIn(
            agent_mod.MALFORMED_TOOL_CALL_MESSAGE,
            str(agent_mod.build_active_context(sid)),
        )

        seen_work: list[dict] = []

        async def answer(_target, _chat_id, work, **_kwargs):
            seen_work.extend(copy.deepcopy(work))
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Hello"}
                        ],
                    }
                ]
            }

        with (
            patch.object(agent_mod, "responses_create_stream", answer),
            patch.object(agent_mod, "send", _noop),
            patch.object(agent_mod, "typing_loop", _noop),
            patch.object(agent_mod, "maybe_auto_compact", _noop),
        ):
            result = await agent_mod.agent_turn(1, "hi")

        self.assertTrue(result)
        self.assertEqual(seen_work[0]["content"], "hi")
        self.assertNotIn("do it", str(seen_work))
        self.assertNotIn(agent_mod.MALFORMED_TOOL_CALL_MESSAGE, str(seen_work))

    async def test_root_turn_marker_spans_exact_task_lifetime(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_turn(*_args, **_kwargs):
            started.set()
            await release.wait()
            return True

        with (
            patch.object(controller, "agent_turn", fake_turn),
            patch.object(controller, "ensure_goal_pin", _noop),
            patch.object(controller, "sync_goal_pin", _noop),
            patch.object(controller, "save_state"),
        ):
            task = controller.start_root_session(1, "work")
            await started.wait()
            marker = session.state["active_root_turn"]
            self.assertEqual(marker["id"], turns.snapshot.id)
            self.assertEqual(
                marker["session_id"],
                session.state["current_session_id"],
            )
            release.set()
            await task

        self.assertIsNone(session.state["active_root_turn"])

    async def test_delivery_failure_does_not_spin_controller(self) -> None:
        with (
            patch.object(
                controller,
                "flush_pending_delivery",
                return_value=False,
            ),
            patch.object(
                controller,
                "_pending_controller_work",
                return_value=True,
            ),
            patch.object(controller, "start_root_session") as restart,
        ):
            await controller.run_root_session(1, None)

        restart.assert_not_called()

    async def test_failed_telegram_send_keeps_assistant_transcript(
        self,
    ) -> None:
        sid = session.state["current_session_id"]

        async def fake_stream(
            _target,
            chat_id,
            items,
            extra_instructions="",
            **_kwargs,
        ):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "reply-text",
                            }
                        ],
                    }
                ]
            }

        async def boom(*_a, **_k):
            raise RuntimeError("telegram down")

        with (
            patch.object(
                agent_mod,
                "responses_create_stream",
                fake_stream,
            ),
            patch.object(agent_mod, "send", boom),
            patch.object(agent_mod, "typing_loop", _noop),
            patch.object(agent_mod, "maybe_auto_compact", _noop),
        ):
            ok = await agent_mod.agent_turn(1, "hello")
        self.assertIsInstance(ok, agent_mod.RetryableTurnFailure)
        self.assertEqual(ok.reason, "response delivery failed")
        texts = chat_store.render_compaction_records(
            chat_store.item_events(sid)
        )
        self.assertIn("reply-text", texts)
        self.assertIn("hello", texts)
        reference = session.state["pending_delivery"]
        self.assertEqual(reference["session_id"], sid)
        restart_state = self._chat_dir / "restart-state.json"
        restart_state.write_text(
            json.dumps(
                {
                    "schema_version": session.STATE_SCHEMA_VERSION,
                    "model": session.state.get("model"),
                    "runtime": {},
                    "chats": {
                        session.current_runtime().key: {
                            "conversation_id": 1,
                            "state": dict(session.state),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        with patch.object(session, "STATE_PATH", restart_state):
            restarted = session.load_state()
        restarted_chat = restarted["chats"][session.current_runtime().key]
        self.assertEqual(
            restarted_chat["state"]["pending_delivery"],
            reference,
        )

        delivered: list[str] = []

        async def recover(_chat_id, text):
            delivered.append(text)

        with (
            patch.object(agent_mod, "send", recover),
            patch.object(
                agent_mod,
                "detached_delivery_context",
                return_value=nullcontext(),
            ),
        ):
            recovered = await agent_mod.flush_pending_delivery(1)

        self.assertTrue(recovered)
        self.assertEqual(delivered, ["reply-text"])
        self.assertIsNone(session.state["pending_delivery"])

    async def test_prepared_delivery_without_event_is_abandoned(self) -> None:
        sid = str(session.state["current_session_id"])
        seq = chat_store.next_event_seq(sid)
        session.set_pending_delivery(sid, seq)
        restart_state = self._chat_dir / "prepared-state.json"
        restart_state.write_text(
            json.dumps(
                {
                    "schema_version": session.STATE_SCHEMA_VERSION,
                    "model": session.state.get("model"),
                    "runtime": {},
                    "chats": {
                        session.current_runtime().key: {
                            "conversation_id": 1,
                            "state": dict(session.state),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        with chat_store.chat_path(sid).open("ab") as handle:
            handle.write(b'{"type":"item","seq":')
        chat_store._SEQ_CACHE.clear()
        with patch.object(session, "STATE_PATH", restart_state):
            restarted = session.load_state()
        restarted_chat = restarted["chats"][session.current_runtime().key]
        self.assertEqual(
            restarted_chat["state"]["pending_delivery"],
            {"session_id": sid, "seq": seq},
        )

        sent: list[str] = []

        async def capture(_chat_id, text):
            sent.append(text)

        with patch.object(agent_mod, "send", capture):
            recovered = await agent_mod.flush_pending_delivery(1)

        self.assertTrue(recovered)
        self.assertEqual(sent, [])
        self.assertIsNone(session.state["pending_delivery"])

    async def test_recovery_prompt_does_not_embed_raw_tool_events(
        self,
    ) -> None:
        huge_output = "SECRET-RAW-TOOL-OUTPUT" * 20_000
        session.state["interrupted_subagents"] = [
            {
                "id": "sub-1",
                "session_id": "sub-session",
                "interrupted_at": 124.0,
                "tool_events": [{"output": huge_output}],
            }
        ]
        seen: dict[str, str] = {}

        async def fake_stream(
            _target,
            chat_id,
            items,
            extra_instructions="",
            **_kwargs,
        ):
            seen["instructions"] = extra_instructions
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "recovered",
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
            patch.object(agent_mod, "send", _noop),
            patch.object(agent_mod, "typing_loop", _noop),
            patch.object(agent_mod, "maybe_auto_compact", _noop),
        ):
            ok = await agent_mod.agent_turn(1, "resume")

        self.assertTrue(ok)
        instructions = seen["instructions"]
        self.assertNotIn("SECRET-RAW-TOOL-OUTPUT", instructions)
        self.assertIn('"tool_event_count": 1', instructions)
        self.assertIn('"session_id": "sub-session"', instructions)

    def test_unclean_root_turn_closes_unmatched_tool_call(self) -> None:
        sid = str(session.state["current_session_id"])
        chat_store.append_item(
            sid,
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "call-1",
                "arguments": '{"command":"work"}',
            },
            source="assistant",
        )
        session.state["active_root_turn"] = {
            "id": "turn-1",
            "session_id": sid,
        }

        repaired = agent_mod.repair_interrupted_root_turn()

        self.assertTrue(repaired)
        self.assertIsNone(session.state["active_root_turn"])
        items = [event["item"] for event in chat_store.item_events(sid)]
        self.assertEqual(items[-2]["type"], "function_call_output")
        self.assertEqual(items[-2]["call_id"], "call-1")
        self.assertEqual(items[-1]["role"], "user")
        self.assertIn("turn_aborted", items[-1]["content"])

    def test_subagent_checkpoint_reference_is_bounded(self) -> None:
        huge_output = "x" * 500_000
        previous = {
            "session_id": "old-session",
            "interrupted_at": 123.0,
            "tool_events": [{"output": huge_output}],
            "resumed_from": {
                "tool_events": [{"output": huge_output}],
            },
        }

        ref = agent_mod._subagent_checkpoint_ref(previous)

        self.assertEqual(
            ref,
            {
                "session_id": "old-session",
                "interrupted_at": 123.0,
                "tool_event_count": 1,
            },
        )
        self.assertLess(len(str(ref)), 200)

    def test_subagent_recovery_context_is_bounded(self) -> None:
        huge_output = "x" * 500_000

        payload = subagents._recovery_prompt_payload(
            {
                "status": "interrupted",
                "last_task": "inspect target",
                "error": "process exited",
                "tool_events": [{"output": huge_output}],
            }
        )

        self.assertEqual(
            payload,
            {
                "status": "interrupted",
                "last_task": "inspect target",
                "error": "process exited",
                "tool_event_count": 1,
            },
        )
        self.assertLess(len(str(payload)), 200)


if __name__ == "__main__":
    unittest.main()
