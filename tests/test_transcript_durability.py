"""Transcript durability during agent turns."""
import asyncio
import copy
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-durable-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ["MODEL_CONTEXT_TOKENS"] = "131072"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import agent as agent_mod
import chat_store
import controller
from response_errors import MalformedToolCallError
import session
from turn_runtime import turns
import subagents


async def _noop(*_a, **_k):
    return None


class TranscriptDurabilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._chat_dir = Path(tempfile.mkdtemp(prefix="chats-"))
        self._chat_patch = patch.object(
            chat_store,
            "CHAT_DIR",
            self._chat_dir,
        )
        self._chat_patch.start()
        self.addCleanup(self._chat_patch.stop)
        chat_store._SEQ_CACHE.clear()
        session.state.clear()
        session.state.update(copy.deepcopy(session.DEFAULT_STATE))
        sid = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
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

    async def test_items_written_during_turn(self) -> None:
        sid = session.state["current_session_id"]
        seen_before_tool: list[int] = []

        async def fake_stream(chat_id, items, extra_instructions=""):
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

        async def answer(_chat_id, work, **_kwargs):
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

        async def fake_stream(chat_id, items, extra_instructions=""):
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

        async def fake_stream(chat_id, items, extra_instructions=""):
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
