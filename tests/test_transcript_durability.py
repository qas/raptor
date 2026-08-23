"""Transcript durability during agent turns."""
import copy
import os
import sys
import tempfile
import unittest
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
import session
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
        sid = chat_store.create_session(kind="main")
        session.state["current_session_id"] = sid
        session.active_task = None
        session.active_since = None
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

    async def test_recovery_prompt_does_not_embed_raw_tool_events(
        self,
    ) -> None:
        huge_output = "SECRET-RAW-TOOL-OUTPUT" * 20_000
        session.state["interrupted_agent"] = {
            "session_id": "root-session",
            "interrupted_at": 123.0,
            "tool_events": [{"output": huge_output}],
            "resumed_from": {
                "tool_events": [{"output": huge_output}],
            },
        }
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
        self.assertIn('"session_id": "root-session"', instructions)
        self.assertIn('"session_id": "sub-session"', instructions)

    def test_reinterruption_keeps_resumed_from_bounded(self) -> None:
        huge_output = "x" * 500_000
        previous = {
            "session_id": "old-session",
            "interrupted_at": 123.0,
            "tool_events": [{"output": huge_output}],
            "resumed_from": {
                "tool_events": [{"output": huge_output}],
            },
        }

        ref = agent_mod._recovery_checkpoint_ref(previous)

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
