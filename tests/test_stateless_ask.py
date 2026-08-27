"""Tests for the transcript-free ``/ask`` side channel."""
import copy
import contextlib
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-ask-tests-"))
os.environ["TG_BOT_TOKEN"] = "test-token"
os.environ["TG_USER_ID"] = "1"
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import chat_store
import commands
import responses
import session
from model_providers import ModelTarget


class _Response:
    is_error = False

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"output": []}


class _Client:
    def __init__(self) -> None:
        self.payload = None

    async def post(self, _url, **kwargs):
        self.payload = kwargs["json"]
        return _Response()


class StatelessAskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        session.set_default_model_target(ModelTarget("local", "model-a"))
        session.set_default_chat("ask-tests")
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
        session.state["current_session_id"] = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            model_target={"provider_id": "local", "model": "model-a"},
        )
        session.state["model_target"] = {
            "provider_id": "local",
            "model": "model-a",
        }

    def test_payload_has_one_user_message_and_tools(self) -> None:
        tools = [{"type": "function", "name": "read_file"}]
        payload = responses.build_stateless_response_payload(
            [{"role": "user", "content": "side question"}],
            "model-a",
            tools=tools,
        )
        self.assertEqual(
            payload,
            {
                "model": "model-a",
                "input": [
                    {"role": "user", "content": "side question"}
                ],
                "stream": False,
                "tools": tools,
                "parallel_tool_calls": False,
            },
        )
        self.assertNotIn("instructions", payload)

    async def test_request_does_not_mutate_session_state(self) -> None:
        client = _Client()
        before = copy.deepcopy(session.state)

        with patch.object(session, "responses", client, create=True):
            await responses.stateless_response(
                ModelTarget("local", "model-a"),
                [{"role": "user", "content": "side question"}]
            )

        self.assertEqual(session.state, before)
        self.assertEqual(
            client.payload["input"],
            [{"role": "user", "content": "side question"}],
        )

    async def test_command_returns_answer_without_transcript_writes(
        self,
    ) -> None:
        session_id = str(session.state["current_session_id"])
        before_events = chat_store.read_events(session_id)
        reply = {
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "isolated answer",
                }],
            }]
        }
        send = AsyncMock()
        with (
            patch.object(
                commands,
                "stateless_response",
                AsyncMock(return_value=reply),
            ) as create,
            patch.object(commands, "send", send),
            patch.object(commands, "capture_delivery_context", return_value=None),
            patch.object(
                commands,
                "bound_delivery_context",
                side_effect=lambda *_args: contextlib.nullcontext(),
            ),
        ):
            handled = await commands.command(1, "/ask side question")
            task = session.current_runtime().turns.task
            self.assertIsNotNone(task)
            await task

        self.assertTrue(handled)
        create.assert_awaited_once_with(
            ModelTarget("local", "model-a"),
            [{"role": "user", "content": "side question"}]
        )
        send.assert_awaited_once_with(1, "isolated answer")
        self.assertEqual(chat_store.read_events(session_id), before_events)

    async def test_command_executes_tools_only_in_memory(self) -> None:
        session_id = str(session.state["current_session_id"])
        before_events = chat_store.read_events(session_id)
        call = {
            "type": "function_call",
            "name": "read_file",
            "call_id": "call-1",
            "arguments": '{"path":"README.md"}',
        }
        first = {"output": [call]}
        final = {
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "tool-backed answer",
                }],
            }]
        }
        create = AsyncMock(side_effect=[first, final])
        execute = AsyncMock(return_value={"ok": True, "content": "doc"})
        send = AsyncMock()
        with (
            patch.object(commands, "stateless_response", create),
            patch.object(commands, "execute_tool_with_approval", execute),
            patch.object(commands, "send", send),
            patch.object(commands, "capture_delivery_context", return_value=None),
            patch.object(
                commands,
                "bound_delivery_context",
                side_effect=lambda *_args: contextlib.nullcontext(),
            ),
        ):
            handled = await commands.command(1, "/ask inspect the readme")
            task = session.current_runtime().turns.task
            self.assertIsNotNone(task)
            await task

        self.assertTrue(handled)
        execute.assert_awaited_once()
        second_work = create.await_args_list[1].args[1]
        self.assertEqual(second_work[0]["content"], "inspect the readme")
        self.assertEqual(second_work[1], call)
        self.assertEqual(second_work[2]["type"], "function_call_output")
        send.assert_awaited_once_with(1, "tool-backed answer")
        self.assertEqual(chat_store.read_events(session_id), before_events)

    async def test_command_returns_before_stateless_request_finishes(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def respond(_target, _work):
            started.set()
            await release.wait()
            return {
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                }]
            }

        with (
            patch.object(commands, "stateless_response", respond),
            patch.object(commands, "send", AsyncMock()),
            patch.object(commands, "capture_delivery_context", return_value=None),
            patch.object(
                commands,
                "bound_delivery_context",
                side_effect=lambda *_args: contextlib.nullcontext(),
            ),
        ):
            handled = await asyncio.wait_for(
                commands.command(1, "/ask side question"), timeout=0.1
            )
            await started.wait()
            task = session.current_runtime().turns.task
            self.assertTrue(handled)
            self.assertIsNotNone(task)
            self.assertFalse(task.done())
            release.set()
            await task


if __name__ == "__main__":
    unittest.main()
