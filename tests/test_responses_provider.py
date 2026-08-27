"""Inbound Responses-compatible provider tests."""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-responses-provider-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chat_provider import (
    ActionButton,
    IncomingAction,
    IncomingMessage,
    PollResult,
)
from responses_provider import ResponsesApiProvider, input_text
from responses import build_response_payload, reasoning_summary_delta
from responses import responses_create_stream
from subagents import build_subagent_payload
from model_providers import ModelTarget
import session
from turn_runtime import TurnKind, turns


class ResponsesInputTests(unittest.TestCase):
    def test_reasoning_effort_defaults_are_provider_driven(self) -> None:
        self.assertIsNone(
            responses_create_stream.__kwdefaults__["reasoning_effort"]
        )
        payload = build_subagent_payload(
            ModelTarget("local", "test-model"),
            [{"role": "user", "content": "hello"}],
            allow_subagents=False,
            depth=1,
        )
        self.assertNotIn("reasoning", payload)

    def test_requests_summary_without_exposing_raw_reasoning(self) -> None:
        payload = build_response_payload(
            [{"role": "user", "content": "hello"}],
            reasoning_summary="auto",
            stream=True,
        )
        self.assertEqual(payload["reasoning"], {"summary": "auto"})
        self.assertEqual(
            reasoning_summary_delta({
                "type": "response.reasoning_summary_text.delta",
                "delta": "Safe summary",
            }),
            "Safe summary",
        )
        self.assertEqual(
            reasoning_summary_delta({
                "type": "response.reasoning_text.delta",
                "delta": "Private reasoning",
                "encrypted_content": "ciphertext",
            }),
            "",
        )

    def test_accepts_string_and_responses_message_content(self) -> None:
        self.assertEqual(input_text("hello"), "hello")
        self.assertEqual(
            input_text([
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "/status"},
                    ],
                },
            ]),
            "/status",
        )

    def test_rejects_input_without_user_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "no user text"):
            input_text([{"role": "assistant", "content": "old"}])

    def test_full_history_uses_only_latest_user_turn(self) -> None:
        self.assertEqual(
            input_text([
                {"role": "user", "content": "/new"},
                {
                    "role": "assistant",
                    "content": "New session created.",
                },
                {"role": "user", "content": "hey"},
            ]),
            "hey",
        )


class ResponsesApiProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        session.set_default_chat("responses-test")
        turns.finish()
        self.provider = ResponsesApiProvider(
            host="127.0.0.1",
            port=0,
            api_key="",
        )
        await self.provider.initialize((("stop", "Abort current run"),))
        self.base_url = f"http://127.0.0.1:{self.provider.bound_port}"

    async def asyncTearDown(self) -> None:
        await self.provider.close()
        turns.finish()

    async def _poll(self) -> PollResult:
        batch = await self.provider.poll(None, timeout=1)
        for event in batch.events:
            self.provider.prepare_event(event)
        return batch

    async def test_nonloopback_bind_requires_api_key(self) -> None:
        await self.provider.close()
        self.provider = ResponsesApiProvider(
            host="0.0.0.0",
            port=0,
            api_key="",
        )

        with self.assertRaisesRegex(RuntimeError, "API_KEY is required"):
            await self.provider.initialize(())

    async def _answer_once(self, text: str = "agent answer") -> IncomingMessage:
        batch = await self._poll()
        event = batch.events[0]
        self.assertIsInstance(event, IncomingMessage)
        await self.provider.send_text(event.conversation_id, text)
        return event

    async def test_nonstream_response_round_trip(self) -> None:
        answer = asyncio.create_task(self._answer_once())
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url + "/v1/responses",
                json={"model": "ignored", "input": "/status"},
            )
        event = await answer
        self.assertEqual(event.text, "/status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["output_text"], "agent answer")
        self.assertEqual(
            body["output"][0]["content"][0]["text"],
            "agent answer",
        )

    async def test_background_output_is_retained_without_active_request(
        self,
    ) -> None:
        await self.provider.send_text("background", "finished later")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.base_url + "/v1/status",
                params={"conversation": "background"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["text"], "finished later")
        self.assertTrue(response.json()["data"][0]["asynchronous"])

    async def test_async_inbox_does_not_evict_live_status(self) -> None:
        await self.provider.close()
        self.provider = ResponsesApiProvider(
            host="127.0.0.1",
            port=0,
            max_status_messages=1,
        )
        await self.provider.initialize(())
        message_id = await self.provider.create_message("alpha", "live")
        await self.provider.pin_message("alpha", message_id)

        await self.provider.send_text("alpha", "background one")
        await self.provider.send_text("alpha", "background two")
        await self.provider.send_text("beta", "other conversation")

        self.assertIn(message_id, self.provider.messages)
        self.assertEqual(len(self.provider.inbox["alpha"]), 1)
        self.assertEqual(len(self.provider.inbox["beta"]), 1)

    async def test_stream_event_buffer_is_bounded_and_keeps_completion(
        self,
    ) -> None:
        await self.provider.close()
        self.provider = ResponsesApiProvider(
            host="127.0.0.1",
            port=0,
            max_stream_events=4,
        )
        await self.provider.initialize(())
        pending = await self.provider._queue_message({
            "input": "hello",
            "stream": True,
        })
        await self._poll()
        for index in range(20):
            await self.provider.send_reasoning_summary("default", str(index))
        await self.provider.send_text("default", "done")

        self.assertLessEqual(pending.events.qsize(), 4)
        events = []
        while not pending.events.empty():
            events.append(pending.events.get_nowait())
            pending.events.task_done()
        self.assertIn("response.output_text.done", [item["type"] for item in events])
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_pending_request_capacity_returns_stable_overload(
        self,
    ) -> None:
        await self.provider.close()
        self.provider = ResponsesApiProvider(
            host="127.0.0.1",
            port=0,
            api_key="",
            max_pending=1,
        )
        await self.provider.initialize(())
        self.base_url = f"http://127.0.0.1:{self.provider.bound_port}"

        async with httpx.AsyncClient() as client:
            first = asyncio.create_task(
                client.post(
                    self.base_url + "/v1/responses",
                    json={"input": "first"},
                )
            )
            batch = await self._poll()
            overloaded = await client.post(
                self.base_url + "/v1/responses",
                json={"input": "second"},
            )
            await self.provider.send_text(
                batch.events[0].conversation_id,
                "done",
            )
            await first

        self.assertEqual(overloaded.status_code, 429)
        self.assertEqual(
            overloaded.json()["error"]["code"],
            "agent_overloaded",
        )

    async def test_request_types_are_not_silently_coerced(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url + "/v1/responses",
                json={"input": "hello", "stream": "false"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("stream must be a boolean", response.text)

    async def test_request_read_timeout_is_bounded(self) -> None:
        await self.provider.close()
        self.provider = ResponsesApiProvider(
            host="127.0.0.1",
            port=0,
            api_key="",
            read_timeout=0.01,
        )
        await self.provider.initialize(())
        reader, writer = await asyncio.open_connection(
            "127.0.0.1",
            self.provider.bound_port,
        )
        try:
            response = await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()

        self.assertIn(b"408 Request Timeout", response)

    async def test_connection_capacity_returns_service_unavailable(self) -> None:
        self.provider._active_connections = self.provider.max_connections
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.base_url + "/healthz")
        finally:
            self.provider._active_connections = 0

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["error"]["code"],
            "server_overloaded",
        )

    async def test_internal_errors_are_correlated_but_not_exposed(
        self,
    ) -> None:
        with patch.object(
            self.provider,
            "_queue_message",
            AsyncMock(side_effect=RuntimeError("private filesystem path")),
        ):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url + "/v1/responses",
                    json={"input": "hello"},
                )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("private filesystem path", response.text)
        self.assertRegex(response.json()["error"]["request_id"], r"^req_")

    async def test_status_message_retention_is_bounded(self) -> None:
        await self.provider.close()
        self.provider = ResponsesApiProvider(
            host="127.0.0.1",
            port=0,
            api_key="",
            max_status_messages=2,
        )
        await self.provider.initialize(())
        self.base_url = f"http://127.0.0.1:{self.provider.bound_port}"

        await self.provider.create_message("default", "one")
        await self.provider.create_message("default", "two")
        await self.provider.create_message("default", "three")

        self.assertEqual(
            [message["text"] for message in self.provider.messages.values()],
            ["two", "three"],
        )

    async def test_stream_emits_deltas_status_and_completion(self) -> None:
        async def answer() -> None:
            batch = await self._poll()
            event = batch.events[0]
            await self.provider.create_message(
                event.conversation_id,
                "Working",
            )
            await self.provider.send_reasoning_summary(
                event.conversation_id,
                "Checking the request",
            )
            await self.provider.send_draft(
                event.conversation_id,
                1,
                "partial",
            )
            await self.provider.send_text(
                event.conversation_id,
                "partial answer",
            )

        task = asyncio.create_task(answer())
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.base_url + "/v1/responses",
                json={"input": "hello", "stream": True},
            ) as response:
                wire = "\n".join([line async for line in response.aiter_lines()])
        await task
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: response.created", wire)
        self.assertIn("event: raptor.status", wire)
        self.assertIn("event: response.reasoning_summary_text.delta", wire)
        self.assertIn("event: response.reasoning_summary_text.done", wire)
        self.assertIn('"type":"summary_text"', wire)
        self.assertIn('"text":"Checking the request"', wire)
        self.assertIn('"delta":"partial"', wire)
        self.assertIn('"delta":" answer"', wire)
        self.assertIn("event: response.output_text.done", wire)
        self.assertIn("event: response.completed", wire)
        self.assertIn(
            'event: response.reasoning_summary_text.delta\n'
            'data: {"type":"response.reasoning_summary_text.delta",'
            '"item_id":',
            wire,
        )
        self.assertIn('"output_index":1', wire)

    async def test_stream_sends_heartbeat_while_agent_is_silent(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.data = bytearray()

            def write(self, value: bytes) -> None:
                self.data.extend(value)

            async def drain(self) -> None:
                return None

        pending = await self.provider._queue_message({
            "input": "long task",
            "stream": True,
        })
        writer = Writer()
        with patch("responses_provider.SSE_HEARTBEAT_SECONDS", 0.01):
            task = asyncio.create_task(
                self.provider._write_sse(writer, pending)
            )
            await asyncio.sleep(0.025)
            pending.events.put_nowait({
                "type": "response.completed",
                "response": {"status": "completed"},
            })
            await task
        self.assertIsNone(pending.completed)
        self.assertIn(b": keep-alive\n\n", writer.data)

    async def test_action_endpoint_queues_provider_action(self) -> None:
        async with httpx.AsyncClient() as client:
            request = asyncio.create_task(
                client.post(
                    self.base_url + "/v1/actions",
                    json={"data": "approval:abc:approve"},
                )
            )
            batch = await self._poll()
            event = batch.events[0]
            self.assertIsInstance(event, IncomingAction)
            self.assertEqual(event.data, "approval:abc:approve")
            await self.provider.answer_action(event.action_id, "Approved")
            self.assertFalse(request.done())
            await self.provider.finish_event(event)
            response = await request
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "accepted")
        self.assertEqual(response.json()["message"], "Approved")

    async def test_routes_named_conversation(self) -> None:
        async with httpx.AsyncClient() as client:
            request = asyncio.create_task(
                client.post(
                    self.base_url + "/v1/responses",
                    json={"input": "hello", "conversation": "other"},
                )
            )
            batch = await self._poll()
            event = batch.events[0]
            self.assertEqual(event.conversation_id, "other")
            await self.provider.send_text(event.conversation_id, "answer")
            response = await request
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["output_text"], "answer")

    async def test_send_without_active_request_is_retained_as_status(
        self,
    ) -> None:
        await self.provider.send_text("default", "orphaned")

        self.assertEqual(self.provider.messages, {})
        self.assertEqual(
            [
                message["text"]
                for message in self.provider.inbox["default"].values()
            ],
            ["orphaned"],
        )

    async def test_delivery_context_is_scoped_to_its_conversation(self) -> None:
        pending = await self.provider._queue_message({
            "input": "hello",
            "conversation": "alpha",
        })
        batch = await self._poll()
        with self.assertRaisesRegex(ValueError, "another conversation"):
            self.provider.capture_delivery_context("beta")
        context = self.provider.capture_delivery_context("alpha")
        with self.assertRaisesRegex(ValueError, "another conversation"):
            self.provider.activate_delivery_context("beta", context)
        await self.provider.send_text("alpha", "done")
        assert pending.completed is not None
        await pending.completed

    async def test_status_messages_are_scoped_to_their_conversation(
        self,
    ) -> None:
        message_id = await self.provider.create_message("alpha", "Working")
        operations = (
            self.provider.edit_message("beta", message_id, "Changed"),
            self.provider.delete_message("beta", message_id),
            self.provider.pin_message("beta", message_id),
            self.provider.unpin_message("beta", message_id),
        )
        for operation in operations:
            with self.assertRaisesRegex(ValueError, "another conversation"):
                await operation
        self.assertEqual(
            self.provider.messages[message_id]["conversation_id"],
            "alpha",
        )

    async def test_busy_noncommand_waits_for_steered_response(self) -> None:
        turns.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
        )
        blocker = turns.task
        assert blocker is not None
        try:
            async with httpx.AsyncClient() as client:
                request = asyncio.create_task(
                    client.post(
                        self.base_url + "/v1/responses",
                        json={"input": "change direction"},
                    )
                )
                batch = await self._poll()
                self.assertEqual(batch.events[0].text, "change direction")
                delivery_context = self.provider.capture_delivery_context(
                    batch.events[0].conversation_id,
                )
                await self.provider.acknowledge_queued_message(
                    batch.events[0].conversation_id,
                )
                self.assertFalse(request.done())
                token = self.provider.activate_delivery_context(
                    batch.events[0].conversation_id,
                    delivery_context,
                )
                try:
                    await self.provider.send_text(
                        batch.events[0].conversation_id,
                        "steered answer",
                    )
                finally:
                    self.provider.restore_delivery_context(token)
                response = await request
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["output_text"], "steered answer")
        finally:
            blocker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await blocker
            turns.finish(blocker)

    async def test_concurrent_requests_are_dispatched_in_order(
        self,
    ) -> None:
        async with httpx.AsyncClient() as client:
            first = asyncio.create_task(
                client.post(
                    self.base_url + "/v1/responses",
                    json={"input": "first"},
                )
            )
            second = asyncio.create_task(
                client.post(
                    self.base_url + "/v1/responses",
                    json={"input": "second"},
                )
            )
            first_batch = await self._poll()
            await self.provider.send_text(
                first_batch.events[0].conversation_id,
                "first answer",
            )
            second_batch = await self._poll()
            second_context = self.provider.capture_delivery_context(
                second_batch.events[0].conversation_id,
            )
            await self.provider.acknowledge_queued_message(
                second_batch.events[0].conversation_id,
            )
            token = self.provider.activate_delivery_context(
                second_batch.events[0].conversation_id,
                second_context,
            )
            try:
                await self.provider.send_text(
                    second_batch.events[0].conversation_id,
                    "second answer",
                )
            finally:
                self.provider.restore_delivery_context(token)
            first_response = await first
            second_response = await second
        self.assertEqual(first_batch.events[0].text, "first")
        self.assertEqual(second_batch.events[0].text, "second")
        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(first_response.json()["output_text"], "first answer")
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(second_response.json()["output_text"], "second answer")

    async def test_streaming_steering_waits_for_real_completion(self) -> None:
        pending = await self.provider._queue_message({
            "input": "first",
            "stream": True,
        })
        await self._poll()
        delivery_context = self.provider.capture_delivery_context(
            self.provider.primary_conversation_id,
        )
        await self.provider.acknowledge_queued_message(
            self.provider.primary_conversation_id,
        )
        self.assertTrue(pending.events.empty())
        token = self.provider.activate_delivery_context(
            self.provider.primary_conversation_id,
            delivery_context,
        )
        try:
            await self.provider.send_text(
                self.provider.primary_conversation_id,
                "final",
            )
        finally:
            self.provider.restore_delivery_context(token)
        events = []
        while True:
            event = await pending.events.get()
            pending.events.task_done()
            events.append(event)
            if event["type"] == "response.completed":
                break
        self.assertEqual(events[-1]["response"]["output_text"], "final")
        self.assertIsNone(pending.completed)

    async def test_undelivered_event_fails_instead_of_hanging(self) -> None:
        pending = await self.provider._queue_message({"input": "hello"})
        batch = await self._poll()

        await self.provider.finish_event(batch.events[0])

        assert pending.completed is not None
        with self.assertRaisesRegex(RuntimeError, "without a response"):
            await pending.completed

    async def test_status_snapshot_preserves_pin_and_actions(self) -> None:
        message_id = await self.provider.create_message(
            self.provider.primary_conversation_id,
            "Approval required",
            ((ActionButton("Approve", "approval:a:approve"),),),
        )
        await self.provider.pin_message(
            self.provider.primary_conversation_id,
            message_id,
        )
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url + "/v1/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["conversation"],
            self.provider.primary_conversation_id,
        )
        self.assertEqual(response.json()["data"], [{
            "conversation_id": self.provider.primary_conversation_id,
            "message_id": message_id,
            "text": "Approval required",
            "actions": [{
                "label": "Approve",
                "data": "approval:a:approve",
            }],
            "pinned": True,
        }])

    async def test_status_snapshot_is_scoped_to_named_conversation(
        self,
    ) -> None:
        default_id = await self.provider.create_message("default", "Default")
        named_id = await self.provider.create_message("project-a", "Named")
        async with httpx.AsyncClient() as client:
            default_response = await client.get(self.base_url + "/v1/status")
            named_response = await client.get(
                self.base_url + "/v1/status",
                params={"conversation": "project-a"},
            )
        self.assertEqual(
            [item["message_id"] for item in default_response.json()["data"]],
            [default_id],
        )
        self.assertEqual(
            [item["message_id"] for item in named_response.json()["data"]],
            [named_id],
        )

    async def test_lists_the_agent_model(self) -> None:
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url + "/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "raptor")

    async def test_bearer_auth_when_configured(self) -> None:
        await self.provider.close()
        self.provider = ResponsesApiProvider(
            host="127.0.0.1",
            port=0,
            api_key="secret",
        )
        await self.provider.initialize(())
        self.base_url = f"http://127.0.0.1:{self.provider.bound_port}"
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url + "/healthz")
            allowed = await client.get(
                self.base_url + "/healthz",
                headers={"Authorization": "Bearer secret"},
            )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(allowed.status_code, 200)


if __name__ == "__main__":
    unittest.main()
