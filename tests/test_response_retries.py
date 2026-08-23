"""Transient Responses backend retry tests."""
import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import responses


class ResponseRetryTests(unittest.IsolatedAsyncioTestCase):
    def test_response_payload_rejects_instruction_roles_in_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "instructions field"):
            responses.build_response_payload(
                [
                    {"role": "user", "content": "hello"},
                    {"role": "developer", "content": "late instruction"},
                ]
            )

    async def test_incomplete_stream_retries_then_succeeds(self) -> None:
        completed = {"status": "completed", "output": []}
        request = AsyncMock(
            side_effect=[
                responses.IncompleteResponsesStreamError("early close"),
                completed,
            ]
        )
        with (
            patch.object(responses, "RESPONSES_MAX_RETRIES", 3),
            patch.object(responses, "RESPONSES_RETRY_BASE_SECONDS", 0),
            patch.object(responses, "_responses_create_stream_once", request),
            patch.object(responses, "log_event"),
        ):
            result = await responses.responses_create_stream(1, [])
        self.assertEqual(result, completed)
        self.assertEqual(request.await_count, 2)

    async def test_exhausted_incomplete_stream_is_transient(self) -> None:
        request = AsyncMock(
            side_effect=responses.IncompleteResponsesStreamError("early close")
        )
        with (
            patch.object(responses, "RESPONSES_MAX_RETRIES", 1),
            patch.object(responses, "RESPONSES_RETRY_BASE_SECONDS", 0),
            patch.object(responses, "_responses_create_stream_once", request),
            patch.object(responses, "log_event"),
        ):
            with self.assertRaises(responses.TransientResponsesError):
                await responses.responses_create_stream(1, [])
        self.assertEqual(request.await_count, 2)

    async def test_context_overflow_bypasses_transport_retries(self) -> None:
        request = AsyncMock(
            side_effect=responses.ContextLengthError("too large")
        )
        with (
            patch.object(responses, "RESPONSES_MAX_RETRIES", 3),
            patch.object(responses, "_responses_create_stream_once", request),
        ):
            with self.assertRaises(responses.ContextLengthError):
                await responses.responses_create_stream(1, [])
        self.assertEqual(request.await_count, 1)

    def test_retryable_http_statuses_are_classified_narrowly(self) -> None:
        request = httpx.Request("POST", "http://backend/v1/responses")
        retryable = httpx.HTTPStatusError(
            "unavailable",
            request=request,
            response=httpx.Response(503, request=request),
        )
        terminal = httpx.HTTPStatusError(
            "bad request",
            request=request,
            response=httpx.Response(400, request=request),
        )
        self.assertTrue(responses.is_transient_responses_error(retryable))
        self.assertFalse(responses.is_transient_responses_error(terminal))

    async def test_remote_disconnect_retries_three_times(self) -> None:
        request = AsyncMock(
            side_effect=httpx.RemoteProtocolError("server disconnected")
        )
        with (
            patch.object(responses, "RESPONSES_MAX_RETRIES", 3),
            patch.object(responses, "RESPONSES_RETRY_BASE_SECONDS", 0),
            patch.object(responses, "_responses_create_stream_once", request),
            patch.object(responses, "log_event"),
        ):
            with self.assertRaises(responses.TransientResponsesError):
                await responses.responses_create_stream(1, [])
        self.assertEqual(request.await_count, 4)

    async def test_model_listing_uses_common_retry_policy(self) -> None:
        request = AsyncMock(
            side_effect=[httpx.ConnectError("offline"), ["model-a"]]
        )
        with (
            patch.object(responses, "RESPONSES_RETRY_BASE_SECONDS", 0),
            patch.object(responses, "_list_models_once", request),
            patch.object(responses, "log_event"),
        ):
            result = await responses.list_models()
        self.assertEqual(result, ["model-a"])
        self.assertEqual(request.await_count, 2)

    async def test_stateless_request_uses_common_retry_policy(self) -> None:
        expected = {"status": "completed", "output": []}
        request = AsyncMock(
            side_effect=[httpx.ReadError("disconnected"), expected]
        )
        with (
            patch.object(responses, "RESPONSES_RETRY_BASE_SECONDS", 0),
            patch.object(responses, "_stateless_response_once", request),
            patch.object(responses, "log_event"),
        ):
            result = await responses.stateless_response([])
        self.assertEqual(result, expected)
        self.assertEqual(request.await_count, 2)

    async def test_slow_draft_does_not_block_stream_consumption(self) -> None:
        stream_consumed = asyncio.Event()
        release_draft = asyncio.Event()

        class FakeResponse:
            is_error = False

            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                yield (
                    'data: {"type":"response.output_text.delta",'
                    '"delta":"hello"}'
                )
                stream_consumed.set()
                yield (
                    'data: {"type":"response.completed","response":'
                    '{"status":"completed","output":[]}}'
                )

        class FakeStream:
            async def __aenter__(self):
                return FakeResponse()

            async def __aexit__(self, *_args):
                return False

        class FakeClient:
            def stream(self, *_args, **_kwargs):
                return FakeStream()

        async def slow_draft(*_args):
            await release_draft.wait()

        with (
            patch.object(
                responses.session,
                "responses",
                FakeClient(),
                create=True,
            ),
            patch.object(responses, "ensure_model", AsyncMock(return_value="m")),
            patch.object(responses, "send_draft", slow_draft),
            patch.object(responses, "CHAT_STREAMING", True),
            patch.object(responses, "CHAT_STREAM_INTERVAL", 0),
        ):
            task = asyncio.create_task(
                responses._responses_create_stream_once(1, [])
            )
            await asyncio.wait_for(stream_consumed.wait(), timeout=1)
            self.assertFalse(task.done())
            release_draft.set()
            result = await asyncio.wait_for(task, timeout=1)

        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
