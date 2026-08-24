"""Transient Responses backend retry tests."""
import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from response_errors import MalformedToolCallError

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

    def test_malformed_tool_arguments_are_model_output_failure(self) -> None:
        response = httpx.Response(
            500,
            json={
                "error": {
                    "message": (
                        "Failed to parse tool call arguments as JSON: "
                        "invalid string"
                    )
                }
            },
        )

        error = responses.parse_http_response_error(response)

        self.assertIsInstance(error, MalformedToolCallError)
        assert error is not None
        self.assertFalse(responses.is_transient_responses_error(error))

    async def test_malformed_tool_call_is_not_retried(self) -> None:
        request = AsyncMock(
            side_effect=MalformedToolCallError("invalid generated arguments")
        )
        with (
            patch.object(responses, "RESPONSES_MAX_RETRIES", 3),
            patch.object(responses, "_responses_create_stream_once", request),
        ):
            with self.assertRaises(MalformedToolCallError):
                await responses.responses_create_stream(1, [])
        self.assertEqual(request.await_count, 1)

    async def test_stream_classifies_malformed_tool_call_response(self) -> None:
        def reject(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                500,
                request=request,
                json={
                    "error": {
                        "message": (
                            "Failed to parse function_call arguments as "
                            "JSON: syntax error"
                        )
                    }
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(reject)
        ) as client:
            with patch.object(
                responses.session,
                "responses",
                client,
                create=True,
            ):
                with self.assertRaises(MalformedToolCallError):
                    await responses.stream_response_payload(
                        url="http://backend/v1/responses",
                        headers={},
                        payload={"stream": True},
                    )

    async def test_stateless_request_classifies_malformed_tool_call(self) -> None:
        response = httpx.Response(
            400,
            json={
                "error": {
                    "message": (
                        "Could not parse tool_call arguments as JSON: "
                        "invalid value"
                    )
                }
            },
        )
        client = AsyncMock()
        client.post.return_value = response

        with (
            patch.object(
                responses.session,
                "responses",
                client,
                create=True,
            ),
            patch.object(responses, "state", {"model": "model-a"}),
        ):
            with self.assertRaises(MalformedToolCallError):
                await responses._stateless_response_once([])

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

    async def test_stream_exposes_cumulative_public_output(self) -> None:
        class FakeResponse:
            is_error = False

            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                yield (
                    'data: {"type":"response.reasoning_summary_text.delta",'
                    '"delta":"Check"}'
                )
                yield (
                    'data: {"type":"response.reasoning_summary_text.delta",'
                    '"delta":" files"}'
                )
                yield (
                    'data: {"type":"response.output_text.delta",'
                    '"delta":"Found"}'
                )
                yield (
                    'data: {"type":"response.output_text.delta",'
                    '"delta":" it"}'
                )
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

        text = AsyncMock()
        reasoning = AsyncMock()
        with patch.object(
            responses.session,
            "responses",
            FakeClient(),
            create=True,
        ):
            result = await responses.stream_response_payload(
                url="http://backend/v1/responses",
                headers={},
                payload={"stream": True},
                on_text=text,
                on_reasoning_summary=reasoning,
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            [call.args[0] for call in reasoning.await_args_list],
            ["Check", "Check files"],
        )
        self.assertEqual(
            [call.args[0] for call in text.await_args_list],
            ["Found", "Found it"],
        )


if __name__ == "__main__":
    unittest.main()
