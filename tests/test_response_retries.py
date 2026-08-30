"""Transient Responses backend retry tests."""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
from response_errors import MalformedToolCallError

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-response-retry-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import responses
from model_providers import (
    ModelConfiguration,
    ModelProvider,
    ModelTarget,
)

TARGET = ModelTarget("test", "model-a")


class ResponseRetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        configuration = ModelConfiguration(
            providers={
                "test": ModelProvider(
                    id="test",
                    base_url="http://backend/v1",
                    default_model="model-a",
                    request_max_retries=3,
                    retry_base_seconds=0,
                )
            },
            default_target=TARGET,
        )
        provider_patch = patch.object(
            responses,
            "MODEL_CONFIGURATION",
            configuration,
        )
        provider_patch.start()
        self.addCleanup(provider_patch.stop)

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
            patch.object(responses, "_responses_create_stream_once", request),
            patch.object(responses, "log_event"),
        ):
            result = await responses.responses_create_stream(TARGET, 1, [])
        self.assertEqual(result, completed)
        self.assertEqual(request.await_count, 2)

    async def test_stream_forwards_tool_activity_callback(self) -> None:
        completed = {"status": "completed", "output": []}
        request = AsyncMock(return_value=completed)
        tool_call = AsyncMock()
        with patch.object(
            responses,
            "_responses_create_stream_once",
            request,
        ):
            result = await responses.responses_create_stream(
                TARGET,
                1,
                [],
                on_tool_call=tool_call,
            )

        self.assertEqual(result, completed)
        self.assertIs(request.await_args.kwargs["on_tool_call"], tool_call)

    async def test_exhausted_incomplete_stream_is_transient(self) -> None:
        request = AsyncMock(
            side_effect=responses.IncompleteResponsesStreamError("early close")
        )
        with (
            patch.object(responses, "_responses_create_stream_once", request),
            patch.object(responses, "log_event"),
            patch.object(
                responses,
                "model_provider",
                return_value=Mock(
                    request_max_retries=1,
                    retry_base_seconds=0,
                ),
            ),
        ):
            with self.assertRaises(responses.TransientResponsesError):
                await responses.responses_create_stream(TARGET, 1, [])
        self.assertEqual(request.await_count, 2)

    async def test_context_overflow_bypasses_transport_retries(self) -> None:
        request = AsyncMock(
            side_effect=responses.ContextLengthError("too large")
        )
        with patch.object(responses, "_responses_create_stream_once", request):
            with self.assertRaises(responses.ContextLengthError):
                await responses.responses_create_stream(TARGET, 1, [])
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
        with patch.object(responses, "_responses_create_stream_once", request):
            with self.assertRaises(MalformedToolCallError):
                await responses.responses_create_stream(TARGET, 1, [])
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
        ):
            with self.assertRaises(MalformedToolCallError):
                await responses._stateless_response_once(TARGET, [])

    async def test_remote_disconnect_retries_three_times(self) -> None:
        request = AsyncMock(
            side_effect=httpx.RemoteProtocolError("server disconnected")
        )
        with (
            patch.object(responses, "_responses_create_stream_once", request),
            patch.object(responses, "log_event"),
        ):
            with self.assertRaises(responses.TransientResponsesError):
                await responses.responses_create_stream(TARGET, 1, [])
        self.assertEqual(request.await_count, 4)

    async def test_retry_after_delays_retry_without_extending_attempts(
        self,
    ) -> None:
        request_value = httpx.Request(
            "POST",
            "http://backend/v1/responses",
        )
        error = httpx.HTTPStatusError(
            "rate limited",
            request=request_value,
            response=httpx.Response(
                429,
                headers={"Retry-After": "7"},
                request=request_value,
            ),
        )
        request = AsyncMock(side_effect=[error, "done"])
        sleep = AsyncMock()

        with patch.object(responses.asyncio, "sleep", sleep):
            result = await responses.retry_transient_response(
                request,
                operation="test",
                max_retries=1,
                retry_base_seconds=0.5,
            )

        self.assertEqual(result, "done")
        sleep.assert_awaited_once_with(7.0)

    async def test_partial_public_stream_is_not_replayed(self) -> None:
        async def request(*_args, replay_guard, **_kwargs):
            replay_guard.public_output_seen = True
            raise responses.IncompleteResponsesStreamError("disconnected")

        attempt = AsyncMock(side_effect=request)
        with (
            patch.object(responses, "_responses_create_stream_once", attempt),
        ):
            with self.assertRaises(responses.PartialResponsesStreamError):
                await responses.responses_create_stream(TARGET, 1, [])

        self.assertEqual(attempt.await_count, 1)

    async def test_model_listing_uses_common_retry_policy(self) -> None:
        request = AsyncMock(
            side_effect=[httpx.ConnectError("offline"), ["model-a"]]
        )
        with (
            patch.object(responses, "_list_models_once", request),
            patch.object(responses, "log_event"),
        ):
            result = await responses.list_models("test")
        self.assertEqual(result, ["model-a"])
        self.assertEqual(request.await_count, 2)

    async def test_model_listing_can_disable_retries_for_status(self) -> None:
        request = AsyncMock(side_effect=httpx.ConnectError("offline"))
        with (
            patch.object(responses, "_list_models_once", request),
            patch.object(responses, "log_event"),
        ):
            with self.assertRaises(responses.TransientResponsesError):
                await responses.list_models("test", max_retries=0)
        self.assertEqual(request.await_count, 1)

    async def test_model_listing_has_a_bounded_request_timeout(self) -> None:
        response = Mock()
        response.json.return_value = {"data": [{"id": "model-a"}]}
        client = Mock(get=AsyncMock(return_value=response))
        with patch.object(responses.session, "responses", client, create=True):
            result = await responses._list_models_once("test")
        self.assertEqual(result, ["model-a"])
        timeout = client.get.await_args.kwargs["timeout"]
        self.assertEqual(timeout.connect, 10.0)
        self.assertEqual(timeout.read, 10.0)
        self.assertEqual(timeout.write, 10.0)
        self.assertEqual(timeout.pool, 10.0)

    async def test_stateless_request_uses_common_retry_policy(self) -> None:
        expected = {"status": "completed", "output": []}
        request = AsyncMock(
            side_effect=[httpx.ReadError("disconnected"), expected]
        )
        with (
            patch.object(responses, "_stateless_response_once", request),
            patch.object(responses, "log_event"),
        ):
            result = await responses.stateless_response(TARGET, [])
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
            patch.object(responses, "send_draft", slow_draft),
            patch.object(responses, "CHAT_STREAMING", True),
            patch.object(responses, "CHAT_STREAM_INTERVAL", 0),
        ):
            task = asyncio.create_task(
                responses._responses_create_stream_once(TARGET, 1, [])
            )
            await asyncio.wait_for(stream_consumed.wait(), timeout=1)
            self.assertFalse(task.done())
            release_draft.set()
            result = await asyncio.wait_for(task, timeout=1)

        self.assertEqual(result["status"], "completed")

    async def test_failed_stream_cancels_owned_draft_task(self) -> None:
        draft_started = asyncio.Event()
        draft_cancelled = asyncio.Event()

        async def fail_stream(*_args, on_text, **_kwargs):
            await on_text("partial")
            await draft_started.wait()
            raise httpx.ReadError("disconnected")

        async def blocked_draft(*_args):
            draft_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                draft_cancelled.set()
                raise

        with (
            patch.object(responses, "stream_response_payload", fail_stream),
            patch.object(responses, "send_draft", blocked_draft),
            patch.object(responses, "CHAT_STREAMING", True),
            patch.object(responses, "CHAT_STREAM_INTERVAL", 0),
        ):
            with self.assertRaises(httpx.ReadError):
                await responses._responses_create_stream_once(TARGET, 1, [])

        self.assertTrue(draft_cancelled.is_set())

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

    async def test_stream_exposes_cumulative_tool_call_arguments(self) -> None:
        class FakeResponse:
            is_error = False

            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                yield (
                    'data: {"type":"response.output_item.added",'
                    '"output_index":0,"item":{"type":"function_call",'
                    '"id":"fc1","call_id":"c1","name":"shell",'
                    '"arguments":""}}'
                )
                yield (
                    'data: {"type":"response.function_call_arguments.delta",'
                    '"output_index":0,"item_id":"fc1",'
                    '"delta":"{\\"command\\":"}'
                )
                yield (
                    'data: {"type":"response.function_call_arguments.delta",'
                    '"output_index":0,"item_id":"fc1",'
                    '"delta":"\\"pwd\\"}"}'
                )
                yield (
                    'data: {"type":"response.function_call_arguments.done",'
                    '"output_index":0,"item_id":"fc1",'
                    '"arguments":"{\\"command\\":\\"pwd\\"}"}'
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

        tool_call = AsyncMock()
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
                on_tool_call=tool_call,
            )

        self.assertEqual(result["status"], "completed")
        updates = [call.args for call in tool_call.await_args_list]
        self.assertEqual(
            [update[0]["arguments"] for update in updates],
            ["", '{"command":', '{"command":"pwd"}', '{"command":"pwd"}'],
        )
        self.assertEqual(
            [update[1] for update in updates],
            [False, False, False, True],
        )


if __name__ == "__main__":
    unittest.main()
