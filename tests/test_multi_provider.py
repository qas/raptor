"""Concurrent chat-provider multiplexer tests."""
import asyncio
import os
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chat_provider import (
    IncomingAction,
    IncomingMessage,
    PollResult,
    ProviderCapabilities,
)
from multi_provider import MultiProvider
from responses_provider import ResponsesApiProvider
from activity import ActivitySnapshot


class QueueProvider:
    capabilities = ProviderCapabilities(
        drafts=True,
        pins=True,
        controls=True,
        typing=True,
    )

    def __init__(self, name: str, conversation_id: str) -> None:
        self.name = name
        self.authorized_user_id = name + ":user"
        self.primary_conversation_id = conversation_id
        self.events: asyncio.Queue = asyncio.Queue()
        self.calls: list[tuple] = []
        self.cursor = 0

    @staticmethod
    def encode_conversation_id(conversation_id) -> str:
        return str(conversation_id)

    @staticmethod
    def decode_conversation_id(value: str) -> str:
        return value

    def prepare_event(self, event) -> None:
        self.calls.append(("prepare_event", event))

    async def initialize(self, commands) -> None:
        self.calls.append(("initialize", commands))

    async def close(self) -> None:
        self.calls.append(("close",))

    async def poll(self, cursor, *, timeout: int) -> PollResult:
        del cursor, timeout
        event = await self.events.get()
        self.events.task_done()
        self.cursor += 1
        return PollResult((event,), self.cursor)

    async def send_text(self, conversation_id, text: str) -> None:
        self.calls.append(("send", conversation_id, text))

    async def send_draft(self, conversation_id, draft_id, text) -> None:
        self.calls.append(("draft", conversation_id, draft_id, text))

    async def send_reasoning_summary(
        self, conversation_id, delta: str,
    ) -> None:
        self.calls.append(("reasoning", conversation_id, delta))

    async def create_message(self, conversation_id, text, controls=()):
        self.calls.append(("create", conversation_id, text, controls))
        return self.name + ":message"

    async def edit_message(
        self, conversation_id, message_id, text, controls=(),
    ) -> None:
        self.calls.append(
            ("edit", conversation_id, message_id, text, controls)
        )

    async def delete_message(self, conversation_id, message_id) -> None:
        self.calls.append(("delete", conversation_id, message_id))

    async def pin_message(self, conversation_id, message_id) -> None:
        self.calls.append(("pin", conversation_id, message_id))

    async def unpin_message(self, conversation_id, message_id) -> None:
        self.calls.append(("unpin", conversation_id, message_id))

    async def set_typing(self, conversation_id, active: bool) -> None:
        self.calls.append(("typing", conversation_id, active))

    async def reject_busy_message(self, conversation_id) -> bool:
        self.calls.append(("reject_busy", conversation_id))
        return False

    async def acknowledge_queued_message(self, conversation_id) -> None:
        self.calls.append(("acknowledge_queued", conversation_id))

    async def finish_event(self, event) -> None:
        self.calls.append(("finish_event", event))

    def capture_delivery_context(self, conversation_id):
        self.calls.append(("capture_delivery_context", conversation_id))
        return self.name + ":delivery"

    def activate_delivery_context(self, conversation_id, delivery_context):
        self.calls.append(
            ("activate_delivery_context", conversation_id, delivery_context)
        )
        return self.name + ":token"

    def restore_delivery_context(self, token) -> None:
        self.calls.append(("restore_delivery_context", token))

    async def answer_action(
        self, action_id, text="", *, alert: bool = False,
    ) -> None:
        self.calls.append(("answer", action_id, text, alert))

    async def open_activity_surface(
        self,
        conversation_id,
        snapshot,
        existing_surface_id=None,
    ):
        self.calls.append(
            (
                "activity_open",
                conversation_id,
                snapshot,
                existing_surface_id,
            )
        )
        return "topic/message"

    async def update_activity_surface(
        self,
        conversation_id,
        surface_id,
        snapshot,
    ) -> None:
        self.calls.append(
            ("activity_update", conversation_id, surface_id, snapshot)
        )

    async def close_activity_surface(
        self,
        conversation_id,
        surface_id,
        snapshot,
    ) -> None:
        self.calls.append(
            ("activity_close", conversation_id, surface_id, snapshot)
        )

    def restore_activity_surface(self, conversation_id, surface_id) -> None:
        self.calls.append(("activity_restore", conversation_id, surface_id))


class MultiProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.telegram = QueueProvider("telegram", "123")
        self.api = QueueProvider("responses_api", "api:default")
        self.api.capabilities = ProviderCapabilities(
            drafts=True,
            reasoning_summaries=True,
            pins=True,
            controls=True,
            typing=True,
        )
        self.multi = MultiProvider((self.telegram, self.api))

    async def asyncTearDown(self) -> None:
        await self.multi.close()

    async def test_routes_message_and_reply_to_origin_provider(self) -> None:
        await self.api.events.put(
            IncomingMessage(
                conversation_id="api:default",
                sender_id=self.api.authorized_user_id,
                message_id="request-1",
                text="/status",
            )
        )
        batch = await self.multi.poll(None, timeout=1)
        event = batch.events[0]
        self.assertEqual(event.conversation_id, "responses_api:api:default")
        self.assertEqual(event.sender_id, self.multi.authorized_user_id)
        self.multi.prepare_event(event)
        prepared = next(
            call for call in self.api.calls if call[0] == "prepare_event"
        )[1]
        self.assertEqual(prepared.conversation_id, "api:default")
        await self.multi.send_text(event.conversation_id, "status result")
        self.assertIn(
            ("send", "api:default", "status result"),
            self.api.calls,
        )
        self.assertNotIn(
            ("send", "123", "status result"),
            self.telegram.calls,
        )
        await self.multi.send_reasoning_summary(
            event.conversation_id,
            "Safe summary",
        )
        self.assertIn(
            ("reasoning", "api:default", "Safe summary"),
            self.api.calls,
        )

    async def test_finishes_action_on_origin_provider(self) -> None:
        await self.api.events.put(
            IncomingAction(
                action_id="action-1",
                conversation_id="api:default",
                sender_id=self.api.authorized_user_id,
                message_id="status-1",
                data="steer:abcd:cancel",
            )
        )
        event = (await self.multi.poll(None, timeout=1)).events[0]

        await self.multi.finish_event(event)

        finished = next(
            call for call in self.api.calls if call[0] == "finish_event"
        )
        raw_event = finished[1]
        self.assertEqual(raw_event.action_id, "action-1")
        self.assertEqual(raw_event.conversation_id, "api:default")
        self.assertFalse(
            any(call[0] == "finish_event" for call in self.telegram.calls)
        )

    async def test_ready_events_from_both_providers_are_not_lost(self) -> None:
        await self.telegram.events.put(
            IncomingMessage(
                "123",
                self.telegram.authorized_user_id,
                "tg-1",
                "telegram text",
            )
        )
        await self.api.events.put(
            IncomingMessage(
                "api:default",
                self.api.authorized_user_id,
                "api-1",
                "api text",
            )
        )
        first = await self.multi.poll(None, timeout=1)
        second = await self.multi.poll(first.cursor, timeout=1)
        self.assertEqual(
            {first.events[0].text, second.events[0].text},
            {"telegram text", "api text"},
        )

    async def test_action_acknowledgement_routes_by_namespaced_id(self) -> None:
        await self.api.events.put(
            IncomingAction(
                action_id="action-1",
                conversation_id="api:default",
                sender_id=self.api.authorized_user_id,
                message_id="status-1",
                data="approval:abc:approve",
            )
        )
        batch = await self.multi.poll(None, timeout=1)
        event = batch.events[0]
        self.assertEqual(event.action_id, "responses_api:action-1")
        await self.multi.answer_action(event.action_id, "accepted")
        self.assertIn(
            ("answer", "action-1", "accepted", False),
            self.api.calls,
        )

    async def test_does_not_promote_unauthorized_provider_sender(self) -> None:
        await self.telegram.events.put(
            IncomingMessage(
                conversation_id="123",
                sender_id="intruder",
                message_id="message-1",
                text="run this",
            )
        )

        event = (await self.multi.poll(None, timeout=1)).events[0]

        self.assertNotEqual(event.sender_id, self.multi.authorized_user_id)
        self.assertEqual(event.sender_id, "telegram:intruder")

    async def test_unqualified_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiplexed conversation"):
            await self.multi.send_text("123", "ambiguous")
        with self.assertRaisesRegex(ValueError, "multiplexed action"):
            await self.multi.answer_action("action-1")

    async def test_activity_surface_stays_on_its_provider(self) -> None:
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Task",
            status="running",
        )
        conversation_id = "telegram:123"

        surface_id = await self.multi.open_activity_surface(
            conversation_id,
            snapshot,
        )
        await self.multi.update_activity_surface(
            conversation_id,
            str(surface_id),
            snapshot,
        )
        await self.multi.close_activity_surface(
            conversation_id,
            str(surface_id),
            snapshot,
        )

        self.assertEqual(surface_id, "telegram:topic/message")
        operations = [call[0] for call in self.telegram.calls]
        self.assertEqual(
            operations[-3:],
            ["activity_open", "activity_update", "activity_close"],
        )
        self.assertFalse(
            any(call[0].startswith("activity_") for call in self.api.calls)
        )

    async def test_existing_activity_surface_reopens_on_same_provider(
        self,
    ) -> None:
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Continue",
            status="running",
        )

        surface_id = await self.multi.open_activity_surface(
            "telegram:123",
            snapshot,
            "telegram:topic/message",
        )

        self.assertEqual(surface_id, "telegram:topic/message")
        self.assertEqual(
            self.telegram.calls[-1],
            (
                "activity_open",
                "123",
                snapshot,
                "topic/message",
            ),
        )

    async def test_existing_activity_surface_cannot_cross_providers(
        self,
    ) -> None:
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Continue",
            status="running",
        )

        with self.assertRaisesRegex(
            ValueError,
            "activity surface and conversation differ",
        ):
            await self.multi.open_activity_surface(
                "telegram:123",
                snapshot,
                "responses_api:topic/message",
            )

        self.assertFalse(
            any(
                call[0] == "activity_open"
                for provider in (self.telegram, self.api)
                for call in provider.calls
            )
        )

    async def test_responses_request_context_survives_multiplexed_poll(self) -> None:
        responses = ResponsesApiProvider(host="127.0.0.1", port=0)
        multi = MultiProvider((self.telegram, responses))
        self.addAsyncCleanup(multi.close)
        pending = await responses._queue_message({"input": "hello"})
        batch = await multi.poll(None, timeout=1)
        multi.prepare_event(batch.events[0])
        await multi.send_text(batch.events[0].conversation_id, "world")
        assert pending.completed is not None
        result = await pending.completed
        self.assertEqual(result["output_text"], "world")


if __name__ == "__main__":
    unittest.main()
