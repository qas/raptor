"""Chat-provider contract and provider-neutral orchestration tests."""
import asyncio
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")
os.environ.setdefault("TG_CHAT_IDS", "1")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chat_provider import (
    ActionButton,
    ChatProvider,
    Controls,
    IncomingAction,
    IncomingMessage,
    PollResult,
    ProviderCapabilities,
)
from chat_runtime import (
    get_chat_provider,
    load_chat_provider,
    load_chat_providers,
    set_chat_provider,
)
import session
from turn_runtime import TurnKind, turns


class FakeProvider:
    name = "fake"
    authorized_user_id = "@operator:example.org"
    primary_conversation_id = "!agent:example.org"

    def __init__(self, *, pins: bool = True) -> None:
        self.capabilities = ProviderCapabilities(
            drafts=True,
            pins=pins,
            controls=True,
            typing=True,
        )
        self.calls: list[tuple] = []
        self.next_message = 0
        self.reject_busy = False

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
        self.calls.append(("poll", cursor, timeout))
        return PollResult((), cursor)

    async def send_text(self, conversation_id, text: str) -> None:
        self.calls.append(("send_text", conversation_id, text))

    async def send_draft(
        self,
        conversation_id,
        draft_id: int,
        text: str,
    ) -> None:
        self.calls.append(
            ("send_draft", conversation_id, draft_id, text)
        )

    async def send_reasoning_summary(
        self,
        conversation_id,
        delta: str,
    ) -> None:
        self.calls.append(
            ("send_reasoning_summary", conversation_id, delta)
        )

    async def create_message(
        self,
        conversation_id,
        text: str,
        controls: Controls = (),
    ) -> str:
        self.next_message += 1
        message_id = f"$event-{self.next_message}"
        self.calls.append(
            ("create", conversation_id, message_id, text, controls)
        )
        return message_id

    async def edit_message(
        self,
        conversation_id,
        message_id,
        text: str,
        controls: Controls = (),
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
        return self.reject_busy

    async def acknowledge_queued_message(self, conversation_id) -> None:
        self.calls.append(("acknowledge_queued", conversation_id))

    async def finish_event(self, event) -> None:
        self.calls.append(("finish_event", event))

    def capture_delivery_context(self, conversation_id):
        self.calls.append(("capture_delivery_context", conversation_id))
        return None

    def activate_delivery_context(self, conversation_id, delivery_context):
        self.calls.append(
            ("activate_delivery_context", conversation_id, delivery_context)
        )
        return None

    def restore_delivery_context(self, token) -> None:
        self.calls.append(("restore_delivery_context", token))

    async def answer_action(
        self,
        action_id: str,
        text: str = "",
        *,
        alert: bool = False,
    ) -> None:
        self.calls.append(("answer", action_id, text, alert))


class ChatProviderContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.provider = FakeProvider()
        self.previous_provider = set_chat_provider(self.provider)
        session.current_runtime().pinned_status_conversation_id = None
        session.current_runtime().pinned_status_message_id = None
        session.current_runtime().pinned_status_owner = None
        session.current_runtime().goal_pin_message_id = None
        session.current_runtime().goal_pin_goal_id = None
        session.pending_approvals.clear()
        session.pending_steers.clear()
        while True:
            try:
                session.steer_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                session.steer_queue.task_done()
        turns.finish()

    def tearDown(self) -> None:
        set_chat_provider(self.previous_provider)

    async def test_persistent_status_supports_opaque_provider_ids(self) -> None:
        from presentation import clear_pinned_status, show_pinned_status

        message_id = await show_pinned_status(
            "!room:example.org",
            "goal:g1",
            "Goal active",
        )
        self.assertEqual(message_id, "$event-1")

        same_id = await show_pinned_status(
            "!room:example.org",
            "approval:a1",
            "Approval required",
        )
        self.assertEqual(same_id, message_id)
        await clear_pinned_status(
            "!room:example.org",
            owner="approval:a1",
        )

        methods = [call[0] for call in self.provider.calls]
        self.assertEqual(
            methods,
            ["create", "pin", "edit", "unpin", "delete"],
        )

    async def test_provider_without_pins_uses_capability_path(self) -> None:
        from presentation import clear_pinned_status, show_pinned_status

        provider = FakeProvider(pins=False)
        set_chat_provider(provider)
        provider.capabilities = ProviderCapabilities(
            drafts=False,
            pins=False,
            controls=False,
            typing=False,
        )
        await show_pinned_status(
            "!room",
            "goal:g1",
            "Goal active",
            controls=((ActionButton("Approve", "approve"),),),
        )
        await clear_pinned_status("!room", owner="goal:g1")

        methods = [call[0] for call in provider.calls]
        self.assertEqual(methods, ["create", "delete"])
        self.assertEqual(provider.calls[0][-1], ())

    async def test_steering_message_is_never_pinned(self) -> None:
        from presentation import (
            clear_steering_indicator,
            steering_indicator,
        )

        message_id = await steering_indicator(
            "!room:example.org",
            "abcd",
        )
        await clear_steering_indicator(
            "!room:example.org",
            message_id,
            "abcd",
        )

        methods = [call[0] for call in self.provider.calls]
        self.assertEqual(methods, ["create", "delete"])
        controls = self.provider.calls[0][-1]
        self.assertEqual(
            [button.action for button in controls[0]],
            ["steer:abcd:apply", "steer:abcd:cancel"],
        )

    async def test_cancel_steering_deletes_user_message(self) -> None:
        from steering import handle_steering_action

        session.state["pending_inputs"] = ["cancel me"]
        session.pending_steers["abcd"] = {
            "id": "abcd",
            "chat_id": "!room:example.org",
            "text": "cancel me",
            "source_message_id": "$user-message",
            "message_id": "$steering-controls",
            "status": "queued",
        }
        event = IncomingAction(
            action_id="$cancel-action",
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$steering-controls",
            data="steer:abcd:cancel",
        )

        with patch("steering.session.save_state"):
            handled = await handle_steering_action(event)

        self.assertTrue(handled)
        self.assertNotIn("abcd", session.pending_steers)
        self.assertEqual(session.state["pending_inputs"], [])
        self.assertIn(
            ("delete", "!room:example.org", "$steering-controls"),
            self.provider.calls,
        )
        self.assertIn(
            ("delete", "!room:example.org", "$user-message"),
            self.provider.calls,
        )

    async def test_slow_forced_steer_waits_for_root_ownership(self) -> None:
        from controller import _dequeue_steer
        from steering import handle_steering_action

        entry = {
            "id": "abcd",
            "chat_id": "!room:example.org",
            "text": "apply after cancellation",
            "message_id": "$steering-controls",
            "status": "queued",
        }
        session.pending_steers["abcd"] = entry
        await session.steer_queue.put(entry)
        event = IncomingAction(
            action_id="$apply-action",
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$steering-controls",
            data="steer:abcd:apply",
        )

        with (
            patch(
                "steering.interrupt_root_turn",
                AsyncMock(
                    return_value=types.SimpleNamespace(
                        completed=False,
                        error=None,
                    )
                ),
            ),
            patch("steering.ensure_root_session") as ensure,
        ):
            handled = await handle_steering_action(event)

        self.assertTrue(handled)
        self.assertEqual(entry["status"], "force_pending")
        ensure.assert_called_once_with("!room:example.org", None)

        selected = await _dequeue_steer()
        self.assertIs(selected, entry)
        self.assertEqual(entry["status"], "applied")
        self.assertNotIn("abcd", session.pending_steers)

    async def test_cancelled_steer_claim_returns_to_queue(self) -> None:
        from controller import _dequeue_steer

        entry = {
            "id": "abcd",
            "chat_id": "!room:example.org",
            "text": "keep this request",
            "message_id": "$steering-controls",
            "status": "queued",
        }
        session.pending_steers["abcd"] = entry
        await session.steer_queue.put(entry)

        async def cancel_cleanup(*_args, **_kwargs):
            raise asyncio.CancelledError

        with patch(
            "controller.clear_steering_indicator",
            cancel_cleanup,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await _dequeue_steer()

        self.assertEqual(entry["status"], "queued")
        self.assertIs(session.pending_steers["abcd"], entry)
        self.assertIs(session.steer_queue.get_nowait(), entry)
        session.steer_queue.task_done()

    async def test_global_stop_discards_queued_steering(self) -> None:
        from steering import cancel_pending_steers

        session.state["pending_inputs"] = ["queued"]
        session.pending_steers["abcd"] = {
            "id": "abcd",
            "chat_id": "!room:example.org",
            "text": "queued",
            "message_id": "$steering-controls",
            "status": "queued",
        }
        await session.steer_queue.put(session.pending_steers["abcd"])

        with patch("steering.session.save_state"):
            cancelled = await cancel_pending_steers()

        self.assertEqual(cancelled, 1)
        self.assertEqual(session.pending_steers, {})
        self.assertEqual(session.state["pending_inputs"], [])
        self.assertTrue(session.steer_queue.empty())

    async def test_interruption_cleanup_preserves_only_forced_steer(self) -> None:
        from steering import cancel_unforced_steers

        queued = {
            "id": "queued",
            "chat_id": "!room:example.org",
            "text": "discard me",
            "message_id": "$queued-controls",
            "status": "queued",
        }
        forced = {
            "id": "forced",
            "chat_id": "!room:example.org",
            "text": "keep me",
            "message_id": "$forced-controls",
            "status": "force_pending",
        }
        session.pending_steers.update(
            {"queued": queued, "forced": forced}
        )
        await session.steer_queue.put(queued)
        await session.steer_queue.put(forced)

        with patch("steering.session.save_state"):
            cancelled = await cancel_unforced_steers()

        self.assertEqual(cancelled, 1)
        self.assertEqual(session.pending_steers, {"forced": forced})
        self.assertEqual(session.state["pending_inputs"], ["keep me"])
        self.assertIs(await session.steer_queue.get(), forced)
        session.steer_queue.task_done()

    def test_fake_provider_satisfies_runtime_contract(self) -> None:
        self.assertIsInstance(self.provider, ChatProvider)

    def test_provider_access_requires_explicit_process_binding(self) -> None:
        previous_provider = set_chat_provider(None)
        try:
            with self.assertRaisesRegex(RuntimeError, "not been initialized"):
                get_chat_provider()
        finally:
            set_chat_provider(previous_provider)

    def test_external_provider_factory_loads_by_module_attribute(self) -> None:
        module = types.ModuleType("test_external_chat_provider")
        module.create_provider = FakeProvider
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)

        loaded = load_chat_provider(
            "test_external_chat_provider:create_provider"
        )

        self.assertIsInstance(loaded, FakeProvider)

    def test_responses_api_provider_is_builtin(self) -> None:
        from responses_provider import ResponsesApiProvider

        self.assertIsInstance(
            load_chat_provider("responses_api"),
            ResponsesApiProvider,
        )

    def test_configured_providers_are_composed_by_default(self) -> None:
        from config import CHAT_PROVIDERS
        from multi_provider import MultiProvider

        self.assertEqual(CHAT_PROVIDERS, ("telegram", "responses_api"))
        self.assertIsInstance(
            load_chat_providers(CHAT_PROVIDERS),
            MultiProvider,
        )

    def test_single_configured_provider_is_not_wrapped(self) -> None:
        from telegram import TelegramProvider

        self.assertIsInstance(
            load_chat_providers(("telegram",)),
            TelegramProvider,
        )

    def test_unknown_provider_spec_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_chat_provider("unknown")

    async def test_normalized_message_reaches_core_with_string_id(self) -> None:
        from loop import handle_event

        event = IncomingMessage(
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$message",
            text="continue",
        )
        with (
            patch("loop.command", AsyncMock(return_value=False)),
            patch("loop.start_root_session") as start,
        ):
            await handle_event(event)

        start.assert_called_once_with(
            "!room:example.org",
            "continue",
            delivery_context=None,
        )

    async def test_request_provider_can_reject_busy_input_before_steering(
        self,
    ) -> None:
        from loop import handle_event

        turns.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
        )
        blocker = turns.task
        assert blocker is not None
        self.provider.reject_busy = True
        event = IncomingMessage(
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$message",
            text="concurrent input",
        )
        try:
            with (
                patch("loop.command", AsyncMock(return_value=False)),
                patch("loop.start_root_session") as start,
            ):
                await handle_event(event)
            start.assert_not_called()
            self.assertIn(
                ("reject_busy", "!room:example.org"),
                self.provider.calls,
            )
            self.assertEqual(session.pending_steers, {})
        finally:
            blocker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await blocker
            turns.finish(blocker)

    async def test_busy_chat_input_is_queued_and_transport_acknowledged(
        self,
    ) -> None:
        from loop import handle_event

        turns.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
        )
        blocker = turns.task
        assert blocker is not None
        previous_session_id = session.state.get("current_session_id")
        session.state["current_session_id"] = None
        event = IncomingMessage(
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$message",
            text="change direction",
        )
        try:
            with patch("loop.command", AsyncMock(return_value=False)):
                await handle_event(event)
            self.assertEqual(len(session.pending_steers), 1)
            self.assertIn(
                ("acknowledge_queued", "!room:example.org"),
                self.provider.calls,
            )
        finally:
            blocker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await blocker
            turns.finish(blocker)
            session.state["current_session_id"] = previous_session_id
            while not session.steer_queue.empty():
                session.steer_queue.get_nowait()
                session.steer_queue.task_done()
            session.pending_steers.clear()

    async def test_busy_chat_rejects_input_when_steering_queue_is_full(
        self,
    ) -> None:
        from loop import handle_event

        turns.start(
            asyncio.Event().wait(),
            kind=TurnKind.REGULAR,
        )
        blocker = turns.task
        assert blocker is not None
        session.pending_steers["existing"] = {"status": "queued"}
        event = IncomingMessage(
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$message",
            text="one too many",
        )
        try:
            with (
                patch("loop.command", AsyncMock(return_value=False)),
                patch("loop.MAX_PENDING_STEERS", 1),
            ):
                await handle_event(event)
            self.assertIn(
                (
                    "send_text",
                    "!room:example.org",
                    "Steering queue is full (1).",
                ),
                self.provider.calls,
            )
            self.assertEqual(set(session.pending_steers), {"existing"})
        finally:
            blocker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await blocker
            turns.finish(blocker)
            session.pending_steers.clear()

    async def test_ask_prompt_is_not_copied_into_event_log(self) -> None:
        from loop import handle_event

        event = IncomingMessage(
            conversation_id="!room:example.org",
            sender_id="@operator:example.org",
            message_id="$message",
            text="/ask private side question",
        )
        with (
            patch("loop.command", AsyncMock(return_value=True)),
            patch("loop.log_event") as logged,
        ):
            await handle_event(event)

        received = logged.call_args.args[2]
        self.assertEqual(received["command"], "/ask")
        self.assertNotIn("text", received)

    async def test_unknown_normalized_action_is_acknowledged(self) -> None:
        from loop import handle_event

        await handle_event(
            IncomingAction(
                action_id="$action",
                conversation_id="!room:example.org",
                sender_id="@operator:example.org",
                message_id="$message",
                data="provider-specific-unknown-action",
            )
        )
        self.assertIn(
            ("answer", "$action", "", False),
            self.provider.calls,
        )


class TelegramNormalizationTests(unittest.TestCase):
    def test_callback_is_normalized_at_adapter_boundary(self) -> None:
        from telegram import telegram_provider

        event = telegram_provider.normalize_update(
            {
                "callback_query": {
                    "id": "callback-1",
                    "from": {"id": 1},
                    "data": "approval:abc:approve",
                    "message": {
                        "message_id": 9,
                        "chat": {"id": 1, "type": "private"},
                    },
                },
            }
        )
        self.assertEqual(
            event,
            IncomingAction(
                action_id="callback-1",
                conversation_id="1",
                sender_id=1,
                message_id=9,
                data="approval:abc:approve",
            ),
        )

    def test_forum_topic_is_an_independent_conversation(self) -> None:
        import telegram

        provider = telegram.TelegramProvider()
        event = provider.normalize_update(
            {
                "message": {
                    "message_id": 10,
                    "message_thread_id": 42,
                    "is_topic_message": True,
                    "from": {"id": 1},
                    "chat": {"id": 1, "type": "supergroup"},
                    "text": "hello",
                }
            }
        )

        self.assertIsInstance(event, IncomingMessage)
        self.assertEqual(event.conversation_id, "1/42")
        self.assertTrue(event.interactive)

    def test_activity_topic_input_is_noninteractive(self) -> None:
        import telegram

        provider = telegram.TelegramProvider()
        provider._chats[1].activity_topic_ids.add(42)
        event = provider.normalize_update(
            {
                "message": {
                    "message_id": 10,
                    "message_thread_id": 42,
                    "is_topic_message": True,
                    "from": {"id": 1},
                    "chat": {"id": 1, "type": "supergroup"},
                    "text": "ignored",
                }
            }
        )

        self.assertIsInstance(event, IncomingMessage)
        self.assertFalse(event.interactive)

    def test_chat_and_topic_membership_are_isolated(self) -> None:
        import telegram

        with patch.object(telegram, "TG_CHAT_IDS", (1, 2)):
            provider = telegram.TelegramProvider()
        provider._chats[1].activity_topic_ids.add(42)

        first_chat = provider.normalize_update(
            {
                "message": {
                    "message_id": 10,
                    "message_thread_id": 42,
                    "is_topic_message": True,
                    "from": {"id": 1},
                    "chat": {"id": 1, "type": "supergroup"},
                    "text": "activity input",
                }
            }
        )
        second_chat = provider.normalize_update(
            {
                "message": {
                    "message_id": 11,
                    "message_thread_id": 42,
                    "is_topic_message": True,
                    "from": {"id": 1},
                    "chat": {"id": 2, "type": "supergroup"},
                    "text": "main input",
                }
            }
        )
        unknown_chat = provider.normalize_update(
            {
                "message": {
                    "message_id": 12,
                    "from": {"id": 1},
                    "chat": {"id": 3, "type": "private"},
                    "text": "unknown input",
                }
            }
        )

        self.assertFalse(first_chat.interactive)
        self.assertTrue(second_chat.interactive)
        self.assertFalse(unknown_chat.interactive)


class TelegramMultiChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_discovers_every_chat_in_order(self) -> None:
        import telegram

        call = AsyncMock(
            side_effect=[
                True,
                {"type": "private"},
                {"type": "supergroup", "is_forum": True},
                {"id": 99},
                {
                    "status": "administrator",
                    "can_manage_topics": True,
                },
                True,
            ]
        )
        client = AsyncMock()
        with (
            patch.object(telegram, "TG_CHAT_IDS", (7, -1002)),
            patch.object(telegram, "_client", client),
            patch.object(telegram, "tg_call", call),
        ):
            provider = telegram.TelegramProvider()
            await provider.initialize(())

        self.assertEqual(provider.primary_conversation_id, "7")
        self.assertEqual(
            provider._chats[7].chat_type,
            "private",
        )
        self.assertEqual(provider._chats[-1002].chat_type, "supergroup")
        self.assertTrue(provider._chats[-1002].is_forum)
        self.assertTrue(provider.capabilities.drafts)
        self.assertEqual(
            [entry.args for entry in call.await_args_list],
            [
                ("deleteWebhook", {"drop_pending_updates": False}),
                ("getChat", {"chat_id": 7}),
                ("getChat", {"chat_id": -1002}),
                ("getMe",),
                (
                    "getChatMember",
                    {"chat_id": -1002, "user_id": 99},
                ),
                ("setMyCommands", {"commands": []}),
            ],
        )

    async def test_drafts_are_routed_only_to_private_chats(self) -> None:
        import telegram

        with patch.object(telegram, "TG_CHAT_IDS", (7, -1002)):
            provider = telegram.TelegramProvider()
        provider._chats[7].chat_type = "private"
        provider._chats[-1002].chat_type = "supergroup"
        draft = AsyncMock()

        with patch.object(telegram, "send_draft", draft):
            await provider.send_draft(7, 1, "private draft")
            await provider.send_draft(-1002, 2, "group draft")

        draft.assert_awaited_once_with(7, 1, "private draft")

    async def test_activity_topics_are_isolated_by_chat(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        with patch.object(telegram, "TG_CHAT_IDS", (-1001, -1002)):
            provider = telegram.TelegramProvider()
        provider._chats[-1001].is_forum = True
        provider._chats[-1002].is_forum = True
        provider._chats[-1001].activity_topic_ids.add(42)
        provider._chats[-1002].activity_topic_ids.add(42)
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="completed",
        )

        with (
            patch.object(telegram, "_edit_rich_message", AsyncMock()),
            patch.object(telegram, "_delete_forum_topic", AsyncMock()),
        ):
            await provider.close_activity_surface(
                "-1001/10",
                "42/77",
                snapshot,
            )

        self.assertNotIn(42, provider._chats[-1001].activity_topic_ids)
        self.assertIn(42, provider._chats[-1002].activity_topic_ids)


class TelegramTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_activity_topic_is_created_and_deleted(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        provider = telegram.TelegramProvider()
        provider._chats[1].is_forum = True
        call = AsyncMock(
            side_effect=[{"message_thread_id": 42}, True]
        )
        rich = AsyncMock(return_value={"message_id": 77})
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="running",
        )
        with (
            patch.object(telegram, "tg_call", call),
            patch.object(telegram, "send_rich", rich),
        ):
            surface_id = await provider.open_activity_surface("1/10", snapshot)
            await provider.close_activity_surface(
                "1/10",
                str(surface_id),
                ActivitySnapshot(
                    activity_id="worker",
                    title="Inspect target",
                    status="completed",
                    result="done",
                ),
            )

        self.assertEqual(surface_id, "42/77")
        self.assertEqual(call.await_args_list[0].args[0], "createForumTopic")
        self.assertEqual(
            call.await_args_list[0].args[1]["name"],
            "Subagent: worker",
        )
        self.assertIn("Subagent: worker", rich.await_args_list[0].args[2])
        self.assertEqual(call.await_args_list[1].args[0], "deleteForumTopic")
        self.assertNotIn(42, provider._chats[1].activity_topic_ids)

    def test_activity_text_includes_reasoning_and_reply(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        text = telegram._activity_text(
            ActivitySnapshot(
                activity_id="worker",
                title="Inspect target",
                status="running",
                reasoning_summary="Checking files",
                reply="I found the issue",
            )
        )

        self.assertIn("Reasoning\nChecking files", text)
        self.assertIn("Reply\nI found the issue", text)

    async def test_unchanged_message_edit_is_a_successful_noop(self) -> None:
        import telegram

        error = telegram.TelegramApiError(
            "editMessageText",
            status_code=200,
            error_code=400,
            description=(
                "Bad Request: message is not modified: specified new message "
                "content and reply markup are exactly the same"
            ),
        )
        rich = AsyncMock(side_effect=error)

        with patch.object(telegram, "send_rich", rich):
            await telegram.TelegramProvider().edit_message(1, 7, "unchanged")

        rich.assert_awaited_once()

    async def test_activity_update_propagates_real_edit_errors(self) -> None:
        import telegram
        from activity import ActivitySnapshot

        error = telegram.TelegramApiError(
            "editMessageText",
            status_code=400,
            error_code=400,
            description="Bad Request: message to edit not found",
        )
        snapshot = ActivitySnapshot(
            activity_id="worker",
            title="Inspect target",
            status="running",
        )

        with (
            patch.object(telegram, "send_rich", AsyncMock(side_effect=error)),
            self.assertRaises(telegram.TelegramApiError),
        ):
            await telegram.TelegramProvider().update_activity_surface(
                "1/10",
                "42/77",
                snapshot,
            )

    async def test_chat_requests_are_spaced(self) -> None:
        import telegram

        with (
            patch.object(telegram, "_CHAT_REQUEST_INTERVAL", 0.025),
            patch.object(telegram, "_GLOBAL_REQUEST_INTERVAL", 0.0),
        ):
            started = asyncio.get_running_loop().time()
            await telegram._reserve_telegram_request("sendMessage", 1)
            await telegram._reserve_telegram_request("editMessageText", 1)
            elapsed = asyncio.get_running_loop().time() - started

        self.assertGreaterEqual(elapsed, 0.02)

    async def test_429_wait_metadata_is_applied_before_retry(self) -> None:
        import telegram

        request_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            request_count += 1
            if request_count == 1:
                return httpx.Response(
                    429,
                    json={
                        "ok": False,
                        "error_code": 429,
                        "description": "Too Many Requests",
                        "parameters": {"retry_after": 9},
                    },
                )
            return httpx.Response(200, json={"ok": True, "result": "sent"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        reserve = AsyncMock()
        defer = AsyncMock()
        try:
            with (
                patch.object(telegram, "_client", client),
                patch.object(telegram, "_reserve_telegram_request", reserve),
                patch.object(telegram, "_defer_telegram_requests", defer),
            ):
                result = await telegram.tg_call(
                    "sendMessage",
                    {"chat_id": 1, "text": "hello"},
                )
        finally:
            await client.aclose()

        self.assertEqual(result, "sent")
        self.assertEqual(request_count, 2)
        defer.assert_awaited_once_with(1, 9.0)

    async def test_rich_text_does_not_retry_rate_limit_as_plain_text(
        self,
    ) -> None:
        import telegram

        error = telegram.TelegramApiError(
            "sendMessage",
            status_code=429,
            error_code=429,
            description="Too Many Requests",
            retry_after=9,
        )
        call = AsyncMock(side_effect=error)
        with (
            patch.object(telegram, "TELEGRAM_MARKDOWN", True),
            patch.object(telegram, "tg_call", call),
            self.assertRaises(telegram.TelegramApiError),
        ):
            await telegram.send_rich(
                "sendMessage",
                {"chat_id": 1},
                "hello",
            )
        self.assertEqual(call.await_count, 1)

    async def test_rich_text_falls_back_only_for_entity_parse_error(
        self,
    ) -> None:
        import telegram

        error = telegram.TelegramApiError(
            "sendMessage",
            status_code=400,
            error_code=400,
            description="Bad Request: can't parse entities",
        )
        call = AsyncMock(side_effect=[error, "sent"])
        with (
            patch.object(telegram, "TELEGRAM_MARKDOWN", True),
            patch.object(telegram, "tg_call", call),
        ):
            result = await telegram.send_rich(
                "sendMessage",
                {"chat_id": 1},
                "hello",
            )

        self.assertEqual(result, "sent")
        self.assertEqual(call.await_count, 2)
        self.assertEqual(
            call.await_args_list[1].args[1],
            {"chat_id": 1, "text": "hello"},
        )

    async def test_edit_clears_controls_in_same_request(self) -> None:
        import telegram

        call = AsyncMock(return_value=True)
        with patch.object(telegram, "tg_call", call):
            await telegram.TelegramProvider().edit_message(1, 7, "updated")

        call.assert_awaited_once_with(
            "editMessageText",
            {
                "chat_id": 1,
                "message_id": 7,
                "reply_markup": {"inline_keyboard": []},
                "text": "updated",
                "parse_mode": "HTML",
            },
        )
