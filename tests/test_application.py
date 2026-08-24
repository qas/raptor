"""Application resource-ownership tests."""

import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import application
from chat_provider import IncomingMessage
from chat_runtime import get_chat_provider, set_chat_provider


class ApplicationLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_ignored_event_is_finalized_without_binding_chat_state(
        self,
    ) -> None:
        provider = Mock(authorized_user_id="operator")
        provider.prepare_event = Mock()
        provider.finish_event = AsyncMock()
        event = IncomingMessage(
            conversation_id="unconfigured",
            sender_id="intruder",
            message_id="1",
            text="hello",
        )

        with patch.object(application.session, "bound_chat") as bound_chat:
            await application.dispatch_event(provider, event)

        provider.prepare_event.assert_called_once_with(event)
        provider.finish_event.assert_awaited_once_with(event)
        bound_chat.assert_not_called()

    async def test_accepted_event_is_bound_dispatched_and_finalized(
        self,
    ) -> None:
        provider = Mock(authorized_user_id="operator")
        provider.prepare_event = Mock()
        provider.finish_event = AsyncMock()
        event = IncomingMessage(
            conversation_id="configured",
            sender_id="operator",
            message_id="1",
            text="hello",
        )

        with (
            patch.object(
                application.session,
                "bound_chat",
                return_value=nullcontext(),
            ) as bound_chat,
            patch.object(application, "handle_event", AsyncMock()) as handle,
        ):
            await application.dispatch_event(provider, event)

        provider.prepare_event.assert_called_once_with(event)
        bound_chat.assert_called_once_with("configured")
        handle.assert_awaited_once_with(event)
        provider.finish_event.assert_awaited_once_with(event)

    async def test_failed_provider_startup_releases_every_owned_resource(
        self,
    ) -> None:
        provider = Mock()
        provider.initialize = AsyncMock(side_effect=RuntimeError("startup"))
        provider.close = AsyncMock()
        client = Mock()
        client.aclose = AsyncMock()
        close_skills = AsyncMock()
        previous_provider = set_chat_provider(None)
        self.addCleanup(set_chat_provider, previous_provider)

        with (
            patch.object(
                application,
                "load_chat_providers",
                return_value=provider,
            ),
            patch.object(application.httpx, "AsyncClient", return_value=client),
            patch.object(application, "start_skill_discovery"),
            patch.object(application, "close_skill_discovery", close_skills),
            patch.object(application, "cancel_background_subagents", AsyncMock()),
            patch.object(application, "cancel_shell_sessions", AsyncMock()),
        ):
            with self.assertRaisesRegex(RuntimeError, "startup"):
                await application.main()

        provider.close.assert_awaited_once()
        close_skills.assert_awaited_once()
        client.aclose.assert_awaited_once()
        with self.assertRaisesRegex(RuntimeError, "not been initialized"):
            get_chat_provider()


if __name__ == "__main__":
    unittest.main()
