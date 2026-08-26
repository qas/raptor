"""Application resource-ownership tests."""

import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

_ROOT = Path(__file__).resolve().parent.parent
_HOME = Path(tempfile.mkdtemp(prefix="raptor-application-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import application
from chat_provider import IncomingMessage, ProviderCapabilities
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

    async def test_failed_event_does_not_drop_later_batch_events(self) -> None:
        provider = Mock(name="provider", authorized_user_id="operator")
        provider.name = "provider"
        provider.prepare_event = Mock()
        provider.finish_event = AsyncMock()
        first = IncomingMessage("first", "operator", "1", "one")
        second = IncomingMessage("second", "operator", "2", "two")

        with (
            patch.object(
                application,
                "dispatch_event",
                AsyncMock(side_effect=[RuntimeError("failed"), None]),
            ) as dispatch,
            patch.object(application, "log_exception") as log_exception,
        ):
            await application.dispatch_events(provider, (first, second))

        self.assertEqual(
            [call.args[1] for call in dispatch.await_args_list],
            [first, second],
        )
        log_exception.assert_called_once()

    async def test_prepare_failure_still_finalizes_and_continues(self) -> None:
        provider = Mock(name="provider", authorized_user_id="operator")
        provider.name = "provider"
        provider.prepare_event = Mock(
            side_effect=[RuntimeError("prepare failed"), None]
        )
        provider.finish_event = AsyncMock()
        first = IncomingMessage("first", "operator", "1", "one")
        second = IncomingMessage("second", "operator", "2", "two")

        with (
            patch.object(
                application.session,
                "bound_chat",
                return_value=nullcontext(),
            ),
            patch.object(application, "handle_event", AsyncMock()),
            patch.object(application, "log_exception"),
        ):
            await application.dispatch_events(provider, (first, second))

        self.assertEqual(
            [call.args[0] for call in provider.finish_event.await_args_list],
            [first, second],
        )

    async def test_cleanup_failure_does_not_skip_later_operation(self) -> None:
        completed: list[str] = []

        async def fail() -> None:
            raise RuntimeError("cleanup failed")

        async def succeed() -> None:
            completed.append("done")

        with patch.object(application, "log_exception") as logged:
            await application._cleanup("first", fail())
            await application._cleanup("second", succeed())

        self.assertEqual(completed, ["done"])
        logged.assert_called_once()

    async def test_ready_event_precedes_external_ready_callback(self) -> None:
        provider = Mock(
            name="provider",
            authorized_user_id="operator",
            primary_conversation_id="conversation",
            capabilities=ProviderCapabilities(),
        )
        provider.initialize = AsyncMock()
        provider.close = AsyncMock()
        runtime = Mock(
            key="conversation",
            conversation_id="conversation",
        )
        client = Mock()
        client.aclose = AsyncMock()
        client_factory = Mock(return_value=client)

        observed: list[str] = []

        def on_ready() -> None:
            observed.append("callback")
            raise RuntimeError("stop after readiness")

        def record(_source, event, _data) -> None:
            if event == "ready":
                observed.append("log")

        previous_provider = set_chat_provider(None)
        self.addCleanup(set_chat_provider, previous_provider)
        patches = (
            patch.object(application, "load_chat_providers", return_value=provider),
            patch.object(
                application,
                "outbound_http_client",
                client_factory,
            ),
            patch.object(application, "ensure_model", AsyncMock()),
            patch.object(application, "start_skill_discovery"),
            patch.object(application, "close_skill_discovery", AsyncMock()),
            patch.object(application, "ensure_chat_dirs"),
            patch.object(
                application.session,
                "set_default_chat",
                return_value=runtime,
            ),
            patch.object(
                application,
                "bootstrap_runtime_storage",
                return_value={"repaired_chats": 0, "created_sessions": 0},
            ),
            patch.object(
                application.session,
                "all_chat_runtimes",
                return_value=(runtime,),
            ),
            patch.object(
                application.session,
                "bound_runtime",
                return_value=nullcontext(),
            ),
            patch.object(application, "rehydrate_pending_inputs", return_value=0),
            patch.object(application, "state", {"model": "test"}),
            patch.object(application, "repair_interrupted_root_turn"),
            patch.object(
                application,
                "flush_pending_delivery",
                AsyncMock(return_value=False),
            ),
            patch.object(
                application,
                "prepare_goal_on_startup",
                return_value=None,
            ),
            patch.object(application, "thread_active", return_value=False),
            patch.object(application, "reconcile_activity_surfaces", AsyncMock()),
            patch.object(
                application,
                "restore_pending_subagent_completions",
                return_value=0,
            ),
            patch.object(application, "goal_is_active", return_value=False),
            patch.object(application, "interrupt_root_turn", AsyncMock()),
            patch.object(application, "cancel_background_subagents", AsyncMock()),
            patch.object(application, "cancel_shell_sessions", AsyncMock()),
            patch.object(application, "close_activity_projections", AsyncMock()),
            patch.object(application, "log_event", side_effect=record),
        )
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            with self.assertRaisesRegex(RuntimeError, "stop after readiness"):
                await application.main(on_ready=on_ready)

        self.assertEqual(observed, ["log", "callback"])
        client_factory.assert_called_once_with(
            timeout=application.httpx.Timeout(None, connect=10.0),
        )

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
            patch.object(
                application,
                "outbound_http_client",
                return_value=client,
            ),
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
