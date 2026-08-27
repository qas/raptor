"""Operator-facing model provider command tests."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-model-command-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")

import chat_store
import commands
import session
from model_providers import ModelConfiguration, ModelProvider, ModelTarget
from turn_runtime import turns


class ModelCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="model-command-chats-"))
        chat_patch = patch.object(chat_store, "CHAT_DIR", self.directory)
        chat_patch.start()
        self.addCleanup(chat_patch.stop)
        chat_store._SEQ_CACHE.clear()

        self.alpha = ModelTarget("alpha", "alpha-model")
        self.beta = ModelTarget("beta", "beta-model")
        self.configuration = ModelConfiguration(
            providers={
                "alpha": ModelProvider(
                    id="alpha",
                    base_url="http://alpha.example/v1",
                    default_model="alpha-model",
                ),
                "beta": ModelProvider(
                    id="beta",
                    base_url="http://beta.example/v1",
                    default_model="beta-model",
                ),
            },
            default_target=self.alpha,
        )
        configuration_patch = patch.object(
            commands,
            "MODEL_CONFIGURATION",
            self.configuration,
        )
        configuration_patch.start()
        self.addCleanup(configuration_patch.stop)
        session_configuration_patch = patch.object(
            session,
            "MODEL_CONFIGURATION",
            self.configuration,
        )
        session_configuration_patch.start()
        self.addCleanup(session_configuration_patch.stop)
        provider_patch = patch.object(
            commands,
            "model_provider",
            side_effect=lambda target: self.configuration.provider(
                target.provider_id
            ),
        )
        provider_patch.start()
        self.addCleanup(provider_patch.stop)

        session.set_default_model_target(self.alpha)
        self.runtime_context = session.bound_chat(
            f"model-command:{self.directory.name}"
        )
        self.runtime_context.__enter__()
        self.addCleanup(self.runtime_context.__exit__, None, None, None)
        session.state.clear()
        session.state.update(copy.deepcopy(session.DEFAULT_STATE))
        session.state["model_target"] = self.alpha.to_dict()
        self.session_id = chat_store.create_session(
            kind="main",
            chat_key=session.current_runtime().key,
            model_target=self.alpha.to_dict(),
        )
        session.state["current_session_id"] = self.session_id
        turns.finish()
        session.subagent_tasks.clear()

    async def test_models_lists_the_requested_provider(self) -> None:
        send = AsyncMock()
        listing = AsyncMock(return_value=["beta-model", "beta-small"])
        with (
            patch.object(commands, "send", send),
            patch.object(commands, "list_models", listing),
        ):
            handled = await commands.command(1, "/models beta")

        self.assertTrue(handled)
        listing.assert_awaited_once_with("beta")
        self.assertIn("Models from beta", send.await_args.args[1])
        self.assertIn("beta-small", send.await_args.args[1])

    async def test_model_switch_archives_and_starts_a_fresh_target(self) -> None:
        send = AsyncMock()
        with (
            patch.object(commands, "send", send),
            patch.object(commands, "session_transition_busy", return_value=False),
        ):
            handled = await commands.command(1, "/model beta beta-model")

        self.assertTrue(handled)
        new_session_id = str(session.state["current_session_id"])
        self.assertNotEqual(new_session_id, self.session_id)
        self.assertEqual(session.current_model_target(), self.beta)
        old_end = [
            event
            for event in chat_store.read_events(self.session_id)
            if event.get("type") == "session_end"
        ]
        self.assertEqual(old_end[-1]["reason"], "model_target_changed")
        self.assertEqual(
            chat_store.session_summary(new_session_id)["model_target"],
            self.beta.to_dict(),
        )
        self.assertIn("fresh session", send.await_args.args[1])


if __name__ == "__main__":
    unittest.main()
