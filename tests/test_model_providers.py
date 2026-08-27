"""Model-provider registry and routing tests."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-model-provider-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")

import responses
from model_providers import (
    ModelTarget,
    load_model_configuration,
)


CONFIG = """
model_provider = "alpha"
model = "alpha-large"

[model_providers.alpha]
base_url = "https://alpha.example/v1/"
api_key_env = "ALPHA_API_KEY"
default_model = "alpha-small"
request_max_retries = 1
retry_base_seconds = 0
context_window = 100000
reasoning_effort = "high"
reasoning_summary = "auto"

[model_providers.alpha.models."alpha-large"]
context_window = 200000

[model_providers.alpha.models."alpha-quiet"]
reasoning_effort = ""
reasoning_summary = ""

[model_providers.beta]
base_url = "http://beta.example/v1"
default_model = "beta-default"
context_window = 32000
"""


class _Response:
    is_error = False

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class ModelProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="raptor-provider-tests-"))
        self.path = directory / "config.toml"
        self.path.write_text(CONFIG, encoding="utf-8")
        self.configuration = load_model_configuration(self.path)

    def test_loads_provider_and_per_model_settings(self) -> None:
        self.assertEqual(
            self.configuration.default_target,
            ModelTarget("alpha", "alpha-large"),
        )
        provider = self.configuration.provider("alpha")
        self.assertEqual(provider.base_url, "https://alpha.example/v1")
        settings = provider.settings_for("alpha-large")
        self.assertEqual(settings.context_window, 200000)
        self.assertEqual(settings.reasoning_effort, "high")
        self.assertEqual(settings.reasoning_summary, "auto")
        quiet = provider.settings_for("alpha-quiet")
        self.assertEqual(quiet.context_window, 100000)
        self.assertIsNone(quiet.reasoning_effort)
        self.assertIsNone(quiet.reasoning_summary)

    def test_target_selection_inherits_or_uses_provider_default(self) -> None:
        parent = ModelTarget("alpha", "alpha-large")
        self.assertEqual(
            self.configuration.select_target(parent=parent),
            parent,
        )
        self.assertEqual(
            self.configuration.select_target(
                parent=parent,
                provider_id="beta",
            ),
            ModelTarget("beta", "beta-default"),
        )
        self.assertEqual(
            self.configuration.select_target(parent=parent, model="alpha-small"),
            ModelTarget("alpha", "alpha-small"),
        )

    def test_secret_is_lazy_and_never_part_of_target(self) -> None:
        provider = self.configuration.provider("alpha")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ALPHA_API_KEY"):
                provider.headers()
        with patch.dict(os.environ, {"ALPHA_API_KEY": "secret"}, clear=True):
            self.assertEqual(
                provider.headers(),
                {"Authorization": "Bearer secret"},
            )
        self.assertEqual(
            ModelTarget("alpha", "alpha-large").to_dict(),
            {"provider_id": "alpha", "model": "alpha-large"},
        )

    def test_invalid_provider_configuration_fails_fast(self) -> None:
        self.path.write_text(
            "[model_providers.bad]\nbase_url = 'file:///tmp/model'\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "HTTP"):
            load_model_configuration(self.path)

    def test_existing_config_cannot_silently_fall_back_to_local(self) -> None:
        self.path.write_text("model_providres = {}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unknown Raptor config fields"):
            load_model_configuration(self.path)

        self.path.write_text("model = 'typo-proof'\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must define"):
            load_model_configuration(self.path)

    async def test_requests_route_through_each_selected_provider(self) -> None:
        client = AsyncMock()
        client.post.side_effect = [
            _Response({"id": "alpha-response"}),
            _Response({"id": "beta-response"}),
        ]
        with (
            patch.object(responses, "MODEL_CONFIGURATION", self.configuration),
            patch.object(responses.session, "responses", client, create=True),
            patch.dict(os.environ, {"ALPHA_API_KEY": "secret"}),
        ):
            alpha, beta = await __import__("asyncio").gather(
                responses.responses_create(
                    ModelTarget("alpha", "alpha-large"),
                    [{"role": "user", "content": "one"}],
                    tools=[],
                ),
                responses.responses_create(
                    ModelTarget("beta", "beta-default"),
                    [{"role": "user", "content": "two"}],
                    tools=[],
                ),
            )

        self.assertEqual({alpha["id"], beta["id"]}, {"alpha-response", "beta-response"})
        urls = {call.args[0] for call in client.post.await_args_list}
        self.assertEqual(
            urls,
            {
                "https://alpha.example/v1/responses",
                "http://beta.example/v1/responses",
            },
        )


if __name__ == "__main__":
    unittest.main()
