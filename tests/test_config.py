import os
import re
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")
os.environ.setdefault("TG_CHAT_IDS", "1")

import config
import config_document


ROOT = Path(__file__).resolve().parent.parent


class TelegramConfigurationTests(unittest.TestCase):
    def test_telegram_chat_ids_preserve_configured_order(self) -> None:
        with patch.dict(
            os.environ,
            {"TG_USER_ID": "7", "TG_CHAT_IDS": "-1002, 7, -1001"},
            clear=True,
        ):
            values = runpy.run_path(str(ROOT / "config.py"))

        self.assertEqual(values["TG_CHAT_IDS"], (-1002, 7, -1001))

    def test_telegram_chat_ids_default_to_authorized_user(self) -> None:
        with patch.dict(os.environ, {"TG_USER_ID": "7"}, clear=True):
            values = runpy.run_path(str(ROOT / "config.py"))

        self.assertEqual(values["TG_CHAT_IDS"], (7,))

    def test_rejects_duplicate_telegram_chat_ids(self) -> None:
        with patch.dict(
            os.environ,
            {"TG_USER_ID": "7", "TG_CHAT_IDS": "7,-1001,7"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "TG_CHAT_IDS entries must be unique",
            ):
                runpy.run_path(str(ROOT / "config.py"))

    def test_rejects_invalid_telegram_chat_ids(self) -> None:
        for value in ("", "7,,8", "7,chat", "7,0"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"TG_USER_ID": "7", "TG_CHAT_IDS": value},
                clear=True,
            ):
                with self.assertRaises(ValueError):
                    runpy.run_path(str(ROOT / "config.py"))


class ResponsesServerConfigurationTests(unittest.TestCase):
    def test_inbound_api_defaults_to_unauthenticated_loopback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            values = runpy.run_path(str(ROOT / "config.py"))

        self.assertEqual(values["RESPONSES_SERVER_HOST"], "127.0.0.1")
        self.assertEqual(values["RESPONSES_SERVER_API_KEY"], "")


class TomlConfigurationTests(unittest.TestCase):
    def _load(self, document, environment=None):
        with (
            patch.object(
                config_document,
                "load_config_document",
                return_value=document,
            ),
            patch.dict(os.environ, environment or {}, clear=True),
        ):
            return runpy.run_path(str(ROOT / "config.py"))

    def test_non_secret_toml_settings_are_loaded(self) -> None:
        values = self._load(
            {
                "network": {
                    "proxy": "http://proxy.example:8080",
                    "no_proxy": ["models.example"],
                },
                "permissions": {
                    "filesystem": {
                        "deny_read": [".env", "**/*.pem"],
                        "glob_scan_max_depth": 12,
                    },
                },
                "chat": {
                    "providers": ["telegram"],
                    "streaming": False,
                    "stream_interval": 0.2,
                    "max_pending_steers": 6,
                    "max_runtimes": 12,
                },
                "telegram": {
                    "user_id": 7,
                    "chat_ids": [7, -1001],
                    "max_retries": 4,
                    "markdown": False,
                    "subagent_topics_silent": False,
                },
                "responses_server": {
                    "host": "localhost",
                    "port": 9000,
                    "max_body": 2048,
                    "max_connections": 9,
                    "max_pending": 8,
                    "max_status_messages": 7,
                    "max_stream_events": 6,
                    "read_timeout": 1.5,
                },
                "subagents": {
                    "max_depth": 5,
                    "max_records": 20,
                    "max_tool_events": 21,
                    "max_pending_inputs": 22,
                    "max_background": 23,
                },
                "tools": {"max_rounds": 2, "max_output": 4096},
                "shell": {"timeout": 90},
                "state": {"max_load_bytes": 8192},
                "compaction": {
                    "output_tokens": 512,
                    "generation_tokens": 1024,
                    "reasoning_effort": "medium",
                    "keep_recent_tokens": 100,
                    "user_anchor_tokens": 101,
                    "max_record_chars": 2048,
                    "context_ratio": 0.75,
                    "context_safety_tokens": 102,
                },
            }
        )

        expected = {
            "RAPTOR_PROXY": "http://proxy.example:8080",
            "RAPTOR_NO_PROXY": ("models.example",),
            "FILESYSTEM_POLICY": config.FileAccessPolicy.create(
                config.AGENT_WORKDIR,
                [".env", "**/*.pem"],
                12,
            ),
            "CHAT_PROVIDERS": ("telegram",),
            "CHAT_STREAMING": False,
            "CHAT_STREAM_INTERVAL": 0.2,
            "MAX_PENDING_STEERS": 6,
            "MAX_CHAT_RUNTIMES": 12,
            "TG_USER_ID": 7,
            "TG_CHAT_IDS": (7, -1001),
            "TG_MAX_RETRIES": 4,
            "TELEGRAM_MARKDOWN": False,
            "TELEGRAM_SUBAGENT_TOPICS_SILENT": False,
            "RESPONSES_SERVER_HOST": "localhost",
            "RESPONSES_SERVER_PORT": 9000,
            "RESPONSES_SERVER_MAX_BODY": 2048,
            "RESPONSES_SERVER_MAX_CONNECTIONS": 9,
            "RESPONSES_SERVER_MAX_PENDING": 8,
            "RESPONSES_SERVER_MAX_STATUS_MESSAGES": 7,
            "RESPONSES_SERVER_MAX_STREAM_EVENTS": 6,
            "RESPONSES_SERVER_READ_TIMEOUT": 1.5,
            "MAX_SUBAGENT_DEPTH": 5,
            "MAX_SUBAGENT_RECORDS": 20,
            "MAX_SUBAGENT_TOOL_EVENTS": 21,
            "MAX_SUBAGENT_PENDING_INPUTS": 22,
            "MAX_BACKGROUND_SUBAGENTS": 23,
            "MAX_TOOL_ROUNDS": 2,
            "MAX_TOOL_OUTPUT": 4096,
            "SHELL_TIMEOUT": 90,
            "MAX_STATE_LOAD_BYTES": 8192,
            "COMPACTION_OUTPUT_TOKENS": 512,
            "COMPACTION_GENERATION_TOKENS": 1024,
            "COMPACTION_REASONING_EFFORT": "medium",
            "COMPACT_KEEP_RECENT_TOKENS": 100,
            "COMPACTION_USER_ANCHOR_TOKENS": 101,
            "COMPACTION_MAX_RECORD_CHARS": 2048,
            "CONTEXT_COMPACT_RATIO": 0.75,
            "CONTEXT_SAFETY_TOKENS": 102,
        }
        for name, expected_value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(values[name], expected_value)

    def test_environment_overrides_toml(self) -> None:
        values = self._load(
            {
                "chat": {"streaming": False},
                "telegram": {
                    "chat_ids": [8],
                    "subagent_topics_silent": True,
                },
            },
            {
                "CHAT_STREAMING": "true",
                "TG_CHAT_IDS": "9,-1002",
                "TELEGRAM_SUBAGENT_TOPICS_SILENT": "false",
            },
        )

        self.assertTrue(values["CHAT_STREAMING"])
        self.assertEqual(values["TG_CHAT_IDS"], (9, -1002))
        self.assertFalse(values["TELEGRAM_SUBAGENT_TOPICS_SILENT"])

    def test_unknown_or_secret_toml_fields_are_rejected(self) -> None:
        for document in (
            {"telegram": {"bot_token": "secret"}},
            {"chat": {"streamng": True}},
            {"permissions": {"filesystem": {"deny_reads": []}}},
        ):
            with self.subTest(document=document), self.assertRaisesRegex(
                ValueError,
                "Unknown",
            ):
                self._load(document)

    def test_proxy_credentials_remain_environment_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "use RAPTOR_PROXY"):
            self._load(
                {"network": {"proxy": "https://user:pass@proxy.example"}}
            )

    def test_toml_types_are_not_coerced(self) -> None:
        with self.assertRaisesRegex(ValueError, "MAX_SUBAGENT_DEPTH"):
            self._load({"subagents": {"max_depth": 3.5}})


class ProxyConfigurationTests(unittest.TestCase):
    def test_accepts_https_and_remote_dns_socks_proxies(self) -> None:
        for proxy in (
            "https://user:pass@proxy.example:8443",
            "socks5h://proxy.example:1080",
        ):
            with self.subTest(proxy=proxy), patch.dict(
                os.environ,
                {"RAPTOR_PROXY": proxy},
                clear=True,
            ):
                values = runpy.run_path(str(ROOT / "config.py"))
            self.assertEqual(values["RAPTOR_PROXY"], proxy)

    def test_rejects_proxy_schemes_that_can_bypass_remote_dns(self) -> None:
        for proxy in (
            "socks5://proxy.example:1080",
            "ftp://proxy.example",
            "proxy.example:8080",
        ):
            with self.subTest(proxy=proxy), patch.dict(
                os.environ,
                {"RAPTOR_PROXY": proxy},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "RAPTOR_PROXY"):
                    runpy.run_path(str(ROOT / "config.py"))

    def test_rejects_proxy_url_suffixes(self) -> None:
        for proxy in (
            "https://proxy.example/path",
            "https://proxy.example?bypass=true",
            "https://proxy.example#fragment",
        ):
            with self.subTest(proxy=proxy), patch.dict(
                os.environ,
                {"RAPTOR_PROXY": proxy},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "RAPTOR_PROXY"):
                    runpy.run_path(str(ROOT / "config.py"))

    def test_accepts_exact_and_wildcard_proxy_bypasses(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAPTOR_PROXY": "socks5h://proxy.example:1080",
                "RAPTOR_NO_PROXY": "127.0.0.1, Models.Example ,*.google.com",
            },
            clear=True,
        ):
            values = runpy.run_path(str(ROOT / "config.py"))
        self.assertEqual(
            values["RAPTOR_NO_PROXY"],
            ("127.0.0.1", "models.example", "*.google.com"),
        )

    def test_rejects_invalid_proxy_bypasses(self) -> None:
        for bypass in (
            "google.com,",
            "*",
            "api.*.example",
            "*.127.0.0.1",
            "https://google.com",
            "google.com:443",
            "bad_host.example",
            "google.com,GOOGLE.COM",
        ):
            with self.subTest(bypass=bypass), patch.dict(
                os.environ,
                {
                    "RAPTOR_PROXY": "socks5h://proxy.example:1080",
                    "RAPTOR_NO_PROXY": bypass,
                },
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "RAPTOR_NO_PROXY"):
                    runpy.run_path(str(ROOT / "config.py"))

    def test_proxy_bypasses_require_a_proxy(self) -> None:
        with patch.dict(
            os.environ,
            {"RAPTOR_NO_PROXY": "models.example"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "RAPTOR_NO_PROXY requires RAPTOR_PROXY",
            ):
                runpy.run_path(str(ROOT / "config.py"))


class ShellConfigurationTests(unittest.TestCase):
    def test_shell_timeout_defaults_to_unlimited(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            values = runpy.run_path(str(ROOT / "config.py"))

        self.assertEqual(values["SHELL_TIMEOUT"], 0)

    def test_shell_timeout_accepts_a_positive_deadline(self) -> None:
        with patch.dict(os.environ, {"SHELL_TIMEOUT": "900"}, clear=True):
            values = runpy.run_path(str(ROOT / "config.py"))

        self.assertEqual(values["SHELL_TIMEOUT"], 900)


class ContextBudgetTests(unittest.TestCase):

    def test_model_backend_environment_is_not_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MODEL_CONTEXT_TOKENS": "100000",
                "RESPONSES_BASE_URL": "http://main.example/v1",
                "RESPONSES_API_KEY": "main-key",
                "RESPONSES_MODEL": "main-model",
                "RESPONSES_REASONING_EFFORT": "high",
                "RESPONSES_MAX_RETRIES": "9",
                "RESPONSES_RETRY_BASE_SECONDS": "4.0",
            },
            clear=True,
        ):
            values = runpy.run_path(str(ROOT / "config.py"))

        self.assertNotIn("MODEL_CONTEXT_TOKENS", values)
        self.assertNotIn("RESPONSES_BASE_URL", values)
        self.assertNotIn("SUBAGENT_RESPONSES_MODEL", values)

    def test_budget_is_derived_from_each_supplied_model_window(self) -> None:
        with (
            patch.object(config, "CONTEXT_COMPACT_RATIO", 0.82),
            patch.object(config, "CONTEXT_SAFETY_TOKENS", 4_096),
        ):
            self.assertEqual(config.model_context_input_budget(100_000), 82_000)
            self.assertEqual(config.model_context_input_budget(20_000), 15_904)

    def test_old_model_retry_environment_is_ignored(self) -> None:
        with patch.dict(
            os.environ,
            {"RESPONSES_MAX_RETRIES": "-1"},
            clear=True,
        ):
            values = runpy.run_path(str(ROOT / "config.py"))
        self.assertNotIn("RESPONSES_MAX_RETRIES", values)

    def test_rejects_unknown_boolean_environment_values(self) -> None:
        with patch.dict(
            os.environ,
            {"CHAT_STREAMING": "sometimes"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "CHAT_STREAMING must be a boolean",
            ):
                runpy.run_path(str(ROOT / "config.py"))

    def test_rejects_nonfinite_float_environment_values(self) -> None:
        with patch.dict(
            os.environ,
            {"CHAT_STREAM_INTERVAL": "nan"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "CHAT_STREAM_INTERVAL must be finite",
            ):
                runpy.run_path(str(ROOT / "config.py"))


class ReadmeEnvironmentTests(unittest.TestCase):
    def test_every_config_environment_variable_is_documented(self) -> None:
        source = (ROOT / "config.py").read_text()
        readme = (ROOT / "README.md").read_text()
        names = set(
            re.findall(
                r'os\.getenv\(\s*["\']([A-Z0-9_]+)',
                source,
            )
        )
        names.update(
            re.findall(
                r'_env_[a-z_]+\(\s*["\']([A-Z0-9_]+)',
                source,
            )
        )

        missing = sorted(
            name for name in names if f"`{name}`" not in readme
        )

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
