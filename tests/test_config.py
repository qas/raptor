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

    def test_subagent_settings_do_not_inherit_main_environment(self) -> None:
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

        self.assertEqual(values["MODEL_CONTEXT_TOKENS"], 100_000)
        self.assertEqual(values["SUBAGENT_MODEL_CONTEXT_TOKENS"], 0)
        self.assertEqual(
            values["SUBAGENT_RESPONSES_BASE_URL"],
            "http://127.0.0.1:8000/v1",
        )
        self.assertEqual(values["SUBAGENT_RESPONSES_API_KEY"], "")
        self.assertEqual(values["SUBAGENT_RESPONSES_MODEL"], "")
        self.assertIsNone(values["SUBAGENT_RESPONSES_REASONING_EFFORT"])
        self.assertEqual(
            values["SUBAGENT_RESPONSES_REASONING_SUMMARY"],
            "auto",
        )
        self.assertEqual(values["SUBAGENT_RESPONSES_MAX_RETRIES"], 3)
        self.assertEqual(
            values["SUBAGENT_RESPONSES_RETRY_BASE_SECONDS"],
            0.5,
        )

    def test_main_and_subagent_windows_are_independent(self) -> None:
        with (
            patch.object(config, "MODEL_CONTEXT_TOKENS", 100_000),
            patch.object(config, "SUBAGENT_MODEL_CONTEXT_TOKENS", 20_000),
            patch.object(config, "CONTEXT_COMPACT_RATIO", 0.82),
            patch.object(config, "CONTEXT_SAFETY_TOKENS", 4_096),
        ):
            self.assertEqual(config.context_input_budget(), 82_000)
            self.assertEqual(config.subagent_context_input_budget(), 15_904)

    def test_rejects_out_of_range_environment_values(self) -> None:
        with patch.dict(
            os.environ,
            {"RESPONSES_MAX_RETRIES": "-1"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "RESPONSES_MAX_RETRIES must be at least 0",
            ):
                runpy.run_path(str(ROOT / "config.py"))

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
