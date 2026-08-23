import os
import re
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import config


ROOT = Path(__file__).resolve().parent.parent


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

        missing = sorted(
            name for name in names if f"`{name}`" not in readme
        )

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
