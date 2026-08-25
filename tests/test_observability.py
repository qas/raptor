import json
import unittest
from unittest.mock import patch

from observability import log_event, log_shell_start, redact_sensitive


class ObservabilityTests(unittest.TestCase):
    def test_redacts_credentials_without_hiding_token_metrics(self) -> None:
        value = {
            "authorization": "Bearer top-secret",
            "responses_api_key": "backend-secret",
            "tg_bot_token": "telegram-secret",
            "input_tokens": 42,
            "message": (
                "POST https://api.telegram.org/bot123456:abcdef/sendMessage "
                "Authorization: Bearer another-secret"
            ),
        }

        redacted = redact_sensitive(value)

        self.assertEqual(redacted["authorization"], "[REDACTED]")
        self.assertEqual(redacted["responses_api_key"], "[REDACTED]")
        self.assertEqual(redacted["tg_bot_token"], "[REDACTED]")
        self.assertEqual(redacted["input_tokens"], 42)
        self.assertNotIn("123456:abcdef", redacted["message"])
        self.assertNotIn("another-secret", redacted["message"])

    def test_log_event_redacts_nested_values(self) -> None:
        with patch("builtins.print") as output:
            log_event(
                "test",
                "credential",
                {"nested": {"token": "secret-value"}},
            )

        record = json.loads(output.call_args.args[0])
        self.assertEqual(record["data"]["nested"]["token"], "[REDACTED]")

    def test_log_event_bounds_large_payloads(self) -> None:
        with patch("builtins.print") as output:
            log_event("test", "large", {"message": "x" * 10_000})

        record = json.loads(output.call_args.args[0])
        message = record["data"]["message"]
        self.assertLess(len(message), 2_100)
        self.assertIn("chars omitted", message)

    def test_audit_event_preserves_intentional_payload(self) -> None:
        command = "RESPONSES_API_KEY=operator-secret run-model"
        with patch("builtins.print") as output:
            log_shell_start(
                session_id="shell-1",
                command=command,
                chat_id="telegram:123",
                parent_session_id="main-1",
                pid=1234,
                timeout=None,
                tty=False,
            )

        record = json.loads(output.call_args.args[0])
        self.assertTrue(record["audit"])
        self.assertEqual(record["event"], "authorized")
        self.assertEqual(record["data"]["command"], command)

    def test_shell_audit_rejects_oversized_command(self) -> None:
        with patch("observability.MAX_TOOL_OUTPUT", 10):
            with self.assertRaisesRegex(ValueError, "exceeds 10"):
                log_shell_start(
                    session_id="shell-1",
                    command="x" * 11,
                    chat_id="telegram:123",
                    parent_session_id=None,
                    pid=1234,
                    timeout=None,
                    tty=False,
                )

    def test_redacts_environment_and_header_credentials(self) -> None:
        value = (
            "RESPONSES_API_KEY=backend-secret "
            "TG_BOT_TOKEN=telegram-secret "
            "X-API-Key: proxy-secret"
        )

        redacted = redact_sensitive(value)

        self.assertNotIn("backend-secret", redacted)
        self.assertNotIn("telegram-secret", redacted)
        self.assertNotIn("proxy-secret", redacted)


if __name__ == "__main__":
    unittest.main()
