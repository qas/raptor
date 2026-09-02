"""Tests for the lock-free version command."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["RAPTOR_HOME"] = tempfile.mkdtemp(prefix="raptor-version-home-")
os.environ["AGENT_WORKDIR"] = os.environ["RAPTOR_HOME"]

from raptor import entrypoint
from version import display_version


class VersionCliTests(unittest.TestCase):
    def test_version_matches_project_and_skips_runtime_lock(self) -> None:
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["raptor", "--version"]),
            patch("raptor.entrypoint.acquire_runtime_lock") as acquire,
            patch.dict(sys.modules, {"raptor.app.application": None}),
            contextlib.redirect_stdout(output),
        ):
            result = entrypoint.run()
        self.assertEqual(result, 0)
        self.assertEqual(
            output.getvalue().strip(),
            f"raptor {display_version()}",
        )
        acquire.assert_not_called()

    def test_short_version_alias(self) -> None:
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["raptor", "-V"]),
            contextlib.redirect_stdout(output),
        ):
            result = entrypoint.run()
        self.assertEqual(result, 0)
        self.assertIn(display_version(), output.getvalue())

    def test_version_rejects_another_process_action(self) -> None:
        with (
            patch.object(sys, "argv", ["raptor", "--version", "--status"]),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as stopped:
                entrypoint.run()
        self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
