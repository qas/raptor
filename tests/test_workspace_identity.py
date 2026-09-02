"""Workspace identity bootstrap tests."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-workspace-identity-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME / ".raptor")
os.environ["AGENT_WORKDIR"] = str(_HOME)

from raptor.model import responses
import workspace_identity


class WorkspaceIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = workspace_identity._workspace_identity

    def tearDown(self) -> None:
        workspace_identity._workspace_identity = self.previous

    def test_initialization_creates_and_loads_missing_templates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            created = workspace_identity.initialize_workspace_identity(root)

            self.assertEqual(created, ("AGENTS.md", "MEMORY.md"))
            self.assertTrue((root / "AGENTS.md").is_file())
            self.assertTrue((root / "MEMORY.md").is_file())
            instructions = responses.instructions()
            self.assertIn("WORKSPACE AGENTS.md:", instructions)
            self.assertIn("WORKSPACE MEMORY.md:", instructions)

    def test_initialization_preserves_existing_operator_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "AGENTS.md"
            memory = root / "MEMORY.md"
            agents.write_text("custom agent", encoding="utf-8")
            memory.write_text("custom memory", encoding="utf-8")

            created = workspace_identity.initialize_workspace_identity(root)

            self.assertEqual(created, ())
            self.assertEqual(agents.read_text(encoding="utf-8"), "custom agent")
            self.assertEqual(memory.read_text(encoding="utf-8"), "custom memory")
            instructions = responses.instructions()
            self.assertIn("custom agent", instructions)
            self.assertIn("custom memory", instructions)

    def test_initialization_rejects_oversized_identity_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "AGENTS.md").write_text("too large", encoding="utf-8")
            with patch.object(
                workspace_identity,
                "MAX_WORKSPACE_IDENTITY_FILE_BYTES",
                3,
            ):
                with self.assertRaisesRegex(RuntimeError, "AGENTS.md exceeds"):
                    workspace_identity.initialize_workspace_identity(root)

            self.assertFalse((root / "MEMORY.md").exists())

    def test_initialization_rejects_non_utf8_identity_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "AGENTS.md").write_bytes(b"\xff")

            with self.assertRaisesRegex(RuntimeError, "must be UTF-8"):
                workspace_identity.initialize_workspace_identity(root)

    def test_initialization_rejects_symlinked_identity_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.write_text("external instructions", encoding="utf-8")
            (root / "AGENTS.md").symlink_to(outside)

            with self.assertRaisesRegex(RuntimeError, "regular workspace file"):
                workspace_identity.initialize_workspace_identity(root)
