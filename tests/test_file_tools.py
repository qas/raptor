import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_HOME = Path(tempfile.mkdtemp(prefix="raptor-file-tools-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)
os.environ.setdefault("TG_BOT_TOKEN", "test-token")
os.environ.setdefault("TG_USER_ID", "1")

import tools
from config import TOOLS
from raptor.shell.filesystem_permissions import FileAccessPolicy


class FileToolTests(unittest.TestCase):
    def test_read_file_streams_and_bounds_one_long_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "large.txt").write_text("x" * 10_000)
            with (
                patch.object(tools, "AGENT_WORKDIR", root),
                patch.object(tools, "MAX_TOOL_OUTPUT", 1024),
                patch.object(tools, "FILE_READ_CHUNK_CHARS", 128),
            ):
                result = tools.read_file_tool({"path": "large.txt"})

        self.assertTrue(result["ok"])
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["text"]), 1024)
        self.assertIn("[truncated]", result["text"])

    def test_read_file_marks_unread_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lines.txt").write_text("first\nsecond\nthird\n")
            with patch.object(tools, "AGENT_WORKDIR", root):
                result = tools.read_file_tool(
                    {
                        "path": "lines.txt",
                        "start_line": 2,
                        "max_lines": 1,
                    }
                )

        self.assertEqual(result["text"], "second")
        self.assertTrue(result["truncated"])

    def test_list_directory_reports_bounded_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("c", "a", "b"):
                (root / name).write_text(name)
            with patch.object(tools, "AGENT_WORKDIR", root):
                result = tools.list_dir_tool({"max_entries": 2})

        self.assertEqual(
            [entry["name"] for entry in result["entries"]],
            ["a", "b"],
        )
        self.assertTrue(result["truncated"])

    def test_edit_rejects_file_above_memory_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "large.txt").write_text("unchanged")
            with (
                patch.object(tools, "AGENT_WORKDIR", root),
                patch.object(tools, "MAX_EDIT_FILE_BYTES", 4),
            ):
                result = tools.edit_file_tool(
                    {
                        "path": "large.txt",
                        "old_text": "unchanged",
                        "new_text": "changed",
                    }
                )

        self.assertFalse(result["ok"])
        self.assertIn("edit limit", result["error"])

    def test_edit_rejects_output_expansion_above_memory_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file.txt").write_text("aaaa")
            with (
                patch.object(tools, "AGENT_WORKDIR", root),
                patch.object(tools, "MAX_EDIT_FILE_BYTES", 8),
            ):
                result = tools.edit_file_tool(
                    {
                        "path": "file.txt",
                        "old_text": "a",
                        "new_text": "abc",
                        "replace_all": True,
                    }
                )

        self.assertFalse(result["ok"])
        self.assertIn("edited file exceeds", result["error"])

    def test_mutations_reject_oversized_direct_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "file.txt").write_text("old")
            with (
                patch.object(tools, "AGENT_WORKDIR", root),
                patch.object(tools, "MAX_TOOL_OUTPUT", 4),
            ):
                written = tools.write_file_tool(
                    {"path": "new.txt", "content": "12345"}
                )
                edited = tools.edit_file_tool(
                    {
                        "path": "file.txt",
                        "old_text": "old",
                        "new_text": "12345",
                    }
                )

        self.assertFalse(written["ok"])
        self.assertFalse(edited["ok"])

    def test_file_mutation_arguments_match_tool_budget(self) -> None:
        schemas = {tool["name"]: tool for tool in TOOLS}
        write = schemas["write_file"]["parameters"]["properties"]
        edit = schemas["edit_file"]["parameters"]["properties"]

        self.assertEqual(write["content"]["maxLength"], tools.MAX_TOOL_OUTPUT)
        self.assertEqual(edit["old_text"]["maxLength"], tools.MAX_TOOL_OUTPUT)
        self.assertEqual(edit["new_text"]["maxLength"], tools.MAX_TOOL_OUTPUT)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(tools, "AGENT_WORKDIR", Path(directory)):
                result = tools.list_dir_tool({"max_entries": 1})
        self.assertLessEqual(len(json.dumps(result)), tools.MAX_TOOL_OUTPUT)

    def test_workspace_path_accepts_logical_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / "real"
            real.mkdir()
            (real / "file.txt").write_text("ok")
            link = Path(directory) / "link"
            link.symlink_to(real)
            with patch.object(tools, "AGENT_WORKDIR", link):
                path = tools.workspace_path("file.txt")
                listed = tools.list_dir_tool({"path": "."})
        self.assertEqual(path, (real / "file.txt").resolve())
        self.assertEqual(listed["path"], ".")
        self.assertEqual(listed["entries"][0]["name"], "file.txt")

    def test_workspace_path_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(tools, "AGENT_WORKDIR", root):
                with self.assertRaises(ValueError):
                    tools.workspace_path("../outside")

    def test_deny_read_blocks_reads_and_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret = root / ".env"
            secret.write_text("secret")
            policy = FileAccessPolicy.create(root, ["**/.env"])
            with (
                patch.object(tools, "AGENT_WORKDIR", root),
                patch.object(tools, "FILESYSTEM_POLICY", policy),
            ):
                with self.assertRaisesRegex(PermissionError, "deny_read"):
                    tools.read_file_tool({"path": ".env"})
                with self.assertRaisesRegex(PermissionError, "deny_read"):
                    tools.write_file_tool(
                        {"path": ".env", "content": "replacement"}
                    )

            self.assertEqual(secret.read_text(), "secret")

    def test_directory_listing_omits_denied_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("secret")
            (root / "visible.txt").write_text("ok")
            policy = FileAccessPolicy.create(root, [".env"])
            with (
                patch.object(tools, "AGENT_WORKDIR", root),
                patch.object(tools, "FILESYSTEM_POLICY", policy),
            ):
                result = tools.list_dir_tool({"path": "."})

            self.assertEqual(
                [entry["name"] for entry in result["entries"]],
                ["visible.txt"],
            )


if __name__ == "__main__":
    unittest.main()
