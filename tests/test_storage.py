import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage import write_text_atomic


class AtomicWriteTests(unittest.TestCase):
    def test_replaces_content_and_applies_mode(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="raptor-storage-"))
        path = root / "state.json"

        write_text_atomic(path, "complete", mode=0o600)

        self.assertEqual(path.read_text(), "complete")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(root.glob("*.tmp")), [])

    def test_removes_temporary_file_after_replace_failure(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="raptor-storage-"))
        path = root / "state.json"

        with patch.object(os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                write_text_atomic(path, "incomplete")

        self.assertFalse(path.exists())
        self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
