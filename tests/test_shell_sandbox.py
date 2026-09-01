import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import shell_sandbox
from filesystem_permissions import FileAccessPolicy


class ShellSandboxTests(unittest.TestCase):
    def test_bubblewrap_launch_masks_denied_file_with_no_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            secret = root / ".env"
            secret.write_text("secret")
            policy = FileAccessPolicy.create(root, [".env"])
            with (
                patch.object(shell_sandbox.sys, "platform", "linux"),
                patch.object(
                    shell_sandbox,
                    "_trusted_bubblewrap",
                    return_value=Path("/usr/bin/bwrap"),
                ),
            ):
                launch = shell_sandbox.build_shell_sandbox_launch(
                    "cat .env", policy
                )
            try:
                self.assertEqual(launch.argv[:2], [
                    "/usr/bin/bwrap",
                    "--args",
                ])
                options_fd = int(launch.argv[2])
                options = os.pread(options_fd, 1_000_000, 0).decode(
                    "utf-8"
                ).split("\0")
                mask_index = options.index("--ro-bind-data")
                self.assertEqual(options[mask_index - 2 : mask_index], [
                    "--perms",
                    "000",
                ])
                self.assertEqual(options[mask_index + 2], str(secret))
                self.assertEqual(
                    launch.argv[3:],
                    ["--", "/bin/bash", "-c", "cat .env"],
                )
            finally:
                launch.cleanup()

    def test_rejects_writable_bubblewrap_executable(self) -> None:
        metadata = unittest.mock.Mock(
            st_mode=shell_sandbox.stat.S_IFREG | 0o777,
            st_uid=0,
        )
        with (
            patch.object(
                shell_sandbox.shutil,
                "which",
                return_value="/tmp/bwrap",
            ),
            patch.object(
                shell_sandbox.Path,
                "resolve",
                return_value=Path("/tmp/bwrap"),
            ),
            patch.object(
                shell_sandbox.Path,
                "stat",
                return_value=metadata,
            ),
            patch.object(shell_sandbox.os, "access", return_value=True),
            self.assertRaisesRegex(RuntimeError, "root-owned"),
        ):
            shell_sandbox._trusted_bubblewrap()

    def test_denies_secret_when_platform_helper_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "workspace"
            root.mkdir()
            external = base / "external"
            external.mkdir()
            (external / ".env").write_text("TOP_SECRET")
            (root / "linked").symlink_to(
                external, target_is_directory=True
            )
            (root / "looped-link").symlink_to("looped-link")
            (root / "visible.txt").write_text("VISIBLE")
            policy = FileAccessPolicy.create(root, ["**/.env"])
            helper_available = (
                shell_sandbox.sys.platform.startswith("linux")
                and shutil.which("bwrap") is not None
            ) or (
                shell_sandbox.sys.platform == "darwin"
                and Path("/usr/bin/sandbox-exec").is_file()
            )
            if not helper_available:
                with self.assertRaisesRegex(RuntimeError, "requires"):
                    shell_sandbox.build_shell_sandbox_launch("true", policy)
                return
            launch = shell_sandbox.build_shell_sandbox_launch(
                "cat linked/.env 2>/dev/null || printf DENIED; "
                "cat visible.txt",
                policy,
            )
            try:
                result = subprocess.run(
                    launch.argv,
                    cwd=root,
                    capture_output=True,
                    check=False,
                    pass_fds=tuple(launch.inherited_fds),
                )
            finally:
                launch.cleanup()

            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(result.stdout, b"DENIEDVISIBLE")
            self.assertNotIn(b"TOP_SECRET", result.stdout + result.stderr)

    def test_missing_exact_path_placeholder_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            policy = FileAccessPolicy.create(root, ["private/secret.env"])
            with (
                patch.object(shell_sandbox.sys, "platform", "linux"),
                patch.object(
                    shell_sandbox,
                    "_trusted_bubblewrap",
                    return_value=Path("/usr/bin/bwrap"),
                ),
            ):
                launch = shell_sandbox.build_shell_sandbox_launch(
                    "true", policy
                )

            self.assertTrue((root / "private").exists())
            launch.cleanup()
            self.assertFalse((root / "private").exists())


if __name__ == "__main__":
    unittest.main()
