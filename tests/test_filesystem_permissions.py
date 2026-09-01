import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import filesystem_permissions
from filesystem_permissions import FileAccessPolicy


class FileAccessPolicyTests(unittest.TestCase):
    def test_recursive_glob_matches_root_and_nested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            policy = FileAccessPolicy.create(root, ["**/.env"])

            self.assertTrue(policy.denies(root / ".env"))
            self.assertTrue(policy.denies(root / "service" / ".env"))
            self.assertFalse(policy.denies(root / ".env.example"))

    def test_denied_directory_blocks_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            policy = FileAccessPolicy.create(root, ["private"])

            self.assertTrue(policy.denies(root / "private" / "token.txt"))
            self.assertFalse(policy.denies(root / "public" / "token.txt"))

    def test_rejects_root_wide_globs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "non-root"):
                FileAccessPolicy.create(Path(directory), ["/**/*.env"])

    def test_rejects_duplicate_patterns_after_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unique"):
                FileAccessPolicy.create(Path(directory), [".env", " .env "])

    def test_glob_expansion_fails_closed_at_depth_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / ".env").write_text("root")
            nested = root / "one" / "two"
            nested.mkdir(parents=True)
            (nested / ".env").write_text("nested")
            policy = FileAccessPolicy.create(
                root,
                ["**/.env"],
                glob_scan_max_depth=1,
            )

            with self.assertRaisesRegex(RuntimeError, "exceeded depth"):
                policy.prepare_denied_paths()

    def test_glob_expansion_fails_closed_at_directory_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target"
            target.mkdir()
            (target / ".env").write_text("secret")
            (root / "linked").symlink_to(target, target_is_directory=True)
            policy = FileAccessPolicy.create(root, ["**/.env"])

            with self.assertRaisesRegex(
                RuntimeError, "through directory symlink"
            ):
                policy.prepare_denied_paths()

    def test_glob_expansion_ignores_file_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "raptor-real"
            target.write_text("binary")
            bin_dir = root / ".local" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "raptor").symlink_to(target)
            secret = root / ".env"
            secret.write_text("secret")
            policy = FileAccessPolicy.create(root, ["**/.env"])

            denied = policy.prepare_denied_paths()

            self.assertEqual(
                denied,
                (filesystem_permissions.PreparedDeniedPath(secret, False),),
            )

    def test_non_recursive_glob_ignores_irrelevant_deep_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "nested" / "deep").mkdir(parents=True)
            secret = root / "secret.pem"
            secret.write_text("secret")
            policy = FileAccessPolicy.create(
                root,
                ["*.pem"],
                glob_scan_max_depth=0,
            )

            denied = policy.prepare_denied_paths()

            self.assertEqual(
                denied,
                (filesystem_permissions.PreparedDeniedPath(secret, False),),
            )

    def test_overlapping_matches_collapse_to_the_denied_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            private = root / "private"
            private.mkdir()
            (private / "token.env").write_text("secret")
            policy = FileAccessPolicy.create(
                root, ["private", "private/*.env"]
            )

            denied = policy.prepare_denied_paths()

            self.assertEqual(
                denied,
                (filesystem_permissions.PreparedDeniedPath(private, True),),
            )

    def test_shell_payload_round_trips_validated_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = FileAccessPolicy.create(
                Path(directory), [".env", "**/*.pem"], 7
            )

            restored = FileAccessPolicy.from_shell_payload(
                policy.shell_payload()
            )

            self.assertEqual(restored, policy)

    def test_bubblewrap_launch_masks_denied_file_with_no_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            secret = root / ".env"
            secret.write_text("secret")
            policy = FileAccessPolicy.create(root, [".env"])
            with (
                patch.object(filesystem_permissions.sys, "platform", "linux"),
                patch.object(
                    filesystem_permissions,
                    "_trusted_bubblewrap",
                    return_value=Path("/usr/bin/bwrap"),
                ),
            ):
                launch = filesystem_permissions.build_shell_sandbox_launch(
                    "cat .env", policy
                )
            try:
                arguments = launch.argv
                mask_index = arguments.index("--ro-bind-data")
                self.assertEqual(arguments[mask_index - 2 : mask_index], [
                    "--perms",
                    "000",
                ])
                self.assertEqual(arguments[mask_index + 2], str(secret))
            finally:
                launch.cleanup()

    def test_rejects_writable_bubblewrap_executable(self) -> None:
        metadata = unittest.mock.Mock(
            st_mode=filesystem_permissions.stat.S_IFREG | 0o777,
            st_uid=0,
        )
        with (
            patch.object(
                filesystem_permissions.shutil,
                "which",
                return_value="/tmp/bwrap",
            ),
            patch.object(
                filesystem_permissions.Path,
                "resolve",
                return_value=Path("/tmp/bwrap"),
            ),
            patch.object(
                filesystem_permissions.Path,
                "stat",
                return_value=metadata,
            ),
            patch.object(filesystem_permissions.os, "access", return_value=True),
            self.assertRaisesRegex(RuntimeError, "root-owned"),
        ):
            filesystem_permissions._trusted_bubblewrap()

    def test_shell_sandbox_denies_secret_when_platform_helper_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / ".env").write_text("TOP_SECRET")
            (root / "visible.txt").write_text("VISIBLE")
            policy = FileAccessPolicy.create(root, [".env"])
            helper_available = (
                filesystem_permissions.sys.platform.startswith("linux")
                and shutil.which("bwrap") is not None
            ) or (
                filesystem_permissions.sys.platform == "darwin"
                and Path("/usr/bin/sandbox-exec").is_file()
            )
            if not helper_available:
                with self.assertRaisesRegex(RuntimeError, "requires"):
                    filesystem_permissions.build_shell_sandbox_launch(
                        "true", policy
                    )
                return
            launch = filesystem_permissions.build_shell_sandbox_launch(
                "cat .env 2>/dev/null || printf DENIED; cat visible.txt",
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
                patch.object(filesystem_permissions.sys, "platform", "linux"),
                patch.object(
                    filesystem_permissions,
                    "_trusted_bubblewrap",
                    return_value=Path("/usr/bin/bwrap"),
                ),
            ):
                launch = filesystem_permissions.build_shell_sandbox_launch(
                    "true", policy
                )

            self.assertTrue((root / "private").exists())
            launch.cleanup()
            self.assertFalse((root / "private").exists())


if __name__ == "__main__":
    unittest.main()
