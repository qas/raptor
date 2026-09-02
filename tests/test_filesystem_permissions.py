import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from raptor.shell import filesystem_permissions
from raptor.shell.filesystem_permissions import FileAccessPolicy


class FileAccessPolicyTests(unittest.TestCase):
    def test_compiled_glob_matches_path_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cases = (
                ("**/.env", ".env", True),
                ("**/.env", "service/.env", True),
                ("**/.env", "service/.env.example", False),
                ("services/*/config?.toml", "services/api/config1.toml", True),
                ("services/*/config?.toml", "services/api/config.toml", False),
                ("private/**", "private", True),
                ("private/**", "private/nested/token", True),
                ("data/[ab].pem", "data/a.pem", True),
                ("data/[ab].pem", "data/c.pem", False),
            )
            for configured, candidate, expected in cases:
                with self.subTest(pattern=configured, path=candidate):
                    policy = FileAccessPolicy.create(root, [configured])
                    self.assertEqual(
                        policy.patterns[0].matches(root / candidate),
                        expected,
                    )

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

    def test_glob_expansion_masks_subtree_at_depth_limit(self) -> None:
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

            denied = policy.prepare_denied_paths()

            self.assertEqual(
                denied,
                (
                    filesystem_permissions.PreparedDeniedPath(
                        root / ".env", False
                    ),
                    filesystem_permissions.PreparedDeniedPath(
                        root / "one" / "two", False
                    ),
                ),
            )

    def test_glob_expansion_masks_unreadable_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            restricted = root / "restricted"
            restricted.mkdir()
            real_scandir = filesystem_permissions.os.scandir

            def scandir(path: str | Path):
                if Path(path) == restricted:
                    raise PermissionError("restricted")
                return real_scandir(path)

            policy = FileAccessPolicy.create(root, ["**/.env"])
            with patch.object(
                filesystem_permissions.os,
                "scandir",
                side_effect=scandir,
            ):
                denied = policy.prepare_denied_paths()

            self.assertEqual(
                denied,
                (
                    filesystem_permissions.PreparedDeniedPath(
                        restricted, False
                    ),
                ),
            )

    def test_glob_expansion_masks_subtree_when_iteration_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            unstable = root / "unstable"
            unstable.mkdir()
            real_scandir = filesystem_permissions.os.scandir

            class FailingScan:
                def __enter__(self):
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

                def __next__(self):
                    raise OSError("directory changed during scan")

            def scandir(path: str | Path):
                if Path(path) == unstable:
                    return FailingScan()
                return real_scandir(path)

            policy = FileAccessPolicy.create(root, ["**/.env"])
            with patch.object(
                filesystem_permissions.os,
                "scandir",
                side_effect=scandir,
            ):
                denied = policy.prepare_denied_paths()

            self.assertEqual(
                denied,
                (
                    filesystem_permissions.PreparedDeniedPath(
                        unstable, False
                    ),
                ),
            )

    def test_patterns_with_shared_scan_root_use_one_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            nested = root / "nested"
            nested.mkdir()
            secret = nested / ".env"
            secret.write_text("secret")
            certificate = nested / "certificate.pem"
            certificate.write_text("certificate")
            real_scandir = filesystem_permissions.os.scandir
            root_scans = 0

            def scandir(path: str | Path):
                nonlocal root_scans
                if Path(path) == root:
                    root_scans += 1
                return real_scandir(path)

            policy = FileAccessPolicy.create(
                root,
                ["**/.env", "**/*.pem"],
            )
            with patch.object(
                filesystem_permissions.os,
                "scandir",
                side_effect=scandir,
            ):
                denied = policy.prepare_denied_paths()

            self.assertEqual(root_scans, 1)
            self.assertEqual(
                {item.path for item in denied},
                {secret, certificate},
            )

    def test_glob_expansion_ignores_irrelevant_directory_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "workspace"
            root.mkdir()
            target = base / "external"
            target.mkdir()
            (target / "visible.txt").write_text("visible")
            secret = root / ".env"
            secret.write_text("secret")
            (root / "linked").symlink_to(target, target_is_directory=True)
            policy = FileAccessPolicy.create(root, ["**/.env"])

            denied = policy.prepare_denied_paths()

            self.assertEqual(
                denied,
                (filesystem_permissions.PreparedDeniedPath(secret, False),),
            )

    def test_glob_does_not_cross_directory_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "workspace"
            root.mkdir()
            target = base / "external"
            target.mkdir()
            secret = target / "restricted.txt"
            secret.write_text("secret")
            (root / "linked").symlink_to(target, target_is_directory=True)
            policy = FileAccessPolicy.create(root, ["**/restricted.txt"])

            self.assertEqual(policy.prepare_denied_paths(), ())

    def test_exact_match_through_symlink_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "workspace"
            root.mkdir()
            target = base / "missing-target"
            (root / "linked").symlink_to(target)
            policy = FileAccessPolicy.create(root, ["linked"])

            with self.assertRaisesRegex(RuntimeError, "through symlink"):
                policy.prepare_denied_paths()

    def test_glob_does_not_cross_directory_symlink_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            secret = root / ".env"
            secret.write_text("secret")
            (root / "loop").symlink_to(root, target_is_directory=True)
            policy = FileAccessPolicy.create(root, ["**/loop/.env"])

            self.assertEqual(policy.prepare_denied_paths(), ())

    def test_glob_with_explicit_symlink_prefix_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "workspace"
            root.mkdir()
            target = base / "external"
            target.mkdir()
            (root / "linked").symlink_to(target, target_is_directory=True)
            policy = FileAccessPolicy.create(root, ["linked/**/*.pem"])

            with self.assertRaisesRegex(RuntimeError, "through symlink"):
                policy.prepare_denied_paths()

    def test_glob_expansion_ignores_file_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target-file"
            target.write_text("binary")
            (root / "linked-file").symlink_to(target)
            secret = root / ".env"
            secret.write_text("secret")
            policy = FileAccessPolicy.create(root, ["**/.env"])

            denied = policy.prepare_denied_paths()

            self.assertEqual(
                denied,
                (filesystem_permissions.PreparedDeniedPath(secret, False),),
            )

    def test_glob_expansion_ignores_unresolvable_file_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "looped-link").symlink_to("looped-link")
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

    def test_denied_path_resolution_enforces_match_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = FileAccessPolicy.create(
                Path(directory),
                ["first.secret", "second.secret"],
            )

            with (
                patch.object(
                    filesystem_permissions,
                    "MAX_DENY_READ_MATCHES",
                    1,
                ),
                self.assertRaisesRegex(RuntimeError, "more than 1 paths"),
            ):
                policy.prepare_denied_paths()

    def test_supervisor_payload_round_trips_validated_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = FileAccessPolicy.create(
                Path(directory), [".env", "**/*.pem"], 7
            )

            restored = FileAccessPolicy.from_supervisor_payload(
                policy.supervisor_payload()
            )

            self.assertEqual(restored, policy)

if __name__ == "__main__":
    unittest.main()
