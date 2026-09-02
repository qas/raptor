import json
import tempfile
import unittest
from pathlib import Path

from raptor.build_info import (
    BuildInfo,
    load_build_info,
    nightly_build_info,
    release_build_info,
    write_build_info,
)


_REVISION = "a83797d0" + ("1" * 32)
_BUILT_AT = "2026-08-30T15:04:05Z"


class BuildInfoTests(unittest.TestCase):
    def test_stable_release_uses_tag_version_and_revision(self) -> None:
        info = release_build_info("v2.5.0", _REVISION, _BUILT_AT)
        self.assertEqual(info.channel, "stable")
        self.assertEqual(info.version, "2.5.0")
        self.assertEqual(info.display(), "2.5.0 (a83797d0)")

    def test_prerelease_stages_are_supported(self) -> None:
        for stage in ("alpha.1", "beta.2", "rc.3"):
            with self.subTest(stage=stage):
                info = release_build_info(
                    f"v2.5.0-{stage}",
                    _REVISION,
                    _BUILT_AT,
                )
                self.assertEqual(info.channel, "prerelease")
                self.assertEqual(info.version, f"2.5.0-{stage}")

    def test_release_rejects_noncanonical_tags(self) -> None:
        for tag in (
            "2.5.0",
            "v2.5",
            "v02.5.0",
            "v2.5.0-preview.1",
            "v2.5.0-rc.01",
        ):
            with self.subTest(tag=tag):
                with self.assertRaises(ValueError):
                    release_build_info(tag, _REVISION, _BUILT_AT)

    def test_nightly_display_contains_revision_and_date(self) -> None:
        info = nightly_build_info(_REVISION, _BUILT_AT)
        self.assertEqual(info.channel, "nightly")
        self.assertIsNone(info.version)
        self.assertEqual(info.display(), "nightly (a83797d0, 2026-08-30)")

    def test_generated_identity_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = release_build_info(
                "v2.5.0-rc.1",
                _REVISION,
                _BUILT_AT,
            )
            write_build_info(expected, root / ".raptor-build.json")
            self.assertEqual(load_build_info(root), expected)

    def test_source_checkout_uses_project_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                '[project]\nversion = "3.1.4"\n',
                encoding="utf-8",
            )
            self.assertEqual(
                load_build_info(root),
                BuildInfo("source", "3.1.4"),
            )

    def test_generated_identity_rejects_channel_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".raptor-build.json").write_text(
                json.dumps(
                    {
                        "built_at": _BUILT_AT,
                        "channel": "stable",
                        "revision": _REVISION,
                        "version": "2.5.0-rc.1",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_build_info(root)

    def test_generated_identity_requires_full_revision(self) -> None:
        with self.assertRaises(ValueError):
            nightly_build_info("a83797d0", _BUILT_AT)


if __name__ == "__main__":
    unittest.main()
