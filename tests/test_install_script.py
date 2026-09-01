import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_INSTALL = _ROOT / "install.sh"


def _helper_source(name: str) -> str:
    source = _INSTALL.read_text(encoding="utf-8")
    start = source.index(f"{name}()")
    end = source.index("\n}\n", start)
    return source[start:end + 2]


def _run_helper(
    name: str,
    *arguments: str,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    quoted = " ".join(f'"${index}"' for index in range(1, len(arguments) + 1))
    script = _helper_source(name) + f"\n{name} {quoted}\n"
    return subprocess.run(
        ["sh", "-s", *arguments],
        input=script,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_helpers(
    names: tuple[str, ...],
    invocation: str,
    *,
    arguments: tuple[str, ...],
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    source = "\n".join(_helper_source(name) for name in names)
    script = source + f"\n{invocation}\n"
    return subprocess.run(
        ["sh", "-s", *arguments],
        input=script,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _run_uninstall(
    install_root: Path,
    bin_dir: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["RAPTOR_INSTALL_ROOT"] = str(install_root)
    env["RAPTOR_BIN_DIR"] = str(bin_dir)
    return subprocess.run(
        ["sh", str(_INSTALL), "--uninstall"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _fake_install(root: Path) -> tuple[Path, Path, Path]:
    install_root = root / "share" / "raptor"
    bin_dir = root / "bin"
    version_dir = install_root / "versions" / "v0.1.0"
    version_dir.mkdir(parents=True)
    binary = version_dir / "raptor"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    bin_dir.mkdir()
    (bin_dir / "raptor").symlink_to(binary)
    return install_root, bin_dir, version_dir


class InstallScriptTests(unittest.TestCase):
    def test_release_version_accepts_semver_tags(self) -> None:
        completed = _run_helper("is_release_version", "v0.1.0", cwd=_ROOT)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_release_version_accepts_prerelease_tags(self) -> None:
        for version in (
            "v2.5.0-alpha.1",
            "v2.5.0-beta.2",
            "v2.5.0-rc.3",
        ):
            with self.subTest(version=version):
                completed = _run_helper(
                    "is_release_version",
                    version,
                    cwd=_ROOT,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_release_version_accepts_nightly_channel(self) -> None:
        completed = _run_helper("is_release_version", "nightly", cwd=_ROOT)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_release_version_rejects_unsupported_identifiers(self) -> None:
        for version in (
            "latest",
            "v1.2",
            "v1.2.3-preview.1",
            "v1.2.3-rc",
            "v01.2.3",
            "v1.2.3-rc.01",
        ):
            with self.subTest(version=version):
                completed = _run_helper(
                    "is_release_version",
                    version,
                    cwd=_ROOT,
                )
                self.assertNotEqual(completed.returncode, 0)

    def test_release_version_rejects_path_escape(self) -> None:
        completed = _run_helper(
            "is_release_version",
            "v1.0.0/../../..",
            cwd=_ROOT,
        )
        self.assertNotEqual(completed.returncode, 0)

    def test_absolute_dir_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            completed = _run_helper("absolute_dir", "rel-install", cwd=cwd)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            resolved = Path(completed.stdout.strip())
            self.assertTrue(resolved.is_absolute())
            self.assertEqual(resolved, (cwd / "rel-install").resolve())

    def test_resolve_dir_does_not_create(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            completed = _run_helper("resolve_dir", "missing", cwd=cwd)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((cwd / "missing").exists())
            self.assertEqual(
                Path(completed.stdout.strip()),
                (cwd / "missing").resolve(),
            )

    def test_physical_dir_resolves_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real)
            completed = _run_helper("physical_dir", str(link), cwd=root)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(Path(completed.stdout.strip()), real.resolve())

    def test_is_under_accepts_child(self) -> None:
        completed = _run_helper(
            "is_under",
            "/opt/raptor/versions/v0.1.0/raptor",
            "/opt/raptor",
            cwd=_ROOT,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_is_under_rejects_sibling_prefix(self) -> None:
        completed = _run_helper(
            "is_under",
            "/opt/raptor-evil/bin",
            "/opt/raptor",
            cwd=_ROOT,
        )
        self.assertNotEqual(completed.returncode, 0)

    def test_linux_sandbox_install_command_is_distribution_specific(
        self,
    ) -> None:
        for distribution, expected in (
            ("ubuntu", "sudo apt-get install bubblewrap"),
            ("fedora", "sudo dnf install bubblewrap"),
            ("arch", "sudo pacman -S bubblewrap"),
            ("unknown", "Install bubblewrap with your system package manager"),
        ):
            with self.subTest(distribution=distribution):
                completed = _run_helper(
                    "linux_sandbox_install_command",
                    distribution,
                    cwd=_ROOT,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout.strip(), expected)

    def test_linux_distribution_id_reads_data_without_sourcing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "executed"
            os_release = root / "os-release"
            os_release.write_text(
                f'ID="ubuntu"\nNAME=$(touch {marker})\n',
                encoding="utf-8",
            )

            completed = _run_helper(
                "linux_distribution_id",
                str(os_release),
                cwd=_ROOT,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "ubuntu")
            self.assertFalse(marker.exists())

    def test_linux_sandbox_rejects_untrusted_bubblewrap_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stat = root / "stat"
            stat.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$FAKE_STAT_METADATA\"\n",
                encoding="utf-8",
            )
            stat.chmod(0o755)
            base_env = os.environ.copy()
            base_env["PATH"] = f"{root}:{base_env['PATH']}"

            for metadata in ("1000 755", "0 775", "0 757", "root 755"):
                with self.subTest(metadata=metadata):
                    env = base_env | {"FAKE_STAT_METADATA": metadata}
                    completed = _run_helpers(
                        ("trusted_linux_bwrap",),
                        'trusted_linux_bwrap "$1"',
                        arguments=("/usr/bin/bwrap",),
                        cwd=_ROOT,
                        env=env,
                    )
                    self.assertNotEqual(completed.returncode, 0)

            env = base_env | {"FAKE_STAT_METADATA": "0 755"}
            completed = _run_helpers(
                ("trusted_linux_bwrap",),
                'trusted_linux_bwrap "$1"',
                arguments=("/usr/bin/bwrap",),
                cwd=_ROOT,
                env=env,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_installer_identifies_ubuntu_user_namespace_denial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            bwrap = bin_dir / "bwrap"
            bwrap.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' "
                "'bwrap: setting up uid map: Permission denied' >&2\n"
                "exit 1\n",
                encoding="utf-8",
            )
            bwrap.chmod(0o755)
            stat = bin_dir / "stat"
            stat.write_text(
                "#!/bin/sh\nprintf '%s\\n' '0 755'\n",
                encoding="utf-8",
            )
            stat.chmod(0o755)
            os_release = root / "os-release"
            os_release.write_text("ID=ubuntu\n", encoding="utf-8")
            restriction = root / "restriction"
            restriction.write_text("1\n", encoding="utf-8")
            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            completed = _run_helpers(
                (
                    "linux_sandbox_install_command",
                    "linux_distribution_id",
                    "ubuntu_userns_restriction_enabled",
                    "trusted_linux_bwrap",
                    "report_linux_sandbox",
                ),
                'report_linux_sandbox "$1" "$2" "$3"',
                arguments=(
                    str(root),
                    str(os_release),
                    str(restriction),
                ),
                cwd=_ROOT,
                env=env,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Ubuntu AppArmor denied", completed.stdout)
            self.assertIn("#ubuntu-apparmor", completed.stdout)

    def test_unknown_argument_is_rejected(self) -> None:
        completed = subprocess.run(
            ["sh", str(_INSTALL), "--nope"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Unknown argument", completed.stderr)

    def test_uninstall_rejects_extra_arguments(self) -> None:
        completed = subprocess.run(
            ["sh", str(_INSTALL), "--uninstall", "extra"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Unexpected arguments", completed.stderr)

    def test_uninstall_removes_owned_link_and_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install_root, bin_dir, version_dir = _fake_install(root)
            workspace = root / "project" / ".raptor"
            workspace.mkdir(parents=True)
            (workspace / "state.json").write_text("{}", encoding="utf-8")
            completed = _run_uninstall(install_root, bin_dir)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((bin_dir / "raptor").exists())
            self.assertFalse(version_dir.exists())
            self.assertFalse((install_root / "versions").exists())
            self.assertFalse(install_root.exists())
            self.assertTrue((workspace / "state.json").exists())

    def test_uninstall_leaves_foreign_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install_root, bin_dir, _version_dir = _fake_install(root)
            other = root / "other" / "raptor"
            other.parent.mkdir()
            other.write_text("other\n", encoding="utf-8")
            link = bin_dir / "raptor"
            link.unlink()
            link.symlink_to(other)
            completed = _run_uninstall(install_root, bin_dir)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), other.resolve())
            self.assertIn("Leaving", completed.stdout)

    def test_uninstall_refuses_while_binary_is_running(self) -> None:
        sleep = shutil.which("sleep")
        if sleep is None:
            self.skipTest("sleep is required")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install_root, bin_dir, version_dir = _fake_install(root)
            binary = version_dir / "raptor"
            shutil.copy(sleep, binary)
            binary.chmod(0o755)
            process = subprocess.Popen([str(binary), "30"])
            try:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    exe = Path("/proc") / str(process.pid) / "exe"
                    if exe.exists() or process.poll() is not None:
                        break
                    time.sleep(0.01)
                completed = _run_uninstall(install_root, bin_dir)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("running", completed.stderr)
                self.assertTrue((install_root / "versions").exists())
                self.assertTrue((bin_dir / "raptor").is_symlink())
            finally:
                process.kill()
                process.wait(timeout=2)

    def test_uninstall_refuses_running_binary_through_symlinked_root(self) -> None:
        sleep = shutil.which("sleep")
        if sleep is None:
            self.skipTest("sleep is required")
        if not Path("/proc").is_dir():
            self.skipTest("requires /proc")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_root = root / "real" / "raptor"
            version_dir = real_root / "versions" / "v0.1.0"
            version_dir.mkdir(parents=True)
            binary = version_dir / "raptor"
            shutil.copy(sleep, binary)
            binary.chmod(0o755)
            link_root = root / "link" / "raptor"
            link_root.parent.mkdir()
            link_root.symlink_to(real_root)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "raptor").symlink_to(
                link_root / "versions" / "v0.1.0" / "raptor"
            )
            process = subprocess.Popen([str(binary), "30"])
            try:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    exe = Path("/proc") / str(process.pid) / "exe"
                    if exe.exists() or process.poll() is not None:
                        break
                    time.sleep(0.01)
                completed = _run_uninstall(link_root, bin_dir)
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("running", completed.stderr)
                self.assertTrue(version_dir.exists())
                self.assertTrue((bin_dir / "raptor").is_symlink())
            finally:
                process.kill()
                process.wait(timeout=2)
