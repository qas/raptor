"""OS-enforced launch plans for managed shell commands."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from raptor.shell.filesystem_permissions import FileAccessPolicy, PreparedDeniedPath


SANDBOX_PROBE_TIMEOUT_SECONDS = 5
SANDBOX_PROBE_ERROR_BYTES = 4096


@dataclass
class ShellSandboxLaunch:
    argv: list[str]
    inherited_fds: list[int] = field(default_factory=list)
    placeholders: list[tuple[Path, int, int]] = field(default_factory=list)
    profile_path: Path | None = None

    def cleanup(self) -> None:
        for fd in self.inherited_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.inherited_fds.clear()
        for path, device, inode in reversed(self.placeholders):
            try:
                metadata = path.stat(follow_symlinks=False)
                if (
                    metadata.st_dev == device
                    and metadata.st_ino == inode
                    and stat.S_ISREG(metadata.st_mode)
                    and metadata.st_size == 0
                ):
                    path.unlink()
            except FileNotFoundError:
                pass
        self.placeholders.clear()
        if self.profile_path is not None:
            try:
                self.profile_path.unlink()
            except FileNotFoundError:
                pass
            self.profile_path = None


def _first_missing_component(path: Path) -> Path | None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            return current
    return None


def _symlink_component(path: Path) -> Path | None:
    if path == Path(path.anchor):
        return None
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return current
    return None


def _materialize_missing_target(
    path: Path,
    launch: ShellSandboxLaunch,
) -> Path | None:
    missing = _first_missing_component(path)
    if missing is None:
        return path
    try:
        fd = os.open(missing, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0)
    except FileNotFoundError:
        return None
    else:
        os.close(fd)
    metadata = missing.stat(follow_symlinks=False)
    launch.placeholders.append((missing, metadata.st_dev, metadata.st_ino))
    return missing


def _trusted_bubblewrap() -> Path:
    configured = shutil.which("bwrap")
    if configured is None:
        raise RuntimeError(
            "permissions.filesystem.deny_read requires bubblewrap on Linux"
        )
    try:
        executable = Path(configured).resolve(strict=True)
        metadata = executable.stat()
    except OSError as exc:
        raise RuntimeError("cannot validate bubblewrap executable") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_mode & 0o022
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError(
            "bubblewrap must be root-owned and not group/world writable"
        )
    return executable


def probe_linux_shell_sandbox() -> None:
    """Verify Bubblewrap from the current executable's security context."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Linux shell sandbox probing requires Linux")
    executable = _trusted_bubblewrap()
    try:
        completed = subprocess.run(
            [str(executable), "--ro-bind", "/", "/", "/bin/true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=SANDBOX_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("bubblewrap probe timed out") from exc
    if completed.returncode == 0:
        return
    error = completed.stderr[:SANDBOX_PROBE_ERROR_BYTES].decode(
        "utf-8",
        errors="replace",
    ).strip()
    raise RuntimeError(error or "bubblewrap probe failed without an error")


def _make_inheritable(fd: int) -> int:
    try:
        os.set_inheritable(fd, True)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _bubblewrap_options_fd(options: list[str]) -> int:
    with tempfile.TemporaryFile(mode="w+b") as options_file:
        for option in options:
            options_file.write(os.fsencode(option))
            options_file.write(b"\0")
        options_file.flush()
        options_file.seek(0)
        return _make_inheritable(os.dup(options_file.fileno()))


def _bubblewrap_launch(
    command: str,
    policy: FileAccessPolicy,
    denied: tuple[PreparedDeniedPath, ...],
) -> ShellSandboxLaunch:
    executable = _trusted_bubblewrap()
    launch = ShellSandboxLaunch(
        [
            str(executable),
            "--die-with-parent",
            "--bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--bind-try",
            "/dev/shm",
            "/dev/shm",
        ]
    )
    try:
        for item in denied:
            symlink = _symlink_component(item.path)
            if symlink is not None:
                raise RuntimeError(
                    "cannot enforce deny-read path through symlink: "
                    f"{symlink}"
                )
            target = item.path
            if not target.exists() and not target.is_symlink():
                if not item.materialize_if_missing:
                    continue
                materialized = _materialize_missing_target(target, launch)
                if materialized is None:
                    raise RuntimeError(
                        f"cannot materialize deny-read path: {target}"
                    )
                target = materialized
            if target.is_dir():
                launch.argv.extend(
                    ["--perms", "000", "--tmpfs", str(target)]
                )
                continue
            fd = _make_inheritable(os.open("/dev/null", os.O_RDONLY))
            launch.inherited_fds.append(fd)
            launch.argv.extend(
                ["--perms", "000", "--ro-bind-data", str(fd), str(target)]
            )
        launch.argv.extend(["--chdir", str(policy.workspace)])
        options_fd = _bubblewrap_options_fd(launch.argv[1:])
        launch.inherited_fds.append(options_fd)
        launch.argv = [
            str(executable),
            "--args",
            str(options_fd),
            "--",
            "/bin/bash",
            "-c",
            command,
        ]
        return launch
    except BaseException:
        launch.cleanup()
        raise


def _seatbelt_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _seatbelt_launch(
    command: str,
    policy: FileAccessPolicy,
    denied: tuple[PreparedDeniedPath, ...],
) -> ShellSandboxLaunch:
    executable = "/usr/bin/sandbox-exec"
    if not os.path.isfile(executable):
        raise RuntimeError(
            "permissions.filesystem.deny_read requires sandbox-exec on macOS"
        )
    rules = ["(version 1)", "(allow default)"]
    for item in denied:
        symlink = _symlink_component(item.path)
        if symlink is not None:
            raise RuntimeError(
                "cannot enforce deny-read path through symlink: "
                f"{symlink}"
            )
        kind = "subpath" if item.path.is_dir() else "literal"
        rules.append(
            "(deny file-read* file-write* "
            f'({kind} "{_seatbelt_quote(str(item.path))}"))'
        )
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="raptor-seatbelt-",
        suffix=".sb",
        delete=False,
    )
    try:
        handle.write("\n".join(rules) + "\n")
        handle.flush()
    finally:
        handle.close()
    profile_path = Path(handle.name)
    os.chmod(profile_path, 0o600)
    return ShellSandboxLaunch(
        [executable, "-f", str(profile_path), "/bin/bash", "-c", command],
        profile_path=profile_path,
    )


def build_shell_sandbox_launch(
    command: str,
    policy: FileAccessPolicy,
) -> ShellSandboxLaunch:
    if not policy.enabled:
        return ShellSandboxLaunch(["/bin/bash", "-c", command])
    denied = policy.prepare_denied_paths()
    if sys.platform.startswith("linux"):
        return _bubblewrap_launch(command, policy, denied)
    if sys.platform == "darwin":
        return _seatbelt_launch(command, policy, denied)
    raise RuntimeError(
        "permissions.filesystem.deny_read is unsupported on this platform"
    )
