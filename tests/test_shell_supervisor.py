import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from raptor.shell import shell_supervisor
from raptor.shell.filesystem_permissions import FileAccessPolicy


class SupervisorWaitTests(unittest.TestCase):
    def test_signal_group_ignores_unsignalable_zombie_group(self) -> None:
        with patch.object(os, "killpg", side_effect=PermissionError):
            shell_supervisor._signal_group(123, signal.SIGTERM)

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_terminate_group_preserves_exited_leader_status(self) -> None:
        pid = os.fork()
        if pid == 0:
            os.setpgid(0, 0)
            os._exit(3)
        time.sleep(0.05)
        status = shell_supervisor._terminate_group(
            pid,
            lambda: True,
        )
        self.assertEqual(os.waitstatus_to_exitcode(status), 3)

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_terminate_group_reaps_live_leader(self) -> None:
        pid = os.fork()
        if pid == 0:
            os.setpgid(0, 0)
            time.sleep(30)
            os._exit(0)
        os.setpgid(pid, pid)
        status = shell_supervisor._terminate_group(pid, lambda: False)
        self.assertNotEqual(os.waitstatus_to_exitcode(status), 0)

    def test_command_comes_from_inherited_fd(self) -> None:
        command_read, command_write = os.pipe()
        os.write(command_write, b"printf secret-command")
        os.close(command_write)
        liveness_read, liveness_write = os.pipe()
        start_read, start_write = os.pipe()
        policy_read, policy_write = os.pipe()
        ready_read, ready_write = os.pipe()
        try:
            with patch.object(
                shell_supervisor, "_run", return_value=7
            ) as run:
                result = shell_supervisor.main([
                    "supervisor",
                    str(liveness_read),
                    str(start_read),
                    str(policy_read),
                    str(ready_write),
                    str(command_read),
                ])
            self.assertEqual(result, 7)
            self.assertEqual(
                run.call_args.args[0],
                "printf secret-command",
            )
        finally:
            for fd in (
                liveness_write,
                start_read,
                start_write,
                policy_read,
                policy_write,
                ready_read,
            ):
                os.close(fd)

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires linux")
    @unittest.skipUnless(shutil.which("bwrap"), "requires bwrap")
    def test_sigterm_removes_missing_deny_read_placeholder(self) -> None:
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve()
            placeholder = workspace / ".env"
            command = "sleep 30"
            liveness_read, liveness_write = os.pipe()
            start_read, start_write = os.pipe()
            ready_read, ready_write = os.pipe()
            policy_file = tempfile.TemporaryFile(mode="w+b")
            policy_file.write(
                FileAccessPolicy.create(
                    workspace, [".env"]
                ).supervisor_payload().encode("utf-8")
            )
            policy_file.seek(0)
            command_file = tempfile.TemporaryFile(mode="w+b")
            command_file.write(command.encode("utf-8"))
            command_file.seek(0)
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "raptor.py"),
                    shell_supervisor.SUPERVISOR_MODE,
                    str(liveness_read),
                    str(start_read),
                    str(policy_file.fileno()),
                    str(ready_write),
                    str(command_file.fileno()),
                ],
                cwd=workspace,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(
                    liveness_read,
                    start_read,
                    policy_file.fileno(),
                    ready_write,
                    command_file.fileno(),
                ),
            )
            os.close(liveness_read)
            os.close(start_read)
            os.close(ready_write)
            try:
                os.write(start_write, b"1")
                os.close(start_write)
                start_write = -1
                readable, _, _ = select.select([ready_read], [], [], 5)
                if not readable:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=2)
                        self.fail("supervisor did not become ready")
                    stdout, stderr = process.communicate(timeout=2)
                    self.fail(stderr.decode() or stdout.decode())
                self.assertEqual(os.read(ready_read, 1), b"1")
                self.assertTrue(placeholder.is_file())
                metadata = placeholder.stat()
                self.assertEqual(metadata.st_size, 0)
                self.assertEqual(metadata.st_mode & 0o777, 0)
                cmdline = Path(
                    f"/proc/{process.pid}/cmdline"
                ).read_bytes().replace(b"\0", b" ")
                self.assertNotIn(command.encode(), cmdline)
                os.kill(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=2)
                    self.fail("supervisor ignored SIGTERM")
            finally:
                os.close(liveness_write)
                if start_write >= 0:
                    os.close(start_write)
                os.close(ready_read)
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=2)
                policy_file.close()
                command_file.close()
            self.assertEqual(
                process.returncode,
                128 + signal.SIGTERM,
            )
            self.assertFalse(placeholder.exists())
