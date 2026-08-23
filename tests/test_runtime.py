import copy
import io
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import process_lock
import raptor
import runtime
import session


class RuntimeLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_state = copy.deepcopy(session.state.get("runtime"))
        process_lock.release_runtime_lock()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lock_path = Path(self.temp_dir.name) / "runtime.lock"
        self.path_patch = patch.object(
            process_lock,
            "RUNTIME_LOCK_PATH",
            self.lock_path,
        )
        self.path_patch.start()

    def tearDown(self) -> None:
        if self.runtime_state is None:
            session.state.pop("runtime", None)
        else:
            session.state["runtime"] = self.runtime_state
        process_lock.release_runtime_lock()
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_process_lock_has_atomic_lifetime(self) -> None:
        self.assertTrue(process_lock.acquire_runtime_lock())
        self.assertTrue(process_lock.runtime_lock_held())
        self.assertEqual(self.lock_path.read_text().strip(), str(os.getpid()))

        process_lock.release_runtime_lock()

        self.assertFalse(process_lock.runtime_lock_held())

    def test_owner_publication_failure_releases_lock(self) -> None:
        with patch.object(process_lock.os, "pwrite", return_value=0):
            with self.assertRaises(OSError):
                process_lock.acquire_runtime_lock()

        self.assertFalse(process_lock.runtime_lock_held())

    def test_process_control_is_independent_of_application_config(self) -> None:
        environment = os.environ.copy()
        environment["RESPONSES_SERVER_PORT"] = "invalid"
        environment["RAPTOR_HOME"] = str(self.lock_path.parent / "home")
        command = (
            "import sys; import runtime; "
            "runtime.runtime_info(); "
            "assert 'config' not in sys.modules; "
            "assert 'session' not in sys.modules"
        )

        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_daemon_owner_survives_launcher_exit(self) -> None:
        home = Path(self.temp_dir.name) / "daemon-home"
        marker = Path(self.temp_dir.name) / "daemon.pid"
        environment = os.environ.copy()
        environment.update(
            {
                "AGENT_WORKDIR": str(Path(__file__).resolve().parent.parent),
                "RAPTOR_HOME": str(home),
                "RAPTOR_LOG": str(home / "raptor.log"),
                "TEST_DAEMON_MARKER": str(marker),
            }
        )
        daemon_command = """
import os
import signal
from pathlib import Path

from process_lock import acquire_runtime_lock, release_runtime_lock
from runtime import daemonize

def raise_exit():
    raise SystemExit(0)

assert acquire_runtime_lock()
daemonize()
signal.signal(signal.SIGTERM, lambda *_args: raise_exit())
Path(os.environ["TEST_DAEMON_MARKER"]).write_text(str(os.getpid()))
try:
    signal.pause()
finally:
    release_runtime_lock()
"""

        launched = subprocess.run(
            [sys.executable, "-c", daemon_command],
            cwd=Path(__file__).resolve().parent.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        self.assertEqual(launched.returncode, 0, launched.stderr)

        daemon_pid: int | None = None
        daemon_stopped = False

        def run_control(source: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, "-c", source],
                cwd=Path(__file__).resolve().parent.parent,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

        try:
            deadline = time.monotonic() + 2
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.exists(), "daemon did not publish its PID")
            daemon_pid = int(marker.read_text())

            status = run_control(
                "import runtime; "
                "raise SystemExit(runtime.cli_runtime_status())"
            )
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertIn(f"pid={daemon_pid}", status.stdout)

            stopped = run_control(
                "import runtime; raise SystemExit(runtime.stop_daemon())"
            )
            self.assertEqual(stopped.returncode, 0, stopped.stderr)

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                probe = run_control(
                    "import runtime; "
                    "raise SystemExit(runtime.cli_runtime_status())"
                )
                if probe.returncode == 1:
                    daemon_stopped = True
                    break
                time.sleep(0.01)
            else:
                self.fail("daemon ownership was not released")
        finally:
            if daemon_pid is None:
                lock_path = home / "runtime.lock"
                if lock_path.exists():
                    raw_pid = lock_path.read_text().strip()
                    daemon_pid = int(raw_pid) if raw_pid else None
            if daemon_pid is not None and not daemon_stopped:
                try:
                    os.kill(daemon_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_fork_parent_detaches_without_unlocking_child(self) -> None:
        self.assertTrue(process_lock.acquire_runtime_lock())
        read_fd, write_fd = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(write_fd)
            try:
                os.read(read_fd, 1)
            finally:
                os.close(read_fd)
            os._exit(0)

        os.close(read_fd)
        try:
            process_lock.detach_runtime_lock()
            self.assertTrue(process_lock.runtime_lock_status().held)
        finally:
            os.write(write_fd, b"x")
            os.close(write_fd)
            os.waitpid(child_pid, 0)

        self.assertFalse(process_lock.runtime_lock_status().held)

    def test_runtime_metadata_refreshes_lock_owner_after_daemon_fork(self) -> None:
        self.assertTrue(process_lock.acquire_runtime_lock())
        with (
            patch.object(process_lock.os, "getpid", return_value=4321),
            patch.object(session, "save_state"),
        ):
            runtime.set_runtime(daemon=True)

        self.assertEqual(self.lock_path.read_text().strip(), "4321")
        self.assertEqual(session.state["runtime"]["pid"], 4321)

    def test_entrypoint_acquires_ownership_before_application_import(self) -> None:
        order: list[str] = []
        runtime_module = types.ModuleType("runtime")
        runtime_module.parse_args = lambda: (
            order.append("parse")
            or Namespace(status=False, stop_daemon=False, daemon=False)
        )
        runtime_module.cli_runtime_status = lambda: 0
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: None
        runtime_module.set_runtime = lambda **_kw: order.append("publish")
        runtime_module.clear_runtime_if_ours = lambda: order.append("clear")
        application_module = types.ModuleType("application")

        async def application_main() -> None:
            order.append("application")

        application_module.main = application_main
        session_module = types.ModuleType("session")
        session_module.DAEMON_MODE = False

        def acquire() -> bool:
            order.append("lock")
            return True

        with (
            patch.object(sys, "argv", ["raptor.py"]),
            patch.object(raptor, "acquire_runtime_lock", side_effect=acquire),
            patch.object(raptor, "release_runtime_lock"),
            patch.dict(
                sys.modules,
                {
                    "runtime": runtime_module,
                    "application": application_module,
                    "session": session_module,
                },
            ),
        ):
            result = raptor.run()

        self.assertEqual(result, 0)
        self.assertEqual(
            order,
            ["parse", "lock", "publish", "application", "clear"],
        )

    def test_status_does_not_acquire_application_ownership(self) -> None:
        runtime_module = types.ModuleType("runtime")
        runtime_module.parse_args = lambda: Namespace(
            status=True,
            stop_daemon=False,
            daemon=False,
        )
        runtime_module.cli_runtime_status = lambda: 7
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: None
        runtime_module.set_runtime = lambda **_kw: None
        runtime_module.clear_runtime_if_ours = lambda: None
        with (
            patch.object(sys, "argv", ["raptor.py", "--status"]),
            patch.object(raptor, "acquire_runtime_lock") as acquire,
            patch.dict(sys.modules, {"runtime": runtime_module}),
        ):
            result = raptor.run()

        self.assertEqual(result, 7)
        acquire.assert_not_called()

    def test_daemon_parent_detaches_its_lock_copy(self) -> None:
        with (
            patch.object(runtime.os, "pipe", return_value=(10, 11)),
            patch.object(runtime.os, "fork", return_value=123),
            patch.object(runtime.os, "read", return_value=b"1") as read,
            patch.object(runtime.os, "close"),
            patch.object(runtime, "detach_runtime_lock") as detach,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as stopped:
                runtime.daemonize()

        self.assertEqual(stopped.exception.code, 0)
        read.assert_called_once_with(10, 1)
        detach.assert_called_once_with()

    def test_daemon_parent_rejects_failed_child_startup(self) -> None:
        with (
            patch.object(runtime.os, "pipe", return_value=(10, 11)),
            patch.object(runtime.os, "fork", return_value=123),
            patch.object(runtime.os, "read", return_value=b""),
            patch.object(runtime.os, "close"),
            patch.object(runtime, "release_runtime_lock") as release,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as stopped:
                runtime.daemonize()

        self.assertEqual(stopped.exception.code, 1)
        release.assert_called_once_with()

    def test_status_uses_lock_owner_without_rewriting_live_state(self) -> None:
        session.state["runtime"] = {"pid": 9999, "daemon": True}
        with (
            patch.object(
                runtime,
                "runtime_lock_status",
                return_value=process_lock.RuntimeLockStatus(True, 4321),
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            result = runtime.cli_runtime_status()

        self.assertEqual(result, 0)
        self.assertIn("pid=4321", output.getvalue())
        self.assertIn("daemon=unknown", output.getvalue())

    def test_stop_signals_authoritative_lock_owner(self) -> None:
        session.state["runtime"] = {"pid": 9999}
        with (
            patch.object(
                runtime,
                "runtime_lock_status",
                return_value=process_lock.RuntimeLockStatus(True, 4321),
            ),
            patch.object(runtime.os, "kill") as kill,
            redirect_stdout(io.StringIO()),
        ):
            result = runtime.stop_daemon()

        self.assertEqual(result, 0)
        kill.assert_called_once_with(4321, runtime.signal.SIGTERM)

    def test_status_survives_unavailable_session_metadata(self) -> None:
        with (
            patch.object(
                runtime,
                "runtime_lock_status",
                return_value=process_lock.RuntimeLockStatus(True, 4321),
            ),
            patch.object(runtime, "runtime_info", return_value={}),
            redirect_stdout(io.StringIO()) as output,
        ):
            result = runtime.cli_runtime_status()

        self.assertEqual(result, 0)
        self.assertIn("pid=4321", output.getvalue())
        self.assertIn("daemon=unknown", output.getvalue())


if __name__ == "__main__":
    unittest.main()
