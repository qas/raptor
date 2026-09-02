import copy
import io
import json
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

_HOME = Path(tempfile.mkdtemp(prefix="raptor-runtime-tests-"))
os.environ["RAPTOR_HOME"] = str(_HOME)
os.environ["AGENT_WORKDIR"] = str(_HOME)

from raptor.app import process_lock
from raptor import entrypoint
from raptor.app import runtime
from raptor.state import session
from raptor.app.application_control import ExitRequest


class RuntimeLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_state = copy.deepcopy(session.state.get("runtime"))
        process_lock.release_runtime_lock()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lock_path = Path(self.temp_dir.name) / "raptor.app.runtime.lock"
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
        self.assertFalse(
            os.get_inheritable(process_lock._runtime_lock_fd)
        )

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
            "import sys; from raptor.app import runtime; "
            "runtime.runtime_info(); "
            "assert 'config' not in sys.modules; "
            "assert 'raptor.state.session' not in sys.modules"
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

    def test_runtime_info_rejects_oversized_state(self) -> None:
        state_path = Path(self.temp_dir.name) / "state.json"
        state_path.write_bytes(b"x" * 1025)
        with (
            patch.object(runtime, "STATE_PATH", state_path),
            patch.dict(os.environ, {"MAX_STATE_LOAD_BYTES": "1024"}),
        ):
            self.assertEqual(runtime.runtime_info(), {})

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

from raptor.app.process_lock import acquire_runtime_lock, release_runtime_lock
from raptor.app.runtime import daemonize, signal_daemon_ready

def raise_exit():
    raise SystemExit(0)

assert acquire_runtime_lock()
ready_fd = daemonize()
signal.signal(signal.SIGTERM, lambda *_args: raise_exit())
Path(os.environ["TEST_DAEMON_MARKER"]).write_text(str(os.getpid()))
signal_daemon_ready(ready_fd)
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
                "from raptor.app import runtime; "
                "raise SystemExit(runtime.cli_runtime_status())"
            )
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertIn(f"pid={daemon_pid}", status.stdout)

            stopped = run_control(
                "from raptor.app import runtime; "
                "raise SystemExit(runtime.stop_daemon())"
            )
            self.assertEqual(stopped.returncode, 0, stopped.stderr)

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                probe = run_control(
                    "from raptor.app import runtime; "
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
                lock_path = home / "raptor.app.runtime.lock"
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
        runtime_module = types.ModuleType("raptor.app.runtime")
        runtime_module.parse_args = lambda: (
            order.append("parse")
            or Namespace(
                status=False,
                stop_daemon=False,
                check_proxy=False,
                check_sandbox=False,
                daemon=False,
            )
        )
        runtime_module.cli_runtime_status = lambda: 0
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: None
        runtime_module.set_runtime = lambda **_kw: order.append("publish")
        runtime_module.clear_runtime_if_ours = lambda: order.append("clear")
        application_module = types.ModuleType("raptor.app.application")

        async def application_main() -> None:
            order.append("application")

        application_module.main = application_main
        session_module = types.ModuleType("raptor.state.session")
        session_module.DAEMON_MODE = False

        def acquire() -> bool:
            order.append("lock")
            return True

        with (
            patch.object(sys, "argv", ["raptor.py"]),
            patch.object(entrypoint, "acquire_runtime_lock", side_effect=acquire),
            patch.object(entrypoint, "release_runtime_lock"),
            patch.dict(
                sys.modules,
                {
                    "raptor.app.runtime": runtime_module,
                    "raptor.app.application": application_module,
                    "raptor.state.session": session_module,
                },
            ),
        ):
            result = entrypoint.run()

        self.assertEqual(result, 0)
        self.assertEqual(
            order,
            ["parse", "lock", "publish", "application", "clear"],
        )

    def test_restart_executes_only_after_runtime_state_is_cleared(self) -> None:
        order: list[str] = []
        runtime_module = types.ModuleType("raptor.app.runtime")
        runtime_module.parse_args = lambda: Namespace(
            status=False,
            stop_daemon=False,
            check_proxy=False,
            check_sandbox=False,
            daemon=False,
        )
        runtime_module.cli_runtime_status = lambda: 0
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: None
        runtime_module.set_runtime = lambda **_kw: order.append("publish")
        runtime_module.clear_runtime_if_ours = lambda: order.append("clear")
        application_module = types.ModuleType("raptor.app.application")

        async def application_main() -> None:
            order.append("application")

        application_module.main = application_main
        session_module = types.ModuleType("raptor.state.session")
        session_module.DAEMON_MODE = False

        with (
            patch.object(sys, "argv", ["raptor.py"]),
            patch.object(entrypoint, "acquire_runtime_lock", return_value=True),
            patch.object(
                entrypoint,
                "release_runtime_lock",
                side_effect=lambda: order.append("release"),
            ),
            patch.object(
                entrypoint.os,
                "execv",
                side_effect=lambda *_args: order.append("exec"),
            ) as execv,
            patch(
                "raptor.app.application_control.take_exit_request",
                return_value=ExitRequest.RESTART,
            ),
            patch.dict(
                sys.modules,
                {
                    "raptor.app.runtime": runtime_module,
                    "raptor.app.application": application_module,
                    "raptor.state.session": session_module,
                },
            ),
        ):
            result = entrypoint.run()

        self.assertEqual(result, 0)
        self.assertEqual(
            order,
            ["publish", "application", "clear", "exec", "release"],
        )
        execv.assert_called_once_with(
            os.path.realpath(sys.executable),
            [
                os.path.realpath(sys.executable),
                str(Path(entrypoint.__file__).resolve().parent.parent / "raptor.py"),
            ],
        )

    def test_frozen_restart_reuses_original_process_arguments(self) -> None:
        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", "/opt/raptor/raptor"),
            patch.object(sys, "argv", ["raptor", "--daemon"]),
        ):
            argv = entrypoint._restart_argv()

        self.assertEqual(argv, ["/opt/raptor/raptor", "--daemon"])

    def test_daemon_entrypoint_signals_after_application_ready(self) -> None:
        order: list[str] = []
        runtime_module = types.ModuleType("raptor.app.runtime")
        runtime_module.parse_args = lambda: Namespace(
            status=False,
            stop_daemon=False,
            check_proxy=False,
            check_sandbox=False,
            daemon=True,
        )
        runtime_module.cli_runtime_status = lambda: 0
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: order.append("detach") or 99
        runtime_module.set_runtime = lambda **_kw: order.append("publish")
        runtime_module.clear_runtime_if_ours = lambda: order.append("clear")
        runtime_module.signal_daemon_ready = (
            lambda fd: order.append(f"ready:{fd}")
        )
        application_module = types.ModuleType("raptor.app.application")

        async def application_main(*, on_ready) -> None:
            order.append("initialized")
            on_ready()
            order.append("running")

        application_module.main = application_main
        session_module = types.ModuleType("raptor.state.session")
        session_module.DAEMON_MODE = False

        with (
            patch.object(sys, "argv", ["raptor.py", "--daemon"]),
            patch.object(entrypoint, "acquire_runtime_lock", return_value=True),
            patch.object(entrypoint, "release_runtime_lock"),
            patch.dict(
                sys.modules,
                {
                    "raptor.app.runtime": runtime_module,
                    "raptor.app.application": application_module,
                    "raptor.state.session": session_module,
                },
            ),
        ):
            result = entrypoint.run()

        self.assertEqual(result, 0)
        self.assertEqual(
            order,
            [
                "detach",
                "publish",
                "initialized",
                "ready:99",
                "running",
                "clear",
            ],
        )

    def test_daemon_entrypoint_transfers_readiness_fd_ownership(self) -> None:
        runtime_module = types.ModuleType("raptor.app.runtime")
        runtime_module.parse_args = lambda: Namespace(
            status=False,
            stop_daemon=False,
            check_proxy=False,
            check_sandbox=False,
            daemon=True,
        )
        runtime_module.cli_runtime_status = lambda: 0
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: 99
        runtime_module.set_runtime = lambda **_kw: None
        runtime_module.clear_runtime_if_ours = lambda: None

        def fail_ready(_fd: int) -> None:
            raise BrokenPipeError("launcher exited")

        runtime_module.signal_daemon_ready = fail_ready
        application_module = types.ModuleType("raptor.app.application")

        async def application_main(*, on_ready) -> None:
            on_ready()

        application_module.main = application_main
        session_module = types.ModuleType("raptor.state.session")
        session_module.DAEMON_MODE = False

        with (
            patch.object(sys, "argv", ["raptor.py", "--daemon"]),
            patch.object(entrypoint, "acquire_runtime_lock", return_value=True),
            patch.object(entrypoint, "release_runtime_lock"),
            patch.object(entrypoint.os, "close") as close,
            patch.dict(
                sys.modules,
                {
                    "raptor.app.runtime": runtime_module,
                    "raptor.app.application": application_module,
                    "raptor.state.session": session_module,
                },
            ),
        ):
            with self.assertRaises(BrokenPipeError):
                entrypoint.run()

        close.assert_not_called()

    def test_supervisor_mode_does_not_acquire_runtime_lock(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                [
                    "raptor",
                    "_shell-supervisor",
                    "3",
                    "4",
                    "policy",
                    "5",
                    "true",
                ],
            ),
            patch.object(entrypoint, "acquire_runtime_lock") as acquire,
            patch("raptor.shell.shell_supervisor.main", return_value=9) as supervisor,
        ):
            result = entrypoint.run()
        self.assertEqual(result, 9)
        acquire.assert_not_called()
        supervisor.assert_called_once_with(
            ["raptor", "3", "4", "policy", "5", "true"]
        )

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_entrypoint_supervisor_mode_runs_command(self) -> None:
        root = Path(__file__).resolve().parent.parent
        liveness_read, liveness_write = os.pipe()
        start_read, start_write = os.pipe()
        ready_read, ready_write = os.pipe()
        policy_file = tempfile.TemporaryFile(mode="w+b")
        policy_file.write(
            json.dumps(
                {
                    "workspace": str(root),
                    "patterns": [],
                    "glob_scan_max_depth": 32,
                }
            ).encode("utf-8")
        )
        policy_file.seek(0)
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(root / "raptor.py"),
                    "_shell-supervisor",
                    str(liveness_read),
                    str(start_read),
                    str(policy_file.fileno()),
                    str(ready_write),
                    "printf ok",
                ],
                cwd=root,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(
                    liveness_read,
                    start_read,
                    policy_file.fileno(),
                    ready_write,
                ),
            )
        except BaseException:
            os.close(liveness_read)
            os.close(liveness_write)
            os.close(start_read)
            os.close(start_write)
            os.close(ready_read)
            os.close(ready_write)
            policy_file.close()
            raise
        os.close(liveness_read)
        os.close(start_read)
        os.close(ready_write)
        try:
            os.write(start_write, b"1")
            os.close(start_write)
            start_write = -1
            self.assertEqual(os.read(ready_read, 1), b"1")
            stdout, stderr = process.communicate(timeout=5)
        finally:
            os.close(liveness_write)
            if start_write >= 0:
                os.close(start_write)
            os.close(ready_read)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            policy_file.close()
        self.assertEqual(process.returncode, 0, stderr.decode())
        self.assertEqual(stdout, b"ok")

    def test_status_does_not_acquire_application_ownership(self) -> None:
        runtime_module = types.ModuleType("raptor.app.runtime")
        runtime_module.parse_args = lambda: Namespace(
            status=True,
            stop_daemon=False,
            check_proxy=False,
            check_sandbox=False,
            daemon=False,
        )
        runtime_module.cli_runtime_status = lambda: 7
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: None
        runtime_module.set_runtime = lambda **_kw: None
        runtime_module.clear_runtime_if_ours = lambda: None
        with (
            patch.object(sys, "argv", ["raptor.py", "--status"]),
            patch.object(entrypoint, "acquire_runtime_lock") as acquire,
            patch.dict(sys.modules, {"raptor.app.runtime": runtime_module}),
        ):
            result = entrypoint.run()

        self.assertEqual(result, 7)
        acquire.assert_not_called()

    def test_sandbox_check_runs_without_runtime_ownership(self) -> None:
        runtime_module = types.ModuleType("raptor.app.runtime")
        runtime_module.parse_args = lambda: Namespace(
            status=False,
            stop_daemon=False,
            check_proxy=False,
            check_sandbox=True,
            daemon=False,
        )
        runtime_module.cli_runtime_status = lambda: 0
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: None
        runtime_module.set_runtime = lambda **_kw: None
        runtime_module.clear_runtime_if_ours = lambda: None
        sandbox_module = types.ModuleType("raptor.shell.shell_sandbox")
        checked: list[bool] = []
        sandbox_module.probe_linux_shell_sandbox = lambda: checked.append(True)
        with (
            patch.object(entrypoint, "acquire_runtime_lock") as acquire,
            patch.dict(
                sys.modules,
                {
                    "raptor.app.runtime": runtime_module,
                    "raptor.shell.shell_sandbox": sandbox_module,
                },
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            result = entrypoint.run()

        self.assertEqual(result, 0)
        self.assertEqual(checked, [True])
        self.assertEqual(output.getvalue(), "Linux shell sandbox: ready\n")
        acquire.assert_not_called()

    def test_sandbox_check_reports_failure_without_runtime_ownership(
        self,
    ) -> None:
        runtime_module = types.ModuleType("raptor.app.runtime")
        runtime_module.parse_args = lambda: Namespace(
            status=False,
            stop_daemon=False,
            check_proxy=False,
            check_sandbox=True,
            daemon=False,
        )
        runtime_module.cli_runtime_status = lambda: 0
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: None
        runtime_module.set_runtime = lambda **_kw: None
        runtime_module.clear_runtime_if_ours = lambda: None
        sandbox_module = types.ModuleType("raptor.shell.shell_sandbox")

        def fail_probe() -> None:
            raise RuntimeError("permission denied")

        sandbox_module.probe_linux_shell_sandbox = fail_probe
        with (
            patch.object(entrypoint, "acquire_runtime_lock") as acquire,
            patch.dict(
                sys.modules,
                {
                    "raptor.app.runtime": runtime_module,
                    "raptor.shell.shell_sandbox": sandbox_module,
                },
            ),
            redirect_stderr(io.StringIO()) as error,
        ):
            result = entrypoint.run()

        self.assertEqual(result, 1)
        self.assertEqual(error.getvalue(), "permission denied\n")
        acquire.assert_not_called()

    def test_proxy_check_reports_egress_without_runtime_ownership(self) -> None:
        runtime_module = types.ModuleType("raptor.app.runtime")
        runtime_module.parse_args = lambda: Namespace(
            status=False,
            stop_daemon=False,
            check_proxy=True,
            check_sandbox=False,
            daemon=False,
        )
        runtime_module.cli_runtime_status = lambda: 0
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: None
        runtime_module.set_runtime = lambda **_kw: None
        runtime_module.clear_runtime_if_ours = lambda: None
        network_module = types.ModuleType("raptor.network")
        network_module.ProxyNotConfiguredError = type(
            "ProxyNotConfiguredError",
            (RuntimeError,),
            {},
        )
        async def proxy_egress_ip() -> str:
            return "203.0.113.10"
        network_module.proxy_egress_ip = proxy_egress_ip
        with (
            patch.object(entrypoint, "acquire_runtime_lock") as acquire,
            patch.dict(
                sys.modules,
                {"raptor.app.runtime": runtime_module, "raptor.network": network_module},
            ),
            redirect_stdout(io.StringIO()) as output,
        ):
            result = entrypoint.run()
        self.assertEqual(result, 0)
        self.assertEqual(
            output.getvalue(),
            "Proxy: reachable\nEgress IP: 203.0.113.10\n",
        )
        acquire.assert_not_called()

    def test_proxy_check_redacts_connection_failures(self) -> None:
        runtime_module = types.ModuleType("raptor.app.runtime")
        runtime_module.parse_args = lambda: Namespace(
            status=False,
            stop_daemon=False,
            check_proxy=True,
            check_sandbox=False,
            daemon=False,
        )
        runtime_module.cli_runtime_status = lambda: 0
        runtime_module.stop_daemon = lambda: 0
        runtime_module.daemonize = lambda: None
        runtime_module.set_runtime = lambda **_kw: None
        runtime_module.clear_runtime_if_ours = lambda: None
        network_module = types.ModuleType("raptor.network")
        network_module.ProxyNotConfiguredError = type(
            "ProxyNotConfiguredError",
            (RuntimeError,),
            {},
        )
        async def proxy_egress_ip() -> str:
            raise RuntimeError("proxy-secret")
        network_module.proxy_egress_ip = proxy_egress_ip
        with (
            patch.object(entrypoint, "acquire_runtime_lock") as acquire,
            patch.dict(
                sys.modules,
                {"raptor.app.runtime": runtime_module, "raptor.network": network_module},
            ),
            redirect_stderr(io.StringIO()) as error,
        ):
            result = entrypoint.run()
        self.assertEqual(result, 1)
        self.assertEqual(error.getvalue(), "Proxy: unreachable\n")
        self.assertNotIn("proxy-secret", error.getvalue())
        acquire.assert_not_called()

    def test_daemon_parent_detaches_its_lock_copy(self) -> None:
        with (
            patch.object(runtime.os, "pipe", return_value=(10, 11)),
            patch.object(runtime.os, "fork", return_value=123),
            patch.object(
                runtime,
                "_await_daemon_ready",
                return_value=(True, 456),
            ) as ready,
            patch.object(runtime.os, "close"),
            patch.object(runtime, "detach_runtime_lock") as detach,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as stopped:
                runtime.daemonize()

        self.assertEqual(stopped.exception.code, 0)
        ready.assert_called_once_with(10)
        detach.assert_called_once_with()

    def test_daemon_parent_rejects_failed_child_startup(self) -> None:
        with (
            patch.object(runtime.os, "pipe", return_value=(10, 11)),
            patch.object(runtime.os, "fork", return_value=123),
            patch.object(
                runtime,
                "_await_daemon_ready",
                return_value=(False, 456),
            ),
            patch.object(runtime.os, "close"),
            patch.object(runtime.os, "kill") as kill,
            patch.object(runtime, "release_runtime_lock") as release,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as stopped:
                runtime.daemonize()

        self.assertEqual(stopped.exception.code, 1)
        kill.assert_called_once_with(456, signal.SIGTERM)
        release.assert_called_once_with()

    def test_daemon_readiness_deadline_covers_silent_pid_handshake(self) -> None:
        read_fd, write_fd = os.pipe()
        try:
            started = time.monotonic()
            with patch.object(runtime, "DAEMON_START_TIMEOUT_SECONDS", 0.01):
                ready, pid = runtime._await_daemon_ready(read_fd)
            elapsed = time.monotonic() - started
        finally:
            os.close(read_fd)
            os.close(write_fd)

        self.assertFalse(ready)
        self.assertIsNone(pid)
        self.assertLess(elapsed, 0.2)

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
