import os
import copy
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import process_lock
import raptor
import runtime


class RuntimeLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime_state = copy.deepcopy(runtime.state.get("runtime"))
        process_lock.release_runtime_lock()
        self.lock_path = Path(tempfile.mkdtemp()) / "runtime.lock"
        self.path_patch = patch.object(
            process_lock,
            "RUNTIME_LOCK_PATH",
            self.lock_path,
        )
        self.path_patch.start()

    def tearDown(self) -> None:
        if self.runtime_state is None:
            runtime.state.pop("runtime", None)
        else:
            runtime.state["runtime"] = self.runtime_state
        process_lock.release_runtime_lock()
        self.path_patch.stop()

    def test_process_lock_has_atomic_lifetime(self) -> None:
        self.assertTrue(process_lock.acquire_runtime_lock())
        self.assertTrue(process_lock.runtime_lock_held())
        self.assertEqual(self.lock_path.read_text(), str(os.getpid()))

        process_lock.release_runtime_lock()

        self.assertFalse(process_lock.runtime_lock_held())

    def test_runtime_metadata_refreshes_lock_owner_after_daemon_fork(self) -> None:
        self.assertTrue(process_lock.acquire_runtime_lock())
        with (
            patch.object(process_lock.os, "getpid", return_value=4321),
            patch.object(runtime, "save_state"),
        ):
            runtime.set_runtime(daemon=True)

        self.assertEqual(self.lock_path.read_text(), "4321")
        self.assertEqual(runtime.state["runtime"]["pid"], 4321)

    def test_entrypoint_acquires_ownership_before_application_import(self) -> None:
        order: list[str] = []
        runtime_module = types.ModuleType("runtime")
        runtime_module.parse_args = lambda: (
            order.append("parse")
            or Namespace(status=False, stop_daemon=False, daemon=False)
        )
        runtime_module.clear_stale_runtime = lambda: None
        runtime_module.cli_runtime_status = lambda: 0
        runtime_module.stop_daemon_from_state = lambda: 0
        runtime_module.daemonize = lambda: None
        application_module = types.ModuleType("application")
        application_module.main = AsyncMock(return_value=None)
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
        self.assertEqual(order, ["lock", "parse"])
        application_module.main.assert_awaited_once()

    def test_status_does_not_acquire_application_ownership(self) -> None:
        runtime_module = types.ModuleType("runtime")
        runtime_module.parse_args = lambda: Namespace(
            status=True,
            stop_daemon=False,
            daemon=False,
        )
        runtime_module.clear_stale_runtime = lambda: None
        runtime_module.cli_runtime_status = lambda: 7
        runtime_module.stop_daemon_from_state = lambda: 0
        runtime_module.daemonize = lambda: None
        with (
            patch.object(sys, "argv", ["raptor.py", "--status"]),
            patch.object(raptor, "acquire_runtime_lock") as acquire,
            patch.dict(sys.modules, {"runtime": runtime_module}),
        ):
            result = raptor.run()

        self.assertEqual(result, 7)
        acquire.assert_not_called()


if __name__ == "__main__":
    unittest.main()
