import os
import time
import unittest

import shell_supervisor


class SupervisorWaitTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_reap_if_exited_returns_cached_status(self) -> None:
        pid = os.fork()
        if pid == 0:
            os._exit(3)
        deadline = time.monotonic() + 2
        status = None
        while time.monotonic() < deadline:
            status = shell_supervisor._reap_if_exited(pid)
            if status is not None:
                break
            time.sleep(0.01)
        self.assertIsNotNone(status)
        self.assertEqual(os.waitstatus_to_exitcode(status), 3)
        self.assertEqual(
            shell_supervisor._terminate_group(pid, status),
            status,
        )

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork")
    def test_terminate_group_reaps_live_leader(self) -> None:
        pid = os.fork()
        if pid == 0:
            os.setpgid(0, 0)
            time.sleep(30)
            os._exit(0)
        os.setpgid(pid, pid)
        status = shell_supervisor._terminate_group(pid, None)
        self.assertNotEqual(os.waitstatus_to_exitcode(status), 0)
