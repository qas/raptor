import os
import signal
import time
import unittest
from unittest.mock import patch

import shell_supervisor


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
