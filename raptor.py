# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.28,<1",
# ]
# ///
"""Raptor process entry point."""

import asyncio
import sys

from process_lock import acquire_runtime_lock, release_runtime_lock


_READ_ONLY_CLI_FLAGS = {"--help", "--status", "--stop-daemon", "-h"}


def run() -> int:
    """Parse the CLI and start Raptor with ownership established first."""
    owns_runtime = not any(
        argument in _READ_ONLY_CLI_FLAGS for argument in sys.argv[1:]
    )
    if owns_runtime and not acquire_runtime_lock():
        print("Raptor is already running", file=sys.stderr)
        return 1

    try:
        from runtime import (
            clear_stale_runtime,
            cli_runtime_status,
            daemonize,
            parse_args,
            stop_daemon_from_state,
        )

        args = parse_args()
        clear_stale_runtime()
        if args.status:
            return cli_runtime_status()
        if args.stop_daemon:
            return stop_daemon_from_state()

        import application
        import session

        if args.daemon:
            session.DAEMON_MODE = True
            daemonize()
        asyncio.run(application.main())
        return 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 0
    finally:
        if owns_runtime:
            release_runtime_lock()


if __name__ == "__main__":
    raise SystemExit(run())
