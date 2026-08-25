"""Raptor process entry point."""

import asyncio
import os
import sys

from process_lock import acquire_runtime_lock, release_runtime_lock


def run() -> int:
    """Parse the CLI and establish ownership before loading the application."""
    from runtime import (
        clear_runtime_if_ours,
        cli_runtime_status,
        daemonize,
        parse_args,
        set_runtime,
        stop_daemon,
    )

    args = parse_args()
    owns_runtime = not (args.status or args.stop_daemon)
    if owns_runtime and not acquire_runtime_lock():
        print("Raptor is already running", file=sys.stderr)
        return 1
    try:
        if args.status:
            return cli_runtime_status()
        if args.stop_daemon:
            return stop_daemon()

        ready_fd: int | None = None
        if args.daemon:
            ready_fd = daemonize()
        set_runtime(daemon=args.daemon)
        try:
            import application
            import session

            session.DAEMON_MODE = args.daemon
            if ready_fd is None:
                asyncio.run(application.main())
            else:
                from runtime import signal_daemon_ready

                def on_ready() -> None:
                    nonlocal ready_fd
                    assert ready_fd is not None
                    owned_fd = ready_fd
                    ready_fd = None
                    signal_daemon_ready(owned_fd)

                asyncio.run(application.main(on_ready=on_ready))
        finally:
            if ready_fd is not None:
                os.close(ready_fd)
            clear_runtime_if_ours()
        return 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 0
    finally:
        if owns_runtime:
            release_runtime_lock()


if __name__ == "__main__":
    raise SystemExit(run())
