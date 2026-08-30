"""Raptor process entry point."""

import asyncio
import os
import sys

from process_lock import acquire_runtime_lock, release_runtime_lock
from shell_supervisor import SUPERVISOR_MODE


def run() -> int:
    """Parse the CLI and establish ownership before loading the application."""
    if len(sys.argv) > 1 and sys.argv[1] == SUPERVISOR_MODE:
        from shell_supervisor import main
        return main([sys.argv[0], *sys.argv[2:]])
    from runtime import (
        clear_runtime_if_ours,
        cli_runtime_status,
        daemonize,
        parse_args,
        set_runtime,
        stop_daemon,
    )

    args = parse_args()
    if getattr(args, "version", False):
        from version import display_version

        print(f"raptor {display_version()}")
        return 0
    owns_runtime = not (args.status or args.stop_daemon or args.check_proxy)
    if owns_runtime and not acquire_runtime_lock():
        print("Raptor is already running", file=sys.stderr)
        return 1
    try:
        if args.check_proxy:
            from network import ProxyNotConfiguredError, proxy_egress_ip
            try:
                address = asyncio.run(proxy_egress_ip())
            except ProxyNotConfiguredError:
                print("Proxy: disabled", file=sys.stderr)
                return 1
            except Exception:
                print("Proxy: unreachable", file=sys.stderr)
                return 1
            print("Proxy: reachable")
            print(f"Egress IP: {address}")
            return 0
        if args.status:
            return cli_runtime_status()
        if args.stop_daemon:
            return stop_daemon()

        from workspace_identity import initialize_workspace_identity

        initialize_workspace_identity()
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
