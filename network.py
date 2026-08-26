"""Owned outbound HTTP client construction."""

from typing import Any

import httpx

from config import RAPTOR_NO_PROXY, RAPTOR_PROXY


def _proxy_bypass_mount(host: str) -> str:
    if ":" in host:
        return f"all://[{host}]"
    return f"all://{host}"


def outbound_http_client(**options: Any) -> httpx.AsyncClient:
    """Create a fail-closed client with owned destination bypasses."""
    if RAPTOR_NO_PROXY:
        mounts = dict(options.pop("mounts", None) or {})
        mounts.update(
            {
                _proxy_bypass_mount(host): None
                for host in RAPTOR_NO_PROXY
            }
        )
        options["mounts"] = mounts
    return httpx.AsyncClient(
        proxy=RAPTOR_PROXY,
        trust_env=False,
        **options,
    )
