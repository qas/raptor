"""Owned outbound HTTP client construction."""

from typing import Any

import httpx

from config import RAPTOR_PROXY


def outbound_http_client(**options: Any) -> httpx.AsyncClient:
    """Create a fail-closed client with no environment proxy bypasses."""
    return httpx.AsyncClient(
        proxy=RAPTOR_PROXY,
        trust_env=False,
        **options,
    )
