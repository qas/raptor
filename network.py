"""Owned outbound HTTP client construction."""

import asyncio
import ipaddress
from typing import Any

import httpx

from config import RAPTOR_NO_PROXY, RAPTOR_PROXY


PROXY_CHECK_TIMEOUT_SECONDS = 10.0
_PROXY_CHECK_URL = "https://api.ipify.org"


class ProxyNotConfiguredError(RuntimeError):
    pass


def _normalized_host(host: str) -> str:
    return host.encode("idna").decode("ascii").lower()


class _ProxyBypassTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        proxy: str,
        bypasses: tuple[str, ...],
        **options: Any,
    ) -> None:
        self._direct = httpx.AsyncHTTPTransport(**options)
        self._proxied = httpx.AsyncHTTPTransport(proxy=proxy, **options)
        self._exact = frozenset(
            host for host in bypasses if not host.startswith("*.")
        )
        self._suffixes = tuple(
            host[1:] for host in bypasses if host.startswith("*.")
        )

    def _bypasses(self, host: str) -> bool:
        normalized = _normalized_host(host)
        return normalized in self._exact or any(
            normalized.endswith(suffix) for suffix in self._suffixes
        )

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        transport = (
            self._direct
            if self._bypasses(request.url.host)
            else self._proxied
        )
        return await transport.handle_async_request(request)

    async def aclose(self) -> None:
        try:
            await self._direct.aclose()
        finally:
            await self._proxied.aclose()


def outbound_http_client(**options: Any) -> httpx.AsyncClient:
    """Create a fail-closed client with owned destination bypasses."""
    if RAPTOR_NO_PROXY:
        if "transport" in options or "mounts" in options:
            raise ValueError(
                "Proxy bypasses cannot be combined with custom transports"
            )
        assert RAPTOR_PROXY is not None
        transport_options = {
            name: options[name]
            for name in ("verify", "cert", "http1", "http2", "limits")
            if name in options
        }
        transport_options["trust_env"] = False
        transport = _ProxyBypassTransport(
            RAPTOR_PROXY,
            RAPTOR_NO_PROXY,
            **transport_options,
        )
        return httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            **options,
        )
    return httpx.AsyncClient(
        proxy=RAPTOR_PROXY,
        trust_env=False,
        **options,
    )


async def proxy_egress_ip() -> str:
    """Return the public IP observed through the configured proxy."""
    if RAPTOR_PROXY is None:
        raise ProxyNotConfiguredError("RAPTOR_PROXY is not configured")
    async def request_ip() -> str:
        async with httpx.AsyncClient(
            proxy=RAPTOR_PROXY,
            trust_env=False,
            timeout=httpx.Timeout(PROXY_CHECK_TIMEOUT_SECONDS),
        ) as client:
            response = await client.get(_PROXY_CHECK_URL)
            response.raise_for_status()
            return str(ipaddress.ip_address(response.text.strip()))

    return await asyncio.wait_for(
        request_ip(),
        timeout=PROXY_CHECK_TIMEOUT_SECONDS,
    )
