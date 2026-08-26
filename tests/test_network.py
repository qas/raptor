import unittest
from unittest.mock import patch

import network


class OutboundHttpClientTests(unittest.TestCase):
    def test_proxy_is_explicit_and_environment_bypasses_are_disabled(
        self,
    ) -> None:
        with (
            patch.object(
                network,
                "RAPTOR_PROXY",
                "socks5h://proxy.example:1080",
            ),
            patch.object(network, "RAPTOR_NO_PROXY", ()),
            patch.object(network.httpx, "AsyncClient") as client,
        ):
            network.outbound_http_client(timeout=None)
        client.assert_called_once_with(
            proxy="socks5h://proxy.example:1080",
            trust_env=False,
            timeout=None,
        )

    def test_direct_mode_still_ignores_ambient_proxy_variables(self) -> None:
        with (
            patch.object(network, "RAPTOR_PROXY", None),
            patch.object(network, "RAPTOR_NO_PROXY", ()),
            patch.object(network.httpx, "AsyncClient") as client,
        ):
            network.outbound_http_client()
        client.assert_called_once_with(proxy=None, trust_env=False)

    def test_configured_bypasses_become_direct_mounts(self) -> None:
        with (
            patch.object(
                network,
                "RAPTOR_PROXY",
                "socks5h://proxy.example:1080",
            ),
            patch.object(
                network,
                "RAPTOR_NO_PROXY",
                ("models.example", "*.google.com", "::1"),
            ),
            patch.object(network.httpx, "AsyncClient") as client,
        ):
            network.outbound_http_client(
                mounts={"all://existing.example": "transport"},
            )
        client.assert_called_once_with(
            proxy="socks5h://proxy.example:1080",
            trust_env=False,
            mounts={
                "all://existing.example": "transport",
                "all://models.example": None,
                "all://*.google.com": None,
                "all://[::1]": None,
            },
        )


class ProxyBypassRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_and_wildcard_hosts_route_directly(self) -> None:
        with (
            patch.object(
                network,
                "RAPTOR_PROXY",
                "socks5h://127.0.0.1:1080",
            ),
            patch.object(
                network,
                "RAPTOR_NO_PROXY",
                ("models.example", "*.google.com", "::1"),
            ),
        ):
            client = network.outbound_http_client()
        try:
            direct = client._transport
            self.assertIs(
                client._transport_for_url(
                    network.httpx.URL("https://models.example/v1/models")
                ),
                direct,
            )
            self.assertIs(
                client._transport_for_url(
                    network.httpx.URL("https://api.google.com")
                ),
                direct,
            )
            self.assertIs(
                client._transport_for_url(
                    network.httpx.URL("http://[::1]:8000/v1/models")
                ),
                direct,
            )
            self.assertIsNot(
                client._transport_for_url(
                    network.httpx.URL("https://sub.models.example")
                ),
                direct,
            )
            self.assertIsNot(
                client._transport_for_url(
                    network.httpx.URL("https://google.com")
                ),
                direct,
            )
            self.assertIsNot(
                client._transport_for_url(
                    network.httpx.URL("https://other.example")
                ),
                direct,
            )
        finally:
            await client.aclose()
