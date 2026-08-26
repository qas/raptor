import unittest
from unittest.mock import AsyncMock, Mock, call, patch

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

    def test_configured_bypasses_create_explicit_routing_transport(self) -> None:
        direct = Mock()
        proxied = Mock()
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
            patch.object(
                network.httpx,
                "AsyncHTTPTransport",
                side_effect=(direct, proxied),
            ) as transport_factory,
            patch.object(network.httpx, "AsyncClient") as client,
        ):
            network.outbound_http_client(timeout=None)
        transport_factory.assert_has_calls([
            call(trust_env=False),
            call(
                proxy="socks5h://proxy.example:1080",
                trust_env=False,
            ),
        ])
        routing_transport = client.call_args.kwargs["transport"]
        client.assert_called_once_with(
            transport=routing_transport,
            trust_env=False,
            timeout=None,
        )


class ProxyBypassRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_and_wildcard_hosts_route_directly(self) -> None:
        direct = Mock(
            handle_async_request=AsyncMock(return_value=Mock()),
            aclose=AsyncMock(),
        )
        proxied = Mock(
            handle_async_request=AsyncMock(return_value=Mock()),
            aclose=AsyncMock(),
        )
        with patch.object(
            network.httpx,
            "AsyncHTTPTransport",
            side_effect=(direct, proxied),
        ):
            transport = network._ProxyBypassTransport(
                "socks5h://127.0.0.1:1080",
                ("models.example", "*.google.com", "::1"),
            )
        try:
            for url in (
                "https://models.example/v1/models",
                "https://api.google.com",
                "http://[::1]:8000/v1/models",
            ):
                with self.subTest(url=url):
                    await transport.handle_async_request(
                        network.httpx.Request("GET", url)
                    )
            self.assertEqual(direct.handle_async_request.await_count, 3)
            self.assertEqual(proxied.handle_async_request.await_count, 0)
            for url in (
                "https://sub.models.example",
                "https://google.com",
                "https://other.example",
            ):
                with self.subTest(url=url):
                    await transport.handle_async_request(
                        network.httpx.Request("GET", url)
                    )
            self.assertEqual(direct.handle_async_request.await_count, 3)
            self.assertEqual(proxied.handle_async_request.await_count, 3)
        finally:
            await transport.aclose()
        direct.aclose.assert_awaited_once()
        proxied.aclose.assert_awaited_once()


class ProxyCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_check_forces_proxy_and_returns_valid_ip(self) -> None:
        response = Mock(text="203.0.113.10\n")
        client = Mock(
            get=AsyncMock(return_value=response),
            __aenter__=AsyncMock(),
            __aexit__=AsyncMock(return_value=None),
        )
        client.__aenter__.return_value = client
        with (
            patch.object(
                network,
                "RAPTOR_PROXY",
                "socks5h://proxy.example:1080",
            ),
            patch.object(network.httpx, "AsyncClient", return_value=client) as factory,
        ):
            address = await network.proxy_egress_ip()
        self.assertEqual(address, "203.0.113.10")
        factory.assert_called_once_with(
            proxy="socks5h://proxy.example:1080",
            trust_env=False,
            timeout=network.httpx.Timeout(
                network.PROXY_CHECK_TIMEOUT_SECONDS
            ),
        )
        client.get.assert_awaited_once_with(network._PROXY_CHECK_URL)
        response.raise_for_status.assert_called_once_with()

    async def test_proxy_check_requires_configured_proxy(self) -> None:
        with (
            patch.object(network, "RAPTOR_PROXY", None),
            patch.object(network.httpx, "AsyncClient") as factory,
        ):
            with self.assertRaises(network.ProxyNotConfiguredError):
                await network.proxy_egress_ip()
        factory.assert_not_called()

    async def test_proxy_check_rejects_invalid_ip_response(self) -> None:
        response = Mock(text="not-an-ip")
        client = Mock(
            get=AsyncMock(return_value=response),
            __aenter__=AsyncMock(),
            __aexit__=AsyncMock(return_value=None),
        )
        client.__aenter__.return_value = client
        with (
            patch.object(
                network,
                "RAPTOR_PROXY",
                "socks5h://proxy.example:1080",
            ),
            patch.object(network.httpx, "AsyncClient", return_value=client),
        ):
            with self.assertRaises(ValueError):
                await network.proxy_egress_ip()
