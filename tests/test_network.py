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
            patch.object(network.httpx, "AsyncClient") as client,
        ):
            network.outbound_http_client()
        client.assert_called_once_with(proxy=None, trust_env=False)
