"""Bounded HTTP adapter for the shared portable OpenAI-compatible wire protocol."""

from __future__ import annotations

from typing import Any

import httpx

from .inference_client import decode_response, infer, validate_config


class OpenAICompletion:
    """Explicit owner-authorized route. No proxies, redirects, model guessing or logs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        connection_authorized: bool = False,
        auth_env: str | None = None,
        timeout: float = 60.0,
        max_response_bytes: int = 2_000_000,
        max_tokens: int = 1024,
        max_prompt_bytes: int = 120_000,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.auth_env = auth_env
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.max_tokens = max_tokens
        self.max_prompt_bytes = max_prompt_bytes
        self.transport = transport
        validate_config(self.metadata())
        if not connection_authorized:
            raise ValueError("connection_not_authorized")

    def metadata(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "auth_env": self.auth_env,
            "timeout": self.timeout,
            "max_response_bytes": self.max_response_bytes,
            "max_tokens": self.max_tokens,
            "max_prompt_bytes": self.max_prompt_bytes,
        }

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        try:
            with httpx.Client(
                timeout=self.timeout,
                trust_env=False,
                follow_redirects=False,
                transport=self.transport,
            ) as client:

                def send(
                    method: str, path: str, headers: dict[str, str], body: bytes | None
                ) -> dict[str, Any]:
                    with client.stream(
                        method, self.base_url + path, headers=headers, content=body
                    ) as response:
                        if response.status_code != 200:
                            return decode_response(response.status_code, b"")
                        data = bytearray()
                        for chunk in response.iter_bytes(chunk_size=65536):
                            data.extend(chunk)
                            if len(data) > self.max_response_bytes:
                                raise ValueError("inference_response_limit")
                    return decode_response(response.status_code, bytes(data))

                return infer(self.metadata(), messages, send)
        except httpx.TimeoutException:
            raise ValueError("inference_timeout") from None
        except httpx.HTTPError:
            raise ValueError("inference_connection") from None
