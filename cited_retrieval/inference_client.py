"""Portable stdlib-only Windows client. Copy this file alone; Python >=3.11.

Reads an exported private packet, connects ONLY after explicit authorization, and
writes a NEW private response file. No model/index/embedding dependencies. This
module also supplies shared wire validation for the integrated HTTP adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def validate_config(config: dict[str, Any]) -> None:
    try:
        base = config["base_url"]
        url = urlsplit(base)
        model = config["model"]
        timeout = config["timeout"]
        auth = config["auth_env"]
        valid = (
            isinstance(base, str)
            and url.scheme in {"http", "https"}
            and url.hostname
            and url.port != 0
            and url.username is None
            and url.password is None
            and not url.query
            and not url.fragment
            and "?" not in base
            and "#" not in base
            and not any(c.isspace() or ord(c) < 32 for c in base)
            and isinstance(model, str)
            and model
            and model == model.strip()
            and not any(ord(c) < 32 for c in model)
            and type(timeout) in {int, float}
            and math.isfinite(timeout)
            and timeout > 0
            and all(
                type(config[k]) is int and config[k] > 0
                for k in ("max_response_bytes", "max_tokens", "max_prompt_bytes")
            )
            and (
                auth is None
                or (isinstance(auth, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", auth))
            )
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        valid = False
    if not valid:
        raise ValueError("invalid_inference_config")


def packet_id(provider: dict[str, Any], messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"provider": provider, "messages": messages}, ensure_ascii=True, sort_keys=True
        ).encode()
    ).hexdigest()


def private_path(path: str | Path) -> Path:
    """Refuse private artifacts inside any main or linked Git checkout."""
    resolved = Path(path).resolve()
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValueError("unsafe_answer_path")
    return resolved


def completion_body(config: dict[str, Any], messages: list[dict[str, str]]) -> bytes:
    """One serialization and size policy for preparation and both HTTP clients."""
    validate_config(config)
    body = json.dumps(
        {
            "model": config["model"],
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "max_tokens": config["max_tokens"],
        }
    ).encode()
    if len(body) > config["max_prompt_bytes"]:
        raise ValueError("inference_context_limit")
    return body


def infer(
    config: dict[str, Any],
    messages: list[dict[str, str]],
    send: Callable[[str, str, dict[str, str], bytes | None], dict[str, Any]],
) -> dict[str, Any]:
    body = completion_body(config, messages)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if config["auth_env"] is not None:
        secret = os.environ.get(config["auth_env"])
        if not secret or any(ord(c) < 32 or ord(c) > 126 for c in secret):
            raise ValueError("inference_auth")
        headers["Authorization"] = "Bearer " + secret
    advertised = send("GET", "/models", headers, None)
    if not isinstance(advertised.get("data"), list) or not all(
        isinstance(item, dict) and isinstance(item.get("id"), str) for item in advertised["data"]
    ):
        raise ValueError("inference_protocol")
    if config["model"] not in [item["id"] for item in advertised["data"]]:
        raise ValueError("inference_model_mismatch")
    return send("POST", "/chat/completions", headers, body)


def decode_response(status: int, data: bytes) -> dict[str, Any]:
    if status in {401, 403}:
        raise ValueError("inference_auth")
    if status != 200:
        raise ValueError("inference_http_status")
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeError):
        raise ValueError("inference_protocol") from None
    if not isinstance(payload, dict):
        raise ValueError("inference_protocol")  # noqa: TRY004 - stable wire failure code
    return payload


def run_packet(packet: dict[str, Any], *, connection_authorized: bool = False) -> dict[str, Any]:
    """Perform actual HTTP using stdlib only; no proxies or redirects."""
    if not connection_authorized:
        raise ValueError("connection_not_authorized")
    try:
        config = packet["provider"]
        validate_config(config)
        messages = packet["messages"]
        if (
            packet["schema"] != 1
            or packet["request_id"] != packet_id(config, messages)
            or not isinstance(messages, list)
            or len(messages) != 2
            or [m["role"] for m in messages] != ["system", "user"]
            or not all(isinstance(m["content"], str) for m in messages)
        ):
            raise ValueError("invalid_request_packet")
    except (KeyError, TypeError):
        raise ValueError("invalid_request_packet") from None
    url = urlsplit(config["base_url"])

    def send(method: str, path: str, headers: dict[str, str], body: bytes | None) -> dict[str, Any]:
        connection_type = (
            http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_type(str(url.hostname), url.port, timeout=config["timeout"])
        try:
            connection.request(method, url.path.rstrip("/") + path, body=body, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                return decode_response(response.status, b"")
            data = response.read(config["max_response_bytes"] + 1)
            if len(data) > config["max_response_bytes"]:
                raise ValueError("inference_response_limit")
            return decode_response(response.status, data)
        except TimeoutError:
            raise ValueError("inference_timeout") from None
        except (OSError, http.client.HTTPException):
            raise ValueError("inference_connection") from None
        finally:
            connection.close()

    try:
        raw = infer(config, messages, send)
        return {"schema": 1, "request_id": packet["request_id"], "response": raw}
    except ValueError as exc:
        return {"schema": 1, "request_id": packet["request_id"], "error": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    parser.add_argument("--response", required=True)
    parser.add_argument("--connection-authorized", action="store_true", required=True)
    args = parser.parse_args(argv)
    # Reserve the output BEFORE network I/O. Never overwrite an input/user file.
    try:
        with open(private_path(args.request), encoding="utf-8") as stream:
            packet = json.load(stream)
        fd = os.open(private_path(args.response), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            result = run_packet(packet, connection_authorized=args.connection_authorized)
            json.dump(result, stream, ensure_ascii=True, sort_keys=True)
            stream.write("\n")
        print(
            json.dumps(
                {
                    "status": "error" if "error" in result else "response_saved",
                    **({"error": result["error"]} if "error" in result else {}),
                }
            )
        )
        return 2 if "error" in result else 0
    except Exception:  # noqa: BLE001 - private CLI input and provider privacy seam
        print('{"status":"error","error":"client_failed"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
