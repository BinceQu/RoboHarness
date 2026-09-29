from __future__ import annotations

import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request

from .config import behavior_urlopen as urlopen, validate_owned_origin

from .errors import ConfigurationError, RemoteAPIError, TransportError


class RestClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        validate_owned_origin(base_url)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._origin = urlparse(self.base_url)

    def get_json(self, path: str) -> Any:
        return self._request_json("GET", path, None)

    def get_memory(self, *, timeout_s: float) -> Any:
        """Read live interface memory with a caller-owned bounded timeout."""
        return self._request_json(
            "GET", "/api/memory", None, timeout_s=timeout_s
        )

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        return self._request_json("POST", path, payload)

    def get_rollout_budget(self, *, timeout_s: float) -> Any:
        return self._request_json(
            'GET', '/api/rollout_budget', None, timeout_s=timeout_s
        )

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        self._validate_api_path(path, method)
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        raw, status, _ = self._request(
            Request(self.base_url + path, data=body, headers=headers, method=method),
            timeout_s=timeout_s,
        )
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteAPIError(
                "BEHAVIOR server returned a non-JSON response.",
                status=status,
                response=raw[:1000].decode("utf-8", errors="replace"),
            ) from exc

    def get_media(self, reference: str, max_bytes: int) -> tuple[bytes, str]:
        parsed = urlparse(reference)
        if parsed.scheme:
            if (parsed.scheme, parsed.hostname, parsed.port) != (
                self._origin.scheme,
                self._origin.hostname,
                self._origin.port,
            ):
                raise ConfigurationError(
                    "Refusing media URL outside the configured BEHAVIOR origin."
                )
            url = reference
        else:
            if not reference.startswith("/"):
                raise ConfigurationError("Relative media URL must start with '/'.")
            url = urljoin(self.base_url + "/", reference.lstrip("/"))
        raw, _, headers = self._request(
            Request(url, headers={"Accept": "image/*"}, method="GET"),
            max_bytes=max_bytes,
        )
        return raw, headers.get("Content-Type", "").split(";", 1)[0]

    def _request(
        self,
        request: Request,
        *,
        max_bytes: int | None = None,
        timeout_s: float | None = None,
    ) -> tuple[bytes, int, Any]:
        request_timeout_s = self.timeout_s if timeout_s is None else timeout_s
        if request_timeout_s <= 0:
            raise ConfigurationError("HTTP request timeout must be positive.")
        try:
            with urlopen(request, timeout=request_timeout_s) as response:
                limit = max_bytes + 1 if max_bytes is not None else None
                raw = response.read(limit)
                if max_bytes is not None and len(raw) > max_bytes:
                    raise ConfigurationError(
                        f"Returned media exceeds {max_bytes} bytes."
                    )
                return raw, int(response.status), response.headers
        except HTTPError as exc:
            raw = exc.read(64 * 1024)
            try:
                response: Any = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                response = raw.decode("utf-8", errors="replace")
            message = "BEHAVIOR API request failed."
            if isinstance(response, dict):
                message = str(response.get("error") or response.get("message") or message)
            raise RemoteAPIError(
                message, status=int(exc.code), response=response
            ) from exc
        except (URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            reason = getattr(exc, "reason", exc)
            raise TransportError(
                f"Unable to reach BEHAVIOR server at {self.base_url}.",
                details=str(reason),
            ) from exc

    @staticmethod
    def _validate_api_path(path: str, method: str) -> None:
        if method == "GET" and path in {
            "/api/memory",
            '/api/rollout_budget',
            "/api/state",
            "/api/v2/tools",
        }:
            return
        if method == "POST" and path.startswith("/api/v2/"):
            suffix = path.removeprefix("/api/v2/")
            if suffix and all(ch.islower() or ch.isdigit() or ch == "_" for ch in suffix):
                return
        raise ConfigurationError(f"REST path is outside the adapter boundary: {path}")
