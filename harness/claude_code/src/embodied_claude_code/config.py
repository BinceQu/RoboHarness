from __future__ import annotations

from dataclasses import dataclass
from http.client import HTTPConnection
import ipaddress
import os
import re
import socket
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPHandler, HTTPRedirectHandler, ProxyHandler, build_opener, urlopen

from .errors import ConfigurationError


TRUE_VALUES = {"1", "true", "yes", "on"}
MAX_MEMORY_TIMEOUT_S = 5.0
DEFAULT_MODEL_IMAGE_MAX_BYTES = 256 * 1024
DEFAULT_MODEL_IMAGE_MAX_EDGE = 720
DEFAULT_MODEL_IMAGE_JPEG_QUALITY = 95
DEFAULT_IMAGE_CONVERTER = "/usr/bin/convert"


def official_task_index_for_port(port: int) -> int | None:
    """Catalog ports: task10–45 on 15010–15045, task0–9 on 15060–15069."""
    if os.environ.get('ROBOHARNESS_HTTP_PORT') == str(port):
        task = os.environ.get('ROBOHARNESS_TASK_ID', '')
        if task.isdigit() and 0 <= int(task) < 50:
            return int(task)
        return None
    if 15010 <= port <= 15045:
        return port - 15000
    if 15060 <= port <= 15069:
        return port - 15060
    return None


def validate_owned_origin(url: str, session_id: str | None = None) -> None:
    """A harness-owned agent can only address its assigned loopback port."""
    raw = os.environ.get("BEHAVIOR_EVAL_OWNER_PORT", "")
    if not raw:
        return
    if not raw.isdigit():
        raise ConfigurationError("Invalid official port ownership.")
    port = int(raw)
    task_index = official_task_index_for_port(port)
    if task_index is None:
        raise ConfigurationError("Invalid official port ownership.")
    parsed = urlparse(url)
    try:
        valid = (parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
                 and parsed.port == port and not parsed.username and not parsed.password)
    except ValueError:
        valid = False
    if not valid:
        raise ConfigurationError(f"Official agent is restricted to http://127.0.0.1:{port}.")
    sid = os.environ.get("BEHAVIOR_SESSION_ID", "") if session_id is None else session_id
    if len(sid) > 64 or not re.fullmatch(rf"t{task_index:02d}p{port}i[0-9]-[A-Za-z0-9_.-]+", sid):
        raise ConfigurationError("Official session does not match its task/port owner.")


class OwnedPortRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_owned_origin(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class OwnedPortHTTPConnection(HTTPConnection):
    """Retry loopback TCP establishment, never a transmitted robot request."""

    connect_attempts = 3
    connect_timeout_s = 3.0
    connect_budget_s = 10.0
    connect_backoff_s = 0.1

    def connect(self):
        # A tunnel sends HTTP bytes during connect(); it is not safe to replay.
        if self._tunnel_host:
            return super().connect()
        original_timeout = self.timeout
        response_timeout = (socket.getdefaulttimeout()
                            if original_timeout is socket._GLOBAL_DEFAULT_TIMEOUT
                            else original_timeout)
        budget = self.connect_budget_s
        if response_timeout is not None:
            budget = min(budget, response_timeout)
        deadline = time.monotonic() + budget
        try:
            for attempt in range(self.connect_attempts):
                self.timeout = min(self.connect_timeout_s, max(0.001, deadline - time.monotonic()))
                try:
                    super().connect()
                except OSError:
                    # close() would also reset the pending HTTP request state.
                    if self.sock is not None:
                        self.sock.close()
                        self.sock = None
                    remaining = deadline - time.monotonic()
                    if attempt + 1 == self.connect_attempts or remaining <= 0:
                        raise
                    time.sleep(min(self.connect_backoff_s, remaining))
                else:
                    self.sock.settimeout(response_timeout)
                    return
        finally:
            self.timeout = original_timeout


class OwnedPortHTTPHandler(HTTPHandler):
    def http_open(self, req):
        return self.do_open(OwnedPortHTTPConnection, req)


def behavior_urlopen(request, *, timeout):
    if not os.environ.get("BEHAVIOR_EVAL_OWNER_PORT"):
        return urlopen(request, timeout=timeout)
    validate_owned_origin(request.full_url)
    request.add_header("X-Behavior-Session-ID", os.environ["BEHAVIOR_SESSION_ID"])
    # Do not let environment proxy settings or redirects leave the bound port.
    return build_opener(ProxyHandler({}), OwnedPortRedirectHandler(),
                        OwnedPortHTTPHandler()).open(request, timeout=timeout)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc


def _is_loopback(hostname: str | None) -> bool:
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _default_record_root() -> Path:
    if value := (
        os.environ.get("CLAUDE_PLUGIN_DATA") or os.environ.get("PLUGIN_DATA")
    ):
        return Path(value).expanduser() / "trajectories"
    data_home = Path(
        os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
    )
    return data_home / "embodied-claude-code" / "trajectories"


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://127.0.0.1:5011"
    http_timeout_s: float = 1900.0
    memory_timeout_s: float = 1.0
    allow_remote: bool = False
    profile_path: Path | None = None
    record_root: Path = Path("trajectories")
    record: bool = True
    session_id: str = ""
    record_label: str = ""
    max_image_bytes: int = 20 * 1024 * 1024
    max_images_per_call: int = 1
    model_image_max_bytes: int = DEFAULT_MODEL_IMAGE_MAX_BYTES
    model_image_max_edge: int = DEFAULT_MODEL_IMAGE_MAX_EDGE
    model_image_jpeg_quality: int = DEFAULT_MODEL_IMAGE_JPEG_QUALITY
    image_converter: str = DEFAULT_IMAGE_CONVERTER

    def __post_init__(self) -> None:
        validate_owned_origin(self.base_url, self.session_id)
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConfigurationError(
                "BEHAVIOR_BASE_URL must be an absolute http(s) origin."
            )
        if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
            raise ConfigurationError(
                "BEHAVIOR_BASE_URL must not contain a path, query, or fragment."
            )
        if not self.allow_remote and not _is_loopback(parsed.hostname):
            raise ConfigurationError(
                "Non-loopback BEHAVIOR_BASE_URL requires BEHAVIOR_ALLOW_REMOTE=1."
            )
        if self.http_timeout_s <= 0:
            raise ConfigurationError("BEHAVIOR_HTTP_TIMEOUT_S must be positive.")
        if not 0 < self.memory_timeout_s <= MAX_MEMORY_TIMEOUT_S:
            raise ConfigurationError(
                "BEHAVIOR_MEMORY_TIMEOUT_S must be greater than 0 and at most "
                f"{MAX_MEMORY_TIMEOUT_S:g}."
            )
        if self.max_image_bytes <= 0 or self.max_images_per_call <= 0:
            raise ConfigurationError("Image limits must be positive.")
        if self.model_image_max_bytes <= 0 or self.model_image_max_edge <= 0:
            raise ConfigurationError("Model image limits must be positive.")
        if not 1 <= self.model_image_jpeg_quality <= 95:
            raise ConfigurationError(
                "BEHAVIOR_MODEL_IMAGE_JPEG_QUALITY must be between 1 and 95."
            )
        if not self.image_converter.strip():
            raise ConfigurationError("BEHAVIOR_IMAGE_CONVERTER must not be empty.")

    @classmethod
    def from_env(cls) -> "Settings":
        profile = os.environ.get("BEHAVIOR_PROFILE", "").strip()
        record_root = os.environ.get("BEHAVIOR_RECORD_ROOT", "").strip()
        try:
            timeout = float(os.environ.get("BEHAVIOR_HTTP_TIMEOUT_S", "1900"))
        except ValueError as exc:
            raise ConfigurationError(
                "BEHAVIOR_HTTP_TIMEOUT_S must be numeric."
            ) from exc
        try:
            memory_timeout = float(
                os.environ.get("BEHAVIOR_MEMORY_TIMEOUT_S", "1")
            )
        except ValueError as exc:
            raise ConfigurationError(
                "BEHAVIOR_MEMORY_TIMEOUT_S must be numeric."
            ) from exc
        return cls(
            base_url=os.environ.get(
                "BEHAVIOR_BASE_URL", "http://127.0.0.1:5011"
            ).rstrip("/"),
            http_timeout_s=timeout,
            memory_timeout_s=memory_timeout,
            allow_remote=os.environ.get("BEHAVIOR_ALLOW_REMOTE", "0").lower()
            in TRUE_VALUES,
            profile_path=Path(profile).expanduser() if profile else None,
            record_root=(
                Path(record_root).expanduser()
                if record_root
                else _default_record_root()
            ),
            record=os.environ.get("BEHAVIOR_RECORD", "1").lower()
            in TRUE_VALUES,
            session_id=os.environ.get("BEHAVIOR_SESSION_ID", "").strip(),
            record_label=os.environ.get("BEHAVIOR_RECORD_LABEL", "").strip(),
            model_image_max_bytes=_env_int(
                "BEHAVIOR_MODEL_IMAGE_MAX_BYTES",
                DEFAULT_MODEL_IMAGE_MAX_BYTES,
            ),
            model_image_max_edge=_env_int(
                "BEHAVIOR_MODEL_IMAGE_MAX_EDGE",
                DEFAULT_MODEL_IMAGE_MAX_EDGE,
            ),
            model_image_jpeg_quality=_env_int(
                "BEHAVIOR_MODEL_IMAGE_JPEG_QUALITY",
                DEFAULT_MODEL_IMAGE_JPEG_QUALITY,
            ),
            image_converter=os.environ.get(
                "BEHAVIOR_IMAGE_CONVERTER", DEFAULT_IMAGE_CONVERTER
            ).strip(),
        )
