from __future__ import annotations

from typing import Any


class EmbodiedError(RuntimeError):
    """A model-facing adapter error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Any = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details is not None:
            result["details"] = self.details
        return result


class ConfigurationError(EmbodiedError):
    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__("configuration_error", message, details=details)


class EpisodeStateError(EmbodiedError):
    def __init__(self, message: str) -> None:
        super().__init__("episode_state_error", message)


class ToolPolicyError(EmbodiedError):
    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__("tool_policy_error", message, details=details)


class CameraFrameError(EmbodiedError):
    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(
            "camera_frame_error",
            message,
            details=details,
            retryable=True,
        )


class TransportError(EmbodiedError):
    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(
            "transport_error", message, details=details, retryable=True
        )


class RemoteAPIError(EmbodiedError):
    def __init__(
        self,
        message: str,
        *,
        status: int,
        response: Any = None,
    ) -> None:
        super().__init__(
            "remote_api_error",
            message,
            details={"http_status": status, "response": response},
            retryable=status >= 500,
        )
        self.status = status
