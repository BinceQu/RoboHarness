"""Small copy of the official evaluator msgpack wire codec.

Keeping this codec local makes the sidecar independent from the mutable
interface tree.  The representation matches the v3.9.x evaluator: NumPy
arrays are encoded as a map containing ``__ndarray__``, dtype, shape, and raw
bytes.  The proxy forwards frames unchanged; decoding is used only for the
read-only reset/action checks.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np


def pack_data(obj: Any) -> Any:
    """Encode the NumPy values used by the evaluator protocol."""

    # Keep the same boundary as the official codec.  The proxy never creates
    # wire frames itself in production, but matching this conversion makes
    # local probes and future protocol helpers interchangeable with it.
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for the sidecar
        torch = None
    if torch is not None and isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_data(obj: dict[Any, Any]) -> Any:
    """Decode one NumPy extension map."""

    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
packb = functools.partial(msgpack.packb, default=pack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)


def mapping_get(payload: Any, key: str, default: Any = None) -> Any:
    """Read a protocol key regardless of msgpack's str/bytes key mode."""

    if not isinstance(payload, Mapping):
        return default
    if key in payload:
        return payload[key]
    encoded = key.encode("utf-8")
    return payload.get(encoded, default)


def has_reset_key(frame: bytes | bytearray | memoryview) -> bool:
    """Scan only top-level msgpack keys for the evaluator reset control.

    Observations contain large RGB-D arrays.  Decoding an entire observation
    merely to decide whether it is the tiny reset frame would duplicate the
    official interface's work in the sidecar.  ``Unpacker.skip`` walks each
    value without materializing it, while preserving the same key-presence
    rule as :func:`is_reset_payload`.  Malformed or non-map frames fail open;
    the unchanged backend remains responsible for their protocol error.
    """

    if not isinstance(frame, (bytes, bytearray, memoryview)):
        return False
    # A malformed control frame must be handled by the unchanged backend. Do
    # not let any msgpack implementation-specific decode exception tear down
    # the evaluator connection from the sidecar.
    try:
        unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)
        unpacker.feed(frame)
        size = unpacker.read_map_header()
        found = False
        for _ in range(size):
            key = unpacker.unpack()
            if key in ("reset", b"reset"):
                found = True
            unpacker.skip()
        # ``unpackb`` (used by the official interface) rejects trailing data;
        # match that behavior instead of recognizing a valid-looking prefix.
        # ``tell`` exists in supported msgpack releases; keep a guarded
        # fallback for older deployments so reset handling still works even
        # if only the optional trailing-data check is unavailable.
        tell = getattr(unpacker, "tell", None)
        if callable(tell) and tell() != len(frame):
            return False
        return found
    except Exception:
        return False


def is_reset_payload(payload: Any) -> bool:
    """Return whether a decoded evaluator frame is the reset control frame.

    The v3.9.1 interface treats the presence of ``reset`` as the control
    signal (it doesn't inspect the value).  Preserve that wire-level rule so a
    malformed-but-recognizable reset frame isn't accidentally forwarded into
    the request/response path and left waiting for an action that will never
    arrive.
    """

    if not isinstance(payload, Mapping):
        return False
    return "reset" in payload or b"reset" in payload
