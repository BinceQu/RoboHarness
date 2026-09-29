"""Simulator-independent BEHAVIOR evaluator websocket protocol helpers."""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np


def pack_data(obj: Any) -> Any:
    """Encode NumPy values using the representation used by the v3.9.1 evaluator."""
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
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


def unpack_data(obj: dict) -> Any:
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
