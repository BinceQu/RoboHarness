"""Evaluator-compatible MsgPack, also used on the private local control socket."""
import socket
import struct

import msgpack
import numpy as np

MAX_MESSAGE = 128 * 1024 * 1024


def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind in "Vc":
            raise TypeError("unsupported array dtype")
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(type(value).__name__)


def decode(value):
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.hasobject or dtype.kind in "Vc":
            raise ValueError("unsupported array dtype")
        # IPC must retain the controller's normal writable-array semantics.
        # A view over immutable MsgPack bytes breaks recovery code that adjusts
        # returned joint targets, such as locking torso yaw before a hold.
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"]).copy()
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack(value):
    return msgpack.packb(value, default=encode, use_bin_type=True)


def unpack(data):
    return msgpack.unpackb(data, object_hook=decode, raw=False, strict_map_key=False)


def receive_exact(stream, length):
    pieces = []
    while length:
        chunk = stream.recv(min(length, 1024 * 1024))
        if not chunk:
            raise EOFError("Control connection closed")
        pieces.append(chunk)
        length -= len(chunk)
    return b"".join(pieces)


def receive(stream):
    length = struct.unpack("!I", receive_exact(stream, 4))[0]
    if length > MAX_MESSAGE:
        raise ValueError("message too large")
    return unpack(receive_exact(stream, length))


def send(stream, value):
    data = pack(value)
    if len(data) > MAX_MESSAGE:
        raise ValueError("message too large")
    stream.sendall(struct.pack("!I", len(data)) + data)


def request(path, value, timeout=180):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(timeout)
        stream.connect(str(path))
        send(stream, value)
        reply = receive(stream)
        if not reply.get("ok"):
            message = reply.get("error", "control request failed")
            if reply.get("error_type") in {"ValueError", "TypeError", "KeyError", "TimeoutError"}:
                kind = {"ValueError": ValueError, "TypeError": TypeError,
                        "KeyError": KeyError, "TimeoutError": TimeoutError}[reply["error_type"]]
                raise kind(message)
            if reply.get("error_type") == "TrackObjectDistanceBindingError":
                from .geometry.tracked_object_distance import TrackObjectDistanceBindingError
                raise TrackObjectDistanceBindingError(reply.get("reason") or "remote_binding_error", message)
            raise RuntimeError(message)
        return reply["value"]
