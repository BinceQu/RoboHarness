"""Local IPC must preserve NumPy values and normal mutable return semantics."""
import numpy as np
import pytest

from tools.util.protocol import pack, unpack


@pytest.mark.parametrize("value", [
    np.array([-0.9921228289604187, 2.5305089950561523,
              0.15791626274585724, 0.0], dtype=np.float64),
    np.arange(27, dtype=np.float32),
    np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
    np.arange(12, dtype=np.float32).reshape(3, 4),
])
def test_numpy_roundtrip_returns_independent_writable_arrays(value):
    encoded = pack({"nested": [value]})
    received = unpack(encoded)["nested"][0]
    assert received.dtype == value.dtype and received.shape == value.shape
    assert np.array_equal(received, value)
    # The original in-process controller returns writable arrays. Tool code
    # is allowed to adjust a returned joint target without changing the
    # controller's measured state or any other decoded observation.
    received.flat[-1] = 7
    assert received.flat[-1] == 7
    assert np.array_equal(unpack(encoded)["nested"][0], value)

