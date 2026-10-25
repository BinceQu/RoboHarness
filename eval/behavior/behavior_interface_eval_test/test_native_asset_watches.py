from __future__ import annotations

import ctypes
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from behavior_interface_eval_test import native_asset_watches as watches


class NativeAssetWatchesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "libomniclient.so"
        self.path.write_bytes(b"native ABI fixture")
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.library = SimpleNamespace(
            omniClientGetVersionString=mock.Mock(
                return_value=watches.PINNED_VERSION.encode("ascii")
            ),
            testSetWatchesEnabled=mock.Mock(),
        )
        patch = mock.patch.object(watches, "_loaded_libraries", {})
        patch.start()
        self.addCleanup(patch.stop)

    def test_unknown_binary_is_rejected_before_loading_native_code(self):
        with mock.patch.object(watches.ctypes, "CDLL") as load:
            with self.assertRaisesRegex(RuntimeError, "Unsupported OmniClient"):
                watches.disable_native_asset_watches(self.path)
            load.assert_not_called()

    def test_pinned_abi_disables_subscriptions_once_with_bool_signature(self):
        with mock.patch.object(watches, "PINNED_SHA256", self.digest), mock.patch.object(
            watches.ctypes, "CDLL", return_value=self.library
        ) as load:
            first = watches.disable_native_asset_watches(self.path)
            self.assertEqual(first, watches.disable_native_asset_watches(self.path))
            load.assert_called_once_with(str(self.path.resolve()), mode=ctypes.RTLD_LOCAL)
        self.library.testSetWatchesEnabled.assert_called_once_with(False)
        self.assertEqual(self.library.testSetWatchesEnabled.argtypes, [ctypes.c_bool])
        self.assertIsNone(self.library.testSetWatchesEnabled.restype)
        self.assertEqual(first["sha256"], self.digest)

    def test_wrong_loaded_version_cannot_change_native_switch(self):
        self.library.omniClientGetVersionString.return_value = b"unknown"
        with mock.patch.object(watches, "PINNED_SHA256", self.digest), mock.patch.object(
            watches.ctypes, "CDLL", return_value=self.library
        ):
            with self.assertRaisesRegex(RuntimeError, "Unexpected loaded"):
                watches.disable_native_asset_watches(self.path)
        self.library.testSetWatchesEnabled.assert_not_called()
        self.assertFalse(watches._loaded_libraries)

    def test_missing_symbol_is_a_startup_failure(self):
        del self.library.testSetWatchesEnabled
        with mock.patch.object(watches, "PINNED_SHA256", self.digest), mock.patch.object(
            watches.ctypes, "CDLL", return_value=self.library
        ):
            with self.assertRaisesRegex(RuntimeError, "missing the tested native ABI"):
                watches.disable_native_asset_watches(self.path)
        self.assertFalse(watches._loaded_libraries)


if __name__ == "__main__":
    unittest.main()
