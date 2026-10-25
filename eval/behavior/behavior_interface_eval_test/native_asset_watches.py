"""Process-local workaround for the pinned Kit asset reload mutex abort.

The extension and MDL watcher settings do not disable OmniClient's local
file subscriptions. In the archived runtime, a texture metadata event can
abort carb.assets on a tasking worker. Evaluation reads immutable assets, so
we disable those subscriptions before SimulationApp starts. Initial reads
and explicit scene loads still use the unmodified native implementation.

This is an internal OmniClient entry point, NOT a supported public API. Its
ABI is accepted only for the exact binary exercised by our native regression
test. Unknown binaries fail before dlopen; never silently run unprotected.
No installed SDK file or user/global configuration is modified.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path


PINNED_SHA256 = "96d901619a7ed20db00e96b8cf41c523865289d11d426cd520bdf9f7c6e60302"
PINNED_VERSION = "2.67.0-release.6072+gl.6293b5e9"
_loaded_libraries: dict[Path, object] = {}


def disable_native_asset_watches(library_path: Path | None = None) -> dict[str, str]:
    """Disable native watches before Kit startup, or fail with a clear error."""
    if library_path is None:
        import isaacsim

        library_path = (
            Path(isaacsim.__file__).parent
            / "kit/extscore/omni.client.lib/bin/libomniclient.so"
        )
    path = Path(library_path).resolve(strict=True)
    if path not in _loaded_libraries:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != PINNED_SHA256:
            raise RuntimeError(
                "Unsupported OmniClient binary for archived evaluation: "
                f"{path} sha256={actual}; expected {PINNED_SHA256}. "
                "Install the pinned Isaac Sim 5.1.0 runtime. The asset-watch "
                "workaround requires native regression testing before accepting "
                "a different binary; do not bypass this check."
            )
        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        try:
            get_version = library.omniClientGetVersionString
            set_watches = library.testSetWatchesEnabled
        except AttributeError as exc:
            raise RuntimeError("Pinned OmniClient is missing the tested native ABI") from exc
        get_version.argtypes = []
        get_version.restype = ctypes.c_char_p
        version = get_version().decode("ascii")
        if version != PINNED_VERSION:
            raise RuntimeError(f"Unexpected loaded OmniClient version: {version}")
        set_watches.argtypes = [ctypes.c_bool]
        set_watches.restype = None
        set_watches(False)
        # Keep the process-local library reference for the lifetime of Kit.
        _loaded_libraries[path] = library
    return {
        "policy": "immutable-assets-no-native-watches",
        "library": str(path),
        "sha256": PINNED_SHA256,
        "version": PINNED_VERSION,
    }
