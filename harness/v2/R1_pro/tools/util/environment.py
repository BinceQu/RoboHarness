"""Configure and inspect the Python environment inside this exact workspace."""
import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import sysconfig

WORKSPACE = Path(__file__).resolve().parents[2]


def configure():
    # A relative .pth entry keeps `python tools/name.py` importable even after
    # copying/renaming the complete workspace, without external PYTHONPATH.
    site = Path(sysconfig.get_path("purelib"))
    if not site.resolve().is_relative_to(WORKSPACE):
        raise ValueError("Run this using this workspace's .venv/bin/python")
    (site / "_r1_pro_workspace.pth").write_text(os.path.relpath(WORKSPACE, site) + "\n")
    for path in WORKSPACE.rglob("*"):
        if path.is_symlink():
            target = Path(os.readlink(path))
            if target.is_absolute() and target.is_relative_to(WORKSPACE):
                path.unlink()
                path.symlink_to(os.path.relpath(target, path.parent))
    config = WORKSPACE / ".venv/pyvenv.cfg"
    if "include-system-site-packages = false" not in config.read_text():
        raise ValueError("System site-packages must be disabled")


def check():
    for name, value in (("prefix", sys.prefix), ("base_prefix", sys.base_prefix), ("executable", sys.executable)):
        if not Path(value).resolve().is_relative_to(WORKSPACE):
            raise ValueError(f"Python {name} escapes this workspace: {value}")
    modules = {}
    for name in ("numpy", "scipy", "PIL", "cv2", "msgpack", "websockets", "jsonschema",
                 "trimesh", "moderngl", "torch", "torchvision", "warp", "curobo", "sam2",
                 "tools.util.robot", "tools.util.driver"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(WORKSPACE):
            raise ValueError(f"Module is outside this workspace: {name}: {path}")
        modules[name] = str(path.relative_to(WORKSPACE))
    for path in WORKSPACE.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(WORKSPACE):
            raise ValueError(f"External symlink: {path}")
    print(json.dumps({"ok": True, "workspace": str(WORKSPACE), "modules": modules,
                      "torch": metadata.version("torch"), "system_site_packages": False}, indent=2))


def lock():
    # Vendored CUDA/SAM packages are installed separately from local paths.
    # Imports such as setuptools may append vendored dependencies to sys.path.
    # Lock installed distributions only, not those private bundled copies.
    versions = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions(
                       path=[sysconfig.get_path("purelib")])
                       if d.metadata["Name"].lower().replace("_", "-") not in {"nvidia-curobo", "sam-2"}})
    (WORKSPACE / "requirements.lock").write_text("# Exact third-party environment; no external project installations.\n" + "\n".join(versions) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configure", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--lock", action="store_true")
    args = parser.parse_args()
    if args.configure:
        configure()
    if args.check:
        check()
    if args.lock:
        lock()
