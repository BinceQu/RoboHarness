"""Install a relative workspace import path; no parent repository imports."""

from pathlib import Path
import argparse
import importlib
import json
import sys
import sysconfig


def configure():
    site = Path(sysconfig.get_paths()["purelib"])
    root = Path(__file__).resolve().parents[2]
    if not site.resolve().is_relative_to(root):
        raise RuntimeError("Use this workspace .venv/bin/python")
    (site / "workspace.pth").write_text("../../../..\n")
    print("Local Python environment ready:", root)


def check():
    root = Path(__file__).resolve().parents[2]
    for name, path in [
        ("interpreter", sys.executable),
        ("standard_library", sys.base_prefix),
        ("site_packages", sysconfig.get_paths()["purelib"]),
    ]:
        if not Path(path).resolve().is_relative_to(root):
            raise RuntimeError(f"{name} escapes workspace: {path}")
    modules = {}
    for name in [
        "numpy",
        "cv2",
        "PIL",
        "scipy",
        "trimesh",
        "moderngl",
        "msgpack",
        "msgpack_numpy",
        "pydantic",
        "websockets",
        "yaml",
        "h5py",
        "tools.util.runtime",
        "robot_config.control",
    ]:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"External module implementation: {name}: {path}")
        modules[name] = str(path.relative_to(root))
    escaping = [
        str(path)
        for path in root.rglob("*")
        if path.is_symlink() and not path.resolve().is_relative_to(root)
    ]
    if escaping:
        raise RuntimeError(f"External workspace links: {escaping}")
    print(
        json.dumps({"ok": True, "workspace": str(root), "modules": modules}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else configure()
