"""Prove a copied workspace runs using only its own Python, code and assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.util.session import clone_workspace

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    copy = clone_workspace(ROOT, args.destination)
    env = {"PATH": str(copy / ".venv/bin") + ":/usr/bin:/bin", "HOME": os.environ["HOME"],
           "CUDA_VISIBLE_DEVICES": "5", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    executable = str(copy / ".venv/bin/python")
    report = json.loads(subprocess.check_output([executable, "tools/util/environment.py", "--check"],
                        cwd=copy, env=env, text=True))
    passed = []
    for path in sorted((copy / "tools").glob("*.py")):
        if path.stem in {"__init__", "activate_skill", "deactivate_skill"}:
            continue
        schema = json.loads(subprocess.check_output([executable, str(path), "--schema"],
                           cwd=copy, env=env, text=True, timeout=60))
        assert schema["type"] == "object", path
        passed.append(path.stem)
    # An unrelated working directory must not resolve the original package.
    code = "import tools.util.robot as r; import sys,json; print(json.dumps({'module':r.__file__,'prefix':sys.base_prefix}))"
    outside = json.loads(subprocess.check_output([executable, "-c", code], cwd="/tmp", env=env, text=True))
    assert Path(outside["module"]).resolve().is_relative_to(copy)
    assert Path(outside["prefix"]).resolve().is_relative_to(copy)
    for relative in ("tools/capture_head_camera.py", "robot_config/meshes/base_link.obj", ".venv/lib/python3.10/site-packages/numpy/__init__.py"):
        original, duplicate = ROOT / relative, copy / relative
        assert original.stat().st_ino != duplicate.stat().st_ino, relative
        assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(duplicate.read_bytes()).digest()
    report.update(tool_schemas=passed, tool_count=len(passed), external_cwd=outside,
                  copies_have_distinct_inodes=True, pythonpath=None)
    (copy.parent / "standalone-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
