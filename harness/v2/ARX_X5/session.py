"""Create an independent copy of this complete robot workspace."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo


def create(destination, session_id, task=""):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", session_id):
        raise ValueError("session_id must be 1..64 safe characters")
    template = Path(__file__).resolve().parent
    target = Path(destination).absolute()
    if target.exists() or target.is_relative_to(template):
        raise ValueError("Choose a new destination outside the template")
    os.umask(0o077)

    def ignore(directory, names):
        return {
            n
            for n in names
            if n
            in {".runtime", ".session.json", ".dev", "__pycache__", ".pytest_cache"}
            or (
                Path(directory) == template and n in {"perception", "tool_call_history"}
            )
        }

    shutil.copytree(template, target, symlinks=True, ignore=ignore)
    for path in target.rglob("*"):
        if path.is_symlink():
            link = os.readlink(path)
            if os.path.isabs(link) and (
                link == str(template) or link.startswith(str(template) + "/")
            ):
                new = str(target) + link[len(str(template)) :]
                path.unlink()
                path.symlink_to(os.path.relpath(new, path.parent))
            if not path.resolve().is_relative_to(target.resolve()):
                raise ValueError(f"Workspace symlink escapes: {path}")
    # uv relocatable environments may retain an absolute interpreter home.
    for path in [target / ".venv/pyvenv.cfg", *list((target / ".venv/bin").iterdir())]:
        if path.is_file() and not path.is_symlink():
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            path.write_text(text.replace(str(template), str(target)))
    config = {
        "session_id": session_id,
        "task": task,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    (target / ".session.json").write_text(json.dumps(config, indent=2) + "\n")
    for directory in [
        ".runtime",
        "tool_call_history",
        "perception/proprioception",
        "perception/exteroception",
        ".claude",
        ".agents",
    ]:
        (target / directory).mkdir(parents=True, exist_ok=True)
    for directory in ["perception", "tool_call_history"]:
        readme = template / directory / "README.md"
        if readme.is_file():
            shutil.copy2(readme, target / directory / "README.md")
    for directory in [".claude", ".agents"]:
        link = target / directory / "skills"
        if not link.exists():
            link.symlink_to("../skills")
    (target / ".runtime/state.json").write_text(
        json.dumps({"counter": 0, "latest_image_id": None, "active_skill": None}) + "\n"
    )
    subprocess.run(
        [str(target / ".venv/bin/python"), str(target / "tools/util/environment.py")],
        cwd=target,
        check=True,
    )
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--task", default="")
    args = parser.parse_args()
    print(create(args.destination, args.id, args.task))
