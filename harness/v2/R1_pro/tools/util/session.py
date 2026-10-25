"""Clone the complete harness and bind its private state to one episode."""
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil

from .files import atomic_json
from .paths import WORKSPACE
from .protocol import request


def clone_workspace(template, destination, *, with_environment=True):
    """Copy bytes, never hardlink another session's writable code or libraries.

    `with_environment=False` is for isolated protocol unit tests only. Normal
    sessions copy their interpreter and installed third-party libraries too.
    Internal symlinks are relocated; escaping symlinks are rejected.
    """
    template, destination = Path(template).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Use a new directory for each session")
    if (template / ".session.json").exists():
        raise ValueError("Cannot clone a live session as a clean template")
    if with_environment and not (template / ".venv/bin/python").is_file():
        raise RuntimeError("Run setup.sh inside the template first")
    excluded = ["__pycache__", "*.pyc", ".pytest_cache", ".runtime", ".dev"]
    if not with_environment:
        excluded += [".venv", ".python"]
    shutil.copytree(template, destination, symlinks=True,
                    ignore=shutil.ignore_patterns(*excluded))
    for path in destination.rglob("*"):
        if not path.is_symlink():
            continue
        target = Path(os.readlink(path))
        if target.is_absolute() and target.is_relative_to(template):
            replacement = destination / target.relative_to(template)
            path.unlink()
            path.symlink_to(os.path.relpath(replacement, path.parent))
        if not path.resolve().is_relative_to(destination):
            raise ValueError(f"Workspace symlink escapes its own directory: {path}")
    if with_environment:
        config = destination / ".venv/pyvenv.cfg"
        config.write_text(config.read_text().replace(str(template), str(destination)))
        for path in (destination / ".venv/bin").iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            try:
                text = path.read_text()
            except UnicodeError:
                continue
            if str(template) in text:
                path.write_text(text.replace(str(template), str(destination)))
    return destination


def default_trajectory(destination):
    destination = Path(destination)
    return (destination.parent / "trajectory" if destination.name == "workspace"
            else destination.with_name(destination.name + ".trajectory"))


def create(template, destination, socket, *, session_id, task_name, gpu=5,
           with_environment=True, trajectory_dir=None):
    template, destination = Path(template).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Use a new directory for each session")
    if not task_name.strip():
        raise ValueError("task_name is required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", session_id):
        raise ValueError("session_id must contain 1–64 safe filename characters")
    status = request(socket, {"operation": "status"})
    if not status["connected"] or status["session_id"] or not status["initialization"].get("ready"):
        raise RuntimeError("Evaluator must have a fresh initialized, unbound episode")
    clone_workspace(template, destination, with_environment=with_environment)
    state_dir = destination / ".runtime"
    trajectory = Path(trajectory_dir or default_trajectory(destination)).resolve()
    if trajectory.is_relative_to(destination):
        raise ValueError("trajectory_dir must be outside the model workspace")
    for name in ("claude-home", "compile"):
        (state_dir / name).mkdir(parents=True)
    for name in ("recordings", "calls"):
        (trajectory / name).mkdir(parents=True, exist_ok=True)
    (state_dir / "plan.key").write_text(secrets.token_hex(32) + "\n")
    config = {"workspace": str(destination), "state_dir": str(state_dir),
        "socket": str(Path(socket).resolve()), "session_id": session_id,
        "episode_id": status["episode_id"], "gpu": gpu,
        "mesh_root": str(destination / "robot_config"),
        "numerical_python": str(destination / ".venv/bin/python"),
        "compile_tmp": str(state_dir / "compile"), "task_name": task_name,
        "trajectory_dir": str(trajectory)}
    atomic_json(destination / ".session.json", config)
    atomic_json(state_dir / "state.json", {"session_id": session_id, "counter": 0,
        "active_skill": None, "skill_revision": 0, "latest_image_id": ""})
    atomic_json(destination / "skills/active.json", {"baseline": "behavior-v2-baseline", "active_skill": None, "revision": 0})
    for host in (".claude", ".agents"):
        directory = destination / host
        directory.mkdir(exist_ok=True)
        (directory / "skills").symlink_to("../skills", target_is_directory=True)
    request(socket, {"operation": "bind", "session_id": session_id, "episode_id": status["episode_id"]})
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--task-name", required=True)
    args = parser.parse_args()
    config = create(WORKSPACE, args.workspace, args.socket,
        session_id=args.session_id, task_name=args.task_name)
    import subprocess
    env = dict(os.environ, ROBOT_WORKSPACE=config["workspace"])
    env.pop("PYTHONPATH", None)
    result = subprocess.run([config["numerical_python"], "tools/capture_head_camera.py"],
        cwd=config["workspace"], env=env, text=True, capture_output=True, check=True)
    print(json.dumps({"workspace": config["workspace"],
                     "initial_observation": json.loads(result.stdout)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
