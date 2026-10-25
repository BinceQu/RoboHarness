"""V2 workspace harness: ``./run.sh --harness v2 --task task01 [...]``.

The tasks, prompts, budgets, evaluator and scoring are the ones the V1
harnesses use; only the agent side differs. Each case gets a fresh copy of the
R1_pro workspace, whose Python tools send atomic commands to the evaluator
through a local driver. The atomic driver needs the workspace's libraries, so
the runner re-executes itself under the workspace Python before starting.
"""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import uuid

from . import core as runner

WORKSPACE = runner.ROOT / "harness/v2/R1_pro"
V1_ONLY = ("supplemental_instances", "context_instance", "port", "write_video")


def main(args, argv):
    for key in V1_ONLY:
        if getattr(args, key):
            raise ValueError(f"--{key.replace('_', '-')} is not available with --harness v2")
    task = runner.load_task(args.task)
    config = runner.read_config(args.config)
    for key in ("model_url", "model"):
        if getattr(args, key):
            config[key] = getattr(args, key)
    python = Path(config.get("workspace_python") or WORKSPACE / ".venv/bin/python")
    if not python.is_absolute():
        python = runner.ROOT / python
    config["workspace_python"] = str(python)
    if config["unmask_evaluator_cuda"]:
        raise ValueError("Workspace runs require unmask_evaluator_cuda=false")
    if config["session_timeout_s"] != 0:
        raise ValueError("Workspace runs have no episode wall-clock cap")
    ports = runner.listener_ports(task, config)
    token = f"workspace-{task['task']}-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = args.run_dir.resolve() if args.run_dir else runner.ROOT / "runs" / token
    first = str(task["cases"][0]["instance_id"])
    plan = {"run_id": token, "run_dir": str(path), "port": ports["http"], "policy_port": ports["policy"],
        "gate_port": ports["gate"], "gpu": args.gpu, "harness": "v2", "model": config["model"],
        "model_url": config["model_url"], "diagnostic_only": bool(args.instances or args.smoke), "write_video": False,
        "smoke": args.smoke,
        "task_config": task, "cases": runner.select_cases(task, first if args.smoke else args.instances),
        "architecture": "agent Python -> atomic action driver -> evaluator; separate read-only monitor"}
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if path.exists():
        raise ValueError(f"Run directory must be new: {path}")
    if Path(sys.prefix).resolve() != python.parent.parent.resolve():
        if not python.exists():
            raise ValueError(f"Missing workspace environment {python}; run harness/v2/R1_pro/setup.sh")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            filter(None, [str(runner.ROOT / "eval"), os.environ.get("PYTHONPATH")])))
        os.execve(str(python), [str(python), "-m", "runner", *argv], env)
    from .workspace_run import WorkspaceRun, workspace_sources
    runner.preflight(config, "claude_code")
    from .assets import prepare_robot_asset
    prepare_robot_asset()
    locks = Path(config["cache_dir"]) / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        for port in sorted(ports.values()):
            handle = stack.enter_context((locks / f"port-{port}.lock").open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with socket.socket() as sock:
                # A completed run may leave TCP TIME_WAIT sockets. Match the
                # servers' reuse behavior while still rejecting a live listener.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
        def stop(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, stop)
        WorkspaceRun(plan, config).execute()
    report = runner.read_json(path / "summary.json")
    complete = (not args.smoke and report["status"] == "complete"
        and report["n_finished"] == len(plan["cases"])
        and not any(case.get("workspace_source_changes") for case in report["cases"])
        and runner.read_json(path / "workspace-source.json") == workspace_sources())
    print(json.dumps({"results": str(path / "summary.json"), "mean_q": report["mean_q"],
                      "complete": complete}, indent=2))
    return 0 if complete else 2
