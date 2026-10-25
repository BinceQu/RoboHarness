#!/usr/bin/env python3
"""Run archived BEHAVIOR cases with the V2 workspace harness.

The task definitions, prompts, budgets, evaluator and scoring are the same as
the V1 runner (`./run.sh`); only the agent side differs: each case gets a fresh
copy of the R1_pro workspace, and its Python tools send atomic commands to the
evaluator through a local driver.
"""
import argparse
import contextlib
import fcntl
import json
from pathlib import Path
import signal
import socket
import sys
import time
import uuid

V2 = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(V2 / "R1_pro"), str(V2.parents[1])]
from roboharness import runner
from direct_run import WorkspaceRun, workspace_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="task00", help="task00, 0, or task name")
    parser.add_argument("--instances", help="Actual IDs, e.g. 301,304 (default: all five archived cases)")
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU for the evaluator and workspace")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Physical diagnostic only; never used as a scored run")
    args = parser.parse_args()
    task = runner.load_task(args.task)
    config = runner.read_config(args.config)
    if config["unmask_evaluator_cuda"]:
        raise ValueError("Single-GPU evaluation requires unmask_evaluator_cuda=false")
    if config["session_timeout_s"] != 0:
        raise ValueError("This evaluation has no episode wall-clock cap")
    ports = runner.listener_ports(task, config)
    token = f"workspace-{task['task']}-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = args.run_dir.resolve() if args.run_dir else V2 / "behavior_evaluator/runs" / token
    first = str(task["cases"][0]["instance_id"])
    plan = {"run_id": token, "run_dir": str(path), "port": ports["http"], "policy_port": ports["policy"],
        "gate_port": ports["gate"], "gpu": args.gpu, "harness": "workspace", "model": config["model"],
        "model_url": config["model_url"], "diagnostic_only": bool(args.instances or args.smoke), "write_video": False,
        "smoke": args.smoke,
        "task_config": task, "cases": runner.select_cases(task, first if args.smoke else args.instances),
        "architecture": "agent Python -> atomic action driver -> evaluator; separate read-only monitor"}
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    if path.exists():
        raise FileExistsError("Use a fresh run directory; evidence is never overwritten")
    runner.preflight(config, "claude_code")
    from roboharness.assets import prepare_robot_asset
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
    raise SystemExit(0 if complete else 2)


if __name__ == "__main__":
    main()
