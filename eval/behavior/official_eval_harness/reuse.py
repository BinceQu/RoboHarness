"""检测已有官方栈是否可复用，避免测试脚本重复冷启动。"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

LIVE_MARKER_DIR = Path("/tmp")
_SS_PID_RE = re.compile(r"pid=(\d+)")


def live_marker_path(http_port: int) -> Path:
    return LIVE_MARKER_DIR / f"behavior_eval_harness_live_p{int(http_port)}.json"


def parse_slots(raw: str | None) -> list[int]:
    if raw is None or not str(raw).strip():
        return []
    return [int(part) for part in str(raw).replace(",", " ").split() if part.strip()]


def read_proc_environ(pid: int) -> dict[str, str]:
    path = Path(f"/proc/{int(pid)}/environ")
    if not path.is_file():
        return {}
    try:
        raw = path.read_bytes()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return env


def read_proc_cmdline(pid: int) -> str:
    path = Path(f"/proc/{int(pid)}/cmdline")
    if not path.is_file():
        return ""
    try:
        return path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def find_listen_pid(port: int) -> int | None:
    try:
        proc = subprocess.run(
            ["ss", "-ltnp"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    needle = re.compile(rf":{int(port)}\b")
    for line in proc.stdout.splitlines():
        if not needle.search(line):
            continue
        match = _SS_PID_RE.search(line)
        if match:
            return int(match.group(1))
    return None


def _cmdline_looks_like_evaluator(cmdline: str) -> bool:
    text = cmdline.lower()
    return any(
        token in text
        for token in (
            "launch_official_evaluator",
            "official_evaluator_entrypoint",
            "omnigibson/eval/eval.py",
            "omnigibson.eval.eval",
        )
    )


def _cmdline_looks_like_interface(cmdline: str) -> bool:
    text = cmdline.lower()
    return any(
        token in text
        for token in (
            "launch_official_policy_interface",
            "official_policy_interface",
        )
    )


def find_evaluator_pid(http_port: int) -> int | None:
    wanted = str(int(http_port))
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        env = read_proc_environ(pid)
        if env.get("BEHAVIOR_EVAL_TEST_PORT") != wanted:
            continue
        cmdline = read_proc_cmdline(pid)
        if _cmdline_looks_like_evaluator(cmdline):
            return pid
        if _cmdline_looks_like_interface(cmdline):
            continue
        # bash 包装器：环境对了，命令行是 launch 脚本。
        if "launch_official_evaluator_v391.sh" in cmdline:
            return pid
    return None


def stack_fingerprint(plan: Any, worker: Any) -> dict[str, Any]:
    return {
        "http_port": int(worker.http_port),
        "gpu": int(worker.gpu),
        "task": str(plan.task.name),
        "task_index": int(plan.task.index),
        "timeout_steps": int(plan.task.timeout_steps),
        "slots": [int(slot) for slot in worker.slots],
        "instance_ids": [int(item) for item in worker.instance_ids],
        "idle_gate": bool(plan.idle_gate),
        "spatial_map": bool(plan.spatial_map),
        "annotator": str(plan.annotator),
        "policy_port": int(worker.policy_port),
        "gate_port": int(worker.gate_port),
        "num_rollouts": int(getattr(plan, "num_rollouts", 1)),
    }


def reuse_mismatch(expected: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """比较计划与现场；空列表表示可以复用。"""

    reasons: list[str] = []
    checks = (
        ("http_port", "HTTP 口"),
        ("gpu", "GPU"),
        ("task", "任务"),
        ("timeout_steps", "超时帧"),
        ("slots", "instance 列表"),
        ("idle_gate", "idle-gate"),
        ("spatial_map", "小地图"),
        ("policy_port", "policy 口"),
        ("num_rollouts", "num-rollouts"),
    )
    for key, label in checks:
        if key not in live or live[key] is None:
            # 现场还读不到的字段不挡复用，靠 pid / health 再收紧。
            continue
        if key == "slots":
            live_slots = {int(slot) for slot in live[key]}
            planned_slots = {int(slot) for slot in expected[key]}
            if planned_slots and planned_slots.issubset(live_slots):
                continue
        if live[key] != expected[key]:
            reasons.append(f"{label} 不匹配: 计划={expected[key]} 现场={live[key]}")
    if expected.get("idle_gate") and live.get("gate_port") != expected.get("gate_port"):
        reasons.append(
            f"idle-gate 口不匹配: 计划={expected.get('gate_port')} 现场={live.get('gate_port')}"
        )
    if expected.get("idle_gate") and live.get("eval_policy_port") not in {
        None,
        expected.get("gate_port"),
    }:
        reasons.append(
            f"evaluator 没连 gate: 计划={expected.get('gate_port')} "
            f"现场={live.get('eval_policy_port')}"
        )
    if not expected.get("idle_gate") and live.get("eval_policy_port") not in {
        None,
        expected.get("policy_port"),
    }:
        reasons.append(
            f"evaluator 没直连 policy: 计划={expected.get('policy_port')} "
            f"现场={live.get('eval_policy_port')}"
        )
    return reasons


def live_from_environ(
    env: dict[str, str],
    *,
    http_port: int,
    policy_port: int,
    gate_port: int,
) -> dict[str, Any]:
    gpu_raw = (
        env.get("BEHAVIOR_EVAL_TEST_PHYSICAL_GPU")
        or env.get("BEHAVIOR_INTERFACE_PHYSICAL_GPU")
        or env.get("CUDA_VISIBLE_DEVICES")
        or ""
    ).strip()
    try:
        gpu = int(gpu_raw.split(",")[0])
    except (TypeError, ValueError):
        gpu = None
    slots = parse_slots(env.get("BEHAVIOR_EVAL_TEST_INSTANCE_INDICES"))
    if not slots:
        index = env.get("BEHAVIOR_EVAL_TEST_INSTANCE_INDEX", "").strip()
        if index.isdigit():
            slots = [int(index)]
    max_steps_raw = env.get("BEHAVIOR_EVAL_TEST_MAX_STEPS", "").strip()
    try:
        timeout_steps = int(max_steps_raw) if max_steps_raw else None
    except ValueError:
        timeout_steps = None
    eval_policy_raw = env.get("BEHAVIOR_EVAL_TEST_POLICY_PORT", "").strip()
    try:
        eval_policy_port = int(eval_policy_raw) if eval_policy_raw else None
    except ValueError:
        eval_policy_port = None
    idle_gate = eval_policy_port == int(gate_port) if eval_policy_port is not None else None
    rollouts_raw = env.get("BEHAVIOR_EVAL_TEST_NUM_ROLLOUTS", "1").strip()
    try:
        num_rollouts = int(rollouts_raw)
    except ValueError:
        num_rollouts = None
    return {
        "http_port": int(http_port),
        "gpu": gpu,
        "task": (env.get("TASK") or "").strip(),
        "timeout_steps": timeout_steps,
        "slots": slots,
        "idle_gate": idle_gate,
        "spatial_map": (env.get("BEHAVIOR_SPATIAL_MAP") or "0").strip() == "1",
        "annotator": (env.get("BEHAVIOR_EVAL_TEST_RGBD_ANNOTATOR_DEVICE") or "").strip(),
        "policy_port": int(policy_port),
        "gate_port": int(gate_port),
        "eval_policy_port": eval_policy_port,
        "num_rollouts": num_rollouts,
        "output_dir": (env.get("BEHAVIOR_EVAL_TEST_OUTPUT_DIR") or "").strip(),
        "runtime_dir": (env.get("BEHAVIOR_EVAL_TEST_RUNTIME_DIR") or "").strip(),
    }


def classify_live_stack(
    expected: dict[str, Any],
    live: dict[str, Any] | None,
    *,
    health_ready: bool,
    pids_alive: bool,
    force_restart: bool = False,
) -> str:
    """返回 ready / starting / mismatch / absent。"""

    if force_restart:
        return "restart" if live else "absent"
    if not live or not pids_alive:
        return "absent"
    if reuse_mismatch(expected, live):
        return "mismatch"
    if health_ready:
        return "ready"
    return "starting"


def read_live_marker(http_port: int) -> dict[str, Any] | None:
    path = live_marker_path(http_port)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_live_marker(payload: dict[str, Any]) -> Path:
    path = live_marker_path(int(payload["http_port"]))
    body = dict(payload)
    body["written_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def clear_live_marker(http_port: int) -> None:
    path = live_marker_path(http_port)
    if path.exists():
        path.unlink()
