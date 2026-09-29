"""测试页右上角：instance 列表 + 停本口 agent session。

停 session 只杀 Claude/agent，不拆 interface / evaluator / idle-gate。
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

from official_eval_harness.catalog import instance_id_for_slot, result_json_name
from official_eval_harness.reuse import read_live_marker, read_proc_cmdline, read_proc_environ


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_RUNS_ROOT = REPO_ROOT / "experimental_gpu3_interface" / "eval_runs"

_AGENT_CMDLINE = (
    "embodied_claude_code",
    "embodied-claude",
    "scripts/run --port",
    "./scripts/run",
)
_SKIP_CMDLINE = (
    "official_policy_interface",
    "official_evaluator",
    "official_idle_step_gate",
    "launch_official_policy_interface",
    "launch_official_evaluator",
    "behavior_interface.server",
    "operator_controls",
)


def resolve_http_port(server: Any | None = None) -> int | None:
    for raw in (
        getattr(server, "web_port", None),
        os.environ.get("BEHAVIOR_EVAL_TEST_PORT"),
        os.environ.get("BEHAVIOR_PORT"),
    ):
        text = str(raw or "").strip()
        if text.isdigit():
            return int(text)
    return None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    stat = Path(f"/proc/{int(pid)}/stat")
    if not stat.is_file():
        return False
    try:
        text = stat.read_text(encoding="utf-8")
        state = text.rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError, ValueError):
        return False
    return state not in {"Z"}


def _read_pid_file(path: Path) -> int | None:
    if not path.is_file():
        return None
    raw = "".join(ch for ch in path.read_text(encoding="utf-8") if ch.isdigit())
    return int(raw) if raw else None


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _discover_instance_ids(server: Any | None, port: int | None) -> tuple[list[int], list[int]]:
    if port:
        marker = read_live_marker(int(port)) or {}
        raw_ids = marker.get("instance_ids") or []
        raw_slots = marker.get("slots") or []
        slots = [int(item) for item in raw_slots]
        ids = [int(item) for item in raw_ids]
        if not ids and slots:
            ids = [instance_id_for_slot(slot) for slot in slots]
        if ids:
            return ids, slots

    getter = getattr(server, "_eval_instance_ids", None)
    if callable(getter):
        try:
            ids = [int(item) for item in getter()]
        except Exception:
            ids = []
        if ids:
            return ids, []

    return list(range(301, 311)), list(range(10))


def infer_current_instance_id(
    port: int | None,
    instance_ids: list[int],
    *,
    operator_status: dict[str, Any] | None = None,
    marker: dict[str, Any] | None = None,
) -> int | None:
    """官方口 interface 自己不持有仿真，从侧信道 / 已写出的 JSON 推断当前 instance。"""
    status = operator_status if operator_status is not None else _operator_status(port)
    if status.get("current_instance_id") is not None:
        try:
            return int(status["current_instance_id"])
        except (TypeError, ValueError):
            pass
    if port and marker is None:
        marker = read_live_marker(int(port))
    marker = marker or {}
    ids = [int(item) for item in instance_ids]
    task = str(marker.get("task") or "").strip()
    output_dir = str(marker.get("output_dir") or "").strip()
    if task and output_dir and ids:
        json_dir = Path(output_dir) / "json"
        for iid in ids:
            path = json_dir / result_json_name(task, iid)
            if not (path.is_file() and path.stat().st_size > 20):
                return int(iid)
        return int(ids[-1])
    if ids:
        return int(ids[0])
    return None


def _operator_status(port: int | None) -> dict[str, Any]:
    if not port:
        return {}
    try:
        from behavior_interface_eval_test.operator_scene_control import read_status

        return read_status(int(port))
    except Exception:
        return {}


def _claude_runtime_dirs(port: int) -> list[Path]:
    dirs: list[Path] = []
    marker = read_live_marker(int(port)) or {}
    run_dir = str(marker.get("run_dir") or "").strip()
    if run_dir:
        dirs.append(Path(run_dir) / f"p{int(port)}" / "runtime")
    runtime_dir = str(marker.get("runtime_dir") or "").strip()
    if runtime_dir:
        dirs.append(Path(runtime_dir))
    # 只认当前 live marker 的 runtime。扫全部 eval_runs 会把旧局死 pid
    # 当成 session，STOP 还可能误杀复用了那个 pid 的进程。
    if not dirs and EVAL_RUNS_ROOT.is_dir():
        dirs.extend(sorted(EVAL_RUNS_ROOT.glob(f"*/p{int(port)}/runtime")))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in dirs:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _looks_like_agent(cmdline: str, env: dict[str, str], port: int) -> bool:
    if any(marker in cmdline for marker in _SKIP_CMDLINE):
        return False
    if str(env.get("BEHAVIOR_PORT") or "").strip() != str(port):
        return False
    if not str(env.get("BEHAVIOR_SESSION_ID") or "").strip():
        return False
    lowered = cmdline.lower()
    if any(marker in cmdline for marker in _AGENT_CMDLINE):
        return True
    return "claude" in lowered


def _scan_agent_pids(port: int) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    my_pid = os.getpid()
    try:
        my_pgid = os.getpgid(my_pid)
    except OSError:
        my_pgid = None
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == my_pid:
            continue
        try:
            pgid = os.getpgid(pid)
        except OSError:
            continue
        if my_pgid is not None and pgid == my_pgid:
            continue
        env = read_proc_environ(pid)
        cmdline = read_proc_cmdline(pid)
        if not _looks_like_agent(cmdline, env, port):
            continue
        found.append(
            {
                "pid": pid,
                "pgid": pgid,
                "session_id": str(env.get("BEHAVIOR_SESSION_ID") or "").strip(),
                "source": "proc",
            }
        )
    return found


def agent_session_status(port: int | None) -> dict[str, Any]:
    if not port:
        return {"running": False, "sessions": []}
    sessions: list[dict[str, Any]] = []
    seen_pids: set[int] = set()
    for runtime in _claude_runtime_dirs(int(port)):
        pid = _read_pid_file(runtime / "claude_session.pid")
        session_id = _read_text(runtime / "claude_session.id")
        if not pid:
            continue
        sessions.append(
            {
                "pid": int(pid),
                "session_id": session_id,
                "alive": _pid_alive(pid),
                "source": str(runtime / "claude_session.pid"),
            }
        )
        seen_pids.add(int(pid))
    for item in _scan_agent_pids(int(port)):
        if int(item["pid"]) in seen_pids:
            continue
        item["alive"] = _pid_alive(item["pid"])
        sessions.append(item)
    live = [item for item in sessions if item.get("alive")]
    current = live[0] if live else None
    return {
        "running": bool(live),
        "pid": None if current is None else current.get("pid"),
        "session_id": "" if current is None else str(current.get("session_id") or ""),
        "sessions": sessions,
    }


def _stop_pid(pid: int) -> dict[str, Any]:
    info = {"pid": int(pid), "stopped": False, "error": ""}
    if not _pid_alive(pid):
        info["stopped"] = True
        return info
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except OSError:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError as exc:
            info["error"] = str(exc)
            return info
    deadline = time.time() + 8
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.2)
    if _pid_alive(pid):
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except OSError:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except OSError as exc:
                info["error"] = str(exc)
                return info
    info["stopped"] = not _pid_alive(pid)
    return info


def stop_agent_sessions(port: int | None) -> dict[str, Any]:
    """停本口 Claude/agent，不动官方仿真栈。"""
    if not port:
        return {"ok": False, "error": "unknown http port", "stopped": []}
    before = agent_session_status(int(port))
    targets: dict[int, dict[str, Any]] = {}
    for item in before.get("sessions") or []:
        pid = item.get("pid")
        if not pid or not item.get("alive"):
            continue
        targets[int(pid)] = item
    stopped: list[dict[str, Any]] = []
    for pid, item in targets.items():
        result = _stop_pid(int(pid))
        result["session_id"] = item.get("session_id") or ""
        result["source"] = item.get("source") or ""
        stopped.append(result)
        runtime = Path(str(item.get("source") or ""))
        if runtime.name == "claude_session.pid" and result.get("stopped"):
            try:
                runtime.unlink()
            except OSError:
                pass
    after = agent_session_status(int(port))
    return {
        "ok": True,
        "stopped": stopped,
        "still_running": bool(after.get("running")),
        "agent_session": after,
    }


def eval_control_snapshot(server: Any | None = None, *, official: bool | None = None) -> dict[str, Any]:
    port = resolve_http_port(server)
    current = getattr(server, "current_instance_id", None) if server is not None else None
    op_status = _operator_status(port)
    instance_ids, slots = _discover_instance_ids(server, port)
    inferred = infer_current_instance_id(
        port,
        instance_ids,
        operator_status=op_status,
        marker=read_live_marker(int(port)) if port else None,
    )
    if inferred is not None:
        current = inferred
    if current is not None and int(current) not in {int(item) for item in instance_ids}:
        instance_ids = [int(current), *instance_ids]
    if official is None:
        mode = ""
        getter = getattr(server, "snapshot_state", None)
        if callable(getter):
            try:
                mode = str((getter() or {}).get("mode") or "")
            except Exception:
                mode = ""
        official = "official" in mode or bool(op_status.get("listener"))
    return {
        "ok": True,
        "port": port,
        "official": bool(official),
        "reset_owner": "evaluator" if official else "interface",
        "current_instance_id": None if current is None else int(current),
        "instance_ids": [int(item) for item in instance_ids],
        "slots": [int(item) for item in slots],
        "operator_reset": op_status,
        "agent_session": agent_session_status(port),
    }
