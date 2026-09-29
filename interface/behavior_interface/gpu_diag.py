"""Small GPU diagnostics used around high-risk simulation / cuRobo paths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, Optional


# Physical GPU ownership for every managed interface family.  CUDA-visible
# ordinals are deliberately not used here: launchers mask one physical card,
# so all in-process CUDA consumers use local ``cuda:0`` instead.
FIXED_GPU_BY_PORT = {
    15050: "0",  # protected primary simulator
    15051: "0",
    15052: "1",
    15053: "2",
    15054: "3",
    15055: "4",
    15056: "5",
    # 官方评测口：task_id = port-15060；012→GPU0，345→GPU1，678→GPU2，9→GPU3。
    15060: "0",
    15061: "0",
    15062: "0",
    15063: "1",
    15064: "1",
    15065: "1",
    15066: "2",
    15067: "2",
    15068: "2",
    15069: "3",
    # Training interfaces: four bounded instances per GPU2-5.
    5019: "2",
    5020: "2",
    5021: "2",
    5022: "2",
    5023: "3",
    5024: "3",
    5025: "3",
    5026: "3",
    5027: "4",
    5028: "4",
    5029: "4",
    5030: "4",
    5031: "5",
    5032: "5",
    5033: "5",
    5034: "5",
}


def fixed_gpu_for_port(port: int) -> Optional[str]:
    """Return the physical owner for a managed port, if it has one."""

    try:
        if os.environ.get('ROBOHARNESS_HTTP_PORT') == str(int(port)):
            gpu = os.environ.get('ROBOHARNESS_GPU', '')
            if not gpu.isdigit():
                raise ValueError('ROBOHARNESS_GPU must be a nonnegative physical GPU index')
            return gpu
        return FIXED_GPU_BY_PORT.get(int(port))
    except (TypeError, ValueError):
        return None


def enforce_fixed_gpu_environment(port: int) -> Optional[str]:
    """Apply and validate the one-card environment for a managed port.

    This helper is intentionally independent of torch/Omni imports so every
    Python entrypoint can enforce ownership before creating a CUDA context.
    It raises ``ValueError`` for contradictory caller-provided variables.
    """

    gpu = fixed_gpu_for_port(port)
    if gpu is None:
        return None
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "BEHAVIOR_INTERFACE_PHYSICAL_GPU",
        "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU",
        "IK_FILTER_CUDA_VISIBLE_DEVICES",
        "BEHAVIOR_RTABMAP_CUDA_DEVICE",
        "BEHAVIOR_RTABMAP_RETRIEVAL_CUDA_DEVICE",
    ):
        value = os.environ.get(name, "").strip()
        if value and value != gpu:
            raise ValueError(
                f"port {int(port)} requires {name}={gpu}, got {value!r}"
            )
        os.environ[name] = gpu
    local_gpu = os.environ.get("OMNIGIBSON_GPU_ID", "").strip()
    if local_gpu and local_gpu != "0":
        raise ValueError(
            f"port {int(port)} requires local OMNIGIBSON_GPU_ID=0, got {local_gpu!r}"
        )
    os.environ["OMNIGIBSON_GPU_ID"] = "0"
    for name in (
        "BEHAVIOR_SLAM_CUDA_DEVICE",
        "BEHAVIOR_SLAM_GEOMETRY_DEVICE",
        "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE",
    ):
        value = os.environ.get(name, "").strip()
        if value and value != "cuda:0":
            raise ValueError(
                f"port {int(port)} requires {name}=cuda:0, got {value!r}"
            )
        os.environ[name] = "cuda:0"
    return gpu


def apply_requested_cpu_affinity() -> Optional[list[int]]:
    """Apply an explicit interface/evaluator CPU set before heavy work starts."""

    raw = (
        os.environ.get("BEHAVIOR_EVAL_TEST_CPUSET", "").strip()
        or os.environ.get("INTERFACE_CPUSET", "").strip()
        or os.environ.get("TARGET_CPUSET", "").strip()
    )
    if not raw:
        return None
    cpus: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"invalid CPU set {raw!r}")
        if "-" in token:
            bounds = token.split("-", 1)
            if len(bounds) != 2 or not all(item.isdigit() for item in bounds):
                raise ValueError(f"invalid CPU range {token!r}")
            lo, hi = (int(item) for item in bounds)
            if lo > hi:
                raise ValueError(f"invalid CPU range {token!r}")
            cpus.update(range(lo, hi + 1))
        elif token.isdigit():
            cpus.add(int(token))
        else:
            raise ValueError(f"invalid CPU set token {token!r}")
    if not cpus:
        raise ValueError(f"CPU set {raw!r} is empty")
    setter = getattr(os, "sched_setaffinity", None)
    if setter is None:
        raise RuntimeError("explicit CPU set requested but sched_setaffinity is unavailable")
    try:
        setter(0, cpus)
    except OSError as exc:
        raise ValueError(f"cannot apply CPU set {raw!r}: {exc}") from exc
    return sorted(cpus)


ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "OMNIGIBSON_GPU_ID",
    "IK_FILTER_CUDA_VISIBLE_DEVICES",
    "BEHAVIOR_INTERFACE_PHYSICAL_GPU",
    "BEHAVIOR_EVAL_TEST_PHYSICAL_GPU",
    "BEHAVIOR_RTABMAP_CUDA_DEVICE",
    "BEHAVIOR_RTABMAP_RETRIEVAL_CUDA_DEVICE",
    "BEHAVIOR_SLAM_CUDA_DEVICE",
    "BEHAVIOR_SLAM_GEOMETRY_DEVICE",
    "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE",
    "EEF_NEAR_SAM2_DEVICE",
    "BEHAVIOR_EVAL_TEST_CPUSET",
    "INTERFACE_CPUSET",
    "TARGET_CPUSET",
    "BEHAVIOR_KDTREE_WORKERS",
    "BEHAVIOR_RGBD_POINT_WORKERS",
    "BEHAVIOR_IK_CPU_THREADS",
    "BEHAVIOR_OPENCV_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
    "CARB_TASKING_THREADS",
    "OMNIGIBSON_APPDATA_PATH",
    "OMNIGIBSON_HEADLESS",
    "OMNIGIBSON_LAUNCH_FROM_HOST",
    "VK_ICD_FILENAMES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PORT",
    "BEHAVIOR_INTERFACE_ROLE",
    "INTERFACE_TOOL_VERSION",
    "INTERFACE_TASK_SWITCH_REEXEC",
    "INTERFACE_TASK_SWITCH_TARGET",
)


def resource_ownership_snapshot() -> Dict[str, Any]:
    """Return the process-local GPU/CPU ownership contract without CUDA calls.

    This deliberately reads only environment and scheduler state.  It is safe
    to expose from a health endpoint before a CUDA context exists and remains
    useful when the NVIDIA driver is unhealthy.  ``physical_gpu`` is the
    launcher-level owner, while ``local_device`` is the ordinal visible inside
    a single-card ``CUDA_VISIBLE_DEVICES`` mask.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    port_raw = (
        os.environ.get("PORT", "").strip()
        or os.environ.get("BEHAVIOR_EVAL_TEST_PORT", "").strip()
    )
    expected_physical = fixed_gpu_for_port(port_raw) if port_raw.isdigit() else None
    physical = (
        os.environ.get("BEHAVIOR_INTERFACE_PHYSICAL_GPU", "").strip()
        or os.environ.get("BEHAVIOR_EVAL_TEST_PHYSICAL_GPU", "").strip()
        or (visible if visible.isdigit() else "")
    )
    local = os.environ.get("OMNIGIBSON_GPU_ID", "").strip() or "0"
    errors: list[str] = []
    visible_entries = [item.strip() for item in visible.split(",") if item.strip()]
    if not visible:
        errors.append("CUDA_VISIBLE_DEVICES is unset; GPU visibility is unmasked")
    if visible and (
        len(visible_entries) != 1 or not visible_entries[0].isdigit()
    ):
        errors.append("CUDA_VISIBLE_DEVICES must contain one numeric GPU")
    if visible_entries and physical and physical not in visible_entries:
        errors.append("physical owner is absent from CUDA_VISIBLE_DEVICES")
    if expected_physical is not None and physical != expected_physical:
        errors.append(
            f"port {port_raw} requires physical GPU {expected_physical}, got {physical or '<unset>'}"
        )
    if visible_entries and local != "0":
        errors.append("OMNIGIBSON_GPU_ID must be local ordinal 0 under a mask")
    physical_child_vars = (
        "IK_FILTER_CUDA_VISIBLE_DEVICES",
        "BEHAVIOR_RTABMAP_CUDA_DEVICE",
        "BEHAVIOR_RTABMAP_RETRIEVAL_CUDA_DEVICE",
    )
    if physical and visible_entries:
        for name in physical_child_vars:
            value = os.environ.get(name, "").strip()
            if value and value != physical:
                errors.append(f"{name} does not match physical owner")
    local_cuda_vars = (
        "BEHAVIOR_SLAM_CUDA_DEVICE",
        "BEHAVIOR_SLAM_GEOMETRY_DEVICE",
        "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE",
    )
    if visible_entries:
        for name in local_cuda_vars:
            value = os.environ.get(name, "").strip()
            if value and value != "cuda:0":
                errors.append(f"{name} must be cuda:0 under a mask")
    try:
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = []
    return {
        "physical_gpu": physical,
        "port": port_raw,
        "expected_physical_gpu": expected_physical or "",
        "cuda_visible_devices": visible,
        "local_device": "cuda:0" if visible else f"cuda:{local} (unmasked)",
        "omnibus_local_gpu_id": local,
        "ownership_ok": not errors,
        "ownership_errors": errors,
        "ik_filter_visible_devices": os.environ.get(
            "IK_FILTER_CUDA_VISIBLE_DEVICES", ""
        ),
        "rtabmap_cuda_device": os.environ.get(
            "BEHAVIOR_RTABMAP_CUDA_DEVICE", ""
        ),
        "rtabmap_retrieval_cuda_device": os.environ.get(
            "BEHAVIOR_RTABMAP_RETRIEVAL_CUDA_DEVICE", ""
        ),
        "slam_device": os.environ.get("BEHAVIOR_SLAM_CUDA_DEVICE", ""),
        "slam_geometry_device": os.environ.get(
            "BEHAVIOR_SLAM_GEOMETRY_DEVICE", ""
        ),
        "occupancy_device": os.environ.get(
            "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE", ""
        ),
        "cpu_set": affinity,
        "requested_cpu_set": (
            os.environ.get("BEHAVIOR_EVAL_TEST_CPUSET", "").strip()
            or os.environ.get("INTERFACE_CPUSET", "").strip()
            or os.environ.get("TARGET_CPUSET", "").strip()
        ),
        "cpu_thread_limits": {
            key: os.environ.get(key, "")
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "TBB_NUM_THREADS",
                "CARB_TASKING_THREADS",
                "BEHAVIOR_KDTREE_WORKERS",
                "BEHAVIOR_RGBD_POINT_WORKERS",
                "BEHAVIOR_IK_CPU_THREADS",
                "BEHAVIOR_OPENCV_THREADS",
            )
            if key in os.environ
        },
    }


def _flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _enabled() -> bool:
    return _flag("BEHAVIOR_GPU_DIAG", True)


def _max_lines() -> int:
    try:
        return max(1, int(os.environ.get("BEHAVIOR_GPU_DIAG_MAX_LINES", "40")))
    except Exception:
        return 40


def _timeout_s() -> float:
    try:
        return max(0.2, float(os.environ.get("BEHAVIOR_GPU_DIAG_TIMEOUT_S", "2.0")))
    except Exception:
        return 2.0


def _json_default(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unrepr {type(value).__name__}>"


def _compact_json(payload: Dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _diag_file() -> Optional[str]:
    if not _flag("BEHAVIOR_GPU_DIAG_WRITE_FILE", True):
        return None
    return os.environ.get("BEHAVIOR_GPU_DIAG_FILE") or f"/tmp/behavior_gpu_diag_{os.getpid()}.jsonl"


def _write_jsonl(payload: Dict[str, Any]) -> None:
    path = _diag_file()
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(_compact_json(payload) + "\n")
    except Exception:
        pass


def _emit(log_fn: Optional[Callable[[str], None]], payload: Dict[str, Any]) -> None:
    if not _enabled():
        return
    payload = dict(payload)
    payload.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    payload.setdefault("ts_unix", round(time.time(), 3))
    line = "GPU_DIAG " + _compact_json(payload)
    try:
        if log_fn is print:
            print(line, flush=True)
        elif log_fn is not None:
            log_fn(line)
        else:
            print(line, flush=True)
    except Exception:
        try:
            print(line, flush=True)
        except Exception:
            pass
    _write_jsonl(payload)


def _env_snapshot() -> Dict[str, str]:
    return {key: os.environ[key] for key in ENV_KEYS if key in os.environ}


def _torch_snapshot() -> Optional[Dict[str, Any]]:
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return None
    try:
        cuda = torch_mod.cuda
        info: Dict[str, Any] = {
            "loaded": True,
            "is_available": bool(cuda.is_available()),
            "device_count": int(cuda.device_count()) if cuda.is_available() else 0,
        }
        if info["is_available"]:
            try:
                dev = int(cuda.current_device())
            except Exception:
                dev = 0
            info["current_device"] = dev
            try:
                info["device_name"] = str(cuda.get_device_name(dev))
            except Exception:
                pass
            try:
                free, total = cuda.mem_get_info(dev)
                info["mem_free_mib"] = round(float(free) / (1024 * 1024), 1)
                info["mem_total_mib"] = round(float(total) / (1024 * 1024), 1)
            except Exception as exc:
                info["mem_get_info_error"] = f"{type(exc).__name__}: {exc}"
            for name, fn in (
                ("allocated_mib", cuda.memory_allocated),
                ("reserved_mib", cuda.memory_reserved),
                ("max_allocated_mib", cuda.max_memory_allocated),
                ("max_reserved_mib", cuda.max_memory_reserved),
            ):
                try:
                    info[name] = round(float(fn(dev)) / (1024 * 1024), 1)
                except Exception:
                    pass
        return info
    except Exception as exc:
        return {"loaded": True, "error": f"{type(exc).__name__}: {exc}"}


def _run_nvidia_smi(args: Iterable[str]) -> Dict[str, Any]:
    cmd = ["nvidia-smi", *args]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_timeout_s(),
            check=False,
        )
    except FileNotFoundError:
        return {"cmd": cmd, "error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "error": f"timeout after {_timeout_s():.1f}s"}
    except Exception as exc:
        return {"cmd": cmd, "error": f"{type(exc).__name__}: {exc}"}

    rows = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
    stderr = (proc.stderr or "").strip()
    max_lines = _max_lines()
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "rows": rows[:max_lines],
        "truncated": len(rows) > max_lines,
        "stderr": stderr[:1000],
    }


def enable_faulthandler(log_fn: Optional[Callable[[str], None]] = None) -> None:
    if not _enabled():
        return
    try:
        import faulthandler

        faulthandler.enable(all_threads=True)
        _emit(log_fn, {"section": "faulthandler", "enabled": True, "pid": os.getpid()})
    except Exception as exc:
        _emit(log_fn, {"section": "faulthandler", "enabled": False, "error": str(exc), "pid": os.getpid()})


def log_gpu_diag(
    log_fn: Optional[Callable[[str], None]],
    tag: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
    include_nvidia: bool = True,
    include_torch: bool = True,
) -> None:
    if not _enabled():
        return
    base = {
        "tag": tag,
        "section": "process",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "argv": sys.argv,
        "env": _env_snapshot(),
    }
    if extra:
        base["extra"] = dict(extra)
    _emit(log_fn, base)

    if include_torch:
        torch_info = _torch_snapshot()
        if torch_info is not None:
            _emit(
                log_fn,
                {
                    "tag": tag,
                    "section": "torch",
                    "pid": os.getpid(),
                    **torch_info,
                },
            )

    if include_nvidia and _flag("BEHAVIOR_GPU_DIAG_NVIDIA_SMI", True):
        _emit(
            log_fn,
            {
                "tag": tag,
                "section": "nvidia_compute_apps",
                "pid": os.getpid(),
                **_run_nvidia_smi(
                    (
                        "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
                        "--format=csv,noheader,nounits",
                    )
                ),
            },
        )
        _emit(
            log_fn,
            {
                "tag": tag,
                "section": "nvidia_gpus",
                "pid": os.getpid(),
                **_run_nvidia_smi(
                    (
                        "--query-gpu=index,pci.bus_id,memory.used,memory.total,utilization.gpu,pstate",
                        "--format=csv,noheader,nounits",
                    )
                ),
            },
        )
