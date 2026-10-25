"""BehaviorInterface：主线程跑仿真 + skill 执行；Flask 在另一个线程暴露 web。

支持两种运行模式：
  * dry-run: 不启动 OmniGibson，用合成画面 + mock 底盘动力学。
             用来快速调通 web / cli / skill 注册链路。
  * real:    启动真实仿真（R1Pro + Behavior Challenge 场景），
             需要 GPU 与 OmniGibson 数据集。
"""

from __future__ import annotations

import copy
import gc
import importlib
import math
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Some top-level interface modules import OmniGibson as a side effect. Configure
# process-owned temporary storage before any third-party or interface imports so
# OG's decrypted USD directory is recoverable after SIGKILL or a crash.
from .runtime_tmp import (
    configure_process_runtime_tmp as _configure_runtime_tmp,
    process_runtime_port_hint as _runtime_port_hint,
)

_configure_runtime_tmp(_runtime_port_hint())
del _configure_runtime_tmp, _runtime_port_hint

import cv2
import numpy as np

# OpenCV's implicit pool otherwise defaults to all visible CPUs.  Interface
# image conversion runs beside Kit/PhysX, so keep it bounded; operators can
# raise the cap for an isolated benchmark without changing the code path.
try:
    _opencv_threads = int(os.environ.get("BEHAVIOR_OPENCV_THREADS", "1"))
except (TypeError, ValueError):
    _opencv_threads = 1
cv2.setNumThreads(max(1, min(_opencv_threads, 16)))

from . import agent_runs
from .camera_frames import (
    CameraFeedHealth,
    convert_rgb_frame,
    frame_description,
    rgb_array,
)
from .challenge_dataset import (
    HIDDEN_TEST_INSTANCE_IDS,
    PUBLIC_TEST_INSTANCE_IDS,
    build_stable_backed_task_scene as challenge_build_stable_backed_task_scene,
    eval_instance_ids as challenge_eval_instance_ids,
    mode_dir as challenge_mode_dir,
    mode_for_instance as challenge_mode_for_instance,
    template_path as challenge_template_path,
    validate_task_assets as validate_challenge_task_assets,
)
from .challenge_robot_config import (
    DEFAULT_CHALLENGE_ROBOT_CONFIG,
    build_interface_robot_config,
    load_challenge_robot_config,
    resolve_challenge_robot_config_path,
)
from .robot_variant import (
    normalize_robot_dof,
    robot_config_path_for_dof,
    robot_type_for_dof,
)
from .camera_render_control import (
    rebuild_robot_camera_render_products,
    set_robot_camera_render_updates,
)
from .overlay import (
    draw_axes_overlay,
    draw_topbar,
    label_image,
    make_placeholder,
)
from .skills import SKILL_REGISTRY, load_all_skills, list_skills
from .world_api import WorldAPI
from .gpu_diag import (
    apply_requested_cpu_affinity,
    enable_faulthandler,
    enforce_fixed_gpu_environment,
    FIXED_GPU_BY_PORT,
    fixed_gpu_for_port,
    log_gpu_diag,
    resource_ownership_snapshot,
)


def _env_truthy(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none"}


def _configure_omniverse_thread_limit() -> int | None:
    """Apply the interface's Kit tasking limit before OG starts SimulationApp.

    OmniGibson's simulator clears ``sys.argv`` before constructing
    ``SimulationApp``.  Consequently, a ``--/plugins/carb.tasking...`` option
    supplied to the outer launcher is discarded.  Isaac's supported
    ``limit_cpu_threads`` launcher setting is the reliable control point and
    also sets PXR/OPENBLAS and omni.tbb limits consistently.
    """
    raw = os.environ.get("CARB_TASKING_THREADS", "16").strip()
    if raw == "0":
        return None
    try:
        limit = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("CARB_TASKING_THREADS must be an integer in 1..64 or 0") from exc
    if not 1 <= limit <= 64:
        raise ValueError("CARB_TASKING_THREADS must be in 1..64 or 0")

    import omnigibson.lazy as lazy

    app_cls = lazy.isaacsim.SimulationApp
    config = dict(app_cls.DEFAULT_LAUNCHER_CONFIG)
    config["limit_cpu_threads"] = limit
    app_cls.DEFAULT_LAUNCHER_CONFIG = config
    return limit


def _env_float(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    """读浮点环境变量，非法值回落到 default。"""
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _skip_headless_rtx_warmup() -> bool:
    return _env_truthy("OMNIGIBSON_HEADLESS", False) and _env_truthy(
        "OMNIGIBSON_SKIP_HEADLESS_RTX_WARMUP", False
    )


def _skip_headless_obs_prime() -> bool:
    return _env_truthy("OMNIGIBSON_HEADLESS", False) and _env_truthy(
        "OMNIGIBSON_SKIP_HEADLESS_OBS_PRIME", False
    )


def _task_goals_only_observation_mode() -> bool:
    from .memory import task_goals_only_memory_enabled

    return task_goals_only_memory_enabled()


def _sanitize_model_payload_for_mode(
    payload: Any,
    *,
    task_name: str,
    goals: Dict[str, Any],
) -> Any:
    if not _task_goals_only_observation_mode():
        return payload
    from .memory import sanitize_model_payload

    return sanitize_model_payload(payload, task_name=task_name, goals=goals)


_CAMERA_CAPTURE_SKILLS = frozenset(
    {
        "capture",
        "capture_left_wrist_camera",
        "capture_right_wrist_camera",
    }
)


def _process_memory_mib() -> Dict[str, float]:
    values: Dict[str, float] = {}
    try:
        with open("/proc/self/smaps_rollup", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                key, sep, rest = line.partition(":")
                if not sep or key not in {"Rss", "Pss", "Pss_Anon", "Private_Dirty", "Swap"}:
                    continue
                parts = rest.split()
                if parts:
                    values[key.lower()] = float(parts[0]) / 1024.0
    except OSError:
        pass
    return values


def _release_native_allocator_caches() -> Dict[str, Any]:
    """Release free allocator caches without touching live simulation objects."""
    import ctypes

    result: Dict[str, Any] = {"python_gc_collected": int(gc.collect())}
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        result["glibc_malloc_trim"] = int(malloc_trim(0))
    except Exception as e:
        result["glibc_error"] = f"{type(e).__name__}: {e}"
    return result


def _trim_loaded_warp_mempools() -> Dict[str, Any]:
    """Release free blocks from this interface's Warp CUDA mempool only.

    Fixed launchers normally mask a process to one card, but compatibility
    callers can still run unmasked.  Iterating every ``runtime.cuda_devices``
    in that case synchronizes and trims allocator state owned by other
    interfaces.  Resolve the owner to Warp's process-local ordinal and skip
    every other device.
    """
    result: Dict[str, Any] = {
        "loaded": False,
        "initialized": False,
        "trimmed_devices": [],
        "errors": [],
    }
    warp = sys.modules.get("warp")
    if warp is None:
        return result
    result["loaded"] = True

    context = getattr(warp, "context", None)
    runtime = getattr(context, "runtime", None)
    if runtime is None:
        return result
    result["initialized"] = True

    get_threshold = getattr(warp, "get_mempool_release_threshold", None)
    set_threshold = getattr(warp, "set_mempool_release_threshold", None)
    synchronize_device = getattr(warp, "synchronize_device", None)
    if not all(callable(fn) for fn in (get_threshold, set_threshold, synchronize_device)):
        result["errors"].append("Warp mempool APIs unavailable")
        return result

    devices = list(getattr(runtime, "cuda_devices", ()))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    physical = os.environ.get("BEHAVIOR_INTERFACE_PHYSICAL_GPU", "").strip()
    owned_ordinal = 0
    if visible:
        entries = [item.strip() for item in visible.split(",") if item.strip()]
        if physical in entries:
            owned_ordinal = entries.index(physical)
    elif physical.isdigit():
        owned_ordinal = int(physical)

    for index, device in enumerate(devices):
        try:
            device_ordinal = int(getattr(device, "ordinal", index))
        except (TypeError, ValueError):
            device_ordinal = index
        if device_ordinal != owned_ordinal:
            continue
        if (
            not getattr(device, "has_context", False)
            or not getattr(device, "is_mempool_supported", False)
        ):
            continue
        label = str(device)
        old_threshold = None
        threshold_changed = False
        try:
            old_threshold = int(get_threshold(device))
            set_threshold(device, 0)
            threshold_changed = True
            synchronize_device(device)
            result["trimmed_devices"].append(label)
        except Exception as e:
            result["errors"].append(f"{label}: {type(e).__name__}: {e}")
        finally:
            if threshold_changed:
                try:
                    set_threshold(device, old_threshold)
                except Exception as e:
                    result["errors"].append(
                        f"{label} restore: {type(e).__name__}: {e}"
                    )
    return result


_PREDICATE_LABELS = {
    "inside": "{a} 在 {b} 内",
    "ontop": "{a} 在 {b} 上",
    "under": "{a} 在 {b} 下",
    "nextto": "{a} 靠近 {b}",
    "touching": "{a} 接触 {b}",
    "covered": "{a} 被 {b} 覆盖",
    "contains": "{a} 含有 {b}",
    "filled": "{a} 装有 {b}",
    "attached": "{a} 连接到 {b}",
    "toggled_on": "{a} 已开启",
    "open": "{a} 已打开",
    "cooked": "{a} 已烹饪",
    "frozen": "{a} 已冷冻",
    "on_fire": "{a} 着火",
    "real": "{a} 已生成",
    "hot": "{a} 已加热",
}


def _bddl_term_label(term: Any, variables: Optional[Dict[str, str]] = None) -> str:
    """Return a compact human label for a BDDL object / variable term."""
    s = str(term).strip()
    key = s.lstrip("?")
    if variables and key in variables:
        return variables[key]

    idx = ""
    base = key
    if "_" in key:
        maybe_base, maybe_idx = key.rsplit("_", 1)
        if maybe_idx.isdigit():
            base, idx = maybe_base, maybe_idx

    # Drop WordNet suffixes like .n.01 and simplify common synthetic separators.
    base = base.split(".")[0]
    base = base.replace("__of__", " of ")
    base = base.replace("__", " ")
    base = base.replace("_", " ")
    base = " ".join(base.split())
    if base.startswith("electric refrigerator"):
        base = base.replace("electric refrigerator", "refrigerator", 1)
    if idx:
        return f"{base} {idx}"
    return base


def _shorten_label(text: str, max_len: int = 96) -> str:
    text = " ".join(str(text).replace("\n", " ").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


_LEGACY_CACHED_SCOPE_ALIASES = {
    ("cook_a_brisket", "countertop.n.01_1"): ("cabinet.n.01_1",),
}


def _bddl_problem_path(task_name: str, definition_id: int = 0) -> str:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    return os.path.join(
        root,
        "bddl3",
        "bddl",
        "activity_definitions",
        str(task_name),
        f"problem{int(definition_id)}.bddl",
    )


def _bddl_object_instances_for_task(task_name: str, definition_id: int = 0) -> list[str]:
    """Read object instance names from the local BDDL problem file."""
    import re

    path = _bddl_problem_path(task_name, definition_id)
    instances: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            in_objects = False
            for line in f:
                if "(:objects" in line:
                    in_objects = True
                    continue
                match = re.match(r"\s*([\w\.]+_\d+)\s+-\s+[\w\.]+", line)
                if in_objects and match:
                    instances.append(match.group(1))
                    continue
                if in_objects and ")" in line:
                    break
    except OSError:
        return []
    return instances


def _bddl_has_wildcard_objects(task_name: str, definition_id: int = 0) -> bool:
    """Return True when the BDDL object list contains wildcard instances."""
    import re

    path = _bddl_problem_path(task_name, definition_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            in_objects = False
            for line in f:
                if "(:objects" in line:
                    in_objects = True
                    continue
                if in_objects and re.search(r"[\w\.]+_\*", line):
                    return True
                if in_objects and ")" in line:
                    break
    except OSError:
        return False
    return False


def _legacy_template_scope_compatible(
    task_name: str,
    definition_id: int,
    cached_scene_path: str,
) -> tuple[bool, str]:
    """Return whether a cached 2025 template can satisfy the current BDDL scope."""
    import json

    if _bddl_has_wildcard_objects(task_name, definition_id):
        return False, "BDDL declares wildcard object instances; avoid stale legacy cached scope"
    bddl_instances = _bddl_object_instances_for_task(task_name, definition_id)
    if not bddl_instances:
        return False, "could not read BDDL object instances"
    try:
        with open(cached_scene_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        return False, f"could not read cached template metadata: {e}"
    inst_to_name = (
        ((payload.get("metadata") or {}).get("task") or {}).get("inst_to_name") or {}
    )
    if not isinstance(inst_to_name, dict) or not inst_to_name:
        return False, "cached template has no metadata.task.inst_to_name"
    missing: list[str] = []
    for obj_inst in bddl_instances:
        if obj_inst in inst_to_name:
            continue
        aliases = _LEGACY_CACHED_SCOPE_ALIASES.get((str(task_name), obj_inst), ())
        if any(alias in inst_to_name for alias in aliases):
            continue
        missing.append(obj_inst)
    if missing:
        return False, f"missing current BDDL scope entries: {missing}"
    return True, "compatible"


def _substitute_goal_expr(expr: Any, variables: Optional[Dict[str, str]] = None) -> Any:
    """Substitute already-grounded BDDL variables inside a parsed expression."""
    variables = dict(variables or {})
    if isinstance(expr, str):
        key = expr.lstrip("?")
        return variables.get(key, expr)
    if not isinstance(expr, (list, tuple)):
        return expr
    if not expr:
        return []

    token = str(expr[0])
    body = list(expr[1:])
    if token in {"forall", "exists"} and len(body) >= 2:
        iterable = list(body[0])
        bound_var = str(iterable[0]).lstrip("?")
        inner_vars = dict(variables)
        inner_vars.pop(bound_var, None)
        return [token, iterable, _substitute_goal_expr(body[1], inner_vars)]
    if token == "forn" and len(body) >= 3:
        iterable = list(body[1])
        bound_var = str(iterable[0]).lstrip("?")
        inner_vars = dict(variables)
        inner_vars.pop(bound_var, None)
        return [token, body[0], iterable, _substitute_goal_expr(body[2], inner_vars)]
    if token in {"forpairs", "fornpairs"}:
        prefix_n = 1 if token == "fornpairs" else 0
        if len(body) >= 3 + prefix_n:
            iterable1 = list(body[prefix_n])
            iterable2 = list(body[prefix_n + 1])
            bound1 = str(iterable1[0]).lstrip("?")
            bound2 = str(iterable2[0]).lstrip("?")
            inner_vars = dict(variables)
            inner_vars.pop(bound1, None)
            inner_vars.pop(bound2, None)
            if token == "fornpairs":
                return [
                    token,
                    body[0],
                    iterable1,
                    iterable2,
                    _substitute_goal_expr(body[3], inner_vars),
                ]
            return [
                token,
                iterable1,
                iterable2,
                _substitute_goal_expr(body[2], inner_vars),
            ]
    return [token] + [_substitute_goal_expr(x, variables) for x in body]


def _expand_goal_judges(
    expr: Any,
    object_map: Optional[Dict[str, list]],
    variables: Optional[Dict[str, str]] = None,
) -> list[Any]:
    """Split mandatory conjunctions / universals into independently displayed judges.

    ``and`` and ``forall`` are true iff every child is true, so each child can be
    shown as its own red/green box. Constructs such as ``exists``, ``or``,
    ``forn`` and ``forpairs`` are kept intact because splitting them would change
    the meaning of "choose one", "exactly N", or matching.
    """
    variables = dict(variables or {})
    if not isinstance(expr, (list, tuple)) or not expr:
        return [_substitute_goal_expr(expr, variables)]

    token = str(expr[0])
    body = list(expr[1:])
    if token == "and":
        out: list[Any] = []
        for child in body:
            out.extend(_expand_goal_judges(child, object_map, variables))
        return out
    if token == "forall" and len(body) >= 2:
        iterable, subexpr = body[0], body[1]
        var = str(iterable[0]).lstrip("?")
        category = str(iterable[2])
        instances = list((object_map or {}).get(category, []))
        out: list[Any] = []
        for inst in instances:
            child_vars = dict(variables)
            child_vars[var] = str(inst)
            out.extend(_expand_goal_judges(subexpr, object_map, child_vars))
        if out:
            return out
    return [_substitute_goal_expr(expr, variables)]


def _render_goal_expr(expr: Any, variables: Optional[Dict[str, str]] = None) -> str:
    """Render a parsed BDDL goal expression into a short chip label."""
    variables = dict(variables or {})
    if not isinstance(expr, (list, tuple)) or not expr:
        return _bddl_term_label(expr, variables)

    token = str(expr[0])
    body = list(expr[1:])

    if token == "and":
        return " + ".join(_render_goal_expr(e, variables) for e in body)
    if token == "or":
        return " / ".join(_render_goal_expr(e, variables) for e in body)
    if token == "not":
        inner = body[0] if body else []
        if isinstance(inner, (list, tuple)) and inner:
            pred = str(inner[0])
            args = [_bddl_term_label(arg, variables) for arg in inner[1:]]
            if pred == "open" and args:
                return f"{args[0]} 关闭"
            if pred == "real" and args:
                return f"{args[0]} 不存在"
            if pred == "covered" and len(args) >= 2:
                return f"{args[0]} 未被 {args[1]} 覆盖"
            if pred == "contains" and len(args) >= 2:
                return f"{args[0]} 不含 {args[1]}"
            if pred == "inside" and len(args) >= 2:
                return f"{args[0]} 不在 {args[1]} 内"
            if pred == "touching" and len(args) >= 2:
                return f"{args[0]} 不接触 {args[1]}"
        return "未满足: " + _render_goal_expr(body[0], variables)
    if token == "forall" and len(body) >= 2:
        iterable, subexpr = body[0], body[1]
        var = str(iterable[0]).lstrip("?")
        cat = _bddl_term_label(iterable[2], variables)
        variables[var] = cat
        return f"全部 {cat}: {_render_goal_expr(subexpr, variables)}"
    if token == "exists" and len(body) >= 2:
        iterable, subexpr = body[0], body[1]
        var = str(iterable[0]).lstrip("?")
        cat = _bddl_term_label(iterable[2], variables)
        variables[var] = cat
        return f"存在 {cat}: {_render_goal_expr(subexpr, variables)}"
    if token == "forn" and len(body) >= 3:
        n, iterable, subexpr = body[0], body[1], body[2]
        n_val = str(n[0] if isinstance(n, (list, tuple)) and n else n)
        var = str(iterable[0]).lstrip("?")
        cat = _bddl_term_label(iterable[2], variables)
        variables[var] = cat
        return f"恰好 {n_val} 个 {cat}: {_render_goal_expr(subexpr, variables)}"
    if token == "forpairs" and len(body) >= 3:
        iterable1, iterable2, subexpr = body[0], body[1], body[2]
        var1 = str(iterable1[0]).lstrip("?")
        var2 = str(iterable2[0]).lstrip("?")
        cat1 = _bddl_term_label(iterable1[2], variables)
        cat2 = _bddl_term_label(iterable2[2], variables)
        variables[var1] = cat1
        variables[var2] = cat2
        return f"{cat1} 与 {cat2} 配对: {_render_goal_expr(subexpr, variables)}"
    if token == "fornpairs" and len(body) >= 4:
        n, iterable1, iterable2, subexpr = body[0], body[1], body[2], body[3]
        n_val = str(n[0] if isinstance(n, (list, tuple)) and n else n)
        var1 = str(iterable1[0]).lstrip("?")
        var2 = str(iterable2[0]).lstrip("?")
        cat1 = _bddl_term_label(iterable1[2], variables)
        cat2 = _bddl_term_label(iterable2[2], variables)
        variables[var1] = cat1
        variables[var2] = cat2
        return f"{n_val} 组 {cat1}-{cat2}: {_render_goal_expr(subexpr, variables)}"

    if token in _PREDICATE_LABELS:
        args = [_bddl_term_label(arg, variables) for arg in body]
        if len(args) == 1:
            return _PREDICATE_LABELS[token].format(a=args[0], b="")
        if len(args) >= 2:
            return _PREDICATE_LABELS[token].format(a=args[0], b=args[1])

    args = " ".join(_bddl_term_label(arg, variables) for arg in body)
    return f"{token} {args}".strip()


_RESET_GRASP_PREP_ARM_QPOS = np.array(
    [0.0, 0.0, 0.0, -2.0943951024, 0.0, -1.0471975512, 0.0],
    dtype=np.float64,
)
_RESET_TRUNK_UPRIGHT_QPOS = np.array(
    [-0.02, 0.04667, 0.02667, 0.0001],
    dtype=np.float64,
)

# Challenge 2026 self-eval uses public test instance IDs 301..320; reported
# results should use the first 10 public indices. Keep the count configurable in
# one place because reset/random instance selection and startup share it.
_NUM_EVAL_INSTANCES = 10
_CHALLENGE_2026_PUBLIC_TEST_IDS = list(PUBLIC_TEST_INSTANCE_IDS)
_CHALLENGE_2026_HIDDEN_TEST_IDS = list(HIDDEN_TEST_INSTANCE_IDS)
_CHALLENGE_MODES = {"public_test", "hidden_test", "train"}


def _challenge_mode() -> str:
    mode = (
        os.environ.get("BEHAVIOR_CHALLENGE_MODE")
        or os.environ.get("INTERFACE_CHALLENGE_MODE")
        or "public_test"
    )
    mode = str(mode).strip()
    return mode if mode in _CHALLENGE_MODES else "public_test"


def _challenge_year() -> int:
    raw = os.environ.get("BEHAVIOR_CHALLENGE_YEAR") or os.environ.get("INTERFACE_CHALLENGE_YEAR") or "2026"
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 2026


def _challenge_2026_mode_dir(mode: str) -> str:
    return challenge_mode_dir(mode if mode in _CHALLENGE_MODES else "public_test")


def _challenge_2026_mode_for_instance(instance_id: int) -> str:
    return challenge_mode_for_instance(int(instance_id), _challenge_mode())

# reset 是否携带 instance 切换的哨兵：
#   _INSTANCE_UNSET -> 普通 reset，回到当前 instance 的 baseline（不换 instance）
#   None            -> reset 时随机换一个评测 instance
#   int             -> reset 时切换到指定 instance
_INSTANCE_UNSET = object()


def _robot_controller_contract_errors(
    robot: Any,
    robot_dof: Optional[int] = None,
) -> List[str]:
    """Return controller mismatches that make interface actions unsafe."""
    controllers = getattr(robot, "controllers", {}) or {}
    if robot_dof is None:
        robot_dof = (
            8
            if "tool_roll_left" in controllers or "tool_roll_right" in controllers
            else 7
        )
    robot_dof = normalize_robot_dof(robot_dof)
    expected = {
        "base": ("HolonomicBaseJointController", 3, "velocity", None),
        "trunk": ("JointController", 4, "position", False),
        "arm_left": ("JointController", 7, "position", False),
        "arm_right": ("JointController", 7, "position", False),
        "gripper_left": ("MultiFingerGripperController", 2, "effort", None),
        "gripper_right": ("MultiFingerGripperController", 2, "effort", None),
    }
    if robot_dof == 8:
        expected.update({
            "tool_roll_left": ("JointController", 1, "position", False),
            "tool_roll_right": ("JointController", 1, "position", False),
        })
    errors = []
    for name, (class_name, command_dim, motor_type, use_delta) in expected.items():
        controller = controllers.get(name)
        if controller is None:
            errors.append(f"{name}: missing")
            continue
        actual_class = type(controller).__name__
        actual_dim = int(getattr(controller, "command_dim", -1))
        actual_motor = getattr(
            controller,
            "motor_type",
            getattr(controller, "_motor_type", None),
        )
        if actual_class != class_name:
            errors.append(f"{name}: class={actual_class}, expected={class_name}")
        if actual_dim != command_dim:
            errors.append(f"{name}: command_dim={actual_dim}, expected={command_dim}")
        if actual_motor != motor_type:
            errors.append(f"{name}: motor_type={actual_motor}, expected={motor_type}")
        if use_delta is not None and bool(getattr(controller, "use_delta_commands", True)) != use_delta:
            errors.append(f"{name}: use_delta_commands must be {use_delta}")
        if name.startswith("gripper_") and getattr(controller, "_mode", None) != "independent":
            errors.append(f"{name}: mode={getattr(controller, '_mode', None)}, expected=independent")
    action_dim = int(getattr(robot, "action_dim", -1))
    expected_action_dim = 27 if robot_dof == 8 else 25
    if action_dim != expected_action_dim:
        errors.append(f"action_dim={action_dim}, expected={expected_action_dim}")
    return errors


def _mat3_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> xyzw 四元数（数值稳定的 Shepperd 法）。"""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw], dtype=np.float64)


# ---------------------------------------------------------------------------
# Skill 执行上下文：传给每个 skill 的 ctx 参数
# ---------------------------------------------------------------------------


@dataclass
class SkillContext:
    """暴露给 skill 的运行时上下文。
    - world: 统一世界 API（机器人/物体/动作构造）
    - log(msg): 写一行到 web 日志
    - set_status(msg): 设置 skill 状态短消息（显示在 top bar）
    - set_result(payload): 把 skill 完成后的结构化结果（dict）回传给 server，
                           web/CLI 可在 job 状态里看到，便于和后续 skill 串联
    - get_last_result(skill_name): 读上一次某 skill 的 result（跨 skill 共享，例如 get_grasp_position → execute_grasp）
    """

    world: WorldAPI
    _logger: Any
    _set_status: Any
    _set_result: Any = None
    _set_internal_result: Any = None
    _get_last_result: Any = None
    _cancel_check: Any = None
    _adjust_camera: Any = None
    task_name: str = ""

    def log(self, msg: str) -> None:
        self._logger(msg)

    def set_status(self, msg: str) -> None:
        self._set_status(msg)

    def set_result(self, payload: Dict[str, Any]) -> Any:
        if self._set_result is not None:
            return self._set_result(payload)
        return None

    def set_internal_result(self, payload: Dict[str, Any]) -> None:
        if self._set_internal_result is not None:
            self._set_internal_result(payload)

    def get_last_result(self, skill_name: str) -> Optional[Dict[str, Any]]:
        if self._get_last_result is not None:
            return self._get_last_result(skill_name)
        return None

    def is_cancelled(self) -> bool:
        if self._cancel_check is not None:
            try:
                return bool(self._cancel_check())
            except Exception:
                return False
        return False

    def raise_if_cancelled(self, where: str = "") -> None:
        from behavior_interface.errors import SkillCancelled

        if self.is_cancelled():
            msg = "skill 已取消"
            if where:
                msg = f"{msg}（{where}）"
            raise SkillCancelled(msg)


@dataclass
class SkillJob:
    request_id: str
    name: str
    args: Dict[str, Any]
    gen: Any = None
    status: str = "queued"  # queued / running / done / cancelled / failed
    message: str = ""
    result: Optional[Dict[str, Any]] = None  # skill 通过 ctx.set_result(...) 写入
    queued_ts: float = field(default_factory=time.time)
    started_ts: Optional[float] = None
    ended_ts: Optional[float] = None


# ---------------------------------------------------------------------------
# 服务器主类
# ---------------------------------------------------------------------------


class BehaviorInterface:
    def __init__(
        self,
        task: str = "make_microwave_popcorn",
        robot: Optional[str] = None,
        robot_dof: Optional[int] = None,
        scene_model: str = "house_double_floor_lower",
        main_size=(960, 540),      # GTA 主视图大小（w, h）；仅第三人称展示，可降分辨率省 RTX
        sub_size=(320, 240),       # 副视图大小
        target_hz: float = 20.0,
        dry_run: bool = False,
        physics_per_render: int = 4,
        activity_instance_id: Optional[int] = None,
        robot_config_path: Optional[str] = None,
        cli_argv: Optional[List[str]] = None,
    ):
        from .challenge_tasks import challenge_task_by_name

        task_name = str(task or "").strip()
        task_meta = challenge_task_by_name(task_name)
        if task_meta is None:
            raise ValueError(f"unknown BEHAVIOR 2026 task: {task_name!r}")
        expected_scene = str(task_meta["scene"])
        if str(scene_model) != expected_scene:
            raise ValueError(
                f"scene mismatch for BEHAVIOR 2026 task id={task_meta['id']} "
                f"{task_name}: expected {expected_scene}, got {scene_model}"
            )
        self.task_id = int(task_meta["id"])
        self.task_name = task_name
        self.scene_model = expected_scene
        self.started_ts = time.time()
        self.main_w, self.main_h = main_size
        self.sub_w, self.sub_h = sub_size
        from behavior_interface.head_capture import HEAD_IMAGE_WIDTH, HEAD_IMAGE_HEIGHT
        self.head_w, self.head_h = HEAD_IMAGE_WIDTH, HEAD_IMAGE_HEIGHT
        self.target_hz = target_hz
        self.dry_run = dry_run
        requested_dof = normalize_robot_dof(robot_dof)
        self.robot_config_path = str(resolve_challenge_robot_config_path(
            robot_config_path,
            robot_dof=requested_dof,
        ))
        robot_config = load_challenge_robot_config(
            self.robot_config_path,
            robot_dof=requested_dof,
        )
        self.robot_dof = normalize_robot_dof(robot_config["arm_dof"])
        configured_robot = str(robot_config["robot_type"])
        if robot is not None and str(robot).strip().lower() != configured_robot.lower():
            raise ValueError(
                f"requested robot={robot!r} does not match "
                f"robot config type={configured_robot!r}"
            )
        self.robot_name = configured_robot
        # 真实仿真时单次主循环 tick 里推进的物理 step 数。提高这个数能让物理时钟
        # 跑得更快（4 路 RGB 渲染才是瓶颈），代价是显示 fps 不变但每帧间隔机器人移动更多。
        self.physics_per_render = max(1, int(physics_per_render))
        self._cli_argv = list(cli_argv or sys.argv)

        # === GTA 主视图相机微调参数（可通过 /api/camera 调） ===
        # distance       : 相机与机器人 base 的水平距离（米）
        # height         : 相机在 base 上方的高度（米）
        # yaw_offset_deg : 相机方位相对机器人正后方的偏移角（正=向左转，负=向右转）
        # look_z_offset  : 看的目标点相对 base 的高度（米），决定俯仰
        #                  减小 -> 相机更俯，增大 -> 相机更仰
        self.cam_defaults = {
            "distance": 2.5,
            "height": 2.8,
            "yaw_offset_deg": 0.0,
            "look_z_offset": 1.2,
        }
        self.cam_params = dict(self.cam_defaults)
        self.cam_lock = threading.Lock()
        self.camera_io_lock = threading.RLock()

        # === 公共状态 ===
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.skill_lock = threading.Lock()

        # 4 路画面 BGR uint8
        self.frames: Dict[str, np.ndarray] = {
            "main": make_placeholder(self.main_w, self.main_h, "waiting for sim..."),
            "head": make_placeholder(self.head_w, self.head_h, "head cam"),
            "left_wrist": make_placeholder(self.sub_w, self.sub_h, "left wrist"),
            "right_wrist": make_placeholder(self.sub_w, self.sub_h, "right wrist"),
        }
        self.frame_ids: Dict[str, int] = {k: 0 for k in self.frames}
        self.frame_updated_ts: Dict[str, float] = {k: time.time() for k in self.frames}
        self.frame_id = 0

        # === 视图按需渲染 ===
        # 左栏四路视图默认只开 head，其余（GTA 主视图 / 左右腕）关闭以省 GPU/CPU：
        #   - GTA 主视图是 rgb-only 外部相机，关闭时暂停其 render product，og.sim.render()
        #     不再为它跑 RTX（最贵的一路），运动期间甚至可整帧跳过渲染 → 直接提速、减少 timeout。
        #   - head/wrist 是机器人相机，绝不暂停其 render product（会让 Replicator 失效），
        #     关闭时仅跳过取帧+JPEG 编码（省 CPU/GPU 回读），渲染质量与分辨率完全不变。
        # 用户在 UI 手动开某路时，主循环会同步预热（enable + 多渲染几帧再取帧），保证打开即有画面。
        self._feed_active: Dict[str, bool] = {
            "main": False,
            "head": True,
            "left_wrist": False,
            "right_wrist": False,
        }
        self._feed_active_lock = threading.Lock()
        self._feed_warmup_pending: set[str] = set()
        self._gta_render_enabled = True  # GTA render product 当前是否在渲染
        # goal chip 后台刷新限频：turn 输入的 goals 走 /api/state，但 goals 只在 skill
        # 改变世界后才变；空闲时 20Hz 反复跑 BDDL 评估纯属浪费，限频到 ~0.5s，并在
        # skill 刚结束的那一 tick 强制刷新，保证完成判定不迟滞。
        self._goal_cache_interval_s = 0.5
        self._goal_cache_last_ts = 0.0
        self._prev_had_job = False
        self._last_finished_skill_name: Optional[str] = None
        self.camera_health: Dict[str, CameraFeedHealth] = {}
        self._vision_safe_state = False
        self._vision_safe_reason = ""
        self._camera_last_summary_log: Dict[str, float] = {}

        self.log_lines: deque = deque(maxlen=200)
        self.tick = 0
        self.fps = 0.0
        self._fps_ema = 20.0
        self._last_tick_ts = time.time()

        # 主循环把 PhysX 数据序列化成纯 dict 存到这里；web 线程只读 cache，绝不直接碰 PhysX。
        # 这是为了避免 Flask 处理 /api/state 与主 sim 线程在 PhysX 上发生 read/write 冲突
        # （会导致 articulation_view 失效，后续 env.step 全部抛 'NoneType has no attribute view'）。
        self._cached_robot_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0}
        self._cached_eef_pose: Dict[str, Any] = {}
        self._cached_tro: Dict[str, Any] = {}
        self._cached_memory: Dict[str, Any] = {}
        self._cached_memory_text = "(memory 未构建)"
        self._cached_goals: Dict[str, Any] = {
            "items": [],
            "satisfied": 0,
            "total": 0,
            "complete": False,
            "ok": False,
        }

        # Scene Graph 缓存：主线程定时重建，web/skill 只读。
        # SceneGraph 比 TRO 重很多（遍历几百物体 + 关系计算），所以每 5 秒重建一次。
        # 显示给 web 的文本和 SceneGraph 对象本身都缓存（避免 web 每秒序列化几百物体）。
        self._cached_scene_graph = None              # SceneGraph 实例
        self._cached_scene_graph_text = "(未构建)"
        self._cached_scene_graph_summary: Dict[str, Any] = {}
        self._scene_graph_interval_s = 15.0          # skill 运行中的重建间隔
        self._scene_graph_last_build_ts = 0.0
        self._scene_graph_robot_radius = 0.25
        # 空闲省算力：记录上次重建时的机器人位姿，空闲静止时跳过 ~10s 的重建阻塞，
        # 仅当机器人相对上次重建移动超过阈值时才在空闲期重建。
        self._scene_graph_last_pose: Optional[Dict[str, float]] = None
        self._scene_graph_idle_move_xy = 0.05        # m，空闲重建的平移阈值
        self._scene_graph_idle_move_yaw = 2.0        # deg，空闲重建的转角阈值
        # 空闲省算力：未绑定 capture 图时后台 build_memory 会每拍读 head 深度（720×720
        # 回读），是 pristine 空闲卡顿 + head 变 stale 阻塞 skill 的根因。周期刷新时机器人
        # 静止且距上次全量重建不足保活间隔就复用旧缓存，避免每拍读深度。turn 的 memory
        # 时效由 capture skill 自身输出注入保证，不依赖此后台刷新，故限频不影响 harness。
        self._memory_periodic_interval_s = 5.0       # 静止时后台 memory 的保活重建间隔
        self._memory_periodic_last_ts = 0.0
        self._memory_periodic_last_pose: Optional[Dict[str, float]] = None
        self._memory_periodic_move_xy = 0.02         # m，触发重建的平移阈值
        self._memory_periodic_move_yaw = 1.0         # deg，触发重建的转角阈值
        try:
            self._native_trim_interval_s = max(
                0.0,
                float(os.environ.get("BEHAVIOR_INTERFACE_NATIVE_TRIM_INTERVAL_S", "0")),
            )
        except ValueError:
            self._native_trim_interval_s = 0.0
        self._native_trim_last_ts = 0.0
        # 磁盘守卫：低水位先告警，再低则回收陈旧会话产物（见 agent_runs 保留策略）
        self._disk_guard_last_ts = 0.0
        self._disk_guard_interval_s = _env_float(
            "BEHAVIOR_INTERFACE_DISK_GUARD_INTERVAL_S", 600.0, minimum=0.0
        )
        self._disk_warn_mib = _env_float(
            "BEHAVIOR_INTERFACE_DISK_WARN_MIB", 98304.0, minimum=0.0
        )
        self._disk_gc_mib = _env_float(
            "BEHAVIOR_INTERFACE_DISK_GC_MIB", 65536.0, minimum=0.0
        )
        self._runs_max_age_h = _env_float(
            "BEHAVIOR_INTERFACE_RUNS_MAX_AGE_H", 48.0, minimum=0.0
        )
        self._idle_quiescence_enabled = _env_truthy(
            "BEHAVIOR_INTERFACE_IDLE_QUIESCENCE", True
        )
        try:
            self._idle_settle_s = max(
                0.0,
                float(os.environ.get("BEHAVIOR_INTERFACE_IDLE_SETTLE_S", "2.0")),
            )
        except ValueError:
            self._idle_settle_s = 2.0
        try:
            self._idle_poll_s = max(
                0.01,
                float(os.environ.get("BEHAVIOR_INTERFACE_IDLE_POLL_S", "0.1")),
            )
        except ValueError:
            self._idle_poll_s = 0.1
        self._idle_settle_deadline_mono = time.monotonic() + self._idle_settle_s
        self._idle_quiescent = False
        self._idle_quiescent_since_ts: Optional[float] = None
        self._idle_suppressed_ticks = 0
        self._idle_wake_event = threading.Event()
        self._idle_vision_prime_job_token: Optional[str] = None

        # 空闲渲染限频：静止场景每拍白渲染一次 og.sim.render()(~1.5s)是空闲期最大头。
        # 机器人静止、相机位姿未变、无 skill 时画面不变 → 跳过渲染+抓帧、复用上一帧
        # （web 按 frame_id 不重复 JPEG 编码）；相机被调/机器人移动/视图刚开启/保活间隔到
        # 才重渲染。capture 是 job（idle_tick=False，走完整渲染），不受此限频影响，harness
        # 图像时效不变。
        self._gta_cam_dirty = True                   # 相机位姿脏标记：首帧/调相机后强制渲染
        # 保活间隔必须显著小于 head 的 stale_after_s=5.0：否则空闲复用时 head 帧超时被判
        # stale → 触发 vision-safe 阻断 skill。2.0s 下最坏刷新间隔≈4s<5s，安全。
        self._render_idle_keepalive_s = 2.0          # 静止时保活重渲染间隔(s)
        self._render_idle_last_ts = 0.0
        self._render_idle_last_pose: Optional[Dict[str, float]] = None
        self._render_idle_move_xy = 0.01             # m，触发重渲染的平移阈值
        self._render_idle_move_yaw = 0.5             # deg，触发重渲染的转角阈值

        # skill 队列
        self.skill_queue: "queue.Queue[SkillJob]" = queue.Queue()
        self._pending_skill_hint: Optional[Dict[str, Any]] = None
        self.current_job: Optional[SkillJob] = None
        self.cancel_flag = False
        self.skill_status_msg = ""
        # 各 skill 的最近一次 result（dict 形式），允许 skill 之间共享数据
        # 例如 get_grasp_position 把候选 grasp 写入，execute_grasp 读取
        self._last_skill_results: Dict[str, Dict[str, Any]] = {}
        self._last_skill_internal_results: Dict[str, Dict[str, Any]] = {}
        # Chronological skill/tool history for agent handoff.  This is a
        # passive audit trail only: it does not affect skill execution.
        self._skill_history: deque = deque(maxlen=300)
        self._skill_history_seq = 0

        # restart 请求标志：web 线程置位，主 sim 线程下一拍执行 env.reset()
        # （PhysX 操作必须在主线程，否则会触发并发污染）
        self.reset_request = False
        self.reset_count = 0
        self._after_world_reset_hooks: list = []
        # task 热切换请求：web 线程只入队，实际 env reload 在主 sim 线程执行。
        self.task_switch_request: Optional[Dict[str, Any]] = None
        self.task_switch_count = 0
        self.task_switch_in_progress = False
        self.task_switch_error: Optional[str] = None
        self.task_switch_reexec_requested = False
        self.task_switch_reexec_target: Optional[Dict[str, Any]] = None

        # === task instance（与 BEHAVIOR challenge 评测一致）===
        # requested_instance_id 为 None 表示每次启动/切换随机选一个评测 instance；
        # 指定 int 则固定加载该 instance。current_instance_id 是当前实际加载的 instance。
        self.requested_instance_id: Optional[int] = (
            int(activity_instance_id) if activity_instance_id is not None else None
        )
        self.current_instance_id: Optional[int] = None
        # reset 是否携带 instance 切换（见模块级 _INSTANCE_UNSET 注释）。
        self._reset_instance_request: Any = _INSTANCE_UNSET
        # 独立于 scene._initial_file 的完整世界快照。普通 restart 必须从这里恢复，
        # 避免运行期代码意外改写 OG baseline 后只能复位机器人 / TRO。
        self._world_reset_scene_file: Optional[Dict[str, Any]] = None
        self._simulation_degraded = False
        self._simulation_degraded_reason = ""
        self._simulation_degraded_since_ts: Optional[float] = None

        # 仿真句柄
        self.env = None
        self.robot = None
        self.gta_sensor = None  # external VisionSensor 用作 GTA 视角
        # GTA 相机 3x3 旋转矩阵（cam->world），由 _update_gta_camera_pose 写入，被 HUD 用来
        # 把世界三轴投影到 2D 屏幕（让指南针真正反映"从当前视角看世界 +X/+Y/+Z 在哪边"）
        self._gta_cam_R = np.eye(3, dtype=np.float64)
        self.world = WorldAPI(dry_run=dry_run, robot_dof=self.robot_dof)
        self.world.control_hz = float(self.target_hz)
        self.world._gpu_diag_log = self.log

        self._stopped = False
        self._stdout_broken = False
        self._reset_camera_health()

    # ------------------------------------------------------------------ utils

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self.state_lock:
            self.log_lines.append(line)
        if self._stdout_broken:
            return
        try:
            print(line, flush=True)
        except (BrokenPipeError, OSError, ValueError):
            self._stdout_broken = True

    def _log_gpu_diag(
        self,
        tag: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        include_nvidia: bool = True,
    ) -> None:
        try:
            log_gpu_diag(self.log, tag, extra=extra, include_nvidia=include_nvidia)
        except Exception:
            pass

    def _should_log_skill_gpu_diag(self, skill_name: str) -> bool:
        if os.environ.get("BEHAVIOR_GPU_DIAG_SKILL_EVENTS", "1").strip().lower() in {"0", "false", "no", "off"}:
            return False
        name = str(skill_name)
        risky_tokens = ("move", "eef", "grasp", "reachability", "curobo", "ik")
        return any(token in name for token in risky_tokens)

    def _set_skill_status(self, msg: str) -> None:
        with self.state_lock:
            self.skill_status_msg = msg

    # ---- camera adjust ----

    # 单击按钮的步长（HTML 一致）
    CAMERA_STEPS = {
        "distance": 0.3,        # 米
        "height": 0.3,          # 米
        "yaw_offset_deg": 15.0, # 度
        "look_z_offset": 0.2,   # 米
    }
    # 各参数的合法范围（防止把相机调到地下或撞机器人）
    CAMERA_LIMITS = {
        "distance": (0.6, 8.0),
        "height": (-0.5, 6.0),
        "yaw_offset_deg": (-180.0, 180.0),
        "look_z_offset": (-1.0, 3.0),
    }

    def adjust_camera(self, deltas: Optional[Dict[str, float]] = None,
                      overrides: Optional[Dict[str, float]] = None,
                      reset: bool = False) -> Dict[str, float]:
        """调整 GTA 主视图相机参数，支持三种用法：
        - reset=True：恢复默认值。
        - deltas={key: +/- step_count}：在当前值基础上按预设步长增减若干步。
        - overrides={key: absolute_value}：直接覆盖某个参数。
        优先级：reset > overrides > deltas（同一次调用同时给会按这个顺序套用）。
        返回当前生效的相机参数字典。
        """
        with self.cam_lock:
            if reset:
                self.cam_params = dict(self.cam_defaults)
            if deltas:
                for k, n_steps in deltas.items():
                    if k not in self.cam_params or k not in self.CAMERA_STEPS:
                        continue
                    new_val = self.cam_params[k] + float(n_steps) * self.CAMERA_STEPS[k]
                    lo, hi = self.CAMERA_LIMITS[k]
                    self.cam_params[k] = max(lo, min(hi, new_val))
            if overrides:
                for k, v in overrides.items():
                    if k not in self.cam_params:
                        continue
                    lo, hi = self.CAMERA_LIMITS[k]
                    self.cam_params[k] = max(lo, min(hi, float(v)))
            snap = dict(self.cam_params)
        # 置脏：空闲渲染限频据此在下一拍强制重渲染，保证调相机后立即刷新画面。
        self._gta_cam_dirty = True
        self._signal_sim_wake()
        self.log(f"camera = {snap}")
        return snap

    def get_camera_state(self) -> Dict[str, Any]:
        with self.cam_lock:
            state = {
                "params": dict(self.cam_params),
                "defaults": dict(self.cam_defaults),
                "steps": dict(self.CAMERA_STEPS),
                "limits": dict(self.CAMERA_LIMITS),
            }
        state["streams"] = {
            feed: health.snapshot()
            for feed, health in self.camera_health.items()
        }
        state["vision_safe_state"] = self._vision_safe_state
        state["vision_safe_reason"] = self._vision_safe_reason
        return state

    @staticmethod
    def _robot_camera_feed(sensor_name: str) -> Optional[str]:
        name = str(sensor_name).lower()
        if "zed_link" in name or "zed" in name or "head" in name:
            return "head"
        if "left_realsense" in name or "left_wrist" in name:
            return "left_wrist"
        if "right_realsense" in name or "right_wrist" in name:
            return "right_wrist"
        return None

    def _reset_camera_health(self) -> None:
        for health in getattr(self, "camera_health", {}).values():
            health.stop()
        defaults = {
            "main": ("gta_view", "OmniGibson VisionSensor/Replicator"),
            "head": ("robot_head", "OmniGibson VisionSensor/Replicator"),
            "left_wrist": ("robot_left_wrist", "OmniGibson VisionSensor/Replicator"),
            "right_wrist": ("robot_right_wrist", "OmniGibson VisionSensor/Replicator"),
        }
        self.camera_health = {
            feed: CameraFeedHealth(
                source,
                backend,
                failure_threshold=3,
                stale_after_s=5.0,
                reconnect_backoff_s=0.5,
                reconnect_backoff_max_s=8.0,
            )
            for feed, (source, backend) in defaults.items()
        }
        self._camera_failure_placeholders = {}
        self._camera_first_frame_logged = set()
        self._vision_safe_state = False
        self._vision_safe_reason = ""

    def _bind_camera_health_sources(self) -> None:
        self._reset_camera_health()
        if self.world is not None:
            self.world._codex_camera_io_lock = self.camera_io_lock

        found = set()
        if self.gta_sensor is not None:
            health = CameraFeedHealth(
                getattr(self.gta_sensor, "prim_path", "gta_view"),
                type(self.gta_sensor).__name__,
            )
            health.record_open(True)
            self.camera_health["main"] = health
            found.add("main")

        sensors = getattr(self.robot, "sensors", {}) if self.robot is not None else {}
        try:
            items = list(sensors.items())
        except Exception:
            items = []
        for sensor_name, sensor in items:
            feed = self._robot_camera_feed(sensor_name)
            if feed is None or "rgb" not in list(getattr(sensor, "modalities", []) or []):
                continue
            source = getattr(sensor, "prim_path", None) or sensor_name
            health = CameraFeedHealth(source, type(sensor).__name__)
            opened = bool(getattr(sensor, "initialized", True))
            health.record_open(opened, "sensor is not initialized" if not opened else "")
            self.camera_health[feed] = health
            found.add(feed)

        for feed, health in self.camera_health.items():
            if feed not in found:
                health.record_open(False, f"{feed} sensor not found or rgb modality missing")
            snap = health.snapshot()
            self.log(
                "camera init "
                f"feed={feed} source={snap['source']} backend={snap['backend']} "
                f"open={snap['opened']}"
                + (f" error={snap['last_error']}" if snap["last_error"] else "")
            )
        if self.camera_health["head"].snapshot()["degraded"]:
            self._enter_vision_safe_state("head camera initialization failed")

    def _enter_vision_safe_state(self, reason: str) -> None:
        if self._stopped or self.task_switch_in_progress or self.reset_request:
            return
        first_entry = not self._vision_safe_state
        self._vision_safe_state = True
        self._vision_safe_reason = str(reason)
        if self.world is not None:
            self.world._codex_fast_motion_no_obs = False
        if self.current_job is not None:
            self.cancel_flag = True
        if first_entry:
            self.log(
                "ERROR vision degraded; entering safe hold "
                f"reason={self._vision_safe_reason}"
            )

    def _leave_vision_safe_state(self) -> None:
        if not self._vision_safe_state:
            return
        self._vision_safe_state = False
        old_reason = self._vision_safe_reason
        self._vision_safe_reason = ""
        self.log(f"camera head recovered; leaving safe hold previous_reason={old_reason}")

    def _record_camera_failure(self, feed: str, frame: Any, reason: str) -> None:
        health = self.camera_health[feed]
        entered_degraded = health.record_failure(reason)
        snap = health.snapshot()
        failures = int(snap["consecutive_failures"])
        should_log = (
            failures == 1
            or entered_degraded
            or failures in {5, 10}
            or failures % 25 == 0
        )
        if should_log and not self._stopped:
            self.log(
                "WARN camera read failed "
                f"feed={feed} source={snap['source']} backend={snap['backend']} "
                f"consecutive_failures={failures}/{snap['failure_threshold']} "
                f"last_success_age_s={snap['last_success_age_s']} "
                f"frame={frame_description(frame)} reason={reason}"
            )
        if feed == "head" and (snap["degraded"] or health.stale()):
            self._enter_vision_safe_state(
                f"head stream unavailable after {failures} consecutive failures"
            )

    def _record_camera_success(self, feed: str, frame: Any) -> bool:
        health = self.camera_health[feed]
        before = health.snapshot()
        if not health.record_success(frame):
            self._record_camera_failure(feed, frame, "frame validation failed")
            return False
        if before["consecutive_failures"] or before["degraded"]:
            snap = health.snapshot()
            self.log(
                "camera read recovered "
                f"feed={feed} source={snap['source']} backend={snap['backend']} "
                f"previous_failures={before['consecutive_failures']} "
                f"reconnect_count={snap['reconnect_count']} frame={snap['frame']}"
            )
        elif feed not in self._camera_first_frame_logged:
            snap = health.snapshot()
            self._camera_first_frame_logged.add(feed)
            self.log(
                "camera first frame "
                f"feed={feed} source={snap['source']} backend={snap['backend']} "
                f"read=True frame={snap['frame']}"
            )
        if feed == "head":
            self._leave_vision_safe_state()
        return True

    def _maybe_reconnect_robot_camera(
        self,
        feed: str,
        sensor: Any,
        *,
        reason: str,
    ) -> bool:
        health = self.camera_health[feed]
        attempt = health.begin_reconnect()
        if attempt is None or self._stopped or self.task_switch_in_progress:
            return False
        snap = health.snapshot()
        self.log(
            "camera reconnect start "
            f"feed={feed} source={snap['source']} backend={snap['backend']} "
            f"attempt={attempt} reason={reason}"
        )
        try:
            with self.camera_io_lock:
                report = rebuild_robot_camera_render_products(
                    self.world,
                    sensor=sensor,
                )
            ok = bool(report.get("ok"))
            error = "; ".join(report.get("errors") or [])
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
        health.finish_reconnect(ok, error)
        self.log(
            "camera reconnect result "
            f"feed={feed} attempt={attempt} ok={ok}"
            + (f" error={error}" if error else "")
        )
        return True

    # ------------------------------------------------------------------ skill

    def submit_skill(self, name: str, args: Dict[str, Any]) -> str:
        if getattr(self, "_simulation_degraded", False):
            raise RuntimeError(
                "simulation is degraded; reset or switch task before submitting skills: "
                f"{self._simulation_degraded_reason}"
            )
        # 热重载后 skills 模块会换新 SKILL_REGISTRY 字典，这里每次动态取最新注册表
        from behavior_interface.skills import SKILL_REGISTRY as reg
        if name not in reg:
            raise ValueError(f"未注册 skill: {name}")
        from behavior_interface.v2_display import skill_display_name

        req_id = f"job-{int(time.time()*1000)}"
        job = SkillJob(request_id=req_id, name=name, args=dict(args))
        self.skill_queue.put(job)
        label = skill_display_name(name, args)
        with self.state_lock:
            self._pending_skill_hint = {
                "name": label,
                "skill": name,
                "request_id": req_id,
                "args": dict(args),
                "queue_depth": self.skill_queue.qsize(),
            }
        self._signal_sim_wake()
        self.log(f"queued {label}({args}) -> {req_id}")
        return req_id

    def _history_safe(self, obj: Any, *, max_string: int = 1200, max_list: int = 80, depth: int = 0) -> Any:
        """Return a JSON-safe, prompt-safe copy for chronological skill history."""
        if depth > 8:
            return "<max-depth>"
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                key = str(k)
                if key in {"rgb_main", "rgb_overlay", "rgb", "data_url"} and isinstance(v, str) and v.startswith("data:"):
                    out[key] = f"<data-url {len(v)} chars>"
                else:
                    out[key] = self._history_safe(v, max_string=max_string, max_list=max_list, depth=depth + 1)
            return out
        if isinstance(obj, (list, tuple)):
            vals = list(obj)
            out = [self._history_safe(v, max_string=max_string, max_list=max_list, depth=depth + 1) for v in vals[:max_list]]
            if len(vals) > max_list:
                out.append(f"<truncated {len(vals) - max_list} items>")
            return out
        if isinstance(obj, str):
            if obj.startswith("data:"):
                return f"<data-url {len(obj)} chars>"
            if len(obj) > max_string:
                return obj[:max_string] + f"... <truncated {len(obj) - max_string} chars>"
            return obj
        if isinstance(obj, (int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _record_skill_history(self, job: SkillJob, *, status: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Append one completed/cancelled/failed skill call to the chronological audit trail."""
        try:
            from behavior_interface.v2_display import skill_display_name
            display = skill_display_name(job.name, job.args)
        except Exception:
            display = job.name
        now = time.time()
        job.ended_ts = job.ended_ts or now
        args = dict(job.args or {})
        session_id = str(args.get("session_id") or "")
        with self.state_lock:
            self._skill_history_seq += 1
            rec = {
                "seq": self._skill_history_seq,
                "request_id": job.request_id,
                "skill": job.name,
                "display_name": display,
                "status": status,
                "session_id": session_id,
                "source": "web" if session_id == "web" else ("unknown" if not session_id else "session"),
                "args": self._history_safe(args),
                "result": self._history_safe(result if result is not None else job.result),
                "queued_ts": round(float(job.queued_ts or now), 3),
                "started_ts": round(float(job.started_ts or now), 3) if job.started_ts else None,
                "ended_ts": round(float(job.ended_ts or now), 3),
                "elapsed_s": round(float((job.ended_ts or now) - (job.started_ts or job.queued_ts or now)), 3),
                "reset_count": self.reset_count,
                "tick": self.tick,
            }
            self._skill_history.append(rec)

    def wait_for_skill_result(
        self,
        skill_name: str,
        timeout_s: float = 180.0,
        poll_s: float = 0.25,
        request_id: str | None = None,
    ) -> Dict[str, Any]:
        """阻塞等待指定 skill 完成并返回其 result（供 REST 同步接口使用）。

        若提供 request_id，仅当 _last_skill_results 中该 skill 的结果带有相同 job 时才返回，
        避免把上一次运行的陈旧结果当成本次完成（表现为 HTTP 已返回 done 但机器人未动）。
        """
        t0 = time.time()
        with self.state_lock:
            prev = self._last_skill_results.get(skill_name)
        while time.time() - t0 < timeout_s:
            time.sleep(poll_s)
            with self.state_lock:
                cur = self._last_skill_results.get(skill_name)
            if request_id is not None:
                if cur is not None and cur.get("job") == request_id:
                    return dict(cur)
            elif cur is not None and cur != prev:
                return dict(cur)
        raise TimeoutError(f"等待 skill {skill_name!r} 超时 ({timeout_s}s)")

    def cancel_current_skill(self) -> None:
        with self.skill_lock:
            if self.current_job is not None:
                self.cancel_flag = True
                self.log(f"cancel requested: {self.current_job.name}")

    def _gripper_joint_open_targets(self, arm: str) -> list[tuple[int, str, float]]:
        """Return (joint_index, joint_name, open_qpos) for R1Pro finger joints."""
        if self.robot is None:
            return []
        names = list(self.robot.joints.keys())
        joint_names: list[str] = []
        try:
            joint_names.extend(list(getattr(self.robot, "finger_joint_names", {}).get(arm, [])))
        except Exception:
            pass
        joint_names.extend([
            f"{arm}_gripper_finger_joint1",
            f"{arm}_gripper_finger_joint2",
        ])
        try:
            for fj in getattr(self.robot, "finger_joints", {}).get(arm, []):
                jn = getattr(fj, "joint_name", None) or getattr(fj, "name", None)
                if jn:
                    joint_names.append(str(jn))
        except Exception:
            pass

        out: list[tuple[int, str, float]] = []
        seen: set[str] = set()
        for jn in joint_names:
            jn = str(jn)
            if not jn or jn in seen or jn not in names:
                continue
            seen.add(jn)
            joint = self.robot.joints[jn]
            try:
                open_q = float(getattr(joint, "upper_limit"))
            except Exception:
                open_q = 0.05
            if not np.isfinite(open_q):
                open_q = 0.05
            out.append((int(names.index(jn)), jn, open_q))
        return out

    def _set_open_grippers_in_qpos(self, q, arms: tuple[str, ...]) -> tuple[list[int], dict[str, list[float]]]:
        """Write open finger qpos into an existing joint-position vector."""
        touched: list[int] = []
        qpos: dict[str, list[float]] = {}
        for arm in arms:
            vals: list[float] = []
            for j_idx, _, open_q in self._gripper_joint_open_targets(arm):
                q[int(j_idx)] = float(open_q)
                touched.append(int(j_idx))
                vals.append(float(open_q))
            qpos[arm] = vals
            try:
                if self.world is not None and vals:
                    self.world.set_gripper_pin_qpos(arm, vals)
            except Exception:
                pass
        return touched, qpos

    def _force_open_gripper_joints(self, arms: tuple[str, ...] = ("left", "right")) -> dict[str, list[float]]:
        """Hard-set finger joints to upper limits; used during reset before pins lock."""
        if self.dry_run or self.robot is None:
            return {}
        q0 = self.robot.get_joint_positions()
        q = q0.clone() if hasattr(q0, "clone") else np.asarray(q0, dtype=np.float64).copy()
        touched, _ = self._set_open_grippers_in_qpos(q, arms)
        if touched:
            self.robot.set_joint_positions(q)
            try:
                v0 = self.robot.get_joint_velocities()
                v = v0.clone() if hasattr(v0, "clone") else np.asarray(v0, dtype=np.float64).copy()
                for j in touched:
                    v[int(j)] = 0.0
                self.robot.set_joint_velocities(v)
            except Exception:
                pass
        return self._gripper_qpos_debug(arms)

    def _gripper_qpos_debug(self, arms: tuple[str, ...] = ("left", "right")) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        if self.dry_run or self.robot is None:
            return out
        try:
            q = self.robot.get_joint_positions()
        except Exception:
            return out
        for arm in arms:
            vals = []
            for j_idx, _, _ in self._gripper_joint_open_targets(arm):
                vals.append(round(float(q[int(j_idx)]), 5))
            out[arm] = vals
        return out

    def _gripper_open_action_kwargs(self, arms: tuple[str, ...] = ("left", "right")) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        if self.world is None:
            return out
        for arm in arms:
            vals = [float(q_open) for _, _, q_open in self._gripper_joint_open_targets(arm)]
            try:
                idx = self.world.controller_action_idx(f"gripper_{arm}")
                out[f"gripper_{arm}"] = vals if vals and len(idx) == len(vals) else [1.0]
            except Exception:
                out[f"gripper_{arm}"] = [1.0]
        return out

    def request_reset(self, instance_id: Any = _INSTANCE_UNSET) -> int:
        """供 web 线程调用：把 reset 标志置上，主 sim 线程下一拍真正执行 env.reset()。

        instance_id：
          _INSTANCE_UNSET（默认）-> 普通 reset，回到当前 instance 的 baseline；
          None                   -> reset 时随机换一个评测 instance；
          int                    -> reset 时切换到指定 instance。
        """
        if instance_id is not _INSTANCE_UNSET and not self.dry_run:
            requested = None if instance_id is None else int(instance_id)
            self._validate_task_instance_assets(
                task_name=self.task_name,
                scene_model=self.scene_model,
                requested_instance_id=requested,
            )
        with self.skill_lock:
            self.reset_request = True
            self._reset_instance_request = instance_id
            pending = self.reset_count + 1
        self._signal_sim_wake()
        if instance_id is _INSTANCE_UNSET:
            self.log("RESET 请求已入队（保持当前 instance），等待主线程下一拍执行...")
        else:
            self.log(
                f"RESET 请求已入队（instance="
                f"{'随机' if instance_id is None else instance_id}），等待主线程下一拍执行..."
            )
        return pending

    def request_task_switch(self, task_name: str, scene_model: Optional[str] = None) -> int:
        """Queue a Behavior task hot-switch. The sim thread performs the actual env reload."""
        from .challenge_tasks import challenge_task_by_name

        task_name = str(task_name or "").strip()
        if not task_name:
            raise ValueError("missing task_name")
        task_meta = challenge_task_by_name(task_name)
        if task_meta is None:
            raise ValueError(f"unknown task: {task_name}")
        expected_scene = str(task_meta["scene"])
        scene_model = str(scene_model or expected_scene or self.scene_model).strip() or self.scene_model
        if expected_scene and scene_model != expected_scene:
            raise ValueError(
                f"scene mismatch for task id={task_meta['id']} {task_name}: "
                f"expected {expected_scene}, got {scene_model}"
            )
        if not self.dry_run:
            self._validate_task_instance_assets(
                task_name=task_name,
                scene_model=scene_model,
                requested_instance_id=self.requested_instance_id,
            )
        with self.skill_lock:
            self.task_switch_request = {
                "task_id": int(task_meta["id"]),
                "task": task_name,
                "scene": scene_model,
                "requested_at": time.time(),
            }
            self.task_switch_error = None
            pending = self.task_switch_count + 1
        self._signal_sim_wake()
        self.log(
            f"TASK SWITCH 请求已入队: {self.task_name} -> {task_name} "
            f"(scene={scene_model})"
        )
        return pending

    def _clear_pending_skill_runtime(self) -> int:
        """Clear queued/running skills without touching sim state. Must be called on sim thread."""
        cleared = 0
        while True:
            try:
                self.skill_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        try:
            if self.current_job is not None and getattr(self.current_job, "gen", None) is not None:
                close = getattr(self.current_job.gen, "close", None)
                if callable(close):
                    close()
        except Exception:
            pass
        self.current_job = None
        self._last_finished_skill_name = None
        self.cancel_flag = False
        self.skill_status_msg = ""
        with self.state_lock:
            self._pending_skill_hint = None
        return cleared

    def _release_reset_runtime_caches(self) -> None:
        """Release reset-invalidated Python and GPU caches on the simulation thread."""
        if self.dry_run:
            return
        results: Dict[str, Any] = {}
        cleanup_specs = (
            (
                "opening_volume",
                "behavior_interface.skills.plan_grasp_opening_volume",
                "clear_opening_volume_caches",
            ),
            (
                "vertical_plan",
                "behavior_interface.trunk_vertical_lift",
                "clear_vertical_plan_cache",
            ),
            (
                "v7_gpu",
                "behavior_interface.skills.grasp_obj_v7_gpu",
                "release_v7_gpu_memory",
            ),
        )
        for label, module_name, function_name in cleanup_specs:
            try:
                module = importlib.import_module(module_name)
                cleanup = getattr(module, function_name)
                value = cleanup()
                results[label] = "ok" if value is None else value
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                results[label] = error
                self.log(f"WARN reset cache cleanup failed [{label}]: {error}")

        self.log(f"RESET_CACHE_CLEANUP result={results}")

    def _apply_reset_grasp_prep_pose(self) -> None:
        """env.reset() 后立刻把躯干/双臂放到可控初值，并写入 pin 目标。"""
        if self.dry_run or self.robot is None:
            return
        names = list(self.robot.joints.keys())
        q0 = self.robot.get_joint_positions()
        q = q0.clone() if hasattr(q0, "clone") else np.asarray(q0, dtype=np.float64).copy()
        touched: list[int] = []
        trunk_changed = False
        try:
            trunk_idx_raw = self.robot.trunk_control_idx
            if hasattr(trunk_idx_raw, "detach"):
                trunk_idx_raw = trunk_idx_raw.detach().cpu().numpy()
            trunk_idx = [int(i) for i in np.asarray(trunk_idx_raw).reshape(-1)[:4]]
            for i, j in enumerate(trunk_idx):
                q[int(j)] = float(_RESET_TRUNK_UPRIGHT_QPOS[i])
                touched.append(int(j))
            trunk_changed = len(trunk_idx) >= 4
        except Exception as exc:
            self.log(f"RESET trunk init warning: {exc}")
        changed: list[str] = []
        tool_roll_pins: dict[str, float] = {}
        for arm in ("left", "right"):
            try:
                idx = [names.index(f"{arm}_arm_joint{i+1}") for i in range(7)]
            except ValueError:
                continue
            for i, j in enumerate(idx):
                q[int(j)] = float(_RESET_GRASP_PREP_ARM_QPOS[i])
                touched.append(int(j))
            changed.append(arm)
            try:
                j8 = names.index(f"{arm}_arm_joint8")
                tool_roll_pins[arm] = float(q[int(j8)])
            except ValueError:
                pass
            try:
                grip_touched, _ = self._set_open_grippers_in_qpos(q, (arm,))
                touched.extend(grip_touched)
            except Exception as exc:
                self.log(f"RESET gripper init warning[{arm}]: {exc}")
        if touched:
            self.robot.set_joint_positions(q)
            try:
                v0 = self.robot.get_joint_velocities()
                v = v0.clone() if hasattr(v0, "clone") else np.asarray(v0, dtype=np.float64).copy()
                for j in touched:
                    v[int(j)] = 0.0
                self.robot.set_joint_velocities(v)
            except Exception as exc:
                self.log(f"RESET joint velocity zero warning: {exc}")
            if self.world is not None:
                try:
                    if trunk_changed:
                        self.world.set_trunk_pin_qpos(_RESET_TRUNK_UPRIGHT_QPOS)
                    else:
                        self.world.set_trunk_pin_qpos()
                except Exception as exc:
                    self.log(f"RESET trunk pin warning: {exc}")
                for arm in changed:
                    try:
                        self.world.set_arm_pin_qpos(arm, _RESET_GRASP_PREP_ARM_QPOS)
                    except Exception as exc:
                        self.log(f"RESET arm pin warning[{arm}]: {exc}")
                for arm, pin_qpos in tool_roll_pins.items():
                    try:
                        self.world.set_tool_roll_pin_qpos(arm, pin_qpos)
                    except Exception as exc:
                        self.log(f"RESET tool roll pin warning[{arm}]: {exc}")
            if trunk_changed:
                try:
                    ch = self.world.chest_pose() if self.world is not None else {}
                    self.log(
                        "RESET trunk init: upright LUT "
                        f"q={[round(float(x), 4) for x in _RESET_TRUNK_UPRIGHT_QPOS]} "
                        f"chest_z={float(ch.get('z', 0.0)):.3f} "
                        f"θz={float(ch.get('theta_z_deg', 0.0)):.1f}°"
                    )
                except Exception:
                    self.log(
                        "RESET trunk init: upright LUT "
                        f"q={[round(float(x), 4) for x in _RESET_TRUNK_UPRIGHT_QPOS]}"
                    )
            if changed:
                grip_qpos = self._force_open_gripper_joints(("left", "right"))
                self.log(
                    "RESET arm init: grasp prep "
                    f"arms={changed} q={[round(float(x), 4) for x in _RESET_GRASP_PREP_ARM_QPOS]} "
                    f"gripper=open qpos={grip_qpos}"
                )

    def _warmup_reset_hold_pins(self, n_steps: int = 8) -> None:
        """Reset/startup 后播放几拍 pinned action，防止无命令关节被物理步积分带走。"""
        if self.dry_run or self.env is None or self.world is None:
            return
        try:
            import torch as th
            import omnigibson as _og

            # warmup 只为让关节物理稳定，既不需要相机 obs，也不需要 RTX 渲染。
            # 1) no-obs：跳过每步昂贵的 _post_step/get_obs（含 head/wrist 分割 remap）；
            # 2) render_on_step(False)：让 og.sim.step 只跑纯物理子步、不渲染——
            #    这是 reset 里最大耗时块（24 步 ~0.25s/步，主要花在 RTX 渲染）的根因。
            # warmup 结束后 _prime_obs_after_warmup 会重新渲染并恢复相机。
            with _og.sim.render_on_step(False):
                for _ in range(max(1, int(n_steps))):
                    self._force_open_gripper_joints(("left", "right"))
                    hold = self.world.make_action(**self._gripper_open_action_kwargs(("left", "right")))
                    hold_t = th.from_numpy(hold.astype(np.float32))
                    self._step_action_no_obs(hold_t)
            grip_qpos = self._force_open_gripper_joints(("left", "right"))
            self.log(
                f"RESET hold pins: trunk/arms locked and grippers opened "
                f"for {int(n_steps)} warmup physics steps (no-obs/no-render) qpos={grip_qpos}"
            )
        except Exception as e:
            self.log(f"WARN reset hold pins failed: {e}")

    def _reset_env_for_interface(self) -> None:
        """Reset OG while avoiding a flaky flattened image observation check."""
        if self.env is None:
            return
        self.env.reset(get_obs=False)

    def _release_all_assisted_grasps(self) -> List[str]:
        """Remove every assisted-grasp constraint and stale private bookkeeping."""
        clear_keepalive = getattr(
            self.world,
            "clear_gripper_close_keepalive",
            None,
        )
        if callable(clear_keepalive):
            clear_keepalive()
        robot = self.robot
        if self.dry_run or robot is None:
            return []
        released: List[str] = []
        arms = list(getattr(robot, "arm_names", ()) or ("left", "right"))
        held_map = getattr(robot, "_ag_obj_in_hand", None)
        constraints = getattr(robot, "_ag_obj_constraints", None)
        params = getattr(robot, "_ag_obj_constraint_params", None)
        for arm in arms:
            held = held_map.get(arm) if isinstance(held_map, dict) else None
            if held is not None:
                released.append(
                    str(
                        getattr(held, "name", "")
                        or getattr(held, "prim_path", "")
                        or type(held).__name__
                    )
                )
            constraint = constraints.get(arm) if isinstance(constraints, dict) else None
            constraint_params = params.get(arm) if isinstance(params, dict) else {}
            try:
                if constraint is not None:
                    robot.release_grasp_immediately(arm=arm)
                elif constraint_params:
                    robot._release_grasp(arm=arm)
            except Exception as exc:
                self.log(f"WARN reset release grasp failed[{arm}]: {exc}")
            for attr, value in (
                ("_ag_obj_in_hand", None),
                ("_ag_obj_constraints", None),
                ("_ag_obj_constraint_params", {}),
                ("_ag_freeze_gripper", False),
                ("_ag_release_counter", None),
                ("_ag_grasp_counter", None),
            ):
                mapping = getattr(robot, attr, None)
                if isinstance(mapping, dict):
                    mapping[arm] = value.copy() if isinstance(value, dict) else value
        if released:
            self.log(f"RESET released assisted grasps: {released}")
        return released

    def _capture_world_reset_snapshot(self, reason: str) -> None:
        """Save an immutable full-scene restart checkpoint, including non-TRO objects."""
        if self.dry_run or self.env is None:
            return
        self._rebind_robot_handles_after_scene_restore(
            stage=f"capture_snapshot:{reason}"
        )
        scene_file = self.env.scene.save(as_dict=True)
        self._world_reset_scene_file = copy.deepcopy(scene_file)
        self.env.scene.update_initial_file(scene_file=copy.deepcopy(scene_file))
        registry = (
            ((scene_file.get("state") or {}).get("registry") or {}).get("object_registry")
            or {}
        )
        self.log(
            f"RESET_WORLD_SNAPSHOT captured reason={reason} objects={len(registry)} "
            f"instance={self.current_instance_id}"
        )

    def _world_reset_pose_errors(
        self,
        *,
        tolerance_m: float,
        tolerance_rad: float = math.radians(1.0),
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Compare every current non-robot object pose with the restart snapshot."""
        snapshot = self._world_reset_scene_file
        if self.dry_run or self.env is None or not isinstance(snapshot, dict):
            return []
        registry = (
            ((snapshot.get("state") or {}).get("registry") or {}).get("object_registry")
            or {}
        )
        robot_name = str(getattr(self.robot, "name", "") or "")
        errors: List[Dict[str, Any]] = []
        scene = self.env.scene
        try:
            current_names = set(scene.object_registry.get_dict("name").keys())
        except Exception:
            current_names = {
                str(getattr(obj, "name", ""))
                for obj in list(getattr(scene, "objects", ()) or ())
                if getattr(obj, "name", None)
            }
        expected_names = set(registry.keys())
        for name in sorted(current_names - expected_names - {robot_name}):
            errors.append({"name": name, "error": "unexpected"})
        for name, expected_state in registry.items():
            if name == robot_name:
                continue
            expected_root = (
                expected_state.get("root_link")
                if isinstance(expected_state, dict)
                else None
            )
            expected_pos = expected_root.get("pos") if isinstance(expected_root, dict) else None
            expected_ori = expected_root.get("ori") if isinstance(expected_root, dict) else None
            if expected_pos is None or expected_ori is None:
                continue
            obj = scene.object_registry("name", name)
            if obj is None:
                errors.append({"name": name, "error": "missing"})
                continue
            try:
                current_pos, current_ori = obj.get_position_orientation()
                if hasattr(current_pos, "detach"):
                    current_pos = current_pos.detach().cpu().numpy()
                if hasattr(current_ori, "detach"):
                    current_ori = current_ori.detach().cpu().numpy()
                if hasattr(expected_pos, "detach"):
                    expected_pos = expected_pos.detach().cpu().numpy()
                if hasattr(expected_ori, "detach"):
                    expected_ori = expected_ori.detach().cpu().numpy()
                delta = np.asarray(current_pos, dtype=np.float64).reshape(-1)[:3] - np.asarray(
                    expected_pos, dtype=np.float64
                ).reshape(-1)[:3]
                err_m = float(np.linalg.norm(delta))
                current_quat = np.asarray(current_ori, dtype=np.float64).reshape(-1)[:4]
                expected_quat = np.asarray(expected_ori, dtype=np.float64).reshape(-1)[:4]
                current_norm = float(np.linalg.norm(current_quat))
                expected_norm = float(np.linalg.norm(expected_quat))
                if current_norm <= 1e-12 or expected_norm <= 1e-12:
                    raise ValueError("zero-length orientation quaternion")
                quat_dot = abs(
                    float(
                        np.dot(
                            current_quat / current_norm,
                            expected_quat / expected_norm,
                        )
                    )
                )
                err_rad = 2.0 * math.acos(max(0.0, min(1.0, quat_dot)))
            except Exception as exc:
                errors.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if err_m > float(tolerance_m) or err_rad > float(tolerance_rad):
                errors.append(
                    {
                        "name": name,
                        "err_m": round(err_m, 6),
                        "err_deg": round(math.degrees(err_rad), 4),
                    }
                )

        def severity(item: Dict[str, Any]) -> float:
            if "error" in item:
                return float("inf")
            return max(
                float(item.get("err_m", 0.0)) / max(float(tolerance_m), 1e-12),
                math.radians(float(item.get("err_deg", 0.0)))
                / max(float(tolerance_rad), 1e-12),
            )

        errors.sort(key=severity, reverse=True)
        return errors[: max(1, int(limit))]

    @staticmethod
    def _scene_init_entry_is_robot(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        class_module = str(entry.get("class_module") or "")
        class_name = str(entry.get("class_name") or "")
        return class_module.startswith("omnigibson.robots.") or (
            class_module == "omnigibson.robots.robot" and class_name == "Robot"
        )

    @staticmethod
    def _rewrite_scene_agent_metadata(
        scene_file: Dict[str, Any],
        robot_name: str,
    ) -> None:
        metadata = scene_file.get("metadata")
        if not isinstance(metadata, dict):
            return
        candidates = [metadata]
        nested_task = metadata.get("task")
        if isinstance(nested_task, dict):
            candidates.append(nested_task)
        for container in candidates:
            inst_to_name = container.get("inst_to_name")
            if not isinstance(inst_to_name, dict):
                continue
            for key in list(inst_to_name):
                if "agent.n." in str(key):
                    inst_to_name[key] = robot_name

    def _scene_file_with_current_robot(
        self,
        scene_file: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a restore payload containing exactly the current configured robot."""
        if self.env is None:
            raise RuntimeError("cannot preserve robot without an active environment")

        robot_handle = self.robot or getattr(self.world, "robot", None)
        expected_name = str(getattr(robot_handle, "name", "") or "")
        sources: List[tuple[str, Dict[str, Any], bool]] = []
        try:
            current = self.env.scene.save(as_dict=True)
            if isinstance(current, dict):
                sources.append(("current_scene", current, False))
        except Exception as exc:
            self.log(
                "WARN cannot snapshot current robot before scene restore; "
                f"falling back to reset baseline: {type(exc).__name__}: {exc}"
            )
        baseline = getattr(self, "_world_reset_scene_file", None)
        if isinstance(baseline, dict):
            sources.append(("world_reset_baseline", baseline, True))

        robot_name = ""
        current_init = None
        current_state = None
        robot_source = ""
        for source_name, snapshot, allow_discovery in sources:
            init_info = (
                ((snapshot.get("objects_info") or {}).get("init_info") or {})
            )
            registry = (
                (((snapshot.get("state") or {}).get("registry") or {}).get(
                    "object_registry"
                ) or {})
            )
            candidate_names = [expected_name] if expected_name else []
            if allow_discovery:
                candidate_names.extend(
                    str(name)
                    for name, entry in init_info.items()
                    if self._scene_init_entry_is_robot(entry)
                    and str(name) not in candidate_names
                )
            for candidate_name in candidate_names:
                candidate_init = init_info.get(candidate_name)
                candidate_state = registry.get(candidate_name)
                if isinstance(candidate_init, dict) and candidate_state is not None:
                    robot_name = candidate_name
                    current_init = candidate_init
                    current_state = candidate_state
                    robot_source = source_name
                    break
            if robot_name:
                break

        if not robot_name or not isinstance(current_init, dict) or current_state is None:
            raise RuntimeError(
                "current scene and reset baseline are missing a usable robot "
                f"init/state entry (expected={expected_name!r})"
            )
        if robot_source != "current_scene":
            self.log(
                "RESET robot self-heal source="
                f"{robot_source} robot={robot_name}"
            )

        restored = copy.deepcopy(scene_file)
        restore_init = restored.setdefault("objects_info", {}).setdefault(
            "init_info", {}
        )
        restore_registry = (
            restored.setdefault("state", {})
            .setdefault("registry", {})
            .setdefault("object_registry", {})
        )
        old_robot_names = {
            str(name)
            for name, entry in list(restore_init.items())
            if self._scene_init_entry_is_robot(entry)
        }
        for old_name in old_robot_names:
            restore_init.pop(old_name, None)
            restore_registry.pop(old_name, None)

        restore_init[robot_name] = copy.deepcopy(current_init)
        restore_registry[robot_name] = copy.deepcopy(current_state)
        self._rewrite_scene_agent_metadata(restored, robot_name)
        return restored

    def _simulation_integrity_error(self) -> Optional[str]:
        if self.dry_run:
            return None
        if self.env is None:
            return "environment handle is missing"
        try:
            robots = list(self.env.robots)
        except Exception as exc:
            return f"cannot read env.robots: {type(exc).__name__}: {exc}"
        if len(robots) != 1:
            return f"expected exactly one scene robot, found {len(robots)}"

        robot = robots[0]
        robot_name = str(getattr(robot, "name", "") or "")
        if not robot_name:
            return "scene robot has no name"
        try:
            registered = self.env.scene.object_registry("name", robot_name)
        except Exception as exc:
            return (
                f"cannot query robot registry for {robot_name}: "
                f"{type(exc).__name__}: {exc}"
            )
        if registered is not robot:
            return f"robot registry entry {robot_name!r} does not match env.robots[0]"

        articulation_view = getattr(robot, "_articulation_view", None)
        if articulation_view is None:
            return f"robot {robot_name!r} has no articulation view"
        try:
            prim_count = int(getattr(articulation_view, "count"))
        except Exception as exc:
            return (
                f"cannot read robot articulation prim count for {robot_name}: "
                f"{type(exc).__name__}: {exc}"
            )
        if prim_count <= 0:
            return (
                f"robot {robot_name!r} articulation prim view is empty "
                f"(count={prim_count})"
            )

        try:
            physics_handle_valid = bool(articulation_view.is_physics_handle_valid())
        except Exception as exc:
            return (
                f"cannot validate robot physics handle for {robot_name}: "
                f"{type(exc).__name__}: {exc}"
            )
        if not physics_handle_valid:
            return f"robot {robot_name!r} articulation physics handle is invalid"

        physics_view = getattr(articulation_view, "_physics_view", None)
        if physics_view is None:
            return f"robot {robot_name!r} articulation physics view is missing"
        try:
            physics_count = int(getattr(physics_view, "count"))
        except Exception as exc:
            return (
                f"cannot read robot PhysX articulation count for {robot_name}: "
                f"{type(exc).__name__}: {exc}"
            )
        if physics_count <= 0:
            return (
                f"robot {robot_name!r} PhysX articulation registry is empty "
                f"(count={physics_count})"
            )
        try:
            root_transforms = physics_view.get_root_transforms()
            if root_transforms is None:
                return (
                    f"robot {robot_name!r} PhysX articulation root transforms "
                    "are unavailable"
                )
            shape = tuple(int(dim) for dim in root_transforms.shape)
            if not shape or shape[0] < physics_count:
                return (
                    f"robot {robot_name!r} PhysX articulation root transforms "
                    f"are empty or truncated (shape={shape}, count={physics_count})"
                )
        except Exception as exc:
            return (
                f"cannot read robot PhysX articulation root transforms for "
                f"{robot_name}: {type(exc).__name__}: {exc}"
            )
        return None

    def _rebind_robot_handles_after_scene_restore(self, *, stage: str) -> None:
        """Validate the robot registry after restore/reset and refresh aliases."""
        error = self._simulation_integrity_error()
        if error is not None:
            raise RuntimeError(
                f"simulation integrity check failed after {stage}: {error}"
            )
        robot = self.env.robots[0]
        self.robot = robot
        if self.world is not None:
            self.world.env = self.env
            self.world.robot = robot
        self._validate_robot_controller_contract()
        self.log(
            f"SIM_INTEGRITY_OK stage={stage} robot={robot.name} "
            f"prim_count={int(robot._articulation_view.count)} "
            f"physics_count={int(robot._articulation_view._physics_view.count)}"
        )

    def _simulation_degraded_frames(self, reason: str) -> Dict[str, np.ndarray]:
        detail = str(reason or "simulation integrity failure")
        return {
            "main": make_placeholder(
                self.main_w,
                self.main_h,
                f"simulation degraded: {detail}",
            ),
            "head": make_placeholder(
                self.head_w,
                self.head_h,
                "simulation degraded; reset or switch task",
            ),
            "left_wrist": make_placeholder(
                self.sub_w,
                self.sub_h,
                "simulation degraded",
            ),
            "right_wrist": make_placeholder(
                self.sub_w,
                self.sub_h,
                "simulation degraded",
            ),
        }

    def _enter_simulation_degraded(self, reason: str) -> None:
        first_entry = not getattr(self, "_simulation_degraded", False)
        self._simulation_degraded = True
        self._simulation_degraded_reason = str(reason)
        if getattr(self, "_simulation_degraded_since_ts", None) is None:
            self._simulation_degraded_since_ts = time.time()
        if self.world is not None:
            self.world._codex_fast_motion_no_obs = False
        cleared = self._clear_pending_skill_runtime()
        if all(
            hasattr(self, attr)
            for attr in ("main_w", "main_h", "head_w", "head_h", "sub_w", "sub_h")
        ):
            self._update_frames(
                self._simulation_degraded_frames(self._simulation_degraded_reason)
            )
        if first_entry:
            self.log(
                "ERROR simulation degraded; entering zero-step hold "
                f"reason={self._simulation_degraded_reason} "
                f"cleared_pending_skill={cleared}"
            )
        self._signal_sim_wake()

    def _leave_simulation_degraded(self) -> None:
        if not getattr(self, "_simulation_degraded", False):
            return
        old_reason = self._simulation_degraded_reason
        self._simulation_degraded = False
        self._simulation_degraded_reason = ""
        self._simulation_degraded_since_ts = None
        self.log(
            "simulation integrity recovered; leaving zero-step hold "
            f"previous_reason={old_reason}"
        )

    def _restore_world_reset_snapshot(self) -> None:
        """Reset task bookkeeping and restore the independently cached full scene."""
        snapshot = self._world_reset_scene_file
        if not isinstance(snapshot, dict):
            raise RuntimeError("full world reset snapshot is unavailable")
        self._release_all_assisted_grasps()
        self.env.scene.update_initial_file(scene_file=copy.deepcopy(snapshot))
        self._reset_env_for_interface()
        self._rebind_robot_handles_after_scene_restore(stage="world_snapshot_reset")
        errors = self._world_reset_pose_errors(tolerance_m=0.005)
        if errors:
            self.log(
                f"WARN RESET full snapshot first restore residual={errors}; "
                "forcing scene.restore()"
            )
            self._release_all_assisted_grasps()
            self.env.scene.restore(
                scene_file=copy.deepcopy(snapshot),
                update_initial_file=True,
            )
            self._rebind_robot_handles_after_scene_restore(
                stage="world_snapshot_forced_restore"
            )
            errors = self._world_reset_pose_errors(tolerance_m=0.005)
        if errors:
            raise RuntimeError(f"full world reset verification failed: {errors}")
        self.log("RESET_WORLD_VERIFY stage=restore residual=0")

    def _load_challenge_template(self, path: str, *, as_torch: bool) -> Dict[str, Any]:
        """Build a coherent task cache on top of the compatible stable scene."""
        import json as _json

        from omnigibson.macros import gm

        with open(path, "r", encoding="utf-8") as f:
            template = _json.load(f)
        if not isinstance(template, dict):
            raise RuntimeError(f"invalid challenge template at {path}")

        template, stats = challenge_build_stable_backed_task_scene(
            template, gm.DATA_PATH, self.scene_model
        )
        self.log(
            "challenge task scene built from stable baseline "
            f"shared={stats['shared_objects']} "
            f"task_only={len(stats['task_only_objects'])} "
            f"objects={stats['object_count']} "
            f"stable={stats['stable']} "
            f"task_hashes={stats['corrected_task_hashes']}"
        )

        if as_torch:
            from omnigibson.utils.python_utils import recursively_convert_to_torch

            template = recursively_convert_to_torch(template)
        return template

    def _challenge_template_path_for_instance(self, instance_id: int) -> str:
        """Resolve the on-disk challenge task template for a given instance."""
        from omnigibson.macros import gm

        mode = _challenge_2026_mode_for_instance(int(instance_id))
        path = challenge_template_path(
            gm.DATA_PATH,
            self.task_name,
            self.scene_model,
            mode,
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"challenge task template missing for reset: {path}"
            )
        return path

    def _restore_challenge_template_scene(self, instance_id: int) -> str:
        """Restore full-scene object poses from the authoritative template JSON.

        Memory snapshots can be polluted (e.g. after sim.stop/play). Template
        restore brings furniture and non-TRO objects back before TRO reload.
        """
        path = self._challenge_template_path_for_instance(int(instance_id))
        template = self._load_challenge_template(path, as_torch=True)
        template = self._scene_file_with_current_robot(template)
        self._release_all_assisted_grasps()
        self.env.scene.restore(
            scene_file=copy.deepcopy(template),
            update_initial_file=True,
        )
        self._rebind_robot_handles_after_scene_restore(
            stage=f"challenge_template_restore:{instance_id}"
        )
        self.log(
            "RESET_WORLD_TEMPLATE restored "
            f"task={self.task_name} scene={self.scene_model} instance={instance_id} "
            f"path={path}"
        )
        return path

    def _reset_world_to_task_initial(self) -> int:
        """Authoritative Restart World: template scene + current instance TRO.

        Unlike snapshot-only restore, this always rebuilds from on-disk challenge
        assets so polluted in-memory baselines cannot leave objects displaced.
        """
        if self.current_instance_id is None:
            raise RuntimeError("no current instance; cannot reset world")
        inst = int(self.current_instance_id)
        self._restore_challenge_template_scene(inst)
        if not self._load_task_instance(inst):
            raise RuntimeError(
                f"failed to reload task instance={inst} during reset world"
            )
        return inst

    def _hide_ceilings_for_current_env(self) -> None:
        """Hide ceilings for the elevated GTA camera."""
        if self.env is None:
            return
        try:
            hidden = 0
            for name in ("ceilings", "ceiling"):
                obj = self.env.scene.object_registry("name", name)
                if obj is not None:
                    obj.visible = False
                    hidden += 1
            for obj in self.env.scene.objects:
                if getattr(obj, "category", None) in ("ceilings", "roof"):
                    obj.visible = False
                    hidden += 1
            self.log(f"hid ceilings: {hidden} objects")
        except Exception as e:
            self.log(f"WARN hide ceiling failed: {e}")

    def _bind_world_handles_for_current_env(
        self,
        *,
        initializing: bool = False,
    ) -> None:
        """Refresh robot/world/sensor handles after creating a new OG Environment."""
        if self.env is None:
            self.robot = None
            self.gta_sensor = None
            self.world = WorldAPI(dry_run=self.dry_run, robot_dof=self.robot_dof)
            self.world.control_hz = float(self.target_hz)
            return
        self.robot = self.env.robots[0]
        self._validate_robot_controller_contract()
        try:
            from behavior_interface.challenge_base_mass import apply_challenge_base_mass

            self.challenge_base_mass = apply_challenge_base_mass(
                self.robot,
                log_fn=self.log,
                preserve_sim_state=not initializing,
            )
        except Exception as e:
            self.challenge_base_mass = {"ok": False, "error": str(e)}
            self.log(f"WARN challenge base mass align failed: {e}")
        effort_audit = getattr(self.robot, "_joint_effort_override_audit", {}) or {}
        if effort_audit:
            applied = [float(item["after"]) for item in effort_audit.values()]
            self.log(
                "robot strength profile applied "
                f"config={self.robot_config_path} joints={len(effort_audit)} "
                f"max_effort=[{min(applied):.1f},{max(applied):.1f}]"
            )
        self.world = WorldAPI(
            env=self.env,
            robot=self.robot,
            dry_run=False,
            robot_dof=self.robot_dof,
        )
        self.world.control_hz = float(self.target_hz)
        self.world._gpu_diag_log = self.log
        try:
            from behavior_interface.skills.eef import _ensure_world_pinned_actions

            _ensure_world_pinned_actions(self.world)
        except Exception as e:
            raise RuntimeError(
                "failed to install controller-only world action policy"
            ) from e
        if not getattr(self.world, "_codex_pinned_actions_v13", False):
            raise RuntimeError("controller-only world action policy did not activate")
        self.log(
            "runtime action policy installed: controller_action_only v13 "
            "official_ag_close_keepalive"
        )
        self.gta_sensor = None
        if getattr(self.env, "_external_sensors", None):
            for name, sensor in self.env._external_sensors.items():
                if name == "gta_view":
                    self.gta_sensor = sensor
                    break
        self._bind_camera_health_sources()
        self.log(f"sim ready; robot={self.robot.name} action_dim={self.robot.action_dim}")

    def _validate_robot_controller_contract(self) -> None:
        """Fail before stepping when a cached scene supplied incompatible controllers."""
        if self.dry_run or self.robot is None:
            return
        errors = _robot_controller_contract_errors(self.robot, self.robot_dof)
        if errors:
            detail = "; ".join(errors)
            raise RuntimeError(
                "R1Pro controller contract mismatch; refusing to step an unsafe robot: "
                f"{detail}. Cached scene robots must be filtered so robots[].controller_config is applied."
            )

    def _clear_task_dependent_caches(self) -> None:
        """Drop cached state that belongs to the previous task/world."""
        with self.state_lock:
            self._cached_robot_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0}
            self._cached_eef_pose = {}
            self._cached_tro = {}
            self._cached_memory = {}
            self._cached_memory_text = "(memory 未构建)"
            self._cached_goals = {
                "items": [],
                "satisfied": 0,
                "total": 0,
                "complete": False,
                "ok": False,
            }
            self._cached_scene_graph = None
            self._cached_scene_graph_text = "(未构建)"
            self._cached_scene_graph_summary = {}
            self._last_skill_results = {}
            self._last_skill_internal_results = {}
            self._skill_history.clear()
            self._skill_history_seq = 0
        self._world_reset_scene_file = None
        self._scene_graph_last_build_ts = 0.0

    def _post_task_env_loaded(self, *, render_warmup: int = 15) -> None:
        """Run the same interface-side preparation used after a cold env load/reset."""
        self._bind_world_handles_for_current_env()
        self._reset_env_for_interface()
        # 切换任务后同样加载 task instance（指定或随机），并设为 reset baseline。
        resolved_inst = self._resolve_instance_id(self.requested_instance_id)
        self._load_task_instance(resolved_inst)
        self._apply_reset_grasp_prep_pose()
        try:
            from behavior_interface.head_capture import setup_head_after_env_reset
            setup_head_after_env_reset(self.world, env=self.env, log_fn=self.log)
        except Exception as e:
            self.log(f"WARN head 相机配置/挂载失败: {e}")
        self._warmup_reset_hold_pins(n_steps=24)
        self._capture_world_reset_snapshot("task_switch")
        self._hide_ceilings_for_current_env()
        if _skip_headless_rtx_warmup():
            self.log("skip RTX warmup during headless startup")
        else:
            try:
                import omnigibson as _og
                self.log("warming up RTX render pipeline for external sensor...")
                for _ in range(max(0, int(render_warmup))):
                    _og.sim.render()
                self.log("RTX warmup done.")
            except Exception as e:
                self.log(f"WARN RTX warmup failed: {e}")
        # no-obs warmup 后预热相机 annotator，避免首个 env.step 取到空 seg buffer
        self._prime_obs_after_warmup()
        self._refresh_state_cache()
        self._refresh_goal_cache()
        self._refresh_memory_cache()
        self._maybe_rebuild_scene_graph(force=True)

    def _create_env_for_current_task(self) -> None:
        import omnigibson as og

        cfg = self._build_env_cfg()
        self.log(f"env cfg task={self.task_name} scene={self.scene_model}")
        self.env = og.Environment(configs=cfg)
        self._post_task_env_loaded(render_warmup=15)

    def _build_task_config_for_current_task(self) -> Dict[str, Any]:
        return dict(self._build_env_cfg().get("task") or {})

    def _switch_task_same_scene_in_place(self) -> None:
        """Update BehaviorTask without clearing the USD stage."""
        if self.env is None:
            raise RuntimeError("env is not initialized")
        import omnigibson as og

        cfg = self._build_task_config_for_current_task()
        self.log(f"env update_task activity_name={self.task_name}")
        if getattr(og, "sim", None) is not None and not og.sim.is_playing():
            og.sim.play()
        self.env.update_task(cfg)
        self._post_task_env_loaded(render_warmup=8)

    def _preclear_vision_sensor_backends(self) -> None:
        """Best-effort cleanup for Replicator annotators before og.clear().

        Isaac can leave a Replicator NodeObj invalid after long-running sensor use.
        Calling VisionSensor.remove() then fails while detaching annotators. For a
        full task recreate we are about to destroy all sensors anyway, so it is
        safer to detach what still works and then mark modalities/annotators empty.
        """
        try:
            from omnigibson.sensors.vision_sensor import VisionSensor
        except Exception as e:
            self.log(f"WARN task switch import VisionSensor failed: {e}")
            return
        try:
            sensors = list(getattr(VisionSensor, "SENSORS", {}).values())
        except Exception:
            sensors = []
        touched = 0
        for sensor in sensors:
            try:
                ann = getattr(sensor, "_annotators", None) or {}
                rp = getattr(sensor, "_render_product", None)
                for annotator in list(ann.values()):
                    if annotator is None:
                        continue
                    for target in (rp, [getattr(rp, "path", None)] if rp is not None else None):
                        if target is None:
                            continue
                        try:
                            annotator.detach(target)
                            break
                        except Exception:
                            pass
                for key in list(ann.keys()):
                    ann[key] = None
                try:
                    sensor._modalities = set()
                except Exception:
                    pass
                touched += 1
            except Exception as e:
                self.log(f"WARN task switch sensor preclear failed: {e}")
        if touched:
            self.log(f"TASK SWITCH pre-cleared VisionSensor backends: {touched}")

    def _recreate_og_env_for_current_task(self) -> None:
        """Clear the current OG stage and create an Environment for self.task_name."""
        import omnigibson as og

        self.env = None
        self.robot = None
        self.gta_sensor = None
        self.world = WorldAPI(dry_run=self.dry_run, robot_dof=self.robot_dof)
        self.world.control_hz = float(self.target_hz)
        try:
            if getattr(og, "sim", None) is not None:
                og.sim.stop()
        except Exception as e:
            self.log(f"WARN og.sim.stop before task switch failed: {e}")
        self._preclear_vision_sensor_backends()
        try:
            if getattr(og, "sim", None) is not None:
                og.clear()
        except AttributeError as e:
            # This mirrors OG's own batch task sampler: sporadic AttributeError
            # during clear can be benign, and Environment creation may still work.
                self.log(f"WARN og.clear AttributeError during task switch: {e}")
        self._create_env_for_current_task()

    def _argv_for_task_reexec(self, task_name: str, scene_model: str) -> List[str]:
        """Return argv for restarting this interface process with a new task/scene."""
        old_argv = list(self._cli_argv or sys.argv)
        argv = [sys.executable, "-u", "-m", "behavior_interface.server"] + old_argv[1:]

        def set_opt(args: List[str], opt: str, value: str) -> None:
            for i, arg in enumerate(args):
                if arg == opt:
                    if i + 1 < len(args):
                        args[i + 1] = value
                    else:
                        args.append(value)
                    return
                if arg.startswith(opt + "="):
                    args[i] = opt + "=" + value
                    return
            args.extend([opt, value])

        set_opt(argv, "--task", task_name)
        set_opt(argv, "--scene", scene_model)
        set_opt(argv, "--robot", self.robot_name)
        set_opt(argv, "--robot-dof", str(self.robot_dof))
        set_opt(argv, "--robot-config", self.robot_config_path)
        return argv

    def _reexec_for_task_switch(self, task_name: str, scene_model: str) -> None:
        """Cleanly restart the current process for task switching in real sim mode.

        Isaac / Replicator / BDDL keep process-global caches that do not survive
        dozens of in-process og.clear() cycles reliably. Start a clean Python
        process with only stdio inherited, then terminate this old runtime.
        """
        argv = self._argv_for_task_reexec(task_name, scene_model)
        from .runtime_tmp import prepare_child_environment

        env = prepare_child_environment(os.environ.copy())
        env["INTERFACE_TASK_SWITCH_REEXEC"] = "1"
        env["INTERFACE_TASK_SWITCH_TARGET"] = f"{task_name}@{scene_model}"
        env["BEHAVIOR_ROBOT_DOF"] = str(self.robot_dof)
        env["ROBOT"] = self.robot_name
        env["BEHAVIOR_ROBOT_CONFIG"] = self.robot_config_path
        env.setdefault("INTERFACE_TASK_SWITCH_READY_TIMEOUT_S", "3600")
        if not _env_truthy("INTERFACE_TASK_SWITCH_DISABLE_APPDATA_ISOLATION", False):
            env["INTERFACE_TASK_SWITCH_ISOLATE_APPDATA"] = "retry"
        try:
            port = int(_supervisor_arg_value(argv, "--port", os.environ.get("PORT", "5000")) or "5000")
        except (TypeError, ValueError):
            port = 5000
        appdata_base = _task_switch_appdata_base(env, port)
        env["INTERFACE_TASK_SWITCH_APPDATA_BASE"] = appdata_base
        env["OMNIGIBSON_APPDATA_PATH"] = appdata_base
        env.setdefault("INTERFACE_LOG_PATH", _interface_log_path_for_port(port))
        env.pop("WERKZEUG_SERVER_FD", None)
        env.pop("LISTEN_FDS", None)
        env.pop("LISTEN_PID", None)
        self.task_switch_reexec_requested = True
        self.task_switch_reexec_target = {"task": task_name, "scene": scene_model}
        supervisor_argv = [
            sys.executable,
            "-u",
            "-m",
            "behavior_interface.server",
            "--task-switch-supervisor",
            "--",
            *argv,
        ]
        self.log(f"TASK SWITCH restart: argv={' '.join(argv)}")
        self.log(f"TASK SWITCH supervisor: argv={' '.join(supervisor_argv)}")
        sys.stdout.flush()
        sys.stderr.flush()
        import subprocess

        stdio = _open_interface_stdio_for_port(port)
        try:
            child = subprocess.Popen(
                supervisor_argv,
                cwd=os.getcwd(),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdio,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
        finally:
            stdio.close()
        self.log(f"TASK SWITCH spawned supervisor pid={child.pid}; old pid exiting")
        sys.stdout.flush()
        sys.stderr.flush()
        # os._exit deliberately bypasses atexit and OmniGibson's cleanup().
        # Remove only this process's token-verified TMPDIR before handing over.
        try:
            from .runtime_tmp import cleanup_process_runtime_tmp

            cleanup_process_runtime_tmp()
        except Exception as exc:
            self.log(f"TASK SWITCH runtime TMPDIR cleanup skipped: {exc}")
        os._exit(0)

    def _maybe_do_task_switch(self) -> bool:
        """主 sim 线程执行 task 热切换；返回 True 表示本拍处理了切换请求。"""
        with self.skill_lock:
            req = self.task_switch_request
            if req is None:
                return False
            self.task_switch_request = None
            self.task_switch_in_progress = True
            self.reset_request = False

        # 任务切换同样要清除躯干保持器，避免它把切换后的关节状态覆盖回旧姿态。
        self._clear_normal_trunk_hold("task_switch")

        new_task = str(req.get("task") or "").strip()
        new_scene = str(req.get("scene") or self.scene_model).strip() or self.scene_model
        old_task_id = self.task_id
        old_task = self.task_name
        old_scene = self.scene_model
        cleared = 0
        switch_t0 = time.time()
        try:
            from .challenge_tasks import challenge_task_by_name

            new_task_meta = challenge_task_by_name(new_task)
            if new_task_meta is None:
                raise ValueError(f"unknown BEHAVIOR 2026 task: {new_task!r}")
            expected_scene = str(new_task_meta["scene"])
            if new_scene != expected_scene:
                raise ValueError(
                    f"scene mismatch for BEHAVIOR 2026 task id={new_task_meta['id']} "
                    f"{new_task}: expected {expected_scene}, got {new_scene}"
                )
            new_task_id = int(new_task_meta["id"])
            cleared = self._clear_pending_skill_runtime()
            self._clear_task_dependent_caches()
            self._update_frames({
                "main": make_placeholder(self.main_w, self.main_h, f"switching task: {new_task}"),
                "head": make_placeholder(self.head_w, self.head_h, f"switching task: {new_task}"),
                "left_wrist": make_placeholder(self.sub_w, self.sub_h, "switching task"),
                "right_wrist": make_placeholder(self.sub_w, self.sub_h, "switching task"),
            })
            self.log(
                f"TASK SWITCH 开始: {old_task} -> {new_task} "
                f"(scene {old_scene} -> {new_scene}); cleared pending skill={cleared}"
            )

            self.task_id = new_task_id
            self.task_name = new_task
            self.scene_model = new_scene
            if self.dry_run:
                self.task_switch_count += 1
                self._refresh_goal_cache()
                self._refresh_memory_cache()
                self.log(f"TASK SWITCH 完成 dry-run (#{self.task_switch_count}) task={self.task_name}")
                return True

            same_scene_inplace = (
                new_scene == old_scene
                and os.environ.get("INTERFACE_TASK_SWITCH_SAME_SCENE_INPLACE", "0").strip().lower()
                not in {"0", "false", "no", "off"}
            )
            if same_scene_inplace:
                try:
                    try:
                        from behavior_interface.skills.plan_grasp_core import clear_plan_viz_prims

                        clear_plan_viz_prims()
                    except Exception:
                        pass
                    self.log(
                        f"TASK SWITCH same-scene in-place start: "
                        f"{old_task} -> {new_task} scene={new_scene}"
                    )
                    self._switch_task_same_scene_in_place()
                    self.task_switch_count += 1
                    self._leave_simulation_degraded()
                    sg_sum = self.get_scene_graph_summary()
                    goals = self._cached_goals
                    elapsed = time.time() - switch_t0
                    self.log(
                        f"TASK SWITCH 完成 in-place (#{self.task_switch_count}) "
                        f"task={self.task_name} scene={self.scene_model} "
                        f"elapsed={elapsed:.1f}s "
                        f"goals={goals.get('satisfied', 0)}/{goals.get('total', 0)} "
                        f"memory_objects={self.get_memory_summary().get('objects', 0)} "
                        f"scene_graph_objects={sg_sum.get('objects')}"
                    )
                    return True
                except Exception as inplace_exc:
                    tb = traceback.format_exc()
                    self.log(
                        f"WARN TASK SWITCH same-scene in-place failed; "
                        f"falling back to re-exec: {inplace_exc}\n{tb}"
                    )
                    self._clear_task_dependent_caches()

            self._reexec_for_task_switch(new_task, new_scene)

            try:
                from behavior_interface.skills.plan_grasp_core import clear_plan_viz_prims
                clear_plan_viz_prims()
            except Exception:
                pass
            # Behavior cached scene_instance depends on activity_name, not just
            # scene_model. A different task can require different task-relevant
            # objects / metadata in the same house, so equivalent restart means
            # full env recreate whenever task changes.
            self._recreate_og_env_for_current_task()
            self.task_switch_count += 1
            self._leave_simulation_degraded()
            sg_sum = self.get_scene_graph_summary()
            goals = self._cached_goals
            self.log(
                f"TASK SWITCH 完成 (#{self.task_switch_count}) task={self.task_name} "
                f"goals={goals.get('satisfied', 0)}/{goals.get('total', 0)} "
                f"memory_objects={self.get_memory_summary().get('objects', 0)} "
                f"scene_graph_objects={sg_sum.get('objects')}"
            )
        except Exception as e:
            tb = traceback.format_exc()
            self.task_switch_error = str(e)
            self.log(f"TASK SWITCH 失败: {e}\n{tb}")
            if not self.dry_run:
                try:
                    self.log(f"TASK SWITCH rollback: restore {old_task} scene={old_scene}")
                    self.task_id = old_task_id
                    self.task_name = old_task
                    self.scene_model = old_scene
                    self._recreate_og_env_for_current_task()
                    self._refresh_state_cache()
                    self._refresh_goal_cache()
                    self._refresh_memory_cache()
                    self._maybe_rebuild_scene_graph(force=True)
                    self._leave_simulation_degraded()
                    self.log("TASK SWITCH rollback 完成")
                except Exception as e2:
                    tb2 = traceback.format_exc()
                    self.log(f"TASK SWITCH rollback 失败: {e2}\n{tb2}")
            return True
        finally:
            with self.skill_lock:
                self.task_switch_in_progress = False
        return True

    def _clear_normal_trunk_hold(self, reason: str = "") -> None:
        """清除 normal 模式躯干+双臂持续保持器。reset / task switch / skill 启动等任何
        会改变机器人关节的入口都必须调用，否则保持器会每步把关节重设回旧 qpos，
        把 env.reset / 新动作覆盖掉。"""
        if self.world is None:
            return
        if getattr(self.world, "_codex_normal_trunk_hold", False):
            try:
                self.world.shortcut_post_step_stabilize_now = None
            except Exception:
                pass
            self.world._codex_normal_trunk_hold = False
            self.log(f"[normal_hold] 清除躯干保持器（{reason}）")

    def _maybe_do_reset(self) -> bool:
        """主 sim 线程调用。如果有 reset 请求，就清空 skill 队列、cancel 当前 skill、reset env。
        返回 True 表示本拍执行了 reset。"""
        with self.skill_lock:
            if not self.reset_request:
                return False
            self.reset_request = False
            inst_req = self._reset_instance_request
            self._reset_instance_request = _INSTANCE_UNSET
        # 关键：先清除躯干保持器，否则它每个物理步会把关节重设回上次 move 的姿态，
        # 把 env.reset 的关节复位覆盖掉（表现为 reset 后机器人没回直立）。
        self._clear_normal_trunk_hold("reset")
        reset_cache_cleanup_done = False
        try:
            self.log("RESET 开始：清队列 + cancel 当前 skill + env.reset()")
            cleared = self._clear_pending_skill_runtime()
            with self.state_lock:
                self._last_skill_results = {}
                self._last_skill_internal_results = {}
                self._skill_history.clear()
                self._skill_history_seq = 0

            if not self.dry_run and self.env is not None:
                try:
                    from behavior_interface.skills.plan_grasp_core import clear_plan_viz_prims
                    clear_plan_viz_prims()
                except Exception:
                    pass
                switching_instance = inst_req is not _INSTANCE_UNSET
                if switching_instance:
                    self._release_all_assisted_grasps()
                    resolved_inst = int(self._resolve_instance_id(inst_req))
                    # 先从磁盘 template 恢复整场，再叠 TRO，避免脏 baseline 残留
                    self._restore_challenge_template_scene(resolved_inst)
                    self._load_task_instance(resolved_inst)
                    snapshot_reason = "instance_switch"
                else:
                    # Restart World 按钮：必须走权威 template+instance，不能只 restore
                    # 可能已被 stop/play 污染的内存 snapshot。
                    self._reset_world_to_task_initial()
                    snapshot_reason = "reset_world"
                self._apply_reset_grasp_prep_pose()
                try:
                    from behavior_interface.head_capture import setup_head_after_env_reset
                    setup_head_after_env_reset(self.world, env=self.env, log_fn=self.log)
                except Exception as e:
                    self.log(f"WARN head 相机配置/挂载失败: {e}")
                self._warmup_reset_hold_pins(n_steps=24)
                # 用刚刚权威重建后的状态刷新不可变 baseline，供后续漂移二次 restore。
                self._capture_world_reset_snapshot(snapshot_reason)
                errors = self._world_reset_pose_errors(tolerance_m=0.02)
                if errors:
                    self.log(
                        f"WARN RESET post-warmup world drift={errors}; "
                        "restoring full snapshot once more"
                    )
                    self._release_all_assisted_grasps()
                    self.env.scene.restore(
                        scene_file=copy.deepcopy(self._world_reset_scene_file),
                        update_initial_file=True,
                    )
                    self._rebind_robot_handles_after_scene_restore(
                        stage="post_warmup_snapshot_restore"
                    )
                    self._apply_reset_grasp_prep_pose()
                    errors = self._world_reset_pose_errors(tolerance_m=0.005)
                if errors:
                    raise RuntimeError(f"post-warmup world reset verification failed: {errors}")
                self.log("RESET_WORLD_VERIFY stage=post_warmup residual=0")
                # no-obs warmup 后预热相机 annotator（render + get_obs），既避免 GTA
                # 灰图，也避免主循环首个 env.step 取到空 seg buffer 崩溃。
                self._prime_obs_after_warmup()

            self.reset_count += 1
            for fn in list(getattr(self, "_after_world_reset_hooks", []) or []):
                try:
                    fn()
                except Exception as exc:
                    self.log(f"WARN after_world_reset hook: {exc}")
            self._refresh_state_cache()
            self._refresh_goal_cache()
            # reset 之后立刻重建一次 Scene Graph（物体位置变了）
            self._maybe_rebuild_scene_graph(force=True)
            self._release_reset_runtime_caches()
            reset_cache_cleanup_done = True
            self._leave_simulation_degraded()
            self.log(f"RESET 完成 (#{self.reset_count})，清空 pending skill={cleared}")
        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"RESET 失败: {e}\n{tb}")
            self._enter_simulation_degraded(f"reset failed: {e}")
        finally:
            if not reset_cache_cleanup_done:
                try:
                    self._release_reset_runtime_caches()
                except Exception as e:
                    self.log(f"WARN reset cache cleanup failed: {e}")
            try:
                self._maybe_release_native_memory(reason="reset", force=True)
            except Exception as e:
                self.log(f"WARN reset memory cleanup failed: {e}")
        return True

    def _maybe_start_next_skill(self) -> None:
        if self.current_job is not None:
            return
        if getattr(self, "_simulation_degraded", False):
            return
        head_health = self.camera_health.get("head")
        fast_no_obs = bool(
            getattr(self.world, "_codex_fast_motion_no_obs", False)
        )
        if head_health is not None and head_health.stale() and not fast_no_obs:
            self._enter_vision_safe_state("head frame is stale")
        if self._vision_safe_state:
            return
        try:
            job = self.skill_queue.get_nowait()
        except queue.Empty:
            return
        from behavior_interface.skills import SKILL_REGISTRY as _skill_reg
        spec = _skill_reg.get(job.name)
        if spec is None:
            job.status = "failed"
            job.message = f"unknown skill {job.name}"
            self.log(job.message)
            return
        # normal 模式躯干+双臂保持器：move 结束后每步运动学重设关节，防止位置控制器
        # 撑不住深俯角而下垂/塌低。除 capture（仅渲染、需保持该姿态）外，任何 skill
        # 开始时都清除它，避免保持器与新动作（尤其动臂/动躯干）打架。
        if job.name != "capture":
            self._clear_normal_trunk_hold(f"skill={job.name}")
        self._last_skill_internal_results.pop(job.name, None)
        # 闭包式 setter/getter：捕获当前 job + last_results
        def _set_result_for_job(payload: Dict[str, Any], _job=job):
            out = dict(payload)
            out.setdefault("job", _job.request_id)
            was_detached_capture = bool(
                isinstance(_job.result, dict)
                and _job.result.get("_official_capture_pending_token")
            )
            restricted_memory = _task_goals_only_observation_mode()
            if restricted_memory:
                with self.state_lock:
                    goals = dict(self._cached_goals)
                out = _sanitize_model_payload_for_mode(
                    out,
                    task_name=self.task_name,
                    goals=goals,
                )
            _job.result = out
            self._last_skill_results[_job.name] = out
            if _job.name == "capture":
                mem = out.get("memory")
                text = out.get("memory_text")
                if isinstance(mem, dict) and isinstance(text, str):
                    mem = dict(mem)
                    if not restricted_memory:
                        mem.setdefault("task", self.task_name)
                    with self.state_lock:
                        self._cached_memory = mem
                        self._cached_memory_text = text
            # A detached capture records its job history as soon as the
            # evaluator callback is released.  Replace only that private
            # pending marker when the HTTP waiter installs the real result;
            # all other historical records remain immutable.
            if was_detached_capture:
                safe_out = self._history_safe(out)
                with self.state_lock:
                    for record in reversed(self._skill_history):
                        if record.get("request_id") == _job.request_id:
                            record["result"] = safe_out
                            break
            return out

        def _set_internal_result_for_job(payload: Dict[str, Any], _job=job):
            out = dict(payload)
            out.setdefault("job", _job.request_id)
            self._last_skill_internal_results[_job.name] = out

        def _get_last_result(skill_name: str):
            return (
                self._last_skill_internal_results.get(skill_name)
                or self._last_skill_results.get(skill_name)
            )

        def _cancel_check():
            with self.skill_lock:
                return bool(self.cancel_flag)

        def _adjust_camera(
            deltas=None, overrides=None, reset=False,
        ):
            return self.adjust_camera(
                deltas=deltas, overrides=overrides, reset=reset,
            )

        ctx = SkillContext(
            world=self.world,
            _logger=self.log,
            _set_status=self._set_skill_status,
            _set_result=_set_result_for_job,
            _set_internal_result=_set_internal_result_for_job,
            _get_last_result=_get_last_result,
            _cancel_check=_cancel_check,
            _adjust_camera=_adjust_camera,
            task_name=self.task_name,
        )
        try:
            gen = spec.fn(ctx, **job.args)
        except TypeError as e:
            job.status = "failed"
            job.message = f"参数错误: {e}"
            self.log(job.message)
            job.result = {
                "ok": False,
                "error": job.message,
                "skill": job.name,
                "job": job.request_id,
            }
            job.ended_ts = time.time()
            self._last_skill_results[job.name] = job.result
            self._record_skill_history(job, status="failed", result=job.result)
            return
        # skill 必须返回 generator（yield action）
        if not hasattr(gen, "__next__"):
            job.status = "failed"
            job.message = f"skill {job.name} 没有 yield action（必须是 generator）"
            self.log(job.message)
            job.result = {
                "ok": False,
                "error": job.message,
                "skill": job.name,
                "job": job.request_id,
            }
            job.ended_ts = time.time()
            self._last_skill_results[job.name] = job.result
            self._record_skill_history(job, status="failed", result=job.result)
            return
        job.gen = gen
        job.status = "running"
        job.started_ts = time.time()
        self.current_job = job
        self.cancel_flag = False
        with self.state_lock:
            self._pending_skill_hint = None
        from behavior_interface.v2_display import skill_display_name
        job.display_name = skill_display_name(job.name, job.args)
        self._set_skill_status(f"start {job.display_name}")
        if self._should_log_skill_gpu_diag(job.name):
            self._log_gpu_diag(
                "skill.start",
                extra={
                    "skill": job.name,
                    "display": job.display_name,
                    "job": job.request_id,
                    "args": repr(job.args)[:800],
                },
                include_nvidia=False,
            )

    def _tick_skill(self) -> Optional[np.ndarray]:
        """从当前 skill 拿一步 action（若有）。完成 / 异常时清理。"""
        if self.current_job is None:
            return None
        if self.cancel_flag:
            job = self.current_job
            job.status = "cancelled"
            dn = getattr(job, "display_name", None) or job.name
            cancel_error = "vision_degraded" if self._vision_safe_state else "cancelled"
            self.log(
                f"cancelled {dn}"
                + (
                    f" due to {self._vision_safe_reason}"
                    if self._vision_safe_state
                    else ""
                )
            )
            if self._should_log_skill_gpu_diag(job.name):
                self._log_gpu_diag(
                    "skill.cancelled",
                    extra={"skill": job.name, "display": dn, "job": job.request_id},
                    include_nvidia=False,
                )
            try:
                if job.gen is not None:
                    job.gen.close()
            except Exception:
                pass
            self._last_skill_results[job.name] = {
                "ok": False,
                "error": cancel_error,
                "skill": job.name,
                "job": job.request_id,
                "detail": self._vision_safe_reason if self._vision_safe_state else "",
            }
            job.result = self._last_skill_results[job.name]
            job.ended_ts = time.time()
            self._record_skill_history(job, status="cancelled", result=job.result)
            self._last_finished_skill_name = job.name
            self.current_job = None
            self.cancel_flag = False
            self.skill_status_msg = ""
            return self.world.set_base_velocity(0.0, 0.0, 0.0)
        try:
            action = next(self.current_job.gen)
            return action
        except StopIteration:
            job = self.current_job
            dn = getattr(job, "display_name", None) or job.name
            self.log(f"done {dn}")
            if self._should_log_skill_gpu_diag(job.name):
                self._log_gpu_diag(
                    "skill.done",
                    extra={"skill": job.name, "display": dn, "job": job.request_id},
                    include_nvidia=False,
                )
            if job.result is not None and job.result.get("ok") is False:
                err = job.result.get("error") or job.result.get("message") or "skill 返回 ok=false"
                self.log(f"  ↳ {err}")
            job.status = "done"
            # 生成器型 skill（如 move/move_to）若未通过 ctx.set_result 写结果，
            # 这里补一个带唯一 job id 的默认成功结果，
            # 否则 wait_for_skill_result 会一直等不到「新结果」而超时。
            if job.result is None:
                self._last_skill_results[job.name] = {
                    "ok": True,
                    "skill": job.name,
                    "job": job.request_id,
                }
                job.result = self._last_skill_results[job.name]
            job.ended_ts = time.time()
            self._record_skill_history(job, status="done", result=job.result)
            self._last_finished_skill_name = job.name
            self.current_job = None
            return None
        except Exception as e:
            from behavior_interface.errors import SkillCancelled

            job = self.current_job
            if isinstance(e, SkillCancelled):
                job.status = "cancelled"
                self.log(f"cancelled {getattr(job, 'display_name', None) or job.name}: {e}")
                if self._should_log_skill_gpu_diag(job.name):
                    self._log_gpu_diag(
                        "skill.cancelled",
                        extra={
                            "skill": job.name,
                            "display": getattr(job, "display_name", None) or job.name,
                            "job": job.request_id,
                            "detail": str(e),
                        },
                        include_nvidia=False,
                    )
                self._last_skill_results[job.name] = {
                    "ok": False,
                    "error": "cancelled",
                    "skill": job.name,
                    "job": job.request_id,
                    "detail": str(e),
                }
                job.result = self._last_skill_results[job.name]
                job.ended_ts = time.time()
                self._record_skill_history(job, status="cancelled", result=job.result)
                self._last_finished_skill_name = job.name
                self.current_job = None
                self.cancel_flag = False
                self.skill_status_msg = ""
                return self.world.set_base_velocity(0.0, 0.0, 0.0)
            tb = traceback.format_exc()
            self.log(f"skill error: {e}\n{tb}")
            job.status = "failed"
            job.message = str(e)
            if self._should_log_skill_gpu_diag(job.name):
                self._log_gpu_diag(
                    "skill.failed",
                    extra={
                        "skill": job.name,
                        "display": getattr(job, "display_name", None) or job.name,
                        "job": job.request_id,
                        "error": f"{type(e).__name__}: {e}",
                    },
                )
            # 写入失败结果，避免 CLI/批量脚本空等到超时。
            # 带唯一 job id，否则相同报错时 wait_for_skill_result 因 cur==prev 判不出「新结果」而超时。
            self._last_skill_results[job.name] = {
                "ok": False,
                "error": str(e),
                "step": str(job.args.get("step", "")),
                "job": job.request_id,
            }
            job.result = self._last_skill_results[job.name]
            job.ended_ts = time.time()
            self._record_skill_history(job, status="failed", result=job.result)
            self._last_finished_skill_name = job.name
            self.current_job = None
            return self.world.set_base_velocity(0.0, 0.0, 0.0)

    def _step_action_no_obs(self, action_t) -> None:
        """Physics/controller step without camera obs.

        This keeps ordinary action execution intact (robot.apply_action +
        og.sim.step) but skips env._post_step/get_obs, which is too expensive
        for closed-loop base motion and can fail on transient segmentation
        buffers. Skills opt in with world._codex_fast_motion_no_obs.
        """
        import omnigibson as _og

        _sd = getattr(self, "_step_diag", None)
        _p0 = time.time()
        self.env._pre_step(action_t)
        if isinstance(_sd, dict):
            _sd["prestep"] += time.time() - _p0
        # 性能修复：sim._render_on_step 默认 True，导致这里每个物理子步都会把
        # 全部相机 render product（GTA 主视图 + head + 双腕）完整 RTX 渲染一遍，
        # 而这些像素没人消费（取帧走 tick 末尾的显式 sim.render()）。本函数日志
        # 一直自称 no-obs/no-render，但渲染从未真正关掉。用 OG 自带的
        # render_on_step(False) 关掉：两条分支推进的仿真时间一致，物理语义不变。
        _s0 = time.time()
        with _og.sim.render_on_step(False):
            _og.sim.step()
        hard_lock_tool_roll = getattr(self.world, "hard_lock_tool_roll_pins", None)
        if callable(hard_lock_tool_roll):
            hard_lock_tool_roll()
        if isinstance(_sd, dict):
            _sd["simstep"] += time.time() - _s0
        cur = int(getattr(self.env, "_current_step", 0) or 0)
        self.env._current_step = cur + 1

    def _sync_fast_motion_camera_rendering(
        self,
        fast_no_obs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Keep head/wrist Hydra textures alive across fast no-observation motion."""
        if self.dry_run or self.world is None:
            return {"ok": True, "changed": False, "enabled": True}
        if fast_no_obs is None:
            fast_no_obs = bool(getattr(self.world, "_codex_fast_motion_no_obs", False))
        report = set_robot_camera_render_updates(self.world, enabled=True)
        if report.get("changed"):
            state = "resumed" if report.get("enabled") else "paused"
            if report.get("previous") is not None:
                self.log(
                    f"fast no_obs robot camera render products {state}: "
                    f"{', '.join(report.get('sensors') or [])}"
                )
            if report.get("enabled"):
                try:
                    import omnigibson as _og

                    with self.camera_io_lock:
                        for _ in range(3):
                            _og.sim.render()
                except Exception as exc:
                    self.log(f"WARN robot camera resume warmup failed: {exc}")
        elif report.get("errors"):
            self.log(
                "WARN fast no_obs robot camera render switch failed: "
                + "; ".join(report["errors"])
            )
        return report

    def _prime_obs_after_warmup(
        self,
        n_render: int = 3,
        n_attempts: int = 5,
        *,
        allow_headless_skip: bool = True,
    ) -> None:
        """no-obs warmup 之后预热相机 annotator。

        长时间不取 obs（reset 的 no-obs warmup / 运动）后，分割 annotator 第一次
        get_obs 会返回空张量，主循环首个 normal env.step 会在 _remap 里 th.max(empty)
        崩溃。这里渲染若干帧再 get_obs（带重试），把 annotator 预热到可用状态。
        """
        if self.dry_run or self.env is None:
            return
        set_robot_camera_render_updates(self.world, True)
        if allow_headless_skip and _skip_headless_obs_prime():
            self.log("skip obs prime during headless startup")
            self._log_gpu_diag("obs_prime.skipped", extra={"reason": "headless_startup"})
            return
        try:
            import omnigibson as _og
        except Exception:
            return
        for attempt in range(max(1, int(n_attempts))):
            try:
                with self.camera_io_lock:
                    for _ in range(max(1, int(n_render))):
                        _og.sim.render()
                    self.env.get_obs()
                return
            except Exception as e:
                if attempt == 0:
                    self.log(f"RESET obs 预热重试: {e}")
        self.log("WARN RESET obs 预热未完全成功")

    # ------------------------------------------------------------------ state

    def snapshot_state(self) -> Dict[str, Any]:
        """供 web 线程调用，只读 _cached_*；绝不能在这里访问 PhysX。"""
        with self.state_lock:
            base_pose = dict(self._cached_robot_pose)
            eef_pose = dict(self._cached_eef_pose)
            tro = dict(self._cached_tro)
            skill_blob = None
            if self.current_job is not None:
                dn = getattr(self.current_job, "display_name", None) or self.current_job.name
                skill_blob = {
                    "name": dn,
                    "skill": self.current_job.name,
                    "args": dict(self.current_job.args),
                    "status": self.skill_status_msg,
                    "request_id": self.current_job.request_id,
                    "state": "running",
                }
            elif getattr(self, "_pending_skill_hint", None):
                skill_blob = dict(self._pending_skill_hint)
                skill_blob["state"] = "queued"
            # 把每个 skill 的最近一次 result 也对外暴露，
            # web/CLI 可以读 get_grasp_position 的候选列表给 execute_grasp 用
            last_results = dict(self._last_skill_results)
            skill_history = list(self._skill_history)
            goals = dict(self._cached_goals)
        with self.skill_lock:
            task_switch_pending = self.task_switch_request is not None or self.task_switch_in_progress
            task_switch_error = self.task_switch_error
            task_switch_reexec = self.task_switch_reexec_requested
            task_switch_reexec_target = self.task_switch_reexec_target
        if _task_goals_only_observation_mode():
            tro = {}
            last_results = _sanitize_model_payload_for_mode(
                last_results,
                task_name=self.task_name,
                goals=goals,
            )
            skill_history = _sanitize_model_payload_for_mode(
                skill_history,
                task_name=self.task_name,
                goals=goals,
            )
            scene_graph_summary = {}
        else:
            scene_graph_summary = self.get_scene_graph_summary()
        return {
            "pid": os.getpid(),
            "started_ts": self.started_ts,
            "task_id": self.task_id,
            "task": self.task_name,
            "robot": self.robot_name,
            "robot_dof": self.robot_dof,
            "scene": self.scene_model,
            "instance_id": self.current_instance_id,
            "mode": "dry-run" if self.dry_run else "real",
            "tick": self.tick,
            "fps": self.fps,
            # Environment/scheduler-only ownership proof. It does not touch
            # PhysX or create a CUDA context from the Flask thread.
            "gpu_mapping": resource_ownership_snapshot(),
            "base_pose": base_pose,
            "eef_pose": eef_pose,
            "active_skill": skill_blob,
            "tro": tro,
            "memory": self.get_memory_summary(),
            "camera": self.get_camera_state(),
            "vision_degraded": self._vision_safe_state,
            "vision_degraded_reason": self._vision_safe_reason,
            "simulation_degraded": getattr(
                self,
                "_simulation_degraded",
                False,
            ),
            "simulation_degraded_reason": getattr(
                self,
                "_simulation_degraded_reason",
                "",
            ),
            "simulation_degraded_since_ts": getattr(
                self,
                "_simulation_degraded_since_ts",
                None,
            ),
            "reset_count": self.reset_count,
            "reset_pending": self.reset_request,
            "task_switch_count": self.task_switch_count,
            "task_switch_pending": task_switch_pending,
            "task_switch_error": task_switch_error,
            "task_switch_reexec": task_switch_reexec,
            "task_switch_reexec_target": task_switch_reexec_target,
            "scene_graph": scene_graph_summary,
            "last_skill_results": last_results,
            "skill_history": skill_history,
            "goals": goals,
            "feeds": self.get_feed_active(),
            "simulation_activity": self._simulation_activity_snapshot(),
        }

    def _refresh_state_cache(self) -> None:
        """在主 sim 线程里调用：把 PhysX 读出来的数据序列化进 _cached_*。"""
        try:
            pose = self.world.robot_pose()
            eef_pose = {
                arm: self.world.eef_pose(arm=arm)
                for arm in ("left", "right")
            }
            tro = self.world.task_relevant_state()
        except Exception as e:
            self.log(f"WARN state cache refresh failed: {e}")
            return
        with self.state_lock:
            self._cached_robot_pose = {
                "x": float(pose.pos[0]),
                "y": float(pose.pos[1]),
                "z": float(pose.pos[2]),
                "yaw_deg": math.degrees(pose.yaw),
            }
            self._cached_eef_pose = eef_pose
            self._cached_tro = tro

    def _refresh_goal_cache(self) -> None:
        """Evaluate BDDL goal conditions on the sim thread and cache UI-safe chip state."""
        try:
            if self.dry_run or self.env is None or getattr(self.env, "task", None) is None:
                parsed = []
                if self.dry_run:
                    parsed = [
                        ["real", "cooked__popcorn.n.01_1"],
                        ["contains", "popcorn__bag.n.01_1", "cooked__popcorn.n.01_1"],
                    ]
                items = [
                    {
                        "index": i,
                        "label": _shorten_label(_render_goal_expr(expr)),
                        "full": _render_goal_expr(expr),
                        "satisfied": False,
                    }
                    for i, expr in enumerate(parsed)
                ]
                payload = {
                    "items": items,
                    "satisfied": 0,
                    "total": len(items),
                    "complete": False,
                    "ok": bool(items),
                }
            else:
                from bddl.condition_evaluation import compile_state

                task = self.env.task
                activity = getattr(task, "activity_conditions", None)
                object_map = getattr(activity, "parsed_objects", None) or {}
                parsed = list(getattr(activity, "parsed_goal_conditions", None) or [])
                judges = []
                for expr in parsed:
                    judges.extend(_expand_goal_judges(expr, object_map))
                conditions = compile_state(
                    judges,
                    task.backend,
                    scope=task.object_scope,
                    object_map=object_map,
                    generate_ground_options=False,
                )
                items = []
                for i, (expr, cond) in enumerate(zip(judges, conditions)):
                    try:
                        satisfied = bool(cond.evaluate())
                    except Exception as e:
                        satisfied = False
                        self.log(f"WARN goal condition {i} eval failed: {e}")
                    full = _render_goal_expr(expr)
                    items.append(
                        {
                            "index": i,
                            "label": _shorten_label(full),
                            "full": full,
                            "satisfied": satisfied,
                        }
                    )
                sat = sum(1 for item in items if item["satisfied"])
                payload = {
                    "items": items,
                    "satisfied": sat,
                    "total": len(items),
                    "complete": bool(items) and sat == len(items),
                    "ok": True,
                }
        except Exception as e:
            payload = {
                "items": [],
                "satisfied": 0,
                "total": 0,
                "complete": False,
                "ok": False,
                "error": str(e),
            }
        with self.state_lock:
            self._cached_goals = payload

    def _refresh_memory_cache(self, periodic: bool = False) -> None:
        """在主 sim 线程里构建 web memory；web 线程只读缓存。

        periodic=True 表示来自主循环的空闲/常态周期刷新：未绑定 capture 图时会走全量
        build_memory（含 head 深度回读），此路径按「机器人静止 + 保活间隔」限频，避免每拍
        读 head 深度造成空闲卡顿。reset/任务切换/回滚等一次性调用用默认 periodic=False，
        始终全量刷新。
        """
        try:
            from .memory import (
                build_memory,
                build_task_goals_only_memory,
                format_memory,
                task_goals_only_memory_enabled,
            )

            if task_goals_only_memory_enabled():
                with self.state_lock:
                    goals = dict(self._cached_goals)
                mem = build_task_goals_only_memory(
                    task_name=self.task_name,
                    goals=goals,
                )
                text = format_memory(mem)
                with self.state_lock:
                    self._cached_memory = mem
                    self._cached_memory_text = text
                return

            with self.state_lock:
                robot_pose = dict(self._cached_robot_pose)
                tro = dict(self._cached_tro)
                existing_memory = dict(self._cached_memory) if isinstance(self._cached_memory, dict) else {}
                existing_text = self._cached_memory_text
            # If the latest MEMORY is explicitly bound to a capture image and
            # the robot base has not moved since that capture, keep it.  This
            # prevents the periodic server-cache refresh from replacing an
            # image_id-bound UVD table with an unbound projection between
            # capture and the model/tool consumer.
            try:
                head = existing_memory.get("head_camera") if isinstance(existing_memory.get("head_camera"), dict) else {}
                mem_pose = existing_memory.get("robot_pose") if isinstance(existing_memory.get("robot_pose"), dict) else {}
                if head.get("binding") == "capture_image" and head.get("image_id"):
                    dx = float(robot_pose.get("x")) - float(mem_pose.get("x"))
                    dy = float(robot_pose.get("y")) - float(mem_pose.get("y"))
                    dxy = math.hypot(dx, dy)
                    dyaw = abs((float(robot_pose.get("yaw_deg")) - float(mem_pose.get("yaw_deg")) + 180.0) % 360.0 - 180.0)
                    if dxy <= 0.01 and dyaw <= 0.5 and isinstance(existing_text, str) and existing_text:
                        return
            except Exception:
                pass
            # 未绑定 capture 图 → 下面会全量 build（每拍读 head 深度）。周期刷新时限频：
            # 机器人静止且距上次全量重建不足保活间隔，就复用现有缓存，避免每拍读深度。
            if periodic and isinstance(existing_text, str) and existing_text:
                now_ts = time.time()
                moved = True
                last = self._memory_periodic_last_pose
                if isinstance(last, dict):
                    try:
                        dxy = math.hypot(
                            float(robot_pose.get("x", 0.0)) - float(last.get("x", 0.0)),
                            float(robot_pose.get("y", 0.0)) - float(last.get("y", 0.0)),
                        )
                        dyaw = abs(
                            (float(robot_pose.get("yaw_deg", 0.0)) - float(last.get("yaw_deg", 0.0)) + 180.0) % 360.0 - 180.0
                        )
                        moved = dxy > self._memory_periodic_move_xy or dyaw > self._memory_periodic_move_yaw
                    except Exception:
                        moved = True
                if (not moved) and (now_ts - self._memory_periodic_last_ts) < self._memory_periodic_interval_s:
                    return
                self._memory_periodic_last_ts = now_ts
                self._memory_periodic_last_pose = dict(robot_pose)
            mem = build_memory(
                self.world,
                task_name=self.task_name,
                robot_pose=robot_pose,
                tro=tro,
                tick=self.tick,
            )
            text = format_memory(mem)
        except Exception as e:
            mem = {"ok": False, "error": str(e), "summary": {}}
            text = f"(memory build failed: {e})"
        with self.state_lock:
            self._cached_memory = mem
            self._cached_memory_text = text

    def _live_mapper(self):
        """测试口实时建图器；未开启 BEHAVIOR_SPATIAL_MAP 时返回 None。"""
        mapper = getattr(self, "_spatial_live_mapper", None)
        if mapper is not None:
            return None if mapper is False else mapper
        try:
            from behavior_interface.spatial_map import spatial_map_enabled

            if not spatial_map_enabled() or self.dry_run:
                self._spatial_live_mapper = False
                return None
            from behavior_interface.rtabmap_slam.live import (
                get_live_mapper,
                live_backend_selected,
            )

            if live_backend_selected():
                mapper = get_live_mapper()
            else:
                from behavior_interface.spatial_map_live import LiveMapper

                mapper = LiveMapper()
        except Exception as e:
            self.log(f"WARN spatial live mapper unavailable: {e}")
            self._spatial_live_mapper = False
            return None
        self._spatial_live_mapper = mapper
        return mapper

    def _physics_dt(self) -> float:
        """物理子步时长；实时建图与连续录制共用同一时间基准。

        只有读到合理值才缓存，避免 world 尚未就绪时把兜底值钉死。
        """
        dt = getattr(self, "_spatial_physics_dt", None)
        if dt is not None:
            return dt
        try:
            from behavior_interface.skills.move_to_object_v2 import _base_physics_dt

            value = float(_base_physics_dt(self.world))
        except Exception:
            return 0.05
        if not (0.0 < value <= 1.0):
            return 0.05
        self._spatial_physics_dt = value
        return value

    def _continuous_capture(self):
        """连续录制器；模块不可用时返回 None 并不再重试。"""
        capture = getattr(self, "_capture_singleton", None)
        if capture is None:
            try:
                from behavior_interface.continuous_capture import CAPTURE

                capture = CAPTURE
            except Exception as e:
                self.log(f"WARN continuous capture unavailable: {e}")
                capture = False
            self._capture_singleton = capture
        return None if capture is False else capture

    def _capture_substep(self, dt: float) -> None:
        """连续录制：逐子步收 qvel，并按节流落一帧 RGB-D。

        必须挂在子步循环里，不能挂 _spatial_map_tick——后者位于
        fast_no_obs 分支之外，底盘一动就整段跳过，而底盘运动恰恰是
        必须录到的部分。异常绝不打断仿真。
        """
        capture = self._continuous_capture()
        if capture is None:
            return
        try:
            capture.on_tick(self.world, dt)
            capture.on_frame(self.world)
        except Exception:
            pass

    def _spatial_odom_tick(self) -> None:
        """逐物理子步把 base_qvel 积分进实时地图。异常绝不打断仿真。"""
        if self.dry_run or self.world is None:
            return
        dt = self._physics_dt()
        self._capture_substep(dt)
        mapper = self._live_mapper()
        if mapper is None or mapper.disabled:
            return
        try:
            mapper.odom_tick(self.world, dt)
        except Exception:
            pass

    def _spatial_map_tick(self, now: float) -> None:
        """自适应频率把 head depth 并进占用栅格。异常绝不打断仿真。"""
        mapper = self._live_mapper()
        if mapper is None or mapper.disabled:
            return
        try:
            with self.camera_io_lock:
                mapper.map_tick(self.world, now)
        except Exception:
            pass

    def _maybe_rebuild_scene_graph(self, force: bool = False) -> None:
        """主线程定时（默认 5s）重建 SceneGraph，缓存供 web/skill 用。"""
        if self.dry_run:
            return
        now = time.time()
        if not force and (now - self._scene_graph_last_build_ts) < self._scene_graph_interval_s:
            return
        try:
            from .scene_graph import format_scene_graph
            sg = self.world.build_scene_graph(robot_radius=self._scene_graph_robot_radius)
            if sg is None:
                return
            text = format_scene_graph(sg)
            summary = {
                "objects": len(sg.objects),
                "relations": len(sg.relations),
                "obstacles": len(sg.free_region.obstacles),
                "bounds": list(sg.free_region.bounds),
                "robot_radius": sg.free_region.robot_radius,
                "build_ms": sg.build_ms,
                # 暴露胸口 5D pose（chest_x/y/z + theta_x_deg + theta_z_deg）
                # 方便 web 客户端 + plan/grasp skill 直接读
                "robot_pose": dict(sg.robot_pose),
            }
        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"WARN build_scene_graph failed: {e}\n{tb}")
            return
        with self.state_lock:
            self._cached_scene_graph = sg
            self._cached_scene_graph_text = text
            self._cached_scene_graph_summary = summary
        # 把最新 sg 注入到 WorldAPI，skill 直接通过 ctx.world.current_scene_graph 读
        try:
            self.world.current_scene_graph = sg
        except Exception:
            pass
        # The interval starts after the synchronous build completes. Measuring
        # from its start turns a slow build into an immediate rebuild loop.
        self._scene_graph_last_build_ts = time.time()
        # 快照本次重建时的机器人位姿，供空闲期「移动才重建」判断
        try:
            with self.state_lock:
                self._scene_graph_last_pose = dict(self._cached_robot_pose)
        except Exception:
            self._scene_graph_last_pose = None

    def _maybe_refresh_scene_graph_after_step(
        self,
        *,
        has_job: bool,
        skill_just_finished: bool,
    ) -> None:
        """Refresh the heavy scene graph without blocking active skill frames."""
        if has_job:
            return
        if skill_just_finished:
            self._maybe_rebuild_scene_graph(force=True)
        elif self._robot_moved_since_scene_graph():
            self._maybe_rebuild_scene_graph()

    def _maybe_guard_disk(self, reason: str, force: bool = False) -> None:
        """低磁盘时先告警、再回收陈旧会话产物。

        写满分区会让 capture 落盘的 npy 被截断，下游读到坏数据；这里在还有余量
        时就出手，避免静默损坏。只回收早于保留窗口的会话，活跃会话不受影响。
        """
        interval_s = getattr(self, "_disk_guard_interval_s", 0.0)
        now = time.time()
        if not force:
            if interval_s <= 0:
                return
            if (now - getattr(self, "_disk_guard_last_ts", 0.0)) < interval_s:
                return
        self._disk_guard_last_ts = now
        warn_mib = getattr(self, "_disk_warn_mib", 0.0)
        gc_mib = getattr(self, "_disk_gc_mib", 0.0)
        free_mib = agent_runs.disk_free_mib()
        if free_mib < 0 or free_mib >= warn_mib:
            return
        if free_mib >= gc_mib:
            self.log(
                f"DISK_LOW reason={reason} free={free_mib:.0f}MiB "
                f"warn_at={warn_mib:.0f}MiB gc_at={gc_mib:.0f}MiB"
            )
            return
        max_age_h = getattr(self, "_runs_max_age_h", 48.0)
        report = agent_runs.prune_all_stale_sessions(max_age_h)
        after_mib = agent_runs.disk_free_mib()
        self.log(
            f"DISK_GC reason={reason} free={free_mib:.0f}->{after_mib:.0f}MiB "
            f"removed={report['removed']} freed={report['freed_mib']:.0f}MiB "
            f"max_age_h={max_age_h:.1f} errors={report['errors'][:3]}"
        )

    def _maybe_release_native_memory(self, reason: str, force: bool = False) -> None:
        if self.dry_run:
            return
        # 磁盘守卫是尽力而为的附加维护，任何失败都不该拖累内存释放主流程
        try:
            self._maybe_guard_disk(reason, force=force)
        except Exception as e:
            self.log(f"DISK_GUARD_SKIP reason={reason} {type(e).__name__}: {e}")
        now = time.time()
        if not force:
            if self._native_trim_interval_s <= 0:
                return
            if (now - self._native_trim_last_ts) < self._native_trim_interval_s:
                return
        self._native_trim_last_ts = now
        started = time.time()
        before = _process_memory_mib()
        result: Dict[str, Any] = {
            "native": _release_native_allocator_caches(),
        }
        try:
            result["warp"] = _trim_loaded_warp_mempools()
        except Exception as e:
            result["warp"] = {
                "loaded": True,
                "initialized": True,
                "trimmed_devices": [],
                "errors": [f"{type(e).__name__}: {e}"],
            }
        after = _process_memory_mib()
        self.log(
            "MEMORY_RELEASE "
            f"reason={reason} elapsed={time.time() - started:.3f}s "
            f"pss={before.get('pss', 0.0):.1f}->{after.get('pss', 0.0):.1f}MiB "
            f"anon={before.get('pss_anon', 0.0):.1f}->{after.get('pss_anon', 0.0):.1f}MiB "
            f"private_dirty={before.get('private_dirty', 0.0):.1f}->{after.get('private_dirty', 0.0):.1f}MiB "
            f"result={result}"
        )

    def _signal_sim_wake(self) -> None:
        event = getattr(self, "_idle_wake_event", None)
        if event is not None:
            event.set()

    def _arm_idle_settle(self, reason: str) -> None:
        if self.dry_run or not self._idle_quiescence_enabled:
            return
        now = time.monotonic()
        self._idle_settle_deadline_mono = now + self._idle_settle_s
        if self._idle_quiescent:
            idle_for = 0.0
            if self._idle_quiescent_since_ts is not None:
                idle_for = max(0.0, time.time() - self._idle_quiescent_since_ts)
            self.log(f"IDLE_WAKE reason={reason} quiescent_for={idle_for:.1f}s")
        self._idle_quiescent = False
        self._idle_quiescent_since_ts = None
        self._signal_sim_wake()

    @staticmethod
    def _skill_changes_world_state(skill_name: Optional[str]) -> bool:
        return skill_name not in _CAMERA_CAPTURE_SKILLS

    def _finish_idle_transition_after_skill(self, skill_name: Optional[str]) -> None:
        if self._skill_changes_world_state(skill_name):
            self._arm_idle_settle("skill_finished")
            return
        if self.dry_run or not self._idle_quiescence_enabled:
            return
        self._idle_settle_deadline_mono = time.monotonic()
        self.log(f"IDLE_SETTLE_SKIPPED skill={skill_name}")

    def _queued_skill_token(self) -> Optional[str]:
        with self.skill_queue.mutex:
            if not self.skill_queue.queue:
                return None
            job = self.skill_queue.queue[0]
        return str(getattr(job, "request_id", id(job)))

    def _queued_skill_name(self) -> Optional[str]:
        with self.skill_queue.mutex:
            if not self.skill_queue.queue:
                return None
            job = self.skill_queue.queue[0]
        name = str(getattr(job, "name", "") or "")
        return name or None

    def _idle_wake_reason(self, now_mono: Optional[float] = None) -> Optional[str]:
        if self.current_job is not None:
            return "active_skill"
        if now_mono is None:
            now_mono = time.monotonic()
        queued_token = self._queued_skill_token()
        if queued_token is not None:
            # A broken camera must not turn a queued job into an infinite idle
            # physics loop. Give camera recovery a bounded settling window, then
            # return to zero-step quiescence while leaving the job queued.
            if self._vision_safe_state:
                first_prime = self._idle_vision_prime_job_token != queued_token
                if self._idle_quiescent and first_prime:
                    return "queued_skill"
                if now_mono >= self._idle_settle_deadline_mono:
                    return None
            return "queued_skill"
        self._idle_vision_prime_job_token = None
        with self.skill_lock:
            if self.reset_request:
                return "reset"
            if self.task_switch_request is not None or self.task_switch_in_progress:
                return "task_switch"
        with self._feed_active_lock:
            if self._feed_warmup_pending:
                return "feed_warmup"
            main_active = bool(self._feed_active.get("main", False))
        if self._gta_cam_dirty and main_active:
            return "camera_update"
        if bool(getattr(self.world, "_codex_fast_motion_no_obs", False)):
            return "fast_motion"
        return None

    def _prime_vision_after_quiescence(self) -> None:
        if self.dry_run or self.env is None or self.robot is None:
            return
        queued_skill = self._queued_skill_name()
        if queued_skill not in _CAMERA_CAPTURE_SKILLS:
            try:
                self._prime_obs_after_warmup(
                    n_render=3,
                    n_attempts=2,
                    allow_headless_skip=False,
                )
            except Exception as e:
                self.log(f"WARN idle wake obs prime failed: {e}")

        # env.get_obs() refreshes annotators but does not update CameraFeedHealth.
        # Read the head RGB once so a deliberately static quiescent frame is not
        # mistaken for a dead camera when _maybe_start_next_skill() runs.
        try:
            for sensor_name, sensor in list(self.robot.sensors.items()):
                if self._robot_camera_feed(sensor_name) != "head":
                    continue
                arr = self._read_camera_rgb("head", sensor)
                if arr is None:
                    return
                img = convert_rgb_frame(
                    arr,
                    lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
                self._update_frames({"head": label_image(img, "HEAD")})
                return
        except Exception as e:
            self.log(f"WARN idle wake head refresh failed: {e}")

    def _maybe_hold_quiescent_idle(self) -> bool:
        """Return True when this loop should skip every simulation operation."""
        if (
            self.dry_run
            or not self._idle_quiescence_enabled
            or self.env is None
            or self.robot is None
            or self.world is None
            or getattr(self.world, "robot", None) is None
        ):
            return False

        now_mono = time.monotonic()
        reason = self._idle_wake_reason(now_mono)
        if reason is not None:
            if self._idle_quiescent:
                idle_for = 0.0
                if self._idle_quiescent_since_ts is not None:
                    idle_for = max(0.0, time.time() - self._idle_quiescent_since_ts)
                self._idle_quiescent = False
                self._idle_quiescent_since_ts = None
                self._idle_settle_deadline_mono = now_mono + self._idle_settle_s
                self.log(f"IDLE_WAKE reason={reason} quiescent_for={idle_for:.1f}s")
                if reason in {"queued_skill", "feed_warmup"}:
                    if reason == "queued_skill":
                        self._idle_vision_prime_job_token = self._queued_skill_token()
                    self._prime_vision_after_quiescence()
            return False

        if now_mono < self._idle_settle_deadline_mono:
            return False

        if not self._idle_quiescent:
            self._idle_quiescent = True
            self._idle_quiescent_since_ts = time.time()
            self.log(
                "IDLE_QUIESCENT "
                f"settle_s={self._idle_settle_s:.2f}; suppressing og.sim.step()"
            )
            self._maybe_release_native_memory(reason="quiescent", force=True)
        self._idle_suppressed_ticks += 1
        return True

    def _wait_for_idle_wake(self) -> None:
        event = self._idle_wake_event
        event.wait(timeout=self._idle_poll_s)
        event.clear()

    def _simulation_activity_snapshot(self) -> Dict[str, Any]:
        now_mono = time.monotonic()
        if self.dry_run:
            state = "dry_run"
        elif getattr(self, "_simulation_degraded", False):
            state = "degraded"
        elif self.current_job is not None:
            state = "active"
        elif self._idle_quiescent:
            state = "quiescent"
        else:
            state = "settling"
        return {
            "state": state,
            "idle_quiescence_enabled": bool(self._idle_quiescence_enabled),
            "settle_s": float(self._idle_settle_s),
            "settle_remaining_s": max(
                0.0,
                self._idle_settle_deadline_mono - now_mono,
            ),
            "quiescent_since_ts": self._idle_quiescent_since_ts,
            "suppressed_loop_count": int(self._idle_suppressed_ticks),
            "degraded_reason": getattr(
                self,
                "_simulation_degraded_reason",
                "",
            ),
        }

    def _robot_moved_since_scene_graph(self) -> bool:
        """空闲期用：机器人相对上次场景图重建是否移动超过阈值。无历史位姿时视为已移动。"""
        last = self._scene_graph_last_pose
        if not isinstance(last, dict):
            return True
        try:
            with self.state_lock:
                cur = dict(self._cached_robot_pose)
            dxy = math.hypot(
                float(cur.get("x", 0.0)) - float(last.get("x", 0.0)),
                float(cur.get("y", 0.0)) - float(last.get("y", 0.0)),
            )
            dyaw = abs(
                (float(cur.get("yaw_deg", 0.0)) - float(last.get("yaw_deg", 0.0)) + 180.0) % 360.0 - 180.0
            )
        except Exception:
            return True
        return dxy > self._scene_graph_idle_move_xy or dyaw > self._scene_graph_idle_move_yaw

    def _idle_should_render(self, now: float) -> bool:
        """空闲静止时是否需要重渲染主循环画面（否则复用上一帧、跳过 og.sim.render）。"""
        # 相机位姿刚被调 / 首帧 → 必渲
        if self._gta_cam_dirty:
            return True
        # 视图刚被开启（预热）→ 必渲，保证「打开即有画面」
        with self._feed_active_lock:
            if self._feed_warmup_pending:
                return True
        if not self.frames:
            return True
        # 保活间隔到 → 渲一帧，兜住任何缓慢变化
        if (now - self._render_idle_last_ts) >= self._render_idle_keepalive_s:
            return True
        # 机器人移动超阈值 → 渲（用上一拍缓存位姿，一拍滞后可接受）
        last = self._render_idle_last_pose
        if not isinstance(last, dict):
            return True
        try:
            with self.state_lock:
                cur = dict(self._cached_robot_pose)
            dxy = math.hypot(
                float(cur.get("x", 0.0)) - float(last.get("x", 0.0)),
                float(cur.get("y", 0.0)) - float(last.get("y", 0.0)),
            )
            dyaw = abs(
                (float(cur.get("yaw_deg", 0.0)) - float(last.get("yaw_deg", 0.0)) + 180.0) % 360.0 - 180.0
            )
        except Exception:
            return True
        return dxy > self._render_idle_move_xy or dyaw > self._render_idle_move_yaw

    def get_scene_graph(self):
        """供 skill / WorldAPI.* 调用：返回缓存的 SceneGraph。"""
        with self.state_lock:
            return self._cached_scene_graph

    def get_scene_graph_text(self) -> str:
        with self.state_lock:
            return self._cached_scene_graph_text

    def get_scene_graph_summary(self) -> Dict[str, Any]:
        with self.state_lock:
            return dict(self._cached_scene_graph_summary)

    def get_memory(self) -> Dict[str, Any]:
        with self.state_lock:
            return dict(self._cached_memory)

    def get_memory_text(self) -> str:
        with self.state_lock:
            return self._cached_memory_text

    def get_memory_summary(self) -> Dict[str, Any]:
        with self.state_lock:
            memory = self._cached_memory or {}
            if _task_goals_only_observation_mode():
                return dict(memory.get("bddl_completion") or {})
            return dict(memory.get("summary") or {})

    # ------------------------------------------------------------------ 视图按需渲染

    def get_feed_active(self) -> Dict[str, bool]:
        with self._feed_active_lock:
            return dict(self._feed_active)

    def set_feed_active(self, feed: str, active: bool) -> Dict[str, Any]:
        """开/关某一路视图。开启会标记预热，由主循环在下一 tick 内补渲染+取帧。"""
        with self._feed_active_lock:
            if feed not in self._feed_active:
                return {"ok": False, "error": f"unknown feed {feed!r}", "feeds": dict(self._feed_active)}
            prev = bool(self._feed_active.get(feed, False))
            self._feed_active[feed] = bool(active)
            if bool(active) and not prev:
                self._feed_warmup_pending.add(feed)
            elif not bool(active):
                self._feed_warmup_pending.discard(feed)
            feeds = dict(self._feed_active)
        if prev != bool(active):
            self._signal_sim_wake()
        return {"ok": True, "feed": feed, "active": bool(active), "feeds": feeds}

    def _feed_is_active(self, feed: str) -> bool:
        with self._feed_active_lock:
            return bool(self._feed_active.get(feed, False))

    def _take_feed_warmups(self) -> set:
        with self._feed_active_lock:
            pending = set(self._feed_warmup_pending)
            self._feed_warmup_pending.clear()
            return pending

    def _sync_gta_render(self, want_enabled: bool) -> None:
        """按主视图开关控制 GTA 渲染开销——但绝不暂停其 render product。

        经验证：暂停 external GTA sensor 的 hydra_texture 更新会像机器人相机一样
        让 Replicator render vars 失效（日志出现 removePath Replicator_02 /
        LdrColorSD missing，渲染管线变慢、head 帧被判 stale 触发 vision-safe 抖动）。
        因此这里保持 no-op：关闭主视图省资源完全靠「不抓帧/不编码」+「快速运动整帧
        跳过 render()」实现，不动 render product，保证稳定与「打开即有画面」。
        """
        return

    def _update_frames(self, frames: Dict[str, np.ndarray]) -> None:
        """覆盖共享 frame，并为真正更新的 feed 单独递增版本号。"""
        now = time.time()
        with self.frame_lock:
            for k, v in frames.items():
                old = self.frames.get(k)
                self.frames[k] = v
                if v is not old:
                    self.frame_ids[k] = int(self.frame_ids.get(k, 0)) + 1
                    self.frame_updated_ts[k] = now
            self.frame_id += 1

    # ------------------------------------------------------------------ sim init

    def init_simulation(self) -> None:
        """启 OmniGibson，加载场景 + R1Pro + GTA 外部相机。"""
        if self.dry_run:
            self.log("dry-run 模式：跳过 OmniGibson 启动")
            return

        self.log("启动 OmniGibson...")
        self._log_gpu_diag(
            "init_simulation.begin",
            extra={
                "task": self.task_name,
                "scene": self.scene_model,
                "robot": self.robot_name,
                "pid": os.getpid(),
            },
        )
        # 延迟 import，避免 dry-run 也加载
        import omnigibson as og
        from omnigibson.macros import gm
        from .skills import _install_empty_image_remapper_hotfix

        sim_thread_limit = _configure_omniverse_thread_limit()
        if sim_thread_limit is None:
            self.log("Kit tasking thread limit disabled by CARB_TASKING_THREADS=0")
        else:
            self.log(f"Kit tasking/PXR thread limit={sim_thread_limit}")

        _install_empty_image_remapper_hotfix()

        # 默认设置：禁用 ROS / 录制等
        gm.RENDER_VIEWER_CAMERA = False
        gm.ENABLE_OBJECT_STATES = True
        gm.ENABLE_TRANSITION_RULES = True

        self._validate_task_instance_assets(
            task_name=self.task_name,
            scene_model=self.scene_model,
            requested_instance_id=self.requested_instance_id,
        )
        cfg = self._build_env_cfg()
        self.log(f"env cfg task={self.task_name} scene={self.scene_model}")
        self._log_gpu_diag(
            "og_environment.before",
            extra={"task": self.task_name, "scene": self.scene_model},
        )
        try:
            self.env = og.Environment(configs=cfg)
        except Exception as e:
            self._log_gpu_diag(
                "og_environment.exception",
                extra={
                    "task": self.task_name,
                    "scene": self.scene_model,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            raise
        self._bind_world_handles_for_current_env(initializing=True)
        self._log_gpu_diag(
            "og_environment.after",
            extra={
                "task": self.task_name,
                "scene": self.scene_model,
                "robot": getattr(self.robot, "name", None),
            },
        )

        self._reset_env_for_interface()
        # 加载 task instance（与 BEHAVIOR challenge 评测一致：指定或随机），并设为 reset baseline。
        resolved_inst = self._resolve_instance_id(self.requested_instance_id)
        self._load_task_instance(resolved_inst)
        self._apply_reset_grasp_prep_pose()
        try:
            from behavior_interface.head_capture import setup_head_after_env_reset
            setup_head_after_env_reset(self.world, env=self.env, log_fn=self.log)
        except Exception as e:
            self.log(f"WARN head 相机配置/挂载快照失败: {e}")
        self._warmup_reset_hold_pins(n_steps=24)
        self._capture_world_reset_snapshot("startup")

        # 隐藏天花板：GTA 俯视相机会架在机器人头顶上方，否则被天花板遮挡，画面全是灰色的天花板背面。
        try:
            hidden = 0
            for name in ("ceilings", "ceiling"):
                obj = self.env.scene.object_registry("name", name)
                if obj is not None:
                    obj.visible = False
                    hidden += 1
            # 兜底：按 category 找
            for obj in self.env.scene.objects:
                if getattr(obj, "category", None) in ("ceilings", "roof"):
                    obj.visible = False
                    hidden += 1
            self.log(f"hid ceilings: {hidden} objects")
        except Exception as e:
            self.log(f"WARN hide ceiling failed: {e}")

        # 关键：让 RTX 渲染管线把 external sensor 的渲染产品热身一遍。
        # 没有这步时 GTA cam 的 LdrColor 渲染变量可能持续是 placeholder（输出全灰画面）。
        if _skip_headless_rtx_warmup():
            self.log("skip RTX warmup during headless startup")
            self._log_gpu_diag("rtx_warmup.skipped", extra={"reason": "headless_startup"})
        else:
            try:
                self.log("warming up RTX render pipeline for external sensor...")
                self._log_gpu_diag("rtx_warmup.before", extra={"renders": 15})
                for i in range(15):
                    og.sim.render()
                self.log("RTX warmup done.")
                self._log_gpu_diag("rtx_warmup.after", extra={"renders": 15})
            except Exception as e:
                self.log(f"WARN RTX warmup failed: {e}")
                self._log_gpu_diag(
                    "rtx_warmup.exception",
                    extra={"error": f"{type(e).__name__}: {e}", "renders": 15},
                )

        self._bind_camera_health_sources()

        # 预填一次 state cache + 首次构建 Scene Graph
        self._refresh_state_cache()
        self._refresh_goal_cache()
        self._refresh_memory_cache()
        self.log("首次构建 Scene Graph...")
        self._maybe_rebuild_scene_graph(force=True)
        sg_sum = self.get_scene_graph_summary()
        if sg_sum:
            self.log(f"Scene Graph: objects={sg_sum.get('objects')} "
                     f"relations={sg_sum.get('relations')} "
                     f"obstacles={sg_sum.get('obstacles')} "
                     f"build_ms={sg_sum.get('build_ms')}")
        self._arm_idle_settle("startup")

    # ------------------------------------------------------------------ task instance

    def _eval_instance_ids(self) -> List[int]:
        """Return candidate eval instance IDs for the current task.

        2026 challenge 的 train / public / hidden 实例范围是全任务共享的；
        旧 2025 数据集仍使用 metadata/test_instances.csv 逐任务列出 instance。
        2026 数据缺失时必须硬失败，不能回退 template 或旧数据集。
        """
        try:
            from omnigibson.macros import gm

            if _challenge_year() >= 2026:
                root_2026 = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances")
                if not os.path.isdir(root_2026):
                    raise FileNotFoundError(
                        f"2026 challenge instances not found at {root_2026}"
                    )
                return challenge_eval_instance_ids(
                    _challenge_mode(),
                    limit=_NUM_EVAL_INSTANCES,
                )

            import csv as _csv

            csv_path = os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv")
            if os.path.isfile(csv_path):
                with open(csv_path, "r") as f:
                    rows = list(_csv.reader(f))[1:]
                for row in rows:
                    if len(row) >= 3 and str(row[1]).strip() == self.task_name:
                        ids = [int(x) for x in str(row[2]).strip().split(",") if x.strip()]
                        if ids:
                            return ids[:_NUM_EVAL_INSTANCES]
                self.log(f"WARN 2025 test_instances.csv 未找到 task={self.task_name}，回退 instance [0]")
        except Exception as e:
            if _challenge_year() >= 2026:
                raise
            self.log(f"WARN 读取评测 instance 列表失败，回退 instance [0]: {e}")
        return [0]

    def _validate_task_instance_assets(
        self,
        *,
        task_name: str,
        scene_model: str,
        requested_instance_id: Optional[int],
    ) -> Dict[str, Any]:
        """Fail before simulation teardown when the selected 2026 data is incomplete."""
        if self.dry_run or _challenge_year() < 2026:
            return {}
        from omnigibson.macros import gm

        instance_ids = None if requested_instance_id is None else [int(requested_instance_id)]
        result = validate_challenge_task_assets(
            gm.DATA_PATH,
            task_name,
            scene_model,
            mode=_challenge_mode(),
            instance_ids=instance_ids,
            limit=_NUM_EVAL_INSTANCES,
        )
        self.log(
            "challenge assets validated "
            f"task={task_name} scene={scene_model} mode={result.get('mode')} "
            f"instances={result.get('instance_ids')}"
        )
        return result

    def _resolve_instance_id(self, requested: Optional[int]) -> int:
        """requested 指定则用它；否则从评测 instance 列表里随机选一个。"""
        if requested is not None:
            return int(requested)
        import random
        ids = self._eval_instance_ids()
        return int(random.choice(ids))

    def _task_instance_path_candidates(
        self,
        *,
        scene_model: str,
        activity_name: str,
        tro_filename: str,
        instance_id: int,
    ) -> List[str]:
        """Return the one allowed TRO path for the configured challenge year/split."""
        from omnigibson.macros import gm

        rel = os.path.join(
            "json",
            f"{scene_model}_task_{activity_name}_instances",
            f"{tro_filename}-tro_state.json",
        )
        candidates: list[str] = []

        if _challenge_year() >= 2026:
            root_2026 = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances")
            preferred = _challenge_2026_mode_for_instance(int(instance_id))
            candidates.append(
                os.path.join(root_2026, _challenge_2026_mode_dir(preferred), scene_model, rel)
            )
        else:
            candidates.append(
                os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "scenes", scene_model, rel)
            )
        return candidates

    def _resolve_task_instance_path(
        self,
        *,
        scene_model: str,
        activity_name: str,
        tro_filename: str,
        instance_id: int,
    ) -> tuple[Optional[str], List[str]]:
        candidates = self._task_instance_path_candidates(
            scene_model=scene_model,
            activity_name=activity_name,
            tro_filename=tro_filename,
            instance_id=instance_id,
        )
        for path in candidates:
            if os.path.isfile(path):
                return path, candidates
        return None, candidates

    def _verify_task_instance_object_positions(
        self,
        expected_positions: Dict[str, List[float]],
        *,
        tolerance_m: float = 0.05,
        stage: str,
    ) -> Dict[str, Any]:
        """Verify that TRO object state reached the physical scene object."""
        task = self.env.task
        checked = 0
        max_error_m = 0.0
        errors: Dict[str, Any] = {}
        for tro_key, expected_raw in expected_positions.items():
            ent = task.object_scope.get(tro_key)
            if ent is None:
                errors[tro_key] = {"error": "missing object_scope entity"}
                continue
            try:
                if ent.is_system or not ent.exists:
                    continue
            except Exception:
                pass
            try:
                actual_raw, _ = ent.get_position_orientation()
                actual = np.asarray(
                    actual_raw.detach().cpu().numpy()
                    if hasattr(actual_raw, "detach")
                    else actual_raw,
                    dtype=np.float64,
                ).reshape(3)
                expected = np.asarray(expected_raw, dtype=np.float64).reshape(3)
                err_m = float(np.linalg.norm(actual - expected))
            except Exception as exc:
                errors[tro_key] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            checked += 1
            max_error_m = max(max_error_m, err_m)
            if err_m > float(tolerance_m):
                errors[tro_key] = {
                    "name": str(getattr(ent, "name", "") or ""),
                    "expected": [round(float(x), 6) for x in expected.tolist()],
                    "actual": [round(float(x), 6) for x in actual.tolist()],
                    "err_m": round(err_m, 6),
                }
        report = {
            "stage": str(stage),
            "checked": checked,
            "max_error_m": round(max_error_m, 6),
            "tolerance_m": float(tolerance_m),
            "errors": errors,
        }
        if errors:
            raise RuntimeError(f"TRO physical pose verification failed: {report}")
        self.log(
            "TASK_INSTANCE_POSE_VERIFY "
            f"stage={stage} checked={checked} max_err={max_error_m*1000:.1f}mm"
        )
        return report

    def _load_task_instance(self, instance_id: int) -> bool:
        """加载指定 task instance 的 TRO 状态 + 机器人预采样位姿，并设为 reset baseline。

        逻辑与官方 OmniGibson/omnigibson/learning/eval.py 的
        Evaluator.load_task_instance 完全一致：
          1) 读 {scene}_task_{activity}_0_{id}_template-tro_state.json；
          2) 对 task.object_scope 里的每个 TRO 逐项 load_state；机器人 base 位姿从
             robot_poses[model][0] 设置，并写入 scene task metadata；
          3) 25 步 step_physics + keep_still 消除加载抖动；
          4) scene.update_initial_file() 把当前态设为新的 reset baseline，再 scene.reset()。
        这样「启动加载 instance」和之后的 env.reset() 都会回到同一 baseline，
        即启动与 reset 效果一致。
        """
        if self.dry_run or self.env is None or self.robot is None:
            return False
        try:
            import omnigibson as og
            from omnigibson.utils.python_utils import recursively_convert_to_torch
            import json as _json

            task = self.env.task
            scene_model = task.scene_name
            tro_filename = task.get_cached_activity_scene_filename(
                scene_model=scene_model,
                activity_name=task.activity_name,
                activity_definition_id=task.activity_definition_id,
                activity_instance_id=int(instance_id),
            )
            tro_file_path, candidates = self._resolve_task_instance_path(
                scene_model=scene_model,
                activity_name=task.activity_name,
                tro_filename=tro_filename,
                instance_id=int(instance_id),
            )
            if tro_file_path is None:
                message = (
                    f"instance {instance_id} tro_state missing for task={task.activity_name} "
                    f"scene={scene_model}; checked={candidates[:3]}"
                )
                if _challenge_year() >= 2026:
                    raise FileNotFoundError(message)
                self.log(f"WARN {message}; keeping template baseline")
                return False
            with open(tro_file_path, "r") as f:
                tro_raw = _json.load(f)
            tro_all = recursively_convert_to_torch(tro_raw)
            expected_positions: Dict[str, List[float]] = {}
            for tro_key, tro_state in tro_raw.items():
                if tro_key == "robot_poses" or not isinstance(tro_state, dict):
                    continue
                root_link = tro_state.get("root_link")
                if isinstance(root_link, dict) and root_link.get("pos") is not None:
                    expected_positions[str(tro_key)] = list(root_link["pos"])

            n_obj = 0
            robot_set = False
            for tro_key, tro_state in tro_all.items():
                if tro_key == "robot_poses":
                    model = str(getattr(self.robot, "model_name", None) or getattr(self.robot, "model", "")).lower()
                    pose_map = {str(k).lower(): v for k, v in tro_state.items()}
                    if "robot" in pose_map:
                        available_poses = pose_map["robot"]
                    elif model in pose_map:
                        available_poses = pose_map[model]
                    else:
                        if _challenge_year() >= 2026:
                            raise KeyError(
                                f"instance {instance_id} robot_poses has no robot/{model}"
                            )
                        self.log(f"WARN instance {instance_id} robot_poses 无 robot/{model}，机器人位姿不变")
                        continue
                    robot_pos = available_poses[0]["position"]
                    robot_quat = available_poses[0]["orientation"]
                    self.robot.set_position_orientation(robot_pos, robot_quat)
                    self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
                    robot_set = True
                else:
                    ent = task.object_scope.get(tro_key)
                    if ent is None:
                        if _challenge_year() >= 2026:
                            raise KeyError(
                                f"instance {instance_id} TRO object {tro_key!r} "
                                "is absent from the loaded task template"
                            )
                        continue
                    ent.load_state(tro_state, serialized=False)
                    n_obj += 1

            # 加载 state 后可能有抖动（小质量/薄物体），按官方做法稳定 TRO。
            try:
                og.sim.update_handles()
            except Exception:
                pass
            for _ in range(25):
                og.sim.step_physics()
                for entity in task.object_scope.values():
                    try:
                        if not entity.is_system and entity.exists:
                            entity.keep_still()
                    except Exception:
                        pass

            self._verify_task_instance_object_positions(
                expected_positions,
                tolerance_m=0.05,
                stage="after_tro_settle",
            )

            # 把当前（含 instance TRO + 机器人位姿）状态设为新的 reset baseline，再 reset 一次。
            self.env.scene.update_initial_file()
            self.env.scene.reset()
            self._rebind_robot_handles_after_scene_restore(
                stage=f"task_instance_scene_reset:{instance_id}"
            )
            self._verify_task_instance_object_positions(
                expected_positions,
                tolerance_m=0.05,
                stage="after_scene_reset",
            )
            self.current_instance_id = int(instance_id)
            self.log(
                f"已加载 task instance={instance_id}（task={task.activity_name} "
                f"scene={scene_model} mode={_challenge_2026_mode_for_instance(int(instance_id))} "
                f"TRO物体={n_obj} robot_pose={'set' if robot_set else 'keep'}），"
                f"并设为 reset baseline"
            )
            return True
        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"加载 task instance {instance_id} 失败: {e}\n{tb}")
            if _challenge_year() >= 2026:
                raise
            return False

    def _build_env_cfg(self) -> Dict[str, Any]:
        """主仿真配置：R1Pro + BehaviorTask + GTA 外部相机。"""
        viewer_w, viewer_h = 640, 480
        if not self.dry_run:
            try:
                import omnigibson as _og
                sim = getattr(_og, "sim", None)
                if sim is not None:
                    viewer_w = int(getattr(sim, "viewer_width", viewer_w))
                    viewer_h = int(getattr(sim, "viewer_height", viewer_h))
            except Exception:
                pass
        robot_cfg = build_interface_robot_config(
            path=getattr(self, "robot_config_path", None),
            robot_type=self.robot_name,
            robot_dof=self.robot_dof,
            image_width=self.sub_w,
            image_height=self.sub_h,
        )
        cfg = {
            "env": {
                "device": None,
                "automatic_reset": False,
                "flatten_action_space": False,
                "flatten_obs_space": False,
                "use_external_obs": True,
                "initial_pos_z_offset": 0.1,
                "external_sensors": [
                    {
                        "sensor_type": "VisionSensor",
                        "name": "gta_view",
                        "relative_prim_path": "/gta_view",
                        "modalities": ["rgb"],
                        "sensor_kwargs": {
                            "image_height": self.main_h,
                            "image_width": self.main_w,
                        },
                        "position": [0.0, 0.0, 2.0],
                        "orientation": [0.0, 0.0, 0.0, 1.0],
                        "pose_frame": "parent",
                    }
                ],
            },
            "render": {
                "viewer_width": viewer_w,
                "viewer_height": viewer_h,
            },
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": self.scene_model,
                "trav_map_resolution": 0.1,
                "default_erosion_radius": 0.0,
                "trav_map_with_objects": True,
                "scene_source": "OG",
                # Challenge templates contain a legacy Robot entry whose default
                # arm controllers are 6D IK. Exclude it so _load_robots creates
                # the explicitly configured R1Pro below (7D absolute joint arms).
                "include_robots": False,
            },
            "robots": [robot_cfg],
            "objects": [],
            "task": {
                "type": "BehaviorTask",
                "activity_name": self.task_name,
                "activity_definition_id": 0,
                "activity_instance_id": 0,
                "predefined_problem": None,
                "online_object_sampling": False,
                "debug_object_sampling": False,
                "highlight_task_relevant_objects": False,
                "use_presampled_robot_pose": True,
                "termination_config": {"max_steps": 100000},
                "reward_config": {"r_potential": 1.0},
            },
        }
        if not self.dry_run:
            try:
                from omnigibson.macros import gm

                if _challenge_year() >= 2026:
                    requested = self.requested_instance_id
                    template_mode = (
                        _challenge_2026_mode_for_instance(int(requested))
                        if requested is not None
                        else _challenge_mode()
                    )
                    cached_scene_path = challenge_template_path(
                        gm.DATA_PATH,
                        self.task_name,
                        self.scene_model,
                        template_mode,
                    )
                    if not os.path.isfile(cached_scene_path):
                        raise FileNotFoundError(
                            "2026 challenge task template missing; refusing online sampling: "
                            f"{cached_scene_path}"
                        )
                    cfg["scene"]["scene_file"] = self._load_challenge_template(
                        cached_scene_path, as_torch=False
                    )
                    self.log(
                        "using 2026 challenge task template "
                        f"task={self.task_name} scene={self.scene_model} "
                        f"mode={template_mode} path={cached_scene_path}"
                    )
                    return cfg

                cached_scene_instance = (
                    f"{self.scene_model}_task_{self.task_name}_0_0_template"
                )
                cached_scene_path = os.path.join(
                    gm.DATA_PATH,
                    "2025-challenge-task-instances",
                    "scenes",
                    self.scene_model,
                    "json",
                    f"{cached_scene_instance}.json",
                )
                try:
                    from .challenge_tasks import challenge_task_by_name

                    task_meta = challenge_task_by_name(self.task_name)
                    challenge_task_id = (
                        int(task_meta.get("id", -1)) if task_meta is not None else -1
                    )
                except Exception:
                    challenge_task_id = -1

                if not os.path.isfile(cached_scene_path):
                    cfg["task"]["online_object_sampling"] = True
                    cfg["task"]["use_presampled_robot_pose"] = False
                    self.log(
                        "WARN cached task template missing; enabling online object sampling "
                        f"task={self.task_name} scene={self.scene_model} checked={cached_scene_path}"
                    )
                elif challenge_task_id >= 50:
                    cache_ok, cache_reason = _legacy_template_scope_compatible(
                        self.task_name,
                        int(cfg["task"].get("activity_definition_id", 0)),
                        cached_scene_path,
                    )
                    if cache_ok:
                        self.log(
                            "using compatible cached task template "
                            f"task={self.task_name} id={challenge_task_id} scene={self.scene_model}"
                        )
                    else:
                        cfg["task"]["online_object_sampling"] = True
                        cfg["task"]["use_presampled_robot_pose"] = False
                        self.log(
                            "WARN cached task template incompatible with current BDDL; "
                            "enabling online object sampling "
                            f"task={self.task_name} id={challenge_task_id} scene={self.scene_model} "
                            f"reason={cache_reason}"
                        )
            except Exception as e:
                if _challenge_year() >= 2026:
                    raise
                self.log(f"WARN cached task template check failed: {e}")
        return cfg

    # ------------------------------------------------------------------ frames

    def _update_gta_camera_pose(self) -> None:
        """每步用机器人位姿计算 GTA 视角 pose。

        相机摆位：机器人 base 正后方 DISTANCE 米、上方 HEIGHT 米，可绕机器人左右转 YAW_OFFSET 度。
        相机视线（USD 相机约定 = -Z）看向 (robot, LOOK_Z_OFFSET)；
        相机 up（USD 相机约定 = +Y）投影到与视线正交后尽量贴近世界 +Z，
        这样画面不会绕视线轴滚转，前/后/左/右与世界坐标一致。
        """
        if self.gta_sensor is None or self.robot is None:
            return
        try:
            pos, quat = self.robot.get_position_orientation()
            pos = pos.detach().cpu().numpy() if hasattr(pos, "detach") else np.asarray(pos)
            quat = quat.detach().cpu().numpy() if hasattr(quat, "detach") else np.asarray(quat)
            x, y, z, w = quat
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

            with self.cam_lock:
                distance = float(self.cam_params["distance"])
                height = float(self.cam_params["height"])
                yaw_off = math.radians(float(self.cam_params["yaw_offset_deg"]))
                look_z = float(self.cam_params["look_z_offset"])

            # 相机方位：从机器人"正后方"出发，加上水平 yaw_off 偏移
            cam_yaw = yaw + yaw_off
            cam_pos = np.array([
                float(pos[0] - distance * math.cos(cam_yaw)),
                float(pos[1] - distance * math.sin(cam_yaw)),
                float(pos[2] + height),
            ])
            target = np.array([float(pos[0]), float(pos[1]), float(pos[2]) + look_z])

            # lookAt：USD camera 视线 = -Z，up = +Y
            forward = target - cam_pos
            forward /= max(np.linalg.norm(forward), 1e-9)
            world_up = np.array([0.0, 0.0, 1.0])
            # 当 forward 几乎竖直时退化，用机器人前向当近似 up
            if abs(np.dot(forward, world_up)) > 0.95:
                world_up = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
            right = np.cross(world_up, -forward)  # camera +X
            right /= max(np.linalg.norm(right), 1e-9)
            cam_up = np.cross(-forward, right)    # camera +Y
            cam_up /= max(np.linalg.norm(cam_up), 1e-9)
            # 相机 +Z 在世界系下 = -forward
            R = np.column_stack([right, cam_up, -forward]).astype(np.float64)
            quat_xyzw = _mat3_to_quat_xyzw(R)

            import torch as th
            self.gta_sensor.set_position_orientation(
                position=th.tensor(cam_pos, dtype=th.float32),
                orientation=th.tensor(quat_xyzw, dtype=th.float32),
            )
            # 暴露给 HUD：用来把世界三轴投影到 2D 屏幕
            self._gta_cam_R = R
        except Exception as e:
            self.log(f"WARN update GTA cam pose failed: {e}")

    def _read_camera_rgb(self, feed: str, sensor: Any) -> Optional[np.ndarray]:
        health = self.camera_health[feed]
        if health.snapshot()["degraded"]:
            attempted = self._maybe_reconnect_robot_camera(
                feed,
                sensor,
                reason=health.snapshot()["last_error"] or "consecutive read failures",
            )
            if not attempted:
                return None
        try:
            with self.camera_io_lock:
                # UI 监控视图只需 rgb：若该 sensor 还挂着 seg/depth/normal（capture/工具
                # 加的），临时把 modality 收窄成 rgb，跳过 seg/depth 昂贵的 get_data 回读+
                # remap。annotator 仍挂在 render product 上不动，工具后续直接 sensor.get_obs()
                # 仍拿到完整 seg/depth。sim 主线程串行 + 同一 camera_io_lock，收窄/还原无并发风险。
                saved_mods = getattr(sensor, "_modalities", None)
                narrow = (
                    isinstance(saved_mods, (set, frozenset))
                    and "rgb" in saved_mods
                    and len(saved_mods) > 1
                )
                if narrow:
                    sensor._modalities = {"rgb"}
                try:
                    sensor_obs, _ = sensor.get_obs()
                finally:
                    if narrow:
                        sensor._modalities = saved_mods
        except Exception as exc:
            self._record_camera_failure(
                feed,
                None,
                f"get_obs failed: {type(exc).__name__}: {exc}",
            )
            return None
        rgb = sensor_obs.get("rgb") if isinstance(sensor_obs, dict) else None
        arr = rgb_array(rgb)
        if arr is None:
            self._record_camera_failure(
                feed,
                rgb,
                "get_obs returned missing, empty, or malformed rgb",
            )
            return None
        if not self._record_camera_success(feed, rgb):
            return None
        return arr

    def _grab_real_frames(self, obs) -> Dict[str, np.ndarray]:
        """Read simulator camera feeds independently and recover broken render products.

        仅抓取处于「开启」状态的视图：关闭的视图跳过 get_obs + 颜色转换 + 后续 JPEG
        编码，省 CPU/GPU 回读；head/wrist 的渲染分辨率/质量不受影响（这里只是不取）。
        """
        result: Dict[str, np.ndarray] = {}
        active = self.get_feed_active()
        # Reaching this function means the main loop has completed the render
        # requested for newly enabled feeds. Consume the one-shot wake marker so
        # it cannot keep the simulation out of quiescence indefinitely.
        self._take_feed_warmups()

        # 主视图：GTA 俯瞰（给人看；agent/capture/move_to_object 用 head，不读此路）
        if self.gta_sensor is not None and active.get("main", False):
            arr = self._read_camera_rgb("main", self.gta_sensor)
            if arr is not None:
                img = convert_rgb_frame(
                    arr,
                    lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
                if img.shape[1] != self.main_w or img.shape[0] != self.main_h:
                    img = cv2.resize(img, (self.main_w, self.main_h))
                result["main"] = img

        # 机器人相机：head / 腕部 → 副视图（与 capture 主图同源）
        try:
            sensor_items = list(self.robot.sensors.items())
        except Exception as exc:
            if not self._stopped:
                self.log(f"WARN robot camera enumeration failed: {type(exc).__name__}: {exc}")
            sensor_items = []
        seen_feeds = set()
        for sensor_name, sensor in sensor_items:
            feed_key = self._robot_camera_feed(sensor_name)
            if feed_key is None:
                continue
            seen_feeds.add(feed_key)
            # 关闭的副视图跳过取帧（不影响其渲染质量与分辨率，仅省一次 get_obs+编码）
            if not active.get(feed_key, False):
                continue
            try:
                if "rgb" not in getattr(sensor, "modalities", []):
                    self._record_camera_failure(
                        feed_key,
                        None,
                        "rgb modality missing from sensor",
                    )
                    continue
                arr = self._read_camera_rgb(feed_key, sensor)
                if arr is None:
                    continue
                img = convert_rgb_frame(
                    arr,
                    lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
                if feed_key == "head":
                    # head 保持原生 720×720，与 capture / VLM 反投影一致
                    result[feed_key] = img
                else:
                    result[feed_key] = cv2.resize(img, (self.sub_w, self.sub_h))
            except Exception as exc:
                self._record_camera_failure(
                    feed_key,
                    None,
                    f"frame conversion failed: {type(exc).__name__}: {exc}",
                )
        for feed_key in ("head", "left_wrist", "right_wrist"):
            if feed_key not in seen_feeds:
                self._record_camera_failure(
                    feed_key,
                    None,
                    "sensor disappeared from robot.sensors",
                )

        return result

    def _grab_gta_main_frame(self) -> Dict[str, np.ndarray]:
        """只取 GTA 第三人称主视图 RGB，不碰机器人 head/wrist 的 seg obs。

        供 no_obs 快速运动期间保持主视图实时跟随机器人；GTA sensor 仅 rgb，
        不触发分割 remap，因而不会有空 buffer 崩溃，也比整套 obs 便宜得多。
        """
        result: Dict[str, np.ndarray] = {}
        if self.gta_sensor is None:
            return result
        try:
            arr = self._read_camera_rgb("main", self.gta_sensor)
            if arr is not None:
                img = convert_rgb_frame(
                    arr,
                    lambda frame: cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
                if img.shape[1] != self.main_w or img.shape[0] != self.main_h:
                    img = cv2.resize(img, (self.main_w, self.main_h))
                result["main"] = img
        except Exception as e:
            self.log(f"WARN gta main frame failed: {e}")
        return result

    def _grab_mock_frames(self) -> Dict[str, np.ndarray]:
        """dry-run：合成动画帧（机器人小圆点 + 朝向）。"""
        pose = self.world.robot_pose()
        bx, by = float(pose.pos[0]), float(pose.pos[1])
        yaw = float(pose.yaw)

        # 主图：俯视棋盘
        main = np.full((self.main_h, self.main_w, 3), (24, 28, 38), dtype=np.uint8)
        # 棋盘格
        step = 60
        for x in range(0, self.main_w, step):
            cv2.line(main, (x, 0), (x, self.main_h), (40, 44, 56), 1)
        for y in range(0, self.main_h, step):
            cv2.line(main, (0, y), (self.main_w, y), (40, 44, 56), 1)

        # 世界 -> 像素：中心是 (0,0)，1m = 60px，y 向下取反
        def w2p(wx, wy):
            px = int(self.main_w / 2 + wx * 60.0)
            py = int(self.main_h / 2 - wy * 60.0)
            return px, py

        # 几个假物体
        for label, (wx, wy), color in [
            ("microwave", (6.1, -0.7), (50, 200, 200)),
            ("popcorn_bag", (7.5, -0.5), (180, 180, 60)),
            ("countertop", (5.5, 0.5), (120, 120, 120)),
        ]:
            px, py = w2p(wx, wy)
            cv2.rectangle(main, (px - 20, py - 20), (px + 20, py + 20), color, -1)
            cv2.putText(main, label, (px - 30, py - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 机器人
        rpx, rpy = w2p(bx, by)
        cv2.circle(main, (rpx, rpy), 14, (60, 230, 60), -1)
        nx = int(rpx + 30 * math.cos(yaw))
        ny = int(rpy - 30 * math.sin(yaw))
        cv2.arrowedLine(main, (rpx, rpy), (nx, ny), (255, 255, 255), 2, tipLength=0.4)

        # 副视图：纯色 + 文字
        head = make_placeholder(self.head_w, self.head_h, "head (mock)", bg=(20, 30, 50))
        lw = make_placeholder(self.sub_w, self.sub_h, "L wrist (mock)", bg=(40, 20, 50))
        rw = make_placeholder(self.sub_w, self.sub_h, "R wrist (mock)", bg=(20, 50, 30))
        # 随时间变化的圆点表明帧在更新
        ph = int((time.time() * 60) % self.sub_w)
        cv2.circle(head, (int((time.time() * 60) % self.head_w), self.head_h // 2), 6, (255, 255, 255), -1)
        for img in (lw, rw):
            cv2.circle(img, (ph, self.sub_h // 2), 6, (255, 255, 255), -1)

        return {"main": main, "head": head, "left_wrist": lw, "right_wrist": rw}

    def _decorate(self, frames: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """主图加 HUD + 标题，副图加 label。"""
        state = self.snapshot_state()
        main = frames.get("main")
        if main is not None:
            yaw = math.radians(state["base_pose"]["yaw_deg"])
            main = draw_topbar(main, {
                "task": state["task"], "robot": state["robot"],
                "tick": state["tick"], "fps": state["fps"],
                "base_pose": state["base_pose"],
                "active_skill": state["active_skill"],
            })
            main = draw_axes_overlay(main, yaw_rad=yaw, radius=70, cam_R=self._gta_cam_R)
            frames["main"] = main
        for k, label in [
            ("head", "HEAD"),
            ("left_wrist", "LEFT WRIST"),
            ("right_wrist", "RIGHT WRIST"),
        ]:
            img = frames.get(k)
            if img is not None:
                frames[k] = label_image(img, label)
        return frames

    # ------------------------------------------------------------------ main loop

    def step_once(self) -> None:
        """单步：1) 取下一 skill action 2) 物理 step N 次 3) 渲染并更新画面/状态。"""
        # 任何动作之前先看一下是否有 task switch / RESET 请求；PhysX 操作必须在主线程做。
        if self._maybe_do_task_switch():
            self._arm_idle_settle("task_switch")
            return
        if self._maybe_do_reset():
            self._arm_idle_settle("reset")
            return
        if getattr(self, "_simulation_degraded", False):
            self._wait_for_idle_wake()
            return
        if self._maybe_hold_quiescent_idle():
            self._wait_for_idle_wake()
            return

        # Shared by dry-run and real-simulation frame finalization below.
        _idle_reuse = False
        if self.dry_run:
            self._maybe_start_next_skill()
            action = self._tick_skill()
            time.sleep(1.0 / self.target_hz)
            frames = self._grab_mock_frames()
        else:
            if self.env is None or self.robot is None or self.world is None or getattr(self.world, "robot", None) is None:
                self._enter_simulation_degraded(
                    "simulation handles are unavailable; restart or task switch needed"
                )
                self._wait_for_idle_wake()
                return
            else:
                integrity_error = self._simulation_integrity_error()
                if integrity_error is not None:
                    self._enter_simulation_degraded(integrity_error)
                    self._wait_for_idle_wake()
                    return
                try:
                    import torch as th
                    obs = None
                    # 视图按需渲染：读当前开关。关闭的视图不抓帧/不编码；快速运动路径下
                    # 主视图关闭还会整帧跳过 render()（不暂停任何 render product，稳定优先）。
                    active_feeds = self.get_feed_active()
                    main_active = bool(active_feeds.get("main", False))
                    any_active = any(active_feeds.values())
                    fast_no_obs = bool(getattr(self.world, "_codex_fast_motion_no_obs", False))
                    # 空闲 tick：无 skill 时不需要 seg/obs，避免 env.step 的完整观测管线；
                    # 纯物理子步 + 一次 RTX render 取 RGB，帧内容不变、去掉双次渲染。
                    idle_tick = (self.current_job is None) and (not fast_no_obs)
                    _diag = {"t0": time.time(), "idle": bool(idle_tick), "had_job": self.current_job is not None}
                    # 空闲静止复用上一帧标记：为 True 时本拍跳过补帧/装饰/写帧，web 服上一帧。
                    # 细分计时累加器（诊断空闲慢的真正热点）；_step_action_no_obs 内部会写
                    # prestep/simstep 两项。
                    self._step_diag = {"sync": 0.0, "start": 0.0, "tick": 0.0,
                                       "prestep": 0.0, "simstep": 0.0, "envstep": 0.0}
                    _sd = self._step_diag
                    # 空闲且机器人静止时只推进 1 个物理子步（其余 skill/运动路径仍满额
                    # physics_per_render）：空闲物体不动，多推的子步纯属浪费 CPU，
                    # 每 tick 从 ~4×1.3s 降到 ~1×1.3s，相机/视角调节立即跟手；
                    # skill/reset 时恢复满额，物理保真度不变。
                    _n_sub = 1 if idle_tick else self.physics_per_render
                    for _pi in range(_n_sub):
                        # Apply the state left by the previous generator yield before
                        # the next skill line can read a robot camera.
                        fast_no_obs = bool(getattr(self.world, "_codex_fast_motion_no_obs", False))
                        _c0 = time.time()
                        self._sync_fast_motion_camera_rendering(fast_no_obs)
                        _sd["sync"] += time.time() - _c0
                        _c0 = time.time()
                        self._maybe_start_next_skill()
                        _sd["start"] += time.time() - _c0
                        hard_lock_tool_roll = getattr(
                            self.world,
                            "hard_lock_tool_roll_pins",
                            None,
                        )
                        if callable(hard_lock_tool_roll):
                            hard_lock_tool_roll()
                        # skill 刚入队：本 tick 剩余子步改走正常路径，避免漏掉首帧 obs
                        if idle_tick and self.current_job is not None:
                            idle_tick = False
                        _c0 = time.time()
                        action = self._tick_skill()
                        _sd["tick"] += time.time() - _c0
                        if action is None:
                            action = (
                                self.world.hold_action()
                                if self._vision_safe_state
                                else self.world.set_base_velocity(0.0, 0.0, 0.0)
                            )
                        enforce_gripper_pins = getattr(
                            self.world,
                            "enforce_gripper_close_keepalive",
                            None,
                        )
                        if not callable(enforce_gripper_pins):
                            enforce_gripper_pins = getattr(
                                self.world,
                                "enforce_gripper_effort_pins",
                                None,
                            )
                        if callable(enforce_gripper_pins):
                            action = enforce_gripper_pins(action)
                        # skill generator may enter or leave no_obs while producing this
                        # action, so switch render products before advancing physics.
                        fast_no_obs = bool(getattr(self.world, "_codex_fast_motion_no_obs", False))
                        _c0 = time.time()
                        self._sync_fast_motion_camera_rendering(fast_no_obs)
                        _sd["sync"] += time.time() - _c0
                        if isinstance(action, np.ndarray):
                            action_t = th.from_numpy(action.astype(np.float32))
                        else:
                            action_t = action
                        _last_sub = (_pi == _n_sub - 1)
                        if fast_no_obs or idle_tick or (not _last_sub):
                            self._step_action_no_obs(action_t)
                            if fast_no_obs:
                                obs = None
                        else:
                            # 有 skill 且最后一子步：完整 env.step 产出 obs（含 seg）
                            _c0 = time.time()
                            obs, _, _, _, _ = self.env.step(action_t)
                            _sd["envstep"] += time.time() - _c0
                            hard_lock_tool_roll = getattr(
                                self.world,
                                "hard_lock_tool_roll_pins",
                                None,
                            )
                            if callable(hard_lock_tool_roll):
                                hard_lock_tool_roll()
                        shortcut_stabilize = getattr(self.world, "shortcut_post_step_stabilize_now", None)
                        if callable(shortcut_stabilize):
                            try:
                                shortcut_stabilize()
                            except Exception:
                                pass
                        # 实时建图里程：逐物理子步积分 base_qvel（合规本体感知）
                        self._spatial_odom_tick()
                    fast_no_obs = bool(getattr(self.world, "_codex_fast_motion_no_obs", False))
                    _diag["phys"] = time.time()
                    _diag["render"] = _diag["phys"]
                    _diag["grab"] = _diag["phys"]
                    if fast_no_obs:
                        # no_obs 运动期间唯一的渲染就是给 GTA 第三人称主视图用。
                        # 主视图开启时才 render+抓帧，让画面实时跟随机器人；主视图关闭时
                        # 整帧跳过 RTX，运动 tick 退化为纯物理推进 → 显著提速、减少 tool timeout。
                        frames = {}
                        if main_active:
                            try:
                                self._update_gta_camera_pose()
                                import omnigibson as _og2
                                _og2.sim.render()
                                frames.update(self._grab_gta_main_frame())
                            except Exception as e:
                                self.log(f"WARN no_obs GTA refresh failed: {e}")
                        shortcut_stabilize = getattr(self.world, "shortcut_post_step_stabilize_now", None)
                        if callable(shortcut_stabilize):
                            try:
                                shortcut_stabilize()
                            except Exception:
                                pass
                    elif idle_tick:
                        # 空闲：静止且画面不变时跳过 og.sim.render()+抓帧、复用上一帧；
                        # 相机被调/机器人移动/视图刚开/保活到才重渲染。全部视图关闭则整帧免渲染。
                        if main_active:
                            self._update_gta_camera_pose()
                        if any_active and self._idle_should_render(time.time()):
                            try:
                                import omnigibson as _og2
                                _og2.sim.render()
                            except Exception:
                                pass
                            _diag["render"] = time.time()
                            frames = self._grab_real_frames(None)
                            _diag["grab"] = time.time()
                            self._render_idle_last_ts = time.time()
                            self._gta_cam_dirty = False
                            try:
                                with self.state_lock:
                                    self._render_idle_last_pose = dict(self._cached_robot_pose)
                            except Exception:
                                self._render_idle_last_pose = None
                        else:
                            # 静止复用上一帧：不渲染、不抓帧、不写帧（保持旧 frame_id）
                            frames = {}
                            if any_active:
                                _idle_reuse = True
                    else:
                        if main_active:
                            self._update_gta_camera_pose()
                        # 关键：相机位置变了之后让 RTX 再走一帧，否则 annotator 数据是上一帧
                        if any_active:
                            try:
                                import omnigibson as _og2
                                _og2.sim.render()
                            except Exception:
                                pass
                        shortcut_stabilize = getattr(self.world, "shortcut_post_step_stabilize_now", None)
                        if callable(shortcut_stabilize):
                            try:
                                shortcut_stabilize()
                            except Exception:
                                pass
                        frames = self._grab_real_frames(obs) if any_active else {}
                except Exception as e:
                    tb = traceback.format_exc()
                    self.log(f"env.step error: {e}\n{tb}")
                    integrity_error = self._simulation_integrity_error()
                    if integrity_error is not None:
                        self._enter_simulation_degraded(
                            f"{integrity_error}; "
                            f"last_step_error={type(e).__name__}: {e}"
                        )
                        self._wait_for_idle_wake()
                        return
                    frames = {"main": make_placeholder(self.main_w, self.main_h, f"sim error: {e}")}

        # 缺帧用 placeholder 补。fast no_obs 下副视图故意不写回：
        # Head / Wrist 保持旧 frame_id，web 不会重复 JPEG 编码和推流。
        # 关闭的视图完全不写回（保持旧 frame_id、不 bump、不编码），前端会隐藏它。
        fast_no_obs = bool(getattr(self.world, "_codex_fast_motion_no_obs", False))
        fill_active = self.get_feed_active()
        if (not _idle_reuse) and fill_active.get("main", False) and "main" not in frames:
            if fast_no_obs:
                with self.frame_lock:
                    frames["main"] = self.frames.get("main")
            if frames.get("main") is None:
                frames["main"] = make_placeholder(self.main_w, self.main_h, "no main feed")
        for k in ("head", "left_wrist", "right_wrist"):
            if _idle_reuse:
                break
            if not fill_active.get(k, False):
                continue
            if k not in frames:
                if fast_no_obs:
                    continue
                health = self.camera_health.get(k)
                snap = health.snapshot() if health is not None else {}
                status = (
                    "degraded"
                    if snap.get("degraded")
                    else ("stale" if snap.get("stale") else "unavailable")
                )
                placeholder = self._camera_failure_placeholders.get((k, status))
                if placeholder is None:
                    width = self.head_w if k == "head" else self.sub_w
                    height = self.head_h if k == "head" else self.sub_h
                    placeholder = make_placeholder(
                        width,
                        height,
                        f"{k} camera {status}",
                    )
                    self._camera_failure_placeholders[(k, status)] = placeholder
                frames[k] = placeholder

        if not _idle_reuse:
            frames = self._decorate(frames)
        try:
            _diag["frame"] = time.time()
        except Exception:
            _diag = None

        # 3) tick + fps
        self.tick += 1
        now = time.time()
        dt = max(now - self._last_tick_ts, 1e-3)
        self._last_tick_ts = now
        inst = 1.0 / dt
        self._fps_ema = 0.9 * self._fps_ema + 0.1 * inst
        self.fps = self._fps_ema

        # 在主 sim 线程里刷新 state cache（web 线程只读 cache，绝不访问 PhysX）
        _t_pre_state = time.time()
        self._refresh_state_cache()
        _t_post_state = time.time()
        _t_goal_dur = 0.0
        _t_mem_dur = 0.0
        _t_sg_dur = 0.0
        fast_no_obs = bool(getattr(self.world, "_codex_fast_motion_no_obs", False))
        has_job = self.current_job is not None
        if not fast_no_obs:
            skill_just_finished = self._prev_had_job and not has_job
            finished_skill_name = (
                getattr(self, "_last_finished_skill_name", None)
                if skill_just_finished
                else None
            )
            world_state_changed = (
                skill_just_finished
                and self._skill_changes_world_state(finished_skill_name)
            )
            # 顶部任务完成条件 chip：限频刷新，避免空闲时 20Hz 反复跑 BDDL 评估浪费算力。
            # goals 只在 skill 改变世界后才变；skill 刚结束的那一 tick 强制刷新，保证
            # harness 下一 turn 读 /api/state 时完成判定不迟滞（turn 图像/memory 时效由
            # capture skill 自身保证，不依赖此后台刷新）。
            if world_state_changed or (now - self._goal_cache_last_ts) >= self._goal_cache_interval_s:
                _g0 = time.time()
                self._refresh_goal_cache()
                _t_goal_dur = time.time() - _g0
                self._goal_cache_last_ts = now
            # 右侧 Memory 面板：已绑定 capture 图时内部早退；未绑定时按「静止+保活间隔」
            # 限频，避免每拍读 head 深度导致空闲卡顿（periodic=True）。
            _m0 = time.time()
            self._refresh_memory_cache(periodic=True)
            _t_mem_dur = time.time() - _m0
            # Scene Graph 重建（单次 ~10s，同步阻塞主循环）：
            #  - skill 刚结束：强制重建一次，保证下一 turn 导航/overlay 看到最新布局；
            #  - skill 运行中：禁止后台重建；skill generator 必须连续推进物理帧，
            #    否则同步构图会把一次动作帧拉长到十几秒并触发 wall-clock timeout；
            #  - 纯空闲：仅当机器人相对上次重建移动过才重建，静止时完全跳过。
            _sg0 = time.time()
            self._maybe_refresh_scene_graph_after_step(
                has_job=has_job,
                skill_just_finished=world_state_changed,
            )
            _t_sg_dur = time.time() - _sg0
            # 实时建图：底盘在动时约 5Hz、静止时约 0.5Hz 并入一帧 head depth
            self._spatial_map_tick(now)
            if skill_just_finished:
                self._finish_idle_transition_after_skill(finished_skill_name)
                self._last_finished_skill_name = None
            if not has_job:
                self._maybe_release_native_memory(reason="idle")
        self._prev_had_job = has_job

        # 分段计时诊断：慢 tick（>1s）打印各段耗时，定位空闲卡顿真正瓶颈。
        try:
            _total = time.time() - _diag["t0"] if isinstance(_diag, dict) and "t0" in _diag else 0.0
            if _total > 1.0:
                _phys = (_diag.get("phys", _diag["t0"]) - _diag["t0"]) if isinstance(_diag, dict) else 0.0
                _render = (_diag.get("render", _diag.get("phys", 0)) - _diag.get("phys", 0)) if isinstance(_diag, dict) else 0.0
                _grab = (_diag.get("grab", _diag.get("render", 0)) - _diag.get("render", 0)) if isinstance(_diag, dict) else 0.0
                _sd = getattr(self, "_step_diag", {}) or {}
                self.log(
                    "STEP_DIAG total=%.2fs idle=%s had_job=%s | phys=%.2f[prestep=%.2f simstep=%.2f sync=%.2f tick=%.2f start=%.2f envstep=%.2f] render=%.2f grab=%.2f | state=%.2f goal=%.2f mem=%.2f sg=%.2f"
                    % (
                        _total,
                        _diag.get("idle") if isinstance(_diag, dict) else "?",
                        _diag.get("had_job") if isinstance(_diag, dict) else "?",
                        _phys,
                        _sd.get("prestep", 0.0),
                        _sd.get("simstep", 0.0),
                        _sd.get("sync", 0.0),
                        _sd.get("tick", 0.0),
                        _sd.get("start", 0.0),
                        _sd.get("envstep", 0.0),
                        _render,
                        _grab,
                        (_t_post_state - _t_pre_state),
                        _t_goal_dur,
                        _t_mem_dur,
                        _t_sg_dur,
                    )
                )
        except Exception:
            pass

        if not _idle_reuse:
            self._update_frames(frames)

        # 空闲限速：无 skill 运行时把主循环压到 target_hz，省 GPU/CPU；skill 执行期间
        # （含 fast_no_obs 运动）不限速，保证物理推进速度、减少 tool timeout。
        if (not self.dry_run) and (not fast_no_obs) and (not has_job):
            sleep_s = (1.0 / self.target_hz) - (time.time() - now)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def run_forever(self) -> None:
        try:
            while not self._stopped:
                self.step_once()
        except KeyboardInterrupt:
            self.log("KeyboardInterrupt -> stop")
        finally:
            self.stop()

    def stop(self) -> None:
        with self.camera_io_lock:
            self._stopped = True
            for health in self.camera_health.values():
                health.stop()


# ---------------------------------------------------------------------------
# main 入口
# ---------------------------------------------------------------------------


def _supervisor_arg_value(argv: List[str], opt: str, default: Optional[str] = None) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == opt and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(opt + "="):
            return arg.split("=", 1)[1]
    return default


def _cmdline_for_pid(pid: int) -> List[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            data = f.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return []
    return [part.decode("utf-8", "replace") for part in data.split(b"\0") if part]


def _pid_is_running(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="replace") as f:
            fields = f.read().split()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return False
    return len(fields) >= 3 and fields[2] != "Z"


def _interface_pids_for_port(port: int) -> List[tuple[int, List[str]]]:
    matches: List[tuple[int, List[str]]] = []
    try:
        names = os.listdir("/proc")
    except OSError:
        return matches
    want_port = str(port)
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        argv = _cmdline_for_pid(pid)
        if not argv:
            continue
        joined = " ".join(argv)
        if "behavior_interface.server" not in joined:
            continue
        if _supervisor_arg_value(argv, "--port") != want_port:
            continue
        matches.append((pid, argv))
    return matches


def _signal_pid_or_group(pid: int, sig: int) -> None:
    pgid = os.getpgid(pid)
    if pgid == pid:
        os.killpg(pgid, sig)
    else:
        os.kill(pid, sig)


def _terminate_pid(pid: int, timeout_s: float = 20.0) -> bool:
    import signal

    if pid <= 1 or pid == os.getpid() or not _pid_is_running(pid):
        return True
    try:
        _signal_pid_or_group(pid, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError):
        return True
    except Exception as e:
        print(f"TASK SWITCH supervisor cleanup warn: SIGTERM pid={pid} failed: {e}", flush=True)
        return False

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.5)

    try:
        _signal_pid_or_group(pid, signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError):
        return True
    except Exception as e:
        print(f"TASK SWITCH supervisor cleanup warn: SIGKILL pid={pid} failed: {e}", flush=True)
        return False

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.25)
    return not _pid_is_running(pid)


def _cleanup_stale_port_processes(port: int, keep_pids: set[int], reason: str) -> None:
    stale: List[tuple[int, str, str]] = []
    keep = set(keep_pids)
    keep.add(os.getpid())
    for pid, argv in _interface_pids_for_port(port):
        if pid in keep:
            continue
        task = _supervisor_arg_value(argv, "--task", "") or ""
        scene = _supervisor_arg_value(argv, "--scene", "") or ""
        label = f"{task}@{scene}" if task or scene else "unknown"
        stale.append((pid, label, " ".join(argv)))

    if not stale:
        return

    print(
        f"TASK SWITCH supervisor cleanup stale port={port} reason={reason}: "
        + ", ".join(f"pid={pid} target={label}" for pid, label, _ in stale),
        flush=True,
    )
    for pid, _label, _cmd in stale:
        ok = _terminate_pid(pid)
        print(f"TASK SWITCH supervisor cleanup pid={pid} ok={ok}", flush=True)


def _safe_path_component(text: str, max_len: int = 96) -> str:
    safe = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("._")
    return (out or "unknown")[:max_len]


def _task_switch_appdata_base(base_env: Dict[str, str], port: int) -> str:
    base = (
        base_env.get("INTERFACE_TASK_SWITCH_APPDATA_BASE")
        or base_env.get("OMNIGIBSON_APPDATA_PATH")
        or f"/tmp/og_appdata_{base_env.get('USER') or 'bince'}_interface_{port}"
    )
    marker = f"{os.sep}task_switch_retries{os.sep}"
    if marker in base:
        base = base.split(marker, 1)[0]
    return base


def _interface_log_path_for_port(port: int) -> str:
    return os.environ.get("INTERFACE_LOG_PATH") or f"/tmp/interface_{port}.log"


def _open_interface_stdio_for_port(port: int):
    log_path = _interface_log_path_for_port(port)
    try:
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return open(log_path, "ab", buffering=0)
    except OSError:
        return open(os.devnull, "ab", buffering=0)


def _attempt_appdata_path(base_env: Dict[str, str], port: int, target_task: str, target_scene: str, attempt: int) -> Optional[str]:
    disable_raw = base_env.get("INTERFACE_TASK_SWITCH_DISABLE_APPDATA_ISOLATION")
    disable = (
        _env_truthy("INTERFACE_TASK_SWITCH_DISABLE_APPDATA_ISOLATION", False)
        if disable_raw is None
        else str(disable_raw).strip().lower() not in {"", "0", "false", "no", "off", "none"}
    )
    if disable:
        return None
    mode = str(base_env.get("INTERFACE_TASK_SWITCH_ISOLATE_APPDATA", "retry")).strip().lower()
    if mode in {"0", "false", "no", "off", "none"}:
        mode = "retry"
    if mode in {"retry", "retries", "failed", "after_failure"} and attempt <= 1:
        return None

    base = _task_switch_appdata_base(base_env, port)
    target = _safe_path_component(f"{target_task}_{target_scene}", max_len=120)
    path = os.path.join(base, "task_switch_retries", target, f"supervisor_{os.getpid()}", f"attempt_{attempt}")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"TASK SWITCH supervisor warn: cannot create appdata path={path}: {e}", flush=True)
        return None
    return path


def _retry_delay_for_reason(reason: str, retry_delay_s: float, gpu_crash_delay_s: float) -> float:
    text = str(reason).lower()
    if "rc=-11" in text or "sigsegv" in text or "gpu crash" in text or "device_lost" in text:
        return max(retry_delay_s, gpu_crash_delay_s)
    return retry_delay_s


def _ensure_runtime_env_defaults() -> None:
    # Avoid duplicate Vulkan ICD enumeration, which can make Kit see the same
    # NVIDIA GPU twice and crash during early GPU foundation startup.
    nvidia_icd = "/etc/vulkan/icd.d/nvidia_icd.json"
    if not os.environ.get("VK_ICD_FILENAMES") and os.path.exists(nvidia_icd):
        os.environ["VK_ICD_FILENAMES"] = nvidia_icd
    # Reuse the port's validated appdata across task-switch retries. A fresh
    # isolated appdata lacks the Kit extension registry and can fail before the
    # simulator is able to publish a ready signal.
    os.environ.setdefault("INTERFACE_TASK_SWITCH_DISABLE_APPDATA_ISOLATION", "1")


# Compatibility alias retained for callers/tests that inspect the server's
# mapping.  The authoritative table lives in the lightweight gpu_diag module
# so all Python entrypoints validate the same ownership contract.
_FIXED_GPU_BY_INTERFACE_PORT = dict(FIXED_GPU_BY_PORT)


def _enforce_fixed_gpu_for_port(port: int) -> None:
    gpu = fixed_gpu_for_port(port)
    if gpu is None:
        return
    try:
        enforce_fixed_gpu_environment(port)
    except ValueError as exc:
        raise SystemExit(f"[server] GPU ownership validation failed: {exc}") from exc
    user = os.environ.get("USER") or "bince"
    os.environ.setdefault("OMNIGIBSON_APPDATA_PATH", f"/var/tmp/og_appdata_{user}_interface_{port}_gpu{gpu}")


def _wait_for_child_ready(
    proc: Any,
    port: int,
    timeout_s: float,
    target_task: str,
    target_scene: str,
    target_robot_dof: int,
) -> tuple[bool, str]:
    import json
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/api/state"
    deadline = time.time() + timeout_s
    last_error = ""
    last_cleanup_ts = 0.0
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            return False, f"child exited before ready rc={rc}"
        try:
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                status = int(resp.status)
                body = resp.read()
                if 200 <= status < 500:
                    try:
                        state = json.loads(body.decode("utf-8"))
                    except Exception as e:
                        state = {}
                        last_error = f"state json error={e}"
                    if isinstance(state, dict):
                        task = str(state.get("task") or "")
                        scene = str(state.get("scene") or "")
                        try:
                            robot_dof = int(state.get("robot_dof"))
                        except (TypeError, ValueError):
                            robot_dof = 0
                        state_pid_raw = state.get("pid")
                        try:
                            state_pid = int(state_pid_raw) if state_pid_raw is not None else 0
                        except (TypeError, ValueError):
                            state_pid = 0
                        if (
                            task == target_task
                            and scene == target_scene
                            and robot_dof == int(target_robot_dof)
                        ):
                            if not state_pid or state_pid == proc.pid:
                                return True, (
                                    f"ready status={status} pid={state_pid or 'unknown'} "
                                    f"task={task} scene={scene} robot_dof={robot_dof}"
                                )
                            last_error = (
                                f"target served by unexpected pid={state_pid} "
                                f"child_pid={proc.pid} task={task} scene={scene}"
                            )
                        else:
                            last_error = (
                                f"stale listener status={status} pid={state_pid or 'unknown'} "
                                f"task={task!r} scene={scene!r} robot_dof={robot_dof!r} "
                                f"expected_robot_dof={target_robot_dof}"
                            )
                    if last_error and time.time() - last_cleanup_ts >= 10.0:
                        _cleanup_stale_port_processes(
                            port,
                            keep_pids={proc.pid},
                            reason=last_error,
                        )
                        last_cleanup_ts = time.time()
                else:
                    last_error = f"status={status}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = str(e)
        time.sleep(5.0)
    return False, f"ready timeout after {timeout_s:.0f}s last_error={last_error}"


def _terminate_child(proc: Any) -> None:
    import subprocess
    import signal

    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=20.0)


def _run_task_switch_supervisor(child_argv: List[str]) -> int:
    import subprocess

    if not child_argv:
        print("TASK SWITCH supervisor error: missing child argv", flush=True)
        return 2
    port_text = _supervisor_arg_value(child_argv, "--port", "5000") or "5000"
    try:
        port = int(port_text)
    except ValueError:
        print(f"TASK SWITCH supervisor error: invalid --port {port_text!r}", flush=True)
        return 2
    target_task = _supervisor_arg_value(child_argv, "--task", "") or ""
    target_scene = _supervisor_arg_value(child_argv, "--scene", "") or ""
    target_robot_dof = normalize_robot_dof(
        _supervisor_arg_value(
            child_argv,
            "--robot-dof",
            os.environ.get("BEHAVIOR_ROBOT_DOF", "8"),
        )
    )

    attempts = max(1, int(os.environ.get("INTERFACE_TASK_SWITCH_MAX_ATTEMPTS", "8")))
    ready_timeout_s = max(60.0, float(os.environ.get("INTERFACE_TASK_SWITCH_READY_TIMEOUT_S", "3600")))
    retry_delay_s = max(0.0, float(os.environ.get("INTERFACE_TASK_SWITCH_RETRY_DELAY_S", "45")))
    gpu_crash_delay_s = max(
        retry_delay_s,
        float(os.environ.get("INTERFACE_TASK_SWITCH_GPU_CRASH_COOLDOWN_S", "60")),
    )
    base_env = os.environ.copy()
    base_env.pop("WERKZEUG_SERVER_FD", None)
    base_env.pop("LISTEN_FDS", None)
    base_env.pop("LISTEN_PID", None)
    if not _env_truthy("INTERFACE_TASK_SWITCH_DISABLE_APPDATA_ISOLATION", False):
        base_env["INTERFACE_TASK_SWITCH_ISOLATE_APPDATA"] = "retry"
    appdata_base = _task_switch_appdata_base(base_env, port)
    base_env["INTERFACE_TASK_SWITCH_APPDATA_BASE"] = appdata_base
    base_env["OMNIGIBSON_APPDATA_PATH"] = appdata_base
    base_env["BEHAVIOR_ROBOT_DOF"] = str(target_robot_dof)
    base_env["ROBOT"] = str(
        _supervisor_arg_value(child_argv, "--robot", robot_type_for_dof(target_robot_dof))
    )
    base_env["BEHAVIOR_ROBOT_CONFIG"] = str(
        _supervisor_arg_value(
            child_argv,
            "--robot-config",
            robot_config_path_for_dof(target_robot_dof),
        )
    )
    base_env.setdefault("INTERFACE_LOG_PATH", _interface_log_path_for_port(port))

    target = base_env.get("INTERFACE_TASK_SWITCH_TARGET", "")
    print(
        f"TASK SWITCH supervisor start target={target or f'{target_task}@{target_scene}'} port={port} "
        f"attempts={attempts} timeout={ready_timeout_s:.0f}s",
        flush=True,
    )
    last_reason = ""
    for attempt in range(1, attempts + 1):
        _cleanup_stale_port_processes(
            port,
            keep_pids=set(),
            reason=f"before attempt {attempt} target={target_task}@{target_scene}",
        )
        from .runtime_tmp import prepare_child_environment, prune_stale_runtime_dirs

        prune_stale_runtime_dirs()
        env = prepare_child_environment(base_env.copy())
        env["INTERFACE_TASK_SWITCH_SUPERVISED"] = "1"
        env["INTERFACE_TASK_SWITCH_ATTEMPT"] = str(attempt)
        appdata_path = _attempt_appdata_path(env, port, target_task, target_scene, attempt)
        if appdata_path:
            env["OMNIGIBSON_APPDATA_PATH"] = appdata_path
            print(
                f"TASK SWITCH supervisor attempt {attempt}/{attempts}: "
                f"OMNIGIBSON_APPDATA_PATH={appdata_path}",
                flush=True,
            )
        try:
            from .runtime_storage import preflight_storage

            storage_report = preflight_storage(env["OMNIGIBSON_APPDATA_PATH"])
            print(
                "TASK SWITCH storage preflight: "
                f"free={storage_report['free_before'] / (1024 ** 3):.1f}"
                f"->{storage_report['free_after'] / (1024 ** 3):.1f}GiB "
                f"removed={len(storage_report['removed'])}",
                flush=True,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"TASK SWITCH storage preflight failed: {exc}", flush=True)
            return 1
        print(f"TASK SWITCH supervisor attempt {attempt}/{attempts}: {' '.join(child_argv)}", flush=True)
        stdio = _open_interface_stdio_for_port(port)
        try:
            stdio.write(
                (
                    f"\n=== TASK SWITCH child attempt {attempt}/{attempts} "
                    f"target={target_task}@{target_scene} pid=supervisor:{os.getpid()} ===\n"
                ).encode("utf-8", "replace")
            )
            proc = subprocess.Popen(
                child_argv,
                cwd=os.getcwd(),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdio,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
        finally:
            stdio.close()
        ready, reason = _wait_for_child_ready(
            proc,
            port=port,
            timeout_s=ready_timeout_s,
            target_task=target_task,
            target_scene=target_scene,
            target_robot_dof=target_robot_dof,
        )
        if ready:
            print(
                f"TASK SWITCH supervisor ready pid={proc.pid} reason={reason}; supervisor exiting",
                flush=True,
            )
            return 0

        last_reason = reason
        print(
            f"TASK SWITCH supervisor attempt {attempt}/{attempts} failed: {reason}",
            flush=True,
        )
        _terminate_child(proc)
        _cleanup_stale_port_processes(
            port,
            keep_pids=set(),
            reason=f"after failed attempt {attempt} target={target_task}@{target_scene}",
        )
        delay_s = _retry_delay_for_reason(last_reason, retry_delay_s, gpu_crash_delay_s)
        if attempt < attempts and delay_s > 0:
            print(f"TASK SWITCH supervisor retrying after {delay_s:.0f}s", flush=True)
            time.sleep(delay_s)

    print(
        f"TASK SWITCH supervisor failed target={target} attempts={attempts} last={last_reason}",
        flush=True,
    )
    return 1


def main():
    # This must run before importing OmniGibson / Isaac Sim. OG creates its
    # decrypted-USD temp directory during import and otherwise uses /tmp/tmp*.
    from .runtime_tmp import configure_process_runtime_tmp

    runtime_port = _supervisor_arg_value(
        sys.argv,
        "--port",
        os.environ.get("PORT", "5000"),
    )
    configure_process_runtime_tmp(runtime_port or "5000")
    enable_faulthandler()
    _ensure_runtime_env_defaults()

    if "--task-switch-supervisor" in sys.argv:
        marker = sys.argv.index("--task-switch-supervisor")
        if marker + 1 >= len(sys.argv) or sys.argv[marker + 1] != "--":
            print("TASK SWITCH supervisor error: expected '--' after --task-switch-supervisor", flush=True)
            sys.exit(2)
        sys.exit(_run_task_switch_supervisor(sys.argv[marker + 2:]))

    import argparse

    parser = argparse.ArgumentParser(description="BEHAVIOR-1K Web 测试 interface")
    parser.add_argument("--task", default="make_microwave_popcorn",
                        help="BehaviorTask activity_name（默认 make_microwave_popcorn）")
    parser.add_argument("--robot", default=os.environ.get("ROBOT"))
    parser.add_argument(
        "--robot-dof",
        type=int,
        choices=(7, 8),
        default=normalize_robot_dof(),
        help="R1Pro arm variant: 7 or 8 (default 8)",
    )
    parser.add_argument(
        "--robot-config",
        default=os.environ.get("BEHAVIOR_ROBOT_CONFIG"),
        help=(
            "Challenge robot YAML; defaults to "
            f"{DEFAULT_CHALLENGE_ROBOT_CONFIG}"
        ),
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="Challenge scene model; omitted means use the official scene for --task",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--main-w", type=int, default=960)
    parser.add_argument("--main-h", type=int, default=540)
    parser.add_argument("--sub-w", type=int, default=320)
    parser.add_argument("--sub-h", type=int, default=240)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--physics-per-render", type=int, default=4,
                        help="每次主循环（含 4 路渲染）推进的物理 step 数，"
                             "调大让机器人移动更快（默认 4）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不启动真实仿真，用 mock 数据验证 web/cli")
    parser.add_argument("--tool-version", choices=("v0", "v1", "v1_shortcut", "v2", "v3"),
                        default=os.environ.get("INTERFACE_TOOL_VERSION", "v2"),
                        help="加载工具集版本：v0=完整当前工具集，v1=精简工具集，v1_shortcut=精简工具集+执行类工具直达关节目标，v2=v1重命名工具面，v3=v2去掉capture+新增manipulate族")
    _env_inst = os.environ.get("INTERFACE_TASK_INSTANCE", "").strip()
    _env_inst_default = int(_env_inst) if _env_inst.lstrip("-").isdigit() else None
    parser.add_argument("--instance-id", type=int, default=_env_inst_default,
                        help="BEHAVIOR challenge 预采样 task instance id；不指定则每次启动随机选一个评测 instance"
                             "（与官方评测同一套实例数据）")
    args, unknown_kit_args = parser.parse_known_args()
    _enforce_fixed_gpu_for_port(args.port)
    try:
        apply_requested_cpu_affinity()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"[server] CPU affinity validation failed: {exc}") from exc
    if not args.dry_run:
        from .runtime_storage import preflight_storage

        user = os.environ.get("USER") or str(os.getuid())
        appdata_path = os.environ.setdefault(
            "OMNIGIBSON_APPDATA_PATH",
            f"/var/tmp/og_appdata_{user}_interface_{args.port}_gpu0",
        )
        storage_report = preflight_storage(
            appdata_path,
            reserve_for_pid=os.getpid(),
        )
        print(
            "[storage-reservation] "
            f"appdata={appdata_path} "
            f"projected_free={storage_report['projected_free_bytes'] / (1024 ** 3):.1f}GiB "
            f"reserved={storage_report['requested_reservation_bytes'] / (1024 ** 3):.1f}GiB",
            flush=True,
        )
    if args.scene is None:
        from .challenge_tasks import challenge_task_scene

        args.scene = challenge_task_scene(args.task)
        if args.scene is None:
            parser.error(f"unknown BEHAVIOR 2026 task: {args.task!r}")
    os.environ["INTERFACE_TOOL_VERSION"] = args.tool_version
    import importlib
    import behavior_interface.skills as skills_pkg

    skills_pkg = importlib.reload(skills_pkg)
    globals()["SKILL_REGISTRY"] = skills_pkg.SKILL_REGISTRY
    globals()["load_all_skills"] = skills_pkg.load_all_skills
    globals()["list_skills"] = skills_pkg.list_skills

    server = BehaviorInterface(
        task=args.task,
        robot=args.robot,
        robot_dof=args.robot_dof,
        scene_model=args.scene,
        main_size=(args.main_w, args.main_h),
        sub_size=(args.sub_w, args.sub_h),
        target_hz=args.hz,
        dry_run=args.dry_run,
        physics_per_render=args.physics_per_render,
        activity_instance_id=args.instance_id,
        robot_config_path=args.robot_config,
        cli_argv=sys.argv,
    )
    enable_faulthandler(server.log)
    if os.environ.get("INTERFACE_TASK_SWITCH_REEXEC"):
        server.log(f"started after task switch re-exec target={os.environ.get('INTERFACE_TASK_SWITCH_TARGET', '')}")

    loaded = load_all_skills()
    server.log(
        f"tool_version={args.tool_version} skills loaded: "
        f"{loaded} -> {[s['name'] for s in list_skills()]}"
    )
    if unknown_kit_args:
        server.log(f"passing through kit args: {unknown_kit_args}")
    server._log_gpu_diag(
        "server.start",
        extra={
            "task": args.task,
            "scene": args.scene,
            "robot": args.robot,
            "robot_dof": server.robot_dof,
            "robot_config": server.robot_config_path,
            "port": args.port,
            "tool_version": args.tool_version,
            "dry_run": bool(args.dry_run),
        },
    )

    server.init_simulation()
    if _env_truthy("BEHAVIOR_INTERFACE_NATIVE_TRIM_POST_INIT", False):
        server._maybe_release_native_memory(reason="post_init", force=True)

    from .web import run_web_in_thread
    run_web_in_thread(server, host=args.host, port=args.port)
    server.log(f"web running at http://{args.host}:{args.port}/")

    server.run_forever()


if __name__ == "__main__":
    main()
