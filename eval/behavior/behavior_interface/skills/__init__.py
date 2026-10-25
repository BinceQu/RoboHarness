"""
Skill 注册表（装饰器风格）。

使用：

    from behavior_interface.skills import register_skill

    @register_skill("move_to", description="...")
    def move_to(ctx, x: float, y: float):
        while not done:
            action = ...
            yield action  # 主循环 step env

启动时 `load_all_skills()` 仅加载 move / move_to / plan_grasp（依赖模块会导入但不注册）。
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from behavior_interface.coordinate_contract import (
    COORDINATE_MAX,
    COORDINATE_SYSTEM,
    public_skill_wrapper,
)
from behavior_interface.skills.arm_force_guard import guard_arm_force_skill
from behavior_interface.tool import active_tool_version, load_tool_profile


def _install_empty_image_remapper_hotfix() -> bool:
    """Patch an already-running OmniGibson process without resetting its remapper caches."""
    # Importing OG launches substantial native/runtime initialization and may
    # allocate cache space. Server startup must run its storage preflight first.
    if "omnigibson" not in sys.modules:
        return False
    try:
        from omnigibson.utils import vision_utils
    except Exception:
        return False

    if getattr(vision_utils, "REMAPPER_HANDLES_EMPTY_IMAGES", False):
        return False

    original_remap = vision_utils.Remapper.remap

    def remap_with_empty_image_support(self, old_mapping, new_mapping, image, image_keys=None):
        if image.numel() == 0:
            return image.to(dtype=vision_utils.th.int32), {}
        return original_remap(self, old_mapping, new_mapping, image, image_keys)

    vision_utils.Remapper.remap = remap_with_empty_image_support
    vision_utils.REMAPPER_HANDLES_EMPTY_IMAGES = True
    print("[skills.reload] installed empty segmentation remapper hotfix", flush=True)
    return True


_EMPTY_IMAGE_REMAPPER_HOTFIX_INSTALLED = _install_empty_image_remapper_hotfix()


# 对外暴露的 skill（其余依赖模块 import 时可能短暂注册，加载后会 prune 掉）
_BASE_PUBLIC = (
    "move_in_world_coord", "move_in_robot_coord", "face_to_point", "move_eef", "plan_move_eef", "plan_move_eef_to_point", "adjust_plan_pose", "move_to", "plan_grasp", "plan_eef",
    "capture",
    "capture_left_wrist_camera", "capture_right_wrist_camera",
    "move_base_to_point",
    "diag_vertical_lift", "diag_vertical_lift_v2", "fold_deepest_phase2_q3",
    "diag_vertical_three_phase_gta", "diag_reverse_vertical_z_ik_gta",
    "diag_reverse_upward_two_phase_gta",
    "plan_eef_v2", "exec_move_v2", "exec_eef_pose_v2",
    "mark_object_v2", "move_to_object_v2", "move_to_point_v2", "move_to_point_v3", "mesure_shoulder_distance", "arm_reset", "set_arm_to_grasp_position", "set_arm_to_grasp_position_shortcut", "diag_set_arm_to_grasp_position_random_fk", "diag_keep_ori_wrist_preflight", "reset_body",
    "rotate_eef",
    "diag_gripper_frame",
    "stage_countertop_grasp_audit", "diag_opening_volume", "viz_grasp_obj_multiview",
    "diag_object_surface", "diag_grasp_obj_pipeline", "diag_grasp_obj_pipeline_v2",
    "diag_grasp_obj_pipeline_v3", "diag_grasp_point_pipeline_v1", "diag_move_to_point_ray",
    "diag_bench_v7_cpu_gpu", "diag_bench_inflate_overlap_cpu_gpu",
    "diag_popcorn_seg",
    "diag_rerender_grasp_views",
    "diag_tuck_trajectory",
    "diag_tuck_rigid_replay",
    "diag_tuck_chest_invariance",
    "diag_tuck_chest_fk_audit",
    "diag_tuck_chest_now",
    "diag_tuck_elbow_scan",
    "diag_tuck_path_strategies",
    "diag_tuck_verify_path",
    "diag_tuck_two_phase",
    "diag_tuck_joint_probe",
    "diag_tuck_lift_only",
    "diag_tuck_phase1_vertical_pose",
    "diag_tuck_phase1_j1_sign_audit",
    "diag_tuck_phase1_backswing_video",
    "diag_tuck_phase1_v2_pose",
    "diag_tuck_pillar_safe_zone",
    "diag_tuck_waist_j34_forbidden_zone",
    "diag_tuck_waist_j34_forbidden_zone_trimetric",
    "diag_tuck_phase2_feasibility",
    "diag_tuck_phase2_waist_fwd",
    "diag_tuck_phase2_three_constraints",
    "diag_tuck_phase2_four_constraints",
    "diag_tuck_phase2_j34_constraints",
    "diag_tuck_phase2_j34_breakdown",
    "diag_tuck_phase2_fwd25_lat3_audit",
    "diag_tuck_phase2_fwd25_lat3_video",
    "diag_tuck_phase2_fwd_minimum",
    "diag_tuck_hang_j34_relaxed_audit",
    "diag_tuck_hang_j34_fwd_minimum",
    "diag_tuck_hang_j34_fwd_refine_video",
    "diag_tuck_hang_j34_fwd_penalty_video",
    "diag_tuck_legacy_hard_fwd",
    "diag_tuck_rrt_waist_fwd",
    "diag_tuck_rrt_three_constraints",
    "diag_tuck_rrt_four_constraints",
    "diag_tuck_rrt_j34_constraints",
    "diag_tuck_rrt_j34_relaxed",
    "diag_eef_pose_reachability", "diag_eef_ik_accuracy", "diag_eef_common_random_ik",
    "diag_eef_compare_ik_methods",
    "diag_gripper_camera_flip_viz",
    "diag_grasp_obj_filter_camera_flip_head_viz",
    "diag_gripper_arm_adapt_viz",
    "diag_eef_basis_pose",
    "diag_q4_lock_smoke",
)
_EXTRA_PUBLIC = tuple(
    s.strip() for s in os.environ.get("INTERFACE_EXTRA_SKILLS", "").split(",") if s.strip()
)
_V0_PUBLIC_SKILLS: frozenset[str] = frozenset(_BASE_PUBLIC + _EXTRA_PUBLIC)

# 启动时需要 import 的模块（move 与 move_to 同在 move_to.py）
_V0_LOAD_MODULES: Sequence[str] = ("move_to", "face_to_point", "move_eef", "plan_move_eef", "plan_move_eef_to_point", "adjust_plan_pose", "rotate_eef", "plan_grasp", "plan_eef", "capture", "wrist_capture", "move_base_to_point", "plan_eef_v2", "exec_move_v2", "exec_eef_pose_v2", "diag_eef_pose_reachability", "diag_gripper_camera_flip", "mark_object_v2", "move_to_object_v2", "move_to_point_v2", "move_to_point_v3", "mesure_shoulder_distance", "arm_reset", "reset_body") + tuple(
    m for m in _EXTRA_PUBLIC if m not in (
        "move_in_world_coord", "move_in_robot_coord", "face_to_point", "move_eef", "plan_move_eef", "plan_move_eef_to_point", "adjust_plan_pose", "move_to", "plan_grasp", "plan_eef",
        "capture", "wrist_capture", "move_base_to_point", "plan_eef_v2", "exec_move_v2", "exec_eef_pose_v2", "mark_object_v2", "move_to_object_v2",
        "move_to_point_v2", "move_to_point_v3", "mesure_shoulder_distance", "reset_body", "rotate_eef",
    )
)

# hot-reload 时按依赖顺序 reload（plan_grasp 的 transitive import）
_V0_RELOAD_MODULES: Sequence[str] = (
    "grasp",
    "stage_countertop_grasp_audit",
    "tuck_trajectory",
    "eef",
    "vlm_grasp_verify",
    "vlm_lawn_dual",
    "vlm_scene_grasp",
    "viz_eef_v2",
    "plan_grasp_gripper_geom",
    "plan_grasp_gripper_fit",
    "plan_grasp_opening_volume",
    "viz_gripper_overlay",
    "gripper_camera_face",
    "diag_opening_volume",
    "viz_grasp_obj_multiview",
    "diag_object_surface",
    "grasp_obj_pipeline_core",
    "grasp_obj_v7_gpu",
    "grasp_point_pipeline_core",
    "plan_grasp_object",
    "plan_grasp_point",
    "diag_grasp_obj_pipeline",
    "diag_grasp_point_pipeline",
    "diag_rerender_grasp_views",
    "diag_popcorn_seg",
    "plan_grasp_core",
    "plan_eef_core",
    "plan_eef_lawn_capture",
    "trunk_vertical_lift",
    "move_to",
    "face_to_point",
    "move_eef",
    "plan_move_eef",
    "plan_move_eef_to_point",
    "adjust_plan_pose",
    "rotate_eef",
    "plan_grasp",
    "plan_eef",
    "viz_base_path_overlay",
    "capture",
    "wrist_grasp_zone_overlay",
    "wrist_capture",
    "move_base_to_point",
    "plan_eef_v2",
    "exec_move_v2",
    "exec_eef_pose_v2",
    "diag_eef_pose_reachability",
    "diag_gripper_camera_flip",
    "mark_object_v2",
    "base_chord_reach",
    "move_to_object_geom",
    "reach_point_pitch_recovery",
    "move_to_object_v2",
    "move_to_point_v2",
    "move_to_point_v3",
    "mesure_shoulder_distance",
    "diag_move_to_point_ray",
    "wrist_j567_solver",
    "arm_reset",
    "reset_body",
    "rotate_eef",
)


def _profile_sequence(profile, name: str, default: Sequence[str]) -> Sequence[str]:
    value = getattr(profile, name, None)
    return tuple(default) if value is None else tuple(str(x) for x in value)


def _append_extra_modules(modules: Sequence[str]) -> Sequence[str]:
    out = list(modules)
    seen = {str(x) for x in out}
    for mod_name in _EXTRA_PUBLIC:
        if mod_name not in seen:
            out.append(mod_name)
            seen.add(mod_name)
    return tuple(out)


def _profile_public(profile) -> frozenset[str]:
    value = getattr(profile, "PUBLIC_SKILLS", None)
    if value is None:
        return _V0_PUBLIC_SKILLS
    return frozenset(str(x) for x in value) | frozenset(_EXTRA_PUBLIC)


TOOL_VERSION = active_tool_version()
# v2/v3 profile 从 tool.v1 导入 RELOAD_MODULES；热加载时先刷 v1 再刷当前 profile
try:
    import behavior_interface.tool.v1 as _tool_v1_mod

    importlib.reload(_tool_v1_mod)
except Exception:
    pass
_ACTIVE_PROFILE = load_tool_profile(TOOL_VERSION)
try:
    _ACTIVE_PROFILE = importlib.reload(_ACTIVE_PROFILE)
except Exception:
    pass
PUBLIC_SKILLS: frozenset[str] = _profile_public(_ACTIVE_PROFILE)
LOAD_MODULES: Sequence[str] = _append_extra_modules(
    _profile_sequence(_ACTIVE_PROFILE, "LOAD_MODULES", _V0_LOAD_MODULES)
)
RELOAD_MODULES: Sequence[str] = _append_extra_modules(
    _profile_sequence(_ACTIVE_PROFILE, "RELOAD_MODULES", _V0_RELOAD_MODULES)
)
POST_LOAD_MODULES: Sequence[str] = _profile_sequence(_ACTIVE_PROFILE, "POST_LOAD_MODULES", ())
POST_RELOAD_MODULES: Sequence[str] = _profile_sequence(_ACTIVE_PROFILE, "POST_RELOAD_MODULES", POST_LOAD_MODULES)


@dataclass
class SkillSpec:
    """单个 skill 元数据。"""

    name: str
    fn: Callable
    description: str = ""
    params: List[Dict[str, Any]] = field(default_factory=list)


SKILL_REGISTRY: Dict[str, SkillSpec] = {}


def _extract_params(fn: Callable) -> List[Dict[str, Any]]:
    """从函数签名抽取参数元数据，忽略第一个 ctx 参数。"""
    params = []
    sig = inspect.signature(fn)
    for i, (pname, p) in enumerate(sig.parameters.items()):
        if i == 0:
            # 第一个总是 ctx（运行时由 server 注入）
            continue
        ann = p.annotation if p.annotation is not inspect._empty else "any"
        default = None if p.default is inspect._empty else p.default
        required = p.default is inspect._empty
        params.append(
            {
                "name": pname,
                "type": getattr(ann, "__name__", str(ann)),
                "default": default,
                "required": required,
            }
        )
    return params


def register_skill(name: str, description: str = ""):
    """装饰器：注册一个 skill。被装饰函数必须以 ctx 为第一个参数，
    返回值应为生成器，yield 出每一步 action（np.ndarray / th.Tensor / None）。
    """

    def deco(fn: Callable):
        # 热重载时允许覆盖同名 skill（importlib.reload 会再次执行装饰器）
        public_fn = public_skill_wrapper(fn)
        public_fn = guard_arm_force_skill(name, public_fn)
        SKILL_REGISTRY[name] = SkillSpec(
            name=name,
            fn=public_fn,
            description=description or (fn.__doc__ or "").strip(),
            params=_extract_params(fn),
        )
        return fn

    return deco


def _prune_registry() -> None:
    """去掉依赖模块注册的 skill，只保留 PUBLIC_SKILLS。"""
    for name in list(SKILL_REGISTRY.keys()):
        if name not in PUBLIC_SKILLS:
            del SKILL_REGISTRY[name]


def _import_module(mod_name: str) -> None:
    full = str(mod_name) if "." in str(mod_name) else f"behavior_interface.skills.{mod_name}"
    try:
        importlib.import_module(full)
    except (PermissionError, OSError) as exc:
        # admin-only files in NFS — surface in log but don't crash startup.
        print(f"[skills] skip {full}: {exc.__class__.__name__}: {exc}")


def _reload_module(mod_name: str) -> None:
    import sys
    full = str(mod_name) if "." in str(mod_name) else f"behavior_interface.skills.{mod_name}"
    old_module = sys.modules.get(full)
    if old_module is not None:
        cleanup = getattr(old_module, "shutdown_background_workers", None)
        if not callable(cleanup):
            cleanup = getattr(old_module, "_close_persistent_ik_workers", None)
        if callable(cleanup):
            cleanup()
    # 删除缓存后重新 import，避免 reload 半失败仍执行旧 skill 函数
    sys.modules.pop(full, None)
    try:
        importlib.import_module(full)
    except (PermissionError, OSError) as exc:
        print(f"[skills] reload skip {full}: {exc.__class__.__name__}: {exc}")


def load_all_skills() -> List[str]:
    """仅加载 move / move_to / plan_grasp 三个 skill。"""
    global _EMPTY_IMAGE_REMAPPER_HOTFIX_INSTALLED

    loaded = []
    for mod_name in tuple(LOAD_MODULES) + tuple(POST_LOAD_MODULES):
        _import_module(mod_name)
        loaded.append(mod_name)
    _EMPTY_IMAGE_REMAPPER_HOTFIX_INSTALLED = (
        _install_empty_image_remapper_hotfix()
        or _EMPTY_IMAGE_REMAPPER_HOTFIX_INSTALLED
    )
    _prune_registry()
    return loaded


def reload_all_skills() -> List[str]:
    """热重载 skill 模块并刷新注册表。"""
    import sys
    loaded: List[str] = []
    try:
        import behavior_interface.agent_runs as agent_runs_mod
        importlib.reload(agent_runs_mod)
        loaded.append("agent_runs")
    except Exception:
        pass
    try:
        import behavior_interface.coordinate_contract as coordinate_contract_mod
        importlib.reload(coordinate_contract_mod)
        loaded.append("coordinate_contract")
    except Exception:
        pass
    try:
        import behavior_interface.trunk_vertical_lift as tvl_mod
        importlib.reload(tvl_mod)
        loaded.append("trunk_vertical_lift")
    except Exception:
        pass
    # scene_graph 非 skills 子包，但 move_to_object 规划依赖其新 API，须一并 reload
    try:
        import behavior_interface.scene_graph as sg_mod
        importlib.reload(sg_mod)
        loaded.append("scene_graph")
    except Exception:
        pass
    import behavior_interface.skills as skills_pkg
    importlib.reload(skills_pkg)
    globals().update({
        k: getattr(skills_pkg, k)
        for k in (
            "TOOL_VERSION",
            "PUBLIC_SKILLS",
            "LOAD_MODULES",
            "RELOAD_MODULES",
            "_EXTRA_PUBLIC",
            "_BASE_PUBLIC",
            "_V0_PUBLIC_SKILLS",
            "_V0_LOAD_MODULES",
            "_V0_RELOAD_MODULES",
            "POST_LOAD_MODULES",
            "POST_RELOAD_MODULES",
        )
        if hasattr(skills_pkg, k)
    })
    SKILL_REGISTRY.clear()
    for mod_name in RELOAD_MODULES:
        try:
            _reload_module(mod_name)
            loaded.append(mod_name)
        except ModuleNotFoundError:
            pass
    for mod_name in POST_RELOAD_MODULES:
        try:
            _reload_module(mod_name)
            loaded.append(mod_name)
        except ModuleNotFoundError:
            pass
    _prune_registry()
    return loaded


def list_skills() -> List[Dict[str, Any]]:
    """供 web/CLI 列出当前所有 skill。"""
    def public_params(spec: SkillSpec) -> List[Dict[str, Any]]:
        params = []
        for raw in spec.params:
            param = dict(raw)
            name = str(param.get("name") or "")
            if name in ("u", "v"):
                param.update({
                    "description": (
                        f"Qwen3-VL relative image coordinate {name}; "
                        f"top-left origin, range 0..{COORDINATE_MAX}"
                    ),
                    "minimum": 0,
                    "maximum": COORDINATE_MAX,
                    "coordinate_system": COORDINATE_SYSTEM,
                })
            elif name == "points":
                param.update({
                    "description": (
                        "List of Qwen3-VL relative image-coordinate [u, v] pairs; "
                        f"each axis uses range 0..{COORDINATE_MAX}"
                    ),
                    "coordinate_system": COORDINATE_SYSTEM,
                    "coordinate_range": {
                        "u": [0, COORDINATE_MAX],
                        "v": [0, COORDINATE_MAX],
                    },
                })
            params.append(param)
        return params

    return [
        {
            "name": s.name,
            "description": s.description,
            "params": public_params(s),
            "coordinate_system": COORDINATE_SYSTEM,
            "coordinate_range": {
                "u": [0, COORDINATE_MAX],
                "v": [0, COORDINATE_MAX],
            },
        }
        for s in SKILL_REGISTRY.values()
    ]
