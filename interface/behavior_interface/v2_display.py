"""v2 工具在日志/UI 中的显示名（与 embodied-agent tools.py 一致）。"""

from __future__ import annotations

from typing import Any, Dict

# plan_eef_v2 的 mode → agent 工具名
_MODE_TO_TOOL: Dict[str, str] = {
    "grasp_point": "plan_grasp_point",
    "grasp_point_filter": "plan_grasp_point_filter",
    "grasp_point_filter_rgbd": "plan_grasp_point_filter_rgbd",
    "grasp_point_filter_rgbd_lite": "plan_grasp_point_filter_rgbd_lite",
    "grasp_obj": "plan_grasp_object",
    "grasp_obj_filter": "plan_grasp_object_filter",
    "open": "plan_open",
    "close": "plan_close",
    "press": "plan_press",
    "press_point": "plan_press_point",
    "place": "plan_place",
}

_SKILL_ALIAS: Dict[str, str] = {
    "plan_eef_v2": "plan",  # 无 mode 时的兜底
    "plan_eef_rgbd_batch": "plan_grasp_point_filter_rgbd",
    "plan_eef_rgbd_lite": "plan_grasp_point_filter_rgbd_lite",
    "mark_object_v2": "mark_object",
    "move_to_object_v2": "move_to_object",
    "move_to_point_v2": "move_to_point",
    "move_to_point_v3": "move_to_point",
    "mesure_shoulder_distance": "mesure_shoulder_distance",
    "exec_move_v2": "exec_move",
    "exec_eef_pose_v2": "exec_eef_pose",
    "move_in_world_coord": "move_in_world_coord",
    "move_in_robot_coord": "move_in_robot_coord",
    "face_to_point": "face_to_point",
    "move_eef": "move_eef",
    "plan_move_eef": "plan_move_eef",
    "plan_move_eef_to_point": "plan_move_eef_to_point",
    "adjust_plan_pose": "adjust_plan_pose",
    "arm_reset": "arm_reset",
    "set_arm_to_grasp_position": "set_arm_to_grasp_position",
    "set_arm_to_grasp_position_shortcut": "set_arm_to_grasp_position_shortcut",
    "reset_body": "reset_body",
    "rotate_eef": "rotate_eef",
}


def mode_to_tool(mode: str) -> str:
    """plan_eef_v2 的 mode 字符串 → agent 工具名。"""
    m = (mode or "").strip()
    if m.startswith("push_"):
        return "plan_push"
    return _MODE_TO_TOOL.get(m, f"plan_{m}" if m else "plan")


def skill_display_name(skill_name: str, args: Dict[str, Any] | None = None) -> str:
    """仿真 skill 注册名 → 人机界面/日志里显示的工具名。"""
    args = args or {}
    if skill_name == "plan_eef_v2":
        mode = args.get("mode")
        if mode:
            return mode_to_tool(str(mode))
    if skill_name in _SKILL_ALIAS and skill_name != "plan_eef_v2":
        return _SKILL_ALIAS[skill_name]
    return skill_name
