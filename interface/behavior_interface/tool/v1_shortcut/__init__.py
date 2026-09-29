"""v1_shortcut profile.

Same compact public tool surface as v1.  Planning / capture / adjustment tools
keep the normal implementation.  Motion/execution tools are overridden after the
normal modules load by ``behavior_interface.tool.v1_shortcut.shortcut_tools``.
"""

from behavior_interface.tool.v1 import (  # noqa: F401
    PUBLIC_SKILLS,
    RELOAD_MODULES as _V1_RELOAD_MODULES,
    V2_TOOL_NAMES,
)

LOAD_MODULES = (
    "move_to",
    "face_to_point",
    "move_eef",
    "plan_move_eef",
    "adjust_plan_pose",
    "rotate_eef",
    "viz_base_path_overlay",
    "capture",
    "wrist_grasp_zone_overlay",
    "wrist_capture",
    "move_base_to_point",
    "plan_eef_v2",
    "exec_move_v2",
    "exec_eef_pose_v2",
    "move_to_object_v2",
    "move_to_point_v2",
    "move_to_point_v3",
    "mesure_shoulder_distance",
    "arm_reset",
    "reset_body",
    "diag_reset_object_diff",
    "diag_safe_back_scan",
)

POST_LOAD_MODULES = ("behavior_interface.tool.v1_shortcut.shortcut_tools",)
POST_RELOAD_MODULES = POST_LOAD_MODULES
RELOAD_MODULES = tuple(_V1_RELOAD_MODULES)
