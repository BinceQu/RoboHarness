"""v3 tool profile：在 v2 基础上新增 manipulate 工具。

保留 v2 的全部模型可见工具，并新增 manipulate 族：
manipulate_add_vector_to_point / manipulate_move_vector_to_vector。

实现层沿用 v2 的全部 skill 模块，另加 manipulate_vector 模块。
"""

from behavior_interface.tool.v2 import LOAD_MODULES as _V2_LOAD_MODULES
from behavior_interface.tool.v2 import PUBLIC_SKILLS as _V2_PUBLIC_SKILLS
from behavior_interface.tool.v2 import RELOAD_MODULES as _V2_RELOAD_MODULES


PUBLIC_SKILLS = tuple(dict.fromkeys(
    tuple(_V2_PUBLIC_SKILLS)
    + ("manipulate_add_vector_to_point", "manipulate_move_vector_to_vector")
))

LOAD_MODULES = tuple(dict.fromkeys(tuple(_V2_LOAD_MODULES) + ("manipulate_vector",)))
RELOAD_MODULES = tuple(dict.fromkeys(tuple(_V2_RELOAD_MODULES) + ("manipulate_vector",)))

# 模型可见工具：保留 v2 列表，末尾追加两个 manipulate。
V2_TOOL_NAMES = (
    "capture_head_camera",
    "capture_left_wrist_camera",
    "capture_right_wrist_camera",
    "read_depth",
    "move_chassis_to_floor_point",
    "adjust_chassis",
    "adjust_pitch",
    "adjust_height",
    "spin_to_facing_point",
    "open_gripper",
    "close_gripper",
    "adjust_left_eef_pose_in_head_frame",
    "adjust_right_eef_pose_in_head_frame",
    "adjust_left_eef_pose_in_wrist_frame",
    "adjust_right_eef_pose_in_wrist_frame",
    "plan_eef_translation_to_uvd_point",
    "adjust_plan_pose",
    "move_to_reach_point",
    "measure_shoulder_distance",
    "plan_grasp_point_filter",
    "plan_grasp_point_filter_rgbd",
    "plan_grasp_point_filter_rgbd_lite",
    "plan_press_point",
    "exec_plan_pose",
    "set_arm_to_grasp_position",
    "reset_body",
    "manipulate_add_vector_to_point",
    "manipulate_move_vector_to_vector",
)
