"""v2 tool profile: v1 implementation surface with renamed public tools."""

from behavior_interface.tool.v1 import LOAD_MODULES as _V1_LOAD_MODULES
from behavior_interface.tool.v1 import RELOAD_MODULES as _V1_RELOAD_MODULES


_WRIST_CAPTURE_DEPENDENCIES = (
    "plan_grasp_gripper_geom",
    "wrist_grasp_zone_overlay",
    "wrist_metric_ruler",
    "wrist_capture",
)


def _with_wrist_capture_dependencies(modules):
    ordered = list(dict.fromkeys(modules))

    def ensure_before(module_name, dependent_name):
        if dependent_name not in ordered:
            ordered.append(dependent_name)
        if module_name not in ordered:
            ordered.insert(ordered.index(dependent_name), module_name)
            return
        module_index = ordered.index(module_name)
        dependent_index = ordered.index(dependent_name)
        if module_index > dependent_index:
            ordered.pop(module_index)
            ordered.insert(ordered.index(dependent_name), module_name)

    for module_name, dependent_name in zip(
        _WRIST_CAPTURE_DEPENDENCIES,
        _WRIST_CAPTURE_DEPENDENCIES[1:],
    ):
        ensure_before(module_name, dependent_name)
    return tuple(ordered)


def _with_reach_point_recovery_dependency(modules):
    ordered = list(dict.fromkeys(modules))
    dependency = "reach_point_pitch_recovery"
    dependent = "move_to_object_v2"
    if dependent not in ordered:
        ordered.append(dependent)
    if dependency not in ordered:
        ordered.insert(ordered.index(dependent), dependency)
    elif ordered.index(dependency) > ordered.index(dependent):
        ordered.remove(dependency)
        ordered.insert(ordered.index(dependent), dependency)
    return tuple(ordered)


# Keep the implementation skills that the v2 wrappers call.  Old v1 names that
# are only implementation details stay registered, but V2_TOOL_NAMES below is
# the only model-visible surface for the web / harness tool list.
PUBLIC_SKILLS = (
    "capture",
    "capture_left_wrist_camera",
    "capture_right_wrist_camera",
    "read_depth",
    "move_base_to_point",
    "move_eef",
    "adjust_left_eef_pose_in_head_frame",
    "adjust_right_eef_pose_in_head_frame",
    "adjust_left_eef_pose_in_wrist_frame",
    "adjust_right_eef_pose_in_wrist_frame",
    "move_point_to_point",
    "plan_move_eef",
    "move_in_robot_coord",
    "face_to_point",
    "adjust_plan_pose",
    "move_to_point_v2",
    "move_to_point_v3",
    "mesure_shoulder_distance",
    "plan_eef_v2",
    "plan_eef_rgbd_batch",
    "plan_eef_rgbd_lite",
    "exec_eef_pose_v2",
    "set_arm_to_grasp_position",
    "diag_keep_ori_wrist_preflight",
    "reset_body",
    "diag_reset_object_diff",
    "diag_safe_back_scan",
    "diag_depth_mesh_overlap",
)


LOAD_MODULES = tuple(dict.fromkeys(
    _with_reach_point_recovery_dependency(
        _with_wrist_capture_dependencies(_V1_LOAD_MODULES)
    )
    + (
        "plan_press_point",
        "adjust_eef_pose",
        "behavior_interface.web_runtime_reload",
        "read_depth",
        "adjust_eef_pose_in_wrist_frame",
        "move_point_to_point",
        "depth_mesh_reconstruction",
        "diag_depth_mesh_overlap",
        "grasp_point_filter_rgbd",
        "grasp_point_filter_rgbd_lite",
    )
))
RELOAD_MODULES = tuple(dict.fromkeys(
    _with_reach_point_recovery_dependency(
        _with_wrist_capture_dependencies(_V1_RELOAD_MODULES)
    )
    + (
        "plan_press_point",
        "adjust_eef_pose",
        "behavior_interface.web_runtime_reload",
        "read_depth",
        "adjust_eef_pose_in_wrist_frame",
        "move_point_to_point",
        "depth_mesh_reconstruction",
        "diag_depth_mesh_overlap",
        "grasp_point_filter_rgbd",
        "grasp_point_filter_rgbd_lite",
    )
))

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
    "move_point_to_point",
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
)
