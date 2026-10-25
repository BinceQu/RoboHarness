"""Tool dispatch inside this workspace; no HTTP tools or MCP server."""

from importlib import import_module
from tools.util.capabilities import (
    PUBLIC_TOOLS,
    validate_submission as validate_original,
)


class Functions(dict):
    def __getitem__(self, name):
        if name not in (*PUBLIC_TOOLS, "joint_command"):
            raise KeyError(name)
        return getattr(import_module("tools." + name), name)


PUBLIC_TOOL_FUNCTIONS = Functions()


def validate_submission(name, arguments):
    if name == "joint_command":
        from tools.joint_command import validate_command

        validate_command(arguments.get("action"))
        if set(arguments) - {"action", "session_id"}:
            raise ValueError("joint_command accepts only action")
        return dict(arguments)
    return validate_original(name, arguments)


def strip_context_arguments(name, arguments):
    result = dict(arguments)
    # Runtime checks ownership and the scheduler retains capture_session_id
    # before removing transport metadata from these strict function signatures.
    if name in {
        "move_tracked_point",
        "adjust_left_eef_pose_in_head_frame",
        "adjust_right_eef_pose_in_head_frame",
        "adjust_left_eef_pose_in_wrist_frame",
        "adjust_right_eef_pose_in_wrist_frame",
    }:
        result.pop("session_id", None)
    return result
