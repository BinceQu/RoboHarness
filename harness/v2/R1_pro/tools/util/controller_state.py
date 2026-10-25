"""Joint target pins and policy-local odometry; no environment or scene API."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class Pose:
    pos: np.ndarray
    quat: np.ndarray

    @property
    def yaw(self):
        x, y, z, w = self.quat
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def as_dict(self):
        return {"pos": self.pos.tolist(), "quat": self.quat.tolist(), "yaw_deg": math.degrees(self.yaw)}


class ControllerState:
    def __init__(self, **unused):
        self.robot_dof = 8
        self._mock_base = np.zeros(3)
        self._arm_pin_qpos = {}
        self._trunk_pin_qpos = None
        self._gripper_pin_qpos = {}
        self._gripper_pin_effort = {}
        self._gripper_close_keepalive = set()
        self._tool_roll_pin_qpos = {"left": 0., "right": 0.}
        self._tool_roll_motion_enabled = set()

    def arm_pin_qpos_list(self, arm):
        value = self._arm_pin_qpos.get(arm)
        return None if value is None else list(value)

    def clear_arm_pin_qpos(self, arm=None):
        if arm is None:
            self._arm_pin_qpos.clear()
        else:
            self._arm_pin_qpos.pop(arm, None)

    def set_tool_roll_pin_qpos(self, arm, qpos):
        arm = str(arm).lower().strip()
        if arm not in ("left", "right") or not math.isfinite(float(qpos)):
            raise ValueError("Invalid arm or J8 target")
        # A pin is internal controller state, not an issued action. The
        # baseline reset helper first pins the measured residual, then issues
        # zero targets and revokes motion. Explicit atomic controls can also
        # establish a nonzero pin; validate_action checks its physical limits.
        self._tool_roll_pin_qpos[arm] = float(qpos)

    def tool_roll_pin_qpos(self, arm):
        if arm not in ("left", "right"):
            raise ValueError("Unknown arm")
        return self._tool_roll_pin_qpos[arm]

    def begin_tool_roll_motion(self, arm):
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError("Unknown arm")
        self._tool_roll_motion_enabled.add(arm)

    def end_tool_roll_motion(self, arm):
        self._tool_roll_motion_enabled.discard(arm)

    def has_tool_roll(self, arm=None):
        return True

    def set_trunk_pin_qpos(self, qpos=None):
        values = self.trunk_qpos() if qpos is None else np.asarray(qpos, dtype=float)
        if values.size != 4 or not np.isfinite(values).all():
            raise ValueError("Trunk target requires four finite angles")
        self._trunk_pin_qpos = values.tolist()

    def trunk_pin_qpos_list(self):
        if self._trunk_pin_qpos is None:
            self.set_trunk_pin_qpos()
        return list(self._trunk_pin_qpos)

    def robot_pose(self):
        x, y, yaw = self._mock_base
        return Pose(np.array([x, y, 0.]), np.array([0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]))
