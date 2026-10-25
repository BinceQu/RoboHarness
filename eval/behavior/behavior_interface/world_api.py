"""
统一世界 API：所有 skill 通过 ctx.world 访问。
- 一套世界系坐标（OmniGibson world frame，米 + xyzw 四元数）
- 对象命名 = 场景里的 obj.name（DatasetObject 名）+ BDDL synset alias
- 提供常用查询/操控：机器人位姿、物体位姿、设置 base 速度等
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .robot_variant import normalize_robot_dof, robot_has_tool_roll


_TOOL_ROLL_RESET_REPORT_TOL_RAD = 1e-7
_GRIPPER_EFFORT_LIMIT_N = 20.0
_GRIPPER_QPOS_SERVO_KP_N_PER_M = 160.0
_GRIPPER_QPOS_SERVO_KD_N_S_PER_M = 3.0
_GRIPPER_QPOS_SERVO_CLOSE_LIMIT_N = 3.0
_GRIPPER_QPOS_SERVO_OPEN_LIMIT_N = 8.0
_GRIPPER_PIN_OPEN_EPS_M = 0.001
_GRIPPER_ASSIST_KEEPALIVE_N = 2.0
_GRIPPER_ASSIST_MIN_KEEPALIVE_N = 0.05


def _challenge_action_only_enabled() -> bool:
    return str(
        os.environ.get("BEHAVIOR_CHALLENGE_MODE", "")
    ).lower().strip() in {"train", "public_test", "hidden_test"}


def _to_np(x) -> np.ndarray:
    """torch.Tensor / list / np.ndarray -> 1D numpy float."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


@dataclass
class Pose:
    """世界系位姿：xyz + 四元数 xyzw。"""

    pos: np.ndarray  # shape (3,)
    quat: np.ndarray  # shape (4,) xyzw

    @property
    def yaw(self) -> float:
        """从 quat 取 yaw（绕 z）。"""
        x, y, z, w = self.quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pos": self.pos.tolist(),
            "quat": self.quat.tolist(),
            "yaw_deg": math.degrees(self.yaw),
        }


class WorldAPI:
    """对仿真的薄封装。所有方法返回原生 Python / numpy 类型，避免 skill 直接接触 torch。"""

    def __init__(
        self,
        env=None,
        robot=None,
        dry_run: bool = False,
        robot_dof: Optional[int] = None,
    ):
        self.env = env
        self.robot = robot
        self.dry_run = dry_run
        self.robot_dof = normalize_robot_dof(robot_dof)
        # dry-run 用：维护一个假的底盘状态 (x, y, yaw)
        self._mock_base = np.array([0.0, 0.0, 0.0])
        # 当前 Scene Graph 缓存（由 server 在主循环里通过 .current_scene_graph = sg 写入）
        # skill 在 ctx.world.current_scene_graph 即可读到最新结构，无需自己重建
        self.current_scene_graph = None
        # 底盘 / 躯干 / capture hold 动作需要一个显式的手臂保持目标。
        # arm skill 每次发绝对关节目标时会刷新它；reset 会把它设为 grasp prep。
        self._arm_pin_qpos: Dict[str, List[float]] = {}
        self._trunk_pin_qpos: Optional[List[float]] = None
        self._gripper_pin_qpos: Dict[str, List[float]] = {}
        self._gripper_pin_effort: Dict[str, List[float]] = {}
        # Official assisted grasping is command-driven: an effort gripper must
        # receive a negative command on every policy step or OG releases the
        # constraint. This latch records the semantic close request separately
        # from finger qpos snapshots used by arm / body motion helpers.
        self._gripper_close_keepalive: set[str] = set()
        # J8 is a separate tool-roll controller, excluded from all 7DOF arm IK.
        # Every action explicitly holds the current per-arm lock target.
        self._tool_roll_pin_qpos: Dict[str, float] = {}
        if self.dry_run and self.robot_dof == 8:
            self._tool_roll_pin_qpos.update({"left": 0.0, "right": 0.0})
        elif self.robot is not None:
            try:
                names = list(self.robot.joints.keys())
                qpos = self.robot.get_joint_positions()
                for arm in ("left", "right"):
                    joint_name = f"{arm}_arm_joint8"
                    if joint_name in names:
                        self._tool_roll_pin_qpos[arm] = float(
                            qpos[names.index(joint_name)]
                        )
            except Exception:
                pass
        self._tool_roll_motion_enabled: set[str] = set()

    # ---- 机器人 ----

    def chest_pose(self) -> Dict[str, float]:
        """胸口 (torso_link4) 的世界 pose，统一格式：
            {"x","y","z","theta_x_deg","theta_z_deg","forward":[fx,fy,fz]}
        - x,y,z          胸口位置
        - theta_x_deg    胸口 forward 与世界 +X 夹角（水平 yaw）
        - theta_z_deg    胸口 forward 与世界 +Z 夹角（极角，0=朝上 / 180=朝下）
        - forward        胸口 forward 单位向量（世界系）
        """
        if self.dry_run:
            return {"x": 0.0, "y": 0.0, "z": 1.2,
                    "theta_x_deg": 0.0, "theta_z_deg": 90.0,
                    "forward": [1.0, 0.0, 0.0]}
        try:
            chest_link = self.robot.links.get("torso_link4")
            if chest_link is None:
                # 回退：base 上方 1.2m，朝 base yaw
                base = self.robot_pose()
                return {"x": float(base.pos[0]), "y": float(base.pos[1]), "z": float(base.pos[2]) + 1.2,
                        "theta_x_deg": math.degrees(base.yaw), "theta_z_deg": 90.0,
                        "forward": [math.cos(base.yaw), math.sin(base.yaw), 0.0]}
            cpos_t, cquat_t = chest_link.get_position_orientation()
            cpos = _to_np(cpos_t)
            q = _to_np(cquat_t)
            qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            fx = 1.0 - 2.0 * (qy * qy + qz * qz)
            fy = 2.0 * (qx * qy + qw * qz)
            fz = 2.0 * (qx * qz - qw * qy)
            n = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
            fx /= n; fy /= n; fz /= n
            return {
                "x": float(cpos[0]), "y": float(cpos[1]), "z": float(cpos[2]),
                "theta_x_deg": math.degrees(math.atan2(fy, fx)),
                "theta_z_deg": math.degrees(math.acos(max(-1.0, min(1.0, fz)))),
                "forward": [fx, fy, fz],
            }
        except Exception:
            base = self.robot_pose()
            return {"x": float(base.pos[0]), "y": float(base.pos[1]), "z": float(base.pos[2]) + 1.2,
                    "theta_x_deg": math.degrees(base.yaw), "theta_z_deg": 90.0,
                    "forward": [math.cos(base.yaw), math.sin(base.yaw), 0.0]}

    def trunk_qpos(self) -> np.ndarray:
        """读 trunk 4 个 joint 当前关节角度（弧度），shape=(4,)."""
        if self.dry_run:
            return np.zeros(4)
        try:
            qpos = self.robot.get_joint_positions()
            idx = self.robot.trunk_control_idx
            return _to_np(qpos[idx]).astype(np.float64)
        except Exception:
            return np.zeros(4)

    def base_qvel(self) -> np.ndarray:
        """Read base velocity as robot-frame ``(forward, left, yaw)``.

        OmniGibson's holonomic controller stores the virtual x/y joint
        velocities in the articulation's canonical frame.  Controller inputs,
        however, are expressed in the moving robot frame.  Returning the raw
        virtual-joint values makes a lateral command look like forward motion
        whenever the base is rotated, and makes the requested axis appear
        stalled.  Rotate by the virtual yaw joint so every caller sees the
        same robot-frame convention as ``set_base_velocity``.
        """
        if self.dry_run:
            return np.zeros(3, dtype=np.float64)
        try:
            qvel = self.robot.get_joint_velocities()
            idx = self.robot.base_control_idx
            canonical = _to_np(qvel[idx]).astype(np.float64).reshape(3)
        except Exception:
            return np.zeros(3, dtype=np.float64)
        try:
            qpos = self.robot.get_joint_positions()
            base_yaw = float(
                _to_np(qpos[idx]).astype(np.float64).reshape(3)[2]
            )
        except Exception:
            return canonical
        cos_y, sin_y = math.cos(base_yaw), math.sin(base_yaw)
        return np.array(
            [
                cos_y * canonical[0] + sin_y * canonical[1],
                -sin_y * canonical[0] + cos_y * canonical[1],
                canonical[2],
            ],
            dtype=np.float64,
        )

    def eef_pose(self, arm: str = "right") -> Dict[str, Any]:
        """末端执行器（eef link）的世界 pose。返回 {"pos":[x,y,z], "quat":[x,y,z,w]}。"""
        if self.dry_run:
            return {"pos": [0.5, 0.0 if arm == "right" else 0.0, 1.0],
                    "quat": [0.0, 0.0, 0.0, 1.0]}
        try:
            pos_t = self.robot.get_eef_position(arm=arm)
            quat_t = self.robot.get_eef_orientation(arm=arm)
            return {"pos": _to_np(pos_t).tolist(), "quat": _to_np(quat_t).tolist()}
        except Exception:
            return {"pos": [0.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]}

    def shoulder_pose(self, arm: str = "right") -> Dict[str, float]:
        """肩部 link (arm_link1) 的世界位置 —— reachable 启发式用。
        距离肩部 < 0.75m 通常可达。
        """
        if self.dry_run:
            return {"x": 0.0, "y": -0.2 if arm == "right" else 0.2, "z": 1.1}
        try:
            link_name = f"{arm}_arm_link1"
            link = self.robot.links.get(link_name)
            if link is None:
                # 退化：用 chest + 局部偏移
                ch = self.chest_pose()
                side = -0.2 if arm == "right" else 0.2
                return {"x": ch["x"], "y": ch["y"] + side, "z": ch["z"] - 0.05}
            pos_t, _ = link.get_position_orientation()
            p = _to_np(pos_t)
            return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
        except Exception:
            ch = self.chest_pose()
            side = -0.2 if arm == "right" else 0.2
            return {"x": ch["x"], "y": ch["y"] + side, "z": ch["z"] - 0.05}

    def arm_qpos_list(self, arm: str) -> List[float]:
        """单臂 7 关节当前角（弧度）。"""
        if self.dry_run:
            return [0.0] * 7
        names = list(self.robot.joints.keys())
        idx = [names.index(f"{arm}_arm_joint{i+1}") for i in range(7)]
        qpos = self.robot.get_joint_positions()
        return [float(qpos[i]) for i in idx]

    def set_arm_pin_qpos(self, arm: str, qpos) -> None:
        """设置底盘/躯干/hold 时该手臂应保持的绝对 7 关节目标。"""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if arr.size < 7:
            raise ValueError(f"arm pin qpos for {arm} expects 7 values, got {arr.size}")
        self._arm_pin_qpos[arm] = [float(x) for x in arr[:7]]

    def arm_pin_qpos_list(self, arm: str) -> Optional[List[float]]:
        """读取该手臂的保持目标；未设置时返回 None。"""
        pin = self._arm_pin_qpos.get(str(arm).lower().strip())
        return list(pin) if pin is not None else None

    def clear_arm_pin_qpos(self, arm: Optional[str] = None) -> None:
        """清除手臂保持目标；调试/特殊 skill 可用。"""
        if arm is None:
            self._arm_pin_qpos.clear()
            return
        self._arm_pin_qpos.pop(str(arm).lower().strip(), None)

    def tool_roll_qpos(self, arm: str) -> float:
        """Read the independent J8 tool-roll position in radians."""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        if self.dry_run:
            if not self.has_tool_roll(arm):
                raise RuntimeError("independent J8 tool roll is not active on the 7DOF robot")
            return float(self._tool_roll_pin_qpos.get(arm, 0.0))
        if not self.has_tool_roll(arm):
            raise RuntimeError("independent J8 tool roll is not active on the 7DOF robot")
        names = list(self.robot.joints.keys())
        idx = names.index(f"{arm}_arm_joint8")
        return float(self.robot.get_joint_positions()[idx])

    def tool_roll_joint_limits(self, arm: str) -> Tuple[float, float]:
        """Return the URDF lower/upper limits for the independent J8 joint."""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        if self.dry_run:
            if not self.has_tool_roll(arm):
                raise RuntimeError("independent J8 tool roll is not active on the 7DOF robot")
            return -math.pi, math.pi
        if not self.has_tool_roll(arm):
            raise RuntimeError("independent J8 tool roll is not active on the 7DOF robot")
        joint = self.robot.joints[f"{arm}_arm_joint8"]
        return float(joint.lower_limit), float(joint.upper_limit)

    def set_tool_roll_pin_qpos(self, arm: str, qpos: float) -> None:
        """Set the absolute J8 target that every subsequent action must hold."""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        value = float(qpos)
        if not math.isfinite(value):
            raise ValueError(f"tool roll pin qpos for {arm} must be finite")
        self._tool_roll_pin_qpos[arm] = value

    def tool_roll_pin_qpos(self, arm: str) -> float:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        return float(self._tool_roll_pin_qpos.get(arm, 0.0))

    def begin_tool_roll_motion(self, arm: str) -> None:
        """Temporarily authorize J8 motion for the wrist-frame roll skill."""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        self._tool_roll_motion_enabled.add(arm)

    def end_tool_roll_motion(self, arm: str) -> None:
        """Revoke J8 motion authorization; the current pin remains active."""
        self._tool_roll_motion_enabled.discard(str(arm).lower().strip())

    def force_set_tool_roll_qpos(self, arm: str, qpos: float) -> None:
        """Diagnostic-only J8 injection; forbidden during challenge execution."""
        arm = str(arm).lower().strip()
        target = float(qpos)
        if not self.has_tool_roll(arm):
            raise RuntimeError("independent J8 tool roll is not active on the 7DOF robot")
        if self.dry_run:
            self.set_tool_roll_pin_qpos(arm, target)
            return
        if _challenge_action_only_enabled():
            raise RuntimeError(
                "direct J8 qpos injection is disabled in challenge mode; "
                "use tool_roll controller actions"
            )
        names = list(self.robot.joints.keys())
        idx = names.index(f"{arm}_arm_joint8")
        q0 = self.robot.get_joint_positions()
        q = q0.clone() if hasattr(q0, "clone") else np.asarray(q0, dtype=np.float64).copy()
        q[int(idx)] = target
        self.robot.set_joint_positions(q)
        try:
            v0 = self.robot.get_joint_velocities()
            v = v0.clone() if hasattr(v0, "clone") else np.asarray(v0, dtype=np.float64).copy()
            v[int(idx)] = 0.0
            self.robot.set_joint_velocities(v)
        except Exception:
            pass
        self.set_tool_roll_pin_qpos(arm, target)

    def hard_lock_tool_roll_pins(self) -> Dict[str, Any]:
        """Report J8 tracking without writing joint state outside PhysX."""
        report: Dict[str, Any] = {"locked": [], "authorized": [], "action_only": True}
        if self.dry_run or self.robot is None:
            return report

        names = list(self.robot.joints.keys())
        q0 = self.robot.get_joint_positions()
        for arm in ("left", "right"):
            if not self.has_tool_roll(arm):
                continue
            if arm in self._tool_roll_motion_enabled:
                report["authorized"].append(arm)
                continue
            idx = names.index(f"{arm}_arm_joint8")
            target = float(self.tool_roll_pin_qpos(arm))
            report["locked"].append({
                "arm": arm,
                "target_rad": target,
                "actual_rad": float(q0[int(idx)]),
                "error_rad": float(q0[int(idx)]) - target,
            })
        return report

    def reset_tool_roll_to_zero(self, arm: str) -> Dict[str, Any]:
        """Legacy synchronous reset; challenge execution must use action steps."""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        if not self.has_tool_roll(arm):
            return {
                "arm": arm,
                "required": False,
                "reset": False,
                "reason": "7dof robot has no independent J8",
            }
        before = float(self.tool_roll_qpos(arm))
        pin_before = float(self.tool_roll_pin_qpos(arm))
        reset = (
            abs(before) > _TOOL_ROLL_RESET_REPORT_TOL_RAD
            or abs(pin_before) > _TOOL_ROLL_RESET_REPORT_TOL_RAD
        )
        if _challenge_action_only_enabled():
            if reset:
                raise RuntimeError(
                    "synchronous J8 reset is disabled in challenge mode; "
                    "use the action-only skill reset"
                )
            self.end_tool_roll_motion(arm)
            self.set_tool_roll_pin_qpos(arm, 0.0)
            return {
                "arm": arm,
                "reset": False,
                "before_rad": before,
                "pin_before_rad": pin_before,
                "after_rad": before,
                "pin_after_rad": 0.0,
                "locked": True,
                "action_only": True,
            }

        self.end_tool_roll_motion(arm)
        self.force_set_tool_roll_qpos(arm, 0.0)

        after = float(self.tool_roll_qpos(arm))
        pin_after = float(self.tool_roll_pin_qpos(arm))
        if abs(after) > _TOOL_ROLL_RESET_REPORT_TOL_RAD or abs(pin_after) > _TOOL_ROLL_RESET_REPORT_TOL_RAD:
            raise RuntimeError(
                f"{arm} J8 reset failed: q={after:.8f}rad pin={pin_after:.8f}rad"
            )
        return {
            "arm": arm,
            "reset": bool(reset),
            "before_rad": before,
            "pin_before_rad": pin_before,
            "after_rad": after,
            "pin_after_rad": pin_after,
            "locked": True,
        }

    def has_tool_roll(self, arm: Optional[str] = None) -> bool:
        if self.dry_run and self.robot is None:
            return self.robot_dof == 8
        return robot_has_tool_roll(self.robot, arm)

    def set_trunk_pin_qpos(self, qpos=None) -> None:
        """设置后续 pinned action 要保持的 trunk 4 关节目标；None 表示取当前值。"""
        arr = self.trunk_qpos() if qpos is None else np.asarray(qpos, dtype=np.float64)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size < 4:
            raise ValueError(f"trunk pin qpos expects 4 values, got {arr.size}")
        self._trunk_pin_qpos = [float(x) for x in arr[:4]]

    def trunk_pin_qpos_list(self) -> List[float]:
        """读取 trunk 保持目标；未设置时初始化为当前 trunk。"""
        if self._trunk_pin_qpos is None:
            self.set_trunk_pin_qpos()
        return list(self._trunk_pin_qpos or self.trunk_qpos().tolist())

    def gripper_qpos_list(self, arm: str) -> Optional[List[float]]:
        """读取夹爪当前 finger 关节角；失败时返回 None。"""
        if self.dry_run:
            return [0.0]
        arm = str(arm).lower().strip()
        try:
            names = list(self.robot.joints.keys())
            qpos = self.robot.get_joint_positions()
            joint_names = []
            try:
                joint_names.extend(list(getattr(self.robot, "finger_joint_names", {}).get(arm, [])))
            except Exception:
                pass
            joint_names.extend([
                f"{arm}_gripper_finger_joint1",
                f"{arm}_gripper_finger_joint2",
            ])
            seen = set()
            out: List[float] = []
            for jn in joint_names:
                jn = str(jn)
                if jn in seen or jn not in names:
                    continue
                seen.add(jn)
                out.append(float(qpos[names.index(jn)]))
            if out:
                return out
            # Last-resort fallback for non-R1 robots whose gripper action indices
            # happen to match qpos indices.
            idx = self.controller_action_idx(f"gripper_{arm}")
            return [float(qpos[int(i)]) for i in idx]
        except Exception:
            return None

    def gripper_qvel_list(self, arm: str) -> Optional[List[float]]:
        """Read the finger velocities in joint-coordinate order."""
        if self.dry_run:
            return [0.0, 0.0]
        arm = str(arm).lower().strip()
        try:
            names = list(self.robot.joints.keys())
            qvel = self.robot.get_joint_velocities()
            joint_names = []
            try:
                joint_names.extend(
                    list(getattr(self.robot, "finger_joint_names", {}).get(arm, []))
                )
            except Exception:
                pass
            joint_names.extend([
                f"{arm}_gripper_finger_joint1",
                f"{arm}_gripper_finger_joint2",
            ])
            seen = set()
            out: List[float] = []
            for joint_name in joint_names:
                joint_name = str(joint_name)
                if joint_name in seen or joint_name not in names:
                    continue
                seen.add(joint_name)
                out.append(float(qvel[names.index(joint_name)]))
            return out or None
        except Exception:
            return None

    def gripper_motor_type(self, arm: str) -> str:
        """Return the configured gripper motor type without importing OG enums."""
        if self.dry_run or self.robot is None:
            return "position"
        controller = self.robot.controllers.get(f"gripper_{str(arm).lower().strip()}")
        return str(
            getattr(controller, "motor_type", getattr(controller, "_motor_type", "position"))
        ).lower()

    def gripper_uses_effort(self, arm: str) -> bool:
        return self.gripper_motor_type(arm) == "effort"

    def _gripper_close_keepalive_arms(self) -> set[str]:
        """Return the close latch set, creating it for hot-reloaded worlds."""
        keepalive = getattr(self, "_gripper_close_keepalive", None)
        if not isinstance(keepalive, set):
            keepalive = set(keepalive or ())
            self._gripper_close_keepalive = keepalive
        return keepalive

    def gripper_close_keepalive_active(self, arm: str) -> bool:
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        return arm in self._gripper_close_keepalive_arms()

    def latch_gripper_close_keepalive(self, arm: str, effort=None) -> None:
        """Latch close intent until an explicit open or world reset.

        This does not create or modify an assisted-grasp constraint. It only
        persists the controller command that OmniGibson uses both while seeking
        contact and to retain an established official constraint.
        """
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")

        if self.gripper_uses_effort(arm):
            dim = len(self.controller_action_idx(f"gripper_{arm}"))
            source = effort
            if source is None:
                source = self._gripper_pin_effort.get(arm)
            if source is None:
                arr = np.full(dim, -_GRIPPER_ASSIST_KEEPALIVE_N, dtype=np.float64)
            else:
                arr = np.asarray(source, dtype=np.float64).reshape(-1)
                if arr.size == 1 and dim > 1:
                    arr = np.repeat(arr, dim)
                if arr.size != dim or not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"gripper close keepalive for {arm} expects {dim} finite values, "
                        f"got {arr.size}"
                    )
                magnitude = np.where(
                    arr < -1e-6,
                    np.abs(arr),
                    _GRIPPER_ASSIST_MIN_KEEPALIVE_N,
                )
                arr = -np.clip(
                    magnitude,
                    _GRIPPER_ASSIST_MIN_KEEPALIVE_N,
                    _GRIPPER_EFFORT_LIMIT_N,
                )
            self._gripper_pin_effort[arm] = [float(x) for x in arr.tolist()]
            self._gripper_pin_qpos.pop(arm, None)
        elif arm not in self._gripper_pin_qpos:
            current = self.gripper_qpos_list(arm)
            if current:
                self._gripper_pin_qpos[arm] = [float(x) for x in current]

        self._gripper_close_keepalive_arms().add(arm)

    def release_gripper_close_keepalive(self, arm: str) -> None:
        """Authorize an explicit open and remove the persistent close command."""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        self._gripper_close_keepalive_arms().discard(arm)
        self._gripper_pin_effort.pop(arm, None)
        self._gripper_pin_qpos.pop(arm, None)

    def clear_gripper_close_keepalive(self, arm: Optional[str] = None) -> None:
        """Clear close latches for an authoritative scene / session reset."""
        if arm is not None:
            self.release_gripper_close_keepalive(arm)
            return
        for side in ("left", "right"):
            self.release_gripper_close_keepalive(side)

    def gripper_keepalive_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            arm: {
                "active": self.gripper_close_keepalive_active(arm),
                "motor_type": self.gripper_motor_type(arm),
                "effort": self.gripper_pin_effort_list(arm),
                "qpos": self.gripper_pin_qpos_list(arm),
            }
            for arm in ("left", "right")
        }

    def set_gripper_pin_qpos(self, arm: str, qpos) -> None:
        """设置后续 pinned action 要保持的 gripper finger 目标。"""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if arr.size <= 0:
            raise ValueError(f"gripper pin qpos for {arm} is empty")

        # A finger position observed while carrying is not an open command.
        # OG freezes the physical effort at zero once assisted grasping is
        # active, so these qpos values can drift by millimetres during a body
        # trajectory. Only release_gripper_close_keepalive() may authorize the
        # transition from a latched close to qpos control.
        if self.gripper_close_keepalive_active(arm):
            return

        # A live assisted grasp requires a negative gripper control every
        # physics step. Legacy motion code frequently snapshots the current
        # finger qpos; that is a hold request, not an explicit release.
        carry = self._gripper_pin_effort.get(arm)
        if carry is not None and any(float(x) < -1e-6 for x in carry):
            try:
                dim = len(self.controller_action_idx(f"gripper_{arm}"))
                target = self._gripper_qpos_target(arm, arr, dim)
                current_raw = self.gripper_qpos_list(arm)
                current = (
                    None
                    if current_raw is None
                    else np.asarray(current_raw, dtype=np.float64).reshape(-1)
                )
                explicit_open = (
                    current is not None
                    and current.size == target.size
                    and bool(np.any(target > current + _GRIPPER_PIN_OPEN_EPS_M))
                )
                if current is None:
                    explicit_open = bool(arr.size == 1 and float(arr[0]) > 0.0)
                if not explicit_open:
                    return
            except Exception:
                # If the target cannot be classified safely, retain the live
                # constraint; callers can always use clear/open explicitly.
                return

        self._gripper_pin_qpos[arm] = [float(x) for x in arr.tolist()]
        self._gripper_pin_effort.pop(arm, None)

    def gripper_pin_qpos_list(self, arm: str) -> Optional[List[float]]:
        pin = self._gripper_pin_qpos.get(str(arm).lower().strip())
        return list(pin) if pin is not None else None

    def set_gripper_pin_effort(self, arm: str, effort) -> None:
        """Persist a direct per-finger effort command, in newtons for R1Pro."""
        arm = str(arm).lower().strip()
        if arm not in ("left", "right"):
            raise ValueError(f"bad arm '{arm}'")
        arr = np.asarray(effort, dtype=np.float64).reshape(-1)
        if arr.size <= 0 or not np.all(np.isfinite(arr)):
            raise ValueError(f"gripper pin effort for {arm} must be finite and non-empty")
        dim = len(self.controller_action_idx(f"gripper_{arm}"))
        if arr.size == 1 and dim > 1:
            arr = np.repeat(arr, dim)
        if arr.size != dim:
            raise ValueError(f"gripper pin effort for {arm} expects {dim} values, got {arr.size}")
        arr = np.clip(arr, -_GRIPPER_EFFORT_LIMIT_N, _GRIPPER_EFFORT_LIMIT_N)
        # Once a verified assisted grasp is latched, no incidental trajectory
        # update may change either finger command. Explicit open first calls
        # release_gripper_close_keepalive(), which makes the new command legal.
        if self.gripper_close_keepalive_active(arm):
            return
        self._gripper_pin_effort[arm] = [float(x) for x in arr.tolist()]
        self._gripper_pin_qpos.pop(arm, None)

    def gripper_pin_effort_list(self, arm: str) -> Optional[List[float]]:
        pin = self._gripper_pin_effort.get(str(arm).lower().strip())
        return list(pin) if pin is not None else None

    def clear_gripper_pin_effort(self, arm: Optional[str] = None) -> None:
        if arm is None:
            active = self._gripper_close_keepalive_arms()
            self._gripper_pin_effort = {
                side: pin
                for side, pin in self._gripper_pin_effort.items()
                if side in active
            }
            return
        arm = str(arm).lower().strip()
        if self.gripper_close_keepalive_active(arm):
            return
        self._gripper_pin_effort.pop(arm, None)

    def clear_gripper_pin_qpos(self, arm: Optional[str] = None) -> None:
        if arm is None:
            active = self._gripper_close_keepalive_arms()
            self._gripper_pin_qpos = {
                side: pin
                for side, pin in self._gripper_pin_qpos.items()
                if side in active
            }
            self._gripper_pin_effort = {
                side: pin
                for side, pin in self._gripper_pin_effort.items()
                if side in active
            }
            return
        arm = str(arm).lower().strip()
        if self.gripper_close_keepalive_active(arm):
            return
        self._gripper_pin_qpos.pop(arm, None)
        self._gripper_pin_effort.pop(arm, None)

    def enforce_gripper_close_keepalive(self, action):
        """Overlay close commands at the final outgoing action boundary.

        Negative pins are also honored while a close is still being attempted,
        before the verified latch is armed. This remains action-only and never
        creates, moves, or repairs an object constraint directly.
        """
        if action is None:
            return None
        is_tensor = hasattr(action, "clone") and hasattr(action, "new_tensor")
        out = action.clone() if is_tensor else np.array(action, copy=True)
        for arm in ("left", "right"):
            pin = self.gripper_pin_effort_list(arm)
            active = self.gripper_close_keepalive_active(arm)
            if active and self.gripper_uses_effort(arm):
                if pin is None or not any(float(x) < -1e-6 for x in pin):
                    dim = len(self.controller_action_idx(f"gripper_{arm}"))
                    pin = [-_GRIPPER_ASSIST_KEEPALIVE_N] * dim
                    self._gripper_pin_effort[arm] = list(pin)
            if pin is None or not any(float(x) < -1e-6 for x in pin):
                continue
            idx = self.controller_action_idx(f"gripper_{arm}").tolist()
            if len(idx) != len(pin):
                raise ValueError(
                    f"gripper effort pin for {arm} has {len(pin)} values, expected {len(idx)}"
                )
            if is_tensor:
                out[idx] = out.new_tensor(pin)
            else:
                out[np.asarray(idx, dtype=int)] = np.asarray(pin, dtype=out.dtype)
        return out

    def enforce_gripper_effort_pins(self, action):
        """Compatibility alias for pre-v13 server loops."""
        return self.enforce_gripper_close_keepalive(action)

    def _gripper_qpos_target(self, arm: str, command, dim: int) -> np.ndarray:
        """Normalize legacy open/close or qpos commands into finger q targets."""
        arr = np.asarray(command, dtype=np.float64).reshape(-1)
        joints = list(getattr(self.robot, "finger_joints", {}).get(arm, []))
        lower = np.asarray(
            [float(joint.lower_limit) for joint in joints[:dim]], dtype=np.float64
        )
        upper = np.asarray(
            [float(joint.upper_limit) for joint in joints[:dim]], dtype=np.float64
        )
        if lower.size != dim or upper.size != dim:
            lower = np.zeros(dim, dtype=np.float64)
            upper = np.full(dim, 0.05, dtype=np.float64)
        if arr.size == 1:
            return upper.copy() if float(arr[0]) > 0.0 else lower.copy()
        if arr.size > dim:
            arr = arr[:dim]
        elif arr.size < dim:
            arr = np.pad(arr, (0, dim - arr.size), mode="edge")
        return np.clip(arr, lower, upper)

    def _gripper_qpos_servo_effort(self, arm: str, target_qpos) -> np.ndarray:
        """Convert a legacy qpos target to a deliberately low-force effort command."""
        target = np.asarray(target_qpos, dtype=np.float64).reshape(-1)
        qpos_raw = self.gripper_qpos_list(arm)
        if qpos_raw is None:
            raise RuntimeError(f"cannot read {arm} gripper qpos")
        qpos = np.asarray(qpos_raw, dtype=np.float64).reshape(-1)
        qvel_raw = self.gripper_qvel_list(arm)
        qvel = np.zeros_like(qpos) if qvel_raw is None else np.asarray(qvel_raw, dtype=np.float64).reshape(-1)
        if qpos.size != target.size or qvel.size != target.size:
            raise RuntimeError(
                f"cannot servo {arm} gripper: qpos={qpos.size} qvel={qvel.size} target={target.size}"
            )
        effort = (
            _GRIPPER_QPOS_SERVO_KP_N_PER_M * (target - qpos)
            - _GRIPPER_QPOS_SERVO_KD_N_S_PER_M * qvel
        )
        return np.clip(
            effort,
            -_GRIPPER_QPOS_SERVO_CLOSE_LIMIT_N,
            _GRIPPER_QPOS_SERVO_OPEN_LIMIT_N,
        )

    def _legacy_gripper_command_keeps_effort(
        self,
        arm: str,
        command,
        target_qpos,
    ) -> Optional[np.ndarray]:
        """Keep a negative carry pin when legacy code only asks to stay closed."""
        pin = self.gripper_pin_effort_list(arm)
        if self.gripper_close_keepalive_active(arm) and pin is None:
            dim = len(self.controller_action_idx(f"gripper_{arm}"))
            pin = [-_GRIPPER_ASSIST_KEEPALIVE_N] * dim
            self._gripper_pin_effort[arm] = list(pin)
        if pin is None:
            return None
        pin_np = np.asarray(pin, dtype=np.float64).reshape(-1)
        if pin_np.size == 0 or not np.any(pin_np < -1e-6):
            return None
        if self.gripper_close_keepalive_active(arm):
            return pin_np

        raw = np.asarray(command, dtype=np.float64).reshape(-1)
        if raw.size == 1:
            return pin_np if float(raw[0]) <= 0.0 else None

        target = np.asarray(target_qpos, dtype=np.float64).reshape(-1)
        current_raw = self.gripper_qpos_list(arm)
        if current_raw is None:
            return None
        current = np.asarray(current_raw, dtype=np.float64).reshape(-1)
        if current.size != target.size:
            return None

        joints = list(getattr(self.robot, "finger_joints", {}).get(arm, []))
        upper = np.asarray(
            [float(joint.upper_limit) for joint in joints[: target.size]],
            dtype=np.float64,
        )
        if upper.size != target.size:
            upper = np.full(target.size, 0.05, dtype=np.float64)
        if np.all(target >= upper - 1e-6):
            return None
        if np.any(target > current + _GRIPPER_PIN_OPEN_EPS_M):
            return None
        return pin_np

    def limb_pin_kwargs(self) -> Dict[str, Any]:
        """锁定用：trunk、双臂、双夹爪，以及独立 J8 tool roll。"""
        out: Dict[str, Any] = {"trunk": self.trunk_pin_qpos_list()}
        for arm in ("left", "right"):
            try:
                self.controller_action_idx(f"arm_{arm}")
                out[f"arm_{arm}"] = self.arm_pin_qpos_list(arm) or self.arm_qpos_list(arm)
                effort = self.gripper_pin_effort_list(arm) if self.gripper_uses_effort(arm) else None
                if effort is not None:
                    out[f"gripper_effort_{arm}"] = effort
                else:
                    grip = self.gripper_pin_qpos_list(arm) or self.gripper_qpos_list(arm)
                    if grip is not None:
                        out[f"gripper_{arm}"] = grip
            except Exception:
                pass
            try:
                self.controller_action_idx(f"tool_roll_{arm}")
                out[f"tool_roll_{arm}"] = [self.tool_roll_pin_qpos(arm)]
            except Exception:
                pass
        return out

    def pinned_action(self, **overrides: np.ndarray) -> np.ndarray:
        """构造 action：默认锁住 base/trunk/双臂/双夹爪，再覆盖调用方真正要动的 slice。"""
        if self.dry_run:
            return self.make_action_unpinned(**overrides)
        kw = self.limb_pin_kwargs()
        kw["base"] = [0.0, 0.0, 0.0]
        for arm in ("left", "right"):
            if f"gripper_{arm}" in overrides:
                kw.pop(f"gripper_effort_{arm}", None)
            if f"gripper_effort_{arm}" in overrides:
                kw.pop(f"gripper_{arm}", None)
        kw.update(overrides)
        return self.make_action_unpinned(**kw)

    # ── action 索引 / 命令构造 ──

    def controller_action_idx(self, controller_name: str) -> np.ndarray:
        """读 robot.controller_action_idx[controller_name]，返回 int array."""
        if self.dry_run:
            layout = {
                "base": [0, 1, 2],
                "trunk": [3, 4, 5, 6],
                "arm_left": [7, 8, 9, 10, 11, 12, 13],
                "gripper_left": [14, 15],
                "arm_right": [16, 17, 18, 19, 20, 21, 22],
                "gripper_right": [23, 24],
            }
            if self.robot_dof == 8:
                layout.update({
                    "tool_roll_left": [25],
                    "tool_roll_right": [26],
                })
            return np.array(layout.get(controller_name, []), dtype=int)
        idx = self.robot.controller_action_idx[controller_name]
        return _to_np(idx).astype(int)

    def make_action_unpinned(self, **overrides: np.ndarray) -> np.ndarray:
        """构造一个 controller noop action，并把 overrides 里指定的 slice 替换。
        例: make_action(base=[vx,vy,wz], trunk=[dq1,dq2,dq3,dq4], arm_right=[dx,dy,dz,drx,dry,drz])
        """
        if self.dry_run:
            return self._mock_base_step(
                float(overrides.get("base", [0, 0, 0])[0] if "base" in overrides else 0),
                float(overrides.get("base", [0, 0, 0])[1] if "base" in overrides else 0),
                float(overrides.get("base", [0, 0, 0])[2] if "base" in overrides else 0),
            )
        overrides = dict(overrides)
        direct_gripper_effort: Dict[str, np.ndarray] = {}
        for arm in ("left", "right"):
            alias = f"gripper_effort_{arm}"
            if alias not in overrides:
                continue
            controller_name = f"gripper_{arm}"
            direct_gripper_effort[controller_name] = np.asarray(
                overrides.pop(alias), dtype=np.float32
            ).reshape(-1)
            overrides[controller_name] = direct_gripper_effort[controller_name]
        # Even "unpinned" legacy paths must never leave J8 without an absolute
        # high-force hold target.
        for arm in ("left", "right"):
            ctrl = f"tool_roll_{arm}"
            if ctrl not in overrides:
                try:
                    self.controller_action_idx(ctrl)
                    overrides[ctrl] = [self.tool_roll_pin_qpos(arm)]
                except Exception:
                    pass
        action = self._compute_noop_action_np()
        gripper_qpos_targets: Dict[str, np.ndarray] = {}
        for ctrl, vec in overrides.items():
            idx = self.controller_action_idx(ctrl)
            vec_np = np.asarray(vec, dtype=np.float32).reshape(-1)
            if ctrl.startswith("tool_roll_") and len(vec_np) == 1:
                arm = ctrl.split("tool_roll_", 1)[1]
                requested = float(vec_np[0])
                locked = self.tool_roll_pin_qpos(arm)
                if (
                    arm not in self._tool_roll_motion_enabled
                    and abs(requested - locked) > 1e-7
                ):
                    raise PermissionError(
                        f"{ctrl} is locked; only adjust_eef_pose_in_wrist_frame roll may move J8"
                    )
            if ctrl.startswith("gripper_") and self.gripper_uses_effort(ctrl.split("_", 1)[1]):
                arm = ctrl.split("_", 1)[1]
                if ctrl in direct_gripper_effort:
                    if vec_np.size == 1 and len(idx) > 1:
                        vec_np = np.repeat(vec_np, len(idx))
                    vec_np = np.clip(
                        vec_np,
                        -_GRIPPER_EFFORT_LIMIT_N,
                        _GRIPPER_EFFORT_LIMIT_N,
                    ).astype(np.float32)
                else:
                    target_qpos = self._gripper_qpos_target(arm, vec_np, len(idx))
                    carry_effort = self._legacy_gripper_command_keeps_effort(
                        arm,
                        vec_np,
                        target_qpos,
                    )
                    if carry_effort is not None:
                        direct_gripper_effort[ctrl] = carry_effort.astype(np.float32)
                        vec_np = direct_gripper_effort[ctrl]
                    else:
                        gripper_qpos_targets[arm] = target_qpos
                        vec_np = self._gripper_qpos_servo_effort(arm, target_qpos).astype(np.float32)
            # gripper 控制器在 JointController + absolute position 模式下，
            # 每个 finger 是独立 joint。允许调用方传 1D 命令（语义 open/close），
            # 这里自动 broadcast 到 N 个 finger joints。
            # 取当前 finger 的 lower/upper limit，根据 cmd 正负设极值。
            elif ctrl.startswith("gripper_") and len(vec_np) == 1 and len(idx) > 1:
                arm = ctrl.split("_", 1)[1]
                cmd_val = float(vec_np[0])
                expanded = np.zeros(len(idx), dtype=np.float32)
                try:
                    finger_joints = self.robot.finger_joints.get(arm, [])
                    for i, fj in enumerate(finger_joints[:len(idx)]):
                        lo = float(fj.lower_limit)
                        up = float(fj.upper_limit)
                        # cmd > 0 → open（取较大幅值）；cmd < 0 → close（取较小幅值）
                        expanded[i] = up if cmd_val > 0 else lo
                except Exception:
                    expanded[:] = cmd_val
                vec_np = expanded
            elif ctrl.startswith("gripper_") and len(idx) != len(vec_np):
                if len(idx) == 1 and len(vec_np) > 1:
                    vec_np = np.asarray([float(np.mean(vec_np))], dtype=np.float32)
                elif len(vec_np) > len(idx):
                    vec_np = vec_np[:len(idx)]
                else:
                    expanded = np.asarray(action[idx], dtype=np.float32).reshape(-1)
                    expanded[: len(vec_np)] = vec_np
                    vec_np = expanded
            if ctrl in ("arm_left", "arm_right") and len(idx) != len(vec_np):
                if len(vec_np) > len(idx):
                    vec_np = vec_np[:len(idx)]
                else:
                    expanded = np.asarray(action[idx], dtype=np.float32).reshape(-1)
                    expanded[: len(vec_np)] = vec_np
                    vec_np = expanded
            if len(idx) != len(vec_np):
                raise ValueError(f"action 长度不符：{ctrl} 期望 {len(idx)}, 给了 {len(vec_np)}")
            action[idx] = vec_np
            if ctrl in ("arm_left", "arm_right") and len(vec_np) >= 7:
                # R1Pro arm controller 在本服务里配置为 absolute position。
                # 记录最后一次显式 arm 目标，后续底盘/躯干动作才能继续锁住手臂。
                self._arm_pin_qpos[ctrl.split("_", 1)[1]] = [
                    float(x) for x in vec_np[:7].astype(np.float64).tolist()
                ]
            elif ctrl == "trunk" and len(vec_np) >= 4:
                self._trunk_pin_qpos = [
                    float(x) for x in vec_np[:4].astype(np.float64).tolist()
                ]
            elif ctrl.startswith("gripper_") and len(vec_np) > 0:
                arm = ctrl.split("_", 1)[1]
                if ctrl in direct_gripper_effort:
                    self.set_gripper_pin_effort(arm, vec_np)
                elif arm in gripper_qpos_targets:
                    self.set_gripper_pin_qpos(arm, gripper_qpos_targets[arm])
                else:
                    self.set_gripper_pin_qpos(arm, vec_np)
            elif ctrl.startswith("tool_roll_") and len(vec_np) == 1:
                self.set_tool_roll_pin_qpos(
                    ctrl.split("tool_roll_", 1)[1],
                    float(vec_np[0]),
                )
        return self.enforce_gripper_close_keepalive(action)

    def make_action(self, **overrides: np.ndarray) -> np.ndarray:
        """默认安全 action：没有被 overrides 指定的 base/trunk/arm/gripper 全部显式锁住。"""
        return self.pinned_action(**overrides)

    def robot_pose(self) -> Pose:
        if self.dry_run:
            x, y, yaw = self._mock_base
            half = yaw / 2.0
            quat = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
            return Pose(pos=np.array([x, y, 0.0]), quat=quat)
        # 真实仿真：偶发脏读（Web 线程和 sim 线程之间读 USD prim 缓冲），
        # 表现为 pos=(0,0,0) + quat 为 identity。容错策略：
        #   1) quat 模长 ~0：直接用缓存。
        #   2) pos 突变 > 1.5m 且 quat 是 identity：判定为脏读，用缓存。
        try:
            pos, quat = self.robot.get_position_orientation()
            pos_np = _to_np(pos)
            quat_np = _to_np(quat)
            if np.linalg.norm(quat_np) < 1e-3:
                raise ValueError("invalid quat (zero norm)")
            last = getattr(self, "_last_pose", None)
            if last is not None:
                jump = float(np.linalg.norm(pos_np[:2] - last.pos[:2]))
                is_identity_quat = bool(
                    abs(quat_np[0]) < 1e-6 and abs(quat_np[1]) < 1e-6
                    and abs(quat_np[2]) < 1e-6 and abs(abs(quat_np[3]) - 1.0) < 1e-6
                )
                is_zero_pos = float(np.linalg.norm(pos_np)) < 1e-6
                if jump > 1.5 and is_identity_quat and is_zero_pos:
                    # 几乎肯定是脏读（位姿对应 reset 默认值）
                    return last
            self._last_pose = Pose(pos=pos_np, quat=quat_np)
            return self._last_pose
        except Exception:
            if getattr(self, "_last_pose", None) is not None:
                return self._last_pose
            return Pose(pos=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]))

    def robot_action_dim(self) -> int:
        if self.dry_run:
            return 27
        return int(self.robot.action_dim)

    def empty_action(self) -> np.ndarray:
        """安全 idle action。

        这里不能返回全零向量：R1Pro 的 trunk/arm controller 是 absolute
        position，零向量会把没被要求动的关节目标打到 0，reset/capture/plan
        这类“等一帧”的路径会把躯干和手臂松掉。真实仿真里 empty_action
        等价于 pinned hold；只有 dry-run/mock 保留全零。
        """
        if self.dry_run or self.robot is None:
            return np.zeros(self.robot_action_dim(), dtype=np.float32)
        return self.hold_action_pinned()

    def hold_action(self) -> np.ndarray:
        """全锁定：底盘零速 + trunk/双臂保持当前绝对关节角。"""
        return self.hold_action_pinned()

    def hold_action_pinned(self) -> np.ndarray:
        """底盘零速，躯干/双臂显式锁在当前关节角。"""
        if self.dry_run:
            return self.empty_action()
        kw = self.limb_pin_kwargs()
        kw["base"] = [0.0, 0.0, 0.0]
        return self.make_action_unpinned(**kw)

    def make_action_trunk_locked(self, trunk_q) -> np.ndarray:
        """躯干段：底盘锁零速，双臂锁当前角，只改 trunk。"""
        if self.dry_run:
            return self.empty_action()
        kw = self.limb_pin_kwargs()
        kw["base"] = [0.0, 0.0, 0.0]
        kw["trunk"] = np.asarray(trunk_q, dtype=np.float64).reshape(-1).tolist()
        return self.make_action_unpinned(**kw)

    def base_action_indices(self) -> Tuple[int, int, int]:
        """返回 base 控制器在 action 向量中的 (vx, vy, wz) 三个索引。"""
        if self.dry_run:
            return (0, 1, 2)
        idx = self.robot.controller_action_idx["base"]
        idx = _to_np(idx).astype(int)
        assert len(idx) == 3, f"R1Pro base 期望 3 维，实际 {len(idx)}"
        return tuple(int(i) for i in idx)

    def set_base_velocity(self, vx: float, vy: float, wz: float) -> np.ndarray:
        """底盘 (vx,vy,wz)；躯干 q1–q3/q4 与双臂每步显式锁在当前关节角。"""
        if self.dry_run:
            return self._mock_base_step(vx, vy, wz)

        kw = self.limb_pin_kwargs()
        kw["base"] = [float(vx), float(vy), float(wz)]
        return self.make_action_unpinned(**kw)

    def _compute_noop_action_np(self) -> np.ndarray:
        """调用每个 controller 的 compute_no_op_action 构造完整 noop action（numpy）。"""
        import torch as th

        action = th.zeros(int(self.robot.action_dim))
        try:
            control_dict = self.robot.get_control_dict()
        except Exception:
            return action.cpu().numpy().astype(np.float32)
        for name, controller in self.robot.controllers.items():
            try:
                partial = controller.compute_no_op_action(control_dict)
                idx = self.robot.controller_action_idx[name]
                action[idx] = partial
            except Exception:
                pass
        return action.detach().cpu().numpy().astype(np.float32)

    def _mock_base_step(self, vx: float, vy: float, wz: float, dt: float = 0.05) -> np.ndarray:
        """dry-run 用：直接按局部速度积分 mock 底盘状态。"""
        x, y, yaw = self._mock_base
        # 局部 (vx, vy) -> 世界
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        wx = cos_y * vx - sin_y * vy
        wy = sin_y * vx + cos_y * vy
        self._mock_base = np.array([x + wx * dt, y + wy * dt, yaw + wz * dt])
        action = np.zeros(self.robot_action_dim(), dtype=np.float32)
        action[0] = vx
        action[1] = vy
        action[2] = wz
        return action

    # ---- 物体 / TRO ----

    def list_objects(self) -> List[str]:
        if self.dry_run:
            return ["microwave.n.02_1", "popcorn__bag.n.01_1", "countertop.n.01_1"]
        names = []
        try:
            for obj in self.env.scene.objects:
                names.append(obj.name)
        except Exception:
            pass
        return names

    def get_object_pose(self, name: str) -> Optional[Pose]:
        if self.dry_run:
            return Pose(pos=np.array([1.0, 0.0, 0.5]), quat=np.array([0.0, 0.0, 0.0, 1.0]))
        obj = self._resolve_object(name)
        if obj is None:
            return None
        pos, quat = obj.get_position_orientation()
        return Pose(pos=_to_np(pos), quat=_to_np(quat))

    def _resolve_object(self, name: str):
        """先按 BDDL scope，再按 scene.object_registry by name。"""
        try:
            task = self.env.task
            if hasattr(task, "object_scope") and task.object_scope:
                if name in task.object_scope:
                    ent = task.object_scope[name]
                    return getattr(ent, "unwrapped", ent)
        except Exception:
            pass
        try:
            obj = self.env.scene.object_registry("name", name)
            if obj is not None:
                return obj
        except Exception:
            pass
        return None

    # ---- Scene Graph ----

    def build_scene_graph(self, robot_radius: float = 0.25):
        """构建 SceneGraph。**必须在主 sim 线程调用**（会读 PhysX）。"""
        from behavior_interface.scene_graph import build_scene_graph
        if self.dry_run:
            return None
        if self.env is None or self.robot is None:
            return None
        return build_scene_graph(self.env, self.robot, robot_radius=robot_radius)

    def plan_path(self, free_region, start, goal,
                  resolution: float = 0.15, extra_inflate: float = 0.0):
        """A* 规划。free_region 由 build_scene_graph 提供。"""
        from behavior_interface.scene_graph import plan_path
        return plan_path(free_region, tuple(start), tuple(goal),
                         resolution=resolution, extra_inflate=extra_inflate)

    def is_point_free(self, free_region, x: float, y: float,
                      extra_inflate: float = 0.0):
        from behavior_interface.scene_graph import is_point_free
        return is_point_free(free_region, x, y, extra_inflate=extra_inflate)

    def is_segment_free(self, free_region, p0, p1,
                        step: float = 0.05, extra_inflate: float = 0.0):
        from behavior_interface.scene_graph import is_segment_free
        return is_segment_free(free_region, tuple(p0), tuple(p1),
                               step=step, extra_inflate=extra_inflate)

    # ---- TRO（保留原接口，scene_graph 不替代它，给老代码兜底）----

    def task_relevant_state(self) -> Dict[str, Any]:
        """简化版 TRO：列出 BDDL object_scope 中实体位姿。
        真实仿真下读 env.task.object_scope。
        """
        if self.dry_run:
            return {
                "microwave.n.02_1": {
                    "pos": [6.12, -0.70, 1.01],
                    "yaw_deg": 90.0,
                    "open": False,
                },
                "popcorn__bag.n.01_1": {
                    "pos": [7.53, -0.52, 0.98],
                    "yaw_deg": 90.0,
                },
            }
        out: Dict[str, Any] = {}
        try:
            task = self.env.task
            for key, ent in (task.object_scope or {}).items():
                obj = getattr(ent, "unwrapped", ent)
                if obj is None or not hasattr(obj, "get_position_orientation"):
                    continue
                try:
                    if not getattr(ent, "exists", True):
                        continue
                except Exception:
                    pass
                try:
                    pos, quat = obj.get_position_orientation()
                except Exception:
                    continue
                pose = Pose(pos=_to_np(pos), quat=_to_np(quat))
                entry: Dict[str, Any] = {
                    "pos": pose.pos.tolist(),
                    "yaw_deg": round(math.degrees(pose.yaw), 2),
                }
                # 顺手记录 Open 状态
                try:
                    from omnigibson.object_states import Open  # type: ignore

                    if Open in obj.states:
                        entry["open"] = bool(obj.states[Open].get_value())
                except Exception:
                    pass
                out[key] = entry
        except Exception:
            pass
        return out
