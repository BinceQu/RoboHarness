"""胸前 tuck：两阶段在线规划 + 弧长平滑播放。

Phase1 垂直抬升：j1 后摆 + j4 屈肘协同（三角形 C 扫描），尽量少向前、到达垂直演示位。
Phase2 贴胸收拢：自然收向胸前 pose；EEF 世界系 Z 不降；夹爪相对胸廓 forward 投影 <20cm。
exec 只动 arm 7 关节；规划 FK 段内包络，播放沿关节弧长余弦缓动。
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from behavior_interface.skills import register_skill

TUCK_CHEST_FORWARD_MAX_M = 0.15  # 旧离线标定折线
TUCK_EXEC_FORWARD_MAX_M = 0.20   # 验收：夹爪相对胸廓 forward 投影 <20cm
TUCK_PLAN_FORWARD_MAX_M = 0.16   # FK 段内包络（留动态跟踪裕度）
# 播放：沿关节弧长余弦缓动，限制每帧 |Δq|
TUCK_MAX_DQ_PER_STEP = 0.58  # 仅 arm_reset 等复用
TUCK_HOLD_FRAMES = 6
TUCK_PLAY_DQ_PER_FRAME = 0.015  # 弧长播放每帧约 max|Δq|
TUCK_PHASE_HOLD_FRAMES = 8  # Phase1 末短暂停稳再进 Phase2
TUCK_PLAY_FRAMES_MIN = 72
TUCK_EXEC_MAX_WAYPOINTS = 360
TUCK_EXEC_MAX_PLAY_FRAMES = 540
TUCK_EXEC_MAX_ARC_RAD = 8.0
TUCK_FRAMES_PER_WP = 32  # 旧路点播放保留
TUCK_POS_TOL_RAD = 0.048
TUCK_DENSIFY_MAX_DQ = 0.030  # 规划/播放前段内加密
# R1Pro 腰链（eval_utils.JOINT_RANGE）；torso_joint4 为链顶 +Z 转台（腰部最高电机）
_R1PRO_TRUNK_LIMITS = (
    (-1.1345, 1.8326),   # torso_joint1
    (-2.7925, 2.5307),   # torso_joint2
    (-1.8326, 1.5708),   # torso_joint3
    (-3.0543, 3.0543),   # torso_joint4（最高）
)
_R1PRO_TRUNK_Q4_RANGE = (-3.0543, 3.0543)
_PILLAR_LINKS = ("torso_link1", "torso_link2", "torso_link3")
_GRIPPER_PILLAR_SOFT_M = 0.09    # 腰柱净空软目标（非硬拒绝）
TUCK_PHASE2_MIN_LATERAL_M = 0.0   # lateral 仅 diag 汇报
TUCK_PHASE2_WAIST_MIN_M = 0.15     # Phase2 腰顶枢轴距离下限（默认 15cm）
TUCK_PHASE2_LATERAL_OUTWARD_MAX_M = 0.05  # 左臂向左/右臂向右相对 Phase1 末 ≤5cm
TUCK_PHASE2_Z_DROP_MAX_M = 0.03          # Phase2 EEF Z ≥ Phase1末 − 3cm
TUCK_PHASE2_Z_DESCEND_MIN_M = 0.005      # Phase2 终点至少比 Phase1 末低 5mm
# torso_joint4 在父系 torso_link3 下的关节原点（URDF：绕 +Z，y=0 在对称面上）
_TRUNK_J4_PIVOT_IN_LINK3 = np.array([0.0, 0.0, 0.1], dtype=np.float64)
# torso_joint3 在 torso_link3 局部原点（第二高腰电机）
_TRUNK_J3_PIVOT_IN_LINK3 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
# link3 局部 z：下层电机 + 侧立柱（j3 原点=0，j4 枢轴≈0.1）
_WAIST_LINK3_LOCAL_Z = (-0.10, 0.12)
# link4 局部 z：上层电机 + 腰台；超过此值接胸廓白盖板
_WAIST_LINK4_LOCAL_Z_MAX = 0.09
# 世界 z 裁剪：以 j3 底、j4 顶各扩约一颗电机半径
_WAIST_J34_Z_MARGIN_BELOW_M = 0.10
_WAIST_J34_Z_MARGIN_ABOVE_M = 0.10
# link 局部 |y| 上限：剔除肩台/臂座，只留腰柱侧板内（半宽 10cm）
_WAIST_LINK_LOCAL_Y_ABS_MAX = 0.10
# j4 腰台测宽（包住白色腰台侧板，略宽于纯电机壳）
_WAIST_J34_SHELL_Y_ABS_MAX = 0.09
# 禁入盒：测宽后再略放大；沿 j3→j4 轴向下平移「半高」
_WAIST_J34_FORBIDDEN_WIDTH_SCALE = 1.1
_WAIST_J34_FORBIDDEN_SHIFT_DOWN_H_FRAC = 0.5
# 禁入区沿 forward 以腰台深度 L 为基准，前/后各延伸 N×L（需碰到本体）
_WAIST_J34_FORBIDDEN_FWD_BACK_EXTEND_FACTOR = 3.0

# 胸前目标位形（R1Pro 7-DOF；胸口 forward≤15cm，较上一版更贴胸）
CHEST_TUCK_ARM = {
    "right": np.array([-0.73, 0.09, 1.245, -2.04, -0.18, 0.03, 0.08], dtype=np.float64),
    "left": np.array([-0.73, -0.09, -1.245, -2.04, 0.26, -0.03, -0.08], dtype=np.float64),
}

# hang(全零) → 胸前：肩后收 + 屈肘递进，段内关节线性插值
# 标定目标：各段插值采样点胸口 forward ≤ 15cm（见 diag_tuck_trajectory 验证）
def _wp(rows: List[List[float]]) -> List[np.ndarray]:
    return [np.array(r, dtype=np.float64) for r in rows]


def _mirror_tuck_q_for_left(q_right: np.ndarray) -> np.ndarray:
    """右臂胸前关节角镜像到左臂（R1Pro 左右对称约定）。"""
    q = np.asarray(q_right, dtype=np.float64).reshape(7)
    return np.array([q[0], -q[1], -q[2], q[3], -q[4], -q[5], -q[6]], dtype=np.float64)


# 仿真贪心标定（diag_tuck_trajectory replan=true）：段内插值 max chest_fwd≤15cm
_TUCK_TRAJECTORY: Dict[str, List[np.ndarray]] = {
    "right": _wp([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.275, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.275, -0.275, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.495, -0.33, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.77, -0.385, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.935, -0.495, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.045, -0.66, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.155, -0.77, -0.055, 0.0, 0.0],
        [0.0, 0.055, 1.21, -0.935, -0.11, 0.0, 0.0],
        [0.0, 0.055, 1.21, -1.155, -0.11, 0.0, 0.055],
        [0.0, 0.11, 1.265, -1.265, -0.165, 0.0, 0.055],
        [-0.055, 0.11, 1.265, -1.485, -0.165, 0.0, 0.055],
        [-0.11, 0.11, 1.265, -1.76, -0.165, 0.0, 0.055],
        [-0.22, 0.11, 1.265, -1.925, -0.165, 0.0, 0.055],
        [-0.44, 0.11, 1.265, -1.98, -0.165, 0.0, 0.055],
        [-0.73, 0.09, 1.245, -2.04, -0.18, 0.03, 0.08],
    ]),
    "left": _wp([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.275, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.275, -0.275, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.55, -0.33, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.77, -0.385, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.935, -0.55, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.045, -0.66, 0.055, 0.0, 0.0],
        [0.0, 0.0, -1.155, -0.715, 0.165, 0.0, 0.0],
        [0.0, -0.055, -1.21, -0.88, 0.22, 0.0, 0.0],
        [0.0, -0.055, -1.21, -1.1, 0.22, 0.0, -0.055],
        [0.0, -0.11, -1.265, -1.265, 0.275, 0.0, -0.055],
        [-0.055, -0.11, -1.265, -1.485, 0.275, 0.0, -0.055],
        [-0.055, -0.11, -1.265, -1.76, 0.275, 0.0, -0.055],
        [-0.22, -0.11, -1.265, -1.925, 0.275, 0.0, -0.055],
        [-0.44, -0.11, -1.265, -1.98, 0.275, 0.0, -0.055],
        [-0.73, -0.09, -1.245, -2.04, 0.26, -0.03, -0.08],
    ]),
}


def _chest_forward_axis(world):
    ch = world.chest_pose()
    chest = np.array([ch["x"], ch["y"], ch["z"]], dtype=np.float64)
    fwd = np.asarray(ch["forward"], dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(fwd))
    if n > 1e-9:
        fwd = fwd / n
    return chest, fwd


def _probe_eef_pos_at_arm_qpos(world, arm: str, arm_q: np.ndarray) -> Optional[np.ndarray]:
    from behavior_interface.skills.grasp import _get_arm_dof_idx

    saved = None
    robot = None
    try:
        robot = world.robot
        idx = _get_arm_dof_idx(world, arm)
        saved = robot.get_joint_positions().clone()
        q = saved.clone()
        aq = np.asarray(arm_q, dtype=np.float64).reshape(7)
        for i, j in enumerate(idx):
            q[int(j)] = float(aq[i])
        robot.set_joint_positions(q)
        eef = world.eef_pose(arm=arm)
        return np.asarray(eef["pos"], dtype=np.float64).reshape(3)
    except Exception:
        return None
    finally:
        if saved is not None and robot is not None:
            try:
                robot.set_joint_positions(saved)
            except Exception:
                pass


def eef_chest_forward_dist_m(world, arm: str, arm_q: np.ndarray) -> Optional[float]:
    epos = _probe_eef_pos_at_arm_qpos(world, arm, arm_q)
    if epos is None:
        return None
    chest, fwd = _chest_forward_axis(world)
    return float(np.dot(epos - chest, fwd))


def _eef_world_z_m(world, arm: str, arm_q: np.ndarray) -> Optional[float]:
    epos = _probe_eef_pos_at_arm_qpos(world, arm, arm_q)
    if epos is None:
        return None
    return float(epos[2])


def _eef_world_xy(world, arm: str, arm_q: np.ndarray) -> Optional[np.ndarray]:
    epos = _probe_eef_pos_at_arm_qpos(world, arm, arm_q)
    if epos is None:
        return None
    return np.asarray(epos[:2], dtype=np.float64)


def _eef_chest_horiz_components(
    world, arm: str, arm_q: np.ndarray,
) -> Optional[tuple]:
    """胸口坐标：forward(沿胸朝前)、lateral(水平左右)、world_z。"""
    epos = _probe_eef_pos_at_arm_qpos(world, arm, arm_q)
    if epos is None:
        return None
    chest, fwd = _chest_forward_axis(world)
    rel = np.asarray(epos, dtype=np.float64).reshape(3) - chest
    fwd_c = float(np.dot(rel, fwd))
    lat_dir = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
    ln = float(np.linalg.norm(lat_dir))
    if ln < 1e-9:
        lat_dir = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        ln = float(np.linalg.norm(lat_dir))
    if ln > 1e-9:
        lat_dir = lat_dir / ln
    lat_c = float(np.dot(rel, lat_dir))
    return fwd_c, lat_c, float(epos[2])


def _probe_eef_pose_at_arm_qpos(
    world, arm: str, arm_q: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """FK 探针：EEF 位姿 (pos, quat_xyzw)。"""
    from behavior_interface.skills.grasp import _get_arm_dof_idx

    saved = None
    robot = None
    try:
        robot = world.robot
        idx = _get_arm_dof_idx(world, arm)
        saved = robot.get_joint_positions().clone()
        q = saved.clone()
        aq = np.asarray(arm_q, dtype=np.float64).reshape(7)
        for i, j in enumerate(idx):
            q[int(j)] = float(aq[i])
        robot.set_joint_positions(q)
        eef = world.eef_pose(arm=arm)
        pos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(eef.get("quat", eef.get("orientation")), dtype=np.float64).reshape(4)
        return pos, quat
    except Exception:
        return None
    finally:
        if saved is not None and robot is not None:
            try:
                robot.set_joint_positions(saved)
            except Exception:
                pass


_GRIPPER_PROBE_LOCAL_PTS: Optional[np.ndarray] = None


def _gripper_probe_points_local() -> np.ndarray:
    """夹爪 AABB 角点 + 稀疏 mesh 顶点（本地系，缓存）。"""
    global _GRIPPER_PROBE_LOCAL_PTS
    if _GRIPPER_PROBE_LOCAL_PTS is not None:
        return _GRIPPER_PROBE_LOCAL_PTS
    from behavior_interface.skills.viz_eef_v2 import gripper_trimesh_eef

    tm = gripper_trimesh_eef()
    if tm is None:
        _GRIPPER_PROBE_LOCAL_PTS = np.zeros((1, 3), dtype=np.float64)
        return _GRIPPER_PROBE_LOCAL_PTS
    verts = np.asarray(tm.vertices, dtype=np.float64)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
    ], dtype=np.float64)
    sparse = verts[:: max(1, len(verts) // 24)]
    _GRIPPER_PROBE_LOCAL_PTS = np.vstack([corners, sparse])
    return _GRIPPER_PROBE_LOCAL_PTS


def _gripper_mesh_vertices_world(
    world, arm: str, arm_q: np.ndarray, *, stride: int = 3,
) -> Optional[np.ndarray]:
    """夹爪 mesh 顶点世界坐标（子采样，用于禁飞区碰撞复核）。"""
    from behavior_interface.skills.viz_eef_v2 import gripper_trimesh_eef

    pose = _probe_eef_pose_at_arm_qpos(world, arm, arm_q)
    tm = gripper_trimesh_eef()
    if pose is None or tm is None:
        return None
    pos, quat = pose
    R = _quat_xyzw_to_rot(quat)
    verts = np.asarray(tm.vertices, dtype=np.float64)
    if int(stride) > 1:
        verts = verts[:: int(stride)]
    return (R @ verts.T).T + pos.reshape(1, 3)


def _gripper_probe_points_world(
    world, arm: str, arm_q: np.ndarray,
) -> Optional[np.ndarray]:
    """夹爪探针点世界坐标（RRT 快速碰撞）。"""
    pose = _probe_eef_pose_at_arm_qpos(world, arm, arm_q)
    if pose is None:
        return None
    pos, quat = pose
    R = _quat_xyzw_to_rot(quat)
    local = _gripper_probe_points_local()
    return (R @ local.T).T + pos.reshape(1, 3)


def _point_in_waist_j34_forbidden_box(
    p_world: np.ndarray,
    forbidden_box: dict,
    world,
    *,
    touch_eps_m: float = 1e-4,
) -> bool:
    """点是否在腰 j3↔j4 禁飞区内（含贴边 touch_eps）。"""
    basis = _module_basis_from_box(world, forbidden_box)
    if basis is None:
        return False
    origin, fwd_h, lat, up = basis
    rel = np.asarray(p_world, dtype=np.float64).reshape(3) - origin
    f = float(rel @ fwd_h)
    la = float(rel @ lat)
    hh = float(rel @ up)
    eps = float(touch_eps_m)
    return (
        float(forbidden_box["f_lo_m"]) - eps <= f <= float(forbidden_box["f_hi_m"]) + eps
        and float(forbidden_box["lat_lo_m"]) - eps <= la <= float(forbidden_box["lat_hi_m"]) + eps
        and float(forbidden_box["z_lo_m"]) - eps <= hh <= float(forbidden_box["z_hi_m"]) + eps
    )


def _gripper_hits_waist_j34_forbidden(
    world, arm: str, arm_q: np.ndarray, forbidden_box: Optional[dict],
    *,
    fine_mesh: bool = False,
) -> bool:
    """夹爪任一探针点在禁飞区内 → 触碰。"""
    if not forbidden_box:
        return False
    if fine_mesh:
        verts = _gripper_mesh_vertices_world(world, arm, arm_q, stride=2)
    else:
        verts = _gripper_probe_points_world(world, arm, arm_q)
    if verts is None:
        return True
    for p in verts:
        if _point_in_waist_j34_forbidden_box(p, forbidden_box, world):
            return True
    return False


def _waist_j34_forbidden_box_pinned(world) -> Optional[dict]:
    """当前 trunk 俯身角下冻结的 j3↔j4 禁飞区。"""
    est = _waist_j34_axis_symmetric_box(world, world.trunk_qpos().copy())
    return est.get("forbidden_box")


def _self_collision_pairs(world, impulse_thresh=1e-6):
    """返回当前帧机器人自碰撞的 (linkA, linkB) 对集合。"""
    import numpy as _np

    robot = world.robot
    root = robot.prim_path
    pairs = set()
    try:
        contacts = robot.contact_list()
    except Exception:
        return pairs
    for c in contacts:
        b0 = str(c.body0)
        b1 = str(c.body1)
        if not (b0.startswith(root) and b1.startswith(root)):
            continue
        imp = c.impulse
        try:
            mag = float(_np.linalg.norm(_np.asarray(imp, dtype=float)))
        except Exception:
            mag = 1.0
        if mag < impulse_thresh:
            continue
        n0 = b0.split("/")[-1]
        n1 = b1.split("/")[-1]
        if n0 == n1:
            continue
        pairs.add(tuple(sorted((n0, n1))))
    return pairs


def _arm_self_collision_at_q(world, arm: str, arm_q: np.ndarray) -> bool:
    """FK 探针后检查手臂相关自碰。"""
    if _probe_eef_pos_at_arm_qpos(world, arm, arm_q) is None:
        return True
    prefix = f"{arm}_"
    for a, b in _self_collision_pairs(world):
        if a.startswith(prefix) or b.startswith(prefix):
            return True
        if a.startswith("torso_") and b.startswith(prefix):
            return True
    return False


TUCK_LIFT_FORWARD_SOFT_M = 0.05  # Phase1 软目标：尽量少向前（非硬约束）
TUCK_LIFT_LATERAL_TOL_M = 0.025  # Phase1：胸口左右漂移 |Δlateral| 软惩罚
# R1Pro 肘 j4 限位（eval_utils.JOINT_RANGE）
_R1PRO_ARM_J4_LIMITS = {
    "left": (-2.0944, 0.3491),
    "right": (-2.0944, 0.3491),
}
# R1Pro 7-DOF 臂关节限位（eval_utils.JOINT_RANGE）
_R1PRO_ARM_LIMITS = {
    "left": (
        np.array([-4.4506, -0.1745, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708], dtype=np.float64),
        np.array([1.3090, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708], dtype=np.float64),
    ),
    "right": (
        np.array([-4.4506, -3.1416, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708], dtype=np.float64),
        np.array([1.3090, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708], dtype=np.float64),
    ),
}


def _arm_joint_pivot_world(world, arm: str, joint_i: int) -> Optional[np.ndarray]:
    """关节转轴原点世界坐标（用对应 link 原点近似）。"""
    link_i = {0: 1, 3: 4}.get(joint_i, joint_i + 1)
    return _link_world_pos(world, f"{arm}_arm_link{link_i}")


def _elbow_joint_angle_rad(p_sh: np.ndarray, p_el: np.ndarray, p_ee: np.ndarray) -> float:
    """肘关节夹角 C：大臂 a 与小臂 b 的夹角；完全伸直为 π(180°)，屈肘减小。"""
    u = np.asarray(p_el, dtype=np.float64) - np.asarray(p_sh, dtype=np.float64)
    v = np.asarray(p_ee, dtype=np.float64) - np.asarray(p_el, dtype=np.float64)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-9 or nv < 1e-9:
        return float(np.pi)
    cos_uv = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.pi - np.arccos(cos_uv))


def _elbow_interior_angle_rad(p_sh: np.ndarray, p_el: np.ndarray, p_ee: np.ndarray) -> float:
    """同 _elbow_joint_angle_rad（历史别名）。"""
    return _elbow_joint_angle_rad(p_sh, p_el, p_ee)


def _measure_lift_triangle_ab(world, arm: str, q_ref: np.ndarray) -> Optional[tuple]:
    """测量肩(j1/link1)枢轴→肘(j4/link4) 距离 a，肘→EEF 距离 b。"""
    p_sh = _arm_joint_pivot_world(world, arm, 0)
    p_el = _arm_joint_pivot_world(world, arm, 3)
    p_ee = _probe_eef_pos_at_arm_qpos(world, arm, q_ref)
    if p_sh is None or p_el is None or p_ee is None:
        return None
    a = float(np.linalg.norm(p_el - p_sh))
    b = float(np.linalg.norm(p_ee - p_el))
    return a, b, p_sh, p_el, p_ee


def _triangle_c_and_shoulder_angle(a: float, b: float, C: float) -> tuple:
    """余弦定理：已知 a、b 及肘夹角 C(伸直=π)，求 c 与肩端夹角 A（对边为 a 的角）。"""
    C = float(np.clip(C, 1e-6, np.pi))
    c = float(np.sqrt(max(a * a + b * b - 2.0 * a * b * np.cos(C), 1e-12)))
    cos_a = float(np.clip((a * a + c * c - b * b) / (2.0 * a * c + 1e-12), -1.0, 1.0))
    A = float(np.arccos(cos_a))
    cos_b = float(np.clip((b * b + c * c - a * a) / (2.0 * b * c + 1e-12), -1.0, 1.0))
    B = float(np.arccos(cos_b))
    return c, A, B


def _solve_j1_for_j4_triangle(
    world,
    arm: str,
    j4: float,
    a: float,
    b: float,
    C_tgt: float,
    *,
    j1_lo: float = -2.5,
    j1_hi: float = 0.5,
    n_scan: int = 48,
) -> Optional[float]:
    """给定 j4 与三角形目标角 C，搜 j1 使 FK 肘夹角≈C 且 |肩→EEF|≈c。"""
    c_tgt, _, _ = _triangle_c_and_shoulder_angle(a, b, C_tgt)
    p_sh0 = _arm_joint_pivot_world(world, arm, 0)
    if p_sh0 is None:
        return None
    best_j1: Optional[float] = None
    best_err = float("inf")
    for j1 in np.linspace(float(j1_lo), float(j1_hi), int(n_scan)):
        q = np.zeros(7, dtype=np.float64)
        q[0] = float(j1)
        q[3] = float(j4)
        p_sh = _arm_joint_pivot_world(world, arm, 0)
        p_el = _arm_joint_pivot_world(world, arm, 3)
        p_ee = _probe_eef_pos_at_arm_qpos(world, arm, q)
        if p_sh is None or p_el is None or p_ee is None:
            continue
        C_act = _elbow_joint_angle_rad(p_sh, p_el, p_ee)
        c_act = float(np.linalg.norm(p_ee - p_sh))
        err = abs(C_act - C_tgt) + 0.35 * abs(c_act - c_tgt) / max(c_tgt, 1e-6)
        if err < best_err:
            best_err = err
            best_j1 = float(j1)
    return best_j1


def _shoulder_j1_backswing_bounds(arm: str) -> tuple:
    """离胸最近肩电机 j1 后摆：hang(0) → j1+（与收胸 j1≈-0.73 相反）。

    左右臂收胸位形 CHEST_TUCK_ARM 的 j1 均为负；后摆应增大 j1，而非右臂取负。
    """
    _ = (arm or "right").strip().lower()
    return 0.0, 1.55


def _interpolate_arm_qpath(
    q_start: np.ndarray,
    q_end: np.ndarray,
    n_steps: int,
) -> List[np.ndarray]:
    """关节空间线性插值（仅用于 Phase1 短路径加密）。"""
    q0 = np.asarray(q_start, dtype=np.float64).reshape(7)
    q1 = np.asarray(q_end, dtype=np.float64).reshape(7)
    n = max(2, int(n_steps))
    return [q0 + (q1 - q0) * (i / (n - 1)) for i in range(n)]


def _search_lift_endpoint(
    world,
    arm: str,
    q0: np.ndarray,
    *,
    fwd0: float,
    lat0: float,
    z0: float,
    max_forward_delta_m: float,
    lateral_tol_m: float,
) -> Optional[np.ndarray]:
    """在 (j1,j4) 上网格搜索 Phase1 终点：Z 最高且满足 forward/lateral 约束。"""
    arm = (arm or "right").strip().lower()
    j4_lo, _ = _R1PRO_ARM_J4_LIMITS.get(arm, (-2.0944, 0.3491))
    j1_lo, j1_hi = _shoulder_j1_backswing_bounds(arm)
    j1_vals = np.linspace(float(j1_lo), float(j1_hi), 14)
    best_q: Optional[np.ndarray] = None
    best_z = float(z0)
    for j1 in j1_vals:
        for j4 in np.linspace(0.0, float(j4_lo), 28):
            qtry = np.zeros(7, dtype=np.float64)
            qtry[0] = float(j1)
            qtry[3] = float(j4)
            h1 = _eef_chest_horiz_components(world, arm, qtry)
            zz = _eef_world_z_m(world, arm, qtry)
            if h1 is None or zz is None:
                continue
            dfwd = float(h1[0] - fwd0)
            dlat = abs(float(h1[1] - lat0))
            if dfwd > float(max_forward_delta_m) + 1e-4 or dfwd < -0.02:
                continue
            if dlat > float(lateral_tol_m):
                continue
            if zz > best_z + 1e-5:
                best_z = float(zz)
                best_q = qtry
    return best_q


def _solve_phase1_vertical_c_pose(world, arm: str) -> Optional[dict]:
    """Phase1 终点：j4 屈至限位 + j1 后摆，c 竖直；软惩罚 forward/XY，无硬上限。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "right").strip().lower()
    j4_lo, _ = _R1PRO_ARM_J4_LIMITS.get(arm, (-2.0944, 0.3491))
    q_hang = np.asarray(_HANG_ARM_QPOS, dtype=np.float64).reshape(7).copy()
    xy_hang = _eef_world_xy(world, arm, q_hang)
    z_hang = _eef_world_z_m(world, arm, q_hang)
    h_hang = _eef_chest_horiz_components(world, arm, q_hang)
    if xy_hang is None or z_hang is None or h_hang is None:
        return None
    fwd0, lat0, _ = h_hang
    meas = _measure_lift_triangle_ab(world, arm, q_hang)
    a_cm = round(meas[0] * 100, 2) if meas else None
    b_cm = round(meas[1] * 100, 2) if meas else None
    j1_lo, j1_hi = _shoulder_j1_backswing_bounds(arm)
    j1_vals = np.linspace(float(j1_lo), float(j1_hi), 240)

    best: Optional[dict] = None
    best_score = float("inf")
    for j1 in j1_vals:
        q = q_hang.copy()
        q[0] = float(j1)
        q[3] = float(j4_lo)
        p_sh = _arm_joint_pivot_world(world, arm, 0)
        p_el = _arm_joint_pivot_world(world, arm, 3)
        p_ee = _probe_eef_pos_at_arm_qpos(world, arm, q)
        xy = _eef_world_xy(world, arm, q)
        h1 = _eef_chest_horiz_components(world, arm, q)
        if p_sh is None or p_el is None or p_ee is None or xy is None or h1 is None:
            continue
        dxy = float(np.linalg.norm(xy - xy_hang))
        dfwd = float(h1[0] - fwd0)
        dlat = abs(float(h1[1] - lat0))
        c_vec = np.asarray(p_ee - p_sh, dtype=np.float64)
        horiz = float(np.linalg.norm(c_vec[:2]))
        c_len = float(np.linalg.norm(c_vec))
        if c_len < 1e-6:
            continue
        vert_ratio = horiz / c_len
        C = _elbow_joint_angle_rad(p_sh, p_el, p_ee)
        zz = float(p_ee[2])
        # 软目标：c 竖直、少向前、少 XY 漂移；必须到达肘限位后摆位
        score = (
            vert_ratio * 400.0
            + dxy * 600.0
            + max(dfwd, 0.0) * 250.0
            + max(dfwd - TUCK_LIFT_FORWARD_SOFT_M, 0.0) * 800.0
            + dlat * 150.0
            - zz * 0.02
        )
        if score < best_score:
            best_score = score
            best = {
                "q": q.copy(),
                "j1": float(j1),
                "j4": float(j4_lo),
                "a_cm": a_cm,
                "b_cm": b_cm,
                "dxy_cm": round(dxy * 100, 2),
                "delta_forward_cm": round(dfwd * 100, 2),
                "delta_lateral_cm": round(dlat * 100, 2),
                "c_horiz_cm": round(horiz * 100, 3),
                "c_len_cm": round(c_len * 100, 2),
                "c_tilt_deg": round(float(np.degrees(np.arctan2(horiz, abs(c_vec[2]) + 1e-9))), 3),
                "C_deg": round(float(np.degrees(C)), 2),
                "eef_z_m": round(zz, 4),
                "dz_cm": round((zz - z_hang) * 100, 2),
                "c_vec_m": [round(float(v), 4) for v in c_vec],
                "self_collide_fk": bool(_arm_self_collision_at_q(world, arm, q)),
            }
    return best


def _next_tuck_video_path() -> str:
    """下一版 tuck 过程视频路径：behavior_interface/skills/test/exec_move/tuck_vN.mp4。"""
    base = os.path.join(os.path.dirname(__file__), "test", "exec_move")
    os.makedirs(base, exist_ok=True)
    mx = 0
    for p in glob.glob(os.path.join(base, "tuck_v*.mp4")):
        m = re.search(r"tuck_v(\d+)", os.path.basename(p))
        if m:
            mx = max(mx, int(m.group(1)))
    return os.path.join(base, f"tuck_v{mx + 1}.mp4")


def _gta_rgb_array(world) -> Optional[np.ndarray]:
    """读取 gta_view 当前帧 RGB (H,W,3) uint8。"""
    try:
        gta = world.env._external_sensors.get("gta_view")
        if gta is None:
            return None
        obs, _ = gta.get_obs()
        rgb = None
        for k, v in obs.items():
            if "rgb" in k.lower():
                rgb = v
                break
        if rgb is None:
            return None
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr
    except Exception:
        return None


class _GtaVideoRecorder:
    """GTA 主视角逐帧录像。"""

    def __init__(self, fpath: str, fps: int = 8):
        self.fpath = str(fpath)
        self.fps = int(fps)
        self._frames: List[np.ndarray] = []

    def capture(self, world) -> None:
        arr = _gta_rgb_array(world)
        if arr is not None:
            self._frames.append(arr)

    def save(self, ctx) -> bool:
        if not self._frames:
            return False
        try:
            import imageio
            os.makedirs(os.path.dirname(self.fpath) or ".", exist_ok=True)
            imageio.mimwrite(
                self.fpath, self._frames, fps=self.fps, macro_block_size=1,
            )
            ctx.log(f"[tuck_video] {self.fpath} frames={len(self._frames)} fps={self.fps}")
            return True
        except Exception as e:
            ctx.log(f"[tuck_video] save failed: {e}")
            return False


# tuck 录像统一 GTA 视角：dist=1.90m h=2.80m tilt(look_z)=1.20m yaw_off=135°
TUCK_GTA_CAMERA_OVERRIDES = {
    "distance": 1.90,
    "height": 2.80,
    "look_z_offset": 1.20,
    "yaw_offset_deg": 135.0,
}


def _ensure_tuck_gta_camera(ctx) -> None:
    """录像前固定 GTA 主视角（由 server 每帧 _update_gta_camera_pose 应用）。"""
    fn = getattr(ctx, "_adjust_camera", None)
    if fn is None:
        return
    snap = fn(overrides=dict(TUCK_GTA_CAMERA_OVERRIDES))
    ctx.log(
        "[tuck_cam] GTA "
        f"dist={snap.get('distance')}m h={snap.get('height')}m "
        f"tilt(look_z)={snap.get('look_z_offset')}m "
        f"yaw_off={snap.get('yaw_offset_deg')}°"
    )


def _make_tuck_video_recorder(ctx, fpath: str, fps: int = 10) -> _GtaVideoRecorder:
    _ensure_tuck_gta_camera(ctx)
    return _GtaVideoRecorder(fpath, fps=fps)


def _capture_gta_png(world, fpath: str, ctx) -> bool:
    """抓取 gta_view RGB 存盘（拍照前勿再 yield，避免外参被重置）。"""
    try:
        import cv2
        gta = world.env._external_sensors.get("gta_view")
        if gta is None:
            return False
        obs, _ = gta.get_obs()
        rgb = None
        for k, v in obs.items():
            if "rgb" in k.lower():
                rgb = v
                break
        if rgb is None:
            return False
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
        cv2.imwrite(fpath, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        ctx.log(f"[gta_cap] {fpath} shape={arr.shape}")
        return True
    except Exception as e:
        ctx.log(f"[gta_cap] failed: {e}")
        return False


def _gta_camera_meta(world) -> Optional[dict]:
    """读取 gta_view 内外参。"""
    try:
        gta = world.env._external_sensors.get("gta_view")
        if gta is None:
            return None
        pos, quat = gta.get_position_orientation()
        return {
            "cam_pos": np.asarray(pos, dtype=np.float64).reshape(3),
            "cam_quat": np.asarray(quat, dtype=np.float64).reshape(4),
            "focal_length": float(getattr(gta, "focal_length", 17.0)),
            "horizontal_aperture": float(getattr(gta, "horizontal_aperture", 20.995)),
            "w": int(getattr(gta, "image_width", 1280)),
            "h": int(getattr(gta, "image_height", 720)),
        }
    except Exception:
        return None


def _project_world_to_uv(
    pts_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """世界点 → 像素 uv[N,2] 与有效掩码。"""
    from behavior_interface.skills.viz_gripper_overlay import _project_points

    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    uv, zc = _project_points(pts, cam_pos, cam_quat, w, h, fl, ha)
    valid = np.isfinite(uv).all(axis=1) & (zc < -1e-5)
    return uv, valid


def _link_local_to_world(world, link_name: str, local_pts: np.ndarray) -> np.ndarray:
    """link 局部点 → 世界坐标。"""
    link = world.robot.links.get(link_name)
    if link is None:
        return np.zeros((0, 3), dtype=np.float64)
    lpos_t, lquat_t = link.get_position_orientation()
    lpos = np.asarray(lpos_t, dtype=np.float64).reshape(3)
    R = _quat_xyzw_to_rot(np.asarray(lquat_t, dtype=np.float64))
    loc = np.asarray(local_pts, dtype=np.float64).reshape(-1, 3)
    return (R @ loc.T).T + lpos


def _aabb_corners_world(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    corners = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                corners.append([
                    lo[0] if ix == 0 else hi[0],
                    lo[1] if iy == 0 else hi[1],
                    lo[2] if iz == 0 else hi[2],
                ])
    mid = 0.5 * (lo + hi)
    corners.append(mid.tolist())
    return np.asarray(corners, dtype=np.float64)


def _probe_set_trunk_qpos(world, trunk_q: np.ndarray):
    """写入 trunk 关节并刷新渲染；返回恢复用 saved qpos。"""
    import omnigibson as og

    robot = world.robot
    idx = np.asarray(robot.trunk_control_idx, dtype=int).reshape(-1)
    saved = robot.get_joint_positions().clone()
    q = saved.clone()
    tq = np.asarray(trunk_q, dtype=np.float64).reshape(len(idx))
    for ti, j in enumerate(idx):
        q[int(j)] = float(tq[ti])
    robot.set_joint_positions(q)
    for _ in range(3):
        og.sim.render()
    return saved


def _collect_pillar_world_points(world, trunk_q: np.ndarray) -> np.ndarray:
    """当前 trunk 下支柱 link 碰撞 AABB 采样点（世界系）。"""
    saved = _probe_set_trunk_qpos(world, trunk_q)
    pts: List[np.ndarray] = []
    try:
        for lname in _PILLAR_LINKS:
            link = world.robot.links.get(lname)
            if link is None:
                continue
            try:
                lo, hi = link.aabb
                pts.append(_aabb_corners_world(_to_np(lo), _to_np(hi)))
            except Exception:
                p = _link_world_pos(world, lname)
                if p is not None:
                    pts.append(p.reshape(1, 3))
        j4 = _link_world_pos(world, "torso_link4")
        if j4 is not None:
            pts.append(j4.reshape(1, 3))
    finally:
        try:
            world.robot.set_joint_positions(saved)
        except Exception:
            pass
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(pts)


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _sweep_pillar_envelope(
    world,
    trunk_base_q: np.ndarray,
    *,
    q4_vals: Optional[np.ndarray] = None,
    n_q4: int = 25,
) -> np.ndarray:
    """扫 torso_joint4 全行程，合并支柱世界点云。"""
    tbase = np.asarray(trunk_base_q, dtype=np.float64).reshape(4).copy()
    if q4_vals is None:
        q4_vals = np.linspace(_R1PRO_TRUNK_Q4_RANGE[0], _R1PRO_TRUNK_Q4_RANGE[1], int(n_q4))
    chunks: List[np.ndarray] = []
    for q4 in q4_vals:
        tq = tbase.copy()
        tq[3] = float(q4)
        pts = _collect_pillar_world_points(world, tq)
        if len(pts):
            chunks.append(pts)
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(chunks)


def _horiz_axes_from_chest(world) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """胸口原点、forward 水平轴、lateral 水平轴。"""
    chest, fwd = _chest_forward_axis(world)
    lat = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
    ln = float(np.linalg.norm(lat))
    if ln < 1e-9:
        lat = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        ln = float(np.linalg.norm(lat))
    if ln > 1e-9:
        lat = lat / ln
    return chest, fwd, lat


def _world_from_chest_horiz(
    world, fwd_m: float, lat_m: float, z_m: float,
) -> np.ndarray:
    chest, fwd, lat = _horiz_axes_from_chest(world)
    p = chest + fwd * float(fwd_m) + lat * float(lat_m)
    p[2] = float(z_m)
    return p


def _estimate_pillar_forbidden_horiz(
    world,
    pillar_pts_world: np.ndarray,
    *,
    z_ref_m: float,
    fwd_range: Tuple[float, float] = (0.0, 0.28),
    lat_range: Tuple[float, float] = (-0.22, 0.22),
    grid_n: int = 42,
    clearance_m: float = _GRIPPER_PILLAR_SOFT_M,
) -> dict:
    """在胸口水平面 z=z_ref，网格估计夹爪中心禁入区（fwd,lat）。"""
    pillar = np.asarray(pillar_pts_world, dtype=np.float64).reshape(-1, 3)
    if len(pillar) < 1:
        return {"forbidden": [], "safe_lat_min_left": None}
    fwd_vals = np.linspace(fwd_range[0], fwd_range[1], int(grid_n))
    lat_vals = np.linspace(lat_range[0], lat_range[1], int(grid_n))
    forbidden: List[List[float]] = []
    for f in fwd_vals:
        for lat in lat_vals:
            p = _world_from_chest_horiz(world, f, lat, z_ref_m)
            dmin = float(np.min(np.linalg.norm(pillar - p, axis=1)))
            if dmin < float(clearance_m):
                forbidden.append([float(f), float(lat), round(dmin * 100, 2)])
    # 左臂：fwd∈[0.05,0.22] 时 lateral 需 > 安全下限（远离柱体）
    lat_safe = None
    band = [r for r in forbidden if 0.05 <= r[0] <= 0.22]
    if band:
        lat_safe = max(float(r[1]) for r in band if r[1] > 0.0)
        lat_safe = float(lat_safe) + float(clearance_m) * 0.35
    return {
        "forbidden": forbidden,
        "safe_lat_min_left_m": lat_safe,
        "z_ref_m": float(z_ref_m),
        "clearance_m": float(clearance_m),
    }


def _overlay_pillar_zone_gta(
    rgb: np.ndarray,
    world,
    pillar_pts_world: np.ndarray,
    forbidden: Sequence[Sequence[float]],
    *,
    z_ref_m: float,
    eef_world: Optional[np.ndarray] = None,
) -> np.ndarray:
    """GTA 图上红色标出支柱扫掠包络与禁入网格。"""
    import cv2

    meta = _gta_camera_meta(world)
    if meta is None:
        return rgb
    cam_pos = meta["cam_pos"]
    cam_quat = meta["cam_quat"]
    w, h = meta["w"], meta["h"]
    fl, ha = meta["focal_length"], meta["horizontal_aperture"]
    out = rgb.copy()
    overlay = out.copy()

    # 支柱点云：红色小圆点
    if len(pillar_pts_world) > 0:
        uv, ok = _project_world_to_uv(pillar_pts_world, cam_pos, cam_quat, w, h, fl, ha)
        for i in np.where(ok)[0]:
            u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(overlay, (u, v), 3, (255, 40, 40), -1, lineType=cv2.LINE_AA)

    # 禁入区：fwd-lat 网格抬升到 z_ref 的方块投影
    if forbidden:
        step = max(1, len(forbidden) // 800)
        for row in forbidden[::step]:
            p = _world_from_chest_horiz(world, row[0], row[1], z_ref_m)
            uv1, ok1 = _project_world_to_uv(p.reshape(1, 3), cam_pos, cam_quat, w, h, fl, ha)
            if not ok1[0]:
                continue
            u, v = int(round(uv1[0, 0])), int(round(uv1[0, 1]))
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(overlay, (u, v), 5, (220, 20, 20), -1, lineType=cv2.LINE_AA)

    # 当前夹爪：绿色
    if eef_world is not None:
        uv_e, ok_e = _project_world_to_uv(
            np.asarray(eef_world).reshape(1, 3), cam_pos, cam_quat, w, h, fl, ha,
        )
        if ok_e[0]:
            u, v = int(round(uv_e[0, 0])), int(round(uv_e[0, 1]))
            cv2.drawMarker(
                overlay, (u, v), (40, 220, 80), cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA,
            )

    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    return out


def _waist_joint3_axis_center_world(world) -> Optional[np.ndarray]:
    """腰部第二高关节 torso_joint3 转轴中心（torso_link3 局部原点）。"""
    pts = _link_local_to_world(
        world, "torso_link3", _TRUNK_J3_PIVOT_IN_LINK3.reshape(1, 3),
    )
    if len(pts) < 1:
        return None
    return np.asarray(pts[0], dtype=np.float64).reshape(3)


def _collect_link_aabb_points_world(
    world,
    link_name: str,
    *,
    local_z_filter: Optional[Tuple[float, float]] = None,
    local_z_max: Optional[float] = None,
    local_y_abs_max: Optional[float] = None,
    grid_n: int = 6,
) -> np.ndarray:
    """link 碰撞 AABB 密集体素采样（世界系）；可选 link 局部 y/z 过滤。"""
    link = world.robot.links.get(link_name)
    if link is None:
        return np.zeros((0, 3), dtype=np.float64)
    try:
        lo, hi = link.aabb
        lo = _to_np(lo)
        hi = _to_np(hi)
    except Exception:
        p = _link_world_pos(world, link_name)
        return p.reshape(1, 3) if p is not None else np.zeros((0, 3), dtype=np.float64)
    gn = max(3, int(grid_n))
    xs = np.linspace(lo[0], hi[0], gn)
    ys = np.linspace(lo[1], hi[1], gn)
    zs = np.linspace(lo[2], hi[2], gn)
    grid = np.array(
        [[x, y, z] for x in xs for y in ys for z in zs],
        dtype=np.float64,
    )
    pts = np.vstack([_aabb_corners_world(lo, hi), grid])
    kept: List[np.ndarray] = []
    for p in pts:
        loc = _pos_in_link_frame(world, link_name, p)
        if loc is None:
            continue
        ly, lz = float(loc[1]), float(loc[2])
        if local_y_abs_max is not None and abs(ly) > float(local_y_abs_max) + 1e-4:
            continue
        if local_z_filter is not None:
            if lz < float(local_z_filter[0]) - 1e-4 or lz > float(local_z_filter[1]) + 1e-4:
                continue
        if local_z_max is not None and lz > float(local_z_max) + 1e-4:
            continue
        kept.append(np.asarray(p, dtype=np.float64).reshape(3))
    if not kept:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(kept)


def _clip_waist_j34_world_z(
    pts: np.ndarray,
    j3: np.ndarray,
    j4: np.ndarray,
) -> np.ndarray:
    """按 j3/j4 枢轴世界高度裁剪，保留两颗电机整壳（非仅 10cm 轴间距）。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 1:
        return pts
    z_lo = float(j3[2]) - float(_WAIST_J34_Z_MARGIN_BELOW_M)
    z_hi = float(j4[2]) + float(_WAIST_J34_Z_MARGIN_ABOVE_M)
    m = (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
    return pts[m] if np.any(m) else pts


def _collect_waist_j34_segment_points(world, trunk_q: np.ndarray) -> np.ndarray:
    """胸廓下 j3↔j4 腰段：torso_link3 全宽 + torso_link4 腰台（排除胸廓上盖）。"""
    saved = _probe_set_trunk_qpos(world, trunk_q)
    chunks: List[np.ndarray] = []
    merged = np.zeros((0, 3), dtype=np.float64)
    try:
        j3 = _waist_joint3_axis_center_world(world)
        j4 = _waist_top_joint4_axis_center_world(world)
        # link3：下层电机 + 侧立柱（局部 z 带）
        ycap = float(_WAIST_LINK_LOCAL_Y_ABS_MAX)
        l3 = _collect_link_aabb_points_world(
            world, "torso_link3",
            grid_n=6,
            local_z_filter=_WAIST_LINK3_LOCAL_Z,
            local_y_abs_max=ycap,
        )
        if len(l3):
            chunks.append(l3)
        # link4：上层电机 + 腰台；local z 以上接胸廓白盖板
        l4 = _collect_link_aabb_points_world(
            world, "torso_link4",
            grid_n=6,
            local_z_max=_WAIST_LINK4_LOCAL_Z_MAX,
            local_y_abs_max=ycap,
        )
        if len(l4):
            chunks.append(l4)
        merged = np.vstack(chunks)
        if j3 is not None and j4 is not None:
            merged = _clip_waist_j34_world_z(merged, j3, j4)
    finally:
        try:
            world.robot.set_joint_positions(saved)
        except Exception:
            pass
    return merged if chunks else np.zeros((0, 3), dtype=np.float64)


def _sweep_waist_j34_envelope(
    world,
    trunk_base_q: np.ndarray,
    *,
    q4_vals: Optional[np.ndarray] = None,
    n_q4: int = 25,
) -> np.ndarray:
    """扫 torso_joint4 全行程，合并 j3↔j4 腰段世界点云。"""
    tbase = np.asarray(trunk_base_q, dtype=np.float64).reshape(4).copy()
    if q4_vals is None:
        q4_vals = np.linspace(_R1PRO_TRUNK_Q4_RANGE[0], _R1PRO_TRUNK_Q4_RANGE[1], int(n_q4))
    chunks: List[np.ndarray] = []
    for q4 in q4_vals:
        tq = tbase.copy()
        tq[3] = float(q4)
        pts = _collect_waist_j34_segment_points(world, tq)
        if len(pts):
            chunks.append(pts)
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(chunks)


def _chest_flz_from_world(
    world, pts_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """世界点 → 胸口系 (forward, lateral, world_z)。"""
    chest, fwd = _chest_forward_axis(world)
    lat_dir = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
    ln = float(np.linalg.norm(lat_dir))
    if ln < 1e-9:
        lat_dir = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
        ln = float(np.linalg.norm(lat_dir))
    if ln > 1e-9:
        lat_dir = lat_dir / ln
    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    rel = pts - chest.reshape(1, 3)
    fwd_c = rel @ fwd
    lat_c = rel @ lat_dir
    zz = pts[:, 2]
    return fwd_c, lat_c, zz


def _robot_base_axis_world(world, local_axis: np.ndarray) -> Optional[np.ndarray]:
    """机器人 base 局部轴方向 → 世界系单位向量。"""
    try:
        _bpos, base_quat = world.robot.get_position_orientation()
        R = _quat_xyzw_to_rot(np.asarray(base_quat, dtype=np.float64))
        v = R @ np.asarray(local_axis, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else None
    except Exception:
        return None


def _waist_j34_link_lateral_axis(world, link_name: str, up: np.ndarray) -> Optional[np.ndarray]:
    """link 局部 +y 在 ⊥up 平面内的单位方向。"""
    p0 = _link_local_to_world(world, link_name, np.array([[0.0, 0.0, 0.0]]))
    p1 = _link_local_to_world(world, link_name, np.array([[0.0, 1.0, 0.0]]))
    if len(p0) < 1 or len(p1) < 1:
        return None
    lat = np.asarray(p1[0] - p0[0], dtype=np.float64).reshape(3)
    lat = lat - up * float(np.dot(lat, up))
    ln = float(np.linalg.norm(lat))
    if ln < 1e-6:
        return None
    return lat / ln


def _module_j34_axes_world(world) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """腰 j3↔j4 模块系：原点=轴线中点，up= j3→j4，lat=link4 局部 y，fwd=矢状面前向。"""
    j3 = _waist_joint3_axis_center_world(world)
    j4 = _waist_top_joint4_axis_center_world(world)
    if j3 is None or j4 is None:
        return None
    j3a = np.asarray(j3, dtype=np.float64)
    j4a = np.asarray(j4, dtype=np.float64)
    origin = 0.5 * (j3a + j4a)
    up = j4a - j3a
    up_n = float(np.linalg.norm(up))
    if up_n < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up = up / up_n
    # j3/j4 枢轴在 torso_link3 的 y=0 对称面 → 宽向用 link3 的 +y
    lat = _waist_j34_link_lateral_axis(world, "torso_link3", up)
    if lat is None:
        base_lat = _robot_base_axis_world(world, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        if base_lat is None:
            _, _cf, chest_lat = _horiz_axes_from_chest(world)
            base_lat = chest_lat
        lat = base_lat - up * float(np.dot(base_lat, up))
        ln = float(np.linalg.norm(lat))
        if ln < 1e-6:
            return None
        lat = lat / ln
    fwd_h = np.cross(lat, up)
    fwd_h = fwd_h / max(float(np.linalg.norm(fwd_h)), 1e-9)
    _, chest_fwd, _ = _horiz_axes_from_chest(world)
    if float(np.dot(fwd_h, chest_fwd)) < 0.0:
        fwd_h = -fwd_h
        lat = -lat
    return origin, fwd_h, lat, up


def _collect_j4_waist_shell_points(world, trunk_q: np.ndarray) -> np.ndarray:
    """近胸廓 torso_link4 腰台壳体采样（排除肩台/胸廓上盖）。"""
    saved = _probe_set_trunk_qpos(world, trunk_q)
    try:
        return _collect_link_aabb_points_world(
            world, "torso_link4",
            grid_n=8,
            local_z_max=_WAIST_LINK4_LOCAL_Z_MAX,
            local_y_abs_max=float(_WAIST_LINK_LOCAL_Y_ABS_MAX),
        )
    finally:
        try:
            world.robot.set_joint_positions(saved)
        except Exception:
            pass


def _measure_j4_shell_width_m(world, trunk_q: np.ndarray) -> Optional[float]:
    """j4 腰台壳体宽：link4 局部 |y| 最大半宽 ×2（关于 y=0 对称）。"""
    saved = _probe_set_trunk_qpos(world, trunk_q)
    try:
        pts = _collect_link_aabb_points_world(
            world, "torso_link4",
            grid_n=8,
            local_z_max=_WAIST_LINK4_LOCAL_Z_MAX,
            local_y_abs_max=float(_WAIST_J34_SHELL_Y_ABS_MAX),
        )
    finally:
        try:
            world.robot.set_joint_positions(saved)
        except Exception:
            pass
    if len(pts) < 1:
        return None
    ys_abs: List[float] = []
    for p in pts:
        loc = _pos_in_link_frame(world, "torso_link4", p)
        if loc is not None:
            ys_abs.append(abs(float(loc[1])))
    if not ys_abs:
        return None
    return float(2.0 * max(ys_abs))


def _waist_j34_axis_symmetric_box(
    world,
    trunk_q: np.ndarray,
    *,
    length_extend_factor: float = 1.0,
) -> dict:
    """以 j3→j4 中轴为对称轴构盒：宽=j4 腰台壳；高=轴段+下移半高；禁入前后各 N×L 且包络本体。"""
    w_j4 = _measure_j4_shell_width_m(world, trunk_q)
    seg_pts = _collect_waist_j34_segment_points(world, trunk_q)
    axes = _module_j34_axes_world(world)
    j3 = _waist_joint3_axis_center_world(world)
    j4 = _waist_top_joint4_axis_center_world(world)
    if axes is None or j3 is None or j4 is None or w_j4 is None or w_j4 < 1e-4:
        return {"measure": None, "body_box": None, "forbidden_box": None}
    origin, fwd_h, lat, up = axes
    j3a = np.asarray(j3, dtype=np.float64)
    j4a = np.asarray(j4, dtype=np.float64)
    h_j3 = float(np.dot(j3a - origin, up))
    h_j4 = float(np.dot(j4a - origin, up))
    h_lo = min(h_j3, h_j4) - float(_WAIST_J34_Z_MARGIN_BELOW_M)
    h_hi = max(h_j3, h_j4) + float(_WAIST_J34_Z_MARGIN_ABOVE_M)
    height_m = h_hi - h_lo
    dh = -float(_WAIST_J34_FORBIDDEN_SHIFT_DOWN_H_FRAC) * height_m
    h_lo += dh
    h_hi += dh
    width_m = float(w_j4) * float(_WAIST_J34_FORBIDDEN_WIDTH_SCALE)
    half_w = 0.5 * width_m
    # j3↔j4 轴线为横向中线：向两侧各 W/2（轴线过盒子中心，不是侧面）
    axis_origin = 0.5 * (j3a + j4a)
    origin = axis_origin
    lat_lo, lat_hi = -half_w, half_w
    # 腰段 j3↔j4 点云沿 forward 包络（中轴带内全深度，贴本体）
    f_shell_lo, f_shell_hi = 0.0, 0.12
    strip_shell: List[np.ndarray] = []
    for p in np.asarray(seg_pts, dtype=np.float64).reshape(-1, 3):
        if float(np.abs((p - origin) @ lat)) > half_w + 1e-3:
            continue
        strip_shell.append(p)
    if strip_shell:
        rel_s = np.asarray(strip_shell, dtype=np.float64).reshape(-1, 3) - origin.reshape(1, 3)
        f_seg = rel_s @ fwd_h
        f_shell_lo = float(np.min(f_seg))
        f_shell_hi = float(np.max(f_seg))
    length_m = max(f_shell_hi - f_shell_lo, 1e-4)
    height_m = h_hi - h_lo
    ext = max(0.0, float(length_extend_factor))
    ext_mul = ext * float(_WAIST_J34_FORBIDDEN_FWD_BACK_EXTEND_FACTOR)
    f_forbid_lo = f_shell_lo - length_m * ext_mul
    f_forbid_hi = f_shell_hi + length_m * ext_mul
    meas = {
        "frame": "module_j34_axis_sym",
        "f_shell_lo_m": round(f_shell_lo, 4),
        "f_shell_hi_m": round(f_shell_hi, 4),
        "f_forbid_lo_m": round(f_forbid_lo, 4),
        "f_forbid_hi_m": round(f_forbid_hi, 4),
        "f_min_m": round(f_shell_lo, 4),
        "f_max_m": round(f_shell_hi, 4),
        "lat_min_m": round(lat_lo, 4),
        "lat_max_m": round(lat_hi, 4),
        "h_min_m": round(h_lo, 4),
        "h_max_m": round(h_hi, 4),
        "length_m": round(length_m, 4),
        "width_m": round(width_m, 4),
        "height_m": round(height_m, 4),
        "length_cm": round(length_m * 100.0, 2),
        "width_cm": round(width_m * 100.0, 2),
        "height_cm": round(height_m * 100.0, 2),
        "width_source": "torso_link4_shell_local_y_sym",
        "width_scale": float(_WAIST_J34_FORBIDDEN_WIDTH_SCALE),
        "shift_down_h_frac": float(_WAIST_J34_FORBIDDEN_SHIFT_DOWN_H_FRAC),
        "fwd_back_extend_factor": float(_WAIST_J34_FORBIDDEN_FWD_BACK_EXTEND_FACTOR),
        "symmetry_axis": "j3_j4",
        "origin_world_m": origin.round(4).tolist(),
    }
    basis = (origin, fwd_h, lat, up)
    body_box = _chest_box_dict(
        f_shell_lo, f_shell_hi, lat_lo, lat_hi, h_lo, h_hi,
        module_frame=True, module_basis=basis,
    )
    forbidden_box = _chest_box_dict(
        f_forbid_lo, f_forbid_hi,
        lat_lo, lat_hi, h_lo, h_hi,
        module_frame=True, module_basis=basis,
    )
    return {
        "measure": meas,
        "body_box": body_box,
        "forbidden_box": forbidden_box,
        "length_extend_factor": ext,
        "fwd_back_extend_mul": ext_mul,
        "projection_length_cm": round((f_forbid_hi - f_forbid_lo) * 100.0, 2),
    }


def _chest_frame_aabb_from_points(
    world, pts_world: np.ndarray,
) -> Optional[dict]:
    """胸口系下点云 AABB（对照）。"""
    seg = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    if len(seg) < 1:
        return None
    f_seg, l_seg, z_seg = _chest_flz_from_world(world, seg)
    f_lo, f_hi = float(np.min(f_seg)), float(np.max(f_seg))
    lat_lo, lat_hi = float(np.min(l_seg)), float(np.max(l_seg))
    z_lo, z_hi = float(np.min(z_seg)), float(np.max(z_seg))
    length_m = f_hi - f_lo
    width_m = lat_hi - lat_lo
    height_m = z_hi - z_lo
    return {
        "f_min_m": round(f_lo, 4),
        "f_max_m": round(f_hi, 4),
        "lat_min_m": round(lat_lo, 4),
        "lat_max_m": round(lat_hi, 4),
        "z_min_m": round(z_lo, 4),
        "z_max_m": round(z_hi, 4),
        "length_m": round(length_m, 4),
        "width_m": round(width_m, 4),
        "height_m": round(height_m, 4),
        "length_cm": round(length_m * 100.0, 2),
        "width_cm": round(width_m * 100.0, 2),
        "height_cm": round(height_m * 100.0, 2),
        "frame": "chest",
    }


def _chest_box_dict(
    f_lo: float, f_hi: float,
    lat_lo: float, lat_hi: float,
    z_lo: float, z_hi: float,
    *,
    module_frame: bool = False,
    module_basis: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
) -> dict:
    out = {
        "f_lo_m": float(f_lo),
        "f_hi_m": float(f_hi),
        "lat_lo_m": float(lat_lo),
        "lat_hi_m": float(lat_hi),
        "z_lo_m": float(z_lo),
        "z_hi_m": float(z_hi),
        "_module_frame": bool(module_frame),
    }
    if module_basis is not None:
        o, fwd, lat, up = module_basis
        out["_mf_origin"] = np.asarray(o, dtype=np.float64).reshape(3).tolist()
        out["_mf_fwd"] = np.asarray(fwd, dtype=np.float64).reshape(3).tolist()
        out["_mf_lat"] = np.asarray(lat, dtype=np.float64).reshape(3).tolist()
        out["_mf_up"] = np.asarray(up, dtype=np.float64).reshape(3).tolist()
    return out


def _module_basis_from_box(
    world, box: dict,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if box.get("_mf_origin") is not None:
        return (
            np.asarray(box["_mf_origin"], dtype=np.float64).reshape(3),
            np.asarray(box["_mf_fwd"], dtype=np.float64).reshape(3),
            np.asarray(box["_mf_lat"], dtype=np.float64).reshape(3),
            np.asarray(box["_mf_up"], dtype=np.float64).reshape(3),
        )
    return _module_j34_axes_world(world)


def _waist_j34_body_and_forward_projection(
    world,
    trunk_q: np.ndarray,
    *,
    length_extend_factor: float = 1.0,
) -> dict:
    """兼容入口：见 _waist_j34_axis_symmetric_box。"""
    return _waist_j34_axis_symmetric_box(
        world, trunk_q,
        length_extend_factor=length_extend_factor,
    )


def _chest_box_faces_world(
    world,
    box: Optional[dict],
) -> List[np.ndarray]:
    """长方体 6 面（世界系）；支持胸口系或 j3↔j4 模块系。"""
    if not box:
        return []
    f0 = float(box["f_lo_m"])
    f1 = float(box["f_hi_m"])
    lat0, lat1 = float(box["lat_lo_m"]), float(box["lat_hi_m"])
    z0, z1 = float(box["z_lo_m"]), float(box["z_hi_m"])
    if box.get("_module_frame"):
        basis = _module_basis_from_box(world, box)
        if basis is None:
            return []
        origin, fwd_h, lat, up = basis

        def _p(f, la, hh):
            return origin + fwd_h * f + lat * la + up * hh

        return [
            np.vstack([_p(f0, lat0, z0), _p(f0, lat1, z0), _p(f0, lat1, z1), _p(f0, lat0, z1)]),
            np.vstack([_p(f1, lat0, z0), _p(f1, lat1, z0), _p(f1, lat1, z1), _p(f1, lat0, z1)]),
            np.vstack([_p(f0, lat0, z0), _p(f1, lat0, z0), _p(f1, lat0, z1), _p(f0, lat0, z1)]),
            np.vstack([_p(f0, lat1, z0), _p(f1, lat1, z0), _p(f1, lat1, z1), _p(f0, lat1, z1)]),
            np.vstack([_p(f0, lat0, z0), _p(f1, lat0, z0), _p(f1, lat1, z0), _p(f0, lat1, z0)]),
            np.vstack([_p(f0, lat0, z1), _p(f1, lat0, z1), _p(f1, lat1, z1), _p(f0, lat1, z1)]),
        ]
    corners: Dict[Tuple[float, float, float], np.ndarray] = {}
    for f in (f0, f1):
        for lat in (lat0, lat1):
            for z in (z0, z1):
                corners[(f, lat, z)] = _world_from_chest_horiz(world, f, lat, z)
    def _q(f, la, z):
        return corners[(f, la, z)]
    return [
        np.vstack([_q(f0, lat0, z0), _q(f0, lat1, z0), _q(f0, lat1, z1), _q(f0, lat0, z1)]),
        np.vstack([_q(f1, lat0, z0), _q(f1, lat1, z0), _q(f1, lat1, z1), _q(f1, lat0, z1)]),
        np.vstack([_q(f0, lat0, z0), _q(f1, lat0, z0), _q(f1, lat0, z1), _q(f0, lat0, z1)]),
        np.vstack([_q(f0, lat1, z0), _q(f1, lat1, z0), _q(f1, lat1, z1), _q(f0, lat1, z1)]),
        np.vstack([_q(f0, lat0, z0), _q(f1, lat0, z0), _q(f1, lat1, z0), _q(f0, lat1, z0)]),
        np.vstack([_q(f0, lat0, z1), _q(f1, lat0, z1), _q(f1, lat1, z1), _q(f0, lat1, z1)]),
    ]


def _draw_chest_box_on_gta(
    overlay: np.ndarray,
    world,
    box: Optional[dict],
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    *,
    fill_rgb: Tuple[int, int, int],
    edge_rgb: Tuple[int, int, int],
    fill_alpha: float = 0.42,
) -> None:
    """在 RGB overlay 上绘制长方体（面填充 + 棱边）。"""
    import cv2

    if not box:
        return
    faces = _chest_box_faces_world(world, box)
    # 按相机深度排序，先画远面
    face_depth: List[Tuple[float, np.ndarray]] = []
    for face in faces:
        ctr = np.mean(face, axis=0)
        uv, ok = _project_world_to_uv(ctr.reshape(1, 3), cam_pos, cam_quat, w, h, fl, ha)
        if ok[0]:
            face_depth.append((float(ctr[2]), face))
    face_depth.sort(key=lambda x: x[0])
    for _, face in face_depth:
        uv, ok = _project_world_to_uv(face, cam_pos, cam_quat, w, h, fl, ha)
        if int(np.sum(ok)) < 3:
            continue
        poly = np.round(uv[ok]).astype(np.int32).reshape(-1, 1, 2)
        if fill_alpha > 1e-6:
            layer = overlay.copy()
            cv2.fillPoly(layer, [poly], fill_rgb, lineType=cv2.LINE_AA)
            cv2.addWeighted(layer, fill_alpha, overlay, 1.0 - fill_alpha, 0, overlay)
        cv2.polylines(overlay, [poly], True, edge_rgb, 2, cv2.LINE_AA)
    # 12 条棱
    f0, f1 = float(box["f_lo_m"]), float(box["f_hi_m"])
    lat0, lat1 = float(box["lat_lo_m"]), float(box["lat_hi_m"])
    z0, z1 = float(box["z_lo_m"]), float(box["z_hi_m"])
    if box.get("_module_frame"):
        basis = _module_basis_from_box(world, box)
        if basis is None:
            return
        origin, fwd_h, lat_ax, up = basis
        corners = [
            origin + fwd_h * f + lat_ax * la + up * hh
            for f in (f0, f1) for la in (lat0, lat1) for hh in (z0, z1)
        ]
        corners = np.asarray(corners, dtype=np.float64)
    else:
        corners = np.asarray([
            _world_from_chest_horiz(world, f, lat, z)
            for f in (f0, f1) for lat in (lat0, lat1) for z in (z0, z1)
        ], dtype=np.float64)
    uv, ok = _project_world_to_uv(corners, cam_pos, cam_quat, w, h, fl, ha)
    edges = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ]
    for i, j in edges:
        if not (ok[i] and ok[j]):
            continue
        p0 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        p1 = (int(round(uv[j, 0])), int(round(uv[j, 1])))
        cv2.line(overlay, p0, p1, edge_rgb, 2, cv2.LINE_AA)


def _draw_module_box_f_plane_on_gta(
    overlay: np.ndarray,
    world,
    box: dict,
    f_m: float,
    cam_pos: np.ndarray,
    cam_quat: np.ndarray,
    w: int,
    h: int,
    fl: float,
    ha: float,
    *,
    fill_rgb: Tuple[int, int, int] = (70, 140, 255),
    edge_rgb: Tuple[int, int, int] = (30, 100, 230),
    fill_alpha: float = 0.58,
) -> None:
    """模块系下在固定 forward 坐标处绘制矩形截面（lat×up）。"""
    import cv2

    basis = _module_basis_from_box(world, box)
    if basis is None:
        return
    origin, fwd_h, lat_ax, up = basis
    lat0 = float(box["lat_lo_m"])
    lat1 = float(box["lat_hi_m"])
    z0 = float(box["z_lo_m"])
    z1 = float(box["z_hi_m"])
    quad = np.vstack([
        origin + fwd_h * float(f_m) + lat_ax * lat0 + up * z0,
        origin + fwd_h * float(f_m) + lat_ax * lat1 + up * z0,
        origin + fwd_h * float(f_m) + lat_ax * lat1 + up * z1,
        origin + fwd_h * float(f_m) + lat_ax * lat0 + up * z1,
    ])
    uv, ok = _project_world_to_uv(quad, cam_pos, cam_quat, w, h, fl, ha)
    if int(np.sum(ok)) < 3:
        return
    poly = np.round(uv[ok]).astype(np.int32).reshape(-1, 1, 2)
    layer = overlay.copy()
    cv2.fillPoly(layer, [poly], fill_rgb, lineType=cv2.LINE_AA)
    cv2.addWeighted(layer, fill_alpha, overlay, 1.0 - fill_alpha, 0, overlay)
    cv2.polylines(overlay, [poly], True, edge_rgb, 3, cv2.LINE_AA)
    ctr = np.mean(quad, axis=0).reshape(1, 3)
    uv_c, ok_c = _project_world_to_uv(ctr, cam_pos, cam_quat, w, h, fl, ha)
    if ok_c[0]:
        u, v = int(round(uv_c[0, 0])), int(round(uv_c[0, 1]))
        cv2.drawMarker(
            overlay, (u, v), edge_rgb, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA,
        )


def _waist_j34_body_mid_f_m(
    forbidden_box: Optional[dict],
    measure: Optional[dict],
) -> Optional[float]:
    """禁飞区与本体相交段沿 forward 的中点（对称延伸时等于整盒长度中点）。"""
    m = measure or {}
    if m.get("f_shell_lo_m") is not None and m.get("f_shell_hi_m") is not None:
        return 0.5 * (float(m["f_shell_lo_m"]) + float(m["f_shell_hi_m"]))
    if forbidden_box:
        return 0.5 * (float(forbidden_box["f_lo_m"]) + float(forbidden_box["f_hi_m"]))
    return None


def _overlay_waist_j34_forbidden_gta(
    rgb: np.ndarray,
    world,
    *,
    body_box: Optional[dict] = None,
    forbidden_box: Optional[dict] = None,
    measure: Optional[dict] = None,
    compact: bool = False,
) -> np.ndarray:
    """GTA：浅红实心=禁入区（腰台包络 + 前后各 N×L）。"""
    import cv2

    meta = _gta_camera_meta(world)
    if meta is None:
        return rgb
    cam_pos = meta["cam_pos"]
    cam_quat = meta["cam_quat"]
    w, h = meta["w"], meta["h"]
    fl, ha = meta["focal_length"], meta["horizontal_aperture"]
    out = rgb.copy()
    overlay = out.copy()

    # j3↔j4 中轴线（黄线，核对左右对称）
    j3 = _waist_joint3_axis_center_world(world)
    j4 = _waist_top_joint4_axis_center_world(world)
    if j3 is not None and j4 is not None:
        seg = np.vstack([j3, j4])
        uv_ax, ok_ax = _project_world_to_uv(seg, cam_pos, cam_quat, w, h, fl, ha)
        if ok_ax.all():
            p0 = (int(round(uv_ax[0, 0])), int(round(uv_ax[0, 1])))
            p1 = (int(round(uv_ax[1, 0])), int(round(uv_ax[1, 1])))
            cv2.line(overlay, p0, p1, (255, 230, 60), 3, cv2.LINE_AA)
        m = measure or {}
        w_m = float(m.get("width_m") or 0.0)
        basis = _module_basis_from_box(world, forbidden_box or body_box or {})
        if basis is not None and w_m > 1e-4:
            origin, fwd_h, lat, up = basis
            mid = 0.5 * (np.asarray(j3, dtype=np.float64) + np.asarray(j4, dtype=np.float64))
            hw = 0.5 * w_m
            h_mid = 0.5 * (float(m.get("h_min_m") or 0.0) + float(m.get("h_max_m") or 0.0))
            axis_ctr = origin + up * h_mid
            lat_marks = np.vstack([axis_ctr - lat * hw, axis_ctr + lat * hw])
            uv_lm, ok_lm = _project_world_to_uv(lat_marks, cam_pos, cam_quat, w, h, fl, ha)
            for i in range(2):
                if ok_lm[i]:
                    u, v = int(round(uv_lm[i, 0])), int(round(uv_lm[i, 1]))
                    cv2.circle(overlay, (u, v), 6, (60, 200, 255), -1, lineType=cv2.LINE_AA)
            uv_c, ok_c = _project_world_to_uv(axis_ctr.reshape(1, 3), cam_pos, cam_quat, w, h, fl, ha)
            if ok_c[0]:
                u, v = int(round(uv_c[0, 0])), int(round(uv_c[0, 1]))
                cv2.drawMarker(
                    overlay, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA,
                )
            # 禁入区前缘：绿线标出轴线两侧各 W/2（核对中线对称）
            if forbidden_box:
                f_end = float(forbidden_box.get("f_hi_m", 0.0))
                h0 = float(forbidden_box.get("z_lo_m", 0.0))
                h1 = float(forbidden_box.get("z_hi_m", 0.0))
                hh = 0.5 * (h0 + h1)
                p_l = axis_ctr - lat * hw + fwd_h * f_end
                p_r = axis_ctr + lat * hw + fwd_h * f_end
                uv_w, ok_w = _project_world_to_uv(
                    np.vstack([p_l, p_r]), cam_pos, cam_quat, w, h, fl, ha,
                )
                if ok_w.all():
                    cv2.line(
                        overlay,
                        (int(round(uv_w[0, 0])), int(round(uv_w[0, 1]))),
                        (int(round(uv_w[1, 0])), int(round(uv_w[1, 1]))),
                        (80, 255, 120), 3, cv2.LINE_AA,
                    )

    _draw_chest_box_on_gta(
        overlay, world, forbidden_box, cam_pos, cam_quat, w, h, fl, ha,
        fill_rgb=(255, 90, 90), edge_rgb=(255, 60, 60),
        fill_alpha=0.45,
    )
    # 两侧竖棱（lat=±W/2）加亮，避免只看到轴一侧的填充
    if forbidden_box:
        basis = _module_basis_from_box(world, forbidden_box)
        m = measure or {}
        hw = 0.5 * float(m.get("width_m") or 0.0)
        if basis is not None and hw > 1e-4:
            o, fwd_h, lat, up = basis
            f0 = float(forbidden_box["f_lo_m"])
            f1 = float(forbidden_box["f_hi_m"])
            z0 = float(forbidden_box["z_lo_m"])
            z1 = float(forbidden_box["z_hi_m"])
            for la in (-hw, hw):
                seg = np.vstack([
                    o + fwd_h * f0 + lat * la + up * z0,
                    o + fwd_h * f1 + lat * la + up * z0,
                    o + fwd_h * f1 + lat * la + up * z1,
                    o + fwd_h * f0 + lat * la + up * z1,
                ])
                uv_s, ok_s = _project_world_to_uv(seg, cam_pos, cam_quat, w, h, fl, ha)
                for i in range(4):
                    j = (i + 1) % 4
                    if ok_s[i] and ok_s[j]:
                        cv2.line(
                            overlay,
                            (int(round(uv_s[i, 0])), int(round(uv_s[i, 1]))),
                            (int(round(uv_s[j, 0])), int(round(uv_s[j, 1]))),
                            (255, 220, 60), 2, cv2.LINE_AA,
                        )

    # 本体相交段长度中点：蓝色截面（垂直于 forward，lat×H）
    f_mid = _waist_j34_body_mid_f_m(forbidden_box, measure)
    if forbidden_box is not None and f_mid is not None:
        _draw_module_box_f_plane_on_gta(
            overlay, world, forbidden_box, f_mid,
            cam_pos, cam_quat, w, h, fl, ha,
        )

    cv2.addWeighted(overlay, 0.62, out, 0.38, 0, out)
    if not compact:
        legend = out.copy()
        cv2.rectangle(legend, (12, 12), (620, 118), (20, 20, 20), -1)
        cv2.addWeighted(legend, 0.65, out, 0.35, 0, out)
        m = measure or {}
        size_txt = (
            f"L={m.get('length_cm')} W={m.get('width_cm')} H={m.get('height_cm')} cm"
            if m else "L/W/H n/a"
        )
        cv2.putText(
            out, f"light red: forbid +/-{int(_WAIST_J34_FORBIDDEN_FWD_BACK_EXTEND_FACTOR)}xL ({size_txt})",
            (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 100, 100), 1, cv2.LINE_AA,
        )
        cv2.putText(
            out, "blue plane=body intersect mid (length center)",
            (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 180, 255), 1, cv2.LINE_AA,
        )
        cv2.putText(
            out, "yellow=j3-j4 | white cross=axis | cyan=+-W/2 | green=width",
            (22, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA,
        )
    return out


def _gta_cam_pose_save(world) -> Optional[dict]:
    """保存 gta_view 外参以便三视图后恢复。"""
    meta = _gta_camera_meta(world)
    if meta is None:
        return None
    return {
        "cam_pos": meta["cam_pos"].copy(),
        "cam_quat": meta["cam_quat"].copy(),
    }


def _gta_cam_pose_restore(world, pose: Optional[dict]) -> None:
    if not pose:
        return
    try:
        import torch as th

        gta = world.env._external_sensors.get("gta_view")
        if gta is None:
            return
        gta.set_position_orientation(
            position=th.tensor(pose["cam_pos"], dtype=th.float32),
            orientation=th.tensor(
                np.asarray(pose["cam_quat"], dtype=np.float64).reshape(4),
                dtype=th.float32,
            ),
        )
    except Exception:
        pass


def _gta_render_flush(world, n: int = 5) -> None:
    try:
        import omnigibson as og
        for _ in range(max(1, int(n))):
            og.sim.render()
    except Exception:
        pass


def _waist_j34_trimetric_focus_world(
    world,
    forbidden_box: Optional[dict],
    measure: Optional[dict],
) -> Optional[np.ndarray]:
    """三视图注视点：腰模块几何中心。"""
    basis = _module_basis_from_box(world, forbidden_box or {})
    if basis is None:
        return None
    origin, _fwd, _lat, up = basis
    m = measure or {}
    h0 = float(m.get("h_min_m") or 0.0)
    h1 = float(m.get("h_max_m") or 0.0)
    return origin + up * (0.5 * (h0 + h1))


def _waist_j34_trimetric_cameras(
    focus: np.ndarray,
    fwd: np.ndarray,
    lat: np.ndarray,
    up: np.ndarray,
    *,
    dist_front: float = 2.0,
    dist_top: float = 2.6,
    dist_side: float = 2.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """正视(+fwd 看向腰)、俯视(+up)、侧视(-lat)。"""
    focus = np.asarray(focus, dtype=np.float64).reshape(3)
    fwd = np.asarray(fwd, dtype=np.float64).reshape(3)
    lat = np.asarray(lat, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    return {
        "front": (focus + fwd * float(dist_front), focus.copy()),
        "top": (focus + up * float(dist_top), focus.copy()),
        "side": (focus - lat * float(dist_side), focus.copy()),
    }


def _annotate_trimetric_panel(
    rgb: np.ndarray,
    title: str,
    subtitle: str = "",
) -> np.ndarray:
    import cv2

    out = rgb.copy()
    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 420), 56 if subtitle else 36), (16, 16, 16), -1)
    cv2.addWeighted(out, 0.55, rgb, 0.45, 0, out)
    cv2.putText(
        out, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            out, subtitle, (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA,
        )
    return out


def _render_waist_j34_forbidden_oblique_gta(
    world,
    *,
    body_box: Optional[dict],
    forbidden_box: Optional[dict],
    measure: Optional[dict],
    compact: bool = False,
) -> Optional[np.ndarray]:
    """GTA 默认斜上视角（不移动相机）。"""
    _gta_render_flush(world, 4)
    rgb = _gta_rgb_array(world)
    if rgb is None:
        return None
    marked = _overlay_waist_j34_forbidden_gta(
        rgb, world,
        body_box=body_box,
        forbidden_box=forbidden_box,
        measure=measure,
        compact=compact,
    )
    return _annotate_trimetric_panel(
        marked, "斜上视 (GTA)", "默认外参 | 场景透视",
    )


def _stitch_oblique_over_trimetric(
    oblique: np.ndarray,
    trimetric_row: np.ndarray,
    *,
    oblique_h: int = 500,
    trimetric_h: int = 420,
    gap: int = 8,
) -> np.ndarray:
    """上：斜上透视；下：正视/俯视/侧视三视图。"""
    import cv2

    ob = np.asarray(oblique, dtype=np.uint8)
    tri = np.asarray(trimetric_row, dtype=np.uint8)
    if ob.ndim != 3 or tri.ndim != 3:
        return tri if tri.ndim == 3 else ob
    ow = max(1, int(round(ob.shape[1] * float(oblique_h) / max(ob.shape[0], 1))))
    ob_s = cv2.resize(ob, (ow, oblique_h), interpolation=cv2.INTER_AREA)
    tw = max(1, int(round(tri.shape[1] * float(trimetric_h) / max(tri.shape[0], 1))))
    tri_s = cv2.resize(tri, (tw, trimetric_h), interpolation=cv2.INTER_AREA)
    out_w = max(ob_s.shape[1], tri_s.shape[1])
    if ob_s.shape[1] < out_w:
        pad = np.zeros((oblique_h, out_w - ob_s.shape[1], 3), dtype=np.uint8)
        ob_s = np.hstack([ob_s, pad])
    if tri_s.shape[1] < out_w:
        pad = np.zeros((trimetric_h, out_w - tri_s.shape[1], 3), dtype=np.uint8)
        tri_s = np.hstack([tri_s, pad])
    sep = np.zeros((gap, out_w, 3), dtype=np.uint8)
    return np.vstack([ob_s, sep, tri_s])


def _stitch_trimetric_horizontal(
    panels: Sequence[Tuple[str, np.ndarray]],
    *,
    panel_h: int = 480,
    gap: int = 6,
) -> np.ndarray:
    import cv2

    if not panels:
        return np.zeros((panel_h, 640, 3), dtype=np.uint8)
    scaled: List[np.ndarray] = []
    for _title, img in panels:
        img = np.asarray(img, dtype=np.uint8)
        if img.ndim != 3:
            continue
        h0, w0 = img.shape[:2]
        panel_w = max(1, int(round(w0 * float(panel_h) / max(h0, 1))))
        scaled.append(cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA))
    if not scaled:
        return np.zeros((panel_h, 640, 3), dtype=np.uint8)
    total_w = sum(im.shape[1] for im in scaled) + gap * (len(scaled) - 1)
    out = np.zeros((panel_h, total_w, 3), dtype=np.uint8)
    x = 0
    for im in scaled:
        out[:, x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    return out


def _render_waist_j34_forbidden_trimetric(
    world,
    *,
    body_box: Optional[dict],
    forbidden_box: Optional[dict],
    measure: Optional[dict],
    dist_front: float = 2.0,
    dist_top: float = 2.6,
    dist_side: float = 2.0,
) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    """GTA 三视角：机器人 + 禁入区；返回拼接图与各视角单帧。"""
    from behavior_interface.skills.viz_grasp_obj_multiview import _move_gta_cam

    basis = _module_basis_from_box(world, forbidden_box or body_box or {})
    focus = _waist_j34_trimetric_focus_world(world, forbidden_box, measure)
    if basis is None or focus is None:
        return None, {}
    origin, fwd, lat, up = basis
    gta = world.env._external_sensors.get("gta_view")
    if gta is None:
        return None, {}

    saved_pose = _gta_cam_pose_save(world)
    view_labels = {
        "front": ("正视 (+fwd)", "lat×up | 宽×高"),
        "top": ("俯视 (+up)", "fwd×lat | 前×宽"),
        "side": ("侧视 (-lat)", "fwd×up | 前×高"),
    }
    cams = _waist_j34_trimetric_cameras(
        focus, fwd, lat, up,
        dist_front=dist_front, dist_top=dist_top, dist_side=dist_side,
    )
    singles: Dict[str, np.ndarray] = {}
    panel_list: List[Tuple[str, np.ndarray]] = []
    try:
        for vname in ("front", "top", "side"):
            cam_pos, look_at = cams[vname]
            _move_gta_cam(gta, cam_pos, look_at)
            _gta_render_flush(world, 6)
            rgb = _gta_rgb_array(world)
            if rgb is None:
                continue
            marked = _overlay_waist_j34_forbidden_gta(
                rgb, world,
                body_box=body_box,
                forbidden_box=forbidden_box,
                measure=measure,
                compact=True,
            )
            title, sub = view_labels.get(vname, (vname, ""))
            panel = _annotate_trimetric_panel(marked, title, sub)
            singles[vname] = panel
            panel_list.append((vname, panel))
    finally:
        _gta_cam_pose_restore(world, saved_pose)
        _gta_render_flush(world, 2)

    if not panel_list:
        return None, singles
    return _stitch_trimetric_horizontal(panel_list), singles


def _eef_pillar_distance_m(
    world, arm: str, arm_q: np.ndarray, pillar_pts_world: np.ndarray,
) -> Optional[float]:
    """FK 探针：夹爪到支柱点云最近距离。"""
    epos = _probe_eef_pos_at_arm_qpos(world, arm, arm_q)
    pillar = np.asarray(pillar_pts_world, dtype=np.float64).reshape(-1, 3)
    if epos is None or len(pillar) < 1:
        return None
    return float(np.min(np.linalg.norm(pillar - epos.reshape(1, 3), axis=1)))


def _phase2_point_penalty(
    world,
    arm: str,
    arm_q: np.ndarray,
    *,
    max_forward_m: float,
    z_floor_m: Optional[float] = None,
    pillar_pts_world: Optional[np.ndarray] = None,
) -> float:
    """Phase2 单点软惩罚：forward 超标、Z 下降、靠腰柱过近。"""
    pen = 0.0
    fd = eef_chest_forward_dist_m(world, arm, arm_q)
    if fd is None:
        return 200.0
    over_plan = max(0.0, float(fd) - float(max_forward_m))
    over_exec = max(0.0, float(fd) - float(TUCK_EXEC_FORWARD_MAX_M))
    pen += over_plan * over_plan * 10.0 + over_exec * over_exec * 35.0
    zz = _eef_world_z_m(world, arm, arm_q)
    if z_floor_m is not None and zz is not None:
        z_short = max(0.0, float(z_floor_m) - float(zz))
        pen += z_short * z_short * 90.0
    if pillar_pts_world is not None and len(pillar_pts_world) > 0:
        d = _eef_pillar_distance_m(world, arm, arm_q, pillar_pts_world)
        if d is None:
            pen += 25.0
        else:
            short = max(0.0, float(_GRIPPER_PILLAR_SOFT_M) - float(d))
            pen += short * short * 55.0
    return float(pen)


def _segment_phase2_penalty(
    world,
    arm: str,
    qa: np.ndarray,
    qb: np.ndarray,
    *,
    max_forward_m: float,
    z_floor_m: Optional[float] = None,
    pillar_pts_world: Optional[np.ndarray] = None,
    samples: int = 8,
) -> float:
    """段内采样取最大软惩罚（不硬拒绝）。"""
    worst = 0.0
    for i in range(int(samples) + 1):
        t = i / float(max(samples, 1))
        q = (1.0 - t) * qa + t * qb
        worst = max(
            worst,
            _phase2_point_penalty(
                world, arm, q,
                max_forward_m=max_forward_m,
                z_floor_m=z_floor_m,
                pillar_pts_world=pillar_pts_world,
            ),
        )
    return float(worst)


def _lift_upper_forearm_signs(arm: str) -> tuple:
    """大臂(j3=index2) / 小臂(j4=index3) 垂直抬升协同方向。

  R1Pro 悬垂位 hang 下（对照旧 tuck 标定折线前段）：
    - 小臂向前 = 肘屈 j4 负向（两臂相同）
    - 大臂向后 = 左臂 j3 负向、右臂 j3 正向（左右镜像）
  协同比例 |Δj3|:|Δj4| ≈ 1:1 时 EEF 世界系 Z 升高且 XY 漂移最小。
    """
    arm = (arm or "right").strip().lower()
    if arm == "left":
        return -1.0, -1.0
    return +1.0, -1.0


def _plan_phase1_synergy_path(
    world,
    arm: str,
    q_start: np.ndarray,
    q_target: np.ndarray,
    *,
    n_c_steps: int = 28,
) -> List[np.ndarray]:
    """Phase1 过程：j1/j4 肩肘协同沿 C 扫向 q_target，软惩罚 forward、保证到达终点。"""
    arm = (arm or "right").strip().lower()
    q0 = np.asarray(q_start, dtype=np.float64).reshape(7).copy()
    qT = np.asarray(q_target, dtype=np.float64).reshape(7).copy()
    path: List[np.ndarray] = [q0.copy()]
    h0 = _eef_chest_horiz_components(world, arm, q0)
    xy0 = _eef_world_xy(world, arm, q0)
    meas = _measure_lift_triangle_ab(world, arm, q0)
    if h0 is None or meas is None:
        return [q0.copy(), qT.copy()]
    a, b, _, _, _ = meas
    fwd0, lat0, _ = h0
    j1_lo, j1_hi = _shoulder_j1_backswing_bounds(arm)
    j4_lo = float(qT[3])
    p_sh_t = _arm_joint_pivot_world(world, arm, 0)
    p_el_t = _arm_joint_pivot_world(world, arm, 3)
    p_ee_t = _probe_eef_pos_at_arm_qpos(world, arm, qT)
    C_end = (
        _elbow_joint_angle_rad(p_sh_t, p_el_t, p_ee_t)
        if p_sh_t is not None and p_el_t is not None and p_ee_t is not None
        else float(np.pi + j4_lo)
    )
    C_vals = np.linspace(float(np.pi), float(C_end), int(n_c_steps))
    j1_prev, j4_prev = float(q0[0]), float(q0[3])
    dist_prev = float(np.linalg.norm(qT - q0))

    for C_tgt in C_vals[1:]:
        c_tgt, _, _ = _triangle_c_and_shoulder_angle(a, b, C_tgt)
        j4_nom = float(np.clip(C_tgt - np.pi, j4_lo, 0.0))
        j4_cands = sorted({
            j4_nom,
            float(j4_prev + 0.5 * (j4_nom - j4_prev)),
            float(j4_prev + 0.85 * (j4_nom - j4_prev)),
            float(qT[3]),
        })
        best_q: Optional[np.ndarray] = None
        best_score = -float("inf")
        for j4 in j4_cands:
            if j4 > j4_prev + 0.02:
                continue
            j1_solved = _solve_j1_for_j4_triangle(
                world, arm, float(j4), a, b, C_tgt,
                j1_lo=j1_lo, j1_hi=j1_hi, n_scan=48,
            )
            j1_cands: List[float] = [float(qT[0])]
            if j1_solved is not None:
                j1_cands.append(float(j1_solved))
            j1_end = float(j1_solved) if j1_solved is not None else float(qT[0])
            j1_cands += [
                float(v) for v in np.linspace(float(j1_prev), max(float(j1_prev), j1_end), 8)
            ]
            seen: set = set()
            for j1 in j1_cands:
                j1k = round(float(j1), 4)
                if j1k in seen:
                    continue
                seen.add(j1k)
                # 后摆：j1 单调增大（左右臂同向，与收胸 j1 负 相反）
                if j1 < float(j1_prev) - 0.03:
                    continue
                qtry = path[-1].copy()
                qtry[0] = float(j1)
                qtry[3] = float(j4)
                h1 = _eef_chest_horiz_components(world, arm, qtry)
                zz = _eef_world_z_m(world, arm, qtry)
                xy = _eef_world_xy(world, arm, qtry)
                if h1 is None or zz is None or xy is None:
                    continue
                dfwd = max(0.0, float(h1[0] - fwd0))
                dlat = abs(float(h1[1] - lat0))
                dxy = float(np.linalg.norm(xy - xy0)) if xy0 is not None else 0.0
                seg_dfwd = max(
                    0.0,
                    _segment_max_forward_delta(world, arm, path[-1], qtry, fwd0, samples=10),
                )
                dist_tgt = float(np.linalg.norm(qT - qtry))
                if dist_tgt > dist_prev + 0.02:
                    continue
                p_sh = _arm_joint_pivot_world(world, arm, 0)
                p_el = _arm_joint_pivot_world(world, arm, 3)
                p_ee = _probe_eef_pos_at_arm_qpos(world, arm, qtry)
                if p_sh is None or p_el is None or p_ee is None:
                    continue
                C_act = _elbow_joint_angle_rad(p_sh, p_el, p_ee)
                tri_err = abs(C_act - C_tgt) + 0.2 * abs(
                    float(np.linalg.norm(p_ee - p_sh)) - c_tgt,
                ) / max(c_tgt, 1e-6)
                score = (
                    -dist_tgt * 3.0
                    + float(zz) * 0.02
                    - dfwd * 2.5
                    - max(dfwd - TUCK_LIFT_FORWARD_SOFT_M, 0.0) * 12.0
                    - seg_dfwd * 2.0
                    - dxy * 0.8
                    - dlat * 0.4
                    - tri_err * 0.08
                )
                if score > best_score:
                    best_score = score
                    best_q = qtry
        if best_q is None:
            continue
        if float(np.linalg.norm(best_q - path[-1], ord=np.inf)) < 0.005:
            continue
        path.append(best_q.copy())
        j1_prev, j4_prev = float(best_q[0]), float(best_q[3])
        dist_prev = float(np.linalg.norm(qT - best_q))

    if float(np.linalg.norm(path[-1] - qT, ord=np.inf)) > 0.008:
        path.append(qT.copy())
    else:
        path[-1] = qT.copy()
    return _densify_joint_path(path, max_dq=0.038)


def _plan_vertical_lift_path(
    world,
    arm: str,
    q_start: np.ndarray,
    **kwargs,
) -> List[np.ndarray]:
    """Phase1 入口：先解终点再肩肘协同扫向该位。"""
    sol = _solve_phase1_vertical_c_pose(world, arm)
    if sol is None:
        return [np.asarray(q_start, dtype=np.float64).reshape(7).copy()]
    return _plan_phase1_synergy_path(world, arm, q_start, sol["q"])


def _plan_phase1_path(world, arm: str, q_start: np.ndarray) -> List[np.ndarray]:
    """Phase1：到达 tuck_phase1_vertical 终点，过程尽量少向前。"""
    return _plan_vertical_lift_path(world, arm, q_start)


# Phase1 v2：肩肘协同后再屈腕抬高夹爪
# URDF 链 link4(肘)→link5 经 arm_joint5(q[4])，link5→link6 经 arm_joint6(q[5])
_PHASE1_WRIST_NEAR_ELBOW_I = 4   # {arm}_arm_joint5：离肘最近腕关节（小臂滚转）
_PHASE1_WRIST_PITCH_I = 5        # {arm}_arm_joint6：腕俯仰，Phase1 末态下抬升夹爪 Z


def _phase1_wrist_j5_joint_name(arm: str) -> str:
    """URDF 链：link4(肘)→link5 经 arm_joint5。"""
    return f"{(arm or 'right').strip().lower()}_arm_joint5"


def _phase1_wrist_j6_joint_name(arm: str) -> str:
    """URDF 链：link5→link6 经 arm_joint6（腕俯仰）。"""
    return f"{(arm or 'right').strip().lower()}_arm_joint6"


def _probe_joint_dz_sign(
    world,
    arm: str,
    q_base: np.ndarray,
    joint_i: int,
    *,
    dq: float = 0.15,
) -> tuple:
    """FK 探针：关节 joint_i 的 ±dq 哪侧使 EEF Z 更高。"""
    arm = (arm or "right").strip().lower()
    q0 = np.asarray(q_base, dtype=np.float64).reshape(7).copy()
    lo, hi = _R1PRO_ARM_LIMITS[arm]
    z0 = _eef_world_z_m(world, arm, q0)
    if z0 is None:
        return +1.0, 0.0, 0.0
    qp = q0.copy()
    qn = q0.copy()
    qp[joint_i] = float(np.clip(q0[joint_i] + dq, lo[joint_i], hi[joint_i]))
    qn[joint_i] = float(np.clip(q0[joint_i] - dq, lo[joint_i], hi[joint_i]))
    pp = _probe_eef_pos_at_arm_qpos(world, arm, qp)
    pn = _probe_eef_pos_at_arm_qpos(world, arm, qn)
    dz_pos = float((pp[2] - z0) * 100) if pp is not None else -1e9
    dz_neg = float((pn[2] - z0) * 100) if pn is not None else -1e9
    sign = +1.0 if dz_pos >= dz_neg else -1.0
    return sign, dz_pos, dz_neg


def _probe_wrist_j5_lift_sign(world, arm: str, q_base: np.ndarray, *, dq: float = 0.15) -> tuple:
    """arm_joint5 探针（小臂滚转，Phase1 竖直末态下通常 Δz≈0）。"""
    return _probe_joint_dz_sign(world, arm, q_base, _PHASE1_WRIST_NEAR_ELBOW_I, dq=dq)


def _probe_wrist_j6_lift_sign(world, arm: str, q_base: np.ndarray, *, dq: float = 0.15) -> tuple:
    """arm_joint6 探针（腕俯仰，实际抬升夹爪高度）。"""
    return _probe_joint_dz_sign(world, arm, q_base, _PHASE1_WRIST_PITCH_I, dq=dq)


def _solve_phase1_wrist_bend(
    world,
    arm: str,
    q_base: np.ndarray,
    *,
    sol_v1: Optional[dict] = None,
) -> Optional[dict]:
    """Phase1 v1 末态上扫描 j6(腕俯仰) 抬升夹爪；并记录 j5 探针。"""
    arm = (arm or "right").strip().lower()
    q0 = np.asarray(q_base, dtype=np.float64).reshape(7).copy()
    lo, hi = _R1PRO_ARM_LIMITS[arm]
    j6_i = _PHASE1_WRIST_PITCH_I
    h0 = _eef_chest_horiz_components(world, arm, q0)
    z0 = _eef_world_z_m(world, arm, q0)
    if h0 is None or z0 is None:
        return None
    fwd0, lat0, _ = h0
    j5_sign, j5_dz_pos, j5_dz_neg = _probe_wrist_j5_lift_sign(world, arm, q0)
    lift_sign, dz_pos_cm, dz_neg_cm = _probe_wrist_j6_lift_sign(world, arm, q0)
    if lift_sign > 0:
        j6_vals = np.linspace(0.0, float(hi[j6_i]), 80)
    else:
        j6_vals = np.linspace(0.0, float(lo[j6_i]), 80)

    best: Optional[dict] = None
    best_score = -float("inf")
    for j6 in j6_vals:
        if abs(float(j6)) < 1e-4:
            continue
        q = q0.copy()
        q[j6_i] = float(j6)
        zz = _eef_world_z_m(world, arm, q)
        h1 = _eef_chest_horiz_components(world, arm, q)
        if zz is None or h1 is None:
            continue
        dz = float(zz) - z0
        dfwd = max(0.0, float(h1[0] - fwd0))
        dlat = abs(float(h1[1] - lat0))
        collide = _arm_self_collision_at_q(world, arm, q)
        # 腕俯仰阶段：优先 Δz，适度容忍 forward；自碰作软惩罚
        score = (
            dz * 800.0
            - dfwd * 60.0
            - max(dfwd - TUCK_LIFT_FORWARD_SOFT_M, 0.0) * 100.0
            - dlat * 30.0
            - (50.0 if collide else 0.0)
        )
        if score > best_score:
            best_score = score
            best = {
                "q": q.copy(),
                "j6": float(j6),
                "j6_lift_sign": float(lift_sign),
                "dz_cm": round(dz * 100, 2),
                "eef_z_m": round(float(zz), 4),
                "delta_forward_cm": round(dfwd * 100, 2),
                "delta_lateral_cm": round(dlat * 100, 2),
                "self_collide_fk": bool(collide),
                "j6_probe_dz_pos_cm": round(dz_pos_cm, 2),
                "j6_probe_dz_neg_cm": round(dz_neg_cm, 2),
                "j5_probe_dz_pos_cm": round(j5_dz_pos, 2),
                "j5_probe_dz_neg_cm": round(j5_dz_neg, 2),
                "j5_lift_sign": float(j5_sign),
            }
    if best is None:
        return {
            "q": q0.copy(),
            "j6": 0.0,
            "j6_lift_sign": float(lift_sign),
            "dz_cm": 0.0,
            "eef_z_m": round(float(z0), 4),
            "delta_forward_cm": 0.0,
            "delta_lateral_cm": 0.0,
            "j6_probe_dz_pos_cm": round(dz_pos_cm, 2),
            "j6_probe_dz_neg_cm": round(dz_neg_cm, 2),
            "j5_probe_dz_pos_cm": round(j5_dz_pos, 2),
            "j5_probe_dz_neg_cm": round(j5_dz_neg, 2),
            "j5_lift_sign": float(j5_sign),
            "no_gain": True,
        }
    if sol_v1 is not None:
        best["dz_total_from_hang_cm"] = round(
            float(best["eef_z_m"]) * 100 - float(sol_v1.get("eef_z_m", z0)) * 100
            + float(sol_v1.get("dz_cm", 0.0)),
            2,
        )
    return best


# 兼容旧名
def _solve_phase1_wrist_j5_bend(world, arm, q_base, *, sol_v1=None):
    return _solve_phase1_wrist_bend(world, arm, q_base, sol_v1=sol_v1)


def _solve_phase1_vertical_c_pose_v2(world, arm: str) -> Optional[dict]:
    """Phase1 v2：v1 肩肘后摆 + j6(arm_joint6) 腕俯仰抬高夹爪。"""
    sol_v1 = _solve_phase1_vertical_c_pose(world, arm)
    if sol_v1 is None:
        return None
    wrist = _solve_phase1_wrist_bend(world, arm, sol_v1["q"], sol_v1=sol_v1)
    if wrist is None:
        return None
    arm = (arm or "right").strip().lower()
    out = dict(sol_v1)
    out["q"] = wrist["q"].copy()
    out["j6"] = wrist["j6"]
    out["wrist_near_elbow_joint"] = _phase1_wrist_j5_joint_name(arm)
    out["wrist_pitch_joint"] = _phase1_wrist_j6_joint_name(arm)
    out["j6_lift_sign"] = wrist["j6_lift_sign"]
    out["j5_probe"] = {
        "dz_pos_cm": wrist.get("j5_probe_dz_pos_cm"),
        "dz_neg_cm": wrist.get("j5_probe_dz_neg_cm"),
        "note": "arm_joint5 小臂滚转，Phase1 竖直末态 Δz≈0",
    }
    out["j6_probe"] = {
        "dz_pos_cm": wrist.get("j6_probe_dz_pos_cm"),
        "dz_neg_cm": wrist.get("j6_probe_dz_neg_cm"),
    }
    out["dz_wrist_cm"] = wrist["dz_cm"]
    out["eef_z_m"] = wrist["eef_z_m"]
    # 相对 hang 总抬升
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS
    z_hang = _eef_world_z_m(world, arm, _HANG_ARM_QPOS)
    if z_hang is not None:
        out["dz_cm"] = round((float(wrist["eef_z_m"]) - float(z_hang)) * 100, 2)
    out["delta_forward_cm"] = wrist.get("delta_forward_cm", out.get("delta_forward_cm"))
    out["delta_lateral_cm"] = wrist.get("delta_lateral_cm", out.get("delta_lateral_cm"))
    out["phase1_version"] = "v2"
    return out


def _plan_vertical_lift_path_v2(
    world,
    arm: str,
    q_start: np.ndarray,
    **kwargs,
) -> List[np.ndarray]:
    """Phase1 v2：肩肘协同→末段 j5 插值屈腕。"""
    sol_v1 = _solve_phase1_vertical_c_pose(world, arm)
    sol_v2 = _solve_phase1_vertical_c_pose_v2(world, arm)
    if sol_v1 is None or sol_v2 is None:
        return [np.asarray(q_start, dtype=np.float64).reshape(7).copy()]
    path = _plan_phase1_synergy_path(world, arm, q_start, sol_v1["q"])
    q_wrist_tgt = sol_v2["q"]
    if float(np.linalg.norm(path[-1] - q_wrist_tgt, ord=np.inf)) > 0.008:
        wrist_seg = _interpolate_arm_qpath(path[-1], q_wrist_tgt, 10)
        path.extend(wrist_seg[1:])
    else:
        path[-1] = q_wrist_tgt.copy()
    return _densify_joint_path(path, max_dq=0.034)


def _plan_phase1_path_v2(world, arm: str, q_start: np.ndarray) -> List[np.ndarray]:
    """Phase1 v2：肩肘后摆 + 腕部 j5 上弯。"""
    return _plan_vertical_lift_path_v2(world, arm, q_start)


def _phase2_joint_order(arm: str) -> List[int]:
    """Phase2 优先调 j1 收肩，再 j3/j4 贴胸，最后腕部。"""
    return [0, 2, 3, 1, 4, 5, 6]


def _plan_chest_approach_path(
    world,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
    z_floor_m: Optional[float] = None,
    pillar_pts_world: Optional[np.ndarray] = None,
    step_rad: float = 0.055,
    fine_step_rad: float = 0.022,
    goal_tol: float = 0.055,
    max_iters: int = 300,
) -> List[np.ndarray]:
    """Phase2：自然收向胸前 pose；forward/Z/腰柱净空约束。"""
    if z_floor_m is None:
        z_floor_m = _eef_world_z_m(world, arm, q_start)
    return _greedy_plan_tuck_path(
        world, arm, q_start, q_goal,
        max_forward_m=max_forward_m,
        z_floor_m=z_floor_m,
        pillar_pts_world=pillar_pts_world,
        step_rad=step_rad,
        fine_step_rad=fine_step_rad,
        goal_tol=goal_tol,
        max_iters=max_iters,
        joint_order=_phase2_joint_order(arm),
    )


def _segment_feasible_phase2(
    world,
    arm: str,
    qa: np.ndarray,
    qb: np.ndarray,
    *,
    max_forward_m: float,
    z_floor_m: Optional[float] = None,
    pillar_pts_world: Optional[np.ndarray] = None,
    samples: int = 12,
    max_penalty: float = 2.5,
) -> bool:
    """段内软惩罚上限（仅用于 FK 细化；规划主路径不硬截断）。"""
    return _segment_phase2_penalty(
        world, arm, qa, qb,
        max_forward_m=max_forward_m,
        z_floor_m=z_floor_m,
        pillar_pts_world=pillar_pts_world,
        samples=samples,
    ) <= float(max_penalty)


def _refine_path_forward_envelope(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    *,
    max_forward_m: float,
    z_floor_m: Optional[float] = None,
    samples: int = 14,
) -> List[np.ndarray]:
    """段内二分加密，保证插值采样满足 forward / Z 包络。"""
    if len(path) < 2:
        return [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
    out: List[np.ndarray] = [np.asarray(path[0], dtype=np.float64).reshape(7).copy()]

    def _append_segment(qa: np.ndarray, qb: np.ndarray) -> None:
        qa = np.asarray(qa, dtype=np.float64).reshape(7)
        qb = np.asarray(qb, dtype=np.float64).reshape(7)
        if float(np.linalg.norm(qb - qa, ord=np.inf)) < 0.006:
            if float(np.linalg.norm(qb - out[-1], ord=np.inf)) > 1e-4:
                out.append(qb.copy())
            return
        ok = _segment_feasible_phase2(
            world, arm, qa, qb,
            max_forward_m=max_forward_m,
            z_floor_m=z_floor_m,
            samples=samples,
        )
        if ok:
            if float(np.linalg.norm(qb - out[-1], ord=np.inf)) > 1e-4:
                out.append(qb.copy())
            return
        mid = 0.5 * (qa + qb)
        _append_segment(qa, mid)
        _append_segment(mid, qb)

    for i in range(1, len(path)):
        _append_segment(out[-1], path[i])
    return out


def _plan_chest_approach_path_legacy(
    world,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
    step_rad: float = 0.055,
    fine_step_rad: float = 0.022,
    goal_tol: float = 0.055,
    max_iters: int = 300,
) -> List[np.ndarray]:
    """旧 Phase2：每步优先减小 forward（保留作对照）。"""
    q = np.asarray(q_start, dtype=np.float64).reshape(7).copy()
    goal = np.asarray(q_goal, dtype=np.float64).reshape(7)
    path: List[np.ndarray] = [q.copy()]

    def _dist_goal(qv: np.ndarray) -> float:
        return float(np.linalg.norm(goal - qv))

    for _ in range(int(max_iters)):
        if _dist_goal(q) <= goal_tol:
            break
        cur_g = _dist_goal(q)
        best_q: Optional[np.ndarray] = None
        best_fd = float("inf")
        best_g = cur_g
        for dq_mag in (step_rad, fine_step_rad):
            for j in range(7):
                for sign in (-1.0, 1.0):
                    qtry = q.copy()
                    qtry[j] += sign * dq_mag
                    fd = eef_chest_forward_dist_m(world, arm, qtry)
                    if fd is None or fd > float(max_forward_m) + 1e-4:
                        continue
                    g2 = _dist_goal(qtry)
                    if g2 > cur_g - 0.004:
                        continue
                    if fd < best_fd - 1e-5 or (
                        fd <= best_fd + 0.004 and g2 < best_g - 1e-5
                    ):
                        best_fd, best_g, best_q = fd, g2, qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        q = best_q
        if float(np.linalg.norm(q - path[-1], ord=np.inf)) > 0.012:
            path.append(q.copy())

    goal_fd = eef_chest_forward_dist_m(world, arm, goal)
    if goal_fd is not None and goal_fd <= float(max_forward_m) + 1e-4:
        if _dist_goal(path[-1]) > goal_tol:
            path.append(goal.copy())
        else:
            path[-1] = goal.copy()
    return path


def _relax_chest_goal_forward(
    world,
    arm: str,
    q_seed: np.ndarray,
    *,
    max_forward_m: float,
    z_floor_m: Optional[float] = None,
    max_iters: int = 280,
) -> np.ndarray:
    """仅当标定 pose 超标时微调关节，不主动贴胸、不降 EEF Z。"""
    q = np.asarray(q_seed, dtype=np.float64).reshape(7).copy()
    for _ in range(int(max_iters)):
        fd = eef_chest_forward_dist_m(world, arm, q)
        if fd is not None and fd <= float(max_forward_m) + 1e-4:
            return q
        best_q: Optional[np.ndarray] = None
        best_fd = float(fd) if fd is not None else 999.0
        for dq_mag in (0.055, 0.028, 0.014):
            for j in range(7):
                for sign in (-1.0, 1.0):
                    qtry = q.copy()
                    qtry[j] += sign * dq_mag
                    fd2 = eef_chest_forward_dist_m(world, arm, qtry)
                    zz = _eef_world_z_m(world, arm, qtry)
                    if fd2 is None or fd2 > float(max_forward_m) + 1e-4:
                        continue
                    if z_floor_m is not None and zz is not None and zz < float(z_floor_m) - 1e-4:
                        continue
                    if fd2 < best_fd - 1e-5:
                        best_fd = float(fd2)
                        best_q = qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        q = best_q
    return q


def _raise_chest_goal_z_floor(
    world,
    arm: str,
    q_seed: np.ndarray,
    *,
    z_floor_m: float,
    max_forward_m: float,
    max_iters: int = 200,
) -> np.ndarray:
    """在 forward 包络内抬高胸前终点，使 EEF Z ≥ z_floor。"""
    q = np.asarray(q_seed, dtype=np.float64).reshape(7).copy()
    for _ in range(int(max_iters)):
        zz = _eef_world_z_m(world, arm, q)
        if zz is None or zz >= float(z_floor_m) - 1e-4:
            return q
        best_q: Optional[np.ndarray] = None
        best_z = float(zz)
        for dq_mag in (0.045, 0.022, 0.011):
            for j in range(7):
                for sign in (-1.0, 1.0):
                    qtry = q.copy()
                    qtry[j] += sign * dq_mag
                    fd = eef_chest_forward_dist_m(world, arm, qtry)
                    zz2 = _eef_world_z_m(world, arm, qtry)
                    if fd is None or zz2 is None:
                        continue
                    if fd > float(max_forward_m) + 1e-4:
                        continue
                    if zz2 > best_z + 1e-5:
                        best_z = float(zz2)
                        best_q = qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        q = best_q
    return q


def _resolve_chest_goal_q(
    world,
    arm: str,
    *,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
    z_floor_m: Optional[float] = None,
) -> np.ndarray:
    """胸前终点：标定 CHEST_TUCK_ARM，校验 forward 与 Z 下限。"""
    seed = CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"]).copy()
    fd = eef_chest_forward_dist_m(world, arm, seed)
    if fd is not None and fd <= float(max_forward_m) + 1e-4:
        q = seed
    else:
        q = _relax_chest_goal_forward(
            world, arm, seed,
            max_forward_m=max_forward_m,
            z_floor_m=z_floor_m,
        )
    if z_floor_m is not None:
        q = _raise_chest_goal_z_floor(
            world, arm, q,
            z_floor_m=float(z_floor_m),
            max_forward_m=max_forward_m,
        )
    return q


def _lower_chest_goal_z_j34(
    world,
    arm: str,
    q_seed: np.ndarray,
    *,
    z_target_m: float,
    z_floor_m: float,
    max_forward_m: float,
    waist_pivot_world: np.ndarray,
    lat0_m: float,
    lateral_outward_max_m: float,
    forbidden_box: Optional[dict],
) -> np.ndarray:
    """在 j34 四约束可行集内压低胸前终点 Z（控制下降）。"""
    q = np.asarray(q_seed, dtype=np.float64).reshape(7).copy()

    def _ok(qtry: np.ndarray) -> bool:
        return _phase2_constraint_point_ok(
            world, arm, qtry,
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            strict_waist_min=False,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=z_floor_m,
            forbidden_box=forbidden_box,
        )

    for _ in range(360):
        zz = _eef_world_z_m(world, arm, q)
        if zz is not None and float(zz) <= float(z_target_m) + 1e-4 and _ok(q):
            return q
        best_q: Optional[np.ndarray] = None
        best_z = float(zz) if zz is not None else 999.0
        for dq_mag in (0.05, 0.028, 0.014, 0.007):
            for j in range(7):
                for sign in (-1.0, 1.0):
                    qtry = q.copy()
                    qtry[j] += sign * dq_mag
                    zz2 = _eef_world_z_m(world, arm, qtry)
                    if zz2 is None or float(zz2) >= best_z - 1e-5:
                        continue
                    if float(zz2) < float(z_floor_m) - 1e-4:
                        continue
                    if not _ok(qtry):
                        continue
                    best_z = float(zz2)
                    best_q = qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        q = best_q
    return q


def _raise_chest_goal_z_j34(
    world,
    arm: str,
    q_seed: np.ndarray,
    *,
    z_target_m: float,
    z_floor_m: float,
    z_ceiling_m: float,
    max_forward_m: float,
    waist_pivot_world: np.ndarray,
    lat0_m: float,
    lateral_outward_max_m: float,
    forbidden_box: Optional[dict],
) -> np.ndarray:
    """在 j34 约束内抬高胸前终点 Z（避免跌穿 z_floor）。"""
    q = np.asarray(q_seed, dtype=np.float64).reshape(7).copy()

    def _ok(qtry: np.ndarray) -> bool:
        return _phase2_constraint_point_ok(
            world, arm, qtry,
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            strict_waist_min=False,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=float(z_floor_m),
            forbidden_box=forbidden_box,
        )

    for _ in range(280):
        zz = _eef_world_z_m(world, arm, q)
        if (
            zz is not None
            and float(zz) >= float(z_floor_m) - 1e-4
            and float(zz) < float(z_ceiling_m)
            and _ok(q)
        ):
            return q
        best_q: Optional[np.ndarray] = None
        best_z = float(zz) if zz is not None else -999.0
        for dq_mag in (0.05, 0.028, 0.014, 0.007):
            for j in range(7):
                for sign in (-1.0, 1.0):
                    qtry = q.copy()
                    qtry[j] += sign * dq_mag
                    zz2 = _eef_world_z_m(world, arm, qtry)
                    if zz2 is None or float(zz2) <= best_z + 1e-5:
                        continue
                    if float(zz2) >= float(z_ceiling_m) - 1e-4:
                        continue
                    if float(zz2) < float(z_floor_m) - 1e-4:
                        continue
                    if not _ok(qtry):
                        continue
                    best_z = float(zz2)
                    best_q = qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        q = best_q
    return q


def _resolve_chest_goal_q_j34(
    world,
    arm: str,
    *,
    max_forward_m: float,
    z_phase1_m: float,
    z_floor_m: float,
    waist_pivot_world: np.ndarray,
    lat0_m: float,
    lateral_outward_max_m: float,
    forbidden_box: Optional[dict],
    z_drop_max_m: float = TUCK_PHASE2_Z_DROP_MAX_M,
) -> np.ndarray:
    """胸前终点：Z 落在 [Phase1末−z_drop, Phase1末) 且 forward/禁飞区/外移合规。"""
    seed = CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"]).copy()
    q = _relax_chest_goal_forward(
        world, arm, seed,
        max_forward_m=max_forward_m,
        z_floor_m=z_floor_m,
    )
    z_ceiling = float(z_phase1_m) - float(TUCK_PHASE2_Z_DESCEND_MIN_M)
    z_target = 0.5 * (float(z_floor_m) + z_ceiling)
    zz0 = _eef_world_z_m(world, arm, q)
    if zz0 is not None and float(zz0) > z_ceiling + 1e-4:
        q = _lower_chest_goal_z_j34(
            world, arm, q,
            z_target_m=z_target,
            z_floor_m=z_floor_m,
            max_forward_m=max_forward_m,
            waist_pivot_world=waist_pivot_world,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            forbidden_box=forbidden_box,
        )
    elif zz0 is not None and float(zz0) < float(z_floor_m) - 1e-4:
        q = _raise_chest_goal_z_j34(
            world, arm, q,
            z_target_m=z_target,
            z_floor_m=float(z_floor_m),
            z_ceiling_m=z_ceiling,
            max_forward_m=max_forward_m,
            waist_pivot_world=waist_pivot_world,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            forbidden_box=forbidden_box,
        )
    return q


def _search_feasible_chest_goal_j34(
    world,
    arm: str,
    *,
    max_forward_m: float,
    z_phase1_m: float,
    z_floor_m: float,
    waist_pivot_world: np.ndarray,
    lat0_m: float,
    lateral_outward_max_m: float,
    forbidden_box: Optional[dict],
    max_trials: int = 3000,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """在 Z∈[z_floor, z_phase1) 内随机搜索满足四约束的胸前终点。"""
    arm = (arm or "left").strip().lower()
    z_ceiling = float(z_phase1_m) - float(TUCK_PHASE2_Z_DESCEND_MIN_M)
    if z_ceiling <= float(z_floor_m) + 1e-4:
        return None
    rng = np.random.default_rng(int(seed))
    lo, hi = _R1PRO_ARM_LIMITS.get(arm, _R1PRO_ARM_LIMITS["left"])
    seeds = [
        CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"]).copy(),
        _relax_chest_goal_forward(
            world, arm,
            CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"]).copy(),
            max_forward_m=max_forward_m,
            z_floor_m=z_floor_m,
        ),
    ]
    best_q: Optional[np.ndarray] = None
    best_score = float("inf")
    for trial in range(int(max_trials)):
        base = seeds[trial % len(seeds)].copy()
        q = np.clip(
            base + rng.normal(0.0, 0.06 if trial < 1200 else 0.11, 7),
            lo, hi,
        )
        zz = _eef_world_z_m(world, arm, q)
        if zz is None:
            continue
        if float(zz) < float(z_floor_m) - 1e-4 or float(zz) >= z_ceiling:
            continue
        if not _phase2_constraint_point_ok(
            world, arm, q,
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            strict_waist_min=False,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=z_floor_m,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=False,
        ):
            continue
        fd = eef_chest_forward_dist_m(world, arm, q)
        if fd is None:
            continue
        score = float(fd)
        if score < best_score:
            best_score = score
            best_q = q.copy()
    return best_q


def build_two_phase_tuck_path(
    world,
    arm: str,
    *,
    q_start: Optional[np.ndarray] = None,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
) -> tuple:
    """hang → Phase1 垂直抬升 → Phase2 自然收胸，合并为一条关节折线。

    Returns:
        (path, lift_wp_n): 折线路点列表与 Phase1 路点数（含起点）。
    """
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "right").strip().lower()
    q0 = np.asarray(
        q_start if q_start is not None else _HANG_ARM_QPOS,
        dtype=np.float64,
    ).reshape(7).copy()
    plan_fwd = min(float(max_forward_m), TUCK_PLAN_FORWARD_MAX_M)

    # Phase1 始终从 hang 几何出发；若当前不在 hang 则先插值回 hang
    path_out: List[np.ndarray] = []
    if float(np.linalg.norm(q0 - _HANG_ARM_QPOS, ord=np.inf)) > 0.04:
        pre = _interpolate_arm_qpath(q0, _HANG_ARM_QPOS, 6)
        path_out.extend(pre)
        q_lift0 = _HANG_ARM_QPOS
    else:
        q_lift0 = q0.copy()

    lift = _plan_phase1_path(world, arm, q_lift0)
    if path_out and float(np.linalg.norm(lift[0] - path_out[-1], ord=np.inf)) < 0.01:
        path_out.extend(lift[1:])
    else:
        path_out.extend(lift)
    lift_wp_n = len(path_out)
    z_floor = _eef_world_z_m(world, arm, path_out[-1])
    q_goal = _resolve_chest_goal_q(
        world, arm, max_forward_m=max_forward_m, z_floor_m=z_floor,
    )

    q_phase1_end = path_out[-1].copy()
    pillar_pts = _sweep_pillar_envelope(world, world.trunk_qpos(), n_q4=17)
    approach = _plan_chest_approach_path(
        world, arm, path_out[-1], q_goal,
        max_forward_m=plan_fwd,
        z_floor_m=z_floor,
        pillar_pts_world=pillar_pts,
        step_rad=0.038,
        fine_step_rad=0.016,
        max_iters=560,
        goal_tol=0.065,
    )
    if len(approach) < 2:
        fallback = _interpolate_arm_qpath(path_out[-1], q_goal, 14)
        approach = _densify_joint_path(fallback, max_dq=0.042)
    phase2 = [
        np.asarray(q, dtype=np.float64).reshape(7).copy() for q in approach[1:]
    ]
    goal_fd = eef_chest_forward_dist_m(world, arm, q_goal)
    goal_zz = _eef_world_z_m(world, arm, q_goal)
    goal_ok = (
        goal_fd is not None
        and goal_fd <= float(plan_fwd) + 1e-4
        and (z_floor is None or goal_zz is None or goal_zz >= float(z_floor) - 1e-4)
    )
    if goal_ok:
        if not phase2 or float(np.linalg.norm(phase2[-1] - q_goal, ord=np.inf)) > 0.02:
            phase2.append(q_goal.copy())
        else:
            phase2[-1] = q_goal.copy()
    merged = path_out + phase2
    path_out, lift_wp_n = _finalize_tuck_playback_path(
        world, arm, merged, q_phase1_end,
        max_forward_m=plan_fwd,
        z_floor_m=z_floor,
    )
    return path_out, lift_wp_n


def build_two_phase_tuck_path_legacy_hard_fwd(
    world,
    arm: str,
    *,
    q_start: Optional[np.ndarray] = None,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
    z_drop_max_m: float = 0.05,
) -> tuple:
    """hang → Phase1 垂直抬升 → Phase2 legacy 硬 forward 过滤收胸。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "right").strip().lower()
    q0 = np.asarray(
        q_start if q_start is not None else _HANG_ARM_QPOS,
        dtype=np.float64,
    ).reshape(7).copy()

    path_out: List[np.ndarray] = []
    if float(np.linalg.norm(q0 - _HANG_ARM_QPOS, ord=np.inf)) > 0.04:
        pre = _interpolate_arm_qpath(q0, _HANG_ARM_QPOS, 6)
        path_out.extend(pre)
        q_lift0 = _HANG_ARM_QPOS
    else:
        q_lift0 = q0.copy()

    lift = _plan_phase1_path(world, arm, q_lift0)
    if path_out and float(np.linalg.norm(lift[0] - path_out[-1], ord=np.inf)) < 0.01:
        path_out.extend(lift[1:])
    else:
        path_out.extend(lift)
    lift_wp_n = len(path_out)
    z_ref = _eef_world_z_m(world, arm, path_out[-1])
    z_floor = (
        float(z_ref) - float(z_drop_max_m) if z_ref is not None else None
    )
    q_goal = _resolve_chest_goal_q(
        world, arm, max_forward_m=max_forward_m, z_floor_m=z_floor,
    )
    q_phase1_end = path_out[-1].copy()
    approach = _plan_chest_approach_path_legacy(
        world, arm, path_out[-1], q_goal,
        max_forward_m=max_forward_m,
        step_rad=0.038,
        fine_step_rad=0.016,
        max_iters=400,
        goal_tol=0.065,
    )
    if len(approach) < 2:
        fallback = _interpolate_arm_qpath(path_out[-1], q_goal, 14)
        approach = _densify_joint_path(fallback, max_dq=0.042)
    phase2 = [
        np.asarray(q, dtype=np.float64).reshape(7).copy() for q in approach[1:]
    ]
    goal_fd = eef_chest_forward_dist_m(world, arm, q_goal)
    goal_ok = goal_fd is not None and goal_fd <= float(max_forward_m) + 1e-4
    if goal_ok:
        if not phase2 or float(np.linalg.norm(phase2[-1] - q_goal, ord=np.inf)) > 0.02:
            phase2.append(q_goal.copy())
        else:
            phase2[-1] = q_goal.copy()
    merged = path_out + phase2
    path_out, lift_wp_n = _finalize_tuck_playback_path(
        world, arm, merged, q_phase1_end,
        max_forward_m=max_forward_m,
        z_floor_m=z_floor,
    )
    return path_out, lift_wp_n


def build_two_phase_tuck_path_rrt_waist_fwd(
    world,
    arm: str,
    *,
    q_start: Optional[np.ndarray] = None,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
) -> tuple:
    """hang → Phase1 垂直抬升 → Phase2 RRT（forward<20cm 且腰距>d0）。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "right").strip().lower()
    q0 = np.asarray(
        q_start if q_start is not None else _HANG_ARM_QPOS,
        dtype=np.float64,
    ).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        raise RuntimeError("无法计算 torso_joint4 转轴中心")

    path_out: List[np.ndarray] = []
    if float(np.linalg.norm(q0 - _HANG_ARM_QPOS, ord=np.inf)) > 0.04:
        pre = _interpolate_arm_qpath(q0, _HANG_ARM_QPOS, 6)
        path_out.extend(pre)
        q_lift0 = _HANG_ARM_QPOS
    else:
        q_lift0 = q0.copy()

    lift = _plan_phase1_path(world, arm, q_lift0)
    if path_out and float(np.linalg.norm(lift[0] - path_out[-1], ord=np.inf)) < 0.01:
        path_out.extend(lift[1:])
    else:
        path_out.extend(lift)
    lift_wp_n = len(path_out)
    q_phase1_end = path_out[-1].copy()
    d0 = _eef_to_waist_top_dist_m(world, arm, q_phase1_end, pivot)
    if d0 is None:
        raise RuntimeError("Phase1 末 EEF 探针失败")

    q_goal = _resolve_chest_goal_q(
        world, arm, max_forward_m=max_forward_m, z_floor_m=None,
    )
    rrt_path: Optional[List[np.ndarray]] = None
    rrt_meta: dict = {}
    for seed in (0, 1, 2, 3, 4):
        ok, path, info = _rrt_connect_phase2(
            world, arm, q_phase1_end, q_goal,
            waist_pivot_world=pivot,
            waist_d0_m=float(d0),
            max_forward_m=max_forward_m,
            max_iters=6000,
            seed=seed,
        )
        if not ok or not path:
            continue
        verify = _verify_phase2_path_constraints(
            world, arm, path,
            waist_pivot_world=pivot,
            waist_d0_m=float(d0),
            max_forward_m=max_forward_m,
        )
        if verify.get("ok"):
            rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
            rrt_meta = {"seed": seed, **info, "verify": verify}
            break
    if rrt_path is None:
        ok2, path2, prm_info = _prm_phase2_connectivity(
            world, arm, q_phase1_end, q_goal,
            waist_pivot_world=pivot,
            waist_d0_m=float(d0),
            max_forward_m=max_forward_m,
            n_samples=2800,
            seed=42,
        )
        if ok2 and path2:
            verify = _verify_phase2_path_constraints(
                world, arm, path2,
                waist_pivot_world=pivot,
                waist_d0_m=float(d0),
                max_forward_m=max_forward_m,
            )
            if verify.get("ok"):
                rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path2]
                rrt_meta = {"via": "prm", **prm_info, "verify": verify}

    if rrt_path is None:
        raise RuntimeError("RRT/PRM 未找到满足 forward+腰距 约束的 Phase2 路径")

    p1d = _densify_joint_path(path_out, max_dq=0.034)
    p2d = _densify_joint_path(rrt_path, max_dq=TUCK_DENSIFY_MAX_DQ)
    if float(np.linalg.norm(p2d[0] - p1d[-1], ord=np.inf)) < 0.01:
        merged = p1d + p2d[1:]
    else:
        merged = p1d + p2d
    lift_wp_n = len(p1d)
    return merged, lift_wp_n, {"waist_d0_m": float(d0), "rrt": rrt_meta, "pivot": pivot}


def build_two_phase_tuck_path_rrt_three_constraints(
    world,
    arm: str,
    *,
    q_start: Optional[np.ndarray] = None,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
    waist_min_m: float = TUCK_PHASE2_WAIST_MIN_M,
    lateral_outward_max_m: float = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M,
) -> tuple:
    """hang → Phase1 → Phase2 RRT（forward/腰距20cm/外移5cm 三约束）。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "right").strip().lower()
    q0 = np.asarray(
        q_start if q_start is not None else _HANG_ARM_QPOS,
        dtype=np.float64,
    ).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        raise RuntimeError("无法计算 torso_joint4 转轴中心")

    path_out: List[np.ndarray] = []
    if float(np.linalg.norm(q0 - _HANG_ARM_QPOS, ord=np.inf)) > 0.04:
        pre = _interpolate_arm_qpath(q0, _HANG_ARM_QPOS, 6)
        path_out.extend(pre)
        q_lift0 = _HANG_ARM_QPOS
    else:
        q_lift0 = q0.copy()

    lift = _plan_phase1_path(world, arm, q_lift0)
    if path_out and float(np.linalg.norm(lift[0] - path_out[-1], ord=np.inf)) < 0.01:
        path_out.extend(lift[1:])
    else:
        path_out.extend(lift)
    q_phase1_end = path_out[-1].copy()
    h1 = _eef_chest_horiz_components(world, arm, q_phase1_end)
    if h1 is None:
        raise RuntimeError("Phase1 末胸口坐标失败")
    lat0 = float(h1[1])

    q_goal = _resolve_chest_goal_q(
        world, arm, max_forward_m=max_forward_m, z_floor_m=None,
    )
    rrt_path: Optional[List[np.ndarray]] = None
    rrt_meta: dict = {}
    for seed in (0, 1, 2, 3, 4):
        ok, path, info = _rrt_connect_phase2(
            world, arm, q_phase1_end, q_goal,
            waist_pivot_world=pivot,
            max_forward_m=max_forward_m,
            waist_min_m=float(waist_min_m),
            lat0_m=lat0,
            lateral_outward_max_m=float(lateral_outward_max_m),
            max_iters=8000,
            seed=seed,
        )
        if not ok or not path:
            continue
        verify = _verify_phase2_path_constraints(
            world, arm, path,
            waist_pivot_world=pivot,
            max_forward_m=max_forward_m,
            waist_min_m=float(waist_min_m),
            lat0_m=lat0,
            lateral_outward_max_m=float(lateral_outward_max_m),
        )
        if verify.get("ok"):
            rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
            rrt_meta = {"seed": seed, **info, "verify": verify}
            break
    if rrt_path is None:
        ok2, path2, prm_info = _prm_phase2_connectivity(
            world, arm, q_phase1_end, q_goal,
            waist_pivot_world=pivot,
            max_forward_m=max_forward_m,
            waist_min_m=float(waist_min_m),
            lat0_m=lat0,
            lateral_outward_max_m=float(lateral_outward_max_m),
            n_samples=4500,
            seed=42,
        )
        if ok2 and path2:
            verify = _verify_phase2_path_constraints(
                world, arm, path2,
                waist_pivot_world=pivot,
                max_forward_m=max_forward_m,
                waist_min_m=float(waist_min_m),
                lat0_m=lat0,
                lateral_outward_max_m=float(lateral_outward_max_m),
            )
            if verify.get("ok"):
                rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path2]
                rrt_meta = {"via": "prm", **prm_info, "verify": verify}

    if rrt_path is None:
        raise RuntimeError("RRT/PRM 未找到满足三约束的 Phase2 路径")

    p1d = _densify_joint_path(path_out, max_dq=0.034)
    p2d = _densify_joint_path(rrt_path, max_dq=TUCK_DENSIFY_MAX_DQ)
    if float(np.linalg.norm(p2d[0] - p1d[-1], ord=np.inf)) < 0.01:
        merged = p1d + p2d[1:]
    else:
        merged = p1d + p2d
    lift_wp_n = len(p1d)
    return merged, lift_wp_n, {
        "lat0_m": lat0,
        "waist_min_m": float(waist_min_m),
        "lateral_outward_max_m": float(lateral_outward_max_m),
        "rrt": rrt_meta,
        "pivot": pivot,
    }


def build_two_phase_tuck_path_rrt_four_constraints(
    world,
    arm: str,
    *,
    q_start: Optional[np.ndarray] = None,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
    waist_min_m: float = TUCK_PHASE2_WAIST_MIN_M,
    lateral_outward_max_m: float = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M,
) -> tuple:
    """hang → Phase1 → Phase2 RRT（forward/腰距15cm/外移5cm/Z不降 四约束）。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "right").strip().lower()
    q0 = np.asarray(
        q_start if q_start is not None else _HANG_ARM_QPOS,
        dtype=np.float64,
    ).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        raise RuntimeError("无法计算 torso_joint4 转轴中心")

    path_out: List[np.ndarray] = []
    if float(np.linalg.norm(q0 - _HANG_ARM_QPOS, ord=np.inf)) > 0.04:
        pre = _interpolate_arm_qpath(q0, _HANG_ARM_QPOS, 6)
        path_out.extend(pre)
        q_lift0 = _HANG_ARM_QPOS
    else:
        q_lift0 = q0.copy()

    lift = _plan_phase1_path(world, arm, q_lift0)
    if path_out and float(np.linalg.norm(lift[0] - path_out[-1], ord=np.inf)) < 0.01:
        path_out.extend(lift[1:])
    else:
        path_out.extend(lift)
    q_phase1_end = path_out[-1].copy()
    z_floor = _eef_world_z_m(world, arm, q_phase1_end)
    if z_floor is None:
        raise RuntimeError("Phase1 末 EEF 高度探针失败")
    h1 = _eef_chest_horiz_components(world, arm, q_phase1_end)
    if h1 is None:
        raise RuntimeError("Phase1 末胸口坐标失败")
    lat0 = float(h1[1])

    q_goal = _resolve_chest_goal_q(
        world, arm,
        max_forward_m=max_forward_m,
        z_floor_m=float(z_floor),
    )
    rrt_path: Optional[List[np.ndarray]] = None
    rrt_meta: dict = {}
    for seed in (0, 1, 2, 3, 4):
        ok, path, info = _rrt_connect_phase2(
            world, arm, q_phase1_end, q_goal,
            waist_pivot_world=pivot,
            max_forward_m=max_forward_m,
            waist_min_m=float(waist_min_m),
            lat0_m=lat0,
            lateral_outward_max_m=float(lateral_outward_max_m),
            z_floor_m=float(z_floor),
            max_iters=8000,
            seed=seed,
        )
        if not ok or not path:
            continue
        verify = _verify_phase2_path_constraints(
            world, arm, path,
            waist_pivot_world=pivot,
            max_forward_m=max_forward_m,
            waist_min_m=float(waist_min_m),
            lat0_m=lat0,
            lateral_outward_max_m=float(lateral_outward_max_m),
            z_floor_m=float(z_floor),
        )
        if verify.get("ok"):
            rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
            rrt_meta = {"seed": seed, **info, "verify": verify}
            break
    if rrt_path is None:
        ok2, path2, prm_info = _prm_phase2_connectivity(
            world, arm, q_phase1_end, q_goal,
            waist_pivot_world=pivot,
            max_forward_m=max_forward_m,
            waist_min_m=float(waist_min_m),
            lat0_m=lat0,
            lateral_outward_max_m=float(lateral_outward_max_m),
            z_floor_m=float(z_floor),
            n_samples=4500,
            seed=42,
        )
        if ok2 and path2:
            verify = _verify_phase2_path_constraints(
                world, arm, path2,
                waist_pivot_world=pivot,
                max_forward_m=max_forward_m,
                waist_min_m=float(waist_min_m),
                lat0_m=lat0,
                lateral_outward_max_m=float(lateral_outward_max_m),
                z_floor_m=float(z_floor),
            )
            if verify.get("ok"):
                rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path2]
                rrt_meta = {"via": "prm", **prm_info, "verify": verify}

    if rrt_path is None:
        raise RuntimeError("RRT/PRM 未找到满足四约束的 Phase2 路径")

    p1d = _densify_joint_path(path_out, max_dq=0.034)
    p2d = _densify_joint_path(rrt_path, max_dq=TUCK_DENSIFY_MAX_DQ)
    if float(np.linalg.norm(p2d[0] - p1d[-1], ord=np.inf)) < 0.01:
        merged = p1d + p2d[1:]
    else:
        merged = p1d + p2d
    lift_wp_n = len(p1d)
    return merged, lift_wp_n, {
        "lat0_m": lat0,
        "z_floor_m": float(z_floor),
        "waist_min_m": float(waist_min_m),
        "lateral_outward_max_m": float(lateral_outward_max_m),
        "rrt": rrt_meta,
        "pivot": pivot,
    }


def build_two_phase_tuck_path_rrt_j34_constraints(
    world,
    arm: str,
    *,
    q_start: Optional[np.ndarray] = None,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
    lateral_outward_max_m: float = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M,
    z_drop_max_m: float = TUCK_PHASE2_Z_DROP_MAX_M,
) -> tuple:
    """hang → Phase1 → Phase2 RRT（fwd<20cm / 外移≤5cm / Z降≤3cm / 禁飞区净空）。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "right").strip().lower()
    q0 = np.asarray(
        q_start if q_start is not None else _HANG_ARM_QPOS,
        dtype=np.float64,
    ).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        raise RuntimeError("无法计算 torso_joint4 转轴中心")
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if forbidden_box is None:
        raise RuntimeError("腰 j3↔j4 禁飞区构盒失败")

    path_out: List[np.ndarray] = []
    if float(np.linalg.norm(q0 - _HANG_ARM_QPOS, ord=np.inf)) > 0.04:
        pre = _interpolate_arm_qpath(q0, _HANG_ARM_QPOS, 6)
        path_out.extend(pre)
        q_lift0 = _HANG_ARM_QPOS
    else:
        q_lift0 = q0.copy()

    lift = _plan_phase1_path(world, arm, q_lift0)
    if path_out and float(np.linalg.norm(lift[0] - path_out[-1], ord=np.inf)) < 0.01:
        path_out.extend(lift[1:])
    else:
        path_out.extend(lift)
    q_phase1_end = path_out[-1].copy()
    z_phase1 = _eef_world_z_m(world, arm, q_phase1_end)
    if z_phase1 is None:
        raise RuntimeError("Phase1 末 EEF 高度探针失败")
    z_floor = float(z_phase1) - float(z_drop_max_m)
    h1 = _eef_chest_horiz_components(world, arm, q_phase1_end)
    if h1 is None:
        raise RuntimeError("Phase1 末胸口坐标失败")
    lat0 = float(h1[1])

    q_goal = _resolve_chest_goal_q_j34(
        world, arm,
        max_forward_m=max_forward_m,
        z_phase1_m=float(z_phase1),
        z_floor_m=z_floor,
        waist_pivot_world=pivot,
        lat0_m=lat0,
        lateral_outward_max_m=float(lateral_outward_max_m),
        forbidden_box=forbidden_box,
    )
    rrt_path: Optional[List[np.ndarray]] = None
    rrt_meta: dict = {}
    p2_kw = dict(
        waist_pivot_world=pivot,
        max_forward_m=max_forward_m,
        waist_min_m=0.0,
        lat0_m=lat0,
        lateral_outward_max_m=float(lateral_outward_max_m),
        z_floor_m=float(z_floor),
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        forbidden_fine_mesh=False,
    )
    v_kw = dict(
        **p2_kw,
        forbidden_fine_mesh=True,
        z_phase1_ref_m=float(z_phase1),
        require_z_descend=True,
    )
    for seed in (0, 1, 2, 3, 4):
        ok, path, info = _rrt_connect_phase2(
            world, arm, q_phase1_end, q_goal,
            max_iters=7000,
            seed=seed,
            **p2_kw,
        )
        if not ok or not path:
            continue
        verify = _verify_phase2_path_constraints(world, arm, path, **v_kw)
        if verify.get("ok"):
            rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
            rrt_meta = {"seed": seed, **info, "verify": verify}
            break
    if rrt_path is None:
        ok2, path2, prm_info = _prm_phase2_connectivity(
            world, arm, q_phase1_end, q_goal,
            n_samples=4000,
            seed=42,
            **p2_kw,
        )
        if ok2 and path2:
            verify = _verify_phase2_path_constraints(world, arm, path2, **v_kw)
            if verify.get("ok"):
                rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path2]
                rrt_meta = {"via": "prm", **prm_info, "verify": verify}

    if rrt_path is None:
        raise RuntimeError("RRT/PRM 未找到满足 j34 四约束的 Phase2 路径")

    p1d = _densify_joint_path(path_out, max_dq=0.034)
    p2d = _densify_joint_path(rrt_path, max_dq=TUCK_DENSIFY_MAX_DQ)
    if float(np.linalg.norm(p2d[0] - p1d[-1], ord=np.inf)) < 0.01:
        merged = p1d + p2d[1:]
    else:
        merged = p1d + p2d
    lift_wp_n = len(p1d)
    return merged, lift_wp_n, {
        "lat0_m": lat0,
        "z_phase1_m": float(z_phase1),
        "z_floor_m": float(z_floor),
        "lateral_outward_max_m": float(lateral_outward_max_m),
        "forbidden_box": forbidden_box,
        "rrt": rrt_meta,
        "pivot": pivot,
    }


TUCK_J34_RELAXED_LATERAL_OUTWARD_MAX_M = 0.03  # hang→胸：外移≤3cm


def build_two_phase_tuck_path_phase2_fwd_lat_forbidden(
    world,
    arm: str,
    *,
    q_start: Optional[np.ndarray] = None,
    max_forward_m: float = 0.25,
    lateral_outward_max_m: float = TUCK_J34_RELAXED_LATERAL_OUTWARD_MAX_M,
    z_drop_max_m: Optional[float] = None,
    rrt_iters: int = 7000,
    prm_samples: int = 4000,
) -> tuple:
    """hang → Phase1 垂直抬升 → Phase2 RRT（forward/外移/禁飞区；Z 可选）。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "right").strip().lower()
    q0 = np.asarray(
        q_start if q_start is not None else _HANG_ARM_QPOS,
        dtype=np.float64,
    ).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        raise RuntimeError("无法计算 torso_joint4 转轴中心")
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if forbidden_box is None:
        raise RuntimeError("腰 j3↔j4 禁飞区构盒失败")

    path_out: List[np.ndarray] = []
    if float(np.linalg.norm(q0 - _HANG_ARM_QPOS, ord=np.inf)) > 0.04:
        pre = _interpolate_arm_qpath(q0, _HANG_ARM_QPOS, 6)
        path_out.extend(pre)
        q_lift0 = _HANG_ARM_QPOS.copy()
    else:
        q_lift0 = q0.copy()

    lift = _plan_phase1_path(world, arm, q_lift0)
    if path_out and float(np.linalg.norm(lift[0] - path_out[-1], ord=np.inf)) < 0.01:
        path_out.extend(lift[1:])
    else:
        path_out.extend(lift)
    q_phase1_end = path_out[-1].copy()
    z_phase1 = _eef_world_z_m(world, arm, q_phase1_end)
    if z_phase1 is None:
        raise RuntimeError("Phase1 末 EEF 高度探针失败")
    h1 = _eef_chest_horiz_components(world, arm, q_phase1_end)
    if h1 is None:
        raise RuntimeError("Phase1 末胸口坐标失败")
    lat0 = float(h1[1])

    use_z = z_drop_max_m is not None
    z_floor = (float(z_phase1) - float(z_drop_max_m)) if use_z else None
    if use_z:
        q_goal = _resolve_chest_goal_q_j34(
            world, arm,
            max_forward_m=max_forward_m,
            z_phase1_m=float(z_phase1),
            z_floor_m=float(z_floor),
            waist_pivot_world=pivot,
            lat0_m=lat0,
            lateral_outward_max_m=float(lateral_outward_max_m),
            forbidden_box=forbidden_box,
            z_drop_max_m=float(z_drop_max_m),
        )
    else:
        q_goal = _resolve_chest_goal_j34_relaxed(
            world, arm,
            max_forward_m=max_forward_m,
            lat0_m=lat0,
            lateral_outward_max_m=float(lateral_outward_max_m),
            forbidden_box=forbidden_box,
            waist_pivot_world=pivot,
        )

    p2_kw = dict(
        waist_pivot_world=pivot,
        max_forward_m=float(max_forward_m),
        waist_min_m=0.0,
        lat0_m=lat0,
        lateral_outward_max_m=float(lateral_outward_max_m),
        z_floor_m=z_floor,
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        forbidden_fine_mesh=False,
    )
    v_kw = dict(p2_kw)
    v_kw["forbidden_fine_mesh"] = True
    v_kw["z_phase1_ref_m"] = float(z_phase1) if use_z else None
    v_kw["require_z_descend"] = bool(use_z)
    rrt_path: Optional[List[np.ndarray]] = None
    rrt_meta: dict = {}

    # 快速路径：Phase1 末→胸前直线插值（j1 后摆修正后常比 RRT 更稳）
    lerp_dense = _densify_joint_path(
        _interpolate_arm_qpath(q_phase1_end, q_goal, 36),
        max_dq=TUCK_DENSIFY_MAX_DQ,
    )
    if len(lerp_dense) >= 2:
        lerp_v = _verify_phase2_path_constraints(world, arm, lerp_dense, **v_kw)
        if lerp_v.get("ok"):
            rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in lerp_dense]
            rrt_meta = {"via": "straight_lerp", "verify": lerp_v}

    for seed in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        if rrt_path is not None:
            break
        ok, path, info = _rrt_connect_phase2(
            world, arm, q_phase1_end, q_goal,
            max_iters=int(rrt_iters),
            seed=seed,
            **p2_kw,
        )
        if not ok or not path:
            continue
        verify = _verify_phase2_path_constraints(world, arm, path, **v_kw)
        if verify.get("ok"):
            rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
            rrt_meta = {"seed": seed, **info, "verify": verify}
    if rrt_path is None:
        ok2, path2, prm_info = _prm_phase2_connectivity(
            world, arm, q_phase1_end, q_goal,
            n_samples=int(prm_samples),
            seed=42,
            **p2_kw,
        )
        if ok2 and path2:
            verify = _verify_phase2_path_constraints(world, arm, path2, **v_kw)
            if verify.get("ok"):
                rrt_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path2]
                rrt_meta = {"via": "prm", **prm_info, "verify": verify}

    if rrt_path is None:
        greedy_raw = _plan_phase2_j34_greedy(
            world, arm, q_phase1_end, q_goal,
            waist_pivot_world=pivot,
            max_forward_m=float(max_forward_m),
            lat0_m=lat0,
            lateral_outward_max_m=float(lateral_outward_max_m),
            z_floor_m=z_floor,
            forbidden_box=forbidden_box,
        )
        if len(greedy_raw) >= 2:
            greedy_v = _verify_phase2_path_constraints(world, arm, greedy_raw, **v_kw)
            if greedy_v.get("ok"):
                rrt_path = [
                    np.asarray(q, dtype=np.float64).reshape(7).copy() for q in greedy_raw
                ]
                rrt_meta = {"via": "greedy_j34", "verify": greedy_v}

    if rrt_path is None:
        raise RuntimeError("RRT/PRM 未找到满足 Phase2 约束的路径")

    p1d = _densify_joint_path(path_out, max_dq=0.034)
    p2d = _densify_joint_path(rrt_path, max_dq=TUCK_DENSIFY_MAX_DQ)
    if float(np.linalg.norm(p2d[0] - p1d[-1], ord=np.inf)) < 0.01:
        merged = p1d + p2d[1:]
    else:
        merged = p1d + p2d
    lift_wp_n = len(p1d)
    p1_j1 = [round(float(q[0]), 3) for q in p1d[::max(1, len(p1d) // 6)]]
    return merged, lift_wp_n, {
        "lat0_m": lat0,
        "z_phase1_m": float(z_phase1),
        "z_floor_m": z_floor,
        "lateral_outward_max_m": float(lateral_outward_max_m),
        "max_forward_m": float(max_forward_m),
        "forbidden_box": forbidden_box,
        "rrt": rrt_meta,
        "pivot": pivot,
        "phase1_path": p1d,
        "phase2_path": p2d,
        "phase1_j1_samples": p1_j1,
        "phase1_raw_wp": len(lift),
    }


def _resolve_chest_goal_j34_relaxed(
    world,
    arm: str,
    *,
    max_forward_m: float,
    lat0_m: float,
    lateral_outward_max_m: float,
    forbidden_box: Optional[dict],
    waist_pivot_world: np.ndarray,
) -> np.ndarray:
    """胸前终点：仅 forward / 外移 / 禁飞区（无 Phase1、无 Z 约束）。"""
    arm = (arm or "left").strip().lower()
    seed = CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"]).copy()
    q = _relax_chest_goal_forward(
        world, arm, seed,
        max_forward_m=max_forward_m,
        z_floor_m=None,
    )

    def _ok(qtry: np.ndarray) -> bool:
        return _phase2_constraint_point_ok(
            world, arm, qtry,
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            strict_waist_min=False,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=None,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=False,
        )

    if _ok(q):
        return q
    rng = np.random.default_rng(17 if arm == "left" else 29)
    lo, hi = _R1PRO_ARM_LIMITS.get(arm, _R1PRO_ARM_LIMITS["left"])
    best_q: Optional[np.ndarray] = None
    best_fd = float("inf")
    for _ in range(2500):
        qtry = np.clip(q + rng.normal(0.0, 0.07, 7), lo, hi)
        if not _ok(qtry):
            continue
        fd = eef_chest_forward_dist_m(world, arm, qtry)
        if fd is not None and fd < best_fd:
            best_fd = float(fd)
            best_q = qtry.copy()
    if best_q is not None:
        return best_q
    return q


def _verify_j34_relaxed_path(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    *,
    waist_pivot_world: np.ndarray,
    max_forward_m: float,
    lat0_m: float,
    lateral_outward_max_m: float,
    forbidden_box: Optional[dict],
) -> dict:
    """复核 hang→胸：forward / 外移 / 禁飞区（起点 hang 可豁免）。"""
    rep = _verify_phase2_path_constraints(
        world, arm, path,
        waist_pivot_world=waist_pivot_world,
        max_forward_m=max_forward_m,
        waist_min_m=0.0,
        lat0_m=lat0_m,
        lateral_outward_max_m=lateral_outward_max_m,
        z_floor_m=None,
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        forbidden_fine_mesh=True,
        require_z_descend=False,
    )
    rep["forbidden_ok"] = bool((rep.get("forbidden_hits") or 0) == 0)
    return rep


def _audit_hang_j34_relaxed(
    world,
    arm: str,
    *,
    lateral_outward_max_m: float = TUCK_J34_RELAXED_LATERAL_OUTWARD_MAX_M,
    rrt_iters: int = 4500,
    prm_samples: int = 2800,
) -> dict:
    """FK+RRT：hang→胸 三约束（无 Phase1）轨迹是否存在。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "left").strip().lower()
    max_fwd_m = TUCK_EXEC_FORWARD_MAX_M
    lat_max_m = float(lateral_outward_max_m)
    q0 = np.asarray(_HANG_ARM_QPOS, dtype=np.float64).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        return {"ok": False, "error": "无法计算 torso_joint4 转轴中心"}
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if forbidden_box is None:
        return {"ok": False, "error": "腰 j3↔j4 禁飞区构盒失败"}

    h0 = _eef_chest_horiz_components(world, arm, q0)
    if h0 is None:
        return {"ok": False, "error": "hang 胸口坐标失败"}
    lat0 = float(h0[1])
    z_hang = _eef_world_z_m(world, arm, q0)

    q_goal = _resolve_chest_goal_j34_relaxed(
        world, arm,
        max_forward_m=max_fwd_m,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        forbidden_box=forbidden_box,
        waist_pivot_world=pivot,
    )
    goal_fd = eef_chest_forward_dist_m(world, arm, q_goal)
    goal_zz = _eef_world_z_m(world, arm, q_goal)
    goal_h = _eef_chest_horiz_components(world, arm, q_goal)
    goal_lat_out = (
        _phase2_lateral_outward_delta_m(arm, goal_h[1], lat0)
        if goal_h is not None else None
    )
    goal_forbidden = _gripper_hits_waist_j34_forbidden(world, arm, q_goal, forbidden_box)
    goal_ok = (
        goal_fd is not None and goal_fd <= max_fwd_m + 1e-4
        and goal_lat_out is not None and goal_lat_out <= lat_max_m + 1e-4
        and not goal_forbidden
    )

    traj = _search_phase2_trajectory_exists(
        world, arm, q0, q_goal,
        waist_pivot_world=pivot,
        max_forward_m=max_fwd_m,
        waist_min_m=0.0,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        z_floor_m=None,
        rrt_iters=int(rrt_iters),
        prm_samples=int(prm_samples),
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        require_z_descend=False,
    )

    return {
        "ok": True,
        "arm": arm,
        "trajectory_exists": bool(traj.get("trajectory_exists")),
        "goal_feasible": bool(goal_ok),
        "constraints": {
            "max_fwd_cm": round(max_fwd_m * 100, 1),
            "lateral_outward_max_cm": round(lat_max_m * 100, 1),
            "forbidden_rule": "夹爪 mesh 不得进入腰 j3↔j4 禁飞区",
            "note": "无 Phase1 / 无 Z 约束，起点为 hang",
        },
        "hang_start": {
            "fwd_cm": round(float(h0[0]) * 100, 2),
            "lat_cm": round(lat0 * 100, 2),
            "z_m": round(float(z_hang), 4) if z_hang is not None else None,
        },
        "chest_goal": {
            "fwd_cm": round(float(goal_fd) * 100, 2) if goal_fd is not None else None,
            "lateral_outward_cm": round(float(goal_lat_out) * 100, 2)
            if goal_lat_out is not None else None,
            "z_m": round(float(goal_zz), 4) if goal_zz is not None else None,
            "forbidden_hit": bool(goal_forbidden),
            "feasible": bool(goal_ok),
        },
        "trajectory_search": traj,
        "conclusion": "轨迹存在" if traj.get("trajectory_exists") else "轨迹不存在",
    }


def _search_hang_j34_path(
    world,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    waist_pivot_world: np.ndarray,
    lat0_m: float,
    lateral_outward_max_m: float,
    forbidden_box: dict,
    max_forward_m: float,
    rrt_iters: int = 3500,
    prm_samples: int = 2000,
) -> dict:
    """hang→胸：外移 + 禁飞区 + forward 上界 路径搜索。"""
    return _search_phase2_trajectory_exists(
        world, arm, q_start, q_goal,
        waist_pivot_world=waist_pivot_world,
        max_forward_m=float(max_forward_m),
        waist_min_m=0.0,
        lat0_m=float(lat0_m),
        lateral_outward_max_m=float(lateral_outward_max_m),
        z_floor_m=None,
        rrt_iters=int(rrt_iters),
        prm_samples=int(prm_samples),
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        require_z_descend=False,
    )


def _prepare_hang_j34_context(
    world,
    arm: str,
    *,
    lateral_outward_max_m: float = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M,
) -> dict:
    """hang→胸：起点/终点/禁飞区/外移基准。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "left").strip().lower()
    q0 = np.asarray(_HANG_ARM_QPOS, dtype=np.float64).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        raise RuntimeError("无法计算 torso_joint4 转轴中心")
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if forbidden_box is None:
        raise RuntimeError("腰 j3↔j4 禁飞区构盒失败")
    h0 = _eef_chest_horiz_components(world, arm, q0)
    if h0 is None:
        raise RuntimeError("hang 胸口坐标失败")
    lat0 = float(h0[1])
    q_goal = _resolve_chest_goal_j34_relaxed(
        world, arm,
        max_forward_m=0.99,
        lat0_m=lat0,
        lateral_outward_max_m=float(lateral_outward_max_m),
        forbidden_box=forbidden_box,
        waist_pivot_world=pivot,
    )
    return {
        "arm": arm,
        "q0": q0,
        "q_goal": q_goal,
        "pivot": pivot,
        "forbidden_box": forbidden_box,
        "lat0_m": lat0,
        "lateral_outward_max_m": float(lateral_outward_max_m),
    }


def _prepare_phase2_after_phase1_context(
    world,
    arm: str,
    *,
    lateral_outward_max_m: float = TUCK_J34_RELAXED_LATERAL_OUTWARD_MAX_M,
) -> dict:
    """hang→Phase1 垂直位→Phase2：起点/终点/禁飞区（lat0 取 Phase1 末）。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "left").strip().lower()
    lift = _plan_phase1_path(world, arm, _HANG_ARM_QPOS.copy())
    if not lift:
        raise RuntimeError("Phase1 路径为空")
    q1 = np.asarray(lift[-1], dtype=np.float64).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        raise RuntimeError("无法计算 torso_joint4 转轴中心")
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if forbidden_box is None:
        raise RuntimeError("腰 j3↔j4 禁飞区构盒失败")
    h1 = _eef_chest_horiz_components(world, arm, q1)
    if h1 is None:
        raise RuntimeError("Phase1 末胸口坐标失败")
    lat0 = float(h1[1])
    z1 = _eef_world_z_m(world, arm, q1)
    q_goal = _resolve_chest_goal_j34_relaxed(
        world, arm,
        max_forward_m=0.99,
        lat0_m=lat0,
        lateral_outward_max_m=float(lateral_outward_max_m),
        forbidden_box=forbidden_box,
        waist_pivot_world=pivot,
    )
    return {
        "arm": arm,
        "q_phase1_end": q1,
        "q_goal": q_goal,
        "pivot": pivot,
        "forbidden_box": forbidden_box,
        "lat0_m": lat0,
        "lateral_outward_max_m": float(lateral_outward_max_m),
        "phase1_fwd_cm": round(float(h1[0]) * 100, 2),
        "phase1_z_m": round(float(z1), 4) if z1 is not None else None,
        "phase1_j1": round(float(q1[0]), 3),
    }


def _audit_phase2_fwd_minimum(
    world,
    arm: str,
    *,
    lateral_outward_max_m: float = TUCK_J34_RELAXED_LATERAL_OUTWARD_MAX_M,
    rrt_iters: int = 3500,
    prm_samples: int = 2000,
    bs_iters: int = 8,
    quick: bool = False,
    log_fn=None,
) -> dict:
    """Phase1 末→胸：外移+禁飞区下 Phase2 forward 峰值下界（RRT 7-DOF 含腕部）。"""
    arm = (arm or "left").strip().lower()
    try:
        hctx = _prepare_phase2_after_phase1_context(
            world, arm, lateral_outward_max_m=float(lateral_outward_max_m),
        )
    except RuntimeError as exc:
        return {"ok": False, "arm": arm, "error": str(exc)}

    skw = dict(
        waist_pivot_world=hctx["pivot"],
        lat0_m=hctx["lat0_m"],
        lateral_outward_max_m=hctx["lateral_outward_max_m"],
        forbidden_box=hctx["forbidden_box"],
        rrt_iters=int(rrt_iters),
        prm_samples=int(prm_samples),
        strict_waist_min=False,
        require_z_descend=False,
    )

    def _exists(fwd_cap_m: float) -> tuple:
        traj = _search_phase2_trajectory_exists(
            world, arm, hctx["q_phase1_end"], hctx["q_goal"],
            max_forward_m=float(fwd_cap_m),
            waist_min_m=0.0,
            z_floor_m=None,
            **skw,
        )
        return bool(traj.get("trajectory_exists")), traj

    ok_loose, traj_loose = _exists(0.99)
    if not ok_loose:
        return {
            "ok": True,
            "arm": arm,
            "path_exists_unrestricted_fwd": False,
            "phase1_end": {
                "fwd_cm": hctx["phase1_fwd_cm"],
                "lat0_cm": round(hctx["lat0_m"] * 100, 2),
                "z_m": hctx["phase1_z_m"],
                "j1": hctx["phase1_j1"],
            },
            "lateral_outward_max_cm": round(hctx["lateral_outward_max_m"] * 100, 1),
            "conclusion": "Phase1 末→胸：外移+禁飞区下无任何连通路径（放宽 forward 仍无解）",
            "rrt_dof_note": "RRT/PRM 在 7 关节全空间搜索（含腕 j4-j6）",
        }

    loose_v = traj_loose.get("path_verify") or {}
    loose_peak_cm = loose_v.get("max_fwd_cm")

    ok_20, traj_20 = _exists(TUCK_EXEC_FORWARD_MAX_M)
    ok_25, traj_25 = _exists(0.25)

    if quick:
        return {
            "ok": True,
            "arm": arm,
            "quick": True,
            "path_exists_unrestricted_fwd": True,
            "phase1_end": {
                "fwd_cm": hctx["phase1_fwd_cm"],
                "lat0_cm": round(hctx["lat0_m"] * 100, 2),
                "z_m": hctx["phase1_z_m"],
                "j1": hctx["phase1_j1"],
            },
            "lateral_outward_max_cm": round(hctx["lateral_outward_max_m"] * 100, 1),
            "loose_forward_peak_cm": loose_peak_cm,
            "feasible_fwd_lt_20cm": bool(ok_20),
            "feasible_fwd_lt_25cm": bool(ok_25),
            "fwd_lt_20cm": {
                "exists": bool(ok_20),
                "path_peak_cm": (traj_20.get("path_verify") or {}).get("max_fwd_cm") if ok_20 else None,
            },
            "fwd_lt_25cm": {
                "exists": bool(ok_25),
                "path_peak_cm": (traj_25.get("path_verify") or {}).get("max_fwd_cm") if ok_25 else None,
            },
            "rrt_dof_note": "RRT/PRM 在 7 关节全空间搜索（含腕 j4-j6）",
            "conclusion": (
                f"Phase1末 j1={hctx['phase1_j1']} fwd={hctx['phase1_fwd_cm']}cm | "
                f"放宽fwd路径峰={loose_peak_cm}cm | "
                f"<20cm={'可行' if ok_20 else '不可行'} <25cm={'可行' if ok_25 else '不可行'}"
            ),
        }

    hi = min(0.99, float(loose_peak_cm or 60.0) / 100.0 + 0.03)
    lo = 0.08
    best_cap_m = hi
    best_traj = traj_loose
    history: List[dict] = []
    for i in range(int(bs_iters)):
        mid = 0.5 * (lo + hi)
        ok_mid, traj_mid = _exists(mid)
        peak = (traj_mid.get("path_verify") or {}).get("max_fwd_cm") if ok_mid else None
        history.append({
            "iter": i,
            "cap_cm": round(mid * 100, 2),
            "exists": bool(ok_mid),
            "peak_cm": peak,
        })
        if log_fn is not None:
            log_fn(
                f"[p2_fwd_min/{arm}] #{i} cap≤{mid*100:.1f}cm → "
                f"{'有路' if ok_mid else '无'} peak={peak}cm"
            )
        if ok_mid:
            best_cap_m = mid
            best_traj = traj_mid
            hi = mid
        else:
            lo = mid

    best_v = best_traj.get("path_verify") or {}
    min_peak_cm = best_v.get("max_fwd_cm")
    return {
        "ok": True,
        "arm": arm,
        "path_exists_unrestricted_fwd": True,
        "phase1_end": {
            "fwd_cm": hctx["phase1_fwd_cm"],
            "lat0_cm": round(hctx["lat0_m"] * 100, 2),
            "z_m": hctx["phase1_z_m"],
            "j1": hctx["phase1_j1"],
        },
        "lateral_outward_max_cm": round(hctx["lateral_outward_max_m"] * 100, 1),
        "loose_forward_peak_cm": loose_peak_cm,
        "min_forward_cap_cm": round(best_cap_m * 100, 2),
        "min_forward_peak_on_tightest_path_cm": min_peak_cm,
        "feasible_fwd_lt_20cm": bool(ok_20),
        "feasible_fwd_lt_25cm": bool(ok_25),
        "found_via": best_traj.get("found_via"),
        "bs_history": history,
        "rrt_dof_note": "RRT/PRM 在 7 关节全空间搜索（含腕 j4-j6）",
        "conclusion": (
            f"Phase1末 j1={hctx['phase1_j1']} | forward 下界≈{min_peak_cm}cm "
            f"(cap={round(best_cap_m*100,1)}cm) 外移max={best_v.get('max_lateral_outward_cm')}cm "
            f"禁触={best_v.get('forbidden_hits', 0)} | <25cm={'可行' if ok_25 else '不可行'}"
        ),
    }


def _refine_hang_j34_fwd_binary(
    world,
    arm: str,
    *,
    lateral_outward_max_m: float = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M,
    lo_cm: float = 25.0,
    hi_cm: float = 66.0,
    bs_iters: int = 8,
    rrt_iters: int = 3500,
    prm_samples: int = 2000,
    log_fn=None,
) -> dict:
    """二分 [lo_cm, hi_cm]：外移+禁飞区下最小 forward 上界及对应路径。"""
    arm = (arm or "left").strip().lower()
    try:
        hctx = _prepare_hang_j34_context(
            world, arm, lateral_outward_max_m=float(lateral_outward_max_m),
        )
    except RuntimeError as exc:
        return {"ok": False, "arm": arm, "error": str(exc)}

    skw = dict(
        waist_pivot_world=hctx["pivot"],
        lat0_m=hctx["lat0_m"],
        lateral_outward_max_m=hctx["lateral_outward_max_m"],
        forbidden_box=hctx["forbidden_box"],
        rrt_iters=int(rrt_iters),
        prm_samples=int(prm_samples),
    )

    def _exists(fwd_cap_m: float) -> tuple:
        traj = _search_hang_j34_path(
            world, arm, hctx["q0"], hctx["q_goal"],
            max_forward_m=float(fwd_cap_m),
            **skw,
        )
        return bool(traj.get("trajectory_exists")), traj

    lo = float(lo_cm) / 100.0
    hi = float(hi_cm) / 100.0
    ok_hi, traj_hi = _exists(hi)
    if not ok_hi:
        ok_hi, traj_hi = _exists(0.99)
        if not ok_hi:
            return {
                "ok": False,
                "arm": arm,
                "error": "外移+禁飞区下无任何连通路径（放宽 forward 仍无解）",
            }
        hi = 0.99

    ok_lo, _ = _exists(lo)
    if log_fn is not None:
        log_fn(
            f"[refine/{arm}] 括号 lo={lo_cm}cm({'有' if ok_lo else '无'}) "
            f"hi={hi*100:.1f}cm({'有' if ok_hi else '无'})"
        )

    best_cap_m = hi
    best_traj = traj_hi
    history: List[dict] = []
    for i in range(int(bs_iters)):
        mid = 0.5 * (lo + hi)
        ok_mid, traj_mid = _exists(mid)
        peak = (traj_mid.get("path_verify") or {}).get("max_fwd_cm") if ok_mid else None
        history.append({
            "iter": i,
            "cap_cm": round(mid * 100, 2),
            "exists": bool(ok_mid),
            "peak_cm": peak,
        })
        if log_fn is not None:
            log_fn(
                f"[refine/{arm}] #{i} cap≤{mid*100:.1f}cm → "
                f"{'有路' if ok_mid else '无'} peak={peak}cm"
            )
        if ok_mid:
            best_cap_m = mid
            best_traj = traj_mid
            hi = mid
        else:
            lo = mid

    best_v = best_traj.get("path_verify") or {}
    return {
        "ok": True,
        "arm": arm,
        "lateral_outward_max_cm": round(hctx["lateral_outward_max_m"] * 100, 1),
        "min_forward_cap_cm": round(best_cap_m * 100, 2),
        "min_forward_peak_cm": best_v.get("max_fwd_cm"),
        "max_lateral_outward_cm": best_v.get("max_lateral_outward_cm"),
        "forbidden_hits": best_v.get("forbidden_hits", 0),
        "path_ok": bool(best_v.get("ok")),
        "found_via": best_traj.get("found_via"),
        "path": best_traj.get("path"),
        "verify": {
            k: best_v.get(k)
            for k in (
                "ok", "max_fwd_cm", "max_lateral_outward_cm", "forbidden_hits",
            )
        },
        "bs_history": history,
        "conclusion": (
            f"forward 下界≈{best_v.get('max_fwd_cm')}cm "
            f"(cap={round(best_cap_m*100,1)}cm) "
            f"外移max={best_v.get('max_lateral_outward_cm')}cm "
            f"禁触={best_v.get('forbidden_hits', 0)}"
        ),
    }


# hang→胸：仅硬约束外移+禁飞区，forward 用强惩罚软优化
TUCK_HANG_FWD_SOFT_CAP_M = 0.99
TUCK_HANG_FWD_PENALTY_WEIGHT = 180.0


def _hang_j34_point_lat_forbidden_ok(
    world,
    arm: str,
    arm_q: np.ndarray,
    *,
    waist_pivot_world: np.ndarray,
    lat0_m: float,
    lateral_outward_max_m: float,
    forbidden_box: Optional[dict],
    forbidden_fine_mesh: bool = False,
) -> bool:
    """仅外移 + 禁飞区硬约束（forward 仅软惩罚）。"""
    return _phase2_constraint_point_ok(
        world, arm, arm_q,
        waist_pivot_world=waist_pivot_world,
        max_forward_m=TUCK_HANG_FWD_SOFT_CAP_M,
        waist_min_m=0.0,
        lat0_m=lat0_m,
        lateral_outward_max_m=lateral_outward_max_m,
        z_floor_m=None,
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        forbidden_fine_mesh=forbidden_fine_mesh,
    )


def _hang_j34_segment_lat_forbidden_ok(
    world,
    arm: str,
    qa: np.ndarray,
    qb: np.ndarray,
    *,
    waist_pivot_world: np.ndarray,
    lat0_m: float,
    lateral_outward_max_m: float,
    forbidden_box: Optional[dict],
    samples: int = 12,
    skip_start: bool = False,
    forbidden_fine_mesh: bool = False,
) -> bool:
    qa = np.asarray(qa, dtype=np.float64).reshape(7)
    qb = np.asarray(qb, dtype=np.float64).reshape(7)
    k0 = 1 if skip_start else 0
    for k in range(k0, int(samples) + 1):
        t = k / float(max(samples, 1))
        q = (1.0 - t) * qa + t * qb
        if not _hang_j34_point_lat_forbidden_ok(
            world, arm, q,
            waist_pivot_world=waist_pivot_world,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=forbidden_fine_mesh,
        ):
            return False
    return True


def _path_fwd_lat_forbidden_metrics(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    *,
    waist_pivot_world: np.ndarray,
    lat0_m: float,
    lateral_outward_max_m: float,
    forbidden_box: Optional[dict],
    samples_per_seg: int = 14,
    skip_start: bool = True,
    forbidden_fine_mesh: bool = True,
) -> dict:
    """路径 FK：forward 峰值 + 外移/禁飞区复核。"""
    max_fwd = 0.0
    max_lat_out = 0.0
    forbidden_hits = 0
    ok = True
    for i in range(len(path) - 1):
        qa = np.asarray(path[i], dtype=np.float64).reshape(7)
        qb = np.asarray(path[i + 1], dtype=np.float64).reshape(7)
        k0 = 1 if (skip_start and i == 0) else 0
        for k in range(k0, int(samples_per_seg) + 1):
            t = k / float(max(samples_per_seg, 1))
            q = (1.0 - t) * qa + t * qb
            fd = eef_chest_forward_dist_m(world, arm, q)
            if fd is not None:
                max_fwd = max(max_fwd, float(fd))
            h = _eef_chest_horiz_components(world, arm, q)
            if h is not None:
                lat_out = _phase2_lateral_outward_delta_m(arm, h[1], float(lat0_m))
                if lat_out is not None:
                    max_lat_out = max(max_lat_out, float(lat_out))
                    if lat_out > float(lateral_outward_max_m) + 1e-4:
                        ok = False
            if forbidden_box is not None and _gripper_hits_waist_j34_forbidden(
                world, arm, q, forbidden_box, fine_mesh=forbidden_fine_mesh,
            ):
                forbidden_hits += 1
                ok = False
    return {
        "ok": bool(ok),
        "max_fwd_cm": round(max_fwd * 100.0, 2),
        "max_lateral_outward_cm": round(max_lat_out * 100.0, 2),
        "forbidden_hits": int(forbidden_hits),
    }


def _greedy_hang_j34_min_fwd_path(
    world,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    hctx: dict,
    *,
    fwd_penalty_weight: float = TUCK_HANG_FWD_PENALTY_WEIGHT,
    step_rad: float = 0.050,
    fine_step_rad: float = 0.020,
    goal_tol: float = 0.060,
    max_iters: int = 520,
    joint_order: Optional[Sequence[int]] = None,
) -> List[np.ndarray]:
    """贪心：外移+禁飞区硬约束，强 forward 惩罚。"""
    pivot = hctx["pivot"]
    lat0 = float(hctx["lat0_m"])
    lat_max = float(hctx["lateral_outward_max_m"])
    forbidden_box = hctx["forbidden_box"]
    q = np.asarray(q_start, dtype=np.float64).reshape(7).copy()
    goal = np.asarray(q_goal, dtype=np.float64).reshape(7)
    path: List[np.ndarray] = [q.copy()]
    j_order = list(joint_order) if joint_order is not None else list(range(7))

    def _dist_goal(qv: np.ndarray) -> float:
        return float(np.linalg.norm(goal - qv))

    def _point_ok(qv: np.ndarray) -> bool:
        return _hang_j34_point_lat_forbidden_ok(
            world, arm, qv,
            waist_pivot_world=pivot,
            lat0_m=lat0,
            lateral_outward_max_m=lat_max,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=False,
        )

    def _seg_ok(qa: np.ndarray, qb: np.ndarray) -> bool:
        return _hang_j34_segment_lat_forbidden_ok(
            world, arm, qa, qb,
            waist_pivot_world=pivot,
            lat0_m=lat0,
            lateral_outward_max_m=lat_max,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=False,
        )

    for _ in range(int(max_iters)):
        if _dist_goal(q) <= goal_tol:
            break
        cur_g = _dist_goal(q)
        best_q: Optional[np.ndarray] = None
        best_score = float("inf")
        for dq_mag in (step_rad, fine_step_rad):
            for j in j_order:
                for sign in (-1.0, 1.0):
                    qtry = q.copy()
                    qtry[j] += sign * dq_mag
                    if not _point_ok(qtry) or not _seg_ok(q, qtry):
                        continue
                    g2 = _dist_goal(qtry)
                    if g2 > cur_g + 0.010:
                        continue
                    fd = eef_chest_forward_dist_m(world, arm, qtry) or 999.0
                    seg_fwd = _segment_max_forward(
                        world, arm, q, qtry, samples=8,
                    )
                    score = (
                        g2
                        + float(fwd_penalty_weight) * float(fd) * float(fd)
                        + 80.0 * float(seg_fwd) * float(seg_fwd)
                        + 0.35 * float(fd)
                    )
                    if score + 1e-8 < best_score:
                        best_score = score
                        best_q = qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        q = best_q
        if float(np.linalg.norm(q - path[-1], ord=np.inf)) > 0.008:
            path.append(q.copy())

    if _dist_goal(path[-1]) > goal_tol:
        tail = _densify_joint_path(
            _interpolate_arm_qpath(path[-1], goal, 14), max_dq=0.040,
        )
        path.extend(tail[1:])
    elif float(np.linalg.norm(path[-1] - goal, ord=np.inf)) > 1e-3:
        path[-1] = goal.copy()
    return path


def _search_hang_j34_min_fwd_penalty(
    world,
    arm: str,
    *,
    lateral_outward_max_m: float = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M,
    fwd_penalty_weight: float = TUCK_HANG_FWD_PENALTY_WEIGHT,
    rrt_iters: int = 3500,
    prm_samples: int = 2000,
    log_fn=None,
) -> dict:
    """外移+禁飞区硬约束，多策略搜 forward 峰值最小路径。"""
    arm = (arm or "left").strip().lower()
    try:
        hctx = _prepare_hang_j34_context(
            world, arm, lateral_outward_max_m=float(lateral_outward_max_m),
        )
    except RuntimeError as exc:
        return {"ok": False, "arm": arm, "error": str(exc)}

    skw = dict(
        waist_pivot_world=hctx["pivot"],
        lat0_m=hctx["lat0_m"],
        lateral_outward_max_m=hctx["lateral_outward_max_m"],
        forbidden_box=hctx["forbidden_box"],
    )
    best_path: Optional[List[np.ndarray]] = None
    best_peak = float("inf")
    best_via = None
    trials: List[dict] = []

    def _consider(path: List[np.ndarray], via: str) -> None:
        nonlocal best_path, best_peak, best_via
        if len(path) < 2:
            return
        met = _path_fwd_lat_forbidden_metrics(world, arm, path, **skw)
        peak = float(met.get("max_fwd_cm") or 999.0)
        trials.append({"via": via, "peak_cm": peak, "ok": met.get("ok")})
        if met.get("ok") and peak < best_peak - 1e-4:
            best_peak = peak
            best_path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
            best_via = via
            if log_fn is not None:
                log_fn(
                    f"[fwd_pen/{arm}] 新最优 {via} peak={peak:.2f}cm "
                    f"外移max={met.get('max_lateral_outward_cm')}cm"
                )

    j_orders = [
        list(range(7)),
        [0, 3, 2, 1, 4, 5, 6],
        [3, 0, 2, 4, 1, 5, 6],
        [2, 3, 0, 1, 4, 6, 5],
        [1, 0, 3, 2, 4, 5, 6],
        [4, 3, 2, 0, 1, 5, 6],
    ]
    for i, jo in enumerate(j_orders):
        gp = _greedy_hang_j34_min_fwd_path(
            world, arm, hctx["q0"], hctx["q_goal"], hctx,
            fwd_penalty_weight=fwd_penalty_weight,
            joint_order=jo,
            max_iters=480 if i < 3 else 360,
        )
        _consider(gp, f"greedy_jo{i}")

    traj = _search_hang_j34_path(
        world, arm, hctx["q0"], hctx["q_goal"],
        max_forward_m=TUCK_HANG_FWD_SOFT_CAP_M,
        rrt_iters=int(rrt_iters),
        prm_samples=int(prm_samples),
        **skw,
    )
    if traj.get("path"):
        _consider(list(traj["path"]), f"rrt_{traj.get('found_via') or 'prm'}")

    if best_path is None:
        return {
            "ok": False,
            "arm": arm,
            "error": "未找到满足外移+禁飞区的路径",
            "trials": trials,
        }

    verify = _path_fwd_lat_forbidden_metrics(
        world, arm, best_path, forbidden_fine_mesh=True, **skw,
    )
    return {
        "ok": True,
        "arm": arm,
        "planner": "hang_j34_fwd_penalty",
        "fwd_penalty_weight": float(fwd_penalty_weight),
        "lateral_outward_max_cm": round(hctx["lateral_outward_max_m"] * 100, 1),
        "min_forward_peak_cm": verify.get("max_fwd_cm"),
        "max_lateral_outward_cm": verify.get("max_lateral_outward_cm"),
        "forbidden_hits": verify.get("forbidden_hits", 0),
        "path_ok": bool(verify.get("ok")),
        "found_via": best_via,
        "path": best_path,
        "verify": verify,
        "trials": trials,
        "conclusion": (
            f"forward 峰值≈{verify.get('max_fwd_cm')}cm（{best_via}）"
            f" 外移max={verify.get('max_lateral_outward_cm')}cm"
        ),
    }


def _audit_hang_j34_fwd_minimum(
    world,
    arm: str,
    *,
    lateral_outward_max_m: float = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M,
    rrt_iters: int = 3500,
    prm_samples: int = 2000,
    bs_iters: int = 7,
    quick: bool = False,
) -> dict:
    """hang→胸：外移≤5cm + 禁飞区净空下，forward 峰值下界（能否 <20cm / <25cm）。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "left").strip().lower()
    lat_max_m = float(lateral_outward_max_m)
    q0 = np.asarray(_HANG_ARM_QPOS, dtype=np.float64).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        return {"ok": False, "error": "无法计算 torso_joint4 转轴中心"}
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if forbidden_box is None:
        return {"ok": False, "error": "腰 j3↔j4 禁飞区构盒失败"}
    h0 = _eef_chest_horiz_components(world, arm, q0)
    if h0 is None:
        return {"ok": False, "error": "hang 胸口坐标失败"}
    lat0 = float(h0[1])

    # 胸前终点：仅外移 + 禁飞区（forward 上界放宽）
    q_goal = _resolve_chest_goal_j34_relaxed(
        world, arm,
        max_forward_m=0.99,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        forbidden_box=forbidden_box,
        waist_pivot_world=pivot,
    )
    skw = dict(
        waist_pivot_world=pivot,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        forbidden_box=forbidden_box,
        rrt_iters=rrt_iters,
        prm_samples=prm_samples,
    )

    def _exists(fwd_cap_m: float) -> tuple:
        traj = _search_hang_j34_path(
            world, arm, q0, q_goal,
            max_forward_m=float(fwd_cap_m),
            **skw,
        )
        return bool(traj.get("trajectory_exists")), traj

    ok_loose, traj_loose = _exists(0.99)
    if not ok_loose:
        return {
            "ok": True,
            "arm": arm,
            "path_exists_unrestricted_fwd": False,
            "conclusion": "外移+禁飞区约束下无任何连通路径（放宽 forward 仍无解）",
            "lateral_outward_max_cm": round(lat_max_m * 100, 1),
        }

    loose_v = traj_loose.get("path_verify") or {}
    loose_peak_cm = loose_v.get("max_fwd_cm")

    ok_20, traj_20 = _exists(TUCK_EXEC_FORWARD_MAX_M)
    ok_25, traj_25 = _exists(0.25)

    if quick:
        best_cap_m = 0.99 if ok_loose else None
        best_traj = traj_loose if ok_loose else {}
        best_v = best_traj.get("path_verify") or {}
        min_peak_cm = best_v.get("max_fwd_cm")
        return {
            "ok": True,
            "arm": arm,
            "quick": True,
            "path_exists_unrestricted_fwd": bool(ok_loose),
            "lateral_outward_max_cm": round(lat_max_m * 100, 1),
            "loose_forward_peak_cm": loose_peak_cm,
            "min_forward_peak_on_tightest_path_cm": min_peak_cm,
            "feasible_fwd_lt_20cm": bool(ok_20),
            "feasible_fwd_lt_25cm": bool(ok_25),
            "fwd_lt_20cm": {
                "exists": bool(ok_20),
                "path_peak_cm": (traj_20.get("path_verify") or {}).get("max_fwd_cm") if ok_20 else None,
            },
            "fwd_lt_25cm": {
                "exists": bool(ok_25),
                "path_peak_cm": (traj_25.get("path_verify") or {}).get("max_fwd_cm") if ok_25 else None,
            },
            "conclusion": (
                (f"放宽forward路径峰={loose_peak_cm}cm | <20cm={'可行' if ok_20 else '不可行'}"
                 f" | <25cm={'可行' if ok_25 else '不可行'}")
                if ok_loose else "外移+禁飞区下无任何连通路径（放宽 forward 仍无解）"
            ),
        }

    hi = min(0.99, float(loose_peak_cm or 60.0) / 100.0 + 0.03)
    lo = 0.08
    best_cap_m = hi
    best_traj = traj_loose
    for _ in range(int(bs_iters)):
        mid = 0.5 * (lo + hi)
        ok_mid, traj_mid = _exists(mid)
        if ok_mid:
            best_cap_m = mid
            best_traj = traj_mid
            hi = mid
        else:
            lo = mid

    best_v = best_traj.get("path_verify") or {}
    min_peak_cm = best_v.get("max_fwd_cm")

    return {
        "ok": True,
        "arm": arm,
        "path_exists_unrestricted_fwd": True,
        "lateral_outward_max_cm": round(lat_max_m * 100, 1),
        "loose_forward_peak_cm": loose_peak_cm,
        "min_forward_cap_cm": round(best_cap_m * 100, 2),
        "min_forward_peak_on_tightest_path_cm": min_peak_cm,
        "feasible_fwd_lt_20cm": bool(ok_20),
        "feasible_fwd_lt_25cm": bool(ok_25),
        "fwd_lt_20cm": {
            "exists": bool(ok_20),
            "path_peak_cm": (traj_20.get("path_verify") or {}).get("max_fwd_cm") if ok_20 else None,
            "found_via": traj_20.get("found_via") if ok_20 else None,
        },
        "fwd_lt_25cm": {
            "exists": bool(ok_25),
            "path_peak_cm": (traj_25.get("path_verify") or {}).get("max_fwd_cm") if ok_25 else None,
            "found_via": traj_25.get("found_via") if ok_25 else None,
        },
        "tightest_path": {
            "found_via": best_traj.get("found_via"),
            "waypoints": best_traj.get("path_waypoints"),
            "verify": {
                k: best_v.get(k)
                for k in (
                    "ok", "max_fwd_cm", "max_lateral_outward_cm", "forbidden_hits",
                )
            },
        },
        "conclusion": (
            f"forward 峰值下界≈{min_peak_cm}cm（cap={round(best_cap_m*100,1)}cm）"
            f" | <20cm={'可行' if ok_20 else '不可行'}"
            f" | <25cm={'可行' if ok_25 else '不可行'}"
        ),
    }


def build_hang_to_chest_j34_relaxed_path(
    world,
    arm: str,
    *,
    q_start: Optional[np.ndarray] = None,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
    lateral_outward_max_m: float = TUCK_J34_RELAXED_LATERAL_OUTWARD_MAX_M,
) -> tuple:
    """悬垂 hang → 胸前：RRT（forward<20cm / 外移≤3cm / 禁飞区净空），无 Phase1。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "right").strip().lower()
    q0 = np.asarray(
        q_start if q_start is not None else _HANG_ARM_QPOS,
        dtype=np.float64,
    ).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        raise RuntimeError("无法计算 torso_joint4 转轴中心")
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if forbidden_box is None:
        raise RuntimeError("腰 j3↔j4 禁飞区构盒失败")

    h0 = _eef_chest_horiz_components(world, arm, q0)
    if h0 is None:
        raise RuntimeError("起点胸口坐标失败")
    lat0 = float(h0[1])

    q_goal = _resolve_chest_goal_j34_relaxed(
        world, arm,
        max_forward_m=max_forward_m,
        lat0_m=lat0,
        lateral_outward_max_m=float(lateral_outward_max_m),
        forbidden_box=forbidden_box,
        waist_pivot_world=pivot,
    )
    traj = _search_phase2_trajectory_exists(
        world, arm, q0, q_goal,
        waist_pivot_world=pivot,
        max_forward_m=max_forward_m,
        waist_min_m=0.0,
        lat0_m=lat0,
        lateral_outward_max_m=float(lateral_outward_max_m),
        z_floor_m=None,
        rrt_iters=4500,
        prm_samples=2800,
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        require_z_descend=False,
    )
    rrt_path = traj.get("path")
    if not traj.get("trajectory_exists") or not rrt_path:
        raise RuntimeError("RRT/PRM 未找到满足三约束的 hang→胸 路径")

    merged = _densify_joint_path(rrt_path, max_dq=TUCK_DENSIFY_MAX_DQ)
    return merged, 1, {
        "lat0_m": lat0,
        "lateral_outward_max_m": float(lateral_outward_max_m),
        "forbidden_box": forbidden_box,
        "rrt": {
            "found_via": traj.get("found_via"),
            "path_verify": traj.get("path_verify"),
            "search_meta": traj.get("search_meta"),
        },
        "pivot": pivot,
        "q_goal": q_goal,
    }


def build_hang_to_chest_j34_relaxed_lerp_demo(
    world,
    arm: str,
    *,
    q_start: Optional[np.ndarray] = None,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
    lateral_outward_max_m: float = TUCK_J34_RELAXED_LATERAL_OUTWARD_MAX_M,
) -> tuple:
    """hang→胸前直线关节插值（仅演示；可能违反路径约束）。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "right").strip().lower()
    q0 = np.asarray(
        q_start if q_start is not None else _HANG_ARM_QPOS,
        dtype=np.float64,
    ).reshape(7).copy()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        raise RuntimeError("无法计算 torso_joint4 转轴中心")
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if forbidden_box is None:
        raise RuntimeError("腰 j3↔j4 禁飞区构盒失败")
    h0 = _eef_chest_horiz_components(world, arm, q0)
    if h0 is None:
        raise RuntimeError("起点胸口坐标失败")
    lat0 = float(h0[1])
    q_goal = _resolve_chest_goal_j34_relaxed(
        world, arm,
        max_forward_m=max_forward_m,
        lat0_m=lat0,
        lateral_outward_max_m=float(lateral_outward_max_m),
        forbidden_box=forbidden_box,
        waist_pivot_world=pivot,
    )
    lerp = _densify_joint_path(
        _interpolate_arm_qpath(q0, q_goal, 36),
        max_dq=TUCK_DENSIFY_MAX_DQ,
    )
    verify = _verify_j34_relaxed_path(
        world, arm, lerp,
        waist_pivot_world=pivot,
        max_forward_m=max_forward_m,
        lat0_m=lat0,
        lateral_outward_max_m=float(lateral_outward_max_m),
        forbidden_box=forbidden_box,
    )
    return lerp, verify, {
        "lat0_m": lat0,
        "lateral_outward_max_m": float(lateral_outward_max_m),
        "pivot": pivot,
        "q_goal": q_goal,
        "lerp_verify": verify,
    }


def tune_chest_goal_raise_eef(
    world,
    arm: str,
    q_goal: np.ndarray,
    *,
    raise_m: float = 0.05,
    max_forward_m: float = TUCK_CHEST_FORWARD_MAX_M,
    max_iters: int = 320,
) -> np.ndarray:
    """在胸口 forward 包络内贪心抬高 EEF 世界系 Z。"""
    goal = np.asarray(q_goal, dtype=np.float64).reshape(7).copy()
    z0 = _eef_world_z_m(world, arm, goal)
    if z0 is None or float(raise_m) <= 1e-4:
        return goal
    target_z = z0 + float(raise_m)

    for _ in range(int(max_iters)):
        z = _eef_world_z_m(world, arm, goal)
        if z is None or z >= target_z - 0.002:
            break
        best_q: Optional[np.ndarray] = None
        best_z = z
        for dq_mag in (0.055, 0.028, 0.014):
            for j in range(7):
                for sign in (-1.0, 1.0):
                    qtry = goal.copy()
                    qtry[j] += sign * dq_mag
                    fd = eef_chest_forward_dist_m(world, arm, qtry)
                    zz = _eef_world_z_m(world, arm, qtry)
                    if fd is None or zz is None:
                        continue
                    if fd > float(max_forward_m) + 1e-4:
                        continue
                    if zz > best_z + 1e-5:
                        best_z = zz
                        best_q = qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        goal = best_q
    return goal


def search_feasible_chest_goal(
    world,
    arm: str,
    q_seed: np.ndarray,
    *,
    max_forward_m: float = TUCK_CHEST_FORWARD_MAX_M,
    z_min_drop_m: float = 0.12,
    max_iters: int = 260,
) -> Optional[np.ndarray]:
    """在胸口 forward 包络内搜索胸前终点：优先更贴胸，必要时略降高度。"""
    seed = np.asarray(q_seed, dtype=np.float64).reshape(7).copy()
    z_ref = _eef_world_z_m(world, arm, seed)
    best_q: Optional[np.ndarray] = None
    best_fd = float("inf")
    best_z = -float("inf")

    for z_drop in (0.0, 0.03, 0.06, 0.09, float(z_min_drop_m)):
        z_floor = (z_ref - z_drop) if z_ref is not None else None
        q = seed.copy()
        for _ in range(int(max_iters)):
            fd = eef_chest_forward_dist_m(world, arm, q)
            zz = _eef_world_z_m(world, arm, q)
            if fd is not None and zz is not None and fd <= float(max_forward_m) + 1e-4:
                if z_floor is None or zz >= z_floor - 0.004:
                    if fd < best_fd - 1e-4 or (fd <= best_fd + 0.008 and zz > best_z):
                        best_fd, best_z, best_q = fd, zz, q.copy()

            best_next: Optional[np.ndarray] = None
            best_score = float("inf")
            cur_fd = fd if fd is not None else 999.0
            for dq_mag in (0.08, 0.055, 0.028, 0.014):
                for j in range(7):
                    for sign in (-1.0, 1.0):
                        qtry = q.copy()
                        qtry[j] += sign * dq_mag
                        fd2 = eef_chest_forward_dist_m(world, arm, qtry)
                        zz2 = _eef_world_z_m(world, arm, qtry)
                        if fd2 is None or zz2 is None:
                            continue
                        if fd2 > float(max_forward_m) + 1e-4:
                            continue
                        if z_floor is not None and zz2 < z_floor:
                            continue
                        z_pen = abs(zz2 - z_ref) * 2.5 if z_ref is not None else 0.0
                        score = fd2 + z_pen + max(0.0, cur_fd - fd2) * (-0.05)
                        if score < best_score - 1e-6:
                            best_score = score
                            best_next = qtry
                if best_next is not None:
                    break
            if best_next is None:
                break
            q = best_next
        if best_q is not None and best_fd <= float(max_forward_m) * 0.95:
            break
    return best_q


def tune_chest_goal_pull_in(
    world,
    arm: str,
    q_goal: np.ndarray,
    *,
    max_forward_m: float = TUCK_CHEST_FORWARD_MAX_M,
    target_forward_m: Optional[float] = None,
    max_iters: int = 400,
) -> np.ndarray:
    """贪心收向胸口：减小 EEF 胸口 forward，尽量保持高度。"""
    found = search_feasible_chest_goal(
        world, arm, q_goal, max_forward_m=max_forward_m, max_iters=int(max_iters),
    )
    if found is not None:
        return found

    goal = np.asarray(q_goal, dtype=np.float64).reshape(7).copy()
    z_ref = _eef_world_z_m(world, arm, goal)
    tgt = float(target_forward_m if target_forward_m is not None else max_forward_m * 0.88)

    for _ in range(int(max_iters)):
        fd = eef_chest_forward_dist_m(world, arm, goal)
        if fd is None:
            break
        if fd <= tgt + 0.003:
            break
        best_q: Optional[np.ndarray] = None
        best_score = float("inf")
        for dq_mag in (0.055, 0.028, 0.014, 0.007):
            for j in range(7):
                for sign in (-1.0, 1.0):
                    qtry = goal.copy()
                    qtry[j] += sign * dq_mag
                    fd2 = eef_chest_forward_dist_m(world, arm, qtry)
                    zz = _eef_world_z_m(world, arm, qtry)
                    if fd2 is None or zz is None:
                        continue
                    if fd2 > float(max_forward_m) + 1e-4:
                        continue
                    z_pen = 0.0 if z_ref is None else abs(zz - z_ref) * 3.0 + max(0.0, z_ref - zz) * 6.0
                    score = fd2 + z_pen
                    if score < best_score - 1e-6:
                        best_score = score
                        best_q = qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        goal = best_q
    return goal


def _segment_max_forward(
    world, arm: str, qa: np.ndarray, qb: np.ndarray, *, samples: int = 12,
) -> float:
    worst = 0.0
    for i in range(int(samples) + 1):
        t = i / float(samples)
        q = (1.0 - t) * qa + t * qb
        fd = eef_chest_forward_dist_m(world, arm, q)
        if fd is not None:
            worst = max(worst, fd)
    return worst


def _segment_max_forward_delta(
    world,
    arm: str,
    qa: np.ndarray,
    qb: np.ndarray,
    fwd_ref: float,
    *,
    samples: int = 14,
) -> float:
    """段内采样：相对 fwd_ref 的最大 forward 前移量（胸口投影）。"""
    worst = 0.0
    for i in range(int(samples) + 1):
        t = i / float(samples)
        q = (1.0 - t) * qa + t * qb
        fd = eef_chest_forward_dist_m(world, arm, q)
        if fd is not None:
            worst = max(worst, float(fd) - float(fwd_ref))
    return worst


def _refine_phase1_path_forward(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    fwd_ref: float,
    *,
    max_dfwd_m: float,
    samples: int = 12,
) -> List[np.ndarray]:
    """Phase1 段内二分，保证插值过程 forward 前移≤上限。"""
    if len(path) < 2:
        return [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
    out: List[np.ndarray] = [np.asarray(path[0], dtype=np.float64).reshape(7).copy()]

    def _append_seg(qa: np.ndarray, qb: np.ndarray) -> None:
        qa = np.asarray(qa, dtype=np.float64).reshape(7)
        qb = np.asarray(qb, dtype=np.float64).reshape(7)
        if float(np.linalg.norm(qb - qa, ord=np.inf)) < 0.005:
            if float(np.linalg.norm(qb - out[-1], ord=np.inf)) > 1e-4:
                out.append(qb.copy())
            return
        if _segment_max_forward_delta(
            world, arm, qa, qb, fwd_ref, samples=samples,
        ) <= float(max_dfwd_m) + 1e-4:
            if float(np.linalg.norm(qb - out[-1], ord=np.inf)) > 1e-4:
                out.append(qb.copy())
            return
        mid = 0.5 * (qa + qb)
        _append_seg(qa, mid)
        _append_seg(mid, qb)

    for i in range(1, len(path)):
        _append_seg(out[-1], path[i])
    return out


def _greedy_plan_tuck_path(
    world,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    max_forward_m: float = TUCK_EXEC_FORWARD_MAX_M,
    z_floor_m: Optional[float] = None,
    step_rad: float = 0.055,
    fine_step_rad: float = 0.022,
    goal_tol: float = 0.055,
    max_iters: int = 220,
    joint_order: Optional[Sequence[int]] = None,
    pillar_pts_world: Optional[np.ndarray] = None,
    yield_fn=None,
):
    """贪心关节搜索：胸口 forward、EEF Z、腰柱净空，自然逼近目标关节角。"""
    q = np.asarray(q_start, dtype=np.float64).reshape(7).copy()
    goal = np.asarray(q_goal, dtype=np.float64).reshape(7)
    path: List[np.ndarray] = [q.copy()]
    j_order = list(joint_order) if joint_order is not None else list(range(7))

    def _dist_goal(qv: np.ndarray) -> float:
        return float(np.linalg.norm(goal - qv))

    for it in range(int(max_iters)):
        if yield_fn is not None and it % 8 == 0:
            yield_fn()
        if _dist_goal(q) <= goal_tol:
            break
        best_q: Optional[np.ndarray] = None
        best_score = float("inf")
        cur_g = _dist_goal(q)
        cur_fd = eef_chest_forward_dist_m(world, arm, q) or 999.0
        for dq_mag in (step_rad, fine_step_rad):
            for j in j_order:
                for sign in (-1.0, 1.0):
                    qtry = q.copy()
                    qtry[j] += sign * dq_mag
                    fd = eef_chest_forward_dist_m(world, arm, qtry)
                    if fd is None:
                        continue
                    g = _dist_goal(qtry)
                    if g > cur_g + 0.012:
                        continue
                    pt_pen = _phase2_point_penalty(
                        world, arm, qtry,
                        max_forward_m=max_forward_m,
                        z_floor_m=z_floor_m,
                        pillar_pts_world=pillar_pts_world,
                    )
                    seg_pen = _segment_phase2_penalty(
                        world, arm, q, qtry,
                        max_forward_m=max_forward_m,
                        z_floor_m=z_floor_m,
                        pillar_pts_world=pillar_pts_world,
                        samples=6,
                    )
                    score = (
                        g
                        + pt_pen * 2.2
                        + seg_pen * 1.0
                        + float(fd) * 0.25
                        + max(0.0, float(fd) - float(cur_fd)) * 0.8
                    )
                    if score + 1e-6 < best_score:
                        best_score = score
                        best_q = qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        q = best_q
        if float(np.linalg.norm(q - path[-1], ord=np.inf)) > 0.010:
            path.append(q.copy())

    if _dist_goal(path[-1]) > goal_tol:
        tail = _densify_joint_path(
            _interpolate_arm_qpath(path[-1], goal, 12), max_dq=0.042,
        )
        path.extend(tail[1:])
    elif float(np.linalg.norm(path[-1] - goal, ord=np.inf)) > 1e-3:
        path[-1] = goal.copy()
    dedup: List[np.ndarray] = [path[0]]
    for qv in path[1:]:
        if float(np.linalg.norm(qv - dedup[-1], ord=np.inf)) > 0.01:
            dedup.append(qv)
    if float(np.linalg.norm(dedup[-1] - goal, ord=np.inf)) > 1e-3:
        dedup.append(goal.copy())
    return dedup


def _densify_joint_path(
    path: Sequence[np.ndarray], *, max_dq: float = 0.042,
) -> List[np.ndarray]:
    """段内线性插值，限制相邻路点 max|Δq|，便于动态跟踪。"""
    if len(path) < 2:
        return [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
    out: List[np.ndarray] = [np.asarray(path[0], dtype=np.float64).reshape(7).copy()]
    for i in range(1, len(path)):
        a = np.asarray(path[i - 1], dtype=np.float64).reshape(7)
        b = np.asarray(path[i], dtype=np.float64).reshape(7)
        d = float(np.linalg.norm(b - a, ord=np.inf))
        n = max(1, int(np.ceil(d / float(max_dq))))
        for k in range(1, n + 1):
            t = k / n
            q = (1.0 - t) * a + t * b
            if float(np.linalg.norm(q - out[-1], ord=np.inf)) > 0.008:
                out.append(q.copy())
    return out


def _lift_boundary_index(path: Sequence[np.ndarray], q_phase1_end: np.ndarray) -> int:
    """在加密/细化后定位 Phase1 末点索引（含起点计数）。"""
    q_ref = np.asarray(q_phase1_end, dtype=np.float64).reshape(7)
    best_i, best_d = 0, float("inf")
    for i, q in enumerate(path):
        d = float(np.linalg.norm(np.asarray(q).reshape(7) - q_ref, ord=np.inf))
        if d < best_d:
            best_d, best_i = d, i
    return int(best_i) + 1


def _finalize_tuck_playback_path(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    q_phase1_end: np.ndarray,
    *,
    max_forward_m: float,
    z_floor_m: Optional[float],
) -> tuple:
    """Phase1 轻加密；Phase2 加密 + forward/Z 包络细化。"""
    qi = _lift_boundary_index(path, q_phase1_end) - 1
    qi = int(np.clip(qi, 0, len(path) - 1))
    p1 = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path[: qi + 1]]
    p2 = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path[qi:]]
    p1d = _densify_joint_path(p1, max_dq=0.034)
    p2d = _densify_joint_path(p2, max_dq=TUCK_DENSIFY_MAX_DQ)
    merged = p1d + p2d[1:]
    return merged, len(p1d)


def _downsample_path(path: Sequence[np.ndarray], max_waypoints: int = 14) -> List[np.ndarray]:
    """保留端点与均匀抽样，供 exec 播放（段数适中）。"""
    if len(path) <= max_waypoints:
        return [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
    idx = np.linspace(0, len(path) - 1, max_waypoints).astype(int)
    out: List[np.ndarray] = []
    for i in idx:
        q = np.asarray(path[int(i)], dtype=np.float64).reshape(7)
        if not out or float(np.linalg.norm(q - out[-1], ord=np.inf)) > 0.01:
            out.append(q.copy())
    if float(np.linalg.norm(out[-1] - path[-1], ord=np.inf)) > 1e-3:
        out.append(np.asarray(path[-1], dtype=np.float64).reshape(7).copy())
    return out


def verify_tuck_path(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    *,
    max_forward_m: float = TUCK_CHEST_FORWARD_MAX_M,
    samples: int = 12,
) -> Dict[str, float]:
    worst = 0.0
    for i in range(len(path) - 1):
        w = _segment_max_forward(world, arm, path[i], path[i + 1], samples=samples)
        worst = max(worst, w)
    return {
        "waypoints_n": float(len(path)),
        "max_forward_m": worst,
        "max_forward_cm": worst * 100.0,
        "ok": float(worst <= max_forward_m + 1e-3),
    }


def _path_arc_cum(path: Sequence[np.ndarray]) -> List[float]:
    """关节空间折线弧长前缀和。"""
    cum = [0.0]
    for i in range(len(path) - 1):
        seg = float(np.linalg.norm(np.asarray(path[i + 1]) - np.asarray(path[i])))
        cum.append(cum[-1] + seg)
    return cum


def _interp_path_at_s(path: Sequence[np.ndarray], cum: Sequence[float], s: float) -> np.ndarray:
    s = float(max(0.0, min(float(s), float(cum[-1]))))
    for i in range(len(path) - 1):
        if cum[i + 1] >= s - 1e-12:
            seg_len = cum[i + 1] - cum[i]
            t = 0.0 if seg_len < 1e-12 else (s - cum[i]) / seg_len
            q0 = np.asarray(path[i], dtype=np.float64)
            q1 = np.asarray(path[i + 1], dtype=np.float64)
            return (1.0 - t) * q0 + t * q1
    return np.asarray(path[-1], dtype=np.float64).copy()


def _project_onto_path_arc(cur: np.ndarray, path: Sequence[np.ndarray], cum: Sequence[float]) -> float:
    """当前关节在路径上的投影弧长（只前进、不折回）。"""
    cur = np.asarray(cur, dtype=np.float64).reshape(7)
    best_s, best_d = 0.0, float("inf")
    for i in range(len(path) - 1):
        a = np.asarray(path[i], dtype=np.float64)
        b = np.asarray(path[i + 1], dtype=np.float64)
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom < 1e-12 else float(np.clip(np.dot(cur - a, ab) / denom, 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(cur - q, ord=np.inf))
        s = float(cum[i] + t * (cum[i + 1] - cum[i]))
        if d < best_d:
            best_d, best_s = d, s
    return best_s


def _quat_xyzw_to_rot(q: np.ndarray) -> np.ndarray:
    """四元数 xyzw → 3×3 旋转矩阵。"""
    x, y, z, w = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(4)]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _pos_in_link_frame(world, link_name: str, pos_world: np.ndarray) -> Optional[np.ndarray]:
    """世界坐标点 → 指定 link 局部坐标。"""
    try:
        link = world.robot.links.get(link_name)
        if link is None:
            return None
        lpos_t, lquat_t = link.get_position_orientation()
        lpos = np.asarray(lpos_t, dtype=np.float64).reshape(3)
        R = _quat_xyzw_to_rot(np.asarray(lquat_t, dtype=np.float64))
        return R.T @ (np.asarray(pos_world, dtype=np.float64).reshape(3) - lpos)
    except Exception:
        return None


def _link_world_pos(world, link_name: str) -> Optional[np.ndarray]:
    try:
        link = world.robot.links.get(link_name)
        if link is None:
            return None
        p, _ = link.get_position_orientation()
        return np.asarray(p, dtype=np.float64).reshape(3)
    except Exception:
        return None


def _world_to_robot_base_local(world, pos_world: np.ndarray) -> Optional[np.ndarray]:
    """世界坐标 → 机器人 base 根坐标系（R1 对称面为 base 局部 y=0）。"""
    try:
        base_pos, base_quat = world.robot.get_position_orientation()
        bpos = np.asarray(base_pos, dtype=np.float64).reshape(3)
        R = _quat_xyzw_to_rot(np.asarray(base_quat, dtype=np.float64))
        return R.T @ (np.asarray(pos_world, dtype=np.float64).reshape(3) - bpos)
    except Exception:
        return None


def _waist_top_joint4_axis_center_world(world) -> Optional[np.ndarray]:
    """腰部最高关节 torso_joint4 转轴中心（世界系）。

    取 URDF 关节原点在 torso_link3 局部 (0,0,0.1)：在左右对称轴 y=0 上，
    且固定在 link3 上，不随 q4 转动（≠ torso_link4 原点，link4 会绕该点转）。
    """
    pts = _link_local_to_world(
        world, "torso_link3", _TRUNK_J4_PIVOT_IN_LINK3.reshape(1, 3),
    )
    if len(pts) < 1:
        return None
    return np.asarray(pts[0], dtype=np.float64).reshape(3)


def _verify_waist_top_joint4_pivot(world) -> dict:
    """校验转轴中心：对称面、link4 偏差、扫 q4 不变性。"""
    pivot = _waist_top_joint4_axis_center_world(world)
    link4 = _link_world_pos(world, "torso_link4")
    base_loc = _world_to_robot_base_local(world, pivot) if pivot is not None else None
    err_l4_cm = None
    if pivot is not None and link4 is not None:
        err_l4_cm = round(float(np.linalg.norm(pivot - link4)) * 100.0, 3)
    sym_y_cm = round(abs(float(base_loc[1])) * 100.0, 3) if base_loc is not None else None
    # 扫 q4：枢轴应不变（在 link3 上）
    q4_span_cm = None
    try:
        tq0 = world.trunk_qpos().copy()
        pivots = []
        for q4 in np.linspace(_R1PRO_TRUNK_Q4_RANGE[0], _R1PRO_TRUNK_Q4_RANGE[1], 9):
            tq = tq0.copy()
            tq[3] = float(q4)
            saved = _probe_set_trunk_qpos(world, tq)
            p = _waist_top_joint4_axis_center_world(world)
            if p is not None:
                pivots.append(p)
            world.robot.set_joint_positions(saved)
        if len(pivots) >= 2:
            arr = np.asarray(pivots, dtype=np.float64)
            q4_span_cm = round(float(np.max(np.linalg.norm(arr - arr[0], axis=1))) * 100.0, 4)
    except Exception:
        pass
    return {
        "joint": "torso_joint4",
        "parent_link": "torso_link3",
        "pivot_local_in_link3_m": _TRUNK_J4_PIVOT_IN_LINK3.tolist(),
        "pivot_world_m": pivot.round(4).tolist() if pivot is not None else None,
        "torso_link4_world_m": link4.round(4).tolist() if link4 is not None else None,
        "pivot_vs_link4_cm": err_l4_cm,
        "base_local_y_cm": sym_y_cm,
        "q4_sweep_pivot_span_cm": q4_span_cm,
        "on_symmetry_plane": bool(sym_y_cm is not None and sym_y_cm < 1.5),
        "invariant_to_q4": bool(q4_span_cm is not None and q4_span_cm < 0.05),
    }


def _eef_to_waist_top_dist_m(
    world, arm: str, arm_q: np.ndarray, waist_pivot_world: np.ndarray,
) -> Optional[float]:
    epos = _probe_eef_pos_at_arm_qpos(world, arm, arm_q)
    if epos is None:
        return None
    pivot = np.asarray(waist_pivot_world, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(epos - pivot))


def _sample_path_fwd_waist_envelope(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    *,
    waist_pivot_world: np.ndarray,
    waist_d0_m: float,
    max_forward_m: float,
    samples_per_seg: int = 16,
    waist_min_gt_d0: bool = True,
    skip_phase1_start: bool = True,
) -> dict:
    """FK 段内：forward 峰值、腰顶距离 min/max。

    waist_min_gt_d0=True：Phase2 腰距最小值须 > Phase1 末 d0（skip 起点）。
    """
    max_fwd = 0.0
    max_waist = 0.0
    min_waist = 999.0
    for i in range(len(path) - 1):
        qa = np.asarray(path[i], dtype=np.float64).reshape(7)
        qb = np.asarray(path[i + 1], dtype=np.float64).reshape(7)
        k_start = 1 if (skip_phase1_start and i == 0) else 0
        for k in range(k_start, int(samples_per_seg) + 1):
            t = k / float(max(samples_per_seg, 1))
            q = (1.0 - t) * qa + t * qb
            fd = eef_chest_forward_dist_m(world, arm, q)
            wd = _eef_to_waist_top_dist_m(world, arm, q, waist_pivot_world)
            if fd is not None:
                max_fwd = max(max_fwd, float(fd))
            if wd is not None:
                max_waist = max(max_waist, float(wd))
                min_waist = min(min_waist, float(wd))
    if waist_min_gt_d0:
        waist_ok = bool(min_waist > float(waist_d0_m) + 1e-3)
    else:
        waist_ok = bool(max_waist <= float(waist_d0_m) + 1e-3)
    fwd_ok = bool(max_fwd <= float(max_forward_m) + 1e-4)
    return {
        "max_fwd_cm": round(max_fwd * 100.0, 2),
        "max_waist_dist_cm": round(max_waist * 100.0, 2),
        "min_waist_dist_cm": round(min_waist * 100.0, 2) if min_waist < 900 else None,
        "waist_d0_cm": round(float(waist_d0_m) * 100.0, 2),
        "waist_rule": "min>d0" if waist_min_gt_d0 else "max<=d0",
        "fwd_ok": fwd_ok,
        "waist_ok": waist_ok,
        "ok": bool(fwd_ok and waist_ok),
    }


def _plan_chest_approach_path_legacy_waist(
    world,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    max_forward_m: float,
    waist_pivot_world: np.ndarray,
    waist_d0_m: float,
    step_rad: float = 0.038,
    fine_step_rad: float = 0.016,
    goal_tol: float = 0.065,
    max_iters: int = 450,
) -> List[np.ndarray]:
    """Phase2：硬 forward + 腰距 > Phase1 末 d0（段内采样）。"""
    q = np.asarray(q_start, dtype=np.float64).reshape(7).copy()
    goal = np.asarray(q_goal, dtype=np.float64).reshape(7)
    path: List[np.ndarray] = [q.copy()]
    pivot = np.asarray(waist_pivot_world, dtype=np.float64).reshape(3)

    def _dist_goal(qv: np.ndarray) -> float:
        return float(np.linalg.norm(goal - qv))

    def _point_ok(qv: np.ndarray, *, allow_phase1_end: bool = False) -> bool:
        fd = eef_chest_forward_dist_m(world, arm, qv)
        wd = _eef_to_waist_top_dist_m(world, arm, qv, pivot)
        if fd is None or wd is None:
            return False
        if float(fd) > float(max_forward_m) + 1e-4:
            return False
        if allow_phase1_end:
            return float(wd) >= float(waist_d0_m) - 1e-3
        return float(wd) > float(waist_d0_m) + 1e-3

    def _seg_ok(qa: np.ndarray, qb: np.ndarray) -> bool:
        for k in range(7):
            t = k / 6.0
            qv = (1.0 - t) * qa + t * qb
            if not _point_ok(qv):
                return False
        return True

    for _ in range(int(max_iters)):
        if _dist_goal(q) <= goal_tol:
            break
        cur_g = _dist_goal(q)
        best_q: Optional[np.ndarray] = None
        best_fd = float("inf")
        best_g = cur_g
        for dq_mag in (step_rad, fine_step_rad):
            for j in range(7):
                for sign in (-1.0, 1.0):
                    qtry = q.copy()
                    qtry[j] += sign * dq_mag
                    if not _point_ok(qtry) or not _seg_ok(q, qtry):
                        continue
                    g2 = _dist_goal(qtry)
                    if g2 > cur_g - 0.004:
                        continue
                    fd = eef_chest_forward_dist_m(world, arm, qtry) or 999.0
                    if fd < best_fd - 1e-5 or (
                        fd <= best_fd + 0.004 and g2 < best_g - 1e-5
                    ):
                        best_fd, best_g, best_q = float(fd), g2, qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        q = best_q
        if float(np.linalg.norm(q - path[-1], ord=np.inf)) > 0.012:
            path.append(q.copy())

    if _point_ok(goal) and _dist_goal(path[-1]) > goal_tol:
        if _seg_ok(path[-1], goal):
            path.append(goal.copy())
        else:
            tail = _densify_joint_path(
                _interpolate_arm_qpath(path[-1], goal, 10), max_dq=0.035,
            )
            ok_tail = [path[-1]]
            for qn in tail[1:]:
                if _point_ok(qn) and _seg_ok(ok_tail[-1], qn):
                    ok_tail.append(qn)
            path.extend(ok_tail[1:])
    return path


def _phase2_lateral_outward_delta_m(arm: str, lat_m: float, lat0_m: float) -> float:
    """相对 Phase1 末：左臂向左(+lat)、右臂向右(-lat) 的外移量（朝胸心方向为负，不受限）。"""
    arm = (arm or "right").strip().lower()
    if arm == "left":
        return max(0.0, float(lat_m) - float(lat0_m))
    return max(0.0, float(lat0_m) - float(lat_m))


def _phase2_constraint_point_ok(
    world,
    arm: str,
    arm_q: np.ndarray,
    *,
    waist_pivot_world: np.ndarray,
    max_forward_m: float,
    waist_min_m: float = TUCK_PHASE2_WAIST_MIN_M,
    lat0_m: Optional[float] = None,
    lateral_outward_max_m: Optional[float] = None,
    z_floor_m: Optional[float] = None,
    strict_waist_min: bool = True,
    forbidden_box: Optional[dict] = None,
    forbidden_fine_mesh: bool = False,
) -> bool:
    fd = eef_chest_forward_dist_m(world, arm, arm_q)
    wd = _eef_to_waist_top_dist_m(world, arm, arm_q, waist_pivot_world)
    zz = _eef_world_z_m(world, arm, arm_q)
    if fd is None or wd is None:
        return False
    if float(fd) > float(max_forward_m) + 1e-4:
        return False
    if strict_waist_min and float(wd) <= float(waist_min_m) + 1e-3:
        return False
    if z_floor_m is not None and zz is not None:
        if float(zz) < float(z_floor_m) - 1e-5:
            return False
    if lat0_m is not None and lateral_outward_max_m is not None:
        h = _eef_chest_horiz_components(world, arm, arm_q)
        if h is None:
            return False
        outward = _phase2_lateral_outward_delta_m(arm, h[1], float(lat0_m))
        if outward > float(lateral_outward_max_m) + 1e-4:
            return False
    if forbidden_box is not None and _gripper_hits_waist_j34_forbidden(
        world, arm, arm_q, forbidden_box, fine_mesh=forbidden_fine_mesh,
    ):
        return False
    return True


def _phase2_constraint_segment_ok(
    world,
    arm: str,
    qa: np.ndarray,
    qb: np.ndarray,
    *,
    waist_pivot_world: np.ndarray,
    max_forward_m: float,
    waist_min_m: float = TUCK_PHASE2_WAIST_MIN_M,
    lat0_m: Optional[float] = None,
    lateral_outward_max_m: Optional[float] = None,
    z_floor_m: Optional[float] = None,
    samples: int = 14,
    skip_start: bool = False,
    strict_waist_min: bool = True,
    forbidden_box: Optional[dict] = None,
    forbidden_fine_mesh: bool = False,
) -> bool:
    qa = np.asarray(qa, dtype=np.float64).reshape(7)
    qb = np.asarray(qb, dtype=np.float64).reshape(7)
    k0 = 1 if skip_start else 0
    for k in range(k0, int(samples) + 1):
        t = k / float(max(samples, 1))
        q = (1.0 - t) * qa + t * qb
        if not _phase2_constraint_point_ok(
            world, arm, q,
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            waist_min_m=waist_min_m,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=z_floor_m,
            strict_waist_min=strict_waist_min,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=forbidden_fine_mesh,
        ):
            return False
    return True


def _plan_phase2_j34_greedy(
    world,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    waist_pivot_world: np.ndarray,
    max_forward_m: float,
    lat0_m: float,
    lateral_outward_max_m: float,
    z_floor_m: Optional[float] = None,
    forbidden_box: Optional[dict] = None,
    step_rad: float = 0.045,
    fine_step_rad: float = 0.018,
    goal_tol: float = 0.06,
    max_iters: int = 650,
) -> List[np.ndarray]:
    """Phase2 贪心：硬约束 forward / 外移 / 禁飞区（段内采样，粗 mesh）。"""
    arm = (arm or "right").strip().lower()
    q = np.asarray(q_start, dtype=np.float64).reshape(7).copy()
    goal = np.asarray(q_goal, dtype=np.float64).reshape(7)
    path: List[np.ndarray] = [q.copy()]
    j_order = _phase2_joint_order(arm)
    lo, hi = _R1PRO_ARM_LIMITS.get(arm, _R1PRO_ARM_LIMITS["left"])

    def _dist_goal(qv: np.ndarray) -> float:
        return float(np.linalg.norm(goal - qv))

    for _ in range(int(max_iters)):
        if _dist_goal(q) <= goal_tol:
            break
        best_q: Optional[np.ndarray] = None
        best_g = _dist_goal(q)
        for dq_mag in (step_rad, fine_step_rad):
            for j in j_order:
                for sign in (-1.0, 1.0):
                    qtry = q.copy()
                    qtry[j] += sign * dq_mag
                    qtry = np.clip(qtry, lo, hi)
                    if not _phase2_constraint_segment_ok(
                        world, arm, q, qtry,
                        waist_pivot_world=waist_pivot_world,
                        max_forward_m=max_forward_m,
                        waist_min_m=0.0,
                        lat0_m=lat0_m,
                        lateral_outward_max_m=lateral_outward_max_m,
                        z_floor_m=z_floor_m,
                        skip_start=True,
                        strict_waist_min=False,
                        forbidden_box=forbidden_box,
                        forbidden_fine_mesh=False,
                        samples=10,
                    ):
                        continue
                    g = _dist_goal(qtry)
                    if g + 1e-6 < best_g:
                        best_g = g
                        best_q = qtry
            if best_q is not None:
                break
        if best_q is None:
            break
        q = best_q
        if float(np.linalg.norm(q - path[-1], ord=np.inf)) > 0.008:
            path.append(q.copy())

    if _dist_goal(path[-1]) > goal_tol:
        tail = _densify_joint_path(
            _interpolate_arm_qpath(path[-1], goal, 16), max_dq=0.035,
        )
        path.extend(tail[1:])
    elif float(np.linalg.norm(path[-1] - goal, ord=np.inf)) > 1e-3:
        path[-1] = goal.copy()
    return path


def _verify_phase2_path_constraints(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    *,
    waist_pivot_world: np.ndarray,
    max_forward_m: float,
    waist_min_m: float = TUCK_PHASE2_WAIST_MIN_M,
    lat0_m: Optional[float] = None,
    lateral_outward_max_m: Optional[float] = None,
    z_floor_m: Optional[float] = None,
    waist_d0_m: Optional[float] = None,
    strict_waist_min: bool = True,
    forbidden_box: Optional[dict] = None,
    forbidden_fine_mesh: bool = True,
    z_phase1_ref_m: Optional[float] = None,
    require_z_descend: bool = False,
) -> dict:
    """复核 Phase2 折线（起点可落在 Phase1 末，其余点须满足全部约束）。"""
    _ = waist_d0_m  # 旧参数兼容
    if len(path) < 2:
        return {"ok": False, "error": "path too short"}
    max_fwd, min_waist, max_lat_out, min_z = 0.0, 999.0, 0.0, 999.0
    forbidden_hits = 0
    for i in range(len(path) - 1):
        skip = i == 0
        if not _phase2_constraint_segment_ok(
            world, arm, path[i], path[i + 1],
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            waist_min_m=waist_min_m,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=z_floor_m,
            skip_start=skip,
            strict_waist_min=strict_waist_min,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=forbidden_fine_mesh,
        ):
            return {"ok": False, "failed_segment": i}
        for qi, q in enumerate((path[i], path[i + 1])):
            if skip and qi == 0:
                continue
            fd = eef_chest_forward_dist_m(world, arm, q)
            wd = _eef_to_waist_top_dist_m(world, arm, q, waist_pivot_world)
            zz = _eef_world_z_m(world, arm, q)
            h = _eef_chest_horiz_components(world, arm, q)
            if fd is not None:
                max_fwd = max(max_fwd, float(fd))
            if wd is not None:
                min_waist = min(min_waist, float(wd))
            if zz is not None:
                min_z = min(min_z, float(zz))
            if h is not None and lat0_m is not None:
                max_lat_out = max(
                    max_lat_out,
                    _phase2_lateral_outward_delta_m(arm, h[1], float(lat0_m)),
                )
            if forbidden_box is not None and _gripper_hits_waist_j34_forbidden(
                world, arm, q, forbidden_box, fine_mesh=forbidden_fine_mesh,
            ):
                forbidden_hits += 1
    z_ok = True
    if z_floor_m is not None and min_z < 900:
        z_ok = bool(min_z >= float(z_floor_m) - 1e-5)
    z_end_m = _eef_world_z_m(world, arm, path[-1])
    z_descend_ok = True
    z_descend_cm = None
    if require_z_descend and z_phase1_ref_m is not None and z_end_m is not None:
        z_descend_cm = round((float(z_phase1_ref_m) - float(z_end_m)) * 100.0, 3)
        z_descend_ok = float(z_end_m) < float(z_phase1_ref_m) - float(TUCK_PHASE2_Z_DESCEND_MIN_M)
    forbidden_ok = forbidden_hits == 0
    return {
        "ok": bool(z_ok and forbidden_ok and z_descend_ok),
        "max_fwd_cm": round(max_fwd * 100.0, 2),
        "min_waist_cm": round(min_waist * 100.0, 2) if min_waist < 900 else None,
        "max_lateral_outward_cm": round(max_lat_out * 100.0, 2),
        "min_z_m": round(min_z, 4) if min_z < 900 else None,
        "z_end_m": round(float(z_end_m), 4) if z_end_m is not None else None,
        "z_floor_m": round(float(z_floor_m), 4) if z_floor_m is not None else None,
        "z_drop_cm": round((float(z_phase1_ref_m or z_floor_m or 0) - min_z) * 100.0, 3)
        if (z_phase1_ref_m is not None or z_floor_m is not None) and min_z < 900 else None,
        "z_descend_cm": z_descend_cm,
        "forbidden_hits": int(forbidden_hits),
        "z_descend_ok": bool(z_descend_ok),
        "forbidden_ok": bool(forbidden_ok),
    }


def _rrt_connect_phase2(
    world,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    waist_pivot_world: np.ndarray,
    max_forward_m: float,
    waist_min_m: float = TUCK_PHASE2_WAIST_MIN_M,
    lat0_m: Optional[float] = None,
    lateral_outward_max_m: Optional[float] = None,
    z_floor_m: Optional[float] = None,
    max_iters: int = 12000,
    step_rad: float = 0.055,
    goal_tol: float = 0.065,
    goal_bias: float = 0.25,
    seed: int = 0,
    waist_d0_m: Optional[float] = None,
    strict_waist_min: bool = True,
    forbidden_box: Optional[dict] = None,
    forbidden_fine_mesh: bool = False,
) -> tuple:
    """双向 RRT-Connect，在 Phase2 可行集内搜 q_start→q_goal 连通路径。"""
    if waist_d0_m is not None:
        waist_min_m = float(waist_d0_m)
    rng = np.random.default_rng(int(seed))
    lo, hi = _R1PRO_ARM_LIMITS.get(arm, _R1PRO_ARM_LIMITS["left"])

    def _steer(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
        d = qb - qa
        n = float(np.linalg.norm(d, ord=np.inf))
        if n < 1e-9:
            return qa.copy()
        if n <= step_rad:
            return np.clip(qb, lo, hi)
        return np.clip(qa + d * (step_rad / n), lo, hi)

    def _nearest(nodes: List[np.ndarray], q: np.ndarray) -> int:
        best_i, best_d = 0, float("inf")
        for i, n in enumerate(nodes):
            d = float(np.linalg.norm(n - q, ord=np.inf))
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _seg_ok(qa: np.ndarray, qb: np.ndarray, skip_start: bool) -> bool:
        return _phase2_constraint_segment_ok(
            world, arm, qa, qb,
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            waist_min_m=waist_min_m,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=z_floor_m,
            skip_start=skip_start,
            strict_waist_min=strict_waist_min,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=forbidden_fine_mesh,
        )

    def _extend(
        nodes: List[np.ndarray],
        parents: List[int],
        q_tgt: np.ndarray,
        root_is_start: bool,
    ) -> Optional[int]:
        i = _nearest(nodes, q_tgt)
        q_new = _steer(nodes[i], q_tgt)
        if float(np.linalg.norm(q_new - nodes[i], ord=np.inf)) < 0.004:
            return None
        skip = root_is_start and i == 0
        if not _seg_ok(nodes[i], q_new, skip_start=skip):
            return None
        nodes.append(q_new)
        parents.append(i)
        return len(nodes) - 1

    def _connect(
        nodes_a: List[np.ndarray],
        ja: int,
        nodes_b: List[np.ndarray],
        parents_b: List[int],
        root_b_is_start: bool,
    ) -> Optional[int]:
        """从 A 树节点 ja 向 B 树贪心连接，成功返回 B 树汇合节点索引。"""
        q_tgt = nodes_a[ja]
        for _ in range(48):
            ib = _nearest(nodes_b, q_tgt)
            q_new = _steer(nodes_b[ib], q_tgt)
            if float(np.linalg.norm(q_new - nodes_b[ib], ord=np.inf)) < 0.004:
                break
            skip = root_b_is_start and ib == 0
            if not _seg_ok(nodes_b[ib], q_new, skip_start=skip):
                break
            nodes_b.append(q_new)
            parents_b.append(ib)
            ib = len(nodes_b) - 1
            if float(np.linalg.norm(q_new - q_tgt, ord=np.inf)) <= goal_tol:
                if _seg_ok(q_new, q_tgt, skip_start=False):
                    nodes_b.append(q_tgt.copy())
                    parents_b.append(ib)
                    return len(nodes_b) - 1
                return ib
            q_tgt = q_new
        return None

    def _reconstruct(
        nodes_s, par_s, ja, nodes_g, par_g, jg,
    ) -> List[np.ndarray]:
        left: List[np.ndarray] = [nodes_s[ja]]
        c = ja
        while par_s[c] >= 0:
            c = par_s[c]
            left.append(nodes_s[c])
        left.reverse()
        right: List[np.ndarray] = []
        c = jg
        while c >= 0:
            right.append(nodes_g[c])
            c = par_g[c]
        return [q.copy() for q in left] + [q.copy() for q in right]

    q_start = np.clip(np.asarray(q_start, dtype=np.float64).reshape(7), lo, hi)
    q_goal = np.clip(np.asarray(q_goal, dtype=np.float64).reshape(7), lo, hi)
    if not _phase2_constraint_point_ok(
        world, arm, q_goal,
        waist_pivot_world=waist_pivot_world,
        max_forward_m=max_forward_m,
        waist_min_m=waist_min_m,
        lat0_m=lat0_m,
        lateral_outward_max_m=lateral_outward_max_m,
        z_floor_m=z_floor_m,
    ):
        return False, [], {"reason": "goal_infeasible"}

    nodes_s = [q_start.copy()]
    par_s = [-1]
    nodes_g = [q_goal.copy()]
    par_g = [-1]

    for it in range(int(max_iters)):
        q_rand = (
            (q_goal if it % 2 == 0 else q_start)
            if rng.random() < goal_bias
            else rng.uniform(lo, hi)
        )
        if it % 2 == 0:
            j = _extend(nodes_s, par_s, q_rand, True)
            if j is not None:
                jg = _connect(nodes_s, j, nodes_g, par_g, False)
                if jg is not None:
                    path = _reconstruct(nodes_s, par_s, j, nodes_g, par_g, jg)
                    return True, path, {"iters": it + 1, "seed": seed}
        else:
            j = _extend(nodes_g, par_g, q_rand, False)
            if j is not None:
                jg = _connect(nodes_g, j, nodes_s, par_s, True)
                if jg is not None:
                    path = _reconstruct(nodes_s, par_s, jg, nodes_g, par_g, j)
                    return True, path, {"iters": it + 1, "seed": seed}

    return False, [], {"iters": max_iters, "seed": seed, "nodes_s": len(nodes_s), "nodes_g": len(nodes_g)}


def _prm_phase2_connectivity(
    world,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    waist_pivot_world: np.ndarray,
    max_forward_m: float,
    waist_min_m: float = TUCK_PHASE2_WAIST_MIN_M,
    lat0_m: Optional[float] = None,
    lateral_outward_max_m: Optional[float] = None,
    z_floor_m: Optional[float] = None,
    n_samples: int = 4500,
    connect_radius: float = 0.42,
    seed: int = 1,
    waist_d0_m: Optional[float] = None,
    strict_waist_min: bool = True,
    forbidden_box: Optional[dict] = None,
    forbidden_fine_mesh: bool = False,
) -> tuple:
    """随机路图：在可行集内采样，BFS 检测 q_start 与 q_goal 是否连通。"""
    if waist_d0_m is not None:
        waist_min_m = float(waist_d0_m)
    rng = np.random.default_rng(int(seed))
    lo, hi = _R1PRO_ARM_LIMITS.get(arm, _R1PRO_ARM_LIMITS["left"])
    q_start = np.asarray(q_start, dtype=np.float64).reshape(7)
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(7)

    nodes: List[np.ndarray] = []
    for _ in range(int(n_samples)):
        q = rng.uniform(lo, hi)
        if _phase2_constraint_point_ok(
            world, arm, q,
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            waist_min_m=waist_min_m,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=z_floor_m,
            strict_waist_min=strict_waist_min,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=forbidden_fine_mesh,
        ):
            nodes.append(q)

    if not _phase2_constraint_point_ok(
        world, arm, q_goal,
        waist_pivot_world=waist_pivot_world,
        max_forward_m=max_forward_m,
        waist_min_m=waist_min_m,
        lat0_m=lat0_m,
        lateral_outward_max_m=lateral_outward_max_m,
        z_floor_m=z_floor_m,
        strict_waist_min=strict_waist_min,
        forbidden_box=forbidden_box,
        forbidden_fine_mesh=forbidden_fine_mesh,
    ):
        return False, [], {"feasible_samples": len(nodes), "reason": "goal_infeasible"}

    n_total = len(nodes) + 2
    idx_start, idx_goal = n_total - 2, n_total - 1
    all_nodes = nodes + [q_start.copy(), q_goal.copy()]
    adj: List[List[int]] = [[] for _ in range(n_total)]

    def _try_edge(i: int, j: int) -> None:
        if i == j:
            return
        skip_i = i == idx_start
        if not _phase2_constraint_segment_ok(
            world, arm, all_nodes[i], all_nodes[j],
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            waist_min_m=waist_min_m,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=z_floor_m,
            skip_start=skip_i,
            strict_waist_min=strict_waist_min,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=forbidden_fine_mesh,
        ):
            return
        adj[i].append(j)
        adj[j].append(i)

    # 起点/终点只连近邻（加速）
    for i in range(n_total):
        for j in range(i + 1, n_total):
            d = float(np.linalg.norm(all_nodes[i] - all_nodes[j], ord=np.inf))
            if d > connect_radius and i < len(nodes) and j < len(nodes):
                continue
            if i >= len(nodes) or j >= len(nodes) or d <= connect_radius:
                _try_edge(i, j)

    # BFS
    from collections import deque
    q_bfs = deque([idx_start])
    prev = {idx_start: -1}
    while q_bfs:
        u = q_bfs.popleft()
        if u == idx_goal:
            path_idx = []
            v = idx_goal
            while v >= 0:
                path_idx.append(v)
                v = prev[v]
            path_idx.reverse()
            path = [all_nodes[i].copy() for i in path_idx]
            return True, path, {
                "feasible_samples": len(nodes),
                "edges": sum(len(a) for a in adj) // 2,
            }
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q_bfs.append(v)

    return False, [], {"feasible_samples": len(nodes), "edges": sum(len(a) for a in adj) // 2}


def _search_phase2_trajectory_exists(
    world,
    arm: str,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    waist_pivot_world: np.ndarray,
    max_forward_m: float,
    waist_min_m: Optional[float] = None,
    lat0_m: Optional[float] = None,
    lateral_outward_max_m: Optional[float] = None,
    z_floor_m: Optional[float] = None,
    waist_d0_m: Optional[float] = None,
    rrt_iters: int = 6000,
    prm_samples: int = 2800,
    strict_waist_min: bool = True,
    forbidden_box: Optional[dict] = None,
    z_phase1_ref_m: Optional[float] = None,
    require_z_descend: bool = False,
) -> dict:
    """判定 Phase2 轨迹是否存在：多 seed RRT-Connect + PRM，找到即证明存在。"""
    wmin = (
        float(waist_min_m) if waist_min_m is not None
        else float(waist_d0_m) if waist_d0_m is not None
        else float(TUCK_PHASE2_WAIST_MIN_M)
    )
    rrt_runs = []
    found_path: Optional[List[np.ndarray]] = None
    found_via = None
    meta = {}

    v_kw = dict(
        waist_pivot_world=waist_pivot_world,
        max_forward_m=max_forward_m,
        waist_min_m=wmin,
        lat0_m=lat0_m,
        lateral_outward_max_m=lateral_outward_max_m,
        z_floor_m=z_floor_m,
        strict_waist_min=strict_waist_min,
        forbidden_box=forbidden_box,
        forbidden_fine_mesh=bool(forbidden_box),
        z_phase1_ref_m=z_phase1_ref_m,
        require_z_descend=require_z_descend,
    )
    for seed in (0, 1, 2, 3, 4, 5, 6, 7):
        ok, path, info = _rrt_connect_phase2(
            world, arm, q_start, q_goal,
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            waist_min_m=wmin,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=z_floor_m,
            max_iters=int(rrt_iters),
            seed=seed,
            strict_waist_min=strict_waist_min,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=False,
        )
        rrt_runs.append({"seed": seed, "found": ok, **info})
        if not ok or not path:
            continue
        verify = _verify_phase2_path_constraints(world, arm, path, **v_kw)
        if verify.get("ok"):
            found_path = path
            found_via = f"rrt_connect_seed{seed}"
            meta = {**info, "verify": verify}
            break

    prm_info = {}
    verify = meta.get("verify") if found_path is not None else None
    if found_path is None:
        ok, path, prm_info = _prm_phase2_connectivity(
            world, arm, q_start, q_goal,
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            waist_min_m=wmin,
            lat0_m=lat0_m,
            lateral_outward_max_m=lateral_outward_max_m,
            z_floor_m=z_floor_m,
            n_samples=int(prm_samples),
            seed=42,
            strict_waist_min=strict_waist_min,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=False,
        )
        if ok and path:
            verify = _verify_phase2_path_constraints(world, arm, path, **v_kw)
            if verify.get("ok"):
                found_path = path
                found_via = f"prm_{prm_samples}"
                meta = {**prm_info, "verify": verify}

    if found_path is None:
        greedy_raw = _plan_phase2_j34_greedy(
            world, arm, q_start, q_goal,
            waist_pivot_world=waist_pivot_world,
            max_forward_m=max_forward_m,
            lat0_m=float(lat0_m) if lat0_m is not None else 0.0,
            lateral_outward_max_m=float(lateral_outward_max_m or 0.0),
            z_floor_m=z_floor_m,
            forbidden_box=forbidden_box,
        )
        if len(greedy_raw) >= 2:
            verify = _verify_phase2_path_constraints(world, arm, greedy_raw, **v_kw)
            if verify.get("ok"):
                found_path = greedy_raw
                found_via = "greedy_j34"
                meta = {"verify": verify}

    exists = bool(found_path is not None and verify is not None and verify.get("ok"))
    return {
        "trajectory_exists": exists,
        "found_via": found_via,
        "path_waypoints": len(found_path) if found_path else 0,
        "path": (
            [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in found_path]
            if found_path is not None else None
        ),
        "path_verify": verify,
        "rrt_runs": rrt_runs,
        "prm": prm_info,
        "search_meta": meta,
        "search_note": (
            "RRT/PRM 找到路径并经段内 FK 复核 → 轨迹存在"
            if exists
            else "5×RRT(6000iter)+PRM(2800样本)均未连通 → 轨迹不存在（FK 模型下可判定）"
        ),
    }


def _audit_phase2_after_phase1(
    world,
    arm: str,
    *,
    max_forward_m: float,
    lateral_outward_max_m: float,
    z_drop_max_m: Optional[float] = None,
    rrt_iters: int = 4500,
    prm_samples: int = 2800,
) -> dict:
    """先 Phase1（hang→垂直位），再判定 Phase2→胸前轨迹是否存在。

    Phase2 硬约束：forward / 外移 / 禁飞区；z_drop_max_m 为 None 时不约束 Z。
    """
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "left").strip().lower()
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        return {"ok": False, "error": "无法计算 torso_joint4 转轴中心"}
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if forbidden_box is None:
        return {"ok": False, "error": "腰 j3↔j4 禁飞区构盒失败"}

    lift = _plan_phase1_path(world, arm, _HANG_ARM_QPOS.copy())
    if not lift:
        return {"ok": False, "error": "Phase1 路径为空"}
    q1 = np.asarray(lift[-1], dtype=np.float64).reshape(7).copy()
    phase1_wp_n = len(lift)

    h1 = _eef_chest_horiz_components(world, arm, q1)
    if h1 is None:
        return {"ok": False, "error": "Phase1 末胸口坐标失败"}
    lat0 = float(h1[1])
    z_phase1 = _eef_world_z_m(world, arm, q1)
    if z_phase1 is None:
        return {"ok": False, "error": "Phase1 末 EEF 高度探针失败"}

    use_z = z_drop_max_m is not None
    z_floor = (float(z_phase1) - float(z_drop_max_m)) if use_z else None

    if use_z:
        q_goal = _resolve_chest_goal_q_j34(
            world, arm,
            max_forward_m=max_forward_m,
            z_phase1_m=float(z_phase1),
            z_floor_m=float(z_floor),
            waist_pivot_world=pivot,
            lat0_m=lat0,
            lateral_outward_max_m=lateral_outward_max_m,
            forbidden_box=forbidden_box,
            z_drop_max_m=float(z_drop_max_m),
        )
    else:
        q_goal = _resolve_chest_goal_j34_relaxed(
            world, arm,
            max_forward_m=max_forward_m,
            lat0_m=lat0,
            lateral_outward_max_m=lateral_outward_max_m,
            forbidden_box=forbidden_box,
            waist_pivot_world=pivot,
        )

    goal_fd = eef_chest_forward_dist_m(world, arm, q_goal)
    goal_zz = _eef_world_z_m(world, arm, q_goal)
    goal_h = _eef_chest_horiz_components(world, arm, q_goal)
    goal_lat_out = (
        _phase2_lateral_outward_delta_m(arm, goal_h[1], lat0)
        if goal_h is not None else None
    )
    goal_forbidden = _gripper_hits_waist_j34_forbidden(world, arm, q_goal, forbidden_box)
    goal_ok = (
        goal_fd is not None and goal_fd <= float(max_forward_m) + 1e-4
        and goal_lat_out is not None
        and goal_lat_out <= float(lateral_outward_max_m) + 1e-4
        and not goal_forbidden
    )
    if use_z:
        goal_ok = goal_ok and (
            goal_zz is not None
            and goal_zz >= float(z_floor) - 1e-5
            and goal_zz < float(z_phase1) - float(TUCK_PHASE2_Z_DESCEND_MIN_M)
        )

    traj = _search_phase2_trajectory_exists(
        world, arm, q1, q_goal,
        waist_pivot_world=pivot,
        max_forward_m=float(max_forward_m),
        waist_min_m=0.0,
        lat0_m=lat0,
        lateral_outward_max_m=float(lateral_outward_max_m),
        z_floor_m=z_floor,
        rrt_iters=int(rrt_iters),
        prm_samples=int(prm_samples),
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        z_phase1_ref_m=float(z_phase1) if use_z else None,
        require_z_descend=bool(use_z),
    )

    z_rules = (
        f"Phase2 Z∈[Phase1末−{round(float(z_drop_max_m) * 100)}cm, Phase1末)，终点须下降"
        if use_z else "Phase2 不约束 Z"
    )
    return {
        "ok": True,
        "arm": arm,
        "trajectory_exists": bool(traj.get("trajectory_exists")),
        "phase1_waypoints": int(phase1_wp_n),
        "constraints": {
            "max_fwd_cm": round(float(max_forward_m) * 100, 1),
            "lateral_outward_max_cm": round(float(lateral_outward_max_m) * 100, 1),
            "z_drop_max_cm": round(float(z_drop_max_m) * 100, 1) if use_z else None,
            "z_rule": z_rules,
            "forbidden_rule": "夹爪 mesh 顶点不得进入腰 j3↔j4 禁飞区",
            "lateral_rule": "左臂Δlat>0、右臂Δlat<0 相对 Phase1 末，朝胸心方向不限",
            "search_mode": "Phase1 完整路径末态 → Phase2 RRT/PRM",
        },
        "phase1_end": {
            "fwd_cm": round(float(h1[0]) * 100, 2),
            "lat_cm": round(lat0 * 100, 2),
            "z_m": round(float(z_phase1), 4),
            "z_floor_m": round(float(z_floor), 4) if z_floor is not None else None,
            "q": np.round(q1, 4).tolist(),
        },
        "chest_goal": {
            "fwd_cm": round(float(goal_fd) * 100, 2) if goal_fd is not None else None,
            "lateral_outward_cm": round(float(goal_lat_out) * 100, 2)
            if goal_lat_out is not None else None,
            "z_m": round(float(goal_zz), 4) if goal_zz is not None else None,
            "z_descend_cm": round((float(z_phase1) - float(goal_zz)) * 100, 2)
            if goal_zz is not None else None,
            "forbidden_hit": bool(goal_forbidden),
            "feasible": bool(goal_ok),
        },
        "trajectory_search": traj,
        "conclusion": "轨迹存在" if traj.get("trajectory_exists") else "轨迹不存在",
    }


def _audit_phase2_three_constraints(world, arm: str) -> dict:
    """FK+RRT：Phase2 三约束轨迹是否存在。

    1. forward < 20cm
    2. 腰距 min > 15cm（相对 torso_joint4 转轴中心）
    3. 左臂向左/右臂向右相对 Phase1 末 ≤ 5cm（向胸心方向不限）
    """
    arm = (arm or "left").strip().lower()
    max_fwd_m = TUCK_EXEC_FORWARD_MAX_M
    waist_min_m = TUCK_PHASE2_WAIST_MIN_M
    lat_max_m = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        return {"ok": False, "error": "无法计算 torso_joint4 转轴中心"}

    sol = _solve_phase1_vertical_c_pose(world, arm)
    if sol is None:
        return {"ok": False, "error": "phase1 终点无解"}
    q1 = sol["q"]
    h1 = _eef_chest_horiz_components(world, arm, q1)
    if h1 is None:
        return {"ok": False, "error": "Phase1 末胸口坐标失败"}
    lat0 = float(h1[1])
    wd1 = _eef_to_waist_top_dist_m(world, arm, q1, pivot)

    q_goal = _resolve_chest_goal_q(
        world, arm, max_forward_m=max_fwd_m, z_floor_m=None,
    )
    goal_fd = eef_chest_forward_dist_m(world, arm, q_goal)
    goal_wd = _eef_to_waist_top_dist_m(world, arm, q_goal, pivot)
    goal_h = _eef_chest_horiz_components(world, arm, q_goal)
    goal_lat_out = (
        _phase2_lateral_outward_delta_m(arm, goal_h[1], lat0)
        if goal_h is not None else None
    )
    goal_ok = (
        goal_fd is not None and goal_fd <= max_fwd_m + 1e-4
        and goal_wd is not None and goal_wd > waist_min_m + 1e-3
        and goal_lat_out is not None and goal_lat_out <= lat_max_m + 1e-4
    )

    traj = _search_phase2_trajectory_exists(
        world, arm, q1, q_goal,
        waist_pivot_world=pivot,
        max_forward_m=max_fwd_m,
        waist_min_m=waist_min_m,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        rrt_iters=8000,
        prm_samples=4500,
    )

    legacy = _plan_chest_approach_path_legacy(
        world, arm, q1, q_goal, max_forward_m=max_fwd_m,
        step_rad=0.038, fine_step_rad=0.016, max_iters=400, goal_tol=0.065,
    )
    legacy_dense = _densify_joint_path(legacy, max_dq=0.038)
    legacy_verify = _verify_phase2_path_constraints(
        world, arm, legacy_dense,
        waist_pivot_world=pivot,
        max_forward_m=max_fwd_m,
        waist_min_m=waist_min_m,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
    )

    return {
        "ok": True,
        "arm": arm,
        "trajectory_exists": bool(traj.get("trajectory_exists")),
        "constraints": {
            "max_fwd_cm": round(max_fwd_m * 100, 1),
            "waist_min_cm": round(waist_min_m * 100, 1),
            "lateral_outward_max_cm": round(lat_max_m * 100, 1),
            "lateral_rule": "左臂Δlat>0、右臂Δlat<0 相对 Phase1 末，朝胸心方向不限",
        },
        "phase1_end": {
            "fwd_cm": round(float(h1[0]) * 100, 2),
            "lat_cm": round(lat0 * 100, 2),
            "waist_dist_cm": round(float(wd1) * 100, 2) if wd1 is not None else None,
        },
        "chest_goal": {
            "fwd_cm": round(float(goal_fd) * 100, 2) if goal_fd is not None else None,
            "waist_dist_cm": round(float(goal_wd) * 100, 2) if goal_wd is not None else None,
            "lateral_outward_cm": round(float(goal_lat_out) * 100, 2)
            if goal_lat_out is not None else None,
            "feasible": bool(goal_ok),
        },
        "trajectory_search": traj,
        "legacy_fwd_only_ref": legacy_verify,
        "conclusion": "轨迹存在" if traj.get("trajectory_exists") else "轨迹不存在",
    }


def _audit_phase2_j34_constraints(
    world, arm: str, *, z_drop_max_m: float = TUCK_PHASE2_Z_DROP_MAX_M,
) -> dict:
    """FK+RRT：Phase2 禁飞区四约束轨迹是否存在。

    1. forward < 20cm
    2. Phase2 控制下降：Z ≥ Phase1末−z_drop，终点低于 Phase1末
    3. 左臂向左/右臂向右外移 ≤ 5cm
    4. 夹爪 mesh 不得触碰腰 j3↔j4 禁飞区
    """
    arm = (arm or "left").strip().lower()
    max_fwd_m = TUCK_EXEC_FORWARD_MAX_M
    lat_max_m = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M
    z_drop_m = float(z_drop_max_m)
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        return {"ok": False, "error": "无法计算 torso_joint4 转轴中心"}
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if forbidden_box is None:
        return {"ok": False, "error": "腰 j3↔j4 禁飞区构盒失败"}

    sol = _solve_phase1_vertical_c_pose(world, arm)
    if sol is None:
        return {"ok": False, "error": "phase1 终点无解"}
    q1 = sol["q"]
    h1 = _eef_chest_horiz_components(world, arm, q1)
    if h1 is None:
        return {"ok": False, "error": "Phase1 末胸口坐标失败"}
    lat0 = float(h1[1])
    z_phase1 = _eef_world_z_m(world, arm, q1)
    if z_phase1 is None:
        return {"ok": False, "error": "Phase1 末 EEF 高度探针失败"}
    z_floor = float(z_phase1) - float(z_drop_m)

    q_goal = _resolve_chest_goal_q_j34(
        world, arm,
        max_forward_m=max_fwd_m,
        z_phase1_m=float(z_phase1),
        z_floor_m=z_floor,
        waist_pivot_world=pivot,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        forbidden_box=forbidden_box,
        z_drop_max_m=z_drop_m,
    )
    goal_fd = eef_chest_forward_dist_m(world, arm, q_goal)
    goal_zz = _eef_world_z_m(world, arm, q_goal)
    goal_h = _eef_chest_horiz_components(world, arm, q_goal)
    goal_lat_out = (
        _phase2_lateral_outward_delta_m(arm, goal_h[1], lat0)
        if goal_h is not None else None
    )
    goal_forbidden = _gripper_hits_waist_j34_forbidden(world, arm, q_goal, forbidden_box)
    goal_ok = (
        goal_fd is not None and goal_fd <= max_fwd_m + 1e-4
        and goal_lat_out is not None and goal_lat_out <= lat_max_m + 1e-4
        and goal_zz is not None and goal_zz >= z_floor - 1e-5
        and goal_zz < float(z_phase1) - float(TUCK_PHASE2_Z_DESCEND_MIN_M)
        and not goal_forbidden
    )

    traj = _search_phase2_trajectory_exists(
        world, arm, q1, q_goal,
        waist_pivot_world=pivot,
        max_forward_m=max_fwd_m,
        waist_min_m=0.0,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        z_floor_m=z_floor,
        rrt_iters=4500,
        prm_samples=2800,
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        z_phase1_ref_m=float(z_phase1),
        require_z_descend=True,
    )

    return {
        "ok": True,
        "arm": arm,
        "trajectory_exists": bool(traj.get("trajectory_exists")),
        "constraints": {
            "max_fwd_cm": round(max_fwd_m * 100, 1),
            "lateral_outward_max_cm": round(lat_max_m * 100, 1),
            "z_drop_max_cm": round(z_drop_m * 100, 1),
            "z_rule": f"Phase2 Z∈[Phase1末−{round(z_drop_m*100)}cm, Phase1末)，终点须下降",
            "forbidden_rule": "夹爪 mesh 顶点不得进入腰 j3↔j4 禁飞区",
            "lateral_rule": "左臂Δlat>0、右臂Δlat<0 相对 Phase1 末，朝胸心方向不限",
        },
        "phase1_end": {
            "fwd_cm": round(float(h1[0]) * 100, 2),
            "lat_cm": round(lat0 * 100, 2),
            "z_m": round(float(z_phase1), 4),
            "z_floor_m": round(z_floor, 4),
        },
        "chest_goal": {
            "fwd_cm": round(float(goal_fd) * 100, 2) if goal_fd is not None else None,
            "lateral_outward_cm": round(float(goal_lat_out) * 100, 2)
            if goal_lat_out is not None else None,
            "z_m": round(float(goal_zz), 4) if goal_zz is not None else None,
            "z_descend_cm": round((float(z_phase1) - float(goal_zz)) * 100, 2)
            if goal_zz is not None else None,
            "forbidden_hit": bool(goal_forbidden),
            "feasible": bool(goal_ok),
        },
        "forbidden_measure": _waist_j34_axis_symmetric_box(
            world, world.trunk_qpos().copy(),
        ).get("measure"),
        "trajectory_search": traj,
        "conclusion": "轨迹存在" if traj.get("trajectory_exists") else "轨迹不存在",
    }


def _audit_phase2_j34_breakdown(
    world, arm: str, *, prm_probe: int = 1200, z_drop_max_m: float = TUCK_PHASE2_Z_DROP_MAX_M,
) -> dict:
    """快速 FK 分解：四约束逐项 + 直线插值 + 可行集采样（不做长 RRT）。"""
    arm = (arm or "left").strip().lower()
    max_fwd_m = TUCK_EXEC_FORWARD_MAX_M
    lat_max_m = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M
    z_drop_m = float(z_drop_max_m)
    pivot = _waist_top_joint4_axis_center_world(world)
    forbidden_box = _waist_j34_forbidden_box_pinned(world)
    if pivot is None or forbidden_box is None:
        return {"ok": False, "arm": arm, "error": "枢轴或禁飞区失败"}

    sol = _solve_phase1_vertical_c_pose(world, arm)
    if sol is None:
        return {"ok": False, "arm": arm, "error": "phase1 终点无解"}
    q1 = sol["q"]
    h1 = _eef_chest_horiz_components(world, arm, q1)
    z_phase1 = _eef_world_z_m(world, arm, q1)
    if h1 is None or z_phase1 is None:
        return {"ok": False, "arm": arm, "error": "Phase1 末探针失败"}
    lat0 = float(h1[1])
    z_floor = float(z_phase1) - float(z_drop_m)

    q_goal = _resolve_chest_goal_q_j34(
        world, arm,
        max_forward_m=max_fwd_m,
        z_phase1_m=float(z_phase1),
        z_floor_m=z_floor,
        waist_pivot_world=pivot,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        forbidden_box=forbidden_box,
        z_drop_max_m=z_drop_m,
    )
    zz_g = _eef_world_z_m(world, arm, q_goal)
    z_ceil = float(z_phase1) - float(TUCK_PHASE2_Z_DESCEND_MIN_M)
    goal_band_ok = (
        zz_g is not None
        and float(zz_g) >= z_floor - 1e-4
        and float(zz_g) < z_ceil
    )
    goal_search_used = False
    if not goal_band_ok:
        q_s = _search_feasible_chest_goal_j34(
            world, arm,
            max_forward_m=max_fwd_m,
            z_phase1_m=float(z_phase1),
            z_floor_m=z_floor,
            waist_pivot_world=pivot,
            lat0_m=lat0,
            lateral_outward_max_m=lat_max_m,
            forbidden_box=forbidden_box,
            max_trials=3000,
            seed=11 if arm == "left" else 23,
        )
        if q_s is not None:
            q_goal = q_s
            goal_search_used = True

    def _pt_report(q: np.ndarray, label: str) -> dict:
        fd = eef_chest_forward_dist_m(world, arm, q)
        zz = _eef_world_z_m(world, arm, q)
        h = _eef_chest_horiz_components(world, arm, q)
        lat_out = (
            _phase2_lateral_outward_delta_m(arm, h[1], lat0)
            if h is not None else None
        )
        forb = _gripper_hits_waist_j34_forbidden(world, arm, q, forbidden_box, fine_mesh=True)
        c_fwd = fd is not None and fd <= max_fwd_m + 1e-4
        c_z_lo = zz is not None and zz >= z_floor - 1e-5
        c_z_desc = (
            zz is not None
            and zz < float(z_phase1) - float(TUCK_PHASE2_Z_DESCEND_MIN_M)
        ) if label == "chest_goal" else True
        c_lat = lat_out is not None and lat_out <= lat_max_m + 1e-4
        c_forb = not forb
        return {
            "label": label,
            "fwd_cm": round(float(fd) * 100, 2) if fd is not None else None,
            "z_m": round(float(zz), 4) if zz is not None else None,
            "z_vs_phase1_cm": round((float(z_phase1) - float(zz)) * 100, 2) if zz is not None else None,
            "lateral_outward_cm": round(float(lat_out) * 100, 2) if lat_out is not None else None,
            "forbidden_hit": bool(forb),
            "c1_fwd_ok": bool(c_fwd),
            "c2_z_floor_ok": bool(c_z_lo),
            "c2_z_descend_ok": bool(c_z_desc) if label == "chest_goal" else None,
            "c3_lat_ok": bool(c_lat),
            "c4_forbidden_ok": bool(c_forb),
            "all_ok": bool(c_fwd and c_z_lo and c_lat and c_forb and (c_z_desc if label == "chest_goal" else True)),
        }

    p1 = _pt_report(q1, "phase1_end")
    cg = _pt_report(q_goal, "chest_goal")

    lerp = _densify_joint_path(_interpolate_arm_qpath(q1, q_goal, 28), max_dq=0.035)
    lerp_verify = _verify_phase2_path_constraints(
        world, arm, lerp,
        waist_pivot_world=pivot,
        max_forward_m=max_fwd_m,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        z_floor_m=z_floor,
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        forbidden_fine_mesh=True,
        z_phase1_ref_m=float(z_phase1),
        require_z_descend=True,
    )

    rng = np.random.default_rng(7 if arm == "left" else 13)
    lo, hi = _R1PRO_ARM_LIMITS.get(arm, _R1PRO_ARM_LIMITS["left"])
    feas_n = 0
    for _ in range(int(prm_probe)):
        q = rng.uniform(lo, hi)
        if _phase2_constraint_point_ok(
            world, arm, q,
            waist_pivot_world=pivot,
            max_forward_m=max_fwd_m,
            strict_waist_min=False,
            lat0_m=lat0,
            lateral_outward_max_m=lat_max_m,
            z_floor_m=z_floor,
            forbidden_box=forbidden_box,
            forbidden_fine_mesh=False,
        ):
            feas_n += 1

    goal_ok = bool(cg.get("all_ok"))
    lerp_ok = bool(lerp_verify.get("ok"))
    traj_possible = bool(goal_ok and lerp_ok)

    traj_search = None
    trajectory_exists_rrt = False
    if goal_ok:
        traj_search = _search_phase2_trajectory_exists(
            world, arm, q1, q_goal,
            waist_pivot_world=pivot,
            max_forward_m=max_fwd_m,
            waist_min_m=0.0,
            lat0_m=lat0,
            lateral_outward_max_m=lat_max_m,
            z_floor_m=z_floor,
            rrt_iters=3500,
            prm_samples=2000,
            strict_waist_min=False,
            forbidden_box=forbidden_box,
            z_phase1_ref_m=float(z_phase1),
            require_z_descend=True,
        )
        trajectory_exists_rrt = bool(traj_search.get("trajectory_exists"))

    blockers: List[str] = []
    if not cg["c1_fwd_ok"]:
        blockers.append("胸前终点 forward≥20cm")
    if not cg["c2_z_floor_ok"]:
        blockers.append(f"胸前终点 Z 低于 Phase1末−{round(z_drop_m*100)}cm")
    if not cg["c2_z_descend_ok"]:
        blockers.append("胸前终点未相对 Phase1末 下降")
    if not cg["c3_lat_ok"]:
        blockers.append("胸前终点外移>5cm")
    if not cg["c4_forbidden_ok"]:
        blockers.append("胸前终点触碰禁飞区")
    if goal_ok and not lerp_ok:
        blockers.append("直线插值路径违反四约束（可行集不连通于起点-终点之间）")

    verdict = "不存在"
    if trajectory_exists_rrt:
        verdict = "存在（RRT/PRM 已找到路径）"
    elif goal_ok and lerp_ok:
        verdict = "可能存在（直线插值已满足，需 RRT 确认一般连通）"
    elif not goal_ok:
        verdict = "不存在（胸前终点不可行）"
    elif goal_ok:
        verdict = "不存在（胸前终点可行但 RRT/PRM 未连通）"

    return {
        "ok": True,
        "arm": arm,
        "z_drop_max_cm": round(z_drop_m * 100, 1),
        "verdict": verdict,
        "trajectory_exists": bool(trajectory_exists_rrt),
        "goal_feasible": goal_ok,
        "straight_lerp_feasible": lerp_ok,
        "trajectory_exists_fk": traj_possible,
        "blockers": blockers,
        "phase1_end": p1,
        "chest_goal": cg,
        "straight_lerp": lerp_verify,
        "trajectory_search": traj_search,
        "goal_search_used": bool(goal_search_used),
        "feasible_sample_rate": round(feas_n / max(int(prm_probe), 1), 4),
        "feasible_samples": feas_n,
        "prm_probe": int(prm_probe),
        "forbidden_lwh_cm": (_waist_j34_axis_symmetric_box(
            world, world.trunk_qpos().copy(),
        ).get("measure") or {}),
    }


def _audit_phase2_four_constraints(world, arm: str) -> dict:
    """FK+RRT：Phase2 四约束轨迹是否存在。

    1. forward < 20cm
    2. 腰距 min > 15cm
    3. 左臂向左/右臂向右相对 Phase1 末 ≤ 5cm
    4. Phase2 EEF 世界系 Z 严格 ≥ Phase1 末 Z
    """
    arm = (arm or "left").strip().lower()
    max_fwd_m = TUCK_EXEC_FORWARD_MAX_M
    waist_min_m = TUCK_PHASE2_WAIST_MIN_M
    lat_max_m = TUCK_PHASE2_LATERAL_OUTWARD_MAX_M
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        return {"ok": False, "error": "无法计算 torso_joint4 转轴中心"}

    sol = _solve_phase1_vertical_c_pose(world, arm)
    if sol is None:
        return {"ok": False, "error": "phase1 终点无解"}
    q1 = sol["q"]
    h1 = _eef_chest_horiz_components(world, arm, q1)
    if h1 is None:
        return {"ok": False, "error": "Phase1 末胸口坐标失败"}
    lat0 = float(h1[1])
    z_floor = _eef_world_z_m(world, arm, q1)
    if z_floor is None:
        return {"ok": False, "error": "Phase1 末 EEF 高度探针失败"}
    wd1 = _eef_to_waist_top_dist_m(world, arm, q1, pivot)

    q_goal = _resolve_chest_goal_q(
        world, arm,
        max_forward_m=max_fwd_m,
        z_floor_m=float(z_floor),
    )
    goal_fd = eef_chest_forward_dist_m(world, arm, q_goal)
    goal_wd = _eef_to_waist_top_dist_m(world, arm, q_goal, pivot)
    goal_zz = _eef_world_z_m(world, arm, q_goal)
    goal_h = _eef_chest_horiz_components(world, arm, q_goal)
    goal_lat_out = (
        _phase2_lateral_outward_delta_m(arm, goal_h[1], lat0)
        if goal_h is not None else None
    )
    goal_ok = (
        goal_fd is not None and goal_fd <= max_fwd_m + 1e-4
        and goal_wd is not None and goal_wd > waist_min_m + 1e-3
        and goal_lat_out is not None and goal_lat_out <= lat_max_m + 1e-4
        and goal_zz is not None and goal_zz >= float(z_floor) - 1e-5
    )

    traj = _search_phase2_trajectory_exists(
        world, arm, q1, q_goal,
        waist_pivot_world=pivot,
        max_forward_m=max_fwd_m,
        waist_min_m=waist_min_m,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        z_floor_m=float(z_floor),
        rrt_iters=8000,
        prm_samples=4500,
    )

    legacy = _plan_chest_approach_path_legacy(
        world, arm, q1, q_goal, max_forward_m=max_fwd_m,
        step_rad=0.038, fine_step_rad=0.016, max_iters=400, goal_tol=0.065,
    )
    legacy_dense = _densify_joint_path(legacy, max_dq=0.038)
    legacy_verify = _verify_phase2_path_constraints(
        world, arm, legacy_dense,
        waist_pivot_world=pivot,
        max_forward_m=max_fwd_m,
        waist_min_m=waist_min_m,
        lat0_m=lat0,
        lateral_outward_max_m=lat_max_m,
        z_floor_m=float(z_floor),
    )

    return {
        "ok": True,
        "arm": arm,
        "trajectory_exists": bool(traj.get("trajectory_exists")),
        "constraints": {
            "max_fwd_cm": round(max_fwd_m * 100, 1),
            "waist_min_cm": round(waist_min_m * 100, 1),
            "lateral_outward_max_cm": round(lat_max_m * 100, 1),
            "z_floor_m": round(float(z_floor), 4),
            "lateral_rule": "左臂Δlat>0、右臂Δlat<0 相对 Phase1 末，朝胸心方向不限",
            "z_rule": "Phase2 全程 EEF Z ≥ Phase1 末 Z（严格）",
        },
        "phase1_end": {
            "fwd_cm": round(float(h1[0]) * 100, 2),
            "lat_cm": round(lat0 * 100, 2),
            "waist_dist_cm": round(float(wd1) * 100, 2) if wd1 is not None else None,
            "z_m": round(float(z_floor), 4),
        },
        "chest_goal": {
            "fwd_cm": round(float(goal_fd) * 100, 2) if goal_fd is not None else None,
            "waist_dist_cm": round(float(goal_wd) * 100, 2) if goal_wd is not None else None,
            "lateral_outward_cm": round(float(goal_lat_out) * 100, 2)
            if goal_lat_out is not None else None,
            "z_m": round(float(goal_zz), 4) if goal_zz is not None else None,
            "feasible": bool(goal_ok),
        },
        "trajectory_search": traj,
        "legacy_fwd_only_ref": legacy_verify,
        "conclusion": "轨迹存在" if traj.get("trajectory_exists") else "轨迹不存在",
    }


def _audit_phase2_fwd_waist_feasibility(world, arm: str) -> dict:
    """FK：forward<20cm + Phase2 腰距最小值 > Phase1 末 d0。"""
    arm = (arm or "left").strip().lower()
    max_fwd_m = TUCK_EXEC_FORWARD_MAX_M
    pivot_verify = _verify_waist_top_joint4_pivot(world)
    pivot = _waist_top_joint4_axis_center_world(world)
    if pivot is None:
        return {"ok": False, "error": "无法计算 torso_joint4 转轴中心"}

    sol = _solve_phase1_vertical_c_pose(world, arm)
    if sol is None:
        return {"ok": False, "error": "phase1 终点无解"}
    q1 = sol["q"]
    d0 = _eef_to_waist_top_dist_m(world, arm, q1, pivot)
    if d0 is None:
        return {"ok": False, "error": "Phase1 末 EEF 探针失败"}

    q_goal = _resolve_chest_goal_q(
        world, arm, max_forward_m=max_fwd_m, z_floor_m=None,
    )
    goal_fd = eef_chest_forward_dist_m(world, arm, q_goal)
    goal_wd = _eef_to_waist_top_dist_m(world, arm, q_goal, pivot)
    goal_ok = (
        goal_fd is not None and goal_fd <= max_fwd_m + 1e-4
        and goal_wd is not None and goal_wd > float(d0) + 1e-3
    )

    lerp = _densify_joint_path(
        _interpolate_arm_qpath(q1, q_goal, 24), max_dq=0.04,
    )
    lerp_env = _sample_path_fwd_waist_envelope(
        world, arm, lerp,
        waist_pivot_world=pivot, waist_d0_m=float(d0), max_forward_m=max_fwd_m,
    )
    legacy = _plan_chest_approach_path_legacy(
        world, arm, q1, q_goal, max_forward_m=max_fwd_m,
        step_rad=0.038, fine_step_rad=0.016, max_iters=400, goal_tol=0.065,
    )
    legacy_dense = _densify_joint_path(legacy, max_dq=0.038)
    legacy_env = _sample_path_fwd_waist_envelope(
        world, arm, legacy_dense,
        waist_pivot_world=pivot, waist_d0_m=float(d0), max_forward_m=max_fwd_m,
    )
    legacy_reached = float(np.linalg.norm(legacy[-1] - q_goal, ord=np.inf)) <= 0.07

    waist_plan = _plan_chest_approach_path_legacy_waist(
        world, arm, q1, q_goal,
        max_forward_m=max_fwd_m,
        waist_pivot_world=pivot,
        waist_d0_m=float(d0),
    )
    waist_dense = _densify_joint_path(waist_plan, max_dq=0.038)
    waist_env = _sample_path_fwd_waist_envelope(
        world, arm, waist_dense,
        waist_pivot_world=pivot, waist_d0_m=float(d0), max_forward_m=max_fwd_m,
    )
    waist_reached = float(np.linalg.norm(waist_plan[-1] - q_goal, ord=np.inf)) <= 0.07

    # 在 forward<20cm 且腰距>d0 下搜索最大 forward
    best_fwd_above_waist = {"fwd_cm": 0.0, "waist_cm": round(float(d0) * 100, 2)}
    q_scan = q1.copy()
    for _ in range(360):
        improved = False
        for dq_mag in (0.04, 0.02, 0.01):
            for j in range(7):
                for sign in (-1.0, 1.0):
                    qtry = q_scan.copy()
                    qtry[j] += sign * dq_mag
                    fd = eef_chest_forward_dist_m(world, arm, qtry)
                    wd = _eef_to_waist_top_dist_m(world, arm, qtry, pivot)
                    if fd is None or wd is None:
                        continue
                    if float(wd) <= float(d0) + 1e-3:
                        continue
                    if float(fd) > max_fwd_m + 1e-4:
                        continue
                    if float(fd) > best_fwd_above_waist["fwd_cm"] / 100.0 + 1e-5:
                        best_fwd_above_waist = {
                            "fwd_cm": round(float(fd) * 100, 2),
                            "waist_cm": round(float(wd) * 100, 2),
                            "q": np.round(qtry, 4).tolist(),
                        }
                        q_scan = qtry
                        improved = True
            if improved:
                break

    traj = _search_phase2_trajectory_exists(
        world, arm, q1, q_goal,
        waist_pivot_world=pivot,
        waist_d0_m=float(d0),
        max_forward_m=max_fwd_m,
    )

    return {
        "ok": True,
        "arm": arm,
        "trajectory_exists": bool(traj.get("trajectory_exists")),
        "trajectory_search": traj,
        "pivot_verify": pivot_verify,
        "phase1_end_waist_dist_cm": round(float(d0) * 100.0, 2),
        "chest_goal": {
            "fwd_cm": round(float(goal_fd) * 100, 2) if goal_fd is not None else None,
            "waist_dist_cm": round(float(goal_wd) * 100, 2) if goal_wd is not None else None,
            "feasible": bool(goal_ok),
        },
        "straight_lerp": {**lerp_env, "waypoints_n": len(lerp)},
        "legacy_fwd_only": {
            **legacy_env,
            "waypoints_n": len(legacy_dense),
            "reached_goal": bool(legacy_reached),
        },
        "legacy_fwd_waist": {
            **waist_env,
            "waypoints_n": len(waist_dense),
            "reached_goal": bool(waist_reached),
        },
        "goal_feasible": bool(goal_ok),
        "path_feasible": bool(
            (legacy_env.get("ok") and legacy_reached)
            or (waist_env.get("ok") and waist_reached)
        ),
        "max_fwd_with_waist_gt_d0": best_fwd_above_waist,
        "conclusion": (
            "轨迹存在"
            if traj.get("trajectory_exists")
            else "轨迹不存在"
        ),
    }


def _grip_chest_metrics(world, arm: str) -> Dict[str, Optional[float]]:
    """夹爪相对胸廓/躯干多参考系距离。

    注意：grip_chest_frame_norm = ||eef||_{link4} 是 link4 原点→夹爪，不是肚子净空；
    标定用的胸口 forward 投影见 grip_chest_fwd_m（与 eef_chest_forward_dist_m 一致）。
    """
    eef = np.asarray(world.eef_pose(arm=arm)["pos"], dtype=np.float64).reshape(3)
    in_chest = _pos_in_link_frame(world, "torso_link4", eef)
    in_shoulder = _pos_in_link_frame(world, f"{arm}_arm_link1", eef)
    in_belly = _pos_in_link_frame(world, "torso_link2", eef)
    cpos = _link_world_pos(world, "torso_link4")
    grip_chest_world_m = (
        float(np.linalg.norm(eef - cpos)) if cpos is not None else None
    )
    chest_fwd = None
    try:
        chest, fwd = _chest_forward_axis(world)
        chest_fwd = float(np.dot(eef - chest, fwd))
    except Exception:
        pass
    belly_world_m = None
    bpos = _link_world_pos(world, "torso_link2")
    if bpos is not None:
        belly_world_m = float(np.linalg.norm(eef - bpos))
    return {
        "grip_chest_frame_norm_m": (
            float(np.linalg.norm(in_chest)) if in_chest is not None else None
        ),
        "grip_chest_world_m": grip_chest_world_m,
        "grip_chest_fwd_m": chest_fwd,
        "grip_belly_frame_norm_m": (
            float(np.linalg.norm(in_belly)) if in_belly is not None else None
        ),
        "grip_belly_world_m": belly_world_m,
        "grip_shoulder_frame_norm_m": (
            float(np.linalg.norm(in_shoulder)) if in_shoulder is not None else None
        ),
        "trunk_theta_z_deg": float(world.chest_pose().get("theta_z_deg", 90.0)),
    }


def _tuck_frame_metrics(world, arm: str) -> Dict[str, Optional[float]]:
    """兼容旧字段名。"""
    m = _grip_chest_metrics(world, arm)
    return {
        **m,
        "eef_link4_norm_m": m.get("grip_chest_frame_norm_m"),
        "eef_shoulder_norm_m": m.get("grip_shoulder_frame_norm_m"),
        "chest_fwd_m": m.get("grip_chest_fwd_m"),
        "eef_lift_norm_m": m.get("grip_chest_frame_norm_m"),
    }


def _fk_probe_one(
    world,
    arm: str,
    arm_q: np.ndarray,
    trunk_q: np.ndarray,
) -> Dict[str, float]:
    """单点 FK 探针，并校验 trunk 关节是否写入成功。"""
    from behavior_interface.skills.grasp import _get_arm_dof_idx, _to_np

    robot = world.robot
    saved = robot.get_joint_positions().clone()
    arm_idx = _get_arm_dof_idx(world, arm)
    trunk_idx = _to_np(robot.trunk_control_idx).astype(int)
    trunk_q = np.asarray(trunk_q, dtype=np.float64).reshape(len(trunk_idx))
    try:
        q = saved.clone()
        for ti, j in enumerate(trunk_idx):
            q[int(j)] = float(trunk_q[ti])
        for i, j in enumerate(arm_idx):
            q[int(j)] = float(np.asarray(arm_q, dtype=np.float64).reshape(7)[i])
        robot.set_joint_positions(q)
        read_trunk = world.trunk_qpos()
        trunk_err = float(np.linalg.norm(read_trunk - trunk_q, ord=np.inf))
        m = _grip_chest_metrics(world, arm)
        m["trunk_set_err_inf"] = trunk_err
        return {k: float(v) for k, v in m.items() if v is not None}
    finally:
        robot.set_joint_positions(saved)


def _fk_grip_chest_max_along_path(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    trunk_q: np.ndarray,
    *,
    samples: int = 48,
) -> Dict[str, float]:
    """FK：固定 trunk 关节 + 沿折线设 arm 关节，统计夹爪相对胸廓距离峰值。"""
    from behavior_interface.skills.grasp import _get_arm_dof_idx, _to_np

    saved = None
    robot = world.robot
    keys = (
        "grip_chest_frame_norm_m",
        "grip_chest_world_m",
        "grip_chest_fwd_m",
        "grip_belly_frame_norm_m",
        "grip_belly_world_m",
        "grip_shoulder_frame_norm_m",
    )
    worst = {k: 0.0 for k in keys}
    try:
        saved = robot.get_joint_positions().clone()
        arm_idx = _get_arm_dof_idx(world, arm)
        trunk_idx = _to_np(robot.trunk_control_idx).astype(int)
        trunk_q = np.asarray(trunk_q, dtype=np.float64).reshape(len(trunk_idx))
        cum = _path_arc_cum(path)
        total = float(cum[-1])
        n = max(int(samples), len(path) * 4)
        for k in range(n + 1):
            s = total * k / n
            aq = _interp_path_at_s(path, cum, s)
            q = saved.clone()
            for ti, j in enumerate(trunk_idx):
                q[int(j)] = float(trunk_q[ti])
            for i, j in enumerate(arm_idx):
                q[int(j)] = float(aq[i])
            robot.set_joint_positions(q)
            m = _grip_chest_metrics(world, arm)
            for key in keys:
                v = m.get(key)
                if v is not None:
                    worst[key] = max(worst[key], float(v))
    finally:
        if saved is not None:
            try:
                robot.set_joint_positions(saved)
            except Exception:
                pass
    return worst


def _kinematic_path_max_in_frames(
    world,
    arm: str,
    path: Sequence[np.ndarray],
) -> Dict[str, float]:
    """无物理：沿路径设 arm 关节，统计峰值（当前 trunk 姿态）。"""
    try:
        trunk_q = world.trunk_qpos()
    except Exception:
        trunk_q = np.zeros(4)
    fk = _fk_grip_chest_max_along_path(world, arm, path, trunk_q)
    return {
        "eef_lift_norm_m": fk["grip_chest_frame_norm_m"],
        "eef_link4_norm_m": fk["grip_chest_frame_norm_m"],
        "eef_shoulder_norm_m": fk["grip_shoulder_frame_norm_m"],
        "chest_fwd_m": fk["grip_chest_fwd_m"],
        **fk,
    }


def _summarize_grip_chest_log(frame_log: List[Dict]) -> Dict[str, float]:
    keys = (
        "grip_chest_frame_norm_m",
        "grip_chest_world_m",
        "grip_chest_fwd_m",
        "grip_belly_frame_norm_m",
        "grip_belly_world_m",
        "grip_shoulder_frame_norm_m",
    )
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(fr[k]) for fr in frame_log if k in fr]
        if vals:
            out[f"max_{k}"] = max(vals)
    return out


def _tuck_make_action(
    world,
    arm: str,
    arm_q,
    gripper_cmd: float,
    trunk_hold: Optional[np.ndarray],
):
    """tuck 播放：锁 trunk + 绝对 arm 关节角。"""
    from behavior_interface.skills.eef import _make_legacy_7dof_action

    kw = {f"arm_{arm}": arm_q, f"gripper_{arm}": [gripper_cmd]}
    if trunk_hold is not None:
        kw["trunk"] = np.asarray(trunk_hold, dtype=np.float64).reshape(-1).tolist()
    return _make_legacy_7dof_action(world, **kw)


def _yield_play_waypoint_path(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    gripper_cmd: float,
    ctx,
    *,
    frames_per_wp: int = TUCK_FRAMES_PER_WP,
    frame_log: Optional[List[Dict[str, float]]] = None,
    video_rec: Optional[_GtaVideoRecorder] = None,
):
    """逐路点发绝对关节角，每点保持若干帧直至收敛。"""
    from behavior_interface.skills.arm_reset import _arm_qpos

    path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
    if not path:
        return
    ctx.log(f"  [tuck] 路点播放 wp={len(path)} frames/wp={frames_per_wp}")
    trunk_hold = world.trunk_qpos().copy()
    trunk0 = trunk_hold.copy() if frame_log is not None else None
    fi = 0
    if video_rec is not None:
        video_rec.capture(world)
    for wi, q_ref in enumerate(path):
        q_cmd = q_ref.tolist()
        err_inf = 999.0
        for _ in range(int(frames_per_wp)):
            yield _tuck_make_action(world, arm, q_cmd, gripper_cmd, trunk_hold)
            try:
                cur = _arm_qpos(world, arm)
                err_inf = float(np.linalg.norm(q_ref - cur, ord=np.inf))
            except Exception:
                err_inf = 999.0
            if video_rec is not None:
                video_rec.capture(world)
            if frame_log is not None:
                fm = _grip_chest_metrics(world, arm)
                epos = world.eef_pose(arm=arm)
                sc_pairs = _self_collision_pairs(world)
                prefix = f"{arm}_"
                arm_sc = {
                    tuple(sorted((a, b)))
                    for a, b in sc_pairs
                    if a.startswith(prefix) or b.startswith(prefix)
                    or (a.startswith("torso_") and b.startswith(prefix))
                }
                frame_log.append({
                    "fi": float(fi), "wp": float(wi), "err_inf": err_inf,
                    "self_collide": float(bool(arm_sc)),
                    "eef_z_m": float(epos["pos"][2]),
                    **{k: v for k, v in fm.items() if v is not None},
                })
                fi += 1
            if err_inf < float(TUCK_POS_TOL_RAD):
                break
    q_fin = path[-1].tolist()
    for _ in range(int(TUCK_HOLD_FRAMES)):
        yield _tuck_make_action(world, arm, q_fin, gripper_cmd, trunk_hold)
        if video_rec is not None:
            video_rec.capture(world)
    if frame_log is not None and trunk0 is not None:
        frame_log.append({
            "trunk_delta_inf": float(np.linalg.norm(world.trunk_qpos() - trunk0, ord=np.inf)),
        })


def _yield_play_tuck_two_segments(
    world,
    arm: str,
    phase1_path: Sequence[np.ndarray],
    phase2_path: Sequence[np.ndarray],
    gripper_cmd: float,
    ctx,
    *,
    frame_log: Optional[List[Dict[str, float]]] = None,
    video_rec: Optional[_GtaVideoRecorder] = None,
    phase1_frames_per_wp: int = TUCK_FRAMES_PER_WP,
):
    """分两段播放：Phase1 逐路点收敛（与 rrt_three_constraints 一致可见），停稳后 Phase2 弧长平滑。"""
    p1 = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in phase1_path]
    p2 = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in phase2_path]
    if not p1 or not p2:
        return
    trunk_hold = world.trunk_qpos().copy()
    if video_rec is not None:
        video_rec.capture(world)
    ctx.log(
        f"  [tuck] 两段播放 Phase1 路点 wp={len(p1)} j1 {p1[0][0]:.3f}→{p1[-1][0]:.3f} "
        f"j4 {p1[0][3]:.3f}→{p1[-1][3]:.3f} frames/wp={phase1_frames_per_wp} | Phase2 wp={len(p2)}"
    )
    yield from _yield_play_waypoint_path(
        world, arm, p1, gripper_cmd, ctx,
        frames_per_wp=int(phase1_frames_per_wp),
        frame_log=frame_log,
        video_rec=video_rec,
    )
    q_hold = p1[-1].tolist()
    for _ in range(int(TUCK_PHASE_HOLD_FRAMES)):
        yield _tuck_make_action(world, arm, q_hold, gripper_cmd, trunk_hold)
        if video_rec is not None:
            video_rec.capture(world)
    p2_play = p2
    if float(np.linalg.norm(p2[0] - p1[-1], ord=np.inf)) < 0.01 and len(p2) >= 2:
        p2_play = p2[1:]
    yield from _yield_play_tuck_smooth(
        world, arm, p2_play, gripper_cmd, ctx,
        frame_log=frame_log,
        video_rec=video_rec,
        lift_wp_n=None,
    )


def _yield_play_tuck_smooth(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    gripper_cmd: float,
    ctx,
    *,
    frame_log: Optional[List[Dict[str, float]]] = None,
    video_rec: Optional[_GtaVideoRecorder] = None,
    lift_wp_n: Optional[int] = None,
):
    """沿关节弧长余弦缓动播放，限制每帧 |Δq|，运动更平顺。"""
    from behavior_interface.skills.arm_reset import _arm_qpos

    path = [np.asarray(q, dtype=np.float64).reshape(7).copy() for q in path]
    if not path:
        return
    cum = _path_arc_cum(path)
    total = float(cum[-1])
    n_play = max(
        int(TUCK_PLAY_FRAMES_MIN),
        int(np.ceil(total / float(TUCK_PLAY_DQ_PER_FRAME))) + 12,
    )
    ctx.log(
        f"  [tuck] 弧长平滑播放 wp={len(path)} 弧长={total:.2f}rad 帧数≈{n_play}"
    )
    trunk_hold = world.trunk_qpos().copy()
    trunk0 = trunk_hold.copy() if frame_log is not None else None
    if video_rec is not None:
        video_rec.capture(world)
    q_prev = path[0].copy()
    s_phase2 = None
    if lift_wp_n is not None and lift_wp_n > 0:
        s_phase2 = float(cum[min(int(lift_wp_n) - 1, len(cum) - 1)])
    phase2_started = False
    s = 0.0
    ds_base = total / float(max(n_play, 1))
    fi = 0
    max_fi = int(n_play * 2.5)
    while s < total - 1e-6 and fi < max_fi:
        if (
            s_phase2 is not None
            and not phase2_started
            and s >= s_phase2 - 1e-6
        ):
            phase2_started = True
            q_hold = _interp_path_at_s(path, cum, s_phase2).tolist()
            for _ in range(int(TUCK_PHASE_HOLD_FRAMES)):
                yield _tuck_make_action(world, arm, q_hold, gripper_cmd, trunk_hold)
                if video_rec is not None:
                    video_rec.capture(world)
        prog = s / total if total > 1e-6 else 1.0
        ease = 0.7 + 0.3 * float(np.sin(np.pi * prog))
        ds = ds_base * ease
        s_tgt = min(total, s + ds)
        q_ref = _interp_path_at_s(path, cum, s_tgt)
        dq_step = float(np.linalg.norm(q_ref - q_prev, ord=np.inf))
        if dq_step > float(TUCK_PLAY_DQ_PER_FRAME) + 1e-6 and fi > 0:
            alpha = float(TUCK_PLAY_DQ_PER_FRAME) / dq_step
            q_ref = q_prev + alpha * (q_ref - q_prev)
            s_tgt = _project_onto_path_arc(q_ref, path, cum)
        q_prev = q_ref.copy()
        s = float(s_tgt)
        q_cmd = q_ref.tolist()
        yield _tuck_make_action(world, arm, q_cmd, gripper_cmd, trunk_hold)
        err_inf = 999.0
        try:
            cur = _arm_qpos(world, arm)
            err_inf = float(np.linalg.norm(q_ref - cur, ord=np.inf))
        except Exception:
            cur = q_ref
        if video_rec is not None:
            video_rec.capture(world)
        fm = _grip_chest_metrics(world, arm)
        if frame_log is not None:
            epos = world.eef_pose(arm=arm)
            sc_pairs = _self_collision_pairs(world)
            prefix = f"{arm}_"
            arm_sc = {
                tuple(sorted((a, b)))
                for a, b in sc_pairs
                if a.startswith(prefix) or b.startswith(prefix)
                or (a.startswith("torso_") and b.startswith(prefix))
            }
            phase = 2 if phase2_started else 1
            frame_log.append({
                "fi": float(fi),
                "s": float(s),
                "phase": float(phase),
                "err_inf": err_inf,
                "self_collide": float(bool(arm_sc)),
                "eef_z_m": float(epos["pos"][2]),
                **{k: v for k, v in fm.items() if v is not None},
            })
        fi += 1

    q_fin = path[-1].tolist()
    for _ in range(int(TUCK_HOLD_FRAMES)):
        yield _tuck_make_action(world, arm, q_fin, gripper_cmd, trunk_hold)
        if video_rec is not None:
            video_rec.capture(world)
    if frame_log is not None and trunk0 is not None:
        frame_log.append({
            "trunk_delta_inf": float(np.linalg.norm(world.trunk_qpos() - trunk0, ord=np.inf)),
        })


def _yield_play_joint_path(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    gripper_cmd: float,
    ctx,
    *,
    tag: str = "",
    frame_log: Optional[List[Dict[str, float]]] = None,
):
    """兼容旧诊断：委托弧长平滑播放。"""
    yield from _yield_play_tuck_smooth(
        world, arm, path, gripper_cmd, ctx,
        frame_log=frame_log,
    )


def yield_tuck_trajectory(world, arm: str, gripper_cmd: float, ctx):
    """收臂至胸前：两阶段在线规划后播放关节折线（只动 arm）。"""
    from behavior_interface.skills.arm_reset import _arm_qpos
    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    arm = (arm or "right").strip().lower()
    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name="tuck_trajectory.prepare"
    )
    chest = world.chest_pose()
    theta_z = float(chest.get("theta_z_deg", 90.0))
    try:
        q_start = _arm_qpos(world, arm)
    except Exception:
        q_start = None

    path, lift_wp_n = build_two_phase_tuck_path(
        world, arm, q_start=q_start, max_forward_m=TUCK_EXEC_FORWARD_MAX_M,
    )
    fk_rep = verify_tuck_path(
        world, arm, path, max_forward_m=TUCK_PLAN_FORWARD_MAX_M, samples=20,
    )
    arc_rad = float(_path_arc_cum(path)[-1]) if path else 0.0
    play_frames = max(
        int(TUCK_PLAY_FRAMES_MIN),
        int(np.ceil(arc_rad / float(TUCK_PLAY_DQ_PER_FRAME))) + 12,
    )
    z0 = _eef_world_z_m(world, arm, path[0])
    lift_i = min(lift_wp_n, len(path)) - 1
    z_lift = _eef_world_z_m(world, arm, path[max(0, lift_i)])
    z_end = _eef_world_z_m(world, arm, path[-1])
    dz_cm = (z_lift - z0) * 100.0 if z0 is not None and z_lift is not None else 0.0
    z_end_s = f"{z_end:.3f}" if z_end is not None else "?"
    ctx.log(
        f"  [tuck] 两阶段规划 wp={len(path)} trunk_θz={theta_z:.1f}° "
        f"FK峰值forward={fk_rep['max_forward_cm']:.1f}cm "
        f"Δz抬升={dz_cm:.1f}cm 终点z={z_end_s}m "
        f"弧长={arc_rad:.2f}rad 预计帧={play_frames}"
    )
    reject_reasons: List[str] = []
    if len(path) > int(TUCK_EXEC_MAX_WAYPOINTS):
        reject_reasons.append(f"wp={len(path)}>{int(TUCK_EXEC_MAX_WAYPOINTS)}")
    if play_frames > int(TUCK_EXEC_MAX_PLAY_FRAMES):
        reject_reasons.append(
            f"frames={play_frames}>{int(TUCK_EXEC_MAX_PLAY_FRAMES)}"
        )
    if arc_rad > float(TUCK_EXEC_MAX_ARC_RAD):
        reject_reasons.append(f"arc={arc_rad:.2f}>{float(TUCK_EXEC_MAX_ARC_RAD):.2f}")
    if float(fk_rep["max_forward_m"]) > float(TUCK_EXEC_FORWARD_MAX_M) + 1e-3:
        reject_reasons.append(
            f"forward={fk_rep['max_forward_cm']:.1f}cm>"
            f"{TUCK_EXEC_FORWARD_MAX_M * 100.0:.0f}cm"
        )
    if z0 is not None and z_lift is not None and (float(z_lift) - float(z0)) < -0.03:
        reject_reasons.append(f"lift_dz={dz_cm:.1f}cm< -3cm")
    if reject_reasons:
        ctx.log(
            "  [tuck] SKIP 异常轨迹，改由 exec fallback: "
            + ", ".join(reject_reasons)
        )
        return False

    frame_log: List[Dict[str, float]] = []
    yield from _yield_play_tuck_smooth(
        world, arm, path, gripper_cmd, ctx,
        frame_log=frame_log,
        lift_wp_n=lift_wp_n,
    )
    dyn = _summarize_grip_chest_log(frame_log)
    peak_fwd = float(dyn.get("max_grip_chest_fwd_m", fk_rep["max_forward_m"]))
    sc_any = any(bool(fr.get("self_collide")) for fr in frame_log)
    ctx.log(
        f"  [tuck] 动态峰值forward={peak_fwd * 100:.1f}cm "
        f"≤20cm={'✅' if peak_fwd <= TUCK_EXEC_FORWARD_MAX_M + 1e-3 else '✗'} "
        f"自碰={'✗' if sc_any else '✅无'}"
    )
    return True


def _summarize_frame_log(frame_log: List[Dict]) -> Dict[str, float]:
    """从逐帧日志汇总峰值。"""
    keys = (
        "eef_lift_norm_m", "eef_link4_norm_m", "eef_shoulder_norm_m",
        "chest_fwd_m", "err_inf", "s",
    )
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(fr[k]) for fr in frame_log if k in fr]
        if vals:
            out[f"max_{k}"] = max(vals)
            if k == "s":
                out["final_s"] = vals[-1]
    if frame_log:
        td = frame_log[-1].get("trunk_delta_inf")
        if td is not None:
            out["trunk_delta_inf"] = float(td)
    return out


@register_skill(
    "diag_tuck_chest_now",
    description="仅当前躯干：播折线测夹爪-胸廓最远距离（不改 trunk、不 reset）",
)
def diag_tuck_chest_now(ctx, arm: str = "left", play: bool = True):
    """快照当前 trunk，FK+动态播折线；直立基线仅用 FK(全零 trunk) 对比，不动仿真腰。"""
    from behavior_interface.skills.arm_reset import arm_reset

    world = ctx.world
    arm = (arm or "left").strip().lower()
    path = _TUCK_TRAJECTORY.get(arm) or _TUCK_TRAJECTORY["right"]

    yield world.hold_action()
    trunk_now = world.trunk_qpos().copy()
    ch = world.chest_pose()
    theta_now = float(ch.get("theta_z_deg", 90.0))
    ctx.log(
        f"[chest_now] 当前躯干 θz={theta_now:.1f}° "
        f"trunk={np.round(trunk_now, 4).tolist()} — 不 reset、不改 pitch"
    )

    fk_now = _fk_grip_chest_max_along_path(world, arm, path, trunk_now)
    trunk_up = np.zeros(4, dtype=np.float64)
    fk_up = _fk_grip_chest_max_along_path(world, arm, path, trunk_up)

    dyn_now: Dict[str, float] = {}
    if play:
        yield from arm_reset(ctx, arm=arm, mode="hang", open_gripper=True)
        trunk_after_reset = world.trunk_qpos().copy()
        ctx.log(
            f"[chest_now] arm_reset 后 trunk="
            f"{np.round(trunk_after_reset, 4).tolist()} "
            f"Δ={float(np.linalg.norm(trunk_after_reset - trunk_now, ord=np.inf)):.4f}rad"
        )
        fl: List[Dict] = []
        yield from _yield_play_joint_path(
            world, arm, path, 1.0, ctx, tag="_now", frame_log=fl,
        )
        dyn_now = _summarize_grip_chest_log(fl)

    def _cm(d: Dict[str, float], k: str) -> float:
        return round(float(d.get(k, 0.0)) * 100.0, 2)

    # ── forward 峰值帧逐关节剖析：证明 28.7cm 来自 q_cur 偏离 q_ref ──────────
    peak = None
    if play and fl:
        frames = [fr for fr in fl if "grip_chest_fwd_m" in fr and "q_cur" in fr]
        if frames:
            pk = max(frames, key=lambda fr: float(fr["grip_chest_fwd_m"]))
            q_ref = np.asarray(pk["q_ref"], dtype=np.float64).reshape(7)
            q_cur = np.asarray(pk["q_cur"], dtype=np.float64).reshape(7)
            per_joint_err = (q_ref - q_cur)
            # 交叉验证：用峰值帧 q_ref / q_cur 在当前 trunk 下做 FK forward
            fwd_from_ref = _fk_probe_one(world, arm, q_ref, trunk_now).get("grip_chest_fwd_m")
            fwd_from_cur = _fk_probe_one(world, arm, q_cur, trunk_now).get("grip_chest_fwd_m")
            peak = {
                "fi": pk.get("fi"),
                "s": round(float(pk.get("s", 0.0)), 3),
                "dyn_fwd_cm": round(float(pk["grip_chest_fwd_m"]) * 100, 2),
                "q_ref": np.round(q_ref, 3).tolist(),
                "q_cur": np.round(q_cur, 3).tolist(),
                "per_joint_err_rad": np.round(per_joint_err, 3).tolist(),
                "err_inf_rad": round(float(np.max(np.abs(per_joint_err))), 3),
                "fk_fwd_from_qref_cm": round(float(fwd_from_ref) * 100, 2)
                if fwd_from_ref is not None else None,
                "fk_fwd_from_qcur_cm": round(float(fwd_from_cur) * 100, 2)
                if fwd_from_cur is not None else None,
            }
            ctx.log(
                f"[chest_now] forward峰值帧 fi={peak['fi']} s={peak['s']} "
                f"dyn_fwd={peak['dyn_fwd_cm']}cm err_inf={peak['err_inf_rad']}rad"
            )
            ctx.log(
                f"[chest_now]   q_ref={peak['q_ref']}"
            )
            ctx.log(
                f"[chest_now]   q_cur={peak['q_cur']}"
            )
            ctx.log(
                f"[chest_now]   逐关节误差(ref-cur)={peak['per_joint_err_rad']}"
            )
            ctx.log(
                f"[chest_now]   FK校验: forward(q_ref)={peak['fk_fwd_from_qref_cm']}cm "
                f"forward(q_cur)={peak['fk_fwd_from_qcur_cm']}cm "
                f"(后者≈动态{peak['dyn_fwd_cm']}cm → 偏差全来自 q_cur≠q_ref)"
            )

    ctx.log(
        f"[chest_now] FK 当前 forward峰值={_cm(fk_now,'grip_chest_fwd_m')}cm "
        f"直立基线={_cm(fk_up,'grip_chest_fwd_m')}cm | "
        f"世界距胸口={_cm(fk_now,'grip_chest_world_m')} vs {_cm(fk_up,'grip_chest_world_m')}cm"
    )
    if dyn_now:
        ctx.log(
            f"[chest_now] 动态 当前 forward峰值="
            f"{dyn_now.get('max_grip_chest_fwd_m', 0)*100:.2f}cm "
            f"link4={dyn_now.get('max_grip_chest_frame_norm_m', 0)*100:.2f}cm "
            f"belly={dyn_now.get('max_grip_belly_world_m', 0)*100:.2f}cm"
        )

    d_fk_fwd = _cm(fk_now, "grip_chest_fwd_m") - _cm(fk_up, "grip_chest_fwd_m")
    d_dyn_fwd = (
        dyn_now.get("max_grip_chest_fwd_m", 0) * 100
        - _cm(fk_up, "grip_chest_fwd_m")
    ) if dyn_now else None
    ctx.log(
        f"[chest_now] Δforward FK(当前-直立基线)={d_fk_fwd:+.2f}cm"
        + (f" 动态峰值-直立FK={d_dyn_fwd:+.2f}cm" if d_dyn_fwd is not None else "")
    )

    payload = {
        "arm": arm,
        "trunk_now": trunk_now.round(4).tolist(),
        "theta_z_now": theta_now,
        "fk_now_cm": {k: _cm(fk_now, k) for k in fk_now},
        "fk_upright_zero_cm": {k: _cm(fk_up, k) for k in fk_up},
        "dynamic_now_cm": {k.replace("_m", "_cm"): round(v * 100, 2) for k, v in dyn_now.items()},
        "delta_fwd_cm": {"fk": d_fk_fwd, "dynamic_vs_upright_fk": d_dyn_fwd},
        "peak_frame": peak,
    }
    ctx.set_result({"ok": True, **payload})
    yield world.hold_action()


def _play_goals_record(world, arm, goals, frames_each, log_sink):
    """依次对每个 sub-goal 发绝对角，逐帧记录 forward/q_cur，返回(峰值forward_m, 终值forward_m, 末q_cur)。"""
    from behavior_interface.skills.arm_reset import _arm_qpos
    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    _prepare_legacy_7dof_motion(world, arm, stage_name="tuck_goals.prepare")
    peak = 0.0
    last_fwd = 0.0
    for gi, g in enumerate(goals):
        qa = np.asarray(g, dtype=np.float64).reshape(7).tolist()
        for _ in range(int(frames_each)):
            yield _tuck_make_action(world, arm, qa, 1.0, None)
            m = _grip_chest_metrics(world, arm)
            fwd = float(m.get("grip_chest_fwd_m") or 0.0)
            peak = max(peak, fwd)
            last_fwd = fwd
            if log_sink is not None:
                log_sink.append({"gi": gi, "fwd_cm": round(fwd * 100, 2)})
    cur = _arm_qpos(world, arm)
    return peak, last_fwd, cur


def _set_trunk_qpos(world, trunk_q):
    """直接把 trunk 关节设到指定角（诊断用，瞬间定位）。"""
    from behavior_interface.skills.grasp import _to_np

    robot = world.robot
    q = robot.get_joint_positions().clone()
    tidx = _to_np(robot.trunk_control_idx).astype(int)
    tq = np.asarray(trunk_q, dtype=np.float64).reshape(len(tidx))
    for i, j in enumerate(tidx):
        q[int(j)] = float(tq[i])
    robot.set_joint_positions(q)


def _sample_path_z_fwd_envelope(
    world,
    arm: str,
    path: Sequence[np.ndarray],
    *,
    z_ref_m: float,
    z_drop_max_m: float,
    max_forward_m: float,
    samples_per_seg: int = 16,
) -> dict:
    """FK 段内采样：Z 相对 z_ref 最大下降、forward 峰值。"""
    z_min = float(z_ref_m)
    max_fwd = 0.0
    worst_fwd_i = 0
    wp_i = 0
    for i in range(len(path) - 1):
        qa = np.asarray(path[i], dtype=np.float64).reshape(7)
        qb = np.asarray(path[i + 1], dtype=np.float64).reshape(7)
        for k in range(int(samples_per_seg) + 1):
            t = k / float(max(samples_per_seg, 1))
            q = (1.0 - t) * qa + t * qb
            fd = eef_chest_forward_dist_m(world, arm, q)
            zz = _eef_world_z_m(world, arm, q)
            if zz is not None:
                z_min = min(z_min, float(zz))
            if fd is not None and float(fd) > max_fwd:
                max_fwd = float(fd)
                worst_fwd_i = wp_i if t < 0.5 else wp_i + 1
        wp_i += 1
    z_drop_m = max(0.0, float(z_ref_m) - z_min)
    return {
        "z_min_m": round(z_min, 4),
        "z_drop_cm": round(z_drop_m * 100.0, 3),
        "max_fwd_cm": round(max_fwd * 100.0, 2),
        "worst_fwd_wp": int(worst_fwd_i),
        "z_ok": bool(z_drop_m <= float(z_drop_max_m) + 1e-4),
        "fwd_ok": bool(max_fwd <= float(max_forward_m) + 1e-4),
        "ok": bool(
            z_drop_m <= float(z_drop_max_m) + 1e-4
            and max_fwd <= float(max_forward_m) + 1e-4
        ),
    }


def _audit_phase2_z_fwd_feasibility(world, arm: str) -> dict:
    """FK：Phase2 Z/forward 包络可行性（含 5cm Z 裕度、20cm forward）。"""
    arm = (arm or "left").strip().lower()
    z_drop_max_m = 0.05
    max_fwd_m = TUCK_EXEC_FORWARD_MAX_M
    sol = _solve_phase1_vertical_c_pose(world, arm)
    if sol is None:
        return {"ok": False, "error": "phase1 终点无解"}
    q1 = sol["q"]
    z_ref = _eef_world_z_m(world, arm, q1)
    z_floor_strict = z_ref
    z_floor_relaxed = float(z_ref) - float(z_drop_max_m) if z_ref is not None else None
    seed = CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"]).copy()
    q_goal_strict = _resolve_chest_goal_q(
        world, arm, max_forward_m=max_fwd_m, z_floor_m=z_floor_strict,
    )
    q_goal_relaxed = _resolve_chest_goal_q(
        world, arm, max_forward_m=max_fwd_m, z_floor_m=z_floor_relaxed,
    )
    q_goal = q_goal_strict
    torso4 = _link_world_pos(world, "torso_link4")
    torso1 = _link_world_pos(world, "torso_link1")

    def _snap(q):
        fd = eef_chest_forward_dist_m(world, arm, q)
        zz = _eef_world_z_m(world, arm, q)
        h = _eef_chest_horiz_components(world, arm, q)
        return {
            "q": np.round(q, 4).tolist(),
            "eef_z_m": round(float(zz), 4) if zz is not None else None,
            "fwd_cm": round(float(fd) * 100, 2) if fd is not None else None,
            "lat_cm": round(float(h[1]) * 100, 2) if h else None,
        }

    p1 = _snap(q1)
    p_seed = _snap(seed)
    p_goal = _snap(q_goal)
    p_goal_relaxed = _snap(q_goal_relaxed)
    # 关节直线 Phase1→goal
    lerp_path = _densify_joint_path(
        _interpolate_arm_qpath(q1, q_goal_relaxed, 24), max_dq=0.04,
    )
    lerp_env = _sample_path_z_fwd_envelope(
        world, arm, lerp_path,
        z_ref_m=float(z_ref), z_drop_max_m=z_drop_max_m, max_forward_m=max_fwd_m,
    )
    # 贪心规划（无腰柱点云，Z 下限 = Phase1−5cm，forward≤20cm）
    greedy = _plan_chest_approach_path(
        world, arm, q1, q_goal_relaxed,
        max_forward_m=max_fwd_m,
        z_floor_m=z_floor_relaxed,
        pillar_pts_world=None,
        step_rad=0.038,
        fine_step_rad=0.016,
        max_iters=560,
        goal_tol=0.065,
    )
    greedy_dense = _densify_joint_path(greedy, max_dq=0.038)
    greedy_env = _sample_path_z_fwd_envelope(
        world, arm, greedy_dense,
        z_ref_m=float(z_ref), z_drop_max_m=z_drop_max_m, max_forward_m=max_fwd_m,
    )
    goal_ok_strict = (
        p_goal.get("fwd_cm") is not None
        and p_goal["fwd_cm"] < max_fwd_m * 100.0 + 0.1
        and z_ref is not None
        and p_goal.get("eef_z_m") is not None
        and p_goal["eef_z_m"] >= float(z_ref) - 1e-4
    )
    goal_ok_relaxed = (
        p_goal_relaxed.get("fwd_cm") is not None
        and p_goal_relaxed["fwd_cm"] < max_fwd_m * 100.0 + 0.1
        and z_floor_relaxed is not None
        and p_goal_relaxed.get("eef_z_m") is not None
        and p_goal_relaxed["eef_z_m"] >= float(z_floor_relaxed) - 1e-4
    )
    greedy_reached = float(np.linalg.norm(greedy[-1] - q_goal_relaxed, ord=np.inf)) <= 0.07
    # 硬 forward 过滤（legacy）+ 规划裕度 18cm 贪心
    legacy = _plan_chest_approach_path_legacy(
        world, arm, q1, q_goal_relaxed,
        max_forward_m=max_fwd_m,
        step_rad=0.038,
        fine_step_rad=0.016,
        max_iters=400,
        goal_tol=0.065,
    )
    legacy_dense = _densify_joint_path(legacy, max_dq=0.038)
    legacy_env = _sample_path_z_fwd_envelope(
        world, arm, legacy_dense,
        z_ref_m=float(z_ref), z_drop_max_m=z_drop_max_m, max_forward_m=max_fwd_m,
    )
    legacy_reached = float(np.linalg.norm(legacy[-1] - q_goal_relaxed, ord=np.inf)) <= 0.07
    greedy18 = _plan_chest_approach_path(
        world, arm, q1, q_goal_relaxed,
        max_forward_m=0.18,
        z_floor_m=z_floor_relaxed,
        pillar_pts_world=None,
        step_rad=0.038,
        fine_step_rad=0.016,
        max_iters=560,
        goal_tol=0.065,
    )
    greedy18_dense = _densify_joint_path(greedy18, max_dq=0.038)
    greedy18_env = _sample_path_z_fwd_envelope(
        world, arm, greedy18_dense,
        z_ref_m=float(z_ref), z_drop_max_m=z_drop_max_m, max_forward_m=max_fwd_m,
    )
    greedy18_reached = float(np.linalg.norm(greedy18[-1] - q_goal_relaxed, ord=np.inf)) <= 0.07
    return {
        "ok": True,
        "arm": arm,
        "constraints": {
            "max_fwd_cm": round(max_fwd_m * 100.0, 1),
            "z_drop_max_cm": round(z_drop_max_m * 100.0, 1),
        },
        "z_ref_m": round(float(z_ref), 4) if z_ref is not None else None,
        "z_min_floor_m": round(float(z_floor_relaxed), 4) if z_floor_relaxed is not None else None,
        "torso_link4_z_m": round(float(torso4[2]), 4) if torso4 is not None else None,
        "torso_link1_z_m": round(float(torso1[2]), 4) if torso1 is not None else None,
        "eef_above_torso4_cm": round((float(z_ref) - float(torso4[2])) * 100, 1)
        if z_ref is not None and torso4 is not None else None,
        "phase1_end": p1,
        "chest_seed": p_seed,
        "chest_goal_strict": p_goal,
        "chest_goal_relaxed": p_goal_relaxed,
        "straight_lerp": {**lerp_env, "waypoints_n": len(lerp_path)},
        "greedy_plan": {
            **greedy_env,
            "waypoints_n": len(greedy_dense),
            "reached_goal": bool(greedy_reached),
        },
        "greedy_plan_18cm": {
            **greedy18_env,
            "waypoints_n": len(greedy18_dense),
            "reached_goal": bool(greedy18_reached),
        },
        "legacy_hard_fwd": {
            **legacy_env,
            "waypoints_n": len(legacy_dense),
            "reached_goal": bool(legacy_reached),
        },
        "goal_pose_feasible_strict_z": bool(goal_ok_strict),
        "goal_pose_feasible_relaxed_z": bool(goal_ok_relaxed),
        "straight_lerp_feasible": bool(lerp_env.get("ok")),
        "greedy_plan_feasible": bool(greedy_env.get("ok") and greedy_reached),
        "any_fk_path_feasible": bool(
            (greedy_env.get("ok") and greedy_reached)
            or (greedy18_env.get("ok") and greedy18_reached)
            or (legacy_env.get("ok") and legacy_reached)
        ),
        "conclusion": (
            "FK 路径可行"
            if (
                (greedy_env.get("ok") and greedy_reached)
                or (greedy18_env.get("ok") and greedy18_reached)
                or (legacy_env.get("ok") and legacy_reached)
            )
            else (
                "终点可行但直线路径 forward 超标；需专门规划"
                if goal_ok_relaxed and not lerp_env.get("ok")
                else "需进一步放宽或调目标"
            )
        ),
    }


@register_skill(
    "diag_tuck_phase2_feasibility",
    description="FK 审计：Phase2 仅 Z 不降 + forward<20cm 是否可达",
)
def diag_tuck_phase2_feasibility(ctx, arm: str = "left"):
    world = ctx.world
    rep = _audit_phase2_z_fwd_feasibility(world, arm)
    if rep.get("ok"):
        ler = rep.get("straight_lerp") or {}
        grd = rep.get("greedy_plan") or {}
        cg = rep.get("chest_goal_relaxed") or {}
        leg = rep.get("legacy_hard_fwd") or {}
        g18 = rep.get("greedy_plan_18cm") or {}
        ctx.log(
            f"[p2_feas] Phase1末 z={rep.get('z_ref_m')}m "
            f"约束: Z降≤{rep['constraints']['z_drop_max_cm']}cm "
            f"fwd≤{rep['constraints']['max_fwd_cm']}cm | "
            f"直线 fwd峰={ler.get('max_fwd_cm')}cm Z降={ler.get('z_drop_cm')}cm "
            f"{'✅' if rep.get('straight_lerp_feasible') else '✗'} | "
            f"贪心20 fwd={grd.get('max_fwd_cm')}cm Z降={grd.get('z_drop_cm')}cm "
            f"{'✅' if rep.get('greedy_plan_feasible') else '✗'} | "
            f"贪心18 fwd={g18.get('max_fwd_cm')}cm 到终点={'✅' if g18.get('reached_goal') else '✗'} "
            f"{'✅' if g18.get('ok') and g18.get('reached_goal') else '✗'} | "
            f"legacy fwd={leg.get('max_fwd_cm')}cm 到终点={'✅' if leg.get('reached_goal') else '✗'} "
            f"{'✅' if leg.get('ok') and leg.get('reached_goal') else '✗'} | "
            f"总判={'✅' if rep.get('any_fk_path_feasible') else '✗'}"
        )
    ctx.set_result(rep)
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase2_three_constraints",
    description="FK+RRT：Phase2 三约束（fwd/腰距15cm/外移5cm）轨迹是否存在",
)
def diag_tuck_phase2_three_constraints(ctx, arm: str = "left"):
    world = ctx.world
    ctx.log("[p2_x3] 开始三约束 RRT+PRM 连通性搜索（约 2–5 分钟）…")
    yield world.hold_action()
    rep = _audit_phase2_three_constraints(world, arm)
    if rep.get("ok"):
        ts = rep.get("trajectory_search") or {}
        tv = ts.get("path_verify") or {}
        cg = rep.get("chest_goal") or {}
        p1 = rep.get("phase1_end") or {}
        leg = rep.get("legacy_fwd_only_ref") or {}
        ctx.log(
            f"[p2_x3] 轨迹={'存在' if rep.get('trajectory_exists') else '不存在'} "
            f"({ts.get('found_via') or ts.get('search_note', '')}) | "
            f"Phase1末 lat={p1.get('lat_cm')}cm 腰距={p1.get('waist_dist_cm')}cm | "
            f"胸前 fwd={cg.get('fwd_cm')}cm 腰距={cg.get('waist_dist_cm')}cm "
            f"外移={cg.get('lateral_outward_cm')}cm | "
            f"搜索路径 fwd峰={tv.get('max_fwd_cm')}cm "
            f"腰距min={tv.get('min_waist_cm')}cm "
            f"外移max={tv.get('max_lateral_outward_cm')}cm | "
            f"legacy对照 ok={leg.get('ok')} "
            f"腰距min={leg.get('min_waist_cm')}cm "
            f"外移max={leg.get('max_lateral_outward_cm')}cm"
        )
    ctx.set_result(rep)
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase2_four_constraints",
    description="FK+RRT：Phase2 四约束（fwd/腰距15cm/外移5cm/Z不降）轨迹是否存在",
)
def diag_tuck_phase2_four_constraints(ctx, arm: str = "left"):
    world = ctx.world
    ctx.log("[p2_x4] 开始四约束 RRT+PRM 连通性搜索（约 2–5 分钟）…")
    yield world.hold_action()
    rep = _audit_phase2_four_constraints(world, arm)
    if rep.get("ok"):
        ts = rep.get("trajectory_search") or {}
        tv = ts.get("path_verify") or {}
        cg = rep.get("chest_goal") or {}
        p1 = rep.get("phase1_end") or {}
        cst = rep.get("constraints") or {}
        leg = rep.get("legacy_fwd_only_ref") or {}
        ctx.log(
            f"[p2_x4] 轨迹={'存在' if rep.get('trajectory_exists') else '不存在'} "
            f"({ts.get('found_via') or ts.get('search_note', '')}) | "
            f"Phase1末 z={p1.get('z_m')}m lat={p1.get('lat_cm')}cm "
            f"腰距={p1.get('waist_dist_cm')}cm | "
            f"胸前 fwd={cg.get('fwd_cm')}cm z={cg.get('z_m')}m "
            f"腰距={cg.get('waist_dist_cm')}cm 外移={cg.get('lateral_outward_cm')}cm | "
            f"搜索路径 fwd峰={tv.get('max_fwd_cm')}cm "
            f"腰距min={tv.get('min_waist_cm')}cm "
            f"外移max={tv.get('max_lateral_outward_cm')}cm "
            f"z_min={tv.get('min_z_m')}m z_floor={cst.get('z_floor_m')}m "
            f"z_drop={tv.get('z_drop_cm')}cm | "
            f"legacy对照 ok={leg.get('ok')} z_drop={leg.get('z_drop_cm')}cm"
        )
    ctx.set_result(rep)
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase2_waist_fwd",
    description="FK：Phase2 forward<20cm + 腰距最小值>Phase1末",
)
def diag_tuck_phase2_waist_fwd(ctx, arm: str = "left"):
    world = ctx.world
    rep = _audit_phase2_fwd_waist_feasibility(world, arm)
    if rep.get("ok"):
        pv = rep.get("pivot_verify") or {}
        cg = rep.get("chest_goal") or {}
        leg = rep.get("legacy_fwd_waist") or {}
        ler = rep.get("straight_lerp") or {}
        lfo = rep.get("legacy_fwd_only") or {}
        ts = rep.get("trajectory_search") or {}
        tv = ts.get("path_verify") or {}
        ctx.log(
            f"[waist_fwd] Phase1末腰距d0={rep.get('phase1_end_waist_dist_cm')}cm | "
            f"轨迹是否存在={'存在' if rep.get('trajectory_exists') else '不存在'} "
            f"({ts.get('found_via') or ts.get('search_note', '')}) | "
            f"搜索复核 fwd峰={tv.get('max_fwd_cm')}cm "
            f"腰距min={tv.get('min_waist_cm')}cm | "
            f"对照: legacy腰距min={lfo.get('min_waist_dist_cm')}cm "
            f"直线腰距min={ler.get('min_waist_dist_cm')}cm"
        )
    ctx.set_result(rep)
    yield world.hold_action()


@register_skill(
    "diag_tuck_pillar_safe_zone",
    description="扫 torso_joint4 全行程标定腰柱禁入区，GTA 红区叠加图",
)
def diag_tuck_pillar_safe_zone(
    ctx,
    arm: str = "left",
    out_path: Optional[str] = None,
    n_q4: int = 25,
):
    """估算 Phase2 夹爪相对腰柱安全区，保存 GTA 红区标注图。"""
    import cv2
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "left").strip().lower()
    trunk_now = world.trunk_qpos().copy()
    q4_lo, q4_hi = _R1PRO_TRUNK_Q4_RANGE
    ctx.log(
        f"[pillar] R1Pro 腰链限位 rad={_R1PRO_TRUNK_LIMITS} "
        f"最高关节 torso_joint4=[{q4_lo:.4f},{q4_hi:.4f}] "
        f"≈[{np.degrees(q4_lo):.1f}°,{np.degrees(q4_hi):.1f}°]"
    )
    ctx.log(f"[pillar] 当前 trunk={np.round(trunk_now, 4).tolist()}")

    sol = _solve_phase1_vertical_c_pose(world, arm)
    if sol is None:
        ctx.set_result({"ok": False, "error": "Phase1 终点无解"})
        yield world.hold_action()
        return
    q_phase1 = sol["q"]
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = trunk_now.copy()
    for _ in range(12):
        yield _tuck_make_action(world, arm, hang, 1.0, trunk_pin)
    for q in _interpolate_arm_qpath(_HANG_ARM_QPOS, q_phase1, 8):
        for _ in range(4):
            yield _tuck_make_action(world, arm, q.tolist(), 1.0, trunk_pin)

    z_ref = _eef_world_z_m(world, arm, q_phase1)
    if z_ref is None:
        z_ref = float(world.eef_pose(arm=arm)["pos"][2])

    pillar_pts = _sweep_pillar_envelope(world, trunk_now, n_q4=int(n_q4))
    est = _estimate_pillar_forbidden_horiz(world, pillar_pts, z_ref_m=float(z_ref))
    safe_lat = est.get("safe_lat_min_left_m")
    if safe_lat is not None:
        ctx.log(
            f"[pillar] 左臂 tuck 高度 z={z_ref:.3f}m "
            f"建议 |lateral|≥{safe_lat*100:.1f}cm（fwd∈5–22cm 带）"
        )

    h_now = _eef_chest_horiz_components(world, arm, q_phase1)
    d_now = _eef_pillar_distance_m(world, arm, q_phase1, pillar_pts)
    eef_w = np.asarray(world.eef_pose(arm=arm)["pos"], dtype=np.float64).reshape(3)

    rgb = _gta_rgb_array(world)
    if rgb is None:
        ctx.set_result({"ok": False, "error": "gta_view 无 RGB"})
        yield world.hold_action()
        return
    marked = _overlay_pillar_zone_gta(
        rgb, world, pillar_pts, est["forbidden"],
        z_ref_m=float(z_ref), eef_world=eef_w,
    )
    default_out = os.path.join(
        os.path.dirname(__file__), "test", f"tuck_pillar_collision_zone_{arm}.png",
    )
    fpath = str(out_path or default_out)
    os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
    cv2.imwrite(fpath, cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
    ctx.log(f"[pillar] 红区图 {fpath} 支柱点={len(pillar_pts)} 禁入格={len(est['forbidden'])}")

    ctx.set_result({
        "ok": True,
        "arm": arm,
        "trunk_joint4_rad": [q4_lo, q4_hi],
        "trunk_joint4_deg": [round(np.degrees(q4_lo), 2), round(np.degrees(q4_hi), 2)],
        "trunk_limits_rad": [list(map(float, lohi)) for lohi in _R1PRO_TRUNK_LIMITS],
        "trunk_q_now": np.round(trunk_now, 4).tolist(),
        "z_ref_m": round(float(z_ref), 4),
        "gripper_pillar_dist_cm": round(float(d_now) * 100, 2) if d_now is not None else None,
        "chest_fwd_cm": round(h_now[0] * 100, 2) if h_now else None,
        "chest_lat_cm": round(h_now[1] * 100, 2) if h_now else None,
        "safe_lat_min_left_m": safe_lat,
        "clearance_m": _GRIPPER_PILLAR_SOFT_M,
        "forbidden_cells_n": len(est["forbidden"]),
        "image_path": fpath,
        "phase1_q": np.round(q_phase1, 4).tolist(),
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_waist_j34_forbidden_zone",
    description="腰 j3↔j4 段沿 forward 正前方禁入区，GTA 红区叠加图",
)
def diag_tuck_waist_j34_forbidden_zone(
    ctx,
    out_path: Optional[str] = None,
    n_q4: int = 25,
    length_extend_factor: float = 1.0,
):
    """测量 j3↔j4 腰模块 L×W×H（左右对称，与臂无关），包络长方体 + forward 拉长投影。"""
    import cv2

    world = ctx.world
    trunk_now = world.trunk_qpos().copy()
    j3 = _waist_joint3_axis_center_world(world)
    j4 = _waist_top_joint4_axis_center_world(world)
    ctx.log(
        f"[waist_j34] 腰段=torso_joint3(第二高)↔torso_joint4(最高) | "
        f"trunk={np.round(trunk_now, 4).tolist()}"
    )
    if j3 is not None and j4 is not None:
        ctx.log(
            f"[waist_j34] j3={j3.round(3).tolist()} j4={j4.round(3).tolist()} "
            f"间距={np.linalg.norm(j4-j3)*100:.1f}cm"
        )

    yield world.hold_action()

    seg_meas = _collect_waist_j34_segment_points(world, trunk_now)
    seg_sweep = _sweep_waist_j34_envelope(world, trunk_now, n_q4=int(n_q4))
    est = _waist_j34_body_and_forward_projection(
        world, trunk_now,
        length_extend_factor=float(length_extend_factor),
    )
    sweep_meas = _chest_frame_aabb_from_points(world, seg_sweep)
    meas = est.get("measure") or {}
    sym_y_cm = None
    if j3 is not None:
        bl = _world_to_robot_base_local(world, j3)
        if bl is not None:
            sym_y_cm = round(abs(float(bl[1])) * 100.0, 3)

    rgb = _gta_rgb_array(world)
    if rgb is None:
        ctx.set_result({"ok": False, "error": "gta_view 无 RGB"})
        yield world.hold_action()
        return
    marked = _overlay_waist_j34_forbidden_gta(
        rgb, world,
        body_box=est.get("body_box"),
        forbidden_box=est.get("forbidden_box"),
        measure=meas,
    )
    default_out = os.path.join(
        os.path.dirname(__file__), "test", "tuck_waist_j34_forbidden_zone.png",
    )
    fpath = str(out_path or default_out)
    os.makedirs(os.path.dirname(fpath) or ".", exist_ok=True)
    cv2.imwrite(fpath, cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
    axes_chk = _module_j34_axes_world(world)
    lat_sym_cm = None
    if axes_chk is not None and j3 is not None:
        _o, _f, _l, _u = axes_chk
        lat_sym_cm = round(abs(float(np.dot(np.asarray(j3, dtype=np.float64) - _o, _l))) * 100.0, 3)
    fb = est.get("forbidden_box") or {}
    ctx.log(
        f"[waist_j34] 红区图 {fpath} | forbidden lat=[{fb.get('lat_lo_m')},{fb.get('lat_hi_m')}]m | "
        f"j4壳体点={len(_collect_j4_waist_shell_points(world, trunk_now))} | "
        f"对称面偏差|base_y(j3)|={sym_y_cm}cm |j3横向|={lat_sym_cm}cm | "
        f"长(L/fwd)={meas.get('length_cm')}cm "
        f"宽(W/lat)={meas.get('width_cm')}cm "
        f"高(H/axis)={meas.get('height_cm')}cm | "
        f"向前投影={est.get('projection_length_cm')}cm "
        f"(factor={est.get('length_extend_factor')}) | "
        f"q4扫掠对照 L={sweep_meas.get('length_cm') if sweep_meas else None}cm "
        f"W={sweep_meas.get('width_cm') if sweep_meas else None}cm"
    )

    ctx.set_result({
        "ok": True,
        "bilateral": True,
        "joints": {
            "second_highest": "torso_joint3",
            "highest": "torso_joint4",
            "j3_pivot_world_m": j3.round(4).tolist() if j3 is not None else None,
            "j4_pivot_world_m": j4.round(4).tolist() if j4 is not None else None,
            "j3_j4_dist_cm": round(float(np.linalg.norm(j4 - j3)) * 100, 2)
            if j3 is not None and j4 is not None else None,
            "symmetry_plane_base_y_cm": sym_y_cm,
        },
        "trunk_q_now": np.round(trunk_now, 4).tolist(),
        "segment_points_n": len(seg_meas),
        "q4_sweep_envelope_cm": sweep_meas,
        "measure_lwh_cm": {
            "length_fwd": meas.get("length_cm"),
            "width_lat": meas.get("width_cm"),
            "height_axis": meas.get("height_cm"),
        },
        "body_box": est.get("body_box"),
        "forbidden_box": est.get("forbidden_box"),
        "projection_length_cm": est.get("projection_length_cm"),
        "length_extend_factor": est.get("length_extend_factor"),
        "image_path": fpath,
        "note": "左右对称禁入区；深红=腰模块包络；浅红=仅 +forward 拉长",
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_waist_j34_forbidden_zone_trimetric",
    description="腰 j3↔j4 禁入区：斜上视 + 正视/俯视/侧视三视图",
)
def diag_tuck_waist_j34_forbidden_zone_trimetric(
    ctx,
    out_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    oblique_path: Optional[str] = None,
    length_extend_factor: float = 1.0,
    dist_front: float = 2.0,
    dist_top: float = 2.6,
    dist_side: float = 2.0,
):
    """斜上 GTA 透视 + 正视/俯视/侧视；并单独输出 oblique png。"""
    import cv2

    world = ctx.world
    trunk_now = world.trunk_qpos().copy()
    yield world.hold_action()

    est = _waist_j34_body_and_forward_projection(
        world, trunk_now,
        length_extend_factor=float(length_extend_factor),
    )
    meas = est.get("measure") or {}
    body_box = est.get("body_box")
    forbidden_box = est.get("forbidden_box")
    if not forbidden_box:
        ctx.set_result({"ok": False, "error": "禁入区构盒失败"})
        yield world.hold_action()
        return

    # 先拍默认斜上视（三视图会临时改相机，结束后再恢复）
    oblique = _render_waist_j34_forbidden_oblique_gta(
        world,
        body_box=body_box,
        forbidden_box=forbidden_box,
        measure=meas,
        compact=False,
    )
    trimetric_row, singles = _render_waist_j34_forbidden_trimetric(
        world,
        body_box=body_box,
        forbidden_box=forbidden_box,
        measure=meas,
        dist_front=float(dist_front),
        dist_top=float(dist_top),
        dist_side=float(dist_side),
    )
    if trimetric_row is None:
        ctx.set_result({"ok": False, "error": "三视图渲染失败（gta_view 或机位）"})
        yield world.hold_action()
        return

    test_dir = os.path.join(os.path.dirname(__file__), "test")
    base_dir = str(out_dir or test_dir)
    os.makedirs(base_dir, exist_ok=True)

    oblique_fpath = str(
        oblique_path or os.path.join(base_dir, "tuck_waist_j34_forbidden_zone.png"),
    )
    if oblique is not None:
        cv2.imwrite(oblique_fpath, cv2.cvtColor(oblique, cv2.COLOR_RGB2BGR))

    combo_path = str(
        out_path or os.path.join(base_dir, "tuck_waist_j34_forbidden_zone_trimetric.png"),
    )
    if oblique is not None:
        combo = _stitch_oblique_over_trimetric(oblique, trimetric_row)
    else:
        combo = trimetric_row
    cv2.imwrite(combo_path, cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))

    single_paths: Dict[str, str] = {}
    if oblique is not None:
        single_paths["oblique"] = oblique_fpath
    for vname, img in singles.items():
        sp = os.path.join(base_dir, f"tuck_waist_j34_forbidden_zone_{vname}.png")
        cv2.imwrite(sp, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        single_paths[vname] = sp

    ctx.log(
        f"[waist_j34] 全视图 {combo_path} | 斜上 {oblique_fpath} | "
        f"L={meas.get('length_cm')} W={meas.get('width_cm')} H={meas.get('height_cm')}cm | "
        f"单帧={list(single_paths.values())}"
    )
    ctx.set_result({
        "ok": True,
        "bilateral": True,
        "measure_lwh_cm": {
            "length_fwd": meas.get("length_cm"),
            "width_lat": meas.get("width_cm"),
            "height_axis": meas.get("height_cm"),
            "projection_span": est.get("projection_length_cm"),
        },
        "forbidden_box": forbidden_box,
        "oblique_path": oblique_fpath if oblique is not None else None,
        "trimetric_path": combo_path,
        "view_paths": single_paths,
        "camera_dist_m": {
            "front": float(dist_front),
            "top": float(dist_top),
            "side": float(dist_side),
        },
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase1_vertical_pose",
    description="Phase1 演示：肘屈限位+j1后摆使肩→EEF对边竖直，GTA截图",
)
def diag_tuck_phase1_vertical_pose(
    ctx,
    arm: str = "left",
    hold_frames: int = 36,
    out_path: Optional[str] = None,
):
    """左臂移动到 Phase1 目标位并截图（相机请先用 /api/camera 设好外参）。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS
    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    world = ctx.world
    arm = (arm or "left").strip().lower()
    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name="diag_tuck_phase1.prepare"
    )
    sol = _solve_phase1_vertical_c_pose(world, arm)
    if sol is None:
        ctx.set_result({"ok": False, "error": "未找到 j1+j4 使 c 竖直的可行解"})
        yield world.hold_action()
        return

    q_tgt = sol["q"]
    ctx.log(
        f"[phase1_pose] j1={sol['j1']:.3f} j4={sol['j4']:.3f} "
        f"C={sol['C_deg']}° c_tilt={sol['c_tilt_deg']}° "
        f"Δz={sol['dz_cm']}cm dxy={sol.get('dxy_cm')}cm "
        f"Δfwd={sol.get('delta_forward_cm')}cm"
    )

    hang = _HANG_ARM_QPOS.tolist()
    for _ in range(14):
        yield _tuck_make_action(world, arm, hang, 1.0, None)

    path = _interpolate_arm_qpath(_HANG_ARM_QPOS, q_tgt, 10)
    per = max(3, int(hold_frames) // max(len(path), 1))
    for q in path:
        for _ in range(per):
            yield _tuck_make_action(world, arm, q.tolist(), 1.0, None)

    default_out = os.path.join(
        os.path.dirname(__file__), "test", f"tuck_phase1_vertical_{arm}.png",
    )
    fpath = str(out_path or default_out)
    ok_cap = _capture_gta_png(world, fpath, ctx)
    ctx.set_result({
        "ok": True,
        "arm": arm,
        "q_target": np.round(q_tgt, 4).tolist(),
        "pose": sol,
        "screenshot": fpath,
        "screenshot_ok": bool(ok_cap),
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase1_v2_pose",
    description="Phase1 v2：肩肘后摆+arm_joint6腕俯仰抬夹爪，左右臂末态 GTA 截图",
)
def diag_tuck_phase1_v2_pose(
    ctx,
    arm: str = "both",
    hold_frames: int = 36,
    out_dir: Optional[str] = None,
):
    """hang→Phase1 v2，保存两臂末态截图 tuck_phase1_v2_end_{arm}.png。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    world = ctx.world
    arms = ["left", "right"] if (arm or "both").strip().lower() == "both" else [(arm or "left").strip().lower()]
    base_dir = out_dir or os.path.join(os.path.dirname(__file__), "test")
    os.makedirs(base_dir, exist_ok=True)
    trunk_pin = world.trunk_qpos().copy()
    hang = _HANG_ARM_QPOS.tolist()
    reports: dict = {}

    for a in arms:
        sol_v1 = _solve_phase1_vertical_c_pose(world, a)
        sol = _solve_phase1_vertical_c_pose_v2(world, a)
        if sol is None:
            reports[a] = {"ok": False, "error": "Phase1 v2 无解"}
            ctx.log(f"[p1_v2/{a}] 无解")
            continue
        jname5 = _phase1_wrist_j5_joint_name(a)
        jname6 = _phase1_wrist_j6_joint_name(a)
        ctx.log(
            f"[p1_v2/{a}] 肘近腕={jname5}(Δz≈0) 俯仰={jname6} sign={sol['j6_lift_sign']:+.0f} "
            f"j1={sol['j1']:.3f} j4={sol['j4']:.3f} j6={sol['j6']:.3f} "
            f"j6_probe(+/-)={sol['j6_probe']['dz_pos_cm']}/{sol['j6_probe']['dz_neg_cm']}cm "
            f"Δz_v1={sol_v1['dz_cm'] if sol_v1 else '?'}cm +Δz_wrist={sol['dz_wrist_cm']}cm "
            f"总Δz={sol['dz_cm']}cm"
        )
        for _ in range(18):
            yield _tuck_make_action(world, a, hang, 1.0, trunk_pin)
        yield world.hold_action()
        path = _plan_phase1_path_v2(world, a, _HANG_ARM_QPOS.copy())
        per = max(3, int(hold_frames) // max(len(path), 1))
        for q in path:
            for _ in range(per):
                yield _tuck_make_action(world, a, q.tolist(), 1.0, trunk_pin)
        for _ in range(8):
            yield _tuck_make_action(world, a, path[-1].tolist(), 1.0, trunk_pin)
        yield world.hold_action()
        fpath = os.path.join(base_dir, f"tuck_phase1_v2_end_{a}.png")
        ok_cap = _capture_gta_png(world, fpath, ctx)
        reports[a] = {
            "ok": True,
            "wrist_near_elbow_joint": jname5,
            "wrist_pitch_joint": jname6,
            "j6_lift_sign": sol["j6_lift_sign"],
            "j5_probe": sol["j5_probe"],
            "j6_probe": sol["j6_probe"],
            "q_target": np.round(sol["q"], 4).tolist(),
            "j1": sol["j1"],
            "j4": sol["j4"],
            "j6": sol["j6"],
            "dz_v1_cm": sol_v1["dz_cm"] if sol_v1 else None,
            "dz_wrist_cm": sol["dz_wrist_cm"],
            "dz_total_cm": sol["dz_cm"],
            "delta_forward_cm": sol.get("delta_forward_cm"),
            "delta_lateral_cm": sol.get("delta_lateral_cm"),
            "waypoints_n": len(path),
            "screenshot": fpath,
            "screenshot_ok": bool(ok_cap),
        }

    ctx.set_result({"ok": True, "reports": reports, "phase1_version": "v2"})
    yield world.hold_action()


@register_skill(
    "diag_tuck_lift_only",
    description="仅 Phase1 垂直抬升：汇报路点数、ΔZ、XY 漂移",
)
def diag_tuck_lift_only(ctx, arm: str = "left"):
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS
    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    world = ctx.world
    arm = (arm or "left").strip().lower()
    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name="diag_tuck_lift.prepare"
    )
    for _ in range(18):
        yield _tuck_make_action(
            world, arm, _HANG_ARM_QPOS.tolist(), 1.0, None
        )
    meas = _measure_lift_triangle_ab(world, arm, _HANG_ARM_QPOS)
    tri_info = {}
    if meas:
        a, b, ps, pe, pw = meas
        j4_lo, _ = _R1PRO_ARM_J4_LIMITS[arm]
        tri_info = {
            "a_cm": round(a * 100, 2), "b_cm": round(b * 100, 2),
            "C_hang_deg": 180.0,
            "C_limit_deg": round(np.degrees(np.pi + float(j4_lo)), 1),
            "C_geom_hang_deg": round(np.degrees(_elbow_joint_angle_rad(ps, pe, pw)), 1),
        }
        ctx.log(f"[lift_only] 三角形 a={tri_info.get('a_cm')}cm b={tri_info.get('b_cm')}cm "
                f"C_hang={tri_info.get('C_hang_deg')}° C_limit={tri_info.get('C_limit_deg')}°")
    lift = _plan_vertical_lift_path(world, arm, _HANG_ARM_QPOS)
    z0 = _eef_world_z_m(world, arm, lift[0])
    z1 = _eef_world_z_m(world, arm, lift[-1])
    h0 = _eef_chest_horiz_components(world, arm, lift[0])
    h1 = _eef_chest_horiz_components(world, arm, lift[-1])
    peak_dfwd = 0.0
    peak_dlat = 0.0
    peak_z = z0 or 0.0
    if h0 is not None:
        f0, l0, _ = h0
        for q in lift:
            h = _eef_chest_horiz_components(world, arm, q)
            zz = _eef_world_z_m(world, arm, q)
            if h is not None:
                peak_dfwd = max(peak_dfwd, float(h[0] - f0))
                peak_dlat = max(peak_dlat, abs(float(h[1] - l0)))
            if zz is not None:
                peak_z = max(peak_z, zz)
    dfwd_end = float(h1[0] - h0[0]) * 100 if h0 and h1 else None
    dlat_end = abs(float(h1[1] - h0[1])) * 100 if h0 and h1 else None
    ctx.log(
        f"[lift_only] wp={len(lift)} Δz={(z1-z0)*100 if z0 and z1 else 0:.1f}cm "
        f"Δforward={dfwd_end:.2f}cm(max={peak_dfwd*100:.2f}) "
        f"Δlat={dlat_end:.2f}cm(max={peak_dlat*100:.2f}) "
        f"q_end={np.round(lift[-1],3).tolist()}"
    )
    frame_log: List[Dict[str, float]] = []
    yield from _yield_play_waypoint_path(world, arm, lift, 1.0, ctx, frame_log=frame_log)
    dyn = _summarize_grip_chest_log(frame_log)
    ctx.set_result({
        "ok": True, "arm": arm, "lift_wp": len(lift),
        "dz_cm": round((z1 - z0) * 100, 2) if z0 and z1 else None,
        "max_dz_cm": round((peak_z - z0) * 100, 2) if z0 else None,
        "delta_forward_cm": round(dfwd_end, 2) if dfwd_end is not None else None,
        "max_delta_forward_cm": round(peak_dfwd * 100, 2),
        "max_delta_lateral_cm": round(peak_dlat * 100, 2),
        "q_end": np.round(lift[-1], 4).tolist(),
        "dyn_peak_fwd_cm": round(float(dyn.get("max_grip_chest_fwd_m", 0)) * 100, 2),
        "triangle": tri_info,
    })
    yield world.hold_action()


def _probe_j1_backswing_fk(world, arm: str) -> dict:
    """FK：hang 下 j1± 与屈肘限位，比较胸口 forward 变化（判定后摆符号）。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    arm = (arm or "left").strip().lower()
    q_hang = np.asarray(_HANG_ARM_QPOS, dtype=np.float64).reshape(7).copy()
    j4_lo, _ = _R1PRO_ARM_J4_LIMITS.get(arm, (-2.0944, 0.3491))
    h0 = _eef_chest_horiz_components(world, arm, q_hang)
    if h0 is None:
        return {"ok": False, "arm": arm, "error": "hang FK 失败"}
    fwd0 = float(h0[0])
    rows = []
    for j1 in (-1.2, -0.6, 0.6, 1.2, 1.55):
        q = q_hang.copy()
        q[0] = float(j1)
        q[3] = float(j4_lo)
        h = _eef_chest_horiz_components(world, arm, q)
        zz = _eef_world_z_m(world, arm, q)
        if h is None:
            continue
        rows.append({
            "j1": round(float(j1), 2),
            "fwd_cm": round(float(h[0]) * 100, 2),
            "delta_fwd_cm": round((float(h[0]) - fwd0) * 100, 2),
            "lat_cm": round(float(h[1]) * 100, 2),
            "z_m": round(float(zz), 4) if zz is not None else None,
        })
    best_pos = min(rows, key=lambda r: r["delta_fwd_cm"]) if rows else None
    return {
        "ok": True,
        "arm": arm,
        "chest_tuck_j1": round(float(CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"])[0]), 3),
        "hang_fwd_cm": round(fwd0 * 100, 2),
        "samples": rows,
        "least_forward_j1": best_pos.get("j1") if best_pos else None,
        "backswing_bounds": _shoulder_j1_backswing_bounds(arm),
    }


@register_skill(
    "diag_tuck_phase1_j1_sign_audit",
    description="FK：左右臂 j1 后摆符号审计（hang→j1±+屈肘限位）",
)
def diag_tuck_phase1_j1_sign_audit(ctx, arm: str = "both"):
    world = ctx.world
    arms = ["left", "right"] if (arm or "both").strip().lower() == "both" else [(arm or "left").strip().lower()]
    reports = {}
    for a in arms:
        yield world.hold_action()
        rep = _probe_j1_backswing_fk(world, a)
        reports[a] = rep
        if rep.get("ok"):
            ctx.log(
                f"[j1_sign/{a}] 收胸j1={rep.get('chest_tuck_j1')} "
                f"后摆界={rep.get('backswing_bounds')} "
                f"最少前探j1={rep.get('least_forward_j1')} | "
                f"samples={rep.get('samples')}"
            )
    ctx.set_result({"ok": True, "reports": reports})
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase1_backswing_video",
    description="arm_reset(hang) 后仅播放 Phase1 垂直抬升并录 GTA 视频",
)
def diag_tuck_phase1_backswing_video(
    ctx,
    arm: str = "both",
    record_video: bool = True,
    video_path: Optional[str] = None,
):
    """左右臂各自 hang→Phase1 后摆录像，用于核对 j1 符号。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    world = ctx.world
    arms = ["left", "right"] if (arm or "both").strip().lower() == "both" else [(arm or "left").strip().lower()]
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    all_reports: dict = {}
    video_paths: dict = {}

    for a in arms:
        for _ in range(25):
            yield _tuck_make_action(world, a, hang, 1.0, trunk_pin)
        yield world.hold_action()
        lift = _plan_phase1_path(world, a, _HANG_ARM_QPOS.copy())
        p1d = _densify_joint_path(lift, max_dq=0.034)
        j1_end = float(p1d[-1][0]) if p1d else None
        z0 = _eef_world_z_m(world, a, p1d[0])
        z1 = _eef_world_z_m(world, a, p1d[-1])
        h0 = _eef_chest_horiz_components(world, a, p1d[0])
        h1 = _eef_chest_horiz_components(world, a, p1d[-1])
        ctx.log(
            f"[p1_vid/{a}] Phase1 wp={len(p1d)} j1 {p1d[0][0]:.3f}→{j1_end:.3f} "
            f"j4 {p1d[0][3]:.3f}→{p1d[-1][3]:.3f} "
            f"Δfwd={(float(h1[0])-float(h0[0]))*100 if h0 and h1 else 0:.1f}cm "
            f"Δz={(float(z1)-float(z0))*100 if z0 and z1 else 0:.1f}cm"
        )
        vpath = None
        video_ok = False
        if record_video:
            if video_path and len(arms) == 1:
                vpath = str(video_path)
            else:
                vpath = os.path.join(
                    os.path.dirname(__file__), "test", "exec_move",
                    f"tuck_phase1_backswing_{a}.mp4",
                )
            os.makedirs(os.path.dirname(vpath), exist_ok=True)
            video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10)
            frame_log: List[Dict[str, float]] = []
            yield from _yield_play_waypoint_path(
                world, a, p1d, 1.0, ctx,
                frames_per_wp=TUCK_FRAMES_PER_WP,
                frame_log=frame_log,
                video_rec=video_rec,
            )
            video_ok = bool(video_rec.save(ctx))
            video_paths[a] = vpath
        all_reports[a] = {
            "waypoints_n": len(p1d),
            "j1_start": round(float(p1d[0][0]), 3),
            "j1_end": round(float(j1_end), 3) if j1_end is not None else None,
            "j4_end": round(float(p1d[-1][3]), 3),
            "delta_fwd_cm": round((float(h1[0]) - float(h0[0])) * 100, 2) if h0 and h1 else None,
            "delta_z_cm": round((float(z1) - float(z0)) * 100, 2) if z0 and z1 else None,
            "video_path": vpath,
            "video_ok": video_ok,
        }

    ctx.set_result({"ok": True, "reports": all_reports, "video_paths": video_paths})
    yield world.hold_action()


@register_skill(
    "diag_tuck_joint_probe",
    description="悬垂位：逐关节±dq 探针，汇报 EEF Δxyz（理解大臂/小臂协同）",
)
def diag_tuck_joint_probe(ctx, arm: str = "left", dq: float = 0.08, q_start: Optional[list] = None):
    """FK 探针：从 hang（或指定 q）看各关节对 EEF 的影响。"""
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "left").strip().lower()
    q0 = (
        np.asarray(q_start, dtype=np.float64).reshape(7).copy()
        if q_start is not None else _HANG_ARM_QPOS.copy()
    )
    p0 = _probe_eef_pos_at_arm_qpos(world, arm, q0)
    if p0 is None:
        ctx.set_result({"ok": False, "error": "eef probe failed"})
        yield world.hold_action()
        return
    rows = []
    names = [f"j{i+1}" for i in range(7)]
    for j in range(7):
        for sign, lab in ((1.0, "+"), (-1.0, "-")):
            q = q0.copy()
            q[j] += float(dq) * sign
            p1 = _probe_eef_pos_at_arm_qpos(world, arm, q)
            if p1 is None:
                continue
            d = p1 - p0
            rows.append({
                "joint": names[j] + lab,
                "dq": round(float(dq) * sign, 3),
                "dx_cm": round(float(d[0]) * 100, 2),
                "dy_cm": round(float(d[1]) * 100, 2),
                "dz_cm": round(float(d[2]) * 100, 2),
                "dxy_cm": round(float(np.linalg.norm(d[:2])) * 100, 2),
            })
    s3, s4 = _lift_upper_forearm_signs(arm)
    qc = q0.copy()
    qc[2] += s3 * float(dq)
    qc[3] += s4 * float(dq)
    pc = _probe_eef_pos_at_arm_qpos(world, arm, qc)
    coupled = None
    if pc is not None:
        dc = pc - p0
        coupled = {
            "j3_dq": round(s3 * float(dq), 3),
            "j4_dq": round(s4 * float(dq), 3),
            "dx_cm": round(float(dc[0]) * 100, 2),
            "dy_cm": round(float(dc[1]) * 100, 2),
            "dz_cm": round(float(dc[2]) * 100, 2),
            "dxy_cm": round(float(np.linalg.norm(dc[:2])) * 100, 2),
        }
    ctx.log(f"[joint_probe] arm={arm} hang eef z={p0[2]:.3f}m coupled={coupled}")
    for r in rows:
        if abs(r["dz_cm"]) > 0.5 or r["dxy_cm"] < 0.5:
            ctx.log(f"  {r['joint']} dq={r['dq']:+.3f} → Δxyz=({r['dx_cm']:+.1f},{r['dy_cm']:+.1f},{r['dz_cm']:+.1f})cm dxy={r['dxy_cm']:.2f}cm")
    in_sh0 = _pos_in_link_frame(world, f"{arm}_arm_link1", p0)
    h0 = _eef_chest_horiz_components(world, arm, q0)
    ctx.set_result({
        "ok": True, "arm": arm, "eef0": p0.tolist(),
        "eef0_in_shoulder": in_sh0.tolist() if in_sh0 is not None else None,
        "chest_fwd_m": h0[0] if h0 else None,
        "chest_lat_m": h0[1] if h0 else None,
        "eef_z_m": h0[2] if h0 else None,
        "per_joint": rows, "coupled_j3_j4": coupled,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_two_phase",
    description="两阶段 tuck：垂直抬升→贴胸收拢，测 FK/动态 forward 峰值与自碰",
)
def diag_tuck_two_phase(
    ctx,
    arm: str = "left",
    record_video: bool = True,
    video_path: Optional[str] = None,
):
    """诊断：两阶段 tuck；可选录 GTA 过程视频到 test/exec_move/tuck_vN.mp4。"""
    from behavior_interface.skills.arm_reset import _arm_qpos, _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "left").strip().lower()
    theta = float(world.chest_pose().get("theta_z_deg", 90.0))

    # 先回悬垂（锁 trunk，与规划 FK 坐标系一致）
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    for _ in range(20):
        yield _tuck_make_action(world, arm, hang, 1.0, trunk_pin)

    path, lift_wp_n = build_two_phase_tuck_path(world, arm, q_start=_HANG_ARM_QPOS)
    fk = verify_tuck_path(
        world, arm, path, max_forward_m=TUCK_PLAN_FORWARD_MAX_M, samples=24,
    )
    z_phase1_end = _eef_world_z_m(world, arm, path[min(lift_wp_n, len(path)) - 1])
    per_wp = []
    for i, q in enumerate(path):
        fd = eef_chest_forward_dist_m(world, arm, q)
        zz = _eef_world_z_m(world, arm, q)
        sc = _arm_self_collision_at_q(world, arm, q)
        per_wp.append({
            "i": i,
            "phase": 1 if i < lift_wp_n else 2,
            "fwd_cm": round(fd * 100, 2) if fd is not None else None,
            "z_m": round(zz, 4) if zz is not None else None,
            "self_collide_fk": bool(sc),
        })

    vpath = None
    if record_video:
        vpath = str(video_path) if video_path else os.path.join(
            os.path.dirname(__file__), "test", "exec_move", "tuck_v2.mp4",
        )
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
    video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10) if record_video and vpath else None
    frame_log: List[Dict[str, float]] = []
    yield from _yield_play_tuck_smooth(
        world, arm, path, 1.0, ctx,
        frame_log=frame_log,
        video_rec=video_rec,
        lift_wp_n=lift_wp_n,
    )
    video_ok = bool(video_rec.save(ctx)) if video_rec is not None else False
    dyn = _summarize_grip_chest_log(frame_log)
    peak_dyn = float(dyn.get("max_grip_chest_fwd_m", 0.0))
    sc_dyn = any(bool(fr.get("self_collide")) for fr in frame_log)
    cur = _arm_qpos(world, arm)
    final_fwd = float((_grip_chest_metrics(world, arm).get("grip_chest_fwd_m") or 0.0))
    # Phase2 动态 Z 是否下降
    z_drop_m = 0.0
    if z_phase1_end is not None and frame_log:
        p2_frames = [
            fr for fr in frame_log
            if int(fr.get("phase", 0)) == 2
        ]
        for fr in p2_frames:
            zz = fr.get("eef_z_m")
            if zz is not None:
                z_drop_m = max(z_drop_m, float(z_phase1_end) - float(zz))

    ctx.log(
        f"[two_phase] θz={theta:.1f}° wp={len(path)} phase1={lift_wp_n} "
        f"FK峰值={fk['max_forward_cm']:.1f}cm 动态峰值={peak_dyn*100:.1f}cm "
        f"phase2_Z降={z_drop_m*100:.2f}cm "
        f"终点={final_fwd*100:.1f}cm j4={cur[3]:.3f} "
        f"f≤20={'✅' if peak_dyn <= TUCK_EXEC_FORWARD_MAX_M + 1e-3 else '✗'} "
        f"无自碰={'✅' if not sc_dyn else '✗'}"
    )
    ctx.set_result({
        "ok": True,
        "arm": arm,
        "theta_z": theta,
        "waypoints_n": len(path),
        "lift_wp_n": lift_wp_n,
        "fk_peak_cm": round(fk["max_forward_cm"], 2),
        "dyn_peak_cm": round(peak_dyn * 100, 2),
        "final_fwd_cm": round(final_fwd * 100, 2),
        "pass_20": bool(peak_dyn <= TUCK_EXEC_FORWARD_MAX_M + 1e-3),
        "phase2_z_drop_cm": round(z_drop_m * 100, 3),
        "z_no_drop_phase2": bool(z_drop_m <= 0.005),
        "self_collide": bool(sc_dyn),
        "video_path": vpath,
        "video_ok": video_ok,
        "per_wp": per_wp,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_legacy_hard_fwd",
    description="两阶段 tuck（Phase2 legacy 硬 forward≤20cm），录 GTA 视频",
)
def diag_tuck_legacy_hard_fwd(
    ctx,
    arm: str = "left",
    record_video: bool = True,
    video_path: Optional[str] = None,
):
    """播放 legacy 硬 forward Phase2 规划路径并保存视频。"""
    from behavior_interface.skills.arm_reset import _arm_qpos, _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "left").strip().lower()
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    for _ in range(20):
        yield _tuck_make_action(world, arm, hang, 1.0, trunk_pin)

    path, lift_wp_n = build_two_phase_tuck_path_legacy_hard_fwd(
        world, arm, q_start=_HANG_ARM_QPOS,
    )
    z_phase1_end = _eef_world_z_m(world, arm, path[min(lift_wp_n, len(path)) - 1])
    fk = verify_tuck_path(
        world, arm, path, max_forward_m=TUCK_EXEC_FORWARD_MAX_M, samples=24,
    )
    fk_env = _sample_path_z_fwd_envelope(
        world, arm, path,
        z_ref_m=float(z_phase1_end) if z_phase1_end is not None else 0.0,
        z_drop_max_m=0.05,
        max_forward_m=TUCK_EXEC_FORWARD_MAX_M,
        samples_per_seg=16,
    )

    if record_video:
        if video_path:
            vpath = str(video_path)
        else:
            vpath = os.path.join(
                os.path.dirname(__file__), "test", "exec_move",
                f"tuck_legacy_hard_fwd_{arm}.mp4",
            )
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
    else:
        vpath = None
    video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10) if record_video and vpath else None
    frame_log: List[Dict[str, float]] = []
    yield from _yield_play_tuck_smooth(
        world, arm, path, 1.0, ctx,
        frame_log=frame_log,
        video_rec=video_rec,
        lift_wp_n=lift_wp_n,
    )
    video_ok = bool(video_rec.save(ctx)) if video_rec is not None else False
    dyn = _summarize_grip_chest_log(frame_log)
    peak_dyn = float(dyn.get("max_grip_chest_fwd_m", 0.0))
    z_drop_m = 0.0
    if z_phase1_end is not None and frame_log:
        for fr in frame_log:
            if int(fr.get("phase", 0)) != 2:
                continue
            zz = fr.get("eef_z_m")
            if zz is not None:
                z_drop_m = max(z_drop_m, float(z_phase1_end) - float(zz))
    final_fwd = float((_grip_chest_metrics(world, arm).get("grip_chest_fwd_m") or 0.0))
    cur = _arm_qpos(world, arm)
    ctx.log(
        f"[legacy_play] wp={len(path)} phase1={lift_wp_n} "
        f"FK峰={fk['max_forward_cm']:.1f}cm 包络fwd={fk_env.get('max_fwd_cm')}cm "
        f"Z降={fk_env.get('z_drop_cm')}cm | "
        f"动态峰={peak_dyn*100:.1f}cm phase2_Z降={z_drop_m*100:.2f}cm "
        f"终点={final_fwd*100:.1f}cm | 视频={vpath}"
    )
    ctx.set_result({
        "ok": True,
        "arm": arm,
        "planner": "legacy_hard_fwd",
        "waypoints_n": len(path),
        "lift_wp_n": lift_wp_n,
        "fk_peak_cm": round(fk["max_forward_cm"], 2),
        "fk_envelope": fk_env,
        "dyn_peak_cm": round(peak_dyn * 100, 2),
        "phase2_z_drop_cm": round(z_drop_m * 100, 3),
        "final_fwd_cm": round(final_fwd * 100, 2),
        "q_end": np.round(cur, 4).tolist(),
        "video_path": vpath,
        "video_ok": video_ok,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_rrt_waist_fwd",
    description="两阶段 tuck（Phase2 RRT：forward<20cm 且腰距>d0），录 GTA 视频",
)
def diag_tuck_rrt_waist_fwd(
    ctx,
    arm: str = "left",
    record_video: bool = True,
    video_path: Optional[str] = None,
):
    """规划 RRT Phase2 路径并播放录视频。"""
    from behavior_interface.skills.arm_reset import _arm_qpos, _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "left").strip().lower()
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    for _ in range(20):
        yield _tuck_make_action(world, arm, hang, 1.0, trunk_pin)

    ctx.log("[rrt_play] 规划 Phase2 RRT 路径（约 1–3 分钟）…")
    yield world.hold_action()
    path, lift_wp_n, meta = build_two_phase_tuck_path_rrt_waist_fwd(
        world, arm, q_start=_HANG_ARM_QPOS,
    )
    pivot = meta["pivot"]
    d0 = float(meta["waist_d0_m"])
    fk_waist = _sample_path_fwd_waist_envelope(
        world, arm, path[lift_wp_n - 1:],
        waist_pivot_world=pivot,
        waist_d0_m=d0,
        max_forward_m=TUCK_EXEC_FORWARD_MAX_M,
    )
    fk_fwd = verify_tuck_path(
        world, arm, path, max_forward_m=TUCK_EXEC_FORWARD_MAX_M, samples=20,
    )

    if record_video:
        if video_path:
            vpath = str(video_path)
        else:
            vpath = os.path.join(
                os.path.dirname(__file__), "test", "exec_move",
                f"tuck_rrt_waist_fwd_{arm}.mp4",
            )
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
    else:
        vpath = None
    video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10) if record_video and vpath else None
    frame_log: List[Dict[str, float]] = []
    yield from _yield_play_tuck_smooth(
        world, arm, path, 1.0, ctx,
        frame_log=frame_log,
        video_rec=video_rec,
        lift_wp_n=lift_wp_n,
    )
    video_ok = bool(video_rec.save(ctx)) if video_rec is not None else False
    dyn = _summarize_grip_chest_log(frame_log)
    peak_dyn = float(dyn.get("max_grip_chest_fwd_m", 0.0))
    cur = _arm_qpos(world, arm)
    final_fwd = float((_grip_chest_metrics(world, arm).get("grip_chest_fwd_m") or 0.0))
    final_wd = _eef_to_waist_top_dist_m(world, arm, cur, pivot)
    ctx.log(
        f"[rrt_play] wp={len(path)} phase1={lift_wp_n} "
        f"RRT seed={meta.get('rrt', {}).get('seed', meta.get('rrt', {}).get('via'))} | "
        f"FK phase2 fwd峰={fk_waist.get('max_fwd_cm')}cm "
        f"腰距min={fk_waist.get('min_waist_dist_cm')}cm (d0={d0*100:.1f}) | "
        f"动态fwd峰={peak_dyn*100:.1f}cm "
        f"终点fwd={final_fwd*100:.1f}cm "
        f"终点腰距={final_wd*100:.1f}cm | 视频={vpath}"
    )
    ctx.set_result({
        "ok": True,
        "arm": arm,
        "planner": "rrt_waist_fwd",
        "waypoints_n": len(path),
        "lift_wp_n": lift_wp_n,
        "phase1_waist_d0_cm": round(d0 * 100, 2),
        "fk_phase2": fk_waist,
        "fk_fwd_peak_cm": round(fk_fwd["max_forward_cm"], 2),
        "dyn_peak_fwd_cm": round(peak_dyn * 100, 2),
        "final_fwd_cm": round(final_fwd * 100, 2),
        "final_waist_dist_cm": round(float(final_wd) * 100, 2) if final_wd is not None else None,
        "rrt_meta": meta.get("rrt"),
        "q_end": np.round(cur, 4).tolist(),
        "video_path": vpath,
        "video_ok": video_ok,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_rrt_three_constraints",
    description="两阶段 tuck（Phase2 三约束 RRT），录 GTA 视频",
)
def diag_tuck_rrt_three_constraints(
    ctx,
    arm: str = "left",
    record_video: bool = True,
    video_path: Optional[str] = None,
):
    """三约束 RRT Phase2：forward<20cm、腰距>20cm、外移≤5cm，播放录视频。"""
    from behavior_interface.skills.arm_reset import _arm_qpos, _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "left").strip().lower()
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    for _ in range(20):
        yield _tuck_make_action(world, arm, hang, 1.0, trunk_pin)

    ctx.log("[rrt_x3] 规划三约束 RRT 路径（约 1–3 分钟）…")
    yield world.hold_action()
    path, lift_wp_n, meta = build_two_phase_tuck_path_rrt_three_constraints(
        world, arm, q_start=_HANG_ARM_QPOS,
    )
    pivot = meta["pivot"]
    lat0 = float(meta["lat0_m"])
    phase2 = path[min(lift_wp_n, len(path)) - 1:]
    fk_verify = _verify_phase2_path_constraints(
        world, arm, phase2 if len(phase2) >= 2 else path[-2:],
        waist_pivot_world=pivot,
        max_forward_m=TUCK_EXEC_FORWARD_MAX_M,
        waist_min_m=float(meta["waist_min_m"]),
        lat0_m=lat0,
        lateral_outward_max_m=float(meta["lateral_outward_max_m"]),
    )

    if record_video:
        if video_path:
            vpath = str(video_path)
        else:
            vpath = os.path.join(
                os.path.dirname(__file__), "test", "exec_move",
                f"tuck_rrt_three_constraints_{arm}.mp4",
            )
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
    else:
        vpath = None
    video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10) if record_video and vpath else None
    frame_log: List[Dict[str, float]] = []
    yield from _yield_play_tuck_smooth(
        world, arm, path, 1.0, ctx,
        frame_log=frame_log,
        video_rec=video_rec,
        lift_wp_n=lift_wp_n,
    )
    video_ok = bool(video_rec.save(ctx)) if video_rec is not None else False
    dyn = _summarize_grip_chest_log(frame_log)
    peak_dyn = float(dyn.get("max_grip_chest_fwd_m", 0.0))
    cur = _arm_qpos(world, arm)
    final_fwd = float((_grip_chest_metrics(world, arm).get("grip_chest_fwd_m") or 0.0))
    final_wd = _eef_to_waist_top_dist_m(world, arm, cur, pivot)
    ctx.log(
        f"[rrt_x3] wp={len(path)} phase1={lift_wp_n} "
        f"RRT seed={meta.get('rrt', {}).get('seed', meta.get('rrt', {}).get('via'))} | "
        f"FK phase2 fwd峰={fk_verify.get('max_fwd_cm')}cm "
        f"腰距min={fk_verify.get('min_waist_cm')}cm "
        f"外移max={fk_verify.get('max_lateral_outward_cm')}cm | "
        f"动态fwd峰={peak_dyn*100:.1f}cm "
        f"终点fwd={final_fwd*100:.1f}cm 腰距={final_wd*100:.1f}cm | 视频={vpath}"
    )
    ctx.set_result({
        "ok": True,
        "arm": arm,
        "planner": "rrt_three_constraints",
        "waypoints_n": len(path),
        "lift_wp_n": lift_wp_n,
        "phase1_lat_cm": round(lat0 * 100, 2),
        "fk_phase2": fk_verify,
        "dyn_peak_fwd_cm": round(peak_dyn * 100, 2),
        "final_fwd_cm": round(final_fwd * 100, 2),
        "final_waist_dist_cm": round(float(final_wd) * 100, 2) if final_wd is not None else None,
        "rrt_meta": meta.get("rrt"),
        "q_end": np.round(cur, 4).tolist(),
        "video_path": vpath,
        "video_ok": video_ok,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_rrt_four_constraints",
    description="两阶段 tuck（Phase2 四约束 RRT），存在则录 GTA 视频",
)
def diag_tuck_rrt_four_constraints(
    ctx,
    arm: str = "left",
    record_video: bool = True,
    video_path: Optional[str] = None,
):
    """四约束 RRT Phase2：forward<20cm、腰距>15cm、外移≤5cm、Z≥Phase1末，播放录视频。"""
    from behavior_interface.skills.arm_reset import _arm_qpos, _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "left").strip().lower()
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    for _ in range(20):
        yield _tuck_make_action(world, arm, hang, 1.0, trunk_pin)

    ctx.log("[rrt_x4] 先审计四约束轨迹是否存在…")
    yield world.hold_action()
    audit = _audit_phase2_four_constraints(world, arm)
    if not audit.get("trajectory_exists"):
        ctx.log(
            f"[rrt_x4] 轨迹不存在，跳过播放：{audit.get('conclusion')} | "
            f"{(audit.get('trajectory_search') or {}).get('search_note', '')}"
        )
        ctx.set_result({
            "ok": False,
            "arm": arm,
            "trajectory_exists": False,
            "audit": audit,
            "video_path": None,
            "video_ok": False,
        })
        yield world.hold_action()
        return

    ctx.log("[rrt_x4] 轨迹存在，规划四约束 RRT 路径并播放（约 1–3 分钟）…")
    yield world.hold_action()
    path, lift_wp_n, meta = build_two_phase_tuck_path_rrt_four_constraints(
        world, arm, q_start=_HANG_ARM_QPOS,
    )
    pivot = meta["pivot"]
    lat0 = float(meta["lat0_m"])
    z_floor = float(meta["z_floor_m"])
    phase2 = path[min(lift_wp_n, len(path)) - 1:]
    fk_verify = _verify_phase2_path_constraints(
        world, arm, phase2 if len(phase2) >= 2 else path[-2:],
        waist_pivot_world=pivot,
        max_forward_m=TUCK_EXEC_FORWARD_MAX_M,
        waist_min_m=float(meta["waist_min_m"]),
        lat0_m=lat0,
        lateral_outward_max_m=float(meta["lateral_outward_max_m"]),
        z_floor_m=z_floor,
    )

    if record_video:
        if video_path:
            vpath = str(video_path)
        else:
            vpath = os.path.join(
                os.path.dirname(__file__), "test", "exec_move",
                f"tuck_rrt_four_constraints_{arm}.mp4",
            )
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
    else:
        vpath = None
    video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10) if record_video and vpath else None
    frame_log: List[Dict[str, float]] = []
    yield from _yield_play_tuck_smooth(
        world, arm, path, 1.0, ctx,
        frame_log=frame_log,
        video_rec=video_rec,
        lift_wp_n=lift_wp_n,
    )
    video_ok = bool(video_rec.save(ctx)) if video_rec is not None else False
    dyn = _summarize_grip_chest_log(frame_log)
    peak_dyn = float(dyn.get("max_grip_chest_fwd_m", 0.0))
    cur = _arm_qpos(world, arm)
    final_fwd = float((_grip_chest_metrics(world, arm).get("grip_chest_fwd_m") or 0.0))
    final_wd = _eef_to_waist_top_dist_m(world, arm, cur, pivot)
    final_zz = _eef_world_z_m(world, arm, cur)
    ctx.log(
        f"[rrt_x4] wp={len(path)} phase1={lift_wp_n} "
        f"RRT seed={meta.get('rrt', {}).get('seed', meta.get('rrt', {}).get('via'))} | "
        f"FK phase2 fwd峰={fk_verify.get('max_fwd_cm')}cm "
        f"腰距min={fk_verify.get('min_waist_cm')}cm "
        f"外移max={fk_verify.get('max_lateral_outward_cm')}cm "
        f"z_min={fk_verify.get('min_z_m')}m z_floor={z_floor:.4f}m | "
        f"动态fwd峰={peak_dyn*100:.1f}cm "
        f"终点fwd={final_fwd*100:.1f}cm 腰距={final_wd*100:.1f}cm "
        f"z={final_zz:.4f}m | 视频={vpath}"
    )
    ctx.set_result({
        "ok": True,
        "arm": arm,
        "planner": "rrt_four_constraints",
        "trajectory_exists": True,
        "waypoints_n": len(path),
        "lift_wp_n": lift_wp_n,
        "phase1_z_floor_m": z_floor,
        "phase1_lat_cm": round(lat0 * 100, 2),
        "fk_phase2": fk_verify,
        "dyn_peak_fwd_cm": round(peak_dyn * 100, 2),
        "final_fwd_cm": round(final_fwd * 100, 2),
        "final_waist_dist_cm": round(float(final_wd) * 100, 2) if final_wd is not None else None,
        "final_z_m": round(float(final_zz), 4) if final_zz is not None else None,
        "rrt_meta": meta.get("rrt"),
        "q_end": np.round(cur, 4).tolist(),
        "video_path": vpath,
        "video_ok": video_ok,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase2_j34_breakdown",
    description="FK 快速分解：j34 四约束逐项 + 直线插值 + 可行集采样",
)
def diag_tuck_phase2_j34_breakdown(ctx, arm: str = "both", z_drop_max_cm: float = 3.0):
    world = ctx.world
    z_drop_m = float(z_drop_max_cm) / 100.0
    arms = ["left", "right"] if (arm or "both").strip().lower() == "both" else [(arm or "left").strip().lower()]
    reports = {}
    for a in arms:
        yield world.hold_action()
        rep = _audit_phase2_j34_breakdown(world, a, z_drop_max_m=z_drop_m)
        reports[a] = rep
        if rep.get("ok"):
            cg = rep.get("chest_goal") or {}
            p1 = rep.get("phase1_end") or {}
            ctx.log(
                f"[j34_bd/{a}] Z降≤{z_drop_max_cm}cm | {rep.get('verdict')} | "
                f"RRT存在={rep.get('trajectory_exists')} goal={rep.get('goal_feasible')} "
                f"直线={rep.get('straight_lerp_feasible')} | "
                f"P1末 z={p1.get('z_m')}m | 胸前 z={cg.get('z_m')}m "
                f"降={cg.get('z_vs_phase1_cm')}cm fwd={cg.get('fwd_cm')}cm | "
                f"阻断={rep.get('blockers')}"
            )
    ctx.set_result({"ok": True, "z_drop_max_cm": float(z_drop_max_cm), "reports": reports})
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase2_j34_constraints",
    description="FK+RRT：Phase2 禁飞区四约束（fwd/Z降3cm/外移5cm/禁飞区）轨迹是否存在",
)
def diag_tuck_phase2_j34_constraints(ctx, arm: str = "left", z_drop_max_cm: float = 3.0):
    world = ctx.world
    z_drop_m = float(z_drop_max_cm) / 100.0
    ctx.log(f"[p2_j34] 开始禁飞区四约束 RRT+PRM（Z降≤{z_drop_max_cm}cm，约 3–8 分钟）…")
    yield world.hold_action()
    rep = _audit_phase2_j34_constraints(world, arm, z_drop_max_m=z_drop_m)
    if rep.get("ok"):
        ts = rep.get("trajectory_search") or {}
        tv = ts.get("path_verify") or {}
        cg = rep.get("chest_goal") or {}
        p1 = rep.get("phase1_end") or {}
        cst = rep.get("constraints") or {}
        ctx.log(
            f"[p2_j34] 轨迹={'存在' if rep.get('trajectory_exists') else '不存在'} "
            f"({ts.get('found_via') or ts.get('search_note', '')}) | "
            f"Phase1末 z={p1.get('z_m')}m z_floor={p1.get('z_floor_m')}m | "
            f"胸前 fwd={cg.get('fwd_cm')}cm z={cg.get('z_m')}m "
            f"降={cg.get('z_descend_cm')}cm 外移={cg.get('lateral_outward_cm')}cm "
            f"禁飞={'hit' if cg.get('forbidden_hit') else 'ok'} | "
            f"路径 fwd峰={tv.get('max_fwd_cm')}cm "
            f"外移max={tv.get('max_lateral_outward_cm')}cm "
            f"z_min={tv.get('min_z_m')}m 禁触={tv.get('forbidden_hits', 0)}"
        )
    ctx.set_result(rep)
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase2_fwd_minimum",
    description="两阶段 Phase2：外移+禁飞区下 forward 峰值下界（7-DOF RRT 含腕部）",
)
def diag_tuck_phase2_fwd_minimum(
    ctx,
    arm: str = "right",
    lateral_outward_max_cm: float = 3.0,
    quick: bool = False,
    bs_iters: int = 8,
):
    """Phase1 垂直抬升后，二分/扫描 Phase2 forward 下界。"""
    world = ctx.world
    lat_m = float(lateral_outward_max_cm) / 100.0
    iters = 3000 if quick else 3500
    prm_n = 1800 if quick else 2200
    arms = ["left", "right"] if (arm or "both").strip().lower() == "both" else [(arm or "left").strip().lower()]
    reports = {}
    for a in arms:
        yield world.hold_action()
        mode = "快速" if quick else f"二分×{bs_iters}"
        ctx.log(
            f"[p2_fwd_min/{a}] {mode}：Phase1末→胸 forward 下界 "
            f"（外移≤{lateral_outward_max_cm}cm+禁飞区，7-DOF RRT）…"
        )
        rep = _audit_phase2_fwd_minimum(
            world, a,
            lateral_outward_max_m=lat_m,
            rrt_iters=iters,
            prm_samples=prm_n,
            bs_iters=0 if quick else int(bs_iters),
            quick=bool(quick),
            log_fn=ctx.log,
        )
        reports[a] = rep
        if rep.get("ok"):
            ctx.log(f"[p2_fwd_min/{a}] {rep.get('conclusion')}")
    ctx.set_result({
        "ok": True,
        "lateral_outward_max_cm": float(lateral_outward_max_cm),
        "quick": bool(quick),
        "reports": reports,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase2_fwd25_lat3_audit",
    description="两阶段：Phase1 后 Phase2（fwd<25cm / 外移≤3cm / 禁飞区）轨迹是否存在",
)
def diag_tuck_phase2_fwd25_lat3_audit(
    ctx,
    arm: str = "both",
    max_fwd_cm: float = 25.0,
    lateral_outward_max_cm: float = 3.0,
    z_drop_max_cm: Optional[float] = None,
):
    """先 hang→Phase1 垂直位，再 RRT 搜 Phase2→胸前（默认不约束 Z）。"""
    world = ctx.world
    max_fwd_m = float(max_fwd_cm) / 100.0
    lat_m = float(lateral_outward_max_cm) / 100.0
    z_drop_m = (float(z_drop_max_cm) / 100.0) if z_drop_max_cm is not None else None
    arms = ["left", "right"] if (arm or "both").strip().lower() == "both" else [(arm or "left").strip().lower()]
    reports = {}
    for a in arms:
        z_note = f" Z降≤{z_drop_max_cm}cm" if z_drop_m is not None else ""
        ctx.log(
            f"[p2_fwd25/{a}] Phase1→Phase2 存在性：fwd<{max_fwd_cm}cm "
            f"外移≤{lateral_outward_max_cm}cm 禁飞区{z_note}（约 3–8 分钟/臂）…"
        )
        yield world.hold_action()
        rep = _audit_phase2_after_phase1(
            world, a,
            max_forward_m=max_fwd_m,
            lateral_outward_max_m=lat_m,
            z_drop_max_m=z_drop_m,
        )
        reports[a] = rep
        if rep.get("ok"):
            ts = rep.get("trajectory_search") or {}
            tv = ts.get("path_verify") or {}
            cg = rep.get("chest_goal") or {}
            p1 = rep.get("phase1_end") or {}
            ctx.log(
                f"[p2_fwd25/{a}] 轨迹={'存在' if rep.get('trajectory_exists') else '不存在'} "
                f"({ts.get('found_via') or ts.get('search_note', '')}) | "
                f"Phase1末 fwd={p1.get('fwd_cm')}cm lat={p1.get('lat_cm')}cm z={p1.get('z_m')}m "
                f"wp={rep.get('phase1_waypoints')} | "
                f"胸前 fwd={cg.get('fwd_cm')}cm 外移={cg.get('lateral_outward_cm')}cm "
                f"禁飞={'hit' if cg.get('forbidden_hit') else 'ok'} 终点可行={cg.get('feasible')} | "
                f"路径 fwd峰={tv.get('max_fwd_cm')}cm "
                f"外移max={tv.get('max_lateral_outward_cm')}cm 禁触={tv.get('forbidden_hits', 0)}"
            )
        # API 结果去掉 ndarray 路径，避免 set_result 序列化失败
        if rep.get("trajectory_search") and rep["trajectory_search"].get("path") is not None:
            rep = dict(rep)
            ts = dict(rep["trajectory_search"])
            raw_path = ts.pop("path", None)
            if raw_path is not None:
                ts["path_q"] = [np.asarray(q, dtype=np.float64).reshape(7).tolist() for q in raw_path]
                ts["path_waypoints"] = len(ts["path_q"])
            rep["trajectory_search"] = ts
            reports[a] = rep
    ctx.set_result({
        "ok": True,
        "max_fwd_cm": float(max_fwd_cm),
        "lateral_outward_max_cm": float(lateral_outward_max_cm),
        "z_drop_max_cm": z_drop_max_cm,
        "reports": reports,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_phase2_fwd25_lat3_video",
    description="两阶段 tuck：Phase1+Phase2（fwd<25cm/外移≤3cm/禁飞区）存在则录 GTA 视频",
)
def diag_tuck_phase2_fwd25_lat3_video(
    ctx,
    arm: str = "right",
    max_fwd_cm: float = 25.0,
    lateral_outward_max_cm: float = 3.0,
    z_drop_max_cm: Optional[float] = None,
    record_video: bool = True,
    video_path: Optional[str] = None,
):
    """hang→Phase1 垂直抬升→Phase2 RRT，两段分播录像。"""
    from behavior_interface.skills.arm_reset import _arm_qpos, _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "right").strip().lower()
    max_fwd_m = float(max_fwd_cm) / 100.0
    lat_m = float(lateral_outward_max_cm) / 100.0
    z_drop_m = (float(z_drop_max_cm) / 100.0) if z_drop_max_cm is not None else None
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    for _ in range(20):
        yield _tuck_make_action(world, arm, hang, 1.0, trunk_pin)

    ctx.log(
        f"[p2_vid/{arm}] 规划 Phase1 垂直抬升→Phase2（fwd<{max_fwd_cm}cm "
        f"外移≤{lateral_outward_max_cm}cm 禁飞区）…"
    )
    yield world.hold_action()
    try:
        path, lift_wp_n, meta = build_two_phase_tuck_path_phase2_fwd_lat_forbidden(
            world, arm,
            q_start=_HANG_ARM_QPOS,
            max_forward_m=max_fwd_m,
            lateral_outward_max_m=lat_m,
            z_drop_max_m=z_drop_m,
        )
    except RuntimeError as exc:
        ctx.log(f"[p2_vid/{arm}] 规划失败：{exc}")
        ctx.set_result({
            "ok": False,
            "arm": arm,
            "trajectory_exists": False,
            "error": str(exc),
            "video_path": None,
            "video_ok": False,
        })
        yield world.hold_action()
        return

    p1d = meta.get("phase1_path") or path[:lift_wp_n]
    p2d = meta.get("phase2_path") or path[min(lift_wp_n, len(path)) - 1:]
    ctx.log(
        f"[p2_vid/{arm}] Phase1 raw={meta.get('phase1_raw_wp')} dense={len(p1d)} "
        f"j1轨迹={meta.get('phase1_j1_samples')} | Phase2 wp={len(p2d)} "
        f"via={meta.get('rrt', {}).get('seed', meta.get('rrt', {}).get('via'))}"
    )

    # 规划后回 hang，播放方式与 diag_tuck_rrt_three_constraints 一致（合并路径 + lift_wp_n）
    for _ in range(20):
        yield _tuck_make_action(world, arm, hang, 1.0, trunk_pin)

    pivot = meta["pivot"]
    lat0 = float(meta["lat0_m"])
    forbidden_box = meta.get("forbidden_box")
    fk_verify = meta.get("rrt", {}).get("verify") or _verify_phase2_path_constraints(
        world, arm, p2d if len(p2d) >= 2 else path[-2:],
        waist_pivot_world=pivot,
        max_forward_m=max_fwd_m,
        waist_min_m=0.0,
        lat0_m=lat0,
        lateral_outward_max_m=lat_m,
        z_floor_m=z_drop_m,
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        forbidden_fine_mesh=True,
        require_z_descend=z_drop_m is not None,
    )

    vpath = None
    if record_video:
        if video_path:
            vpath = str(video_path)
        else:
            vpath = os.path.join(
                os.path.dirname(__file__), "test", "exec_move",
                f"tuck_phase2_fwd25_lat3_{arm}.mp4",
            )
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
    video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10) if record_video and vpath else None
    frame_log: List[Dict[str, float]] = []
    # Phase1：逐路点（与 three_constraints 左臂录像同样可见的后摆）；Phase2：弧长平滑
    yield from _yield_play_tuck_two_segments(
        world, arm, p1d, p2d, 1.0, ctx,
        frame_log=frame_log,
        video_rec=video_rec,
        phase1_frames_per_wp=TUCK_FRAMES_PER_WP,
    )
    video_ok = bool(video_rec.save(ctx)) if video_rec is not None else False
    dyn = _summarize_grip_chest_log(frame_log)
    peak_dyn = float(dyn.get("max_grip_chest_fwd_m", 0.0))
    cur = _arm_qpos(world, arm)
    final_fwd = float((_grip_chest_metrics(world, arm).get("grip_chest_fwd_m") or 0.0))
    ctx.log(
        f"[p2_vid/{arm}] wp={len(path)} phase1={lift_wp_n} "
        f"via={meta.get('rrt', {}).get('seed', meta.get('rrt', {}).get('via'))} | "
        f"FK phase2 fwd峰={fk_verify.get('max_fwd_cm')}cm "
        f"外移max={fk_verify.get('max_lateral_outward_cm')}cm 禁触={fk_verify.get('forbidden_hits', 0)} | "
        f"动态fwd峰={peak_dyn*100:.1f}cm 终点fwd={final_fwd*100:.1f}cm | 视频={vpath}"
    )
    ctx.set_result({
        "ok": True,
        "arm": arm,
        "planner": "phase2_fwd25_lat3",
        "trajectory_exists": True,
        "found_via": meta.get("rrt", {}).get("seed", meta.get("rrt", {}).get("via")),
        "waypoints_n": len(path),
        "lift_wp_n": lift_wp_n,
        "phase1_j1_samples": meta.get("phase1_j1_samples"),
        "fk_phase2": fk_verify,
        "dyn_peak_fwd_cm": round(peak_dyn * 100, 2),
        "final_fwd_cm": round(final_fwd * 100, 2),
        "z_phase1_m": meta.get("z_phase1_m"),
        "q_end": np.round(cur, 4).tolist(),
        "video_path": vpath,
        "video_ok": video_ok,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_rrt_j34_constraints",
    description="两阶段 tuck（Phase2 禁飞区四约束 RRT），存在则录 GTA 视频",
)
def diag_tuck_rrt_j34_constraints(
    ctx,
    arm: str = "left",
    record_video: bool = True,
    video_path: Optional[str] = None,
):
    """禁飞区四约束 Phase2：fwd<20cm、Z降≤3cm、外移≤5cm、夹爪不进禁飞区。"""
    from behavior_interface.skills.arm_reset import _arm_qpos, _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "left").strip().lower()
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    for _ in range(20):
        yield _tuck_make_action(world, arm, hang, 1.0, trunk_pin)

    ctx.log("[rrt_j34] 先审计禁飞区四约束轨迹是否存在…")
    yield world.hold_action()
    audit = _audit_phase2_j34_constraints(world, arm)
    if not audit.get("trajectory_exists"):
        ctx.log(
            f"[rrt_j34] 轨迹不存在，跳过播放：{audit.get('conclusion')} | "
            f"{(audit.get('trajectory_search') or {}).get('search_note', '')}"
        )
        ctx.set_result({
            "ok": False,
            "arm": arm,
            "trajectory_exists": False,
            "audit": audit,
            "video_path": None,
            "video_ok": False,
        })
        yield world.hold_action()
        return

    ctx.log("[rrt_j34] 轨迹存在，规划并播放（约 2–5 分钟）…")
    yield world.hold_action()
    path, lift_wp_n, meta = build_two_phase_tuck_path_rrt_j34_constraints(
        world, arm, q_start=_HANG_ARM_QPOS,
    )
    pivot = meta["pivot"]
    lat0 = float(meta["lat0_m"])
    z_floor = float(meta["z_floor_m"])
    z_phase1 = float(meta["z_phase1_m"])
    forbidden_box = meta.get("forbidden_box")
    phase2 = path[min(lift_wp_n, len(path)) - 1:]
    fk_verify = _verify_phase2_path_constraints(
        world, arm, phase2 if len(phase2) >= 2 else path[-2:],
        waist_pivot_world=pivot,
        max_forward_m=TUCK_EXEC_FORWARD_MAX_M,
        waist_min_m=0.0,
        lat0_m=lat0,
        lateral_outward_max_m=float(meta["lateral_outward_max_m"]),
        z_floor_m=z_floor,
        strict_waist_min=False,
        forbidden_box=forbidden_box,
        z_phase1_ref_m=z_phase1,
        require_z_descend=True,
    )

    if record_video:
        if video_path:
            vpath = str(video_path)
        else:
            vpath = os.path.join(
                os.path.dirname(__file__), "test", "exec_move",
                f"tuck_rrt_j34_constraints_{arm}.mp4",
            )
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
    else:
        vpath = None
    video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10) if record_video and vpath else None
    frame_log: List[Dict[str, float]] = []
    yield from _yield_play_tuck_smooth(
        world, arm, path, 1.0, ctx,
        frame_log=frame_log,
        video_rec=video_rec,
        lift_wp_n=lift_wp_n,
    )
    video_ok = bool(video_rec.save(ctx)) if video_rec is not None else False
    dyn = _summarize_grip_chest_log(frame_log)
    peak_dyn = float(dyn.get("max_grip_chest_fwd_m", 0.0))
    cur = _arm_qpos(world, arm)
    final_fwd = float((_grip_chest_metrics(world, arm).get("grip_chest_fwd_m") or 0.0))
    final_zz = _eef_world_z_m(world, arm, cur)
    ctx.log(
        f"[rrt_j34] wp={len(path)} phase1={lift_wp_n} | "
        f"FK phase2 fwd峰={fk_verify.get('max_fwd_cm')}cm "
        f"外移max={fk_verify.get('max_lateral_outward_cm')}cm "
        f"z_min={fk_verify.get('min_z_m')}m z降={fk_verify.get('z_descend_cm')}cm "
        f"禁触={fk_verify.get('forbidden_hits', 0)} | "
        f"动态fwd峰={peak_dyn*100:.1f}cm 终点fwd={final_fwd*100:.1f}cm "
        f"z={final_zz:.4f}m | 视频={vpath}"
    )
    ctx.set_result({
        "ok": True,
        "arm": arm,
        "planner": "rrt_j34_constraints",
        "trajectory_exists": True,
        "waypoints_n": len(path),
        "lift_wp_n": lift_wp_n,
        "phase1_z_m": z_phase1,
        "phase1_z_floor_m": z_floor,
        "phase1_lat_cm": round(lat0 * 100, 2),
        "fk_phase2": fk_verify,
        "dyn_peak_fwd_cm": round(peak_dyn * 100, 2),
        "final_fwd_cm": round(final_fwd * 100, 2),
        "final_z_m": round(float(final_zz), 4) if final_zz is not None else None,
        "rrt_meta": meta.get("rrt"),
        "q_end": np.round(cur, 4).tolist(),
        "video_path": vpath,
        "video_ok": video_ok,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_hang_j34_fwd_minimum",
    description="hang→胸：外移≤5cm+禁飞区下 forward 峰值下界（<20/<25cm 可行性）",
)
def diag_tuck_hang_j34_fwd_minimum(
    ctx,
    arm: str = "both",
    lateral_outward_max_cm: float = 5.0,
    quick: bool = True,
):
    world = ctx.world
    lat_m = float(lateral_outward_max_cm) / 100.0
    iters = 2500 if quick else 3500
    prm_n = 1800 if quick else 2000
    arms = ["left", "right"] if (arm or "both").strip().lower() == "both" else [(arm or "left").strip().lower()]
    reports = {}
    for a in arms:
        yield world.hold_action()
        ctx.log(
            f"[fwd_min/{a}] {'快速' if quick else '精细'}扫描 forward（外移≤{lateral_outward_max_cm}cm+禁飞区）…"
        )
        rep = _audit_hang_j34_fwd_minimum(
            world, a,
            lateral_outward_max_m=lat_m,
            rrt_iters=iters,
            prm_samples=prm_n,
            bs_iters=0 if quick else 7,
            quick=quick,
        )
        reports[a] = rep
        if rep.get("ok"):
            ctx.log(f"[fwd_min/{a}] {rep.get('conclusion')}")
            if not rep.get("path_exists_unrestricted_fwd"):
                ctx.log(f"[fwd_min/{a}] 放宽 forward 仍无路径")
    ctx.set_result({
        "ok": True,
        "lateral_outward_max_cm": float(lateral_outward_max_cm),
        "reports": reports,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_hang_j34_fwd_refine_video",
    description="hang→胸：外移≤5cm+禁飞区，二分求最小 forward 并录 GTA 视频",
)
def diag_tuck_hang_j34_fwd_refine_video(
    ctx,
    arm: str = "left",
    lateral_outward_max_cm: float = 5.0,
    fwd_lo_cm: float = 25.0,
    fwd_hi_cm: float = 66.0,
    bs_iters: int = 8,
    record_video: bool = True,
    video_path: Optional[str] = None,
):
    """外移≤5cm + 禁飞区净空，二分 [25,66]cm 找最小 forward 路径并播放录像。"""
    from behavior_interface.skills.arm_reset import _arm_qpos, _HANG_ARM_QPOS

    world = ctx.world
    lat_m = float(lateral_outward_max_cm) / 100.0
    arms = ["left", "right"] if (arm or "both").strip().lower() == "both" else [(arm or "left").strip().lower()]
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    all_reports: dict = {}
    video_paths: dict = {}

    for a in arms:
        for _ in range(20):
            yield _tuck_make_action(world, a, hang, 1.0, trunk_pin)
        yield world.hold_action()
        ctx.log(
            f"[fwd_refine/{a}] 二分 forward [{fwd_lo_cm},{fwd_hi_cm}]cm "
            f"（外移≤{lateral_outward_max_cm}cm+禁飞区，约 {bs_iters+1}×RRT/臂）…"
        )
        yield world.hold_action()
        rep = _refine_hang_j34_fwd_binary(
            world, a,
            lateral_outward_max_m=lat_m,
            lo_cm=float(fwd_lo_cm),
            hi_cm=float(fwd_hi_cm),
            bs_iters=int(bs_iters),
            log_fn=ctx.log,
        )
        all_reports[a] = {
            k: rep.get(k)
            for k in (
                "ok", "error", "min_forward_cap_cm", "min_forward_peak_cm",
                "max_lateral_outward_cm", "forbidden_hits", "path_ok",
                "found_via", "verify", "bs_history", "conclusion",
            )
        }
        if not rep.get("ok") or not rep.get("path"):
            ctx.log(f"[fwd_refine/{a}] 失败：{rep.get('error', '无路径')}")
            continue

        ctx.log(f"[fwd_refine/{a}] {rep.get('conclusion')}")
        path = _densify_joint_path(rep["path"], max_dq=TUCK_DENSIFY_MAX_DQ)

        vpath = None
        video_ok = False
        if record_video:
            if video_path and len(arms) == 1:
                vpath = str(video_path)
            else:
                vpath = os.path.join(
                    os.path.dirname(__file__), "test", "exec_move",
                    f"tuck_hang_j34_min_fwd_{a}.mp4",
                )
            os.makedirs(os.path.dirname(vpath), exist_ok=True)
            video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10)
            frame_log: List[Dict[str, float]] = []
            yield from _yield_play_tuck_smooth(
                world, a, path, 1.0, ctx,
                frame_log=frame_log,
                video_rec=video_rec,
                lift_wp_n=None,
            )
            video_ok = bool(video_rec.save(ctx))
            video_paths[a] = vpath

        cur = _arm_qpos(world, a)
        final_fwd = float((_grip_chest_metrics(world, a).get("grip_chest_fwd_m") or 0.0))
        all_reports[a].update({
            "waypoints_n": len(path),
            "video_path": vpath,
            "video_ok": video_ok,
            "final_fwd_cm": round(final_fwd * 100, 2),
            "q_end": np.round(cur, 4).tolist(),
        })
        ctx.log(f"[fwd_refine/{a}] 播放完成 wp={len(path)} 终点fwd={final_fwd*100:.1f}cm 视频={vpath}")

    ctx.set_result({
        "ok": True,
        "lateral_outward_max_cm": float(lateral_outward_max_cm),
        "fwd_bracket_cm": [float(fwd_lo_cm), float(fwd_hi_cm)],
        "bs_iters": int(bs_iters),
        "reports": all_reports,
        "video_paths": video_paths,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_hang_j34_fwd_penalty_video",
    description="hang→胸：外移≤5cm+禁飞区，强forward惩罚求最小峰值并录 GTA 视频",
)
def diag_tuck_hang_j34_fwd_penalty_video(
    ctx,
    arm: str = "left",
    lateral_outward_max_cm: float = 5.0,
    fwd_penalty_weight: float = TUCK_HANG_FWD_PENALTY_WEIGHT,
    record_video: bool = True,
    video_path: Optional[str] = None,
):
    """外移+禁飞区硬约束，强 forward 惩罚搜最小峰值路径并录像。"""
    from behavior_interface.skills.arm_reset import _arm_qpos, _HANG_ARM_QPOS

    world = ctx.world
    lat_m = float(lateral_outward_max_cm) / 100.0
    arms = ["left", "right"] if (arm or "both").strip().lower() == "both" else [(arm or "left").strip().lower()]
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    all_reports: dict = {}
    video_paths: dict = {}

    for a in arms:
        for _ in range(20):
            yield _tuck_make_action(world, a, hang, 1.0, trunk_pin)
        yield world.hold_action()
        ctx.log(
            f"[fwd_pen/{a}] 强惩罚搜最小 forward（外移≤{lateral_outward_max_cm}cm+禁飞区，"
            f"weight={fwd_penalty_weight}）…"
        )
        yield world.hold_action()
        rep = _search_hang_j34_min_fwd_penalty(
            world, a,
            lateral_outward_max_m=lat_m,
            fwd_penalty_weight=float(fwd_penalty_weight),
            log_fn=ctx.log,
        )
        all_reports[a] = {
            k: rep.get(k)
            for k in (
                "ok", "error", "min_forward_peak_cm", "max_lateral_outward_cm",
                "forbidden_hits", "path_ok", "found_via", "verify", "trials",
                "conclusion", "fwd_penalty_weight",
            )
        }
        if not rep.get("ok") or not rep.get("path"):
            ctx.log(f"[fwd_pen/{a}] 失败：{rep.get('error', '无路径')}")
            continue

        ctx.log(f"[fwd_pen/{a}] {rep.get('conclusion')}")
        path = _densify_joint_path(rep["path"], max_dq=TUCK_DENSIFY_MAX_DQ)

        vpath = None
        video_ok = False
        if record_video:
            if video_path and len(arms) == 1:
                vpath = str(video_path)
            else:
                vpath = os.path.join(
                    os.path.dirname(__file__), "test", "exec_move",
                    f"tuck_hang_j34_min_fwd_penalty_{a}.mp4",
                )
            os.makedirs(os.path.dirname(vpath), exist_ok=True)
            video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10)
            frame_log: List[Dict[str, float]] = []
            yield from _yield_play_tuck_smooth(
                world, a, path, 1.0, ctx,
                frame_log=frame_log,
                video_rec=video_rec,
                lift_wp_n=None,
            )
            video_ok = bool(video_rec.save(ctx))
            video_paths[a] = vpath

        cur = _arm_qpos(world, a)
        final_fwd = float((_grip_chest_metrics(world, a).get("grip_chest_fwd_m") or 0.0))
        all_reports[a].update({
            "waypoints_n": len(path),
            "video_path": vpath,
            "video_ok": video_ok,
            "final_fwd_cm": round(final_fwd * 100, 2),
            "q_end": np.round(cur, 4).tolist(),
        })
        ctx.log(f"[fwd_pen/{a}] 播放完成 wp={len(path)} 终点fwd={final_fwd*100:.1f}cm 视频={vpath}")

    ctx.set_result({
        "ok": True,
        "lateral_outward_max_cm": float(lateral_outward_max_cm),
        "fwd_penalty_weight": float(fwd_penalty_weight),
        "reports": all_reports,
        "video_paths": video_paths,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_hang_j34_relaxed_audit",
    description="FK+RRT：hang→胸 三约束（fwd/外移3cm/禁飞区）轨迹是否存在",
)
def diag_tuck_hang_j34_relaxed_audit(
    ctx,
    arm: str = "both",
    lateral_outward_max_cm: float = 3.0,
):
    world = ctx.world
    lat_m = float(lateral_outward_max_cm) / 100.0
    arms = ["left", "right"] if (arm or "both").strip().lower() == "both" else [(arm or "left").strip().lower()]
    reports = {}
    for a in arms:
        yield world.hold_action()
        ctx.log(f"[hang_j34_rel/{a}] 审计 hang→胸 三约束（外移≤{lateral_outward_max_cm}cm）…")
        rep = _audit_hang_j34_relaxed(world, a, lateral_outward_max_m=lat_m)
        reports[a] = rep
        if rep.get("ok"):
            cg = rep.get("chest_goal") or {}
            h0 = rep.get("hang_start") or {}
            tv = (rep.get("trajectory_search") or {}).get("path_verify") or {}
            ctx.log(
                f"[hang_j34_rel/{a}] {rep.get('conclusion')} | goal={rep.get('goal_feasible')} | "
                f"hang z={h0.get('z_m')}m lat={h0.get('lat_cm')}cm | "
                f"胸前 fwd={cg.get('fwd_cm')}cm 外移={cg.get('lateral_outward_cm')}cm "
                f"z={cg.get('z_m')}m 禁飞={'hit' if cg.get('forbidden_hit') else 'ok'} | "
                f"路径 fwd峰={tv.get('max_fwd_cm')}cm 外移max={tv.get('max_lateral_outward_cm')}cm "
                f"禁触={tv.get('forbidden_hits', 0)}"
            )
    ctx.set_result({"ok": True, "lateral_outward_max_cm": float(lateral_outward_max_cm), "reports": reports})
    yield world.hold_action()


@register_skill(
    "diag_tuck_rrt_j34_relaxed",
    description="hang→胸 三约束 RRT（fwd<20cm/外移≤3cm/禁飞区），无 Phase1，录 GTA 视频",
)
def diag_tuck_rrt_j34_relaxed(
    ctx,
    arm: str = "left",
    lateral_outward_max_cm: float = 3.0,
    record_video: bool = True,
    video_path: Optional[str] = None,
    planner: str = "auto",
):
    """悬垂 hang 直接到胸前：forward<20cm、外移≤3cm、夹爪不进禁飞区（无 Phase1/Z 约束）。

    planner: auto（先 RRT，失败则直线插值演示）/ rrt / lerp_demo
    """
    from behavior_interface.skills.arm_reset import _arm_qpos, _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "left").strip().lower()
    lat_max_m = float(lateral_outward_max_cm) / 100.0
    hang = _HANG_ARM_QPOS.tolist()
    trunk_pin = world.trunk_qpos().copy()
    for _ in range(20):
        yield _tuck_make_action(world, arm, hang, 1.0, trunk_pin)

    ctx.log(
        f"[rrt_j34_rel] hang→胸 三约束（外移≤{lateral_outward_max_cm}cm planner={planner}）…"
    )
    yield world.hold_action()
    planner_mode = (planner or "auto").strip().lower()
    trajectory_exists = False
    if planner_mode == "lerp_demo":
        path, fk_verify, meta = build_hang_to_chest_j34_relaxed_lerp_demo(
            world, arm,
            q_start=_HANG_ARM_QPOS,
            lateral_outward_max_m=lat_max_m,
        )
        pivot = meta["pivot"]
        lat0 = float(meta["lat0_m"])
    elif planner_mode == "rrt":
        path, lift_wp_n, meta = build_hang_to_chest_j34_relaxed_path(
            world, arm,
            q_start=_HANG_ARM_QPOS,
            lateral_outward_max_m=lat_max_m,
        )
        planner_mode = "rrt"
        trajectory_exists = True
        pivot = meta["pivot"]
        lat0 = float(meta["lat0_m"])
        fk_verify = _verify_j34_relaxed_path(
            world, arm, path,
            waist_pivot_world=pivot,
            max_forward_m=TUCK_EXEC_FORWARD_MAX_M,
            lat0_m=lat0,
            lateral_outward_max_m=lat_max_m,
            forbidden_box=meta.get("forbidden_box"),
        )
    else:
        try:
            path, lift_wp_n, meta = build_hang_to_chest_j34_relaxed_path(
                world, arm,
                q_start=_HANG_ARM_QPOS,
                lateral_outward_max_m=lat_max_m,
            )
            planner_mode = "rrt"
            trajectory_exists = True
            pivot = meta["pivot"]
            lat0 = float(meta["lat0_m"])
            fk_verify = _verify_j34_relaxed_path(
                world, arm, path,
                waist_pivot_world=pivot,
                max_forward_m=TUCK_EXEC_FORWARD_MAX_M,
                lat0_m=lat0,
                lateral_outward_max_m=lat_max_m,
                forbidden_box=meta.get("forbidden_box"),
            )
        except Exception as exc:
            ctx.log(
                f"[rrt_j34_rel] RRT 未连通（{exc}），回退直线插值演示（路径可能违反约束）…"
            )
            yield world.hold_action()
            path, fk_verify, meta = build_hang_to_chest_j34_relaxed_lerp_demo(
                world, arm,
                q_start=_HANG_ARM_QPOS,
                lateral_outward_max_m=lat_max_m,
            )
            planner_mode = "lerp_demo"
            pivot = meta["pivot"]
            lat0 = float(meta["lat0_m"])

    if record_video:
        if video_path:
            vpath = str(video_path)
        else:
            suffix = "_lerp_demo" if planner_mode == "lerp_demo" else ""
            vpath = os.path.join(
                os.path.dirname(__file__), "test", "exec_move",
                f"tuck_rrt_j34_relaxed_{arm}{suffix}.mp4",
            )
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
    else:
        vpath = None
    video_rec = _make_tuck_video_recorder(ctx, vpath, fps=10) if record_video and vpath else None
    frame_log: List[Dict[str, float]] = []
    yield from _yield_play_tuck_smooth(
        world, arm, path, 1.0, ctx,
        frame_log=frame_log,
        video_rec=video_rec,
        lift_wp_n=None,
    )
    video_ok = bool(video_rec.save(ctx)) if video_rec is not None else False
    dyn = _summarize_grip_chest_log(frame_log)
    peak_dyn = float(dyn.get("max_grip_chest_fwd_m", 0.0))
    cur = _arm_qpos(world, arm)
    final_fwd = float((_grip_chest_metrics(world, arm).get("grip_chest_fwd_m") or 0.0))
    final_zz = _eef_world_z_m(world, arm, cur)
    ctx.log(
        f"[rrt_j34_rel] mode={planner_mode} wp={len(path)} | "
        f"FK fwd峰={fk_verify.get('max_fwd_cm')}cm "
        f"外移max={fk_verify.get('max_lateral_outward_cm')}cm "
        f"禁触={fk_verify.get('forbidden_hits', 0)} ok={fk_verify.get('ok')} | "
        f"动态fwd峰={peak_dyn*100:.1f}cm 终点fwd={final_fwd*100:.1f}cm "
        f"z={final_zz:.4f}m | 视频={vpath}"
    )
    ctx.set_result({
        "ok": True,
        "arm": arm,
        "planner": planner_mode,
        "trajectory_exists": trajectory_exists,
        "path_feasible": bool(fk_verify.get("ok")),
        "waypoints_n": len(path),
        "lateral_outward_max_cm": float(lateral_outward_max_cm),
        "hang_lat_cm": round(lat0 * 100, 2),
        "fk_path": {
            k: fk_verify.get(k)
            for k in (
                "ok", "max_fwd_cm", "max_lateral_outward_cm",
                "forbidden_hits", "forbidden_ok",
            )
        },
        "dyn_peak_fwd_cm": round(peak_dyn * 100, 2),
        "final_fwd_cm": round(final_fwd * 100, 2),
        "final_z_m": round(float(final_zz), 4) if final_zz is not None else None,
        "rrt_meta": meta.get("rrt") if planner_mode == "rrt" else None,
        "q_end": np.round(cur, 4).tolist(),
        "video_path": vpath,
        "video_ok": video_ok,
    })
    yield world.hold_action()


@register_skill(
    "diag_tuck_verify_path",
    description="在指定俯身trunk下pin住，播候选折线，测全程forward峰值(碰撞判据)",
)
def diag_tuck_verify_path(
    ctx,
    arm: str = "left",
    trunk_q: Optional[list] = None,
    frames_each: int = 20,
    strategy: str = "S2",
):
    """pin trunk 到指定俯身角，播候选折线，forward峰值≤22 即该俯身角不碰撞。"""
    from behavior_interface.skills.arm_reset import _arm_qpos
    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    world = ctx.world
    arm = (arm or "left").strip().lower()
    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name="diag_tuck_verify.prepare"
    )
    base = CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"]).copy()

    # 默认最深俯身（实测导航俯身 θz≈118°）
    if trunk_q is None:
        trunk_q = [0.60, -0.59, -0.49, 0.005]
    trunk_q = np.asarray(trunk_q, dtype=np.float64).reshape(4)

    elbow_first = base.copy(); elbow_first[0] = 0.0
    half_elbow = base.copy(); half_elbow[0] = 0.0; half_elbow[3] = base[3] * 0.5
    strat_map = {
        "S0": [elbow_first],            # 只弯肘、肩不后收 —— 贴胸候选终点
        "S1": [base],
        "S2": [elbow_first, base],
        "S3": [half_elbow, elbow_first, base],
    }
    goals = strat_map.get(strategy, strat_map["S2"])

    _set_trunk_qpos(world, trunk_q)
    theta = float(world.chest_pose().get("theta_z_deg", 90.0))
    trunk_hold = trunk_q.tolist()
    ctx.log(
        f"[verify_path] arm={arm} strategy={strategy} pin θz={theta:.1f}° "
        f"trunk={np.round(trunk_q,3).tolist()} goals={len(goals)}"
    )

    # 只关注手臂链 link 参与的自碰（忽略例如双脚-地面等已被过滤的非自碰）
    sc_links = set()  # 全程出现过的自碰 link 对

    def _goto(q_target, frames, peak_box, track_sc):
        qa = np.asarray(q_target, dtype=np.float64).reshape(7).tolist()
        for _ in range(int(frames)):
            yield _tuck_make_action(world, arm, qa, 1.0, trunk_hold)
            m = _grip_chest_metrics(world, arm)
            fwd = float(m.get("grip_chest_fwd_m") or 0.0)
            peak_box[0] = max(peak_box[0], fwd)
            peak_box[1] = fwd
            if track_sc:
                pairs = _self_collision_pairs(world)
                if pairs:
                    peak_box[2] = 1
                    sc_links.update(pairs)

    # 回 hang（不计入自碰统计）
    hbox = [0.0, 0.0, 0]
    yield from _goto(np.zeros(7), 18, hbox, False)

    overall_peak = 0.0
    any_sc = False
    per_wp = []
    for gi, g in enumerate(goals):
        box = [0.0, 0.0, 0]
        yield from _goto(g, frames_each, box, True)
        overall_peak = max(overall_peak, box[0])
        any_sc = any_sc or bool(box[2])
        per_wp.append({"wp": gi, "peak_cm": round(box[0]*100, 2),
                       "final_cm": round(box[1]*100, 2), "self_collide": bool(box[2])})
        ctx.log(f"[verify_path]   wp{gi}: peak={per_wp[-1]['peak_cm']}cm "
                f"final={per_wp[-1]['final_cm']}cm self_collide={bool(box[2])}")

    cur = _arm_qpos(world, arm)
    final_fwd = float((_grip_chest_metrics(world, arm).get("grip_chest_fwd_m") or 0.0))
    pass_22 = overall_peak <= 0.22 + 1e-6
    pass_nosc = not any_sc
    ctx.log(
        f"[verify_path] θz={theta:.1f}° forward峰值={overall_peak*100:.2f}cm "
        f"final={final_fwd*100:.2f}cm final_j4={cur[3]:.3f} | "
        f"forward≤22={'✅' if pass_22 else '✗'} 不自碰={'✅' if pass_nosc else '✗'}"
    )
    if sc_links:
        ctx.log(f"[verify_path] 自碰 link 对: {sorted(sc_links)}")
    ctx.set_result({"ok": True, "arm": arm, "strategy": strategy,
                    "theta_z": theta, "trunk_q": trunk_q.tolist(),
                    "overall_peak_cm": round(overall_peak*100, 2),
                    "final_fwd_cm": round(final_fwd*100, 2),
                    "final_j4": round(float(cur[3]), 3),
                    "per_wp": per_wp, "pass_22": bool(pass_22),
                    "self_collide": bool(any_sc), "pass_no_self_collision": bool(pass_nosc),
                    "sc_links": sorted(sc_links)})
    yield world.hold_action()


@register_skill(
    "diag_tuck_path_strategies",
    description="当前躯干：对比多种肘绕行路径的全程 forward 峰值",
)
def diag_tuck_path_strategies(ctx, arm: str = "left", frames_each: int = 26):
    """只动手臂，测几种关节运动时序，找全程 forward≤22cm 的绕行路径。"""
    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    world = ctx.world
    arm = (arm or "left").strip().lower()
    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name="diag_tuck_paths.prepare"
    )
    base = CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"]).copy()
    theta = float(world.chest_pose().get("theta_z_deg", 90.0))
    hang = np.zeros(7, dtype=np.float64)

    # 中间位形：先弯肘+上臂（肩 j1 暂不后收），再整体收肩到终点
    elbow_first = base.copy()
    elbow_first[0] = 0.0  # 肩 j1 保持 hang，先把肘 j4/上臂 j3 弯起来
    # 中间位形2：肘先弯一半，避免一次到位中途外扫
    half_elbow = base.copy()
    half_elbow[0] = 0.0
    half_elbow[3] = base[3] * 0.5

    strategies = {
        "S1_direct": [base],
        "S2_elbow_then_shoulder": [elbow_first, base],
        "S3_halfelbow_elbow_shoulder": [half_elbow, elbow_first, base],
    }

    ctx.log(f"[path_strat] arm={arm} θz={theta:.1f}° 测 {len(strategies)} 种路径")
    results = {}
    for name, goals in strategies.items():
        def _goto_hang():
            qa = hang.tolist()
            for _ in range(22):
                yield _tuck_make_action(world, arm, qa, 1.0, None)
        yield from _goto_hang()
        peak, last_fwd, cur = yield from _play_goals_record(
            world, arm, goals, frames_each, None,
        )
        rec = {
            "peak_fwd_cm": round(peak * 100, 2),
            "final_fwd_cm": round(last_fwd * 100, 2),
            "final_j4": round(float(cur[3]), 3),
            "ok_22": bool(peak <= 0.22 + 1e-6),
        }
        results[name] = rec
        ctx.log(
            f"[path_strat] {name}: peak_fwd={rec['peak_fwd_cm']}cm "
            f"final_fwd={rec['final_fwd_cm']}cm final_j4={rec['final_j4']} "
            f"{'✅≤22' if rec['ok_22'] else '✗>22'}"
        )

    winners = [n for n, r in results.items() if r["ok_22"]]
    ctx.log(f"[path_strat] 全程≤22cm 的路径: {winners or '无'}")
    ctx.set_result({"ok": True, "arm": arm, "theta_z": theta, "results": results,
                    "winners": winners})
    yield world.hold_action()


@register_skill(
    "diag_tuck_elbow_scan",
    description="当前躯干：扫描肘(j4)各深度的物理可达 q_cur 与 forward",
)
def diag_tuck_elbow_scan(ctx, arm: str = "left", hold_frames: int = 50):
    """在当前躯干姿态下，命令一组肘深度的胸前位形，看真实可达与 forward。"""
    from behavior_interface.skills.arm_reset import _arm_qpos
    from behavior_interface.skills.eef import _prepare_legacy_7dof_motion

    world = ctx.world
    arm = (arm or "left").strip().lower()
    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name="diag_tuck_elbow.prepare"
    )
    base = CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"]).copy()
    theta = float(world.chest_pose().get("theta_z_deg", 90.0))
    hang = np.zeros(7, dtype=np.float64)
    ctx.log(f"[elbow_scan] arm={arm} 当前躯干 θz={theta:.1f}° base_j4={base[3]:.2f}")

    def _goto(q_target, frames):
        qa = np.asarray(q_target, dtype=np.float64).reshape(7).tolist()
        for _ in range(int(frames)):
            yield _tuck_make_action(world, arm, qa, 1.0, None)

    results = []
    j4_vals = [-2.04, -1.7, -1.4, -1.1, -0.8, -0.5]
    for j4 in j4_vals:
        yield from _goto(hang, 22)
        cand = base.copy()
        cand[3] = j4
        yield from _goto(cand, hold_frames)
        cur = _arm_qpos(world, arm)
        m = _grip_chest_metrics(world, arm)
        rec = {
            "j4_cmd": round(float(j4), 3),
            "j4_cur": round(float(cur[3]), 3),
            "j4_err": round(float(j4 - cur[3]), 3),
            "fwd_cm": round(float(m["grip_chest_fwd_m"]) * 100, 2),
            "q_cur": np.round(cur, 3).tolist(),
        }
        results.append(rec)
        ctx.log(
            f"[elbow_scan] j4_cmd={rec['j4_cmd']} → j4_cur={rec['j4_cur']} "
            f"(err={rec['j4_err']}) forward={rec['fwd_cm']}cm"
        )

    feasible = [r for r in results if abs(r["j4_err"]) < 0.15 and r["fwd_cm"] <= 22.0]
    ctx.log(
        f"[elbow_scan] forward≤22cm 且肘到位的候选: "
        + (", ".join(f"j4={r['j4_cmd']}({r['fwd_cm']}cm)" for r in feasible) or "无")
    )
    ctx.set_result({"ok": True, "arm": arm, "theta_z": theta, "scan": results,
                    "feasible": feasible})
    yield world.hold_action()


@register_skill(
    "diag_tuck_chest_fk_audit",
    description="FK 审计：当前/导航俯身 trunk 与直立对比（逐点 + 折线峰值）",
)
def diag_tuck_chest_fk_audit(
    ctx,
    arm: str = "left",
    use_current_trunk: bool = True,
    pitch_deg: float = 0.0,
):
    """不播轨迹；用当前或指定俯身角 trunk，与直立 FK 对比夹爪-胸廓距离。"""
    from behavior_interface.skills.move_to import move_in_robot_coord
    from behavior_interface.skills.reset_body import reset_body

    world = ctx.world
    arm = (arm or "left").strip().lower()
    path = _TUCK_TRAJECTORY.get(arm) or _TUCK_TRAJECTORY["right"]
    q_goal = CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"])
    # 导航常见大俯身（用户日志里的 trunk，θz 未必与 pitch_only 相同）
    trunk_nav_bent = np.array([0.5355, -0.5086, -0.4310, 0.0173], dtype=np.float64)

    yield world.hold_action()

    if bool(use_current_trunk):
        trunk_bent = world.trunk_qpos().copy()
        theta_bent = float(world.chest_pose().get("theta_z_deg", 90.0))
        ctx.log(
            f"[fk_audit] 使用 **当前** trunk θz={theta_bent:.1f}° "
            f"q={np.round(trunk_bent, 4).tolist()}"
        )
    else:
        if abs(float(pitch_deg)) > 1e-3:
            yield from move_in_robot_coord(ctx, pitch=float(pitch_deg))
        trunk_bent = world.trunk_qpos().copy()
        theta_bent = float(world.chest_pose().get("theta_z_deg", 90.0))
        ctx.log(
            f"[fk_audit] pitch={float(pitch_deg):+.1f}° trunk θz={theta_bent:.1f}° "
            f"q={np.round(trunk_bent, 4).tolist()}"
        )

    yield from reset_body(ctx)
    trunk_up = world.trunk_qpos().copy()
    theta_up = float(world.chest_pose().get("theta_z_deg", 90.0))

    fk_up = _fk_grip_chest_max_along_path(world, arm, path, trunk_up)
    fk_bent = _fk_grip_chest_max_along_path(world, arm, path, trunk_bent)
    fk_nav = _fk_grip_chest_max_along_path(world, arm, path, trunk_nav_bent)

    wp_up = _fk_probe_one(world, arm, q_goal, trunk_up)
    wp_bent = _fk_probe_one(world, arm, q_goal, trunk_bent)
    wp_nav = _fk_probe_one(world, arm, q_goal, trunk_nav_bent)

    def _cm(d: Dict[str, float], k: str) -> float:
        return round(float(d.get(k, 0.0)) * 100.0, 2)

    ctx.log(
        f"[fk_audit] 直立 θz={theta_up:.1f}° | 折线峰值 forward={_cm(fk_up,'grip_chest_fwd_m')}cm "
        f"belly_world={_cm(fk_up,'grip_belly_world_m')}cm link4_norm={_cm(fk_up,'grip_chest_frame_norm_m')}cm"
    )
    ctx.log(
        f"[fk_audit] 俯身 θz={theta_bent:.1f}° | 折线峰值 forward={_cm(fk_bent,'grip_chest_fwd_m')}cm "
        f"belly_world={_cm(fk_bent,'grip_belly_world_m')}cm link4_norm={_cm(fk_bent,'grip_chest_frame_norm_m')}cm "
        f"trunk_set_err={wp_bent.get('trunk_set_err_inf', 0):.2e}"
    )
    ctx.log(
        f"[fk_audit] 导航俯身(典型) | 折线峰值 forward={_cm(fk_nav,'grip_chest_fwd_m')}cm "
        f"belly_world={_cm(fk_nav,'grip_belly_world_m')}cm link4_norm={_cm(fk_nav,'grip_chest_frame_norm_m')}cm"
    )
    ctx.log(
        f"[fk_audit] 终点胸前 wp | 直立 fwd={_cm(wp_up,'grip_chest_fwd_m')} "
        f"俯身 fwd={_cm(wp_bent,'grip_chest_fwd_m')} "
        f"导航 fwd={_cm(wp_nav,'grip_chest_fwd_m')} cm"
    )
    d_fwd = _cm(fk_bent, "grip_chest_fwd_m") - _cm(fk_up, "grip_chest_fwd_m")
    d_belly = _cm(fk_bent, "grip_belly_world_m") - _cm(fk_up, "grip_belly_world_m")
    d_l4 = _cm(fk_bent, "grip_chest_frame_norm_m") - _cm(fk_up, "grip_chest_frame_norm_m")
    ctx.log(
        f"[fk_audit] Δ(俯身-直立) 折线峰值: forward={d_fwd:+.2f}cm belly_world={d_belly:+.2f}cm "
        f"link4_norm={d_l4:+.2f}cm"
    )
    if abs(d_fwd) < 0.5 and abs(d_l4) < 0.5:
        ctx.log(
            "[fk_audit] 说明: link4系/forward 在 FK 下随 trunk 不变 → 臂挂在 link4 子树；"
            "若俯身时肉眼更远，请看 belly_world 或动态播放。"
        )

    payload = {
        "arm": arm,
        "upright": {"theta_z_deg": theta_up, "trunk_q": trunk_up.round(4).tolist(), "fk": fk_up, "wp": wp_up},
        "bent": {"theta_z_deg": theta_bent, "trunk_q": trunk_bent.round(4).tolist(), "fk": fk_bent, "wp": wp_bent},
        "nav_bent": {"trunk_q": trunk_nav_bent.round(4).tolist(), "fk": fk_nav, "wp": wp_nav},
        "delta_cm": {"forward": d_fwd, "belly_world": d_belly, "link4_norm": d_l4},
    }
    ctx.set_result({"ok": True, **payload})
    yield world.hold_action()


@register_skill(
    "diag_tuck_chest_invariance",
    description="直立 vs 俯身：同关节折线，夹爪相对胸廓最大距离是否不变",
)
def diag_tuck_chest_invariance(
    ctx,
    arm: str = "left",
    pitch_deg: float = -20.0,
):
    """保持 _TUCK_TRAJECTORY 不变：FK + 动态播放各测一遍直立/俯身峰值。"""
    from behavior_interface.skills.arm_reset import arm_reset
    from behavior_interface.skills.move_to import move_in_robot_coord
    from behavior_interface.skills.reset_body import reset_body

    world = ctx.world
    arm = (arm or "left").strip().lower()
    path = _TUCK_TRAJECTORY.get(arm) or _TUCK_TRAJECTORY["right"]
    pitch_deg = float(pitch_deg)

    ctx.log(f"[diag_chest_inv] arm={arm} 折线 wp={len(path)} pitch={pitch_deg:.1f}°")

    yield from reset_body(ctx)
    trunk_up = world.trunk_qpos().copy()
    theta_up = float(world.chest_pose().get("theta_z_deg", 90.0))
    fk_up = _fk_grip_chest_max_along_path(world, arm, path, trunk_up)
    ctx.log(
        f"[diag_chest_inv] 直立 FK max_chest_frame={fk_up['grip_chest_frame_norm_m']*100:.2f}cm "
        f"max_chest_fwd={fk_up['grip_chest_fwd_m']*100:.2f}cm "
        f"max_shoulder_frame={fk_up['grip_shoulder_frame_norm_m']*100:.2f}cm θz={theta_up:.1f}°"
    )

    yield from arm_reset(ctx, arm=arm, mode="hang", open_gripper=True)
    fl_up: List[Dict] = []
    yield from _yield_play_joint_path(
        world, arm, path, 1.0, ctx, tag="_up", frame_log=fl_up,
    )
    dyn_up = _summarize_grip_chest_log(fl_up)
    ctx.log(
        f"[diag_chest_inv] 直立 动态 max_chest_frame="
        f"{dyn_up.get('max_grip_chest_frame_norm_m', 0)*100:.2f}cm "
        f"max_chest_fwd={dyn_up.get('max_grip_chest_fwd_m', 0)*100:.2f}cm"
    )

    yield from move_in_robot_coord(ctx, pitch=pitch_deg)
    trunk_bent = world.trunk_qpos().copy()
    theta_bent = float(world.chest_pose().get("theta_z_deg", 90.0))
    fk_bent = _fk_grip_chest_max_along_path(world, arm, path, trunk_bent)
    ctx.log(
        f"[diag_chest_inv] 俯身 FK max_chest_frame={fk_bent['grip_chest_frame_norm_m']*100:.2f}cm "
        f"max_chest_fwd={fk_bent['grip_chest_fwd_m']*100:.2f}cm "
        f"max_shoulder_frame={fk_bent['grip_shoulder_frame_norm_m']*100:.2f}cm θz={theta_bent:.1f}°"
    )

    yield from arm_reset(ctx, arm=arm, mode="hang", open_gripper=True)
    fl_bent: List[Dict] = []
    yield from _yield_play_joint_path(
        world, arm, path, 1.0, ctx, tag="_bent", frame_log=fl_bent,
    )
    dyn_bent = _summarize_grip_chest_log(fl_bent)
    ctx.log(
        f"[diag_chest_inv] 俯身 动态 max_chest_frame="
        f"{dyn_bent.get('max_grip_chest_frame_norm_m', 0)*100:.2f}cm "
        f"max_chest_fwd={dyn_bent.get('max_grip_chest_fwd_m', 0)*100:.2f}cm"
    )

    def _delta_cm(a: float, b: float) -> float:
        return round((float(b) - float(a)) * 100.0, 2)

    fk_delta = {
        "chest_frame_cm": _delta_cm(
            fk_up["grip_chest_frame_norm_m"], fk_bent["grip_chest_frame_norm_m"],
        ),
        "chest_fwd_cm": _delta_cm(fk_up["grip_chest_fwd_m"], fk_bent["grip_chest_fwd_m"]),
        "shoulder_frame_cm": _delta_cm(
            fk_up["grip_shoulder_frame_norm_m"], fk_bent["grip_shoulder_frame_norm_m"],
        ),
    }
    dyn_delta = {
        "chest_frame_cm": _delta_cm(
            dyn_up.get("max_grip_chest_frame_norm_m", 0.0),
            dyn_bent.get("max_grip_chest_frame_norm_m", 0.0),
        ),
        "chest_fwd_cm": _delta_cm(
            dyn_up.get("max_grip_chest_fwd_m", 0.0),
            dyn_bent.get("max_grip_chest_fwd_m", 0.0),
        ),
    }
    # 胸廓系 0.5cm 内视为数值噪声
    chest_invariant_fk = abs(fk_delta["chest_frame_cm"]) < 0.5
    chest_invariant_dyn = abs(dyn_delta["chest_frame_cm"]) < 0.5
    fwd_invariant_fk = abs(fk_delta["chest_fwd_cm"]) < 0.5
    fwd_invariant_dyn = abs(dyn_delta["chest_fwd_cm"]) < 0.5

    if chest_invariant_fk:
        verdict = (
            "FK: 同关节折线在 torso_link4 系下 **胸廓距离不变**（折线运动学上相对胸廓）"
        )
    else:
        verdict = "FK: 同关节折线在胸廓系下 **距离随俯仰变化**（非胸廓相对轨迹）"

    ctx.log(
        f"[diag_chest_inv] 结论: {verdict} | "
        f"FK Δ胸廓系={fk_delta['chest_frame_cm']:+.2f}cm "
        f"Δforward={fk_delta['chest_fwd_cm']:+.2f}cm Δ肩系={fk_delta['shoulder_frame_cm']:+.2f}cm | "
        f"动态 Δ胸廓系={dyn_delta['chest_frame_cm']:+.2f}cm "
        f"Δforward={dyn_delta['chest_fwd_cm']:+.2f}cm"
    )
    if chest_invariant_fk and not fwd_invariant_dyn:
        ctx.log(
            "[diag_chest_inv] 动态 forward 峰值俯仰差大 → 问题在 **播放跟踪/碰撞**，"
            "不是折线关节角在胸廓系下定义错了；FK 俯身仍 ≤15cm。"
        )
    if not chest_invariant_fk:
        ctx.log(
            "[diag_chest_inv] 需胸廓系路点：按 trunk θ 用 IK 在 torso_link4 系标定，"
            "或 trunk+arm 联合规划。"
        )

    payload = {
        "arm": arm,
        "pitch_deg": pitch_deg,
        "upright": {
            "theta_z_deg": theta_up,
            "trunk_q": np.round(trunk_up, 4).tolist(),
            "fk_max_cm": {k: round(v * 100, 2) for k, v in fk_up.items()},
            "dynamic_max_cm": {k.replace("_m", "_cm"): round(v * 100, 2)
                               for k, v in dyn_up.items()},
        },
        "bent": {
            "theta_z_deg": theta_bent,
            "trunk_q": np.round(trunk_bent, 4).tolist(),
            "fk_max_cm": {k: round(v * 100, 2) for k, v in fk_bent.items()},
            "dynamic_max_cm": {k.replace("_m", "_cm"): round(v * 100, 2)
                               for k, v in dyn_bent.items()},
        },
        "delta_cm": {"fk": fk_delta, "dynamic": dyn_delta},
        "chest_frame_invariant_fk": bool(chest_invariant_fk),
        "chest_frame_invariant_dynamic": bool(chest_invariant_dyn),
        "chest_fwd_invariant_fk": bool(fwd_invariant_fk),
        "chest_fwd_invariant_dynamic": bool(fwd_invariant_dyn),
        "verdict": verdict,
    }
    ctx.set_result({"ok": True, **payload})
    yield world.hold_action()


@register_skill(
    "diag_tuck_rigid_replay",
    description="诊断 tuck replay：刚体系距离 + 关节跟踪（当前 trunk 姿态）",
)
def diag_tuck_rigid_replay(
    ctx,
    arm: str = "left",
    play: bool = True,
):
    """在当前 trunk 下：ghost 沿路径 FK 峰值 vs 真实播放峰值。"""
    world = ctx.world
    arm = (arm or "left").strip().lower()
    path = _TUCK_TRAJECTORY.get(arm) or _TUCK_TRAJECTORY["right"]
    yield world.hold_action()

    kin = _kinematic_path_max_in_frames(world, arm, path)
    chest = world.chest_pose()
    ctx.log(
        f"[diag_tuck_replay] arm={arm} trunk_θz={chest.get('theta_z_deg', 90):.1f}° "
        f"ghost_lift={kin['eef_lift_norm_m']*100:.1f}cm "
        f"ghost_link4={kin['eef_link4_norm_m']*100:.1f}cm "
        f"ghost_chest_fwd={kin['chest_fwd_m']*100:.1f}cm"
    )

    dyn: Dict[str, float] = {}
    frame_log: List[Dict] = []
    if play:
        yield from _yield_play_joint_path(
            world, arm, path, 1.0, ctx, tag="_replay", frame_log=frame_log,
        )
        dyn = _summarize_grip_chest_log(frame_log)
        ctx.log(
            f"[diag_tuck_replay] dynamic max_link4={dyn.get('max_grip_chest_frame_norm_m', 0)*100:.1f}cm "
            f"max_chest_fwd={dyn.get('max_grip_chest_fwd_m', 0)*100:.1f}cm "
            f"max_belly={dyn.get('max_grip_belly_world_m', 0)*100:.1f}cm "
            f"max_err_inf={dyn.get('max_err_inf', 0):.3f}rad "
            f"trunk_Δ={dyn.get('trunk_delta_inf', 0):.4f}rad"
        )

    payload = {
        "arm": arm,
        "trunk_theta_z_deg": float(chest.get("theta_z_deg", 90.0)),
        "kinematic_max_cm": {k: round(v * 100.0, 2) for k, v in kin.items()},
        "dynamic_max_cm": {k.replace("_m", "_cm"): round(v * 100.0, 2)
                           for k, v in dyn.items() if k.endswith("_m")},
        "dynamic": dyn,
        "frames_n": len(frame_log),
    }
    ctx.set_result({"ok": True, **payload})
    yield world.hold_action()


@register_skill(
    "diag_tuck_trajectory",
    description="验证/贪心重规划 tuck 轨迹（胸口 forward≤15cm）",
)
def diag_tuck_trajectory(
    ctx,
    arm: str = "right",
    out_json: str = "/tmp/tuck_trajectory.json",
    replan: bool = True,
    raise_eef_m: float = 0.0,
    pull_in: bool = True,
    max_forward_cm: float = 0.0,
):
    from behavior_interface.skills.arm_reset import _HANG_ARM_QPOS

    world = ctx.world
    arm = (arm or "right").strip().lower()
    yield world.hold_action()

    max_fwd_m = (
        float(max_forward_cm) / 100.0
        if float(max_forward_cm) > 1e-6
        else float(TUCK_CHEST_FORWARD_MAX_M)
    )

    if bool(replan):
        q_goal = CHEST_TUCK_ARM.get(arm, CHEST_TUCK_ARM["right"]).copy()
        if float(raise_eef_m) > 1e-4:
            z0 = _eef_world_z_m(world, arm, q_goal)
            q_goal = tune_chest_goal_raise_eef(
                world, arm, q_goal, raise_m=float(raise_eef_m), max_forward_m=max_fwd_m,
            )
            z1 = _eef_world_z_m(world, arm, q_goal)
            if z0 is not None and z1 is not None:
                ctx.log(
                    f"[diag_tuck] 抬高 EEF Δz={(z1 - z0) * 100:.1f}cm "
                    f"(目标 +{float(raise_eef_m) * 100:.1f}cm) goal={np.round(q_goal, 3).tolist()}"
                )
        if bool(pull_in):
            q_seed = q_goal.copy()
            if arm == "left":
                q_seed = _mirror_tuck_q_for_left(
                    CHEST_TUCK_ARM.get("right", CHEST_TUCK_ARM["right"]),
                )
            fd0 = eef_chest_forward_dist_m(world, arm, q_seed)
            z0 = _eef_world_z_m(world, arm, q_seed)
            q_found = search_feasible_chest_goal(
                world, arm, q_seed, max_forward_m=max_fwd_m, max_iters=360,
            )
            if q_found is not None:
                q_goal = q_found
            else:
                q_goal = tune_chest_goal_pull_in(
                    world, arm, q_goal, max_forward_m=max_fwd_m,
                )
            fd1 = eef_chest_forward_dist_m(world, arm, q_goal)
            z1 = _eef_world_z_m(world, arm, q_goal)
            if fd0 is not None and fd1 is not None:
                dz = (z1 - z0) * 100.0 if z0 is not None and z1 is not None else 0.0
                ctx.log(
                    f"[diag_tuck] 收臂 goal_fwd {fd0 * 100:.1f}→{fd1 * 100:.1f}cm "
                    f"Δz={dz:+.1f}cm goal={np.round(q_goal, 3).tolist()}"
                )
        ctx.log(
            f"[diag_tuck] 贪心规划 hang→胸前 arm={arm} max_fwd={max_fwd_m * 100:.0f}cm …"
        )
        holds = {"n": 0}

        def _yield_hold():
            holds["n"] += 1

        raw = _greedy_plan_tuck_path(
            world, arm, _HANG_ARM_QPOS.copy(),
            q_goal,
            max_forward_m=max_fwd_m,
            max_iters=320,
            yield_fn=lambda: _yield_hold(),
        )
        for _ in range(holds["n"]):
            yield world.hold_action()
        path = _downsample_path(raw, max_waypoints=16)
        ctx.log(f"[diag_tuck] 规划 {len(raw)} 细点 → 下采样 {len(path)} 路点")
    else:
        path = _TUCK_TRAJECTORY.get(arm, _TUCK_TRAJECTORY["right"])
    yield world.hold_action()

    rep = verify_tuck_path(world, arm, path, max_forward_m=max_fwd_m)
    rounded = [[round(float(v), 4) for v in q] for q in path]
    per_wp = []
    for i, q in enumerate(path):
        fd = eef_chest_forward_dist_m(world, arm, q)
        zz = _eef_world_z_m(world, arm, q)
        per_wp.append({
            "i": i,
            "chest_fwd_cm": round(fd * 100, 2) if fd is not None else None,
            "eef_z_m": round(zz, 4) if zz is not None else None,
        })
        ctx.log(
            f"  wp{i}: chest_fwd={fd * 100:.1f}cm eef_z={zz:.3f}m"
            if fd is not None and zz is not None
            else f"  wp{i}: chest_fwd=?"
        )

    ctx.log(
        f"[diag_tuck] arm={arm} segments={len(path)-1} "
        f"max_fwd={rep['max_forward_cm']:.1f}cm ok={bool(rep['ok'])}"
    )
    payload = {
        "arm": arm,
        "max_forward_cm": rep["max_forward_cm"],
        "ok": bool(rep["ok"]),
        "waypoints": rounded,
        "per_wp": per_wp,
        "replanned": bool(replan),
    }
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        ctx.log(f"[diag_tuck] 写文件失败: {e}")

    ctx.set_result({"ok": True, **payload, "out_json": out_json})
    yield world.hold_action()
