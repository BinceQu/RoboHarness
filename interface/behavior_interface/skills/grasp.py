"""
grasp skills：

get_grasp_position(object_name, arm="right", k="auto")
    输入物体名（Scene Graph 里的统一短名，或 BDDL key，或 scene 内部 ID 也兼容），
    自动算 3-5 个最具代表性的 grasp 候选并返回。
    每个候选包含 dict：
        {
          "id": int,
          "pos":      [x, y, z],        # 夹爪 TCP 的世界坐标
          "approach": [ax, ay, az],     # 夹爪 approach 方向（朝物体的单位向量）
          "arm":      "right" / "left",
          "reachable": bool,
          "score":    float (0..1),
          "label":    str (top / side+x / side-x / ...)
        }

execute_grasp(grasp_id=0)
    取 get_grasp_position 的第 grasp_id 个候选执行。
    - 若 reachable：把 eef 通过 IK 移到 grasp pos -> 闭夹爪 -> 提起 0.1m
                    -> 观察物体 z 变化 > 0.04m -> 返回 grasped=True
    - 若不 reachable：在 free_region 里搜一个 base 位置使物体落入"肩部 0.55m 范围内"
                      并返回该 base pose，方便后续 move_to(x, y) 再执行

设计要点：
- "代表性"：在物体 AABB 表面密采样大量 candidate，按 cosine 距离做 farthest-point
  sampling 选 k 个，覆盖各方向（顶/前/后/左/右）。
- "reachable" 启发式：grasp.pos 距离机器人当前 shoulder 在 0.20 ~ 0.75m 之间
  并且 z 在 0.05 ~ 1.7m 区间。
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from behavior_interface.skills import register_skill


def _challenge_action_only_enabled() -> bool:
    return str(
        os.environ.get("BEHAVIOR_CHALLENGE_MODE", "")
    ).lower().strip() in {"train", "public_test", "hidden_test"}


# ─────────────────────────────────────────────────────────────────────────────
# 物体反查 + AABB 工具
# ─────────────────────────────────────────────────────────────────────────────

def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _resolve_object_handle(world, name: str):
    """Scene Graph 里的统一短名（如 "popcorn_bag"）、BDDL key（"popcorn__bag.n.01_1"）、
    scene 原始名（"popcorn_bag_kbhqxq_0"）三种都尽量接受。
    返回 OmniGibson StatefulObject，找不到返回 None。
    """
    if world.dry_run:
        return None

    # 1) 先用 Scene Graph 查统一短名 → scene_name
    sg = getattr(world, "current_scene_graph", None)
    scene_name_hint: Optional[str] = None
    if sg is not None:
        for o in sg.objects:
            if name in (o.name, o.bddl_key, o.scene_name):
                scene_name_hint = o.scene_name
                break
        # free_region.obstacles 也可能含有
        if scene_name_hint is None:
            for ob in sg.free_region.obstacles:
                if ob.get("name") == name:
                    # obstacles 没保留 scene_name，但 SG.objects 里同 name 的就有
                    for o in sg.objects:
                        if o.name == name:
                            scene_name_hint = o.scene_name
                            break

    # 2) WorldAPI 的反查（BDDL key 优先）
    candidate_names = []
    if scene_name_hint:
        candidate_names.append(scene_name_hint)
    candidate_names.append(name)

    for cand in candidate_names:
        obj = world._resolve_object(cand)  # noqa: SLF001
        if obj is not None:
            return obj
    return None


def _aabb_of(obj) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        lo, hi = obj.aabb
        return _to_np(lo).reshape(-1), _to_np(hi).reshape(-1)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Grasp 候选生成（AABB 周围密采样 + farthest-point 选 k 个）
# ─────────────────────────────────────────────────────────────────────────────

def _sample_grasp_candidates(aabb_min: np.ndarray, aabb_max: np.ndarray,
                              n_top: int = 5, n_side: int = 0
                              ) -> List[Dict[str, Any]]:
    """对物体 AABB 顶面采样若干 top-down grasp 候选。

    设计决策：基于实验数据，**全部用 top-down approach (0,0,-1)** 最稳定可靠：
    1. 与 OmniGibson 自带 `get_grasp_poses_for_object_sticky` 一致；
    2. R1Pro 初始 eef 朝向就是向下，IK 无需大角度调整；
    3. sticky grasping 要求 finger 与物体物理接触，top-down 能让两根 finger
       自然从两侧夹住物体顶部；
    4. side approach 需要旋转 gripper opening 方向（grasp_ori 6-DOF IK），
       否则 gripper opening 方向不对，常失败。我们目前的 IK 只控位置不锁 ori。

    采样 5 个 top-down 点：物体顶面中心 + 4 个偏移（沿 ±x, ±y）。
    n_side 参数保留兼容，但默认 0。
    """
    center = (aabb_min + aabb_max) / 2.0
    extent = aabb_max - aabb_min
    half = extent / 2.0
    candidates: List[Dict[str, Any]] = []

    # 顶部 grasp 高度：AABB top 朝下 0-2cm（让 fingertip 略入物体）
    top_z = aabb_max[2]
    top_grasp_z = top_z - min(0.02, 0.1 * extent[2])

    # 候选 1：物体顶部中心
    candidates.append({
        "pos":      [float(center[0]), float(center[1]), float(top_grasp_z)],
        "approach": [0.0, 0.0, -1.0],
        "label":    "top@center",
    })

    # 候选 2-5：沿 ±x, ±y 偏移 30% half（避免出 AABB）
    off_x = 0.3 * half[0]
    off_y = 0.3 * half[1]
    side_offsets = [
        (+off_x, 0.0,    "top@+x"),
        (-off_x, 0.0,    "top@-x"),
        (0.0,    +off_y, "top@+y"),
        (0.0,    -off_y, "top@-y"),
    ]
    for dx, dy, label in side_offsets:
        candidates.append({
            "pos":      [float(center[0] + dx), float(center[1] + dy), float(top_grasp_z)],
            "approach": [0.0, 0.0, -1.0],
            "label":    label,
        })

    return candidates


def _farthest_point_sampling(candidates: List[Dict[str, Any]], k: int
                              ) -> List[int]:
    """对 candidates 做 farthest-point 采样，挑出 k 个"最分散"的索引。
    距离度量 = 0.5 * 欧氏距离(pos) + 0.5 * (1 - cosine(approach))
    """
    if k >= len(candidates):
        return list(range(len(candidates)))
    if k <= 0:
        return []

    poss = np.array([c["pos"] for c in candidates])
    apps = np.array([c["approach"] for c in candidates])
    apps = apps / (np.linalg.norm(apps, axis=1, keepdims=True) + 1e-9)

    def dist(i: int, j: int) -> float:
        d_pos = float(np.linalg.norm(poss[i] - poss[j]))
        cos_ij = float(np.clip(np.dot(apps[i], apps[j]), -1, 1))
        d_ang = 1.0 - cos_ij
        return 0.5 * d_pos + 0.5 * d_ang

    chosen = [0]  # 从第一个开始
    while len(chosen) < k:
        # 对每个未选的，算它到 chosen 集合的最小距离
        best_idx, best_d = -1, -1.0
        for i in range(len(candidates)):
            if i in chosen:
                continue
            min_d = min(dist(i, c) for c in chosen)
            if min_d > best_d:
                best_d = min_d
                best_idx = i
        if best_idx < 0:
            break
        chosen.append(best_idx)
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Reachable：基于 URDF 链长的精确肩膀-fingertip 球范围
# ─────────────────────────────────────────────────────────────────────────────
#
# R1Pro 单臂从 shoulder (arm_link1) 到 fingertip 的关节链总长（URDF 各段累计）：
#   shoulder → link7 (wrist)：sum |seg| ≈ 0.79 m (各段直线长之和)
#                            实际 wrist 直线最大距离 ≈ 0.59 m (考虑各段方向)
#   wrist → gripper_link → fingertip：≈ 0.21 m
#
# 完全伸直时 shoulder → fingertip 最远距离 ≈ 0.80 m
# 但 7-DOF joint 都有 limits，实际 IK 可达约 85% × 链长。
# 综合考虑 + 物体表面 finger 需深入 ~5cm，取：
#   max_reach (shoulder → grasp point) = 0.78 m
#   min_reach (避免自碰撞 + 肘部死区)   = 0.18 m
_ARM_MAX_REACH = 1.05   # m (R1Pro arm 7-DOF + trunk 一起求解，实测可达 ~1.0m)
_ARM_MIN_REACH = 0.18   # m


def _auto_pick_arm(world, grasp_pos: np.ndarray) -> str:
    """根据 grasp 点的 base frame y 自动选 left（+y）/right（-y）臂。"""
    try:
        rp = world.robot_pose()
        # 转 base frame
        c = math.cos(-rp.yaw); s = math.sin(-rp.yaw)
        dx = grasp_pos[0] - rp.pos[0]; dy = grasp_pos[1] - rp.pos[1]
        y_base = s * dx + c * dy
        return "left" if y_base > 0 else "right"
    except Exception:
        return "right"


def _is_reachable(world, grasp: Dict[str, Any], arm: str) -> Tuple[bool, str]:
    """精确 reach check：grasp 点到对应 arm shoulder 的 3D 距离 ∈ [min, max]。

    `shoulder` 用 world_api.shoulder_pose(arm) 实时读 (arm_link1 的 world 位置)，
    所以 trunk 升降/俯仰后 shoulder 位置也跟着变化 — reach 检测自动适应胸口姿态。

    距离 ≤ _ARM_MAX_REACH (0.78m, URDF shoulder→fingertip 完全伸直长度的 ~98%)
    且   ≥ _ARM_MIN_REACH (0.18m, 避免肘部死区 / 自碰撞) 时判定 reachable。
    """
    px, py, pz = grasp["pos"]
    try:
        sh = world.shoulder_pose(arm=arm)
        sx, sy, sz = float(sh["x"]), float(sh["y"]), float(sh["z"])
    except Exception:
        return False, f"无法读取 {arm} shoulder 位置"
    d3 = math.sqrt((px - sx)**2 + (py - sy)**2 + (pz - sz)**2)
    if d3 < _ARM_MIN_REACH:
        return False, f"too close shoulder ({d3:.2f}m < {_ARM_MIN_REACH}m)"
    if d3 > _ARM_MAX_REACH:
        return False, f"too far shoulder ({d3:.2f}m > {_ARM_MAX_REACH}m)"
    return True, f"ok (d={d3:.2f}m, shoulder={arm})"


def _pick_chest_pose_for_grasp(grasp_z: float) -> Tuple[float, float]:
    """根据目标 grasp 高度（世界系 z）三档选 chest_z（升降）+ theta_z_deg（俯仰）。

    R1Pro chest world z 物理范围约 [0.66, 1.20] m。
    **关键：shoulder 在 chest 上方约 0.30m**（URDF），所以 chest_z = grasp_z - 0.30
    能让 shoulder 几乎和 grasp 同高，臂水平伸出最有效。

    三档：
      地面物体 (z < 0.60)  → chest_z=0.66 (最低), theta_z=120°  弯腰俯视
      桌面物体 (z ∈ [0.60, 1.50]) → chest_z=clip(z - 0.30, [0.66, 1.20]), theta_z=90°
      高架物体 (z > 1.50)  → chest_z=1.20, theta_z=70°    抬头仰视
    """
    if grasp_z < 0.60:
        return 0.90, 120.0
    if grasp_z > 1.50:
        return 1.20, 70.0
    # 让 shoulder z 略高于 grasp z 约 15cm（chest_z + 0.30 = shoulder_z ≈ grasp_z + 0.15）
    # 这样夹爪从上方接近，IK 解最稳定
    chest_z = max(min(grasp_z - 0.15, 1.20), 0.90)
    return chest_z, 90.0


def _suggest_base_pose(world, grasp: Dict[str, Any], arm: str
                       ) -> Optional[Dict[str, float]]:
    """在 free_region 里搜一个 chest 5D pose，**保证** shoulder→grasp 3D 距离在工作空间内有余量。

    搜索策略：
    1. 根据 grasp 高度选 chest_z + theta_z_deg（三档）。
    2. 从 grasp_xy 周围密扫候选 base 位置（多圈半径 × 24 个方向）。
    3. 对每个候选 base，估算该位置时 shoulder 的世界坐标（用 chest ≈ base_xy + chest_z
       + shoulder_offset_local），计算 shoulder→grasp 3D 距离。
    4. 仅当距离落在 [_ARM_MIN_REACH*1.3, _ARM_MAX_REACH*0.85] 时（有 15% 余量）才接受。
       这样保证到达建议位置后 IK 必定有解。
    5. 同时检查 base 点在 free_region 内（含机器人半径）。
    """
    sg = getattr(world, "current_scene_graph", None)
    if sg is None:
        return None
    from behavior_interface.scene_graph import is_point_free

    px, py, pz = float(grasp["pos"][0]), float(grasp["pos"][1]), float(grasp["pos"][2])
    chest_z, theta_z_deg = _pick_chest_pose_for_grasp(pz)

    # shoulder 相对 chest 的局部偏移（来自 R1Pro URDF：
    #   torso_link4 → right_arm_base_link (0, -0.097, +0.303)
    #   right_arm_base_link → right_arm_link1 (0, -0.0735, 0)
    #   合计：y = -0.171, z = +0.303，对 left 对称取 +0.171）
    shoulder_side_y = -0.171 if arm == "right" else +0.171  # chest 局部 y 偏移
    shoulder_z_offset = +0.303  # shoulder 在 chest 上方约 30cm
    sh_z = chest_z + shoulder_z_offset

    # 目标 shoulder→grasp 3D 距离范围（有余量）
    # 用 15% 余量避免边界抖动：实际可达 [0.18, 0.78]，保守用 [0.24, 0.66]
    d_min_safe = _ARM_MIN_REACH * 1.3   # ≈ 0.23m
    d_max_safe = _ARM_MAX_REACH * 0.85  # ≈ 0.66m

    # 扫描：多圈半径，每圈 24 方向
    best = None
    best_d = None
    for target_dist in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80):
        for deg in range(0, 360, 15):
            rad = math.radians(deg)
            bx = px + target_dist * math.cos(rad)
            by = py + target_dist * math.sin(rad)
            if not is_point_free(sg.free_region, bx, by, extra_inflate=0.0)[0]:
                continue
            # 估算到达该 base 时的 shoulder 世界位置
            # base 朝物体方向 yaw，shoulder 偏移需旋转到世界系
            base_yaw = math.atan2(py - by, px - bx)  # 朝物体的 yaw
            # shoulder local y 转世界: rotate (0, shoulder_side_y) by base_yaw
            sh_x = bx + shoulder_side_y * (-math.sin(base_yaw))
            sh_y = by + shoulder_side_y * math.cos(base_yaw)
            d3 = math.sqrt((px - sh_x)**2 + (py - sh_y)**2 + (pz - sh_z)**2)
            if d3 < d_min_safe or d3 > d_max_safe:
                continue
            # 选最靠近最优距离 0.50m 的候选
            score = abs(d3 - 0.50)
            if best is None or score < best_d:
                best = (bx, by, base_yaw, d3)
                best_d = score

    if best is None:
        return None
    bx, by, base_yaw, d3 = best
    return {
        "x": round(bx, 3), "y": round(by, 3),
        "z": round(chest_z, 3),
        "theta_x_deg": round(math.degrees(base_yaw), 1),
        "theta_z_deg": round(theta_z_deg, 1),
        "yaw_deg": round(math.degrees(base_yaw), 1),
        "dist_to_object": round(math.hypot(px - bx, py - by), 2),
        "shoulder_to_grasp_3d": round(d3, 3),
        "grasp_height_class": (
            "ground" if pz < 0.60
            else ("high" if pz > 1.30 else "table")
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Skill: get_grasp_position
# ─────────────────────────────────────────────────────────────────────────────

@register_skill(
    "get_grasp_position",
    description=(
        "对指定物体（用 Scene Graph 短名/BDDL key/scene 名都行）计算 3-5 个最具代表性的 grasp 候选，"
        "并存到 server 的 last_skill_results，供 execute_grasp 调用。"
        "reachable 优先排序。"
    ),
)
def get_grasp_position(
    ctx,
    object_name: str,
    arm: str = "right",
    k: int = 0,              # 0 表示自适应（reachable 不重复角度优先 3-5 个）
    n_top: int = 8,
    n_side: int = 16,
):
    """yield None 一次后立刻 return。所有计算同步完成。"""
    world = ctx.world
    obj = _resolve_object_handle(world, object_name)
    if obj is None:
        msg = f"找不到物体 '{object_name}'（既不在 Scene Graph 也不在 scene_registry）"
        ctx.log(msg)
        ctx.set_result({"ok": False, "error": msg, "candidates": []})
        return

    aabb = _aabb_of(obj)
    if aabb is None:
        msg = f"物体 '{object_name}' 没有有效 AABB"
        ctx.log(msg)
        ctx.set_result({"ok": False, "error": msg, "candidates": []})
        return

    lo, hi = aabb
    obj_center_now = (lo + hi) / 2.0
    # 实时性诊断：与上次 get_grasp_position 比较物体是否移动 + base 距离
    last_call = ctx.get_last_result("get_grasp_position")
    moved_info = ""
    if last_call is not None and last_call.get("object", {}).get("center"):
        prev_center = np.asarray(last_call["object"]["center"], dtype=np.float64)
        delta = float(np.linalg.norm(obj_center_now - prev_center))
        if delta > 0.02:
            moved_info = (f"  ⚠️ object moved {delta*100:.1f}cm since last call "
                          f"(prev_center={prev_center.tolist()})")
        else:
            moved_info = f"  (object stable, Δ={delta*100:.1f}cm)"
    try:
        rp = world.robot_pose()
        d_base_xy = math.hypot(obj_center_now[0]-float(rp.pos[0]),
                                obj_center_now[1]-float(rp.pos[1]))
        sh = world.shoulder_pose(arm=arm)
        d_shoulder = math.sqrt(
            (obj_center_now[0]-sh["x"])**2 +
            (obj_center_now[1]-sh["y"])**2 +
            (obj_center_now[2]-sh["z"])**2
        )
        moved_info += (f"\n  base→obj_xy={d_base_xy:.2f}m  "
                       f"{arm}_shoulder→obj_3d={d_shoulder:.2f}m  "
                       f"(arm max reach={_ARM_MAX_REACH}m)")
    except Exception:
        pass
    ctx.log(
        f"get_grasp_position obj={obj.name} arm={arm} "
        f"aabb_min=({lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f}) "
        f"aabb_max=({hi[0]:.2f},{hi[1]:.2f},{hi[2]:.2f})  "
        f"center=({obj_center_now[0]:.2f},{obj_center_now[1]:.2f},{obj_center_now[2]:.2f})"
        f"{moved_info}"
    )

    # 1) 密采样
    raw = _sample_grasp_candidates(lo, hi, n_top=n_top, n_side=n_side)

    # 2) reachable 评估 + 评分
    for c in raw:
        ok, why = _is_reachable(world, c, arm=arm)
        c["reachable"] = ok
        c["reach_reason"] = why
        c["arm"] = arm
        # 评分：reachable=1.0；越靠近物体中心高度（更稳）越好
        c["score"] = 1.0 if ok else 0.3
        # top-down 抓取得分加一点（一般最稳）
        if c["label"].startswith("top"):
            c["score"] += 0.1

    # 3) 排序：reachable 优先，再按 score
    raw.sort(key=lambda c: (not c["reachable"], -c["score"]))

    # 4) farthest-point 选 k 个
    # 自适应 k：reachable 数量 / 4，clip 到 [3, 5]
    n_reachable = sum(1 for c in raw if c["reachable"])
    if k <= 0:
        if n_reachable >= 5:
            k = 5
        elif n_reachable >= 3:
            k = n_reachable
        else:
            k = max(3, n_reachable)  # 不够 reachable 时也至少给 3 个候选（让用户看到）
        k = max(3, min(5, k))

    # 优先在 reachable 子集做 FPS，不够再从 unreachable 补
    reachable_pool = [c for c in raw if c["reachable"]]
    chosen_pool: List[Dict[str, Any]] = []
    if len(reachable_pool) >= k:
        idxs = _farthest_point_sampling(reachable_pool, k)
        chosen_pool = [reachable_pool[i] for i in idxs]
    else:
        chosen_pool = list(reachable_pool)
        # 从 unreachable 补充，仍做 FPS
        remain = k - len(chosen_pool)
        if remain > 0:
            unreach = [c for c in raw if not c["reachable"]]
            if unreach:
                # 在 unreach 内 FPS 选 remain 个
                if len(unreach) <= remain:
                    chosen_pool.extend(unreach)
                else:
                    idxs = _farthest_point_sampling(unreach, remain)
                    chosen_pool.extend([unreach[i] for i in idxs])

    # 编号
    for i, c in enumerate(chosen_pool):
        c["id"] = i

    # log 摘要
    ctx.log(
        f"get_grasp_position: 采样 {len(raw)} -> reachable {n_reachable} -> 选 {len(chosen_pool)}"
    )
    for c in chosen_pool:
        ctx.log(
            f"  [{c['id']}] {c['label']:14s} pos=({c['pos'][0]:.2f},{c['pos'][1]:.2f},{c['pos'][2]:.2f}) "
            f"approach=({c['approach'][0]:+.2f},{c['approach'][1]:+.2f},{c['approach'][2]:+.2f}) "
            f"reachable={c['reachable']} ({c['reach_reason']})"
        )

    payload = {
        "ok": True,
        "object": {
            "input": object_name,
            "resolved_name": getattr(obj, "name", object_name),
            "aabb_min": lo.tolist(),
            "aabb_max": hi.tolist(),
            "center": ((lo + hi) / 2.0).tolist(),
        },
        "arm": arm,
        "n_candidates_sampled": len(raw),
        "n_reachable": n_reachable,
        "candidates": chosen_pool,
    }
    # 0 个 reachable → 主动给一个 free_region 里的 base pose 建议
    # 选 reachability 最近 unreachable 的 grasp 作为 anchor 来推荐 base
    if n_reachable == 0 and chosen_pool:
        anchor = chosen_pool[0]  # FPS 后第一个候选
        bp = _suggest_base_pose(world, anchor, arm=arm)
        if bp is not None:
            payload["suggested_base_pose"] = bp
            ctx.log(
                f"  → 当前位置不可达，建议 move_to("
                f"x={bp['x']}, y={bp['y']}, z={bp['z']}, "
                f"theta_x_deg={bp['theta_x_deg']}, theta_z_deg={bp['theta_z_deg']}"
                f")  [grasp_height={bp.get('grasp_height_class')}]"
            )
        else:
            ctx.log("  → 当前位置不可达，但 free_region 没找到合适 base pose")
    ctx.set_result(payload)
    # 必须 yield 至少一次，否则 server 不认为是 generator
    yield world.empty_action()


# ─────────────────────────────────────────────────────────────────────────────
# IK 闭环工具：Jacobian DLS（Damped Least Squares）
#
# 不再依赖 OG IK controller —— 实测它在 R1Pro 长链上不稳定（每帧重发同一个
# absolute target 时 eef 在距离目标 0.1m 处不收敛，反而漂离）。
#
# 这里完全手写 IK：每帧从 OG 取 jacobian → 用 DLS 求 dq → 通过 JointController
# (use_delta_commands=True) 直接命令 arm joint delta。优点：
#   - 数值行为 100% 可控（damping λ、step 限速、early stop 都自己写）
#   - 只看 position 行（3D），不强制 ori → 自由度大，arm 7-DOF redundancy 很容易解
#   - PD 控制器只做 joint level 收敛，靠谱
#
# OG 提供的 jacobian:
#   robot.get_jacobian()  shape: (N_links - 1 + 1, 6, N_dof + 6)
#     - 第 0 维是 link 索引（floating base 多一行）
#     - 第 1 维 6 = (vx, vy, vz, wx, wy, wz)
#     - 第 2 维 N_dof+6：前 6 是 floating base joint，后面是真实 joint
#       => arm joint 列 = arm_control_idx + 6
# ─────────────────────────────────────────────────────────────────────────────

def _quat_inverse(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    return np.array([-qx, -qy, -qz, qw], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def _quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q
    u = np.array([qx, qy, qz], dtype=np.float64)
    s = float(qw)
    v = np.asarray(v, dtype=np.float64)
    return 2.0 * np.dot(u, v) * u + (s*s - np.dot(u, u)) * v + 2.0 * s * np.cross(u, v)


def _quat_to_axisangle(q: np.ndarray) -> np.ndarray:
    """xyzw quat → axis-angle (3D 向量，长度 = 旋转弧度，归一到 [-π, π])"""
    qx, qy, qz, qw = q
    qx, qy, qz, qw = float(qx), float(qy), float(qz), float(qw)
    if qw < 0:  # 双 cover：保证最短旋转
        qx, qy, qz, qw = -qx, -qy, -qz, -qw
    norm = math.sqrt(qx*qx + qy*qy + qz*qz)
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(norm, qw)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    axis = np.array([qx, qy, qz], dtype=np.float64) / norm
    return axis * angle


_CV_API = None  # 延迟 import


def _get_cv_api():
    global _CV_API
    if _CV_API is None:
        from omnigibson.utils.usd_utils import ControllableObjectViewAPI  # type: ignore
        _CV_API = ControllableObjectViewAPI
    return _CV_API


def _get_cv_view(robot):
    """拿到 robot 对应的 BatchControlViewAPIImpl 实例（内部有 _link_idx）。"""
    cv = _get_cv_api()
    pattern = cv._get_pattern_from_prim_path(robot.articulation_root_path)  # noqa: SLF001
    return cv._VIEWS_BY_PATTERN[pattern]  # noqa: SLF001


def _get_arm_dof_idx(world, arm: str) -> np.ndarray:
    """arm 7 个关节在 robot.joints 里的 dof index (int array)。"""
    idx = world.robot.arm_control_idx[arm]
    return _to_np(idx).astype(int)


def _arm_qpos(world, arm: str) -> np.ndarray:
    """读取当前 arm 7 个关节的绝对 qpos（rad）。absolute mode 下要发 q_current + dq。"""
    if world.dry_run:
        return np.zeros(7, dtype=np.float64)
    qpos = world.robot.get_joint_positions()
    idx = _get_arm_dof_idx(world, arm)
    out = np.zeros(len(idx), dtype=np.float64)
    for i, j in enumerate(idx):
        out[i] = float(qpos[int(j)])
    return out


def _read_jacobian_arm(world, arm: str) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (J_arm, arm_dof_idx)：
       J_arm shape (6, 7) —— eef link 对应 arm 7 关节的 jacobian
       arm_dof_idx shape (7,) —— 对应的 joint index

    完全照搬 OG 的取法（见 ControllableObject._add_task_frame_control_dict）：
      jac[-(n_links - body_idx), :, start_idx : start_idx + n_joints]
    其中：
      - body_idx = robot._articulation_view.get_body_index(eef_link_name)
      - start_idx = 6 if floating-base else 0
      - 倒数索引兼容 fixed/floating base 第 0 行可能是 root 的差异
    然后再用 arm_dof_idx 切出 7 个 arm joint 的列。
    """
    cv = _get_cv_api()
    jac_full = cv.get_jacobian(world.robot.articulation_root_path)
    if hasattr(jac_full, "detach"):
        jac_full = jac_full.detach().cpu().numpy()
    else:
        jac_full = np.asarray(jac_full)
    robot = world.robot
    eef_link_name = robot.eef_link_names[arm]
    body_idx = robot._articulation_view.get_body_index(eef_link_name)  # noqa: SLF001
    n_links = robot.n_links
    start_idx = 0 if robot.fixed_base else 6
    n_joints = robot.n_joints
    # 等价 OG: jac_full[-(n_links - body_idx), :, start_idx : start_idx + n_joints]
    jac_arm_full = jac_full[-(n_links - body_idx), :, start_idx:start_idx + n_joints]
    arm_dof_idx = _get_arm_dof_idx(world, arm)
    J_arm = jac_arm_full[:, arm_dof_idx]  # (6, 7)
    return J_arm.astype(np.float64), arm_dof_idx


def _dls_solve_dq(J: np.ndarray, dx: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """Damped Least Squares: dq = J^T (J J^T + λ²I)^-1 dx
    J shape (m, n), dx shape (m,)。lam 越大越稳定但越慢。
    """
    m = J.shape[0]
    JJt = J @ J.T + (lam ** 2) * np.eye(m)
    return J.T @ np.linalg.solve(JJt, dx)


def _dls_pinv(J: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """阻尼伪逆 J^+，shape (n, m)。"""
    m, n = J.shape
    if m <= n:
        jj = J @ J.T + (lam ** 2) * np.eye(m)
        return J.T @ np.linalg.solve(jj, np.eye(m))
    jtj = J.T @ J + (lam ** 2) * np.eye(n)
    return np.linalg.solve(jtj, J.T)


def _read_finger_qpos(world, arm: str) -> str:
    """读 left/right gripper 两个 finger joint 当前 qpos（米，[0, 0.05] 范围）。

    返回 "[q1, q2]" 字符串方便 log；读不到返回 "n/a"。
    """
    if world.dry_run:
        return "n/a (dry_run)"
    try:
        robot = world.robot
        qpos = robot.get_joint_positions()
        names = list(robot.joints.keys())
        j1 = f"{arm}_gripper_finger_joint1"
        j2 = f"{arm}_gripper_finger_joint2"
        i1 = names.index(j1); i2 = names.index(j2)
        v1 = float(qpos[i1]); v2 = float(qpos[i2])
        return f"[{v1:.4f}, {v2:.4f}]"
    except Exception as e:
        return f"n/a ({e})"


def _gripper_cmd_array(gripper_cmd) -> Optional[np.ndarray]:
    if gripper_cmd is None:
        return None
    try:
        arr = np.asarray(gripper_cmd, dtype=np.float64).reshape(-1)
    except Exception:
        arr = np.asarray([float(gripper_cmd)], dtype=np.float64)
    if arr.size <= 0:
        return None
    return arr


def _gripper_cmd_override(gripper_cmd) -> Optional[list[float]]:
    arr = _gripper_cmd_array(gripper_cmd)
    if arr is None:
        return None
    return [float(x) for x in arr.tolist()]


def _apply_gripper_cmd_to_joint_pos(
    robot,
    joint_names: list[str],
    finger_joint_idx_list: list[int],
    joint_pos,
    gripper_cmd,
) -> None:
    arr = _gripper_cmd_array(gripper_cmd)
    if arr is None or not finger_joint_idx_list:
        return
    if arr.size == 1:
        cmd = float(arr[0])
        for fi in finger_joint_idx_list:
            joint = robot.joints[joint_names[fi]]
            upper = float(joint.upper_limit)
            lower = float(joint.lower_limit)
            joint_pos[fi] = upper if cmd > 0 else lower
        return
    for local_i, fi in enumerate(finger_joint_idx_list):
        if local_i >= arr.size:
            break
        joint_pos[fi] = float(arr[local_i])


def _quat_to_mat(q_xyzw: np.ndarray) -> np.ndarray:
    """四元数 (x,y,z,w) → 3x3 旋转矩阵。"""
    x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _orientation_error_omega(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """旋转矩阵姿态误差 → 世界系角速度向量 omega（轴 × 角度）。

    omega 满足：把当前 frame 绕世界系 omega/|omega| 转 |omega| 弧度可对齐目标 frame。
    误差矩阵 R_err = R_target @ R_current.T，再取 log map（axis-angle）。
    """
    R_err = R_target @ R_current.T
    # log map：trace = 1 + 2*cos(angle)
    trace = float(np.clip(R_err[0,0] + R_err[1,1] + R_err[2,2], -1.0, 3.0))
    cos_ang = (trace - 1.0) * 0.5
    cos_ang = max(-1.0, min(1.0, cos_ang))
    angle = math.acos(cos_ang)
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float64)
    if angle > math.pi - 1e-3:
        # 接近 180°：用稳定公式
        # 找最大对角元素方向作为轴
        d = np.array([R_err[0,0], R_err[1,1], R_err[2,2]])
        i = int(np.argmax(d))
        axis = np.zeros(3, dtype=np.float64)
        axis[i] = math.sqrt(max(0.0, (R_err[i,i] + 1.0) * 0.5))
        for j in range(3):
            if j != i:
                axis[j] = R_err[i,j] / (2.0 * axis[i] + 1e-9)
        return axis * angle
    s = 2.0 * math.sin(angle)
    axis = np.array([
        R_err[2,1] - R_err[1,2],
        R_err[0,2] - R_err[2,0],
        R_err[1,0] - R_err[0,1],
    ], dtype=np.float64) / s
    return axis * angle


def _eef_watch_link_names(robot, arm: str) -> frozenset:
    """监测碰撞的 link 名：arm_link1-7 + gripper/finger/eef。"""
    names = {f"{arm}_arm_link{i}" for i in range(1, 8)}
    names |= set(robot.gripper_link_names.get(arm, []))
    names |= set(robot.finger_link_names.get(arm, []))
    eef_nm = robot.eef_link_names.get(arm)
    if eef_nm:
        names.add(eef_nm)
    return frozenset(names)


def _disabled_collision_pair_set(robot) -> set:
    out: set = set()
    try:
        for a, b in robot.disabled_collision_pairs:
            out.add(tuple(sorted((str(a), str(b)))))
    except Exception:
        pass
    return out


def _object_contact_prim_paths(obj) -> frozenset:
    """目标物体所有 link 的 prim path（contact 阶段允许触碰）。"""
    if obj is None:
        return frozenset()
    paths: set = set()
    try:
        links = getattr(obj, "links", None) or {}
        for lk in links.values():
            pp = getattr(lk, "prim_path", None)
            if pp:
                paths.add(str(pp))
    except Exception:
        pass
    return frozenset(paths)


def _eef_collision_hits(
    world,
    arm: str,
    *,
    allow_object=None,
    impulse_thresh: float = 1e-6,
) -> List[dict]:
    """读 PhysX contact_list，返回 EEF/手臂相关碰撞。

    监测方式：每步仿真后 robot.contact_list()，筛出 body 含本臂 watch link 的接触；
    - self：另一 body 也是机器人 link（且不在 disabled_collision_pairs）
    - world：另一 body 为场景/物体
  allow_object 非空时，与该物体 link 的接触不计入（用于 contact 段预期触物）。
    """
    robot = world.robot
    watch = _eef_watch_link_names(robot, arm)
    disabled = _disabled_collision_pair_set(robot)
    allow_paths = _object_contact_prim_paths(allow_object)
    robot_paths = frozenset(getattr(robot, "link_prim_paths", []) or [])
    hits: List[dict] = []
    try:
        contacts = robot.contact_list()
    except Exception:
        return hits
    for c in contacts:
        b0, b1 = str(c.body0), str(c.body1)
        n0, n1 = b0.split("/")[-1], b1.split("/")[-1]
        if n0 not in watch and n1 not in watch:
            continue
        other_path = b1 if n0 in watch else b0
        other_name = n1 if n0 in watch else n0
        watch_name = n0 if n0 in watch else n1
        if allow_paths and other_path in allow_paths:
            continue
        imp = getattr(c, "impulse", None)
        try:
            imp_mag = float(np.linalg.norm(np.asarray(imp, dtype=np.float64)))
        except Exception:
            imp_mag = 1.0
        if imp_mag < impulse_thresh:
            continue
        if other_path in robot_paths:
            pair = tuple(sorted((watch_name, other_name)))
            if pair in disabled:
                continue
            kind = "self"
        else:
            kind = "world"
        hits.append({
            "kind": kind,
            "link": watch_name,
            "other": other_name,
            "impulse": round(imp_mag, 6),
        })
    return hits


def _eef_goto_world(world, arm: str, target_world_pos, target_world_quat=None,
                    hold_world_quat=None,
                    max_steps: int = 200, pos_tol: float = 0.025,
                    ori_tol: float = 0.20,  # rad ≈ 11°
                    gripper_cmd: Optional[float] = None,
                    ctx=None, stage_name: str = "",
                    max_dq_per_step: float = 0.04,  # rad/step, ≈ 2.3°
                    max_dx_per_step: float = 0.04,  # m/step, ≈ 4cm
                    max_dw_per_step: float = 0.08,  # rad/step, ≈ 4.6°
                    ori_weight: float = 0.5,  # 方向误差相对位置的权重
                    null_ori_gain: float = 0.45,  # hold_world_quat：零空间姿态修正增益
                    lam: float = 0.10,
                    adaptive_ori: bool = True,  # True: 大姿态差时位置优先（默认）；
                                                # False: 姿态权重恒定（姿态优先，先转到目标朝向锁定 IK 分支）
                    disable_stuck_check: bool = False,
                    collision_watch_arm: Optional[str] = None,
                    abort_on_eef_collision: bool = False,
                    allow_contact_object=None):
    """生成器：用 Jacobian DLS IK 把 eef 收敛到 world frame target_pos (+target_quat)。

    【exec_move 硬约束】本函数及所有 exec_move 路径**只允许命令 arm_{arm} 与 gripper_{arm}**，
    **禁止**通过 make_action 移动底盘(base)或腰部(trunk)。腰/底盘在抓取执行全程保持不动。

    每步：
      1. 读 eef 真实 world pos + quat + jacobian
      2. dx_world = target - eef（限速 max_dx_per_step）
      3. 若 target_world_quat 给出：算 omega_world = R_target × R_cur^T 的 log map
         （限速 max_dw_per_step），组合 6D 误差 [dx; ori_weight * omega]
         用 J[:6,:] 求 dq
      3b. 若 hold_world_quat 给出（且 target_world_quat 为 None）：
         主任务 J[:3,:] 追位置，零空间用 J[3:6,:] 抑制姿态漂移（move_eef 平移）
      3c. 否则只用 J[:3,:] + dx（姿态可自由漂移）
      4. clip dq 到 max_dq_per_step
      5. 发 arm JointController delta = dq

    收敛条件：pos_err < pos_tol AND (target_quat=None OR ori_err < ori_tol)
    gripper_cmd: 不为 None 时每帧顺带发夹爪命令。
    collision_watch_arm + abort_on_eef_collision：每步读 PhysX contact_list 监测 EEF/手臂碰撞。
    """
    from behavior_interface.skills.eef import (
        _assert_legacy_7dof_motion_ready,
        _prepare_legacy_7dof_motion,
    )

    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    target_world_pos = np.asarray(target_world_pos, dtype=np.float64)
    use_ori = target_world_quat is not None
    hold_ori = (not use_ori) and hold_world_quat is not None
    R_target = _quat_to_mat(np.asarray(target_world_quat)) if use_ori else None
    R_hold = _quat_to_mat(np.asarray(hold_world_quat)) if hold_ori else None
    last_err = float("inf")
    stuck_cnt = 0
    last_eef_pos = None
    collision_hits: List[dict] = []

    for step in range(max_steps):
        if abort_on_eef_collision and collision_watch_arm:
            hits = _eef_collision_hits(
                world, collision_watch_arm, allow_object=allow_contact_object,
            )
            if hits:
                collision_hits = hits
                if ctx is not None:
                    ctx.log(
                        f"    [{stage_name}] ABORT eef_collision "
                        f"{hits[0].get('kind')} {hits[0].get('link')}↔{hits[0].get('other')}"
                    )
                return float("inf")

        eef = world.eef_pose(arm=arm)
        epos = np.array(eef["pos"], dtype=np.float64)
        equat = np.array(eef["quat"], dtype=np.float64)
        dx_world = target_world_pos - epos
        err = float(np.linalg.norm(dx_world))
        last_err = err

        # 方向误差
        omega_world = np.zeros(3, dtype=np.float64)
        ori_err_rad = 0.0
        if use_ori:
            R_cur = _quat_to_mat(equat)
            omega_world = _orientation_error_omega(R_target, R_cur)
            ori_err_rad = float(np.linalg.norm(omega_world))
        elif hold_ori:
            R_cur = _quat_to_mat(equat)
            omega_world = _orientation_error_omega(R_hold, R_cur)
            ori_err_rad = float(np.linalg.norm(omega_world))

        # 收敛
        if err < pos_tol and (not use_ori or ori_err_rad < ori_tol):
            if ctx is not None:
                ctx.log(f"    [{stage_name}] converged step={step} "
                        f"pos_err={err:.3f}m ori_err={math.degrees(ori_err_rad):.1f}°")
            return last_err

        # 限速 dx, omega
        if err > max_dx_per_step:
            dx_world = dx_world * (max_dx_per_step / err)
        if ori_err_rad > max_dw_per_step:
            omega_world = omega_world * (max_dw_per_step / ori_err_rad)

        # ori_weight 自适应：远距离/大旋转时位置优先，避免 IK 卡在大姿态调整
        # ori_err > 1.0 rad (~57°): weight × 0.1  （只看位置，等位置到了再调姿态）
        # ori_err 0.3~1.0 rad: weight 线性插值
        # ori_err < 0.3 rad (~17°): full weight  （精准对齐）
        ori_w_eff = ori_weight
        if use_ori and adaptive_ori:
            if ori_err_rad > 1.0:
                ori_w_eff = ori_weight * 0.1
            elif ori_err_rad > 0.3:
                # 0.3 → ori_weight；1.0 → 0.1 * ori_weight
                t = (1.0 - ori_err_rad) / (1.0 - 0.3)  # 1→0
                ori_w_eff = ori_weight * (0.1 + 0.9 * t)

        try:
            J, arm_dof_idx = _read_jacobian_arm(world, arm)
            if use_ori:
                J_use = J[:6, :]
                # 用 effective weight 缩放方向行
                J_scaled = J_use.copy()
                J_scaled[3:, :] = ori_w_eff * J_use[3:, :]
                dx6 = np.concatenate([dx_world, ori_w_eff * omega_world])
                dq = _dls_solve_dq(J_scaled, dx6, lam=lam)
                dx_pred = J[:3, :] @ dq
            elif hold_ori:
                j_pos = J[:3, :]
                dq_prim = _dls_solve_dq(j_pos, dx_world, lam=lam)
                j_pinv = _dls_pinv(j_pos, lam=lam)
                n_dof = j_pos.shape[1]
                null_proj = np.eye(n_dof) - j_pinv @ j_pos
                omega_hold = omega_world.copy()
                ow_norm = float(np.linalg.norm(omega_hold))
                if ow_norm > max_dw_per_step:
                    omega_hold = omega_hold * (max_dw_per_step / ow_norm)
                j_ori = J[3:6, :]
                dq_sec = null_proj @ j_ori.T @ (null_ori_gain * omega_hold)
                dq = dq_prim + dq_sec
                dx_pred = j_pos @ dq
            else:
                J_use = J[:3, :]
                dq = _dls_solve_dq(J_use, dx_world, lam=lam)
                dx_pred = J_use @ dq
            dq_norm_inf = float(np.linalg.norm(dq, ord=np.inf))
            if dq_norm_inf > max_dq_per_step:
                dq = dq * (max_dq_per_step / dq_norm_inf)
        except Exception as e:
            if ctx is not None:
                ctx.log(f"    [{stage_name}] jacobian err: {e}")
            return last_err

        if ctx is not None and (step == 0 or step % 30 == 0):
            dq_str = "[" + ",".join(f"{v:+.3f}" for v in dq) + "]"
            ori_info = (
                f" ori_err={math.degrees(ori_err_rad):.1f}°"
                if (use_ori or hold_ori) else ""
            )
            ctx.log(
                f"    [{stage_name}] step={step:3d} eef=({epos[0]:.3f},"
                f"{epos[1]:.3f},{epos[2]:.3f}) err={err:.3f}m{ori_info} "
                f"dq_inf={dq_norm_inf:.4f} dq={dq_str} "
                f"dx_pred=({dx_pred[0]:+.3f},{dx_pred[1]:+.3f},{dx_pred[2]:+.3f})"
            )

        # stuck 检测：连续 20 步 eef 没动 → IK 解到接近奇异，提前退出
        # （除非显式 disable，e.g. grip stage 需要持续 push 即使 stuck）
        if last_eef_pos is not None and not disable_stuck_check:
            moved = float(np.linalg.norm(epos - last_eef_pos))
            if moved < 0.001:
                stuck_cnt += 1
            else:
                stuck_cnt = 0
            if stuck_cnt >= 20:
                if ctx is not None:
                    ctx.log(f"    [{stage_name}] stuck at step={step} err={err:.3f}m")
                return last_err
        last_eef_pos = epos

        # absolute position 模式：controller 期望接收 arm 7 个关节的绝对目标位置。
        # DLS IK 解出的 dq 是 delta，需要叠加到当前 qpos 上再发出去。
        q_cur_arm = _arm_qpos(world, arm)
        q_tgt_arm = q_cur_arm + dq
        overrides = {f"arm_{arm}": q_tgt_arm.tolist()}
        grip_override = _gripper_cmd_override(gripper_cmd)
        if grip_override is not None:
            overrides[f"gripper_{arm}"] = grip_override
        _assert_legacy_7dof_motion_ready(world, arm)
        yield world.make_action(**overrides)

    if ctx is not None:
        ctx.log(f"    [{stage_name}] timeout {max_steps} steps, final err={last_err:.3f}m")
    return last_err


def _set_arm_qpos_direct(world, arm: str, arm_q) -> None:
    """直接把手臂设到指定位形（规划探针用）。"""
    idx = _get_arm_dof_idx(world, arm)
    q = world.robot.get_joint_positions().clone()
    aq = np.asarray(arm_q, dtype=np.float64).reshape(7)
    for i, j in enumerate(idx):
        q[int(j)] = float(aq[i])
    world.robot.set_joint_positions(q)


def _dls_probe_steps(
    world,
    arm: str,
    target_pos,
    target_quat=None,
    *,
    max_steps: int = 100,
    pos_tol: float = 0.10,
    gripper_cmd: float = 1.0,
    disable_stuck_check: bool = True,
    cancel_check=None,
) -> float:
    """同步 DLS 迭代（不 yield），返回最终位置误差。调用前须已设好起始关节。"""
    from behavior_interface.errors import SkillCancelled
    from behavior_interface.skills.arm_reset import _arm_qpos

    target_world_pos = np.asarray(target_pos, dtype=np.float64)
    use_ori = target_quat is not None
    R_target = _quat_to_mat(np.asarray(target_quat)) if use_ori else None
    last_err = float("inf")
    stuck_cnt = 0
    last_eef_pos = None
    max_dq_per_step = 0.04
    max_dx_per_step = 0.04
    max_dw_per_step = 0.08
    ori_weight = 0.5
    lam = 0.10

    for step in range(max_steps):
        if cancel_check is not None and step % 6 == 0 and cancel_check():
            raise SkillCancelled("DLS 迭代已取消")
        eef = world.eef_pose(arm=arm)
        epos = np.array(eef["pos"], dtype=np.float64)
        equat = np.array(eef["quat"], dtype=np.float64)
        dx_world = target_world_pos - epos
        err = float(np.linalg.norm(dx_world))
        last_err = err

        omega_world = np.zeros(3, dtype=np.float64)
        ori_err_rad = 0.0
        if use_ori:
            R_cur = _quat_to_mat(equat)
            omega_world = _orientation_error_omega(R_target, R_cur)
            ori_err_rad = float(np.linalg.norm(omega_world))

        if err < pos_tol and (not use_ori or ori_err_rad < 0.20):
            return last_err
        # 规划探针加速：远离目标且步数已多，提前放弃
        if step >= 55 and err > pos_tol * 1.35:
            return last_err

        if err > max_dx_per_step:
            dx_world = dx_world * (max_dx_per_step / err)
        if ori_err_rad > max_dw_per_step:
            omega_world = omega_world * (max_dw_per_step / ori_err_rad)

        ori_w_eff = ori_weight
        if use_ori:
            if ori_err_rad > 1.0:
                ori_w_eff = ori_weight * 0.1
            elif ori_err_rad > 0.3:
                t = (1.0 - ori_err_rad) / (1.0 - 0.3)
                ori_w_eff = ori_weight * (0.1 + 0.9 * t)

        try:
            J, _ = _read_jacobian_arm(world, arm)
            if use_ori:
                J_use = J[:6, :]
                J_scaled = J_use.copy()
                J_scaled[3:, :] = ori_w_eff * J_use[3:, :]
                dx6 = np.concatenate([dx_world, ori_w_eff * omega_world])
                dq = _dls_solve_dq(J_scaled, dx6, lam=lam)
            else:
                dq = _dls_solve_dq(J[:3, :], dx_world, lam=lam)
            dq_norm_inf = float(np.linalg.norm(dq, ord=np.inf))
            if dq_norm_inf > max_dq_per_step:
                dq = dq * (max_dq_per_step / dq_norm_inf)
        except Exception:
            return last_err

        if last_eef_pos is not None and not disable_stuck_check:
            moved = float(np.linalg.norm(epos - last_eef_pos))
            stuck_cnt = stuck_cnt + 1 if moved < 0.001 else 0
            if stuck_cnt >= 20:
                return last_err
        last_eef_pos = epos

        q_cur_arm = _arm_qpos(world, arm)
        q_tgt_arm = q_cur_arm + dq
        # 规划探针：仅运动学改关节，不 env.step，避免推动物体
        _set_arm_qpos_direct(world, arm, q_tgt_arm)

    return last_err


def _dls_probe_exec_path(
    world,
    arm: str,
    target_pos,
    target_quat,
    chest_q,
    *,
    back_m: float = 0.10,
    safe_tol: float = 0.10,
    cnt_tol: float = 0.08,
    max_steps_safe: int = 90,
    max_steps_cnt: int = 100,
    cancel_check=None,
) -> Tuple[float, float]:
    """从胸前位形同步模拟 exec tuck→safe→contact，返回 (safe_err, cnt_err)。"""
    from behavior_interface.errors import SkillCancelled

    def _chk(where: str) -> None:
        if cancel_check is not None and cancel_check():
            raise SkillCancelled(f"DLS 探针已取消（{where}）")

    saved = world.robot.get_joint_positions().clone()
    try:
        _chk("tuck")
        _set_arm_qpos_direct(world, arm, chest_q)

        tgt = np.asarray(target_pos, dtype=np.float64).reshape(3)
        tq = np.asarray(target_quat, dtype=np.float64).reshape(4)
        pointing = _quat_to_mat(tq)[:, 2]
        pn = float(np.linalg.norm(pointing))
        pointing = pointing / pn if pn > 1e-9 else np.array([0.0, 0.0, -1.0])
        safe_pos = tgt - float(back_m) * pointing

        _chk("safe")
        safe_err = _dls_probe_steps(
            world, arm, safe_pos, tq,
            max_steps=max_steps_safe, pos_tol=safe_tol,
            cancel_check=cancel_check,
        )
        if safe_err > safe_tol:
            return safe_err, float("inf")

        _chk("contact")
        cnt_err = _dls_probe_steps(
            world, arm, tgt, None,
            max_steps=max_steps_cnt, pos_tol=cnt_tol,
            cancel_check=cancel_check,
        )
        return safe_err, cnt_err
    finally:
        try:
            world.robot.set_joint_positions(saved)
        except Exception:
            pass


def _hold_pose_curobo_placeholder():
    """占位—下方 _hold_pose 之前插 cuRobo 集成。"""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# cuRobo IK 集成（取代自实现 DLS 在复杂场景的 IK）
# ─────────────────────────────────────────────────────────────────────────────

_CUROBO_CACHE: Dict[str, Any] = {"mg": None, "robot_id": None}

# 抓取时腰(trunk)是否锁死。
# False（默认）：官方 ARM embodiment，腰+臂一起规划、仅锁底盘——最自然、可达性最好。
#   cuRobo 已证明锁腰会让 top-down 俯抓够不到（位置差18mm、朝向差38°）。
# True：切 arm_no_torso，锁腰+锁底盘只动手臂（仅适合近距离、不需腰参与的抓取）。
_LOCK_TRUNK: bool = True


def _curobo_gpu_mem_info() -> Tuple[Optional[int], Optional[int]]:
    try:
        import torch as th
        if th.cuda.is_available():
            free, total = th.cuda.mem_get_info()
            return int(free), int(total)
    except Exception:
        pass
    return None, None


def _env_mib(name: str, default: int) -> int:
    try:
        import os
        return max(0, int(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _world_gpu_diag(world, tag: str, *, extra: Optional[Dict[str, Any]] = None,
                    include_nvidia: bool = True) -> None:
    log_fn = getattr(world, "_gpu_diag_log", None)
    if not callable(log_fn):
        return
    try:
        from behavior_interface.gpu_diag import log_gpu_diag
        log_gpu_diag(log_fn, tag, extra=extra, include_nvidia=include_nvidia)
    except Exception:
        pass


def _curobo_gpu_low_mem() -> bool:
    """Isaac 占满显存时 cuRobo warmup 易 OOM，需降 batch / 关 cuda_graph。"""
    free, _ = _curobo_gpu_mem_info()
    if free is not None:
        threshold_mib = _env_mib("CUROBO_LOW_MEM_FREE_MB", 6144)
        return free < threshold_mib * 1024 * 1024
    return False


def _get_curobo_mg(world, *, force_low_mem: bool = False):
    """单例 CuRoboMotionGenerator。第一次调用阻塞 ~30s 构建 motion gen graph，
    之后 O(ms) 复用。

    关键：MG 缓存挂在 `world` 对象上（而非模块级），这样 reload skills 不会丢；
    且**只要该 robot 已有任何可用 MG 就复用**，绝不为了 low_mem 再建第二个
    （第二个会和已存在的那个一起吃显存 → OOM，这正是物体被扫翻的根因之一）。"""
    robot_id = id(world.robot)

    # 1) world 上的持久缓存（reload 不丢）。缓存标记的 lock_trunk 必须与当前 _LOCK_TRUNK
    #    一致，否则是另一种 embodiment 的旧 MG，必须丢弃重建。
    wcache = getattr(world, "_curobo_mg_cache", None)
    if (isinstance(wcache, dict) and wcache.get("robot_id") == robot_id
            and wcache.get("mg") is not None
            and wcache.get("lock_trunk") == _LOCK_TRUNK):
        return wcache["mg"]

    # 2) 模块级缓存兜底（同一 reload 周期内）
    mcached = _CUROBO_CACHE.get("mg")
    if (mcached is not None and _CUROBO_CACHE.get("robot_id") == robot_id
            and _CUROBO_CACHE.get("lock_trunk") == _LOCK_TRUNK):
        try:
            world._curobo_mg_cache = {"robot_id": robot_id, "mg": mcached, "lock_trunk": _LOCK_TRUNK}
        except Exception:
            pass
        return mcached

    low_mem = force_low_mem or _curobo_gpu_low_mem()
    from omnigibson.action_primitives.curobo import (
        CuRoboMotionGenerator, CuRoboEmbodimentSelection,
    )
    mode = "low_mem" if low_mem else "normal"

    free, total = _curobo_gpu_mem_info()
    if free is not None:
        min_free_mib = _env_mib("CUROBO_MG_MIN_FREE_MB", 2048)
        free_mib = free / (1024 * 1024)
        total_mib = (total or 0) / (1024 * 1024)
        _world_gpu_diag(
            world,
            "curobo_mg.before_warmup",
            extra={
                "mode": mode,
                "force_low_mem": bool(force_low_mem),
                "free_mib": round(free_mib, 1),
                "total_mib": round(total_mib, 1),
                "min_free_mib": int(min_free_mib),
                "lock_trunk": bool(_LOCK_TRUNK),
            },
        )
        if free < min_free_mib * 1024 * 1024:
            _world_gpu_diag(
                world,
                "curobo_mg.skip_low_free",
                extra={
                    "mode": mode,
                    "free_mib": round(free_mib, 1),
                    "total_mib": round(total_mib, 1),
                    "min_free_mib": int(min_free_mib),
                },
            )
            raise RuntimeError(
                f"skip cuRobo warmup: GPU free {free_mib:.0f} MiB < "
                f"CUROBO_MG_MIN_FREE_MB={min_free_mib} MiB "
                f"(total={total_mib:.0f} MiB, mode={mode})"
            )
        print(
            f"[curobo] GPU free {free_mib:.0f}/{total_mib:.0f} MiB before warmup "
            f"(mode={mode})",
            flush=True,
        )

    # 丢弃可能存在的旧 MG（如官方 ARM 版），释放显存后再建新的，避免新旧并存 OOM
    try:
        if isinstance(getattr(world, "_curobo_mg_cache", None), dict):
            world._curobo_mg_cache = None
        _CUROBO_CACHE["mg"] = None
        _CUROBO_CACHE.pop("no_torso", None)
    except Exception:
        pass

    print(f"[curobo] 初始化 CuRoboMotionGenerator ({mode}, 耗时 ~30s)...", flush=True)

    # ── embodiment 选择：exec 路径要求没显式动的身体关节锁住。
    # _LOCK_TRUNK=True 使用 arm_no_torso；执行侧 lock_body=True 会冻结底盘 + trunk。
    cfg_path_override = None
    if _LOCK_TRUNK:
        try:
            import os as _os
            base_cfg = dict(world.robot.curobo_path)
            arm_path = base_cfg.get(CuRoboEmbodimentSelection.ARM)
            if arm_path:
                nt_path = arm_path.replace("_arm.yaml", "_arm_no_torso.yaml")
                if nt_path != arm_path and _os.path.exists(nt_path):
                    base_cfg[CuRoboEmbodimentSelection.ARM] = nt_path
                    cfg_path_override = base_cfg
                    print(f"[curobo] ARM→arm_no_torso（锁腰+锁底盘，仅手臂）", flush=True)
        except Exception as _e:
            print(f"[curobo] arm_no_torso 切换失败，沿用官方 ARM: {_e}", flush=True)
    else:
        print("[curobo] 用官方 ARM embodiment（腰+臂规划，仅锁底盘）", flush=True)

    try:
        import gc
        import torch as th
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
            th.cuda.synchronize()
    except Exception:
        pass
    try:
        mg = CuRoboMotionGenerator(
            robot=world.robot,
            robot_cfg_path=cfg_path_override,
            batch_size=1,
            use_cuda_graph=False,
            debug=False,
            use_default_embodiment_only=False,
            collision_activation_distance=0.001,
            motion_cfg_kwargs={"self_collision_check": False},
        )
    except Exception as e:
        err_s = str(e).lower()
        _world_gpu_diag(
            world,
            "curobo_mg.exception",
            extra={
                "mode": mode,
                "force_low_mem": bool(force_low_mem),
                "error": f"{type(e).__name__}: {e}",
            },
        )
        # OOM：先彻底清缓存再重试一次 low_mem，避免半分配残留
        try:
            import gc
            import torch as th
            gc.collect()
            if th.cuda.is_available():
                th.cuda.empty_cache()
                th.cuda.synchronize()
        except Exception:
            pass
        if not low_mem and ("out of memory" in err_s or "cuda" in err_s):
            print("[curobo] OOM，切换 low_mem 重试", flush=True)
            _world_gpu_diag(
                world,
                "curobo_mg.retry_low_mem",
                extra={"from_mode": mode, "error": f"{type(e).__name__}: {e}"},
            )
            return _get_curobo_mg(world, force_low_mem=True)
        raise
    _CUROBO_CACHE["mg"] = mg
    _CUROBO_CACHE["robot_id"] = robot_id
    _CUROBO_CACHE["lock_trunk"] = _LOCK_TRUNK
    try:
        world._curobo_mg_cache = {"robot_id": robot_id, "mg": mg, "lock_trunk": _LOCK_TRUNK}
    except Exception:
        pass
    print(f"[curobo] 初始化完成 ({mode}, lock_trunk={_LOCK_TRUNK})", flush=True)
    _world_gpu_diag(
        world,
        "curobo_mg.after_warmup",
        extra={
            "mode": mode,
            "batch_size": int(getattr(mg, "batch_size", -1)),
            "lock_trunk": bool(_LOCK_TRUNK),
        },
    )
    return mg


_CUROBO_OBSTACLES_LAST_UPDATE: Dict[str, float] = {"t": -1.0}


def _maybe_update_obstacles(mg, ctx=None, force: bool = False, ttl_s: float = 1.0,
                            ignore_objects=None):
    """按 ttl_s 节流地刷新 cuRobo collision world。
    第一次调用、或距上次刷新 > ttl_s、或 force 时执行。
    ignore_objects：从碰撞世界排除的物体列表（如抓取目标，让夹爪能贴近）。
    注意：带 ignore_objects 时强制刷新（不同 ignore 集合的缓存不可复用）。
    """
    import time as _time
    now = _time.time()
    if ignore_objects:
        force = True
    if force or now - _CUROBO_OBSTACLES_LAST_UPDATE["t"] > ttl_s:
        try:
            mg.update_obstacles(ignore_objects=ignore_objects)
            _CUROBO_OBSTACLES_LAST_UPDATE["t"] = now
            if ctx is not None:
                n_ig = len(ignore_objects) if ignore_objects else 0
                ctx.log(f"    [curobo] update_obstacles done (ignore={n_ig})")
        except Exception as e:
            if ctx is not None:
                ctx.log(f"    [curobo] update_obstacles failed: {e}")


def _curobo_arm_ik_reachable(
    world, arm: str, pos, quat,
    *, max_attempts: int = 12, timeout: float = 3.0,
    world_collision: bool = False, lock_other_arm: bool = True,
    seed_arm_qpos=None,
    pos_tol: float = 0.02,
    rot_tol: float = 0.20,
) -> Tuple[bool, float, float]:
    """同步探针：在当前 embodiment（_LOCK_TRUNK 决定是否锁腰）下，用 cuRobo ik_only
    判断给定 eef 世界位姿是否"手臂可达"。返回 (ok, pos_err_m, rot_err_rad)。

    seed_arm_qpos：可选 7 维手臂关节初值（如胸前 tuck），模拟 exec tuck→safe 的起始态。
    world_collision=False：纯运动学可达（默认）——用于规划阶段选朝向，碰撞由执行阶段处理。
    必须在主仿真线程调用（compute_trajectories 访问 og.sim）。"""
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
    try:
        mg = _get_curobo_mg(world)
    except Exception:
        return False, float("inf"), float("inf")
    robot = world.robot
    bs = int(mg.batch_size)
    tp: Dict[str, Any] = {}
    tq: Dict[str, Any] = {}
    link = robot.eef_link_names[arm]
    tp[link] = th.stack([th.tensor(list(pos), dtype=th.float32) for _ in range(bs)])
    tq[link] = th.stack([th.tensor(list(quat), dtype=th.float32) for _ in range(bs)])
    if lock_other_arm:
        for a in robot.arm_names:
            if a == arm:
                continue
            ep = world.eef_pose(arm=a)
            ol = robot.eef_link_names[a]
            tp[ol] = th.stack([th.tensor(list(ep["pos"]), dtype=th.float32) for _ in range(bs)])
            tq[ol] = th.stack([th.tensor(list(ep["quat"]), dtype=th.float32) for _ in range(bs)])
    initial_joint_pos = None
    if seed_arm_qpos is not None:
        try:
            q = robot.get_joint_positions().clone()
            arm_idx = _get_arm_dof_idx(world, arm)
            sq = th.as_tensor(list(seed_arm_qpos), dtype=q.dtype, device=q.device)
            for i, j in enumerate(arm_idx):
                q[int(j)] = sq[i]
            # cuRobo 内部会 stack([initial_joint_pos] * batch_size)，此处须传 1D 全关节向量
            initial_joint_pos = q
        except Exception:
            initial_joint_pos = None
    try:
        res = mg.compute_trajectories(
            target_pos=tp, target_quat=tq, initial_joint_pos=initial_joint_pos, is_local=False,
            max_attempts=max_attempts, timeout=timeout, ik_fail_return=50,
            enable_finetune_trajopt=False, finetune_attempts=0,
            return_full_result=True, success_ratio=1.0 / bs,
            attached_obj=None, attached_obj_scale=None, motion_constraint=None,
            skip_obstacle_update=True, ik_only=True,
            ik_world_collision_check=world_collision,
            emb_sel=CuRoboEmbodimentSelection.ARM,
        )
    except Exception:
        return False, float("inf"), float("inf")
    # full_results 是按 batch 的列表，每个有 position_error/rotation_error 形状 (n_links, batch)。
    # 取"最好 seed"：按 batch 取 max(over links)，再跨 batch 取 min。
    results = res if isinstance(res, (list, tuple)) else [res]
    best_pe = float("inf")
    best_re = float("inf")
    for r in results:
        pe = getattr(r, "position_error", None)
        re_ = getattr(r, "rotation_error", None)
        if pe is None:
            continue
        pe_pb = pe.max(dim=0).values if pe.dim() == 2 else pe
        re_pb = (re_.max(dim=0).values if (re_ is not None and re_.dim() == 2)
                 else (re_ if re_ is not None else None))
        n = pe_pb.numel()
        for b in range(n):
            pv = float(pe_pb[b].item())
            rv = float(re_pb[b].item()) if re_pb is not None else 0.0
            if pv < best_pe:
                best_pe = pv
                best_re = rv
    ok = (best_pe < float(pos_tol)) and (best_re < float(rot_tol))
    return ok, best_pe, best_re


# 与官方 starter_semantic_action_primitives.py 中的常量保持一致
_M_JOINT_POS_DIFF_THRESHOLD = 0.005
_M_LOW_PRECISION_JOINT_POS_DIFF_THRESHOLD = 0.05
_M_DEFAULT_DIST_THRESHOLD = 0.02
_M_DEFAULT_ANGLE_THRESHOLD = 0.05
_M_MAX_STEPS_PER_WAYPOINT = 10    # 每个 waypoint 最多 yield 多少帧（对齐官方 MAX_STEPS_FOR_JOINT_MOTION=10）


def _body_lock_joint_indices(robot) -> List[int]:
    """lock_body=True 时冻结底盘 + trunk；exec 只能动指定手臂/夹爪。"""
    idx: List[int] = []
    try:
        idx.extend(_to_np(robot.base_control_idx).astype(int).tolist())
    except Exception:
        pass
    try:
        idx.extend(_to_np(robot.trunk_control_idx).astype(int).tolist())
    except Exception:
        pass
    return idx


def _freeze_body_on_trajectory(q_traj, lock_idx: List[int], lock_vals) -> None:
    """把轨迹中底盘/躯干关节钉死在执行起点。"""
    if not lock_idx or lock_vals is None or len(lock_vals) == 0:
        return
    import torch as th
    for i in range(len(q_traj)):
        q_traj[i, lock_idx] = lock_vals


def _full_q_controller_action(world, joint_pos, arm: str, gripper_cmd=None):
    """Map a full-q waypoint without requiring every controller to be JointController."""
    robot = world.robot
    q_np = _to_np(joint_pos).reshape(-1)
    overrides: Dict[str, Any] = {}
    for name, controller in robot.controllers.items():
        if name.startswith("gripper_"):
            if name == f"gripper_{arm}" and gripper_cmd is not None:
                overrides[name] = _gripper_cmd_override(gripper_cmd)
            continue
        if name.startswith("tool_roll_"):
            # J8 is independently pinned by WorldAPI and must not be moved by
            # a 7-DoF arm trajectory.
            continue
        dof_idx = _to_np(controller.dof_idx).astype(int).reshape(-1)
        if name == "base":
            if dof_idx.size < 3:
                continue
            target_x, target_y, target_yaw = [float(x) for x in q_np[dof_idx[:3]]]
            pose = world.robot_pose()
            dx = target_x - float(pose.pos[0])
            dy = target_y - float(pose.pos[1])
            cos_yaw = math.cos(float(pose.yaw))
            sin_yaw = math.sin(float(pose.yaw))
            overrides[name] = [
                cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy,
                _wrap_angle_py(target_yaw - float(pose.yaw)),
            ]
            continue
        overrides[name] = [float(x) for x in q_np[dof_idx].tolist()]
    return world.make_action(**overrides)


def _eef_goto_world_curobo(world, arm: str,
                            target_world_pos, target_world_quat,
                            gripper_cmd: Optional[float] = None,
                            ctx=None, stage_name: str = "",
                            max_attempts: int = 20,
                            timeout: float = 15.0,
                            low_precision: bool = False,
                            lock_other_arm: bool = False,
                            allow_contact: bool = True,
                            lock_body: bool = False):
    """allow_contact=True：关掉 world collision check（抓取/开门时手必须"碰"目标物体，
    cuRobo 默认会因 collision 拒绝有效 IK 解；此处放行）。"""
    """cuRobo 规划 + 执行（generator）。完全按 OmniGibson 官方
    `starter_semantic_action_primitives._move_hand` + `_plan_joint_motion`
    + `_execute_motion_plan` 的调用模式实现。

    target_world_pos: 3 维世界坐标
    target_world_quat: 4 维 xyzw 世界四元数（None 时用 current eef quat）
    lock_other_arm: True 时把另一只手的当前 pose 也加进 target，
                    避免 cuRobo padding bug 把另一臂送到 (0,0,0)
    返回（StopIteration.value）: 0.0 表示成功，inf 表示规划失败
    """
    import math as _math
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection
    from behavior_interface.skills.eef import (
        _assert_legacy_7dof_motion_ready,
        _prepare_legacy_7dof_motion,
    )

    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    if target_world_quat is None:
        eef = world.eef_pose(arm=arm)
        target_world_quat = eef["quat"]

    try:
        mg = _get_curobo_mg(world)
    except Exception as e:
        if ctx is not None:
            ctx.log(f"    [{stage_name}] curobo 初始化失败: {e}")
        for _ in range(3):
            yield world.empty_action()
        return float("inf")
    robot = world.robot
    batch_size = int(mg.batch_size)
    body_lock_idx = _body_lock_joint_indices(robot) if lock_body else []
    body_lock_vals = None
    if body_lock_idx:
        _q0 = robot.get_joint_positions().cpu()
        body_lock_vals = _q0[body_lock_idx].clone()
        if ctx is not None:
            ctx.log(
                f"    [{stage_name}] lock_body: 冻结 {len(body_lock_idx)} 个底盘/躯干关节"
            )

    # 1) 更新 collision world（不强制；TTL 节流避免每个 stage 都全场扫一次）
    _maybe_update_obstacles(mg, ctx=ctx, force=False, ttl_s=1.0)

    # 2) 准备 target dict。lock_other_arm=True 时把另一臂的当前 eef pose 也传进去，
    #    防止 cuRobo 内部 default ee_link target=(0,0,0) padding bug 导致整体 IK 失败。
    target_pos_d: Dict[str, "th.Tensor"] = {}
    target_quat_d: Dict[str, "th.Tensor"] = {}
    for a in robot.arm_names:
        link_name = robot.eef_link_names[a]
        if a == arm:
            pos = th.tensor(target_world_pos, dtype=th.float32)
            quat = th.tensor(target_world_quat, dtype=th.float32)
        elif lock_other_arm:
            ep = world.eef_pose(arm=a)
            pos = th.tensor(ep["pos"], dtype=th.float32)
            quat = th.tensor(ep["quat"], dtype=th.float32)
        else:
            continue
        target_pos_d[link_name] = pos
        target_quat_d[link_name] = quat

    # batch_size 次复制（官方做法）
    target_pos_d = {k: th.stack([v for _ in range(batch_size)]) for k, v in target_pos_d.items()}
    target_quat_d = {k: th.stack([v for _ in range(batch_size)]) for k, v in target_quat_d.items()}

    if ctx is not None:
        ctx.log(f"    [{stage_name}] curobo 规划 → ({target_world_pos[0]:.3f},"
                f"{target_world_pos[1]:.3f},{target_world_pos[2]:.3f}) "
                f"quat=({target_world_quat[0]:+.2f},{target_world_quat[1]:+.2f},"
                f"{target_world_quat[2]:+.2f},{target_world_quat[3]:+.2f}) "
                f"locks={list(target_pos_d.keys())}")

    try:
        # 完全照搬官方 _plan_joint_motion 参数，但 allow_contact 时关 collision check：
        #   抓取/开门本来就要让手接触物体，cuRobo 默认会因 collision 拒绝有效 IK 解
        # allow_contact=True 时走 ik_only 路径（trajopt 同样会做 collision check）
        # 用 return_full_result=True 拿到 IKResult.status / pose_error 便于诊断
        full_results = mg.compute_trajectories(
            target_pos=target_pos_d,
            target_quat=target_quat_d,
            initial_joint_pos=None,
            is_local=False,
            max_attempts=max(1, _math.ceil(max_attempts / batch_size)),
            timeout=timeout,
            ik_fail_return=5,
            enable_finetune_trajopt=not allow_contact,
            finetune_attempts=1,
            return_full_result=True,
            success_ratio=1.0 / batch_size,
            attached_obj=None,
            attached_obj_scale=None,
            motion_constraint=None,
            skip_obstacle_update=True,
            ik_only=allow_contact,
            ik_world_collision_check=not allow_contact,
            emb_sel=CuRoboEmbodimentSelection.ARM,
        )
    except Exception as e:
        if ctx is not None:
            ctx.log(f"    [{stage_name}] curobo 异常: {type(e).__name__}: {e}")
        for _ in range(3):
            yield world.empty_action()
        return float("inf")

    # 从 full_results 中提取 successes & traj_paths（兼容 IKResult / MotionGenResult）
    successes_list = []
    traj_paths = []
    for r in full_results:
        s = getattr(r, "success", None)
        if s is None:
            successes_list.append(False)
            traj_paths.append(None)
            continue
        # success: 可能是 tensor，可能是 batch
        s_t = s if isinstance(s, th.Tensor) else th.tensor([bool(s)])
        if allow_contact:
            # IKResult: 用 .js_solution，已经是一个 JointState
            js = getattr(r, "js_solution", None)
            for i in range(s_t.numel()):
                successes_list.append(bool(s_t.flatten()[i].item()))
                # IKResult.js_solution 是 batch JointState
                if js is None:
                    traj_paths.append(None)
                else:
                    # 取第 i 个解
                    try:
                        traj_paths.append(js[i])
                    except Exception:
                        traj_paths.append(js)
        else:
            # MotionGenResult：必须用 get_paths() 取每条 batch 的 JointState 轨迹
            batch_paths: list = []
            if hasattr(r, "get_paths"):
                try:
                    batch_paths = list(r.get_paths())
                except Exception:
                    batch_paths = []
            for i in range(s_t.numel()):
                successes_list.append(bool(s_t.flatten()[i].item()))
                if i < len(batch_paths) and batch_paths[i] is not None:
                    traj_paths.append(batch_paths[i])
                else:
                    traj_paths.append(None)
    successes = th.tensor(successes_list, dtype=th.bool)

    success = bool(successes.any().item()) if len(successes) > 0 else False
    idx_succ = -1
    if success:
        idx_succ = int(successes.nonzero(as_tuple=True)[0][0].item())
    else:
        # 即使 cuRobo 报失败，也看一下 pos_err：如果某个 seed 在所有 link 上 pos_err < 2cm
        # 且 rot_err < 0.1 rad，仍认为是"工程上够用"的 IK 解，避免误判。
        # （cuRobo 内部 success 阈值约 5e-3 m，常常因为 mm 级误差判失败）
        # full_results 是 list of result（按 batch 分），每个 result 有 ik_result.position_error / rotation_error
        if allow_contact:
            for r in full_results:
                pe = getattr(r, "position_error", None)
                re = getattr(r, "rotation_error", None)
                if pe is None:
                    continue
                # pe shape: (n_links, batch_size)；按 batch 找 max(link)
                pe_per_batch = pe.max(dim=0).values if pe.dim() == 2 else pe
                re_per_batch = re.max(dim=0).values if (re is not None and re.dim() == 2) else (re if re is not None else None)
                for b in range(pe_per_batch.numel()):
                    pos_ok = pe_per_batch[b].item() < 0.02
                    rot_ok = (re_per_batch is None) or (re_per_batch[b].item() < 0.20)
                    if pos_ok and rot_ok:
                        if ctx is not None:
                            ctx.log(f"    [{stage_name}] curobo 报 fail 但接受 (pos_err={pe_per_batch[b].item()*1000:.1f}mm)")
                        # 替换 successes / 选这个 batch idx
                        successes[b] = True
                        idx_succ = b
                        success = True
                        break
                if success:
                    break

    if not success:
        if ctx is not None:
            r0 = full_results[0] if full_results else None
            status = getattr(r0, "status", None)
            pose_err = getattr(r0, "position_error", None)
            rot_err = getattr(r0, "rotation_error", None)
            attempts = getattr(r0, "attempts", None)
            ctx.log(f"    [{stage_name}] curobo 规划失败 successes={successes.tolist()} "
                    f"status={status} pos_err={pose_err} rot_err={rot_err} attempts={attempts}")
        for _ in range(3):
            yield world.empty_action()
        return float("inf")

    traj_path = traj_paths[idx_succ]

    # ik_only=True 时 traj_path 是 IKResult 的单个 JointState（只含 IK active joints），
    # 直接 get_full_js 会报 "lock_joints is also listed in self.joint_names"。
    # 参考官方 _ik_solver_cartesian_to_joint_space 的做法：get_full_js=False，
    # 再用当前完整 qpos 作为 base 把 IK 解 overwrite 进去。
    if allow_contact:
        # traj_path 是 IKResult.js_solution 整体 batch JointState
        # 拿 idx_succ 对应的解 → 按 joint name 写入 cur_q_full
        try:
            js_pos_full = traj_path.position
            js_names = list(traj_path.joint_names)
            if js_pos_full.dim() == 2:
                js_pos = js_pos_full[idx_succ].cpu().float()
            else:
                js_pos = js_pos_full.cpu().float()
        except Exception as e:
            if ctx is not None:
                ctx.log(f"    [{stage_name}] 取 IK 解失败: {e}")
            for _ in range(3):
                yield world.empty_action()
            return float("inf")
        cur_q_full = robot.get_joint_positions().to(js_pos.dtype).cpu()
        q_target_full = cur_q_full.clone()
        all_names = list(robot.joints.keys())
        body_lock_set = set(body_lock_idx)
        # 计算每个 joint name 是否被实际 overwrite
        overwritten = []
        skipped = []
        for i, jn in enumerate(js_names):
            if i >= len(js_pos):
                continue
            if jn in all_names:
                idx_full = all_names.index(jn)
                if lock_body and idx_full in body_lock_set:
                    skipped.append(jn)
                    continue
                q_target_full[idx_full] = js_pos[i]
                overwritten.append(jn)
            else:
                skipped.append(jn)
        if ctx is not None and stage_name in ("pre", "cnt"):
            ctx.log(f"    [{stage_name}] IK js_names ({len(js_names)} 个) "
                    f"overwrite={len(overwritten)} skip={len(skipped)} "
                    f"skipped_first3={skipped[:3]} ov_first3={overwritten[:3]}")
            # 验证：q_target_full 应用后 cuRobo FK 给出的 wrist 位置 vs IK target
            try:
                from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as _CES
                _emb = _CES.ARM
                _cu_js = type(traj_path)(
                    position=q_target_full.unsqueeze(0).cuda(),
                    joint_names=mg.robot_joint_names,
                ).get_ordered_joint_state(mg.mg[_emb].kinematics.joint_names)
                _ks = mg.mg[_emb].kinematics.compute_kinematics(_cu_js)
                _eef_link_name = robot.eef_link_names[arm]
                if _eef_link_name in _ks.link_poses:
                    _eef_pose = _ks.link_poses[_eef_link_name]
                    _eef_pos_local = _eef_pose.position[0].cpu().tolist()
                    ctx.log(f"    [{stage_name}] cuRobo FK on q_target ({_eef_link_name}) "
                            f"local={[round(x,3) for x in _eef_pos_local]}")
            except Exception as _fk_e:
                ctx.log(f"    [{stage_name}] cuRobo FK 验证失败: {_fk_e}")
        q_traj = th.stack([cur_q_full, q_target_full], dim=0)
    else:
        # plan_batch 路径：单条 JointState 轨迹 → (T, D)
        q_traj = mg.path_to_joint_trajectory(
            traj_path, get_full_js=True,
            emb_sel=CuRoboEmbodimentSelection.ARM,
        ).cpu().float()
        if q_traj.dim() == 1:
            q_traj = q_traj.unsqueeze(0)
        elif q_traj.dim() > 2:
            q_traj = q_traj.reshape(-1, q_traj.shape[-1])
    q_traj = mg.add_linearly_interpolated_waypoints(traj=q_traj, max_inter_dist=0.05)
    if lock_body and body_lock_idx:
        _freeze_body_on_trajectory(q_traj, body_lock_idx, body_lock_vals)
    n_steps = len(q_traj)

    if ctx is not None:
        # 诊断：q_traj[0] vs q_traj[-1] 差异，看 IK 是否真的改了关节
        try:
            q0 = q_traj[0]
            qN = q_traj[-1]
            d_full = (qN - q0).abs()
            d_max_idx = int(d_full.argmax().item())
            d_max_val = float(d_full.max().item())
            jn_max = list(robot.joints.keys())[d_max_idx] if d_max_idx < len(robot.joints) else "?"
            # 也 log 关键 arm 关节
            arm_idx_list = []
            try:
                arm_idx_list = _to_np(robot.arm_control_idx[arm]).astype(int).tolist()
            except Exception:
                pass
            arm_q0 = [round(q0[i].item(), 3) for i in arm_idx_list] if arm_idx_list else []
            arm_qN = [round(qN[i].item(), 3) for i in arm_idx_list] if arm_idx_list else []
            ctx.log(f"    [{stage_name}] q_traj {n_steps} wp; max_joint_change={d_max_val:.3f}rad on '{jn_max}'; "
                    f"{arm} arm: q0={arm_q0} → qN={arm_qN}")
        except Exception as _e:
            ctx.log(f"    [{stage_name}] q_traj 诊断异常: {_e}")
        ctx.log(f"    [{stage_name}] curobo 轨迹 {n_steps} waypoints, 开始执行")

    # ── 准备 finger override：让指定的 finger qpos 在每个 waypoint 都强制是 open/close ──
    joint_names = list(robot.joints.keys())
    finger_joint_idx_list: List[int] = []
    if arm in robot.finger_joints:
        for fj in robot.finger_joints[arm]:
            try:
                finger_joint_idx_list.append(joint_names.index(fj.joint_name))
            except ValueError:
                pass

    # 用于判断"是否到达 waypoint"的关节索引：lock_body 时只盯手臂，忽略躯干
    articulation_control_idx_list: List[int] = []
    # 总是等腰(trunk)到位：腰自由模式 cuRobo 会驱动腰，必须等它跟上；锁腰模式腰本就在目标。
    try:
        trunk_idx = _to_np(robot.trunk_control_idx).astype(int).tolist()
        articulation_control_idx_list.extend(trunk_idx)
    except Exception:
        pass
    for a in robot.arm_names:
        try:
            articulation_control_idx_list.extend(_to_np(robot.arm_control_idx[a]).astype(int).tolist())
        except Exception:
            pass
    art_thresh = (_M_LOW_PRECISION_JOINT_POS_DIFF_THRESHOLD if low_precision
                  else _M_JOINT_POS_DIFF_THRESHOLD)

    base_idx_list: List[int] = []
    try:
        base_idx_list = _to_np(robot.base_control_idx).astype(int).tolist()
    except Exception:
        pass

    # ── 按官方 _execute_motion_plan 的方式逐 waypoint 执行：每个 waypoint 反复
    # yield 同一 absolute joint target，直到 arm 跟到位（或达到 MAX_STEPS_PER_WAYPOINT）。──
    for i, joint_pos in enumerate(q_traj):
        joint_pos = joint_pos.clone().reshape(-1)
        if lock_body and body_lock_idx:
            joint_pos[body_lock_idx] = body_lock_vals
        # gripper override
        _apply_gripper_cmd_to_joint_pos(
            robot, joint_names, finger_joint_idx_list, joint_pos, gripper_cmd
        )

        base_target_reached = (len(base_idx_list) == 0)
        articulation_target_reached = False
        for j in range(_M_MAX_STEPS_PER_WAYPOINT):
            _assert_legacy_7dof_motion_ready(world, arm)
            action = _full_q_controller_action(world, joint_pos, arm, gripper_cmd)
            yield action

            current_joint_pos = robot.get_joint_positions()
            diff = joint_pos - current_joint_pos

            # articulation diff（trunk + arms）
            if articulation_control_idx_list:
                art_diff = diff[articulation_control_idx_list]
                max_art = float(th.max(th.abs(art_diff)).item())
                if max_art < art_thresh:
                    articulation_target_reached = True

            # base diff（HolonomicBase 用 wrap_angle 处理朝向）
            if base_idx_list:
                base_diff = diff[base_idx_list]
                max_base_pos = float(th.max(th.abs(base_diff[:2])).item()) if len(base_idx_list) >= 2 else 0.0
                max_base_orn = float(abs(_wrap_angle_py(float(base_diff[2].item())))) if len(base_idx_list) >= 3 else 0.0
                if max_base_pos < _M_DEFAULT_DIST_THRESHOLD and max_base_orn < _M_DEFAULT_ANGLE_THRESHOLD:
                    base_target_reached = True

            if base_target_reached and articulation_target_reached:
                break

        # 不抛 error，循环结束就当这个 waypoint 完成
        if not articulation_target_reached and ctx is not None and (i % 20 == 0 or i == n_steps - 1):
            ctx.log(f"    [{stage_name}] wp {i}/{n_steps} not fully reached "
                    f"max_art_diff>{art_thresh:.3f}")

    if ctx is not None:
        ctx.log(f"    [{stage_name}] curobo 执行完成 {n_steps} waypoints")
    return 0.0


def _plan_eef_motion_curobo(
    world, arm: str,
    target_world_pos, target_world_quat,
    *,
    lock_body: bool = True,
    attached_obj=None,
    ignore_objects=None,
    max_attempts: Optional[int] = None,
    timeout: float = 60.0,
    ctx=None,
    stage_name: str = "motion",
    emb_sel_override=None,
    diag_attrib_obj=None,
):
    """
    官方 `_plan_joint_motion` 模式：一次性把"当前位形 → 目标 eef pose"
    规划成一条无碰撞的全关节轨迹（读环境障碍 + trajopt 平滑）。

    与逐段 ik_only 不同：
      - ik_only=False           → 完整 motion planning（非两点线性插值）
      - ik_world_collision_check=True → 全轨迹世界避障
      - emb_sel=ARM             → 只规划手臂，底盘/躯干被锁（不推歪机器人）

    返回 q_traj (torch.Tensor, (T, D)) 或 None（规划失败，调用方自行回退）。
    """
    import math as _math
    import torch as th
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

    try:
        mg = _get_curobo_mg(world)
    except Exception as e:
        if ctx is not None:
            ctx.log(f"    [{stage_name}] curobo 初始化失败: {e}")
        return None

    robot = world.robot
    bs = int(mg.batch_size)
    eef_link = robot.eef_link_names[arm]
    pos_t = th.tensor(list(target_world_pos), dtype=th.float32)
    quat_t = th.tensor(list(target_world_quat), dtype=th.float32)
    target_pos = {eef_link: th.stack([pos_t for _ in range(bs)])}
    target_quat = {eef_link: th.stack([quat_t for _ in range(bs)])}
    # arm_no_torso 不锁另一只手臂：把它当前 eef pose 也设为目标，避免左臂被规划乱甩
    for a in robot.arm_names:
        if a == arm:
            continue
        other_link = robot.eef_link_names[a]
        ep = world.eef_pose(arm=a)
        op = th.tensor(list(ep["pos"]), dtype=th.float32)
        oq = th.tensor(list(ep["quat"]), dtype=th.float32)
        target_pos[other_link] = th.stack([op for _ in range(bs)])
        target_quat[other_link] = th.stack([oq for _ in range(bs)])

    # 每次规划前刷新场景障碍（读环境做避障）；ignore_objects 把抓取目标排除，
    # 让 cuRobo 能贴近目标又严格避开桌子/其它物体（官方抓取做法）。
    _maybe_update_obstacles(mg, ctx=ctx, force=True, ttl_s=0.0, ignore_objects=ignore_objects)

    if max_attempts is None:
        max_attempts = _math.ceil(100 / bs)

    # 诊断：检查起始状态是否已在碰撞中（若是则规划必然失败）
    try:
        import torch as _th
        _q_start = robot.get_joint_positions()
        _coll = mg.check_collisions(_q_start.unsqueeze(0), skip_obstacle_update=True)
        if _coll.any():
            if ctx is not None:
                ctx.log(f"    [{stage_name}] WARN 起始关节状态在碰撞中 (n={_coll.sum().item()}) → cuRobo 将无法规划任何轨迹")
        else:
            if ctx is not None:
                ctx.log(f"    [{stage_name}] 起始状态无碰撞，cuRobo 可规划")
    except Exception as _e:
        if ctx is not None:
            ctx.log(f"    [{stage_name}] 碰撞检查失败: {_e}")

    try:
        successes, traj_paths = mg.compute_trajectories(
            target_pos=target_pos,
            target_quat=target_quat,
            initial_joint_pos=None,
            is_local=False,
            max_attempts=max_attempts,
            timeout=timeout,
            ik_fail_return=50,
            enable_finetune_trajopt=True,
            finetune_attempts=1,
            return_full_result=False,
            success_ratio=1.0 / bs,
            attached_obj=attached_obj,
            attached_obj_scale=None,
            motion_constraint=None,
            skip_obstacle_update=True,
            ik_only=False,
            ik_world_collision_check=True,
            emb_sel=emb_sel_override or CuRoboEmbodimentSelection.ARM,
        )
    except Exception as e:
        if ctx is not None:
            ctx.log(f"    [{stage_name}] curobo motion plan 异常: {type(e).__name__}: {e}")
        return None

    try:
        idx = th.where(successes)[0].cpu()
    except Exception:
        idx = []
    if len(idx) == 0:
        if ctx is not None:
            ctx.log(f"    [{stage_name}] curobo motion plan 失败：无可达无碰撞路径")
            # 纯 IK 探针：关掉世界碰撞，看锁腰下该 pose 到底"够不够得到"
            try:
                probe = mg.compute_trajectories(
                    target_pos=target_pos, target_quat=target_quat,
                    initial_joint_pos=None, is_local=False,
                    max_attempts=max_attempts, timeout=timeout, ik_fail_return=50,
                    enable_finetune_trajopt=False, finetune_attempts=0,
                    return_full_result=True, success_ratio=1.0 / bs,
                    attached_obj=None, attached_obj_scale=None, motion_constraint=None,
                    skip_obstacle_update=True, ik_only=True,
                    ik_world_collision_check=False,
                    emb_sel=CuRoboEmbodimentSelection.ARM,
                )
                r0 = probe[0] if isinstance(probe, (list, tuple)) and probe else probe
                pe = getattr(r0, "position_error", None)
                re_ = getattr(r0, "rotation_error", None)
                st = getattr(r0, "status", None)
                pe_v = float(pe.max().item()) if pe is not None else float("nan")
                re_v = float(re_.max().item()) if re_ is not None else float("nan")
                if pe_v < 0.03 and (re_v != re_v or re_v < 0.25):
                    ctx.log(f"    [{stage_name}] 探针1：纯IK可达 (pos_err={pe_v*1000:.0f}mm "
                            f"rot_err={re_v:.2f}) → 运动学够得到，根因在【世界碰撞】")
                    # 探针2：带世界碰撞的 IK —— 判断"目标位形本身"是否存在无碰撞解
                    #   有解 → 目标无碰撞，是【起点↔目标的路径】被挡（trajopt/graph 没连上）
                    #   无解 → 目标位形【本身嵌在障碍里】（夹爪/手腕穿桌面或物体）→ 必须换 pose
                    try:
                        succ2, _paths2 = mg.compute_trajectories(
                            target_pos=target_pos, target_quat=target_quat,
                            initial_joint_pos=None, is_local=False,
                            max_attempts=max_attempts, timeout=timeout, ik_fail_return=50,
                            enable_finetune_trajopt=False, finetune_attempts=0,
                            return_full_result=False, success_ratio=1.0 / bs,
                            attached_obj=None, attached_obj_scale=None, motion_constraint=None,
                            skip_obstacle_update=True, ik_only=True,
                            ik_world_collision_check=True,
                            emb_sel=emb_sel_override or CuRoboEmbodimentSelection.ARM,
                        )
                        ok2 = bool(succ2.any().item()) if succ2 is not None else False
                        if ok2:
                            ctx.log(f"    [{stage_name}] 探针2：目标位形【有】无碰撞IK解 "
                                    f"→ 根因是【起点→目标路径被挡】（提高 max_attempts / 改起手姿势 / 加 graph）")
                        else:
                            ctx.log(f"    [{stage_name}] 探针2：目标位形【无】任何无碰撞IK解 "
                                    f"→ 根因是【该抓取位姿本身嵌在障碍里】（夹爪/手腕穿桌面或物体）→ 必须换 pose/back_m")
                            # 探针3 归因：排除目标物体后重测，定位元凶是"目标物体本身"还是"台面/其它"
                            if diag_attrib_obj is not None:
                                try:
                                    _maybe_update_obstacles(mg, ctx=None, force=True,
                                                            ttl_s=0.0, ignore_objects=[diag_attrib_obj])
                                    succ3, _p3 = mg.compute_trajectories(
                                        target_pos=target_pos, target_quat=target_quat,
                                        initial_joint_pos=None, is_local=False,
                                        max_attempts=max_attempts, timeout=timeout, ik_fail_return=50,
                                        enable_finetune_trajopt=False, finetune_attempts=0,
                                        return_full_result=False, success_ratio=1.0 / bs,
                                        attached_obj=None, attached_obj_scale=None, motion_constraint=None,
                                        skip_obstacle_update=True, ik_only=True,
                                        ik_world_collision_check=True,
                                        emb_sel=emb_sel_override or CuRoboEmbodimentSelection.ARM,
                                    )
                                    ok3 = bool(succ3.any().item()) if succ3 is not None else False
                                    if ok3:
                                        ctx.log(f"    [{stage_name}] 探针3：排除目标物体后【有】无碰撞IK解 "
                                                f"→ 元凶=【目标物体本身】（夹爪实体与袋体几何重叠，即体积指标选了穿模 pose）")
                                    else:
                                        ctx.log(f"    [{stage_name}] 探针3：排除目标物体后仍【无】解 "
                                                f"→ 元凶=【台面/其它障碍】（非目标物体）")
                                except Exception as _pe3:
                                    ctx.log(f"    [{stage_name}] IK探针3异常: {_pe3}")
                                finally:
                                    # 恢复障碍（含目标），不影响后续 fallback
                                    _maybe_update_obstacles(mg, ctx=None, force=True,
                                                            ttl_s=0.0, ignore_objects=None)
                    except Exception as _pe2:
                        ctx.log(f"    [{stage_name}] IK探针2异常: {_pe2}")
                else:
                    ctx.log(f"    [{stage_name}] 探针1：纯IK也不可达 (pos_err={pe_v*1000:.0f}mm "
                            f"rot_err={re_v:.2f} status={st}) → 锁腰下该 pose【够不到】")
            except Exception as _pe:
                ctx.log(f"    [{stage_name}] IK 探针异常: {_pe}")
        return None

    try:
        q_traj = mg.path_to_joint_trajectory(
            traj_paths[int(idx[0])], get_full_js=True,
            emb_sel=CuRoboEmbodimentSelection.ARM,
        ).cpu().float()
    except Exception as e:
        if ctx is not None:
            ctx.log(f"    [{stage_name}] 轨迹转换失败: {e}")
        return None
    if q_traj.dim() == 1:
        q_traj = q_traj.unsqueeze(0)
    elif q_traj.dim() > 2:
        q_traj = q_traj.reshape(-1, q_traj.shape[-1])
    # 0.01 太密会插出上百个 waypoint、逐点收敛极慢；0.03 足够平滑且大幅减少仿真步数。
    q_traj = mg.add_linearly_interpolated_waypoints(traj=q_traj, max_inter_dist=0.03)

    if lock_body:
        lock_idx = _body_lock_joint_indices(robot)
        if lock_idx:
            lock_vals = robot.get_joint_positions().cpu()[lock_idx].clone()
            _freeze_body_on_trajectory(q_traj, lock_idx, lock_vals)

    if ctx is not None:
        ctx.log(
            f"    [{stage_name}] motion plan ok：{len(q_traj)} waypoints "
            f"（全轨迹避障, emb=ARM, lock_body={lock_body}）"
        )
    return q_traj


def _execute_curobo_q_traj(
    world, arm: str, q_traj,
    *,
    gripper_cmd: Optional[float] = None,
    lock_body: bool = False,
    low_precision: bool = False,
    ctx=None,
    stage_name: str = "motion",
):
    """
    逐 waypoint 执行 cuRobo 关节轨迹（官方 `_execute_motion_plan` 模式）。
    lock_body=True 时底盘/躯干钉死在执行起点，仅手臂运动。生成器，yield action。
    """
    import torch as th
    from behavior_interface.skills.eef import (
        _assert_legacy_7dof_motion_ready,
        _prepare_legacy_7dof_motion,
    )

    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    robot = world.robot
    joint_names = list(robot.joints.keys())

    body_lock_idx = _body_lock_joint_indices(robot) if lock_body else []
    body_lock_vals = None
    if body_lock_idx:
        body_lock_vals = robot.get_joint_positions().cpu()[body_lock_idx].clone()

    finger_joint_idx_list: List[int] = []
    if arm in robot.finger_joints:
        for fj in robot.finger_joints[arm]:
            try:
                finger_joint_idx_list.append(joint_names.index(fj.joint_name))
            except ValueError:
                pass

    # 到位判定：lock_body 时只盯手臂（忽略躯干/底盘）
    articulation_control_idx_list: List[int] = []
    # 总是等腰(trunk)到位：腰自由模式 cuRobo 会驱动腰，必须等它跟上；锁腰模式腰本就在目标。
    try:
        articulation_control_idx_list.extend(_to_np(robot.trunk_control_idx).astype(int).tolist())
    except Exception:
        pass
    for a in robot.arm_names:
        try:
            articulation_control_idx_list.extend(_to_np(robot.arm_control_idx[a]).astype(int).tolist())
        except Exception:
            pass
    art_thresh = (_M_LOW_PRECISION_JOINT_POS_DIFF_THRESHOLD if low_precision
                  else _M_JOINT_POS_DIFF_THRESHOLD)

    base_idx_list: List[int] = []
    if not lock_body:
        try:
            base_idx_list = _to_np(robot.base_control_idx).astype(int).tolist()
        except Exception:
            pass

    n_steps = len(q_traj)
    for i, joint_pos in enumerate(q_traj):
        joint_pos = joint_pos.clone().reshape(-1)
        if lock_body and body_lock_idx:
            joint_pos[body_lock_idx] = body_lock_vals
        _apply_gripper_cmd_to_joint_pos(
            robot, joint_names, finger_joint_idx_list, joint_pos, gripper_cmd
        )

        base_target_reached = (len(base_idx_list) == 0)
        articulation_target_reached = False
        for _j in range(_M_MAX_STEPS_PER_WAYPOINT):
            _assert_legacy_7dof_motion_ready(world, arm)
            action = _full_q_controller_action(world, joint_pos, arm, gripper_cmd)
            yield action

            diff = joint_pos - robot.get_joint_positions()
            if articulation_control_idx_list:
                max_art = float(th.max(th.abs(diff[articulation_control_idx_list])).item())
                if max_art < art_thresh:
                    articulation_target_reached = True
            if base_idx_list:
                base_diff = diff[base_idx_list]
                max_base_pos = float(th.max(th.abs(base_diff[:2])).item()) if len(base_idx_list) >= 2 else 0.0
                max_base_orn = float(abs(_wrap_angle_py(float(base_diff[2].item())))) if len(base_idx_list) >= 3 else 0.0
                if max_base_pos < _M_DEFAULT_DIST_THRESHOLD and max_base_orn < _M_DEFAULT_ANGLE_THRESHOLD:
                    base_target_reached = True
            if base_target_reached and articulation_target_reached:
                break

    if ctx is not None:
        ctx.log(f"    [{stage_name}] 全轨迹执行完成 {n_steps} waypoints")


def _execute_arm_q_traj(
    world, arm: str, q_traj,
    *,
    gripper_cmd: Optional[float] = None,
    low_precision: bool = True,
    ctx=None,
    stage_name: str = "motion",
):
    """make_action 风格逐 waypoint 执行手臂关节轨迹（推荐用于 grasp_obj）。

    只发 arm_{arm}（绝对关节位）+ gripper_{arm}；底盘/躯干/另一条手臂由
    make_action 的 no-op 基底保持：base 速度=0（HolonomicBase velocity 控制器），
    trunk / 另一臂维持当前 qpos。

    相比 robot.q_to_action(full_q)：后者会给 velocity 控制的底盘算出非零速度命令，
    导致底盘漂移 / 被推（正是"机器人没动/被推歪"的根因）。这里彻底不碰底盘。
    生成器，yield action。
    """
    from behavior_interface.skills.eef import (
        _assert_legacy_7dof_motion_ready,
        _prepare_legacy_7dof_motion,
    )

    _prepare_legacy_7dof_motion(
        world, arm, ctx=ctx, stage_name=f"{stage_name}.prepare"
    )
    robot = world.robot
    joint_names = list(robot.joints.keys())
    try:
        arm_joint_idx = [joint_names.index(f"{arm}_arm_joint{i + 1}") for i in range(7)]
    except ValueError as e:
        if ctx is not None:
            ctx.log(f"    [{stage_name}] 找不到 {arm} arm joint: {e}")
        return

    art_thresh = (_M_LOW_PRECISION_JOINT_POS_DIFF_THRESHOLD if low_precision
                  else _M_JOINT_POS_DIFF_THRESHOLD)
    n_steps = len(q_traj)
    for _i, jp in enumerate(q_traj):
        jp = jp.reshape(-1)
        arm_q = [float(jp[k]) for k in arm_joint_idx]
        overrides_base = {f"arm_{arm}": arm_q}
        grip_override = _gripper_cmd_override(gripper_cmd)
        if grip_override is not None:
            overrides_base[f"gripper_{arm}"] = grip_override
        for _j in range(_M_MAX_STEPS_PER_WAYPOINT):
            _assert_legacy_7dof_motion_ready(world, arm)
            yield world.make_action(**overrides_base)
            cur = robot.get_joint_positions()
            max_d = max(
                abs(arm_q[k2] - float(cur[arm_joint_idx[k2]])) for k2 in range(7)
            )
            if max_d < art_thresh:
                break

    if ctx is not None:
        ctx.log(
            f"    [{stage_name}] 手臂轨迹执行完成 {n_steps} waypoints"
            f"（make_action, base/trunk 不动）"
        )


def _wrap_angle_py(a: float) -> float:
    """把角度 wrap 到 [-pi, pi]。"""
    import math as _math
    a = (a + _math.pi) % (2 * _math.pi) - _math.pi
    return a


def _hold_pose(world, arm: str, _unused_target_pos, _unused_target_quat,
               n_frames: int, gripper_cmd: Optional[float] = None):
    """保持 arm 不动：absolute 模式下要发当前 qpos 作为 target（不是 [0]*n）。
    用于 settle / 闭夹爪期间。"""
    if world.dry_run:
        for _ in range(n_frames):
            overrides: Dict[str, Any] = {f"arm_{arm}": [0.0] * 7}
            grip_override = _gripper_cmd_override(gripper_cmd)
            if grip_override is not None:
                overrides[f"gripper_{arm}"] = grip_override
            yield world.make_action(**overrides)
        return
    # 一次性读当前 qpos，整段 hold 期间锁定到这个目标
    q_hold = _arm_qpos(world, arm).tolist()
    for _ in range(n_frames):
        overrides = {f"arm_{arm}": q_hold}
        grip_override = _gripper_cmd_override(gripper_cmd)
        if grip_override is not None:
            overrides[f"gripper_{arm}"] = grip_override
        yield world.make_action(**overrides)


# ─────────────────────────────────────────────────────────────────────────────
# Skill: execute_grasp
# ─────────────────────────────────────────────────────────────────────────────

# 默认 stage 步数 / 容差
_STAGE_MAX_STEPS  = 150     # 每个 stage 最多 150 步
_STAGE_PRE_TOL    = 0.04    # pre-grasp 容差（远点宽松）
_STAGE_GRASP_TOL  = 0.025   # contact 容差（精度更高）
_STAGE_LIFT_TOL   = 0.025
_GRIP_SETTLE_FR   = 30      # 闭夹爪期间维持 arm pose 的帧数
_OPEN_FR          = 8       # 开夹爪 settle 帧数


def _read_obj_z(obj) -> Optional[float]:
    if obj is None:
        return None
    try:
        p, _ = obj.get_position_orientation()
        return float(_to_np(p)[2])
    except Exception:
        return None


def _execute_single_grasp(ctx, last, grasp: Dict[str, Any], arm_arg: str,
                          lift_height: float, lift_check_threshold: float):
    """单个 grasp 的完整执行流程（生成器，yield action）。

    生成器 return 一个 dict（用 StopIteration.value 取回）。
    """
    world = ctx.world
    arm_eff = arm_arg or grasp.get("arm") or "right"
    target_pos = np.array(grasp["pos"], dtype=np.float64)
    approach   = np.array(grasp["approach"], dtype=np.float64)
    norm = float(np.linalg.norm(approach))
    if norm > 1e-9:
        approach = approach / norm

    # 两阶段 contact 设计：
    #   contact_pos:   target_pos 沿 approach 方向略入物体（5cm），保留 target_pos
    #                  的 xy 偏移（确保 5 个不同候选 contact 真的不同）
    #   grip_push_pos: 沿 approach 方向再多入 8cm（强行让 finger 进物体范围）
    # 物理阻挡时 IK stuck，close gripper command 让 finger 关闭夹住物体。
    contact_pos   = target_pos + 0.05 * approach
    grip_push_pos = target_pos + 0.13 * approach

    # pre-grasp 沿 -approach 方向偏 15cm（更宽裕，IK 容易解）
    pre_grasp_pos = target_pos - 0.15 * approach
    # lift 从 contact 位置沿世界 +Z 抬 lift_height
    lift_target_pos = contact_pos + np.array([0.0, 0.0, lift_height])

    # 解析物体 + 起始 z
    obj = _resolve_object_handle(world, last["object"].get("resolved_name")
                                  or last["object"].get("input"))
    obj_z0 = _read_obj_z(obj)

    eef0 = world.eef_pose(arm=arm_eff)
    ctx.log(
        f"── grasp #{grasp.get('id')} label={grasp.get('label')} arm={arm_eff} "
        f"target=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) "
        f"obj_z0={obj_z0}"
    )
    ctx.log(f"  eef0=({eef0['pos'][0]:.3f},{eef0['pos'][1]:.3f},{eef0['pos'][2]:.3f})")

    # 锁定一个 grasp ori（保持当前 eef 朝向；IK 在 nullspace 内移动）
    grasp_ori = np.asarray(eef0["quat"], dtype=np.float64)

    # ── Stage A：打开夹爪 ──
    from behavior_interface.skills.eef import _gripper_limit_cmd

    open_cmd = _gripper_limit_cmd(world, arm_eff, open_gripper=True)
    for _ in range(_OPEN_FR):
        yield world.make_action(**{f"gripper_{arm_eff}": open_cmd})

    # ── Stage B：eef 收敛到 pre-grasp ──
    ctx.log(f"  → pre-grasp ({pre_grasp_pos[0]:.3f},{pre_grasp_pos[1]:.3f},{pre_grasp_pos[2]:.3f})")
    pre_err = yield from _eef_goto_world(
        world, arm_eff, pre_grasp_pos, target_world_quat=grasp_ori,
        max_steps=_STAGE_MAX_STEPS, pos_tol=_STAGE_PRE_TOL,
        gripper_cmd=+1.0, ctx=ctx, stage_name="pre",
    )

    # ── Stage C：推到 grasp 点（沿 approach 略微扎入 + 朝物体中心，确保 gripper 接触）──
    ctx.log(f"  → contact ({contact_pos[0]:.3f},{contact_pos[1]:.3f},{contact_pos[2]:.3f})")
    grasp_err = yield from _eef_goto_world(
        world, arm_eff, contact_pos, target_world_quat=grasp_ori,
        max_steps=_STAGE_MAX_STEPS, pos_tol=_STAGE_GRASP_TOL,
        gripper_cmd=+1.0, ctx=ctx, stage_name="cnt",
    )

    # 诊断：打印 finger link world pos 和 obj_center 距离
    try:
        robot = world.robot
        finger_names = robot.finger_link_names.get(arm_eff, [])
        eef_now = world.eef_pose(arm=arm_eff)
        obj_center_now = None
        # 重新查物体 AABB 用作诊断（contact_pos 计算改用 target_pos，
        # 这里只是 log 用）
        _obj_dbg = _resolve_object_handle(world, last["object"].get("resolved_name")
                                          or last["object"].get("input"))
        if _obj_dbg is not None:
            lo_dbg, hi_dbg = _aabb_of(_obj_dbg)  # type: ignore
            obj_center_now = (lo_dbg + hi_dbg) / 2.0
        for fn in finger_names:
            fl = robot.links.get(fn)
            if fl is None:
                continue
            fp = _to_np(fl.get_position_orientation()[0])
            offset_from_eef = fp - np.asarray(eef_now["pos"], dtype=np.float64)
            d_to_obj = float(np.linalg.norm(fp - obj_center_now)) if obj_center_now is not None else -1.0
            ctx.log(
                f"  DIAG finger={fn} world=({fp[0]:.3f},{fp[1]:.3f},{fp[2]:.3f}) "
                f"offset_from_eef=({offset_from_eef[0]:.3f},{offset_from_eef[1]:.3f},{offset_from_eef[2]:.3f}) "
                f"dist_to_obj_center={d_to_obj:.3f}m"
            )
    except Exception as e:
        ctx.log(f"  DIAG finger probe failed: {e}")

    # ── Stage D：边关夹爪边继续 push 到 contact_pos（禁用 stuck 检测）──
    # sticky grasping 需要 gripper link 与物体物理接触 + close gripper command。
    # eef stuck 时 finger 可能还没接触，所以这里持续 push + close gripper，
    # 让物理仿真中物体被 gripper 关闭过程"夹住"。
    ctx.log(f"  closing gripper while pushing to grip_push_pos "
            f"({grip_push_pos[0]:.3f},{grip_push_pos[1]:.3f},{grip_push_pos[2]:.3f})...")
    # grip 前读 finger qpos（用于验证 close 是否生效）
    finger_q_before = _read_finger_qpos(world, arm_eff)
    yield from _eef_goto_world(
        world, arm_eff, grip_push_pos, target_world_quat=grasp_ori,
        max_steps=_GRIP_SETTLE_FR, pos_tol=0.0,
        gripper_cmd=-1.0, ctx=ctx, stage_name="grip",
        disable_stuck_check=True,
    )
    yield from _hold_pose(world, arm_eff, grip_push_pos, grasp_ori,
                          n_frames=15, gripper_cmd=-1.0)
    try:
        from behavior_interface.skills.eef import _official_grasp_active

        if _official_grasp_active(world, arm_eff):
            latch_keepalive = getattr(world, "latch_gripper_close_keepalive", None)
            if callable(latch_keepalive):
                latch_keepalive(arm_eff)
    except Exception:
        pass
    finger_q_after = _read_finger_qpos(world, arm_eff)
    ctx.log(
        f"  finger qpos (close cmd=-1): before={finger_q_before} after={finger_q_after} "
        f"(0=fully closed, 0.05=fully open)"
    )

    # ── Stage E：抬起 ──
    ctx.log(f"  → lift z={lift_target_pos[2]:.3f}")
    lift_err = yield from _eef_goto_world(
        world, arm_eff, lift_target_pos, target_world_quat=grasp_ori,
        max_steps=_STAGE_MAX_STEPS, pos_tol=_STAGE_LIFT_TOL,
        gripper_cmd=-1.0, ctx=ctx, stage_name="lift",
    )

    # Settle 几帧让物体稳定
    yield from _hold_pose(world, arm_eff, lift_target_pos, grasp_ori,
                          n_frames=8, gripper_cmd=-1.0)

    # ── 验证 ──
    obj_z1 = _read_obj_z(obj)
    grasped = (obj_z0 is not None and obj_z1 is not None
               and (obj_z1 - obj_z0) > lift_check_threshold)
    dz_obj  = (obj_z1 - obj_z0) if (obj_z0 is not None and obj_z1 is not None) else None
    ctx.log(
        f"  result: obj_z {obj_z0} -> {obj_z1} Δ={dz_obj} grasped={grasped}"
        f" | pre_err={pre_err:.3f} grasp_err={grasp_err:.3f} lift_err={lift_err:.3f}"
    )

    return {
        "ok": True,
        "grasp_id": grasp.get("id"),
        "label": grasp.get("label"),
        "arm": arm_eff,
        "target_pos": target_pos.tolist(),
        "contact_pos": contact_pos.tolist(),
        "reachable": True,
        "pre_grasp_err": float(pre_err),
        "grasp_pt_err": float(grasp_err),
        "lift_err": float(lift_err),
        "object_z_before": obj_z0,
        "object_z_after": obj_z1,
        "object_dz": dz_obj,
        "grasped": bool(grasped),
    }


def _release_arm_to_home(world, arm: str, last_lift_pos: np.ndarray,
                         grasp_ori: np.ndarray, ctx=None):
    """release：开夹爪、把 arm 抬回更高 / 远离物体的位置。
    给 grasp_id=-1 模式做 inter-grasp reset 用。
    """
    # 开夹爪
    from behavior_interface.skills.eef import _gripper_limit_cmd

    open_cmd = _gripper_limit_cmd(world, arm, open_gripper=True)
    for _ in range(_OPEN_FR):
        yield world.make_action(**{f"gripper_{arm}": open_cmd})
    # arm 再抬高 10cm + 后退 10cm（避免下一次 grasp 蹭到物体）
    safe_pos = last_lift_pos + np.array([0.0, 0.0, 0.10])
    yield from _eef_goto_world(
        world, arm, safe_pos, target_world_quat=grasp_ori,
        max_steps=80, pos_tol=0.05, gripper_cmd=+1.0, ctx=ctx, stage_name="release",
    )


@register_skill(
    "execute_grasp",
    description=(
        "执行上一次 get_grasp_position 选出来的某个 grasp。"
        "grasp_id=0 = 默认抓第 0 个；grasp_id=-1 = 遍历所有 reachable 候选逐个执行"
        "（每次抓完无论成败都 release，便于一次性验证）。"
        "若 reachable：开夹爪 → eef 收敛到 pre-grasp → 推到 grasp → 闭夹爪 → 提起 lift_height "
        "→ 检查物体 z 变化；若不 reachable：在 free_region 里规划 base pose 写入 result，"
        "用户后续 move_to 过去再 retry。"
    ),
)
def execute_grasp(
    ctx,
    grasp_id: int = 0,
    arm: str = "",
    lift_height: float = 0.15,
    lift_check_threshold: float = 0.04,
):
    """yield 每步 action 到 sim。"""
    world = ctx.world

    # 1) 读上一次 get_grasp_position 的 result
    last = ctx.get_last_result("get_grasp_position")
    if last is None or not last.get("ok"):
        msg = "请先成功执行 get_grasp_position 获得候选"
        ctx.log(msg)
        ctx.set_result({"ok": False, "error": msg})
        yield world.empty_action()
        return

    cands = last["candidates"]

    # ─────────────── grasp_id = -1：遍历所有 reachable ───────────────
    # 为了一次性验证"每个 reachable grasp 都能真的抓住"，本模式：
    #   1. 记录物体初始 world pose
    #   2. 用初始 AABB sample candidates，对当前 robot 评 reachable
    #   3. 对每个 reachable 候选依次执行 grasp
    #   4. 每轮抓完先 release（开夹爪 + 抬手），再把**物体 set 回初始 pose**，
    #      让下一轮的 grasp 候选位置 / reachable 与第一次完全一致
    if grasp_id < 0:
        if _challenge_action_only_enabled():
            ctx.set_result({
                "ok": False,
                "error": (
                    "评测模式禁用 grasp_id<0：该旧遍历模式会在多轮之间直接"
                    "回写物体位姿。请逐个执行候选，并用正式 scene reset"
                    "开始下一轮。"
                ),
            })
            yield world.empty_action()
            return
        obj_name = (last.get("object", {}).get("input")
                    or last.get("object", {}).get("resolved_name"))
        if not obj_name:
            msg = "上次 get_grasp_position 的 object 名缺失"
            ctx.log(msg); ctx.set_result({"ok": False, "error": msg})
            yield world.empty_action()
            return
        arm_eff_default = arm or last.get("arm") or "right"

        obj = _resolve_object_handle(world, obj_name)
        if obj is None:
            msg = f"找不到物体 {obj_name}"
            ctx.log(msg); ctx.set_result({"ok": False, "error": msg})
            yield world.empty_action()
            return
        # 记录初始 pose（用于每轮 reset）
        try:
            init_pos_t, init_quat_t = obj.get_position_orientation()
            init_pos  = _to_np(init_pos_t).reshape(3)
            init_quat = _to_np(init_quat_t).reshape(4)
        except Exception:
            init_pos, init_quat = None, None
            ctx.log("无法读取物体初始 pose，遍历模式可能不准确")

        # 用初始 AABB sample 一组 candidates 并算 reachable
        aabb = _aabb_of(obj)
        if aabb is None:
            msg = f"物体 {obj_name} 无 AABB"
            ctx.log(msg); ctx.set_result({"ok": False, "error": msg})
            yield world.empty_action()
            return
        lo, hi = aabb
        raw = _sample_grasp_candidates(lo, hi, n_top=8, n_side=16)
        for c in raw:
            ok, why = _is_reachable(world, c, arm=arm_eff_default)
            c["reachable"] = ok
            c["reach_reason"] = why
            c["arm"] = arm_eff_default
            c["score"] = (1.0 if ok else 0.3) + (0.1 if c["label"].startswith("top") else 0)
        raw.sort(key=lambda x: (not x["reachable"], -x["score"]))
        reach = [c for c in raw if c["reachable"]]
        if not reach:
            # 没 reachable —— 也给个 base pose 建议，方便用户直接 move_to
            anchor = raw[0] if raw else None
            bp = _suggest_base_pose(world, anchor, arm=arm_eff_default) if anchor else None
            result_payload: Dict[str, Any] = {
                "ok": False,
                "error": "当前没有 reachable 候选，先 move_to 到合适位置再试",
                "n_tried": 0,
            }
            if bp is not None:
                result_payload["suggested_base_pose"] = bp
                result_payload["next_action"] = (
                    f"move_to(x={bp['x']:.3f}, y={bp['y']:.3f}, z={bp['z']:.3f}, "
                    f"theta_x_deg={bp['theta_x_deg']:.1f}, theta_z_deg={bp['theta_z_deg']:.1f})"
                )
                ctx.log(
                    f"  → 没 reachable，建议 {result_payload['next_action']} "
                    f"[grasp_height={bp.get('grasp_height_class')}]"
                )
            else:
                ctx.log("  → 没 reachable，且 free_region 没找到合适 base pose")
            ctx.set_result(result_payload)
            yield world.empty_action()
            return
        # FPS 选 3-5 个最分散的
        k = min(5, len(reach))
        idxs = _farthest_point_sampling(reach, k)
        chosen = [reach[i] for i in idxs]
        for i, c in enumerate(chosen):
            c["id"] = i

        ctx.log(f"=== 遍历模式：将执行 {len(chosen)} 个 reachable 候选，"
                f"每轮抓完把物体重置回初始 pose ===")
        results: List[Dict[str, Any]] = []
        for round_i, c in enumerate(chosen):
            ctx.log(f"\n[round {round_i+1}/{len(chosen)}] grasp #{c['id']} label={c['label']}")

            # 每轮开始前 reset 物体到 initial pose（除了第 0 轮，原本就在那）
            if round_i > 0 and init_pos is not None:
                try:
                    import torch as _th
                    obj.set_position_orientation(
                        position=_th.tensor(init_pos, dtype=_th.float32),
                        orientation=_th.tensor(init_quat, dtype=_th.float32),
                    )
                    ctx.log(f"  物体已 reset 到 ({init_pos[0]:.3f},{init_pos[1]:.3f},"
                            f"{init_pos[2]:.3f})")
                except Exception as e:
                    ctx.log(f"  物体 reset 失败: {e}")

                # 等几帧让物理稳定
                for _ in range(20):
                    yield world.empty_action()

            r = yield from _execute_single_grasp(
                ctx, {"object": last["object"]}, c, arm,
                lift_height, lift_check_threshold,
            )
            results.append(r)

            # release：开夹爪 + 抬手让物体掉下
            arm_eff = r["arm"]
            grasp_ori = np.asarray(world.eef_pose(arm=arm_eff)["quat"],
                                   dtype=np.float64)
            yield from _release_arm_to_home(
                world, arm_eff,
                last_lift_pos=np.array(c["pos"]) + np.array([0, 0, lift_height]),
                grasp_ori=grasp_ori, ctx=ctx,
            )

        n_success = sum(1 for r in results if r.get("grasped"))
        ctx.log(f"\n=== 遍历完成：{n_success}/{len(results)} 个 grasp 成功 ===")
        ctx.set_result({
            "ok": True,
            "mode": "all_reachable",
            "n_tried": len(results),
            "n_success": n_success,
            "per_grasp": results,
        })
        yield world.empty_action()
        return

    # ─────────────── 单个 grasp 模式 ───────────────
    if grasp_id >= len(cands):
        msg = f"grasp_id={grasp_id} 越界（共 {len(cands)} 个候选）"
        ctx.log(msg)
        ctx.set_result({"ok": False, "error": msg})
        yield world.empty_action()
        return

    grasp = cands[grasp_id]
    arm_eff = arm or grasp.get("arm") or "right"

    # 2) 当前是否 reachable（机器人可能已经动过，需要重判）
    ok, why = _is_reachable(world, grasp, arm=arm_eff)
    if not ok:
        ctx.log(f"execute_grasp NOT reachable: {why}；尝试规划 base pose")
        bp = _suggest_base_pose(world, grasp, arm=arm_eff)
        if bp is None:
            msg = "在 free_region 里找不到合适的 base pose"
            ctx.log(msg)
            ctx.set_result({"ok": False, "error": msg, "reachable": False})
        else:
            next_action = (
                f"move_to(x={bp['x']:.3f}, y={bp['y']:.3f}, z={bp['z']:.3f}, "
                f"theta_x_deg={bp['theta_x_deg']:.1f}, theta_z_deg={bp['theta_z_deg']:.1f})"
            )
            ctx.log(
                f"建议 5D chest pose: ({bp['x']:.2f},{bp['y']:.2f},{bp['z']:.2f}) "
                f"theta=({bp['theta_x_deg']:+.1f}°,{bp['theta_z_deg']:.1f}°)  "
                f"→ {next_action}"
            )
            ctx.set_result({
                "ok": True,
                "reachable": False,
                "reason": why,
                "suggested_base_pose": bp,
                "next_action": next_action,
                "grasp_used": grasp,
            })
        yield world.empty_action()
        return

    # 3) 执行
    r = yield from _execute_single_grasp(
        ctx, last, grasp, arm, lift_height, lift_check_threshold,
    )
    ctx.set_result(r)
    for _ in range(3):
        yield world.empty_action()


@register_skill(
    "diag_gripper_frame",
    description="诊断：dump 真实夹爪 link/手指在 EEF 局部系坐标，与 plan_grasp_gripper_fit 的几何常量对比，验证 +Z/-Z 约定是否一致。",
)
def diag_gripper_frame(ctx, arm: str = "right"):
    """对比真机夹爪几何 vs gripper-fit 内部模型（仅日志，不动机器人）。"""
    import numpy as np
    from behavior_interface.skills import plan_grasp_gripper_fit as gf

    world = ctx.world
    robot = world.robot

    eef = world.eef_pose(arm=arm)
    eef_pos = np.asarray(eef["pos"], dtype=np.float64).reshape(3)
    eef_quat = np.asarray(eef["quat"], dtype=np.float64).reshape(4)
    R = _quat_to_mat(eef_quat)  # 列 = EEF 各轴在世界系

    def to_local(world_pt):
        return R.T @ (np.asarray(world_pt, dtype=np.float64).reshape(3) - eef_pos)

    ctx.log("==== diag_gripper_frame ====")
    ctx.log(f"arm={arm} eef_pos={eef_pos.round(4).tolist()} "
            f"eef_quat={eef_quat.round(4).tolist()}")
    ctx.log(f"EEF局部 +Z(世界)={R[:,2].round(3).tolist()}  +Y={R[:,1].round(3).tolist()}")

    # 1) 各夹爪 link 原点在 EEF 局部系
    link_names = []
    try:
        link_names = list(robot.finger_link_names.get(arm, []))
    except Exception:
        pass
    extra = [f"{arm}_gripper_link"]
    for ln in extra + link_names:
        lk = robot.links.get(ln)
        if lk is None:
            ctx.log(f"  link {ln}: (缺失)")
            continue
        try:
            p, _q = lk.get_position_orientation()
            loc = to_local(p)
            ctx.log(f"  link {ln:34s} EEF局部={loc.round(4).tolist()}")
        except Exception as e:
            ctx.log(f"  link {ln}: 读取失败 {e}")

    # 2) 手指碰撞网格顶点在 EEF 局部系的 z/y 范围（判定指尖朝向）
    for ln in link_names:
        lk = robot.links.get(ln)
        if lk is None:
            continue
        try:
            cms = getattr(lk, "collision_meshes", {}) or {}
            allz = []
            ally = []
            for cm in cms.values():
                pts = cm.points.numpy() if hasattr(cm.points, "numpy") else np.asarray(cm.points)
                scale = cm.get_world_scale().numpy() if hasattr(cm.get_world_scale(), "numpy") else np.asarray(cm.get_world_scale())
                cmp_, cmq = cm.get_position_orientation()
                Rcm = _quat_to_mat(np.asarray(cmq if not hasattr(cmq,"numpy") else cmq.numpy(), dtype=np.float64).reshape(4))
                cmp_ = np.asarray(cmp_ if not hasattr(cmp_,"numpy") else cmp_.numpy(), dtype=np.float64).reshape(3)
                world_pts = (Rcm @ (pts * scale).T).T + cmp_
                loc_pts = (R.T @ (world_pts - eef_pos).T).T
                allz.append(loc_pts[:, 2]); ally.append(loc_pts[:, 1])
            if allz:
                z = np.concatenate(allz); y = np.concatenate(ally)
                ctx.log(f"  mesh {ln:30s} EEF局部 z∈[{z.min():.4f},{z.max():.4f}] "
                        f"y∈[{y.min():.4f},{y.max():.4f}] (n={len(z)})")
        except Exception as e:
            ctx.log(f"  mesh {ln}: 失败 {e}")

    # 3) gripper-fit 内部模型常量
    ctx.log("  --- plan_grasp_gripper_fit 内部模型 ---")
    ctx.log(f"  GRIP_GAP_CENTER_LOCAL={gf.GRIP_GAP_CENTER_LOCAL.tolist()} "
            f"_GAP_BOX(ylo,yhi,zlo,zhi,xh)={gf._GAP_BOX}")
    ctx.log(f"  _FINGER_BOXES={gf._FINGER_BOXES}")
    ctx.log(f"  _PALM_BOX={gf._PALM_BOX}")
    ctx.log(f"  指尖判定 gripper_reach_length_m={gf.gripper_reach_length_m():.4f} "
            f"(=|min finger z|)")
    ctx.log("  >>> 对比要点：真机手指 mesh 的 z 范围若为正（指尖在 +Z），"
            "而 _FINGER_BOXES 指尖在 z=-0.095（-Z）→ 约定相反，gripper-fit 模型翻转")
    ctx.log("==== diag_gripper_frame done ====")
    ctx.set_result({"ok": True})
    yield world.empty_action()


# 注册台面 audit 摆放 skill（reload grasp 时一并加载）
import behavior_interface.skills.stage_countertop_grasp_audit  # noqa: F401,E402
