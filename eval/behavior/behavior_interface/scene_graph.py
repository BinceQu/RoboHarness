"""
Scene Graph：场景的文本结构化表示。

包含三块：
  1. nodes      —— task/家具类物体的列表（统一命名 + 全局位姿 + AABB + 状态）
  2. relations  —— TRO ↔ 全部节点 的 on_top / inside / under / next_to 关系
  3. free_region —— 机器人可移动的区域（世界 AABB 减若干障碍 AABB），
                   既给 web 文本展示，又给 move/move_to 提供 A* 避障

设计要点：
- 避免调用慢的 OmniGibson object_states（OnTop 依赖 Touching 碰撞查询，
  每对几十毫秒），全部用 AABB 几何近似关系。
- 主线程定时重建（默认 5 秒一次），web 线程只读 cache，避免 PhysX 并发污染。

外部入口：
    sg = build_scene_graph(env, robot, robot_radius=0.25)
    text = format_scene_graph(sg)
    path = plan_path(sg.free_region, start_xy, goal_xy, robot_radius=0.25)
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 物体过滤：黑名单 + 体积下限
# ─────────────────────────────────────────────────────────────────────────────

# 这些 category 一定不进 Scene Graph（建筑要素 + 装饰 + 灯具）
_EXCLUDE_CATEGORIES = {
    "walls", "wall", "ceilings", "ceiling", "floors", "floor", "floor_lamp_base",
    "roof", "roofs", "stairs", "fence", "railing", "railings",
    "pole", "column",
    # 户外地面铺装（贴地，机器人不会撞）
    "lawn", "grass", "driveway", "pavement", "sidewalk", "road",
    "paver", "pavers", "cobblestone", "tile_flooring", "tile_floor",
    # 绿化
    "bush", "tree", "flower", "garden_plant", "potted_plant",
    "shrub", "hedge",
    # 装饰
    "picture", "painting", "wall_clock", "decoration", "decorative_object",
    "flag", "banner", "carpet", "rug", "curtain", "curtains", "blinds",
    # 灯具（一般小且常吊在天花板上，对避障无意义）
    "pendant_light", "pendant_lamp", "wall_light", "ceiling_light",
    "ceiling_lamp", "wall_lamp", "chandelier",
    # 窗、门框（门有时要保留，但门的运动很难处理，先排除）
    "window", "windows", "skylight",
}

# 这些 category 即使在 _EXCLUDE_CATEGORIES 里，仍然作为障碍参与 free_region 计算
# 例如 tree —— 不是 task 节点，但树干是真实障碍
_OBSTACLE_EVEN_IF_EXCLUDED = {
    "tree", "bush", "shrub", "hedge",
    "door", "doors", "doorframe",
}

# 这些 category 不视为 SceneGraph 节点，但**会**作为障碍参与 free_region 计算
# （即不在文本里出，但避障算上）—— 已并入 _OBSTACLE_EVEN_IF_EXCLUDED
_OBSTACLE_ONLY_CATEGORIES: set = set()

# 机器人撞得到的高度区间。物体 z_max < ROBOT_MIN_Z 算地面（paver/瓷砖），
# z_min > ROBOT_MAX_Z 算头顶上方（屋顶/吊灯），都不算障碍。
# R1Pro 底盘约 0.1m，整体高度约 1.5m，留点余量
_ROBOT_MIN_Z = 0.05    # 顶部低于这个高度的算地面
_ROBOT_MAX_Z = 1.8     # 底部高于这个高度的算屋顶/天花板

# 节点最小体积（m³），过滤掉装饰小物件
_MIN_NODE_VOLUME = 0.002
# 节点最小 xy 占地面积（m²），低于此即使体积够也不进 SceneGraph（小杂物）
_MIN_NODE_FOOTPRINT = 0.01


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _safe_aabb(obj) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """获取 obj 的 world-frame AABB；失败返回 None。"""
    try:
        lo, hi = obj.aabb
        lo = _to_np(lo).reshape(-1)
        hi = _to_np(hi).reshape(-1)
        if lo.shape[0] != 3 or hi.shape[0] != 3:
            return None
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            return None
        return lo, hi
    except Exception:
        return None


def _safe_yaw(obj) -> float:
    try:
        _, quat = obj.get_position_orientation()
        q = _to_np(quat)
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        return math.degrees(math.atan2(siny, cosy))
    except Exception:
        return 0.0


def _is_node_candidate(obj) -> bool:
    """是否进入 SceneGraph 节点列表（家具/task 物体）。"""
    cat = (getattr(obj, "category", "") or "").lower()
    if cat == "robot":
        return False
    if cat in _EXCLUDE_CATEGORIES:
        return False
    if cat.startswith("wall") or cat.startswith("floor") or cat.startswith("ceiling"):
        return False
    return True


def _is_obstacle_candidate(obj) -> bool:
    """是否纳入 free_region 的障碍物列表（包含家具 + 部分排除项如 door / tree）。
    注意：z 高度过滤在 build_scene_graph 里另做（这里只看类别）。
    """
    cat = (getattr(obj, "category", "") or "").lower()
    # 机器人本身绝对不能算障碍，否则起点会被自己挡住，A* 找不到出口
    if cat == "robot":
        return False
    if cat.startswith("wall") or cat.startswith("floor") or cat.startswith("ceiling"):
        return False
    if cat in {"lawn", "grass", "driveway", "pavement", "sidewalk", "road",
               "paver", "pavers", "cobblestone", "tile_flooring", "tile_floor",
               "roof", "roofs"}:
        return False
    if cat in {"picture", "painting", "wall_clock", "flag", "banner",
               "carpet", "rug", "curtain", "curtains", "blinds",
               "pendant_light", "pendant_lamp", "wall_light",
               "ceiling_light", "ceiling_lamp", "wall_lamp", "chandelier",
               "window", "windows", "skylight"}:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SGObject:
    """Scene Graph 节点。`name` 是全 Scene Graph 唯一的引用标识，
    其它地方（relations 的 subject/target、tro 列表、obstacles 列表）
    都只用 `name`，不用 scene_name / bddl_key。
    `scene_name` 和 `bddl_key` 仅作为元数据保留。
    """
    name: str                            # 唯一短名（如 "microwave" / "bar#1" / "door#3"）
    category: str
    bddl_key: Optional[str]              # 如 "microwave.n.02_1"，仅 TRO 有；非 TRO 为 null
    scene_name: str                      # OmniGibson 内部 ID（如 "microwave_hjjxmi_0"）
    pos: List[float]                     # [x, y, z] 世界系米
    yaw_deg: float                       # 绕 Z（度）
    aabb_min: List[float]                # [xmin, ymin, zmin]
    aabb_max: List[float]                # [xmax, ymax, zmax]
    states: Dict[str, Any] = field(default_factory=dict)  # 如 {"Open": True}


@dataclass
class SGRelation:
    """关系的 subject/target 都用 SGObject.name（唯一短名）。"""
    subject: str
    relation: str   # "on_top_of" | "inside" | "under" | "next_to"
    target: str


@dataclass
class FreeRegion:
    """可行区域 = world_bounds \\ union(obstacles)。
    obstacle 的 name 字段与对应 SGObject.name 一致（同一短名体系）。
    """
    bounds: Tuple[float, float, float, float]   # (xmin, ymin, xmax, ymax)
    obstacles: List[Dict[str, Any]]             # [{name, category, x:[a,b], y:[c,d], z:[..]}]
    robot_radius: float


@dataclass
class SceneGraph:
    """SceneGraph 数据结构。
    `robot_pose` 描述的是**胸口前向向量**：起点是机器人胸口（torso_link4 link 位置），
    向量方向是胸口朝前（torso_link4 局部 +X 在世界系下的方向）。
    具体字段：
      - x, y, z         胸口起点（米）
      - theta_x_deg     胸口前向向量与世界 +X 轴的夹角（度，水平面投影方向），
                        0 = 朝 +X，90 = 朝 +Y，绕 +Z 逆时针
      - theta_z_deg     胸口前向向量与世界 +Z 轴的夹角（度，球坐标极角），
                        0 = 朝上，90 = 水平，180 = 朝下
      - base_x, base_y, base_yaw_deg  保留底盘信息，给 move/移动用
    """
    objects: List[SGObject]
    relations: List[SGRelation]
    free_region: FreeRegion
    robot_pose: Dict[str, float]
    tro: List[str]                 # TRO 物体的 name 列表（用统一短名，不是 BDDL key）
    build_ms: float = 0.0          # 构建耗时


# ─────────────────────────────────────────────────────────────────────────────
# AABB 几何关系（替代慢的 OnTop/Inside/Under state，纯几何）
# ─────────────────────────────────────────────────────────────────────────────

def _xy_overlap_area(a: SGObject, b: SGObject) -> float:
    """两个 AABB 在 xy 平面的重叠面积。"""
    xmin = max(a.aabb_min[0], b.aabb_min[0])
    xmax = min(a.aabb_max[0], b.aabb_max[0])
    ymin = max(a.aabb_min[1], b.aabb_min[1])
    ymax = min(a.aabb_max[1], b.aabb_max[1])
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    return (xmax - xmin) * (ymax - ymin)


def _xy_distance(a: SGObject, b: SGObject) -> float:
    """两个 AABB 在 xy 平面的最小欧氏距离（重叠返回 0）。"""
    dx = max(0.0, max(a.aabb_min[0] - b.aabb_max[0], b.aabb_min[0] - a.aabb_max[0]))
    dy = max(0.0, max(a.aabb_min[1] - b.aabb_max[1], b.aabb_min[1] - a.aabb_max[1]))
    return math.hypot(dx, dy)


def _classify_relation(a: SGObject, b: SGObject) -> Optional[str]:
    """根据 AABB 几何判定 a 对 b 的关系（启发式）。
    返回 None 表示无明显关系。
    """
    a_footprint = max(1e-6,
                      (a.aabb_max[0] - a.aabb_min[0]) *
                      (a.aabb_max[1] - a.aabb_min[1]))
    overlap = _xy_overlap_area(a, b)
    if overlap < 0.1 * a_footprint:
        # xy 上几乎不重叠：只能是 next_to 或无关
        if _xy_distance(a, b) < 0.3 and abs(a.aabb_min[2] - b.aabb_min[2]) < 0.3:
            return "next_to"
        return None

    # xy 重叠较大：考察 z 关系
    a_zmin, a_zmax = a.aabb_min[2], a.aabb_max[2]
    b_zmin, b_zmax = b.aabb_min[2], b.aabb_max[2]

    # inside：a 在 b 的 z 范围内 + xy 完全被 b 包住
    if (a_zmin > b_zmin - 0.02 and a_zmax < b_zmax + 0.02 and
            a.aabb_min[0] > b.aabb_min[0] - 0.02 and
            a.aabb_max[0] < b.aabb_max[0] + 0.02 and
            a.aabb_min[1] > b.aabb_min[1] - 0.02 and
            a.aabb_max[1] < b.aabb_max[1] + 0.02):
        return "inside"

    # on_top_of：a 底部接近 b 顶部
    if abs(a_zmin - b_zmax) < 0.08 and a_zmin >= b_zmin:
        return "on_top_of"

    # under：a 顶部接近 b 底部
    if abs(a_zmax - b_zmin) < 0.08 and a_zmax <= b_zmax:
        return "under"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 构建 SceneGraph
# ─────────────────────────────────────────────────────────────────────────────

def _read_object_state_open(obj) -> Optional[bool]:
    try:
        from omnigibson.object_states import Open  # type: ignore
        if Open in obj.states:
            return bool(obj.states[Open].get_value())
    except Exception:
        pass
    return None


def _collect_bddl_keys(env) -> Dict[str, Any]:
    """返回 {scene_obj_name: bddl_key}。"""
    out: Dict[str, str] = {}
    try:
        task = env.task
        scope = getattr(task, "object_scope", None) or {}
        for key, ent in scope.items():
            obj = getattr(ent, "unwrapped", ent)
            if obj is None or not hasattr(obj, "name"):
                continue
            try:
                if not getattr(ent, "exists", True):
                    continue
            except Exception:
                pass
            out[obj.name] = key
    except Exception:
        pass
    return out


def _to_sg_object(obj, bddl_key: Optional[str]) -> Optional[SGObject]:
    aabb = _safe_aabb(obj)
    if aabb is None:
        return None
    lo, hi = aabb
    extent = hi - lo
    volume = float(max(0.0, extent[0]) * max(0.0, extent[1]) * max(0.0, extent[2]))
    footprint = float(max(0.0, extent[0]) * max(0.0, extent[1]))
    if bddl_key is None and (volume < _MIN_NODE_VOLUME or footprint < _MIN_NODE_FOOTPRINT):
        # 装饰小物件直接丢；TRO 强制保留
        return None
    try:
        pos_t, _ = obj.get_position_orientation()
        pos = _to_np(pos_t).tolist()
    except Exception:
        pos = ((lo + hi) / 2.0).tolist()
    yaw = _safe_yaw(obj)
    states: Dict[str, Any] = {}
    open_v = _read_object_state_open(obj)
    if open_v is not None:
        states["Open"] = open_v
    return SGObject(
        name="",   # 在 build_scene_graph 里统一赋值（_assign_unique_names）
        category=str(getattr(obj, "category", "") or ""),
        bddl_key=bddl_key,
        scene_name=str(obj.name),
        pos=[round(float(p), 3) for p in pos],
        yaw_deg=round(yaw, 1),
        aabb_min=[round(float(lo[i]), 3) for i in range(3)],
        aabb_max=[round(float(hi[i]), 3) for i in range(3)],
        states=states,
    )


def _assign_unique_names(scene_names: List[str], categories: Dict[str, str],
                         positions: Dict[str, Tuple[float, float]]) -> Dict[str, str]:
    """给一组 (scene_name, category, (x, y)) 分配全局唯一短名。

    规则：同 category 内按 (x, y) 升序排列。
      - 单实例 → 直接用 `category`
      - 多实例 → `category#1`, `category#2`, ...
    返回 {scene_name: short_name}。
    """
    by_cat: Dict[str, List[str]] = {}
    for sn in scene_names:
        by_cat.setdefault(categories.get(sn, ""), []).append(sn)
    name_map: Dict[str, str] = {}
    for cat, lst in by_cat.items():
        lst.sort(key=lambda sn: positions.get(sn, (0.0, 0.0)))
        if len(lst) == 1:
            name_map[lst[0]] = cat or lst[0]
        else:
            for i, sn in enumerate(lst, 1):
                name_map[sn] = f"{cat}#{i}" if cat else f"{sn}"
    return name_map


def _scene_xy_bounds(objects: List[SGObject], pad: float = 0.5) -> Tuple[float, float, float, float]:
    if not objects:
        return (-10.0, -10.0, 10.0, 10.0)
    xs_lo = [o.aabb_min[0] for o in objects]
    xs_hi = [o.aabb_max[0] for o in objects]
    ys_lo = [o.aabb_min[1] for o in objects]
    ys_hi = [o.aabb_max[1] for o in objects]
    return (min(xs_lo) - pad, min(ys_lo) - pad,
            max(xs_hi) + pad, max(ys_hi) + pad)


def _child_link_name(obj, joint) -> Optional[str]:
    """从铰链 joint 解析 child link 短名。"""
    child_raw = None
    for attr in ("body1", "child", "child_link", "_body1_name", "child_name"):
        v = getattr(joint, attr, None)
        if v is None:
            continue
        child_raw = v if isinstance(v, str) else getattr(v, "name", str(v))
        break
    if child_raw is None:
        return None
    child_short = child_raw.split("/")[-1]
    links = getattr(obj, "links", None) or {}
    if child_short not in links:
        for ln in links.keys():
            if ln in child_raw:
                return ln
        return None
    return child_short


def _append_articulated_link_obstacles(
    scene_objects,
    obstacle_raw: List[Tuple[str, str, np.ndarray, np.ndarray]],
    seen: set,
) -> None:
    """半开烤箱门等：物体级 AABB 不含门板，补 child link 当前 AABB。"""
    try:
        from omnigibson.object_states.open_state import Open, _get_relevant_joints
    except Exception:
        return
    for obj in scene_objects:
        try:
            if Open not in getattr(obj, "states", {}):
                continue
            base_name = str(getattr(obj, "name", ""))
            if not base_name:
                continue
            cat = (getattr(obj, "category", "") or "").lower()
            st = obj.states[Open]
            info = st.relevant_joints_info or _get_relevant_joints(obj)
            _, joints, _dirs = info
            links = getattr(obj, "links", None) or {}
            for joint in joints:
                child_short = _child_link_name(obj, joint)
                if not child_short:
                    continue
                link = links.get(child_short)
                if link is None:
                    continue
                aabb = _safe_aabb(link)
                if aabb is None:
                    try:
                        lo, hi = link.aabb
                        aabb = (_to_np(lo).reshape(3), _to_np(hi).reshape(3))
                    except Exception:
                        continue
                lo, hi = aabb
                z_low, z_high = float(lo[2]), float(hi[2])
                if z_high < _ROBOT_MIN_Z or z_low > _ROBOT_MAX_Z:
                    continue
                fp = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
                if fp < _MIN_NODE_FOOTPRINT:
                    continue
                key = f"{base_name}::{child_short}"
                if key in seen:
                    continue
                seen.add(key)
                obstacle_raw.append((key, f"{cat}_link", lo, hi))
        except Exception:
            continue


def build_scene_graph(env, robot, robot_radius: float = 0.25,
                      include_only_fixed_for_obstacle: bool = True) -> SceneGraph:
    """从 OmniGibson env 构建 SceneGraph。**必须在主 sim 线程调用**。"""
    import time as _t
    t0 = _t.time()

    bddl_keys = _collect_bddl_keys(env)
    robot_name = str(getattr(robot, "name", "")) if robot is not None else ""

    nodes: List[SGObject] = []
    obstacle_raw: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    # ↑ (name, category, lo, hi)

    try:
        scene_objects = list(env.scene.objects)
    except Exception:
        scene_objects = []

    for obj in scene_objects:
        try:
            name = str(getattr(obj, "name", ""))
            if not name or name == robot_name:
                continue
            cat = (getattr(obj, "category", "") or "").lower()

            # 节点（家具/task）
            if _is_node_candidate(obj):
                sg_obj = _to_sg_object(obj, bddl_keys.get(name))
                if sg_obj is not None:
                    nodes.append(sg_obj)

            # 障碍物（包括 node 候选 + door 等）
            if _is_obstacle_candidate(obj):
                # 静态家具才进避障；可拿走的小物体（fixed_base=False）跳过
                if include_only_fixed_for_obstacle:
                    try:
                        if not bool(getattr(obj, "fixed_base", False)):
                            # 但 task 相关大件（如 microwave）即使 fixed_base 可能是 False，
                            # 体积大于 0.05 m³ 时也加入避障，保守一点
                            aabb = _safe_aabb(obj)
                            if aabb is None:
                                continue
                            lo, hi = aabb
                            vol = float(max(0, hi[0] - lo[0]) *
                                        max(0, hi[1] - lo[1]) *
                                        max(0, hi[2] - lo[2]))
                            if vol < 0.05:
                                continue
                    except Exception:
                        continue
                aabb = _safe_aabb(obj)
                if aabb is None:
                    continue
                lo, hi = aabb
                # 太薄/太小的 AABB 跳过（例如平贴墙面的物体）
                if (hi[0] - lo[0]) * (hi[1] - lo[1]) < _MIN_NODE_FOOTPRINT:
                    continue
                # 几何兜底：z 高度不在机器人可能撞到的范围内则跳过
                # （paver 整张地砖、roof 整片屋顶等没在黑名单时也能滤掉）
                z_low, z_high = float(lo[2]), float(hi[2])
                if z_high < _ROBOT_MIN_Z:
                    continue                                            # 贴地铺装
                if z_low > _ROBOT_MAX_Z:
                    continue                                            # 头顶上方
                # 树/灌木的 AABB 是整个树冠，但机器人只会撞树干
                # 用 pos 中心 + 小 footprint 替代
                if cat in {"tree", "bush", "shrub", "hedge"}:
                    try:
                        pos_t, _ = obj.get_position_orientation()
                        cx, cy = float(_to_np(pos_t)[0]), float(_to_np(pos_t)[1])
                    except Exception:
                        cx = float((lo[0] + hi[0]) / 2)
                        cy = float((lo[1] + hi[1]) / 2)
                    trunk = 0.3 if cat == "tree" else 0.2
                    lo = np.array([cx - trunk, cy - trunk, z_low], dtype=np.float64)
                    hi = np.array([cx + trunk, cy + trunk, z_high], dtype=np.float64)
                obstacle_raw.append((name, cat, lo, hi))
        except Exception:
            continue

    # 铰链门板（半开烤箱门等）：补 link 级障碍，避免只认柜体 AABB
    _append_articulated_link_obstacles(scene_objects, obstacle_raw, set())

    # ── 统一命名：nodes ∪ obstacles 一起进入 name_map ──
    # 这样 obstacles 里的 name 和 nodes 里的 name 完全一致，整张图只有一套引用
    all_scene_names: List[str] = []
    cat_map: Dict[str, str] = {}
    pos_map: Dict[str, Tuple[float, float]] = {}
    for n in nodes:
        all_scene_names.append(n.scene_name)
        cat_map[n.scene_name] = n.category
        pos_map[n.scene_name] = (float(n.pos[0]), float(n.pos[1]))
    for (sn, cat, lo, hi) in obstacle_raw:
        if sn in cat_map:
            continue
        all_scene_names.append(sn)
        cat_map[sn] = cat
        pos_map[sn] = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2))
    name_map = _assign_unique_names(all_scene_names, cat_map, pos_map)

    # 把 name 写回 nodes
    for n in nodes:
        n.name = name_map.get(n.scene_name, n.scene_name)

    # free region：obstacles 也用统一 name
    bounds = _scene_xy_bounds(nodes, pad=0.5)
    obstacles = [
        {
            "name": name_map.get(sn, sn),
            "category": cat,
            "x": [round(float(lo[0]), 3), round(float(hi[0]), 3)],
            "y": [round(float(lo[1]), 3), round(float(hi[1]), 3)],
            "z": [round(float(lo[2]), 3), round(float(hi[2]), 3)],
        }
        for (sn, cat, lo, hi) in obstacle_raw
    ]
    free_region = FreeRegion(bounds=bounds, obstacles=obstacles, robot_radius=robot_radius)

    # ── 关系：所有节点两两计算（带距离剪枝），覆盖完整 ──
    relations: List[SGRelation] = []
    REL_DIST_MAX = 1.5
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i == j:
                continue
            if _xy_distance(a, b) > REL_DIST_MAX:
                continue
            rel = _classify_relation(a, b)
            if rel is None:
                continue
            # next_to 对称，只存 i<j 的一份；on_top_of/inside/under 双向各存（但 b 视角是 has_on_top 等，在格式化时生成，不存）
            if rel == "next_to" and i > j:
                continue
            relations.append(SGRelation(
                subject=a.name,   # 统一短名
                relation=rel,
                target=b.name,    # 统一短名
            ))

    # 机器人位姿：用胸口 (torso_link4) link 的世界 pose
    # 胸口前向向量 = torso_link4 局部 +X 旋转到世界系
    # 输出 (x, y, z) = 胸口位置；theta_x_deg = forward 与 +X 夹角（水平 yaw）；
    # theta_z_deg = forward 与 +Z 夹角（球坐标极角，0=朝上 / 90=水平 / 180=朝下）
    robot_pose = {
        "x": 0.0, "y": 0.0, "z": 0.0,
        "theta_x_deg": 0.0, "theta_z_deg": 90.0,
        "base_x": 0.0, "base_y": 0.0, "base_yaw_deg": 0.0,
    }
    try:
        # 1) base 位姿
        bpos_t, bquat_t = robot.get_position_orientation()
        bpos = _to_np(bpos_t)
        bquat = _to_np(bquat_t)
        bx, by, bz, bw = float(bquat[0]), float(bquat[1]), float(bquat[2]), float(bquat[3])
        base_siny = 2.0 * (bw * bz + bx * by)
        base_cosy = 1.0 - 2.0 * (by * by + bz * bz)
        base_yaw_deg = math.degrees(math.atan2(base_siny, base_cosy))

        # 2) 胸口 link 位姿（优先 torso_link4；找不到则回退到 base + 高度偏移）
        chest_pos = None
        chest_quat = None
        try:
            chest_link = robot.links.get("torso_link4")
            if chest_link is not None:
                cpos, cquat = chest_link.get_position_orientation()
                chest_pos = _to_np(cpos)
                chest_quat = _to_np(cquat)
        except Exception:
            chest_pos = None

        if chest_pos is None:
            # 回退：胸口 = base 上方 1.2m（R1Pro 站姿大致高度）
            chest_pos = bpos + np.array([0.0, 0.0, 1.2])
            chest_quat = bquat

        # 3) forward 向量 = chest 局部 +X 旋转到世界
        qx, qy, qz, qw = (float(chest_quat[0]), float(chest_quat[1]),
                          float(chest_quat[2]), float(chest_quat[3]))
        # 旋转 [1, 0, 0]：fx = 1 - 2*(y² + z²), fy = 2*(xy + wz), fz = 2*(xz - wy)
        fx = 1.0 - 2.0 * (qy * qy + qz * qz)
        fy = 2.0 * (qx * qy + qw * qz)
        fz = 2.0 * (qx * qz - qw * qy)
        fnorm = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        fx /= fnorm; fy /= fnorm; fz /= fnorm

        theta_x_deg = math.degrees(math.atan2(fy, fx))           # 水平 yaw
        theta_z_deg = math.degrees(math.acos(max(-1.0, min(1.0, fz))))  # 与 +Z 夹角

        robot_pose = {
            "x": round(float(chest_pos[0]), 3),
            "y": round(float(chest_pos[1]), 3),
            "z": round(float(chest_pos[2]), 3),
            "theta_x_deg": round(theta_x_deg, 1),
            "theta_z_deg": round(theta_z_deg, 1),
            "base_x": round(float(bpos[0]), 3),
            "base_y": round(float(bpos[1]), 3),
            "base_yaw_deg": round(base_yaw_deg, 1),
        }
    except Exception:
        pass

    # TRO 用统一 name（而不是 BDDL key），便于和 objects.name / relations 引用对齐
    tro_names = sorted({n.name for n in nodes if n.bddl_key})

    sg = SceneGraph(
        objects=nodes,
        relations=relations,
        free_region=free_region,
        robot_pose=robot_pose,
        tro=tro_names,
        build_ms=round((_t.time() - t0) * 1000.0, 1),
    )
    return sg


# ─────────────────────────────────────────────────────────────────────────────
# 文本格式化（给 web 显示和 LLM 消费）
# ─────────────────────────────────────────────────────────────────────────────

def _build_inverse_relations(sg: SceneGraph) -> Dict[str, List[Dict[str, str]]]:
    """从 sg.relations 派生每个物体的"对方视角"关系，便于 LLM 读：
      - next_to 对称，双方都列
      - A on_top_of B  ⇒  B has_on_top A
      - A inside    B  ⇒  B contains    A
      - A under     B  ⇒  B above       A
    返回 {name: [{"relation":..., "target":...}, ...]}，已去重排序。
    """
    by_subj: Dict[str, List[Tuple[str, str]]] = {n.name: [] for n in sg.objects}
    for r in sg.relations:
        by_subj.setdefault(r.subject, []).append((r.relation, r.target))
        if r.relation == "next_to":
            by_subj.setdefault(r.target, []).append(("next_to", r.subject))
        elif r.relation == "on_top_of":
            by_subj.setdefault(r.target, []).append(("has_on_top", r.subject))
        elif r.relation == "inside":
            by_subj.setdefault(r.target, []).append(("contains", r.subject))
        elif r.relation == "under":
            by_subj.setdefault(r.target, []).append(("above", r.subject))
    rel_order = {"on_top_of": 0, "inside": 1, "under": 2,
                 "has_on_top": 3, "contains": 4, "above": 5,
                 "next_to": 6}
    out: Dict[str, List[Dict[str, str]]] = {}
    for name, lst in by_subj.items():
        seen = set()
        uniq = []
        for r, t in lst:
            if (r, t) in seen:
                continue
            seen.add((r, t))
            uniq.append((r, t))
        uniq.sort(key=lambda rt: (rel_order.get(rt[0], 99), rt[1]))
        out[name] = [{"relation": r, "target": t} for r, t in uniq]
    return out


def to_dict(sg: SceneGraph) -> Dict[str, Any]:
    """SceneGraph → 纯 dict（可 json.dumps 序列化）。
    每个 object 的 `name` 是全 sg 唯一标识；relations / tro / obstacles 内
    引用一律用这个 name。
    """
    rel_inv = _build_inverse_relations(sg)
    objs_sorted = sorted(
        sg.objects,
        key=lambda o: (o.bddl_key is None, o.category, o.name),
    )
    objects_json = []
    for o in objs_sorted:
        objects_json.append({
            "name": o.name,
            "category": o.category,
            "bddl_key": o.bddl_key,
            "scene_name": o.scene_name,
            "pos": o.pos,
            "yaw_deg": o.yaw_deg,
            "aabb_min": o.aabb_min,
            "aabb_max": o.aabb_max,
            "states": o.states,
            "relations": rel_inv.get(o.name, []),
        })
    fr = sg.free_region
    xmin, ymin, xmax, ymax = fr.bounds
    return {
        "robot": {
            "name": "robot",
            # 胸口前向向量的起点（chest_xyz）
            "chest_pos": [sg.robot_pose["x"], sg.robot_pose["y"], sg.robot_pose["z"]],
            # 胸口 forward 向量与世界 +X 轴的夹角（水平 yaw，度）
            "theta_x_deg": sg.robot_pose["theta_x_deg"],
            # 胸口 forward 向量与世界 +Z 轴的夹角（极角，度）
            "theta_z_deg": sg.robot_pose["theta_z_deg"],
            # 底盘信息（给 move/move_to 用）
            "base_pos": [sg.robot_pose.get("base_x", 0.0),
                         sg.robot_pose.get("base_y", 0.0)],
            "base_yaw_deg": sg.robot_pose.get("base_yaw_deg", 0.0),
        },
        "tro": sg.tro,
        "stats": {
            "objects": len(sg.objects),
            "relations": len(sg.relations),
            "obstacles": len(sg.free_region.obstacles),
            "build_ms": sg.build_ms,
        },
        "objects": objects_json,
        "free_region": {
            "bounds": list(fr.bounds),
            "robot_radius": fr.robot_radius,
            "constraint_text": (
                f"x ∈ [{xmin:.2f}, {xmax:.2f}]  ∧  y ∈ [{ymin:.2f}, {ymax:.2f}]"
                f"  ∧  ¬(∃ i: (x,y) ∈ inflated_obstacle[i])"
            ),
            "obstacles": fr.obstacles,
        },
    }


def format_scene_graph(sg: SceneGraph) -> str:
    """Scene Graph → 给人/LLM 看的 pretty JSON 字符串。
    所有物体引用都用唯一 name（如 "microwave"、"bar#1"），不混用别的标识。
    """
    import json as _json
    d = to_dict(sg)
    return _json.dumps(d, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# 避障：点检测 + A* 路径规划
# ─────────────────────────────────────────────────────────────────────────────

def _inflated_obstacles(free_region: FreeRegion,
                        extra_inflate: float = 0.0) -> List[Tuple[float, float, float, float]]:
    """返回 inflated 障碍 [(xmin, ymin, xmax, ymax), ...]."""
    r = free_region.robot_radius + extra_inflate
    out = []
    for ob in free_region.obstacles:
        x0, x1 = ob["x"]
        y0, y1 = ob["y"]
        out.append((x0 - r, y0 - r, x1 + r, y1 + r))
    return out


def _dist_inflated_rect(
    px: float, py: float,
    ix0: float, iy0: float, ix1: float, iy1: float,
) -> float:
    """点到膨胀矩形的有符号距离：外为正，内为负（越深越负）。"""
    if ix0 <= px <= ix1 and iy0 <= py <= iy1:
        return -min(px - ix0, ix1 - px, py - iy0, iy1 - py)
    dx = max(ix0 - px, 0.0, px - ix1)
    dy = max(iy0 - py, 0.0, py - iy1)
    return math.hypot(dx, dy)


def point_clearance(
    free_region: FreeRegion,
    x: float,
    y: float,
    extra_inflate: float = 0.0,
) -> Tuple[bool, float, Optional[str]]:
    """(是否 free, 到最近膨胀障碍距离 m, 最近障碍名)。在障内 clearance<0。"""
    xmin, ymin, xmax, ymax = free_region.bounds
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return False, -1.0, "<out_of_bounds>"
    r = free_region.robot_radius + extra_inflate
    best_d = float("inf")
    best_name: Optional[str] = None
    for ob in free_region.obstacles:
        x0, x1 = ob["x"]
        y0, y1 = ob["y"]
        d = _dist_inflated_rect(x, y, x0 - r, y0 - r, x1 + r, y1 + r)
        if d < best_d:
            best_d = d
            best_name = ob["name"]
    if best_d == float("inf"):
        return True, 999.0, None
    return best_d > 0.0, float(best_d), best_name


def is_point_free(free_region: FreeRegion, x: float, y: float,
                  extra_inflate: float = 0.0) -> Tuple[bool, Optional[str]]:
    """检查 (x, y) 是否在可行区域内。返回 (ok, blocking_obstacle_name)。"""
    ok, _, blk = point_clearance(free_region, x, y, extra_inflate=extra_inflate)
    return ok, blk


def is_segment_free(free_region: FreeRegion,
                    p0: Tuple[float, float], p1: Tuple[float, float],
                    step: float = 0.05, extra_inflate: float = 0.0) -> Tuple[bool, Optional[str]]:
    """沿线段离散采样检测可行。"""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    n = max(2, int(dist / step) + 1)
    for i in range(n + 1):
        t = i / n
        x = p0[0] + t * dx
        y = p0[1] + t * dy
        ok, name = is_point_free(free_region, x, y, extra_inflate=extra_inflate)
        if not ok:
            return False, name
    return True, None


def plan_path(free_region: FreeRegion,
              start: Tuple[float, float], goal: Tuple[float, float],
              resolution: float = 0.15,
              extra_inflate: float = 0.0,
              max_cells: int = 80000,
              snap_goal: bool = True) -> Tuple[List[Tuple[float, float]], str]:
    """A* 2D grid 规划。返回 (waypoints, status)。
    waypoints 至少含起点与终点；status ∈ {"ok", "no_path", "start_blocked",
    "goal_blocked", "skip_direct"}。
    若起点→终点直线无障碍，跳过 A* 直接返回两点。
    """
    # 优先：直线可通过就不规划
    seg_ok, _ = is_segment_free(free_region, start, goal,
                                step=max(0.05, resolution / 2.0),
                                extra_inflate=extra_inflate)
    if seg_ok:
        return [start, goal], "skip_direct"

    xmin, ymin, xmax, ymax = free_region.bounds
    # 把 start/goal 都纳入 bounds（防止机器人正好在边缘外）
    pad = resolution * 2
    xmin = min(xmin, start[0] - pad, goal[0] - pad)
    ymin = min(ymin, start[1] - pad, goal[1] - pad)
    xmax = max(xmax, start[0] + pad, goal[0] + pad)
    ymax = max(ymax, start[1] + pad, goal[1] + pad)

    nx = max(2, int(math.ceil((xmax - xmin) / resolution)))
    ny = max(2, int(math.ceil((ymax - ymin) / resolution)))
    if nx * ny > max_cells:
        # 太大就降低分辨率
        scale = math.sqrt(nx * ny / max_cells)
        resolution = resolution * scale
        nx = max(2, int(math.ceil((xmax - xmin) / resolution)))
        ny = max(2, int(math.ceil((ymax - ymin) / resolution)))

    def to_idx(x, y):
        ix = int(round((x - xmin) / resolution))
        iy = int(round((y - ymin) / resolution))
        ix = min(max(ix, 0), nx - 1)
        iy = min(max(iy, 0), ny - 1)
        return ix, iy

    def to_xy(ix, iy):
        return (xmin + ix * resolution, ymin + iy * resolution)

    obs_infl = _inflated_obstacles(free_region, extra_inflate=extra_inflate)

    def cell_blocked(ix, iy):
        x, y = to_xy(ix, iy)
        for (x0, y0, x1, y1) in obs_infl:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    sx, sy = to_idx(*start)
    gx, gy = to_idx(*goal)

    # 起点 / 终点本身被障碍占据：尝试在邻近找 free cell
    def nearest_free(ix, iy, radius=8):
        if not cell_blocked(ix, iy):
            return ix, iy
        for r in range(1, radius + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    nx2, ny2 = ix + dx, iy + dy
                    if 0 <= nx2 < nx and 0 <= ny2 < ny and not cell_blocked(nx2, ny2):
                        return nx2, ny2
                for dy in range(-r + 1, r):
                    for dx2 in (-r, r):
                        nx2, ny2 = ix + dx2, iy + dy
                        if 0 <= nx2 < nx and 0 <= ny2 < ny and not cell_blocked(nx2, ny2):
                            return nx2, ny2
        return None

    s = nearest_free(sx, sy)
    if s is None:
        return [start, goal], "start_blocked"
    sx, sy = s
    if snap_goal:
        g = nearest_free(gx, gy)
        if g is None:
            return [start, goal], "goal_blocked"
        gx, gy = g
    elif cell_blocked(gx, gy):
        return [start, goal], "goal_blocked"

    # A*
    def h(ix, iy):
        return math.hypot(ix - gx, iy - gy)

    open_set: List[Tuple[float, int, int]] = []
    heapq.heappush(open_set, (h(sx, sy), sx, sy))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
    DIRS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.41421356), (1, -1, 1.41421356),
            (-1, 1, 1.41421356), (-1, -1, 1.41421356)]
    visited = set()
    while open_set:
        _, cx, cy = heapq.heappop(open_set)
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))
        if (cx, cy) == (gx, gy):
            # 回溯
            path_cells = [(cx, cy)]
            while (cx, cy) in came_from:
                cx, cy = came_from[(cx, cy)]
                path_cells.append((cx, cy))
            path_cells.reverse()
            waypoints = [start] + [to_xy(ix, iy) for ix, iy in path_cells] + [goal]
            return _simplify_path(waypoints, free_region, extra_inflate, resolution), "ok"
        for dx, dy, w in DIRS:
            nx2, ny2 = cx + dx, cy + dy
            if not (0 <= nx2 < nx and 0 <= ny2 < ny):
                continue
            if cell_blocked(nx2, ny2):
                continue
            tg = g_score[(cx, cy)] + w
            if tg < g_score.get((nx2, ny2), float("inf")):
                g_score[(nx2, ny2)] = tg
                came_from[(nx2, ny2)] = (cx, cy)
                heapq.heappush(open_set, (tg + h(nx2, ny2), nx2, ny2))

    return [start, goal], "no_path"


def _simplify_path(waypoints: List[Tuple[float, float]],
                   free_region: FreeRegion,
                   extra_inflate: float,
                   step: float) -> List[Tuple[float, float]]:
    """贪心简化：把能直连的连续点合并。"""
    if len(waypoints) <= 2:
        return waypoints
    out = [waypoints[0]]
    i = 0
    while i < len(waypoints) - 1:
        j = len(waypoints) - 1
        # 找最远能直连的 j
        while j > i + 1:
            ok, _ = is_segment_free(free_region, waypoints[i], waypoints[j],
                                    step=max(0.05, step / 2.0),
                                    extra_inflate=extra_inflate)
            if ok:
                break
            j -= 1
        out.append(waypoints[j])
        i = j
    return out
