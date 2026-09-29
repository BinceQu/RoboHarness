"""capture skill —— agent 工具 `capture()` 的仿真后端。

渲染机器人相机（主视图 = head/zed 第一视角，见设计文档 Q2），把：
  - 主视图 RGB（→ 返回给 agent 的决策图）
  - head 的 depth_linear / seg_instance_id / normal（层 B 反投影资源，落盘备 plan_* 用）
  - head 相机内外参（反投影必需）
  - tro / robot 位姿 / holding（层 A 结构化状态）
落到 server 侧 session（agent_runs），并把主视图以 data URL 回传。

取图前用 `yield world.hold_action()` 稳定仿真并保持 trunk 俯仰（勿用 empty_action 全零）。
同一个 image_id 的 RGB/depth/seg 和 head camera pose 必须来自同一仿真 tick；
保存图像和读取相机外参之间不能再 step，否则机器人抖动时反投影射线会错帧。
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Dict, Optional

import cv2
import numpy as np

from behavior_interface import agent_runs
from behavior_interface.camera_frames import observation_frame_errors
from behavior_interface.camera_render_control import set_robot_camera_render_updates
from behavior_interface.head_capture import HEAD_FOCAL_LENGTH, HEAD_HORIZONTAL_APERTURE
from behavior_interface.skills import register_skill
from behavior_interface.skills.viz_base_path_overlay import render_base_forward_path_overlay
from behavior_interface.skills.vlm_lawn_dual import (
    _save_rgb,
    _save_depth,
    _save_seg,
    save_npy_atomic,
)


# zed = head 第一视角；realsense = 腕部相机
_FEED_KEYS = (
    ("zed_link", "head"),
    ("zed", "head"),
    ("head", "head"),
    ("left_realsense", "left_wrist"),
    ("left_wrist", "left_wrist"),
    ("right_realsense", "right_wrist"),
    ("right_wrist", "right_wrist"),
)
_HEAD_SENSOR_MARKERS = ("zed_link", "zed", "head", "camera")
_WRIST_SENSOR_MARKERS = ("left_realsense", "right_realsense", "left_wrist", "right_wrist", "wrist")
CAPTURE_HEAD_RED_OBSTACLE_OVERLAY_ENABLED = False


def _refresh_camera_render_control():
    """Reload render controls and patch the long-lived server module in place."""
    import importlib

    from behavior_interface import camera_render_control as control

    control = importlib.reload(control)
    globals()["set_robot_camera_render_updates"] = control.set_robot_camera_render_updates
    try:
        from behavior_interface import server as server_module

        server_module.set_robot_camera_render_updates = control.set_robot_camera_render_updates
    except Exception:
        pass
    return control


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _camera_io_guard(world):
    lock = getattr(world, "_codex_camera_io_lock", None)
    return lock if lock is not None else nullcontext()


def _robot_seg_ids_from_info(info: Any) -> Dict[str, Any]:
    raw = info.get("seg_instance_id") if isinstance(info, dict) else None
    id2label: Dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                id2label[int(k)] = str(v)
            except Exception:
                continue
    markers = (
        "/agent",
        "agent.n.01",
        "/robot",
        "robot",
        "r1pro",
        "r1_pro",
        "/r1",
    )
    robot_ids = {
        sid
        for sid, label in id2label.items()
        if any(marker in label.lower() for marker in markers)
    }
    return {
        "robot_ids": sorted(robot_ids),
        "robot_id_labels": {str(sid): id2label[sid] for sid in sorted(robot_ids)},
    }


def warm_obs_annotators(
    world,
    ctx=None,
    log_tag: str = "capture",
    *,
    n_render: int = 3,
    n_attempts: int = 4,
) -> bool:
    """退出 no_obs 前预热相机 annotator，避免恢复 normal env.step 时
    seg buffer 为空触发 `th.max(empty)` 崩溃。

    长时间 no_obs 运动期间从不调用 get_obs，分割 annotator 第一次取数会返回
    空张量（numel()==0），与 env.reset 里「render 3 次再 get_obs」同因。这里在
    仍处于 no_obs 的窗口内主动渲染并取一次 obs（带重试），把 annotator 预热到
    可产出非空 buffer 的状态。返回是否预热成功。
    """
    if getattr(world, "dry_run", False):
        return True
    try:
        import omnigibson as og
    except Exception:
        return False
    robot = getattr(world, "robot", None)
    if robot is None:
        return False
    set_robot_camera_render_updates(world, True)
    sensors = []
    try:
        for name, sensor in robot.sensors.items():
            if "rgb" in getattr(sensor, "modalities", []):
                sensors.append((name, sensor))
    except Exception:
        sensors = []
    if not sensors:
        return False
    for attempt in range(max(1, int(n_attempts))):
        try:
            with _camera_io_guard(world):
                for _ in range(max(1, int(n_render))):
                    og.sim.render()
        except Exception:
            pass
        ok = True
        last_err = None
        for name, sensor in sensors:
            try:
                with _camera_io_guard(world):
                    sensor.get_obs()
            except Exception as e:
                ok = False
                last_err = f"{name}: {e}"
                break
        if ok:
            if ctx is not None and attempt > 0:
                ctx.log(f"[{log_tag}] obs annotator 预热完成 (attempt={attempt + 1})")
            return True
        if ctx is not None and attempt == 0:
            ctx.log(f"[{log_tag}] obs annotator 预热重试中: {last_err}")
    if ctx is not None:
        ctx.log(f"[{log_tag}] WARN obs annotator 预热未完全成功")
    return False


def _sensor_modalities(sensor) -> list[str]:
    try:
        return [str(mod) for mod in (getattr(sensor, "modalities", []) or [])]
    except Exception:
        return []


def _collect_robot_sensors(robot) -> Dict[str, Any]:
    """遍历 robot.sensors，按 feed 名归类视觉传感器；capture 前再补 rgb。"""
    out: Dict[str, Any] = {}
    seen = []
    try:
        for sensor_name, sensor in robot.sensors.items():
            lname = str(sensor_name).lower()
            has_rgb = "rgb" in _sensor_modalities(sensor)
            seen.append((lname, sensor_name, sensor, has_rgb))
            for needle, feed in _FEED_KEYS:
                if needle in lname:
                    prev = out.get(feed)
                    if prev is None or (has_rgb and "rgb" not in _sensor_modalities(prev)):
                        out[feed] = sensor
                    break
    except Exception:
        pass
    if "head" not in out:
        for lname, _sensor_name, sensor, _has_rgb in seen:
            if any(marker in lname for marker in _HEAD_SENSOR_MARKERS) and not any(
                marker in lname for marker in _WRIST_SENSOR_MARKERS
            ):
                out["head"] = sensor
                break
    rgb_sensors = [item for item in seen if item[3]]
    if "head" not in out and len(rgb_sensors) == 1:
        out["head"] = rgb_sensors[0][2]
    return out


def _capture_sensor_items(
    sensors: Dict[str, Any],
    *,
    include_auxiliary_rgb: bool = False,
) -> list[tuple[str, Any]]:
    """Return capture feeds with the required head frame first.

    Head capture must not wait for empty wrist annotators. Dedicated wrist
    capture tools own those feeds; the legacy all-feed snapshot remains
    available through ``include_auxiliary_rgb``.
    """
    items = []
    head = sensors.get("head")
    if head is not None:
        items.append(("head", head))
    if include_auxiliary_rgb:
        for feed in ("left_wrist", "right_wrist"):
            sensor = sensors.get(feed)
            if sensor is not None:
                items.append((feed, sensor))
    return items


def _rgb_sensor_names(robot) -> list[str]:
    try:
        return [
            str(sensor_name)
            for sensor_name, sensor in robot.sensors.items()
            if "rgb" in _sensor_modalities(sensor)
        ]
    except Exception:
        return []


def _sensor_debug_names(robot) -> list[Dict[str, Any]]:
    try:
        return [
            {
                "name": str(sensor_name),
                "modalities": _sensor_modalities(sensor),
                "type": type(sensor).__name__,
            }
            for sensor_name, sensor in robot.sensors.items()
        ]
    except Exception:
        return []


def _external_sensor_debug_names(world) -> list[Dict[str, Any]]:
    try:
        env = getattr(world, "env", None)
        external = getattr(env, "_external_sensors", {}) or {}
        return [
            {
                "name": str(sensor_name),
                "modalities": _sensor_modalities(sensor),
                "type": type(sensor).__name__,
            }
            for sensor_name, sensor in external.items()
        ]
    except Exception:
        return []


def _gta_fallback_sensor(world):
    try:
        env = getattr(world, "env", None)
        external = getattr(env, "_external_sensors", {}) or {}
        return external.get("gta_view")
    except Exception:
        return None


def _ensure_modalities(sensor, mods) -> bool:
    """确保 sensor 开启给定 modality；有新增返回 True（调用方需再 render 让 buffer 生效）。"""
    changed = False
    have = list(getattr(sensor, "modalities", []))
    for mod in mods:
        if mod not in have:
            try:
                sensor.add_modality(mod)
                changed = True
            except Exception:
                pass
    return changed


def _warm_sensor_obs(sensor, ctx=None, log_tag: str = "capture.sensor", *, n_render: int = 3, n_attempts: int = 4) -> bool:
    try:
        import omnigibson as og
    except Exception:
        og = None
    last_err = None
    for attempt in range(max(1, int(n_attempts))):
        try:
            with _camera_io_guard(getattr(ctx, "world", None)):
                if og is not None:
                    for _ in range(max(1, int(n_render))):
                        og.sim.render()
                obs, _ = sensor.get_obs()
            rgb = obs.get("rgb") if isinstance(obs, dict) else None
            if rgb is not None and _to_np(rgb).size == 0:
                raise RuntimeError("rgb buffer empty")
            return True
        except Exception as e:
            last_err = e
            if ctx is not None and attempt == 0:
                ctx.log(f"[{log_tag}] obs annotator 预热重试中: {e}")
    if ctx is not None and last_err is not None:
        ctx.log(f"[{log_tag}] WARN obs annotator 预热失败: {last_err}")
    return False


def _read_complete_frame(
    sensor,
    required_modalities,
    ctx,
    *,
    log_tag: str,
    n_attempts: int = 4,
    allow_rebuild: bool = True,
):
    obs = info = None
    frame_errors = ["camera frame not read"]
    repair_rounds = 2 if allow_rebuild else 1
    for repair_round in range(repair_rounds):
        for attempt in range(max(1, int(n_attempts))):
            try:
                with _camera_io_guard(getattr(ctx, "world", None)):
                    obs, info = sensor.get_obs()
                frame_errors = observation_frame_errors(obs, required_modalities)
                if not frame_errors:
                    return obs, info, []
                if attempt == 0:
                    ctx.log(
                        f"{log_tag} WARN 帧未就绪，重新预热: "
                        + "; ".join(frame_errors)
                    )
            except Exception as e:
                frame_errors = [f"get_obs failed: {e}"]
                if attempt == 0:
                    ctx.log(f"{log_tag} WARN get_obs 失败，重新预热: {e}")
            _warm_sensor_obs(
                sensor,
                ctx,
                log_tag=log_tag,
                n_render=3,
                n_attempts=1,
            )
        if allow_rebuild and repair_round == 0:
            control = _refresh_camera_render_control()
            report = control.rebuild_robot_camera_render_products(
                ctx.world,
                sensor=sensor,
            )
            if report.get("sensors"):
                ctx.log(
                    f"{log_tag} WARN 重建 robot camera render products 后重试: "
                    + ", ".join(report["sensors"])
                )
            if report.get("errors"):
                ctx.log(
                    f"{log_tag} WARN render product 重建异常: "
                    + "; ".join(report["errors"])
                )
    return obs, info, frame_errors


def _holding(robot) -> Dict[str, Optional[str]]:
    """尽力读取每只手当前抓着的物体名（assisted-grasping）。拿不到则 None。"""
    out: Dict[str, Optional[str]] = {"left": None, "right": None}
    ag = getattr(robot, "_ag_obj_in_hand", None)
    if isinstance(ag, dict):
        for arm in ("left", "right"):
            obj = ag.get(arm)
            if obj is not None:
                out[arm] = getattr(obj, "name", str(obj))
    return out


def _camera_meta(sensor) -> Dict[str, Any]:
    """读 head 相机内外参（反投影用）。"""
    from behavior_interface.head_capture import head_intrinsics_dict

    pos, quat = sensor.get_position_orientation()
    pos = _to_np(pos).reshape(-1).tolist()
    quat = _to_np(quat).reshape(-1).tolist()  # xyzw
    intr = head_intrinsics_dict(sensor)
    return {
        "pos": [float(v) for v in pos[:3]],
        "quat": [float(v) for v in quat[:4]],
        **intr,
    }


def render_vectors_overlay(session_id: str, rgb_path: str, cam_meta: dict, out_path: str):
    """把 session 下 plans/vectors.json 里的向量按当前相机重投影叠加到 head 图。

    每个向量画出局部坐标系：
      红=法向/接近向(dir_world)，蓝=主轴(axis_world，双向)，绿=副法向(binormal_world)。
    向量以世界坐标存储，任意相机位姿都能正确重投影，因此跨视角持久可见。
    返回 {ok, count, path?} 。
    """
    import json as _json

    try:
        from behavior_interface.skills.vlm_grasp_verify import _world_to_pixel
    except Exception as e:
        return {"ok": False, "error": f"import _world_to_pixel 失败: {e}"}

    vpath = os.path.join(agent_runs.plans_dir(session_id), "vectors.json")
    if not os.path.isfile(vpath):
        return {"ok": False, "error": "no_vectors"}
    try:
        with open(vpath, encoding="utf-8") as f:
            vectors = _json.load(f)
    except Exception as e:
        return {"ok": False, "error": f"读取 vectors.json 失败: {e}"}
    if not vectors:
        return {"ok": False, "error": "empty_vectors"}

    img = cv2.imread(rgb_path)
    if img is None:
        return {"ok": False, "error": f"读取图像失败: {rgb_path}"}
    h_img, w_img = img.shape[:2]

    cam_pos = np.asarray(cam_meta.get("pos", [0, 0, 0]), dtype=np.float64).reshape(3)
    cam_quat = np.asarray(cam_meta.get("quat", [0, 0, 0, 1]), dtype=np.float64).reshape(4)
    w = int(cam_meta.get("image_width", w_img))
    h = int(cam_meta.get("image_height", h_img))
    fl = float(cam_meta.get("focal_length", HEAD_FOCAL_LENGTH))
    ha = float(cam_meta.get("horizontal_aperture", HEAD_HORIZONTAL_APERTURE))
    sx = w_img / float(max(w, 1))
    sy = h_img / float(max(h, 1))

    def _proj(pt):
        px = _world_to_pixel(cam_pos, cam_quat, np.asarray(pt, dtype=np.float64), w, h, fl, ha)
        if px is None:
            return None
        u, v = px
        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        return (int(round(u * sx)), int(round(v * sy)))

    # BGR：红=法向/接近, 蓝=主轴, 绿=副法向
    COL_NORMAL = (0, 0, 255)
    COL_AXIS = (255, 0, 0)
    COL_BINORMAL = (0, 200, 0)
    count = 0
    for vid, rec in vectors.items():
        try:
            base = np.asarray(rec["base_world"], dtype=np.float64).reshape(3)
        except Exception:
            continue
        L = float(rec.get("length_m", 0.1))
        p0 = _proj(base)
        if p0 is None:
            continue
        n_dir = np.asarray(rec.get("dir_world", [0, 0, 0]), dtype=np.float64).reshape(3)
        a_dir = np.asarray(rec.get("axis_world", [0, 0, 0]), dtype=np.float64).reshape(3)
        b_dir = np.asarray(rec.get("binormal_world", [0, 0, 0]), dtype=np.float64).reshape(3)
        # 法向(接近向) 单向箭头
        pn = _proj(base + n_dir * L)
        if pn is not None:
            cv2.arrowedLine(img, p0, pn, COL_NORMAL, 2, tipLength=0.25)
        # 主轴 双向
        if float(np.linalg.norm(a_dir)) > 1e-6:
            pa1 = _proj(base + a_dir * (L * 0.7))
            pa2 = _proj(base - a_dir * (L * 0.7))
            if pa1 is not None and pa2 is not None:
                cv2.line(img, pa2, pa1, COL_AXIS, 2)
        # 副法向 单向短箭头
        if float(np.linalg.norm(b_dir)) > 1e-6:
            pb = _proj(base + b_dir * (L * 0.6))
            if pb is not None:
                cv2.arrowedLine(img, p0, pb, COL_BINORMAL, 2, tipLength=0.25)
        cv2.circle(img, p0, 4, (0, 255, 255), -1)
        cv2.putText(img, str(vid), (p0[0] + 6, p0[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        count += 1

    if count == 0:
        return {"ok": False, "error": "no_visible_vectors"}
    if not cv2.imwrite(out_path, img):
        return {"ok": False, "error": f"写出叠加图失败: {out_path}"}
    return {"ok": True, "count": count, "path": out_path}


@register_skill(
    "capture",
    description=(
        "渲染机器人相机并返回当前视图。主视图=head/zed 第一视角；"
        "同时落盘 head 的 depth/seg/normal 与相机内外参（供 plan_* 反投影），"
        "以及 tro / robot 位姿 / holding。返回 image_id 与主视图 data URL。"
    ),
)
def capture(
    ctx,
    session_id: str,
    save_depth: bool = True,
    save_seg: bool = True,
    save_normal: bool = True,
    n_settle: int = 2,
    include_auxiliary_rgb: bool = False,
):
    """capture 当前相机视图。一个 session 内图片顺序编号 img_NNNN，落盘并回传编号。"""
    from behavior_interface import agent_runs

    world = ctx.world
    robot = world.robot
    control = _refresh_camera_render_control()
    old_no_obs = bool(getattr(world, "_codex_fast_motion_no_obs", False))
    old_keep_cameras = bool(
        getattr(world, "_codex_keep_robot_camera_render_updates", False)
    )
    world._codex_keep_robot_camera_render_updates = True
    control.set_robot_camera_render_updates(world, True)
    world._codex_fast_motion_no_obs = True

    # —— 渲染稳定（保持 trunk 关节，勿用 empty_action 以免俯仰复位）——
    try:
        for _ in range(max(0, n_settle)):
            yield world.hold_action()
        try:
            import omnigibson as og
            for _ in range(2):
                og.sim.render()
        except Exception:
            og = None

        if robot is None:
            ctx.set_result({"ok": False, "error": "robot 未加载（dry-run？）"})
            yield world.hold_action()
            return

        sensors = _collect_robot_sensors(robot)
        head = sensors.get("head")
        camera_source = "robot_head"
        if head is None:
            fallback = _gta_fallback_sensor(world)
            if fallback is None:
                ctx.set_result({
                    "ok": False,
                    "error": "未找到 head/zed 相机",
                    "rgb_sensors": _rgb_sensor_names(robot),
                    "sensors": _sensor_debug_names(robot),
                    "external_sensors": _external_sensor_debug_names(world),
                })
                yield world.hold_action()
                return
            head = fallback
            sensors["head"] = head
            camera_source = "gta_view_fallback"
            ctx.log("capture WARN 未找到 robot head/zed，相机输入临时回退到 external gta_view")
        if _ensure_modalities(head, ["rgb"]):
            yield world.hold_action()
            if og is not None:
                for _ in range(2):
                    og.sim.render()

    # 运行中热修复 / 老进程可能没在 env.reset 时配到 head；capture 前再对齐一次官方内参。
        from behavior_interface.head_capture import configure_head_sensor, reset_head_sensor_to_mount
        try:
            if configure_head_sensor(head, env=getattr(world, "env", None)):
                ctx.log("capture head 相机已对齐 Challenge：720x720 horizontal_aperture=40")
                yield world.hold_action()
                if og is not None:
                    for _ in range(2):
                        og.sim.render()
        except Exception as e:
            ctx.log(f"capture WARN head 相机配置失败: {e}")

    # 仅当 head 与 zed_link 脱节且有出厂快照时才复位（勿每次 capture 硬写四元数）
        if reset_head_sensor_to_mount(world, ctx=ctx):
            yield world.hold_action()
            if og is not None:
                for _ in range(2):
                    og.sim.render()

    # —— 确保 head 开启 depth/seg/normal ——
        want_mods = ["rgb"]
        if save_depth:
            want_mods.append("depth_linear")
        if save_seg:
            want_mods.append("seg_instance_id")
        if save_normal:
            want_mods.append("normal")
        if _ensure_modalities(head, want_mods):
            yield world.hold_action()
            if og is not None:
                for _ in range(2):
                    og.sim.render()

    # —— 预热相机 annotator —— 若 capture 紧接 no_obs 运动而来，seg buffer
    #    可能为空，直接 get_obs 会拿到空张量。先渲染并取一次 obs 把它预热。
    #    这些渲染 tick 同时驱动纹理流式，也负责执行 vision_sensor 在 headless
    #    下延后到「首次 capture / warmup」的相机参数传播。
        warm_obs_annotators(world, ctx, log_tag="capture")
        _warm_sensor_obs(head, ctx, log_tag=f"capture.{camera_source}")

    # —— 分配 image_id + 准备目录 ——
        agent_runs.ensure_session(session_id)
        image_id = agent_runs.next_image_id(session_id)
        img_dir = agent_runs.images_dir(session_id)
        os.makedirs(img_dir, exist_ok=True)

        rgb_main_path: Optional[str] = None
        modal_paths: Dict[str, str] = {}
        rgb_feeds: Dict[str, str] = {}
        segmentation_meta: Dict[str, Any] = {}
        head_obs_for_memory: Optional[Dict[str, Any]] = None
        try:
            cam_meta = _camera_meta(head)
            cam_meta["feed"] = camera_source
        except Exception as e:
            ctx.log(f"capture WARN 取 head 内外参失败: {e}")
            cam_meta = {"feed": camera_source}

    # —— 逐相机取 obs 并落盘 ——
    # 注意：这里到 meta 落盘之间不能 yield / step。否则 move_to_object 后若机身
    # 仍有轻微抖动，图片/depth 和 camera pose 会错帧，点击反投影会穿到地面。
        for feed, sensor in _capture_sensor_items(
            sensors,
            include_auxiliary_rgb=include_auxiliary_rgb,
        ):
            required_modalities = ["rgb"]
            if feed == "head":
                if save_depth:
                    required_modalities.append("depth_linear")
                if save_seg:
                    required_modalities.append("seg_instance_id")
                if save_normal:
                    required_modalities.append("normal")
            obs, info, frame_errors = _read_complete_frame(
                sensor,
                required_modalities,
                ctx,
                log_tag=f"capture.{feed}",
                n_attempts=4 if feed == "head" else 1,
                allow_rebuild=feed == "head",
            )
            if frame_errors:
                ctx.log(
                    f"capture WARN {feed} 连续预热后仍无完整帧，跳过: "
                    + "; ".join(frame_errors)
                )
                continue
            suffix = ".png" if feed == "head" else f".{feed}.png"
            rgb_p = agent_runs.image_path(session_id, image_id, suffix)
            if _save_rgb(obs, rgb_p):
                rgb_feeds[feed] = rgb_p
                if feed == "head":
                    rgb_main_path = rgb_p
            else:
                ctx.log(f"capture WARN {feed} RGB 无有效像素，跳过该帧")
            if feed == "head":
                head_obs_for_memory = obs
                if save_depth:
                    dnpy = agent_runs.image_path(session_id, image_id, ".depth.npy")
                    dvis = agent_runs.image_path(session_id, image_id, ".depth.png")
                    if _save_depth(obs, dnpy, dvis) is not None:
                        modal_paths["depth"] = dnpy
                if save_seg:
                    snpy = agent_runs.image_path(session_id, image_id, ".seg.npy")
                    svis = agent_runs.image_path(session_id, image_id, ".seg.png")
                    if _save_seg(obs, info, snpy, svis) is not None:
                        modal_paths["seg"] = snpy
                        segmentation_meta.update(_robot_seg_ids_from_info(info))
                if save_normal:
                    nrm = obs.get("normal")
                    if nrm is not None:
                        nnpy = agent_runs.image_path(session_id, image_id, ".normal.npy")
                        if save_npy_atomic(nnpy, _to_np(nrm).astype(np.float32)):
                            modal_paths["normal"] = nnpy

        if rgb_main_path is None:
            ctx.set_result({"ok": False, "error": "head RGB 渲染失败"})
            yield world.hold_action()
            return

    # —— 结构化状态（层 A）——
        try:
            tro = world.task_relevant_state()
        except Exception:
            tro = {}
        try:
            base = world.robot_pose()
            base_pose = {"x": float(base.pos[0]), "y": float(base.pos[1]),
                         "z": float(base.pos[2]), "yaw_deg": float(np.degrees(base.yaw))}
        except Exception:
            base_pose = {}
        try:
            chest = world.chest_pose()
        except Exception:
            chest = {}
        robot_state = {
            "base_pose": base_pose,
            "chest_pose": chest,
            "eef_left": world.eef_pose(arm="left"),
            "eef_right": world.eef_pose(arm="right"),
            "holding": _holding(robot),
        }

    # —— image-bound memory：与本次 capture 的 image_id/RGB/depth/camera pose 同源 ——
        memory_payload = None
        memory_text = None
        try:
            from behavior_interface.memory import (
                build_memory,
                build_task_goals_only_memory,
                format_memory,
                task_goals_only_memory_enabled,
            )

            task_name = str(getattr(ctx, "task_name", "") or "")
            if task_goals_only_memory_enabled():
                # The server adds its current BDDL cache before this payload is
                # exposed. Do not compute privileged object projections here.
                memory_payload = build_task_goals_only_memory(task_name=task_name)
            else:
                depth_for_memory = None
                if isinstance(head_obs_for_memory, dict):
                    depth_for_memory = head_obs_for_memory.get("depth_linear")
                memory_payload = build_memory(
                    world,
                    task_name=task_name,
                    robot_pose=base_pose,
                    tro=tro,
                    tick=None,
                    head=head,
                    head_camera=cam_meta,
                    depth_linear=depth_for_memory,
                    image_id=image_id,
                )
                memory_payload.setdefault("image_binding", {})
                memory_payload["image_binding"].update({
                    "image_id": image_id,
                    "rgb_main_path": rgb_main_path,
                    "camera_source": camera_source,
                    "guarantee": "UVD projected from the same capture camera pose and depth buffer as this image_id",
                })
            memory_text = format_memory(memory_payload)
        except Exception as e:
            ctx.log(f"capture WARN image-bound memory build failed: {e}")

    # —— 落盘 meta（plan_* 反投影时按 image_id 取回）——
        meta = {
            "image_id": image_id,
            "session_id": session_id,
            "rgb": {k: os.path.basename(v) for k, v in rgb_feeds.items()},
            "modalities": {k: os.path.basename(v) for k, v in modal_paths.items()},
            "camera": cam_meta,
            "tro": tro,
            "robot": robot_state,
        }
        if segmentation_meta:
            meta["segmentation"] = segmentation_meta
        rgb_display_path = rgb_main_path
        base_path_overlay = None
        try:
            overlay_path = agent_runs.image_path(session_id, image_id, ".path.png")
            base_path_overlay = render_base_forward_path_overlay(
                rgb_path=rgb_main_path,
                depth_path=modal_paths.get("depth"),
                meta=meta,
                out_path=overlay_path,
                seg_path=modal_paths.get("seg"),
                show_depth_conflict_overlay=CAPTURE_HEAD_RED_OBSTACLE_OVERLAY_ENABLED,
            )
            if base_path_overlay and base_path_overlay.get("ok"):
                rgb_display_path = overlay_path
                meta.setdefault("overlays", {})["base_forward_path"] = os.path.basename(overlay_path)
            else:
                ctx.log(
                    "capture WARN base path overlay failed: "
                    f"{(base_path_overlay or {}).get('error', 'unknown')}"
                )
        except Exception as e:
            base_path_overlay = {"ok": False, "error": str(e)}
            ctx.log(f"capture WARN base path overlay exception: {e}")

        # —— manipulate 向量坐标系叠加（若存在 vectors.json，绿/红/蓝重投影到当前视图）——
        vectors_overlay = None
        try:
            sess_id = session_id
            vec_out = agent_runs.image_path(sess_id, image_id, ".vectors.png")
            vectors_overlay = render_vectors_overlay(
                sess_id, rgb_display_path, cam_meta, vec_out,
            )
            if vectors_overlay and vectors_overlay.get("ok"):
                rgb_display_path = vec_out
                meta.setdefault("overlays", {})["vectors"] = os.path.basename(vec_out)
        except Exception as e:
            vectors_overlay = {"ok": False, "error": str(e)}
            ctx.log(f"capture WARN vectors overlay exception: {e}")

        agent_runs.save_image_meta(session_id, image_id, meta)

        w_native = int(cam_meta.get("image_width", 0)) or None
        h_native = int(cam_meta.get("image_height", 0)) or None
        ctx.set_result({
            "ok": True,
            "tool": "capture",
            "image_id": image_id,
            "image_width": w_native,
            "image_height": h_native,
            "rgb_main": agent_runs.file_to_data_url(rgb_display_path),
            "rgb_main_path": rgb_main_path,
            "rgb_overlay_path": rgb_display_path if rgb_display_path != rgb_main_path else None,
            "camera": cam_meta,
            "tro": tro,
            "robot": robot_state,
            "modalities": list(modal_paths.keys()),
            "base_path_overlay": base_path_overlay,
            "vectors_overlay": vectors_overlay,
            "memory": memory_payload,
            "memory_text": memory_text,
        })
        yield world.hold_action()
    finally:
        world._codex_fast_motion_no_obs = old_no_obs
        world._codex_keep_robot_camera_render_updates = old_keep_cameras
