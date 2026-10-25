"""实时建图：控制 tick 级里程 + 5Hz head RGB-D 融合。

只用官方 evaluator 允许的观测：
- 严格官方模式优先复用策略层用 proprio ``base_qvel`` 校正后的
  ``policy_local_base_pose()`` 相邻增量；普通模式才直接积分机体系
  ``world.base_qvel()``。两条路径都不读 ``WorldAPI.robot_pose``。
- head RGB/``depth_linear`` + 由 trunk proprio 与出厂安装位姿推出的相机
  机体系位姿。相邻 RGB-D 只校正相同时段的 proprio 增量。

不读 scene graph、房间名、分割、接触。整个模块的每个入口都吞异常并在
连续失败后自我禁用，绝不让建图拖垮仿真主循环。
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import numpy as np


# 冻结 benchmark 证明漏掉静止/刹车后的帧会破坏后续几何对齐；只要 evaluator
# 提供新观测，运动和静止都维持约 5Hz。
MOVING_INTERVAL_S = 0.20
IDLE_INTERVAL_S = 0.20
# 判定「在动」的阈值，低于此值认为是噪声
MOVING_SPEED_EPS_MPS = 0.02
MOVING_YAW_EPS_DPS = 1.0
# 停止后仍按移动频率再建图一会儿，把刹车后的视野补全
MOVING_TAIL_S = 0.6
# 两次深度融合之间几乎没有线位移时，只允许墙方向纠 yaw，不让栅格匹配
# 注入平移。阈值远高于 proprio 静止噪声、远低于移动帧间的正常位移。
TRANSLATION_MATCH_MIN_M = 0.01
# 扫描匹配要跟得上漂移：隔太久（原来 12 帧约 2.4s）误差早就超出搜索窗，
# 再也拉不回来，远处的墙就被一层层描成重影。
SCAN_MATCH_EVERY = 4
# 实时路径点数远多于回放，抽稀一点省 CPU
LIVE_DEPTH_STRIDE = 6
MAX_FAILURES = 8


class LiveMapper:
    """挂在 sim 主循环上的实时建图器。所有方法都必须在 sim 主线程调用。"""

    def __init__(self) -> None:
        self.disabled = False
        self.failures = 0
        self.integrated = 0
        self.last_integrate_ts = 0.0
        self.last_moving_ts = 0.0
        self.last_error = ""
        self._enabled_cache: Optional[bool] = None
        self._mount: Optional[tuple] = None
        self._last_policy_pose: Optional[np.ndarray] = None
        self._motion_since_rgbd = (0.0, 0.0, 0.0)
        self._last_rgbd_map_pose: Optional[tuple[float, float, float]] = None
        self._last_evaluator_sequence: Optional[int] = None
        self._rgbd = None
        self._translation_since_map_m = 0.0
        self._turn_since_map_deg = 0.0
        self.odometry_source = "uninitialized"
        self.policy_pose_ticks = 0
        self.qvel_fallback_ticks = 0
        self.rgbd_corrections = 0
        self.rgbd_errors = 0
        # 结构冻结后仍运行相邻 RGB-D 里程，只禁止占据栅格写入。
        self.rgbd_localization_skips = 0
        self.rgbd_localization_updates = 0

    # ------------------------------------------------------------------ 开关

    def _enabled(self) -> bool:
        if self.disabled:
            return False
        if self._enabled_cache is None:
            try:
                from behavior_interface.spatial_map import spatial_map_enabled

                self._enabled_cache = bool(spatial_map_enabled())
            except Exception:
                self._enabled_cache = False
        return bool(self._enabled_cache)

    def _fail(self, error: str) -> None:
        self.failures += 1
        self.last_error = str(error)
        if self.failures >= MAX_FAILURES:
            self.disabled = True

    def reset(self) -> None:
        """episode 重置：清掉节流状态，下一帧立刻重新开始建图。"""
        self.last_integrate_ts = 0.0
        self.last_moving_ts = 0.0
        self.integrated = 0
        self.failures = 0
        self._mount = None
        self._last_policy_pose = None
        self._motion_since_rgbd = (0.0, 0.0, 0.0)
        self._last_rgbd_map_pose = None
        self._last_evaluator_sequence = None
        if self._rgbd is not None:
            self._rgbd.reset()
        self._translation_since_map_m = 0.0
        self._turn_since_map_deg = 0.0
        self.odometry_source = "uninitialized"
        self.policy_pose_ticks = 0
        self.qvel_fallback_ticks = 0
        self.rgbd_corrections = 0
        self.rgbd_errors = 0
        self.rgbd_localization_skips = 0
        self.rgbd_localization_updates = 0

    def _ego(self):
        from behavior_interface.spatial_map import (
            active_session_id,
            get_map,
            mark_active_session,
        )

        ego = get_map(active_session_id() or "default")
        # 让后来的 session（UI 轮询 / agent 工具）能接管到这张图，而不是另起空图
        mark_active_session(ego.session_id)
        return ego

    # ------------------------------------------------------------------ 里程

    @staticmethod
    def _wrap_rad(value: float) -> float:
        return math.atan2(math.sin(float(value)), math.cos(float(value)))

    @classmethod
    def _compose_body_motion(
        cls,
        accumulated: tuple[float, float, float],
        delta: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """把两个依次发生的机体系 SE(2) 增量合成到第一帧机体系。"""
        x, y, yaw_deg = [float(value) for value in accumulated]
        forward, left, turn_deg = [float(value) for value in delta]
        yaw = math.radians(yaw_deg)
        return (
            x + math.cos(yaw) * forward - math.sin(yaw) * left,
            y + math.sin(yaw) * forward + math.cos(yaw) * left,
            math.degrees(cls._wrap_rad(math.radians(yaw_deg + turn_deg))),
        )

    @classmethod
    def _compose_body_twist(
        cls,
        accumulated: tuple[float, float, float],
        delta: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """把一个同时含平移/角速度的 tick 先精确积分再累加。

        ``_compose_body_motion`` 还要兼容已经是有限相对位姿的调用方；
        qvel 路径单独走这里，避免把同一个旋转积分两次。
        """
        from behavior_interface.spatial_map import body_twist_delta

        return cls._compose_body_motion(
            accumulated,
            body_twist_delta(*delta),
        )

    @classmethod
    def _compose_map_pose(
        cls,
        pose: tuple[float, float, float],
        delta: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """把机体系增量落到一个地图系起始位姿上。"""
        x, y, yaw_deg = [float(value) for value in pose]
        forward, left, turn_deg = [float(value) for value in delta]
        yaw = math.radians(yaw_deg)
        return (
            x + math.cos(yaw) * forward - math.sin(yaw) * left,
            y + math.sin(yaw) * forward + math.cos(yaw) * left,
            math.degrees(cls._wrap_rad(math.radians(yaw_deg + turn_deg))),
        )

    def _rgbd_tracker(self):
        if self._rgbd is None:
            from behavior_interface.rgbd_odometry import RgbdOdometry

            # 在线主循环不能为长期定位保留 3000 个特征；相邻里程在
            # 1200--1500 个稳定角点下已经足够，长期关键帧另有独立上限。
            self._rgbd = RgbdOdometry(nfeatures=1400)
        return self._rgbd

    def _policy_pose_delta(self, world) -> Optional[tuple[float, float, float]]:
        """读取官方策略层已经校正过的 local odometry，相邻帧求机体系增量。

        返回 ``None`` 表示当前 world 没有这条官方接口，调用方应走 qvel 兼容
        路径。第一帧只建立参考，返回零增量，不能把 episode 开始前的路径搬进图。
        """
        provider = getattr(world, "policy_local_base_pose", None)
        if not callable(provider):
            return None
        pose = np.asarray(provider(), dtype=np.float64).reshape(-1)
        if pose.size < 3 or not np.all(np.isfinite(pose[:3])):
            raise ValueError("policy_local_base_pose 无效")
        current = pose[:3].copy()
        previous = self._last_policy_pose
        self._last_policy_pose = current
        self.odometry_source = "policy_local_base_pose"
        self.policy_pose_ticks += 1
        if previous is None:
            return (0.0, 0.0, 0.0)
        dx, dy = float(current[0] - previous[0]), float(current[1] - previous[1])
        yaw0 = float(previous[2])
        cos_y, sin_y = math.cos(yaw0), math.sin(yaw0)
        return (
            cos_y * dx + sin_y * dy,
            -sin_y * dx + cos_y * dy,
            math.degrees(self._wrap_rad(float(current[2] - yaw0))),
        )

    def odom_tick(self, world, dt: float) -> None:
        """每个物理子步调用一次：把官方 local odometry 增量并进地图。

        严格官方 world 已经用同一份 proprio 对 ``policy_local_base_pose`` 做过
        command prediction + observation reconciliation，地图直接取它的相邻增量，
        避免另起一套带死区的积分器。普通模式没有该接口时仍兼容 base_qvel。
        这仍然不是 GT；朝向漂移仍要由墙方向和回环修正。
        """
        if not self._enabled():
            return
        try:
            step = float(dt)
            if not math.isfinite(step) or step <= 0.0 or step > 0.5:
                return
            delta = self._policy_pose_delta(world)
            policy_delta = delta is not None
            if delta is None:
                qvel = np.asarray(world.base_qvel(), dtype=np.float64).reshape(3)
                if not np.all(np.isfinite(qvel)):
                    return
                forward = float(qvel[0]) * step
                left = float(qvel[1]) * step
                yaw_deg = math.degrees(float(qvel[2])) * step
                self.odometry_source = "base_qvel_fallback"
                self.qvel_fallback_ticks += 1
            else:
                forward, left, yaw_deg = delta
            # policy_local_base_pose 返回相邻帧的有限 SE(2) 增量；base_qvel
            # 返回速度 tick。两者的平移表达式相同，但后者必须先做一次
            # 常速度刚体积分，不能把已经有限的 policy 增量再积分一遍。
            if policy_delta:
                relative_delta = (forward, left, yaw_deg)
                self._motion_since_rgbd = self._compose_body_motion(
                    self._motion_since_rgbd,
                    relative_delta,
                )
            else:
                from behavior_interface.spatial_map import body_twist_delta

                relative_delta = body_twist_delta(forward, left, yaw_deg)
                self._motion_since_rgbd = self._compose_body_motion(
                    self._motion_since_rgbd,
                    relative_delta,
                )
            distance = math.hypot(relative_delta[0], relative_delta[1])
            speed = distance / step
            yaw_dps = abs(relative_delta[2]) / step
            if speed < MOVING_SPEED_EPS_MPS and yaw_dps < MOVING_YAW_EPS_DPS:
                moving = False
            else:
                moving = True
                self.last_moving_ts = time.time()
            if distance > 1e-9 or abs(relative_delta[2]) > 1e-9:
                self._translation_since_map_m += distance
                self._turn_since_map_deg += abs(relative_delta[2])
                ego = self._ego()
                if policy_delta:
                    advance = getattr(ego, "advance_relative_odometry", None)
                    if not callable(advance):
                        advance = ego.advance_odometry
                    advance(forward, left, yaw_deg)
                else:
                    advance = getattr(ego, "advance_body_twist", None)
                    if not callable(advance):
                        # 兼容旧的测试替身；正式 EgoMap 始终有该入口。
                        advance = ego.advance_odometry
                    advance(forward, left, yaw_deg)
            elif not moving:
                return
        except Exception as exc:
            self._fail(f"odom_tick: {exc}")

    # ------------------------------------------------------------------ 建图

    def map_tick(self, world, now: Optional[float] = None) -> bool:
        """按约 5Hz 读一帧 head RGB-D 并并入占用栅格。"""
        if not self._enabled():
            return False
        now = float(now if now is not None else time.time())
        moving = (now - self.last_moving_ts) <= MOVING_TAIL_S
        interval = MOVING_INTERVAL_S if moving else IDLE_INTERVAL_S
        if (now - self.last_integrate_ts) < interval:
            return False
        self.last_integrate_ts = now
        try:
            bundle = self._read_bundle(world)
            if bundle is None:
                return False
            sequence = bundle.get("evaluator_sequence")
            if sequence is not None:
                sequence = int(sequence)
                if sequence == self._last_evaluator_sequence:
                    return False
                self._last_evaluator_sequence = sequence
            ego = self._ego()
            # 冻结只停止结构栅格写入，不停止视觉里程。相邻 RGB-D 负责连续
            # 传播，长期关键帧负责消除累计漂移；OpenCV 线程数在视觉前端被
            # 硬限制，不能再取满整机核心。
            mapping_state = getattr(ego, "mapping_state", "mapping")
            rgbd_consumed = self._apply_rgbd_odometry(ego, bundle)
            if mapping_state == "localization" and rgbd_consumed:
                self.rgbd_localization_updates += 1
                self.odometry_source = f"{self.odometry_source}_localization"
            image_features = (
                None
                if self._rgbd is None
                else getattr(self._rgbd, "last_features", None)
            )
            if rgbd_consumed:
                # 即使后面的栅格融合抛异常，tracker 也已经前进到当前帧；先留
                # 下匹配前锚点，成功融合后再用扫描匹配后的最终位姿覆盖。
                self._last_rgbd_map_pose = (ego.x, ego.y, ego.yaw_deg)
            self.integrated += 1
            image_id = f"live-{sequence if sequence is not None else self.integrated}"
            # 先做只读重访探测，再决定本帧能否写结构。可靠重访如果在这一
            # 帧触发冻结，当前深度绝不能先落进长期栅格。
            observe_visual = getattr(ego, "observe_visual_frame", None)
            commit_visual = getattr(ego, "commit_visual_keyframe", None)
            split_visual_commit = callable(commit_visual)
            if callable(observe_visual):
                observe_visual(
                    bundle,
                    image_id,
                    self.integrated,
                    image_features=image_features,
                    **(
                        {"commit_mapping_keyframe": False}
                        if split_visual_commit
                        else {}
                    ),
                )

            # 兼容只实现 integrate_capture 的轻量测试替身；真实 EgoMap 总有
            # mapping_state、observe_visual_frame 和 commit_visual_keyframe。
            structural_ok = False
            if getattr(ego, "mapping_state", "mapping") == "mapping":
                do_match = (self.integrated % SCAN_MATCH_EVERY) == 0
                match_translation = (
                    self._translation_since_map_m >= TRANSLATION_MATCH_MIN_M
                )
                structural_ok = ego.integrate_capture(
                    bundle,
                    image_id=image_id,
                    stride=LIVE_DEPTH_STRIDE,
                    scan_match=do_match,
                    match_translation=match_translation,
                )
            if structural_ok and split_visual_commit:
                commit_visual(
                    bundle,
                    image_id,
                    self.integrated,
                    image_features=image_features,
                )
            # 视觉证据可能在本帧刚触发冻结。结构帧已经写过时不重复；后续帧
            # 只把深度写入短时细节层，结构摘要和 submap 保持只读。
            detail_ok = False
            integrate_detail = getattr(ego, "integrate_detail_capture", None)
            if (
                not structural_ok
                and getattr(ego, "mapping_state", "mapping") == "localization"
                and callable(integrate_detail)
            ):
                detail_ok = bool(integrate_detail(
                    bundle,
                    stride=LIVE_DEPTH_STRIDE,
                ))
            if structural_ok:
                self.failures = 0
                self._translation_since_map_m = 0.0
                self._turn_since_map_deg = 0.0
                # 建了图就算有内容：起点在这一刻确立，别等第一次底盘动作
                ego.ensure_start()
            if rgbd_consumed:
                # 扫描匹配可能又移动了当前位姿；下一段视觉相对运动从最终
                # 融合位姿出发，不能从匹配前的临时位姿出发。
                self._last_rgbd_map_pose = (ego.x, ego.y, ego.yaw_deg)
            return bool(structural_ok or detail_ok)
        except Exception as exc:
            self._fail(f"map_tick: {exc}")
            return False

    def _apply_rgbd_odometry(self, ego, bundle: Dict[str, Any]) -> bool:
        """用相邻 RGB-D 替换自上一视觉帧以来的 qvel 终点。

        返回 True 表示 tracker 已消费当前帧；无 RGB 时保留累计运动，等下一
        张完整帧一起估计。视觉证据不足时 tracker 会返回 None，此时 qvel 已经
        把在线位姿推进到正确的回退终点。
        """
        rgb = bundle.get("rgb")
        depth = bundle.get("depth")
        camera = bundle.get("camera")
        if rgb is None or depth is None or not isinstance(camera, dict):
            return False
        try:
            tracker = self._rgbd_tracker()
            estimate = tracker.update(
                rgb,
                depth,
                camera,
                qvel_delta=self._motion_since_rgbd,
            )
        except Exception as exc:
            self.rgbd_errors += 1
            self.last_error = f"rgbd_odometry: {exc}"
            if self._rgbd is not None:
                self._rgbd.reset()
            self._last_rgbd_map_pose = None
            self._motion_since_rgbd = (0.0, 0.0, 0.0)
            self.odometry_source = "rgbd_error_qvel_fallback"
            return True

        if estimate is not None and self._last_rgbd_map_pose is not None:
            desired = self._compose_map_pose(
                self._last_rgbd_map_pose,
                (estimate.forward_m, estimate.left_m, estimate.yaw_deg),
            )
            ego.correct_live_pose(*desired)
            self.rgbd_corrections += 1
            self.odometry_source = "rgbd"
        else:
            self.odometry_source = "rgbd_qvel_fallback"
        self._motion_since_rgbd = (0.0, 0.0, 0.0)
        return True

    def _read_bundle(self, world) -> Optional[Dict[str, Any]]:
        """读一帧 head RGB-D，组出反投影所需的机体系相机位姿与内参。

        严格官方模式下 interface 不拥有仿真器，没有相机 sensor 可读，
        depth 和相机相对位姿都由 evaluator 随观测推过来（走 adapter）。
        """
        adapter = getattr(world, "_official_adapter", None)
        if adapter is not None:
            return self._read_bundle_from_adapter(adapter)
        return self._read_bundle_from_sensor(world)

    def _read_bundle_from_adapter(self, adapter) -> Optional[Dict[str, Any]]:
        """官方模式：从 evaluator 观测里取 head RGB-D + cam_rel_pose。"""
        from behavior_interface.head_capture import (
            HEAD_FOCAL_LENGTH,
            HEAD_HORIZONTAL_APERTURE,
        )

        depth_provider = getattr(adapter, "camera_depth_frame", None)
        depth = (
            depth_provider("head")
            if callable(depth_provider)
            else adapter.camera_depth_frames().get("head")
        )
        bgr = adapter.camera_frame("head")
        relative = (adapter.camera_relative_poses() or {}).get("head")
        if depth is None or bgr is None or not relative:
            return None
        arr = np.asarray(depth).squeeze()
        bgr_arr = np.asarray(bgr)
        if arr.ndim != 2 or bgr_arr.ndim != 3 or bgr_arr.shape[2] < 3:
            return None
        height, width = arr.shape
        if bgr_arr.shape[:2] != (height, width):
            return None
        camera = {
            "focal_length": float(HEAD_FOCAL_LENGTH),
            "horizontal_aperture": float(HEAD_HORIZONTAL_APERTURE),
            "image_width": int(width),
            "image_height": int(height),
            "robot_relative_pose": {
                "pos": list(relative.get("pos") or []),
                "quat": list(relative.get("quat") or []),
            },
        }
        if len(camera["robot_relative_pose"]["pos"]) != 3:
            return None
        if len(camera["robot_relative_pose"]["quat"]) != 4:
            return None
        rgb = np.ascontiguousarray(bgr_arr[:, :, 2::-1]).astype(
            np.uint8,
            copy=False,
        )
        bundle = {
            "rgb": rgb,
            "depth": np.ascontiguousarray(arr, dtype=np.float32),
            "camera": camera,
        }
        metadata = getattr(adapter, "observation_metadata", None)
        if callable(metadata):
            try:
                sequence, _received_ts = metadata()
                bundle["evaluator_sequence"] = int(sequence)
            except Exception:
                pass
        return bundle

    def _read_bundle_from_sensor(self, world) -> Optional[Dict[str, Any]]:
        """普通模式：从同一次 head.get_obs 读取 RGB + depth_linear。"""
        from behavior_interface.head_capture import (
            get_head_sensor,
            head_factory_mount_pose,
            head_intrinsics_dict,
        )
        from behavior_interface.skills.reach_point_pitch_recovery import (
            head_camera_pose_robot_from_proprio,
        )

        head = get_head_sensor(world)
        if head is None:
            return None
        try:
            observation, _info = head.get_obs()
        except Exception:
            return None
        depth = observation.get("depth_linear")
        rgb = observation.get("rgb")
        if depth is None or rgb is None:
            try:
                head.add_modality("depth_linear")
            except Exception:
                pass
            return None

        def to_numpy(value):
            detach = getattr(value, "detach", None)
            if callable(detach):
                value = detach().cpu().numpy()
            return np.asarray(value)

        depth_arr = to_numpy(depth).squeeze()
        rgb_arr = to_numpy(rgb)
        if depth_arr.ndim != 2 or rgb_arr.ndim != 3 or rgb_arr.shape[2] < 3:
            return None
        if rgb_arr.shape[:2] != depth_arr.shape:
            return None
        mount = self._mount or head_factory_mount_pose(world)
        if mount is None:
            return None
        self._mount = mount
        camera_relative = head_camera_pose_robot_from_proprio(
            np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4),
            camera_parent_pos=mount[0],
            camera_parent_quat=mount[1],
        )
        camera = dict(head_intrinsics_dict(head))
        camera["image_width"] = int(depth_arr.shape[1])
        camera["image_height"] = int(depth_arr.shape[0])
        camera["robot_relative_pose"] = camera_relative
        return {
            "rgb": np.ascontiguousarray(rgb_arr[:, :, :3]),
            "depth": np.ascontiguousarray(depth_arr, dtype=np.float32),
            "camera": camera,
        }

    # ------------------------------------------------------------------ 诊断

    def status(self) -> Dict[str, Any]:
        from behavior_interface.spatial_map import BACKEND, BUILD

        tracker = self._rgbd
        last_motion = None if tracker is None else getattr(
            tracker, "last_motion", None
        )
        ego_query = self._ego().query()
        return {
            "backend": BACKEND,
            "build": BUILD,
            "enabled": self._enabled(),
            "disabled": self.disabled,
            "integrated": self.integrated,
            "failures": self.failures,
            "last_error": self.last_error,
            "odometry_source": self.odometry_source,
            "rgbd_corrections": self.rgbd_corrections,
            "rgbd_errors": self.rgbd_errors,
            "rgbd_localization_skips": self.rgbd_localization_skips,
            "rgbd_localization_updates": self.rgbd_localization_updates,
            "rgbd_accepted": 0 if tracker is None else tracker.accepted,
            "rgbd_rejected": 0 if tracker is None else tracker.rejected,
            "rgbd_last_reason": "" if tracker is None else tracker.last_reason,
            "rgbd_observability": (
                None
                if last_motion is None
                else {
                    "geometry_major_span_m": last_motion.geometry_major_span_m,
                    "geometry_minor_span_m": last_motion.geometry_minor_span_m,
                    "pixel_coverage_x": last_motion.pixel_coverage_x,
                    "pixel_coverage_y": last_motion.pixel_coverage_y,
                    "translation_std_m": last_motion.translation_std_m,
                    "yaw_std_deg": last_motion.yaw_std_deg,
                    "normal_matrix_condition": last_motion.normal_matrix_condition,
                }
            ),
            "feature_backend": (
                "uninitialized"
                if tracker is None
                else str(getattr(tracker, "feature_backend", "unknown"))
            ),
            "feature_device": (
                "uninitialized"
                if tracker is None
                else str(getattr(tracker, "feature_device", "unknown"))
            ),
            "feature_fallback_reason": (
                ""
                if tracker is None
                else str(getattr(tracker, "feature_fallback_reason", ""))
            ),
            "matching_backend": (
                "uninitialized"
                if tracker is None
                else str(getattr(tracker, "matching_backend", "unknown"))
            ),
            "matching_device": (
                "uninitialized"
                if tracker is None
                else str(getattr(tracker, "matching_device", "unknown"))
            ),
            "matching_fallback_reason": (
                ""
                if tracker is None
                else str(getattr(tracker, "matching_fallback_reason", ""))
            ),
            "ransac_backend": (
                "uninitialized"
                if tracker is None
                else str(getattr(tracker, "ransac_backend", "unknown"))
            ),
            "ransac_device": (
                "uninitialized"
                if tracker is None
                else str(getattr(tracker, "ransac_device", "unknown"))
            ),
            "ransac_fallback_reason": (
                ""
                if tracker is None
                else str(getattr(tracker, "ransac_fallback_reason", ""))
            ),
            "last_evaluator_sequence": self._last_evaluator_sequence,
            "mapping": ego_query.get("mapping", {}),
            "localization": ego_query.get("localization", {}),
            "scan_matching": ego_query.get("scan_matching", {}),
        }
