"""点 record 时把 evaluator 观测流连续落盘成可直接重放的 capture bundle。

原先的人工录制只在 tool call 边界留关键帧：运动区间既没有逐 tick 的
``base_qvel``，也没有中间的 RGB-D，离线 RTAB-Map 只看得到孤立快照，
只能判定为无法连续重放。

真正的原因不是观测不存在。严格官方模式下 evaluator 在运动期间按约 37Hz
持续推观测，但 interface 的建图钩子挂在 ``fast_no_obs`` 分支之外，
底盘一动就整段跳过——最该建图的时段反而一帧不读。所以这里不新增任何
观测源，只把仿真主线程上本来就在流动的两条合规产物接出来落盘：

- 每个物理子步的 ``base_qvel``（本体感知，``odom_tick`` 已在用）；
- 每次观测更新的 head RGB-D 与 ``cam_rel_pose``（``map_tick`` 已在用）。

线程模型
--------
Flask 线程只写 start/stop 请求标志；仿真主线程独占时钟与待写队列；
PNG 压缩和落盘都在后台线程，避免把物理步拖慢。

时间基准
--------
相机时间戳一律取累积仿真时间（各 tick 的 ``dt`` 之和），不用墙钟。
仿真跑不到实时，墙钟和 dt 累积能差出几倍，而 ``CaptureBundleWriter``
要求帧间隔与该区间内 tick 时长在 8ms 内自洽——用累积仿真时间可以让它恒等。
"""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

_TICK_MAX_DT_S = 1.0
_SENTINEL = object()


def continuous_capture_enabled() -> bool:
    """默认开启；只有显式设 0/false/no/off 才关掉。"""
    raw = str(os.environ.get("BEHAVIOR_CAPTURE_CONTINUOUS", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _env_enabled(name: str, default: bool) -> bool:
    fallback = "1" if default else "0"
    raw = str(os.environ.get(name, fallback)).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or value < low or value > high:
        return default
    return value


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return default if value < low or value > high else value


def _set_current_thread_nice(
    env_name: str,
    *,
    default: int = 19,
) -> None:
    """Lower only the native worker thread's scheduling priority.

    ``os.nice`` is thread-scoped on the Linux/NPTL runtime used by the
    simulator.  Querying the current value first makes this monotonic: a
    deployment can never accidentally *raise* a background worker's priority,
    and the evaluator thread is never touched.
    """

    target = _env_int(env_name, default, 0, 19)
    try:
        getpriority = getattr(os, "getpriority", None)
        if callable(getpriority):
            current = int(
                getpriority(getattr(os, "PRIO_PROCESS", 0), 0)
            )
            if target <= current:
                return
            os.nice(target - current)
            return
        # Non-Linux fallback: keep the best-effort behavior available on
        # platforms that expose ``nice`` but not per-thread priority queries.
        os.nice(target)
    except (AttributeError, OSError, TypeError, ValueError):
        pass


@dataclass(frozen=True)
class _PendingFrame:
    """一帧待落盘的官方观测，连同它入口区间内的全部 qvel tick。

    数组都已与仿真解耦（copy 过），可以安全交给后台线程。
    """

    sequence: int
    timestamp_s: float
    rgb: np.ndarray
    depth_m: np.ndarray
    camera: dict
    sampled_base_qvel: tuple[float, float, float]
    ticks: tuple[tuple[float, tuple[float, float, float]], ...]


def _as_qvel(value: Any) -> Optional[tuple[float, float, float]]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _to_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return np.asarray(value)


def _intrinsics_from_camera(camera: dict) -> Optional[dict]:
    """把 focal_length/horizontal_aperture 换算成 fx/fy/cx/cy。

    口径与 ``spatial_map`` 的反投影一致，否则录下来的帧离线重放时会和
    在线地图对不上。
    """
    try:
        width = int(camera.get("image_width") or 0)
        height = int(camera.get("image_height") or 0)
        focal = float(camera.get("focal_length") or 0.0)
        aperture = float(camera.get("horizontal_aperture") or 0.0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or focal <= 0.0 or aperture <= 1e-9:
        return None
    fx = focal * float(width) / aperture
    if not math.isfinite(fx) or fx <= 0.0:
        return None
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fx,
        "cx": width * 0.5,
        "cy": height * 0.5,
    }


def peek_sequence(world) -> Optional[int]:
    """只读 evaluator 序号，不搬运图像。

    帧节流命中之前不该去 copy 720x720 的 RGB-D，那是纯浪费。
    """
    adapter = getattr(world, "_official_adapter", None)
    if adapter is None:
        return None
    try:
        sequence, _received_ts = adapter.observation_metadata()
    except Exception:
        return None
    return int(sequence)


def read_frame(world) -> Optional[tuple[int, np.ndarray, np.ndarray, dict]]:
    """读一帧 head RGB-D + 相机外参 + evaluator 序号。"""
    adapter = getattr(world, "_official_adapter", None)
    if adapter is not None:
        return _read_frame_from_adapter(adapter)
    return _read_frame_from_sensor(world)


def _read_frame_from_adapter(adapter) -> Optional[tuple[int, np.ndarray, np.ndarray, dict]]:
    """严格官方模式：interface 不拥有仿真器，观测全部由 evaluator 推过来。"""
    from behavior_interface.head_capture import (
        HEAD_FOCAL_LENGTH,
        HEAD_HORIZONTAL_APERTURE,
    )

    try:
        sequence, _received_ts = adapter.observation_metadata()
        depth = adapter.camera_depth_frame("head")
        bgr = adapter.camera_frame("head")
        relative = (adapter.camera_relative_poses() or {}).get("head")
    except Exception:
        return None
    if depth is None or bgr is None or not relative:
        return None
    depth_arr = np.asarray(depth).squeeze()
    bgr_arr = np.asarray(bgr)
    if depth_arr.ndim != 2 or bgr_arr.ndim != 3 or bgr_arr.shape[2] < 3:
        return None
    height, width = depth_arr.shape
    if bgr_arr.shape[0] != height or bgr_arr.shape[1] != width:
        return None
    position = list(relative.get("pos") or [])
    quaternion = list(relative.get("quat") or [])
    if len(position) != 3 or len(quaternion) != 4:
        return None
    camera = {
        "focal_length": float(HEAD_FOCAL_LENGTH),
        "horizontal_aperture": float(HEAD_HORIZONTAL_APERTURE),
        "image_width": int(width),
        "image_height": int(height),
        "robot_relative_pose": {"pos": position, "quat": quaternion},
    }
    # adapter 统一吐 BGR，RTAB-Map 侧要的是 RGB
    rgb = np.ascontiguousarray(bgr_arr[:, :, 2::-1]).astype(np.uint8, copy=False)
    depth_m = np.ascontiguousarray(depth_arr, dtype=np.float32)
    return int(sequence), rgb, depth_m, camera


def _read_frame_from_sensor(world) -> Optional[tuple[int, np.ndarray, np.ndarray, dict]]:
    """普通模式：interface 自己持有 head sensor，直接读 rgb + depth_linear。

    这条路径没有 evaluator 序号，返回 -1 让调用方用单调计数占位。
    """
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
    depth_arr = _to_numpy(depth).squeeze()
    rgb_arr = _to_numpy(rgb)
    if depth_arr.ndim != 2 or rgb_arr.ndim != 3 or rgb_arr.shape[2] < 3:
        return None
    height, width = depth_arr.shape
    if rgb_arr.shape[0] != height or rgb_arr.shape[1] != width:
        return None
    mount = head_factory_mount_pose(world)
    if mount is None:
        return None
    try:
        relative = head_camera_pose_robot_from_proprio(
            np.asarray(world.trunk_qpos(), dtype=np.float64).reshape(4),
            camera_parent_pos=mount[0],
            camera_parent_quat=mount[1],
        )
    except Exception:
        return None
    camera = dict(head_intrinsics_dict(head))
    camera["image_width"] = int(width)
    camera["image_height"] = int(height)
    camera["robot_relative_pose"] = relative
    rgb_u8 = np.ascontiguousarray(rgb_arr[:, :, :3]).astype(np.uint8, copy=False)
    depth_m = np.ascontiguousarray(depth_arr, dtype=np.float32)
    return -1, rgb_u8, depth_m, camera


def verify_offline_ready(path: Path | str) -> dict[str, Any]:
    """离线 RTAB-Map 能不能直接读这份目录。"""
    try:
        from behavior_interface.rtabmap_slam.offline import BehaviorCaptureDataset

        dataset = BehaviorCaptureDataset(path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:400]}
    covered = sum(frame.motion.duration_s for frame in list(dataset)[1:])
    last = dataset.frames[-1].timestamp_s if dataset.frames else 0.0
    return {
        "ok": True,
        "frames": len(dataset),
        "motion_seconds": round(float(covered), 3),
        "timestamp_s": round(float(last), 3),
    }


class ContinuousCapture:
    """进程级单例：把观测流写成 ``rtabmap_slam`` 可直接重放的目录。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sim_lock = threading.Lock()
        self._request_start: Optional[Path] = None
        self._request_stop = False
        # 主线程每个物理子步都会问一次，用无锁标志挡住绝大多数拿锁开销
        self._has_request = False
        self._active = False
        self._stats: dict[str, Any] = self._empty_stats()

        # 以下由 _sim_lock 保护：Flask 点 REC/STOP 与官方 act() 会并发碰到
        self._sim_clock_s = 0.0
        self._pending_ticks: list[tuple[float, tuple[float, float, float]]] = []
        self._origin_written = False
        self._last_frame_clock_s = 0.0
        self._last_sequence: Optional[int] = None
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._frame_interval_s = 0.2

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "frames": 0,
            "ticks": 0,
            "dropped_frames": 0,
            "depth_delta_frames": 0,
            "depth_keyframes": 0,
            "depth_raw_bytes": 0,
            "depth_npy_baseline_bytes": 0,
            "depth_stored_bytes": 0,
            "depth_bytes_saved": 0,
            "depth_storage_ratio": 1.0,
            "sim_seconds": 0.0,
            "path": "",
            "error": "",
            "closed": False,
        }

    # ------------------------------------------------------------------ 外部请求

    def request_start(self, capture_dir: Path | str) -> dict[str, Any]:
        """Flask 线程调用：只登记请求，真正建 writer 在仿真主线程。"""
        target = Path(capture_dir).expanduser()
        with self._lock:
            if self._active or self._request_start is not None:
                return {"ok": False, "error": "continuous capture already active"}
            self._request_start = target
            self._request_stop = False
            self._stats = self._empty_stats()
            self._stats["path"] = str(target)
            self._has_request = True
            return {"ok": True, "path": str(target)}

    def prime(self, world) -> dict[str, Any]:
        """Flask 点 REC 后立刻建 writer、写下 origin 帧。

        不等下一拍 ``act()``：否则点了按钮磁盘上还是空的，看起来像没在录。
        """
        if world is None:
            return self.status()
        with self._sim_lock:
            self._service_requests(world)
        return self.status()

    def request_stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._active and self._request_start is None:
                return {"ok": True, "active": False, **self._stats}
            self._request_stop = True
            self._has_request = True
            return {"ok": True, "active": True, **self._stats}

    def stop_now(self, world=None) -> dict[str, Any]:
        """Flask 点 STOP：当场收尾，不依赖下一拍官方 act()。

        还没建 writer 就直接取消；已经在录则尽量再写最后一帧，然后 close。
        """
        with self._sim_lock:
            with self._lock:
                self._request_start = None
                self._request_stop = True
                self._has_request = True
            if self._queue is None:
                with self._lock:
                    self._request_stop = False
                    self._has_request = False
                    self._active = False
                return self.status()
            if world is not None:
                if not self._origin_written:
                    self._write_origin(world)
                if self._pending_ticks:
                    self._emit_frame(world)
            self._finish()
        status = self.status()
        path = str(status.get("path") or "")
        if path:
            status["offline"] = verify_offline_ready(path)
        return status

    def wait_until_closed(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """阻塞到收尾完成再返回。

        ``offline`` 侧要求 manifest 的 ``closed`` 为 true 才肯重放，所以
        stop 必须等主线程真正把 writer 关掉，不能立刻返回。
        """
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                if not self._active and not self._request_stop:
                    return {"enabled": True, "active": False, **self._stats}
            time.sleep(0.05)
        with self._lock:
            payload = {"enabled": True, "active": self._active, **self._stats}
        payload.setdefault("error", "")
        if not payload["error"]:
            payload["error"] = "timed out waiting for capture to close"
        return payload

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": continuous_capture_enabled(),
                "active": self._active,
                **self._stats,
            }

    # ------------------------------------------------------------------ 仿真主线程

    def on_tick(self, world, dt_s: float) -> None:
        """每个物理子步调用一次，累积一条 body 系速度采样。"""
        if self._queue is None or not self._origin_written:
            # origin 帧之前的 tick 没有对应区间，收了反而让时长对不上
            return
        with self._sim_lock:
            if self._queue is None or not self._origin_written:
                return
            try:
                step = float(dt_s)
                if not math.isfinite(step) or step <= 0.0 or step > _TICK_MAX_DT_S:
                    return
                qvel = _as_qvel(world.base_qvel())
                if qvel is None:
                    return
                self._pending_ticks.append((step, qvel))
                self._sim_clock_s += step
            except Exception as exc:
                self._fail(f"on_tick: {exc}")

    def on_frame(self, world) -> None:
        """每个物理子步调用一次，按仿真时间节流决定是否落一帧。

        必须挂在官方 ``act()`` 或普通仿真子步循环里，不能挂
        ``_spatial_map_tick``：后者位于 ``fast_no_obs`` 分支之外，
        底盘一动就整段不执行，而底盘运动恰恰是必须录到的部分。
        """
        if self._queue is None and not self._has_request:
            return
        with self._sim_lock:
            try:
                if not self._service_requests(world):
                    return
                stopping = self._stop_requested()
                elapsed = self._sim_clock_s - self._last_frame_clock_s
                if not stopping and elapsed < self._frame_interval_s:
                    return
                if not self._pending_ticks:
                    # 仿真时间没走（静止），没有运动会丢，也就不需要新帧
                    if stopping:
                        self._finish()
                    return
                self._emit_frame(world)
                if stopping:
                    self._finish()
            except Exception as exc:
                self._fail(f"on_frame: {exc}")

    # ------------------------------------------------------------------ 内部

    def _stop_requested(self) -> bool:
        with self._lock:
            return self._request_stop

    def _service_requests(self, world) -> bool:
        """处理排队的 start；返回 True 表示当前应继续录。"""
        if self._has_request:
            with self._lock:
                pending = self._request_start
                self._request_start = None
                self._has_request = self._request_stop
            if pending is not None and self._queue is None:
                self._open(pending)
        if self._queue is None:
            return False
        if not self._origin_written:
            self._write_origin(world)
            return False
        return True

    def _open(self, target: Path) -> None:
        from behavior_interface.rtabmap_slam.capture import CaptureBundleWriter

        self._sim_clock_s = 0.0
        self._pending_ticks = []
        self._origin_written = False
        self._last_frame_clock_s = 0.0
        self._last_sequence = None
        hz = _env_float("BEHAVIOR_CAPTURE_HZ", 5.0, 0.5, 30.0)
        self._frame_interval_s = 1.0 / hz
        depth = _env_int("BEHAVIOR_CAPTURE_QUEUE", 48, 4, 512)
        try:
            target.mkdir(parents=True, exist_ok=True)
            writer = CaptureBundleWriter(
                target,
                idle_depth_compression=_env_enabled(
                    "BEHAVIOR_CAPTURE_IDLE_COMPRESSION", True
                ),
                idle_qvel_threshold=_env_float(
                    "BEHAVIOR_CAPTURE_IDLE_QVEL_EPS", 0.001, 0.0, 0.1
                ),
                delta_keyframe_interval=_env_int(
                    "BEHAVIOR_CAPTURE_DELTA_KEYFRAME_FRAMES", 30, 1, 300
                ),
            )
        except Exception as exc:
            self._fail(f"open: {exc}")
            with self._lock:
                self._request_stop = False
            return
        self._queue = queue.Queue(maxsize=depth)
        self._worker = threading.Thread(
            target=self._drain,
            args=(writer, self._queue),
            name="continuous-capture",
            daemon=True,
        )
        self._worker.start()
        with self._lock:
            self._active = True
            self._stats["path"] = str(target)

    def _write_origin(self, world) -> None:
        """origin 帧必须没有入口 tick，写成功后才开始收 tick。"""
        frame = read_frame(world)
        if frame is None:
            return
        sequence, rgb, depth, camera = frame
        if sequence < 0:
            sequence = 0
        payload = _PendingFrame(
            sequence=sequence,
            timestamp_s=0.0,
            rgb=rgb,
            depth_m=depth,
            camera=camera,
            sampled_base_qvel=_as_qvel(world.base_qvel()) or (0.0, 0.0, 0.0),
            ticks=(),
        )
        assert self._queue is not None
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            return
        self._origin_written = True
        self._last_sequence = sequence
        self._last_frame_clock_s = 0.0

    def _emit_frame(self, world) -> None:
        peeked = peek_sequence(world)
        if (
            peeked is not None
            and self._last_sequence is not None
            and peeked <= self._last_sequence
        ):
            # evaluator 还没推新观测：写下去只是同一张图，且违反序号严格递增。
            # tick 留着，等真正的新观测到来时一起提交。
            return
        frame = read_frame(world)
        if frame is None:
            return
        sequence, rgb, depth, camera = frame
        if sequence < 0:
            sequence = (self._last_sequence or 0) + 1
        if self._last_sequence is not None and sequence <= self._last_sequence:
            return
        payload = _PendingFrame(
            sequence=sequence,
            timestamp_s=self._sim_clock_s,
            rgb=rgb,
            depth_m=depth,
            camera=camera,
            sampled_base_qvel=self._pending_ticks[-1][1],
            ticks=tuple(self._pending_ticks),
        )
        assert self._queue is not None
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            # 落盘跟不上时只丢图像，绝不丢 tick：ticks 留到下一帧一起提交，
            # 累积时长依旧等于那时的帧间隔，运动覆盖率不会出现空洞。
            with self._lock:
                self._stats["dropped_frames"] += 1
            return
        self._pending_ticks = []
        self._last_frame_clock_s = self._sim_clock_s
        self._last_sequence = sequence

    def _finish(self) -> None:
        pending = self._queue
        worker = self._worker
        self._queue = None
        self._worker = None
        self._origin_written = False
        # 末尾没配上帧的 tick 直接丢弃，留着会让 writer 拒绝收尾
        self._pending_ticks = []
        if pending is not None:
            try:
                pending.put_nowait(_SENTINEL)
            except queue.Full:
                # 后台线程卡住时不要把仿真主线程一起拖住，丢掉最旧的待写帧腾位置
                try:
                    pending.get_nowait()
                    pending.put_nowait(_SENTINEL)
                except (queue.Empty, queue.Full):
                    pass
        if worker is not None:
            worker.join(timeout=60.0)
        with self._lock:
            self._active = False
            self._request_stop = False
            self._has_request = self._request_start is not None

    def _drain(self, writer, pending: queue.Queue) -> None:
        """后台线程：顺序消费待写帧，PNG 压缩和 fsync 都发生在这里。"""
        # Continuous capture is archival work.  It may use CPU and NAS I/O for
        # seconds after a frame is queued, but it must not preempt the
        # evaluator's observation/action loop on an oversubscribed host.
        _set_current_thread_nice(
            "BEHAVIOR_CAPTURE_WORKER_NICE",
            default=19,
        )
        from behavior_interface.rtabmap_slam.official import (
            CameraIntrinsics,
            CameraRelativePose,
            OfficialObservation,
        )

        try:
            while True:
                item = pending.get()
                if item is _SENTINEL:
                    break
                if not isinstance(item, _PendingFrame):
                    continue
                intrinsics = _intrinsics_from_camera(item.camera)
                if intrinsics is None:
                    continue
                relative = item.camera.get("robot_relative_pose") or {}
                observation = OfficialObservation.create(
                    timestamp_s=item.timestamp_s,
                    head_rgb=item.rgb,
                    head_depth_m=item.depth_m,
                    intrinsics=CameraIntrinsics(
                        intrinsics["width"],
                        intrinsics["height"],
                        intrinsics["fx"],
                        intrinsics["fy"],
                        intrinsics["cx"],
                        intrinsics["cy"],
                    ),
                    camera_relative_pose=CameraRelativePose.create(
                        relative.get("pos") or [],
                        relative.get("quat") or [],
                    ),
                )
                for dt_s, qvel in item.ticks:
                    writer.append_tick(qvel, dt_s)
                writer.append_frame(
                    observation,
                    evaluator_sequence=int(item.sequence),
                    sampled_base_qvel=item.sampled_base_qvel,
                    # Preserve sampling throughput if encoding ever falls behind.
                    allow_idle_compression=(
                        pending.qsize() <= max(2, pending.maxsize // 4)
                    ),
                )
                with self._lock:
                    self._stats["frames"] += 1
                    self._stats["ticks"] += len(item.ticks)
                    self._stats["sim_seconds"] = round(float(item.timestamp_s), 3)
                    self._stats.update(writer.payload_stats)
        except Exception as exc:
            self._fail(f"writer: {exc}")
            # 主线程还在往队列里塞，继续排空直到收尾信号，别让它反复丢帧超时
            while True:
                try:
                    if pending.get(timeout=30.0) is _SENTINEL:
                        break
                except queue.Empty:
                    break
        finally:
            try:
                writer.close()
                with self._lock:
                    self._stats["closed"] = True
            except Exception as exc:
                self._fail(f"close: {exc}")


CAPTURE = ContinuousCapture()
