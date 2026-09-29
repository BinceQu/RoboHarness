"""Strict, evaluator-safe observation boundary for the RTAB-Map worker.

There is deliberately no generic ``dict`` ingestion API here.  A caller must
pass the four allowed observation products explicitly: head RGB, head depth,
head camera pose relative to the base, and integrated base velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Sequence

import numpy as np


_CAMERA_TO_RTAB_OPTICAL = np.diag([1.0, -1.0, -1.0])

# These geometry-only limits are mirrored by the native worker.  They use the
# evaluator-provided camera pose relative to the base, never a world pose.
STRUCTURAL_VIEW_MIN_HORIZONTAL = 0.85
STRUCTURAL_CAMERA_MIN_HEIGHT_M = 1.25
STRUCTURAL_CAMERA_MAX_PLANAR_OFFSET_M = 0.45
# A settled camera can be a little below the strict chassis-centred envelope
# after the head controller finishes an articulation.  These bounds are only
# reachable through the stateful hysteresis below; a transient or manipulation
# view still remains rejected.  Keep them wide enough for the recorded
# post-articulation pose (horizontal view ~= 0.793, height ~= 1.347 m).
STRUCTURAL_RECOVERY_VIEW_MIN_HORIZONTAL = 0.75
STRUCTURAL_RECOVERY_CAMERA_MIN_HEIGHT_M = 1.15
STRUCTURAL_RECOVERY_CAMERA_MAX_PLANAR_OFFSET_M = 0.60
# One 5 cm occupancy cell or two degrees between submitted observations is
# already enough to move a 3.5 m depth endpoint by multiple cells.  Larger
# camera-to-base changes are articulation transients, not rigid-base motion,
# and must not enter either mapping or place retrieval.
STRUCTURAL_CAMERA_MAX_STEP_M = 0.05
STRUCTURAL_CAMERA_MAX_STEP_RAD = math.radians(2.0)
STRUCTURAL_RECOVERY_STABLE_FRAMES = 3


def _finite_vector(value: Sequence[float], length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != length or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    return array


def _wrap_radians(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _quaternion_matrix_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = _finite_vector(quaternion, 4, "quaternion_xyzw")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("quaternion_xyzw must have non-zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("camera dimensions must be positive")
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if float(self.fx) <= 0.0 or float(self.fy) <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class CameraRelativePose:
    """Head optical-camera pose in the robot base frame, quaternion in XYZW."""

    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    @classmethod
    def create(
        cls,
        position_m: Sequence[float],
        quaternion_xyzw: Sequence[float],
    ) -> "CameraRelativePose":
        position = _finite_vector(position_m, 3, "camera position")
        quaternion = _finite_vector(quaternion_xyzw, 4, "camera quaternion")
        _quaternion_matrix_xyzw(quaternion)
        return cls(tuple(float(v) for v in position), tuple(float(v) for v in quaternion))

    def rtabmap_local_transform(self) -> np.ndarray:
        """Return RTAB-Map's optical-to-base 3x4 transform.

        BEHAVIOR camera rays use ``[right, up, -forward]``.  RTAB-Map uses
        ``[right, down, forward]``.  The fixed proper rotation below converts
        the latter before applying evaluator-provided ``cam_rel_pose``.
        """

        rotation = _quaternion_matrix_xyzw(self.quaternion_xyzw)
        output = np.empty((3, 4), dtype=np.float64)
        output[:, :3] = rotation @ _CAMERA_TO_RTAB_OPTICAL
        output[:, 3] = np.asarray(self.position_m, dtype=np.float64)
        return output


@dataclass(frozen=True)
class OfficialObservation:
    timestamp_s: float
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    camera_relative_pose: CameraRelativePose

    @classmethod
    def create(
        cls,
        *,
        timestamp_s: float,
        head_rgb: np.ndarray,
        head_depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        camera_relative_pose: CameraRelativePose,
    ) -> "OfficialObservation":
        stamp = float(timestamp_s)
        if not math.isfinite(stamp) or stamp < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        rgb = np.asarray(head_rgb)
        depth = np.asarray(head_depth_m)
        expected = (int(intrinsics.height), int(intrinsics.width))
        if rgb.shape != expected + (3,):
            raise ValueError(f"head_rgb must have shape {expected + (3,)}")
        if depth.shape != expected:
            raise ValueError(f"head_depth_m must have shape {expected}")
        if rgb.dtype != np.uint8:
            raise ValueError("head_rgb must be uint8 RGB")
        if not np.issubdtype(depth.dtype, np.floating):
            raise ValueError("head_depth_m must be a floating-point meter image")
        rgb = np.ascontiguousarray(rgb)
        depth = np.ascontiguousarray(depth, dtype=np.dtype("<f4"))
        # Invalid pixels remain explicit zeroes.  Range and self-body filtering
        # happen in the native worker before both visual odometry and mapping.
        depth = depth.copy()
        depth[~np.isfinite(depth) | (depth <= 0.0)] = 0.0
        rgb.setflags(write=False)
        depth.setflags(write=False)
        return cls(stamp, rgb, depth, intrinsics, camera_relative_pose)


def _structural_geometry_usable(
    observation: OfficialObservation, *, relaxed: bool = False
) -> bool:
    transform = observation.camera_relative_pose.rtabmap_local_transform()
    horizontal_view = math.hypot(float(transform[0, 2]), float(transform[1, 2]))
    camera_height_m = float(transform[2, 3])
    planar_offset_m = math.hypot(float(transform[0, 3]), float(transform[1, 3]))
    if relaxed:
        view_min = STRUCTURAL_RECOVERY_VIEW_MIN_HORIZONTAL
        height_min = STRUCTURAL_RECOVERY_CAMERA_MIN_HEIGHT_M
        offset_max = STRUCTURAL_RECOVERY_CAMERA_MAX_PLANAR_OFFSET_M
    else:
        view_min = STRUCTURAL_VIEW_MIN_HORIZONTAL
        height_min = STRUCTURAL_CAMERA_MIN_HEIGHT_M
        offset_max = STRUCTURAL_CAMERA_MAX_PLANAR_OFFSET_M
    return (
        horizontal_view >= view_min
        and camera_height_m >= height_min
        and planar_offset_m <= offset_max
    )


def structural_observation_usable(observation: OfficialObservation) -> bool:
    """Whether the current head pose is strictly chassis-centred."""

    return _structural_geometry_usable(observation)


def structural_observation_recovery_usable(
    observation: OfficialObservation,
) -> bool:
    """Whether a *settled* post-articulation pose may recover structure."""

    return _structural_geometry_usable(observation, relaxed=True)


class StructuralObservationGate:
    """Causal camera-to-base stability gate for structural RGB-D updates.

    The absolute limits reject manipulation views.  The adjacent-transform
    limits additionally reject reset-body and articulation transitions that
    happen to pass through an otherwise upright pose.  A fresh session may
    use its first qualified observation immediately; after any interruption,
    three consecutive qualified, low-motion observations are required before
    structural processing resumes.
    """

    def __init__(
        self,
        *,
        maximum_translation_step_m: float = STRUCTURAL_CAMERA_MAX_STEP_M,
        maximum_rotation_step_rad: float = STRUCTURAL_CAMERA_MAX_STEP_RAD,
        recovery_stable_frames: int = STRUCTURAL_RECOVERY_STABLE_FRAMES,
    ) -> None:
        self.maximum_translation_step_m = float(maximum_translation_step_m)
        self.maximum_rotation_step_rad = float(maximum_rotation_step_rad)
        self.recovery_stable_frames = int(recovery_stable_frames)
        if self.maximum_translation_step_m <= 0.0:
            raise ValueError("maximum_translation_step_m must be positive")
        if self.maximum_rotation_step_rad <= 0.0:
            raise ValueError("maximum_rotation_step_rad must be positive")
        if self.recovery_stable_frames < 1:
            raise ValueError("recovery_stable_frames must be positive")
        self.reset()

    def reset(self) -> None:
        self._previous_transform: np.ndarray | None = None
        self._interrupted = False
        self._stable_frames = 0

    def update(self, observation: OfficialObservation) -> bool:
        transform = observation.camera_relative_pose.rtabmap_local_transform()
        strict_usable = structural_observation_usable(observation)
        recovery_usable = structural_observation_recovery_usable(observation)
        previous = self._previous_transform
        self._previous_transform = transform.copy()

        if previous is None:
            if strict_usable:
                self._stable_frames = self.recovery_stable_frames
                return True
            self._interrupted = True
            self._stable_frames = 0
            return False

        translation_step_m = float(
            np.linalg.norm(transform[:, 3] - previous[:, 3])
        )
        relative_rotation = previous[:, :3].T @ transform[:, :3]
        cosine = float(
            np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        rotation_step_rad = math.acos(cosine)
        low_motion = (
            translation_step_m <= self.maximum_translation_step_m
            and rotation_step_rad <= self.maximum_rotation_step_rad
        )
        if not recovery_usable or not low_motion:
            self._interrupted = True
            self._stable_frames = 0
            return False

        # A relaxed geometry frame is never accepted immediately.  It must
        # remain low-motion for the same recovery hysteresis used after a head
        # reset; this is what prevents a passing arm/table view from becoming
        # a permanent map observation.
        if not strict_usable:
            self._interrupted = True

        if not self._interrupted:
            self._stable_frames = self.recovery_stable_frames
            return True

        self._stable_frames += 1
        if self._stable_frames < self.recovery_stable_frames:
            return False
        self._interrupted = False
        return True


@dataclass(frozen=True)
class SE2Pose:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0

    def compose(self, relative: "SE2Pose") -> "SE2Pose":
        """Apply a body-frame relative transform to this world-frame pose."""

        c = math.cos(float(self.yaw_rad))
        s = math.sin(float(self.yaw_rad))
        return SE2Pose(
            float(self.x_m) + c * float(relative.x_m) - s * float(relative.y_m),
            float(self.y_m) + s * float(relative.x_m) + c * float(relative.y_m),
            _wrap_radians(float(self.yaw_rad) + float(relative.yaw_rad)),
        )

    def relative_to(self, other: "SE2Pose") -> "SE2Pose":
        """Transform from this pose to ``other``, expressed in this body frame."""

        dx = float(other.x_m) - float(self.x_m)
        dy = float(other.y_m) - float(self.y_m)
        c = math.cos(float(self.yaw_rad))
        s = math.sin(float(self.yaw_rad))
        return SE2Pose(
            c * dx + s * dy,
            -s * dx + c * dy,
            _wrap_radians(float(other.yaw_rad) - float(self.yaw_rad)),
        )


def align_odometry_path_to_endpoint(
    odometry_poses: Sequence[SE2Pose],
    odometry_endpoint: SE2Pose,
    mapped_endpoint: SE2Pose,
) -> tuple[SE2Pose, ...]:
    """把一段逐 tick 底盘轨迹刚体对齐到同一时刻的 SLAM 位姿。"""

    return tuple(
        mapped_endpoint.compose(odometry_endpoint.relative_to(pose))
        for pose in odometry_poses
    )


class BodyOdometry:
    """Exact SE(2) integration of evaluator-provided body-frame ``base_qvel``."""

    def __init__(self) -> None:
        self._pose = SE2Pose()
        self._frame_pose = SE2Pose()
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._pose = SE2Pose()
            self._frame_pose = SE2Pose()

    def advance(self, base_qvel: Sequence[float], dt_s: float) -> SE2Pose:
        velocity = _finite_vector(base_qvel, 3, "base_qvel")
        dt = float(dt_s)
        if not math.isfinite(dt) or dt < 0.0 or dt > 1.0:
            raise ValueError("dt_s must be finite and within [0, 1] second")
        vx, vy, yaw_rate = (float(v) for v in velocity)
        angle = yaw_rate * dt
        if abs(yaw_rate) < 1e-9:
            local_x = vx * dt
            local_y = vy * dt
        else:
            sine = math.sin(angle)
            one_minus_cosine = 1.0 - math.cos(angle)
            local_x = sine / yaw_rate * vx - one_minus_cosine / yaw_rate * vy
            local_y = one_minus_cosine / yaw_rate * vx + sine / yaw_rate * vy
        with self._lock:
            c = math.cos(self._pose.yaw_rad)
            s = math.sin(self._pose.yaw_rad)
            self._pose = SE2Pose(
                self._pose.x_m + c * local_x - s * local_y,
                self._pose.y_m + s * local_x + c * local_y,
                _wrap_radians(self._pose.yaw_rad + angle),
            )
            return self._pose

    def snapshot(self) -> SE2Pose:
        with self._lock:
            return self._pose

    def frame_delta(self) -> tuple[SE2Pose, SE2Pose]:
        """Return ``(delta, current)`` without consuming the frame baseline."""

        with self._lock:
            return self._frame_pose.relative_to(self._pose), self._pose

    def commit_frame(self, pose: SE2Pose) -> None:
        with self._lock:
            self._frame_pose = pose
