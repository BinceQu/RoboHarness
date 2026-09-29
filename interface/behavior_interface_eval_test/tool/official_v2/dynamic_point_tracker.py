"""Observation-only dynamic RGB-D surface-point tracking.

The tracker consumes synchronized evaluator RGB, linear depth, and a camera
pose assembled from evaluator camera-relative poses and policy-local odometry.
It never resolves model labels to simulator objects and never treats a
predicted point as an observed depth measurement.
"""

from __future__ import annotations

import copy
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


OBSERVED = "observed"
OCCLUDED = "occluded"
LOST = "lost"
AMBIGUOUS = "ambiguous"
DYNAMIC_POINT_TRACKER_BUILD = (
    "registered_rgbd_npoint_eef_prior_patch_reacquisition_gpu_resident_v9"
)
OPENCL_UMAT_CACHE_MAX_ENTRIES = 16


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for one evaluator camera stream."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        if self.width <= 1 or self.height <= 1:
            raise ValueError("camera width and height must be greater than one")
        values = np.asarray(
            [self.fx, self.fy, self.cx, self.cy],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class TrackerFrame:
    """One synchronized, allowlisted evaluator observation."""

    rgb: np.ndarray
    depth_linear: np.ndarray
    intrinsics: CameraIntrinsics
    camera_to_robot_base: np.ndarray
    camera_to_policy: np.ndarray
    sequence: int
    timestamp_s: float
    episode_id: str
    camera_role: str = "head"
    color_order: str = "rgb"
    proprio: np.ndarray | None = None


@dataclass(frozen=True)
class DynamicPointSnapshot:
    """Copy-safe public state for one model-annotated point."""

    track_id: str
    label: str
    camera_role: str
    status: str
    uv: tuple[float, float] | None
    pixel_uv: tuple[float, float] | None
    depth_m: float | None
    point_camera_m: tuple[float, float, float] | None
    point_robot_base_m: tuple[float, float, float] | None
    point_policy_local_m: tuple[float, float, float] | None
    velocity_policy_local_m_s: tuple[float, float, float] | None
    predicted_uv: tuple[float, float] | None
    predicted_depth_m: float | None
    confidence: float
    observation_sequence: int
    last_seen_sequence: int
    missed_steps: int
    episode_id: str
    identity_source: str = "model_annotation"
    identity_verified: bool = False
    depth_source: str | None = "current_evaluator_depth_linear"
    direct_simulator_mutation: bool = False

    def as_dict(self) -> dict[str, Any]:
        def optional_list(value):
            return None if value is None else list(value)

        return {
            "track_id": self.track_id,
            "label": self.label,
            "camera_role": self.camera_role,
            "status": self.status,
            "uv": optional_list(self.uv),
            "pixel_uv": optional_list(self.pixel_uv),
            "depth_m": self.depth_m,
            "point_camera_m": optional_list(self.point_camera_m),
            "point_robot_base_m": optional_list(self.point_robot_base_m),
            "point_policy_local_m": optional_list(
                self.point_policy_local_m
            ),
            "velocity_policy_local_m_s": optional_list(
                self.velocity_policy_local_m_s
            ),
            "predicted_uv": optional_list(self.predicted_uv),
            "predicted_depth_m": self.predicted_depth_m,
            "confidence": self.confidence,
            "observation_sequence": self.observation_sequence,
            "last_seen_sequence": self.last_seen_sequence,
            "missed_steps": self.missed_steps,
            "episode_id": self.episode_id,
            "identity_source": self.identity_source,
            "identity_verified": self.identity_verified,
            "depth_source": self.depth_source,
            "observation_contract": [
                "*::rgb",
                "*::depth_linear",
                "*::cam_rel_poses",
                "*::proprio",
                "policy_local_odometry",
            ],
            "direct_simulator_mutation": self.direct_simulator_mutation,
        }


@dataclass(frozen=True)
class DynamicPointTrackerConfig:
    """Accuracy and failure thresholds for the local tracker."""

    patch_radius_px: int = 24
    max_support_points: int = 64
    min_support_points: int = 6
    min_inlier_points: int = 4
    feature_quality: float = 0.01
    feature_min_distance_px: float = 3.0
    lk_window_px: int = 25
    lk_max_level: int = 3
    lk_max_iterations: int = 30
    forward_backward_max_error_px: float = 0.8
    affine_ransac_error_px: float = 1.5
    affine_max_residual_px: float = 1.2
    max_anchor_step_px: float = 96.0
    min_affine_scale: float = 0.65
    max_affine_scale: float = 1.45
    max_affine_rotation_deg: float = 35.0
    min_patch_correlation: float = 0.45
    depth_search_radius_px: int = 2
    depth_layer_abs_tolerance_m: float = 0.08
    depth_layer_relative_tolerance: float = 0.08
    max_occluded_steps: int = 8
    ambiguity_anchor_distance_px: float = 4.0
    ambiguity_support_distance_px: float = 1.5
    ambiguity_support_fraction: float = 0.50
    fallback_search_radius_px: int = 64
    fallback_component_radius_px: int = 56
    fallback_template_radius_px: int = 12
    fallback_depth_change_abs_m: float = 0.35
    fallback_depth_change_relative: float = 0.25
    fallback_min_score: float = 0.38
    fallback_max_velocity_prediction_s: float = 0.35
    fallback_max_velocity_m_s: float = 3.0
    rigid_pair_individual_max_prediction_error_px: float = 24.0
    rigid_pair_individual_max_depth_error_m: float = 0.030
    rigid_pair_identity_min_margin_px: float = 2.0
    rigid_pair_prior_max_pixel_error_px: float = 24.0
    rigid_pair_prior_max_depth_error_m: float = 0.025
    rigid_pair_prior_max_point_error_m: float = 0.030
    # A static RGB-D surface must not be allowed to random-walk through the
    # feature fallback.  This is deliberately below the public 5 mm budget.
    static_registration_min_patch_correlation: float = 0.90
    static_registration_max_depth_error_m: float = 0.0045
    static_registration_max_point_error_m: float = 0.0045

    def validate(self) -> None:
        integer_positive = (
            self.patch_radius_px,
            self.max_support_points,
            self.min_support_points,
            self.min_inlier_points,
            self.lk_window_px,
            self.lk_max_iterations,
            self.fallback_search_radius_px,
            self.fallback_component_radius_px,
            self.fallback_template_radius_px,
        )
        if any(value <= 0 for value in integer_positive):
            raise ValueError("tracker integer thresholds must be positive")
        if self.lk_window_px % 2 == 0:
            raise ValueError("lk_window_px must be odd")
        if self.max_support_points < self.min_support_points:
            raise ValueError(
                "max_support_points must be at least min_support_points"
            )
        if self.min_support_points < self.min_inlier_points:
            raise ValueError(
                "min_support_points must be at least min_inlier_points"
            )
        if self.lk_max_level < 0 or self.depth_search_radius_px < 0:
            raise ValueError("tracker levels and radii cannot be negative")
        if self.max_occluded_steps < 0:
            raise ValueError("max_occluded_steps cannot be negative")
        if not 0.0 <= self.min_patch_correlation <= 1.0:
            raise ValueError("min_patch_correlation must be in 0..1")
        if not 0.0 <= self.ambiguity_support_fraction <= 1.0:
            raise ValueError("ambiguity_support_fraction must be in 0..1")
        if not 0.0 <= self.fallback_min_score <= 1.0:
            raise ValueError("fallback_min_score must be in 0..1")
        positive_fallback_values = (
            self.fallback_depth_change_abs_m,
            self.fallback_depth_change_relative,
            self.fallback_max_velocity_prediction_s,
            self.fallback_max_velocity_m_s,
            self.rigid_pair_individual_max_prediction_error_px,
            self.rigid_pair_individual_max_depth_error_m,
            self.rigid_pair_identity_min_margin_px,
            self.rigid_pair_prior_max_pixel_error_px,
            self.rigid_pair_prior_max_depth_error_m,
            self.rigid_pair_prior_max_point_error_m,
            self.static_registration_min_patch_correlation,
            self.static_registration_max_depth_error_m,
            self.static_registration_max_point_error_m,
        )
        if any(value <= 0.0 for value in positive_fallback_values):
            raise ValueError("fallback tracking thresholds must be positive")
        if self.static_registration_min_patch_correlation > 1.0:
            raise ValueError(
                "static_registration_min_patch_correlation must be <= 1"
            )


@dataclass
class _ValidatedFrame:
    gray: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    camera_to_robot_base: np.ndarray
    camera_to_policy: np.ndarray
    sequence: int
    timestamp_s: float
    episode_id: str
    camera_role: str


@dataclass
class _DepthComponent:
    centroid_px: np.ndarray
    size_px: np.ndarray
    area_px: int
    median_depth_m: float
    pixels_px: np.ndarray


@dataclass
class _TrackState:
    track_id: str
    label: str
    camera_role: str
    anchor_px: np.ndarray
    support_px: np.ndarray
    reference_gray: np.ndarray
    registration_anchor_px: np.ndarray
    registration_support_px: np.ndarray
    registration_gray: np.ndarray
    registration_point_policy: np.ndarray | None
    registration_depth_m: float | None
    last_depth_m: float
    last_point_policy: np.ndarray
    last_velocity_policy: np.ndarray | None
    last_seen_timestamp_s: float
    last_seen_sequence: int
    missed_steps: int
    confidence: float
    component_offset_px: np.ndarray
    component_size_px: np.ndarray
    component_area_px: int
    snapshot: DynamicPointSnapshot
    last_observation_method: str = "registration"
    last_observation_rejections: tuple[dict[str, Any], ...] = ()
    pending_component_depth_roi: np.ndarray | None = None
    pending_component_origin_px: tuple[int, int] | None = None
    pending_component_anchor_px: np.ndarray | None = None
    pending_component_depth_m: float | None = None


def transform_from_pose(
    position,
    quaternion_xyzw,
) -> np.ndarray:
    """Return a local transform without consulting simulator state."""

    pos = np.asarray(position, dtype=np.float64).reshape(3)
    quat = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
        raise ValueError("pose values must be finite")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("pose quaternion has zero norm")
    x, y, z, w = quat / norm
    rotation = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pos
    return transform


def camera_to_policy_from_odometry(
    base_xy_yaw,
    camera_position_robot,
    camera_quaternion_robot_xyzw,
) -> np.ndarray:
    """Compose policy-local base odometry with evaluator camera-relative pose."""

    x, y, yaw = np.asarray(base_xy_yaw, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite([x, y, yaw])):
        raise ValueError("policy-local base odometry must be finite")
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    policy_from_base = np.eye(4, dtype=np.float64)
    policy_from_base[:3, :3] = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    policy_from_base[:2, 3] = [x, y]
    base_from_camera = transform_from_pose(
        camera_position_robot,
        camera_quaternion_robot_xyzw,
    )
    return policy_from_base @ base_from_camera


class DynamicPointTracker:
    """Track model-clicked material points through evaluator RGB-D frames."""

    def __init__(
        self,
        *,
        camera_role: str = "head",
        config: DynamicPointTrackerConfig | None = None,
    ) -> None:
        role = str(camera_role or "").strip()
        if not role:
            raise ValueError("camera_role is required")
        self.camera_role = role
        self.config = config or DynamicPointTrackerConfig()
        self.config.validate()
        self._lock = threading.RLock()
        self._frame: _ValidatedFrame | None = None
        self._episode_id: str | None = None
        self._tracks: dict[str, _TrackState] = {}
        self._next_track_number = 1
        self._active_rigid_pair: dict[str, Any] | None = None
        self._last_rigid_pair_report: dict[str, Any] = {}
        self._opencl_lk_state = "uninitialized"
        self._opencl_lk_device = ""
        self._opencl_lk_disable_reason = ""
        self._opencl_lk_validation_remaining = 12
        self._opencl_lk_validation_passes = 0
        self._opencl_lk_gpu_calls = 0
        self._opencl_lk_cpu_calls = 0
        self._opencl_umat_cache: dict[int, tuple[np.ndarray, Any]] = {}
        self._opencl_umat_cache_hits = 0
        self._opencl_umat_uploads = 0
        self._opencl_umat_evictions = 0

    @property
    def episode_id(self) -> str | None:
        with self._lock:
            return self._episode_id

    @property
    def current_sequence(self) -> int | None:
        with self._lock:
            return None if self._frame is None else self._frame.sequence

    def reset(self) -> None:
        with self._lock:
            self._frame = None
            self._episode_id = None
            self._tracks.clear()
            self._active_rigid_pair = None
            self._last_rigid_pair_report = {}
            self._opencl_umat_cache = {}

    def upgrade_runtime_state(self) -> dict[str, Any]:
        """Initialize policy-owned fields added by a hot reload."""

        with self._lock:
            if not hasattr(self, "_active_rigid_pair"):
                self._active_rigid_pair = None
            if not hasattr(self, "_last_rigid_pair_report"):
                self._last_rigid_pair_report = {}
            acceleration_defaults = {
                "_opencl_lk_state": "uninitialized",
                "_opencl_lk_device": "",
                "_opencl_lk_disable_reason": "",
                "_opencl_lk_validation_remaining": 12,
                "_opencl_lk_validation_passes": 0,
                "_opencl_lk_gpu_calls": 0,
                "_opencl_lk_cpu_calls": 0,
                "_opencl_umat_cache": {},
                "_opencl_umat_cache_hits": 0,
                "_opencl_umat_uploads": 0,
                "_opencl_umat_evictions": 0,
            }
            for field, default in acceleration_defaults.items():
                if not hasattr(self, field):
                    setattr(self, field, default)
            for state in self._tracks.values():
                if not hasattr(state, "last_observation_rejections"):
                    state.last_observation_rejections = ()
                # Older in-memory states can survive an interface hot reload.
                # Their latest measured point is the only policy-owned
                # registration reference available in that case.
                if not hasattr(state, "registration_point_policy"):
                    state.registration_point_policy = np.asarray(
                        state.last_point_policy,
                        dtype=np.float64,
                    ).copy()
                if not hasattr(state, "registration_depth_m"):
                    state.registration_depth_m = float(state.last_depth_m)
            return {
                "ok": True,
                "build": DYNAMIC_POINT_TRACKER_BUILD,
                "active": self._active_rigid_pair is not None,
                "acceleration": self.acceleration_status(),
            }

    def acceleration_status(self) -> dict[str, Any]:
        """Report the result-preserving sparse-flow acceleration backend."""

        with self._lock:
            state = str(getattr(self, "_opencl_lk_state", "uninitialized"))
            return {
                "backend": (
                    "opencl_gpu"
                    if state in {"validating", "active"}
                    else "cpu"
                ),
                "state": state,
                "device": str(getattr(self, "_opencl_lk_device", "")),
                "validation_remaining": int(
                    getattr(self, "_opencl_lk_validation_remaining", 12)
                ),
                "validation_passes": int(
                    getattr(self, "_opencl_lk_validation_passes", 0)
                ),
                "gpu_calls": int(getattr(self, "_opencl_lk_gpu_calls", 0)),
                "cpu_calls": int(getattr(self, "_opencl_lk_cpu_calls", 0)),
                "umat_cache": {
                    "entries": len(
                        getattr(self, "_opencl_umat_cache", {})
                    ),
                    "max_entries": OPENCL_UMAT_CACHE_MAX_ENTRIES,
                    "hits": int(
                        getattr(self, "_opencl_umat_cache_hits", 0)
                    ),
                    "uploads": int(
                        getattr(self, "_opencl_umat_uploads", 0)
                    ),
                    "evictions": int(
                        getattr(self, "_opencl_umat_evictions", 0)
                    ),
                },
                "disable_reason": str(
                    getattr(self, "_opencl_lk_disable_reason", "")
                ),
                "parity_gate": "exact_status_and_valid_points",
            }

    def _disable_opencl_lk(self, reason: str) -> None:
        self._opencl_lk_state = "disabled"
        self._opencl_lk_disable_reason = str(reason)[:240]
        self._opencl_umat_cache = {}

    def _ensure_opencl_lk(self) -> bool:
        state = str(getattr(self, "_opencl_lk_state", "uninitialized"))
        if state in {"validating", "active"}:
            return True
        if state == "disabled":
            return False

        # A single CUDA-visible device makes OpenCV's default OpenCL device
        # unambiguous. This prevents a live evaluator from borrowing another
        # interface's GPU when several physical devices are installed.
        visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
        if not visible or "," in visible:
            self._disable_opencl_lk("single_visible_gpu_required")
            return False
        try:
            if not hasattr(cv2, "ocl") or not cv2.ocl.haveOpenCL():
                self._disable_opencl_lk("opencv_opencl_unavailable")
                return False
            cv2.ocl.setUseOpenCL(True)
            device = cv2.ocl.Device_getDefault()
            device_name = str(device.name())
            if (
                not cv2.ocl.useOpenCL()
                or not bool(device.available())
                or "NVIDIA" not in device_name.upper()
            ):
                self._disable_opencl_lk("nvidia_opencl_gpu_unavailable")
                return False
        except Exception as exc:
            self._disable_opencl_lk(
                f"opencl_initialization_failed:{type(exc).__name__}:{exc}"
            )
            return False

        self._opencl_lk_device = device_name
        self._opencl_lk_state = "validating"
        self._opencl_lk_disable_reason = ""
        return True

    def _opencl_umat(self, value: np.ndarray) -> Any:
        source = np.asarray(value)
        key = id(source)
        cache = self._opencl_umat_cache
        cached = cache.get(key)
        if cached is not None and cached[0] is source:
            # Refresh insertion order so registration and preceding-frame
            # images stay resident while stale observation images age out.
            cache.pop(key)
            cache[key] = cached
            self._opencl_umat_cache_hits += 1
            return cached[1]
        umat = cv2.UMat(np.ascontiguousarray(source))
        cache[key] = (source, umat)
        self._opencl_umat_uploads += 1
        while len(cache) > OPENCL_UMAT_CACHE_MAX_ENTRIES:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key)
            self._opencl_umat_evictions += 1
        return umat

    @staticmethod
    def _numpy_lk_result(result: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        return tuple(
            value.get() if isinstance(value, cv2.UMat) else value
            for value in result
        )

    @staticmethod
    def _lk_results_match(
        cpu_result: tuple[Any, Any, Any],
        gpu_result: tuple[Any, Any, Any],
    ) -> bool:
        cpu_points, cpu_status, _cpu_error = cpu_result
        gpu_points, gpu_status, _gpu_error = gpu_result
        if (cpu_points is None) != (gpu_points is None):
            return False
        if (cpu_status is None) != (gpu_status is None):
            return False
        if cpu_status is None:
            return True
        cpu_status_array = np.asarray(cpu_status)
        gpu_status_array = np.asarray(gpu_status)
        if not np.array_equal(cpu_status_array, gpu_status_array):
            return False
        if cpu_points is None:
            return True
        cpu_point_array = np.asarray(cpu_points).reshape(-1, 2)
        gpu_point_array = np.asarray(gpu_points).reshape(-1, 2)
        if cpu_point_array.shape != gpu_point_array.shape:
            return False
        valid = cpu_status_array.reshape(-1).astype(bool)
        return np.array_equal(cpu_point_array[valid], gpu_point_array[valid])

    def _calc_sparse_lk(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_points: np.ndarray,
        initial_points: np.ndarray | None,
        **options: Any,
    ) -> tuple[Any, Any, Any]:
        """Run identical SparsePyrLK on GPU only after an exact parity gate."""

        def cpu_result() -> tuple[Any, Any, Any]:
            self._opencl_lk_cpu_calls += 1
            return cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                previous_points,
                initial_points,
                **options,
            )

        if not self._ensure_opencl_lk():
            return cpu_result()
        try:
            gpu_result = self._numpy_lk_result(
                cv2.calcOpticalFlowPyrLK(
                    self._opencl_umat(previous_gray),
                    self._opencl_umat(current_gray),
                    cv2.UMat(np.ascontiguousarray(previous_points)),
                    (
                        None
                        if initial_points is None
                        else cv2.UMat(np.ascontiguousarray(initial_points))
                    ),
                    **options,
                )
            )
            self._opencl_lk_gpu_calls += 1
        except Exception as exc:
            self._disable_opencl_lk(
                f"opencl_lk_failed:{type(exc).__name__}:{exc}"
            )
            return cpu_result()

        if self._opencl_lk_state == "validating":
            reference = cpu_result()
            if not self._lk_results_match(reference, gpu_result):
                self._disable_opencl_lk("opencl_lk_parity_mismatch")
                return reference
            self._opencl_lk_validation_passes += 1
            self._opencl_lk_validation_remaining = max(
                0, self._opencl_lk_validation_remaining - 1
            )
            if self._opencl_lk_validation_remaining == 0:
                self._opencl_lk_state = "active"
            return reference
        return gpu_result

    @staticmethod
    def _clone_track_state(state: _TrackState) -> _TrackState:
        """Copy mutable measurements while sharing immutable frame images."""

        candidate = copy.copy(state)
        for field in (
            "anchor_px",
            "support_px",
            "registration_anchor_px",
            "registration_support_px",
            "registration_point_policy",
            "last_point_policy",
            "last_velocity_policy",
            "component_offset_px",
            "component_size_px",
            "pending_component_depth_roi",
            "pending_component_anchor_px",
        ):
            value = getattr(state, field, None)
            if isinstance(value, np.ndarray):
                setattr(candidate, field, value.copy())
        candidate.last_observation_rejections = copy.deepcopy(
            getattr(state, "last_observation_rejections", ())
        )
        return candidate

    def activate_rigid_pair(
        self,
        track_ids: tuple[str, str] | list[str],
        *,
        reference_distance_m: float,
        max_distance_error_m: float | None = None,
        observation_prior_required: bool = False,
    ) -> dict[str, Any]:
        """Track two registered material points with one image transform."""

        requested = tuple(str(value or "").strip() for value in track_ids)
        if len(requested) != 2 or any(not value for value in requested):
            raise ValueError("rigid pair requires exactly two track ids")
        if requested[0] == requested[1]:
            raise ValueError("rigid pair track ids must be unique")
        reference = float(reference_distance_m)
        if not math.isfinite(reference) or reference <= 1.0e-6:
            raise ValueError("reference_distance_m must be positive and finite")
        maximum = (
            max(0.012, min(0.025, 0.25 * reference))
            if max_distance_error_m is None
            else float(max_distance_error_m)
        )
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_distance_error_m must be positive and finite")
        with self._lock:
            missing = [value for value in requested if value not in self._tracks]
            if missing:
                raise KeyError(
                    "unknown rigid-pair tracks: " + ", ".join(sorted(missing))
                )
            self._active_rigid_pair = {
                "track_ids": requested,
                "reference_distance_m": reference,
                "max_distance_error_m": maximum,
                "observation_prior_required": bool(observation_prior_required),
            }
            self._last_rigid_pair_report = {
                "ok": True,
                "active": True,
                "track_ids": list(requested),
                "reference_distance_m": reference,
                "max_distance_error_m": maximum,
                "observation_prior_required": bool(observation_prior_required),
                "source": "registration",
                "build": DYNAMIC_POINT_TRACKER_BUILD,
            }
            return copy.deepcopy(self._last_rigid_pair_report)

    def require_active_rigid_pair_observation_prior(
        self,
        track_ids: tuple[str, str] | list[str],
        *,
        required: bool = True,
    ) -> dict[str, Any]:
        """Require a synchronized caller-owned prior for an active pair.

        The prior is only a candidate gate. It never becomes an observed point;
        every accepted point still comes from the current RGB-D frame.
        """

        requested = tuple(str(value or "").strip() for value in track_ids)
        with self._lock:
            pair = self._active_rigid_pair
            if pair is None:
                raise ValueError("no rigid pair is active")
            if requested != tuple(pair["track_ids"]):
                raise ValueError("rigid-pair observation-prior ids do not match")
            pair["observation_prior_required"] = bool(required)
            return {
                "ok": True,
                "active": True,
                "track_ids": list(requested),
                "observation_prior_required": bool(required),
                "build": DYNAMIC_POINT_TRACKER_BUILD,
            }

    def deactivate_rigid_pair(self) -> None:
        with self._lock:
            self._active_rigid_pair = None
            self._last_rigid_pair_report = {}

    def rigid_pair_report(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._last_rigid_pair_report)

    def ingest(
        self,
        frame: TrackerFrame,
        *,
        rigid_pair_observation_prior: Mapping[str, Any] | None = None,
    ) -> tuple[DynamicPointSnapshot, ...]:
        """Consume one new synchronized observation and update every track."""

        current = self._validate_frame(frame)
        with self._lock:
            if self._episode_id is None:
                self._episode_id = current.episode_id
                self._frame = current
                return ()
            if current.episode_id != self._episode_id:
                self._tracks.clear()
                self._active_rigid_pair = None
                self._last_rigid_pair_report = {}
                self._opencl_umat_cache = {}
                self._episode_id = current.episode_id
                self._frame = current
                return ()
            if self._frame is not None and current.sequence <= self._frame.sequence:
                raise ValueError(
                    "observation sequence must increase strictly within an episode"
                )

            previous = self._tracks
            tentative: dict[str, _TrackState] = {}
            pair_track_ids: set[str] = set()
            prior_track_ids: set[str] = set()
            normalized_prior: dict[str, Any] | None = None
            prior_rejection = "required_observation_prior_unavailable"
            if rigid_pair_observation_prior is not None:
                try:
                    prior_track_ids = {
                        str(value)
                        for value in rigid_pair_observation_prior.get(
                            "track_ids", ()
                        )
                    }
                except (AttributeError, TypeError):
                    prior_track_ids = set()
                normalized_prior, prior_rejection = (
                    self._normalize_observation_prior(
                        rigid_pair_observation_prior,
                        current,
                    )
                )
            pair = self._active_rigid_pair
            if pair is not None:
                requested = tuple(pair["track_ids"])
                if all(track_id in previous for track_id in requested):
                    pair_track_ids.update(requested)
                    pair_prior = (
                        normalized_prior
                        if normalized_prior is not None
                        and tuple(normalized_prior["track_ids"]) == requested
                        else None
                    )
                    pair_prior_rejection = prior_rejection
                    if normalized_prior is not None and pair_prior is None:
                        pair_prior_rejection = "observation_prior_track_mismatch"
                    if (
                        bool(pair.get("observation_prior_required", False))
                        and pair_prior is None
                    ):
                        self._last_rigid_pair_report = {
                            "ok": False,
                            "active": True,
                            "track_ids": list(requested),
                            "observation_sequence": int(current.sequence),
                            "reason": pair_prior_rejection,
                            "candidate_points_published": False,
                            "build": DYNAMIC_POINT_TRACKER_BUILD,
                        }
                        for track_id in requested:
                            tentative[track_id] = self._mark_unobserved(
                                previous[track_id], current
                            )
                        pair_result = "prior_unavailable"
                    else:
                        pair_result = None
                    if pair_result is None:
                        pair_result = self._advance_rigid_pair(
                            [previous[track_id] for track_id in requested],
                            current,
                            reference_distance_m=float(
                                pair["reference_distance_m"]
                            ),
                            max_distance_error_m=float(
                                pair["max_distance_error_m"]
                            ),
                            observation_prior=pair_prior,
                        )
                    if pair_result is None:
                        self._last_rigid_pair_report = {
                            "ok": False,
                            "active": True,
                            "track_ids": list(requested),
                            "reference_distance_m": float(
                                pair["reference_distance_m"]
                            ),
                            "max_distance_error_m": float(
                                pair["max_distance_error_m"]
                            ),
                            "observation_sequence": int(current.sequence),
                            "reason": "joint_rgbd_observation_unavailable",
                            "build": DYNAMIC_POINT_TRACKER_BUILD,
                        }
                        for track_id in requested:
                            tentative[track_id] = self._mark_unobserved(
                                previous[track_id], current
                            )
                    elif pair_result != "prior_unavailable":
                        pair_states, pair_report = pair_result
                        tentative.update(pair_states)
                        self._last_rigid_pair_report = pair_report
                else:
                    self._active_rigid_pair = None
                    self._last_rigid_pair_report = {
                        "ok": False,
                        "active": False,
                        "reason": "rigid_pair_track_removed",
                        "build": DYNAMIC_POINT_TRACKER_BUILD,
                    }
            for track_id, state in previous.items():
                if track_id in pair_track_ids:
                    continue
                # LOST is a public confidence state, not destruction of the
                # policy-owned registration template. The owning memory decides
                # whether to retire the track; while it deliberately retains a
                # track for motion, keep trying current RGB-D for reacquisition.
                if track_id in prior_track_ids and normalized_prior is None:
                    unavailable = self._mark_unobserved(state, current)
                    unavailable.last_observation_rejections = (
                        {
                            "ok": False,
                            "reason": prior_rejection,
                            "candidate_method": "eef_observation_prior",
                            "prediction_values_published_as_observation": False,
                        },
                    )
                    tentative[track_id] = unavailable
                    continue
                point_prior = (
                    None
                    if normalized_prior is None
                    else normalized_prior["points"].get(track_id)
                )
                tentative[track_id] = self._advance_track(
                    state,
                    current,
                    observation_prior=point_prior,
                )

            ambiguous = self._ambiguous_tracks(tentative)
            updated: dict[str, _TrackState] = {}
            for track_id, candidate in tentative.items():
                if track_id in ambiguous:
                    updated[track_id] = self._mark_unobserved(
                        previous[track_id],
                        current,
                        status=AMBIGUOUS,
                    )
                else:
                    updated[track_id] = candidate
            self._tracks = updated
            self._frame = current
            return self.list_points()

    def _normalize_observation_prior(
        self,
        raw: Mapping[str, Any] | None,
        frame: _ValidatedFrame,
        *,
        expected_track_ids: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Validate and project one caller-owned N-point current-frame prior."""

        if raw is None:
            return None, "required_observation_prior_unavailable"
        try:
            prior_ids = tuple(str(value) for value in raw.get("track_ids", ()))
            if (
                not 1 <= len(prior_ids) <= 6
                or any(not track_id for track_id in prior_ids)
                or len(set(prior_ids)) != len(prior_ids)
            ):
                return None, "observation_prior_track_ids_invalid"
            if expected_track_ids is not None and prior_ids != tuple(
                expected_track_ids
            ):
                return None, "observation_prior_track_mismatch"
            if any(track_id not in self._tracks for track_id in prior_ids):
                return None, "observation_prior_track_mismatch"
            if int(raw.get("observation_sequence", -1)) != int(frame.sequence):
                return None, "observation_prior_sequence_mismatch"
            if str(raw.get("episode_id") or "") != str(frame.episode_id):
                return None, "observation_prior_episode_mismatch"
            source = str(raw.get("source") or "").strip()
            if source != "submission_local_fk_from_current_evaluator_proprio":
                return None, "observation_prior_source_invalid"
            points = np.asarray(raw.get("points_robot_base_m"), dtype=np.float64)
            if points.shape != (len(prior_ids), 3) or not np.all(
                np.isfinite(points)
            ):
                return None, "observation_prior_points_invalid"
        except (TypeError, ValueError):
            return None, "observation_prior_invalid"

        normalized_points: dict[str, dict[str, Any]] = {}
        for track_id, point in zip(prior_ids, points):
            pixel, depth = self._project_robot_base_point(point, frame)
            if pixel is None or depth is None:
                return None, "observation_prior_out_of_view"
            normalized_points[track_id] = {
                "point_robot_base_m": point.copy(),
                "expected_pixel": pixel.copy(),
                "expected_depth_m": float(depth),
            }
        return {
            "track_ids": tuple(prior_ids),
            "observation_sequence": int(frame.sequence),
            "episode_id": str(frame.episode_id),
            "source": source,
            "points": normalized_points,
            "max_pixel_error_px": float(
                self.config.rigid_pair_prior_max_pixel_error_px
            ),
            "max_depth_error_m": float(
                self.config.rigid_pair_prior_max_depth_error_m
            ),
            "max_point_error_m": float(
                self.config.rigid_pair_prior_max_point_error_m
            ),
        }, "ok"

    def _normalize_rigid_pair_observation_prior(
        self,
        raw: Mapping[str, Any] | None,
        track_ids: tuple[str, str],
        frame: _ValidatedFrame,
    ) -> tuple[dict[str, Any] | None, str]:
        """Compatibility wrapper retaining the exact two-point contract."""

        return self._normalize_observation_prior(
            raw,
            frame,
            expected_track_ids=tuple(track_ids),
        )

    def _advance_rigid_pair(
        self,
        states: list[_TrackState],
        frame: _ValidatedFrame,
        *,
        reference_distance_m: float,
        max_distance_error_m: float,
        observation_prior: dict[str, Any] | None,
    ) -> tuple[dict[str, _TrackState], dict[str, Any]] | None:
        """Prefer a shared visual transform, then accept only a joint fallback."""

        rejected_shared: list[dict[str, Any]] = []
        for source in ("registration_keyframe", "previous_observation"):
            result = self._advance_rigid_pair_feature(
                states,
                frame,
                use_registration=(source == "registration_keyframe"),
                reference_distance_m=reference_distance_m,
                max_distance_error_m=max_distance_error_m,
                observation_prior=observation_prior,
            )
            if result is not None:
                updated, feature_report = result
                validation = self._rigid_pair_candidate_report(
                    states,
                    updated,
                    frame,
                    reference_distance_m=reference_distance_m,
                    max_distance_error_m=max_distance_error_m,
                    require_identity_assignment=False,
                    validate_individual_prediction=False,
                    observation_prior=observation_prior,
                )
                if bool(validation.get("ok")):
                    validation.update(
                        {
                            "source": f"shared_similarity_{source}",
                            "shared_similarity": feature_report,
                        }
                    )
                    return updated, validation
                rejected_shared.append(
                    {
                        "source": f"shared_similarity_{source}",
                        "validation": validation,
                    }
                )

        prior_points = (
            {} if observation_prior is None else observation_prior["points"]
        )
        individual = {
            state.track_id: self._advance_track(
                state,
                frame,
                observation_prior=prior_points.get(state.track_id),
            )
            for state in states
        }
        report = self._rigid_pair_candidate_report(
            states,
            individual,
            frame,
            reference_distance_m=reference_distance_m,
            max_distance_error_m=max_distance_error_m,
            require_identity_assignment=True,
            validate_individual_prediction=(observation_prior is None),
            observation_prior=observation_prior,
        )
        if bool(report.get("ok")):
            report["source"] = "jointly_validated_individual_candidates"
            report["rejected_shared_candidates"] = rejected_shared
            return individual, report

        per_point = dict(report.get("per_point") or {})
        reject_all = str(report.get("reason") or "").startswith("rigid_pair_")
        isolated: dict[str, _TrackState] = {}
        published: list[str] = []
        rejected: list[str] = []
        for state in states:
            candidate = individual[state.track_id]
            point_ok = bool(
                candidate.snapshot.status == OBSERVED
                and per_point.get(state.track_id, {}).get("ok")
                and not reject_all
            )
            if point_ok:
                isolated[state.track_id] = candidate
                published.append(state.track_id)
            else:
                isolated[state.track_id] = self._mark_unobserved(state, frame)
                rejected.append(state.track_id)
        report.update(
            {
                "source": "isolated_individual_candidate_rejection",
                "candidate_points_published": published,
                "rejected_track_ids": rejected,
                "rejected_shared_candidates": rejected_shared,
                "prediction_values_published_as_observation": False,
            }
        )
        return isolated, report

    def _advance_rigid_pair_feature(
        self,
        states: list[_TrackState],
        frame: _ValidatedFrame,
        *,
        use_registration: bool,
        reference_distance_m: float,
        max_distance_error_m: float,
        observation_prior: dict[str, Any] | None,
    ) -> tuple[dict[str, _TrackState], dict[str, Any]] | None:
        """Estimate one similarity transform for both rigid point identities."""

        if len(states) != 2:
            return None
        reference_images = [
            state.registration_gray if use_registration else state.reference_gray
            for state in states
        ]
        if (
            reference_images[0].shape != reference_images[1].shape
            or not np.array_equal(reference_images[0], reference_images[1])
        ):
            return None
        source_anchors = [
            state.registration_anchor_px if use_registration else state.anchor_px
            for state in states
        ]
        source_supports = [
            state.registration_support_px if use_registration else state.support_px
            for state in states
        ]
        if sum(len(value) for value in source_supports) < max(
            4, self.config.min_inlier_points
        ):
            return None

        points = np.concatenate(source_supports, axis=0).astype(np.float64)
        group_ids = np.concatenate(
            [
                np.full(len(value), index, dtype=np.int64)
                for index, value in enumerate(source_supports)
            ]
        )
        prior_points = (
            {} if observation_prior is None else observation_prior["points"]
        )
        predicted_anchors = []
        for state in states:
            point_prior = prior_points.get(state.track_id)
            predicted = (
                None
                if point_prior is None
                else np.asarray(point_prior["expected_pixel"], dtype=np.float64)
            )
            if predicted is None:
                predicted, _depth = self._project_state_point(state, frame)
            predicted_anchors.append(
                state.anchor_px.copy() if predicted is None else predicted
            )
        shared_delta = np.mean(predicted_anchors, axis=0) - np.mean(
            source_anchors, axis=0
        )
        previous_points = points.astype(np.float32).reshape(-1, 1, 2)
        initial_points = (
            points + shared_delta.reshape(1, 2)
        ).astype(np.float32).reshape(-1, 1, 2)
        lk_options = {
            "winSize": (self.config.lk_window_px, self.config.lk_window_px),
            "maxLevel": self.config.lk_max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.config.lk_max_iterations,
                0.01,
            ),
        }
        next_points, forward_status, _ = self._calc_sparse_lk(
            reference_images[0],
            frame.gray,
            previous_points,
            initial_points,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            **lk_options,
        )
        if next_points is None or forward_status is None:
            return None
        back_points, backward_status, _ = self._calc_sparse_lk(
            frame.gray,
            reference_images[0],
            next_points,
            None,
            **lk_options,
        )
        if back_points is None or backward_status is None:
            return None

        next_flat = next_points.reshape(-1, 2).astype(np.float64)
        back_flat = back_points.reshape(-1, 2).astype(np.float64)
        forward_backward = np.linalg.norm(back_flat - points, axis=1)
        height, width = frame.gray.shape
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(forward_backward)
            & (forward_backward <= self.config.forward_backward_max_error_px)
            & (next_flat[:, 0] >= 0.0)
            & (next_flat[:, 0] <= width - 1.0)
            & (next_flat[:, 1] >= 0.0)
            & (next_flat[:, 1] <= height - 1.0)
        )
        if int(np.count_nonzero(valid)) < max(4, self.config.min_inlier_points):
            return None
        source_good = points[valid]
        next_good = next_flat[valid]
        group_good = group_ids[valid]
        fb_good = forward_backward[valid]
        matrix, mask = cv2.estimateAffinePartial2D(
            source_good.astype(np.float32),
            next_good.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.affine_ransac_error_px,
            maxIters=3000,
            confidence=0.997,
            refineIters=15,
        )
        if matrix is None or mask is None:
            return None
        inliers = mask.reshape(-1).astype(bool)
        if int(np.count_nonzero(inliers)) < max(4, self.config.min_inlier_points):
            return None
        source_good = source_good[inliers]
        next_good = next_good[inliers]
        group_good = group_good[inliers]
        fb_good = fb_good[inliers]
        affine_ok, scale, rotation_deg = self._affine_is_plausible(matrix)
        if not affine_ok:
            return None
        residual = np.linalg.norm(
            self._apply_affine(matrix, source_good) - next_good,
            axis=1,
        )
        residual_ok = residual <= self.config.affine_max_residual_px
        if int(np.count_nonzero(residual_ok)) < max(
            4, self.config.min_inlier_points
        ):
            return None
        next_good = next_good[residual_ok]
        group_good = group_good[residual_ok]
        fb_good = fb_good[residual_ok]
        residual = residual[residual_ok]

        anchors = [
            self._apply_affine(
                matrix,
                np.asarray(anchor, dtype=np.float64).reshape(1, 2),
            )[0]
            for anchor in source_anchors
        ]
        correlations = []
        depths = []
        for index, (state, source_anchor, anchor, predicted) in enumerate(
            zip(states, source_anchors, anchors, predicted_anchors)
        ):
            if (
                not self._pixel_in_bounds(anchor, width, height)
                or float(np.linalg.norm(anchor - predicted))
                > self.config.max_anchor_step_px
            ):
                return None
            correlation = self._warped_patch_correlation(
                reference_images[index],
                frame.gray,
                source_anchor,
                matrix,
            )
            correlations.append(float(correlation))
            group_depths = [
                self._sample_depth(frame.depth, point)
                for point in next_good[group_good == index]
            ]
            numeric_group_depths = [
                float(value) for value in group_depths if value is not None
            ]
            depth_reference = (
                float(np.median(numeric_group_depths))
                if numeric_group_depths
                else state.last_depth_m
            )
            depth = self._sample_depth(
                frame.depth,
                anchor,
                reference_depth=depth_reference,
            )
            if depth is None:
                return None
            depths.append(float(depth))
        if (
            float(np.mean(correlations)) < self.config.min_patch_correlation
            or min(correlations) < 0.20
        ):
            return None

        point_cameras = [
            self._unproject(anchor, depth, frame.intrinsics)
            for anchor, depth in zip(anchors, depths)
        ]
        measured = np.asarray(
            [
                self._transform_point(frame.camera_to_robot_base, point)
                for point in point_cameras
            ],
            dtype=np.float64,
        )
        measured_distance = float(np.linalg.norm(measured[1] - measured[0]))
        distance_error = abs(measured_distance - reference_distance_m)
        if distance_error > max_distance_error_m:
            return None

        support_ratio = min(
            1.0,
            len(next_good) / max(1.0, float(len(points))),
        )
        quality = float(
            np.clip(
                0.30 * support_ratio
                + 0.20
                * (
                    1.0
                    - min(
                        1.0,
                        float(np.median(fb_good))
                        / self.config.forward_backward_max_error_px,
                    )
                )
                + 0.20
                * (
                    1.0
                    - min(
                        1.0,
                        float(np.median(residual))
                        / self.config.affine_max_residual_px,
                    )
                )
                + 0.20 * float(np.mean(correlations))
                + 0.10
                * (1.0 - min(1.0, distance_error / max_distance_error_m)),
                0.0,
                1.0,
            )
        )
        updated: dict[str, _TrackState] = {}
        for index, state in enumerate(states):
            retained = next_good[group_good == index]
            updated[state.track_id] = self._commit_pair_observation(
                state,
                frame,
                anchor=anchors[index],
                depth=depths[index],
                point_camera=point_cameras[index],
                retained_supports=retained,
                confidence=quality,
                observation_method=(
                    "shared_similarity_registration_keyframe"
                    if use_registration
                    else "shared_similarity_previous_observation"
                ),
            )
        report = {
            "ok": True,
            "active": True,
            "track_ids": [state.track_id for state in states],
            "reference_distance_m": float(reference_distance_m),
            "measured_distance_m": measured_distance,
            "distance_error_m": distance_error,
            "max_distance_error_m": float(max_distance_error_m),
            "observation_sequence": int(frame.sequence),
            "inlier_count": int(len(next_good)),
            "inliers_by_track": {
                states[index].track_id: int(np.count_nonzero(group_good == index))
                for index in range(2)
            },
            "scale": float(scale),
            "rotation_deg": float(rotation_deg),
            "patch_correlation": correlations,
            "build": DYNAMIC_POINT_TRACKER_BUILD,
        }
        return updated, report

    def _commit_pair_observation(
        self,
        state: _TrackState,
        frame: _ValidatedFrame,
        *,
        anchor: np.ndarray,
        depth: float,
        point_camera: np.ndarray,
        retained_supports: np.ndarray,
        confidence: float,
        observation_method: str,
    ) -> _TrackState:
        candidate = self._clone_track_state(state)
        point_policy = self._transform_point(
            frame.camera_to_policy, point_camera
        )
        elapsed = float(frame.timestamp_s - state.last_seen_timestamp_s)
        velocity = None
        if elapsed > 1.0e-9:
            velocity = (point_policy - state.last_point_policy) / elapsed
        fresh = self._select_support_points(
            frame.gray, frame.depth, anchor, depth
        )
        supports = self._merge_support_points(retained_supports, fresh)
        snapshot = self._observed_snapshot(
            track_id=state.track_id,
            label=state.label,
            frame=frame,
            anchor=anchor,
            depth=depth,
            point_camera=point_camera,
            point_policy=point_policy,
            velocity=velocity,
            confidence=confidence,
            missed_steps=0,
        )
        candidate.anchor_px = np.asarray(anchor, dtype=np.float64).copy()
        candidate.support_px = supports.copy()
        candidate.reference_gray = frame.gray
        candidate.last_depth_m = float(depth)
        candidate.last_point_policy = point_policy.copy()
        candidate.last_velocity_policy = (
            None if velocity is None else velocity.copy()
        )
        candidate.last_seen_timestamp_s = float(frame.timestamp_s)
        candidate.last_seen_sequence = int(frame.sequence)
        candidate.missed_steps = 0
        candidate.confidence = float(confidence)
        candidate.last_observation_method = str(observation_method)
        self._defer_depth_component_update(
            candidate, frame.depth, anchor, depth
        )
        candidate.snapshot = snapshot
        return candidate

    def _rigid_pair_candidate_report(
        self,
        previous_states: list[_TrackState],
        candidates: dict[str, _TrackState],
        frame: _ValidatedFrame,
        *,
        reference_distance_m: float,
        max_distance_error_m: float,
        require_identity_assignment: bool,
        validate_individual_prediction: bool,
        observation_prior: dict[str, Any] | None,
    ) -> dict[str, Any]:
        ordered = [candidates[state.track_id] for state in previous_states]
        prior_points = (
            {} if observation_prior is None else observation_prior["points"]
        )
        per_point: dict[str, dict[str, Any]] = {}
        measured_rows: list[np.ndarray] = []
        for previous, candidate in zip(previous_states, ordered):
            track_id = previous.track_id
            point_report: dict[str, Any] = {
                "ok": True,
                "status": candidate.snapshot.status,
                "candidate_method": candidate.last_observation_method,
                "candidate_confidence": float(candidate.snapshot.confidence),
                "prediction_values_published_as_observation": False,
            }
            candidate_rejections = tuple(
                getattr(candidate, "last_observation_rejections", ())
            )
            if candidate_rejections:
                point_report["rejected_candidates_before_acceptance"] = (
                    copy.deepcopy(candidate_rejections)
                )
            if candidate.snapshot.status != OBSERVED:
                point_report.update(
                    {"ok": False, "reason": "pair_candidate_unobserved"}
                )
                per_point[track_id] = point_report
                continue
            anchor = np.asarray(candidate.anchor_px, dtype=np.float64)
            depth = float(candidate.snapshot.depth_m)
            measured_point = np.asarray(
                candidate.snapshot.point_robot_base_m, dtype=np.float64
            )
            measured_rows.append(measured_point)
            point_report.update(
                {
                    "candidate_pixel": anchor.astype(float).tolist(),
                    "candidate_depth_m": depth,
                    "raw_candidate_point_robot_base_m": (
                        measured_point.astype(float).tolist()
                    ),
                }
            )
            if observation_prior is None:
                static_gate = self._static_registration_candidate_gate(
                    previous,
                    candidate,
                    frame,
                )
                point_report["static_registration"] = static_gate
                if not static_gate.get("ok", True):
                    point_report.update(
                        {
                            "ok": False,
                            "reason": static_gate.get(
                                "reason", "static_registration_drift_exceeded"
                            ),
                        }
                    )
            expected = prior_points.get(track_id)
            if expected is not None:
                expected_pixel = np.asarray(
                    expected["expected_pixel"], dtype=np.float64
                )
                expected_point = np.asarray(
                    expected["point_robot_base_m"], dtype=np.float64
                )
                expected_depth = float(expected["expected_depth_m"])
                pixel_error = float(np.linalg.norm(anchor - expected_pixel))
                depth_error = abs(depth - expected_depth)
                point_error = float(np.linalg.norm(measured_point - expected_point))
                point_report.update(
                    {
                        "gate_source": "explicit_on_hand_eef_local_fk",
                        "predicted_pixel": expected_pixel.astype(float).tolist(),
                        "predicted_depth_m": expected_depth,
                        "predicted_point_robot_base_m": (
                            expected_point.astype(float).tolist()
                        ),
                        "pixel_error_px": pixel_error,
                        "depth_layer_error_m": depth_error,
                        "point_error_m": point_error,
                        "pixel_error_limit_px": float(
                            observation_prior["max_pixel_error_px"]
                        ),
                        "depth_layer_error_limit_m": float(
                            observation_prior["max_depth_error_m"]
                        ),
                        "point_error_limit_m": float(
                            observation_prior["max_point_error_m"]
                        ),
                    }
                )
                if pixel_error > float(observation_prior["max_pixel_error_px"]):
                    point_report.update(
                        {"ok": False, "reason": "eef_prior_pixel_mismatch"}
                    )
                elif depth_error > float(observation_prior["max_depth_error_m"]):
                    point_report.update(
                        {"ok": False, "reason": "eef_prior_depth_layer_mismatch"}
                    )
                elif point_error > float(observation_prior["max_point_error_m"]):
                    point_report.update(
                        {"ok": False, "reason": "eef_prior_point_mismatch"}
                    )
            elif validate_individual_prediction:
                predicted_pixel, predicted_depth = self._project_state_point(
                    previous, frame
                )
                if predicted_pixel is None or predicted_depth is None:
                    point_report.update(
                        {"ok": False, "reason": "individual_prediction_unavailable"}
                    )
                else:
                    pixel_error = float(
                        np.linalg.norm(anchor - predicted_pixel)
                    )
                    depth_error = abs(depth - float(predicted_depth))
                    point_report.update(
                        {
                            "gate_source": "per_point_motion_prediction",
                            "predicted_pixel": predicted_pixel.astype(float).tolist(),
                            "predicted_depth_m": float(predicted_depth),
                            "pixel_error_px": pixel_error,
                            "depth_layer_error_m": depth_error,
                            "pixel_error_limit_px": float(
                                self.config.rigid_pair_individual_max_prediction_error_px
                            ),
                            "depth_layer_error_limit_m": float(
                                self.config.rigid_pair_individual_max_depth_error_m
                            ),
                        }
                    )
                    if pixel_error > float(
                        self.config.rigid_pair_individual_max_prediction_error_px
                    ):
                        point_report.update(
                            {
                                "ok": False,
                                "reason": "individual_prediction_pixel_mismatch",
                            }
                        )
                    elif depth_error > float(
                        self.config.rigid_pair_individual_max_depth_error_m
                    ):
                        point_report.update(
                            {
                                "ok": False,
                                "reason": "individual_prediction_depth_layer_mismatch",
                            }
                        )
            per_point[track_id] = point_report

        all_observed = len(measured_rows) == len(ordered)
        measured_distance = (
            float(np.linalg.norm(measured_rows[1] - measured_rows[0]))
            if all_observed
            else None
        )
        distance_error = (
            abs(float(measured_distance) - reference_distance_m)
            if measured_distance is not None
            else None
        )
        identity_ok = True
        named_cost = None
        swapped_cost = None
        if require_identity_assignment:
            predicted = []
            for state in previous_states:
                expected = prior_points.get(state.track_id)
                anchor = (
                    None
                    if expected is None
                    else np.asarray(expected["expected_pixel"], dtype=np.float64)
                )
                if anchor is None:
                    anchor, _depth = self._project_state_point(state, frame)
                predicted.append(state.anchor_px if anchor is None else anchor)
            anchors = [state.anchor_px for state in ordered]
            named_cost = float(
                np.linalg.norm(anchors[0] - predicted[0])
                + np.linalg.norm(anchors[1] - predicted[1])
            )
            swapped_cost = float(
                np.linalg.norm(anchors[0] - predicted[1])
                + np.linalg.norm(anchors[1] - predicted[0])
            )
            identity_ok = bool(
                named_cost <= swapped_cost
                and swapped_cost - named_cost
                >= self.config.rigid_pair_identity_min_margin_px
            )
        point_gates_ok = bool(
            per_point and all(value.get("ok") for value in per_point.values())
        )
        ok = bool(
            all_observed
            and point_gates_ok
            and distance_error is not None
            and distance_error <= max_distance_error_m
            and identity_ok
        )
        if not all_observed:
            reason = "pair_candidate_unobserved"
        elif not point_gates_ok:
            reason = "individual_candidate_gate_rejected"
        elif distance_error is not None and distance_error > max_distance_error_m:
            reason = "rigid_pair_distance_mismatch"
        elif not identity_ok:
            reason = "rigid_pair_identity_ambiguous"
        else:
            reason = None
        return {
            "ok": ok,
            "active": True,
            "track_ids": [state.track_id for state in previous_states],
            "reference_distance_m": float(reference_distance_m),
            "measured_distance_m": measured_distance,
            "distance_error_m": distance_error,
            "max_distance_error_m": float(max_distance_error_m),
            "identity_assignment_ok": identity_ok,
            "named_assignment_cost_px": named_cost,
            "swapped_assignment_cost_px": swapped_cost,
            "identity_assignment_margin_px": (
                None
                if named_cost is None or swapped_cost is None
                else float(swapped_cost - named_cost)
            ),
            "identity_assignment_min_margin_px": float(
                self.config.rigid_pair_identity_min_margin_px
            ),
            "per_point": per_point,
            "raw_candidate_points_robot_base_m": (
                [point.astype(float).tolist() for point in measured_rows]
                if all_observed
                else None
            ),
            "candidate_points_published": (
                [state.track_id for state in previous_states] if ok else []
            ),
            "prediction_values_published_as_observation": False,
            "observation_sequence": int(frame.sequence),
            "reason": reason,
            "build": DYNAMIC_POINT_TRACKER_BUILD,
        }

    def mark_point(
        self,
        *,
        u: float,
        v: float,
        label: str,
        track_id: str | None = None,
    ) -> DynamicPointSnapshot:
        """Mark one 0..1000 UV point in the most recently ingested frame."""

        with self._lock:
            frame = self._frame
            if frame is None:
                raise ValueError("an evaluator frame must be ingested before marking")
            normalized_label = str(label or "").strip()
            if not normalized_label:
                raise ValueError("label is required")
            if len(normalized_label) > 128:
                raise ValueError("label is too long")
            anchor = self._relative_to_pixel(
                u,
                v,
                frame.intrinsics.width,
                frame.intrinsics.height,
            )
            depth = self._sample_depth(frame.depth, anchor)
            if depth is None:
                raise ValueError("clicked point has no valid evaluator depth")
            supports = self._select_support_points(
                frame.gray,
                frame.depth,
                anchor,
                depth,
            )
            component = self._depth_component_at_anchor(
                frame.depth,
                anchor,
                depth,
            )
            if component is None:
                raise ValueError("clicked point has no connected evaluator depth")

            chosen_id = self._allocate_track_id(track_id)
            point_camera = self._unproject(anchor, depth, frame.intrinsics)
            point_policy = self._transform_point(
                frame.camera_to_policy,
                point_camera,
            )
            snapshot = self._observed_snapshot(
                track_id=chosen_id,
                label=normalized_label,
                frame=frame,
                anchor=anchor,
                depth=depth,
                point_camera=point_camera,
                point_policy=point_policy,
                velocity=None,
                confidence=self._registration_confidence(len(supports)),
                missed_steps=0,
            )
            self._tracks[chosen_id] = _TrackState(
                track_id=chosen_id,
                label=normalized_label,
                camera_role=self.camera_role,
                anchor_px=anchor.copy(),
                support_px=supports.copy(),
                reference_gray=frame.gray,
                registration_anchor_px=anchor.copy(),
                registration_support_px=supports.copy(),
                registration_gray=frame.gray,
                registration_point_policy=point_policy.copy(),
                registration_depth_m=float(depth),
                last_depth_m=float(depth),
                last_point_policy=point_policy.copy(),
                last_velocity_policy=None,
                last_seen_timestamp_s=float(frame.timestamp_s),
                last_seen_sequence=int(frame.sequence),
                missed_steps=0,
                confidence=self._registration_confidence(len(supports)),
                component_offset_px=anchor - component.centroid_px,
                component_size_px=component.size_px.copy(),
                component_area_px=int(component.area_px),
                snapshot=snapshot,
            )
            return snapshot

    def get_point(self, track_id: str) -> DynamicPointSnapshot:
        with self._lock:
            key = str(track_id or "").strip()
            if key not in self._tracks:
                raise KeyError(f"unknown dynamic point track {key!r}")
            return self._tracks[key].snapshot

    def list_points(self) -> tuple[DynamicPointSnapshot, ...]:
        with self._lock:
            return tuple(
                self._tracks[key].snapshot for key in sorted(self._tracks)
            )

    def remove_point(self, track_id: str) -> bool:
        with self._lock:
            key = str(track_id or "").strip()
            removed = self._tracks.pop(key, None) is not None
            if removed and not self._tracks:
                self._opencl_umat_cache = {}
            pair = self._active_rigid_pair
            if pair is not None and key in pair["track_ids"]:
                self._active_rigid_pair = None
                self._last_rigid_pair_report = {
                    "ok": False,
                    "active": False,
                    "reason": "rigid_pair_track_removed",
                    "build": DYNAMIC_POINT_TRACKER_BUILD,
                }
            return removed

    def _allocate_track_id(self, requested: str | None) -> str:
        if requested is not None:
            key = str(requested).strip()
            if not key:
                raise ValueError("track_id cannot be empty")
            if len(key) > 128:
                raise ValueError("track_id is too long")
            if key in self._tracks:
                raise ValueError(f"duplicate track_id {key!r}")
            return key
        while True:
            key = f"track_{self._next_track_number:06d}"
            self._next_track_number += 1
            if key not in self._tracks:
                return key

    def _static_registration_candidate(
        self,
        state: _TrackState,
        frame: _ValidatedFrame,
        *,
        observation_prior: dict[str, Any] | None = None,
    ) -> _TrackState | None:
        """Read the registered surface at its current projected pixel.

        This path is intentionally conservative.  It is useful for static or
        low-texture surfaces where optical flow has no reliable support, but it
        is only accepted while the immutable registration appearance, depth,
        and projected 3-D point still agree. An explicit on-hand prior may
        relocate the search pixel, but cannot bypass the immutable appearance
        or current measured depth checks. A failed check returns no observation;
        it never returns a predicted coordinate.
        """

        if observation_prior is None:
            registration_point = getattr(
                state, "registration_point_policy", state.last_point_policy,
            )
        else:
            try:
                expected_base = np.asarray(
                    observation_prior["point_robot_base_m"], dtype=np.float64,
                ).reshape(3)
            except (KeyError, TypeError, ValueError):
                return None
            base_to_policy = frame.camera_to_policy @ np.linalg.inv(frame.camera_to_robot_base)
            registration_point = self._transform_point(base_to_policy, expected_base)
        registration_point = np.asarray(registration_point, dtype=np.float64)
        if registration_point.shape != (3,) or not np.all(
            np.isfinite(registration_point)
        ):
            return None
        expected_pixel, expected_depth = self._project_policy_point(
            registration_point,
            frame,
        )
        if expected_pixel is None or expected_depth is None:
            return None
        correlation = self._translation_patch_similarity(
            state.registration_gray,
            frame.gray,
            state.registration_anchor_px,
            expected_pixel,
        )
        candidates = [(correlation, expected_pixel)]
        if (
            observation_prior is not None
            and correlation < self.config.static_registration_min_patch_correlation
        ):
            # A force-controlled finger can settle a few millimeters without
            # moving the EEF. Search only inside the existing 3-D error envelope;
            # immutable RGB and current depth must both confirm the shifted point.
            radius = min(
                int(math.ceil(max(frame.intrinsics.fx, frame.intrinsics.fy)
                              * self.config.static_registration_max_point_error_m
                              / float(expected_depth))),
                int(self.config.rigid_pair_prior_max_pixel_error_px),
            )
            height, width = frame.gray.shape
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius or not (dx or dy):
                        continue
                    pixel = expected_pixel + [dx, dy]
                    if not self._pixel_in_bounds(pixel, width, height):
                        continue
                    score = self._translation_patch_similarity(
                        state.registration_gray, frame.gray,
                        state.registration_anchor_px, pixel,
                    )
                    if score >= self.config.static_registration_min_patch_correlation:
                        candidates.append((score, pixel))
            candidates.sort(key=lambda item: (
                -item[0], float(np.sum((item[1] - expected_pixel) ** 2)),
            ))
        measured = None
        for score, pixel in candidates:
            if score < self.config.static_registration_min_patch_correlation:
                continue
            depth = self._sample_depth(frame.depth, pixel, reference_depth=float(expected_depth))
            if (depth is None
                    or abs(float(depth) - float(expected_depth))
                    > self.config.static_registration_max_depth_error_m):
                continue
            point_camera = self._unproject(pixel, depth, frame.intrinsics)
            point_policy = self._transform_point(frame.camera_to_policy, point_camera)
            if (float(np.linalg.norm(point_policy - registration_point))
                    <= self.config.static_registration_max_point_error_m):
                measured = (pixel, score, depth, point_camera, point_policy)
                break
        if measured is None:
            return None
        expected_pixel, correlation, depth, point_camera, point_policy = measured
        elapsed = float(frame.timestamp_s - state.last_seen_timestamp_s)
        velocity = None
        if elapsed > 1.0e-9:
            velocity = (point_policy - state.last_point_policy) / elapsed

        # Registration RGB/support arrays are immutable.  A shallow state
        # copy avoids copying the full 720p registration image for every
        # point on every static frame; mutable measurements are replaced with
        # owned copies below.
        candidate = copy.copy(state)
        # Static-anchor validation already checked the current RGB-D sample.
        # Rebuilding a full corner set and flood-filling a depth component on
        # every frame only adds work and can replace good registration support
        # with noisy low-texture pixels.  Keep the immutable registration
        # support/component summary; the anchor and depth below remain current
        # evaluator measurements.
        supports = state.registration_support_px.copy()
        if observation_prior is not None:
            supports += (expected_pixel - state.registration_anchor_px).astype(supports.dtype)
            height, width = frame.gray.shape
            supports = supports[
                (supports[:, 0] >= 0) & (supports[:, 0] < width)
                & (supports[:, 1] >= 0) & (supports[:, 1] < height)
            ]
        snapshot = self._observed_snapshot(
            track_id=state.track_id,
            label=state.label,
            frame=frame,
            anchor=expected_pixel,
            depth=float(depth),
            point_camera=point_camera,
            point_policy=point_policy,
            velocity=velocity,
            confidence=float(
                np.clip(
                    0.60 + 0.40 * correlation,
                    0.0,
                    0.98,
                )
            ),
            missed_steps=0,
        )
        candidate.anchor_px = expected_pixel.copy()
        candidate.support_px = supports.copy()
        candidate.reference_gray = frame.gray
        candidate.last_depth_m = float(depth)
        candidate.last_point_policy = point_policy.copy()
        candidate.last_velocity_policy = (
            None if velocity is None else velocity.copy()
        )
        candidate.last_seen_timestamp_s = float(frame.timestamp_s)
        candidate.last_seen_sequence = int(frame.sequence)
        candidate.missed_steps = 0
        candidate.confidence = float(snapshot.confidence)
        candidate.last_observation_method = (
            "registration_anchor_static" if observation_prior is None
            else "registration_patch_eef_prior_rgbd"
        )
        candidate.snapshot = snapshot
        return candidate

    def _static_registration_candidate_gate(
        self,
        state: _TrackState,
        candidate: _TrackState,
        frame: _ValidatedFrame,
    ) -> dict[str, Any]:
        """Reject drift while the current frame still matches registration."""

        report: dict[str, Any] = {
            "ok": True,
            "candidate_method": candidate.last_observation_method,
            "prediction_values_published_as_observation": False,
        }
        if candidate.snapshot.status != OBSERVED:
            report["reason"] = "candidate_unobserved"
            return report
        registration_point = getattr(
            state,
            "registration_point_policy",
            state.last_point_policy,
        )
        registration_point = np.asarray(registration_point, dtype=np.float64)
        expected_pixel, expected_depth = self._project_policy_point(
            registration_point,
            frame,
        )
        if expected_pixel is None or expected_depth is None:
            report["reason"] = "registration_projection_unavailable"
            return report
        correlation = self._translation_patch_similarity(
            state.registration_gray,
            frame.gray,
            state.registration_anchor_px,
            expected_pixel,
        )
        report.update(
            {
                "registration_pixel": expected_pixel.astype(float).tolist(),
                "registration_depth_m": float(expected_depth),
                "registration_patch_correlation": float(correlation),
            }
        )
        if correlation < self.config.static_registration_min_patch_correlation:
            report["reason"] = "registration_appearance_changed"
            return report
        static_depth = self._sample_depth(
            frame.depth,
            expected_pixel,
            reference_depth=float(expected_depth),
        )
        if static_depth is None or abs(float(static_depth) - float(expected_depth)) > (
            self.config.static_registration_max_depth_error_m
        ):
            # Repetitive/flat RGB can retain high correlation after an object
            # has moved.  A changed depth layer is therefore evidence against
            # the static registration state, and leaves dynamic reacquisition
            # available to the caller.
            report["reason"] = "registration_depth_changed"
            return report
        try:
            measured = np.asarray(
                candidate.snapshot.point_policy_local_m,
                dtype=np.float64,
            ).reshape(3)
        except (TypeError, ValueError):
            report.update({"ok": False, "reason": "candidate_point_invalid"})
            return report
        point_error = float(np.linalg.norm(measured - registration_point))
        report.update(
            {
                "point_error_m": point_error,
                "point_error_limit_m": float(
                    self.config.static_registration_max_point_error_m
                ),
            }
        )
        if point_error > self.config.static_registration_max_point_error_m:
            report.update(
                {
                    "ok": False,
                    "reason": "static_registration_drift_exceeded",
                }
            )
        else:
            report["reason"] = None
        return report

    def _advance_track(
        self,
        state: _TrackState,
        frame: _ValidatedFrame,
        *,
        observation_prior: dict[str, Any] | None = None,
    ) -> _TrackState:
        # First try a direct registration-keyframe observation.  The previous
        # implementation only chained frame-to-frame affine flow and therefore
        # accumulated subpixel errors into large identity drift on repetitive
        # surfaces.  A successful direct round trip is anchored to the model's
        # original click and cannot accumulate that error.  The existing
        # incremental and depth-component paths remain the fallback when the
        # object has moved too far or changed appearance too much.
        rejected_candidates: list[dict[str, Any]] = []
        # Before optical flow, try the immutable registration point.  When a
        # low-texture surface is unchanged this produces a current RGB-D
        # measurement at a fixed identity anchor and prevents fallback drift.
        # Low-texture on-hand points also need this RGB-D-verified path. An EEF
        # gate changes the search location, not the required visual evidence.
        static_candidate = self._static_registration_candidate(
            state, frame, observation_prior=observation_prior,
        )
        if static_candidate is not None:
            gate = self._observation_prior_candidate_gate(static_candidate, observation_prior)
            if bool(gate.get("ok")):
                static_candidate.last_observation_rejections = tuple(
                    rejected_candidates
                )
                return static_candidate
            rejected_candidates.append(gate)
        if (
            len(state.registration_support_px)
            >= self.config.min_inlier_points
        ):
            registered_state = self._clone_track_state(state)
            registered_state.anchor_px = state.registration_anchor_px.copy()
            registered_state.support_px = state.registration_support_px.copy()
            registered_state.reference_gray = state.registration_gray
            registration_candidate = self._advance_feature_track(
                registered_state,
                frame,
                observation_prior=observation_prior,
            )
            registration_candidate.last_observation_method = (
                "registration_feature"
            )
            registration_gate = self._observation_prior_candidate_gate(
                registration_candidate,
                observation_prior,
            )
            static_gate = (
                {"ok": True, "reason": None}
                if observation_prior is not None
                else self._static_registration_candidate_gate(
                    state,
                    registration_candidate,
                    frame,
                )
            )
            if registration_gate["ok"] and static_gate["ok"]:
                registration_candidate.last_observation_rejections = tuple(
                    rejected_candidates
                )
                return registration_candidate
            rejected_candidates.append(registration_gate)
            if not static_gate["ok"]:
                rejected_candidates.append(static_gate)

        if len(state.support_px) >= self.config.min_inlier_points:
            feature_candidate = self._advance_feature_track(
                self._clone_track_state(state),
                frame,
                observation_prior=observation_prior,
            )
            feature_candidate.last_observation_method = (
                "previous_observation_feature"
            )
            feature_gate = self._observation_prior_candidate_gate(
                feature_candidate,
                observation_prior,
            )
            static_gate = (
                {"ok": True, "reason": None}
                if observation_prior is not None
                else self._static_registration_candidate_gate(
                    state,
                    feature_candidate,
                    frame,
                )
            )
            if feature_gate["ok"] and static_gate["ok"]:
                feature_candidate.last_observation_rejections = tuple(
                    rejected_candidates
                )
                return feature_candidate
            rejected_candidates.append(feature_gate)
            if not static_gate["ok"]:
                rejected_candidates.append(static_gate)

        depth_candidate = self._advance_depth_component_track(
            self._clone_track_state(state),
            frame,
            observation_prior=observation_prior,
        )
        if depth_candidate is not None:
            depth_candidate.last_observation_method = "depth_component"
            depth_gate = self._observation_prior_candidate_gate(
                depth_candidate,
                observation_prior,
            )
            static_gate = (
                {"ok": True, "reason": None}
                if observation_prior is not None
                else self._static_registration_candidate_gate(
                    state,
                    depth_candidate,
                    frame,
                )
            )
            if depth_gate["ok"] and static_gate["ok"]:
                depth_candidate.last_observation_rejections = tuple(
                    rejected_candidates
                )
                return depth_candidate
            rejected_candidates.append(depth_gate)
            if not static_gate["ok"]:
                rejected_candidates.append(static_gate)
        unobserved = self._mark_unobserved(state, frame)
        unobserved.last_observation_rejections = tuple(rejected_candidates)
        return unobserved

    def _observation_prior_candidate_gate(
        self,
        candidate: _TrackState,
        observation_prior: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate one measured RGB-D candidate before accepting it.

        The EEF prior is only a gate. Candidate coordinates always come from
        the current evaluator RGB-D observation and are never synthesized from
        the prior.
        """

        report: dict[str, Any] = {
            "ok": candidate.snapshot.status == OBSERVED,
            "candidate_method": candidate.last_observation_method,
            "candidate_status": candidate.snapshot.status,
        }
        if candidate.snapshot.status != OBSERVED:
            report["reason"] = "candidate_unobserved"
            return report
        if observation_prior is None:
            report["reason"] = None
            return report
        try:
            anchor = np.asarray(candidate.anchor_px, dtype=np.float64).reshape(2)
            depth = float(candidate.snapshot.depth_m)
            measured_point = np.asarray(
                candidate.snapshot.point_robot_base_m,
                dtype=np.float64,
            ).reshape(3)
            expected_pixel = np.asarray(
                observation_prior["expected_pixel"],
                dtype=np.float64,
            ).reshape(2)
            expected_depth = float(observation_prior["expected_depth_m"])
            expected_point = np.asarray(
                observation_prior["point_robot_base_m"],
                dtype=np.float64,
            ).reshape(3)
        except (KeyError, TypeError, ValueError):
            report.update({"ok": False, "reason": "observation_prior_invalid"})
            return report
        if not (
            np.all(np.isfinite(anchor))
            and math.isfinite(depth)
            and np.all(np.isfinite(measured_point))
            and np.all(np.isfinite(expected_pixel))
            and math.isfinite(expected_depth)
            and np.all(np.isfinite(expected_point))
        ):
            report.update({"ok": False, "reason": "candidate_or_prior_nonfinite"})
            return report

        pixel_error = float(np.linalg.norm(anchor - expected_pixel))
        depth_error = abs(depth - expected_depth)
        point_error = float(np.linalg.norm(measured_point - expected_point))
        report.update(
            {
                "candidate_pixel": anchor.astype(float).tolist(),
                "candidate_depth_m": depth,
                "candidate_point_robot_base_m": (
                    measured_point.astype(float).tolist()
                ),
                "predicted_pixel": expected_pixel.astype(float).tolist(),
                "predicted_depth_m": expected_depth,
                "predicted_point_robot_base_m": expected_point.astype(float).tolist(),
                "pixel_error_px": pixel_error,
                "depth_layer_error_m": depth_error,
                "point_error_m": point_error,
                "pixel_error_limit_px": float(
                    self.config.rigid_pair_prior_max_pixel_error_px
                ),
                "depth_layer_error_limit_m": float(
                    self.config.rigid_pair_prior_max_depth_error_m
                ),
                "point_error_limit_m": float(
                    self.config.rigid_pair_prior_max_point_error_m
                ),
            }
        )
        if pixel_error > self.config.rigid_pair_prior_max_pixel_error_px:
            report.update({"ok": False, "reason": "eef_prior_pixel_mismatch"})
        elif depth_error > self.config.rigid_pair_prior_max_depth_error_m:
            report.update(
                {"ok": False, "reason": "eef_prior_depth_layer_mismatch"}
            )
        elif point_error > self.config.rigid_pair_prior_max_point_error_m:
            report.update({"ok": False, "reason": "eef_prior_point_mismatch"})
        else:
            report.update({"ok": True, "reason": None})
        return report

    def _advance_feature_track(
        self,
        state: _TrackState,
        frame: _ValidatedFrame,
        *,
        observation_prior: dict[str, Any] | None = None,
    ) -> _TrackState:
        previous_points = state.support_px.astype(np.float32).reshape(-1, 1, 2)
        predicted_anchor = (
            None
            if observation_prior is None
            else np.asarray(
                observation_prior["expected_pixel"], dtype=np.float64
            )
        )
        if predicted_anchor is None:
            predicted_anchor, _ = self._project_state_point(state, frame)
        initial_points = None
        flow_flags = 0
        if predicted_anchor is not None:
            camera_delta = predicted_anchor - state.anchor_px
            initial_points = (
                previous_points
                + camera_delta.astype(np.float32).reshape(1, 1, 2)
            )
            flow_flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        lk_options = {
            "winSize": (self.config.lk_window_px, self.config.lk_window_px),
            "maxLevel": self.config.lk_max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.config.lk_max_iterations,
                0.01,
            ),
        }
        next_points, forward_status, _ = self._calc_sparse_lk(
            state.reference_gray,
            frame.gray,
            previous_points,
            initial_points,
            flags=flow_flags,
            **lk_options,
        )
        if next_points is None or forward_status is None:
            return self._mark_unobserved(state, frame)
        back_points, backward_status, _ = self._calc_sparse_lk(
            frame.gray,
            state.reference_gray,
            next_points,
            None,
            **lk_options,
        )
        if back_points is None or backward_status is None:
            return self._mark_unobserved(state, frame)

        previous_flat = previous_points.reshape(-1, 2).astype(np.float64)
        next_flat = next_points.reshape(-1, 2).astype(np.float64)
        back_flat = back_points.reshape(-1, 2).astype(np.float64)
        forward_backward = np.linalg.norm(back_flat - previous_flat, axis=1)
        height, width = frame.gray.shape
        in_bounds = (
            (next_flat[:, 0] >= 0.0)
            & (next_flat[:, 0] <= width - 1.0)
            & (next_flat[:, 1] >= 0.0)
            & (next_flat[:, 1] <= height - 1.0)
        )
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(forward_backward)
            & (
                forward_backward
                <= self.config.forward_backward_max_error_px
            )
            & in_bounds
        )
        if int(np.count_nonzero(valid)) < self.config.min_inlier_points:
            return self._mark_unobserved(state, frame)

        previous_good = previous_flat[valid]
        next_good = next_flat[valid]
        matrix, ransac_mask = cv2.estimateAffinePartial2D(
            previous_good.astype(np.float32),
            next_good.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.affine_ransac_error_px,
            maxIters=2000,
            confidence=0.995,
            refineIters=10,
        )
        if matrix is None or ransac_mask is None:
            return self._mark_unobserved(state, frame)
        ransac_inliers = ransac_mask.reshape(-1).astype(bool)
        if int(np.count_nonzero(ransac_inliers)) < self.config.min_inlier_points:
            return self._mark_unobserved(state, frame)
        previous_good = previous_good[ransac_inliers]
        next_good = next_good[ransac_inliers]
        forward_backward_good = forward_backward[valid][ransac_inliers]

        affine_ok, scale, _ = self._affine_is_plausible(matrix)
        if not affine_ok:
            return self._mark_unobserved(state, frame)
        predicted_support = self._apply_affine(matrix, previous_good)
        residual = np.linalg.norm(predicted_support - next_good, axis=1)
        residual_ok = residual <= self.config.affine_max_residual_px
        if int(np.count_nonzero(residual_ok)) < self.config.min_inlier_points:
            return self._mark_unobserved(state, frame)
        previous_good = previous_good[residual_ok]
        next_good = next_good[residual_ok]
        forward_backward_good = forward_backward_good[residual_ok]

        anchor = self._apply_affine(
            matrix,
            state.anchor_px.reshape(1, 2),
        )[0]
        if (
            not np.all(np.isfinite(anchor))
            or (
                observation_prior is None
                and float(np.linalg.norm(anchor - state.anchor_px))
                > self.config.max_anchor_step_px
            )
            or not self._pixel_in_bounds(anchor, width, height)
        ):
            return self._mark_unobserved(state, frame)

        correlation = self._warped_patch_correlation(
            state.reference_gray,
            frame.gray,
            state.anchor_px,
            matrix,
        )
        if correlation < self.config.min_patch_correlation:
            return self._mark_unobserved(state, frame)

        support_depths = np.asarray(
            [self._sample_depth(frame.depth, point) for point in next_good],
            dtype=object,
        )
        depth_valid = np.asarray(
            [value is not None for value in support_depths],
            dtype=bool,
        )
        if int(np.count_nonzero(depth_valid)) < self.config.min_inlier_points:
            return self._mark_unobserved(state, frame)
        previous_good = previous_good[depth_valid]
        next_good = next_good[depth_valid]
        forward_backward_good = forward_backward_good[depth_valid]
        numeric_depths = np.asarray(
            [float(value) for value in support_depths[depth_valid]],
            dtype=np.float64,
        )
        median_depth = float(np.median(numeric_depths))
        layer_tolerance = self._depth_tolerance(median_depth)
        layer_valid = np.abs(numeric_depths - median_depth) <= layer_tolerance
        if int(np.count_nonzero(layer_valid)) < self.config.min_inlier_points:
            return self._mark_unobserved(state, frame)
        previous_good = previous_good[layer_valid]
        next_good = next_good[layer_valid]
        forward_backward_good = forward_backward_good[layer_valid]
        numeric_depths = numeric_depths[layer_valid]

        matrix, final_mask = cv2.estimateAffinePartial2D(
            previous_good.astype(np.float32),
            next_good.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.affine_ransac_error_px,
            maxIters=1000,
            confidence=0.995,
            refineIters=10,
        )
        if matrix is None or final_mask is None:
            return self._mark_unobserved(state, frame)
        final_inliers = final_mask.reshape(-1).astype(bool)
        if int(np.count_nonzero(final_inliers)) < self.config.min_inlier_points:
            return self._mark_unobserved(state, frame)
        previous_good = previous_good[final_inliers]
        next_good = next_good[final_inliers]
        forward_backward_good = forward_backward_good[final_inliers]
        numeric_depths = numeric_depths[final_inliers]
        affine_ok, scale, _ = self._affine_is_plausible(matrix)
        if not affine_ok:
            return self._mark_unobserved(state, frame)

        predicted_support = self._apply_affine(matrix, previous_good)
        residual = np.linalg.norm(predicted_support - next_good, axis=1)
        if float(np.max(residual)) > self.config.affine_max_residual_px:
            return self._mark_unobserved(state, frame)
        anchor = self._apply_affine(
            matrix,
            state.anchor_px.reshape(1, 2),
        )[0]
        if not self._pixel_in_bounds(anchor, width, height):
            return self._mark_unobserved(state, frame)
        correlation = self._warped_patch_correlation(
            state.reference_gray,
            frame.gray,
            state.anchor_px,
            matrix,
        )
        if correlation < self.config.min_patch_correlation:
            return self._mark_unobserved(state, frame)

        median_depth = float(np.median(numeric_depths))
        depth = self._sample_depth(
            frame.depth,
            anchor,
            reference_depth=median_depth,
        )
        if depth is None:
            return self._mark_unobserved(state, frame)

        point_camera = self._unproject(anchor, depth, frame.intrinsics)
        point_policy = self._transform_point(
            frame.camera_to_policy,
            point_camera,
        )
        elapsed = float(frame.timestamp_s - state.last_seen_timestamp_s)
        velocity = None
        if elapsed > 1e-9:
            velocity = (point_policy - state.last_point_policy) / elapsed

        supports = next_good.copy()
        fresh = self._select_support_points(
            frame.gray,
            frame.depth,
            anchor,
            depth,
        )
        supports = self._merge_support_points(supports, fresh)
        if len(supports) < self.config.min_support_points:
            return self._mark_unobserved(state, frame)

        support_ratio = min(
            1.0,
            len(next_good) / max(1.0, float(len(state.support_px))),
        )
        fb_quality = 1.0 - min(
            1.0,
            float(np.median(forward_backward_good))
            / self.config.forward_backward_max_error_px,
        )
        residual_quality = 1.0 - min(
            1.0,
            float(np.median(residual)) / self.config.affine_max_residual_px,
        )
        depth_spread = float(np.median(np.abs(numeric_depths - median_depth)))
        depth_quality = 1.0 - min(
            1.0,
            depth_spread / max(layer_tolerance, 1e-9),
        )
        scale_quality = 1.0 - min(1.0, abs(float(scale) - 1.0))
        confidence = float(
            np.clip(
                0.20 * support_ratio
                + 0.20 * fb_quality
                + 0.20 * residual_quality
                + 0.20 * depth_quality
                + 0.10 * correlation
                + 0.10 * scale_quality,
                0.0,
                1.0,
            )
        )
        snapshot = self._observed_snapshot(
            track_id=state.track_id,
            label=state.label,
            frame=frame,
            anchor=anchor,
            depth=depth,
            point_camera=point_camera,
            point_policy=point_policy,
            velocity=velocity,
            confidence=confidence,
            missed_steps=0,
        )
        state.anchor_px = anchor.copy()
        state.support_px = supports.copy()
        state.reference_gray = frame.gray
        state.last_depth_m = float(depth)
        state.last_point_policy = point_policy.copy()
        state.last_velocity_policy = (
            None if velocity is None else velocity.copy()
        )
        state.last_seen_timestamp_s = float(frame.timestamp_s)
        state.last_seen_sequence = int(frame.sequence)
        state.missed_steps = 0
        state.confidence = confidence
        self._defer_depth_component_update(
            state, frame.depth, anchor, depth
        )
        state.snapshot = snapshot
        return state

    def _advance_depth_component_track(
        self,
        state: _TrackState,
        frame: _ValidatedFrame,
        *,
        observation_prior: dict[str, Any] | None = None,
    ) -> _TrackState | None:
        self._resolve_deferred_depth_component(state)
        if observation_prior is None:
            predicted_anchor, predicted_depth = self._project_state_point(
                state,
                frame,
            )
        else:
            predicted_anchor = np.asarray(
                observation_prior["expected_pixel"], dtype=np.float64
            )
            predicted_depth = float(observation_prior["expected_depth_m"])
        if predicted_anchor is None or predicted_depth is None:
            return None

        expected_centroid = predicted_anchor - state.component_offset_px
        components = self._depth_components_near_prediction(
            frame.depth,
            expected_centroid,
            predicted_depth,
            state.component_size_px,
        )
        if not components:
            return None

        ranked: list[tuple[float, np.ndarray, float, _DepthComponent]] = []
        search_radius = float(self.config.fallback_search_radius_px)
        depth_tolerance = self._fallback_depth_tolerance(predicted_depth)
        reference_size = np.maximum(state.component_size_px, 1.0)
        reference_area = max(1.0, float(state.component_area_px))
        for component in components:
            centroid_distance = float(
                np.linalg.norm(component.centroid_px - expected_centroid)
            )
            if centroid_distance > search_radius:
                continue
            scale_xy = np.clip(
                component.size_px / reference_size,
                self.config.min_affine_scale,
                self.config.max_affine_scale,
            )
            candidate_anchor = (
                component.centroid_px + state.component_offset_px * scale_xy
            )
            candidate_anchor = self._nearest_component_pixel(
                candidate_anchor,
                component,
            )
            depth = self._sample_depth(
                frame.depth,
                candidate_anchor,
                reference_depth=component.median_depth_m,
            )
            if depth is None:
                continue

            spatial_score = math.exp(
                -2.0 * (centroid_distance / max(search_radius, 1.0)) ** 2
            )
            depth_score = math.exp(
                -abs(component.median_depth_m - predicted_depth)
                / max(depth_tolerance, 1e-9)
            )
            area_score = math.exp(
                -abs(math.log(max(1.0, component.area_px) / reference_area))
            )
            shape_score = math.exp(
                -float(
                    np.mean(
                        np.abs(
                            np.log(
                                np.maximum(component.size_px, 1.0)
                                / reference_size
                            )
                        )
                    )
                )
            )
            appearance_score = self._translation_patch_similarity(
                state.reference_gray,
                frame.gray,
                state.anchor_px,
                candidate_anchor,
            )
            score = float(
                0.32 * spatial_score
                + 0.23 * depth_score
                + 0.16 * area_score
                + 0.14 * shape_score
                + 0.15 * appearance_score
            )
            ranked.append((score, candidate_anchor, float(depth), component))

        ranked.sort(key=lambda value: value[0], reverse=True)
        if not ranked or ranked[0][0] < self.config.fallback_min_score:
            return None
        if (
            observation_prior is not None
            and len(ranked) > 1
            and ranked[0][0] - ranked[1][0] < 0.06
        ):
            return None
        best = ranked[0]
        score, anchor, depth, component = best
        point_camera = self._unproject(anchor, depth, frame.intrinsics)
        point_policy = self._transform_point(
            frame.camera_to_policy,
            point_camera,
        )
        elapsed = float(frame.timestamp_s - state.last_seen_timestamp_s)
        velocity = None
        if elapsed > 1e-9:
            velocity = (point_policy - state.last_point_policy) / elapsed

        supports = self._select_support_points(
            frame.gray,
            frame.depth,
            anchor,
            depth,
        )
        confidence = float(np.clip(0.45 + 0.50 * score, 0.0, 0.95))
        snapshot = self._observed_snapshot(
            track_id=state.track_id,
            label=state.label,
            frame=frame,
            anchor=anchor,
            depth=depth,
            point_camera=point_camera,
            point_policy=point_policy,
            velocity=velocity,
            confidence=confidence,
            missed_steps=0,
        )
        state.anchor_px = anchor.copy()
        state.support_px = supports.copy()
        state.reference_gray = frame.gray
        state.last_depth_m = float(depth)
        state.last_point_policy = point_policy.copy()
        state.last_velocity_policy = (
            None if velocity is None else velocity.copy()
        )
        state.last_seen_timestamp_s = float(frame.timestamp_s)
        state.last_seen_sequence = int(frame.sequence)
        state.missed_steps = 0
        state.confidence = confidence
        current_component = self._depth_component_at_anchor(
            frame.depth,
            anchor,
            depth,
        )
        if current_component is None:
            current_component = component
        state.component_offset_px = anchor - current_component.centroid_px
        state.component_size_px = current_component.size_px.copy()
        state.component_area_px = int(current_component.area_px)
        state.snapshot = snapshot
        return state

    def _mark_unobserved(
        self,
        state: _TrackState,
        frame: _ValidatedFrame,
        *,
        status: str | None = None,
    ) -> _TrackState:
        state = self._clone_track_state(state)
        missed = state.missed_steps + 1
        chosen_status = status or (
            OCCLUDED if missed <= self.config.max_occluded_steps else LOST
        )
        predicted_pixel, predicted_depth = self._project_state_point(state, frame)
        predicted_uv = None
        if predicted_pixel is not None:
            predicted_uv = self._pixel_to_relative(
                predicted_pixel,
                frame.intrinsics.width,
                frame.intrinsics.height,
            )
        state.missed_steps = missed
        state.confidence = float(np.clip(state.confidence * 0.55, 0.0, 1.0))
        state.snapshot = DynamicPointSnapshot(
            track_id=state.track_id,
            label=state.label,
            camera_role=state.camera_role,
            status=chosen_status,
            uv=None,
            pixel_uv=None,
            depth_m=None,
            point_camera_m=None,
            point_robot_base_m=None,
            point_policy_local_m=None,
            velocity_policy_local_m_s=None,
            predicted_uv=predicted_uv,
            predicted_depth_m=predicted_depth,
            confidence=state.confidence,
            observation_sequence=frame.sequence,
            last_seen_sequence=state.last_seen_sequence,
            missed_steps=missed,
            episode_id=frame.episode_id,
            depth_source=None,
        )
        return state

    def _ambiguous_tracks(
        self,
        states: dict[str, _TrackState],
    ) -> set[str]:
        observed = [
            state for state in states.values() if state.snapshot.status == OBSERVED
        ]
        ambiguous: set[str] = set()
        trusted_pair = set()
        if (
            self._active_rigid_pair is not None
            and bool(self._last_rigid_pair_report.get("ok"))
        ):
            trusted_pair = set(self._active_rigid_pair["track_ids"])
        for left_index, left in enumerate(observed):
            for right in observed[left_index + 1 :]:
                if {left.track_id, right.track_id} == trusted_pair:
                    # The joint estimator already validated current RGB-D
                    # distance and identity assignment.  A rigid segment may
                    # legitimately project to nearly one UV when aligned with
                    # the camera, so 2D overlap alone is not ambiguity here.
                    continue
                anchor_distance = float(
                    np.linalg.norm(left.anchor_px - right.anchor_px)
                )
                if anchor_distance > self.config.ambiguity_anchor_distance_px:
                    continue
                if len(left.support_px) == 0 or len(right.support_px) == 0:
                    ambiguous.update((left.track_id, right.track_id))
                    continue
                distances = np.linalg.norm(
                    left.support_px[:, None, :] - right.support_px[None, :, :],
                    axis=2,
                )
                overlap = int(
                    np.count_nonzero(
                        np.min(distances, axis=1)
                        <= self.config.ambiguity_support_distance_px
                    )
                )
                denominator = max(
                    1,
                    min(len(left.support_px), len(right.support_px)),
                )
                if overlap / denominator >= self.config.ambiguity_support_fraction:
                    ambiguous.update((left.track_id, right.track_id))
        return ambiguous

    def _observed_snapshot(
        self,
        *,
        track_id: str,
        label: str,
        frame: _ValidatedFrame,
        anchor: np.ndarray,
        depth: float,
        point_camera: np.ndarray,
        point_policy: np.ndarray,
        velocity: np.ndarray | None,
        confidence: float,
        missed_steps: int,
    ) -> DynamicPointSnapshot:
        uv = self._pixel_to_relative(
            anchor,
            frame.intrinsics.width,
            frame.intrinsics.height,
        )
        point_robot_base = self._transform_point(
            frame.camera_to_robot_base,
            point_camera,
        )
        return DynamicPointSnapshot(
            track_id=track_id,
            label=label,
            camera_role=self.camera_role,
            status=OBSERVED,
            uv=uv,
            pixel_uv=(float(anchor[0]), float(anchor[1])),
            depth_m=float(depth),
            point_camera_m=tuple(float(value) for value in point_camera),
            point_robot_base_m=tuple(
                float(value) for value in point_robot_base
            ),
            point_policy_local_m=tuple(float(value) for value in point_policy),
            velocity_policy_local_m_s=(
                None
                if velocity is None
                else tuple(float(value) for value in velocity)
            ),
            predicted_uv=None,
            predicted_depth_m=None,
            confidence=float(confidence),
            observation_sequence=int(frame.sequence),
            last_seen_sequence=int(frame.sequence),
            missed_steps=int(missed_steps),
            episode_id=frame.episode_id,
        )

    def _validate_frame(self, frame: TrackerFrame) -> _ValidatedFrame:
        if not isinstance(frame, TrackerFrame):
            raise TypeError("frame must be a TrackerFrame")
        frame.intrinsics.validate()
        role = str(frame.camera_role or "").strip()
        if role != self.camera_role:
            raise ValueError(
                f"tracker for {self.camera_role!r} cannot ingest {role!r} frames"
            )
        episode_id = str(frame.episode_id or "").strip()
        if not episode_id:
            raise ValueError("episode_id is required")
        sequence = int(frame.sequence)
        if sequence < 0 or sequence != frame.sequence:
            raise ValueError("sequence must be a non-negative integer")
        timestamp_s = float(frame.timestamp_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")

        depth = np.asarray(frame.depth_linear, dtype=np.float32).squeeze()
        if depth.ndim != 2:
            raise ValueError(f"depth_linear must be HxW, got {depth.shape}")
        if depth.shape != (
            frame.intrinsics.height,
            frame.intrinsics.width,
        ):
            raise ValueError(
                "depth_linear shape does not match camera intrinsics"
            )
        gray = self._to_gray(frame.rgb, frame.color_order)
        if gray.shape != depth.shape:
            raise ValueError("RGB and depth_linear resolutions do not match")
        camera_to_robot_base = self._validate_transform(
            frame.camera_to_robot_base,
            name="camera_to_robot_base",
        )
        camera_to_policy = self._validate_transform(
            frame.camera_to_policy,
            name="camera_to_policy",
        )
        return _ValidatedFrame(
            # _to_gray returns an owned array, including for an HxW source.
            gray=gray,
            depth=depth.copy(),
            intrinsics=frame.intrinsics,
            camera_to_robot_base=camera_to_robot_base,
            camera_to_policy=camera_to_policy,
            sequence=sequence,
            timestamp_s=timestamp_s,
            episode_id=episode_id,
            camera_role=role,
        )

    @staticmethod
    def _to_gray(rgb, color_order: str) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim == 2:
            gray = DynamicPointTracker._as_uint8_image(image)
            # A grayscale evaluator input can alias caller-owned observation
            # memory. The validated frame must remain immutable internally.
            return (
                gray.copy()
                if np.shares_memory(gray, image)
                else np.ascontiguousarray(gray)
            )
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            order = str(color_order or "").strip().lower()
            if order not in ("rgb", "bgr"):
                raise ValueError("color_order must be rgb or bgr")
            if image.shape[2] == 4:
                conversion = (
                    cv2.COLOR_RGBA2GRAY if order == "rgb" else cv2.COLOR_BGRA2GRAY
                )
            else:
                conversion = (
                    cv2.COLOR_RGB2GRAY if order == "rgb" else cv2.COLOR_BGR2GRAY
                )
            if image.dtype != np.uint8:
                image = DynamicPointTracker._as_uint8_image(image)
            # cvtColor allocates the tracker-owned grayscale destination.
            return cv2.cvtColor(image, conversion)
        else:
            raise ValueError(f"RGB must be HxW, HxWx3, or HxWx4, got {image.shape}")

    @staticmethod
    def _as_uint8_image(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.dtype == np.uint8:
            return arr
        numeric = np.asarray(arr, dtype=np.float32)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("RGB contains non-finite values")
        if numeric.size and float(np.max(numeric)) <= 1.0:
            numeric = numeric * 255.0
        return np.clip(np.rint(numeric), 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _validate_transform(value, *, name: str) -> np.ndarray:
        transform = np.asarray(value, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"{name} must be a 4x4 transform")
        if not np.all(np.isfinite(transform)):
            raise ValueError(f"{name} must be finite")
        if not np.allclose(
            transform[3],
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError(f"{name} has an invalid homogeneous row")
        rotation = transform[:3, :3]
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-5,
            rtol=0.0,
        ) or not math.isclose(
            float(np.linalg.det(rotation)),
            1.0,
            abs_tol=1e-5,
        ):
            raise ValueError(f"{name} rotation is not rigid")
        return transform.copy()

    def _select_support_points(
        self,
        gray: np.ndarray,
        depth: np.ndarray,
        anchor: np.ndarray,
        anchor_depth: float,
    ) -> np.ndarray:
        height, width = gray.shape
        center = (int(round(float(anchor[0]))), int(round(float(anchor[1]))))
        radius = int(self.config.patch_radius_px)
        # Four pixels cover the 3x3 gradient, 5x5 response block, and 3x3
        # non-maximum neighborhood used by goodFeaturesToTrack. Valid masked
        # pixels therefore see the same source neighborhood as a full-frame
        # call, including at the actual image boundary.
        feature_halo = 4
        left = max(0, center[0] - radius - feature_halo)
        right = min(width, center[0] + radius + feature_halo + 1)
        top = max(0, center[1] - radius - feature_halo)
        bottom = min(height, center[1] + radius + feature_halo + 1)
        gray_region = gray[top:bottom, left:right]
        region = np.asarray(depth[top:bottom, left:right], dtype=np.float64)
        mask = np.zeros(gray_region.shape, dtype=np.uint8)
        cv2.circle(
            mask,
            (center[0] - left, center[1] - top),
            radius,
            255,
            -1,
        )
        valid_depth = np.isfinite(region) & (region > 0.0)
        valid_depth &= (
            np.abs(region - float(anchor_depth))
            <= self._depth_tolerance(anchor_depth)
        )
        mask[~valid_depth] = 0
        corners = cv2.goodFeaturesToTrack(
            gray_region,
            maxCorners=self.config.max_support_points,
            qualityLevel=self.config.feature_quality,
            minDistance=self.config.feature_min_distance_px,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        if corners is None:
            return np.empty((0, 2), dtype=np.float64)
        points = corners.reshape(-1, 2).astype(np.float64)
        points += [left, top]
        order = np.argsort(np.linalg.norm(points - anchor.reshape(1, 2), axis=1))
        return points[order]

    def _registration_confidence(self, support_count: int) -> float:
        ratio = min(
            1.0,
            max(0.0, float(support_count))
            / float(self.config.min_support_points),
        )
        return float(0.60 + 0.40 * ratio)

    def _fallback_depth_tolerance(self, depth_m: float) -> float:
        return max(
            self.config.fallback_depth_change_abs_m,
            abs(float(depth_m))
            * self.config.fallback_depth_change_relative,
        )

    def _depth_component_at_anchor(
        self,
        depth: np.ndarray,
        anchor: np.ndarray,
        anchor_depth: float,
    ) -> _DepthComponent | None:
        return self._flood_depth_component(
            depth,
            anchor,
            anchor_depth,
            radius_px=self.config.fallback_component_radius_px,
        )

    @staticmethod
    def _clear_deferred_depth_component(state: _TrackState) -> None:
        state.pending_component_depth_roi = None
        state.pending_component_origin_px = None
        state.pending_component_anchor_px = None
        state.pending_component_depth_m = None

    def _defer_depth_component_update(
        self,
        state: _TrackState,
        depth: np.ndarray,
        anchor: np.ndarray,
        anchor_depth: float,
    ) -> None:
        """Save the exact small ROI needed only by a future depth fallback."""

        height, width = depth.shape
        seed_x = int(round(float(anchor[0])))
        seed_y = int(round(float(anchor[1])))
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            self._clear_deferred_depth_component(state)
            return
        radius = max(1, int(self.config.fallback_component_radius_px))
        left = max(0, seed_x - radius)
        right = min(width, seed_x + radius + 1)
        top = max(0, seed_y - radius)
        bottom = min(height, seed_y + radius + 1)
        state.pending_component_depth_roi = np.asarray(
            depth[top:bottom, left:right]
        ).copy()
        state.pending_component_origin_px = (left, top)
        state.pending_component_anchor_px = np.asarray(
            anchor, dtype=np.float64
        ).copy()
        state.pending_component_depth_m = float(anchor_depth)

    def _resolve_deferred_depth_component(self, state: _TrackState) -> None:
        roi = getattr(state, "pending_component_depth_roi", None)
        origin = getattr(state, "pending_component_origin_px", None)
        anchor = getattr(state, "pending_component_anchor_px", None)
        anchor_depth = getattr(state, "pending_component_depth_m", None)
        self._clear_deferred_depth_component(state)
        if roi is None or origin is None or anchor is None or anchor_depth is None:
            return
        origin_array = np.asarray(origin, dtype=np.float64)
        component = self._flood_depth_component(
            roi,
            np.asarray(anchor, dtype=np.float64) - origin_array,
            float(anchor_depth),
            radius_px=self.config.fallback_component_radius_px,
        )
        if component is None:
            return
        state.component_offset_px = np.asarray(
            anchor, dtype=np.float64
        ) - (component.centroid_px + origin_array)
        state.component_size_px = component.size_px.copy()
        state.component_area_px = int(component.area_px)

    def _depth_components_near_prediction(
        self,
        depth: np.ndarray,
        predicted_centroid: np.ndarray,
        predicted_depth_m: float,
        reference_size_px: np.ndarray,
    ) -> list[_DepthComponent]:
        height, width = depth.shape
        radius = self.config.fallback_search_radius_px
        center_x = int(round(float(predicted_centroid[0])))
        center_y = int(round(float(predicted_centroid[1])))
        left = max(0, center_x - radius)
        right = min(width, center_x + radius + 1)
        top = max(0, center_y - radius)
        bottom = min(height, center_y + radius + 1)
        if left >= right or top >= bottom:
            return []

        region = np.asarray(depth[top:bottom, left:right], dtype=np.float64)
        tolerance = self._fallback_depth_tolerance(predicted_depth_m)
        valid = (
            np.isfinite(region)
            & (region > 0.0)
            & (np.abs(region - float(predicted_depth_m)) <= tolerance)
        )
        if not np.any(valid):
            return []

        # Quantizing depth before connected components prevents a foreground
        # surface from joining the background merely through adjacent pixels.
        bin_width = max(0.025, self._depth_tolerance(predicted_depth_m) * 0.5)
        bins = np.zeros(region.shape, dtype=np.int64)
        bins[valid] = np.rint(region[valid] / bin_width).astype(np.int64)
        components: list[_DepthComponent] = []
        minimum_reference_area = max(
            1,
            min(9, int(round(float(np.prod(reference_size_px)) * 0.02))),
        )
        for depth_bin in np.unique(bins[valid]):
            layer = (valid & (bins == depth_bin)).astype(np.uint8)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                layer,
                connectivity=8,
            )
            for label_index in range(1, count):
                area = int(stats[label_index, cv2.CC_STAT_AREA])
                if area < minimum_reference_area:
                    continue
                ys, xs = np.nonzero(labels == label_index)
                pixels = np.column_stack(
                    (xs.astype(np.float64) + left, ys.astype(np.float64) + top)
                )
                values = region[ys, xs]
                component = _DepthComponent(
                    centroid_px=np.asarray(centroids[label_index], dtype=np.float64)
                    + [left, top],
                    size_px=np.array(
                        [
                            stats[label_index, cv2.CC_STAT_WIDTH],
                            stats[label_index, cv2.CC_STAT_HEIGHT],
                        ],
                        dtype=np.float64,
                    ),
                    area_px=area,
                    median_depth_m=float(np.median(values)),
                    pixels_px=pixels,
                )
                components.append(component)
        return components

    def _flood_depth_component(
        self,
        depth: np.ndarray,
        seed_px: np.ndarray,
        seed_depth_m: float,
        *,
        radius_px: int,
    ) -> _DepthComponent | None:
        height, width = depth.shape
        seed_x = int(round(float(seed_px[0])))
        seed_y = int(round(float(seed_px[1])))
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            return None
        radius = max(1, int(radius_px))
        left = max(0, seed_x - radius)
        right = min(width, seed_x + radius + 1)
        top = max(0, seed_y - radius)
        bottom = min(height, seed_y + radius + 1)
        region = np.asarray(depth[top:bottom, left:right], dtype=np.float64)
        tolerance = self._depth_tolerance(seed_depth_m)
        valid = (
            np.isfinite(region)
            & (region > 0.0)
            & (np.abs(region - float(seed_depth_m)) <= tolerance)
        ).astype(np.uint8)
        local_x = seed_x - left
        local_y = seed_y - top
        if valid[local_y, local_x] == 0:
            valid_depth = np.argwhere(valid > 0)
            if len(valid_depth) == 0:
                return None
            nearest = valid_depth[
                int(
                    np.argmin(
                        np.sum(
                            (valid_depth - [local_y, local_x]) ** 2,
                            axis=1,
                        )
                    )
                )
            ]
            local_y, local_x = (int(nearest[0]), int(nearest[1]))

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            valid,
            connectivity=8,
        )
        label_index = int(labels[local_y, local_x])
        if count <= 1 or label_index <= 0:
            return None
        ys, xs = np.nonzero(labels == label_index)
        pixels = np.column_stack(
            (xs.astype(np.float64) + left, ys.astype(np.float64) + top)
        )
        values = region[ys, xs]
        return _DepthComponent(
            centroid_px=np.asarray(centroids[label_index], dtype=np.float64)
            + [left, top],
            size_px=np.array(
                [
                    stats[label_index, cv2.CC_STAT_WIDTH],
                    stats[label_index, cv2.CC_STAT_HEIGHT],
                ],
                dtype=np.float64,
            ),
            area_px=int(stats[label_index, cv2.CC_STAT_AREA]),
            median_depth_m=float(np.median(values)),
            pixels_px=pixels,
        )

    @staticmethod
    def _nearest_component_pixel(
        candidate_px: np.ndarray,
        component: _DepthComponent,
    ) -> np.ndarray:
        distances = np.sum(
            (component.pixels_px - candidate_px.reshape(1, 2)) ** 2,
            axis=1,
        )
        return component.pixels_px[int(np.argmin(distances))].copy()

    def _translation_patch_similarity(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_anchor: np.ndarray,
        current_anchor: np.ndarray,
    ) -> float:
        radius = self.config.fallback_template_radius_px
        patch_size = (2 * radius + 1, 2 * radius + 1)
        previous_patch = cv2.getRectSubPix(
            previous_gray,
            patch_size,
            (float(previous_anchor[0]), float(previous_anchor[1])),
        ).astype(np.float64)
        current_patch = cv2.getRectSubPix(
            current_gray,
            patch_size,
            (float(current_anchor[0]), float(current_anchor[1])),
        ).astype(np.float64)
        previous_std = float(np.std(previous_patch))
        current_std = float(np.std(current_patch))
        if previous_std >= 2.0 and current_std >= 2.0:
            left = previous_patch.reshape(-1) - float(np.mean(previous_patch))
            right = current_patch.reshape(-1) - float(np.mean(current_patch))
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator > 1e-9:
                correlation = float(np.dot(left, right) / denominator)
                return float(np.clip(0.5 + 0.5 * correlation, 0.0, 1.0))
        mean_error = float(np.mean(np.abs(previous_patch - current_patch)))
        return float(math.exp(-mean_error / 24.0))

    def _sample_depth(
        self,
        depth: np.ndarray,
        pixel: np.ndarray,
        *,
        reference_depth: float | None = None,
    ) -> float | None:
        height, width = depth.shape
        x = float(pixel[0])
        y = float(pixel[1])
        if not self._pixel_in_bounds(pixel, width, height):
            return None
        center_x = int(round(x))
        center_y = int(round(y))
        radius = self.config.depth_search_radius_px
        candidates: list[tuple[float, float]] = []
        y_range = range(
            max(0, center_y - radius),
            min(height, center_y + radius + 1),
        )
        x_range = range(
            max(0, center_x - radius),
            min(width, center_x + radius + 1),
        )
        for py in y_range:
            for px in x_range:
                value = float(depth[py, px])
                if not math.isfinite(value) or value <= 0.0:
                    continue
                distance = math.hypot(float(px) - x, float(py) - y)
                if reference_depth is not None:
                    if abs(value - reference_depth) > self._depth_tolerance(
                        reference_depth
                    ):
                        continue
                    score = abs(value - reference_depth) + distance * 1e-6
                else:
                    score = distance
                candidates.append((score, value))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return float(candidates[0][1])

    def _depth_tolerance(self, depth_m: float) -> float:
        return max(
            self.config.depth_layer_abs_tolerance_m,
            abs(float(depth_m)) * self.config.depth_layer_relative_tolerance,
        )

    def _merge_support_points(
        self,
        retained: np.ndarray,
        fresh: np.ndarray,
    ) -> np.ndarray:
        merged = [np.asarray(point, dtype=np.float64) for point in retained]
        minimum = self.config.feature_min_distance_px * 0.75
        for point in fresh:
            if len(merged) >= self.config.max_support_points:
                break
            if not merged or min(
                float(np.linalg.norm(point - existing)) for existing in merged
            ) >= minimum:
                merged.append(np.asarray(point, dtype=np.float64))
        if not merged:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(merged, dtype=np.float64).reshape(-1, 2)

    def _warped_patch_correlation(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_anchor: np.ndarray,
        matrix: np.ndarray,
    ) -> float:
        height, width = previous_gray.shape
        aligned = cv2.warpAffine(
            current_gray,
            np.asarray(matrix, dtype=np.float64),
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        radius = max(4, int(round(self.config.patch_radius_px * 0.70)))
        patch_size = (2 * radius + 1, 2 * radius + 1)
        center = (float(previous_anchor[0]), float(previous_anchor[1]))
        previous_patch = cv2.getRectSubPix(previous_gray, patch_size, center)
        current_patch = cv2.getRectSubPix(aligned, patch_size, center)
        left = previous_patch.astype(np.float64).reshape(-1)
        right = current_patch.astype(np.float64).reshape(-1)
        left -= float(np.mean(left))
        right -= float(np.mean(right))
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1e-9:
            return 0.0
        return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))

    def _affine_is_plausible(
        self,
        matrix: np.ndarray,
    ) -> tuple[bool, float, float]:
        linear = np.asarray(matrix, dtype=np.float64)[:, :2]
        scale = math.sqrt(max(0.0, float(np.linalg.det(linear))))
        rotation_deg = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
        ok = bool(
            math.isfinite(scale)
            and math.isfinite(rotation_deg)
            and self.config.min_affine_scale <= scale <= self.config.max_affine_scale
            and abs(rotation_deg) <= self.config.max_affine_rotation_deg
        )
        return ok, float(scale), float(rotation_deg)

    @staticmethod
    def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        homogeneous = np.column_stack(
            (values, np.ones(len(values), dtype=np.float64))
        )
        return homogeneous @ np.asarray(matrix, dtype=np.float64).T

    @staticmethod
    def _relative_to_pixel(
        u: float,
        v: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        u_value = float(u)
        v_value = float(v)
        if not math.isfinite(u_value) or not math.isfinite(v_value):
            raise ValueError("u and v must be finite")
        if not 0.0 <= u_value <= 1000.0 or not 0.0 <= v_value <= 1000.0:
            raise ValueError("u and v must use the relative 0..1000 frame")
        return np.array(
            [
                u_value / 1000.0 * float(width - 1),
                v_value / 1000.0 * float(height - 1),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _pixel_to_relative(
        pixel: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        return (
            float(pixel[0]) / float(width - 1) * 1000.0,
            float(pixel[1]) / float(height - 1) * 1000.0,
        )

    @staticmethod
    def _pixel_in_bounds(pixel: np.ndarray, width: int, height: int) -> bool:
        x, y = np.asarray(pixel, dtype=np.float64).reshape(2)
        return bool(0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0)

    @staticmethod
    def _unproject(
        pixel: np.ndarray,
        depth_m: float,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        x, y = np.asarray(pixel, dtype=np.float64).reshape(2)
        depth = float(depth_m)
        return np.array(
            [
                (x - intrinsics.cx) / intrinsics.fx * depth,
                -(y - intrinsics.cy) / intrinsics.fy * depth,
                -depth,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
        homogeneous = np.append(np.asarray(point, dtype=np.float64).reshape(3), 1.0)
        return (np.asarray(transform, dtype=np.float64) @ homogeneous)[:3]

    def _project_policy_point(
        self,
        point_policy: np.ndarray,
        frame: _ValidatedFrame,
    ) -> tuple[np.ndarray | None, float | None]:
        camera_from_policy = np.linalg.inv(frame.camera_to_policy)
        point_camera = self._transform_point(camera_from_policy, point_policy)
        depth = -float(point_camera[2])
        if not math.isfinite(depth) or depth <= 1e-6:
            return None, None
        pixel = np.array(
            [
                frame.intrinsics.cx
                + frame.intrinsics.fx * float(point_camera[0]) / depth,
                frame.intrinsics.cy
                - frame.intrinsics.fy * float(point_camera[1]) / depth,
            ],
            dtype=np.float64,
        )
        if not self._pixel_in_bounds(
            pixel,
            frame.intrinsics.width,
            frame.intrinsics.height,
        ):
            return None, depth
        return pixel, depth

    def _project_robot_base_point(
        self,
        point_robot_base: np.ndarray,
        frame: _ValidatedFrame,
    ) -> tuple[np.ndarray | None, float | None]:
        point = np.asarray(point_robot_base, dtype=np.float64).reshape(3)
        camera_from_robot_base = np.linalg.inv(frame.camera_to_robot_base)
        point_camera = self._transform_point(camera_from_robot_base, point)
        depth = -float(point_camera[2])
        if not math.isfinite(depth) or depth <= 1.0e-6:
            return None, None
        pixel = np.array(
            [
                frame.intrinsics.cx
                + frame.intrinsics.fx * float(point_camera[0]) / depth,
                frame.intrinsics.cy
                - frame.intrinsics.fy * float(point_camera[1]) / depth,
            ],
            dtype=np.float64,
        )
        if not self._pixel_in_bounds(
            pixel,
            frame.intrinsics.width,
            frame.intrinsics.height,
        ):
            return None, depth
        return pixel, depth

    def _project_state_point(
        self,
        state: _TrackState,
        frame: _ValidatedFrame,
    ) -> tuple[np.ndarray | None, float | None]:
        point_policy = state.last_point_policy.copy()
        velocity = state.last_velocity_policy
        elapsed = max(0.0, float(frame.timestamp_s - state.last_seen_timestamp_s))
        if velocity is not None and elapsed > 0.0:
            speed = float(np.linalg.norm(velocity))
            if math.isfinite(speed) and speed <= self.config.fallback_max_velocity_m_s:
                horizon = min(
                    elapsed,
                    self.config.fallback_max_velocity_prediction_s,
                )
                point_policy = point_policy + velocity * horizon
        return self._project_policy_point(point_policy, frame)


__all__ = [
    "AMBIGUOUS",
    "LOST",
    "OBSERVED",
    "OCCLUDED",
    "CameraIntrinsics",
    "DynamicPointSnapshot",
    "DynamicPointTracker",
    "DynamicPointTrackerConfig",
    "TrackerFrame",
    "camera_to_policy_from_odometry",
    "transform_from_pose",
]
