"""BEHAVIOR-compatible persistent RGB-D tracking over RoboDojo observations."""

from __future__ import annotations

from copy import deepcopy
import math
import threading
import time
from typing import Any, Mapping

import numpy as np

from tools.util.dynamic_point_tracker import (
    CameraIntrinsics,
    DynamicPointTracker,
    TrackerFrame,
)
from tools.util.tracked_object_distance import (
    TRACKED_OBJECT_DISTANCES_MEMORY_KEY,
    TRACKING_POLICY_MEMORY_KEY,
    TrackedObjectDistanceMemory,
    tracker_frame_content_digest,
)
from tools.util.geometry import (
    OPTICAL_TO_USD_AXES,
    finite_vector,
    robot_base_to_robodojo_env,
)
from tools.util.observation import ObservationAdapter
from tools.util.storage import Snapshot
from tools.util.tracking_legacy import PointTracker as _LegacyPointTracker
from tools.util.tabletop_tracking import TabletopDynamicPointTracker
from tools.util.tabletop_surface import TabletopSurface
from tools.util.x5_kinematics import matrix_to_quaternion, quaternion_to_matrix

OBSERVED = "observed"
TEMPORARILY_UNOBSERVED = "temporarily_unobserved"
LOST = "lost"

TRACKING_CAMERA_ROLES = ("head", "left_wrist", "right_wrist")
TRACKING_SLOTS = ("head", "wrist")
TRACKING_COLLINEAR_TOLERANCE_M = 0.001
TRACKING_IMAGE_COLLINEAR_TOLERANCE_PX = 2.0
# A tracked pixel is allowed a few pixels of optical-flow/depth quantization
# error, but a camera/base convention error is normally tens or hundreds of
# pixels.  Keep this separate from the line-fit tolerance so the diagnostics
# expose the actual reprojection problem.
TRACKING_SOURCE_REPROJECTION_TOLERANCE_PX = 3.0
TRACKING_REPROJECTION_DEPTH_TOLERANCE_M = 0.025
TRACKING_SAME_FRAME_SEQUENCE_SKEW = 0

# Fixed transform from the X5A ``link6`` EEF frame to the USD camera Xform.
# It is the exact composition of the public RoboDojo X5A.urdf camera joints
# and robot_config.yml camera-local pose.  Runtime wrist poses are deliberately
# reconstructed from evaluator EEF proprioception instead of trusting a second,
# independently sampled camera pose.
X5_EEF_TO_WRIST_CAMERA = np.array(
    [
        [-7.34641021e-06, 5.000295741e-01, -8.660083285e-01, 8.48420522e-02],
        [-1.000000000e00, -3.67336028e-06, 6.36213551e-06, -2.04522898e-07],
        [0.000000000e00, 8.660083285e-01, 5.000295741e-01, 5.09396809e-02],
        [0.000000000e00, 0.000000000e00, 0.000000000e00, 1.000000000e00],
    ],
    dtype=np.float64,
)


def x5_wrist_camera_to_env(eef_pose: Any) -> np.ndarray:
    """Compose one evaluator X5 EEF pose with the official USD hand-eye model."""

    if isinstance(eef_pose, Mapping):
        position = finite_vector(eef_pose.get("pos"), 3, "X5 EEF position")
        quaternion = finite_vector(eef_pose.get("quat"), 4, "X5 EEF quaternion")
    else:
        packed = finite_vector(eef_pose, 7, "X5 EEF pose")
        position, quaternion = packed[:3], packed[3:]
    env_from_eef = np.eye(4, dtype=np.float64)
    env_from_eef[:3, :3] = quaternion_to_matrix(quaternion)
    env_from_eef[:3, 3] = position
    return env_from_eef @ X5_EEF_TO_WRIST_CAMERA


class PointTracker:
    """Expose the old RoboDojo API over the latest official tracking memory.

    The public tools and embodied agent keep using ``register``, ``update``,
    ``result`` and ``observed_active_points_snapshot``.  Internally, exact
    capture binding, gap-free replay and live RGB-D tracking are owned by the
    same ``TrackedObjectDistanceMemory`` implementation as official_v2.
    """

    def __init__(
        self,
        *,
        base_rotation: Any | None = None,
        max_unobserved_steps: int = 8,
    ) -> None:
        self._lock = threading.RLock()
        self._base_rotation = (
            np.eye(3, dtype=np.float64)
            if base_rotation is None
            else np.asarray(base_rotation, dtype=np.float64).reshape(3, 3).copy()
        )
        self._managers: dict[str, TrackedObjectDistanceMemory] = {
            "head": self._new_manager("head")
        }
        # Keep the historical attribute for hot-reload/local inspection code.
        self._manager = self._managers["head"]
        self._legacy = _LegacyPointTracker(
            base_rotation=self._base_rotation,
            max_unobserved_steps=max_unobserved_steps,
        )
        # Existing local tests inject one optical-flow result through this
        # attribute.  The production path still comes from the dynamic tracker.
        self._flow = _LegacyPointTracker._flow
        self._capture_bindings: dict[tuple[str, str], dict[str, Any]] = {}
        self._active_result: dict[str, Any] | None = None
        self._active_results_by_slot: dict[str, dict[str, Any]] = {}
        self._slot_roles: dict[str, str | None] = {
            "head": None,
            "wrist": None,
        }
        self._active_camera_role: str | None = None
        self._last_capture_camera_role: str = "head"
        self._last_error: str | None = None
        self._camera_errors: dict[str, str] = {}
        self._hand_eye_diagnostics: dict[str, dict[str, Any]] = {}
        self._motion_retention_leases: dict[str, dict[str, Any]] = {}
        self._motion_retention_counter = 0
        self._eef_prior_leases: dict[str, dict[str, Any]] = {}
        self._eef_prior_counter = 0

    @staticmethod
    def _validate_camera_role(camera_role: str) -> str:
        role = str(camera_role or "").strip()
        if role not in TRACKING_CAMERA_ROLES:
            raise ValueError(
                "tracking camera role must be head, left_wrist, or right_wrist"
            )
        return role

    @staticmethod
    def _tracking_slot(camera_role: str) -> str:
        role = PointTracker._validate_camera_role(camera_role)
        return "head" if role == "head" else "wrist"

    @staticmethod
    def _slot_camera_roles(slot: str) -> tuple[str, ...]:
        normalized = str(slot or "").strip()
        if normalized == "head":
            return ("head",)
        if normalized == "wrist":
            return ("left_wrist", "right_wrist")
        raise ValueError("tracking slot must be head or wrist")

    def _active_source_items_locked(
        self,
    ) -> list[tuple[str, str, TrackedObjectDistanceMemory]]:
        items: list[tuple[str, str, TrackedObjectDistanceMemory]] = []
        for slot in TRACKING_SLOTS:
            role = self._slot_roles.get(slot)
            manager = None if role is None else self._managers.get(role)
            if role is not None and manager is not None:
                items.append((slot, role, manager))
        return items

    def _registered_source_groups_locked(
        self,
        names: list[str] | tuple[str, ...],
    ) -> tuple[
        list[str],
        list[tuple[str, str, TrackedObjectDistanceMemory, list[str]]],
        list[str],
    ]:
        requested = [str(name or "").strip() for name in names]
        if not requested or any(not name for name in requested):
            raise ValueError("names must contain non-empty point names")
        if len(set(requested)) != len(requested):
            raise ValueError("names must be unique")

        groups: list[tuple[str, str, TrackedObjectDistanceMemory, list[str]]] = []
        found: set[str] = set()
        for slot, role, manager in self._active_source_items_locked():
            registered = {
                str(name) for name in manager.status().get("registered_names") or []
            }
            selected = [name for name in requested if name in registered]
            if selected:
                groups.append((slot, role, manager, selected))
                found.update(selected)
        return requested, groups, [name for name in requested if name not in found]

    def _reselect_fallback_locked(self) -> None:
        if self._active_camera_role in {
            role for role in self._slot_roles.values() if role is not None
        }:
            return
        for slot in reversed(TRACKING_SLOTS):
            role = self._slot_roles.get(slot)
            result = self._active_results_by_slot.get(slot)
            if role is not None and result is not None:
                self._active_camera_role = role
                self._active_result = deepcopy(result)
                return
        self._active_camera_role = None
        self._active_result = None

    def _retire_composite_leases_for_roles_locked(
        self,
        camera_roles: tuple[str, ...],
    ) -> None:
        affected = set(camera_roles)
        for token, composite in list(self._motion_retention_leases.items()):
            if not affected.intersection(composite.get("leases", {})):
                continue
            self._motion_retention_leases.pop(token, None)
            for lease in composite.get("leases", {}).values():
                lease["manager"].end_motion_retention(lease["lease_id"])
        for token, composite in list(self._eef_prior_leases.items()):
            if not affected.intersection(composite.get("leases", {})):
                continue
            self._eef_prior_leases.pop(token, None)
            for lease in composite.get("leases", {}).values():
                lease["manager"].deactivate_on_hand_eef_observation_prior(
                    lease["lease_id"]
                )

    def _new_manager(self, camera_role: str) -> TrackedObjectDistanceMemory:
        role = PointTracker._validate_camera_role(camera_role)
        return TrackedObjectDistanceMemory(
            tracker_factory=lambda role=role: TabletopDynamicPointTracker(
                camera_role=role, table_plane_provider=self._head_table_plane
            )
        )

    def _head_table_plane(self, frame) -> dict[str, Any]:
        # Head has a broad view of the support; a close wrist view of a laptop
        # or flat key must never declare that object's surface to be the table.
        manager = self._managers.get("head")
        head = None if manager is None else getattr(manager, "_latest_frame", None)
        if head is None or head.episode_id != frame.episode_id:
            return {"confident": False}
        return TabletopSurface.from_frame(head).report

    def _manager_for_role(
        self,
        camera_role: str,
        *,
        create: bool = True,
    ) -> TrackedObjectDistanceMemory | None:
        role = self._validate_camera_role(camera_role)
        with self._lock:
            manager = self._managers.get(role)
            if manager is None and create:
                manager = self._new_manager(role)
                self._managers[role] = manager
            return manager

    def _selected_role(self) -> str:
        with self._lock:
            return self._active_camera_role or self._last_capture_camera_role or "head"

    def _selected_manager(self) -> TrackedObjectDistanceMemory:
        manager = self._manager_for_role(self._selected_role())
        assert manager is not None
        return manager

    @property
    def manager(self) -> TrackedObjectDistanceMemory:
        return self._selected_manager()

    def reset(self) -> None:
        with self._lock:
            for manager in self._managers.values():
                manager.reset()
            self._managers = {"head": self._new_manager("head")}
            self._manager = self._managers["head"]
            self._legacy.reset()
            self._capture_bindings.clear()
            self._active_result = None
            self._active_results_by_slot.clear()
            self._slot_roles = {"head": None, "wrist": None}
            self._active_camera_role = None
            self._last_capture_camera_role = "head"
            self._last_error = None
            self._camera_errors.clear()
            self._hand_eye_diagnostics.clear()
            self._motion_retention_leases.clear()
            self._motion_retention_counter = 0
            self._eef_prior_leases.clear()
            self._eef_prior_counter = 0

    def _camera_to_base(self, camera_to_env: Any) -> np.ndarray:
        camera = np.asarray(camera_to_env, dtype=np.float64).reshape(4, 4)
        base_from_env = np.eye(4, dtype=np.float64)
        base_from_env[:3, :3] = self._base_rotation.T
        return base_from_env @ camera

    def _camera_to_env_from_eef(
        self,
        camera_role: str,
        eef_pose: Any,
        evaluator_camera_to_env: Any | None = None,
    ) -> np.ndarray:
        role = self._validate_camera_role(camera_role)
        if role == "head":
            if evaluator_camera_to_env is None:
                raise ValueError("head camera requires evaluator camera geometry")
            return np.asarray(evaluator_camera_to_env, dtype=np.float64).reshape(4, 4)
        derived = x5_wrist_camera_to_env(eef_pose)
        if evaluator_camera_to_env is not None:
            observed = np.asarray(evaluator_camera_to_env, dtype=np.float64).reshape(
                4, 4
            )
            delta = np.linalg.inv(derived) @ observed
            cosine = float(np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
            with self._lock:
                self._hand_eye_diagnostics[role] = {
                    "source": "eef_pose_times_official_x5_hand_eye",
                    "translation_error_m": float(np.linalg.norm(delta[:3, 3])),
                    "rotation_error_deg": math.degrees(math.acos(cosine)),
                    "derived_pose_used": True,
                }
        return derived

    def _eef_pose_base_wxyz_value(self, pose: Any, arm: str) -> np.ndarray:
        if isinstance(pose, Mapping):
            pose_env = np.concatenate(
                [
                    finite_vector(pose.get("pos"), 3, f"{arm} EEF position"),
                    finite_vector(pose.get("quat"), 4, f"{arm} EEF quaternion"),
                ]
            )
        else:
            pose_env = finite_vector(pose, 7, f"{arm} EEF pose")
        position = self._base_rotation.T @ pose_env[:3]
        rotation = self._base_rotation.T @ quaternion_to_matrix(pose_env[3:])
        return np.concatenate([position, matrix_to_quaternion(rotation)])

    def _eef_pose_base_wxyz(
        self,
        adapter: ObservationAdapter,
        arm: str,
    ) -> np.ndarray:
        return self._eef_pose_base_wxyz_value(adapter.state(f"{arm}_ee_pose"), arm)

    def _proprio(self, adapter: ObservationAdapter) -> np.ndarray | None:
        try:
            values: list[np.ndarray] = []
            for arm in ("left", "right"):
                q = finite_vector(
                    adapter.state(f"{arm}_arm_joint_state"),
                    6,
                    f"{arm} arm state",
                )
                eef = self._eef_pose_base_wxyz(adapter, arm)
                gripper_raw = adapter.state(
                    f"{arm}_ee_joint_state",
                    required=False,
                )
                if gripper_raw is None or gripper_raw.size < 1:
                    gripper = np.zeros(1, dtype=np.float64)
                else:
                    gripper = np.asarray(
                        [float(np.clip(gripper_raw[0], 0.0, 1.0))],
                        dtype=np.float64,
                    )
                values.extend([q, eef[:3], eef[3:], gripper])
            packed = np.concatenate(values)
            if packed.shape != (28,) or not np.all(np.isfinite(packed)):
                raise ValueError("RoboDojo proprio packing is invalid")
            return packed
        except (KeyError, RuntimeError, TypeError, ValueError):
            return None

    def _snapshot_proprio(self, snapshot: Snapshot) -> np.ndarray | None:
        try:
            values: list[np.ndarray] = []
            for arm in ("left", "right"):
                q = finite_vector(
                    snapshot.robot[f"arm_{arm}_qpos"],
                    6,
                    f"{arm} arm state",
                )
                eef = self._eef_pose_base_wxyz_value(snapshot.robot[f"eef_{arm}"], arm)
                gripper_raw = finite_vector(
                    snapshot.robot[f"gripper_{arm}_qpos"],
                    1,
                    f"{arm} gripper state",
                )
                gripper = np.asarray(
                    [float(np.clip(gripper_raw[0], 0.0, 1.0))],
                    dtype=np.float64,
                )
                values.extend([q, eef[:3], eef[3:], gripper])
            packed = np.concatenate(values)
            if packed.shape != (28,) or not np.all(np.isfinite(packed)):
                raise ValueError("RoboDojo snapshot proprio packing is invalid")
            return packed
        except (KeyError, TypeError, ValueError):
            return None

    def frame_from_adapter(
        self,
        adapter: ObservationAdapter,
        camera_role: str = "head",
    ) -> TrackerFrame:
        role = self._validate_camera_role(camera_role)
        camera = adapter.camera(role, require_depth=True)
        intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64).reshape(
            3, 3
        )
        height, width = np.asarray(camera["depth"]).shape
        eef_pose = None
        if role != "head":
            arm = role.removesuffix("_wrist")
            eef_pose = adapter.state(f"{arm}_ee_pose")
        camera_to_env = self._camera_to_env_from_eef(
            role,
            eef_pose,
            camera["extrinsic_matrix"],
        )
        camera_to_base = self._camera_to_base(camera_to_env)
        status = adapter.status()
        return TrackerFrame(
            rgb=np.asarray(camera["rgb"]).copy(),
            depth_linear=np.asarray(camera["depth"], dtype=np.float32).copy(),
            intrinsics=CameraIntrinsics(
                width=int(width),
                height=int(height),
                fx=float(intrinsic[0, 0]),
                fy=float(intrinsic[1, 1]),
                cx=float(intrinsic[0, 2]),
                cy=float(intrinsic[1, 2]),
            ),
            camera_to_robot_base=camera_to_base,
            camera_to_policy=camera_to_base.copy(),
            sequence=int(camera["sequence"]),
            timestamp_s=float(status.get("updated_at_monotonic") or time.monotonic()),
            episode_id=str(camera["episode_id"]),
            camera_role=role,
            color_order="rgb",
            proprio=self._proprio(adapter),
        )

    def frame_from_snapshot(self, snapshot: Snapshot) -> TrackerFrame:
        """Build the exact tracker frame represented by a frozen RGB-D capture."""

        role = self._validate_camera_role(snapshot.role)
        if snapshot.depth is None:
            raise ValueError(f"{role} capture has no evaluator depth")
        if snapshot.intrinsic_matrix is None or snapshot.extrinsic_matrix is None:
            raise ValueError(f"{role} capture has no complete camera geometry")
        rgb = np.asarray(snapshot.rgb)
        depth = np.asarray(snapshot.depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"{role} capture RGB must be HxWx3")
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError(f"{role} capture RGB and depth resolutions do not match")
        intrinsic = np.asarray(snapshot.intrinsic_matrix, dtype=np.float64).reshape(
            3, 3
        )
        eef_pose = None
        if role != "head":
            arm = role.removesuffix("_wrist")
            try:
                eef_pose = snapshot.robot[f"eef_{arm}"]
            except KeyError as exc:
                raise ValueError(
                    f"{role} capture has no synchronized {arm} EEF pose"
                ) from exc
        camera_to_env = self._camera_to_env_from_eef(
            role,
            eef_pose,
            snapshot.extrinsic_matrix,
        )
        camera_to_base = self._camera_to_base(camera_to_env)
        return TrackerFrame(
            rgb=rgb[:, :, :3].copy(),
            depth_linear=depth.copy(),
            intrinsics=CameraIntrinsics(
                width=int(depth.shape[1]),
                height=int(depth.shape[0]),
                fx=float(intrinsic[0, 0]),
                fy=float(intrinsic[1, 1]),
                cx=float(intrinsic[0, 2]),
                cy=float(intrinsic[1, 2]),
            ),
            camera_to_robot_base=camera_to_base,
            camera_to_policy=camera_to_base.copy(),
            sequence=int(snapshot.sequence),
            timestamp_s=0.0,
            episode_id=str(snapshot.episode_id),
            camera_role=role,
            color_order="rgb",
            proprio=self._snapshot_proprio(snapshot),
        )

    def update(self, adapter: ObservationAdapter) -> None:
        with self._lock:
            managers = list(self._managers.items())
            # One facade snapshot must never observe head at sequence N and
            # wrist at sequence N-1.  Keep the wrapper lock across ingestion
            # of every instantiated camera tracker.
            for role, manager in managers:
                try:
                    frame = self.frame_from_adapter(adapter, role)
                    manager.ingest(frame)
                    self._camera_errors.pop(role, None)
                except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    self._camera_errors[role] = message
                    manager.note_observation_error(exc)
            selected_role = self._selected_role()
            self._last_error = self._camera_errors.get(selected_role)
        # Keep the narrow legacy surface alive for compatibility callers and
        # deterministic fault-injection tests.  It never feeds the move tool.
        self._legacy._flow = self._flow
        self._legacy.update(adapter)

    def register_capture(self, snapshot: Snapshot) -> dict[str, Any]:
        role = self._validate_camera_role(snapshot.role)
        if snapshot.depth is None:
            raise ValueError(f"{role} capture has no evaluator depth")
        manager = self._manager_for_role(role)
        assert manager is not None
        frame = self.frame_from_snapshot(snapshot)
        latest = getattr(manager, "_latest_frame", None)
        latest_sequence = None if latest is None else int(latest.sequence)
        frame_sequence = int(frame.sequence)
        if latest is not None and latest_sequence > frame_sequence:
            raise ValueError(
                f"{role} tracker advanced beyond the frozen capture sequence"
            )
        if (
            latest is None
            or latest_sequence < frame_sequence
            or tracker_frame_content_digest(latest)
            != tracker_frame_content_digest(frame)
        ):
            # A later Human capture of the current observation is a new freeze,
            # including same-sequence wrist RGB-D after the startup preset
            # already registered that slot.
            manager.ingest(frame)
        report = manager.register_capture(
            session_id=snapshot.session_id,
            image_id=snapshot.image_id,
            observation_sequence=int(snapshot.sequence),
            episode_id=snapshot.episode_id,
            image_shape=tuple(int(value) for value in snapshot.depth.shape),
            camera_role=role,
        )
        report["camera_role"] = role
        with self._lock:
            self._capture_bindings[(snapshot.session_id, snapshot.image_id)] = dict(
                report
            )
            self._last_capture_camera_role = role
        return report

    def register(
        self,
        snapshot: Snapshot,
        points: list[dict[str, Any]],
    ) -> dict[str, Any]:
        key = (snapshot.session_id, snapshot.image_id)
        with self._lock:
            binding = deepcopy(self._capture_bindings.get(key))
        if binding is None:
            binding = self.register_capture(snapshot)
        role = self._validate_camera_role(binding.get("camera_role", snapshot.role))
        cleared = self.clear_for_reselection(camera_role=role)
        manager = self._manager_for_role(role)
        assert manager is not None
        committed = manager.replace_points(
            points,
            session_id=snapshot.session_id,
            image_id=snapshot.image_id,
            capture_observation_sequence=int(snapshot.sequence),
            capture_episode_id=snapshot.episode_id,
            capture_image_shape=tuple(int(value) for value in snapshot.depth.shape),
            capture_frame_digest=str(binding["frame_digest"]),
        )
        # The compatibility mirror models the historical head tracker only.
        if role == "head":
            self._legacy.register(snapshot, points)
        public = {
            "session_id": committed["session_id"],
            "image_id": committed["image_id"],
            "episode_id": committed["episode_id"],
            "capture_observation_sequence": int(
                committed["capture_observation_sequence"]
            ),
            "observation_sequence": int(committed["observation_sequence"]),
            "replayed_observation_count": int(committed["replayed_observation_count"]),
            "frame_binding": committed["frame_binding"],
            "frame_digest": committed["frame_digest"],
            "replaced_names": list(cleared.get("cleared_names") or []),
            "tracked_names": list(committed["entries"]),
            "tracked_object_distances": deepcopy(committed["entries"]),
            "points": list(deepcopy(committed["entries"]).values()),
        }
        with self._lock:
            self._active_result = deepcopy(public)
            slot = self._tracking_slot(role)
            self._active_results_by_slot[slot] = deepcopy(public)
            self._slot_roles[slot] = role
            self._active_camera_role = role
            self._last_capture_camera_role = role
        return public

    def clear_for_reselection(
        self,
        *,
        camera_role: str | None = None,
    ) -> dict[str, Any]:
        """Clear one logical camera slot while retaining frame/replay history.

        Head replaces only head.  Either wrist replaces the shared wrist slot,
        which also retires a prior registration made from the other wrist.
        """

        cleared_names: list[str] = []
        role = self._validate_camera_role(camera_role or self._selected_role())
        slot = self._tracking_slot(role)
        roles = self._slot_camera_roles(slot)
        with self._lock:
            self._retire_composite_leases_for_roles_locked(roles)
            for candidate_role in roles:
                manager = self._managers.get(candidate_role)
                if manager is None:
                    continue
                report = manager.clear_for_reselection()
                for name in report.get("cleared_names") or []:
                    normalized = str(name)
                    if normalized not in cleared_names:
                        cleared_names.append(normalized)
            if slot == "head":
                self._legacy.reset()
            self._active_results_by_slot.pop(slot, None)
            self._slot_roles[slot] = None
            self._reselect_fallback_locked()
            self._last_error = None
        return {
            "cleared_names": cleared_names,
            "cleared_count": len(cleared_names),
            "replacement": "complete_camera_slot",
            "replacement_slot": slot,
            "camera_role": role,
        }

    def replace_points(
        self,
        points: list[dict[str, Any]],
        *,
        session_id: str,
        image_id: str,
        capture_observation_sequence: int,
        capture_episode_id: str,
        capture_image_shape: tuple[int, int],
        capture_frame_digest: str,
        replaced_names: list[str] | tuple[str, ...] = (),
        check_cancelled: Any | None = None,
        commit_guard: Any | None = None,
        commit_result: Any | None = None,
    ) -> dict[str, Any]:
        """Delegate strict replacement and publish the wrapper's active result."""

        replacement_names = [str(name) for name in replaced_names]
        key = (str(session_id), str(image_id))
        with self._lock:
            binding = deepcopy(self._capture_bindings.get(key))
        if binding is None:
            raise ValueError("capture frame binding was not registered")
        role = self._validate_camera_role(binding.get("camera_role", "head"))
        cleared = self.clear_for_reselection(camera_role=role)
        for name in cleared.get("cleared_names") or []:
            normalized = str(name)
            if normalized not in replacement_names:
                replacement_names.append(normalized)
        manager = self._manager_for_role(role)
        assert manager is not None

        def _publish(committed: dict[str, Any]) -> None:
            if commit_result is not None:
                commit_result(committed)
            public = {
                "session_id": committed["session_id"],
                "image_id": committed["image_id"],
                "episode_id": committed["episode_id"],
                "capture_observation_sequence": int(
                    committed["capture_observation_sequence"]
                ),
                "observation_sequence": int(committed["observation_sequence"]),
                "replayed_observation_count": int(
                    committed["replayed_observation_count"]
                ),
                "frame_binding": committed["frame_binding"],
                "frame_digest": committed["frame_digest"],
                "replaced_names": replacement_names,
                "tracked_names": list(committed["entries"]),
                "tracked_object_distances": deepcopy(committed["entries"]),
                "points": list(deepcopy(committed["entries"]).values()),
            }
            with self._lock:
                self._active_result = deepcopy(public)
                slot = self._tracking_slot(role)
                self._active_results_by_slot[slot] = deepcopy(public)
                self._slot_roles[slot] = role
                self._active_camera_role = role
                self._last_capture_camera_role = role
                self._last_error = None

        return manager.replace_points(
            points,
            session_id=session_id,
            image_id=image_id,
            capture_observation_sequence=capture_observation_sequence,
            capture_episode_id=capture_episode_id,
            capture_image_shape=capture_image_shape,
            capture_frame_digest=capture_frame_digest,
            check_cancelled=check_cancelled,
            commit_guard=commit_guard,
            commit_result=_publish,
        )

    def remember_legacy_registration(
        self,
        snapshot: Snapshot,
        points: list[dict[str, Any]],
    ) -> None:
        """Keep the compatibility mirror available to older local callers."""

        if self._validate_camera_role(snapshot.role) == "head":
            self._legacy.register(snapshot, points)

    def _manager_registered(self) -> bool:
        with self._lock:
            managers = [
                manager for _slot, _role, manager in self._active_source_items_locked()
            ]
        return any(bool(manager.status().get("registered")) for manager in managers)

    def _merged_memory_fields(
        self,
        manager: TrackedObjectDistanceMemory | None = None,
        camera_role: str | None = None,
    ) -> dict[str, Any]:
        selected_role = self._validate_camera_role(camera_role or self._selected_role())
        selected_manager = manager or self._manager_for_role(selected_role)
        assert selected_manager is not None
        fields = selected_manager.memory_fields()
        policy = fields.setdefault(TRACKING_POLICY_MEMORY_KEY, {})
        policy.update(
            {
                "camera": selected_role,
                "camera_role": selected_role,
                "supported_camera_roles": list(TRACKING_CAMERA_ROLES),
                "persistent_tracking": True,
                "tabletop_anchors": "registration_rgbd_only_fixed_until_reselection_or_reset",
                "tracking_lifetime": "until_replaced_episode_reset_or_retired",
                "robodojo_env_axes": "x_right_y_forward_z_up",
                "wrist_camera_pose_source": (
                    "not_applicable"
                    if selected_role == "head"
                    else "current_eef_pose_times_official_x5_hand_eye"
                ),
                "observation_contract": [
                    "*::rgb",
                    "*::depth_linear",
                    "*::cam_rel_poses",
                    "*::proprio",
                ],
                "frame_alignment": (
                    "required_session_id_image_id_and_tracker_frame_sha256"
                ),
            }
        )
        with self._lock:
            diagnostic = deepcopy(self._hand_eye_diagnostics.get(selected_role))
        if diagnostic is not None:
            policy["wrist_hand_eye_diagnostic"] = diagnostic
        entries = fields.setdefault(TRACKED_OBJECT_DISTANCES_MEMORY_KEY, {})
        # The official tracker is the sole authority once a camera slot has
        # been registered.  The old optical-flow mirror can be one or more
        # frames ahead/behind and used to overwrite only ``pixel_uv`` and
        # ``relative_uv`` here, leaving ``u/v`` and XYZ from the official
        # tracker.  That produced internally impossible records (a point's
        # displayed pixel did not back-project to its published XYZ).  Keep
        # the mirror available for the legacy-only path below, but never merge
        # it into an official entry.
        latest = getattr(selected_manager, "_latest_frame", None)
        legacy_entries = (
            self._legacy.memory_fields().get(TRACKED_OBJECT_DISTANCES_MEMORY_KEY, {})
            if selected_role == "head"
            else {}
        )
        for name, entry in entries.items():
            legacy = legacy_entries.get(name, {})
            # The legacy optical-flow mirror can legitimately be temporarily
            # unobserved while the official RGB-D manager still sees the
            # point (especially for a wrist-only point shown in a head view).
            # Keep that fact as metadata, but never copy legacy coordinates,
            # pixels, status, or sequence into an official entry.  Mixing the
            # two sources is exactly how an apparently valid 3-D line can be
            # disconnected from the rendered frame.
            if legacy.get("status") in (TEMPORARILY_UNOBSERVED, LOST):
                entry["legacy_tracker_status"] = str(legacy.get("status"))
                entry["legacy_tracker_observation_sequence"] = legacy.get(
                    "observation_sequence"
                )
            entry["observed"] = entry.get("status") == OBSERVED
            entry["retired"] = entry.get("status") == LOST
            if entry.get("status") != OBSERVED or latest is None:
                continue
            # Normalize all public image-coordinate fields from the same
            # authoritative current pixel.  This also makes hot-reloaded
            # tracker states created by an older build self-heal on read.
            try:
                width = int(latest.intrinsics.width)
                height = int(latest.intrinsics.height)
                pixel = np.asarray(entry.get("pixel_uv"), dtype=np.float64).reshape(2)
                if not np.all(np.isfinite(pixel)) or width <= 1 or height <= 1:
                    raise ValueError
                relative = np.asarray(
                    [
                        pixel[0] / float(width - 1) * 1000.0,
                        pixel[1] / float(height - 1) * 1000.0,
                    ],
                    dtype=np.float64,
                )
                entry["pixel_uv"] = pixel.astype(float).tolist()
                entry["u"] = float(relative[0])
                entry["v"] = float(relative[1])
                entry["relative_uv"] = relative.astype(float).tolist()
            except (TypeError, ValueError, AttributeError):
                # The manager's strict _memory_entry validation normally makes
                # this unreachable.  Do not make memory unavailable merely
                # because a legacy hot-reload state lacks intrinsics.
                pass
        return fields

    def reprojection_diagnostics(
        self,
        adapter: ObservationAdapter,
        *,
        camera_roles: tuple[str, ...] = TRACKING_CAMERA_ROLES,
    ) -> dict[str, Any]:
        """Project live base-frame tracks into every available RGB-D view.

        The public tracker coordinates are in ``current_robot_base``.  A
        head image and a wrist image have different optical origins, so their
        raw relative ``(u,v)`` values are not comparable.  This diagnostic is
        deliberately observation-only and gives the UI a same-frame set of
        pixels for visual verification.
        """

        fields = self.memory_fields()
        entries = fields.get(TRACKED_OBJECT_DISTANCES_MEMORY_KEY, {})
        points: dict[str, np.ndarray] = {}
        point_entries: dict[str, Mapping[str, Any]] = {}
        for name, entry in entries.items():
            if not isinstance(entry, Mapping) or entry.get("status") != OBSERVED:
                continue
            try:
                value = np.asarray(
                    entry.get("xyz_in_robot_base_coord_m"), dtype=np.float64
                ).reshape(3)
            except (TypeError, ValueError):
                continue
            if np.all(np.isfinite(value)):
                points[str(name)] = value
                point_entries[str(name)] = entry

        def _line_report(point_map: Mapping[str, np.ndarray]) -> dict[str, Any]:
            report: dict[str, Any] = {
                "point_names": list(point_map),
                "point_count": len(point_map),
                "frame": "current_robot_base",
                "axes": "x_forward_y_left_z_up",
                "tolerance_m": TRACKING_COLLINEAR_TOLERANCE_M,
                "rank": None,
                "direction": None,
                "max_perpendicular_residual_m": None,
                "vertical_tilt_deg": None,
                "ok": False,
            }
            if len(point_map) < 2:
                return report
            matrix = np.asarray(list(point_map.values()), dtype=np.float64)
            center = matrix.mean(axis=0)
            centered = matrix - center
            _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
            rank = int(np.count_nonzero(singular > 1.0e-9))
            direction = vh[0] if vh.size else np.zeros(3, dtype=np.float64)
            if direction[2] < 0.0:
                direction = -direction
            residual = np.linalg.norm(
                np.cross(centered, direction.reshape(1, 3)), axis=1
            )
            max_residual = float(np.max(residual))
            report.update(
                {
                    "rank": rank,
                    "direction": direction.astype(float).tolist(),
                    "max_perpendicular_residual_m": max_residual,
                    "vertical_tilt_deg": float(
                        np.degrees(
                            np.arctan2(
                                np.linalg.norm(direction[:2]),
                                abs(float(direction[2])),
                            )
                        )
                    ),
                    "ok": bool(max_residual <= TRACKING_COLLINEAR_TOLERANCE_M),
                }
            )
            return report

        visual_points: dict[str, np.ndarray] = {}
        for name, entry in point_entries.items():
            try:
                raw_value = np.asarray(
                    entry.get("raw_xyz_in_robot_base_coord_m"),
                    dtype=np.float64,
                ).reshape(3)
            except (TypeError, ValueError):
                raw_value = points[name]
            visual_points[name] = (
                raw_value if np.all(np.isfinite(raw_value)) else points[name]
            )
        collinearity = _line_report(points)
        visual_collinearity = _line_report(visual_points)

        projections: dict[str, Any] = {}
        for role in camera_roles:
            try:
                camera = adapter.camera(str(role), require_depth=False)
                intrinsic = np.asarray(
                    camera.get("intrinsic_matrix"), dtype=np.float64
                ).reshape(3, 3)
                extrinsic = np.asarray(
                    camera.get("extrinsic_matrix"), dtype=np.float64
                ).reshape(4, 4)
                if not np.all(np.isfinite(intrinsic)) or not np.all(
                    np.isfinite(extrinsic)
                ):
                    raise ValueError("camera matrices are non-finite")
                height, width = camera["rgb"].shape[:2]
                depth = camera.get("depth")
                if depth is not None:
                    depth = np.asarray(depth, dtype=np.float64)
                env_from_base = np.eye(4, dtype=np.float64)
                env_from_base[:3, :3] = self._base_rotation
                camera_to_env = extrinsic
                camera_pose_source = "evaluator_camera_extrinsic"
                if str(role) != "head":
                    arm = str(role).removesuffix("_wrist")
                    try:
                        eef_pose = adapter.state(f"{arm}_ee_pose")
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        eef_pose = None
                    if eef_pose is not None:
                        # Keep projection diagnostics on the exact same
                        # hand-eye path as DynamicPointTracker.  The evaluator
                        # wrist extrinsic is retained only for the calibration
                        # delta reported by _camera_to_env_from_eef.
                        camera_to_env = self._camera_to_env_from_eef(
                            str(role), eef_pose, extrinsic
                        )
                        camera_pose_source = "eef_pose_times_official_x5_hand_eye"
                camera_from_env = np.linalg.inv(camera_to_env)

                def _project_base_point(base_point: np.ndarray) -> dict[str, Any]:
                    """Project one public-base point through this exact frame."""

                    env_point = (env_from_base @ np.r_[base_point, 1.0])[:3]
                    usd_point = (camera_from_env @ np.r_[env_point, 1.0])[:3]
                    optical = usd_point * OPTICAL_TO_USD_AXES
                    z_value = float(optical[2])
                    if z_value <= 1.0e-9:
                        return {
                            "valid": False,
                            "reason": "point_behind_camera",
                            "optical": optical,
                            "depth_m": z_value,
                        }
                    projected_x = float(
                        intrinsic[0, 0] * optical[0] / z_value + intrinsic[0, 2]
                    )
                    projected_y = float(
                        intrinsic[1, 1] * optical[1] / z_value + intrinsic[1, 2]
                    )
                    return {
                        "valid": bool(
                            0.0 <= projected_x < width and 0.0 <= projected_y < height
                        ),
                        "pixel_uv": [projected_x, projected_y],
                        "relative_uv": [
                            projected_x / float(max(1, width - 1)) * 1000.0,
                            projected_y / float(max(1, height - 1)) * 1000.0,
                        ],
                        "optical": optical,
                        "depth_m": z_value,
                    }

                role_points: dict[str, Any] = {}
                for name, base_point in points.items():
                    public_projection = _project_base_point(base_point)
                    source_entry = point_entries.get(name, {})
                    raw_geometry_point = base_point
                    raw_geometry_source = "public_xyz"
                    try:
                        raw_candidate = np.asarray(
                            source_entry.get("raw_xyz_in_robot_base_coord_m"),
                            dtype=np.float64,
                        ).reshape(3)
                        if np.all(np.isfinite(raw_candidate)):
                            raw_geometry_point = raw_candidate
                            raw_geometry_source = "raw_rgbd_measurement"
                    except (TypeError, ValueError):
                        pass
                    visual_projection = _project_base_point(raw_geometry_point)
                    z = float(public_projection["depth_m"])
                    if not public_projection.get("valid") and (
                        public_projection.get("reason") == "point_behind_camera"
                    ):
                        role_points[name] = {
                            "valid": False,
                            "reason": "point_behind_camera",
                            "xyz_robot_base_m": base_point.astype(float).tolist(),
                            "visual_valid": bool(visual_projection.get("valid")),
                            "visual_xyz_robot_base_m": raw_geometry_point.astype(
                                float
                            ).tolist(),
                            "visual_xyz_source": raw_geometry_source,
                        }
                        continue
                    px, py = public_projection["pixel_uv"]
                    visible = bool(public_projection.get("valid"))
                    point_report = {
                        "valid": visible,
                        "pixel_uv": [px, py],
                        "relative_uv": public_projection["relative_uv"],
                        "xyz_robot_base_m": base_point.astype(float).tolist(),
                        "camera_depth_m": z,
                        # ``visual_*`` is deliberately based on the direct
                        # current RGB-D XYZ when a rigid-pair correction is
                        # active.  A policy-owned fused point must not make a
                        # visibly non-collinear scene pass the image gate.
                        "visual_valid": bool(visual_projection.get("valid")),
                        "visual_pixel_uv": visual_projection.get("pixel_uv"),
                        "visual_relative_uv": visual_projection.get("relative_uv"),
                        "visual_xyz_robot_base_m": raw_geometry_point.astype(
                            float
                        ).tolist(),
                        "visual_xyz_source": raw_geometry_source,
                        "visual_camera_depth_m": visual_projection.get("depth_m"),
                    }
                    source_camera_role = source_entry.get(
                        "camera_role"
                    ) or source_entry.get("camera")
                    source_pixel = source_entry.get("pixel_uv")
                    try:
                        source_pixel_array = np.asarray(
                            source_pixel, dtype=np.float64
                        ).reshape(2)
                        if not np.all(np.isfinite(source_pixel_array)):
                            raise ValueError
                        source_pixel = source_pixel_array.astype(float).tolist()
                    except (TypeError, ValueError):
                        source_pixel = None
                    source_sequence = source_entry.get("observation_sequence")
                    try:
                        source_sequence = int(source_sequence)
                    except (TypeError, ValueError):
                        source_sequence = None
                    point_report.update(
                        {
                            "source_camera_role": (source_camera_role),
                            "source_observation_sequence": source_sequence,
                            "source_pixel_uv": source_pixel,
                        }
                    )
                    if (
                        source_pixel is not None
                        and visual_projection.get("pixel_uv") is not None
                        and str(source_camera_role or "") == str(role)
                    ):
                        source_pixel_error = float(
                            np.linalg.norm(
                                np.asarray(
                                    visual_projection["pixel_uv"],
                                    dtype=np.float64,
                                )
                                - np.asarray(source_pixel, dtype=np.float64)
                            )
                        )
                        point_report.update(
                            {
                                "source_reprojection_error_px": source_pixel_error,
                                "source_reprojection_tolerance_px": (
                                    TRACKING_SOURCE_REPROJECTION_TOLERANCE_PX
                                ),
                                "source_reprojection_ok": bool(
                                    visual_projection.get("valid")
                                    and 0.0 <= float(source_pixel[0]) < width
                                    and 0.0 <= float(source_pixel[1]) < height
                                    and source_pixel_error
                                    <= TRACKING_SOURCE_REPROJECTION_TOLERANCE_PX
                                ),
                            }
                        )
                    if visible and depth is not None:
                        sample_x = int(np.clip(round(px), 0, width - 1))
                        sample_y = int(np.clip(round(py), 0, height - 1))
                        observed_depth = float(depth[sample_y, sample_x])
                        if np.isfinite(observed_depth) and observed_depth > 0.0:
                            depth_residual = abs(observed_depth - z)
                            point_report.update(
                                {
                                    "rgbd_depth_at_pixel_m": observed_depth,
                                    "rgbd_depth_residual_m": depth_residual,
                                    "rgbd_depth_tolerance_m": (
                                        TRACKING_REPROJECTION_DEPTH_TOLERANCE_M
                                    ),
                                    "rgbd_depth_consistent": bool(
                                        depth_residual
                                        <= TRACKING_REPROJECTION_DEPTH_TOLERANCE_M
                                    ),
                                }
                            )
                    if visual_projection.get("valid") and depth is not None:
                        visual_x, visual_y = visual_projection["pixel_uv"]
                        sample_x = int(np.clip(round(visual_x), 0, width - 1))
                        sample_y = int(np.clip(round(visual_y), 0, height - 1))
                        visual_observed_depth = float(depth[sample_y, sample_x])
                        if (
                            np.isfinite(visual_observed_depth)
                            and visual_observed_depth > 0.0
                        ):
                            visual_depth_residual = abs(
                                visual_observed_depth
                                - float(visual_projection["depth_m"])
                            )
                            point_report.update(
                                {
                                    "visual_rgbd_depth_at_pixel_m": (
                                        visual_observed_depth
                                    ),
                                    "visual_rgbd_depth_residual_m": (
                                        visual_depth_residual
                                    ),
                                    "visual_rgbd_depth_consistent": bool(
                                        visual_depth_residual
                                        <= TRACKING_REPROJECTION_DEPTH_TOLERANCE_M
                                    ),
                                }
                            )
                    role_points[name] = point_report

                visible_pixels = np.asarray(
                    [
                        report["pixel_uv"]
                        for report in role_points.values()
                        if report.get("valid")
                    ],
                    dtype=np.float64,
                )
                image_collinearity: dict[str, Any] = {
                    "point_count": int(len(visible_pixels)),
                    "tolerance_px": TRACKING_IMAGE_COLLINEAR_TOLERANCE_PX,
                    "max_perpendicular_residual_px": None,
                    "ok": False,
                }
                if len(visible_pixels) >= 2:
                    pixel_center = visible_pixels.mean(axis=0)
                    pixel_centered = visible_pixels - pixel_center
                    _pixel_u, _pixel_singular, pixel_vh = np.linalg.svd(
                        pixel_centered, full_matrices=False
                    )
                    pixel_direction = pixel_vh[0]
                    pixel_residual = np.abs(
                        pixel_centered[:, 0] * pixel_direction[1]
                        - pixel_centered[:, 1] * pixel_direction[0]
                    )
                    max_pixel_residual = float(np.max(pixel_residual))
                    image_collinearity.update(
                        {
                            "max_perpendicular_residual_px": max_pixel_residual,
                            "ok": bool(
                                max_pixel_residual
                                <= TRACKING_IMAGE_COLLINEAR_TOLERANCE_PX
                            ),
                        }
                    )
                visual_pixels = np.asarray(
                    [
                        report["visual_pixel_uv"]
                        for report in role_points.values()
                        if report.get("visual_valid")
                        and report.get("visual_pixel_uv") is not None
                    ],
                    dtype=np.float64,
                )
                visual_image_collinearity: dict[str, Any] = {
                    "point_count": int(len(visual_pixels)),
                    "tolerance_px": TRACKING_IMAGE_COLLINEAR_TOLERANCE_PX,
                    "max_perpendicular_residual_px": None,
                    "ok": False,
                }
                if len(visual_pixels) >= 2:
                    visual_center = visual_pixels.mean(axis=0)
                    visual_centered = visual_pixels - visual_center
                    _visual_u, _visual_singular, visual_vh = np.linalg.svd(
                        visual_centered, full_matrices=False
                    )
                    visual_direction = visual_vh[0]
                    visual_residual = np.abs(
                        visual_centered[:, 0] * visual_direction[1]
                        - visual_centered[:, 1] * visual_direction[0]
                    )
                    visual_image_collinearity.update(
                        {
                            "max_perpendicular_residual_px": float(
                                np.max(visual_residual)
                            ),
                            "ok": bool(
                                np.max(visual_residual)
                                <= TRACKING_IMAGE_COLLINEAR_TOLERANCE_PX
                            ),
                        }
                    )
                camera_sequence = int(camera["sequence"])
                point_sequences = {
                    name: report.get("source_observation_sequence")
                    for name, report in role_points.items()
                }
                same_sequence = bool(
                    role_points
                    and all(
                        report.get("visual_valid", report.get("valid"))
                        and report.get("source_observation_sequence") is not None
                        and abs(
                            int(report["source_observation_sequence"]) - camera_sequence
                        )
                        <= TRACKING_SAME_FRAME_SEQUENCE_SKEW
                        for report in role_points.values()
                    )
                )
                depth_checked = any(
                    "rgbd_depth_consistent" in report for report in role_points.values()
                )
                depth_consistent = bool(
                    depth_checked
                    and all(
                        report.get("rgbd_depth_consistent") is True
                        for report in role_points.values()
                        if report.get("valid")
                    )
                )
                visual_depth_checked = any(
                    "visual_rgbd_depth_consistent" in report
                    for report in role_points.values()
                )
                visual_depth_consistent = bool(
                    visual_depth_checked
                    and all(
                        report.get("visual_rgbd_depth_consistent") is True
                        for report in role_points.values()
                        if report.get("visual_valid")
                    )
                )
                projections[str(role)] = {
                    "ok": True,
                    "observation_sequence": camera_sequence,
                    "image_shape": [int(height), int(width)],
                    "camera_pose_source": camera_pose_source,
                    "points": role_points,
                    "image_collinearity": image_collinearity,
                    "visual_image_collinearity": visual_image_collinearity,
                    "same_frame": {
                        "camera_observation_sequence": camera_sequence,
                        "point_observation_sequences": point_sequences,
                        "same_observation_sequence": same_sequence,
                        "sequence_skew_tolerance": TRACKING_SAME_FRAME_SEQUENCE_SKEW,
                    },
                    "depth_consistency": {
                        "checked": depth_checked,
                        "all_valid_points_consistent": depth_consistent,
                    },
                    "visual_depth_consistency": {
                        "checked": visual_depth_checked,
                        "all_valid_points_consistent": visual_depth_consistent,
                    },
                }
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                projections[str(role)] = {"ok": False, "reason": str(exc)}
        # Validate every authoritative source pixel against the projection in
        # the camera that produced it.  A point can still project neatly into
        # another camera when its stored pixel/XYZ pair is stale or has been
        # assembled from the legacy compatibility mirror; that must not be
        # presented as visual alignment evidence.
        source_reprojection_checks: list[dict[str, Any]] = []
        for name, entry in point_entries.items():
            source_role = str(
                entry.get("camera_role") or entry.get("camera") or ""
            ).strip()
            source_pixel = entry.get("pixel_uv")
            if not source_role or source_pixel is None:
                continue
            try:
                source_pixel_array = np.asarray(source_pixel, dtype=np.float64).reshape(
                    2
                )
                if not np.all(np.isfinite(source_pixel_array)):
                    raise ValueError
            except (TypeError, ValueError):
                source_reprojection_checks.append(
                    {
                        "point_name": str(name),
                        "camera_role": source_role,
                        "error_px": None,
                        "ok": False,
                        "reason": "source_pixel_invalid",
                    }
                )
                continue
            source_projection = projections.get(source_role)
            source_report = (
                (source_projection or {}).get("points", {}).get(name)
                if isinstance(source_projection, Mapping)
                else None
            )
            error_px = (
                source_report.get("source_reprojection_error_px")
                if isinstance(source_report, Mapping)
                else None
            )
            source_ok = bool(
                isinstance(source_report, Mapping)
                and source_report.get("visual_valid", source_report.get("valid"))
                and error_px is not None
                and math.isfinite(float(error_px))
                and float(error_px) <= TRACKING_SOURCE_REPROJECTION_TOLERANCE_PX
            )
            source_reprojection_checks.append(
                {
                    "point_name": str(name),
                    "camera_role": source_role,
                    "observation_sequence": (
                        source_projection.get("observation_sequence")
                        if isinstance(source_projection, Mapping)
                        else None
                    ),
                    "source_observation_sequence": (
                        source_report.get("source_observation_sequence")
                        if isinstance(source_report, Mapping)
                        else entry.get("observation_sequence")
                    ),
                    "error_px": (None if error_px is None else float(error_px)),
                    "tolerance_px": TRACKING_SOURCE_REPROJECTION_TOLERANCE_PX,
                    "ok": source_ok,
                    "reason": (
                        None
                        if source_ok
                        else "source_projection_unavailable_or_inconsistent"
                    ),
                }
            )
        source_reprojection_ok = bool(
            not source_reprojection_checks
            or all(item.get("ok") is True for item in source_reprojection_checks)
        )
        return {
            "frame": "current_robot_base",
            "axes": "x_forward_y_left_z_up",
            "vertical_semantics": (
                "vertical_tilt_deg is measured against robot-base +Z; a base +Z "
                "line is generally diagonal in the tilted head image, so image-Y "
                "vertical is not a valid world-vertical test"
            ),
            "collinearity": collinearity,
            "visual_collinearity": visual_collinearity,
            "projections": projections,
            "source_reprojection": {
                "checked": bool(source_reprojection_checks),
                "point_count": len(source_reprojection_checks),
                "tolerance_px": TRACKING_SOURCE_REPROJECTION_TOLERANCE_PX,
                "all_points_consistent": source_reprojection_ok,
                "checks": source_reprojection_checks,
            },
            "comparison_rule": (
                "compare projected pixels within one camera role; raw head and "
                "wrist relative_uv values are not comparable"
            ),
        }

    def visual_alignment_gate(
        self,
        adapter: ObservationAdapter,
        point_names: list[str] | tuple[str, ...],
        *,
        expected_sequence: int | None = None,
        camera_roles: tuple[str, ...] = TRACKING_CAMERA_ROLES,
    ) -> dict[str, Any]:
        """Verify named points are visually collinear in one synchronized view.

        This is intentionally stricter than the base-frame SVD diagnostic.  A
        base-frame line can be assembled from observations captured by
        different cameras or different evaluator sequences; that is useful for
        debugging but must not be presented as a visual alignment proof.
        """

        requested = [str(name) for name in point_names]
        if not requested or len(set(requested)) != len(requested):
            return {
                "ok": False,
                "applicable": False,
                "reason": "visual_alignment_point_names_invalid",
                "point_names": requested,
            }
        diagnostic = self.reprojection_diagnostics(
            adapter,
            camera_roles=tuple(str(role) for role in camera_roles),
        )
        # A compatibility/synthetic provider may expose points only through
        # its tool-facing snapshot API, without publishing official RGB-D
        # memory entries.  In that case there is no honest camera projection
        # to verify; report the gate as not applicable instead of turning a
        # successful non-visual unit test into a visual failure.
        has_projected_requested_point = any(
            any(
                name
                in (
                    (projection.get("points") or {})
                    if isinstance(projection, Mapping)
                    else {}
                )
                for name in requested
            )
            for projection in (diagnostic.get("projections") or {}).values()
        )
        if not has_projected_requested_point:
            return {
                "ok": True,
                "applicable": False,
                "reason": "visual_alignment_observation_metadata_unavailable",
                "point_names": requested,
                "expected_sequence": expected_sequence,
                "vertical_semantics": diagnostic.get("vertical_semantics"),
                "candidates": [],
            }
        # Validate each point against the camera that actually supplied its
        # observed pixel before considering another camera as a common view.
        # Otherwise a bad source pixel could be hidden by a mathematically
        # convenient projection in a different camera role.
        source_reprojection_checks = [
            dict(check)
            for check in (
                (diagnostic.get("source_reprojection") or {}).get("checks") or []
            )
            if str(check.get("point_name") or "") in requested
        ]
        source_reprojection_global_ok = bool(
            not source_reprojection_checks
            or all(item["ok"] for item in source_reprojection_checks)
        )
        candidates: list[dict[str, Any]] = []
        for role in camera_roles:
            role_name = str(role)
            projection = (diagnostic.get("projections") or {}).get(role_name)
            if not isinstance(projection, Mapping) or not projection.get("ok"):
                continue
            point_reports = projection.get("points") or {}
            missing = [name for name in requested if name not in point_reports]
            invalid = [
                name
                for name in requested
                if name in point_reports
                and not bool(point_reports[name].get("visual_valid"))
            ]
            # The final visual gate uses direct RGB-D geometry.  Public
            # ``depth_consistency`` can describe a policy-fused point, which
            # is useful for planning diagnostics but is not evidence that the
            # rendered pixel contains the measured surface.
            depth_report = projection.get("visual_depth_consistency") or {}
            # Recompute the image fit over exactly the requested points.  The
            # global diagnostic may contain unrelated historical tracks, and
            # those must not change this request's acceptance decision.
            requested_pixels = np.asarray(
                [
                    point_reports[name]["visual_pixel_uv"]
                    for name in requested
                    if name in point_reports
                    and bool(point_reports[name].get("visual_valid"))
                    and point_reports[name].get("visual_pixel_uv") is not None
                ],
                dtype=np.float64,
            )
            image_fit: dict[str, Any] = {
                "point_count": int(len(requested_pixels)),
                "tolerance_px": TRACKING_IMAGE_COLLINEAR_TOLERANCE_PX,
                "max_perpendicular_residual_px": None,
                "ok": False,
            }
            if len(requested_pixels) >= 2:
                pixel_center = requested_pixels.mean(axis=0)
                pixel_centered = requested_pixels - pixel_center
                _u, _s, pixel_vh = np.linalg.svd(pixel_centered, full_matrices=False)
                pixel_direction = pixel_vh[0]
                pixel_residual = np.abs(
                    pixel_centered[:, 0] * pixel_direction[1]
                    - pixel_centered[:, 1] * pixel_direction[0]
                )
                image_fit["max_perpendicular_residual_px"] = float(
                    np.max(pixel_residual)
                )
                image_fit["ok"] = bool(
                    image_fit["max_perpendicular_residual_px"]
                    <= TRACKING_IMAGE_COLLINEAR_TOLERANCE_PX
                )
            camera_sequence = projection.get("observation_sequence")
            expected_sequence_ok = expected_sequence is None or (
                camera_sequence is not None
                and int(camera_sequence) == int(expected_sequence)
            )
            requested_reports = [
                point_reports[name] for name in requested if name in point_reports
            ]
            requested_depth_ok = bool(
                requested_reports
                and all(
                    report.get("visual_rgbd_depth_consistent") is True
                    for report in requested_reports
                )
            )
            requested_same_sequence = bool(
                requested_reports
                and all(
                    report.get("visual_valid")
                    and report.get("source_observation_sequence") is not None
                    and camera_sequence is not None
                    and abs(
                        int(report["source_observation_sequence"])
                        - int(camera_sequence)
                    )
                    <= TRACKING_SAME_FRAME_SEQUENCE_SKEW
                    for report in requested_reports
                )
            )
            source_reprojection_reports = [
                point_reports[name]
                for name in requested
                if name in point_reports
                and point_reports[name].get("source_pixel_uv") is not None
                and str(point_reports[name].get("source_camera_role") or "")
                == role_name
            ]
            source_reprojection_ok = bool(
                not source_reprojection_reports
                or all(
                    report.get("source_reprojection_ok") is True
                    for report in source_reprojection_reports
                )
            )
            candidate = {
                "camera_role": role_name,
                "camera_observation_sequence": camera_sequence,
                "point_names": requested,
                "missing_points": missing,
                "invalid_points": invalid,
                "same_observation_sequence": requested_same_sequence,
                "expected_sequence": expected_sequence,
                "expected_sequence_ok": expected_sequence_ok,
                "all_points_visible": not missing and not invalid,
                "image_collinearity": dict(image_fit),
                "source_reprojection": {
                    "checked": bool(source_reprojection_reports),
                    "point_count": len(source_reprojection_reports),
                    "tolerance_px": TRACKING_SOURCE_REPROJECTION_TOLERANCE_PX,
                    "all_points_consistent": bool(
                        source_reprojection_ok and source_reprojection_global_ok
                    ),
                    "global_checks": source_reprojection_checks,
                },
                "depth_consistency": {
                    **dict(depth_report),
                    "requested_points_consistent": requested_depth_ok,
                },
                "projected_points": {
                    name: dict(point_reports[name])
                    for name in requested
                    if name in point_reports
                },
            }
            candidate["ok"] = bool(
                candidate["all_points_visible"]
                and candidate["same_observation_sequence"]
                and candidate["expected_sequence_ok"]
                and bool(image_fit.get("ok"))
                and source_reprojection_ok
                and source_reprojection_global_ok
                and requested_depth_ok
            )
            candidates.append(candidate)
            if candidate["ok"]:
                return {
                    "ok": True,
                    "applicable": True,
                    "reason": None,
                    "point_names": requested,
                    "camera_role": role_name,
                    "camera_observation_sequence": camera_sequence,
                    "vertical_semantics": diagnostic.get("vertical_semantics"),
                    "source_reprojection": {
                        "checked": bool(source_reprojection_checks),
                        "point_count": len(source_reprojection_checks),
                        "tolerance_px": TRACKING_SOURCE_REPROJECTION_TOLERANCE_PX,
                        "all_points_consistent": source_reprojection_global_ok,
                        "checks": source_reprojection_checks,
                    },
                    "candidate": candidate,
                    "candidates": candidates,
                }

        applicable = bool(candidates)
        reason = "no_camera_contains_all_points" if not candidates else None
        if candidates:
            failed = candidates[0]
            if not failed["all_points_visible"]:
                reason = "visual_alignment_point_not_visible"
            elif not failed["same_observation_sequence"]:
                reason = "visual_alignment_cross_frame_points"
            elif not failed["expected_sequence_ok"]:
                reason = "visual_alignment_camera_sequence_mismatch"
            elif not (failed["image_collinearity"] or {}).get("ok"):
                reason = "visual_alignment_image_line_fit_failed"
            elif not source_reprojection_global_ok:
                reason = "visual_alignment_source_reprojection_failed"
            elif not (failed["source_reprojection"] or {}).get("all_points_consistent"):
                reason = "visual_alignment_source_reprojection_failed"
            elif not (failed["depth_consistency"] or {}).get(
                "requested_points_consistent"
            ):
                reason = "visual_alignment_rgbd_depth_mismatch"
        return {
            "ok": False,
            "applicable": applicable,
            "reason": reason,
            "point_names": requested,
            "expected_sequence": expected_sequence,
            "vertical_semantics": diagnostic.get("vertical_semantics"),
            "source_reprojection": {
                "checked": bool(source_reprojection_checks),
                "point_count": len(source_reprojection_checks),
                "tolerance_px": TRACKING_SOURCE_REPROJECTION_TOLERANCE_PX,
                "all_points_consistent": source_reprojection_global_ok,
                "checks": source_reprojection_checks,
            },
            "candidates": candidates,
        }

    def _preferred_camera_roles_locked(self) -> list[str]:
        roles = [role for _slot, role, _manager in self._active_source_items_locked()]
        preferred = self._active_camera_role
        if preferred in roles:
            roles.remove(preferred)
            roles.insert(0, preferred)
        return roles

    def _merge_entry_candidates_locked(
        self,
        name: str,
        candidates: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not candidates:
            raise KeyError(name)
        preferred_roles = self._preferred_camera_roles_locked()
        observed_roles = [
            role
            for role, entry in candidates.items()
            if str(entry.get("status") or "") == OBSERVED
        ]
        eligible = observed_roles or list(candidates)
        selected_role = next(
            (role for role in preferred_roles if role in eligible),
            eligible[0],
        )
        selected = deepcopy(dict(candidates[selected_role]))
        source_roles = [role for role in preferred_roles if role in candidates]
        source_roles.extend(role for role in candidates if role not in source_roles)
        selected["tracking_camera_roles"] = source_roles
        selected["observed_camera_roles"] = [
            role for role in source_roles if role in observed_roles
        ]
        selected["selected_camera_role"] = selected_role
        selected["multicamera_source_count"] = len(source_roles)
        if len(source_roles) > 1:
            selected["source_observations"] = {
                role: deepcopy(dict(candidates[role])) for role in source_roles
            }
            points: list[np.ndarray] = []
            for role in observed_roles:
                try:
                    point = np.asarray(
                        candidates[role].get("xyz_in_robot_base_coord_m"),
                        dtype=np.float64,
                    ).reshape(3)
                except (TypeError, ValueError):
                    continue
                if np.all(np.isfinite(point)):
                    points.append(point)
            spread = 0.0
            for left in range(len(points)):
                for right in range(left + 1, len(points)):
                    spread = max(
                        spread,
                        float(np.linalg.norm(points[left] - points[right])),
                    )
            selected["multicamera_xyz_spread_m"] = spread
        return selected

    def memory_fields(self) -> dict[str, Any]:
        # Snapshot facade state, then talk to managers without this lock.
        # Holding both locks here inverts replace_points' publish order.
        with self._lock:
            sources = list(self._active_source_items_locked())
            active_camera_role = self._active_camera_role
            hand_eye = deepcopy(self._hand_eye_diagnostics)
        if sources:
            source_fields = {
                role: self._merged_memory_fields(manager, role)
                for _slot, role, manager in sources
            }
            role = (
                active_camera_role
                if active_camera_role in source_fields
                else sources[-1][1]
            )
            fields = deepcopy(source_fields[role])
            candidates: dict[str, dict[str, Mapping[str, Any]]] = {}
            for _slot, source_role, _manager in sources:
                for name, entry in (
                    source_fields[source_role]
                    .get(TRACKED_OBJECT_DISTANCES_MEMORY_KEY, {})
                    .items()
                ):
                    candidates.setdefault(str(name), {})[source_role] = entry
            with self._lock:
                fields[TRACKED_OBJECT_DISTANCES_MEMORY_KEY] = {
                    name: self._merge_entry_candidates_locked(name, entries)
                    for name, entries in candidates.items()
                }
            policy = fields.setdefault(TRACKING_POLICY_MEMORY_KEY, {})
            active_slots = {
                slot: {
                    "camera_role": source_role,
                    "names": list(
                        source_fields[source_role].get(
                            TRACKED_OBJECT_DISTANCES_MEMORY_KEY, {}
                        )
                    ),
                }
                for slot, source_role, _manager in sources
            }
            policy.update(
                {
                    "camera": role,
                    "camera_role": role,
                    "active_camera_roles": [
                        source_role for _slot, source_role, _manager in sources
                    ],
                    "active_tracking_slots": active_slots,
                    "replacement_scope": ("one_head_group_plus_one_shared_wrist_group"),
                    "same_name_source_policy": (
                        "prefer_observed_latest_registration_then_fallback"
                    ),
                    "wrist_hand_eye_diagnostics": hand_eye,
                }
            )
            return fields

        role = self._selected_role()
        manager = self._selected_manager()
        manager_status = manager.status()
        # A capture binds the official tracker before any named point is
        # selected. Expose official policy metadata during that window.
        if (
            int(manager_status.get("capture_bindings", 0) or 0) > 0
            or int(manager_status.get("replay_history_frames", 0) or 0) > 0
        ):
            return self._merged_memory_fields(manager, role)
        return self._legacy.memory_fields()

    def result(self, session_id: str, image_id: str) -> dict[str, Any]:
        with self._lock:
            matched = next(
                (
                    (slot, deepcopy(active))
                    for slot, active in self._active_results_by_slot.items()
                    if active["session_id"] == str(session_id)
                    and active["image_id"] == str(image_id)
                ),
                None,
            )
        if matched is None:
            return self._legacy.result(session_id, image_id)
        slot, active = matched
        with self._lock:
            role = self._slot_roles.get(slot)
            manager = None if role is None else self._managers.get(role)
            if role is None or manager is None:
                return self._legacy.result(session_id, image_id)
            fields = self._merged_memory_fields(manager, role)[
                TRACKED_OBJECT_DISTANCES_MEMORY_KEY
            ]
        active["tracked_object_distances"] = deepcopy(fields)
        active["tracked_names"] = list(fields)
        active["points"] = list(deepcopy(fields).values())
        active["camera_role"] = role
        active["tracking_slot"] = slot
        status = manager.status()
        sequence = status.get("observation_sequence")
        if sequence is not None:
            active["observation_sequence"] = int(sequence)
        return active

    def robodojo_env_xyz(
        self,
        session_id: str,
        image_id: str,
        name: str,
    ) -> np.ndarray:
        result = self.result(session_id, image_id)
        entry = result["tracked_object_distances"].get(str(name))
        if not isinstance(entry, dict):
            raise KeyError(f"tracked point {name!r} is unavailable")
        if entry.get("status") != OBSERVED:
            raise ValueError(f"tracked point {name!r} is no longer observed")
        point_base = finite_vector(
            entry.get("xyz_in_robot_base_coord_m"),
            3,
            f"tracked {name} robot-base XYZ",
        )
        return robot_base_to_robodojo_env(point_base, self._base_rotation)

    def all_results(self) -> list[dict[str, Any]]:
        with self._lock:
            active = [
                deepcopy(value) for value in self._active_results_by_slot.values()
            ]
        if not active:
            return self._legacy.all_results()
        return [self.result(item["session_id"], item["image_id"]) for item in active]

    def observed_active_points_snapshot(
        self,
        names: list[str] | tuple[str, ...],
        *,
        episode_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            if not self._manager_registered():
                return self._legacy.observed_active_points_snapshot(
                    names,
                    episode_id=episode_id,
                )
            requested, groups, missing = self._registered_source_groups_locked(names)
            reports: dict[str, dict[str, Any]] = {}
            candidates: dict[str, dict[str, Mapping[str, Any]]] = {
                name: {} for name in requested
            }
            for _slot, role, manager, source_names in groups:
                report = manager.observed_active_points_snapshot(
                    source_names,
                    episode_id=episode_id,
                )
                reports[role] = report
                for name, entry in (report.get("entries") or {}).items():
                    candidates[str(name)][role] = entry

            entries = {
                name: self._merge_entry_candidates_locked(name, candidates[name])
                for name in requested
                if candidates[name]
            }
            unavailable: dict[str, str] = {
                name: "point_not_registered_or_retired" for name in missing
            }
            for name in requested:
                if name in entries:
                    continue
                reasons = [
                    str(report.get("unavailable", {}).get(name) or report.get("reason"))
                    for report in reports.values()
                    if name in report.get("requested_names", ())
                ]
                unavailable[name] = next(
                    (reason for reason in reasons if reason and reason != "None"),
                    "tracked_point_unavailable",
                )

            sequences = {
                int(report["observation_sequence"])
                for report in reports.values()
                if report.get("observation_sequence") is not None
            }
            synchronized = len(sequences) <= 1
            ok = not unavailable and synchronized
            preferred_roles = self._preferred_camera_roles_locked()
            primary_role = next(
                (role for role in preferred_roles if role in reports),
                next(iter(reports), None),
            )
            primary = reports.get(primary_role, {})
            rigid_pair_reports = {
                role: deepcopy(report["rigid_pair_fusion"])
                for role, report in reports.items()
                if isinstance(report.get("rigid_pair_fusion"), Mapping)
                and bool(report["rigid_pair_fusion"])
            }
            return {
                "ok": ok,
                "reason": (
                    None
                    if ok
                    else "tracker_observation_mismatch"
                    if not synchronized
                    else "tracked_point_unavailable"
                ),
                "entries": entries,
                "unavailable": unavailable,
                "requested_names": requested,
                "observation_sequence": (
                    next(iter(sequences)) if len(sequences) == 1 else None
                ),
                "episode_id": str(episode_id),
                "session_id": primary.get("session_id"),
                "image_id": primary.get("image_id"),
                "frame": "current_robot_base",
                "axes": "x_forward_y_left_z_up",
                "measurement_source": (
                    "current_evaluator_multicamera_rgbd_and_cam_rel_poses"
                    if len(reports) > 1
                    else primary.get("measurement_source")
                ),
                "camera_roles": list(reports),
                "source_bindings": {
                    role: {
                        "session_id": report.get("session_id"),
                        "image_id": report.get("image_id"),
                    }
                    for role, report in reports.items()
                },
                "source_reports": reports,
                **(
                    {"rigid_pair_fusion": next(iter(rigid_pair_reports.values()))}
                    if len(rigid_pair_reports) == 1
                    else {}
                ),
                **(
                    {"rigid_pair_fusion_by_camera": rigid_pair_reports}
                    if len(rigid_pair_reports) > 1
                    else {}
                ),
                "predictions_published": False,
            }

    def registered_points_snapshot(
        self,
        names: list[str] | tuple[str, ...],
        *,
        episode_id: str,
    ) -> dict[str, Any]:
        """Expose immutable registration RGB-D points to move_tracked_point."""

        with self._lock:
            if not self._manager_registered():
                return {
                    "ok": True,
                    "available": False,
                    "reason": "official_registration_snapshot_unavailable",
                    "requested_names": [str(name) for name in names],
                }
            requested, groups, missing = self._registered_source_groups_locked(names)
            reports: dict[str, dict[str, Any]] = {}
            candidates: dict[str, dict[str, Mapping[str, Any]]] = {
                name: {} for name in requested
            }
            for _slot, role, manager, source_names in groups:
                report = manager.registered_points_snapshot(
                    source_names,
                    episode_id=episode_id,
                )
                reports[role] = report
                for name, entry in (report.get("entries") or {}).items():
                    candidates[str(name)][role] = entry
            if missing or any(not candidates[name] for name in requested):
                unavailable = {
                    name: "registration_point_unavailable"
                    for name in requested
                    if not candidates[name]
                }
                return {
                    "ok": False,
                    "available": False,
                    "reason": "registration_point_unavailable",
                    "entries": {},
                    "unavailable": unavailable,
                    "requested_names": requested,
                    "source_reports": reports,
                }

            selected_entries: dict[str, dict[str, Any]] = {}
            selected_roles: list[str] = []
            for name in requested:
                entry = self._merge_entry_candidates_locked(name, candidates[name])
                selected_entries[name] = entry
                selected_roles.append(str(entry["selected_camera_role"]))
            distinct_roles = list(dict.fromkeys(selected_roles))
            if len(distinct_roles) != 1:
                return {
                    "ok": True,
                    "available": False,
                    "reason": "multiple_registration_captures",
                    "entries": selected_entries,
                    "unavailable": {},
                    "requested_names": requested,
                    "camera_roles": distinct_roles,
                    "source_reports": reports,
                    "predictions_published": False,
                }
            role = distinct_roles[0]
            report = reports[role]
            return {
                **report,
                "ok": True,
                "available": True,
                "reason": None,
                "entries": selected_entries,
                "requested_names": requested,
                "camera_role": role,
                "source_reports": reports,
                "predictions_published": False,
            }

    def activate_rigid_points(
        self,
        names: list[str] | tuple[str, ...],
        *,
        episode_id: str,
        max_point_correction_m: float | None = None,
    ) -> dict[str, Any]:
        """Activate per-camera rigid pairs without conflating camera slots."""

        with self._lock:
            requested, groups, missing = self._registered_source_groups_locked(names)
            if not groups:
                # Preserve the pre-multicamera injectable facade used by
                # benchmark tests and local policy adapters.  Those providers
                # advertise names through status()/observed snapshot without
                # populating the official camera-slot managers.
                if len(requested) == 2:
                    instance_pair = self.__dict__.get("activate_rigid_pair")
                    if callable(instance_pair):
                        return instance_pair(requested, episode_id=episode_id)
                    return self._selected_manager().activate_rigid_pair(
                        requested,
                        episode_id=episode_id,
                        **(
                            {}
                            if max_point_correction_m is None
                            else {"max_point_correction_m": max_point_correction_m}
                        ),
                    )
                return {
                    "ok": True,
                    "active": False,
                    "reason": "tracker_has_no_n_point_fusion",
                    "requested_names": requested,
                    "activated_names": [],
                    "all_points_same_rigid_object_policy_assertion": True,
                }
            if missing:
                raise ValueError(
                    "rigid points are not registered: " + ", ".join(missing)
                )
            reports: dict[str, dict[str, Any]] = {}
            activated: list[tuple[TrackedObjectDistanceMemory, str]] = []
            active_roles: set[str] = set()
            try:
                for _slot, role, manager, source_names in groups:
                    if len(source_names) == 2:
                        report = manager.activate_rigid_pair(
                            source_names,
                            episode_id=episode_id,
                            **(
                                {}
                                if max_point_correction_m is None
                                else {"max_point_correction_m": max_point_correction_m}
                            ),
                        )
                        activated.append((manager, role))
                        if bool(report.get("ok")):
                            active_roles.add(role)
                    else:
                        manager.deactivate_rigid_pair(
                            reason="rigid_bundle_not_a_two_point_camera_local_pair"
                        )
                        report = {
                            "ok": True,
                            "active": False,
                            "reason": (
                                "single_camera_source_point"
                                if len(source_names) == 1
                                else "tracker_has_no_n_point_fusion"
                            ),
                            "names": list(source_names),
                        }
                    reports[role] = report
            except Exception:
                for manager, _role in activated:
                    manager.deactivate_rigid_pair(
                        reason="multicamera_rigid_activation_rollback"
                    )
                raise
            aggregate = {
                "ok": all(bool(report.get("ok", True)) for report in reports.values()),
                "active": bool(active_roles),
                "reason": (
                    "camera_local_rigid_pairs"
                    if active_roles
                    else "cross_camera_or_n_point_local_geometry_only"
                ),
                "requested_names": requested,
                "activated_names": requested,
                "per_camera": reports,
                "all_points_same_rigid_object_policy_assertion": True,
            }
            if len(reports) == 1 and len(requested) == 2:
                return {
                    **next(iter(reports.values())),
                    "requested_names": requested,
                    "activated_names": requested,
                    "all_points_same_rigid_object_policy_assertion": True,
                }
            return aggregate

    def activate_rigid_pair(
        self,
        names: list[str] | tuple[str, str],
        *,
        episode_id: str,
        max_point_correction_m: float | None = None,
    ) -> dict[str, Any]:
        requested = [str(name) for name in names]
        if len(requested) != 2:
            raise ValueError("rigid pair requires exactly two names")
        return self.activate_rigid_points(
            requested,
            episode_id=episode_id,
            max_point_correction_m=max_point_correction_m,
        )

    def deactivate_rigid_pair(
        self,
        *,
        reason: str = "rigid_pair_request_scope_ended",
    ) -> dict[str, Any]:
        with self._lock:
            sources = self._active_source_items_locked()
            if not sources:
                return self._selected_manager().deactivate_rigid_pair(reason=reason)
            reports = {
                role: manager.deactivate_rigid_pair(reason=reason)
                for _slot, role, manager in sources
            }
            if len(reports) == 1:
                return next(iter(reports.values()))
            return {
                "ok": all(bool(report.get("ok", True)) for report in reports.values()),
                "active": False,
                "reason": str(reason),
                "per_camera": reports,
            }

    def activate_on_hand_eef_observation_prior(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        if args:
            names = args[0]
            if len(args) > 1:
                raise TypeError("unexpected positional arguments")
        else:
            names = kwargs.pop("names", None)
        if names is None:
            raise TypeError("names are required")
        episode_id = str(kwargs.pop("episode_id"))
        # The caller's binding identifies its primary source.  Each camera
        # manager must instead receive its own immutable registration binding.
        kwargs.pop("session_id", None)
        kwargs.pop("image_id", None)
        arm = kwargs.pop("arm")
        anchors = np.asarray(kwargs.pop("anchors_eef_m"), dtype=np.float64)
        if kwargs:
            raise TypeError("unsupported arguments: " + ", ".join(sorted(kwargs)))

        with self._lock:
            requested, groups, missing = self._registered_source_groups_locked(names)
            if missing:
                raise ValueError(
                    "on-hand EEF prior points are not registered: " + ", ".join(missing)
                )
            if anchors.shape != (len(requested), 3) or not np.all(np.isfinite(anchors)):
                raise ValueError(
                    "on-hand EEF anchors must be a finite point_count x 3 array"
                )
            index_by_name = {name: index for index, name in enumerate(requested)}
            reports: dict[str, dict[str, Any]] = {}
            leases: dict[str, dict[str, Any]] = {}
            try:
                for _slot, role, manager, source_names in groups:
                    registration = manager.registered_points_snapshot(
                        source_names,
                        episode_id=episode_id,
                    )
                    if not bool(registration.get("ok")):
                        raise ValueError(
                            f"{role} registration binding is unavailable: "
                            f"{registration.get('reason') or 'unknown'}"
                        )
                    source_anchors = anchors[
                        [index_by_name[name] for name in source_names]
                    ]
                    report = manager.activate_on_hand_eef_observation_prior(
                        source_names,
                        episode_id=episode_id,
                        session_id=str(registration["session_id"]),
                        image_id=str(registration["image_id"]),
                        arm=arm,
                        anchors_eef_m=source_anchors,
                    )
                    reports[role] = report
                    leases[role] = {
                        "manager": manager,
                        "lease_id": str(report["lease_id"]),
                    }
            except Exception:
                for lease in leases.values():
                    lease["manager"].deactivate_on_hand_eef_observation_prior(
                        lease["lease_id"]
                    )
                raise

            if len(leases) == 1:
                return next(iter(reports.values()))
            self._eef_prior_counter += 1
            lease_id = f"multicamera-eef-prior-{self._eef_prior_counter:08d}"
            self._eef_prior_leases[lease_id] = {
                "names": requested,
                "episode_id": episode_id,
                "leases": leases,
            }
            return {
                "ok": True,
                "active": True,
                "lease_id": lease_id,
                "names": requested,
                "episode_id": episode_id,
                "arm": str(arm),
                "per_camera": reports,
                "tracking_mode": "independent_multicamera_eef_candidate_gates",
                "prediction_values_published_as_observation": False,
            }

    def deactivate_on_hand_eef_observation_prior(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        if args:
            lease_id = args[0]
            if len(args) > 1:
                raise TypeError("unexpected positional arguments")
        else:
            lease_id = kwargs.pop("lease_id", None)
        if kwargs:
            raise TypeError("unsupported arguments: " + ", ".join(sorted(kwargs)))
        token = str(lease_id or "").strip()
        if not token:
            raise ValueError("on-hand EEF prior lease_id is required")
        with self._lock:
            composite = self._eef_prior_leases.pop(token, None)
            if composite is not None:
                reports = {
                    role: lease["manager"].deactivate_on_hand_eef_observation_prior(
                        lease["lease_id"]
                    )
                    for role, lease in composite["leases"].items()
                }
                return {
                    "ok": all(
                        bool(report.get("ok", True)) for report in reports.values()
                    ),
                    "released": any(
                        bool(report.get("released")) for report in reports.values()
                    ),
                    "lease_id": token,
                    "per_camera": reports,
                }
            reports = [
                manager.deactivate_on_hand_eef_observation_prior(token)
                for _slot, _role, manager in self._active_source_items_locked()
            ]
            if not reports:
                return (
                    self._selected_manager().deactivate_on_hand_eef_observation_prior(
                        token
                    )
                )
            released = next(
                (report for report in reports if bool(report.get("released"))),
                None,
            )
            if released is not None:
                return released
            return {
                "ok": False,
                "released": False,
                "lease_id": token,
            }

    def begin_motion_retention(
        self,
        names: list[str] | tuple[str, ...],
        *,
        episode_id: str,
        timeout_s: float,
    ) -> dict[str, Any]:
        with self._lock:
            requested, groups, missing = self._registered_source_groups_locked(names)
            if missing:
                raise ValueError(
                    "motion retention points are not registered: " + ", ".join(missing)
                )
            reports: dict[str, dict[str, Any]] = {}
            leases: dict[str, dict[str, Any]] = {}
            try:
                for _slot, role, manager, source_names in groups:
                    report = manager.begin_motion_retention(
                        source_names,
                        episode_id=episode_id,
                        timeout_s=timeout_s,
                    )
                    reports[role] = report
                    leases[role] = {
                        "manager": manager,
                        "lease_id": str(report["lease_id"]),
                    }
            except Exception:
                for lease in leases.values():
                    lease["manager"].end_motion_retention(lease["lease_id"])
                raise

            if len(leases) == 1:
                return next(iter(reports.values()))
            superseded: list[str] = []
            requested_set = set(requested)
            for existing_id, existing in list(self._motion_retention_leases.items()):
                if (
                    str(existing["episode_id"]) == str(episode_id)
                    and set(existing["names"]) == requested_set
                ):
                    self._motion_retention_leases.pop(existing_id, None)
                    superseded.append(existing_id)
            self._motion_retention_counter += 1
            lease_id = (
                f"multicamera-motion-retention-{self._motion_retention_counter:08d}"
            )
            self._motion_retention_leases[lease_id] = {
                "names": requested,
                "episode_id": str(episode_id),
                "leases": leases,
            }
            return {
                "ok": True,
                "lease_id": lease_id,
                "names": requested,
                "episode_id": str(episode_id),
                "duration_s": float(timeout_s),
                "superseded_lease_ids": superseded,
                "per_camera": reports,
                "coordinates_published_while_unobserved": False,
            }

    def end_motion_retention(self, lease_id: str) -> dict[str, Any]:
        token = str(lease_id or "").strip()
        if not token:
            raise ValueError("motion retention lease_id is required")
        with self._lock:
            composite = self._motion_retention_leases.pop(token, None)
            if composite is not None:
                reports = {
                    role: lease["manager"].end_motion_retention(lease["lease_id"])
                    for role, lease in composite["leases"].items()
                }
                return {
                    "ok": all(
                        bool(report.get("ok", True)) for report in reports.values()
                    ),
                    "released": any(
                        bool(report.get("released")) for report in reports.values()
                    ),
                    "lease_id": token,
                    "per_camera": reports,
                }
            reports = [
                manager.end_motion_retention(token)
                for _slot, _role, manager in self._active_source_items_locked()
            ]
            if not reports:
                return self._selected_manager().end_motion_retention(token)
            released = next(
                (report for report in reports if bool(report.get("released"))),
                None,
            )
            if released is not None:
                return released
            return {"ok": False, "released": False, "lease_id": token}

    def summary_fields(self) -> dict[str, Any]:
        entries = self.memory_fields()[TRACKED_OBJECT_DISTANCES_MEMORY_KEY]
        return {
            "tracked_object_distances": sum(
                entry.get("status") == OBSERVED for entry in entries.values()
            ),
            "tracked_object_distance_names": list(entries),
            "tracked_object_distance_registrations": len(entries),
        }

    def text(self) -> str:
        fields = self.memory_fields()
        entries = fields[TRACKED_OBJECT_DISTANCES_MEMORY_KEY]
        if not entries:
            return "Tracked object distances: none."
        policy = fields.get(TRACKING_POLICY_MEMORY_KEY, {})
        roles = policy.get("active_camera_roles") or [self._selected_role()]
        lines = [
            "Tracked object distances and XYZ in current robot base coord "
            f"(live {', '.join(str(role) for role in roles)} RGB-D):"
        ]
        for name, entry in entries.items():
            if entry.get("fixed_tabletop"):
                x, y, z = entry["xyz_in_robot_base_coord_m"]
                visibility = (
                    "visible" if entry.get("anchor_visible") else "occluded/out of view"
                )
                lines.append(
                    f"- {name}: fixed tabletop anchor ({visibility}), "
                    "xyz_in_robot_base_coord="
                    f"({x:.4f}, {y:.4f}, {z:.4f}) m"
                )
            elif entry.get("status") == OBSERVED:
                x, y, z = entry["xyz_in_robot_base_coord_m"]
                lines.append(
                    f"- {name}: depth={entry['depth_m']:.4f} m, "
                    f"uv=({entry['u']:.1f}, {entry['v']:.1f}), "
                    "xyz_in_robot_base_coord="
                    f"({x:.4f}, {y:.4f}, {z:.4f}) m"
                )
            else:
                lines.append(
                    f"- {name}: depth and xyz_in_robot_base_coord unavailable "
                    "(temporarily unobserved)"
                )
        return "\n".join(lines)

    def status(self) -> dict[str, Any]:
        with self._lock:
            sources = list(self._active_source_items_locked())
            active_camera_role = self._active_camera_role
        if not sources:
            return self._legacy.status()
        statuses = {role: manager.status() for _slot, role, manager in sources}
        if not any(bool(status.get("registered")) for status in statuses.values()):
            return self._legacy.status()
        role = active_camera_role if active_camera_role in statuses else sources[-1][1]
        result = deepcopy(statuses[role])
        registered_names = list(
            dict.fromkeys(
                str(name)
                for _slot, source_role, _manager in sources
                for name in statuses[source_role].get("registered_names") or []
            )
        )
        active_pairs = {
            source_role: status.get("active_rigid_pair")
            for source_role, status in statuses.items()
            if isinstance(status.get("active_rigid_pair"), Mapping)
        }
        if len(active_pairs) == 1:
            result["active_rigid_pair"] = deepcopy(next(iter(active_pairs.values())))
        elif active_pairs:
            result["active_rigid_pair"] = {
                "names": list(
                    dict.fromkeys(
                        str(name)
                        for pair in active_pairs.values()
                        for name in pair.get("names", ())
                    )
                ),
                "per_camera": deepcopy(active_pairs),
            }
        else:
            result["active_rigid_pair"] = None
        memory_entries = self.memory_fields().get(
            TRACKED_OBJECT_DISTANCES_MEMORY_KEY, {}
        )
        sequences = {
            int(status["observation_sequence"])
            for status in statuses.values()
            if status.get("observation_sequence") is not None
        }
        episodes = {
            str(status["episode_id"])
            for status in statuses.values()
            if status.get("episode_id") is not None
        }
        retained_names = sorted(
            {
                str(name)
                for status in statuses.values()
                for name in status.get("motion_retention", {}).get("retained_names", ())
            }
        )
        result.update(
            {
                "persistent_tracking": True,
                "camera_role": role,
                "active_camera_roles": list(statuses),
                "active_tracking_slots": {
                    slot: {
                        "camera_role": source_role,
                        "registered_names": list(
                            statuses[source_role].get("registered_names") or []
                        ),
                    }
                    for slot, source_role, _manager in sources
                },
                "replacement_scope": ("one_head_group_plus_one_shared_wrist_group"),
                "supported_camera_roles": list(TRACKING_CAMERA_ROLES),
                "registered": len(registered_names),
                "registered_names": registered_names,
                "active": sum(
                    entry.get("status") == OBSERVED for entry in memory_entries.values()
                ),
                "names": list(memory_entries),
                "observation_sequence": (
                    next(iter(sequences)) if len(sequences) == 1 else None
                ),
                "episode_id": (next(iter(episodes)) if len(episodes) == 1 else None),
                "motion_retention": {
                    "active_leases": sum(
                        int(
                            status.get("motion_retention", {}).get("active_leases", 0)
                            or 0
                        )
                        for status in statuses.values()
                    ),
                    "retained_names": retained_names,
                    "coordinates_published_while_unobserved": False,
                },
                "per_camera": deepcopy(statuses),
            }
        )
        with self._lock:
            result["camera_errors"] = deepcopy(self._camera_errors)
            result["wrist_hand_eye"] = deepcopy(self._hand_eye_diagnostics)
            if self._last_error is not None:
                result["last_error"] = self._last_error
        return result
