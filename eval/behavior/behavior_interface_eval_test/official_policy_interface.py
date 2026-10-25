"""Official evaluator policy endpoint plus the unchanged Behavior Interface UI.

The process has no simulator handle. It accepts the v3.9.1 evaluator's
msgpack websocket protocol, keeps only challenge-allowed observations, updates
the existing browser UI, and returns either a queued Human/tool action, an
optional downstream Model Policy Server action, or an observation-derived hold
action.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import importlib
import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from behavior_interface_eval_test.rollout_budget import (
    ROLLOUT_BUDGET_KEY, sanitize_budget, unavailable as unavailable_budget,
    install_rollout_budget_routes,
)

from behavior_interface.gpu_diag import (
    apply_requested_cpu_affinity,
    enforce_fixed_gpu_environment,
    resource_ownership_snapshot,
)

# This endpoint owns no simulator, but it performs image conversion and
# geometry beside the evaluator.  Set numerical-library defaults before
# importing NumPy/OpenCV so inherited host-wide values cannot multiply across
# all policy interfaces.  Explicit operator values remain authoritative.
for _thread_env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_name, "1")

import cv2
import numpy as np
import requests
import websockets
import websockets.asyncio.server as websocket_server
import websockets.sync.client
from flask import Flask, Response, has_request_context, jsonify, request

try:
    _opencv_threads = int(os.environ.get("BEHAVIOR_OPENCV_THREADS", "1"))
except (TypeError, ValueError):
    _opencv_threads = 1
cv2.setNumThreads(max(1, min(_opencv_threads, 16)))


def _gpu_mapping_status() -> dict[str, Any]:
    """Expose the process-local GPU contract for runtime verification."""
    return resource_ownership_snapshot()


def _capture_artifact_nice_target() -> int:
    """Return the validated target used by detached capture workers."""

    try:
        target = int(
            os.environ.get("BEHAVIOR_OFFICIAL_CAPTURE_NICE", "19")
        )
    except (TypeError, ValueError):
        target = 19
    return target if 0 <= target <= 19 else 19


def _set_background_thread_nice(
    env_name: str,
    *,
    default: int = 19,
) -> None:
    """Lower a presentation worker without changing the evaluator thread.

    Linux/NPTL stores nice values per native thread.  Keep this helper
    monotonic so a deployment override can lower a worker's priority, but can
    never accidentally promote it above the priority inherited at startup.
    """

    try:
        target = int(os.environ.get(env_name, str(default)))
    except (TypeError, ValueError):
        target = int(default)
    target = max(0, min(19, target))
    try:
        getpriority = getattr(os, "getpriority", None)
        if callable(getpriority):
            current = int(getpriority(getattr(os, "PRIO_PROCESS", 0), 0))
            if target <= current:
                return
            os.nice(target - current)
            return
        os.nice(target)
    except (AttributeError, OSError, TypeError, ValueError):
        # Priority is an optimization only; an unsupported platform must keep
        # the existing rendering behavior.
        pass


def _presentation_worker_initializer() -> None:
    """Run HUD/map export below the evaluator scheduling priority."""

    _set_background_thread_nice(
        "BEHAVIOR_OFFICIAL_PRESENTATION_NICE",
        default=19,
    )


def _capture_artifact_status(world: Any) -> dict[str, Any]:
    """Expose capture scheduling without touching simulator state."""

    async_enabled = bool(
        getattr(world, "_official_async_capture_artifacts", False)
    )
    detached_enabled = bool(
        getattr(world, "_official_detached_capture_results", False)
    )
    process_enabled = os.environ.get(
        "BEHAVIOR_OFFICIAL_CAPTURE_PROCESS", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    pending: int | None = None
    resolver_available = False
    try:
        tools = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.tools"
        )
        pending_fn = getattr(tools, "capture_artifact_pending_count", None)
        if callable(pending_fn):
            pending = int(pending_fn())
        resolver_available = callable(
            getattr(tools, "wait_for_capture_artifact", None)
        )
    except Exception:
        # Health reporting must remain available even while a transactional
        # tool reload is between module generations.
        resolver_available = False
    return {
        "async": async_enabled,
        "detached": detached_enabled,
        "process_worker_enabled": process_enabled,
        "worker_policy": (
            f"spawn_process_nice{_capture_artifact_nice_target()}_cv2_threads1"
        )
        if process_enabled
        else "bounded_thread",
        "worker_nice_target": _capture_artifact_nice_target(),
        "resolver_available": resolver_available,
        "pending": pending,
    }

from behavior_interface_eval_test.official_action_world import (
    ACTION_DIM,
    ACTION_SLICES,
    ARM_DOF,
    PROPRIO_DIM,
    PROPRIO_SLICES,
    ROBOT_MODEL,
    ROBOT_PROFILE,
    ObservationBackedActionWorld,
)
from behavior_interface_eval_test.episode_initialization import (
    EpisodeGraspPrepController,
)
from behavior_interface_eval_test.official_bddl_progress import (
    UI_BDDL_PROGRESS_KEY,
    awaiting_bddl_progress,
    decode_bddl_progress,
)
from behavior_interface_eval_test.navigation_map_bridge import (
    NAVIGATION_MAP_SCHEMA,
    NAVIGATION_MAP_SCHEMA_VERSION,
    NavigationMapBridge,
    copy_navigation_map_snapshot,
    normalize_navigation_pose_freshness,
    normalize_navigation_pose_source,
    view_navigation_map_snapshot,
)
from behavior_interface_eval_test.tool.official_v2.task_memory import (
    install_official_task_memory,
)
from behavior_interface_eval_test.tool.official_v2.human_track_object_distance_ui import (
    install_track_object_distance_human_ui,
)
from behavior_interface_eval_test.tool.official_v2.contract import (
    MOVE_TRACKED_POINT_ORDER_DESCRIPTION,
    validate_move_tracked_point_args,
    validate_navigate_to_args,
)
from behavior_interface_eval_test.tool.official_v2.tracked_object_distance import (
    TrackedObjectDistanceMemory,
    tracker_rgb_as_uint8,
    tracker_frame_from_allowed_observation,
)
from behavior_interface_eval_test.robot_contract import enforce_locked_action
from behavior_interface_eval_test.official_protocol import Packer, packb, unpackb
from behavior_interface_eval_test.tool.official_v2 import (
    PUBLIC_TOOLS,
    TOOL_VERSION as OFFICIAL_TOOL_VERSION,
    WRIST_ROLL_TEST_TOOL_ENABLED,
    capability_report,
    ensure_profile_installed,
    install_profile,
    translate_submission,
    validate_submission,
)
from behavior_interface_eval_test.tool.official_v2.tools import (
    ADJUST_EEF_LOCAL_BUILD,
    ADJUST_HEIGHT_IMPLEMENTATION_VERSION,
    CONTROL_HZ,
    GRIPPER_CLOSE_DEFAULT_TIMEOUT_S,
    MOVE_TO_REACH_PRE_LIFT_IMPLEMENTATION_VERSION,
    RESET_BODY_IMPLEMENTATION_VERSION,
    _camera_intrinsics,
    _json_ready,
    grasp_prep_hard_timeout_s,
)
from behavior_interface_eval_test.tool.official_v2.visualization_local import (
    render_head_path_overlay_frame,
)


_ALLOWED_EXACT_KEYS = {"task_id", "need_new_action"}
_ALLOWED_SUFFIXES = (
    "::rgb",
    "::depth_linear",
    "::proprio",
    "::cam_rel_poses",
)
_DEFAULT_OBSERVATION_MAX_AGE_S = 5.0
_FAILURE_EXIT_CAPTURE_TIMEOUT_S = 120.0
_CAPTURE_NONE = "none"
_CAPTURE_HEAD_PATH = "head_path"
_CAPTURE_LEFT_WRIST_GRASP = "left_wrist_grasp"
_CAPTURE_RIGHT_WRIST_GRASP = "right_wrist_grasp"
_CAPTURE_REQUEST_ARM_WRIST_GRASP = "request_arm_wrist_grasp"
_CAPTURE_RESULT_ARM_WRIST_GRASP = "result_arm_wrist_grasp"
_CAPTURE_PLAN_GRIPPER = "plan_gripper"
_LIVE_OVERLAY_RENDER_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="official-live-head-hud",
    initializer=_presentation_worker_initializer,
)
_SPATIAL_MAP_EXPORT_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="official-navigation-map-export",
    initializer=_presentation_worker_initializer,
)

_PLAN_PREVIEW_TOOLS = frozenset(
    {
        "plan_eef_translation_to_uvd_point",
        "adjust_plan_pose",
        "plan_grasp_point_filter",
        "plan_grasp_point_filter_rgbd",
        "plan_grasp_point_filter_rgbd_lite",
        "plan_press_point",
    }
)

# This is the model-visible media contract, not a simulator-side camera policy.
# Every capture named here is implemented by the test package and consumes the
# evaluator observation stream.  Keeping the table exhaustive makes a newly
# exposed public tool fail review instead of silently inheriting a head view.
OFFICIAL_V2_CAPTURE_POLICIES: dict[str, dict[str, str]] = {
    "capture_head_camera": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "capture_left_wrist_camera": {
        "success": _CAPTURE_LEFT_WRIST_GRASP,
        "failure": _CAPTURE_LEFT_WRIST_GRASP,
    },
    "capture_right_wrist_camera": {
        "success": _CAPTURE_RIGHT_WRIST_GRASP,
        "failure": _CAPTURE_RIGHT_WRIST_GRASP,
    },
    "read_depth": {"success": _CAPTURE_NONE, "failure": _CAPTURE_NONE},
    "track_object_distance": {
        "success": _CAPTURE_NONE,
        "failure": _CAPTURE_NONE,
    },
    "cut_object": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "move_chassis_to_floor_point": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "move_chassis_to_directly_facing_surface": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "adjust_chassis": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "navigate_to": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "adjust_pitch": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "adjust_height": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "spin_to_facing_point": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "open_gripper": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "close_gripper": {
        "success": _CAPTURE_REQUEST_ARM_WRIST_GRASP,
        "failure": _CAPTURE_REQUEST_ARM_WRIST_GRASP,
    },
    "adjust_left_eef_pose_in_head_frame": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "adjust_right_eef_pose_in_head_frame": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "adjust_left_eef_pose_in_wrist_frame": {
        "success": _CAPTURE_LEFT_WRIST_GRASP,
        "failure": _CAPTURE_LEFT_WRIST_GRASP,
    },
    "adjust_right_eef_pose_in_wrist_frame": {
        "success": _CAPTURE_RIGHT_WRIST_GRASP,
        "failure": _CAPTURE_RIGHT_WRIST_GRASP,
    },
    "move_point_to_point": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "move_tracked_point": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "plan_eef_translation_to_uvd_point": {
        "success": _CAPTURE_PLAN_GRIPPER,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "adjust_plan_pose": {
        "success": _CAPTURE_PLAN_GRIPPER,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "move_to_reach_point": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "measure_shoulder_distance": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "plan_grasp_point_filter": {
        "success": _CAPTURE_PLAN_GRIPPER,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "plan_grasp_point_filter_rgbd": {
        "success": _CAPTURE_PLAN_GRIPPER,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "plan_grasp_point_filter_rgbd_lite": {
        "success": _CAPTURE_PLAN_GRIPPER,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "plan_press_point": {
        "success": _CAPTURE_PLAN_GRIPPER,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "exec_plan_pose": {
        "success": _CAPTURE_RESULT_ARM_WRIST_GRASP,
        "failure": _CAPTURE_RESULT_ARM_WRIST_GRASP,
    },
    "set_arm_to_grasp_position": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
    "reset_body": {
        "success": _CAPTURE_HEAD_PATH,
        "failure": _CAPTURE_HEAD_PATH,
    },
}
if WRIST_ROLL_TEST_TOOL_ENABLED:
    OFFICIAL_V2_CAPTURE_POLICIES["control_wrist_roll"] = {
        "success": _CAPTURE_NONE,
        "failure": _CAPTURE_NONE,
    }
if set(OFFICIAL_V2_CAPTURE_POLICIES) != set(PUBLIC_TOOLS):
    raise RuntimeError("official_v2 capture policy must cover every public tool")

_OFFICIAL_BOUNDARY_ERROR_MARKERS = (
    f"{OFFICIAL_TOOL_VERSION} blocks ",
    "not in the public Interface v2 tool surface",
    "no public official_v2 equivalent",
    "no public official_v2 tool mapping",
    "not in the public official_v2 surface",
)
_INITIALIZATION_GATED_TOOLS = frozenset(
    {
        "adjust_left_eef_pose_in_head_frame",
        "adjust_right_eef_pose_in_head_frame",
        "adjust_left_eef_pose_in_wrist_frame",
        "adjust_right_eef_pose_in_wrist_frame",
        "cut_object",
        "move_point_to_point",
        "move_tracked_point",
        "adjust_plan_pose",
        "exec_plan_pose",
        "plan_eef_translation_to_uvd_point",
        "plan_grasp_point_filter",
        "plan_grasp_point_filter_rgbd",
        "plan_grasp_point_filter_rgbd_lite",
        "plan_press_point",
    }
)


def observation_key_allowed(key: str) -> bool:
    key = str(key)
    return key in _ALLOWED_EXACT_KEYS or key.endswith(_ALLOWED_SUFFIXES)


def filter_allowed_observation(obs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    allowed = {}
    rejected = []
    for key, value in obs.items():
        if observation_key_allowed(str(key)):
            allowed[str(key)] = value
        else:
            rejected.append(str(key))
    return allowed, sorted(rejected)


def _array_has_immutable_owner(value: np.ndarray) -> bool:
    """Return whether a read-only NumPy view is backed by immutable bytes.

    The official msgpack decoder constructs observations with
    ``np.ndarray(buffer=<bytes>, ...)``.  Such arrays are already immutable and
    copying every RGB-D plane before publishing the snapshot only burns CPU and
    memory bandwidth.  Read-only views with any other owner remain defensive
    copies because their owner could still be mutated outside this process.
    """

    if value.flags.writeable:
        return False
    owner: Any = value
    seen: set[int] = set()
    while isinstance(owner, np.ndarray):
        owner_id = id(owner)
        if owner_id in seen:
            return False
        seen.add(owner_id)
        owner = owner.base
    if isinstance(owner, bytes):
        return True
    return isinstance(owner, memoryview) and owner.readonly


def _copy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Defensively copy an observation while borrowing decoder-owned planes."""

    copied: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray) and _array_has_immutable_owner(value):
            copied[str(key)] = value
        else:
            # Keep the historical deepcopy behavior for writable arrays,
            # nested containers, tensors, and scalar extension types.
            copied[str(key)] = deepcopy(value)
    return copied


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb_to_bgr(value: Any) -> Optional[np.ndarray]:
    arr = _as_numpy(value)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4) or arr.size == 0:
        return None
    try:
        canonical_rgb = tracker_rgb_as_uint8(arr, "rgb")
    except ValueError:
        return None
    return cv2.cvtColor(canonical_rgb, cv2.COLOR_RGB2BGR)


def _camera_role(key: str) -> Optional[str]:
    lowered = key.lower()
    if "left_realsense" in lowered or "left_wrist" in lowered:
        return "left_wrist"
    if "right_realsense" in lowered or "right_wrist" in lowered:
        return "right_wrist"
    if "zed_link" in lowered or "head" in lowered:
        return "head"
    return None


@dataclass
class ObservationSnapshot:
    observation: dict[str, Any]
    rejected_keys: list[str]
    received_ts: float
    sequence: int


class ObservationActionAdapter:
    """Challenge-observation store and action-layout adapter."""

    # ``camera_frame`` performs RGB canonicalization into a new OpenCV array,
    # and ``camera_depth_frame`` returns a new depth copy. Capture workers may
    # therefore retain these arrays until their detached artifact completes;
    # callers that provide a different adapter must opt out and keep a copy.
    _official_camera_arrays_owned = True

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot = ObservationSnapshot({}, [], 0.0, 0)
        self._rollout_budget = unavailable_budget('awaiting_evaluator_observation')
        self._navigation_map: Optional[dict[str, Any]] = None
        self._navigation_map_episode_id: Optional[str] = None
        self._ui_goal_progress = awaiting_bddl_progress()
        self._ui_goal_received_ts = 0.0
        self._final_action_overlay: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._final_action_overlay_owns_locked_joints = False
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        for arm in ("left", "right"):
            gripper_slice = ACTION_SLICES[f"gripper_{arm}"]
            if gripper_slice.stop - gripper_slice.start == 1:
                self._last_action[gripper_slice] = 1.0

    def update(
        self,
        obs: dict[str, Any],
        *,
        copy_snapshot: bool = True,
    ) -> ObservationSnapshot:
        policy_obs = {
            str(key): value
            for key, value in obs.items()
            if str(key) not in {UI_BDDL_PROGRESS_KEY, ROLLOUT_BUDGET_KEY}
        }
        filtered, rejected = filter_allowed_observation(policy_obs)
        ui_goal_progress = None
        if UI_BDDL_PROGRESS_KEY in obs:
            try:
                ui_goal_progress = decode_bddl_progress(
                    obs[UI_BDDL_PROGRESS_KEY]
                )
            except Exception as exc:
                ui_goal_progress = awaiting_bddl_progress(
                    "invalid evaluator BDDL status: "
                    f"{type(exc).__name__}: {exc}"
                )
        with self._lock:
            now = time.time()
            # Replace on EVERY observation so a reset/old evaluator cannot
            # inherit another rollout's last counters.
            self._rollout_budget = sanitize_budget(obs.get(ROLLOUT_BUDGET_KEY))
            self._snapshot = ObservationSnapshot(
                observation=_copy_observation(filtered),
                rejected_keys=rejected,
                received_ts=now,
                sequence=self._snapshot.sequence + 1,
            )
            if ui_goal_progress is not None:
                self._ui_goal_progress = ui_goal_progress
                self._ui_goal_received_ts = now
            if copy_snapshot:
                return self.snapshot()
            # The policy hot path only reads this snapshot before the next
            # update. The stored observation already owns one deep copy, so a
            # second full RGB-D copy would add latency without adding safety.
            return ObservationSnapshot(
                observation=self._snapshot.observation,
                rejected_keys=list(self._snapshot.rejected_keys),
                received_ts=self._snapshot.received_ts,
                sequence=self._snapshot.sequence,
            )

    def reset(self) -> None:
        with self._lock:
            self._rollout_budget = unavailable_budget('awaiting_evaluator_observation')
            self._snapshot = ObservationSnapshot(
                observation={},
                rejected_keys=[],
                received_ts=0.0,
                sequence=self._snapshot.sequence,
            )
            self._ui_goal_progress = awaiting_bddl_progress(
                "waiting for evaluator BDDL status after reset"
            )
            self._ui_goal_received_ts = 0.0
            self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
            for arm in ("left", "right"):
                gripper_slice = ACTION_SLICES[f"gripper_{arm}"]
                if gripper_slice.stop - gripper_slice.start == 1:
                    self._last_action[gripper_slice] = 1.0
            self._navigation_map = None
            self._navigation_map_episode_id = None

    def rollout_budget(self) -> dict[str, Any]:
        with self._lock:
            return {**self._rollout_budget,
                    'observation_sequence': self._snapshot.sequence,
                    'observation_age_s': max(0.0, time.time() - self._snapshot.received_ts)
                    if self._snapshot.received_ts else None}

    def snapshot(self) -> ObservationSnapshot:
        with self._lock:
            return ObservationSnapshot(
                observation=_copy_observation(self._snapshot.observation),
                rejected_keys=list(self._snapshot.rejected_keys),
                received_ts=self._snapshot.received_ts,
                sequence=self._snapshot.sequence,
            )

    def borrow_snapshot_for_reading(self) -> ObservationSnapshot:
        """Borrow the current adapter-owned snapshot for internal read-only work.

        Updates replace the complete stored snapshot instead of mutating it, so
        an in-flight renderer can safely retain this reference while the next
        evaluator observation is published.  Public/tool callers continue to
        use ``snapshot()`` and receive an independent defensive copy.
        """

        with self._lock:
            snapshot = self._snapshot
            return ObservationSnapshot(
                observation=snapshot.observation,
                rejected_keys=list(snapshot.rejected_keys),
                received_ts=snapshot.received_ts,
                sequence=snapshot.sequence,
            )

    def observation_metadata(self) -> tuple[int, float]:
        with self._lock:
            return self._snapshot.sequence, self._snapshot.received_ts

    @staticmethod
    def _adopt_frozen_navigation_map(
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Adopt bridge-owned arrays while copying all mutable containers."""

        payload = dict(snapshot)
        occupancy = payload.get("occupancy")
        layers = payload.get("layers")
        frozen = (
            payload.get("schema") == NAVIGATION_MAP_SCHEMA
            and payload.get("schema_version") == NAVIGATION_MAP_SCHEMA_VERSION
            and isinstance(occupancy, np.ndarray)
            and not occupancy.flags.writeable
            and isinstance(layers, Mapping)
            and all(
                isinstance(value, np.ndarray) and not value.flags.writeable
                for value in layers.values()
            )
        )
        if not frozen:
            copied = copy_navigation_map_snapshot(payload)
            if copied is None:  # pragma: no cover - guarded by the argument
                raise ValueError("navigation map snapshot is required")
            return copied
        payload.update(normalize_navigation_pose_freshness(payload))
        payload.update(normalize_navigation_pose_source(payload))
        if not payload["pose_source_sequence_known"]:
            payload.pop("pose_source_observation_sequence", None)
        payload["origin"] = list(payload.get("origin") or ())
        payload["shape"] = list(payload.get("shape") or ())
        payload["pose"] = dict(payload.get("pose") or {})
        payload["places"] = [
            dict(item) for item in list(payload.get("places") or ())
        ]
        payload["lifecycle"] = deepcopy(
            dict(payload.get("lifecycle") or {})
        )
        payload["lifecycle"]["pose_source_sequence_known"] = payload[
            "pose_source_sequence_known"
        ]
        payload["lifecycle"]["source_lag_known"] = payload[
            "pose_source_sequence_known"
        ]
        if payload.get("policy_local_pose") is not None:
            payload["policy_local_pose"] = dict(
                payload["policy_local_pose"]
            )
        if payload.get("source_frame") is not None:
            payload["source_frame"] = deepcopy(
                dict(payload["source_frame"])
            )
        payload["layers"] = dict(layers)
        return payload

    def publish_navigation_map_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        take_ownership: bool = False,
    ) -> None:
        """Atomically publish one backend-neutral navigation map.

        Normal callers get a full defensive copy.  The policy map bridge may
        set ``take_ownership`` because its arrays are already owned and marked
        read-only; mutable dictionaries and lists are still copied here.
        """

        if take_ownership:
            owned = self._adopt_frozen_navigation_map(snapshot)
        else:
            owned = copy_navigation_map_snapshot(snapshot)
            if owned is None:  # pragma: no cover - guarded by the argument
                raise ValueError("navigation map snapshot is required")
        with self._lock:
            episode_id = str(owned.get("episode_id") or "")
            if (
                self._navigation_map_episode_id is not None
                and episode_id != self._navigation_map_episode_id
            ):
                raise ValueError(
                    "navigation map episode does not match the active episode"
                )
            self._navigation_map_episode_id = episode_id
            self._navigation_map = owned

    def begin_navigation_map_episode(self, episode_id: str) -> bool:
        """Clear stale map state before publishing a different episode."""

        value = str(episode_id or "").strip()
        if not value:
            raise ValueError("navigation map episode_id is required")
        with self._lock:
            changed = (
                self._navigation_map_episode_id is not None
                and self._navigation_map_episode_id != value
            )
            if changed:
                self._navigation_map = None
            self._navigation_map_episode_id = value
            return changed

    def clear_navigation_map_snapshot(self) -> None:
        with self._lock:
            self._navigation_map = None

    def navigation_map_snapshot(self) -> Optional[dict[str, Any]]:
        """Return defensive containers and immutable zero-copy grid views."""

        with self._lock:
            return view_navigation_map_snapshot(self._navigation_map)

    def spatial_map_snapshot(self) -> Optional[dict[str, Any]]:
        """Compatibility alias for diagnostics using the older map name."""

        return self.navigation_map_snapshot()

    def ui_goal_progress(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._ui_goal_progress)

    def ui_goal_status(self) -> dict[str, Any]:
        with self._lock:
            payload = self._ui_goal_progress
            received_ts = self._ui_goal_received_ts
            return {
                "available": bool(received_ts),
                "received_ts": received_ts,
                "age_s": (
                    None
                    if not received_ts
                    else round(time.time() - received_ts, 3)
                ),
                "ok": bool(payload.get("ok", False)),
                "satisfied": int(payload.get("satisfied", 0)),
                "total": int(payload.get("total", 0)),
                "complete": bool(payload.get("complete", False)),
                "error": str(payload.get("error", "")),
                "policy_visible": False,
            }

    def observation_is_fresh(self, max_age_s: float) -> bool:
        with self._lock:
            sequence = self._snapshot.sequence
            received_ts = self._snapshot.received_ts
        return (
            sequence > 0
            and received_ts > 0.0
            and time.time() - received_ts <= float(max_age_s)
        )

    def _proprio(self) -> Optional[np.ndarray]:
        with self._lock:
            for key, value in self._snapshot.observation.items():
                if key.endswith("::proprio"):
                    arr = _as_numpy(value).astype(np.float32, copy=False).reshape(-1)
                    if arr.size >= PROPRIO_DIM:
                        return arr
        return None

    def proprio_vector(self) -> Optional[np.ndarray]:
        proprio = self._proprio()
        return None if proprio is None else proprio.copy()

    def hold_action(self) -> np.ndarray:
        """Capture observed arm/trunk positions and command zero base velocity."""
        with self._lock:
            action = self._last_action.copy()
        action[ACTION_SLICES["base"]] = 0.0
        proprio = self._proprio()
        if proprio is not None:
            action[ACTION_SLICES["trunk"]] = proprio[PROPRIO_SLICES["trunk_qpos"]]
            action[ACTION_SLICES["arm_left"]] = proprio[PROPRIO_SLICES["arm_left_qpos"]]
            action[ACTION_SLICES["arm_right"]] = proprio[PROPRIO_SLICES["arm_right_qpos"]]
        return enforce_locked_action(
            action.astype(np.float32, copy=False),
            copy=False,
        )

    def hold_last_action(self) -> np.ndarray:
        """Repeat the last validated limb targets with zero base velocity."""
        with self._lock:
            action = self._last_action.copy()
        action[ACTION_SLICES["base"]] = 0.0
        return enforce_locked_action(
            action.astype(np.float32, copy=False),
            copy=False,
        )

    def set_final_action_overlay(
        self,
        overlay: Optional[Callable[[np.ndarray], np.ndarray]],
        *,
        owns_locked_joints: bool = False,
    ) -> None:
        with self._lock:
            self._final_action_overlay = overlay
            self._final_action_overlay_owns_locked_joints = bool(
                overlay is not None and owns_locked_joints
            )

    def record_action(self, action: Any) -> np.ndarray:
        arr = _as_numpy(action).astype(np.float32, copy=False).reshape(-1)
        if arr.size != ACTION_DIM:
            raise ValueError(f"official action must have {ACTION_DIM} values, got {arr.size}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("official action contains non-finite values")
        arr = enforce_locked_action(arr, copy=True).astype(np.float32, copy=False)
        with self._lock:
            overlay = self._final_action_overlay
            overlay_owns_locked_joints = (
                self._final_action_overlay_owns_locked_joints
            )
        if overlay is not None:
            arr = np.asarray(overlay(arr), dtype=np.float32).reshape(-1)
            if arr.size != ACTION_DIM:
                raise ValueError(
                    "final action overlay returned "
                    f"{arr.size} values; expected {ACTION_DIM}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError("final action overlay returned non-finite values")
            if not overlay_owns_locked_joints:
                arr = enforce_locked_action(arr, copy=False).astype(
                    np.float32,
                    copy=False,
                )
        with self._lock:
            self._last_action = arr.copy()
        return arr

    def adapt_legacy_action(self, action: Any) -> np.ndarray:
        """Map the legacy split-J8 dry-run layout to the active official profile."""
        arr = _as_numpy(action).astype(np.float32, copy=False).reshape(-1)
        if arr.size == ACTION_DIM:
            return self.record_action(arr)
        if arr.size != 25:
            raise ValueError(f"unsupported Behavior Interface action size: {arr.size}")

        out = self.hold_action()
        out[ACTION_SLICES["base"]] = arr[0:3]
        out[ACTION_SLICES["trunk"]] = arr[3:7]
        out[ACTION_SLICES["arm_left"]][:7] = arr[7:14]
        left_gripper = ACTION_SLICES["gripper_left"]
        out[left_gripper] = (
            arr[14:16]
            if left_gripper.stop - left_gripper.start == 2
            else float(np.mean(arr[14:16]))
        )
        out[ACTION_SLICES["arm_right"]][:7] = arr[16:23]
        right_gripper = ACTION_SLICES["gripper_right"]
        out[right_gripper] = (
            arr[23:25]
            if right_gripper.stop - right_gripper.start == 2
            else float(np.mean(arr[23:25]))
        )
        return self.record_action(out)

    def eef_pose(self) -> dict[str, Any]:
        proprio = self._proprio()
        if proprio is None:
            return {}
        return {
            "left": {
                "pos": proprio[PROPRIO_SLICES["eef_left_pos"]].astype(float).tolist(),
                "quat": proprio[PROPRIO_SLICES["eef_left_quat"]].astype(float).tolist(),
            },
            "right": {
                "pos": proprio[PROPRIO_SLICES["eef_right_pos"]].astype(float).tolist(),
                "quat": proprio[PROPRIO_SLICES["eef_right_quat"]].astype(float).tolist(),
            },
        }

    def camera_depth_frames(self) -> dict[str, np.ndarray]:
        with self._lock:
            items = list(self._snapshot.observation.items())
        frames = {}
        for key, value in items:
            if not key.endswith("::depth_linear"):
                continue
            role = _camera_role(key)
            if role is None:
                continue
            arr = _as_numpy(value).astype(np.float32, copy=False)
            if arr.ndim >= 2 and arr.size:
                frames[role] = arr.squeeze().copy()
        return frames

    def camera_depth_frame(self, role: str) -> Optional[np.ndarray]:
        role = str(role)
        with self._lock:
            value = next(
                (
                    value
                    for key, value in self._snapshot.observation.items()
                    if key.endswith("::depth_linear")
                    and _camera_role(key) == role
                ),
                None,
            )
        if value is None:
            return None
        arr = _as_numpy(value).astype(np.float32, copy=False)
        if arr.ndim < 2 or not arr.size:
            return None
        return arr.squeeze().copy()

    def camera_relative_poses(self) -> dict[str, dict[str, list[float]]]:
        with self._lock:
            values = [
                value
                for key, value in self._snapshot.observation.items()
                if key.endswith("::cam_rel_poses")
            ]
        if not values:
            return {}
        arr = _as_numpy(values[0]).astype(np.float64, copy=False).reshape(-1)
        roles = ("left_wrist", "right_wrist", "head")
        out = {}
        for index, role in enumerate(roles):
            start = 7 * index
            if arr.size < start + 7:
                break
            out[role] = {
                "pos": arr[start : start + 3].astype(float).tolist(),
                "quat": arr[start + 3 : start + 7].astype(float).tolist(),
            }
        return out

    def camera_frames(
        self,
        *,
        roles: Optional[set[str]] = None,
        include_main: bool = True,
    ) -> dict[str, np.ndarray]:
        selected_roles = None if roles is None else {str(role) for role in roles}
        with self._lock:
            items = list(self._snapshot.observation.items())
        frames = {}
        for key, value in items:
            if not key.endswith("::rgb"):
                continue
            role = _camera_role(key)
            if role is None:
                continue
            if selected_roles is not None and role not in selected_roles:
                continue
            frame = _rgb_to_bgr(value)
            if frame is not None:
                frames[role] = frame
        if include_main and "head" in frames:
            frames["main"] = cv2.resize(frames["head"], (960, 540))
        return frames

    def camera_frame(self, role: str) -> Optional[np.ndarray]:
        role = str(role)
        with self._lock:
            value = next(
                (
                    value
                    for key, value in self._snapshot.observation.items()
                    if key.endswith("::rgb") and _camera_role(key) == role
                ),
                None,
            )
        return None if value is None else _rgb_to_bgr(value)

    def status(self) -> dict[str, Any]:
        with self._lock:
            sequence = self._snapshot.sequence
            received_ts = self._snapshot.received_ts
            keys = sorted(self._snapshot.observation)
            rejected_keys = list(self._snapshot.rejected_keys)
        return {
            "sequence": sequence,
            "received_ts": received_ts,
            "age_s": None if not received_ts else round(time.time() - received_ts, 3),
            "keys": keys,
            "rejected_keys": rejected_keys,
            "action_dim": ACTION_DIM,
            "bddl_ui": self.ui_goal_status(),
        }


class DownstreamPolicyClient:
    """Synchronous client for an optional Human/Model Policy Server."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        scheme: str = "ws",
        connect_timeout_s: float = 5.0,
    ) -> None:
        self.uri = f"{scheme}://{host}:{int(port)}"
        self.health_url = f"{'https' if scheme == 'wss' else 'http'}://{host}:{int(port)}/healthz"
        self.connect_timeout_s = float(connect_timeout_s)
        self._lock = threading.RLock()
        self._ws = None
        self.metadata: dict[str, Any] = {}
        self.last_error = ""

    def _connect(self) -> None:
        response = requests.get(self.health_url, timeout=self.connect_timeout_s)
        response.raise_for_status()
        self._ws = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            open_timeout=self.connect_timeout_s,
            ping_interval=60,
            ping_timeout=300,
        )
        self.metadata = unpackb(self._ws.recv(), strict_map_key=False)
        self.last_error = ""

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        with self._lock:
            try:
                if self._ws is None:
                    self._connect()
                self._ws.send(packb(obs))
                response = self._ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(response)
                payload = unpackb(response, strict_map_key=False)
                if "action" not in payload:
                    raise KeyError("downstream response has no 'action'")
                self.last_error = ""
                return _as_numpy(payload["action"]).astype(np.float32, copy=False)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                if self._ws is not None:
                    try:
                        self._ws.close()
                    except Exception:
                        pass
                self._ws = None
                raise

    def reset(self) -> None:
        with self._lock:
            if self._ws is None:
                return
            try:
                self._ws.send(packb({"reset": True}))
            except Exception:
                self._ws = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "configured": True,
                "uri": self.uri,
                "connected": self._ws is not None,
                "metadata": deepcopy(self.metadata),
                "last_error": self.last_error,
            }


class EvaluatorConnectionState:
    """Thread-safe count of official evaluator websocket connections."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._count = 0

    def opened(self) -> None:
        with self._lock:
            self._count += 1

    def closed(self) -> None:
        with self._lock:
            self._count = max(0, self._count - 1)

    def snapshot(self) -> tuple[bool, int]:
        with self._lock:
            return self._count > 0, self._count


class _PipelineTiming:
    """Small thread-safe accumulator for observation/action hot-path timings."""

    def __init__(self, ema_alpha: float = 0.1) -> None:
        self._lock = threading.Lock()
        self._ema_alpha = float(ema_alpha)
        self._steps = 0
        self._phases: dict[str, dict[str, float | int]] = {}

    def reset(self) -> None:
        with self._lock:
            self._steps = 0
            self._phases.clear()

    def record(self, samples_ms: dict[str, float]) -> None:
        with self._lock:
            self._steps += 1
            for name, raw_value in samples_ms.items():
                value = max(0.0, float(raw_value))
                entry = self._phases.get(name)
                if entry is None:
                    self._phases[name] = {
                        "samples": 1,
                        "last_ms": value,
                        "ema_ms": value,
                        "max_ms": value,
                        "total_ms": value,
                    }
                    continue
                entry["samples"] = int(entry["samples"]) + 1
                entry["last_ms"] = value
                entry["ema_ms"] = (
                    (1.0 - self._ema_alpha) * float(entry["ema_ms"])
                    + self._ema_alpha * value
                )
                entry["max_ms"] = max(float(entry["max_ms"]), value)
                entry["total_ms"] = float(entry["total_ms"]) + value

    def status(self) -> dict[str, Any]:
        with self._lock:
            phases = deepcopy(self._phases)
            steps = int(self._steps)
        return {
            "steps": steps,
            "clock": "perf_counter",
            "phases": {
                name: {
                    "samples": int(values["samples"]),
                    "last_ms": round(float(values["last_ms"]), 3),
                    "ema_ms": round(float(values["ema_ms"]), 3),
                    "max_ms": round(float(values["max_ms"]), 3),
                    "mean_ms": round(
                        float(values["total_ms"])
                        / max(1, int(values["samples"])),
                        3,
                    ),
                }
                for name, values in phases.items()
            },
        }


class OfficialEvaluatorDisconnectedError(RuntimeError):
    """Raised when an in-flight tool can no longer receive evaluator frames."""


def _wait_for_skill_result_with_evaluator_liveness(
    wait_for_result: Callable[..., dict[str, Any]],
    connection_snapshot: Callable[[], tuple[bool, int]],
    skill_name: str,
    *,
    timeout_s: float,
    poll_s: float,
    request_id: str | None,
    liveness_interval_s: float = 0.5,
) -> dict[str, Any]:
    """Preserve request-id waiting while detecting a dead evaluator promptly."""

    if request_id is None:
        return wait_for_result(
            skill_name,
            timeout_s=timeout_s,
            poll_s=poll_s,
            request_id=None,
        )

    timeout = float(timeout_s)
    if timeout <= 0.0:
        return wait_for_result(
            skill_name,
            timeout_s=timeout,
            poll_s=poll_s,
            request_id=request_id,
        )

    deadline = time.monotonic() + timeout
    interval = max(0.05, float(liveness_interval_s))
    last_timeout: TimeoutError | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                f"等待 skill {skill_name!r} 超时 ({timeout_s}s)"
            ) from last_timeout
        chunk_timeout = min(remaining, interval)
        chunk_poll = min(max(0.01, float(poll_s)), chunk_timeout)
        try:
            return wait_for_result(
                skill_name,
                timeout_s=chunk_timeout,
                poll_s=chunk_poll,
                request_id=request_id,
            )
        except TimeoutError as exc:
            last_timeout = exc
        connected, connection_count = connection_snapshot()
        if not connected or connection_count <= 0:
            raise OfficialEvaluatorDisconnectedError(
                f"official evaluator disconnected while {skill_name} was running"
            ) from last_timeout


@dataclass(frozen=True)
class _OfficialLiveHeadRenderRequest:
    head: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    camera: Optional[dict[str, Any]]
    robot: Optional[dict[str, Any]]
    sequence: int
    error: Optional[str]


@dataclass(frozen=True)
class _NavigationMapExportRequest:
    """Immutable metadata captured on the evaluator callback thread."""

    mapper: Any
    episode_id: str
    sequence: int
    received_ts: float
    policy_pose: Any
    generation: int


def _prepare_official_live_head_frame(
    snapshot: ObservationSnapshot,
    world: ObservationBackedActionWorld,
) -> _OfficialLiveHeadRenderRequest:
    """Freeze all allowed inputs needed by the asynchronous HUD renderer."""

    head = None
    depth = None
    camera_relative_pose = None
    pose_values = None
    for key, value in snapshot.observation.items():
        role = _camera_role(key)
        if role == "head" and key.endswith("::rgb"):
            head = _rgb_to_bgr(value)
        elif role == "head" and key.endswith("::depth_linear"):
            candidate = _as_numpy(value).astype(np.float32, copy=False)
            if candidate.ndim >= 2 and candidate.size:
                depth = candidate.squeeze().copy()
        elif key.endswith("::cam_rel_poses"):
            pose_values = value

    if pose_values is not None:
        poses = _as_numpy(pose_values).astype(np.float64, copy=False).reshape(-1)
        head_start = 14
        if poses.size >= head_start + 7:
            camera_relative_pose = {
                "pos": poses[head_start : head_start + 3].astype(float).tolist(),
                "quat": poses[head_start + 3 : head_start + 7]
                .astype(float)
                .tolist(),
            }

    if head is None:
        return _OfficialLiveHeadRenderRequest(
            head=None,
            depth=None,
            camera=None,
            robot=None,
            sequence=int(snapshot.sequence),
            error="head RGB is unavailable",
        )
    if depth is None or camera_relative_pose is None:
        return _OfficialLiveHeadRenderRequest(
            head=head,
            depth=None,
            camera=None,
            robot=None,
            sequence=int(snapshot.sequence),
            error=(
                "live v2 base-path HUD requires evaluator depth_linear "
                "and camera-relative pose"
            ),
        )

    try:
        camera_pose = world.local_pose_from_robot_relative(
            camera_relative_pose["pos"],
            camera_relative_pose["quat"],
        )
        height, width = head.shape[:2]
        camera = {
            **camera_pose,
            **_camera_intrinsics("head", width, height),
            "source": "official_evaluator_cam_rel_poses",
        }
        robot = {
            "base_pose": world.robot_pose().as_dict(),
        }
        return _OfficialLiveHeadRenderRequest(
            head=head,
            depth=depth,
            camera=deepcopy(camera),
            robot=deepcopy(robot),
            sequence=int(snapshot.sequence),
            error=None,
        )
    except Exception as exc:
        return _OfficialLiveHeadRenderRequest(
            head=head,
            depth=None,
            camera=None,
            robot=None,
            sequence=int(snapshot.sequence),
            error=f"{type(exc).__name__}: {exc}",
        )


def _render_prepared_official_live_head_frame(
    prepared: _OfficialLiveHeadRenderRequest,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """Render one frozen request without reading mutable runtime state."""

    head = prepared.head
    if head is None or prepared.error is not None:
        return head, {
            "ok": False,
            "overlay_applied": False,
            "sequence": int(prepared.sequence),
            "error": prepared.error or "head RGB is unavailable",
        }
    try:
        output, result = render_head_path_overlay_frame(
            image_bgr=head,
            depth_linear=prepared.depth,
            camera=prepared.camera,
            robot=prepared.robot,
        )
        if output is None or not bool(result.get("ok")):
            stamped = head
            try:
                stamped = _stamp_selected_minimap_on_bgr(head)
            except Exception:
                pass
            return stamped, {
                **result,
                "overlay_applied": False,
                "sequence": int(prepared.sequence),
            }
        try:
            output = _stamp_selected_minimap_on_bgr(output)
        except Exception:
            pass
        return output, {
            **result,
            "overlay_applied": True,
            "sequence": int(prepared.sequence),
        }
    except Exception as exc:
        return head, {
            "ok": False,
            "overlay_applied": False,
            "sequence": int(prepared.sequence),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _stamp_selected_minimap_on_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """Use the same selected mapper for the official head HUD and Web UI."""

    from behavior_interface.rtabmap_slam.live import (
        live_backend_selected,
        stamp_minimap_on_bgr as stamp_rtabmap,
    )

    if live_backend_selected():
        return stamp_rtabmap(image_bgr)
    from behavior_interface.spatial_map import stamp_minimap_on_bgr

    return stamp_minimap_on_bgr(image_bgr)


def _render_official_live_head_frame(
    snapshot: ObservationSnapshot,
    world: ObservationBackedActionWorld,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """Render synchronously for captures/tests from one frozen observation."""

    return _render_prepared_official_live_head_frame(
        _prepare_official_live_head_frame(snapshot, world)
    )


class OfficialPolicyRuntime:
    """Owns the unchanged UI runtime and arbitrates Human/tool vs model actions."""

    def __init__(
        self,
        *,
        task: str,
        scene: str,
        robot_dof: int,
        tool_version: str,
        ui_tool_version: str,
        downstream: Optional[DownstreamPolicyClient],
        rgbd_lite_runtime: Optional[dict[str, Any]] = None,
    ) -> None:
        if int(robot_dof) != ARM_DOF:
            raise ValueError(
                f"robot profile {ROBOT_PROFILE!r} requires arm_dof={ARM_DOF}, "
                f"got {robot_dof}"
            )
        if str(tool_version) != OFFICIAL_TOOL_VERSION:
            raise ValueError(
                f"unsupported strict tool profile {tool_version!r}; "
                f"expected {OFFICIAL_TOOL_VERSION!r}"
            )
        if str(ui_tool_version) != "v2":
            raise ValueError("the existing HTTP/frontend compatibility surface requires ui tool v2")
        self.tool_version = str(tool_version)
        self.ui_tool_version = str(ui_tool_version)
        self.rgbd_lite_runtime = deepcopy(rgbd_lite_runtime or {})
        os.environ["INTERFACE_TOOL_VERSION"] = self.ui_tool_version
        os.environ["BEHAVIOR_ROBOT_DOF"] = str(robot_dof)

        from behavior_interface import server as legacy_server
        import behavior_interface.skills as skills_pkg

        # The coordinator itself is intentionally outside tool/official_v2 so
        # it remains stable while that complete dependency graph is reloaded.
        # These fields are policy-owned process state and survive every
        # successful or rolled-back transaction.
        self._official_reload_lock = threading.RLock()
        self._official_reload_generation = 0
        self._official_reload_receipt = None
        self._official_skills_module = skills_pkg

        self.adapter = ObservationActionAdapter()
        self.evaluator_connections = EvaluatorConnectionState()
        self.observation_max_age_s = float(
            os.environ.get(
                "BEHAVIOR_EVAL_TEST_OBSERVATION_MAX_AGE_S",
                _DEFAULT_OBSERVATION_MAX_AGE_S,
            )
        )
        if self.observation_max_age_s <= 0.0:
            raise ValueError(
                "BEHAVIOR_EVAL_TEST_OBSERVATION_MAX_AGE_S must be positive"
            )
        self.server = legacy_server.BehaviorInterface(
            task=task,
            scene_model=scene,
            robot_dof=robot_dof,
            dry_run=True,
            target_hz=30.0,
        )
        self.server.gripper_control_hz = float(CONTROL_HZ)
        self.server.gripper_close_control_horizon_s = float(
            GRIPPER_CLOSE_DEFAULT_TIMEOUT_S
        )
        with self.server.state_lock:
            self.server._cached_goals = self.adapter.ui_goal_progress()
        self.tool_registry = install_profile(skills_pkg, self.adapter)
        self.server.world = ObservationBackedActionWorld(
            proprio_provider=self.adapter.proprio_vector,
            hold_action_provider=self.adapter.hold_action,
            eef_pose_provider=self.adapter.eef_pose,
            robot_dof=robot_dof,
        )
        # Tool capture keeps observation/provenance work on this thread, while
        # its immutable media artifact is serialized by the v2 worker queue.
        # The detached-result flag lets the evaluator callback release the
        # skill after the frozen frame is queued; the HTTP waiter still returns
        # the exact completed artifact.  Direct/offline worlds do not set these
        # opt-in flags and retain the historical synchronous generator path.
        self.server.world._official_async_capture_artifacts = True
        self.server.world._official_detached_capture_results = True
        self.adapter.set_final_action_overlay(
            self.server.world.enforce_official_action,
            owns_locked_joints=True,
        )
        self.server.world._official_adapter = self.adapter
        self.server.world._gpu_diag_log = self.server.log
        self.tracked_object_distances = TrackedObjectDistanceMemory()
        self.server.world._official_tracked_object_distances = (
            self.tracked_object_distances
        )

        @contextmanager
        def official_request_commit_guard():
            with self.server.skill_lock:
                yield not bool(self.server.cancel_flag)

        self.server.world._official_request_commit_guard = (
            official_request_commit_guard
        )
        self.task_memory = install_official_task_memory(
            self.server,
            task,
            dynamic_memory=self.tracked_object_distances,
        )
        self.episode_initializer = EpisodeGraspPrepController(
            arm_dof=ARM_DOF,
            control_hz=CONTROL_HZ,
        )
        original_submit_skill = self.server.submit_skill
        original_wait_for_skill_result = self.server.wait_for_skill_result
        original_maybe_start_next_skill = self.server._maybe_start_next_skill
        public_jobs: dict[str, str] = {}
        cancelled_public_jobs: set[str] = set()
        public_jobs_lock = threading.RLock()
        public_submit_lock = threading.Lock()
        last_public_request_id = ""
        self._official_public_jobs = public_jobs
        self._official_cancelled_public_jobs = cancelled_public_jobs
        self._official_public_jobs_lock = public_jobs_lock
        self._tool_registry_restore_count = 0

        def ensure_official_tool_registry() -> None:
            restored = ensure_profile_installed(skills_pkg, self.tool_registry)
            if restored:
                self._tool_registry_restore_count += 1
                self.server.log(
                    "OFFICIAL_POLICY restored test-owned tool registry after "
                    "an external module replaced the compatibility registry"
                )

        def strict_maybe_start_next_skill() -> None:
            # The compatibility server resolves a queued job from the package
            # global at execution time. Restore and validate it immediately
            # before that lookup so a lazy production import cannot reroute a
            # public official_v2 name.
            ensure_official_tool_registry()
            original_maybe_start_next_skill()
            current_job = getattr(self.server, "current_job", None)
            current_request_id = str(
                getattr(current_job, "request_id", "") or ""
            )
            with public_jobs_lock:
                cancelled = current_request_id in cancelled_public_jobs
                if cancelled:
                    cancelled_public_jobs.discard(current_request_id)
            if cancelled:
                with self.server.skill_lock:
                    active = getattr(self.server, "current_job", None)
                    if (
                        active is not None
                        and str(getattr(active, "request_id", "") or "")
                        == current_request_id
                    ):
                        self.server.cancel_flag = True

        self._ensure_official_tool_registry = ensure_official_tool_registry
        self.server._maybe_start_next_skill = strict_maybe_start_next_skill

        route_hints = {
            "/api/v2/adjust_hight": "adjust_height",
        }

        def current_public_hint() -> str:
            if not has_request_context():
                return ""
            path = str(request.path or "")
            if path in route_hints:
                return route_hints[path]
            prefix = "/api/v2/"
            if path.startswith(prefix):
                candidate = path[len(prefix):].strip("/")
                if candidate in PUBLIC_TOOLS:
                    return candidate
            return ""

        def strict_submit_skill(name: str, args: dict[str, Any]) -> str:
            nonlocal last_public_request_id
            # Linearize every normal submission against a stack commit. The
            # lock is released as soon as the request is atomically queued;
            # the reloader then observes that queue/current job and returns a
            # conflict instead of publishing half-new callbacks underneath it.
            with self._official_reload_lock:
                if not self.evaluator_ready():
                    raise RuntimeError(
                        "official evaluator observation/action loop is unavailable "
                        "or stale"
                    )
                ensure_official_tool_registry()
                public_name, public_args = translate_submission(
                    name,
                    args,
                    public_hint=current_public_hint(),
                )
                if (
                    public_name in _INITIALIZATION_GATED_TOOLS
                    and not self.server.world.episode_initialized()
                ):
                    raise RuntimeError(
                        "episode grasp-prep initialization is incomplete; "
                        f"{public_name} is blocked until both arms converge"
                    )
                validated_args = validate_submission(public_name, public_args)
                # A normal official tool explicitly taking ownership should
                # release a terminal live-test hold before entering the queue.
                self.release_terminal_live_test()
                with public_submit_lock:
                    while (
                        f"job-{int(time.time() * 1000)}"
                        == last_public_request_id
                    ):
                        time.sleep(0.0002)
                    request_id = original_submit_skill(
                        public_name,
                        validated_args,
                    )
                    last_public_request_id = request_id
                with public_jobs_lock:
                    public_jobs[request_id] = public_name
                return request_id

        def strict_wait_for_skill_result(
            skill_name: str,
            timeout_s: float = 180.0,
            poll_s: float = 0.25,
            request_id: str | None = None,
        ) -> dict[str, Any]:
            public_name = str(skill_name)
            if request_id is not None:
                with public_jobs_lock:
                    public_name = public_jobs.get(request_id, public_name)
            wait_started = time.monotonic()
            try:
                result = _wait_for_skill_result_with_evaluator_liveness(
                    original_wait_for_skill_result,
                    self.evaluator_connections.snapshot,
                    public_name,
                    timeout_s=timeout_s,
                    poll_s=poll_s,
                    request_id=request_id,
                )
                pending_token = str(
                    result.get("_official_capture_pending_token") or ""
                ) if isinstance(result, dict) else ""
                if pending_token:
                    # Resolve through the currently committed tools module so
                    # a successful hot reload cannot strand a token in an old
                    # function-global registry.
                    from behavior_interface_eval_test.tool.official_v2 import (
                        tools as official_v2_tools,
                    )

                    resolver = getattr(
                        official_v2_tools,
                        "wait_for_capture_artifact",
                        None,
                    )
                    if not callable(resolver):
                        raise RuntimeError(
                            "official capture result resolver is unavailable"
                        )
                    remaining = max(
                        0.0,
                        float(timeout_s) - (time.monotonic() - wait_started),
                    )
                    try:
                        final = resolver(
                            pending_token,
                            timeout_s=remaining,
                        )
                    except TimeoutError:
                        # ``current_job`` is intentionally empty after a
                        # detached capture.  Cancelling only the server skill
                        # therefore cannot stop the media worker; cancel the
                        # exact token before propagating the request timeout.
                        cancel_artifact = getattr(
                            official_v2_tools,
                            "cancel_capture_artifact",
                            None,
                        )
                        if callable(cancel_artifact):
                            cancel_artifact(pending_token)
                        raise
                    if isinstance(final, dict):
                        final.pop("_official_capture_pending_token", None)
                        return final
                    return final
                return result
            except OfficialEvaluatorDisconnectedError:
                if request_id is not None:
                    cancel_public_job(request_id)
                raise
            finally:
                if request_id is not None:
                    with public_jobs_lock:
                        public_jobs.pop(request_id, None)

        def cancel_public_job(request_id: str) -> bool:
            """Cancel only the named public request, whether queued or running."""

            target = str(request_id or "").strip()
            if not target:
                return False
            # A detached capture has already released ``current_job``.  Cancel
            # its media future explicitly on timeout/reset so a late worker
            # cannot publish an obsolete image into the next episode.
            pending_token = ""
            try:
                with self.server.state_lock:
                    for capture_name in (
                        "capture",
                        "capture_head_camera",
                        "capture_left_wrist_camera",
                        "capture_right_wrist_camera",
                    ):
                        candidate = self.server._last_skill_results.get(
                            capture_name
                        )
                        if (
                            isinstance(candidate, dict)
                            and candidate.get("job") == target
                        ):
                            pending_token = str(
                                candidate.get(
                                    "_official_capture_pending_token"
                                )
                                or ""
                            )
                            if pending_token:
                                break
            except Exception:
                pending_token = ""
            if pending_token:
                try:
                    from behavior_interface_eval_test.tool.official_v2 import (
                        tools as official_v2_tools,
                    )

                    cancel_artifact = getattr(
                        official_v2_tools,
                        "cancel_capture_artifact",
                        None,
                    )
                    if callable(cancel_artifact):
                        cancel_artifact(pending_token)
                except Exception:
                    pass
            with public_jobs_lock:
                cancelled_public_jobs.add(target)
            with self.server.skill_lock:
                current_job = getattr(self.server, "current_job", None)
                if (
                    current_job is not None
                    and str(getattr(current_job, "request_id", "") or "")
                    == target
                ):
                    if getattr(current_job, "result", None) is not None:
                        with public_jobs_lock:
                            cancelled_public_jobs.discard(target)
                        return False
                    self.server.cancel_flag = True
                    with public_jobs_lock:
                        cancelled_public_jobs.discard(target)
                    return True
            return True

        def acknowledge_public_job_terminal(request_id: str) -> None:
            target = str(request_id or "").strip()
            if not target:
                return
            with public_jobs_lock:
                cancelled_public_jobs.discard(target)
                public_jobs.pop(target, None)

        self.server.submit_skill = strict_submit_skill
        self.server._official_reload_submit_barrier = True
        self.server.wait_for_skill_result = strict_wait_for_skill_result
        self.cancel_public_job = cancel_public_job
        self.acknowledge_public_job_terminal = acknowledge_public_job_terminal
        self.downstream = downstream
        self._step_lock = threading.RLock()
        # Development-only live unit-test runner.  This is intentionally
        # separate from the official skill queue: it reads only the adapter's
        # evaluator observation and returns one official action per evaluator
        # callback.  The module is reloaded for each new run so source edits
        # take effect without restarting this process.
        self._live_test_lock = threading.RLock()
        self._live_test_job = None
        self._live_test_module_name = (
            "behavior_interface_eval_test.live_move_tracked_point_test"
        )
        self._live_test_module_build = None
        self._pipeline_timing = _PipelineTiming()
        self._last_obs_mono = None
        self._fps_ema = 0.0
        self._source = "hold"
        self._last_error = ""
        self._live_overlay_lock = threading.RLock()
        self._live_overlay_generation = 0
        self._live_overlay_future: Future[
            tuple[Optional[np.ndarray], dict[str, Any]]
        ] | None = None
        self._live_overlay_inflight_sequence = -1
        self._live_overlay_pending: tuple[
            int,
            _OfficialLiveHeadRenderRequest,
            float,
        ] | None = None
        self._live_overlay_sequence = -1
        self._live_overlay_updated_ts = 0.0
        self._live_overlay_raw_head: Optional[np.ndarray] = None
        self._live_overlay_frames: dict[str, np.ndarray] = {}
        self._live_overlay_preview_sequence = -1
        self._live_overlay_preview_updated_ts = 0.0
        self._live_overlay_preview_head: Optional[np.ndarray] = None
        self._live_overlay_preview_frames: dict[str, np.ndarray] = {}
        self._live_overlay_dropped_requests = 0
        self._live_overlay_status: dict[str, Any] = {
            "ok": False,
            "overlay_applied": False,
            "sequence": 0,
            "error": "no official evaluator observation received",
            "render_policy": "async_latest_only",
        }
        # RTAB-Map accepts frames on its own worker, but converting its latest
        # result into the canonical navigation-map contract can still take
        # hundreds of milliseconds (and may wait for the mapper state lock).
        # Keep that export off the evaluator callback thread.  The exporter is
        # latest-only because intermediate pose metadata has no consumer once a
        # newer evaluator observation exists.
        self._spatial_map_export_lock = threading.RLock()
        self._spatial_map_export_generation = 0
        self._spatial_map_export_episode: Optional[str] = None
        self._spatial_map_export_future: Future[None] | None = None
        self._spatial_map_export_pending: _NavigationMapExportRequest | None = None
        self._spatial_map_export_inflight_sequence = -1
        self._spatial_map_export_dropped_requests = 0
        self._spatial_map_export_last_error = ""
        self.server.log(
            f"OFFICIAL_POLICY profile={ROBOT_PROFILE} model={ROBOT_MODEL} "
            f"arm_dof={ARM_DOF} action_dim={ACTION_DIM} proprio_dim={PROPRIO_DIM} "
            f"tool={self.tool_version} ui={self.ui_tool_version}: "
            "no simulator handle; evaluator observations are filtered to "
            "RGB/depth/proprio/camera-relative-pose/task-id; BDDL status is "
            "removed into a display-only cache"
        )

        original_snapshot = self.server.snapshot_state

        def snapshot_state():
            payload = original_snapshot()
            payload["mode"] = "official-policy-interface"
            payload["transport"] = self.status()
            try:
                from behavior_interface.operator_controls import eval_control_snapshot

                control = eval_control_snapshot(self.server, official=True)
                if control.get("current_instance_id") is not None:
                    payload["instance_id"] = control["current_instance_id"]
                    self.server.current_instance_id = control["current_instance_id"]
                payload["eval_control"] = control
            except Exception:
                pass
            return payload

        self.server.snapshot_state = snapshot_state

    def evaluator_ready(self) -> bool:
        websocket_connected, _ = self.evaluator_connections.snapshot()
        return websocket_connected and self.adapter.observation_is_fresh(
            self.observation_max_age_s
        )

    # ------------------------------------------------------- dev live tests

    def _live_test_providers(self):
        """Build observation-only callbacks for the development runner.

        The callbacks deliberately expose no ``server.robot``, ``env`` or
        simulator object.  The tracked-point manager is policy-owned memory
        populated from the evaluator RGB-D stream.
        """

        from behavior_interface_eval_test.live_move_tracked_point_test import (
            LiveJobProviders,
        )

        manager = self.tracked_object_distances

        def tracked_points(names):
            return manager.observed_active_points_snapshot(
                list(names),
                episode_id=self.server.world.episode_id(),
            )

        def activate_rigid_pair(names, episode_id):
            return manager.activate_rigid_pair(
                list(names),
                episode_id=str(episode_id),
            )

        def activate_rigid_points(names, episode_id):
            """Use an explicit tracker bundle API when the runtime provides it.

            The stock tracker currently exposes only pair fusion.  Returning a
            policy-owned diagnostic report for larger bundles lets the live
            runner retain all markers in one atomic snapshot and enforce the
            N-point geometry locally without inventing object identity.
            """

            activate_all = getattr(manager, "activate_rigid_points", None)
            if callable(activate_all):
                return activate_all(
                    list(names),
                    episode_id=str(episode_id),
                )
            return {
                "ok": True,
                "active": False,
                "requested_names": [str(name) for name in names],
                "activated_names": [],
                "activation_scope": "n_point_local_geometry_only",
                "all_points_same_rigid_object_policy_assertion": True,
                "reason": (
                    "tracker_has_no_n_point_fusion; all requested points "
                    "are checked atomically"
                ),
            }

        return LiveJobProviders(
            proprio=self.adapter.proprio_vector,
            metadata=self.adapter.observation_metadata,
            episode_id=self.server.world.episode_id,
            tracked_points=tracked_points,
            activate_rigid_pair=activate_rigid_pair,
            max_age_s=self.observation_max_age_s,
            activate_rigid_points=activate_rigid_points,
        )

    def reload_live_test_module(self):
        """Return the harness from the current official-v2 generation.

        Generation zero retains the one-module legacy bootstrap used to
        upgrade an already-running pre-transaction process. Once a full stack
        generation is committed, per-case dev starts reuse it and reject
        uncommitted source changes.
        """

        from behavior_interface_eval_test import official_v2_hot_reload

        return official_v2_hot_reload.reload_live_test_harness(
            self,
            import_module=importlib.import_module,
            reload_module=importlib.reload,
            invalidate_caches=importlib.invalidate_caches,
        )

    def reload_official_v2_tool_stack(self) -> dict[str, Any]:
        """Atomically reload and publish one coherent official-v2 generation."""

        from behavior_interface_eval_test import official_v2_hot_reload

        return official_v2_hot_reload.reload_official_v2_tool_stack(
            self,
            import_module=importlib.import_module,
            reload_module=importlib.reload,
            invalidate_caches=importlib.invalidate_caches,
        )

    def official_v2_reload_status(self) -> dict[str, Any]:
        """Return the last committed official-v2 generation without mutation."""

        from behavior_interface_eval_test import official_v2_hot_reload

        return official_v2_hot_reload.official_v2_reload_status(self)

    def start_live_move_tracked_point_test(self, args: dict[str, Any]) -> dict[str, Any]:
        """Start one dev live test from the current evaluator observation.

        This method never touches a simulator handle.  It is intentionally
        not part of ``submit_skill`` or the public official-v2 registry.
        """

        with self._official_reload_lock:
            return self._start_live_move_tracked_point_test_locked(args)

    def _start_live_move_tracked_point_test_locked(
        self,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.evaluator_ready():
            raise RuntimeError(
                "official evaluator observation/action loop is unavailable or stale"
            )
        if not self.server.world.episode_initialized():
            raise RuntimeError(
                "episode grasp-prep initialization is incomplete; run the normal "
                "episode initialization sequence first"
            )
        with self.server.skill_lock:
            if self.server.current_job is not None or not self.server.skill_queue.empty():
                raise RuntimeError(
                    "a normal skill is running or queued; wait for it to finish"
                )
        with self._live_test_lock:
            if self._live_test_job is not None:
                state = str(self._live_test_job.status().get("state") or "")
                if state not in {"done", "failed", "cancelled"}:
                    raise RuntimeError(
                        "another live unit test is already running: "
                        f"{self._live_test_job.job_id}"
                    )
            module = self.reload_live_test_module()
            job_cls = getattr(module, "LiveMoveTrackedPointJob", None)
            if job_cls is None:
                raise RuntimeError(
                    f"{self._live_test_module_name} has no LiveMoveTrackedPointJob"
                )
            job = job_cls(
                providers=self._live_test_providers(),
                raw_args=dict(args or {}),
                module_build=str(getattr(module, "BUILD", "unknown")),
            )
            self._live_test_job = job
            started = job.start()
            self.server.log(
                "DEV live_move_tracked_point_test started "
                f"job={job.job_id} build={getattr(module, 'BUILD', 'unknown')}"
            )
            return started

    def live_test_action(self) -> np.ndarray | None:
        """Return the next action for the active dev runner, if any."""

        with self._live_test_lock:
            job = self._live_test_job
        if job is None:
            return None
        return job.step()

    def live_test_hold_action(self) -> np.ndarray:
        """Return an observation-backed hold while a live plan is pending.

        ``hold_last_action`` can repeat a target from a completed legacy skill
        while the local planner is still working.  The live test must freeze
        the measured arm/trunk pose instead, so the planning interval cannot
        move the robot or create a false handoff error.
        """

        try:
            with self._live_test_lock:
                job = self._live_test_job
            action = None
            if job is not None:
                build_hold = getattr(job, "hold_action", None)
                if callable(build_hold):
                    action = build_hold()
            if action is None:
                action = self.adapter.hold_action()
            apply_gripper_holds = getattr(
                self.server.world,
                "enforce_tool_gripper_holds",
                None,
            )
            if callable(apply_gripper_holds):
                action = apply_gripper_holds(action)
            return np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            # Keep the official hot path alive if an observation disappears
            # between the live-job read and this fallback.
            return self._stable_hold_action()

    def live_test_status(self) -> dict[str, Any]:
        with self._live_test_lock:
            job = self._live_test_job
            if job is None:
                payload = {
                    "ok": True,
                    "state": "idle",
                    "tool": "live_move_tracked_point_test",
                    "build": self._live_test_module_build,
                }
            else:
                payload = job.status()
        # This compatibility field lets an already-running legacy HTTP route
        # expose the transaction receipt after its one-time bootstrap.
        payload["reload_receipt"] = self.official_v2_reload_status()
        return payload

    def cancel_live_test(self, reason: str = "cancelled by user") -> dict[str, Any]:
        with self._official_reload_lock:
            with self._live_test_lock:
                job = self._live_test_job
            if job is not None:
                job.cancel(reason)
            return self.live_test_status()

    def release_terminal_live_test(self) -> None:
        """Let an explicitly queued normal skill reclaim the action stream."""

        with self._official_reload_lock:
            with self._live_test_lock:
                job = self._live_test_job
                if job is None:
                    return
                state = str(job.status().get("state") or "")
                if state in {"done", "failed", "cancelled"}:
                    self._live_test_job = None

    def reset(self) -> None:
        with self._official_reload_lock:
            self._reset_locked()

    def _reset_locked(self) -> None:
        with self._step_lock:
            with self._live_test_lock:
                live_job = self._live_test_job
                self._live_test_job = None
            if live_job is not None:
                live_job.cancel("evaluator reset")
            self._pipeline_timing.reset()
            self.adapter.reset()
            # A detached capture has already released the server's current
            # skill slot.  Cancel its media future explicitly so a worker from
            # the previous episode cannot install a late result after reset.
            try:
                from behavior_interface_eval_test.tool.official_v2 import (
                    tools as official_v2_tools,
                )

                cancel_all = getattr(
                    official_v2_tools,
                    "cancel_all_capture_artifacts",
                    None,
                )
                if callable(cancel_all):
                    cancel_all()
            except Exception as exc:
                self.server.log(
                    "OFFICIAL_POLICY capture reset cleanup failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            self.server.cancel_current_skill()
            public_jobs_lock = getattr(
                self,
                "_official_public_jobs_lock",
                None,
            )
            if public_jobs_lock is not None:
                with public_jobs_lock:
                    self._official_public_jobs.clear()
                    self._official_cancelled_public_jobs.clear()
            self.server.world.reset_observation_state()
            tracked_object_distances = getattr(
                self,
                "tracked_object_distances",
                None,
            )
            if tracked_object_distances is not None:
                tracked_object_distances.reset()
            self.episode_initializer.reset()
            with self.server.state_lock:
                self.server._cached_goals = self.adapter.ui_goal_progress()
            if self.downstream is not None:
                self.downstream.reset()
            self._source = "hold"
            self._last_error = ""
            with self._live_overlay_lock:
                self._live_overlay_generation += 1
                if self._live_overlay_future is not None:
                    self._live_overlay_future.cancel()
                self._live_overlay_future = None
                self._live_overlay_inflight_sequence = -1
                self._live_overlay_pending = None
                self._live_overlay_sequence = -1
                self._live_overlay_updated_ts = 0.0
                self._live_overlay_raw_head = None
                self._live_overlay_frames.clear()
                self._live_overlay_preview_sequence = -1
                self._live_overlay_preview_updated_ts = 0.0
                self._live_overlay_preview_head = None
                self._live_overlay_preview_frames.clear()
                self._live_overlay_dropped_requests = 0
                self._live_overlay_status = {
                    "ok": False,
                    "overlay_applied": False,
                    "sequence": 0,
                    "error": "official evaluator reset",
                    "render_policy": "async_latest_only",
                }
            # A worker may still be inside bridge.capture(); generation
            # invalidation makes its eventual result harmless.  Do not wait
            # here: reset must retain the evaluator callback's bounded time.
            with self._spatial_map_export_lock:
                self._spatial_map_export_generation += 1
                if self._spatial_map_export_future is not None:
                    self._spatial_map_export_future.cancel()
                self._spatial_map_export_future = None
                self._spatial_map_export_pending = None
                self._spatial_map_export_inflight_sequence = -1
                self._spatial_map_export_episode = None
                self._spatial_map_export_last_error = ""
            self._reset_spatial_map()
            self.server.log("OFFICIAL_POLICY reset received from evaluator")

    def _reset_spatial_map(self) -> None:
        """新 episode 从零建图：机器人被瞬移回起点，旧栅格和轨迹全部作废。"""
        adapter = getattr(self, "adapter", None)
        clear_snapshot = getattr(
            adapter,
            "clear_navigation_map_snapshot",
            None,
        )
        if callable(clear_snapshot):
            clear_snapshot()
        bridge = getattr(self, "_navigation_map_bridge", None)
        if bridge is not None:
            bridge.reset()
        try:
            from behavior_interface.rtabmap_slam.live import (
                live_backend_selected,
                reset_live_mapper,
            )

            if live_backend_selected():
                reset_live_mapper()
                self._spatial_live_mapper = None
                return
        except Exception as exc:
            self.server.log(f"OFFICIAL_POLICY RTAB-Map reset failed: {exc}")
        mapper = getattr(self, "_spatial_live_mapper", None)
        if mapper in (None, False):
            return
        try:
            from behavior_interface.spatial_map import reset_all_maps

            reset_all_maps()
            mapper.reset()
        except Exception as exc:
            self.server.log(f"OFFICIAL_POLICY spatial map reset failed: {exc}")

    def _update_ui(self) -> None:
        now_mono = time.monotonic()
        if self._last_obs_mono is not None:
            fps = 1.0 / max(now_mono - self._last_obs_mono, 1e-6)
            self._fps_ema = fps if self._fps_ema <= 0 else 0.9 * self._fps_ema + 0.1 * fps
        self._last_obs_mono = now_mono

        get_feed_active = getattr(self.server, "get_feed_active", None)
        if callable(get_feed_active):
            active_feeds = get_feed_active()
            # Official head/main routes render the sequence-cached HUD directly
            # from the adapter snapshot.  Rebuilding undecorated head/main here
            # would duplicate RGB conversion and resize work on every action.
            wrist_roles = {
                role
                for role in ("left_wrist", "right_wrist")
                if bool(active_feeds.get(role, False))
            }
            frames = self.adapter.camera_frames(
                roles=wrist_roles,
                include_main=False,
            )
        else:
            # Preserve the compatibility behavior used by lightweight tests and
            # any host without the v2 feed-activation surface.
            frames = self.adapter.camera_frames()
        if frames:
            self.server._update_frames(self.server._decorate(frames))
        self.server.tick += 1
        self.server.fps = self._fps_ema
        with self.server.state_lock:
            self.server._cached_eef_pose = {
                arm: self.server.world.eef_pose(arm=arm)
                for arm in ("left", "right")
            }
            self.server._cached_goals = self.adapter.ui_goal_progress()

    def _update_tracked_object_distances(
        self,
        snapshot: ObservationSnapshot,
    ) -> None:
        manager = getattr(self, "tracked_object_distances", None)
        if manager is None:
            return
        try:
            frame = tracker_frame_from_allowed_observation(
                snapshot.observation,
                sequence=snapshot.sequence,
                timestamp_s=snapshot.received_ts,
                episode_id=self.server.world.episode_id(),
                base_xy_yaw=self.server.world.policy_local_base_pose(),
                copy_arrays=False,
            )
        except (TypeError, ValueError) as exc:
            manager.note_observation_error(exc)
            return
        adapter = getattr(self, "adapter", None)
        # ``ObservationActionAdapter.update(copy_snapshot=False)`` hands us
        # one immutable-by-replacement snapshot.  The official memory manager
        # can retain those arrays directly; lightweight test adapters and
        # legacy managers continue through their defensive-copy path.
        borrowed_ingest = (
            getattr(adapter, "_official_camera_arrays_owned", False) is True
            and getattr(manager, "_supports_borrowed_ingest", False) is True
        )
        if borrowed_ingest:
            manager.ingest(frame, copy_frame=False)
        else:
            manager.ingest(frame)

    def _live_overlay_feed_locked(
        self,
        raw_head: np.ndarray,
        feed: str,
        cache: dict[str, np.ndarray],
    ) -> np.ndarray:
        cached = cache.get(feed)
        if cached is not None:
            return cached
        source = (
            np.asarray(raw_head).copy()
            if feed == "head"
            else cv2.resize(
                raw_head,
                (self.server.main_w, self.server.main_h),
            )
        )
        decorated = self.server._decorate({feed: source})
        frame = decorated.get(feed, source)
        cache[feed] = frame
        return frame

    def _available_live_overlay_frame_locked(
        self,
        feed: str,
    ) -> tuple[Optional[np.ndarray], int, float]:
        if self._live_overlay_raw_head is not None:
            return (
                self._live_overlay_feed_locked(
                    self._live_overlay_raw_head,
                    feed,
                    self._live_overlay_frames,
                ),
                int(self._live_overlay_sequence),
                float(self._live_overlay_updated_ts),
            )
        if self._live_overlay_preview_head is not None:
            return (
                self._live_overlay_feed_locked(
                    self._live_overlay_preview_head,
                    feed,
                    self._live_overlay_preview_frames,
                ),
                -(max(0, int(self._live_overlay_preview_sequence)) + 1),
                float(self._live_overlay_preview_updated_ts),
            )
        return None, -1, 0.0

    def _start_live_overlay_render_locked(
        self,
        generation: int,
        prepared: _OfficialLiveHeadRenderRequest,
        updated_ts: float,
    ) -> None:
        started = time.monotonic()
        future = _LIVE_OVERLAY_RENDER_EXECUTOR.submit(
            _render_prepared_official_live_head_frame,
            prepared,
        )
        self._live_overlay_future = future
        self._live_overlay_inflight_sequence = int(prepared.sequence)

        def completed(
            finished: Future[tuple[Optional[np.ndarray], dict[str, Any]]],
        ) -> None:
            try:
                head, status = finished.result()
            except Exception as exc:
                head = prepared.head
                status = {
                    "ok": False,
                    "overlay_applied": False,
                    "sequence": int(prepared.sequence),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            render_ms = round((time.monotonic() - started) * 1000.0, 3)
            with self._live_overlay_lock:
                if (
                    generation != self._live_overlay_generation
                    or self._live_overlay_future is not finished
                ):
                    return
                self._live_overlay_future = None
                self._live_overlay_inflight_sequence = -1
                if int(prepared.sequence) >= self._live_overlay_sequence:
                    self._live_overlay_raw_head = head
                    self._live_overlay_frames.clear()
                    self._live_overlay_sequence = int(prepared.sequence)
                    self._live_overlay_updated_ts = float(updated_ts)
                    self._live_overlay_status = {
                        **status,
                        "render_ms": render_ms,
                        "render_policy": "async_latest_only",
                    }
                pending = self._live_overlay_pending
                self._live_overlay_pending = None
                if (
                    pending is not None
                    and pending[0] == self._live_overlay_generation
                    and pending[1].sequence > self._live_overlay_sequence
                ):
                    self._start_live_overlay_render_locked(*pending)

        future.add_done_callback(completed)

    def live_display_frame(
        self,
        feed: str,
    ) -> tuple[Optional[np.ndarray], int, float]:
        """Return the newest completed HUD while rendering only the latest request."""
        feed = str(feed)
        if feed not in {"head", "main"}:
            with self.server.frame_lock:
                return (
                    self.server.frames.get(feed),
                    int(
                        getattr(self.server, "frame_ids", {}).get(
                            feed,
                            self.server.frame_id,
                        )
                    ),
                    float(
                        getattr(self.server, "frame_updated_ts", {}).get(
                            feed,
                            0.0,
                        )
                    ),
                )

        sequence, updated_ts = self.adapter.observation_metadata()
        with self._live_overlay_lock:
            if self._live_overlay_sequence == sequence:
                return self._available_live_overlay_frame_locked(feed)
            pending_sequence = (
                -1
                if self._live_overlay_pending is None
                else int(self._live_overlay_pending[1].sequence)
            )
            if sequence in {
                self._live_overlay_inflight_sequence,
                pending_sequence,
            }:
                return self._available_live_overlay_frame_locked(feed)

        snapshot = self.adapter.borrow_snapshot_for_reading()
        sequence = int(snapshot.sequence)
        updated_ts = float(snapshot.received_ts)
        prepared = _prepare_official_live_head_frame(
            snapshot,
            self.server.world,
        )
        with self._live_overlay_lock:
            if self._live_overlay_sequence == sequence:
                return self._available_live_overlay_frame_locked(feed)
            self._live_overlay_preview_sequence = sequence
            self._live_overlay_preview_updated_ts = updated_ts
            self._live_overlay_preview_head = prepared.head
            self._live_overlay_preview_frames.clear()
            generation = self._live_overlay_generation
            if self._live_overlay_future is None:
                self._start_live_overlay_render_locked(
                    generation,
                    prepared,
                    updated_ts,
                )
            elif sequence > self._live_overlay_inflight_sequence:
                if self._live_overlay_pending is not None:
                    self._live_overlay_dropped_requests += 1
                self._live_overlay_pending = (
                    generation,
                    prepared,
                    updated_ts,
                )
            return self._available_live_overlay_frame_locked(feed)

    def _tool_action(self) -> Optional[np.ndarray]:
        # While a live test is planning or executing, it owns the evaluator
        # action stream.  Do not start a queued legacy skill in the meantime;
        # doing so would make the next observation belong to two controllers.
        with self._live_test_lock:
            live_job = self._live_test_job
            live_state = (
                None
                if live_job is None
                else str(live_job.status().get("state") or "")
            )
        if live_state not in {None, "done", "failed", "cancelled"}:
            live_action = self.live_test_action()
            if live_action is not None:
                return live_action
            with self._live_test_lock:
                live_job = self._live_test_job
                live_state = (
                    None
                    if live_job is None
                    else str(live_job.status().get("state") or "")
                )
            if live_state not in {None, "done", "failed", "cancelled"}:
                # Planning is asynchronous.  Return a complete measured hold
                # action so a configured downstream policy cannot move the
                # robot while the frozen local plan is being built.
                return self.live_test_hold_action()

        if live_state in {"failed", "cancelled"}:
            # Keep a failed/cancelled experiment on the measured safe pose.  A
            # queued normal official skill is an explicit ownership handoff;
            # release the terminal record in that case so it can proceed.
            normal_job_queued = False
            try:
                normal_job_queued = bool(
                    self.server.current_job is not None
                    or not self.server.skill_queue.empty()
                )
            except Exception:
                normal_job_queued = False
            if normal_job_queued:
                self.release_terminal_live_test()
            else:
                with self._live_test_lock:
                    terminal_job = self._live_test_job
                build_hold = getattr(terminal_job, "terminal_hold_action", None)
                if callable(build_hold):
                    terminal_action = build_hold()
                    if terminal_action is not None:
                        return terminal_action

        self.server._maybe_start_next_skill()
        had_job = self.server.current_job is not None
        action = self.server._tick_skill()
        if action is None and not had_job:
            return None
        if action is None:
            return self._stable_hold_action()
        return self.adapter.adapt_legacy_action(action)

    def _stable_hold_action(self) -> np.ndarray:
        """Hold the last action target without feeding tracking error back in."""
        action = self.adapter.hold_last_action()
        apply_gripper_holds = getattr(
            self.server.world,
            "enforce_tool_gripper_holds",
            None,
        )
        if callable(apply_gripper_holds):
            action = apply_gripper_holds(action)
        return action

    def _continuous_capture_tick(self, observation_dt: float) -> None:
        """点 record 后把本拍官方观测写进可离线重放的 capture bundle。

        必须挂在 ``act()`` 里。严格官方模式不跑 BehaviorServer 的物理循环，
        ``_spatial_odom_tick`` 永远不会被调用；evaluator 每推一帧观测就进
        这里一次，``observation_dt`` 是仿真控制步长 ``1/CONTROL_HZ``。
        """
        world = self.server.world
        if world is None:
            return
        try:
            from behavior_interface.continuous_capture import CAPTURE

            CAPTURE.on_tick(world, float(observation_dt))
            CAPTURE.on_frame(world)
        except Exception:
            pass

    def _spatial_map_frame(self, observation_dt: float) -> None:
        """测试口实时建图：每帧观测积一次里程，按自适应频率并入一帧 depth。

        只用 evaluator 推过来的 head depth_linear、cam_rel_poses 和 proprio
        里的 base_qvel，全部是官方允许的观测。任何异常都自我禁用，绝不影响
        动作回送。
        """
        mapper = getattr(self, "_spatial_live_mapper", None)
        if mapper is False:
            adapter = getattr(self, "adapter", None)
            clear_snapshot = getattr(
                adapter,
                "clear_navigation_map_snapshot",
                None,
            )
            if callable(clear_snapshot):
                clear_snapshot()
            return
        if mapper is None:
            try:
                from behavior_interface.spatial_map import spatial_map_enabled

                if not spatial_map_enabled():
                    self._spatial_live_mapper = False
                    adapter = getattr(self, "adapter", None)
                    clear_snapshot = getattr(
                        adapter,
                        "clear_navigation_map_snapshot",
                        None,
                    )
                    if callable(clear_snapshot):
                        clear_snapshot()
                    return
                from behavior_interface.rtabmap_slam.live import (
                    get_live_mapper,
                    live_backend_selected,
                )

                if live_backend_selected():
                    mapper = get_live_mapper()
                else:
                    from behavior_interface.spatial_map_live import LiveMapper

                    mapper = LiveMapper()
            except Exception as exc:
                self.server.log(f"OFFICIAL_POLICY spatial map unavailable: {exc}")
                self._spatial_live_mapper = False
                adapter = getattr(self, "adapter", None)
                clear_snapshot = getattr(
                    adapter,
                    "clear_navigation_map_snapshot",
                    None,
                )
                if callable(clear_snapshot):
                    clear_snapshot()
                return
            self._spatial_live_mapper = mapper
        if mapper.disabled:
            self._spatial_live_mapper = False
            adapter = getattr(self, "adapter", None)
            clear_snapshot = getattr(
                adapter,
                "clear_navigation_map_snapshot",
                None,
            )
            if callable(clear_snapshot):
                clear_snapshot()
            self.server.log(
                f"OFFICIAL_POLICY spatial map disabled: {mapper.last_error}"
            )
            return
        world = self.server.world
        mapper.odom_tick(world, observation_dt)
        mapper.map_tick(world)
        # The compatibility bridge may wait on RTAB-Map's state lock and copy
        # large grids/trails.  On a live evaluator connection this is scheduled
        # latest-only so the callback can proceed to action selection.  Offline
        # unit callers retain the historical synchronous behavior.
        if self._navigation_map_async_enabled():
            self._schedule_navigation_map_snapshot(mapper)
        else:
            self._publish_navigation_map_snapshot(mapper)

    def _navigation_map_async_enabled(self) -> bool:
        connection_state = getattr(self, "evaluator_connections", None)
        connected = bool(
            connection_state is not None
            and connection_state.snapshot()[0]
        )
        value = os.environ.get("BEHAVIOR_OFFICIAL_ASYNC_MAP_EXPORT", "1")
        return connected and value.strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }

    def _navigation_map_export_request(
        self,
        mapper: Any,
    ) -> _NavigationMapExportRequest | None:
        """Freeze only callback-owned metadata before submitting export work."""

        adapter = getattr(self, "adapter", None)
        if adapter is None:
            return None
        sequence, received_ts = adapter.observation_metadata()
        world = self.server.world
        episode_provider = getattr(world, "episode_id", None)
        episode_id = (
            str(episode_provider())
            if callable(episode_provider)
            else f"official-policy-{id(world)}"
        )
        policy_pose_provider = getattr(world, "policy_local_base_pose", None)
        policy_pose = (
            deepcopy(policy_pose_provider())
            if callable(policy_pose_provider)
            else None
        )
        bridge = getattr(self, "_navigation_map_bridge", None)
        if bridge is None:
            bridge = NavigationMapBridge()
            self._navigation_map_bridge = bridge
        begin_episode = getattr(adapter, "begin_navigation_map_episode", None)
        if callable(begin_episode) and begin_episode(episode_id):
            # Clear bridge caches on the callback thread before any worker can
            # observe the new episode.  This is a tiny operation; the costly
            # provider capture remains asynchronous.
            bridge.reset()
        with self._spatial_map_export_lock:
            if self._spatial_map_export_episode != episode_id:
                self._spatial_map_export_generation += 1
                self._spatial_map_export_episode = episode_id
                self._spatial_map_export_pending = None
            generation = int(self._spatial_map_export_generation)
        return _NavigationMapExportRequest(
            mapper=mapper,
            episode_id=episode_id,
            sequence=int(sequence),
            received_ts=float(received_ts or time.time()),
            policy_pose=policy_pose,
            generation=generation,
        )

    def _start_navigation_map_export_locked(
        self,
        request: _NavigationMapExportRequest,
    ) -> None:
        future = _SPATIAL_MAP_EXPORT_EXECUTOR.submit(
            self._publish_navigation_map_snapshot,
            request.mapper,
            sequence=request.sequence,
            received_ts=request.received_ts,
            episode_id=request.episode_id,
            policy_pose=request.policy_pose,
            generation=request.generation,
        )
        self._spatial_map_export_future = future
        self._spatial_map_export_inflight_sequence = int(request.sequence)

        def completed(finished: Future[None]) -> None:
            error = ""
            try:
                finished.result()
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                error = f"{type(exc).__name__}: {exc}"
            with self._spatial_map_export_lock:
                if self._spatial_map_export_future is not finished:
                    return
                self._spatial_map_export_future = None
                self._spatial_map_export_inflight_sequence = -1
                if error:
                    self._spatial_map_export_last_error = error
                pending = self._spatial_map_export_pending
                self._spatial_map_export_pending = None
                if (
                    pending is not None
                    and pending.generation == self._spatial_map_export_generation
                ):
                    self._start_navigation_map_export_locked(pending)

        future.add_done_callback(completed)

    def _schedule_navigation_map_snapshot(self, mapper: Any) -> None:
        request = self._navigation_map_export_request(mapper)
        if request is None:
            return
        with self._spatial_map_export_lock:
            if self._spatial_map_export_future is None:
                self._start_navigation_map_export_locked(request)
            else:
                if self._spatial_map_export_pending is not None:
                    self._spatial_map_export_dropped_requests += 1
                self._spatial_map_export_pending = request

    def _publish_navigation_map_snapshot(
        self,
        mapper: Any,
        *,
        sequence: int | None = None,
        received_ts: float | None = None,
        episode_id: str | None = None,
        policy_pose: Any = None,
        generation: int | None = None,
    ) -> None:
        """Publish the selected mapper through the test-owned map contract.

        ``sequence``/episode metadata are optional to keep the old synchronous
        call contract.  A worker supplies frozen values and checks its
        generation both before and after the expensive bridge capture.
        """

        adapter = getattr(self, "adapter", None)
        publish = getattr(adapter, "publish_navigation_map_snapshot", None)
        if not callable(publish):
            return
        try:
            bridge = getattr(self, "_navigation_map_bridge", None)
            if bridge is None:
                bridge = NavigationMapBridge()
                self._navigation_map_bridge = bridge
            world = self.server.world
            if sequence is None or received_ts is None:
                sequence, received_ts = adapter.observation_metadata()
            if episode_id is None:
                episode_provider = getattr(world, "episode_id", None)
                episode_id = (
                    str(episode_provider())
                    if callable(episode_provider)
                    else f"official-policy-{id(world)}"
                )
            if generation is not None:
                with self._spatial_map_export_lock:
                    if (
                        generation != self._spatial_map_export_generation
                        or mapper is not getattr(self, "_spatial_live_mapper", None)
                    ):
                        return
            begin_episode = getattr(
                adapter,
                "begin_navigation_map_episode",
                None,
            )
            if callable(begin_episode) and begin_episode(episode_id):
                bridge.reset()
            if policy_pose is None:
                policy_pose_provider = getattr(
                    world,
                    "policy_local_base_pose",
                    None,
                )
                policy_pose = (
                    policy_pose_provider()
                    if callable(policy_pose_provider)
                    else None
                )
            snapshot = bridge.capture(
                mapper,
                episode_id=episode_id,
                observation_sequence=int(sequence),
                policy_local_pose=policy_pose,
                captured_ts=float(received_ts or time.time()),
            )
            if snapshot is not None:
                if generation is not None:
                    # Serialize the final publication with reset invalidation;
                    # a reset can therefore never be followed by a stale map.
                    with self._spatial_map_export_lock:
                        if (
                            generation != self._spatial_map_export_generation
                            or mapper
                            is not getattr(self, "_spatial_live_mapper", None)
                        ):
                            return
                        publish(snapshot, take_ownership=True)
                else:
                    publish(snapshot, take_ownership=True)
        except Exception as exc:
            # Mapping already follows a fail-open runtime policy.  Keep the
            # previous immutable snapshot on a transient export failure; reset
            # and backend-disable paths above explicitly clear stale state.
            self._spatial_map_export_last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            self.server.log(
                "OFFICIAL_POLICY navigation map snapshot failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def act(self, raw_obs: dict[str, Any]) -> np.ndarray:
        call_started = time.perf_counter()
        with self._step_lock:
            phases_ms = {
                "step_lock_wait": (time.perf_counter() - call_started) * 1000.0,
            }
            action_started: float | None = None
            try:
                # 里程计必须按仿真控制步长积分：一个 action 恒定推进 1/CONTROL_HZ
                # 仿真时间。用墙钟间隔会随渲染帧率（实测 ~6fps）把位移放大数倍，
                # 导致底盘闭环在真实还差十几厘米时就判定「到位」。
                observation_dt = 1.0 / float(CONTROL_HZ)

                phase_started = time.perf_counter()
                snapshot = self.adapter.update(raw_obs, copy_snapshot=False)
                phases_ms["snapshot"] = (
                    time.perf_counter() - phase_started
                ) * 1000.0
                filtered = snapshot.observation

                phase_started = time.perf_counter()
                self.server.world.reconcile_base_odometry_from_proprio(
                    observation_dt
                )
                phases_ms["odometry"] = (
                    time.perf_counter() - phase_started
                ) * 1000.0

                phase_started = time.perf_counter()
                self._update_tracked_object_distances(snapshot)
                phases_ms["tracker"] = (
                    time.perf_counter() - phase_started
                ) * 1000.0

                phase_started = time.perf_counter()
                self._continuous_capture_tick(observation_dt)
                phases_ms["capture"] = (
                    time.perf_counter() - phase_started
                ) * 1000.0

                phase_started = time.perf_counter()
                self._spatial_map_frame(observation_dt)
                phases_ms["spatial_map"] = (
                    time.perf_counter() - phase_started
                ) * 1000.0

                phase_started = time.perf_counter()
                self._update_ui()
                phases_ms["ui"] = (
                    time.perf_counter() - phase_started
                ) * 1000.0

                action_started = time.perf_counter()
                if not self.episode_initializer.ready():
                    try:
                        if self.adapter.proprio_vector() is None:
                            self.episode_initializer.note_missing_proprio()
                            action = self._stable_hold_action()
                        else:
                            action = self.episode_initializer.step(self.server.world)
                        if self.episode_initializer.ready():
                            self.server.world.set_episode_initialized(True)
                            initialization = self.episode_initializer.status()
                            self.server.log(
                                "OFFICIAL_POLICY episode initialization: "
                                f"{initialization['state']}; "
                                f"{initialization.get('warning') or 'grasp-prep converged from evaluator proprioception'}"
                            )
                        self._source = "episode-grasp-prep"
                        self._last_error = ""
                        return self.adapter.record_action(action)
                    except Exception as exc:
                        self._last_error = (
                            "episode grasp-prep failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        self.server.log(f"OFFICIAL_POLICY {self._last_error}")
                        self._source = "episode-grasp-prep-error"
                        return self.adapter.record_action(self._stable_hold_action())

                try:
                    action = self._tool_action()
                    if action is not None:
                        self._source = "human/tool"
                        self._last_error = ""
                        return self.adapter.record_action(action)

                    if self.downstream is not None:
                        action = self.downstream.act(filtered)
                        recorded = self.adapter.record_action(action)
                        clear_gripper_holds = getattr(
                            self.server.world,
                            "clear_tool_gripper_holds",
                            None,
                        )
                        if callable(clear_gripper_holds):
                            clear_gripper_holds()
                        self._source = "downstream-model"
                        self._last_error = ""
                        return recorded
                except Exception as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    self.server.log(
                        f"OFFICIAL_POLICY action fallback: {self._last_error}"
                    )

                self._source = "hold"
                return self.adapter.record_action(self._stable_hold_action())
            finally:
                if action_started is not None:
                    phases_ms["action_select"] = (
                        time.perf_counter() - action_started
                    ) * 1000.0
                phases_ms["total"] = (
                    time.perf_counter() - call_started
                ) * 1000.0
                timing = getattr(self, "_pipeline_timing", None)
                if timing is not None:
                    timing.record(phases_ms)

    def idle_probe(self) -> dict[str, Any]:
        """Return only the state needed by the idle-step sidecar.

        The normal ``status``/``snapshot_state`` contract intentionally
        contains camera, memory, map, replay, and tool-history data for the
        browser. The gate polls much more often than a human does, so feeding
        it that payload makes an otherwise read-only check contend with JSON
        serialization and the tool callback. This method reads the same
        state-machine flags under their existing short-lived locks and never
        touches PhysX, frames, maps, or replay buffers.

        Any unexpected read failure is reported as diagnostic uncertainty.
        The sidecar then follows its existing fail-open path and forwards the
        already-generated action.
        """

        try:
            action_source = str(getattr(self, "_source", "") or "")
            last_error = str(getattr(self, "_last_error", "") or "")
            initialization_ready = bool(self.episode_initializer.ready())
            downstream_configured = self.downstream is not None

            with self._live_test_lock:
                live_job = self._live_test_job
                if live_job is None:
                    live_state = "idle"
                else:
                    job_lock = getattr(live_job, "_lock", None)
                    if job_lock is None:
                        live_state = str(
                            getattr(live_job, "_state", "unknown") or ""
                        )
                    else:
                        with job_lock:
                            live_state = str(
                                getattr(live_job, "_state", "unknown") or ""
                            )

            # Match snapshot_state's lock order: state first, then skill. No
            # simulator access occurs in either section.
            with self.server.state_lock:
                reset_pending = getattr(self.server, "reset_request", False)
                vision_degraded = bool(
                    getattr(self.server, "_vision_safe_state", False)
                )
                simulation_degraded = bool(
                    getattr(self.server, "_simulation_degraded", False)
                )
                pending_skill_hint = getattr(
                    self.server, "_pending_skill_hint", None
                )
            with self.server.skill_lock:
                current_job = getattr(self.server, "current_job", None)
                task_switch_pending = bool(
                    getattr(self.server, "task_switch_request", None) is not None
                    or getattr(self.server, "task_switch_in_progress", False)
                )
            skill_queue = getattr(self.server, "skill_queue", None)
            queued_skill = bool(
                skill_queue is not None and not skill_queue.empty()
            )
            active_skill = bool(
                current_job is not None
                or queued_skill
                or pending_skill_hint is not None
            )
            reset_active = reset_pending not in (None, False)
            finish_active = False
            try:
                from behavior_interface_eval_test.operator_scene_control import (
                    pending_finish,
                    pending_reset,
                )

                port = _official_http_port()
                if port is not None and pending_reset(port):
                    reset_active = True
                    if reset_pending in (None, False):
                        reset_pending = True
                if port is not None and pending_finish(port):
                    finish_active = True
            except Exception:
                pass
        except Exception as exc:
            return {
                "ok": True,
                "protocol": "behavior-interface-idle-probe-v1",
                "diagnostic_ok": False,
                "idle": False,
                "reason": f"idle_probe_read_failed:{type(exc).__name__}",
            }

        live_terminal = live_state in {"idle", "done", "failed", "cancelled"}
        idle = bool(
            action_source == "hold"
            and not last_error
            and initialization_ready
            and not downstream_configured
            and live_terminal
            and not active_skill
            and not reset_active
            and not finish_active
            and not task_switch_pending
            and not vision_degraded
            and not simulation_degraded
        )
        if idle:
            reason = "idle_hold"
        elif action_source != "hold":
            reason = f"action_source={action_source!r}"
        elif last_error:
            reason = "backend_error_present"
        elif not initialization_ready:
            reason = "episode_initialization_active"
        elif downstream_configured:
            reason = "downstream_policy_configured"
        elif not live_terminal:
            reason = f"live_test_{live_state or 'unknown'}"
        elif active_skill:
            reason = "skill_active_or_queued"
        elif finish_active:
            reason = "finish_pending"
        elif reset_active:
            reason = "reset_pending"
        elif task_switch_pending:
            reason = "task_switch_pending"
        else:
            reason = "backend_degraded"
        return {
            "ok": True,
            "protocol": "behavior-interface-idle-probe-v1",
            "diagnostic_ok": True,
            "idle": idle,
            "reason": reason,
            "action_source": action_source,
            "last_error": last_error,
            "episode_initialization": {"ready": initialization_ready},
            "downstream": {"configured": downstream_configured},
            "live_unit_test": {"state": live_state},
            "active_skill": active_skill,
            "reset_pending": reset_pending,
            "task_switch_pending": task_switch_pending,
            "vision_degraded": vision_degraded,
            "simulation_degraded": simulation_degraded,
        }

    def status(self) -> dict[str, Any]:
        evaluator_websocket_connected, evaluator_connection_count = (
            self.evaluator_connections.snapshot()
        )
        observation = self.adapter.status()
        evaluator_connected = (
            evaluator_websocket_connected
            and self.adapter.observation_is_fresh(self.observation_max_age_s)
        )
        with self._live_overlay_lock:
            live_head_overlay = {
                **deepcopy(self._live_overlay_status),
                "render_policy": "async_latest_only",
                "completed_sequence": int(self._live_overlay_sequence),
                "inflight_sequence": (
                    None
                    if self._live_overlay_future is None
                    else int(self._live_overlay_inflight_sequence)
                ),
                "pending_sequence": (
                    None
                    if self._live_overlay_pending is None
                    else int(self._live_overlay_pending[1].sequence)
                ),
                "dropped_superseded_requests": int(
                    self._live_overlay_dropped_requests
                ),
                "cached_feeds": sorted(self._live_overlay_frames),
            }
        with self._spatial_map_export_lock:
            navigation_map_export = {
                "render_policy": "async_latest_only"
                if self._navigation_map_async_enabled()
                else "synchronous_offline",
                "inflight_sequence": (
                    None
                    if self._spatial_map_export_future is None
                    else int(self._spatial_map_export_inflight_sequence)
                ),
                "pending_sequence": (
                    None
                    if self._spatial_map_export_pending is None
                    else int(self._spatial_map_export_pending.sequence)
                ),
                "dropped_superseded_requests": int(
                    self._spatial_map_export_dropped_requests
                ),
                "generation": int(self._spatial_map_export_generation),
                "last_error": str(self._spatial_map_export_last_error),
            }
        pipeline_timing = getattr(self, "_pipeline_timing", None)
        capture_artifacts = _capture_artifact_status(self.server.world)
        try:
            live_unit_test = self.live_test_status()
        except Exception:
            live_unit_test = {
                "ok": False,
                "state": "unavailable",
                "tool": "live_move_tracked_point_test",
            }
        return {
            "layer": "behavior-interface",
            "protocol": "BEHAVIOR v3.9.1 msgpack websocket",
            "robot_profile": ROBOT_PROFILE,
            "robot_model": ROBOT_MODEL,
            "arm_dof": ARM_DOF,
            "action_dim": ACTION_DIM,
            "proprio_dim": PROPRIO_DIM,
            "tool_version": self.tool_version,
            "ui_tool_version": self.ui_tool_version,
            "rgbd_lite_runtime": deepcopy(self.rgbd_lite_runtime),
            "gpu_mapping": _gpu_mapping_status(),
            "reset_body_implementation": RESET_BODY_IMPLEMENTATION_VERSION,
            "simulator_handle": False,
            "evaluator_connected": evaluator_connected,
            "evaluator_websocket_connected": evaluator_websocket_connected,
            "evaluator_connection_count": evaluator_connection_count,
            "observation_max_age_s": self.observation_max_age_s,
            "observation_policy": "allowlist",
            "observation": observation,
            "pipeline_timing": (
                pipeline_timing.status()
                if pipeline_timing is not None
                else {"steps": 0, "clock": "perf_counter", "phases": {}}
            ),
            "capture_artifacts": capture_artifacts,
            "ui_frame_policy": {
                "head_main": "on_demand_async_latest_only_official_routes",
                "wrist": "only_when_ui_feed_active",
                "tool_capture_source": "current_evaluator_observation",
            },
            "bddl_ui": self.adapter.ui_goal_status(),
            "base_odometry": self.server.world.base_odometry_status(),
            "gripper_keepalive": self.server.world.gripper_keepalive_status(),
            "action_source": self._source,
            "last_error": self._last_error,
            "episode_initialization": self.episode_initializer.status(),
            "live_head_overlay": live_head_overlay,
            "navigation_map_export": navigation_map_export,
            "live_unit_test": live_unit_test,
            "tracked_object_distances": (
                self.tracked_object_distances.status()
                if hasattr(self, "tracked_object_distances")
                else {
                    "active": 0,
                    "names": [],
                    "last_error": "tracker is not initialized",
                }
            ),
            "downstream": (
                self.downstream.status()
                if self.downstream is not None
                else {"configured": False}
            ),
            "tools": capability_report()["counts"],
            "tool_registry": {
                "owner": "behavior_interface_eval_test.tool.official_v2",
                "restore_count": int(self._tool_registry_restore_count),
                "blocked_legacy_mutations": int(
                    self.tool_registry.blocked_mutation_count
                ),
                "last_blocked_legacy_mutation": (
                    self.tool_registry.last_blocked_mutation
                ),
            },
        }


async def _health_check(connection, request):
    if getattr(request, "path", None) == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def serve_policy(runtime: OfficialPolicyRuntime, host: str, port: int) -> None:
    metadata = {
        "layer": "behavior-interface",
        "protocol": "BEHAVIOR v3.9.1",
        "robot_profile": ROBOT_PROFILE,
        "robot_model": ROBOT_MODEL,
        "arm_dof": ARM_DOF,
        "tool_version": runtime.tool_version,
        "action_dim": ACTION_DIM,
        "allowed_observations": [
            "RGB",
            "depth_linear",
            "proprioception",
            "camera-relative poses supplied by evaluator",
            "task_id",
        ],
        "display_only_metadata": [
            "BDDL goal completion (removed before tool/downstream policy access)",
        ],
    }

    async def handler(websocket):
        runtime.evaluator_connections.opened()
        try:
            await websocket.send(Packer().pack(metadata))
            while True:
                message = await websocket.recv()
                payload = unpackb(message, strict_map_key=False)
                if "reset" in payload:
                    runtime.reset()
                    continue
                started = time.monotonic()
                action = await asyncio.to_thread(runtime.act, payload)
                await websocket.send(
                    packb(
                        {
                            "action": action,
                            "server_timing": {
                                "infer_ms": (time.monotonic() - started) * 1000.0,
                            },
                        }
                    )
                )
        except websockets.ConnectionClosed:
            pass
        finally:
            runtime.evaluator_connections.closed()

    async with websocket_server.serve(
        handler,
        host,
        port,
        compression=None,
        max_size=None,
        # The official evaluator can spend longer than the websockets default
        # 20s + 20s keepalive window resetting a cold OmniGibson scene.  Its
        # lightweight client services control frames only while recv() runs,
        # so a server ping during reset would incorrectly close a healthy local
        # connection before the first observation arrives.
        ping_interval=None,
        process_request=_health_check,
    ) as server:
        await server.serve_forever()


def install_official_live_frame_routes(
    app: Flask,
    runtime: OfficialPolicyRuntime,
) -> None:
    """Replace only official head/main display routes with the frozen v2 HUD."""
    from behavior_interface.web import _encode_jpeg

    original_video = app.view_functions["video"]
    original_frame_jpg = app.view_functions["api_frame_jpg"]
    jpeg_cache_lock = threading.RLock()
    jpeg_cache: dict[tuple[str, int], bytes] = {}

    def encoded_frame(feed: str, item: np.ndarray, fid: int) -> bytes:
        key = (str(feed), int(fid))
        with jpeg_cache_lock:
            cached = jpeg_cache.get(key)
        if cached is not None:
            return cached
        encoded = _encode_jpeg(item)
        if not encoded:
            return b""
        with jpeg_cache_lock:
            if len(jpeg_cache) >= 8:
                jpeg_cache.clear()
            jpeg_cache[key] = encoded
        return encoded

    def official_video(feed: str):
        if feed not in {"head", "main"}:
            return original_video(feed)
        boundary = b"--frame"

        def gen():
            last_id = -1
            while True:
                item, fid, _updated_ts = runtime.live_display_frame(feed)
                if item is None or fid == last_id:
                    time.sleep(0.01)
                    continue
                last_id = fid
                jpg = encoded_frame(feed, item, fid)
                if not jpg:
                    time.sleep(0.01)
                    continue
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpg)).encode()
                    + b"\r\n\r\n"
                    + jpg
                    + b"\r\n"
                )

        return Response(
            gen(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def official_frame_jpg(feed: str):
        if feed not in {"head", "main"}:
            return original_frame_jpg(feed)
        item, fid, updated_ts = runtime.live_display_frame(feed)
        if item is None:
            return Response(status=503)
        jpg = encoded_frame(feed, item, fid)
        if not jpg:
            return Response(status=503)
        return Response(
            jpg,
            mimetype="image/jpeg",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "X-Frame-Id": str(fid),
                "X-Frame-Age-Ms": (
                    str(max(0, int((time.time() - updated_ts) * 1000)))
                    if updated_ts
                    else "0"
                ),
                "X-Official-Overlay": "frozen-v2-base-path-hud",
            },
        )

    app.view_functions["video"] = official_video
    app.view_functions["api_frame_jpg"] = official_frame_jpg


def install_official_track_object_distance_routes(
    app: Flask,
    runtime: OfficialPolicyRuntime,
) -> None:
    """Expose the test-only live distance tool without editing production web."""

    @app.after_request
    def official_track_object_distance_image_contract(response):
        if not request.path.startswith("/api/v2/") or not response.is_json:
            return response
        payload = response.get_json(silent=True)
        if not isinstance(payload, dict):
            return response
        observation = payload.get("observation")
        observation_has_image = isinstance(observation, dict) and bool(
            observation.get("image_id")
            and (
                observation.get("rgb_main")
                or observation.get("rgb_main_path")
                or observation.get("rgb_path")
            )
        )
        media = observation if observation_has_image else payload
        image_id = str(media.get("image_id") or payload.get("image_id") or "").strip()
        has_image = bool(
            media.get("rgb_main")
            or media.get("rgb_main_path")
            or media.get("rgb_path")
        )
        if not image_id or not has_image:
            return response

        binding = media.get("track_object_distance_binding")
        if not isinstance(binding, dict):
            binding = payload.get("track_object_distance_binding")
        if isinstance(binding, dict):
            binding = dict(binding)
            binding.setdefault("trackable", binding.get("ok") is True)
            binding.setdefault(
                "reason",
                None if binding.get("trackable") is True else "binding_not_registered",
            )
        else:
            feed = str(media.get("feed") or payload.get("feed") or "").lower()
            reason = (
                "unsupported_camera_role"
                if feed and feed not in ("head", "main")
                else "binding_not_registered"
            )
            binding = {
                "ok": False,
                "trackable": False,
                "reason": reason,
                "image_id": image_id,
                "error": (
                    "this returned image has no exact frozen head RGB-D frame binding"
                ),
            }
        media["track_object_distance_binding"] = binding
        payload["track_object_distance_binding"] = deepcopy(binding)
        response.set_data(app.json.dumps(payload))
        return response

    @app.post("/api/v2/track_object_distance")
    def official_track_object_distance():
        from behavior_interface_eval_test.tool.official_v2.tracked_object_distance import (
            TRACK_OBJECT_DISTANCE_REPLAY_TIMEOUT_S,
        )

        body = request.get_json(force=True, silent=True) or {}
        session_id = body.get("session_id")
        image_id = body.get("image_id")
        points = body.get("points")
        try:
            timeout_s = float(body.get("timeout_s", TRACK_OBJECT_DISTANCE_REPLAY_TIMEOUT_S))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "timeout_s must be numeric"}), 400
        if not math.isfinite(timeout_s) or timeout_s <= 0.0 or timeout_s > TRACK_OBJECT_DISTANCE_REPLAY_TIMEOUT_S:
            return jsonify({
                "ok": False,
                "error": f"timeout_s must be finite and in 0..{TRACK_OBJECT_DISTANCE_REPLAY_TIMEOUT_S:g}s",
            }), 400
        request_id = None
        try:
            request_id = runtime.server.submit_skill(
                "track_object_distance",
                {
                    "session_id": session_id,
                    "image_id": image_id,
                    "points": points,
                },
            )
            result = runtime.server.wait_for_skill_result(
                "track_object_distance",
                timeout_s=timeout_s,
                request_id=request_id,
            )
        except TimeoutError as exc:
            cancel_request = getattr(runtime, "cancel_public_job", None)
            if callable(cancel_request) and request_id:
                cancel_request(request_id)
            terminal_result = None
            if request_id:
                try:
                    terminal_result = runtime.server.wait_for_skill_result(
                        "track_object_distance",
                        timeout_s=1.0,
                        poll_s=0.01,
                        request_id=request_id,
                    )
                except Exception:
                    terminal_result = None
            if terminal_result is not None:
                acknowledge = getattr(
                    runtime,
                    "acknowledge_public_job_terminal",
                    None,
                )
                if callable(acknowledge):
                    acknowledge(request_id)
            if terminal_result and terminal_result.get("ok"):
                return jsonify(terminal_result), 200
            timeout_result = {
                "ok": False,
                "error": str(exc),
                "tool": "track_object_distance",
            }
            if terminal_result is not None:
                timeout_result["terminal_result"] = terminal_result
            return jsonify(timeout_result), 504
        except Exception as exc:
            return jsonify({
                "ok": False,
                "error": str(exc),
                "tool": "track_object_distance",
            }), 400
        result = dict(result or {})
        if result.get("ok") is False:
            result.setdefault("tool", "track_object_distance")
        return jsonify(result), (200 if result.get("ok") else 400)

    @app.post("/api/v2/cut_object")
    def official_cut_object():
        body = request.get_json(force=True, silent=True) or {}
        session_id = str(body.get("session_id") or "").strip()

        def respond(result: dict[str, Any], status_code: int):
            if not isinstance(result, Mapping):
                return jsonify({
                    "ok": False,
                    "tool": "cut_object",
                    "failure_stage": "planning",
                    "error": "official server returned a non-object result",
                }), 400
            payload = dict(result or {})
            payload.setdefault("tool", "cut_object")
            if payload.get("ok") is True and session_id:
                _attach_official_head_capture(
                    runtime,
                    payload,
                    session_id=session_id,
                    timeout_s=_FAILURE_EXIT_CAPTURE_TIMEOUT_S,
                )
            return jsonify(payload), status_code

        try:
            timeout_s = float(body.get("timeout_s", 90.0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "timeout_s must be numeric"}), 400
        if not math.isfinite(timeout_s) or not 0.1 <= timeout_s <= 600.0:
            return jsonify({
                "ok": False,
                "error": "timeout_s must be finite and in 0.1..600s",
            }), 400
        args = {
            key: body[key]
            for key in (
                "session_id",
                "image_id",
                "points",
                "pos_tol",
                "ori_tol_deg",
                "max_steps",
                "timeout_s",
            )
            if key in body
        }
        request_id = None
        try:
            request_id = runtime.server.submit_skill("cut_object", args)
            result = runtime.server.wait_for_skill_result(
                "cut_object",
                timeout_s=min(630.0, timeout_s + 30.0),
                request_id=request_id,
            )
        except TimeoutError as exc:
            cancel_request = getattr(runtime, "cancel_public_job", None)
            if callable(cancel_request) and request_id:
                cancel_request(request_id)
            terminal_result = None
            if request_id:
                try:
                    terminal_result = runtime.server.wait_for_skill_result(
                        "cut_object",
                        timeout_s=1.0,
                        poll_s=0.01,
                        request_id=request_id,
                    )
                except Exception:
                    terminal_result = None
            if terminal_result is not None:
                acknowledge = getattr(
                    runtime,
                    "acknowledge_public_job_terminal",
                    None,
                )
                if callable(acknowledge):
                    acknowledge(request_id)
            if terminal_result and terminal_result.get("ok"):
                return respond(terminal_result, 200)
            timeout_result = {
                "ok": False,
                "error": str(exc),
                "reason": "http_timeout",
                "failure_stage": "timeout",
                "tool": "cut_object",
            }
            if terminal_result is not None:
                timeout_result["terminal_result"] = terminal_result
            return jsonify(timeout_result), 504
        except Exception as exc:
            return jsonify({
                "ok": False,
                "error": str(exc),
                "tool": "cut_object",
            }), 400
        return respond(result, 200 if result.get("ok") else 400)

    original_tools = app.view_functions.get("api_v2_tools")
    if original_tools is not None:
        metadata = {
            "name": "track_object_distance",
            "endpoint": "/api/v2/track_object_distance",
            "args": [
                {
                    "name": "image_id",
                    "type": "string",
                    "widget": "image",
                    "required": True,
                },
                {
                    "name": "points",
                    "type": "array",
                    "widget": "named_multi_uv",
                    "required": True,
                    "min_points": 1,
                    "max_points": 32,
                    "name_max_length": 128,
                    "items": {
                        "type": "object",
                        "required": ["name", "u", "v"],
                        "properties": {
                            "name": {"type": "string"},
                            "u": {"type": "number", "minimum": 0, "maximum": 1000},
                            "v": {"type": "number", "minimum": 0, "maximum": 1000},
                        },
                    },
                }
            ],
            "coordinate_system": "Qwen3-VL relative image coordinates 0..1000",
            "desc": (
                "Bind every UV to the exact frozen head capture named by image_id, "
                "then track multiple model-named surface points; publish only "
                "current depth_m and "
                "xyz_in_robot_base_coord_m=[x, y, z] in memory. "
                "xyz_in_robot_base_coord_m is measured in meters in the current "
                "robot chassis frame: +X always points chassis-forward, +Y "
                "chassis-left, and +Z chassis-up. A stale image binding or replay "
                "gap fails instead of applying UV to a newer frame. "
                "capture_replay_history_was_evicted_by_its_bounded_storage_policy "
                "means capture a new head image and select the points on that "
                "new image. Delete an "
                "entry when the point leaves view or tracking is lost."
            ),
        }
        cut_metadata = {
            "name": "cut_object",
            "endpoint": "/api/v2/cut_object",
            "args": [
                {
                    "name": "image_id",
                    "type": "string",
                    "widget": "image",
                    "required": True,
                },
                {
                    "name": "points",
                    "type": "array",
                    "widget": "named_multi_uv",
                    "required": True,
                    "min_points": 2,
                    "max_points": 2,
                    "fixed_names": [
                        "cutting_tool_point",
                        "target_object_point",
                    ],
                    "items": {
                        "type": "object",
                        "required": ["name", "u", "v"],
                        "properties": {
                            "name": {"type": "string"},
                            "u": {"type": "number", "minimum": 0, "maximum": 1000},
                            "v": {"type": "number", "minimum": 0, "maximum": 1000},
                        },
                    },
                },
                {
                    "name": "pos_tol",
                    "type": "number",
                    "widget": "number",
                    "required": False,
                    "default": 0.012,
                    "unit": "m",
                },
                {
                    "name": "ori_tol_deg",
                    "type": "number",
                    "widget": "number",
                    "required": False,
                    "default": 5.0,
                    "unit": "deg",
                },
                {
                    "name": "max_steps",
                    "type": "integer",
                    "widget": "number",
                    "required": False,
                    "default": 360,
                },
                {
                    "name": "timeout_s",
                    "type": "number",
                    "widget": "number",
                    "required": False,
                    "default": 90.0,
                    "unit": "s",
                },
            ],
            "coordinate_system": "Qwen3-VL relative image coordinates 0..1000",
            "desc": (
                "Select cutting_tool_point first and target_object_point second "
                "on the same bound head capture. Track both points live and move "
                "the nearest eligible held-tool EEF, with orientation and grippers "
                "preserved, until their current robot-base XYZ distance converges. "
                "This performs one touch attempt and does not use contact or BDDL truth."
            ),
        }

        def head_adjust_metadata(arm: str) -> dict[str, Any]:
            side = str(arm)

            def translation_arg(
                name: str,
                *,
                frame: str,
                positive: str,
                group: str,
                description: str,
            ) -> dict[str, Any]:
                other_group = (
                    "head_camera_forward_leftward_upward"
                    if group == "robot_base_xyz"
                    else "robot_base_xyz"
                )
                return {
                    "name": name,
                    "type": "number",
                    "widget": "number",
                    "required": False,
                    "unit": "m",
                    "coordinate_frame": frame,
                    "positive_direction": positive,
                    "exclusive_group": group,
                    "mutually_exclusive_with": other_group,
                    "description": description,
                }

            args = [
                translation_arg(
                    "x",
                    frame="robot_base",
                    positive="chassis_forward",
                    group="robot_base_xyz",
                    description=(
                        "Robot-base delta: +X is chassis-forward. Use short +X "
                        "increments for chassis-horizontal pushing. Use only "
                        "with y/z; never combine with camera-frame parameters."
                    ),
                ),
                translation_arg(
                    "y",
                    frame="robot_base",
                    positive="chassis_left",
                    group="robot_base_xyz",
                    description=(
                        "Robot-base delta: +Y is chassis-left. Same positive-left "
                        "sign as leftward, but base-fixed rather than camera-rotated."
                    ),
                ),
                translation_arg(
                    "z",
                    frame="robot_base",
                    positive="chassis_up",
                    group="robot_base_xyz",
                    description=(
                        "Robot-base delta: +Z is chassis-up. Use only with x/y."
                    ),
                ),
                translation_arg(
                    "forward",
                    frame="starting_head_camera",
                    positive="camera_forward",
                    group="head_camera_forward_leftward_upward",
                    description=(
                        "Starting-head-camera delta along the viewing direction. "
                        "This includes a vertical component when the camera is "
                        "pitched and is not chassis-horizontal forward. "
                        "Use only with leftward/upward; never combine with x/y/z."
                    ),
                ),
                translation_arg(
                    "leftward",
                    frame="starting_head_camera",
                    positive="camera_left",
                    group="head_camera_forward_leftward_upward",
                    description=(
                        "Starting-head-camera delta toward image left. Same "
                        "positive-left sign as y, but follows camera orientation."
                    ),
                ),
                translation_arg(
                    "upward",
                    frame="starting_head_camera",
                    positive="camera_up",
                    group="head_camera_forward_leftward_upward",
                    description=(
                        "Starting-head-camera delta toward image up. Use only "
                        "with forward/leftward."
                    ),
                ),
            ]
            args.extend(
                [
                    {
                        "name": name,
                        "type": "number",
                        "widget": "number",
                        "required": False,
                        "default": default,
                        "unit": unit,
                    }
                    for name, default, unit in (
                        ("roll", 0.0, f"deg({side} gripper local axis)"),
                        ("pitch", 0.0, "deg(+ fingertips-up)"),
                        ("yaw", 0.0, f"deg({side} gripper local axis)"),
                        ("pos_tol", 0.012, "m"),
                        ("ori_tol_deg", 3.0, "deg"),
                        ("max_steps", 240, "observation/action steps"),
                        ("timeout_s", 60.0, "s"),
                    )
                ]
            )
            return {
                "name": f"adjust_{side}_eef_pose_in_head_frame",
                "endpoint": f"/api/v2/adjust_{side}_eef_pose_in_head_frame",
                "args": args,
                "translation_parameter_families": {
                    "robot_base_xyz": {
                        "parameters": ["x", "y", "z"],
                        "axes": "+X chassis-forward, +Y chassis-left, +Z chassis-up",
                    },
                    "head_camera_forward_leftward_upward": {
                        "parameters": ["forward", "leftward", "upward"],
                        "axes": "starting head-camera forward, image-left, image-up",
                    },
                },
                "translation_families_mutually_exclusive": True,
                "desc": (
                    f"Adjust the {side} EEF by incremental translation and local-gripper RPY. "
                    "For translation, use only x/y/z in the current robot base "
                    "frame (+X forward, +Y left, +Z up), or only "
                    "forward/leftward/upward in the starting head-camera frame. "
                    "Never mix the two parameter families, even when one value is zero. "
                    "y and leftward have the same positive-left sign but belong to "
                    "different frames. For a horizontal push, issue short +x "
                    "increments and verify external-object motion from the next "
                    "evaluator RGB-D observation; tool success itself means the "
                    "requested EEF pose converged."
                ),
            }

        head_metadata = [
            head_adjust_metadata("left"),
            head_adjust_metadata("right"),
        ]

        def official_v2_tools():
            response = app.make_response(original_tools())
            payload = response.get_json(silent=True) or {}
            replaced_names = {
                metadata["name"],
                cut_metadata["name"],
                *(item["name"] for item in head_metadata),
            }
            tools = [
                item
                for item in list(payload.get("tools") or [])
                if item.get("name") not in replaced_names
            ]
            tools.extend(deepcopy(head_metadata))
            tools.append(deepcopy(cut_metadata))
            tools.append(deepcopy(metadata))
            payload["tools"] = tools
            return jsonify(payload)

        app.view_functions["api_v2_tools"] = official_v2_tools


def install_official_move_tracked_point_routes(
    app: Flask,
    runtime: OfficialPolicyRuntime,
) -> None:
    """Expose live tracked-point motion through the official test web surface."""

    @app.post("/api/v2/move_tracked_point")
    def official_move_tracked_point():
        body = request.get_json(force=True, silent=True) or {}
        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({
                "ok": False,
                "tool": "move_tracked_point",
                "failure_stage": "input validation",
                "error": "session_id is required",
            }), 400

        raw_args = {
            str(key): value
            for key, value in body.items()
            if str(key) != "session_id"
        }
        try:
            args = validate_submission("move_tracked_point", raw_args)
        except (TypeError, ValueError) as exc:
            return jsonify({
                "ok": False,
                "tool": "move_tracked_point",
                "failure_stage": "input validation",
                "error": str(exc),
            }), 400

        def respond(result: dict[str, Any], status_code: int):
            if not isinstance(result, Mapping):
                return jsonify({
                    "ok": False,
                    "tool": "move_tracked_point",
                    "failure_stage": "planning",
                    "error": "official server returned a non-object result",
                }), 400
            payload = dict(result or {})
            payload.setdefault("tool", "move_tracked_point")
            payload.setdefault("execution_mode", args["execution_mode"])
            reload_status_fn = getattr(
                runtime,
                "official_v2_reload_status",
                None,
            )
            reload_status = (
                reload_status_fn() if callable(reload_status_fn) else {}
            )
            payload.setdefault(
                "reload_generation",
                int(reload_status.get("generation", 0)),
            )
            payload.setdefault(
                "official_v2_stack_digest",
                reload_status.get("stack_digest"),
            )
            if payload.get("ok") is True:
                if args["execution_mode"] == "plan":
                    _expose_successful_plan_preview(payload)
                else:
                    _attach_official_head_capture(
                        runtime,
                        payload,
                        session_id=session_id,
                        timeout_s=_FAILURE_EXIT_CAPTURE_TIMEOUT_S,
                    )
            return jsonify(_json_ready(payload)), status_code

        request_id = None
        wait_timeout_s = min(
            630.0,
            max(120.0, float(args["timeout_s"]) + 30.0),
        )
        try:
            request_id = runtime.server.submit_skill(
                "move_tracked_point",
                args,
            )
            result = runtime.server.wait_for_skill_result(
                "move_tracked_point",
                timeout_s=wait_timeout_s,
                request_id=request_id,
            )
        except TimeoutError as exc:
            cancel_request = getattr(runtime, "cancel_public_job", None)
            if callable(cancel_request) and request_id:
                cancel_request(request_id)
            terminal_result = None
            if request_id:
                try:
                    terminal_result = runtime.server.wait_for_skill_result(
                        "move_tracked_point",
                        timeout_s=1.0,
                        poll_s=0.01,
                        request_id=request_id,
                    )
                except Exception:
                    terminal_result = None
            if terminal_result is not None:
                acknowledge = getattr(
                    runtime,
                    "acknowledge_public_job_terminal",
                    None,
                )
                if callable(acknowledge):
                    acknowledge(request_id)
            if terminal_result and terminal_result.get("ok"):
                return respond(terminal_result, 200)
            timeout_result = {
                "ok": False,
                "tool": "move_tracked_point",
                "failure_stage": "timeout",
                "error": str(exc),
            }
            if terminal_result is not None:
                timeout_result["terminal_result"] = terminal_result
            return jsonify(_json_ready(timeout_result)), 504
        except Exception as exc:
            return jsonify({
                "ok": False,
                "tool": "move_tracked_point",
                "failure_stage": "planning",
                "error": str(exc),
            }), 400
        return respond(
            result,
            200 if isinstance(result, Mapping) and result.get("ok") else 400,
        )

    original_tools = app.view_functions.get("api_v2_tools")
    if original_tools is None:
        return
    # The backend performs the authoritative AST/affine validation.  This
    # schema pattern is intentionally a lexical allow-list so the generic
    # interface validator does not reject valid spacing, parentheses, scalar
    # coefficients, or division before the shared test-local parser sees it.
    coordinate_pattern = r"^(?:\?|[A-Za-z0-9_().+*/\-\s]{1,256})$"
    variable_schema = {
        "oneOf": [
            {"type": "number"},
            {
                "type": "string",
                "pattern": coordinate_pattern,
            },
            {
                "type": "object",
                "required": ["var"],
                "additionalProperties": False,
                "properties": {
                    "var": {
                        "type": "string",
                        "pattern": "^[A-Za-z][A-Za-z0-9_]*$",
                    },
                    "expr": {
                        "type": "string",
                        "pattern": coordinate_pattern,
                    },
                    "free": {"const": True},
                },
            },
            {
                "type": "object",
                "required": ["expr"],
                "additionalProperties": False,
                "properties": {
                    "expr": {
                        "type": "string",
                        "pattern": coordinate_pattern,
                    }
                },
            },
            {
                "type": "object",
                "required": ["free"],
                "additionalProperties": False,
                "properties": {"free": {"const": True}},
            },
        ]
    }
    coordinate_object_schema = {
        "type": "object",
        "required": ["x", "y", "z"],
        "additionalProperties": False,
        "properties": {
            axis: deepcopy(variable_schema)
            for axis in ("x", "y", "z")
        },
    }
    coordinate_array_schema = {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": deepcopy(variable_schema),
    }
    metadata = {
        "name": "move_tracked_point",
        "endpoint": "/api/v2/move_tracked_point",
        "args": [
            {
                "name": "execution_mode",
                "type": "string",
                "widget": "select",
                "required": False,
                "default": "exec",
                "options": ["exec", "plan"],
                "description": (
                    "exec moves immediately; plan returns a red final-EEF "
                    "overlay and an arm-only plan_id for exec_plan_pose without "
                    "moving; execution uses current-to-safe then safe-to-final "
                    "with the live trunk pinned"
                ),
            },
            {
                "name": "points",
                "type": "array",
                "widget": "tracked_target_points",
                "required": True,
                "min_points": 1,
                "max_points": 6,
                "quick_max_points": 6,
                "quick_presets": [
                    {
                        "type": "touch",
                        "on_hand_count": 1,
                        "off_hand_count": 1,
                    },
                    {
                        "type": "flatwise",
                        "on_hand_count": 3,
                        "off_hand_count": 0,
                    },
                    {
                        "type": "plane_parallel",
                        "on_hand_count": 3,
                        "off_hand_count": 3,
                    },
                    {
                        "type": "collinear",
                        "on_hand_count": 2,
                        "off_hand_count": [1, 2],
                    },
                    {
                        "type": "line_vertical_to_plane",
                        "on_hand_count": 2,
                        "off_hand_count": 3,
                    },
                    {
                        "type": "vertical_to_ground",
                        "on_hand_count": [2, 3],
                        "off_hand_count": 0,
                    },
                    {
                        "type": "faceto",
                        "on_hand_count": 3,
                        "off_hand_count": 0,
                    },
                    {
                        "type": "reverse_faceto",
                        "on_hand_count": 3,
                        "off_hand_count": 0,
                    },
                ],
                "name_max_length": 128,
                "variable_pattern": "^[A-Za-z][A-Za-z0-9_]*$",
                "coordinate_pattern": coordinate_pattern,
                "coordinate_help": (
                    "finite number, variable identifier, affine expression "
                    "such as z+0.08, 2*z, (a+b)/2, or a-b, or ? for an "
                    "unconstrained coordinate"
                ),
                "predefined_eef_points": [
                    "left_finger_tip",
                    "right_finger_tip",
                    "gripper_slide_center",
                ],
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "additionalProperties": False,
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "An active tracked name, or on-hand predefined "
                                "EEF point left_finger_tip, right_finger_tip, or "
                                "gripper_slide_center (no tracking required). The "
                                "finger-tip names denote the narrow physical front "
                                "contact-cap centers at max +Z_EEF on the exact red "
                                "plan-overlay mesh."
                            ),
                        },
                        "role": {
                            "type": "string",
                            "enum": ["on_hand", "off_hand"],
                        },
                        "target_xyz_m": {
                            "oneOf": [
                                coordinate_object_schema,
                                coordinate_array_schema,
                            ]
                        },
                    },
                },
                "quick_items": {
                    "type": "object",
                    "required": ["name", "role"],
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "role": {
                            "type": "string",
                            "enum": ["on_hand", "off_hand"],
                        },
                        "target_xyz_m": {
                            "oneOf": [
                                coordinate_object_schema,
                                coordinate_array_schema,
                            ]
                        },
                    },
                },
            },
            {
                "name": "relations",
                "type": "array",
                "required": False,
                "default": [],
                "items": {
                    "type": "object",
                    "required": ["type", "point_names"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "common_plane",
                                "oriented_plane_normal",
                                "align_vector",
                                "vertical_to_ground",
                            ],
                        },
                        "point_names": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "normal_robot_base": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "direction_robot_base": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "offset_m": {},
                        "mode": {
                            "type": "string",
                            "enum": ["same", "opposite", "parallel"],
                        },
                    },
                },
            },
            {
                "name": "inequalities",
                "type": "array",
                "required": False,
                "default": [],
                "max_items": 6,
                "description": (
                    "Optional strict affine variable inequalities solved jointly "
                    "with every XYZ and geometry constraint. The left side uses "
                    "the same safe affine grammar as target coordinates, the "
                    "operator is > or <, and the right side is a finite constant. "
                    "For example, a-b > 0 selects the solution where a is above b."
                ),
                "items": {
                    "type": "object",
                    "required": ["lhs", "op", "rhs"],
                    "additionalProperties": False,
                    "properties": {
                        "lhs": {
                            "type": "string",
                            "pattern": coordinate_pattern,
                            "description": (
                                "Variable or affine function, for example a-b, "
                                "2*a-b, or (a+b)/2"
                            ),
                        },
                        "op": {
                            "type": "string",
                            "enum": [">", "<"],
                        },
                        "rhs": {
                            "type": "number",
                            "description": "Finite constant in metres.",
                        },
                    },
                },
            },
            {
                "name": "quick_constraint",
                "type": "object",
                "widget": "quick_constraint",
                "required": False,
                "description": (
                    "Optional role-based geometry preset solved jointly with "
                    "all supplied XYZ and typed relations"
                ),
                "enum_types": [
                    "touch",
                    "flatwise",
                    "plane_parallel",
                    "collinear",
                    "line_vertical_to_plane",
                    "vertical_to_ground",
                    "faceto",
                    "reverse_faceto",
                ],
            },
            {
                "name": "quick_constraints",
                "type": "array",
                "widget": "quick_constraint_groups",
                "required": False,
                "max_items": 6,
                "description": (
                    "Optional named preset groups solved simultaneously with "
                    "all supplied XYZ and typed relations"
                ),
                "items": {
                    "type": "object",
                    "required": ["type", "on_hand_points", "off_hand_points"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "touch",
                                "flatwise",
                                "plane_parallel",
                                "collinear",
                                "line_vertical_to_plane",
                                "vertical_to_ground",
                                "faceto",
                                "reverse_faceto",
                            ],
                        },
                        "on_hand_points": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "off_hand_points": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "axial_mode": {
                            "type": "string",
                            "enum": [
                                "ordered_containment",
                                "segment_overlap",
                                "line_only",
                            ],
                        },
                        "axial_point_order": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            {
                "name": "pos_tol",
                "type": "number",
                "widget": "number",
                "required": False,
                "default": 0.03,
                "unit": "m",
            },
            {
                "name": "ori_tol_deg",
                "type": "number",
                "widget": "number",
                "required": False,
                "default": 20.0,
                "unit": "deg",
            },
            {
                "name": "max_steps",
                "type": "integer",
                "widget": "number",
                "required": False,
                "default": 360,
            },
            {
                "name": "timeout_s",
                "type": "number",
                "widget": "number",
                "required": False,
                "default": 90.0,
                "unit": "s",
            },
        ],
        "coordinate_system": (
            "current robot-base frame in meters: +X chassis-forward, "
            "+Y chassis-left, +Z chassis-up"
        ),
        "desc": (
            "Plan or move one to six point references. Ordinary names must be active "
            "in track_object_distance. The on-hand names left_finger_tip, "
            "right_finger_tip, and gripper_slide_center instead use the selected "
            "gripper's live evaluator proprioception and submission-local FK and do "
            "not need tracking. Each finger-tip name is the center of that finger's "
            "narrow physical front contact cap at max +Z_EEF on the exact red-overlay "
            "mesh, and moves with the observed finger-prismatic joint. "
            "execution_mode defaults to exec; plan returns the frozen-head red final-EEF "
            "overlay and a signed arm-only plan_id executable by exec_plan_pose without moving. "
            "It uses the same current-to-safe then safe-to-final compiler as an RGB-D Lite "
            "grasp plan and pins the live trunk throughout execution. Mark each "
            "point on-hand or frozen off-hand, optionally select touch, flatwise, "
            "plane-parallel, collinear, line-vertical-to-plane, vertical-to-ground, faceto, or reverse-faceto, and optionally "
            "enter target x/y/z as a finite number, "
            "a variable identifier, or an affine expression such as z+0.08, "
            "2*z, (a+b)/2, or a-b. Reusing a variable imposes that relation across any point and "
            "axis; an identifier used once leaves that degree of freedom for "
            "the nearest reachable solution. "
            "Optional strict inequalities compare a variable or affine function "
            "with a finite constant using > or <; for example a-b > 0 selects "
            "the signed solution with a above b. Inequalities are solved jointly "
            "and rechecked against the final live tracked points. "
            "All on-hand points must belong to the same rigid object. Quick geometry "
            "and XYZ constraints are solved jointly. Four-point collinear uses the "
            "submitted row order as the directed axial order (for example 1,3,4,2). "
            "Vertical-to-ground uses two on-hand points for equal X/Y, or three "
            "on-hand points for a plane whose unit normal has Z=0, with no fixed heading. "
            "Faceto and reverse-faceto require exactly three ordered on-hand surface points and no off-hand points. "
            "The displayed row order 1,2,3 is preserved: faceto requires those points to wind clockwise in the frozen head-camera image, "
            "while reverse-faceto requires counterclockwise winding. Their outward triangle normal is aligned as closely as possible "
            "to the frozen head-camera optical-forward direction (or its opposite for reverse-faceto); the reported angle is the acute angle. "
            "Optional typed common_plane, oriented_plane_normal, "
            "and align_vector relations are checked by the local solver. The "
            "nearest EEF is selected from live local FK, and success requires "
            "subsequent proprioception and live tracked-point convergence. "
            + MOVE_TRACKED_POINT_ORDER_DESCRIPTION
        ),
    }

    def official_v2_tools_with_move_tracked_point():
        response = app.make_response(original_tools())
        payload = response.get_json(silent=True) or {}
        tools = [
            item
            for item in list(payload.get("tools") or [])
            if item.get("name") != metadata["name"]
        ]
        # Keep the point-motion tool beside the tracker that supplies its
        # inputs.  This is presentation-only; it does not alter dispatch or
        # planner semantics, but prevents the entry from being hidden at the
        # bottom of a long dropdown.
        insertion_index = next(
            (
                index + 1
                for index, item in enumerate(tools)
                if item.get("name") == "track_object_distance"
            ),
            len(tools),
        )
        tools.insert(insertion_index, deepcopy(metadata))
        payload["tools"] = tools
        return jsonify(payload)

    app.view_functions[
        "api_v2_tools"
    ] = official_v2_tools_with_move_tracked_point


def install_official_surface_facing_route(
    app: Flask,
    runtime: OfficialPolicyRuntime,
) -> None:
    """Expose frozen-head three-point surface-facing chassis motion."""

    tool_name = "move_chassis_to_directly_facing_surface"

    @app.post("/api/v2/move_chassis_to_directly_facing_surface")
    def official_move_chassis_to_directly_facing_surface():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify({
                "ok": False,
                "tool": tool_name,
                "failure_stage": "input validation",
                "error": "request body must be a JSON object",
            }), 400
        request_id = None
        try:
            args = validate_submission(tool_name, body)
            wait_timeout_s = min(
                630.0,
                max(120.0, float(args["nav_timeout_s"]) + 30.0),
            )
            request_id = runtime.server.submit_skill(tool_name, args)
            raw_result = runtime.server.wait_for_skill_result(
                tool_name,
                timeout_s=wait_timeout_s,
                request_id=request_id,
            )
        except TimeoutError as exc:
            cancel_request = getattr(runtime, "cancel_public_job", None)
            cancelled = bool(
                callable(cancel_request)
                and request_id
                and cancel_request(request_id)
            )
            return jsonify({
                "ok": False,
                "tool": tool_name,
                "failure_stage": "timeout",
                "error": str(exc),
                "request_id": request_id,
                "request_scoped_cancellation": cancelled,
            }), 504
        except (TypeError, ValueError) as exc:
            return jsonify({
                "ok": False,
                "tool": tool_name,
                "failure_stage": "input validation",
                "error": str(exc),
            }), 400
        except Exception as exc:
            return jsonify({
                "ok": False,
                "tool": tool_name,
                "failure_stage": "tracking",
                "error": str(exc),
            }), 400

        if not isinstance(raw_result, Mapping):
            return jsonify({
                "ok": False,
                "tool": tool_name,
                "failure_stage": "tracking",
                "error": "official server returned a non-object result",
            }), 400
        result = dict(raw_result)
        result.setdefault("tool", tool_name)
        result.setdefault("request_id", request_id)
        return jsonify(_json_ready(result)), (200 if result.get("ok") else 400)

    original_tools = app.view_functions.get("api_v2_tools")
    if original_tools is None:
        return
    metadata = {
        "name": tool_name,
        "endpoint": "/api/v2/move_chassis_to_directly_facing_surface",
        "args": [
            {
                "name": "image_id",
                "type": "string",
                "widget": "image",
                "required": True,
            },
            {
                "name": "points",
                "type": "array",
                "widget": "multi_uv_arm",
                "required": True,
                "min_points": 3,
                "max_points": 3,
                "arm_options": ["any"],
                "items": {
                    "type": "object",
                    "required": ["u", "v"],
                    "additionalProperties": False,
                    "properties": {
                        "u": {"type": "number", "minimum": 0, "maximum": 1000},
                        "v": {"type": "number", "minimum": 0, "maximum": 1000},
                    },
                },
            },
            {
                "name": "nav_timeout_s",
                "type": "number",
                "widget": "number",
                "required": False,
                "default": 120.0,
                "unit": "s",
            },
            {
                "name": "pos_tol_m",
                "type": "number",
                "widget": "number",
                "required": False,
                "default": 0.04,
                "unit": "m",
            },
        ],
        "coordinate_system": "Qwen3-VL relative image coordinates 0..1000",
        "desc": (
            "在同一张冻结 head RGB-D 图上选择恰好三个不共线的表面点。"
            "工具反投影三点并取三角形中心；法向符号自动选择为与冻结 head "
            "相机视线成钝角的一侧。底盘中心先平移到中心沿该法向 0.8m 后投影"
            "到地面的 XY 点，再原地旋转，使机器人前向与该法向的水平投影平行"
            "且方向相反。"
            "三点顺序不会翻转最终法向；执行只使用 official action，并由后续 "
            "base_qvel proprioception 验证。"
        ),
    }

    def official_tools_with_surface_facing():
        response = app.make_response(original_tools())
        payload = response.get_json(silent=True) or {}
        tools = [
            item
            for item in list(payload.get("tools") or [])
            if item.get("name") != tool_name
        ]
        insertion_index = next(
            (
                index + 1
                for index, item in enumerate(tools)
                if item.get("name") == "move_chassis_to_floor_point"
            ),
            len(tools),
        )
        tools.insert(insertion_index, deepcopy(metadata))
        payload["tools"] = tools
        return jsonify(payload)

    app.view_functions["api_v2_tools"] = official_tools_with_surface_facing


def install_official_head_adjust_routes(
    app: Flask,
    runtime: OfficialPolicyRuntime,
) -> None:
    """Replace legacy wrappers that discard robot-base XYZ arguments."""

    translation_keys = ("x", "y", "z", "forward", "leftward", "upward")
    control_keys = (
        "roll",
        "pitch",
        "yaw",
        "pos_tol",
        "ori_tol_deg",
        "max_steps",
        "timeout_s",
    )

    def body_has_value(body: dict[str, Any], key: str) -> bool:
        value = body.get(key)
        return value is not None and str(value).strip() != ""

    def make_handler(tool_name: str, arm: str):
        def official_head_adjust():
            body = request.get_json(force=True, silent=True) or {}
            session_id = str(body.get("session_id") or "").strip()
            if not session_id:
                return jsonify({"ok": False, "error": "session_id is required"}), 400
            try:
                wait_timeout_s = float(body.get("timeout_s", 90.0))
                capture_timeout_s = float(
                    body.get(
                        "capture_timeout_s",
                        max(120.0, min(wait_timeout_s, 180.0)),
                    )
                )
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "timeouts must be numeric"}), 400
            if (
                not math.isfinite(wait_timeout_s)
                or wait_timeout_s <= 0.0
                or wait_timeout_s > 600.0
                or not math.isfinite(capture_timeout_s)
                or capture_timeout_s <= 0.0
                or capture_timeout_s > 600.0
            ):
                return jsonify({
                    "ok": False,
                    "error": "timeouts must be finite and in 0..600s",
                }), 400

            args = {
                key: body[key]
                for key in translation_keys + control_keys
                if body_has_value(body, key)
            }
            request_id = None
            try:
                request_id = runtime.server.submit_skill(tool_name, args)
                raw_result = runtime.server.wait_for_skill_result(
                    tool_name,
                    timeout_s=wait_timeout_s,
                    request_id=request_id,
                )
            except TimeoutError as exc:
                cancel_request = getattr(runtime, "cancel_public_job", None)
                cancellation_requested = bool(
                    callable(cancel_request)
                    and request_id is not None
                    and cancel_request(request_id)
                )
                return jsonify({
                    "ok": False,
                    "error": str(exc),
                    "tool": tool_name,
                    "request_id": request_id,
                    "request_scoped_cancellation": cancellation_requested,
                }), 504
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc), "tool": tool_name}), 400

            result = dict(raw_result or {})
            result.setdefault("tool", tool_name)
            result.setdefault("arm", arm)
            _attach_official_head_capture(
                runtime,
                result,
                session_id=session_id,
                timeout_s=capture_timeout_s,
            )
            return jsonify(result)

        official_head_adjust.__name__ = f"official_{tool_name}"
        return official_head_adjust

    endpoints = (
        (
            "api_v2_adjust_left_eef_pose_in_head_frame",
            "adjust_left_eef_pose_in_head_frame",
            "left",
        ),
        (
            "api_v2_adjust_right_eef_pose_in_head_frame",
            "adjust_right_eef_pose_in_head_frame",
            "right",
        ),
    )
    for endpoint, tool_name, arm in endpoints:
        if endpoint in app.view_functions:
            app.view_functions[endpoint] = make_handler(tool_name, arm)


def install_official_set_arm_route(
    app: Flask,
    runtime: OfficialPolicyRuntime,
) -> None:
    """Give progress-aware grasp prep enough HTTP time to finish safely."""

    endpoint = "api_v2_set_arm_to_grasp_position"
    if endpoint not in app.view_functions:
        return

    def official_set_arm_to_grasp_position():
        body = request.get_json(force=True, silent=True) or {}
        session_id = str(body.get("session_id") or "web").strip()
        try:
            requested_timeout_s = float(body.get("timeout_s", 15.0))
            capture_timeout_s = float(body.get("capture_timeout_s", 120.0))
        except (TypeError, ValueError):
            return jsonify({
                "ok": False,
                "tool": "set_arm_to_grasp_position",
                "error": "timeouts must be numeric",
            }), 400
        if (
            not math.isfinite(requested_timeout_s)
            or not 0.1 <= requested_timeout_s <= 180.0
            or not math.isfinite(capture_timeout_s)
            or not 0.1 <= capture_timeout_s <= 600.0
        ):
            return jsonify({
                "ok": False,
                "tool": "set_arm_to_grasp_position",
                "error": (
                    "timeout_s must be finite and in 0.1..180s; "
                    "capture_timeout_s must be finite and in 0.1..600s"
                ),
            }), 400

        gripper = str(body.get("gripper") or "keep").strip().lower()
        if body.get("open_gripper") is True:
            gripper = "open"
        args = {
            "arm": str(body.get("arm") or "right").strip().lower(),
            "keep_ori_arm": str(
                body.get("keep_ori_arm") or "none"
            ).strip().lower(),
            "gripper": gripper,
            "open_gripper": gripper == "open",
            "timeout_s": requested_timeout_s,
            "max_dq_per_step": body.get("max_dq_per_step", 0.30),
            "tol": body.get("tol", 0.08),
        }
        try:
            validated_args = validate_submission(
                "set_arm_to_grasp_position",
                args,
            )
        except Exception as exc:
            return jsonify({
                "ok": False,
                "tool": "set_arm_to_grasp_position",
                "error": str(exc),
            }), 400

        hard_skill_timeout_s = grasp_prep_hard_timeout_s(
            requested_timeout_s
        )
        wait_timeout_s = min(600.0, hard_skill_timeout_s + 30.0)
        request_id = None
        try:
            request_id = runtime.server.submit_skill(
                "set_arm_to_grasp_position",
                validated_args,
            )
            raw_result = runtime.server.wait_for_skill_result(
                "set_arm_to_grasp_position",
                timeout_s=wait_timeout_s,
                request_id=request_id,
            )
        except TimeoutError as exc:
            cancel_request = getattr(runtime, "cancel_public_job", None)
            cancelled = bool(
                callable(cancel_request)
                and request_id
                and cancel_request(request_id)
            )
            return jsonify({
                "ok": False,
                "tool": "set_arm_to_grasp_position",
                "error": str(exc),
                "request_id": request_id,
                "request_scoped_cancellation": cancelled,
                "skill_soft_timeout_s": requested_timeout_s,
                "skill_hard_timeout_s": hard_skill_timeout_s,
                "http_wait_timeout_s": wait_timeout_s,
            }), 504
        except Exception as exc:
            return jsonify({
                "ok": False,
                "tool": "set_arm_to_grasp_position",
                "error": str(exc),
            }), 400

        result = dict(raw_result or {})
        result.setdefault("tool", "set_arm_to_grasp_position")
        result.setdefault("request_id", request_id)
        result.setdefault("skill_soft_timeout_s", requested_timeout_s)
        result.setdefault("skill_hard_timeout_s", hard_skill_timeout_s)
        result.setdefault("http_wait_timeout_s", wait_timeout_s)
        _attach_official_head_capture(
            runtime,
            result,
            session_id=session_id,
            timeout_s=capture_timeout_s,
        )
        return jsonify(result), (200 if result.get("ok") else 400)

    official_set_arm_to_grasp_position.__name__ = (
        "official_set_arm_to_grasp_position"
    )
    app.view_functions[endpoint] = official_set_arm_to_grasp_position

    original_tools = app.view_functions.get("api_v2_tools")
    if original_tools is None:
        return

    def official_tools_with_feedback_grasp_prep():
        response = app.make_response(original_tools())
        payload = response.get_json(silent=True) or {}
        tools = list(payload.get("tools") or [])
        for item in tools:
            if item.get("name") != "set_arm_to_grasp_position":
                continue
            item["desc"] = (
                "将选定手臂闭环移动到固定 grasp-prep。每条 action 都由最新 "
                "proprio 的 qpos/qvel 节流；成功需位置和速度连续稳定；仍有实际"
                "进展时 timeout_s 最多续期两次，失败则锁住最后实测姿态。"
                "J8 始终锁定 0rad；当前 official_v2 固定关节替代仅支持 "
                "keep_ori_arm=none。完成或失败后都拍 head camera。"
            )
            item["timeout_policy"] = "progress_aware_soft_deadline"
            item["failure_hold"] = "latest_evaluator_proprioception"
            break
        payload["tools"] = tools
        return jsonify(payload)

    app.view_functions["api_v2_tools"] = (
        official_tools_with_feedback_grasp_prep
    )


def _observation_has_image(observation: Any) -> bool:
    return isinstance(observation, dict) and bool(
        str(observation.get("image_id") or "").strip()
        and any(
            observation.get(key)
            for key in ("rgb_main", "rgb_main_path", "rgb_path")
        )
    )


_CAPTURE_SKILL_FEEDS = {
    "capture_head_camera": frozenset({"head", "main"}),
    "capture_left_wrist_camera": frozenset({"left_wrist"}),
    "capture_right_wrist_camera": frozenset({"right_wrist"}),
}
_CAPTURE_SKILL_PRIMARY_FEED = {
    "capture_head_camera": "head",
    "capture_left_wrist_camera": "left_wrist",
    "capture_right_wrist_camera": "right_wrist",
}
_PLAN_MODE_TO_PUBLIC_TOOL = {
    "grasp_point_filter": "plan_grasp_point_filter",
    "grasp_point_filter_rgbd": "plan_grasp_point_filter_rgbd",
    "grasp_point_filter_rgbd_lite": "plan_grasp_point_filter_rgbd_lite",
    "press_point": "plan_press_point",
}


def _official_v2_request_tool(
    result: dict[str, Any],
    body: dict[str, Any],
) -> str:
    candidates: list[Any] = [result.get("tool")]
    for key in ("result", "terminal_result"):
        nested = result.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("tool"))
    for candidate in candidates:
        tool_name = str(candidate or "").strip()
        if tool_name in PUBLIC_TOOLS:
            return tool_name

    endpoint = request.path.rsplit("/", 1)[-1]
    if endpoint in PUBLIC_TOOLS:
        return endpoint
    if endpoint == "capture":
        return "capture_head_camera"
    if endpoint == "plan":
        return _PLAN_MODE_TO_PUBLIC_TOOL.get(
            str(body.get("mode") or "").strip(),
            "",
        )
    return ""


def _payload_arm(result: dict[str, Any], body: dict[str, Any]) -> str:
    candidates: list[Any] = [result.get("arm")]
    for key in ("result", "terminal_result"):
        nested = result.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("arm"))
    candidates.append(body.get("arm"))
    for candidate in candidates:
        arm = str(candidate or "").strip().lower()
        if arm in {"left", "right"}:
            return arm
    return ""


def _failure_capture_skill(
    tool_name: str,
    result: dict[str, Any],
    body: dict[str, Any],
) -> str | None:
    policy = OFFICIAL_V2_CAPTURE_POLICIES.get(tool_name)
    mode = str((policy or {}).get("failure") or _CAPTURE_NONE)
    if mode == _CAPTURE_NONE:
        return None
    if mode == _CAPTURE_HEAD_PATH:
        return "capture_head_camera"
    if mode == _CAPTURE_LEFT_WRIST_GRASP:
        return "capture_left_wrist_camera"
    if mode == _CAPTURE_RIGHT_WRIST_GRASP:
        return "capture_right_wrist_camera"
    if mode == _CAPTURE_REQUEST_ARM_WRIST_GRASP:
        arm = str(body.get("arm") or result.get("arm") or "right").strip().lower()
        return (
            f"capture_{arm}_wrist_camera"
            if arm in {"left", "right"}
            else "capture_head_camera"
        )
    if mode == _CAPTURE_RESULT_ARM_WRIST_GRASP:
        arm = _payload_arm(result, body)
        return (
            f"capture_{arm}_wrist_camera"
            if arm
            else "capture_head_camera"
        )
    raise RuntimeError(f"unsupported official_v2 capture mode: {mode}")


def _has_expected_exit_capture(
    result: dict[str, Any],
    capture_skill: str,
) -> bool:
    observation = result.get("observation")
    if not _observation_has_image(observation):
        return False
    declared_skill = str(result.get("exit_capture_skill") or "").strip()
    if declared_skill == "capture":
        declared_skill = "capture_head_camera"
    feed = str(observation.get("feed") or "").strip().lower()
    return bool(
        declared_skill == capture_skill
        or feed in _CAPTURE_SKILL_FEEDS[capture_skill]
    )


def _attach_official_capture(
    runtime: OfficialPolicyRuntime,
    result: dict[str, Any],
    *,
    session_id: str,
    timeout_s: float,
    capture_skill: str,
) -> bool:
    """Attach one fresh evaluator-backed observation from a test capture tool."""

    from behavior_interface.web import _promote_post_action_observation_fields

    result["exit_capture_attempted"] = True
    try:
        capture_request_id = runtime.server.submit_skill(
            capture_skill,
            {"session_id": session_id},
        )
        observation = runtime.server.wait_for_skill_result(
            capture_skill,
            timeout_s=timeout_s,
            request_id=capture_request_id,
        )
    except Exception as exc:
        result["observation"] = None
        result["capture_error"] = str(exc)
        return False
    if not _observation_has_image(observation):
        result["observation"] = None
        result["capture_error"] = str(
            (observation or {}).get("error")
            or f"{capture_skill} returned no image"
        )
        return False
    observation = dict(observation)
    observation.setdefault("feed", _CAPTURE_SKILL_PRIMARY_FEED[capture_skill])
    result["observation"] = observation
    result["exit_capture_skill"] = capture_skill
    result.pop("capture_error", None)
    _promote_post_action_observation_fields(result)
    return True


def _attach_official_head_capture(
    runtime: OfficialPolicyRuntime,
    result: dict[str, Any],
    *,
    session_id: str,
    timeout_s: float,
) -> bool:
    return _attach_official_capture(
        runtime,
        result,
        session_id=session_id,
        timeout_s=timeout_s,
        capture_skill="capture_head_camera",
    )


def install_official_adjust_height_route(
    app: Flask,
    runtime: OfficialPolicyRuntime,
) -> None:
    """Expose LUT workspace saturation without changing the legacy v2 module."""

    endpoint = "api_v2_adjust_hight"
    if endpoint not in app.view_functions:
        return

    def official_adjust_height():
        body = request.get_json(force=True, silent=True) or {}
        session_id = str(body.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"ok": False, "error": "session_id is required"}), 400
        upward_raw = body.get(
            "upward",
            body.get("upwardm", body.get("upward_m", 0.0)),
        )
        try:
            if isinstance(upward_raw, (bool, np.bool_)):
                raise ValueError
            upward = float(upward_raw)
            wait_timeout_s = float(body.get("timeout_s", 120.0))
            capture_timeout_s = float(body.get("capture_timeout_s", 120.0))
        except (TypeError, ValueError):
            return jsonify({
                "ok": False,
                "error": "upward and timeouts must be numeric",
            }), 400
        if not math.isfinite(upward):
            return jsonify({"ok": False, "error": "upward must be finite"}), 400
        if (
            not math.isfinite(wait_timeout_s)
            or not 0.1 <= wait_timeout_s <= 600.0
            or not math.isfinite(capture_timeout_s)
            or not 0.1 <= capture_timeout_s <= 600.0
        ):
            return jsonify({
                "ok": False,
                "error": "timeouts must be finite and in 0.1..600s",
            }), 400

        request_id = None
        try:
            request_id = runtime.server.submit_skill(
                "adjust_height",
                {"upward": upward},
            )
            raw_result = runtime.server.wait_for_skill_result(
                "adjust_height",
                timeout_s=wait_timeout_s,
                request_id=request_id,
            )
        except TimeoutError as exc:
            cancel_request = getattr(runtime, "cancel_public_job", None)
            cancelled = bool(
                callable(cancel_request)
                and request_id
                and cancel_request(request_id)
            )
            return jsonify({
                "ok": False,
                "error": str(exc),
                "tool": "adjust_height",
                "request_id": request_id,
                "request_scoped_cancellation": cancelled,
            }), 504
        except Exception as exc:
            return jsonify({
                "ok": False,
                "error": str(exc),
                "tool": "adjust_height",
            }), 400

        result = dict(raw_result or {})
        result.setdefault("tool", "adjust_height")
        _attach_official_head_capture(
            runtime,
            result,
            session_id=session_id,
            timeout_s=capture_timeout_s,
        )
        return jsonify(result), (200 if result.get("ok") else 400)

    official_adjust_height.__name__ = "official_adjust_height"
    app.view_functions[endpoint] = official_adjust_height

    original_tools = app.view_functions.get("api_v2_tools")
    if original_tools is None:
        return

    def official_tools_with_height_saturation():
        response = app.make_response(original_tools())
        payload = response.get_json(silent=True) or {}
        tools = list(payload.get("tools") or [])
        for item in tools:
            if item.get("name") == "adjust_height":
                item["desc"] = (
                    "只调整机器人高度 upward，单位米；超出 submission-local "
                    "LUT 可达范围时自动饱和到最高或最低缓存行，完成后拍 head camera。"
                )
                item["workspace_saturation"] = "submission_local_trunk_lut"
                break
        payload["tools"] = tools
        return jsonify(payload)

    app.view_functions["api_v2_tools"] = (
        official_tools_with_height_saturation
    )


def _expose_successful_plan_preview(result: dict[str, Any]) -> None:
    """Expose the red plan overlay through a media field preferred by MCP."""

    visualization = result.get("gripper_visualization")
    visualization_path = (
        str(visualization.get("path") or "").strip()
        if isinstance(visualization, dict) and visualization.get("ok") is True
        else ""
    )
    preview = (
        visualization_path
        or str(result.get("render_image_path") or "").strip()
        or str(result.get("render_image") or "").strip()
    )
    if preview:
        result["marked_image_url"] = preview
        result["plan_preview_overlay"] = "red_gripper_on_frozen_head"


def _uses_plan_preview_contract(
    tool_name: str,
    payload: Mapping[str, Any],
    body: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether this request/result owns a planned-pose preview."""

    if tool_name in _PLAN_PREVIEW_TOOLS:
        return True
    if tool_name != "move_tracked_point":
        return False
    mode = payload.get("execution_mode")
    if mode is None and body is not None:
        mode = body.get("execution_mode")
    return str(mode or "").strip().lower() == "plan"


def _strip_failed_plan_preview(result: dict[str, Any]) -> None:
    """A failed plan has no valid pose to render as a red gripper."""

    for key in (
        "marked_image_url",
        "render_image",
        "render_image_path",
        "rgb_main",
        "rgb_main_path",
        "rgb_overlay_path",
        "rgb_path",
    ):
        result.pop(key, None)
    debug_images = result.get("debug_images")
    if isinstance(debug_images, dict):
        debug_images = dict(debug_images)
        debug_images.pop("frozen_rgb_overlay", None)
        result["debug_images"] = debug_images
    for nested_key in ("result", "terminal_result"):
        nested = result.get(nested_key)
        if isinstance(nested, dict):
            _strip_failed_plan_preview(nested)
    result["plan_gripper_overlay_available"] = False


def install_official_v2_failure_capture_contract(
    app: Flask,
    runtime: OfficialPolicyRuntime,
) -> None:
    """Apply the per-tool model-visible capture contract to v2 results.

    The Codex REST adapter treats non-2xx responses as transport errors and does
    not run media extraction on their JSON bodies.  Operational failures are
    therefore returned as HTTP 200 with ``ok: false`` after any required capture.
    Request validation, strict-boundary rejection, and evaluator-disconnected
    responses retain their original HTTP status.
    """

    def is_boundary_failure(payload: dict[str, Any]) -> bool:
        error = str(payload.get("error") or "")
        return bool(
            payload.get("code")
            in {"evaluator_not_connected", "evaluator_control_required"}
            or any(marker in error for marker in _OFFICIAL_BOUNDARY_ERROR_MARKERS)
        )

    def is_operational_failure(
        payload: dict[str, Any],
        status_code: int,
    ) -> bool:
        tool = str(payload.get("tool") or "").strip()
        return bool(
            status_code >= 500
            or tool in PUBLIC_TOOLS
            or payload.get("job")
            or payload.get("failure_stage")
            or payload.get("timed_out")
            or payload.get("terminal_result") is not None
            or _observation_has_image(payload.get("observation"))
        )

    @app.after_request
    def apply_official_v2_capture_contract(response):
        if (
            request.method != "POST"
            or not request.path.startswith("/api/v2/")
            or not response.is_json
        ):
            return response
        payload = response.get_json(silent=True)
        if not isinstance(payload, dict):
            return response
        body = request.get_json(silent=True) or {}
        tool_name = _official_v2_request_tool(payload, body)
        uses_plan_preview = _uses_plan_preview_contract(
            tool_name,
            payload,
            body,
        )
        if payload.get("ok") is True:
            if uses_plan_preview:
                _expose_successful_plan_preview(payload)
                response.set_data(app.json.dumps(payload))
            return response
        if payload.get("ok") is not False:
            return response
        original_status = int(response.status_code)
        if is_boundary_failure(payload) or not is_operational_failure(
            payload,
            original_status,
        ):
            return response

        session_id = str(body.get("session_id") or "").strip()
        capture_skill = _failure_capture_skill(tool_name, payload, body)
        if uses_plan_preview:
            _strip_failed_plan_preview(payload)
        has_existing_exit_capture = bool(
            capture_skill
            and _has_expected_exit_capture(payload, capture_skill)
        )
        if capture_skill and not has_existing_exit_capture and not session_id:
            return response
        if has_existing_exit_capture:
            from behavior_interface.web import (
                _promote_post_action_observation_fields,
            )

            _promote_post_action_observation_fields(payload)
        elif capture_skill and session_id:
            try:
                capture_timeout_s = float(
                    body.get(
                        "capture_timeout_s",
                        _FAILURE_EXIT_CAPTURE_TIMEOUT_S,
                    )
                )
            except (TypeError, ValueError):
                capture_timeout_s = _FAILURE_EXIT_CAPTURE_TIMEOUT_S
            if not math.isfinite(capture_timeout_s) or capture_timeout_s <= 0.0:
                capture_timeout_s = _FAILURE_EXIT_CAPTURE_TIMEOUT_S
            _attach_official_capture(
                runtime,
                payload,
                session_id=session_id,
                timeout_s=min(capture_timeout_s, 600.0),
                capture_skill=capture_skill,
            )

        if uses_plan_preview:
            base_path = (
                payload.get("observation", {}).get("base_path_overlay")
                if isinstance(payload.get("observation"), dict)
                else None
            )
            payload["failure_capture_overlay"] = (
                "blue_base_path"
                if isinstance(base_path, dict) and base_path.get("ok") is True
                else "raw_head_fallback"
            )

        if original_status != 200:
            payload["failure_http_status"] = original_status
            payload["failure_transport"] = "application_result"
            response.status_code = 200
        response.set_data(app.json.dumps(payload))
        response.content_type = "application/json"
        return response


def install_official_wrist_roll_test_route(
    app: Flask,
    runtime: OfficialPolicyRuntime,
) -> None:
    """Expose the opt-in 8DOF J8 diagnostic tool on test interfaces only."""
    if not WRIST_ROLL_TEST_TOOL_ENABLED:
        return

    @app.post("/api/v2/control_wrist_roll")
    def official_control_wrist_roll():
        body = request.get_json(force=True, silent=True) or {}
        args = {
            key: body[key]
            for key in ("arm", "mode", "angle_deg", "timeout_s")
            if key in body
        }
        try:
            timeout_s = float(body.get("timeout_s", 15.0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "timeout_s must be numeric"}), 400
        if not math.isfinite(timeout_s) or not 0.2 <= timeout_s <= 60.0:
            return jsonify({
                "ok": False,
                "error": "timeout_s must be finite and in 0.2..60.0s",
            }), 400
        request_id = None
        try:
            request_id = runtime.server.submit_skill(
                "control_wrist_roll",
                args,
            )
            result = runtime.server.wait_for_skill_result(
                "control_wrist_roll",
                timeout_s=timeout_s + 10.0,
                request_id=request_id,
            )
        except TimeoutError as exc:
            cancel_request = getattr(runtime, "cancel_public_job", None)
            if callable(cancel_request) and request_id:
                cancel_request(request_id)
            return jsonify({"ok": False, "error": str(exc)}), 504
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(result), (200 if result.get("ok") else 400)

    original_tools = app.view_functions.get("api_v2_tools")
    if original_tools is None:
        return
    metadata = {
        "name": "control_wrist_roll",
        "endpoint": "/api/v2/control_wrist_roll",
        "args": [
            {
                "name": "arm",
                "type": "string",
                "widget": "select",
                "required": False,
                "default": "right",
                "options": ["left", "right"],
            },
            {
                "name": "mode",
                "type": "string",
                "widget": "select",
                "required": False,
                "default": "relative",
                "options": ["relative", "absolute", "reset"],
            },
            {
                "name": "angle_deg",
                "type": "number",
                "widget": "number",
                "required": False,
                "default": 0.0,
                "unit": "deg",
                "minimum": -360.0,
                "maximum": 360.0,
            },
            {
                "name": "timeout_s",
                "type": "number",
                "widget": "number",
                "required": False,
                "default": 15.0,
                "unit": "s",
                "minimum": 0.2,
                "maximum": 60.0,
            },
        ],
        "desc": (
            "Test-only J8 wrist-roll control. Relative commands accumulate from "
            "the observed angle, absolute commands use a joint angle, and reset "
            "returns J8 to zero. Only the selected J8 moves; J1-J7 and the other "
            "arm stay fixed, then J8 is held at the resulting target."
        ),
        "test_only": True,
    }

    def official_v2_tools_with_wrist_roll():
        response = app.make_response(original_tools())
        payload = response.get_json(silent=True) or {}
        tools = [
            item
            for item in list(payload.get("tools") or [])
            if item.get("name") != metadata["name"]
        ]
        tools.append(deepcopy(metadata))
        payload["tools"] = tools
        return jsonify(payload)

    app.view_functions["api_v2_tools"] = official_v2_tools_with_wrist_roll


def _official_http_port() -> int | None:
    raw = os.environ.get("BEHAVIOR_EVAL_TEST_PORT", "").strip()
    if raw.isdigit():
        return int(raw)
    if has_request_context():
        try:
            value = int(request.environ.get("SERVER_PORT") or 0)
        except (TypeError, ValueError):
            value = 0
        if value:
            return value
    return None


def _parse_reset_instance_id(body: Mapping[str, Any] | None) -> Any:
    if not isinstance(body, Mapping) or "instance_id" not in body:
        return "_unset"
    value = body.get("instance_id")
    if value is None or str(value).strip().lower() in ("random", "rand", ""):
        return "random"
    return int(value)


def install_strict_http_control_boundary(
    app: Flask,
    *,
    evaluator_connected: Optional[Callable[[], bool]] = None,
) -> None:
    """Disable legacy control routes that require evaluator ownership."""

    def reject(operation: str):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": (
                        f"strict official mode does not own {operation}; "
                        "the OmniGibson Evaluator is the sole task/simulator owner"
                    ),
                    "code": "evaluator_control_required",
                }
            ),
            409,
        )

    def official_operator_reset():
        # 官方口不能自己 env.reset()；把请求交给 evaluator.step 侧信道。
        from flask import current_app

        from behavior_interface.operator_controls import eval_control_snapshot
        from behavior_interface_eval_test.operator_scene_control import (
            listener_ready,
            write_reset_request,
        )

        port = _official_http_port()
        if port is None:
            return reject("reset")
        if not listener_ready(port):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": (
                            "官方 evaluator 还不会收手动 reset。"
                            "需要重启本口 evaluator（新代码在 step 里轮询请求文件）"
                        ),
                        "code": "evaluator_operator_unavailable",
                    }
                ),
                409,
            )
        body = request.get_json(force=True, silent=True) or {}
        try:
            parsed = _parse_reset_instance_id(body)
        except (TypeError, ValueError):
            return jsonify(
                {"ok": False, "error": f"instance_id 非法: {body.get('instance_id')!r}"}
            ), 400
        instance_id = None if parsed == "_unset" else parsed
        if instance_id == "random":
            snapshot = eval_control_snapshot(official=True)
            ids = list(snapshot.get("instance_ids") or [])
            if not ids:
                return jsonify({"ok": False, "error": "没有可随机的 instance"}), 409
            import random

            instance_id = int(random.choice(ids))
        receipt = write_reset_request(port, instance_id)
        recorder = current_app.extensions.get("human_trajectory_recorder")
        if recorder is not None:
            try:
                recorder.stop_all(outcome="aborted", reason="world_reset")
            except Exception:
                pass
        store = current_app.extensions.get("agent_monitor_store")
        if store is not None:
            try:
                store.on_world_reset()
            except Exception:
                pass
        return jsonify(
            {
                "ok": True,
                "pending": True,
                "mode": "evaluator_operator",
                "owner": "omnigibson_evaluator",
                "request_id": receipt["request_id"],
                "instance_id": receipt["instance_id"],
                "code": "evaluator_operator_reset",
            }
        )

    if "api_reset" in app.view_functions:
        app.view_functions["api_reset"] = official_operator_reset
    if "api_task_switch" in app.view_functions:
        app.view_functions["api_task_switch"] = lambda: reject("task switch")
    if "api_skills_reload" in app.view_functions:
        app.view_functions["api_skills_reload"] = lambda: reject("skill reload")
    if "api_skills_version" in app.view_functions:

        def official_skills_version():
            # The legacy endpoint imports simulator-backed adjust modules only
            # to read build IDs. Importing them also runs their decorators and
            # used to overwrite the test registry while serving this GET.
            try:
                from behavior_interface.rtabmap_slam.live import (
                    BACKEND as rtabmap_backend,
                    BUILD as rtabmap_build,
                    live_backend_selected,
                )

                if live_backend_selected():
                    spatial_map_backend = str(rtabmap_backend)
                    spatial_map_build = str(rtabmap_build)
                else:
                    from behavior_interface import spatial_map

                    spatial_map_backend = str(
                        getattr(spatial_map, "BACKEND", "unknown")
                    )
                    spatial_map_build = str(
                        getattr(spatial_map, "BUILD", "unknown")
                    )
            except Exception:
                spatial_map_backend = "import_failed"
                spatial_map_build = "import_failed"
            return jsonify(
                {
                    "ok": True,
                    "tool_version": OFFICIAL_TOOL_VERSION,
                    "robot": ROBOT_MODEL,
                    "robot_profile": ROBOT_PROFILE,
                    "robot_dof": ARM_DOF,
                    "adjust_head_frame_build": ADJUST_EEF_LOCAL_BUILD,
                    "adjust_wrist_frame_build": ADJUST_EEF_LOCAL_BUILD,
                    "adjust_height_build": ADJUST_HEIGHT_IMPLEMENTATION_VERSION,
                    "move_to_reach_pre_lift_build": (
                        MOVE_TO_REACH_PRE_LIFT_IMPLEMENTATION_VERSION
                    ),
                    "spatial_map_backend": spatial_map_backend,
                    "spatial_map_build": spatial_map_build,
                    "registry_owner": (
                        "behavior_interface_eval_test.tool.official_v2"
                    ),
                    "simulator_handle": False,
                    "wrist_roll_test_tool_enabled": (
                        WRIST_ROLL_TEST_TOOL_ENABLED
                    ),
                }
            )

        app.view_functions["api_skills_version"] = official_skills_version
    if "api_camera" in app.view_functions:
        original_camera = app.view_functions["api_camera"]

        def strict_camera():
            if request.method == "GET":
                return original_camera()
            return reject("simulator camera control")

        app.view_functions["api_camera"] = strict_camera

    if evaluator_connected is not None:

        @app.before_request
        def reject_tool_submission_without_evaluator():
            is_tool_submission = request.method == "POST" and (
                request.path == "/api/skill"
                or request.path.startswith("/api/v2/")
            )
            if not is_tool_submission or evaluator_connected():
                return None
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": (
                            "official evaluator is not connected; tool actions "
                            "require an active evaluator observation/action loop"
                        ),
                        "code": "evaluator_not_connected",
                    }
                ),
                409,
            )

    @app.after_request
    def preserve_official_tool_boundary_status(response):
        if request.method != "POST" or not request.path.startswith("/api/v2/"):
            return response
        if int(response.status_code) != 200:
            return response
        payload = response.get_json(silent=True)
        if not isinstance(payload, dict) or payload.get("ok") is not False:
            return response
        error = str(payload.get("error", ""))
        if any(marker in error for marker in _OFFICIAL_BOUNDARY_ERROR_MARKERS):
            response.status_code = 400
        return response


def install_live_unit_test_routes(
    app: Flask,
    runtime: OfficialPolicyRuntime,
) -> None:
    """Install the opt-in development live-test surface.

    These routes are outside ``/api/v2`` on purpose.  They are useful while
    developing a local test module against a running evaluator, but they are
    not advertised as challenge tools and never expose a simulator handle.
    Each ``POST`` reloads the fixed test-local module before constructing a
    runner, so editing it does not require an interface restart.
    """

    @app.post("/__official__/dev/live/move_tracked_point")
    def dev_live_move_tracked_point():
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify({
                "ok": False,
                "error": "request body must be a JSON object",
                "failure_stage": "input validation",
            }), 400
        try:
            # The browser/form and the official REST adapter omit optional
            # collection fields when they are empty.  The development route
            # is also used by replay/benchmark clients, which commonly send
            # those fields as explicit ``[]``.  Treat an empty optional
            # collection exactly like an omitted one so the live route has
            # the same input semantics as /api/v2/move_tracked_point; a
            # non-empty quick_constraints array still goes through the strict
            # plural-group validator below.
            normalized_body = dict(body)
            for optional_key in ("quick_constraints", "relations", "inequalities"):
                if normalized_body.get(optional_key) == []:
                    normalized_body.pop(optional_key, None)
            # Validate before checking evaluator readiness so malformed input
            # consistently reports HTTP 400 even when the bridge is offline.
            normalized_body = validate_move_tracked_point_args(normalized_body)
        except (TypeError, ValueError) as exc:
            return jsonify({
                "ok": False,
                "tool": "live_move_tracked_point_test",
                "error": str(exc),
                "failure_stage": "input validation",
            }), 400
        try:
            status = runtime.start_live_move_tracked_point_test(normalized_body)
        except (TypeError, ValueError) as exc:
            return jsonify({
                "ok": False,
                "tool": "live_move_tracked_point_test",
                "error": str(exc),
                "failure_stage": "input validation",
            }), 400
        except RuntimeError as exc:
            return jsonify({
                "ok": False,
                "tool": "live_move_tracked_point_test",
                "error": str(exc),
                "failure_stage": "observation validation",
            }), 409
        except Exception as exc:
            return jsonify({
                "ok": False,
                "tool": "live_move_tracked_point_test",
                "error": f"{type(exc).__name__}: {exc}",
                "failure_stage": "planning",
            }), 500
        return jsonify(status), 202

    @app.get("/__official__/dev/live/status")
    def dev_live_status():
        return jsonify(runtime.live_test_status())

    @app.post("/__official__/dev/live/cancel")
    def dev_live_cancel():
        body = request.get_json(force=True, silent=True) or {}
        reason = str(body.get("reason") or "cancelled by user")
        return jsonify(runtime.cancel_live_test(reason))

    @app.post("/__official__/dev/live/reload")
    def dev_live_reload():
        try:
            receipt = runtime.reload_official_v2_tool_stack()
        except RuntimeError as exc:
            return jsonify({
                "ok": False,
                "tool": "official_v2_hot_reload",
                "error": str(exc),
            }), 409
        except Exception as exc:
            return jsonify({
                "ok": False,
                "tool": "official_v2_hot_reload",
                "error": f"{type(exc).__name__}: {exc}",
            }), 500
        return jsonify(receipt)

    @app.get("/__official__/dev/live/reload/status")
    def dev_live_reload_status():
        return jsonify(runtime.official_v2_reload_status())

def start_http(runtime: OfficialPolicyRuntime, host: str, port: int) -> threading.Thread:
    from behavior_interface.web import build_app

    runtime.server.web_port = int(port)
    app = build_app(runtime.server)
    install_rollout_budget_routes(app, runtime)
    from official_eval_harness.session_guard import install_session_guard

    install_session_guard(app, port, os.environ.get("BEHAVIOR_EVAL_SESSION_GUARD", ""))
    runtime._official_http_app = app
    install_strict_http_control_boundary(
        app,
        evaluator_connected=runtime.evaluator_ready,
    )
    install_official_adjust_height_route(app, runtime)
    install_official_live_frame_routes(app, runtime)
    install_official_head_adjust_routes(app, runtime)
    install_official_set_arm_route(app, runtime)
    install_official_track_object_distance_routes(app, runtime)
    install_official_move_tracked_point_routes(app, runtime)
    install_official_surface_facing_route(app, runtime)
    install_official_wrist_roll_test_route(app, runtime)
    install_official_v2_failure_capture_contract(app, runtime)
    install_live_unit_test_routes(app, runtime)
    install_track_object_distance_human_ui(app)

    @app.get("/__official__/idle_probe")
    def official_idle_probe():
        # Keep this endpoint constant-size and side-effect free.  The idle
        # gate polls it while the evaluator is waiting for a policy response;
        # the browser-facing health/state endpoints remain unchanged.
        return jsonify(runtime.idle_probe())

    @app.get("/__official__/healthz")
    def official_healthz():
        payload = {"ok": True, **runtime.status()}
        payload["session_isolation"] = app.extensions["official_session_guard"]
        port = _official_http_port()
        listener = False
        try:
            from behavior_interface_eval_test.operator_scene_control import listener_ready

            listener = bool(port and listener_ready(port))
        except Exception:
            listener = False
        payload["operator_controls"] = {
            "stop_session": True,
            "reset": "evaluator_operator",
            "listener_ready": listener,
        }
        return jsonify(payload)

    @app.get("/__official__/architecture")
    def official_architecture():
        return jsonify(
            {
                "ok": True,
                "layers": [
                    "OmniGibson Evaluator v3.9.1",
                    "Behavior Interface observation/action gateway",
                    "Human web client or downstream Model Policy Server",
                ],
                "simulator_handle_in_interface": False,
                "tool_version": runtime.tool_version,
                "ui_tool_version": runtime.ui_tool_version,
                "frontend": "existing behavior_interface/templates/index.html",
            }
        )

    @app.get("/__official__/tools")
    def official_tools():
        return jsonify({"ok": True, **capability_report()})

    def run() -> None:
        app.run(
            host=host,
            port=port,
            threaded=True,
            debug=False,
            use_reloader=False,
        )

    thread = threading.Thread(target=run, name="official-interface-http", daemon=True)
    thread.start()
    return thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official BEHAVIOR evaluator policy endpoint with existing Interface UI."
    )
    parser.add_argument("--task", default=os.environ.get("TASK", "make_microwave_popcorn"))
    parser.add_argument("--scene", default=os.environ.get("SCENE", "house_double_floor_lower"))
    parser.add_argument(
        "--robot-dof",
        type=int,
        choices=(ARM_DOF,),
        default=ARM_DOF,
    )
    parser.add_argument(
        "--tool-version",
        choices=(OFFICIAL_TOOL_VERSION,),
        default=os.environ.get(
            "BEHAVIOR_EVAL_TEST_TOOL_VERSION",
            OFFICIAL_TOOL_VERSION,
        ),
    )
    parser.add_argument(
        "--ui-tool-version",
        choices=("v2",),
        default=os.environ.get("INTERFACE_TOOL_VERSION", "v2"),
    )
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=18081)
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=15060)
    parser.add_argument("--downstream-host")
    parser.add_argument("--downstream-port", type=int)
    parser.add_argument("--downstream-scheme", choices=("ws", "wss"), default="ws")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Keep direct Python launches subject to the same port-to-GPU contract as
    # the shell launcher.  This runs before runtime/Omni imports can create a
    # CUDA context, so an accidental cross-card environment fails closed.
    try:
        owned_gpu = enforce_fixed_gpu_environment(args.http_port)
    except ValueError as exc:
        raise SystemExit(f"official interface GPU ownership validation failed: {exc}") from exc
    if owned_gpu is not None:
        print(
            f"Official interface port={args.http_port} physical_gpu={owned_gpu} "
            "local_cuda=cuda:0",
            flush=True,
        )
    try:
        apply_requested_cpu_affinity()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"official interface CPU affinity validation failed: {exc}"
        ) from exc
    # 必须在 import server.py 之前按 HTTP 口建叶子。否则所有官方口都会
    # 落到共享的 port5000_，互相 prune 之后 plan_grasp 会永久 FileNotFoundError。
    from behavior_interface.runtime_tmp import configure_process_runtime_tmp

    os.environ.setdefault("PORT", str(args.http_port))
    os.environ.setdefault("BEHAVIOR_EVAL_TEST_PORT", str(args.http_port))
    configure_process_runtime_tmp(args.http_port)

    from behavior_interface_eval_test.rgbd_lite_verified_runtime import (
        install_verified_rgbd_lite_runtime,
    )

    rgbd_lite_runtime = install_verified_rgbd_lite_runtime()
    print(
        "Official RGB-D Lite runtime "
        f"id={rgbd_lite_runtime['runtime_id']} "
        f"ik_gpu={rgbd_lite_runtime['ik_gpu']}",
        flush=True,
    )
    downstream = None
    if args.downstream_host or args.downstream_port:
        if not args.downstream_host or not args.downstream_port:
            raise ValueError("--downstream-host and --downstream-port must be set together")
        downstream = DownstreamPolicyClient(
            args.downstream_host,
            args.downstream_port,
            scheme=args.downstream_scheme,
        )

    runtime = OfficialPolicyRuntime(
        task=args.task,
        scene=args.scene,
        robot_dof=args.robot_dof,
        tool_version=args.tool_version,
        ui_tool_version=args.ui_tool_version,
        downstream=downstream,
        rgbd_lite_runtime=rgbd_lite_runtime,
    )
    start_http(runtime, args.http_host, args.http_port)
    print(
        "Official Behavior Interface "
        f"tool={args.tool_version} ui={args.ui_tool_version} "
        f"http=http://{args.http_host}:{args.http_port} "
        f"evaluator_policy=ws://{args.policy_host}:{args.policy_port}",
        flush=True,
    )
    asyncio.run(serve_policy(runtime, args.policy_host, args.policy_port))


if __name__ == "__main__":
    main()
