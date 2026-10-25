"""Observation-only persistent named point tracking for RoboDojo RGB-D."""

from __future__ import annotations

from copy import deepcopy
import threading
from typing import Any

import numpy as np

from tools.util.geometry import (
    finite_vector,
    relative_uv_to_pixel,
    robodojo_env_to_robot_base,
    unproject_relative_uv,
)
from tools.util.observation import ObservationAdapter
from tools.util.storage import Snapshot


TRACKED_OBJECT_DISTANCES_MEMORY_KEY = "tracked_object_distances"
TRACKING_POLICY_MEMORY_KEY = "tracked_object_distance_tracking"
OBSERVED = "observed"
TEMPORARILY_UNOBSERVED = "temporarily_unobserved"
LOST = "lost"
TRACKING_CAMERA_ROLES = ("head", "left_wrist", "right_wrist")


class PointTracker:
    """Keep BEHAVIOR-compatible live distance memory over RoboDojo frames."""

    def __init__(
        self,
        *,
        base_rotation: Any | None = None,
        max_unobserved_steps: int = 8,
    ) -> None:
        if int(max_unobserved_steps) < 0:
            raise ValueError("max_unobserved_steps cannot be negative")
        self._lock = threading.RLock()
        self._tracks: dict[str, dict[str, Any]] = {}
        self._base_rotation = (
            None
            if base_rotation is None
            else np.asarray(base_rotation, dtype=np.float64).reshape(3, 3).copy()
        )
        self._max_unobserved_steps = int(max_unobserved_steps)
        self._next_track_id = 1
        self._last_deleted: list[dict[str, Any]] = []
        self._last_error: str | None = None

    @staticmethod
    def _key(session_id: str, image_id: str) -> str:
        return f"{session_id}/{image_id}"

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._next_track_id = 1
            self._last_deleted = []
            self._last_error = None

    def _new_track_id(self) -> str:
        result = f"track_{self._next_track_id:06d}"
        self._next_track_id += 1
        return result

    def _robot_base_xyz(self, robodojo_env_xyz: Any) -> list[float]:
        return robodojo_env_to_robot_base(
            robodojo_env_xyz,
            self._base_rotation,
        ).tolist()

    def _observed_entry(
        self,
        *,
        name: str,
        track_id: str,
        pixel_uv: list[float],
        relative_uv: list[float],
        depth_m: float,
        robodojo_env_xyz: Any,
        observation_sequence: int,
        episode_id: str,
        camera_role: str,
        snapshot: Snapshot | None,
    ) -> dict[str, Any]:
        env_xyz = finite_vector(
            robodojo_env_xyz,
            3,
            "tracked RoboDojo env XYZ",
        )
        u, v = [float(value) for value in relative_uv]
        entry = {
            "name": str(name),
            "depth_m": float(depth_m),
            "unit": "m",
            "u": u,
            "v": v,
            "pixel_uv": [float(value) for value in pixel_uv],
            "relative_uv": [u, v],
            "xyz_in_robot_base_coord_m": self._robot_base_xyz(env_xyz),
            "xyz_frame": "current_robot_base",
            "xyz_axes": "x_forward_y_left_z_up",
            "track_id": str(track_id),
            "status": OBSERVED,
            "camera": str(camera_role),
            "camera_role": str(camera_role),
            "confidence": 1.0,
            "observation_sequence": int(observation_sequence),
            "last_seen_sequence": int(observation_sequence),
            "episode_id": str(episode_id),
            "depth_source": "current_evaluator_depth_linear",
            "xyz_source": ("current_evaluator_depth_linear_and_camera_matrices"),
            "identity_source": "agent_annotation",
            "identity_verified": False,
            "observed": True,
            "retired": False,
            "_tracking_pixel_uv": [float(value) for value in pixel_uv],
            "_xyz_in_robodojo_env_m": env_xyz.tolist(),
            "_missed_steps": 0,
        }
        if snapshot is not None:
            entry.update(
                {
                    "source_session_id": snapshot.session_id,
                    "source_image_id": snapshot.image_id,
                    "source_observation_sequence": int(snapshot.sequence),
                    "initial_grounding": (f"policy_owned_frozen_{camera_role}_capture"),
                }
            )
        return entry

    @staticmethod
    def _public_entry(entry: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(
            {key: value for key, value in entry.items() if not str(key).startswith("_")}
        )

    def register(
        self,
        snapshot: Snapshot,
        points: list[dict[str, Any]],
    ) -> dict[str, Any]:
        camera_role = str(snapshot.role)
        if camera_role not in TRACKING_CAMERA_ROLES:
            raise ValueError("track_object_distance requires a supported RGB-D capture")
        if (
            snapshot.depth is None
            or snapshot.intrinsic_matrix is None
            or snapshot.extrinsic_matrix is None
        ):
            raise ValueError(
                f"frozen {camera_role} capture has no complete RGB-D camera geometry"
            )
        height, width = snapshot.rgb.shape[:2]
        entries: dict[str, Any] = {}
        for point in points:
            name = str(point["name"])
            px, py = relative_uv_to_pixel(point["u"], point["v"], width, height)
            xyz, metadata = unproject_relative_uv(
                snapshot.depth,
                snapshot.intrinsic_matrix,
                snapshot.extrinsic_matrix,
                point["u"],
                point["v"],
            )
            entries[name] = self._observed_entry(
                name=name,
                track_id=self._new_track_id(),
                pixel_uv=[float(px), float(py)],
                relative_uv=[float(point["u"]), float(point["v"])],
                depth_m=float(metadata["depth_m"]),
                robodojo_env_xyz=xyz,
                observation_sequence=snapshot.sequence,
                episode_id=snapshot.episode_id,
                camera_role=camera_role,
                snapshot=snapshot,
            )

        with self._lock:
            replaced_names = [
                name
                for record in self._tracks.values()
                for name, item in record["points"].items()
                if not bool(item.get("retired"))
            ]
            self._last_deleted = [
                {"name": name, "reason": "tracking_replaced"} for name in replaced_names
            ]
            record = {
                "session_id": snapshot.session_id,
                "image_id": snapshot.image_id,
                "episode_id": snapshot.episode_id,
                "capture_sequence": int(snapshot.sequence),
                "last_sequence": int(snapshot.sequence),
                "last_rgb": snapshot.rgb.copy(),
                "camera_role": camera_role,
                "points": entries,
                "replaced_names": replaced_names,
                "frame_digest": str(snapshot.integrity_digest or ""),
            }
            # BEHAVIOR replace_points semantics: one call atomically replaces
            # the previously registered named set.
            self._tracks = {self._key(snapshot.session_id, snapshot.image_id): record}
            self._last_error = None
            return self._public_result(record)

    @staticmethod
    def _gray(rgb: np.ndarray) -> np.ndarray:
        values = np.asarray(rgb, dtype=np.float32)
        return (
            0.299 * values[:, :, 0] + 0.587 * values[:, :, 1] + 0.114 * values[:, :, 2]
        ).astype(np.uint8)

    @staticmethod
    def _flow(
        previous: np.ndarray,
        current: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        flat = points.reshape(-1, 2)
        if previous.shape == current.shape and np.array_equal(previous, current):
            return flat, np.ones(flat.shape[0], dtype=bool)
        try:
            import cv2
        except ImportError:
            return flat, np.zeros(flat.shape[0], dtype=bool)

        moved, forward_status, _error = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            points,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if moved is None or forward_status is None:
            return flat, np.zeros(flat.shape[0], dtype=bool)
        returned, backward_status, _error = cv2.calcOpticalFlowPyrLK(
            current,
            previous,
            moved,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        if returned is None or backward_status is None:
            return moved.reshape(-1, 2), np.zeros(flat.shape[0], dtype=bool)
        error = np.linalg.norm(returned.reshape(-1, 2) - flat, axis=1)
        status = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(error)
            & (error <= 1.5)
        )
        return moved.reshape(-1, 2), status

    def _temporarily_unobserved(
        self,
        entry: dict[str, Any],
        *,
        observation_sequence: int,
        reason: str,
    ) -> None:
        missed_steps = int(entry.get("_missed_steps", 0)) + 1
        entry.update(
            {
                "depth_m": None,
                "u": None,
                "v": None,
                "pixel_uv": None,
                "relative_uv": None,
                "xyz_in_robot_base_coord_m": None,
                "status": TEMPORARILY_UNOBSERVED,
                "tracker_status": str(reason),
                "confidence": max(0.0, float(entry.get("confidence", 1.0)) * 0.7),
                "observation_sequence": int(observation_sequence),
                "observed": False,
                "_missed_steps": missed_steps,
            }
        )
        entry.pop("_xyz_in_robodojo_env_m", None)
        if missed_steps > self._max_unobserved_steps:
            entry.update({"status": LOST, "retired": True})
            self._last_deleted.append(
                {"name": entry["name"], "reason": "tracking_lost"}
            )

    def _retire_out_of_view(
        self,
        entry: dict[str, Any],
        *,
        observation_sequence: int,
    ) -> None:
        self._temporarily_unobserved(
            entry,
            observation_sequence=observation_sequence,
            reason="out_of_view",
        )
        entry.update({"status": LOST, "retired": True})
        self._last_deleted.append({"name": entry["name"], "reason": "out_of_view"})

    def update(self, adapter: ObservationAdapter) -> None:
        with self._lock:
            roles = {
                str(record.get("camera_role") or "head")
                for record in self._tracks.values()
            }
        cameras: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for role in roles:
            try:
                cameras[role] = adapter.camera(role, require_depth=True)
            except (KeyError, RuntimeError, ValueError) as exc:
                errors.append(f"{role}: {type(exc).__name__}: {exc}")
        with self._lock:
            for record in self._tracks.values():
                role = str(record.get("camera_role") or "head")
                camera = cameras.get(role)
                if camera is None:
                    continue
                current_rgb = camera["rgb"]
                current_gray = self._gray(current_rgb)
                if record["episode_id"] != camera["episode_id"]:
                    continue
                if int(camera["sequence"]) <= int(record["last_sequence"]):
                    continue
                previous_gray = self._gray(record["last_rgb"])
                live_names = [
                    name
                    for name, item in record["points"].items()
                    if not bool(item.get("retired"))
                ]
                if live_names:
                    old_points = np.asarray(
                        [
                            record["points"][name]["_tracking_pixel_uv"]
                            for name in live_names
                        ],
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)
                    new_points, statuses = self._flow(
                        previous_gray,
                        current_gray,
                        old_points,
                    )
                    for index, name in enumerate(live_names):
                        item = record["points"][name]
                        sequence = int(camera["sequence"])
                        if not bool(statuses[index]):
                            self._temporarily_unobserved(
                                item,
                                observation_sequence=sequence,
                                reason="occluded",
                            )
                            continue
                        px, py = [float(value) for value in new_points[index]]
                        height, width = current_gray.shape
                        if not (0.0 <= px <= width - 1 and 0.0 <= py <= height - 1):
                            self._retire_out_of_view(
                                item,
                                observation_sequence=sequence,
                            )
                            continue
                        u = 1000.0 * px / max(1, width - 1)
                        v = 1000.0 * py / max(1, height - 1)
                        try:
                            xyz, metadata = unproject_relative_uv(
                                camera["depth"],
                                camera["intrinsic_matrix"],
                                camera["extrinsic_matrix"],
                                u,
                                v,
                            )
                        except ValueError:
                            item["_tracking_pixel_uv"] = [px, py]
                            self._temporarily_unobserved(
                                item,
                                observation_sequence=sequence,
                                reason="invalid_depth",
                            )
                            continue
                        source = {
                            key: item[key]
                            for key in (
                                "source_session_id",
                                "source_image_id",
                                "source_observation_sequence",
                                "initial_grounding",
                            )
                            if key in item
                        }
                        updated = self._observed_entry(
                            name=name,
                            track_id=item["track_id"],
                            pixel_uv=[px, py],
                            relative_uv=[u, v],
                            depth_m=float(metadata["depth_m"]),
                            robodojo_env_xyz=xyz,
                            observation_sequence=sequence,
                            episode_id=camera["episode_id"],
                            camera_role=role,
                            snapshot=None,
                        )
                        updated.update(source)
                        record["points"][name] = updated
                record["last_rgb"] = current_rgb.copy()
                record["last_sequence"] = int(camera["sequence"])
            self._last_error = "; ".join(errors) if errors else None

    def _public_result(self, record: dict[str, Any]) -> dict[str, Any]:
        points = [self._public_entry(item) for item in record["points"].values()]
        entries = {
            str(point["name"]): point
            for point in points
            if not bool(point.get("retired"))
        }
        return {
            "session_id": record["session_id"],
            "image_id": record["image_id"],
            "episode_id": record["episode_id"],
            "capture_observation_sequence": int(record["capture_sequence"]),
            "observation_sequence": int(record["last_sequence"]),
            "replayed_observation_count": 0,
            "frame_binding": (
                f"policy_owned_current_frozen_{record.get('camera_role', 'head')}_capture"
            ),
            "camera_role": str(record.get("camera_role") or "head"),
            "frame_digest": record["frame_digest"],
            "replaced_names": list(record["replaced_names"]),
            "tracked_names": list(entries),
            "tracked_object_distances": entries,
            "points": points,
        }

    def result(self, session_id: str, image_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._tracks.get(self._key(session_id, image_id))
            if record is None:
                raise KeyError(f"no tracked points for {session_id}/{image_id}")
            return self._public_result(record)

    def robodojo_env_xyz(
        self,
        session_id: str,
        image_id: str,
        name: str,
    ) -> np.ndarray:
        with self._lock:
            record = self._tracks.get(self._key(session_id, image_id))
            if record is None:
                raise KeyError(f"no tracked points for {session_id}/{image_id}")
            entry = record["points"].get(str(name))
            if entry is None:
                raise KeyError(f"tracked point {name!r} is unavailable")
            if entry.get("status") != OBSERVED or bool(entry.get("retired")):
                raise ValueError(f"tracked point {name!r} is no longer observed")
            return finite_vector(
                entry.get("_xyz_in_robodojo_env_m"),
                3,
                f"tracked {name} RoboDojo env XYZ",
            ).copy()

    def all_results(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_result(record) for record in self._tracks.values()]

    def observed_active_points_snapshot(
        self,
        names: list[str] | tuple[str, ...],
        *,
        episode_id: str,
    ) -> dict[str, Any]:
        """Return one atomic, current-observation snapshot for live control.

        Stale, predicted, unobserved, retired, or cross-episode measurements
        are excluded. Motion tools must stop instead of steering from an old
        depth value.
        """

        requested = [str(name or "").strip() for name in names]
        if not requested or any(not name for name in requested):
            raise ValueError("names must contain non-empty point names")
        if len(set(requested)) != len(requested):
            raise ValueError("names must be unique")
        expected_episode = str(episode_id or "").strip()
        if not expected_episode:
            raise ValueError("episode_id is required")

        with self._lock:
            record = next(iter(self._tracks.values()), None)
            unavailable: dict[str, str] = {}
            reason: str | None = None
            if record is None:
                reason = "tracking_not_registered"
            elif str(record["episode_id"]) != expected_episode:
                reason = "episode_changed"

            entries: dict[str, dict[str, Any]] = {}
            if reason is None and record is not None:
                sequence = int(record["last_sequence"])
                for name in requested:
                    entry = record["points"].get(name)
                    if entry is None or bool(entry.get("retired")):
                        unavailable[name] = "point_not_registered_or_retired"
                        continue
                    if str(entry.get("status") or "") != OBSERVED:
                        unavailable[name] = "point_unobserved"
                        continue
                    if int(entry.get("observation_sequence", -1)) != sequence:
                        unavailable[name] = "stale_tracking_measurement"
                        continue
                    try:
                        xyz = np.asarray(
                            entry["xyz_in_robot_base_coord_m"], dtype=np.float64
                        ).reshape(3)
                        depth_m = float(entry["depth_m"])
                    except (KeyError, TypeError, ValueError):
                        unavailable[name] = "invalid_tracking_measurement"
                        continue
                    if (
                        not np.all(np.isfinite(xyz))
                        or not np.isfinite(depth_m)
                        or depth_m <= 0.0
                    ):
                        unavailable[name] = "invalid_tracking_measurement"
                        continue
                    entries[name] = self._public_entry(entry)
                if unavailable:
                    reason = "tracked_point_unavailable"

            return {
                "ok": reason is None,
                "reason": reason,
                "entries": entries,
                "unavailable": unavailable,
                "requested_names": requested,
                "observation_sequence": (
                    None if record is None else int(record["last_sequence"])
                ),
                "episode_id": None if record is None else record["episode_id"],
                "session_id": None if record is None else record["session_id"],
                "image_id": None if record is None else record["image_id"],
                "frame": "current_robot_base",
                "axes": "x_forward_y_left_z_up",
                "measurement_source": (
                    "current_evaluator_depth_linear_and_camera_matrices"
                ),
                "predictions_published": False,
            }

    def memory_fields(self) -> dict[str, Any]:
        with self._lock:
            entries: dict[str, dict[str, Any]] = {}
            camera_role = "head"
            for record in self._tracks.values():
                camera_role = str(record.get("camera_role") or "head")
                for name, item in record["points"].items():
                    if not bool(item.get("retired")):
                        entries[name] = self._public_entry(item)
            return {
                TRACKED_OBJECT_DISTANCES_MEMORY_KEY: entries,
                TRACKING_POLICY_MEMORY_KEY: {
                    "camera": camera_role,
                    "camera_role": camera_role,
                    "coordinate_system": (
                        "Qwen3-VL relative image coordinates 0..1000"
                    ),
                    "depth_unit": "m",
                    "depth_source": "current_evaluator_depth_linear",
                    "xyz_in_robot_base_coord_unit": "m",
                    "xyz_in_robot_base_coord_frame": "current_robot_base",
                    "xyz_in_robot_base_coord_axes": "x_forward_y_left_z_up",
                    "xyz_in_robot_base_coord_source": (
                        "current_evaluator_depth_linear_and_camera_matrices"
                    ),
                    "robodojo_env_axes": "x_right_y_forward_z_up",
                    "identity_source": "agent_annotation",
                    "identity_verified": False,
                    "initial_grounding": (f"policy_owned_frozen_{camera_role}_capture"),
                    "frame_alignment": "required_session_id_and_image_id",
                    "replay_policy": (
                        "current_capture_then_strict_consecutive_evaluator_observations"
                    ),
                    "persistent_tracking": True,
                    "tracking_lifetime": ("until_replaced_episode_reset_or_retired"),
                    "deletion_policy": (
                        "remove_measurement_when_unobserved_and_retire_track_"
                        "when_out_of_view_or_lost"
                    ),
                    "stale_depth_published": False,
                    "stale_xyz_in_robot_base_coord_published": False,
                    "observation_contract": [
                        f"vision.cam_{camera_role}.color",
                        f"vision.cam_{camera_role}.depth",
                        f"vision.cam_{camera_role}.intrinsic_matrix",
                        f"vision.cam_{camera_role}.extrinsic_matrix",
                    ],
                    "direct_simulator_mutation": False,
                },
            }

    def summary_fields(self) -> dict[str, Any]:
        fields = self.memory_fields()[TRACKED_OBJECT_DISTANCES_MEMORY_KEY]
        return {
            "tracked_object_distances": sum(
                entry.get("status") == OBSERVED for entry in fields.values()
            ),
            "tracked_object_distance_names": list(fields),
            "tracked_object_distance_registrations": len(fields),
        }

    def text(self) -> str:
        entries = self.memory_fields()[TRACKED_OBJECT_DISTANCES_MEMORY_KEY]
        if not entries:
            return "Tracked object distances: none."
        lines = [
            "Tracked object distances and XYZ in current robot base coord "
            f"(live {next(iter(entries.values())).get('camera', 'head')} RGB-D):"
        ]
        for name, entry in entries.items():
            if entry.get("status") == OBSERVED:
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
        fields = self.memory_fields()[TRACKED_OBJECT_DISTANCES_MEMORY_KEY]
        with self._lock:
            latest = next(iter(self._tracks.values()), None)
            return {
                "camera_role": (
                    None if latest is None else latest.get("camera_role", "head")
                ),
                "active": sum(
                    entry.get("status") == OBSERVED for entry in fields.values()
                ),
                "names": list(fields),
                "registered": len(fields),
                "registered_names": list(fields),
                "temporarily_unobserved": {
                    name: entry
                    for name, entry in fields.items()
                    if entry.get("status") == TEMPORARILY_UNOBSERVED
                },
                "observation_sequence": (
                    None if latest is None else int(latest["last_sequence"])
                ),
                "episode_id": None if latest is None else latest["episode_id"],
                "last_deleted": deepcopy(self._last_deleted),
                "last_error": self._last_error,
                "persistent_tracking": True,
            }
