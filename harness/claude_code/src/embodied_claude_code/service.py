from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import os
import re
import threading
import time
from typing import Any
import uuid

from .catalog import ToolCatalog
from .client import RestClient
from .rollout_budget import normalize_budget, unavailable_budget
from .config import Settings
from .coordinates import (
    VLM_IMAGE_COORDINATE_SYSTEM, arguments_to_interface, contains_uv,
    response_to_pixels,
)
from .errors import (
    CameraFrameError,
    EmbodiedError,
    EpisodeStateError,
    ToolPolicyError,
)
from .media import Media, MediaExtractor, safe_json
from .monitor_cards import publish_from_tool_result
from .profile import ToolProfile
from .recorder import TrajectoryRecorder


# 必须与 official_v2.contract._READ_DEPTH_SESSION_RE 一致。
# adapter 若放行更长 id，track_object_distance / read_depth /
# move_point_to_point 会在机器人已经开跑后才 400。
OFFICIAL_SESSION_ID_MAX_LEN = 64
SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
WRIST_CAPTURE_TOOLS = frozenset(
    {"capture_left_wrist_camera", "capture_right_wrist_camera"}
)
WRIST_CAPTURE_MAX_ATTEMPTS = 3
WRIST_MIN_VALID_DEPTH_RATIO = 0.25
PERSISTENT_TRACKING_MAX_TRACKS = 8
PERSISTENT_TRACKING_MAX_STRING_CHARS = 160
PERSISTENT_TRACKING_UNAVAILABLE_WARNING = (
    "Persistent tracking is unavailable because /api/memory could not be read "
    "after this tool call."
)

PERSISTENT_TRACKING_METADATA_FIELDS = (
    "camera",
    "coordinate_system",
    "depth_unit",
    "xyz_in_robot_base_coord_axes",
    "xyz_in_robot_base_coord_frame",
)
PERSISTENT_TRACKING_TEXT_FIELDS = (
    "camera",
    "episode_id",
    "source_image_id",
    "source_session_id",
    "status",
    "track_id",
)
PERSISTENT_TRACKING_BOOL_FIELDS = (
    "identity_verified",
    "lost",
    "valid",
)
PERSISTENT_TRACKING_NUMBER_FIELDS = (
    "confidence",
    "depth_m",
    "observation_sequence",
    "source_observation_sequence",
    "u",
    "v",
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _wrist_frame_quality(response: Any) -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None

    overlay = response.get("grasp_zone_overlay")
    overlay = overlay if isinstance(overlay, dict) else {}
    camera = response.get("camera")
    camera = camera if isinstance(camera, dict) else {}

    valid = _finite_number(
        response.get(
            "valid_depth_pixel_count",
            overlay.get("valid_depth_pixel_count"),
        )
    )
    width = _finite_number(response.get("image_width", camera.get("image_width")))
    height = _finite_number(
        response.get("image_height", camera.get("image_height"))
    )
    if valid is None or width is None or height is None or width <= 0 or height <= 0:
        return None

    total = width * height
    ratio = valid / total
    return {
        "image_id": str(response.get("image_id") or ""),
        "valid_depth_pixel_count": int(valid),
        "total_pixel_count": int(total),
        "valid_depth_ratio": round(ratio, 6),
        "acceptable": (
            0 <= valid <= total and ratio >= WRIST_MIN_VALID_DEPTH_RATIO
        ),
    }


def _tracking_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:PERSISTENT_TRACKING_MAX_STRING_CHARS] if text else None


def _tracking_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _tracking_xyz(value: Any) -> list[int | float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    coordinates = [_tracking_number(item) for item in value]
    if any(item is None for item in coordinates):
        return None
    return [item for item in coordinates if item is not None]


def _compact_persistent_tracking(memory: Any) -> dict[str, Any]:
    """Select only live point identity, binding, status, and coordinates."""
    if not isinstance(memory, dict):
        raise ValueError("/api/memory response must be an object")
    raw = memory.get("raw", memory)
    if not isinstance(raw, dict):
        raise ValueError("/api/memory raw field must be an object")

    compact: dict[str, Any] = {"available": True}
    metadata = raw.get("tracked_object_distance_tracking")
    if isinstance(metadata, dict):
        for field in PERSISTENT_TRACKING_METADATA_FIELDS:
            value = _tracking_text(metadata.get(field))
            if value is not None:
                compact[field] = value

    raw_tracks = raw.get("tracked_object_distances")
    if raw_tracks is None:
        raw_tracks = {}
    if not isinstance(raw_tracks, dict):
        raise ValueError("tracked_object_distances must be an object")

    tracks: dict[str, Any] = {}
    named_tracks = [
        (name, track)
        for name, track in raw_tracks.items()
        if isinstance(name, str) and isinstance(track, dict)
    ]
    named_tracks.sort(key=lambda item: item[0])
    for raw_name, track in named_tracks[:PERSISTENT_TRACKING_MAX_TRACKS]:
        name = _tracking_text(raw_name)
        if name is None:
            continue
        selected: dict[str, Any] = {}
        for field in PERSISTENT_TRACKING_TEXT_FIELDS:
            value = _tracking_text(track.get(field))
            if value is not None:
                selected[field] = value
        for field in PERSISTENT_TRACKING_BOOL_FIELDS:
            value = track.get(field)
            if isinstance(value, bool):
                selected[field] = value
        for field in PERSISTENT_TRACKING_NUMBER_FIELDS:
            value = _tracking_number(track.get(field))
            if value is not None:
                selected[field] = value
        xyz = _tracking_xyz(track.get("xyz_in_robot_base_coord_m"))
        if xyz is not None:
            selected["xyz_in_robot_base_coord_m"] = xyz
        tracks[name] = selected

    compact["tracks"] = tracks
    if len(named_tracks) > PERSISTENT_TRACKING_MAX_TRACKS:
        compact["total_track_count"] = len(named_tracks)
        compact["tracks_truncated"] = True

    # mark_on_map 标的地点和 track 点走同一条 persistent_tracking 通道
    compact["map_marks"] = _compact_map_marks(raw.get("marked_places"))
    return compact


PERSISTENT_MAP_MARK_MAX = 16


def _compact_map_marks(raw_marks: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_marks, list):
        return []
    marks: list[dict[str, Any]] = []
    for item in raw_marks[:PERSISTENT_MAP_MARK_MAX]:
        if not isinstance(item, dict):
            continue
        name = _tracking_text(item.get("name"))
        if name is None:
            continue
        mark: dict[str, Any] = {"name": name}
        direction = _tracking_text(item.get("direction"))
        if direction is not None:
            mark["direction"] = direction
        for field in ("range_m", "spin_deg_to_face"):
            value = _tracking_number(item.get(field))
            if value is not None:
                mark[field] = value
        marks.append(mark)
    return marks


@dataclass
class ToolResult:
    summary: str
    data: dict[str, Any]
    media: list[Media] = field(default_factory=list)
    rest_media_candidate_count: int = 0
    is_error: bool = False

    def public(self) -> dict[str, Any]:
        result = dict(self.data)
        if self.media:
            result["media"] = [item.public() for item in self.media]
        if self.rest_media_candidate_count or self.media:
            result["media_counts"] = {
                "rest_candidates": self.rest_media_candidate_count,
                "selected_for_model": len(self.media),
            }
        return result


@dataclass
class Episode:
    session_id: str
    catalog: ToolCatalog
    initial_state: Any
    recorder: TrajectoryRecorder | None


class EmbodiedService:
    """Thread-safe, task-independent BEHAVIOR v2 adapter."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: RestClient | None = None,
        profile: ToolProfile | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.client = client or RestClient(
            self.settings.base_url, self.settings.http_timeout_s
        )
        self.profile = profile or ToolProfile.load(self.settings.profile_path)
        self.media = MediaExtractor(
            self.client,
            max_bytes=self.settings.max_image_bytes,
            max_images=self.settings.max_images_per_call,
            max_model_bytes=self.settings.model_image_max_bytes,
            max_model_edge=self.settings.model_image_max_edge,
            jpeg_quality=self.settings.model_image_jpeg_quality,
            converter_path=self.settings.image_converter,
        )
        self._lock = threading.RLock()
        self._episode: Episode | None = None
        self._image_bindings: dict[str, dict[str, Any]] = {}
        self._latest_image_id = ""

    def prepare_episode(
        self,
        *,
        session_id: str = "",
        record: bool = True,
        label: str = "",
    ) -> ToolCatalog:
        """Load the live catalog and open an adapter-owned MCP episode."""
        with self._lock:
            episode = self._open_episode(
                session_id=session_id, record=record, label=label
            )
            return episode.catalog

    def start_episode(
        self,
        *,
        session_id: str = "",
        record: bool = True,
        label: str = "",
    ) -> ToolResult:
        with self._lock:
            started = time.monotonic()
            episode = self._open_episode(
                session_id=session_id, record=record, label=label
            )
            result = ToolResult(
                summary=(
                    f"Episode {episode.session_id} started with "
                    f"{len(episode.catalog.tools)} advertised v2 tools."
                ),
                data={
                    "session_id": episode.session_id,
                    "tool_version": episode.catalog.tool_version,
                    "available_tools": sorted(episode.catalog.tools),
                    "state": safe_json(episode.initial_state),
                    "profile": self.profile.public(),
                    "recording": (
                        episode.recorder.status() if episode.recorder else None
                    ),
                },
            )
            self._record_success(
                "behavior_start_episode",
                {"session_id": session_id, "record": record, "label": label},
                started,
                result,
            )
            if episode.recorder is not None:
                result.data["recording"] = episode.recorder.status()
            return result

    def list_tools(self) -> ToolResult:
        with self._lock:
            started = time.monotonic()
            arguments: dict[str, Any] = {}
            try:
                episode = self._require_episode()
                result = ToolResult(
                    summary=f"{len(episode.catalog.tools)} v2 tools are available.",
                    data=episode.catalog.public(),
                )
            except Exception as exc:
                self._record_error("behavior_list_tools", arguments, started, exc)
                raise
            self._record_success("behavior_list_tools", arguments, started, result)
            return result

    def get_state(self) -> ToolResult:
        with self._lock:
            started = time.monotonic()
            arguments: dict[str, Any] = {}
            try:
                self._require_episode()
                state = self.client.get_json("/api/state")
                result = ToolResult(
                    summary="Read current BEHAVIOR task state.",
                    data={"state": safe_json(state)},
                )
            except Exception as exc:
                self._record_error("behavior_get_state", arguments, started, exc)
                raise
            self._record_success("behavior_get_state", arguments, started, result)
            return result

    def call(self, *, tool_name: str, arguments: dict[str, Any] | None) -> ToolResult:
        with self._lock:
            started = time.monotonic()
            model_arguments = dict(arguments or {})
            try:
                episode = self._require_episode()
                spec = episode.catalog.get(tool_name)
                call_arguments = dict(model_arguments)
                if "session_id" in call_arguments:
                    raise ToolPolicyError(
                        "Omit session_id; the adapter injects the active session."
                    )
                spec.validate_arguments(call_arguments)
                if contains_uv(call_arguments):
                    self._assert_click_image(call_arguments.get("image_id"))
                    call_arguments = arguments_to_interface(call_arguments)
                if spec.adapter_image_id:
                    call_arguments.pop("image_id", None)
                call_arguments = self.profile.apply_arguments(
                    tool_name, call_arguments
                )
                for key, value in spec.fixed_arguments.items():
                    if key in call_arguments and call_arguments[key] != value:
                        raise ToolPolicyError(
                            f"Argument {key!r} is fixed by the server tool metadata.",
                            details={"tool_name": tool_name, "expected": value},
                        )
                    call_arguments[key] = value
                call_arguments["session_id"] = episode.session_id
                # A failed or imageless motion can still change the camera pose.
                if not tool_name.startswith(("capture_", "plan_", "read_", "measure_", "mark_", "track_")):
                    self._latest_image_id = ""
                response, capture_quality = self._post_tool(
                    tool_name=tool_name,
                    endpoint=spec.endpoint,
                    arguments=call_arguments,
                )
                selection = self.media.extract(response, tool_name=tool_name)
                warnings = selection.warnings
                if capture_quality is not None:
                    warnings.append(
                        f"Discarded {capture_quality['discarded_bad_frames']} "
                        f"low-quality {tool_name} frame(s) before accepting "
                        "a fresh frame."
                    )
                failed = isinstance(response, dict) and response.get("ok") is False
                data = {
                    "tool_name": tool_name,
                    "response": safe_json(response),
                    "warnings": warnings,
                }
                if capture_quality is not None:
                    data["capture_quality"] = capture_quality
                result = ToolResult(
                    summary=(
                        f"{tool_name} reported failure."
                        if failed
                        else f"{tool_name} completed."
                    ),
                    data=data,
                    media=selection.selected,
                    rest_media_candidate_count=selection.rest_candidate_count,
                    is_error=failed,
                )
                self._attach_persistent_tracking(result)
                self._bind_model_image(result)
                model_size = ((result.media[0].width, result.media[0].height)
                              if result.media else (720, 720))
                if None in model_size:
                    model_size = (720, 720)
                result.data["response"] = response_to_pixels(
                    result.data["response"], width=model_size[0], height=model_size[1]
                )
                result.data["persistent_tracking"] = response_to_pixels(
                    result.data["persistent_tracking"]
                )
            except Exception as exc:
                self._record_error(tool_name, model_arguments, started, exc)
                raise
            self._record_success(tool_name, model_arguments, started, result)
            return result

    def _assert_click_image(self, image_id: Any) -> None:
        binding = self._image_bindings.get(image_id) if isinstance(image_id, str) else None
        if (binding is None or not binding["clickable"]
                or binding["view"] != "head"
                or (binding["width"], binding["height"]) != (720, 720)
                or (binding["source_width"], binding["source_height"]) != (720, 720)
                or image_id != self._latest_image_id):
            raise ToolPolicyError(
                "Pixel click rejected: image_id must name the latest emitted head view, "
                "with both original and model image dimensions exactly 720 x 720. "
                "Capture a fresh head image and re-locate the target; wrist, unknown, "
                "stale, resized, or incorrectly sized views cannot be clicked.",
                details={"image_id": image_id, "latest_image_id": self._latest_image_id,
                         "image_geometry": binding},
            )

    def _bind_model_image(self, result: ToolResult) -> None:
        if not result.media:
            return
        response = result.data.get("response", {})
        observation = response.get("observation")
        observation = observation if isinstance(observation, dict) else {}
        media = result.media[0]
        image_id = str(response.get("image_id") or observation.get("image_id") or media.source_image_id)
        tool_name = str(result.data.get("tool_name") or "")
        camera = response.get("camera", observation.get("camera"))
        camera = camera if isinstance(camera, dict) else {}
        feed = str(media.source_feed or response.get("feed") or observation.get("feed") or camera.get("feed") or "").lower()
        head = "wrist" not in tool_name and (
            feed in {"head", "main"} or (not feed and tool_name == "capture_head_camera")
        )
        actual_size = (media.width, media.height)
        source_size = (media.source_width, media.source_height)
        declared_size = (response.get("image_width", observation.get("image_width", camera.get("image_width"))),
                         response.get("image_height", observation.get("image_height", camera.get("image_height"))))
        declared_ok = all(size is None or size == 720 for size in declared_size)
        binding = {
            "image_id": image_id, "view": "head" if head else feed or "unknown",
            "width": media.width, "height": media.height,
            "source_width": media.source_width, "source_height": media.source_height,
            "sha256": media.sha256,
            "clickable": bool(image_id and head and actual_size == (720, 720)
                              and source_size == (720, 720) and declared_ok
                              and media.source_image_id in {"", image_id}),
            "coordinate_system": VLM_IMAGE_COORDINATE_SYSTEM if head else "observation_only",
        }
        self._latest_image_id = image_id
        if image_id:
            self._image_bindings[image_id] = binding
        result.data["image_geometry"] = binding

    def _attach_persistent_tracking(self, result: ToolResult) -> None:
        result.data['rollout_budget'] = unavailable_budget()
        try:
            memory = self.client.get_memory(
                timeout_s=self.settings.memory_timeout_s
            )
            if isinstance(memory, dict):
                result.data['rollout_budget'] = normalize_budget(memory.get('rollout_budget'))
            tracking = _compact_persistent_tracking(memory)
        except (EmbodiedError, ValueError):
            tracking = {
                "available": False,
                "warning": PERSISTENT_TRACKING_UNAVAILABLE_WARNING,
            }
            warnings = result.data.get("warnings")
            if isinstance(warnings, list):
                warnings.append(PERSISTENT_TRACKING_UNAVAILABLE_WARNING)
        result.data["persistent_tracking"] = tracking
        # Keep live metric tracking; stale image UV is not a fresh visual target.
        for track in tracking.get("tracks", {}).values():
            track.pop("u", None)
            track.pop("v", None)
        if result.media:
            # BEHAVIOR_MINIMAP_HUD=0 时不叠右上角 SLAM 小地图，点选仍对着原 head 图。
            hud_raw = os.environ.get("BEHAVIOR_MINIMAP_HUD", "1").strip().lower()
            if hud_raw not in {"0", "false", "off", "no"}:
                overlaid = self.media.overlay_minimap_hud(result.media[0])
                if overlaid is not None:
                    result.media[0] = overlaid
                    tracking["minimap_hud"] = True
            # 卡片必须用模型这张图，不能再用磁盘上的地板蓝带原图。
            publish_from_tool_result(
                result,
                session_id=self.settings.session_id,
                base_url=self.settings.base_url,
            )

    def read_rollout_budget(self) -> dict[str, Any]:
        """Read counters for skill/error replies without advancing simulation."""
        with self._lock:
            try:
                payload = self.client.get_rollout_budget(
                    timeout_s=min(self.settings.memory_timeout_s, 1.0)
                )
                return normalize_budget(payload)
            except Exception:
                return unavailable_budget()

    def _post_tool(
        self,
        *,
        tool_name: str,
        endpoint: str,
        arguments: dict[str, Any],
    ) -> tuple[Any, dict[str, Any] | None]:
        if tool_name not in WRIST_CAPTURE_TOOLS:
            return self.client.post_json(endpoint, arguments), None

        discarded: list[dict[str, Any]] = []
        for attempt in range(1, WRIST_CAPTURE_MAX_ATTEMPTS + 1):
            response = self.client.post_json(endpoint, arguments)
            if isinstance(response, dict) and response.get("ok") is False:
                return response, None

            quality = _wrist_frame_quality(response)
            if quality is None and not discarded:
                return response, None
            if quality is not None and quality["acceptable"]:
                if not discarded:
                    return response, None
                accepted = dict(quality)
                accepted.pop("acceptable")
                return response, {
                    "attempts": attempt,
                    "discarded_bad_frames": len(discarded),
                    "minimum_valid_depth_ratio": WRIST_MIN_VALID_DEPTH_RATIO,
                    "discarded_frames": discarded,
                    "accepted_frame": accepted,
                }

            rejected = dict(quality or {})
            rejected.pop("acceptable", None)
            rejected["attempt"] = attempt
            if quality is None:
                rejected["reason"] = "quality_metadata_missing_after_bad_frame"
            else:
                rejected["reason"] = "valid_depth_ratio_below_minimum"
            discarded.append(rejected)

        raise CameraFrameError(
            (
                f"{tool_name} returned no usable frame after "
                f"{WRIST_CAPTURE_MAX_ATTEMPTS} serial capture attempts; "
                "no rejected image was exposed to the model."
            ),
            details={
                "tool_name": tool_name,
                "attempts": WRIST_CAPTURE_MAX_ATTEMPTS,
                "minimum_valid_depth_ratio": WRIST_MIN_VALID_DEPTH_RATIO,
                "discarded_frames": discarded,
            },
        )

    def finish_episode(
        self, *, outcome: str = "unknown", note: str = ""
    ) -> dict[str, Any] | None:
        """Finalize recording locally without adding a model-visible turn."""
        with self._lock:
            episode = self._episode
            if episode is None:
                return None
            if episode.recorder is not None:
                episode.recorder.finish(outcome=outcome, note=note)
            status = {
                "session_id": episode.session_id,
                "outcome": outcome,
                "recording": (
                    episode.recorder.status() if episode.recorder else None
                ),
                "server_process_changed": False,
            }
            self._episode = None
            return status

    def stop_episode(self, *, outcome: str = "unknown", note: str = "") -> ToolResult:
        with self._lock:
            started = time.monotonic()
            arguments = {"outcome": outcome, "note": note}
            try:
                episode = self._require_episode()
                recording = episode.recorder.status() if episode.recorder else None
                result = ToolResult(
                    summary=(
                        f"Episode {episode.session_id} closed locally; "
                        "the BEHAVIOR process was not changed."
                    ),
                    data={
                        "session_id": episode.session_id,
                        "outcome": outcome,
                        "recording": recording,
                        "server_process_changed": False,
                    },
                )
            except Exception as exc:
                self._record_error("behavior_stop_episode", arguments, started, exc)
                raise
            self._record_success(
                "behavior_stop_episode", arguments, started, result
            )
            if episode.recorder is not None:
                episode.recorder.finish(outcome=outcome, note=note)
                result.data["recording"] = episode.recorder.status()
            self._episode = None
            return result

    def _record_success(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        started: float,
        result: ToolResult,
    ) -> None:
        if self._episode is not None and self._episode.recorder is not None:
            self._episode.recorder.record(
                tool_name=tool_name,
                arguments=arguments,
                started_monotonic=started,
                result=result.public(),
                media=result.media,
                is_error=result.is_error,
            )

    def _record_error(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        started: float,
        exc: Exception,
    ) -> None:
        if self._episode is None or self._episode.recorder is None:
            return
        error = (
            exc.to_dict()
            if isinstance(exc, EmbodiedError)
            else {"code": "adapter_error", "message": str(exc)}
        )
        self._episode.recorder.record(
            tool_name=tool_name,
            arguments=arguments,
            started_monotonic=started,
            error=error,
            is_error=True,
        )

    def _require_episode(self) -> Episode:
        if self._episode is None:
            raise EpisodeStateError("The MCP episode is not active.")
        return self._episode

    def _open_episode(
        self, *, session_id: str, record: bool, label: str
    ) -> Episode:
        if self._episode is not None:
            raise EpisodeStateError("An episode is already active.")
        resolved_session = session_id.strip() or self._new_session_id()
        _validate_official_session_id(resolved_session)
        state = self.client.get_json("/api/state")
        catalog_payload = self.client.get_json("/api/v2/tools")
        catalog = ToolCatalog.from_payload(catalog_payload, self.profile)
        recorder = None
        if record:
            recorder = TrajectoryRecorder(
                root=self.settings.record_root,
                session_id=resolved_session,
                label=label,
                profile=self.profile.public(),
                tool_catalog=catalog.public(),
                initial_state=state,
            )
        episode = Episode(
            session_id=resolved_session,
            catalog=catalog,
            initial_state=state,
            recorder=recorder,
        )
        self._episode = episode
        self._image_bindings.clear()
        self._latest_image_id = ""
        return episode

    @staticmethod
    def _new_session_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"claude-{stamp}-{uuid.uuid4().hex[:8]}"


def _validate_official_session_id(session_id: str) -> str:
    """拒绝官方 track 合同无法接受的 session_id，避免开跑后再 400。"""
    sid = str(session_id or "").strip()
    if not sid:
        raise ToolPolicyError("session_id is required.")
    if len(sid) > OFFICIAL_SESSION_ID_MAX_LEN:
        raise ToolPolicyError(
            f"session_id is {len(sid)} chars; official track/read_depth/"
            f"move_point_to_point max is {OFFICIAL_SESSION_ID_MAX_LEN}."
        )
    if not SESSION_RE.fullmatch(sid):
        raise ToolPolicyError(
            "session_id must use 1-64 letters, digits, '.', '_' or '-'."
        )
    return sid
