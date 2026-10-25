from __future__ import annotations

from typing import Any
from tools.util.contract import (
    validate_track_object_distance_args,
)
from tools.util.tracked_object_distance import (
    TrackObjectDistanceBindingError,
    tracker_frame_content_digest,
)

from tools.util.motion_constants import TRACKING_CAMERA_ROLES
from tools.util.motion_common import _current_snapshot, _set_result


def _check_cancelled(ctx, where: str) -> None:
    fn = getattr(ctx, "raise_if_cancelled", None)
    if callable(fn):
        fn(where)


def _load_tracker_bound_capture(
    ctx,
    session_id: str,
    image_id: str,
) -> dict[str, Any]:
    """Load the immutable frame and hash it with the live X5 tracker path."""

    capture = _current_snapshot(
        ctx,
        ctx.snapshots.load(str(session_id), str(image_id)),
    )
    role = str(capture.role)
    if role not in TRACKING_CAMERA_ROLES:
        raise ValueError("selected image_id is not a supported RGB-D camera capture")
    if capture.rgb is None or capture.depth is None:
        raise ValueError(f"frozen {role} capture has incomplete RGB-D artifacts")
    if capture.rgb.shape[:2] != capture.depth.shape:
        raise ValueError(f"frozen {role} RGB and depth resolutions do not match")
    frame = ctx.tracker.frame_from_snapshot(capture)
    return {
        "capture": capture,
        "episode_id": str(capture.episode_id),
        "camera_role": role,
        "frame_digest": tracker_frame_content_digest(frame),
    }


def track_object_distance(
    ctx,
    session_id: str,
    image_id: str,
    points: list[dict[str, Any]],
):
    """Ground a complete named set on one frozen frame, then track it live."""

    common = {
        "execution": "official_action_only",
        "observation_contract": [
            "*::rgb",
            "*::depth_linear",
            "*::cam_rel_poses",
            "*::proprio",
        ],
        "direct_simulator_mutation": False,
        "session_id": str(session_id or "").strip(),
        "image_id": str(image_id or "").strip(),
        "memory_key": "tracked_object_distances",
        "depth_unit": "m",
        "xyz_in_robot_base_coord_unit": "m",
        "xyz_in_robot_base_coord_frame": "current_robot_base",
        "xyz_in_robot_base_coord_axes": "x_forward_y_left_z_up",
        "xyz_in_robot_base_coord_source": (
            "current_evaluator_depth_linear_and_camera_matrices"
        ),
        "deletion_policy": (
            "remove_measurement_when_unobserved_and_retire_track_"
            "when_out_of_view_or_lost"
        ),
        "stale_depth_published": False,
        "stale_xyz_in_robot_base_coord_published": False,
        "identity_source": "model_annotation",
        "identity_verified": False,
        "initial_grounding": "policy_owned_frozen_rgbd_capture",
        "frame_alignment": "required_session_id_image_id_and_tracker_frame_sha256",
        "replacement_policy": "complete_named_set_on_success",
        "cleared_names": [],
    }

    try:
        request = validate_track_object_distance_args(
            {
                "session_id": session_id,
                "image_id": image_id,
                "points": points,
            }
        )
    except ValueError as exc:
        _set_result(
            ctx,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": "invalid_request",
                "error": str(exc),
                "failure_stage": "input validation",
            },
        )
        yield ctx.world.hold_action()
        return

    manager = getattr(ctx.world, "_official_tracked_object_distances", None)
    if manager is None or not callable(getattr(manager, "replace_points", None)):
        _set_result(
            ctx,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": "tracker_unavailable",
                "error": "official dynamic distance tracker is unavailable",
                "failure_stage": "observation validation",
            },
        )
        yield ctx.world.hold_action()
        return

    try:
        capture_binding = _load_tracker_bound_capture(
            ctx,
            request["session_id"],
            request["image_id"],
        )
        capture = capture_binding["capture"]
        capture_episode_id = capture_binding["episode_id"]
        capture_frame_digest = capture_binding["frame_digest"]
        camera_role = capture_binding["camera_role"]
        common["camera_role"] = camera_role
        common["initial_grounding"] = f"policy_owned_frozen_{camera_role}_capture"
        common["replacement_slot"] = "head" if camera_role == "head" else "wrist"
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        _set_result(
            ctx,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": "capture_artifact_invalid",
                "error": str(exc),
                "failure_stage": "observation validation",
            },
        )
        yield ctx.world.hold_action()
        return

    try:
        clear_report = manager.clear_for_reselection(camera_role=camera_role)
        common["cleared_names"] = list(clear_report.get("cleared_names") or [])
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        _set_result(
            ctx,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": "tracking_reset_failed",
                "error": str(exc),
                "failure_stage": "tracking",
            },
        )
        yield ctx.world.hold_action()
        return

    def publish_committed_result(tracked: dict[str, Any]) -> None:
        _set_result(
            ctx,
            "track_object_distance",
            {
                **common,
                "ok": True,
                "error": None,
                "tracked_object_distances": tracked["entries"],
                "tracked_names": list(tracked["entries"]),
                "replaced_names": list(common.get("cleared_names") or []),
                "cleared_names": list(common.get("cleared_names") or []),
                "replacement_policy": "complete_named_set_on_success",
                "replacement_scope": ("one_head_group_plus_one_shared_wrist_group"),
                "replacement_slot": common.get("replacement_slot"),
                "session_id": tracked["session_id"],
                "image_id": tracked["image_id"],
                "capture_observation_sequence": tracked["capture_observation_sequence"],
                "observation_sequence": tracked["observation_sequence"],
                "replayed_observation_count": tracked["replayed_observation_count"],
                "episode_id": tracked["episode_id"],
                "frame_binding": tracked["frame_binding"],
                "frame_digest": tracked["frame_digest"],
                "source": (
                    f"policy_owned_frozen_{camera_role}_capture_then_live_"
                    f"evaluator_{camera_role}_rgbd"
                ),
            },
        )

    try:
        manager.replace_points(
            request["points"],
            session_id=request["session_id"],
            image_id=request["image_id"],
            capture_observation_sequence=int(capture.sequence),
            capture_episode_id=capture_episode_id,
            capture_image_shape=tuple(int(value) for value in capture.depth.shape),
            capture_frame_digest=capture_frame_digest,
            check_cancelled=lambda: _check_cancelled(
                ctx,
                "track_object_distance frozen-frame replay",
            ),
            commit_guard=getattr(
                ctx.world,
                "_official_request_commit_guard",
                None,
            ),
            commit_result=publish_committed_result,
            replaced_names=list(common.get("cleared_names") or []),
        )
        # The official manager is authoritative for tracking and motion.  Keep
        # the narrow legacy façade synchronized after that atomic commit so the
        # unchanged BEHAVIOR UI can still read pixel_uv/relative_uv and its
        # existing compatibility hooks continue to observe the same selection.
        remember_legacy = getattr(ctx.tracker, "remember_legacy_registration", None)
        if callable(remember_legacy):
            try:
                remember_legacy(capture, request["points"])
            except (KeyError, TypeError, ValueError, RuntimeError):
                # A compatibility mirror must never invalidate an already
                # committed official tracking registration.
                pass
    except TrackObjectDistanceBindingError as exc:
        _set_result(
            ctx,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": exc.reason,
                "error": str(exc),
                "failure_stage": "tracking",
            },
        )
        yield ctx.world.hold_action()
        return
    except (KeyError, TypeError, ValueError) as exc:
        _set_result(
            ctx,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": "tracking_failed",
                "error": str(exc),
                "failure_stage": "tracking",
            },
        )
        yield ctx.world.hold_action()
        return

    yield ctx.world.hold_action()


if __name__ == "__main__":
    from tools.util.cli import run_cli

    run_cli("track_object_distance", track_object_distance)
