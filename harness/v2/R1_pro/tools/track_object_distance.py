"""track_object_distance

Bind every UV to the exact frozen head capture named by image_id, then track multiple model-named surface points; publish only current depth_m and xyz_in_robot_base_coord_m=[x, y, z] in memory. xyz_in_robot_base_coord_m is measured in meters in the current robot chassis frame: +X always points chassis-forward, +Y chassis-left, and +Z chassis-up. A stale image binding or replay gap fails instead of applying UV to a newer frame. capture_replay_history_was_evicted_by_its_bounded_storage_policy means capture a new head image and select the points on that new image. Delete an entry when the point leaves view or tracking is lost.

How this tool works:
Bind each named point to the exact frozen RGB-D frame, then track its observed image/depth evidence across new observations. Report visibility, age and distance. A track is perception memory, not ground-truth object state.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.geometry.contract import validate_track_object_distance_args
from tools.util.geometry.tracked_object_distance import TrackObjectDistanceBindingError
from tools.util.motion_common import _check_cancelled
from tools.util.motion_common import _load_tracker_bound_capture
from tools.util.motion_common import _set_result
from typing import Any
from tools.util.tool import main

NAME = 'track_object_distance'
SCHEMA = {'type': 'object',
 'properties': {'image_id': {'type': 'string',
                             'description': 'The latest clickable head-camera image_id, actually '
                                            '720 x 720; re-ground the point when image_id changes. '
                                            'Wrist images are not clickable.'},
                'points': {'type': 'array',
                           'minItems': 1,
                           'maxItems': 32,
                           'items': {'properties': {'name': {'type': 'string'},
                                                    'u': {'type': 'integer',
                                                          'minimum': 0,
                                                          'maximum': 719,
                                                          'description': 'Integer original-image '
                                                                         'pixel column in the '
                                                                         'latest 720 x 720 head '
                                                                         'image: 0 is the left '
                                                                         'edge and 719 is the '
                                                                         'right edge. Do not '
                                                                         'normalize or rescale.'},
                                                    'v': {'type': 'integer',
                                                          'minimum': 0,
                                                          'maximum': 719,
                                                          'description': 'Integer original-image '
                                                                         'pixel row in the latest '
                                                                         '720 x 720 head image: 0 '
                                                                         'is the top edge and 719 '
                                                                         'is the bottom edge. Do '
                                                                         'not normalize or '
                                                                         'rescale.'}},
                                     'required': ['name', 'u', 'v'],
                                     'type': 'object'}}},
 'additionalProperties': False,
 'required': ['image_id', 'points']}


def actions(
    robot,
    session_id: str,
    image_id: str,
    points: list[dict[str, Any]],
):
    """Ground named points on one frozen image, then track them live.

    A successful request replaces the complete previously registered named
    set; it never merges the new selection with old points.
    """

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
        "deletion_policy": (
            "remove_measurement_when_unobserved_and_retire_track_"
            "when_out_of_view_or_lost"
        ),
        "stale_depth_published": False,
        "stale_xyz_in_robot_base_coord_published": False,
        "identity_source": "model_annotation",
        "identity_verified": False,
        "initial_grounding": "policy_owned_frozen_head_capture",
        "frame_alignment": (
            "required_session_id_image_id_and_tracker_frame_sha256"
        ),
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
            robot,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": "invalid_request",
                "error": str(exc),
                "failure_stage": "input validation",
            },
        )
        yield robot.hold_action()
        return

    manager = getattr(robot, "_official_tracked_object_distances", None)
    if manager is None:
        _set_result(
            robot,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": "tracker_unavailable",
                "error": "official dynamic distance tracker is unavailable",
                "failure_stage": "observation validation",
            },
        )
        yield robot.hold_action()
        return

    try:
        clear_report = manager.clear_for_reselection()
        common["cleared_names"] = list(
            clear_report.get("cleared_names") or []
        )
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        _set_result(
            robot,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": "tracking_reset_failed",
                "error": str(exc),
                "failure_stage": "tracking",
            },
        )
        yield robot.hold_action()
        return

    try:
        capture_binding = _load_tracker_bound_capture(
            request["session_id"],
            request["image_id"],
        )
        capture = capture_binding["capture"]
        capture_episode_id = capture_binding["episode_id"]
        capture_frame_digest = capture_binding["frame_digest"]
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
        _set_result(
            robot,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": "capture_artifact_invalid",
                "error": str(exc),
                "failure_stage": "observation validation",
            },
        )
        yield robot.hold_action()
        return

    def publish_committed_result(tracked: dict[str, Any]) -> None:
        _set_result(
            robot,
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
                "session_id": tracked["session_id"],
                "image_id": tracked["image_id"],
                "capture_observation_sequence": tracked[
                    "capture_observation_sequence"
                ],
                "observation_sequence": tracked["observation_sequence"],
                "replayed_observation_count": tracked[
                    "replayed_observation_count"
                ],
                "episode_id": tracked["episode_id"],
                "frame_binding": tracked["frame_binding"],
                "frame_digest": tracked["frame_digest"],
                "source": (
                    "policy_owned_frozen_head_capture_then_live_evaluator_head_rgbd"
                ),
            },
        )

    try:
        tracked = manager.replace_points(
            request["points"],
            session_id=request["session_id"],
            image_id=request["image_id"],
            capture_observation_sequence=capture.evaluator_sequence,
            capture_episode_id=capture_episode_id,
            capture_image_shape=capture.depth.shape,
            capture_frame_digest=capture_frame_digest,
            check_cancelled=lambda: _check_cancelled(
                robot,
                "track_object_distance frozen-frame replay",
            ),
            commit_guard=getattr(
                robot,
                "_official_request_commit_guard",
                None,
            ),
            commit_result=publish_committed_result,
        )
    except TrackObjectDistanceBindingError as exc:
        _set_result(
            robot,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": exc.reason,
                "error": str(exc),
                "failure_stage": "tracking",
            },
        )
        yield robot.hold_action()
        return
    except (KeyError, TypeError, ValueError) as exc:
        _set_result(
            robot,
            "track_object_distance",
            {
                **common,
                "ok": False,
                "reason": "tracking_failed",
                "error": str(exc),
                "failure_stage": "tracking",
            },
        )
        yield robot.hold_action()
        return

    yield robot.hold_action()




if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
