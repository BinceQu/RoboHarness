"""measure_shoulder_distance

Measure shoulder distances to a visible surface selected in the latest head image.
Requires image_id, u and v. Object names and privileged object AABBs are not inputs.

How this tool works:
Use the selected pixel's frozen metric depth and camera transform. Compute its 3D distance to both measured shoulder locations and the shoulder midpoint. This measures the clicked surface; it does not identify the object for you.

Read actions() below for the control sequence. Each yielded array is one
27-value command sent directly to the evaluator. RGB-D/IK helpers live
in this file and tools/util; no HTTP macro-tool is called.
"""
from __future__ import annotations

from tools.util.action_constants import TRUNK_LIMITS
from tools.util.motion_common import _camera_intrinsics
from tools.util.motion_common import _depth_value
from tools.util.motion_common import _load_frozen_capture
from tools.util.motion_common import _point_from_relative_uv
from tools.util.motion_common import _point_in_robot_frame
from tools.util.motion_common import _r1pro_chest_pose_robot
from tools.util.motion_common import _r1pro_shoulder_positions_robot
from tools.util.motion_common import _relative_uv_to_pixel
from tools.util.motion_common import _set_result
from tools.util.motion_common import _shoulder_distances_robot
from tools.util.motion_common import _unproject_pixel
from typing import Any
import math
import numpy as np
from tools.util.tool import main

NAME = 'measure_shoulder_distance'
SCHEMA = {'type': 'object',
 'properties': {'image_id': {'type': 'string',
                             'description': 'The latest clickable head-camera image_id, actually '
                                            '720 x 720; re-ground the point when image_id changes. '
                                            'Wrist images are not clickable.'},
                'u': {'type': 'integer',
                      'minimum': 0,
                      'maximum': 719,
                      'description': 'Integer original-image pixel column in the latest 720 x 720 '
                                     'head image: 0 is the left edge and 719 is the right edge. Do '
                                     'not normalize or rescale.'},
                'v': {'type': 'integer',
                      'minimum': 0,
                      'maximum': 719,
                      'description': 'Integer original-image pixel row in the latest 720 x 720 '
                                     'head image: 0 is the top edge and 719 is the bottom edge. Do '
                                     'not normalize or rescale.'}},
 'additionalProperties': False,
 'required': ['image_id', 'u', 'v']}


def actions(
    robot,
    session_id: str,
    image_id: str,
    u: int,
    v: int,
    object_name: str = "",
    **kwargs,
):
    del object_name, kwargs
    capture = _load_frozen_capture(session_id, image_id)
    camera_relative_pose = capture.camera.get("robot_relative_pose")
    if camera_relative_pose:
        target_robot, point_meta = _point_from_current_depth_robot_base(
            depth=capture.depth,
            u=u,
            v=v,
            role=capture.role,
            camera_relative_pose=camera_relative_pose,
        )
    else:
        point_local, point_meta = _point_from_relative_uv(capture, u, v)
        target_robot = _point_in_robot_frame(robot, point_local)
    shoulders, distances = _shoulder_distances_robot(
        target_robot,
        robot.trunk_qpos(),
    )
    shoulder_mid = 0.5 * (shoulders["left"] + shoulders["right"])
    base_horizontal = float(np.linalg.norm(target_robot[:2]))
    shoulder_horizontal = float(
        np.linalg.norm(target_robot[:2] - shoulder_mid[:2])
    )
    current_shoulder_line_pitch = math.degrees(
        math.atan2(
            float(shoulder_mid[2] - target_robot[2]),
            max(1e-9, shoulder_horizontal),
        )
    )
    pitch_solution = _r1pro_pitch_for_aligned_target(
        target_robot,
        robot.trunk_qpos(),
    )
    _set_result(
        robot,
        "measure_shoulder_distance",
        {
            "ok": True,
            "target_source": "evaluator_depth_click",
            "target_robot_m": target_robot.astype(float).tolist(),
            "point_observation": point_meta,
            "left_shoulder_robot_m": shoulders["left"].astype(float).tolist(),
            "right_shoulder_robot_m": shoulders["right"].astype(float).tolist(),
            "shoulder_mid_robot_m": shoulder_mid.astype(float).tolist(),
            "left_shoulder_to_object_m": distances["left"],
            "right_shoulder_to_object_m": distances["right"],
            "base_horizontal_to_object_m": base_horizontal,
            "shoulder_mid_horizontal_to_object_m": shoulder_horizontal,
            "current_shoulder_line_pitch_deg": current_shoulder_line_pitch,
            "recommended_pitch": pitch_solution,
            "static_model": "exact submission-local R1Pro URDF FK",
        },
    )
    yield robot.hold_action()


def _point_from_current_depth_robot_base(
    *,
    depth: np.ndarray,
    u: float,
    v: float,
    role: str,
    camera_relative_pose: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unproject a current evaluator RGB-D pixel directly into robot base.

    This intentionally bypasses command odometry. It is the correct geometry
    path after a collision, where commanded base velocity may differ from
    actual base motion.
    """
    arr = np.asarray(depth, dtype=np.float32).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"depth must be 2-D, got {arr.shape}")
    height, width = arr.shape
    px, py = _relative_uv_to_pixel(u, v, width, height)
    depth_m = _depth_value(arr, px, py)
    relative = dict(camera_relative_pose or {})
    if not relative.get("pos") or not relative.get("quat"):
        raise ValueError("camera_relative_pose must contain pos and quat")
    camera = {
        **_camera_intrinsics(str(role), width, height),
        "pos": relative["pos"],
        "quat": relative["quat"],
    }
    point = _unproject_pixel(
        px=px,
        py=py,
        depth_m=depth_m,
        camera=camera,
    )
    return point, {
        "relative_uv": [float(u), float(v)],
        "pixel_uv": [int(px), int(py)],
        "depth_m": float(depth_m),
        "frame": "robot_base",
        "source": "current_depth_and_cam_rel_pose",
    }



def _r1pro_pitch_for_aligned_target(
    target_robot,
    trunk_q,
) -> dict[str, Any]:
    """Solve q3 so the shoulder-mid forward ray passes through the target.

    The target is reduced to its base-relative horizontal radius, which is
    equivalent to first aligning base yaw with the target. q1/q2 remain fixed
    and q4 is set to zero for the sagittal-plane solve.
    """
    target = np.asarray(target_robot, dtype=np.float64).reshape(3)
    q_start = np.asarray(trunk_q, dtype=np.float64).reshape(4)
    q1, q2, q3_start, _ = q_start
    target_aligned = np.array(
        [float(np.linalg.norm(target[:2])), 0.0, float(target[2])],
        dtype=np.float64,
    )
    a12 = float(q1 + q2)
    pitch_min = a12 - float(TRUNK_LIMITS[2, 1])
    pitch_max = a12 - float(TRUNK_LIMITS[2, 0])
    current_pitch = a12 - float(q3_start)

    def evaluate(pitch: float) -> dict[str, Any]:
        q3 = a12 - float(pitch)
        q = np.array([q1, q2, q3, 0.0], dtype=np.float64)
        shoulders = _r1pro_shoulder_positions_robot(q)
        shoulder_mid = 0.5 * (shoulders["left"] + shoulders["right"])
        _, rotation = _r1pro_chest_pose_robot(q)
        forward = rotation[:, 0]
        delta = target_aligned - shoulder_mid
        along = float(np.dot(delta, forward))
        residual_vec = delta - along * forward
        return {
            "pitch": float(pitch),
            "q3": float(q3),
            "cross": float(
                delta[0] * forward[2] - delta[2] * forward[0]
            ),
            "along": along,
            "residual_m": float(np.linalg.norm(residual_vec)),
            "shoulder_mid": shoulder_mid,
        }

    samples = np.linspace(pitch_min, pitch_max, 721)
    evaluated = [evaluate(float(pitch)) for pitch in samples]
    candidates: list[dict[str, Any]] = []
    for left, right in zip(evaluated[:-1], evaluated[1:]):
        if left["along"] > 0.0 and abs(left["cross"]) <= 1e-10:
            candidates.append(left)
        if left["cross"] * right["cross"] > 0.0:
            continue
        lo, hi = float(left["pitch"]), float(right["pitch"])
        f_lo = float(left["cross"])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            probe = evaluate(mid)
            if abs(probe["cross"]) <= 1e-12:
                lo = hi = mid
                break
            if f_lo * float(probe["cross"]) <= 0.0:
                hi = mid
            else:
                lo = mid
                f_lo = float(probe["cross"])
        candidate = evaluate(0.5 * (lo + hi))
        if candidate["along"] > 0.0:
            candidates.append(candidate)

    if not candidates:
        candidates = [
            min(
                (item for item in evaluated if item["along"] > 0.0),
                key=lambda item: item["residual_m"],
                default=min(evaluated, key=lambda item: item["residual_m"]),
            )
        ]
    best = min(
        candidates,
        key=lambda item: (
            item["residual_m"],
            abs(item["pitch"] - current_pitch),
        ),
    )
    return {
        "chest_pitch_deg": math.degrees(best["pitch"]),
        "q3_rad": best["q3"],
        "pitch_delta_deg": math.degrees(best["pitch"] - current_pitch),
        "ray_residual_m": best["residual_m"],
        "ray_forward_distance_m": best["along"],
        "target_aligned_robot_m": target_aligned.astype(float).tolist(),
        "shoulder_mid_at_solution_robot_m": best["shoulder_mid"]
        .astype(float)
        .tolist(),
        "assumption": "base yaw aligned to target; q1/q2 fixed; q4=0",
    }


if __name__ == "__main__":
    main(NAME, SCHEMA, actions)
