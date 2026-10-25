"""Unit tests for the submission-local navigation footprint envelope."""

from __future__ import annotations

from copy import deepcopy
import math
import unittest
import json
from pathlib import Path

import numpy as np

from behavior_interface_eval_test.robot_contract import BASE_FOOTPRINT_RADIUS_M
from behavior_interface_eval_test.tool.official_v2.navigation_footprint_local import (
    WHOLE_BODY_FOOTPRINT_SCHEMA,
    whole_body_footprint_envelope,
    base_navigation_footprint_envelope,
    base_navigation_vertical_bounds_m,
)
from behavior_interface_eval_test.tool.official_v2 import navigation_footprint_local as footprint
from behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local import LocalRobotState, link_transforms


def _zero_qpos(arm_dof: int) -> dict[str, list[float]]:
    return {
        "trunk": [0.0] * 4,
        "arm_left": [0.0] * arm_dof,
        "arm_right": [0.0] * arm_dof,
        "gripper_left": [0.0] * 2,
        "gripper_right": [0.0] * 2,
    }


class NavigationFootprintLocalTest(unittest.TestCase):
    def test_chassis_vertical_bound_encloses_authored_geometry_at_all_wheel_angles(self):
        model = json.loads(Path(footprint._GEOMETRY_ASSET_PATH).read_text())
        lower, upper = base_navigation_vertical_bounds_m()
        self.assertAlmostEqual(upper, 0.410844918, places=8)
        state = LocalRobotState(
            arm_dof=8, base_pos=np.zeros(3), base_quat=np.array([0., 0., 0., 1.]),
            trunk_q=np.zeros(4), arm_left_q=np.zeros(8), arm_right_q=np.zeros(8),
            gripper_left_q=np.zeros(2), gripper_right_q=np.zeros(2),
            robot_forward=np.array([1., 0., 0.]),
        )
        for steer in np.linspace(-math.pi, math.pi, 5):
            for wheel in np.linspace(-math.pi, math.pi, 9):
                angles = {f"steer_motor_joint{i}": steer for i in (1, 2, 3)}
                angles.update({f"wheel_motor_joint{i}": wheel for i in (1, 2, 3)})
                transforms = link_transforms(state, q_overrides=angles)
                for link in model["navigation_contract"]["base_and_wheels_links"]:
                    for box in model["links"][link]:
                        corners = footprint._aabb_corners(box["aabb_min_m"], box["aabb_max_m"], label=link)
                        tf = transforms[link]
                        z = (corners @ tf[:3, :3].T + tf[:3, 3])[:, 2]
                        self.assertGreaterEqual(float(z.min()), lower - 1e-9)
                        self.assertLessEqual(float(z.max()), upper + 1e-9)
        envelope = base_navigation_footprint_envelope()
        self.assertEqual(envelope["radius_m"], BASE_FOOTPRINT_RADIUS_M)
        self.assertEqual(envelope["z_max_m"], upper)
        self.assertFalse(envelope["arm_posture_dependent"])

    def test_zero_pose_is_bounded_by_the_base_for_both_arm_contracts(self) -> None:
        for arm_dof in (7, 8):
            with self.subTest(arm_dof=arm_dof):
                envelope = whole_body_footprint_envelope(
                    _zero_qpos(arm_dof), arm_dof=arm_dof
                )

                self.assertEqual(envelope["schema"], WHOLE_BODY_FOOTPRINT_SCHEMA)
                self.assertEqual(envelope["radius_m"], BASE_FOOTPRINT_RADIUS_M)
                self.assertEqual(
                    envelope["base_radius_m"], BASE_FOOTPRINT_RADIUS_M
                )
                self.assertEqual(envelope["limiting_link"], "base_footprint")
                self.assertEqual(
                    envelope["limiting_primitive"],
                    "base_and_wheels_contract",
                )
                self.assertGreater(envelope["collision_aabb_count"], 0)
                self.assertEqual(len(envelope["geometry_sha256"]), 64)
                self.assertEqual(len(envelope["kinematics_sha256"]), 64)
                self.assertEqual(len(envelope["source_usd_sha256"]), 64)
                self.assertFalse(envelope["payload_envelope_included"])

    def test_legal_left_and_right_extension_expand_the_swept_circle(self) -> None:
        radii: dict[str, float] = {}
        for side in ("left", "right"):
            with self.subTest(side=side):
                qpos = _zero_qpos(8)
                qpos[f"arm_{side}"][0] = -math.pi / 2.0
                envelope = whole_body_footprint_envelope(qpos, arm_dof=8)

                radius_m = float(envelope["radius_m"])
                radii[side] = radius_m
                self.assertGreater(radius_m, 0.70)
                self.assertLess(radius_m, 0.85)
                self.assertTrue(
                    str(envelope["limiting_link"]).startswith(f"{side}_"),
                    envelope,
                )

        self.assertAlmostEqual(radii["left"], radii["right"], delta=0.002)

    def test_seven_dof_and_locked_eighth_joint_contracts(self) -> None:
        seven_dof = _zero_qpos(7)
        seven_dof["arm_left"][0] = -math.pi / 2.0
        self.assertGreater(
            whole_body_footprint_envelope(seven_dof, arm_dof=7)["radius_m"],
            0.70,
        )

        eight_dof = _zero_qpos(8)
        eight_dof["arm_left"][0] = -math.pi / 2.0
        self.assertGreater(
            whole_body_footprint_envelope(eight_dof, arm_dof=8)["radius_m"],
            0.70,
        )
        eight_dof["arm_right"][7] = 1.0e-3
        with self.assertRaisesRegex(ValueError, "J8 joints locked to zero"):
            whole_body_footprint_envelope(eight_dof, arm_dof=8)

    def test_missing_nonfinite_and_short_proprio_fail_closed(self) -> None:
        valid = _zero_qpos(8)
        cases: list[tuple[str, dict[str, list[float]]]] = []

        missing = deepcopy(valid)
        missing.pop("trunk")
        cases.append(("missing", missing))

        nonfinite = deepcopy(valid)
        nonfinite["arm_left"][2] = math.nan
        cases.append(("nonfinite", nonfinite))

        short_arm = deepcopy(valid)
        short_arm["arm_right"] = short_arm["arm_right"][:-1]
        cases.append(("short_arm", short_arm))

        short_gripper = deepcopy(valid)
        short_gripper["gripper_left"] = [0.0]
        cases.append(("short_gripper", short_gripper))

        for label, qpos in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "finite values"):
                    whole_body_footprint_envelope(qpos, arm_dof=8)


if __name__ == "__main__":
    unittest.main()
