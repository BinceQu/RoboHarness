from __future__ import annotations

import ast
import copy
import json
import math
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from behavior_interface_eval_test.tool.official_v2.dynamic_point_tracker import (
    AMBIGUOUS,
    LOST,
    OBSERVED,
    OCCLUDED,
    CameraIntrinsics,
    DynamicPointTracker,
    DynamicPointTrackerConfig,
    OPENCL_UMAT_CACHE_MAX_ENTRIES,
    TrackerFrame,
    camera_to_policy_from_odometry,
    transform_from_pose,
)


class OfficialDynamicPointTrackerTest(unittest.TestCase):
    WIDTH = 192
    HEIGHT = 144
    INTRINSICS = CameraIntrinsics(
        width=WIDTH,
        height=HEIGHT,
        fx=150.0,
        fy=150.0,
        cx=96.0,
        cy=72.0,
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.can_texture = cls._texture(17, "can")
        cls.bin_texture = cls._texture(29, "bin")

    @staticmethod
    def _texture(seed: int, kind: str) -> np.ndarray:
        rng = np.random.default_rng(seed)
        texture = rng.integers(25, 230, size=(43, 43), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.45)
        if kind == "can":
            cv2.circle(texture, (21, 21), 16, 250, 2)
            cv2.line(texture, (8, 12), (34, 30), 15, 2)
            cv2.line(texture, (9, 31), (32, 8), 210, 1)
        else:
            cv2.rectangle(texture, (5, 5), (37, 37), 245, 2)
            cv2.line(texture, (5, 21), (37, 21), 10, 2)
            cv2.line(texture, (21, 5), (21, 37), 205, 2)
        return texture

    @classmethod
    def _render(cls, objects) -> tuple[np.ndarray, np.ndarray]:
        gray = np.full((cls.HEIGHT, cls.WIDTH), 8, dtype=np.uint8)
        depth = np.full((cls.HEIGHT, cls.WIDTH), 4.0, dtype=np.float32)
        for center, object_depth, texture in objects:
            center_x, center_y = (int(round(value)) for value in center)
            half = texture.shape[0] // 2
            left = center_x - half
            top = center_y - half
            right = left + texture.shape[1]
            bottom = top + texture.shape[0]
            if left < 0 or top < 0 or right > cls.WIDTH or bottom > cls.HEIGHT:
                raise ValueError("test object is outside the synthetic image")
            gray[top:bottom, left:right] = texture
            depth[top:bottom, left:right] = np.float32(object_depth)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), depth

    @classmethod
    def _render_transformed(
        cls,
        *,
        center,
        object_depth: float,
        texture: np.ndarray,
        angle_deg: float,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        gray = np.full((cls.HEIGHT, cls.WIDTH), 8, dtype=np.uint8)
        depth = np.full((cls.HEIGHT, cls.WIDTH), 4.0, dtype=np.float32)
        texture_center = (
            0.5 * float(texture.shape[1] - 1),
            0.5 * float(texture.shape[0] - 1),
        )
        transform = cv2.getRotationMatrix2D(
            texture_center,
            float(angle_deg),
            float(scale),
        )
        transform[:, 2] += np.asarray(center, dtype=np.float64) - np.asarray(
            texture_center,
            dtype=np.float64,
        )
        warped = cv2.warpAffine(
            texture,
            transform,
            (cls.WIDTH, cls.HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = cv2.warpAffine(
            np.full(texture.shape, 255, dtype=np.uint8),
            transform,
            (cls.WIDTH, cls.HEIGHT),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        visible = mask > 0
        gray[visible] = warped[visible]
        depth[visible] = np.float32(object_depth)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), depth

    @classmethod
    def _frame(
        cls,
        *,
        sequence: int,
        timestamp_s: float,
        objects,
        camera_to_robot_base: np.ndarray | None = None,
        camera_to_policy: np.ndarray | None = None,
        episode_id: str = "episode-a",
    ) -> TrackerFrame:
        rgb, depth = cls._render(objects)
        return TrackerFrame(
            rgb=rgb,
            depth_linear=depth,
            intrinsics=cls.INTRINSICS,
            camera_to_robot_base=(
                np.eye(4, dtype=np.float64)
                if camera_to_robot_base is None
                else np.asarray(camera_to_robot_base, dtype=np.float64)
            ),
            camera_to_policy=(
                np.eye(4, dtype=np.float64)
                if camera_to_policy is None
                else np.asarray(camera_to_policy, dtype=np.float64)
            ),
            sequence=sequence,
            timestamp_s=timestamp_s,
            episode_id=episode_id,
            camera_role="head",
        )

    @classmethod
    def _blank_frame(
        cls,
        *,
        sequence: int,
        timestamp_s: float,
        episode_id: str = "episode-a",
    ) -> TrackerFrame:
        return cls._frame(
            sequence=sequence,
            timestamp_s=timestamp_s,
            objects=[],
            episode_id=episode_id,
        )

    def _on_hand_misassociation_case(self):
        fixture_path = (
            Path(__file__).parent
            / "fixtures"
            / "move_tracked_point_on_hand_misassociation_15061.json"
        )
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        width, height = payload["image_size"]
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            **payload["intrinsics"],
        )

        def frame(sequence, samples):
            rng = np.random.default_rng(5840 + sequence)
            gray = rng.integers(35, 220, size=(height, width), dtype=np.uint8)
            gray = cv2.GaussianBlur(gray, (3, 3), 0.4)
            depth = np.full((height, width), 3.0, dtype=np.float32)
            for sample in samples:
                x, y = (int(round(value)) for value in sample["pixel"])
                depth[max(0, y - 8) : y + 9, max(0, x - 8) : x + 9] = (
                    np.float32(sample["depth_m"])
                )
            return TrackerFrame(
                rgb=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
                depth_linear=depth,
                intrinsics=intrinsics,
                camera_to_robot_base=np.eye(4, dtype=np.float64),
                camera_to_policy=np.eye(4, dtype=np.float64),
                sequence=sequence,
                timestamp_s=0.1 * sequence,
                episode_id="episode-15061-repro",
                camera_role="head",
            )

        tracker = DynamicPointTracker()
        tracker.ingest(frame(1, payload["expected"]))
        for index, sample in enumerate(payload["expected"], start=1):
            x, y = sample["pixel"]
            tracker.mark_point(
                u=float(x) / float(width - 1) * 1000.0,
                v=float(y) / float(height - 1) * 1000.0,
                label=f"on-hand-{index}",
                track_id=f"on-hand-track-{index}",
            )
        states = [
            copy.deepcopy(tracker._tracks[f"on-hand-track-{index}"])
            for index in (1, 2)
        ]

        def candidates(samples, validated):
            result = {}
            for state, sample in zip(states, samples):
                anchor = np.asarray(sample["pixel"], dtype=np.float64)
                depth_m = tracker._sample_depth(validated.depth, anchor)
                self.assertIsNotNone(depth_m)
                point_camera = tracker._unproject(
                    anchor, float(depth_m), validated.intrinsics
                )
                result[state.track_id] = tracker._commit_pair_observation(
                    state,
                    validated,
                    anchor=anchor,
                    depth=float(depth_m),
                    point_camera=point_camera,
                    retained_supports=state.support_px,
                    confidence=0.99,
                    observation_method="high_correlation_individual_fallback",
                )
            return result

        return payload, tracker, states, frame, candidates

    def _candidate_fallback_case(self):
        payload, tracker, states, frame_factory, make_candidates = (
            self._on_hand_misassociation_case()
        )
        frame = tracker._validate_frame(
            frame_factory(
                2,
                payload["expected"] + payload["wrong_individual_candidates"],
            )
        )
        good = make_candidates(payload["expected"], frame)
        wrong = make_candidates(payload["wrong_individual_candidates"], frame)
        state = states[1]
        good_candidate = good[state.track_id]
        point_prior = {
            "point_robot_base_m": np.asarray(
                good_candidate.snapshot.point_robot_base_m,
                dtype=np.float64,
            ),
            "expected_pixel": np.asarray(good_candidate.anchor_px, dtype=np.float64),
            "expected_depth_m": float(good_candidate.snapshot.depth_m),
        }
        return (
            tracker,
            state,
            frame,
            good_candidate,
            wrong[state.track_id],
            point_prior,
        )

    def test_advance_track_rejects_registration_candidate_then_uses_previous(
        self,
    ) -> None:
        tracker, state, frame, good, wrong, prior = self._candidate_fallback_case()
        with mock.patch.object(
            tracker,
            "_advance_feature_track",
            side_effect=[copy.deepcopy(wrong), copy.deepcopy(good)],
        ) as feature, mock.patch.object(
            tracker,
            "_advance_depth_component_track",
        ) as depth:
            result = tracker._advance_track(
                copy.deepcopy(state),
                frame,
                observation_prior=prior,
            )

        self.assertEqual(feature.call_count, 2)
        depth.assert_not_called()
        self.assertEqual(result.snapshot.status, OBSERVED)
        self.assertEqual(result.last_observation_method, "previous_observation_feature")
        np.testing.assert_allclose(result.anchor_px, good.anchor_px, atol=1e-12)
        self.assertEqual(len(result.last_observation_rejections), 1)
        rejection = result.last_observation_rejections[0]
        self.assertFalse(rejection["ok"])
        self.assertEqual(rejection["candidate_method"], "registration_feature")
        self.assertEqual(rejection["reason"], "eef_prior_pixel_mismatch")

    def test_eef_prior_retains_patch_reacquisition_for_low_texture_on_hand_points(self):
        tracker = DynamicPointTracker()
        initial = self._frame(sequence=1, timestamp_s=0., objects=[
            ((96, 72), 1.25, np.full((43, 43), 100, dtype=np.uint8)),
        ])
        tracker.ingest(initial)
        mark = self._mark(tracker, np.array([96., 72.]), label="on", track_id="on")
        state = copy.deepcopy(tracker._tracks["on"])
        state.support_px = np.empty((0, 2), dtype=np.float32)
        state.registration_support_px = np.empty((0, 2), dtype=np.float32)
        state.anchor_px += [20., 10.]
        policy_transform = np.eye(4)
        policy_transform[:3, 3] = [.3, -.2, .1]
        current = tracker._validate_frame(self._frame(
            sequence=2, timestamp_s=.1, camera_to_policy=policy_transform,
            objects=[((96, 72), 1.252, np.full((43, 43), 100, dtype=np.uint8))],
        ))
        prior = {"point_robot_base_m": np.array(mark.point_robot_base_m),
                 "expected_pixel": np.array([96., 72.]), "expected_depth_m": 1.25}
        with mock.patch.object(tracker, "_advance_feature_track") as feature, mock.patch.object(
            tracker, "_advance_depth_component_track"
        ) as component:
            recovered = tracker._advance_track(state, current, observation_prior=prior)
        self.assertEqual(recovered.snapshot.status, OBSERVED)
        self.assertEqual(recovered.last_observation_method, "registration_patch_eef_prior_rgbd")
        feature.assert_not_called()
        component.assert_not_called()
        # The emitted depth/XYZ is the fresh measurement, not the FK prediction.
        self.assertAlmostEqual(recovered.snapshot.depth_m, 1.252, places=6)
        self.assertGreater(np.linalg.norm(np.array(recovered.snapshot.point_robot_base_m)
                                         - prior["point_robot_base_m"]), .001)
        np.testing.assert_allclose(np.array(recovered.snapshot.point_policy_local_m),
                                   np.array(recovered.snapshot.point_robot_base_m)+policy_transform[:3, 3])
        np.testing.assert_array_equal(recovered.registration_anchor_px, state.registration_anchor_px)

    def test_prior_patch_reacquisition_requires_current_rgb_and_depth_evidence(self):
        tracker = DynamicPointTracker()
        tracker.ingest(self._frame(sequence=1, timestamp_s=0., objects=[
            ((96, 72), 1.25, self.can_texture),
        ]))
        mark = self._mark(tracker, np.array([96., 72.]), label="on", track_id="on")
        state = tracker._tracks["on"]
        prior = {"point_robot_base_m": np.array(mark.point_robot_base_m),
                 "expected_pixel": np.array([96., 72.]), "expected_depth_m": 1.25}
        for case in ("appearance", "depth", "invalid_depth", "nonfinite_prior", "out_of_view"):
            with self.subTest(case=case):
                frame = self._frame(sequence=2, timestamp_s=.1, objects=[
                    ((96, 72), 1.25 if case != "depth" else 1.3,
                     self.bin_texture if case == "appearance" else self.can_texture),
                ])
                if case == "invalid_depth":
                    frame.depth_linear[:] = np.nan
                current_prior = copy.deepcopy(prior)
                if case == "nonfinite_prior":
                    current_prior["point_robot_base_m"][:] = np.nan
                elif case == "out_of_view":
                    current_prior["point_robot_base_m"][0] = 100.
                result = tracker._static_registration_candidate(
                    state, tracker._validate_frame(frame), observation_prior=current_prior,
                )
                self.assertIsNone(result)

    def test_prior_patch_search_recovers_small_finger_settling_but_not_large_drift(self):
        tracker = DynamicPointTracker()
        tracker.ingest(self._frame(sequence=1, timestamp_s=0., objects=[
            ((96, 72), .18, self.can_texture),
        ]))
        mark = self._mark(tracker, np.array([96., 72.]), label="finger", track_id="finger")
        state = tracker._tracks["finger"]
        prior = {"point_robot_base_m": np.array(mark.point_robot_base_m),
                 "expected_pixel": np.array([96., 72.]), "expected_depth_m": .18}
        for offset, accepted in ((2, True), (10, False)):
            with self.subTest(offset=offset):
                current = tracker._validate_frame(self._frame(sequence=2, timestamp_s=.1, objects=[
                    ((96+offset, 72), .18, self.can_texture),
                ]))
                result = tracker._static_registration_candidate(state, current, observation_prior=prior)
                self.assertEqual(result is not None, accepted)
                if accepted:
                    np.testing.assert_allclose(result.anchor_px, [98., 72.], atol=1e-4)
                    error = np.linalg.norm(np.array(result.snapshot.point_robot_base_m) - prior["point_robot_base_m"])
                    self.assertGreater(error, .002)
                    self.assertLessEqual(error, tracker.config.static_registration_max_point_error_m)

    def test_advance_track_rejects_feature_candidates_then_uses_depth(
        self,
    ) -> None:
        tracker, state, frame, good, wrong, prior = self._candidate_fallback_case()
        with mock.patch.object(
            tracker,
            "_advance_feature_track",
            side_effect=[copy.deepcopy(wrong), copy.deepcopy(wrong)],
        ), mock.patch.object(
            tracker,
            "_advance_depth_component_track",
            return_value=copy.deepcopy(good),
        ) as depth:
            result = tracker._advance_track(
                copy.deepcopy(state),
                frame,
                observation_prior=prior,
            )

        depth.assert_called_once()
        self.assertEqual(result.snapshot.status, OBSERVED)
        self.assertEqual(result.last_observation_method, "depth_component")
        np.testing.assert_allclose(result.anchor_px, good.anchor_px, atol=1e-12)
        self.assertEqual(len(result.last_observation_rejections), 2)
        self.assertTrue(
            all(
                item["reason"] == "eef_prior_pixel_mismatch"
                for item in result.last_observation_rejections
            )
        )

    def test_advance_track_rejects_all_prior_mismatches_without_publishing_xyz(
        self,
    ) -> None:
        tracker, state, frame, _good, wrong, prior = self._candidate_fallback_case()
        with mock.patch.object(
            tracker,
            "_advance_feature_track",
            side_effect=[copy.deepcopy(wrong), copy.deepcopy(wrong)],
        ), mock.patch.object(
            tracker,
            "_advance_depth_component_track",
            return_value=copy.deepcopy(wrong),
        ):
            result = tracker._advance_track(
                copy.deepcopy(state),
                frame,
                observation_prior=prior,
            )

        self.assertNotEqual(result.snapshot.status, OBSERVED)
        self.assertIsNone(result.snapshot.pixel_uv)
        self.assertIsNone(result.snapshot.depth_m)
        self.assertIsNone(result.snapshot.point_robot_base_m)
        self.assertEqual(len(result.last_observation_rejections), 3)
        self.assertEqual(
            [item["candidate_method"] for item in result.last_observation_rejections],
            [
                "registration_feature",
                "previous_observation_feature",
                "depth_component",
            ],
        )

    def test_advance_track_without_prior_preserves_registration_first_behavior(
        self,
    ) -> None:
        tracker, state, frame, _good, wrong, _prior = self._candidate_fallback_case()
        with mock.patch.object(
            tracker,
            "_advance_feature_track",
            return_value=copy.deepcopy(wrong),
        ) as feature, mock.patch.object(
            tracker,
            "_advance_depth_component_track",
        ) as depth:
            result = tracker._advance_track(
                copy.deepcopy(state),
                frame,
                observation_prior=None,
            )

        feature.assert_called_once()
        depth.assert_not_called()
        self.assertEqual(result.snapshot.status, OBSERVED)
        self.assertEqual(result.last_observation_method, "registration_feature")
        np.testing.assert_allclose(result.anchor_px, wrong.anchor_px, atol=1e-12)
        self.assertEqual(result.last_observation_rejections, ())

    @classmethod
    def _flat_surface_frame(
        cls,
        *,
        sequence: int,
        timestamp_s: float,
        center,
        object_depth: float,
        camera_to_robot_base: np.ndarray | None = None,
        camera_to_policy: np.ndarray | None = None,
        visible: bool = True,
        episode_id: str = "episode-a",
    ) -> TrackerFrame:
        gray = np.full((cls.HEIGHT, cls.WIDTH), 80, dtype=np.uint8)
        depth = np.full((cls.HEIGHT, cls.WIDTH), 4.0, dtype=np.float32)
        if visible:
            center_x, center_y = (int(round(value)) for value in center)
            depth[
                center_y - 15 : center_y + 16,
                center_x - 20 : center_x + 21,
            ] = np.float32(object_depth)
        return TrackerFrame(
            rgb=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
            depth_linear=depth,
            intrinsics=cls.INTRINSICS,
            camera_to_robot_base=(
                np.eye(4, dtype=np.float64)
                if camera_to_robot_base is None
                else np.asarray(camera_to_robot_base, dtype=np.float64)
            ),
            camera_to_policy=(
                np.eye(4, dtype=np.float64)
                if camera_to_policy is None
                else np.asarray(camera_to_policy, dtype=np.float64)
            ),
            sequence=sequence,
            timestamp_s=timestamp_s,
            episode_id=episode_id,
            camera_role="head",
        )

    @classmethod
    def _relative_uv(cls, pixel) -> tuple[float, float]:
        x, y = np.asarray(pixel, dtype=np.float64).reshape(2)
        return (
            x / float(cls.WIDTH - 1) * 1000.0,
            y / float(cls.HEIGHT - 1) * 1000.0,
        )

    @classmethod
    def _point_camera(cls, pixel, depth_m: float) -> np.ndarray:
        x, y = np.asarray(pixel, dtype=np.float64).reshape(2)
        depth = float(depth_m)
        return np.array(
            [
                (x - cls.INTRINSICS.cx) / cls.INTRINSICS.fx * depth,
                -(y - cls.INTRINSICS.cy) / cls.INTRINSICS.fy * depth,
                -depth,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _point_policy(camera_to_policy, point_camera) -> np.ndarray:
        point = np.append(np.asarray(point_camera, dtype=np.float64), 1.0)
        return (np.asarray(camera_to_policy, dtype=np.float64) @ point)[:3]

    def _mark(
        self,
        tracker: DynamicPointTracker,
        center,
        *,
        label: str,
        track_id: str,
    ):
        u, v = self._relative_uv(center)
        return tracker.mark_point(
            u=u,
            v=v,
            label=label,
            track_id=track_id,
        )

    def test_two_independent_objects_track_uv_depth_3d_and_velocity(self) -> None:
        tracker = DynamicPointTracker()
        can_center = np.array([52.0, 60.0])
        bin_center = np.array([142.0, 82.0])
        can_depth = 1.20
        bin_depth = 2.10
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[
                    (can_center, can_depth, self.can_texture),
                    (bin_center, bin_depth, self.bin_texture),
                ],
            )
        )
        can_initial = self._mark(
            tracker,
            can_center,
            label="can",
            track_id="can-track",
        )
        bin_initial = self._mark(
            tracker,
            bin_center,
            label="bin",
            track_id="bin-track",
        )

        self.assertEqual(can_initial.status, OBSERVED)
        self.assertEqual(bin_initial.status, OBSERVED)
        self.assertAlmostEqual(can_initial.depth_m, can_depth, places=6)
        self.assertAlmostEqual(bin_initial.depth_m, bin_depth, places=6)

        max_can_pixel_error = 0.0
        max_bin_pixel_error = 0.0
        for sequence in range(2, 15):
            step = sequence - 1
            next_can = can_center + np.array([4.0 * step, -2.0 * step])
            next_bin = bin_center + np.array([-3.0 * step, 1.0 * step])
            next_can_depth = can_depth - 0.025 * step
            next_bin_depth = bin_depth + 0.015 * step
            tracker.ingest(
                self._frame(
                    sequence=sequence,
                    timestamp_s=0.1 * step,
                    objects=[
                        (next_can, next_can_depth, self.can_texture),
                        (next_bin, next_bin_depth, self.bin_texture),
                    ],
                )
            )
            can = tracker.get_point("can-track")
            bin_point = tracker.get_point("bin-track")
            self.assertEqual(can.status, OBSERVED, can.as_dict())
            self.assertEqual(bin_point.status, OBSERVED, bin_point.as_dict())
            self.assertEqual(can.label, "can")
            self.assertEqual(bin_point.label, "bin")
            max_can_pixel_error = max(
                max_can_pixel_error,
                float(np.linalg.norm(np.asarray(can.pixel_uv) - next_can)),
            )
            max_bin_pixel_error = max(
                max_bin_pixel_error,
                float(np.linalg.norm(np.asarray(bin_point.pixel_uv) - next_bin)),
            )
            self.assertAlmostEqual(can.depth_m, next_can_depth, places=5)
            self.assertAlmostEqual(bin_point.depth_m, next_bin_depth, places=5)

            can_camera = self._point_camera(next_can, next_can_depth)
            bin_camera = self._point_camera(next_bin, next_bin_depth)
            np.testing.assert_allclose(
                can.point_camera_m,
                can_camera,
                atol=0.008,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                bin_point.point_camera_m,
                bin_camera,
                atol=0.012,
                rtol=0.0,
            )
            self.assertGreater(can.confidence, 0.70)
            self.assertGreater(bin_point.confidence, 0.70)

        self.assertLess(max_can_pixel_error, 0.60)
        self.assertLess(max_bin_pixel_error, 0.60)
        can_payload = tracker.get_point("can-track").as_dict()
        self.assertEqual(
            can_payload["depth_source"],
            "current_evaluator_depth_linear",
        )
        self.assertEqual(can_payload["identity_source"], "model_annotation")
        self.assertFalse(can_payload["identity_verified"])
        self.assertFalse(can_payload["direct_simulator_mutation"])

    def test_registration_keyframe_prevents_round_trip_identity_drift(self) -> None:
        tracker = DynamicPointTracker()
        origin = np.array([92.0, 70.0])
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[(origin, 1.15, self.can_texture)],
            )
        )
        self._mark(
            tracker,
            origin,
            label="round-trip",
            track_id="round-trip-track",
        )
        centers = []
        for index in range(1, 61):
            phase = 2.0 * np.pi * index / 60.0
            center = origin + np.array(
                [28.0 * np.sin(phase), 13.0 * np.sin(2.0 * phase)]
            )
            centers.append(center)
            tracker.ingest(
                self._frame(
                    sequence=index + 1,
                    timestamp_s=index / 30.0,
                    objects=[(center, 1.15, self.can_texture)],
                )
            )
            current = tracker.get_point("round-trip-track")
            self.assertEqual(current.status, OBSERVED, current.as_dict())
            self.assertLess(
                float(np.linalg.norm(np.asarray(current.pixel_uv) - center)),
                0.75,
            )
        np.testing.assert_allclose(centers[-1], origin, atol=1e-10)
        final = tracker.get_point("round-trip-track")
        self.assertLess(
            float(np.linalg.norm(np.asarray(final.pixel_uv) - origin)),
            0.25,
        )

    def test_static_registration_anchor_prevents_low_texture_random_walk(self) -> None:
        """A static RGB-D patch must not drift through optical-flow fallback."""

        rng = np.random.default_rng(20260905)
        center = np.array([96.0, 72.0])

        def static_frame(sequence: int) -> TrackerFrame:
            # Small renderer noise is enough to make LK support intermittent,
            # while the material and its depth layer remain unchanged.
            gray = np.full((self.HEIGHT, self.WIDTH), 80, dtype=np.int16)
            gray[52:93, 76:117] = 132
            gray += rng.integers(-1, 2, size=gray.shape, dtype=np.int16)
            gray = np.clip(gray, 0, 255).astype(np.uint8)
            depth = np.full((self.HEIGHT, self.WIDTH), 4.0, dtype=np.float32)
            depth[52:93, 76:117] = 1.25
            return TrackerFrame(
                rgb=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
                depth_linear=depth,
                intrinsics=self.INTRINSICS,
                camera_to_robot_base=np.eye(4, dtype=np.float64),
                camera_to_policy=np.eye(4, dtype=np.float64),
                sequence=sequence,
                timestamp_s=0.1 * (sequence - 1),
                episode_id="static-low-texture",
                camera_role="head",
            )

        tracker = DynamicPointTracker()
        tracker.ingest(static_frame(1))
        first = self._mark(
            tracker,
            center,
            label="static-low-texture",
            track_id="static-low-texture-track",
        )
        second = self._mark(
            tracker,
            center + [18.0, 0.0],
            label="static-low-texture-2",
            track_id="static-low-texture-track-2",
        )
        reference_distance = float(
            np.linalg.norm(
                np.asarray(second.point_robot_base_m)
                - np.asarray(first.point_robot_base_m)
            )
        )
        tracker.activate_rigid_pair(
            [first.track_id, second.track_id],
            reference_distance_m=reference_distance,
        )

        for sequence in range(2, 72):
            tracker.ingest(static_frame(sequence))
            for track_id, initial in (
                (first.track_id, first),
                (second.track_id, second),
            ):
                current = tracker.get_point(track_id)
                self.assertEqual(current.status, OBSERVED, current.as_dict())
                self.assertLess(
                    float(
                        np.linalg.norm(
                            np.asarray(current.point_policy_local_m)
                            - np.asarray(initial.point_policy_local_m)
                        )
                    ),
                    0.0045,
                    current.as_dict(),
                )
                self.assertEqual(
                    tracker._tracks[track_id].last_observation_method,
                    "registration_anchor_static",
                )
        self.assertTrue(tracker.rigid_pair_report()["ok"])

    def test_registered_rigid_pair_recovers_from_collapsed_incremental_state(
        self,
    ) -> None:
        rng = np.random.default_rng(811)
        texture = rng.integers(20, 235, size=(45, 91), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.5)
        cv2.rectangle(texture, (3, 3), (87, 41), 250, 2)
        cv2.line(texture, (9, 11), (79, 35), 8, 2)
        cv2.circle(texture, (29, 25), 8, 245, 2)
        cv2.circle(texture, (63, 18), 7, 15, 2)

        def transformed_frame(sequence, center, angle_deg):
            rgb, depth = self._render_transformed(
                center=center,
                object_depth=1.20,
                texture=texture,
                angle_deg=angle_deg,
                scale=1.0,
            )
            return TrackerFrame(
                rgb=rgb,
                depth_linear=depth,
                intrinsics=self.INTRINSICS,
                camera_to_robot_base=np.eye(4, dtype=np.float64),
                camera_to_policy=np.eye(4, dtype=np.float64),
                sequence=sequence,
                timestamp_s=0.1 * (sequence - 1),
                episode_id="episode-a",
                camera_role="head",
            )

        tracker = DynamicPointTracker()
        initial_center = np.array([91.0, 72.0])
        first_anchor = initial_center + [-10.0, 0.0]
        second_anchor = initial_center + [10.0, 0.0]
        tracker.ingest(transformed_frame(1, initial_center, 0.0))
        first = self._mark(
            tracker,
            first_anchor,
            label="head",
            track_id="head-track",
        )
        second = self._mark(
            tracker,
            second_anchor,
            label="tail",
            track_id="tail-track",
        )
        reference_distance = float(
            np.linalg.norm(
                np.asarray(second.point_robot_base_m)
                - np.asarray(first.point_robot_base_m)
            )
        )
        activation = tracker.activate_rigid_pair(
            ["head-track", "tail-track"],
            reference_distance_m=reference_distance,
        )
        self.assertTrue(activation["ok"])

        # Reproduce the online failure mode: independent incremental states
        # have converged to one patch, while immutable registration anchors and
        # support points still carry the two model-selected identities.
        with tracker._lock:
            collapsed_support = tracker._tracks["head-track"].support_px.copy()
            for track_id in ("head-track", "tail-track"):
                tracker._tracks[track_id].anchor_px = initial_center.copy()
                tracker._tracks[track_id].support_px = collapsed_support.copy()

        current_center = initial_center + [18.0, -7.0]
        angle_deg = 8.0
        tracker.ingest(transformed_frame(2, current_center, angle_deg))

        transform = cv2.getRotationMatrix2D(
            tuple(0.5 * (np.asarray(texture.shape[::-1]) - 1.0)),
            angle_deg,
            1.0,
        )
        texture_center = 0.5 * (np.asarray(texture.shape[::-1]) - 1.0)
        transform[:, 2] += current_center - texture_center
        local_first = first_anchor - initial_center + texture_center
        local_second = second_anchor - initial_center + texture_center
        expected_first = np.append(local_first, 1.0) @ transform.T
        expected_second = np.append(local_second, 1.0) @ transform.T

        recovered_first = tracker.get_point("head-track")
        recovered_second = tracker.get_point("tail-track")
        pair_report = tracker.rigid_pair_report()
        self.assertEqual(recovered_first.status, OBSERVED, recovered_first.as_dict())
        self.assertEqual(recovered_second.status, OBSERVED, recovered_second.as_dict())
        self.assertLess(
            float(
                np.linalg.norm(
                    np.asarray(recovered_first.pixel_uv) - expected_first
                )
            ),
            1.0,
            pair_report,
        )
        self.assertLess(
            float(
                np.linalg.norm(
                    np.asarray(recovered_second.pixel_uv) - expected_second
                )
            ),
            1.0,
            pair_report,
        )
        recovered_distance = float(
            np.linalg.norm(
                np.asarray(recovered_second.point_robot_base_m)
                - np.asarray(recovered_first.point_robot_base_m)
            )
        )
        self.assertLess(abs(recovered_distance - reference_distance), 0.005)
        self.assertTrue(pair_report["ok"], pair_report)
        self.assertEqual(
            pair_report["source"],
            "shared_similarity_registration_keyframe",
        )

    def test_registered_rigid_pair_stays_distinct_over_long_shared_motion(
        self,
    ) -> None:
        rng = np.random.default_rng(9917)
        texture = rng.integers(20, 235, size=(45, 91), dtype=np.uint8)
        texture = cv2.GaussianBlur(texture, (3, 3), 0.5)
        cv2.rectangle(texture, (3, 3), (87, 41), 250, 2)
        cv2.line(texture, (7, 35), (82, 9), 10, 2)
        cv2.circle(texture, (27, 17), 7, 245, 2)
        cv2.circle(texture, (66, 29), 8, 12, 2)

        def transformed_frame(sequence, center, angle_deg):
            rgb, depth = self._render_transformed(
                center=center,
                object_depth=1.20,
                texture=texture,
                angle_deg=angle_deg,
                scale=1.0,
            )
            return TrackerFrame(
                rgb=rgb,
                depth_linear=depth,
                intrinsics=self.INTRINSICS,
                camera_to_robot_base=np.eye(4, dtype=np.float64),
                camera_to_policy=np.eye(4, dtype=np.float64),
                sequence=sequence,
                timestamp_s=(sequence - 1) / 30.0,
                episode_id="episode-a",
                camera_role="head",
            )

        tracker = DynamicPointTracker()
        initial_center = np.array([96.0, 72.0])
        initial_anchors = np.asarray(
            [initial_center + [-12.0, 0.0], initial_center + [12.0, 0.0]],
            dtype=np.float64,
        )
        tracker.ingest(transformed_frame(1, initial_center, 0.0))
        snapshots = [
            self._mark(
                tracker,
                initial_anchors[index],
                label=name,
                track_id=f"{name}-long-track",
            )
            for index, name in enumerate(("head", "tail"))
        ]
        reference_distance = float(
            np.linalg.norm(
                np.asarray(snapshots[1].point_robot_base_m)
                - np.asarray(snapshots[0].point_robot_base_m)
            )
        )
        activation = tracker.activate_rigid_pair(
            ["head-long-track", "tail-long-track"],
            reference_distance_m=reference_distance,
        )
        self.assertTrue(activation["ok"], activation)

        texture_center = 0.5 * (np.asarray(texture.shape[::-1]) - 1.0)
        local_anchors = (
            initial_anchors - initial_center.reshape(1, 2) + texture_center
        )
        maximum_pixel_error = 0.0
        maximum_distance_error = 0.0
        previous_pixels = initial_anchors.copy()
        for step in range(1, 73):
            phase = 2.0 * np.pi * step / 72.0
            center = initial_center + np.array(
                [8.0 * np.sin(phase), 4.0 * np.cos(2.0 * phase)]
            )
            angle_deg = float(step)
            tracker.ingest(transformed_frame(step + 1, center, angle_deg))

            transform = cv2.getRotationMatrix2D(
                tuple(texture_center), angle_deg, 1.0
            )
            transform[:, 2] += center - texture_center
            expected_pixels = np.asarray(
                [np.append(anchor, 1.0) @ transform.T for anchor in local_anchors]
            )
            current = [
                tracker.get_point("head-long-track"),
                tracker.get_point("tail-long-track"),
            ]
            for index, snapshot in enumerate(current):
                self.assertEqual(snapshot.status, OBSERVED, snapshot.as_dict())
                pixel_error = float(
                    np.linalg.norm(
                        np.asarray(snapshot.pixel_uv) - expected_pixels[index]
                    )
                )
                maximum_pixel_error = max(maximum_pixel_error, pixel_error)
                self.assertLess(pixel_error, 2.0, tracker.rigid_pair_report())
            current_pixels = np.asarray(
                [snapshot.pixel_uv for snapshot in current], dtype=np.float64
            )
            self.assertFalse(
                np.allclose(current_pixels[0], current_pixels[1], atol=1.0)
            )
            self.assertLess(
                float(np.max(np.linalg.norm(current_pixels - previous_pixels, axis=1))),
                5.0,
            )
            previous_pixels = current_pixels
            current_distance = float(
                np.linalg.norm(
                    np.asarray(current[1].point_robot_base_m)
                    - np.asarray(current[0].point_robot_base_m)
                )
            )
            maximum_distance_error = max(
                maximum_distance_error,
                abs(current_distance - reference_distance),
            )
            report = tracker.rigid_pair_report()
            self.assertTrue(report["ok"], report)
            self.assertTrue(
                str(report["source"]).startswith("shared_similarity_"),
                report,
            )

        self.assertLess(maximum_pixel_error, 2.0)
        self.assertLess(maximum_distance_error, 0.006)

    def test_registered_rigid_pair_never_publishes_collapsed_candidates(self) -> None:
        tracker = DynamicPointTracker()
        center = np.array([92.0, 70.0])
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[(center, 1.15, self.can_texture)],
            )
        )
        first = self._mark(
            tracker,
            center + [-8.0, 0.0],
            label="first",
            track_id="first-track",
        )
        second = self._mark(
            tracker,
            center + [8.0, 0.0],
            label="second",
            track_id="second-track",
        )
        reference_distance = float(
            np.linalg.norm(
                np.asarray(second.point_robot_base_m)
                - np.asarray(first.point_robot_base_m)
            )
        )
        tracker.activate_rigid_pair(
            ["first-track", "second-track"],
            reference_distance_m=reference_distance,
        )

        blank = self._blank_frame(sequence=2, timestamp_s=0.1)
        tracker.ingest(blank)
        for track_id in ("first-track", "second-track"):
            snapshot = tracker.get_point(track_id)
            self.assertNotEqual(snapshot.status, OBSERVED, snapshot.as_dict())
            self.assertIsNone(snapshot.point_robot_base_m)
            self.assertIsNone(snapshot.depth_m)
        report = tracker.rigid_pair_report()
        self.assertFalse(report["ok"])
        self.assertEqual(report["reason"], "pair_candidate_unobserved")
        self.assertFalse(report["prediction_values_published_as_observation"])

    def test_individual_fallback_rejects_15061_wrong_depth_layer(self) -> None:
        payload, tracker, states, frame_factory, make_candidates = (
            self._on_hand_misassociation_case()
        )
        current = tracker._validate_frame(
            frame_factory(2, payload["wrong_individual_candidates"])
        )
        wrong = make_candidates(payload["wrong_individual_candidates"], current)
        tracker.activate_rigid_pair(
            [state.track_id for state in states],
            reference_distance_m=payload["reference_pair_distance_m"],
            max_distance_error_m=payload["pair_distance_gate_m"],
        )

        def individual(state, _frame, **_kwargs):
            return copy.deepcopy(wrong[state.track_id])

        with mock.patch.object(
            tracker, "_advance_rigid_pair_feature", return_value=None
        ), mock.patch.object(tracker, "_advance_track", side_effect=individual):
            updated, report = tracker._advance_rigid_pair(
                states,
                current,
                reference_distance_m=payload["reference_pair_distance_m"],
                max_distance_error_m=payload["pair_distance_gate_m"],
                observation_prior=None,
            )

        self.assertLess(
            report["distance_error_m"], payload["pair_distance_gate_m"]
        )
        second = report["per_point"]["on-hand-track-2"]
        self.assertFalse(second["ok"])
        self.assertEqual(second["reason"], "individual_prediction_pixel_mismatch")
        self.assertGreater(second["pixel_error_px"], 40.0)
        self.assertGreater(second["depth_layer_error_m"], 0.038)
        self.assertEqual(
            second["candidate_method"], "high_correlation_individual_fallback"
        )
        self.assertAlmostEqual(second["candidate_confidence"], 0.99)
        self.assertEqual(updated["on-hand-track-1"].snapshot.status, OBSERVED)
        rejected = updated["on-hand-track-2"].snapshot
        self.assertNotEqual(rejected.status, OBSERVED)
        self.assertIsNone(rejected.depth_m)
        self.assertIsNone(rejected.point_robot_base_m)
        self.assertFalse(report["prediction_values_published_as_observation"])

    def test_eef_prior_accepts_true_perspective_pair_without_shared_similarity(
        self,
    ) -> None:
        payload, tracker, states, frame_factory, make_candidates = (
            self._on_hand_misassociation_case()
        )
        initial = np.asarray(
            [state.snapshot.point_robot_base_m for state in states],
            dtype=np.float64,
        )
        angle = np.deg2rad(18.0)
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ],
            dtype=np.float64,
        )
        midpoint = np.mean(initial, axis=0)
        transformed = (
            (rotation @ (initial - midpoint).T).T
            + midpoint
            + np.array([0.015, -0.006, -0.020])
        )
        samples = []
        for point in transformed:
            depth = -float(point[2])
            samples.append(
                {
                    "pixel": [
                        360.0 + 306.0 * float(point[0]) / depth,
                        360.0 - 306.0 * float(point[1]) / depth,
                    ],
                    "depth_m": depth,
                }
            )
        current = tracker._validate_frame(frame_factory(2, samples))
        observed = make_candidates(samples, current)
        prior = {
            "points": {
                state.track_id: {
                    "point_robot_base_m": transformed[index],
                    "expected_pixel": np.asarray(samples[index]["pixel"]),
                    "expected_depth_m": float(samples[index]["depth_m"]),
                }
                for index, state in enumerate(states)
            },
            "max_pixel_error_px": 24.0,
            "max_depth_error_m": 0.025,
            "max_point_error_m": 0.030,
        }

        def individual(state, _frame, **_kwargs):
            return copy.deepcopy(observed[state.track_id])

        with mock.patch.object(
            tracker, "_advance_rigid_pair_feature", return_value=None
        ), mock.patch.object(tracker, "_advance_track", side_effect=individual):
            updated, report = tracker._advance_rigid_pair(
                states,
                current,
                reference_distance_m=payload["reference_pair_distance_m"],
                max_distance_error_m=payload["pair_distance_gate_m"],
                observation_prior=prior,
            )
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["source"], "jointly_validated_individual_candidates")
        self.assertTrue(
            all(state.snapshot.status == OBSERVED for state in updated.values())
        )
        self.assertTrue(
            all(
                item["gate_source"] == "explicit_on_hand_eef_local_fk"
                for item in report["per_point"].values()
            )
        )

    def test_rejected_eef_candidate_reacquires_from_real_rgbd_without_stale_xyz(
        self,
    ) -> None:
        payload, tracker, states, frame_factory, make_candidates = (
            self._on_hand_misassociation_case()
        )
        expected_points = np.asarray(
            [state.snapshot.point_robot_base_m for state in states],
            dtype=np.float64,
        )

        def prior_for(samples):
            return {
                "points": {
                    state.track_id: {
                        "point_robot_base_m": expected_points[index],
                        "expected_pixel": np.asarray(samples[index]["pixel"]),
                        "expected_depth_m": float(samples[index]["depth_m"]),
                    }
                    for index, state in enumerate(states)
                },
                "max_pixel_error_px": 24.0,
                "max_depth_error_m": 0.025,
                "max_point_error_m": 0.030,
            }

        wrong_frame = tracker._validate_frame(
            frame_factory(2, payload["wrong_individual_candidates"])
        )
        wrong = make_candidates(payload["wrong_individual_candidates"], wrong_frame)
        with mock.patch.object(
            tracker, "_advance_rigid_pair_feature", return_value=None
        ), mock.patch.object(
            tracker,
            "_advance_track",
            side_effect=lambda state, _frame, **_kwargs: copy.deepcopy(
                wrong[state.track_id]
            ),
        ):
            isolated, rejected_report = tracker._advance_rigid_pair(
                states,
                wrong_frame,
                reference_distance_m=payload["reference_pair_distance_m"],
                max_distance_error_m=payload["pair_distance_gate_m"],
                observation_prior=prior_for(payload["expected"]),
            )
        self.assertFalse(rejected_report["ok"])
        self.assertIsNone(isolated["on-hand-track-2"].snapshot.depth_m)
        np.testing.assert_array_equal(
            isolated["on-hand-track-2"].registration_gray,
            states[1].registration_gray,
        )

        good_frame = tracker._validate_frame(frame_factory(3, payload["expected"]))
        good = make_candidates(payload["expected"], good_frame)
        isolated_states = [isolated[state.track_id] for state in states]
        with mock.patch.object(
            tracker, "_advance_rigid_pair_feature", return_value=None
        ), mock.patch.object(
            tracker,
            "_advance_track",
            side_effect=lambda state, _frame, **_kwargs: copy.deepcopy(
                good[state.track_id]
            ),
        ):
            recovered, recovered_report = tracker._advance_rigid_pair(
                isolated_states,
                good_frame,
                reference_distance_m=payload["reference_pair_distance_m"],
                max_distance_error_m=payload["pair_distance_gate_m"],
                observation_prior=prior_for(payload["expected"]),
            )
        self.assertTrue(recovered_report["ok"], recovered_report)
        self.assertTrue(
            all(state.snapshot.status == OBSERVED for state in recovered.values())
        )

    def test_featureless_moving_surface_tracks_clicked_offset_and_depth(self) -> None:
        tracker = DynamicPointTracker()
        initial_center = np.array([58.0, 76.0])
        clicked_offset = np.array([8.0, -6.0])
        tracker.ingest(
            self._flat_surface_frame(
                sequence=1,
                timestamp_s=0.0,
                center=initial_center,
                object_depth=1.55,
            )
        )
        initial = self._mark(
            tracker,
            initial_center + clicked_offset,
            label="featureless-box",
            track_id="featureless-track",
        )
        self.assertEqual(initial.status, OBSERVED)
        self.assertLess(initial.confidence, 1.0)

        pixel_errors = []
        for sequence in range(2, 12):
            step = sequence - 1
            center = initial_center + [4.0 * step, -1.0 * step]
            expected_anchor = center + clicked_offset
            expected_depth = 1.55 - 0.025 * step
            tracker.ingest(
                self._flat_surface_frame(
                    sequence=sequence,
                    timestamp_s=0.1 * step,
                    center=center,
                    object_depth=expected_depth,
                )
            )
            current = tracker.get_point("featureless-track")
            self.assertEqual(current.status, OBSERVED, current.as_dict())
            pixel_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(current.pixel_uv) - expected_anchor
                    )
                )
            )
            self.assertAlmostEqual(current.depth_m, expected_depth, places=5)
            expected_camera = self._point_camera(
                expected_anchor,
                expected_depth,
            )
            np.testing.assert_allclose(
                current.point_camera_m,
                expected_camera,
                atol=0.015,
                rtol=0.0,
            )
        self.assertLess(max(pixel_errors), 0.75)

    def test_featureless_static_surface_uses_camera_pose_projection(self) -> None:
        tracker = DynamicPointTracker()
        first_center = np.array([126.0, 72.0])
        clicked_offset = np.array([-7.0, 4.0])
        depth = 2.0
        tracker.ingest(
            self._flat_surface_frame(
                sequence=1,
                timestamp_s=0.0,
                center=first_center,
                object_depth=depth,
            )
        )
        initial = self._mark(
            tracker,
            first_center + clicked_offset,
            label="featureless-static",
            track_id="featureless-static-track",
        )

        camera_translation_x = 0.60
        pixel_shift = -self.INTRINSICS.fx * camera_translation_x / depth
        second_center = first_center + [pixel_shift, 0.0]
        camera_to_policy = np.eye(4, dtype=np.float64)
        camera_to_policy[0, 3] = camera_translation_x
        tracker.ingest(
            self._flat_surface_frame(
                sequence=2,
                timestamp_s=0.1,
                center=second_center,
                object_depth=depth,
                camera_to_policy=camera_to_policy,
            )
        )
        current = tracker.get_point("featureless-static-track")
        self.assertEqual(current.status, OBSERVED, current.as_dict())
        np.testing.assert_allclose(
            current.pixel_uv,
            second_center + clicked_offset,
            atol=0.75,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            current.point_policy_local_m,
            initial.point_policy_local_m,
            atol=0.015,
            rtol=0.0,
        )

    def test_featureless_surface_loss_never_publishes_background_depth(self) -> None:
        tracker = DynamicPointTracker()
        center = np.array([92.0, 70.0])
        tracker.ingest(
            self._flat_surface_frame(
                sequence=1,
                timestamp_s=0.0,
                center=center,
                object_depth=1.25,
            )
        )
        self._mark(
            tracker,
            center,
            label="featureless-can",
            track_id="featureless-can-track",
        )
        tracker.ingest(
            self._flat_surface_frame(
                sequence=2,
                timestamp_s=0.1,
                center=center,
                object_depth=1.25,
                visible=False,
            )
        )
        hidden = tracker.get_point("featureless-can-track")
        self.assertEqual(hidden.status, OCCLUDED)
        self.assertIsNone(hidden.depth_m)
        self.assertIsNone(hidden.depth_source)
        self.assertAlmostEqual(hidden.predicted_depth_m, 1.25, places=5)

    def test_camera_motion_compensation_preserves_static_policy_point(self) -> None:
        tracker = DynamicPointTracker()
        first_center = np.array([96.0, 72.0])
        depth = 2.0
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[(first_center, depth, self.can_texture)],
            )
        )
        initial = self._mark(
            tracker,
            first_center,
            label="static-can",
            track_id="static-track",
        )

        camera_to_policy = np.eye(4, dtype=np.float64)
        camera_to_policy[0, 3] = 0.08
        second_center = np.array([90.0, 72.0])
        tracker.ingest(
            self._frame(
                sequence=2,
                timestamp_s=0.1,
                objects=[(second_center, depth, self.can_texture)],
                camera_to_policy=camera_to_policy,
            )
        )
        current = tracker.get_point("static-track")

        self.assertEqual(current.status, OBSERVED, current.as_dict())
        self.assertLess(
            float(
                np.linalg.norm(
                    np.asarray(current.point_policy_local_m)
                    - np.asarray(initial.point_policy_local_m)
                )
            ),
            0.008,
        )
        self.assertLess(
            float(np.linalg.norm(current.velocity_policy_local_m_s)),
            0.08,
        )
        np.testing.assert_allclose(current.pixel_uv, second_center, atol=0.45)

    def test_large_camera_motion_uses_policy_projection_as_flow_prior(self) -> None:
        tracker = DynamicPointTracker()
        first_center = np.array([122.0, 72.0])
        depth = 2.0
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[(first_center, depth, self.bin_texture)],
            )
        )
        initial = self._mark(
            tracker,
            first_center,
            label="static-bin",
            track_id="static-bin-track",
        )

        camera_translation_x = 0.60
        expected_pixel_shift = (
            -self.INTRINSICS.fx * camera_translation_x / depth
        )
        second_center = first_center + [expected_pixel_shift, 0.0]
        camera_to_policy = np.eye(4, dtype=np.float64)
        camera_to_policy[0, 3] = camera_translation_x
        tracker.ingest(
            self._frame(
                sequence=2,
                timestamp_s=0.1,
                objects=[(second_center, depth, self.bin_texture)],
                camera_to_policy=camera_to_policy,
            )
        )
        current = tracker.get_point("static-bin-track")

        self.assertEqual(current.status, OBSERVED, current.as_dict())
        np.testing.assert_allclose(current.pixel_uv, second_center, atol=0.55)
        np.testing.assert_allclose(
            current.point_policy_local_m,
            initial.point_policy_local_m,
            atol=0.008,
            rtol=0.0,
        )

    def test_combined_object_affine_and_camera_motion_matches_3d_oracle(self) -> None:
        tracker = DynamicPointTracker()
        initial_center = np.array([88.0, 69.0])
        initial_depth = 1.55
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[
                    (initial_center, initial_depth, self.can_texture),
                ],
            )
        )
        initial = self._mark(
            tracker,
            initial_center,
            label="moving-can",
            track_id="moving-can-track",
        )

        next_center = np.array([97.0, 63.0])
        next_depth = 1.32
        camera_to_robot_base = transform_from_pose(
            [0.01, 0.0, 0.02],
            [0.0, 0.0, 0.0, 1.0],
        )
        camera_to_policy = camera_to_policy_from_odometry(
            [0.04, -0.02, math.radians(3.0)],
            [0.01, 0.0, 0.02],
            [0.0, 0.0, 0.0, 1.0],
        )
        rgb, depth = self._render_transformed(
            center=next_center,
            object_depth=next_depth,
            texture=self.can_texture,
            angle_deg=9.0,
            scale=1.08,
        )
        tracker.ingest(
            TrackerFrame(
                rgb=rgb,
                depth_linear=depth,
                intrinsics=self.INTRINSICS,
                camera_to_robot_base=camera_to_robot_base,
                camera_to_policy=camera_to_policy,
                sequence=2,
                timestamp_s=0.2,
                episode_id="episode-a",
                camera_role="head",
            )
        )
        current = tracker.get_point("moving-can-track")

        self.assertEqual(current.status, OBSERVED, current.as_dict())
        np.testing.assert_allclose(current.pixel_uv, next_center, atol=0.75)
        self.assertAlmostEqual(current.depth_m, next_depth, places=5)
        expected_camera = self._point_camera(next_center, next_depth)
        expected_policy = self._point_policy(
            camera_to_policy,
            expected_camera,
        )
        np.testing.assert_allclose(
            current.point_camera_m,
            expected_camera,
            atol=0.008,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            current.point_policy_local_m,
            expected_policy,
            atol=0.008,
            rtol=0.0,
        )
        expected_velocity = (
            expected_policy - np.asarray(initial.point_policy_local_m)
        ) / 0.2
        np.testing.assert_allclose(
            current.velocity_policy_local_m_s,
            expected_velocity,
            atol=0.04,
            rtol=0.0,
        )

    def test_deterministic_affine_depth_pose_stress_accuracy(self) -> None:
        rng = np.random.default_rng(20260812)
        pixel_errors = []
        camera_point_errors = []
        policy_point_errors = []
        for case in range(24):
            tracker = DynamicPointTracker()
            initial_center = np.array([96.0, 72.0])
            initial_depth = float(rng.uniform(0.85, 2.40))
            tracker.ingest(
                self._frame(
                    sequence=1,
                    timestamp_s=0.0,
                    objects=[
                        (initial_center, initial_depth, self.can_texture),
                    ],
                    episode_id=f"stress-{case}",
                )
            )
            self._mark(
                tracker,
                initial_center,
                label="stress-can",
                track_id="stress-track",
            )

            next_center = initial_center + rng.uniform(
                [-14.0, -11.0],
                [14.0, 11.0],
            )
            next_depth = float(rng.uniform(0.85, 2.40))
            angle_deg = float(rng.uniform(-12.0, 12.0))
            scale = float(rng.uniform(0.90, 1.12))
            base_pose = [
                float(rng.uniform(-0.12, 0.12)),
                float(rng.uniform(-0.12, 0.12)),
                math.radians(float(rng.uniform(-8.0, 8.0))),
            ]
            camera_relative = [
                float(rng.uniform(-0.03, 0.03)),
                float(rng.uniform(-0.03, 0.03)),
                float(rng.uniform(0.0, 0.05)),
            ]
            camera_to_robot_base = transform_from_pose(
                camera_relative,
                [0.0, 0.0, 0.0, 1.0],
            )
            camera_to_policy = camera_to_policy_from_odometry(
                base_pose,
                camera_relative,
                [0.0, 0.0, 0.0, 1.0],
            )
            rgb, depth = self._render_transformed(
                center=next_center,
                object_depth=next_depth,
                texture=self.can_texture,
                angle_deg=angle_deg,
                scale=scale,
            )
            tracker.ingest(
                TrackerFrame(
                    rgb=rgb,
                    depth_linear=depth,
                    intrinsics=self.INTRINSICS,
                    camera_to_robot_base=camera_to_robot_base,
                    camera_to_policy=camera_to_policy,
                    sequence=2,
                    timestamp_s=0.1,
                    episode_id=f"stress-{case}",
                    camera_role="head",
                )
            )
            current = tracker.get_point("stress-track")
            self.assertEqual(current.status, OBSERVED, current.as_dict())
            expected_camera = self._point_camera(next_center, next_depth)
            expected_policy = self._point_policy(
                camera_to_policy,
                expected_camera,
            )
            pixel_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(current.pixel_uv) - next_center
                    )
                )
            )
            camera_point_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(current.point_camera_m) - expected_camera
                    )
                )
            )
            policy_point_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(current.point_policy_local_m)
                        - expected_policy
                    )
                )
            )
            self.assertAlmostEqual(current.depth_m, next_depth, places=5)

        self.assertLess(max(pixel_errors), 0.80)
        self.assertLess(max(camera_point_errors), 0.012)
        self.assertLess(max(policy_point_errors), 0.012)

    def test_occlusion_never_reports_background_depth_as_target_depth(self) -> None:
        config = DynamicPointTrackerConfig(max_occluded_steps=2)
        tracker = DynamicPointTracker(config=config)
        center = np.array([92.0, 70.0])
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[(center, 1.25, self.can_texture)],
            )
        )
        self._mark(
            tracker,
            center,
            label="can",
            track_id="can-track",
        )

        tracker.ingest(self._blank_frame(sequence=2, timestamp_s=0.1))
        hidden = tracker.get_point("can-track")
        self.assertEqual(hidden.status, OCCLUDED)
        self.assertIsNone(hidden.uv)
        self.assertIsNone(hidden.pixel_uv)
        self.assertIsNone(hidden.depth_m)
        self.assertIsNone(hidden.point_camera_m)
        self.assertIsNone(hidden.point_robot_base_m)
        self.assertIsNone(hidden.point_policy_local_m)
        self.assertAlmostEqual(hidden.predicted_depth_m, 1.25, places=5)
        self.assertIsNone(hidden.depth_source)

        tracker.ingest(self._blank_frame(sequence=3, timestamp_s=0.2))
        self.assertEqual(tracker.get_point("can-track").status, OCCLUDED)
        tracker.ingest(self._blank_frame(sequence=4, timestamp_s=0.3))
        self.assertEqual(tracker.get_point("can-track").status, LOST)
        tracker.ingest(self._blank_frame(sequence=5, timestamp_s=0.4))
        permanently_lost = tracker.get_point("can-track")
        self.assertEqual(permanently_lost.status, LOST)
        self.assertEqual(permanently_lost.observation_sequence, 5)

    def test_short_occlusion_reacquires_same_track_with_current_depth(self) -> None:
        tracker = DynamicPointTracker(
            config=DynamicPointTrackerConfig(max_occluded_steps=3)
        )
        first_center = np.array([84.0, 66.0])
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[(first_center, 1.45, self.can_texture)],
            )
        )
        self._mark(
            tracker,
            first_center,
            label="can",
            track_id="can-track",
        )
        tracker.ingest(self._blank_frame(sequence=2, timestamp_s=0.1))
        self.assertEqual(tracker.get_point("can-track").status, OCCLUDED)

        reappeared_center = first_center + [7.0, -3.0]
        tracker.ingest(
            self._frame(
                sequence=3,
                timestamp_s=0.2,
                objects=[
                    (reappeared_center, 1.30, self.can_texture),
                ],
            )
        )
        reacquired = tracker.get_point("can-track")
        self.assertEqual(reacquired.status, OBSERVED, reacquired.as_dict())
        self.assertEqual(reacquired.track_id, "can-track")
        self.assertEqual(reacquired.missed_steps, 0)
        self.assertEqual(reacquired.last_seen_sequence, 3)
        np.testing.assert_allclose(
            reacquired.pixel_uv,
            reappeared_center,
            atol=0.55,
            rtol=0.0,
        )
        self.assertAlmostEqual(reacquired.depth_m, 1.30, places=5)
        self.assertEqual(
            reacquired.depth_source,
            "current_evaluator_depth_linear",
        )

    def test_lost_track_can_reacquire_from_retained_registration_template(self) -> None:
        tracker = DynamicPointTracker(
            config=DynamicPointTrackerConfig(max_occluded_steps=1)
        )
        center = np.array([84.0, 66.0])
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[(center, 1.45, self.can_texture)],
            )
        )
        self._mark(
            tracker,
            center,
            label="can",
            track_id="can-track",
        )
        tracker.ingest(self._blank_frame(sequence=2, timestamp_s=0.1))
        tracker.ingest(self._blank_frame(sequence=3, timestamp_s=0.2))
        self.assertEqual(tracker.get_point("can-track").status, LOST)

        reappeared = center + [6.0, -2.0]
        tracker.ingest(
            self._frame(
                sequence=4,
                timestamp_s=0.3,
                objects=[(reappeared, 1.20, self.can_texture)],
            )
        )
        recovered = tracker.get_point("can-track")
        self.assertEqual(recovered.status, OBSERVED, recovered.as_dict())
        self.assertEqual(recovered.track_id, "can-track")
        self.assertEqual(recovered.last_seen_sequence, 4)
        self.assertEqual(recovered.missed_steps, 0)
        self.assertAlmostEqual(recovered.depth_m, 1.20, places=5)

    def test_overlapping_tracks_are_ambiguous_not_silently_merged(self) -> None:
        tracker = DynamicPointTracker()
        center = np.array([96.0, 72.0])
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[(center, 1.4, self.can_texture)],
            )
        )
        self._mark(
            tracker,
            center + [-1.0, 0.0],
            label="can-a",
            track_id="track-a",
        )
        self._mark(
            tracker,
            center + [1.0, 0.0],
            label="can-b",
            track_id="track-b",
        )

        tracker.ingest(
            self._frame(
                sequence=2,
                timestamp_s=0.1,
                objects=[(center + [4.0, 1.0], 1.4, self.can_texture)],
            )
        )
        left = tracker.get_point("track-a")
        right = tracker.get_point("track-b")
        self.assertEqual(left.status, AMBIGUOUS)
        self.assertEqual(right.status, AMBIGUOUS)
        self.assertIsNone(left.depth_m)
        self.assertIsNone(right.depth_m)

    def test_episode_change_clears_policy_owned_tracks(self) -> None:
        tracker = DynamicPointTracker()
        center = np.array([82.0, 68.0])
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[(center, 1.0, self.can_texture)],
            )
        )
        self._mark(
            tracker,
            center,
            label="can",
            track_id="can-track",
        )

        updates = tracker.ingest(
            self._frame(
                sequence=2,
                timestamp_s=0.1,
                objects=[(center, 1.0, self.can_texture)],
                episode_id="episode-b",
            )
        )
        self.assertEqual(updates, ())
        self.assertEqual(tracker.list_points(), ())
        self.assertEqual(tracker.episode_id, "episode-b")
        with self.assertRaises(KeyError):
            tracker.get_point("can-track")

    def test_input_validation_rejects_stale_mixed_or_invalid_observations(self) -> None:
        tracker = DynamicPointTracker()
        center = np.array([82.0, 68.0])
        frame = self._frame(
            sequence=1,
            timestamp_s=0.0,
            objects=[(center, 1.0, self.can_texture)],
        )
        tracker.ingest(frame)
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            tracker.ingest(frame)
        with self.assertRaisesRegex(ValueError, "0..1000"):
            tracker.mark_point(u=-1, v=500, label="can")

        rgb, depth = self._render([(center, 1.0, self.can_texture)])
        bad_resolution = TrackerFrame(
            rgb=rgb[:-1],
            depth_linear=depth,
            intrinsics=self.INTRINSICS,
            camera_to_robot_base=np.eye(4),
            camera_to_policy=np.eye(4),
            sequence=2,
            timestamp_s=0.1,
            episode_id="episode-a",
        )
        with self.assertRaisesRegex(ValueError, "resolutions do not match"):
            tracker.ingest(bad_resolution)

        invalid_depth = depth.copy()
        invalid_depth[66:71, 80:85] = np.nan
        tracker_with_invalid_depth = DynamicPointTracker()
        tracker_with_invalid_depth.ingest(
            TrackerFrame(
                rgb=rgb,
                depth_linear=invalid_depth,
                intrinsics=self.INTRINSICS,
                camera_to_robot_base=np.eye(4),
                camera_to_policy=np.eye(4),
                sequence=1,
                timestamp_s=0.0,
                episode_id="episode-a",
            )
        )
        u, v = self._relative_uv(center)
        with self.assertRaisesRegex(ValueError, "no valid evaluator depth"):
            tracker_with_invalid_depth.mark_point(u=u, v=v, label="can")

    def test_policy_camera_transform_composes_odometry_and_relative_pose(self) -> None:
        transform = camera_to_policy_from_odometry(
            [2.0, 3.0, math.pi / 2.0],
            [1.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            transform[:3, 3],
            [2.0, 4.0, 0.5],
            atol=1e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            transform[:3, :3],
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            atol=1e-12,
            rtol=0.0,
        )

    def test_xyz_in_current_robot_base_coord_uses_camera_relative_pose_only(
        self,
    ) -> None:
        tracker = DynamicPointTracker()
        center = np.array([82.0, 61.0])
        depth_m = 1.35
        camera_position_robot = [0.31, -0.22, 1.08]
        half_sqrt = math.sqrt(0.5)
        camera_quaternion_robot = [0.0, 0.0, half_sqrt, half_sqrt]
        camera_to_robot_base = transform_from_pose(
            camera_position_robot,
            camera_quaternion_robot,
        )
        camera_to_policy = camera_to_policy_from_odometry(
            [4.0, -3.0, math.radians(37.0)],
            camera_position_robot,
            camera_quaternion_robot,
        )
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[(center, depth_m, self.can_texture)],
                camera_to_robot_base=camera_to_robot_base,
                camera_to_policy=camera_to_policy,
            )
        )

        snapshot = self._mark(
            tracker,
            center,
            label="can",
            track_id="can-track",
        )
        point_camera = self._point_camera(center, depth_m)
        expected_robot_base = np.array(
            [
                camera_position_robot[0] - point_camera[1],
                camera_position_robot[1] + point_camera[0],
                camera_position_robot[2] + point_camera[2],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            snapshot.point_robot_base_m,
            expected_robot_base,
            atol=1e-7,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            snapshot.as_dict()["point_robot_base_m"],
            expected_robot_base,
            atol=1e-7,
            rtol=0.0,
        )
        self.assertGreater(
            float(
                np.linalg.norm(
                    np.asarray(snapshot.point_policy_local_m)
                    - expected_robot_base
                )
            ),
            1.0,
        )

    def test_rejects_invalid_camera_to_robot_base_transform(self) -> None:
        frame = self._blank_frame(sequence=1, timestamp_s=0.0)
        invalid = np.asarray(frame.camera_to_robot_base).copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "camera_to_robot_base must be finite"):
            DynamicPointTracker().ingest(
                TrackerFrame(
                    rgb=frame.rgb,
                    depth_linear=frame.depth_linear,
                    intrinsics=frame.intrinsics,
                    camera_to_robot_base=invalid,
                    camera_to_policy=frame.camera_to_policy,
                    sequence=frame.sequence,
                    timestamp_s=frame.timestamp_s,
                    episode_id=frame.episode_id,
                    camera_role=frame.camera_role,
                )
            )

    def test_deferred_depth_component_matches_eager_result(self) -> None:
        tracker = DynamicPointTracker()
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[((84, 70), 1.25, self.can_texture)],
            )
        )
        snapshot = tracker.mark_point(
            u=84 / 191 * 1000,
            v=70 / 143 * 1000,
            label="can",
        )
        state = tracker._tracks[snapshot.track_id]
        depth = tracker._frame.depth.copy()
        anchor = np.array([86.2, 71.4], dtype=np.float64)
        expected = tracker._depth_component_at_anchor(depth, anchor, 1.25)
        self.assertIsNotNone(expected)

        state.component_offset_px = np.array([99.0, 98.0])
        state.component_size_px = np.array([97.0, 96.0])
        state.component_area_px = 95
        tracker._defer_depth_component_update(state, depth, anchor, 1.25)
        np.testing.assert_array_equal(state.component_offset_px, [99.0, 98.0])
        tracker._resolve_deferred_depth_component(state)

        np.testing.assert_array_equal(
            state.component_offset_px,
            anchor - expected.centroid_px,
        )
        np.testing.assert_array_equal(state.component_size_px, expected.size_px)
        self.assertEqual(state.component_area_px, expected.area_px)
        self.assertIsNone(state.pending_component_depth_roi)
        self.assertIsNone(state.pending_component_origin_px)
        self.assertIsNone(state.pending_component_anchor_px)
        self.assertIsNone(state.pending_component_depth_m)

    def test_support_depth_roi_mask_matches_full_frame_reference(self) -> None:
        tracker = DynamicPointTracker()
        validated = tracker._validate_frame(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[((84, 70), 1.25, self.can_texture)],
            )
        )
        anchor = np.array([84.0, 70.0], dtype=np.float64)
        actual = tracker._select_support_points(
            validated.gray,
            validated.depth,
            anchor,
            1.25,
        )

        mask = np.zeros(validated.gray.shape, dtype=np.uint8)
        center = (84, 70)
        cv2.circle(mask, center, tracker.config.patch_radius_px, 255, -1)
        valid_depth = np.isfinite(validated.depth) & (validated.depth > 0.0)
        valid_depth &= (
            np.abs(validated.depth.astype(np.float64) - 1.25)
            <= tracker._depth_tolerance(1.25)
        )
        mask[~valid_depth] = 0
        corners = cv2.goodFeaturesToTrack(
            validated.gray,
            maxCorners=tracker.config.max_support_points,
            qualityLevel=tracker.config.feature_quality,
            minDistance=tracker.config.feature_min_distance_px,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        expected = corners.reshape(-1, 2).astype(np.float64)
        order = np.argsort(
            np.linalg.norm(expected - anchor.reshape(1, 2), axis=1)
        )
        np.testing.assert_array_equal(actual, expected[order])

    def test_opencl_lk_parity_gate_ignores_only_rejected_points(self) -> None:
        status = np.array([[1], [0]], dtype=np.uint8)
        cpu = (
            np.array([[[10.0, 20.0]], [[30.0, 40.0]]], dtype=np.float32),
            status,
            np.zeros((2, 1), dtype=np.float32),
        )
        gpu = (
            np.array([[[10.0, 20.0]], [[300.0, 400.0]]], dtype=np.float32),
            status.copy(),
            np.ones((2, 1), dtype=np.float32),
        )
        self.assertTrue(DynamicPointTracker._lk_results_match(cpu, gpu))
        gpu[0][0, 0, 0] += np.float32(0.25)
        self.assertFalse(DynamicPointTracker._lk_results_match(cpu, gpu))

    def test_opencl_umat_cache_is_bounded_and_reuses_live_images(self) -> None:
        tracker = DynamicPointTracker()
        images = [
            np.full((8, 8), index, dtype=np.uint8)
            for index in range(OPENCL_UMAT_CACHE_MAX_ENTRIES + 2)
        ]

        first = tracker._opencl_umat(images[0])
        self.assertIs(tracker._opencl_umat(images[0]), first)
        for image in images[1:]:
            tracker._opencl_umat(image)

        status = tracker.acceleration_status()["umat_cache"]
        self.assertEqual(status["entries"], OPENCL_UMAT_CACHE_MAX_ENTRIES)
        self.assertEqual(status["hits"], 1)
        self.assertEqual(status["uploads"], len(images))
        self.assertEqual(status["evictions"], 2)
        self.assertNotIn(id(images[0]), tracker._opencl_umat_cache)

        tracker.reset()
        self.assertEqual(
            tracker.acceleration_status()["umat_cache"]["entries"], 0
        )

    def test_track_images_share_the_owned_validated_frame(self) -> None:
        frame = self._frame(
            sequence=1,
            timestamp_s=0.0,
            objects=[((84, 70), 1.25, self.can_texture)],
        )
        tracker = DynamicPointTracker()
        tracker.ingest(frame)
        owned_gray = tracker._frame.gray
        snapshot = tracker.mark_point(
            u=84 / 191 * 1000,
            v=70 / 143 * 1000,
            label="can",
        )
        state = tracker._tracks[snapshot.track_id]

        self.assertIs(state.reference_gray, owned_gray)
        self.assertIs(state.registration_gray, owned_gray)
        expected = owned_gray.copy()
        frame.rgb[...] = 0
        np.testing.assert_array_equal(owned_gray, expected)

    def test_validated_grayscale_input_is_owned(self) -> None:
        rgb, depth = self._render(
            [((84, 70), 1.25, self.can_texture)]
        )
        grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        frame = TrackerFrame(
            rgb=grayscale,
            depth_linear=depth,
            intrinsics=self.INTRINSICS,
            camera_to_robot_base=np.eye(4, dtype=np.float64),
            camera_to_policy=np.eye(4, dtype=np.float64),
            sequence=1,
            timestamp_s=0.0,
            episode_id="episode-a",
            color_order="rgb",
        )
        tracker = DynamicPointTracker()
        tracker.ingest(frame)
        expected = tracker._frame.gray.copy()

        grayscale[...] = 0
        np.testing.assert_array_equal(tracker._frame.gray, expected)

    def test_lightweight_track_clone_shares_only_immutable_images(self) -> None:
        tracker = DynamicPointTracker()
        tracker.ingest(
            self._frame(
                sequence=1,
                timestamp_s=0.0,
                objects=[((84, 70), 1.25, self.can_texture)],
            )
        )
        snapshot = tracker.mark_point(
            u=84 / 191 * 1000,
            v=70 / 143 * 1000,
            label="can",
        )
        state = tracker._tracks[snapshot.track_id]
        candidate = tracker._clone_track_state(state)

        self.assertIs(candidate.reference_gray, state.reference_gray)
        self.assertIs(candidate.registration_gray, state.registration_gray)
        self.assertIsNot(candidate.anchor_px, state.anchor_px)
        self.assertIsNot(candidate.support_px, state.support_px)
        original_anchor = state.anchor_px.copy()
        candidate.anchor_px += 10.0
        np.testing.assert_array_equal(state.anchor_px, original_anchor)

    def test_tracker_module_has_no_simulator_or_production_tool_imports(self) -> None:
        path = (
            Path(__file__).resolve().parent
            / "tool"
            / "official_v2"
            / "dynamic_point_tracker.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        forbidden_prefixes = (
            "omnigibson",
            "behavior_interface.skills",
            "behavior_interface.tool.v2",
            "behavior_interface.world_api",
            "behavior_interface.server",
        )
        self.assertFalse(
            [
                module
                for module in imported
                if module.startswith(forbidden_prefixes)
            ]
        )


if __name__ == "__main__":
    unittest.main()
