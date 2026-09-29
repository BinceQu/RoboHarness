"""Process isolation, cancellation and recorded coordinate-goal regressions."""

import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from unittest import mock
from types import SimpleNamespace

import numpy as np

from behavior_interface_eval_test.robot_contract import ARM_DOF, ROBOT_PROFILE
from behavior_interface_eval_test.tool.official_v2 import tools
from behavior_interface_eval_test.tool.official_v2.contract import validate_move_tracked_point_args
from behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local import LocalRobotState, eef_pose
from behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local import PlanningDeadlineExceeded, plan_endpoint
from behavior_interface_eval_test.tool.official_v2.tracked_point_planning_worker import _decode, _encode, run_plan_worker


def tv_door_fixture():
    path = Path(__file__).parent / "fixtures" / "move_tracked_tv_door_15061.json"
    fixture = json.loads(path.read_text())
    robot = dict(fixture["robot"])
    robot["base_pose"] = {"pos": [0, 0, 0], "quat": [0, 0, 0, 1]}
    state = LocalRobotState.from_capture(robot, arm_dof=8)
    args = validate_move_tracked_point_args(fixture["request"])
    kwargs = dict(
        state=state, arm="right", q_start=state.arm_right_q.copy(),
        source_points=np.array(fixture["source_points_robot_base_m"]),
        target_points=tools._move_tracked_planning_targets(args["points"], ["1", "2", "3"]),
        off_hand_points=np.array(fixture["fixed_points_robot_base_m"]),
        off_hand_target_points=tools._move_tracked_planning_targets(args["points"], ["4"]),
        on_hand_names=["1", "2", "3"], off_hand_names=["4"],
        relations=args["relations"], pos_tol_m=args["pos_tol"],
        ori_tol_deg=args["ori_tol_deg"], max_steps=args["max_steps"],
    )
    return fixture, kwargs


class PlanningWorkerCodecTest(unittest.TestCase):
    def test_numpy_round_trip_and_json_finite_diagnostics(self):
        value = {"q": np.array([0., 1.]), "optional": float("inf"), "rank": np.int64(1)}
        wire = json.dumps(_encode(value), allow_nan=False)
        decoded = _decode(json.loads(wire))
        np.testing.assert_array_equal(decoded["q"], value["q"])
        self.assertIsNone(decoded["optional"])
        self.assertEqual(decoded["rank"], 1)

    def test_nonfinite_input_and_array_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _encode({"q": np.array([np.nan])}, strict=True)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _decode({"__planning_ndarray__": [None]})


class FrozenPlanCaptureTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((12, 16, 3), 91, dtype=np.uint8)
        self.depth = np.full((12, 16), 0.8, dtype=np.float32)
        self.pose = {"pos": [0.2, 0., 0.5], "quat": [0., 0., 0., 1.]}
        self.adapter = SimpleNamespace(
            observation_metadata=mock.Mock(return_value=(9, 1.0)),
            camera_frame=mock.Mock(side_effect=lambda role: self.rgb),
            camera_depth_frame=mock.Mock(side_effect=lambda role: self.depth),
            camera_relative_poses=mock.Mock(side_effect=lambda: {"head": self.pose}),
        )
        self.ctx = SimpleNamespace(world=SimpleNamespace(_official_adapter=self.adapter))

    def freeze(self):
        return tools._move_tracked_freeze_plan_capture(
            self.ctx, session_id="frozen-unit", registration_image_id="img_old",
            observation_sequence=9,
        )

    def test_freeze_uses_current_pixels_pose_proprio_and_preserves_original_binding(self):
        robot = {"episode_id": "unit-episode", "arm_right_qpos": [0.] * ARM_DOF}
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": root}), mock.patch.object(
            tools, "_capture_robot_state", return_value=robot
        ):
            result = self.freeze()
            self.assertNotEqual(result["image_id"], "img_old")
            saved = tools._load_frozen_capture("frozen-unit", result["image_id"])
            self.assertEqual(saved.evaluator_sequence, 9)
            self.assertEqual(saved.robot, robot)
            self.assertEqual(saved.camera["robot_relative_pose"]["pos"], self.pose["pos"])
            self.rgb[:] = 0
            self.depth[:] = 0
            self.assertTrue(np.all(saved.rgb == 91))
            self.assertTrue(np.all(saved.depth == np.float32(.8)))
            self.assertTrue(result["all_inputs_same_observation"])
            self.assertEqual(result["registration_image_id"], "img_old")

    def test_missing_or_mixed_observation_is_rejected_before_storage(self):
        for case in ("rgb", "depth", "pose", "size", "invalid_depth", "invalid_pose", "changed_sequence"):
            with self.subTest(case=case):
                self.setUp()
                if case == "rgb":
                    self.rgb = None
                elif case == "depth":
                    self.depth = None
                elif case == "pose":
                    self.pose = None
                elif case == "size":
                    self.depth = np.ones((3, 4))
                elif case == "invalid_depth":
                    self.depth[:] = np.nan
                elif case == "invalid_pose":
                    self.pose["pos"][0] = np.nan
                else:
                    self.adapter.observation_metadata.side_effect = [(9, 1.), (10, 2.)]
                with mock.patch.object(tools, "_capture_robot_state", return_value={}), mock.patch.object(
                    tools, "_ensure_session"
                ) as create:
                    with self.assertRaises(ValueError):
                        self.freeze()
                    create.assert_not_called()


class RegisteredBundleReacquisitionTest(unittest.TestCase):
    def setUp(self):
        self.sequence = 10
        self.live_overrides = {}
        self.ctx = SimpleNamespace(world=SimpleNamespace(episode_id=lambda: "episode"))
        self.adapter = SimpleNamespace(observation_metadata=lambda: (self.sequence, 1.))
        self.lease = {}
        self.clean = {
            "ok": True, "applicable": True, "applied": False,
            "outlier_names": [], "max_live_to_registered_eef_error_m": .001,
        }
        self.report = dict(self.clean)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        def patch(name, **kwargs):
            return self.stack.enter_context(mock.patch.object(tools, name, **kwargs))
        self.bootstrap = patch("_move_tracked_bootstrap_on_hand_eef_prior", return_value={
            "ok": True, "active": True, "lease_id": "recovery-lease",
            "activation_observation_sequence": 10,
        })
        patch("_move_tracked_planning_hold_action", return_value="hold")
        patch("_adjust_adapter", side_effect=lambda ctx: (self.adapter, self.sequence))
        patch("_adjust_kinematic_observation", return_value={})
        patch("_move_tracked_capture_state", side_effect=lambda ctx, obs: {"sequence": self.sequence})
        self.snapshot = patch("_move_tracked_live_snapshot", side_effect=lambda *a, **k: {
            "ok": True, "episode_id": "episode", "session_id": "session",
            "image_id": "img_registered", "observation_sequence": self.sequence,
            "entries": {name: {"xyz_in_robot_base_coord_m": [self.sequence, i, 0.]}
                        for i, name in enumerate(["1", "4", "2", "3"])},
            **self.live_overrides,
        })
        self.reconcile = patch("_move_tracked_reconcile_registered_on_hand_source",
                               side_effect=lambda *a, **k: dict(self.report))

    def start(self, deadline=None):
        return tools._move_tracked_reacquire_registered_bundle(
            self.ctx, object(), names=["1", "4", "2", "3"], controlled_names=["1", "2", "3"],
            episode_id="episode", session_id="session", image_id="img_registered",
            observation_sequence=10, hold_state={}, initial_report={"ok": False},
            lease_state=self.lease, deadline_monotonic=deadline or time.monotonic()+30,
        )

    def drain(self, generator):
        for _ in range(30):
            try:
                self.assertEqual(next(generator), "hold")
                self.sequence += 1
            except StopIteration as stopped:
                return stopped.value
        self.fail("reacquisition exceeded step budget")

    def test_two_new_visual_frames_freeze_on_and_off_hand_together(self):
        result = self.drain(self.start())
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["observation_sequence"], 12)
        self.assertEqual(result["report"]["hold_steps"], 2)
        self.assertEqual(self.lease["lease_id"], "recovery-lease")
        self.assertEqual(result["capture_state"]["sequence"], 12)
        self.assertEqual(result["source_by_name"]["4"][0], 12)
        self.assertTrue(np.all(result["source_points"][:, 0] == 12))
        self.assertFalse(result["report"]["predictions_published_as_observation"])
        self.snapshot.assert_called_with(mock.ANY, ["1", "4", "2", "3"], episode_id="episode")

    def test_model_repair_and_persistent_drift_never_count_as_visual_confirmation(self):
        for report in ({**self.clean, "applied": True}, {"ok": False, "outlier_names": ["1"]}):
            with self.subTest(report=report):
                self.sequence = 10
                self.report = report
                result = self.drain(self.start())
                self.assertFalse(result["ok"])
                self.assertEqual(result["reason"], "registered_bundle_reacquisition_timeout")
                self.assertEqual(result["hold_steps"], tools.MOVE_TRACKED_POINT_REGISTRATION_REACQUIRE_MAX_STEPS)

    def test_unavailable_frame_resets_consecutive_confirmation(self):
        generator = self.start()
        self.assertEqual(next(generator), "hold")
        self.sequence += 1
        self.assertEqual(next(generator), "hold")
        self.sequence += 1
        self.live_overrides["ok"] = False
        self.assertEqual(next(generator), "hold")
        self.sequence += 1
        self.live_overrides.clear()
        result = self.drain(generator)
        self.assertTrue(result["ok"])
        self.assertEqual(result["report"]["hold_steps"], 4)

    def test_stale_sequence_mixed_frame_binding_and_episode_fail_closed(self):
        for case in ("stale", "mixed", "binding", "episode"):
            with self.subTest(case=case):
                self.sequence = 10
                self.live_overrides.clear()
                self.ctx.world.episode_id = lambda: "episode"
                generator = self.start()
                self.assertEqual(next(generator), "hold")
                if case != "stale":
                    self.sequence += 1
                if case == "mixed":
                    self.live_overrides["observation_sequence"] = 9
                elif case == "binding":
                    self.live_overrides["image_id"] = "new_registration"
                elif case == "episode":
                    self.ctx.world.episode_id = lambda: "new_episode"
                with self.assertRaises(StopIteration) as stop:
                    next(generator)
                self.assertFalse(stop.exception.value["ok"])
                self.assertNotEqual(stop.exception.value["reason"], "registered_bundle_reacquisition_timeout")

    def test_timeout_and_cancel_do_not_lose_lease(self):
        result = self.drain(self.start(deadline=time.monotonic()-1))
        self.assertFalse(result["ok"])
        self.bootstrap.assert_not_called()
        generator = self.start()
        self.assertEqual(next(generator), "hold")
        self.ctx.raise_if_cancelled = mock.Mock(side_effect=RuntimeError("cancelled"))
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            next(generator)
        self.assertEqual(self.lease["lease_id"], "recovery-lease")


@unittest.skipUnless(ARM_DOF == 8, "recorded evaluator uses the 8-DOF contract")
class PlanningWorkerTest(unittest.TestCase):
    def simple_request(self):
        _fixture, kwargs = tv_door_fixture()
        position, _quaternion = eef_pose(kwargs["state"], "right", kwargs["q_start"])
        kwargs.update(source_points=np.array([position]),
                      target_points=[{"name": "p", "target_xyz_m": position.tolist()}],
                      off_hand_points=None, off_hand_target_points=None,
                      on_hand_names=["p"], off_hand_names=[], relations=[])
        return kwargs

    def assert_reaped(self, reports):
        self.assertTrue(reports[-1]["worker_reaped"])
        with self.assertRaises(ProcessLookupError):
            os.kill(reports[-1]["worker_pid"], 0)

    def test_worker_preserves_local_solution_and_records_frozen_input(self):
        kwargs = self.simple_request()
        reports = []
        direct = tools._move_tracked_plan_frozen(**kwargs)
        with tempfile.TemporaryDirectory() as root:
            # No overlap is needed for this process/protocol test.
            with mock.patch("tempfile.tempdir", root):
                result = run_plan_worker(kwargs, hard_deadline_monotonic=time.monotonic()+30,
                                         progress_callback=reports.append)
            self.assertTrue(result["ok"], result)
            np.testing.assert_allclose(result["endpoint"]["q_final"], direct["endpoint"]["q_final"], atol=1e-10)
            saved = json.loads(Path(reports[-1]["request_path"]).read_text())
            self.assertEqual(saved["robot_profile"], ROBOT_PROFILE)
            self.assertNotIn("world", saved["kwargs"])
            self.assertEqual(float(result["endpoint"]["q_final"][-1]), 0.0)
        self.assert_reaped(reports)

    def test_cancel_before_launch_does_not_start_process(self):
        event = threading.Event()
        event.set()
        reports = []
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            run_plan_worker(self.simple_request(), cancel_event=event,
                            hard_deadline_monotonic=time.monotonic()+10, progress_callback=reports.append)
        self.assertEqual(reports, [])

    def test_cancellation_reaps_the_started_worker(self):
        event = threading.Event()
        reports = []
        def report(value):
            reports.append(value)
            event.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            run_plan_worker(self.simple_request(), cancel_event=event,
                            hard_deadline_monotonic=time.monotonic()+10, progress_callback=report)
        self.assert_reaped(reports)

    def test_startup_timeout_is_bounded_and_reaps_worker(self):
        reports = []
        started = time.monotonic()
        with self.assertRaises(PlanningDeadlineExceeded):
            run_plan_worker(self.simple_request(), hard_deadline_monotonic=time.monotonic()+0.05,
                            progress_callback=reports.append)
        self.assertLess(time.monotonic()-started, 3.0)
        self.assert_reaped(reports)

    def test_native_stage_timeout_and_cancellation_reap_the_worker(self):
        real_popen = subprocess.Popen
        script = (
            "import faulthandler, json, signal, time\n"
            "faulthandler.register(signal.SIGUSR1)\n"
            "print(json.dumps({'kind':'progress','report':{'stage':'overlap.original_overlap'}}), flush=True)\n"
            "time.sleep(60)\n"
        )
        for mode in ("timeout", "cancel"):
            with self.subTest(mode=mode):
                reports = []
                event = threading.Event()
                def report(value):
                    reports.append(value)
                    if mode == "cancel" and value["stage"] == "overlap.original_overlap":
                        event.set()
                def launch(argv, **options):
                    return real_popen([sys.executable, "-u", "-c", script], **options)
                with tempfile.TemporaryDirectory() as root, mock.patch.object(tempfile, "tempdir", root), mock.patch(
                    "behavior_interface_eval_test.tool.official_v2.tracked_point_planning_worker.subprocess.Popen",
                    side_effect=launch,
                ):
                    started = time.monotonic()
                    with self.assertRaisesRegex(RuntimeError, "timed out|cancelled"):
                        run_plan_worker(self.simple_request(), cancel_event=event,
                                        hard_deadline_monotonic=time.monotonic()+0.7,
                                        progress_callback=report)
                    self.assertLess(time.monotonic()-started, 3.0)
                    self.assertEqual(reports[-1]["stage"], "overlap.original_overlap")
                    self.assert_reaped(reports)

    def test_recorded_tv_door_input_has_precise_ordered_endpoint(self):
        _fixture, kwargs = tv_door_fixture()
        endpoint = plan_endpoint(
            state=kwargs["state"], arm="right", q_start=kwargs["q_start"],
            source_points_robot_base_m=kwargs["source_points"], target_points=kwargs["target_points"],
            fixed_points_robot_base_m=kwargs["off_hand_points"], fixed_target_points=kwargs["off_hand_target_points"],
            relations=kwargs["relations"], pos_tol_m=.03, ori_tol_deg=20,
            planning_deadline_monotonic=time.monotonic()+45,
        )
        report = endpoint["constraints"]
        self.assertTrue(endpoint["selected_is_precise"], report)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["relations"][0]["observed_axial_order"], ["1", "4", "2"])
        self.assertLessEqual(report["max_collinear_error_m"], .001)
        points = report["resolved_points_xyz_m"]
        self.assertLessEqual(abs(points["3"][2]-points["4"][2]), .003)


if __name__ == "__main__":
    unittest.main()
