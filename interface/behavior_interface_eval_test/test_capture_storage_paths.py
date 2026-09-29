"""Capture producers and map consumers must agree on policy-owned storage."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from behavior_interface import agent_runs
from behavior_interface.rtabmap_slam.live import (
    _load_capture_bundle,
    _unproject_uv_robot_xy,
)
from behavior_interface_eval_test.tool.official_v2 import tools


class CaptureStoragePathsTest(unittest.TestCase):
    def test_default_root_agrees_with_map_capture_reader(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("BEHAVIOR_AGENT_RUNS", None)
            self.assertEqual(tools._runs_root(), agent_runs.DEFAULT_RUNS_ROOT)

    def test_explicit_root_override_is_preserved(self):
        with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": "/tmp/capture-test"}):
            self.assertEqual(tools._runs_root(), "/tmp/capture-test")

    def test_legacy_session_stays_together_after_default_root_change(self):
        with tempfile.TemporaryDirectory() as legacy, tempfile.TemporaryDirectory() as current:
            session = "old-session"
            old_dir = Path(legacy) / session
            old_dir.mkdir()
            (old_dir / "session.json").write_text('{"image_counter":451,"plan_counter":165}')
            with mock.patch.dict(os.environ), mock.patch.object(tools, "_LEGACY_RUNS_ROOT", legacy), mock.patch.object(
                tools, "_runs_root", return_value=current
            ):
                os.environ.pop("BEHAVIOR_AGENT_RUNS", None)
                self.assertEqual(tools._run_dir(session), str(old_dir))
                self.assertEqual(tools._images_dir(session), str(old_dir / "images"))
                self.assertEqual(tools._plans_dir(session), str(old_dir / "plans"))
                self.assertEqual(tools._load_session_meta(session)["image_counter"], 451)
                self.assertEqual(tools._run_dir("new-session"), str(Path(current) / "new-session"))
                # Even a partly created new location cannot shadow old state.
                (Path(current) / session).mkdir()
                self.assertEqual(tools._run_dir(session), str(old_dir))
                os.environ["BEHAVIOR_AGENT_RUNS"] = current
                self.assertEqual(tools._run_dir(session), str(Path(current) / session))

    def test_session_path_cannot_escape_either_root(self):
        for session in ("../outside", "/tmp/outside", ".", ".."):
            with self.subTest(session=session), self.assertRaises(ValueError):
                tools._run_dir(session)

    def test_official_capture_bundle_is_readable_by_mark_projection(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"BEHAVIOR_AGENT_RUNS": root}):
                with mock.patch.object(agent_runs, "RUNS_ROOT", root):
                    session, image = "capture-path-contract", "img_0007"
                    tools._ensure_session(session)
                    depth_path = Path(tools._image_path(session, image, ".depth.npy"))
                    np.save(depth_path, np.full((12, 12), 2.0, dtype=np.float32))
                    tools._save_image_meta(session, image, {
                        "camera": {
                            "fx": 12.0, "fy": 12.0, "cx": 6.0, "cy": 6.0,
                            "robot_relative_pose": {
                                "pos": [0.2, 0.0, 1.4],
                                "quat": [0.0, 0.0, 0.0, 1.0],
                            },
                        },
                    })
                    bundle = _load_capture_bundle(session, image)
                    self.assertIsNotNone(bundle)
                    self.assertIsNotNone(_unproject_uv_robot_xy(bundle, 536, 626))


if __name__ == "__main__":
    unittest.main()
