"""The model workspace keeps a minimal call record; bulk evidence lands in <case>/trajectory."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.util import history as history_module, perception  # noqa: E402


class FakeAdapter:
    def __init__(self):
        self.current = SimpleNamespace(sequence=7, received_ts=1.0)
        self.offset = {"head": 0, "left_wrist": 1, "right_wrist": 2}

    def camera_rgb(self, role):
        return np.full((8, 8, 3), 9 + self.offset[role], np.uint8)

    def camera_depth_frame(self, role):
        return np.full((8, 8), 1.2345 + self.offset[role], np.float32)

    def proprio_vector(self):
        return np.zeros(65)

    def camera_relative_poses(self):
        return {}


class SlimHistoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.workspace, self.trajectory = root / "workspace", root / "trajectory"
        for sub in (self.workspace / ".runtime", self.workspace / "tool_call_history", self.trajectory / "calls"):
            sub.mkdir(parents=True)
        images = self.trajectory / "recordings/s1/images"
        images.mkdir(parents=True)
        Image.new("RGB", (720, 720), "gray").save(images / "img_0001.png")
        self.adapter = FakeAdapter()
        self.robot = SimpleNamespace(workspace=self.workspace, state_dir=self.workspace / ".runtime",
            session_id="s1", config={"episode_id": 0, "trajectory_dir": str(self.trajectory)},
            _official_adapter=self.adapter, logs=[], rpc=lambda *a, **k: {})
        perception.response_to_pixels = lambda value: value

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, arguments):
        history = history_module.CallHistory(self.robot, "plan", arguments)
        return history

    def test_click_record_is_minimal_and_bulk_is_outside(self):
        history = self.call({"image_id": "img_0001", "u": 100, "v": 200})
        history.action([0.0, 1.0], 1, 2)
        history.finish({"ok": True}, {"full": True})
        names = sorted(p.name for p in history.path.iterdir())
        self.assertEqual(names, ["click_img_0001_crosshair.png", "input.json", "result.json"])
        self.assertEqual(sorted(p.name for p in history.calls.iterdir()),
                         ["joint_commands.jsonl", "output.json", "selected_points.json"])
        self.assertFalse(history.calls.is_relative_to(self.workspace))

    def test_publish_current_only_real_files_and_deduplicated_archive(self):
        head = {"image_id": "img_0001", "rgb_path": str(self.trajectory / "recordings/s1/images/img_0001.png")}
        first = self.call({})
        perception.publish(self.robot, first, head)
        second = self.call({})
        perception.publish(self.robot, second, head)
        exteroception = self.workspace / "perception/exteroception"
        self.assertTrue((exteroception / "head_rgb.png").is_file())
        self.assertFalse(any(p.is_symlink() for p in (self.workspace / "perception").rglob("*")))
        self.assertFalse(list(self.workspace.rglob("*.npy")) + list(self.workspace.rglob("*.npz")))
        archived = json.loads((second.calls / "observation/manifest.json").read_text())["archive"]
        self.assertTrue(archived["left_wrist_depth"]["duplicate_of_earlier_call"])
        self.assertFalse(list((second.calls / "observation").glob("*.npz")))
        depth = np.load(first.calls / "observation/left_wrist_depth_mm.npz")["depth_mm"]
        self.assertEqual(int(depth[0, 0]), 2234)
        latest = json.loads((self.workspace / "perception/latest.json").read_text())
        self.assertTrue(Path(latest["images"]["head_rgb"]["path"]).is_relative_to(self.workspace))
        state = json.loads((self.workspace / ".runtime/state.json").read_text())
        self.assertEqual(state["latest_head_path"], str(exteroception / "head_rgb.png"))


if __name__ == "__main__":
    unittest.main()
