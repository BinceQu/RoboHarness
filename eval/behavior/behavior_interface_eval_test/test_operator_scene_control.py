from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from behavior_interface_eval_test.operator_scene_control import (
    LISTENER_NAME,
    install_operator_reset_on_evaluator_class,
    listener_ready,
    pending_finish,
    pending_operator_work,
    pending_reset,
    read_status,
    write_finish_request,
    write_reset_request,
    write_status,
)


class FakeEvaluator:
    def __init__(self) -> None:
        self.resets = 0
        self.loaded: list[int] = []
        self.steps = 0

    def reset(self) -> None:
        self.resets += 1

    def load_task_instance(self, instance_id: int) -> None:
        self.loaded.append(int(instance_id))

    def step(self):
        self.steps += 1
        return False, False


class OperatorSceneControlTest(unittest.TestCase):
    def test_step_applies_reset_and_instance_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_EVAL_OPERATOR_DIR": tmp},
                clear=False,
            ):
                class LocalEvaluator(FakeEvaluator):
                    pass

                self.assertTrue(install_operator_reset_on_evaluator_class(LocalEvaluator, 15061))
                self.assertFalse(install_operator_reset_on_evaluator_class(LocalEvaluator, 15061))
                self.assertTrue(listener_ready(15061))

                ev = LocalEvaluator()
                ev.load_task_instance(301)
                self.assertEqual(ev.loaded, [301])
                self.assertEqual(read_status(15061)["current_instance_id"], 301)

                write_reset_request(15061, None)
                ev.step()
                self.assertEqual(ev.resets, 1)
                self.assertEqual(ev.loaded, [301])
                self.assertEqual(read_status(15061)["state"], "ready")

                write_reset_request(15061, 304)
                ev.step()
                self.assertEqual(ev.loaded, [301, 304])
                self.assertEqual(ev.resets, 3)
                self.assertEqual(ev._operator_current_instance_id, 304)
                self.assertEqual(read_status(15061)["current_instance_id"], 304)

                write_reset_request(15061, 304)
                ev.step()
                self.assertEqual(ev.loaded, [301, 304])
                self.assertEqual(ev.resets, 4)

    def test_step_submits_finish_as_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_EVAL_OPERATOR_DIR": tmp},
                clear=False,
            ):
                class LocalEvaluator(FakeEvaluator):
                    pass

                self.assertTrue(install_operator_reset_on_evaluator_class(LocalEvaluator, 15063))
                ev = LocalEvaluator()
                ev.load_task_instance(304)
                write_finish_request(15063, reason="model_done:completed")
                self.assertTrue(pending_finish(15063))
                self.assertFalse(pending_reset(15063))
                self.assertTrue(pending_operator_work(15063))
                terminated, truncated = ev.step()
                self.assertEqual((terminated, truncated), (False, True))
                # Finish must not issue one extra action or block on a dead
                # policy connection before writing the result.
                self.assertEqual(ev.steps, 0)
                self.assertEqual(ev.resets, 0)
                self.assertEqual(read_status(15063)["state"], "submitted")
                self.assertFalse(pending_finish(15063))
                self.assertFalse(pending_operator_work(15063))

    def test_pending_reset_until_applied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_EVAL_OPERATOR_DIR": tmp},
                clear=False,
            ):
                self.assertFalse(pending_reset(15062))
                write_reset_request(15062, 304)
                self.assertTrue(pending_reset(15062))
                write_status(15062, applied_request_id=read_status(15062)["last_request_id"])
                self.assertFalse(pending_reset(15062))

    def test_listener_ready_requires_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {"BEHAVIOR_EVAL_OPERATOR_DIR": tmp},
                clear=False,
            ):
                self.assertFalse(listener_ready(15062))
                write_status(15062, listener=LISTENER_NAME, state="idle")
                self.assertTrue(listener_ready(15062))


if __name__ == "__main__":
    unittest.main()
