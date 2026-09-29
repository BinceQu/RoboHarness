import unittest
import numpy as np
from behavior_interface_eval_test.episode_initialization import EpisodeGraspPrepController

class World:
    def __init__(self):
        self.q = {s: np.zeros(8) for s in ('left', 'right')}
        self.pins = {s: q.copy() for s, q in self.q.items()}
    def arm_qpos_list(self, side): return self.q[side].tolist()
    def set_arm_pin_qpos(self, side, q): self.pins[side] = np.asarray(q).copy()
    def hold_action(self): return np.concatenate([self.pins[s] for s in ('left', 'right')])
    def make_action(self, **commands):
        for side in self.q: self.pins[side] = np.asarray(commands['arm_' + side]).copy()
        return self.hold_action()

class InitializationTest(unittest.TestCase):
    def test_blocked_arm_stops_forcing_target_and_reports_observed_hold(self):
        world = World()
        controller = EpisodeGraspPrepController(arm_dof=8, timeout_s=2, max_step_rad=1)
        for _ in range(61): controller.step(world)
        self.assertFalse(controller.ready())
        self.assertEqual(controller.status()['state'], 'holding_after_timeout')
        for _ in range(3): np.testing.assert_array_equal(controller.step(world), np.zeros(16))
        status = controller.status()
        self.assertTrue(status['ready'])
        self.assertTrue(status['timed_out'])
        self.assertFalse(status['grasp_prep_reached'])
        self.assertIn('not reached', status['warning'])
        self.assertEqual(status['action_steps'], 60)
        controller.reset()
        self.assertFalse(controller.ready())
        self.assertIsNone(controller.status()['hold_target_qpos'])

    def test_moving_or_invalid_pose_cannot_be_reported_ready(self):
        world = World()
        controller = EpisodeGraspPrepController(arm_dof=8, timeout_s=0.1)
        for _ in range(4): controller.step(world)
        world.q['left'][3] = 0.2
        for _ in range(10): controller.step(world)
        self.assertFalse(controller.ready())
        world.q['left'][0] = np.nan
        with self.assertRaises(ValueError): controller.step(world)
        self.assertFalse(controller.ready())

    def test_normal_convergence_is_unchanged(self):
        world = World()
        controller = EpisodeGraspPrepController(arm_dof=8, timeout_s=5)
        for _ in range(100):
            action = controller.step(world)
            world.q = {'left': action[:8].copy(), 'right': action[8:].copy()}
            if controller.ready(): break
        self.assertTrue(controller.ready())
        self.assertTrue(controller.status()['grasp_prep_reached'])
        self.assertFalse(controller.status()['timed_out'])

    def test_stalled_target_is_released_before_hard_timeout(self):
        world = World()
        controller = EpisodeGraspPrepController(arm_dof=8, timeout_s=80)
        for _ in range(200):
            controller.step(world)
            if controller.ready(): break
        self.assertTrue(controller.ready())
        self.assertFalse(controller.status()['grasp_prep_reached'])
        self.assertFalse(controller.status()['timed_out'])
        self.assertEqual(controller.status()['hold_reason'], 'no_progress_for_5_control_seconds')

if __name__ == '__main__': unittest.main()
