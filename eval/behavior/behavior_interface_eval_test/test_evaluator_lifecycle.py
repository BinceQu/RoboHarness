"""Official scene lifecycle ordering, using an explicit deleted-object fixture."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavior_interface_eval_test.evaluator_lifecycle import (
    task_object_names, restore_removed_task_objects, guard_property_scroll,
)
from behavior_interface_eval_test import official_evaluator_v393
from behavior_interface_eval_test.official_evaluator_v393 import route_evaluator_class


class LifecycleTests(unittest.TestCase):
    def fixture(self):
        original = {'log1': SimpleNamespace(name='wood_1'), 'log2': SimpleNamespace(name='wood_2')}
        objects = {v.name: v for v in original.values()}
        scope = dict(original, water=SimpleNamespace(name='water'))
        scene = SimpleNamespace(object_registry=lambda field, name: objects.get(name))
        def reset(*, hard):
            self.assertTrue(hard)
            objects.clear()
            objects.update({v.name: SimpleNamespace(name=v.name) for v in original.values()})
            # Simulate ambiguous category-based object-added callbacks.
            scope.update(log1=objects['wood_2'], log2=objects['wood_1'])
        scene.reset = Mock(side_effect=reset)
        accessor = SimpleNamespace(scene=scene, object_scope=scope)
        return accessor, objects

    def test_native_reset_restores_deleted_object_and_exact_scope_identity(self):
        accessor, objects = self.fixture()
        names = task_object_names(accessor)
        self.assertEqual(names, {'log1': 'wood_1', 'log2': 'wood_2'})
        del objects['wood_1']
        accessor.object_scope['log1'] = None
        self.assertEqual(restore_removed_task_objects(accessor, names), ['log1'])
        accessor.scene.reset.assert_called_once_with(hard=True)
        self.assertEqual(accessor.object_scope['log1'].name, 'wood_1')
        self.assertEqual(accessor.object_scope['log2'].name, 'wood_2')
        self.assertEqual(accessor.object_scope['water'].name, 'water')

    def test_intact_scene_never_gets_an_extra_reset(self):
        accessor, _ = self.fixture()
        self.assertEqual(restore_removed_task_objects(accessor, task_object_names(accessor)), [])
        accessor.scene.reset.assert_not_called()

    def test_failed_restore_is_not_reported_as_ready(self):
        accessor, objects = self.fixture()
        names = task_object_names(accessor)
        del objects['wood_1']
        accessor.scene.reset.side_effect = None
        with self.assertRaisesRegex(RuntimeError, 'did not restore.*log1'):
            restore_removed_task_objects(accessor, names)

    def stock(self, accessor, calls, check=lambda: None):
        class Stock:
            def __init__(self, cfg):
                self.instance_eval_states = [SimpleNamespace(env_accessor=accessor)]
            def load_batch(self, mapping, **kwargs):
                check()
                calls.append((mapping, kwargs))
                return 'stock-result'
        return Stock

    def test_route_restores_before_stock_loader_and_preserves_load_arguments(self):
        accessor, objects = self.fixture()
        calls = []
        def restored():
            assert accessor.object_scope['log1'] is objects['wood_1']
        with patch.object(official_evaluator_v393, 'RESTORE_TASK_OBJECTS', True):
            route = route_evaluator_class(self.stock(accessor, calls, restored))({'num_envs': 1, 'headless': False})
            del objects['wood_1']
            accessor.object_scope['log1'] = None
            self.assertEqual(route.load_batch({0: 303}, write_video=True, rollout_id=0), 'stock-result')
        self.assertEqual(calls, [({0: 303}, {'write_video': True, 'rollout_id': 0})])

    def test_default_route_keeps_the_official_loading_path(self):
        # Without the opt-in (V1 runs) the scene is never reset or rebound.
        accessor, objects = self.fixture()
        calls = []
        with patch.object(official_evaluator_v393, 'RESTORE_TASK_OBJECTS', False):
            route = route_evaluator_class(self.stock(accessor, calls))({'num_envs': 1, 'headless': False})
            del objects['wood_1']
            accessor.object_scope['log1'] = None
            self.assertEqual(route.load_batch({0: 303}), 'stock-result')
        accessor.scene.reset.assert_not_called()
        self.assertIsNone(accessor.object_scope['log1'])
        self.assertFalse(hasattr(route, '_route_object_names'))

    def test_headless_guard_preserves_real_frame_behavior_and_is_idempotent(self):
        calls = []
        class Window:
            properties_frame = None
            def save_scroll_pos(self, reset=False):
                calls.append(reset)
                if reset:
                    return 'cleared'
                return self.properties_frame.scroll_y_max
        guard_property_scroll(Window)
        installed = Window.save_scroll_pos
        guard_property_scroll(Window)
        self.assertIs(Window.save_scroll_pos, installed)
        window = Window()
        self.assertIsNone(window.save_scroll_pos())
        self.assertEqual(calls, [])
        self.assertEqual(window.save_scroll_pos(reset=True), 'cleared')
        window.properties_frame = SimpleNamespace(scroll_y_max=42)
        self.assertEqual(window.save_scroll_pos(), 42)
        self.assertEqual(calls, [True, False])


if __name__ == '__main__':
    unittest.main()
