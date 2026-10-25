"""V2 operator lifecycle compatibility; never used by the robot policy.

Restore the official scene snapshot only at an instance-loading boundary.
Do not change an active rollout, TRO states, metrics, physics or tick budgets.
"""
from functools import wraps
import importlib


def task_object_names(accessor):
    """Remember physical-object identity before transition rules remove it."""
    names = {}
    for key, entity in accessor.object_scope.items():
        name = getattr(entity, 'name', None)
        if isinstance(name, str) and accessor.scene.object_registry('name', name) is entity:
            names[key] = name
    return names


def restore_removed_task_objects(accessor, names):
    """v3.9.3 applies TRO poses before its usual hard scene reset.

    Cutting can delete a task object and set its scope entry to None. Restore
    the previous official initial snapshot before applying the NEXT TRO, then
    rebind by saved object name: category-only callbacks cannot distinguish two
    logs of the same category. Leave intact scenes on the unmodified fast path.
    """
    missing = [key for key, name in names.items()
               if accessor.scene.object_registry('name', name) is None]
    if not missing:
        return []
    accessor.scene.reset(hard=True)
    restored = {key: accessor.scene.object_registry('name', name) for key, name in names.items()}
    absent = [key for key, entity in restored.items() if entity is None]
    if absent:
        raise RuntimeError('Official scene reset did not restore task objects: ' + ', '.join(absent))
    accessor.object_scope.update(restored)
    return missing


def guard_property_scroll(window_class):
    """Kit headless property windows have no scroll frame to save."""
    original = window_class.save_scroll_pos
    if getattr(original, '_v2_missing_frame_guard', False):
        return

    @wraps(original)
    def save_scroll_pos(self, reset=False):
        if not reset and self.properties_frame is None:
            return None
        return original(self, reset=reset)

    save_scroll_pos._v2_missing_frame_guard = True
    window_class.save_scroll_pos = save_scroll_pos


def configure_headless_property_window():
    try:
        window = importlib.import_module('omni.kit.window.property.window')
    except ImportError:
        return False  # The optional GUI extension is not loaded in this build.
    guard_property_scroll(window.PropertyWindow)
    return True
