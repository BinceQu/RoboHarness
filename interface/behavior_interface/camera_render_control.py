"""Render-product controls for robot-mounted interface cameras."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict


_ROBOT_CAMERA_MARKERS = (
    "zed_link",
    "zed",
    "head",
    "left_realsense",
    "left_wrist",
    "right_realsense",
    "right_wrist",
)
_STATE_ATTR = "_codex_robot_camera_render_updates_enabled"


def _robot_camera_items(world):
    robot = getattr(world, "robot", None)
    sensors = getattr(robot, "sensors", {}) if robot is not None else {}
    try:
        items = list(sensors.items())
    except Exception:
        items = []
    return [
        (sensor_name, sensor)
        for sensor_name, sensor in items
        if any(marker in str(sensor_name).lower() for marker in _ROBOT_CAMERA_MARKERS)
    ]


def _camera_io_guard(world):
    lock = getattr(world, "_codex_camera_io_lock", None)
    return lock if lock is not None else nullcontext()


def set_robot_camera_render_updates(
    world,
    enabled: bool,
    *,
    allow_pause: bool = False,
) -> Dict[str, Any]:
    """Keep robot camera Hydra updates enabled.

    Pausing and resuming an OmniGibson VisionSensor render product can invalidate
    Replicator render vars. Pause requests are therefore always ignored.
    """
    requested = bool(enabled)
    pause_ignored = not requested
    enabled = True
    previous = getattr(world, _STATE_ATTR, None)
    if previous is enabled:
        return {
            "ok": True,
            "changed": False,
            "previous": previous,
            "enabled": enabled,
            "requested": requested,
            "pause_ignored": pause_ignored,
            "sensors": [],
            "errors": [],
        }

    touched = []
    errors = []
    with _camera_io_guard(world):
        for sensor_name, sensor in _robot_camera_items(world):
            try:
                render_product = getattr(sensor, "render_product", None)
                if render_product is None:
                    render_product = getattr(sensor, "_render_product", None)
                hydra_texture = getattr(render_product, "hydra_texture", None)
                setter = getattr(hydra_texture, "set_updates_enabled", None)
                if not callable(setter):
                    continue
                setter(enabled)
                touched.append(str(sensor_name))
            except Exception as exc:
                errors.append(f"{sensor_name}: {type(exc).__name__}: {exc}")

    ok = not errors
    if ok:
        setattr(world, _STATE_ATTR, enabled)
    return {
        "ok": ok,
        "changed": bool(touched) and previous is not enabled,
        "previous": previous,
        "enabled": enabled,
        "requested": requested,
        "pause_ignored": pause_ignored,
        "sensors": touched,
        "errors": errors,
    }


def set_sensor_render_updates(sensor, enabled: bool) -> Dict[str, Any]:
    """Enable/disable a single VisionSensor 的 Hydra render product 更新。

    仅用于 external GTA 视角这类 rgb-only 的辅助相机：关闭后 og.sim.render() 不再
    为它跑 RTX，从而在主视图无人观看时省下最贵的一路渲染。机器人 head/wrist 相机
    绝不走这里（见 set_robot_camera_render_updates 的注释：暂停机器人相机 render
    product 会让 Replicator render vars 失效）。恢复时调用方需再 render 几帧预热。
    """
    requested = bool(enabled)
    touched = False
    error = ""
    try:
        render_product = getattr(sensor, "render_product", None)
        if render_product is None:
            render_product = getattr(sensor, "_render_product", None)
        hydra_texture = getattr(render_product, "hydra_texture", None)
        setter = getattr(hydra_texture, "set_updates_enabled", None)
        if callable(setter):
            setter(requested)
            touched = True
    except Exception as exc:  # noqa: BLE001 - 渲染开关失败不应打断主循环
        error = f"{type(exc).__name__}: {exc}"
    return {
        "ok": not error,
        "changed": touched,
        "enabled": requested,
        "error": error,
    }


def rebuild_robot_camera_render_products(world, *, sensor=None) -> Dict[str, Any]:
    """Recreate robot camera render products and reattach their annotators.

    When ``sensor`` is provided, rebuild only that sensor. A failed auxiliary
    camera must not invalidate a healthy head camera render product.
    """
    try:
        import omnigibson.lazy as lazy
        from omnigibson.sensors.vision_sensor import render
    except Exception as exc:
        return {
            "ok": False,
            "sensors": [],
            "errors": [f"backend import failed: {type(exc).__name__}: {exc}"],
        }

    rebuilt = []
    errors = []
    items = _robot_camera_items(world)
    if sensor is not None:
        matched = [(name, item) for name, item in items if item is sensor]
        if matched:
            items = matched
        else:
            sensor_name = (
                getattr(sensor, "name", None)
                or getattr(sensor, "prim_path", None)
                or type(sensor).__name__
            )
            items = [(str(sensor_name), sensor)]
    with _camera_io_guard(world):
        for sensor_name, camera_sensor in items:
            render_product = getattr(camera_sensor, "_render_product", None)
            annotators = getattr(camera_sensor, "_annotators", None)
            prim_path = getattr(camera_sensor, "prim_path", None)
            if render_product is None or not isinstance(annotators, dict) or not prim_path:
                errors.append(f"{sensor_name}: sensor backend is not initialized")
                continue
            try:
                width = int(getattr(camera_sensor, "image_width"))
                height = int(getattr(camera_sensor, "image_height"))
                old_path = getattr(render_product, "path", None)
                for annotator in annotators.values():
                    if annotator is None:
                        continue
                    for target in ([old_path] if old_path else None, render_product):
                        if target is None:
                            continue
                        try:
                            annotator.detach(target)
                            break
                        except Exception:
                            pass
                try:
                    render_product.destroy()
                except Exception:
                    pass
                new_product = lazy.omni.replicator.core.create.render_product(
                    prim_path,
                    (width, height),
                    force_new=True,
                )
                camera_sensor._render_product = new_product
                raw_sensor_types = getattr(camera_sensor, "_RAW_SENSOR_TYPES", {})
                for modality, old_annotator in list(annotators.items()):
                    if old_annotator is None:
                        continue
                    annotator_type = raw_sensor_types.get(modality)
                    if annotator_type is None:
                        raise KeyError(f"missing backend annotator type for {modality}")
                    annotator = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(
                        annotator_type
                    )
                    annotator.attach([new_product])
                    annotators[modality] = annotator
                rebuilt.append(str(sensor_name))
            except Exception as exc:
                errors.append(f"{sensor_name}: {type(exc).__name__}: {exc}")

    if rebuilt:
        setattr(world, _STATE_ATTR, True)
        try:
            with _camera_io_guard(world):
                for _ in range(6):
                    render()
        except Exception as exc:
            errors.append(f"render warmup: {type(exc).__name__}: {exc}")
    return {
        "ok": bool(rebuilt) and not errors,
        "sensors": rebuilt,
        "errors": errors,
    }
