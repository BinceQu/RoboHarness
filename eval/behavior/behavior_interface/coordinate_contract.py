"""Public 2D image-coordinate contract.

All public skill inputs and outputs use Qwen3-VL relative coordinates:
top-left=(0, 0), bottom-right=(1000, 1000). Skill implementations continue to
use native image pixels; the registry wrapper converts at the boundary.
"""

from __future__ import annotations

import inspect
import math
import os
from typing import Any, Callable, Dict, Mapping, Optional, Tuple


COORDINATE_MAX = 1000
COORDINATE_SYSTEM = "qwen3vl_relative_0_1000"

_COORDINATE_SEQUENCE_KEYS = {
    "uv",
    "input_uv",
    "input_uv_px",
    "pick_uv",
    "point_uv",
    "projected_uv",
    "projected_uv_clipped",
}
_COORDINATE_MAPPING_KEYS = {
    "pixel",
    "vlm_pixel",
    "reproj_pixel",
    "contact_pixel",
    "point_px",
    "grasp_point_px",
    "corners_uv",
}
_COORDINATE_BOX_KEYS = {
    "uv_bbox",
}
_U_COORDINATE_KEYS = {
    "center_u_px",
}
_V_COORDINATE_KEYS = {
    "center_v_px",
}
_PUBLIC_KEY_ALIASES = {
    "pixel": "resolved_uv",
    "vlm_pixel": "vlm_uv",
    "reproj_pixel": "reproj_uv",
    "contact_pixel": "contact_uv",
    "point_px": "point_uv",
    "grasp_point_px": "grasp_point_uv",
    "input_uv_px": "input_uv",
    "center_u_px": "center_u",
    "center_v_px": "center_v",
}
_ALREADY_RELATIVE_KEYS = {
    "qwen_uv",
    "qwen_u",
    "qwen_v",
    "qwen_coord_system",
}


class CoordinateContractError(ValueError):
    pass


def relative_to_pixel(value: Any, image_size: int) -> int:
    """Map one public 0..1000 coordinate to a native pixel index."""
    try:
        coord = float(value)
    except (TypeError, ValueError) as exc:
        raise CoordinateContractError(f"coordinate must be numeric, got {value!r}") from exc
    if not math.isfinite(coord):
        raise CoordinateContractError(f"coordinate must be finite, got {value!r}")
    if coord < 0.0 or coord > float(COORDINATE_MAX):
        raise CoordinateContractError(
            f"coordinate {coord:g} is outside public range 0..{COORDINATE_MAX}"
        )
    size = int(image_size)
    if size <= 1:
        return 0
    return int(round(coord / float(COORDINATE_MAX) * float(size - 1)))


def pixel_to_relative(value: Any, image_size: int) -> int:
    """Map one native pixel coordinate to public 0..1000 coordinates."""
    try:
        pixel = float(value)
    except (TypeError, ValueError) as exc:
        raise CoordinateContractError(f"pixel coordinate must be numeric, got {value!r}") from exc
    if not math.isfinite(pixel):
        raise CoordinateContractError(f"pixel coordinate must be finite, got {value!r}")
    size = int(image_size)
    if size <= 1:
        return 0
    coord = int(round(pixel / float(size - 1) * float(COORDINATE_MAX)))
    return max(0, min(COORDINATE_MAX, coord))


def relative_pair_to_pixels(u: Any, v: Any, width: int, height: int) -> Tuple[int, int]:
    return relative_to_pixel(u, width), relative_to_pixel(v, height)


def pixel_pair_to_relative(u: Any, v: Any, width: int, height: int) -> Tuple[int, int]:
    return pixel_to_relative(u, width), pixel_to_relative(v, height)


def _positive_dimensions(width: Any, height: Any) -> Optional[Tuple[int, int]]:
    try:
        w = int(width)
        h = int(height)
    except (TypeError, ValueError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def _dimensions_from_mapping(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, Mapping):
        return None
    direct = _positive_dimensions(value.get("image_width"), value.get("image_height"))
    if direct is not None:
        return direct
    for key in ("camera", "head_camera", "render_meta", "observation", "memory"):
        nested = _dimensions_from_mapping(value.get(key))
        if nested is not None:
            return nested
    return None


def _dimensions_from_agent_capture(arguments: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    session_id = str(arguments.get("session_id") or "").strip()
    image_id = str(arguments.get("image_id") or "").strip()
    if not session_id:
        return None
    try:
        from behavior_interface import agent_runs

        if image_id:
            meta = agent_runs.load_image_meta(session_id, image_id)
            dims = _dimensions_from_mapping(meta)
            if dims is not None:
                return dims
        image_dir = agent_runs.images_dir(session_id)
        if os.path.isdir(image_dir):
            names = sorted(
                name[:-10]
                for name in os.listdir(image_dir)
                if name.startswith("img_") and name.endswith(".meta.json")
            )
            if names:
                meta = agent_runs.load_image_meta(session_id, names[-1])
                dims = _dimensions_from_mapping(meta)
                if dims is not None:
                    return dims
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        pass
    return None


def _dimensions_from_plan_session(arguments: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    session_id = str(arguments.get("session_id") or "").strip()
    image_id = str(arguments.get("image_id") or "").strip()
    try:
        if image_id:
            from behavior_interface.skills.plan_eef_core import lookup_image

            entry = lookup_image(image_id)
            session_id = str(entry.get("session_id") or session_id).strip()
        if session_id:
            from behavior_interface.skills.plan_grasp_core import load_session

            session = load_session(session_id)
            dims = _dimensions_from_mapping(session)
            if dims is not None:
                return dims
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
        pass
    return None


def _dimensions_from_world(ctx: Any) -> Optional[Tuple[int, int]]:
    world = getattr(ctx, "world", None)
    if world is None:
        return None
    try:
        from behavior_interface.head_capture import get_head_sensor, head_camera_size

        sensor = get_head_sensor(world)
        if sensor is not None:
            return _positive_dimensions(*head_camera_size(sensor))
    except Exception:
        pass
    try:
        from behavior_interface.head_capture import head_intrinsics_fallback

        _fl, _ha, width, height = head_intrinsics_fallback()
        return _positive_dimensions(width, height)
    except Exception:
        return None


def resolve_image_dimensions(
    *,
    ctx: Any = None,
    arguments: Optional[Mapping[str, Any]] = None,
    payload: Optional[Mapping[str, Any]] = None,
) -> Optional[Tuple[int, int]]:
    """Resolve the native image size associated with a public coordinate."""
    args = arguments or {}
    explicit = args.get("_public_image_dimensions")
    if isinstance(explicit, (list, tuple)) and len(explicit) >= 2:
        dims = _positive_dimensions(explicit[0], explicit[1])
        if dims is not None:
            return dims
    dims = _dimensions_from_mapping(payload)
    if dims is not None:
        return dims
    dims = _dimensions_from_agent_capture(args)
    if dims is not None:
        return dims
    dims = _dimensions_from_plan_session(args)
    if dims is not None:
        return dims
    return _dimensions_from_world(ctx)


def _convert_points_argument(points: Any, width: int, height: int) -> Any:
    if not isinstance(points, (list, tuple)):
        raise CoordinateContractError("points must be a list of [u, v] coordinates")
    converted = []
    for point in points:
        if isinstance(point, Mapping) and point.get("u") is not None and point.get("v") is not None:
            u_px, v_px = relative_pair_to_pixels(point["u"], point["v"], width, height)
            item = dict(point)
            item["u"] = u_px
            item["v"] = v_px
            converted.append(item)
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            u_px, v_px = relative_pair_to_pixels(point[0], point[1], width, height)
            converted.append([u_px, v_px, *list(point[2:])])
        else:
            raise CoordinateContractError(f"invalid point coordinate {point!r}")
    return converted


def convert_skill_arguments(
    fn: Callable,
    ctx: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[tuple[Any, ...], Dict[str, Any], Dict[str, Any]]:
    """Convert public relative coordinates in a skill call to native pixels."""
    signature = inspect.signature(fn)
    bound = signature.bind_partial(ctx, *args, **dict(kwargs))
    supplied = set(bound.arguments)
    u_supplied = "u" in supplied and bound.arguments.get("u") is not None
    v_supplied = "v" in supplied and bound.arguments.get("v") is not None
    if u_supplied != v_supplied:
        raise CoordinateContractError("public image coordinates require both u and v")
    coordinate_supplied = u_supplied and v_supplied
    if not u_supplied and not v_supplied and "u" in signature.parameters and "v" in signature.parameters:
        u_default = signature.parameters["u"].default
        v_default = signature.parameters["v"].default
        if (
            u_default is not inspect.Parameter.empty
            and v_default is not inspect.Parameter.empty
            and u_default is not None
            and v_default is not None
        ):
            bound.arguments["u"] = u_default
            bound.arguments["v"] = v_default
            coordinate_supplied = True
    points_supplied = "points" in supplied and bound.arguments.get("points") is not None
    if (
        (coordinate_supplied or points_supplied)
        and "image_id" in signature.parameters
        and not str(bound.arguments.get("image_id") or "").strip()
    ):
        session_id = str(bound.arguments.get("session_id") or "").strip()
        if session_id:
            try:
                from behavior_interface import agent_runs

                latest_image_id = agent_runs.latest_capture_image_id(
                    session_id,
                    require_camera=True,
                    require_depth=True,
                )
                if latest_image_id:
                    bound.arguments["image_id"] = latest_image_id
            except (FileNotFoundError, OSError, TypeError, ValueError):
                pass
    ctx_name = next(iter(signature.parameters))
    public_arguments = {
        key: value for key, value in bound.arguments.items() if key != ctx_name
    }
    if not coordinate_supplied and not points_supplied:
        return bound.args, dict(bound.kwargs), public_arguments

    dimensions = resolve_image_dimensions(ctx=ctx, arguments=public_arguments)
    if dimensions is None:
        raise CoordinateContractError(
            "cannot resolve image dimensions for public 0..1000 coordinates"
        )
    width, height = dimensions
    if coordinate_supplied:
        bound.arguments["u"], bound.arguments["v"] = relative_pair_to_pixels(
            bound.arguments["u"], bound.arguments["v"], width, height
        )
    if points_supplied:
        bound.arguments["points"] = _convert_points_argument(
            bound.arguments["points"], width, height
        )
    internal_arguments = {
        key: value for key, value in bound.arguments.items() if key != ctx_name
    }
    internal_arguments["_public_image_dimensions"] = dimensions
    return bound.args, dict(bound.kwargs), internal_arguments


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    """像素坐标是否可用于换算。

    投影退化（物体落在相机平面上或后方、深度为 0）会算出 nan/inf。这类值语义上
    是「不可见/无解」，应当输出 None，而不是让整份 skill 结果的坐标换算抛异常，
    把 capture 这种只是顺带带了投影字段的技能整体拖挂。
    """
    return _is_number(value) and math.isfinite(float(value))


def _contains_pixel_coordinate(value: Any, parent_key: str = "") -> bool:
    if isinstance(value, (list, tuple)):
        if parent_key in (_COORDINATE_SEQUENCE_KEYS | _COORDINATE_MAPPING_KEYS):
            if len(value) >= 2 and _is_number(value[0]) and _is_number(value[1]):
                return True
        if parent_key in _COORDINATE_BOX_KEYS:
            if len(value) >= 4 and all(_is_number(item) for item in value[:4]):
                return True
        return any(_contains_pixel_coordinate(item, parent_key) for item in value)
    if isinstance(value, Mapping):
        source = dict(value)
        already_relative = str(source.get("coord_system") or "").strip().lower() == COORDINATE_SYSTEM
        if already_relative:
            return False
        if (
            _is_number(source.get("u"))
            and _is_number(source.get("v"))
            and parent_key not in {"uv_range", "pixel_uv_range", "qwen_uv_range"}
        ):
            return True
        for key, item in source.items():
            if key in _ALREADY_RELATIVE_KEYS:
                continue
            if key in (_U_COORDINATE_KEYS | _V_COORDINATE_KEYS) and _is_number(item):
                return True
            if _contains_pixel_coordinate(item, key):
                return True
        return False
    return False


def _convert_coordinate_sequence(value: Any, width: int, height: int) -> tuple[Any, bool]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if _is_number(value[0]) and _is_number(value[1]):
            if not (_is_finite_number(value[0]) and _is_finite_number(value[1])):
                return [None, None, *list(value[2:])], True
            u, v = pixel_pair_to_relative(value[0], value[1], width, height)
            return [u, v, *list(value[2:])], True
    return value, False


def _convert_coordinate_box(value: Any, width: int, height: int) -> tuple[Any, bool]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        if all(_is_number(item) for item in value[:4]):
            if not all(_is_finite_number(item) for item in value[:4]):
                return [None, None, None, None, *list(value[4:])], True
            u0, v0 = pixel_pair_to_relative(value[0], value[1], width, height)
            u1, v1 = pixel_pair_to_relative(value[2], value[3], width, height)
            return [u0, v0, u1, v1, *list(value[4:])], True
    return value, False


def _publicize_value(
    value: Any,
    *,
    width: int,
    height: int,
    parent_key: str = "",
) -> tuple[Any, bool]:
    if isinstance(value, Mapping):
        source = dict(value)
        out: Dict[str, Any] = {}
        changed = False
        already_relative = str(source.get("coord_system") or "").strip().lower() == COORDINATE_SYSTEM
        uv_candidate = (
            not already_relative
            and _is_number(source.get("u"))
            and _is_number(source.get("v"))
            and parent_key not in {"uv_range", "pixel_uv_range", "qwen_uv_range"}
        )
        # 投影退化出的 nan/inf 视为不可见，置 None
        uv_degenerate = uv_candidate and not (
            _is_finite_number(source.get("u")) and _is_finite_number(source.get("v"))
        )
        direct_uv = uv_candidate and not uv_degenerate
        relative_u = relative_v = None
        if direct_uv:
            relative_u, relative_v = pixel_pair_to_relative(
                source["u"], source["v"], width, height
            )
            changed = True
        elif uv_degenerate:
            changed = True
        for key, item in source.items():
            out_key = key
            if key in _ALREADY_RELATIVE_KEYS:
                out[key] = item
                continue
            if uv_candidate and key == "u":
                out[key] = relative_u
                continue
            if uv_candidate and key == "v":
                out[key] = relative_v
                continue
            if key in _COORDINATE_SEQUENCE_KEYS and not already_relative:
                converted, item_changed = _convert_coordinate_sequence(item, width, height)
                if item_changed:
                    out_key = _PUBLIC_KEY_ALIASES.get(key, key)
                out[out_key] = converted
                changed = changed or item_changed
                continue
            if key in _COORDINATE_BOX_KEYS and not already_relative:
                converted, item_changed = _convert_coordinate_box(item, width, height)
                out[key] = converted
                changed = changed or item_changed
                continue
            if key in _U_COORDINATE_KEYS and _is_number(item) and not already_relative:
                out[_PUBLIC_KEY_ALIASES.get(key, key)] = (
                    pixel_to_relative(item, width)
                    if _is_finite_number(item) else None
                )
                changed = True
                continue
            if key in _V_COORDINATE_KEYS and _is_number(item) and not already_relative:
                out[_PUBLIC_KEY_ALIASES.get(key, key)] = (
                    pixel_to_relative(item, height)
                    if _is_finite_number(item) else None
                )
                changed = True
                continue
            converted, item_changed = _publicize_value(
                item,
                width=width,
                height=height,
                parent_key=key,
            )
            if item_changed:
                out_key = _PUBLIC_KEY_ALIASES.get(key, key)
            out[out_key] = converted
            changed = changed or item_changed
        if changed and "coord_system" in out:
            out["coord_system"] = COORDINATE_SYSTEM
        return out, changed
    if parent_key in _COORDINATE_MAPPING_KEYS:
        converted, changed = _convert_coordinate_sequence(value, width, height)
        if changed:
            return converted, True
    if isinstance(value, list):
        out_list = []
        changed = False
        for item in value:
            converted, item_changed = _publicize_value(
                item,
                width=width,
                height=height,
                parent_key=parent_key,
            )
            out_list.append(converted)
            changed = changed or item_changed
        return out_list, changed
    if isinstance(value, tuple):
        converted, changed = _publicize_value(
            list(value), width=width, height=height, parent_key=parent_key
        )
        return converted, changed
    return value, False


def publicize_result(
    payload: Mapping[str, Any],
    *,
    ctx: Any = None,
    arguments: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert pixel-valued fields in a public skill result to 0..1000."""
    raw = dict(payload or {})
    dimensions = resolve_image_dimensions(ctx=ctx, arguments=arguments, payload=raw)
    if dimensions is None:
        if _contains_pixel_coordinate(raw):
            return {
                "ok": False,
                "error": "cannot resolve image dimensions for public coordinate result",
                "coordinate_system": COORDINATE_SYSTEM,
                "coordinate_range": {
                    "u": [0, COORDINATE_MAX],
                    "v": [0, COORDINATE_MAX],
                },
            }
        return raw
    width, height = dimensions
    out, changed = _publicize_value(raw, width=width, height=height)
    if changed:
        out["coordinate_system"] = COORDINATE_SYSTEM
        out["coordinate_range"] = {
            "u": [0, COORDINATE_MAX],
            "v": [0, COORDINATE_MAX],
        }
    return out


class PublicCoordinateContext:
    """Context proxy that exposes normalized results while retaining raw internals."""

    def __init__(self, parent: Any, arguments: Mapping[str, Any]):
        self._parent = parent
        self.world = parent.world
        self._arguments = dict(arguments)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parent, name)

    def set_result(self, payload: Dict[str, Any]) -> None:
        raw = dict(payload or {})
        set_internal = getattr(self._parent, "set_internal_result", None)
        if callable(set_internal):
            set_internal(raw)
        self._parent.set_result(
            publicize_result(raw, ctx=self._parent, arguments=self._arguments)
        )


def public_skill_wrapper(fn: Callable) -> Callable:
    """Wrap one registry entry with the public coordinate boundary."""
    import functools

    @functools.wraps(fn)
    def wrapped(ctx: Any, *args: Any, **kwargs: Any):
        try:
            call_args, call_kwargs, internal_arguments = convert_skill_arguments(
                fn, ctx, args, kwargs
            )
        except CoordinateContractError as exc:
            ctx.set_result({
                "ok": False,
                "error": str(exc),
                "coordinate_system": COORDINATE_SYSTEM,
                "coordinate_range": {
                    "u": [0, COORDINATE_MAX],
                    "v": [0, COORDINATE_MAX],
                },
            })
            world = getattr(ctx, "world", None)
            hold = getattr(world, "hold_action", None)
            empty = getattr(world, "empty_action", None)
            if callable(hold):
                yield hold()
            elif callable(empty):
                yield empty()
            else:
                yield None
            return
        public_ctx = PublicCoordinateContext(ctx, internal_arguments)
        yield from fn(public_ctx, *call_args[1:], **call_kwargs)

    wrapped._uses_public_coordinate_contract = True
    return wrapped
