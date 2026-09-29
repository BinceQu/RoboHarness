"""Evaluator-owned BDDL progress for the test Interface display.

The evaluator wrapper is the only caller that reads the active task.  It
serializes a small, display-only payload into the observation transport.  The
policy interface removes that envelope before its observation allowlist is
made available to tools or a downstream policy.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional


UI_BDDL_PROGRESS_KEY = "__behavior_interface_ui_bddl_progress__"
_MAX_GOALS = 512
_MAX_LABEL_LENGTH = 512

_PREDICATE_LABELS = {
    "inside": "{a} 在 {b} 内",
    "ontop": "{a} 在 {b} 上",
    "under": "{a} 在 {b} 下",
    "nextto": "{a} 靠近 {b}",
    "touching": "{a} 接触 {b}",
    "covered": "{a} 被 {b} 覆盖",
    "contains": "{a} 含有 {b}",
    "filled": "{a} 装有 {b}",
    "attached": "{a} 连接到 {b}",
    "toggled_on": "{a} 已开启",
    "open": "{a} 已打开",
    "cooked": "{a} 已烹饪",
    "frozen": "{a} 已冷冻",
    "on_fire": "{a} 着火",
    "real": "{a} 已生成",
    "hot": "{a} 已加热",
}


def _bounded_text(value: Any, limit: int = _MAX_LABEL_LENGTH) -> str:
    text = " ".join(str(value).replace("\n", " ").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _term_label(term: Any, variables: Optional[dict[str, str]] = None) -> str:
    value = str(term).strip()
    key = value.lstrip("?")
    if variables and key in variables:
        return variables[key]

    index = ""
    base = key
    if "_" in key:
        maybe_base, maybe_index = key.rsplit("_", 1)
        if maybe_index.isdigit():
            base, index = maybe_base, maybe_index
    base = base.split(".")[0]
    base = base.replace("__of__", " of ").replace("__", " ").replace("_", " ")
    base = " ".join(base.split())
    if base.startswith("electric refrigerator"):
        base = base.replace("electric refrigerator", "refrigerator", 1)
    return f"{base} {index}" if index else base


def render_goal_expression(
    expression: Any,
    variables: Optional[dict[str, str]] = None,
) -> str:
    """Render a parsed BDDL condition in the same compact style as the UI."""
    variables = dict(variables or {})
    if not isinstance(expression, (list, tuple)) or not expression:
        return _term_label(expression, variables)

    token = str(expression[0])
    body = list(expression[1:])
    if token == "and":
        return " + ".join(render_goal_expression(item, variables) for item in body)
    if token == "or":
        return " / ".join(render_goal_expression(item, variables) for item in body)
    if token == "not":
        inner = body[0] if body else []
        if isinstance(inner, (list, tuple)) and inner:
            predicate = str(inner[0])
            args = [_term_label(arg, variables) for arg in inner[1:]]
            if predicate == "open" and args:
                return f"{args[0]} 关闭"
            if predicate == "real" and args:
                return f"{args[0]} 不存在"
            if predicate in {"covered", "contains", "inside", "touching"} and len(args) >= 2:
                negated = {
                    "covered": "未被",
                    "contains": "不含",
                    "inside": "不在",
                    "touching": "不接触",
                }[predicate]
                suffix = " 内" if predicate == "inside" else ""
                return f"{args[0]} {negated} {args[1]}{suffix}"
        return "未满足: " + render_goal_expression(inner, variables)

    if token in {"forall", "exists"} and len(body) >= 2:
        iterable, subexpression = body[0], body[1]
        variable = str(iterable[0]).lstrip("?")
        category = _term_label(iterable[2], variables)
        variables[variable] = category
        prefix = "全部" if token == "forall" else "存在"
        return f"{prefix} {category}: {render_goal_expression(subexpression, variables)}"
    if token == "forn" and len(body) >= 3:
        count, iterable, subexpression = body[0], body[1], body[2]
        count_value = str(count[0] if isinstance(count, (list, tuple)) and count else count)
        variable = str(iterable[0]).lstrip("?")
        category = _term_label(iterable[2], variables)
        variables[variable] = category
        return f"恰好 {count_value} 个 {category}: {render_goal_expression(subexpression, variables)}"
    if token in {"forpairs", "fornpairs"}:
        offset = 1 if token == "fornpairs" else 0
        if len(body) >= 3 + offset:
            iterable_a, iterable_b = body[offset], body[offset + 1]
            subexpression = body[offset + 2]
            category_a = _term_label(iterable_a[2], variables)
            category_b = _term_label(iterable_b[2], variables)
            variables[str(iterable_a[0]).lstrip("?")] = category_a
            variables[str(iterable_b[0]).lstrip("?")] = category_b
            prefix = ""
            if token == "fornpairs":
                count = body[0]
                count_value = str(count[0] if isinstance(count, (list, tuple)) and count else count)
                prefix = f"{count_value} 组 "
            return (
                f"{prefix}{category_a} 与 {category_b} 配对: "
                f"{render_goal_expression(subexpression, variables)}"
            )

    if token in _PREDICATE_LABELS:
        args = [_term_label(arg, variables) for arg in body]
        if len(args) == 1:
            return _PREDICATE_LABELS[token].format(a=args[0], b="")
        if len(args) >= 2:
            return _PREDICATE_LABELS[token].format(a=args[0], b=args[1])
    args = " ".join(_term_label(arg, variables) for arg in body)
    return f"{token} {args}".strip()


def awaiting_bddl_progress(message: str = "waiting for evaluator BDDL status") -> dict[str, Any]:
    return {
        "items": [],
        "satisfied": 0,
        "total": 0,
        "complete": False,
        "ok": False,
        "error": _bounded_text(message),
        "source": "evaluator-ui-only",
    }


def sanitize_bddl_progress(value: Any) -> dict[str, Any]:
    """Reduce an incoming envelope to the exact primitive UI schema."""
    if not isinstance(value, Mapping):
        raise ValueError("BDDL progress must be an object")
    raw_items = value.get("items", [])
    if not isinstance(raw_items, (list, tuple)):
        raise ValueError("BDDL progress items must be a list")
    if len(raw_items) > _MAX_GOALS:
        raise ValueError(f"BDDL progress exceeds {_MAX_GOALS} goals")

    items = []
    for position, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"BDDL goal {position} must be an object")
        full = _bounded_text(raw_item.get("full", raw_item.get("label", "")))
        label = _bounded_text(raw_item.get("label", full), limit=96)
        items.append(
            {
                "index": position,
                "label": label or f"goal {position + 1}",
                "full": full or label or f"goal {position + 1}",
                "satisfied": bool(raw_item.get("satisfied", False)),
            }
        )
    satisfied = sum(1 for item in items if item["satisfied"])
    payload = {
        "items": items,
        "satisfied": satisfied,
        "total": len(items),
        "complete": bool(items) and satisfied == len(items),
        "ok": bool(value.get("ok", bool(items))),
        "source": "evaluator-ui-only",
    }
    error = value.get("error")
    if error:
        payload["error"] = _bounded_text(error)
    return payload


def encode_bddl_progress(value: Any) -> str:
    return json.dumps(
        sanitize_bddl_progress(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_bddl_progress(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError("BDDL progress envelope must be a JSON string")
    return sanitize_bddl_progress(json.loads(value))


def build_evaluator_bddl_progress(
    task: Any,
    goal_status: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build display state from the evaluator's already-computed goal status."""
    try:
        compiled_task = getattr(task, "compiled_task", None)
        conditions = getattr(compiled_task, "conditions", None)
        parsed = list(getattr(conditions, "parsed_goal_conditions", None) or [])
        natural = list(
            getattr(task, "activity_natural_language_goal_conditions", None) or []
        )
        if not parsed and not natural:
            return awaiting_bddl_progress("evaluator task has no BDDL goal conditions")

        satisfied_indices = set()
        if isinstance(goal_status, Mapping):
            for index in goal_status.get("satisfied", []) or []:
                try:
                    satisfied_indices.add(int(index))
                except (TypeError, ValueError):
                    continue

        total = max(len(parsed), len(natural))
        items = []
        for index in range(total):
            if index < len(parsed):
                full = render_goal_expression(parsed[index])
            else:
                full = str(natural[index])
            if not full and index < len(natural):
                full = str(natural[index])
            items.append(
                {
                    "index": index,
                    "label": _bounded_text(full, limit=96),
                    "full": _bounded_text(full),
                    "satisfied": index in satisfied_indices,
                }
            )
        return sanitize_bddl_progress({"items": items, "ok": True})
    except Exception as exc:
        return awaiting_bddl_progress(
            f"evaluator BDDL status failed: {type(exc).__name__}: {exc}"
        )
