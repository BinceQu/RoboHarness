"""Small registry types independent of the legacy skill implementations."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    """BehaviorInterface-compatible tool metadata."""

    name: str
    fn: Callable
    description: str = ""
    params: list[dict[str, Any]] = field(default_factory=list)


def tool_spec(name: str, fn: Callable, description: str) -> ToolSpec:
    params: list[dict[str, Any]] = []
    for index, (param_name, param) in enumerate(
        inspect.signature(fn).parameters.items()
    ):
        if index == 0:
            continue
        annotation = (
            "any"
            if param.annotation is inspect.Signature.empty
            else getattr(param.annotation, "__name__", str(param.annotation))
        )
        params.append(
            {
                "name": param_name,
                "type": annotation,
                "default": (
                    None
                    if param.default is inspect.Signature.empty
                    else param.default
                ),
                "required": param.default is inspect.Signature.empty,
            }
        )
    return ToolSpec(
        name=str(name),
        fn=fn,
        description=str(description),
        params=params,
    )
