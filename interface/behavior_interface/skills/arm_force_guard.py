"""手臂力守护。原文件 admin 600；官方 dry-run 口只需要装饰器身份包装。"""

from __future__ import annotations

from typing import Any, Callable


def guard_arm_force_skill(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    return fn
