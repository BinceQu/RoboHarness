"""官方 evaluator 的操作员 reset / finish 侧信道。

官方协议里 interface 不能 env.reset()：/api/reset 原先直接 409。
evaluator 主循环独占仿真，reset 只从 evaluator → policy 的 websocket 发出。

本模块在 evaluator.step() 里轮询 /tmp 请求文件。Web 按钮写入请求后，
下一拍由 evaluator 自己 reset / load_task_instance，不破坏官方所有权。

``finish`` 让当前 rollout 按官方 ``truncated`` 路径收束：eval.py 立刻写
``q_score`` JSON，再热切下一个 instance。给模型收工 / 超 max ticks 用。
中途 ``reset`` 会打乱当前 rollout 的官方 JSON 计分；只给测试口手动切世界用。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable


LISTENER_NAME = "evaluator_step"
_OP_RESET = "reset"
_OP_FINISH = "finish"
_OPERATOR_OPS = frozenset({_OP_RESET, _OP_FINISH})


def operator_dir() -> Path:
    raw = os.environ.get("BEHAVIOR_EVAL_OPERATOR_DIR", "/tmp").strip()
    return Path(raw or "/tmp")


def request_path(http_port: int) -> Path:
    return operator_dir() / f"behavior_eval_operator_request_p{int(http_port)}.json"


def status_path(http_port: int) -> Path:
    return operator_dir() / f"behavior_eval_operator_status_p{int(http_port)}.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_request(http_port: int) -> dict[str, Any] | None:
    return _read_json(request_path(int(http_port)))


def read_status(http_port: int) -> dict[str, Any]:
    payload = _read_json(status_path(int(http_port))) or {}
    payload.setdefault("state", "absent")
    payload.setdefault("listener", "")
    return payload


def write_status(http_port: int, **fields: Any) -> dict[str, Any]:
    port = int(http_port)
    payload = read_status(port)
    payload.update(fields)
    payload["http_port"] = port
    payload["updated_ts"] = time.time()
    _atomic_write_json(status_path(port), payload)
    return payload


def write_reset_request(
    http_port: int,
    instance_id: int | None = None,
) -> dict[str, Any]:
    """Web 按钮调用：写下一条 reset，等 evaluator 下一拍消费。"""
    return _write_operator_request(
        http_port,
        op=_OP_RESET,
        instance_id=None if instance_id is None else int(instance_id),
        state="queued",
    )


def write_finish_request(
    http_port: int,
    *,
    reason: str = "model_done",
) -> dict[str, Any]:
    """测试脚本调用：当前 rollout 立刻按官方 truncated 交卷，不 reset 世界。"""
    return _write_operator_request(
        http_port,
        op=_OP_FINISH,
        reason=str(reason or "model_done"),
        state="finish_queued",
    )


def _write_operator_request(
    http_port: int,
    *,
    op: str,
    instance_id: int | None = None,
    reason: str = "",
    state: str,
) -> dict[str, Any]:
    port = int(http_port)
    request_id = uuid.uuid4().hex[:16]
    payload = {
        "request_id": request_id,
        "op": op,
        "instance_id": None if instance_id is None else int(instance_id),
        "http_port": port,
        "reason": str(reason or ""),
        "ts": time.time(),
    }
    _atomic_write_json(request_path(port), payload)
    write_status(
        port,
        state=state,
        last_request_id=request_id,
        requested_instance_id=payload["instance_id"],
        last_op=op,
        error="",
    )
    return payload


def listener_ready(http_port: int) -> bool:
    status = read_status(int(http_port))
    return str(status.get("listener") or "") == LISTENER_NAME


def _pending_request(http_port: int, op: str | None = None) -> dict[str, Any] | None:
    request = read_request(int(http_port))
    if not request:
        return None
    found = str(request.get("op") or "").strip()
    if found not in _OPERATOR_OPS:
        return None
    if op is not None and found != op:
        return None
    request_id = str(request.get("request_id") or "").strip()
    if not request_id:
        return None
    status = read_status(int(http_port))
    if status.get("applied_request_id") == request_id:
        return None
    return request


def pending_reset(http_port: int) -> bool:
    """还有未落地的手动 reset 时，idle-gate 必须放行一拍，让 step() 能消费请求。"""
    return _pending_request(int(http_port), _OP_RESET) is not None


def pending_finish(http_port: int) -> bool:
    """模型收工后的交卷请求还没被 step() 消费。"""
    return _pending_request(int(http_port), _OP_FINISH) is not None


def pending_operator_work(http_port: int) -> bool:
    """reset 或 finish 都要冲破 idle-gate，否则 evaluator 看不到请求。"""
    return _pending_request(int(http_port)) is not None


def install_operator_reset_on_evaluator_class(
    evaluator_cls: type,
    http_port: int,
) -> bool:
    """改 Evaluator 类方法，让官方 eval.py 的 step 循环能收手动 reset。"""
    if getattr(evaluator_cls, "_operator_reset_installed", False):
        return False
    port = int(http_port)
    original_step = evaluator_cls.step
    original_load = evaluator_cls.load_task_instance
    original_reset = evaluator_cls.reset

    def load_task_instance(self, instance_id: int) -> Any:
        result = original_load(self, instance_id)
        self._operator_current_instance_id = int(instance_id)
        write_status(
            port,
            listener=LISTENER_NAME,
            state="ready",
            current_instance_id=int(instance_id),
            error="",
        )
        return result

    def step(self) -> Any:
        # FINISH must be handled first and short-circuit before policy.forward.
        # Short-circuiting avoids blocking on a dead agent websocket.
        if _pending_request(port, _OP_FINISH) is not None:
            write_status(
                port,
                listener=LISTENER_NAME,
                state="submitted",
                applied_request_id=str(
                    (_pending_request(port, _OP_FINISH) or {}).get("request_id") or ""
                ),
                last_op=_OP_FINISH,
                error="",
            )
            return False, True
        # RESET: let the current step complete (policy.forward + env.step) before
        # resetting the world.  This matches the original step contract.
        if _pending_request(port, _OP_RESET) is not None:
            result = original_step(self)
            _maybe_apply_operator_request(
                self,
                http_port=port,
                original_reset=original_reset,
                original_load=original_load,
            )
            return result
        return original_step(self)

    evaluator_cls.load_task_instance = load_task_instance
    evaluator_cls.step = step
    evaluator_cls._operator_reset_installed = True
    write_status(port, listener=LISTENER_NAME, state="idle", error="")
    return True


def _maybe_apply_operator_request(
    evaluator: Any,
    *,
    http_port: int,
    original_reset: Callable[..., Any],
    original_load: Callable[..., Any],
) -> bool:
    """Handle a pending RESET after the current step completes.  FINISH is handled in step()."""
    request = _pending_request(http_port)
    if not request:
        return False
    request_id = str(request.get("request_id") or "").strip()
    op = str(request.get("op") or "").strip()
    if op == _OP_FINISH:
        # FINISH is now handled before original_step — this branch is dead.
        return False
    if op != _OP_RESET:
        return False
    raw_instance = request.get("instance_id")
    target = None if raw_instance is None else int(raw_instance)
    current = getattr(evaluator, "_operator_current_instance_id", None)
    write_status(
        http_port,
        listener=LISTENER_NAME,
        state="resetting",
        last_request_id=request_id,
        requested_instance_id=target,
        last_op=_OP_RESET,
        error="",
    )
    try:
        original_reset(evaluator)
        if target is not None and int(target) != (None if current is None else int(current)):
            original_load(evaluator, int(target))
            evaluator._operator_current_instance_id = int(target)
            original_reset(evaluator)
        write_status(
            http_port,
            listener=LISTENER_NAME,
            state="ready",
            last_request_id=request_id,
            applied_request_id=request_id,
            last_op=_OP_RESET,
            current_instance_id=getattr(
                evaluator, "_operator_current_instance_id", target
            ),
            error="",
        )
    except Exception as exc:
        write_status(
            http_port,
            listener=LISTENER_NAME,
            state="error",
            last_request_id=request_id,
            applied_request_id=request_id,
            last_op=_OP_RESET,
            error=f"{type(exc).__name__}: {exc}",
        )
    return False
