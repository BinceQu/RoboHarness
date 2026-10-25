"""Execution context presented to official_v2 generator functions."""

from __future__ import annotations

import threading
from typing import Any, Callable

from tools.util.action_world import RoboDojoActionWorld
from tools.util.storage import PlanStore, SnapshotStore
from tools.util.tracking import PointTracker


class ToolCancelled(RuntimeError):
    pass


class ToolContext:
    def __init__(
        self,
        *,
        world: RoboDojoActionWorld,
        snapshots: SnapshotStore,
        plans: PlanStore,
        tracker: PointTracker,
        task_name: str,
        cancelled: threading.Event,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.world = world
        self.snapshots = snapshots
        self.plans = plans
        self.tracker = tracker
        self.task_name = str(task_name or "")
        self._cancelled = cancelled
        self._logger = logger
        self.result: dict[str, Any] | None = None

    def set_result(self, result: dict[str, Any]) -> None:
        self.result = dict(result)

    def raise_if_cancelled(self, where: str = "tool execution") -> None:
        if self._cancelled.is_set():
            raise ToolCancelled(f"tool cancelled during {where}")

    def log(self, message: str) -> None:
        if self._logger is not None:
            self._logger(str(message))
