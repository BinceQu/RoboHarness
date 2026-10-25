"""Synchronous public requests over an observation/action generator loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time
import traceback
import uuid
import re
from typing import Any, Generator

from tools.util.registry import validate_submission
from tools.util.registry import strip_context_arguments
from tools.util.motion_constants import (
    EXEC_MOVE_TRACKED_DENSE_ACTION_BUDGET,
    EXEC_TOTAL_ACTION_BUDGET,
)
from tools.util.registry import PUBLIC_TOOL_FUNCTIONS
from tools.util.motion_common import attach_exit_capture
from .action_world import DeferAction, RoboDojoActionWorld, WaitAction
from .context import ToolContext
from .observation import ObservationAdapter
from .storage import PlanStore, SnapshotStore
from .tracking import PointTracker


# These tools are observation/planning calls. A future implementation bug must
# fail inside the policy boundary instead of silently spending evaluator steps.
_ACTION_FREE_TOOLS = frozenset(
    {
        "capture_head_camera",
        "capture_left_wrist_camera",
        "capture_right_wrist_camera",
        "read_depth",
        "track_object_distance",
        "move_chassis_to_floor_point",
        "adjust_chassis",
        "adjust_pitch",
        "adjust_height",
        "spin_to_facing_point",
        "plan_eef_translation_to_uvd_point",
        "move_to_reach_point",
        "measure_shoulder_distance",
        "plan_grasp_point_filter_rgbd_lite",
        "plan_press_point",
        "reset_body",
    }
)
_ACTION_BUDGETS = {"exec_plan_pose": EXEC_TOTAL_ACTION_BUDGET}


@dataclass
class ToolJob:
    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    created_at: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    generator: Generator[Any, None, None] | None = None
    context: ToolContext | None = None
    evaluator_actions_dispatched: int = 0
    action_budget_resolved: bool = False
    action_budget: int | None = None
    release_requested: bool = False


class ToolScheduler:
    def __init__(
        self,
        *,
        adapter: ObservationAdapter,
        world: RoboDojoActionWorld,
        snapshots: SnapshotStore,
        plans: PlanStore,
        tracker: PointTracker,
        task_name: str = "",
    ) -> None:
        self.adapter = adapter
        self.world = world
        self.snapshots = snapshots
        self.plans = plans
        self.tracker = tracker
        self.task_name = str(task_name or "")
        self._lock = threading.RLock()
        self._queue: deque[ToolJob] = deque()
        self._jobs: dict[str, ToolJob] = {}
        self._current: ToolJob | None = None
        self._awaiting_observation_sequence: int | None = None
        self._last_capture_session_id = "official_v2"

    @staticmethod
    def _valid_capture_session_id(value: Any) -> str:
        session_id = str(value or "").strip()
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", session_id):
            return session_id
        return ""

    def set_capture_session_id(self, session_id: str) -> None:
        normalized = self._valid_capture_session_id(session_id)
        if normalized:
            with self._lock:
                self._last_capture_session_id = normalized

    def submit(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        raw_arguments = dict(arguments or {})
        # Keep direct runtime/embodied-agent submissions aligned with the HTTP
        # gateway: session_id is transport metadata for this sessionless public
        # callable and must not cross its strict signature boundary.
        capture_session_id = self._valid_capture_session_id(
            raw_arguments.get("session_id")
        )
        normalized_arguments = strip_context_arguments(tool_name, raw_arguments)
        normalized = validate_submission(tool_name, normalized_arguments)
        capture_session_id = (
            self._valid_capture_session_id(normalized.get("session_id"))
            or capture_session_id
        )
        job = ToolJob(
            request_id=uuid.uuid4().hex,
            tool_name=str(tool_name),
            arguments=normalized,
        )
        with self._lock:
            if capture_session_id:
                self._last_capture_session_id = capture_session_id
            self._queue.append(job)
            self._jobs[job.request_id] = job
        return job.request_id

    def wait(self, request_id: str, timeout_s: float = 120.0) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(request_id))
        if job is None:
            raise KeyError(f"unknown request_id {request_id}")
        if not job.done.wait(timeout=max(0.0, float(timeout_s))):
            raise TimeoutError(
                f"tool {job.tool_name} timed out waiting for evaluator actions"
            )
        return dict(job.result or {"ok": False, "error": "tool returned no result"})

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(str(request_id))
        if job is None or job.done.is_set():
            return False
        job.cancelled.set()
        return True

    def release(self, request_id: str) -> None:
        """Drop a delivered receipt, or retire a timed-out job once it finishes.

        The runtime persists full receipts to tool_call_history. Retaining every
        completed job here also retains results, contexts and generator frames
        for the entire episode. An active/cancelled job stays owned until done.
        """
        with self._lock:
            job = self._jobs.get(str(request_id))
            if job is not None:
                job.release_requested = True
                if job.done.is_set():
                    self._jobs.pop(job.request_id, None)

    def cancel_current(self) -> bool:
        with self._lock:
            job = self._current
        if job is None or job.done.is_set():
            return False
        job.cancelled.set()
        return True

    def job_status(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(str(request_id))
        if job is None:
            raise KeyError(f"unknown request_id {request_id}")
        return {
            "request_id": job.request_id,
            "tool": job.tool_name,
            "done": job.done.is_set(),
            "cancelled": job.cancelled.is_set(),
            "result": None if job.result is None else dict(job.result),
        }

    def reset(self) -> None:
        with self._lock:
            jobs = list(self._queue)
            self._queue.clear()
            if self._current is not None:
                jobs.append(self._current)
                self._current = None
            self._awaiting_observation_sequence = None
        for job in jobs:
            job.cancelled.set()
            self._finish(
                job,
                {
                    "ok": False,
                    "error": "episode reset before tool completion",
                    "code": "episode_reset",
                    "tool": job.tool_name,
                    "tool_version": "official_v2",
                },
            )

    def ingest(self, observation: dict[str, Any]) -> None:
        self.adapter.ingest(observation)
        self.tracker.update(self.adapter)

    @staticmethod
    def _allows_evaluator_action(job: ToolJob) -> bool:
        if job.tool_name in _ACTION_FREE_TOOLS:
            return False
        if job.tool_name == "move_tracked_point":
            return str(job.arguments.get("execution_mode", "exec")) != "plan"
        return True

    def _action_budget_for_job(self, job: ToolJob) -> int | None:
        """Resolve the per-request evaluator-action ceiling once.

        Ordinary ``exec_plan_pose`` requests retain the historical 50-action
        guard.  A plan signed by ``move_tracked_point`` is different: the
        current RoboDojo port explicitly replays its dense signed waypoints,
        so it uses the port's large dense-replay ceiling.  Loading the signed
        plan here also means a forged ``source_tool`` cannot opt into the
        exception, and the lookup is cached on the job (not repeated for every
        evaluator action).
        """

        if job.action_budget_resolved:
            return job.action_budget
        budget = _ACTION_BUDGETS.get(job.tool_name)
        if job.tool_name == "exec_plan_pose":
            plan_id = str(job.arguments.get("plan_id", "") or "").strip()
            session_id = str(job.arguments.get("session_id", "") or "").strip()
            if plan_id:
                try:
                    plan = self.plans.load_any(plan_id, session_id or None)
                except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
                    # The callback will report the canonical plan error.  Keep
                    # the conservative legacy budget when it cannot be read.
                    plan = None
                if (
                    isinstance(plan, dict)
                    and str(plan.get("source_tool") or "") == "move_tracked_point"
                ):
                    budget = EXEC_MOVE_TRACKED_DENSE_ACTION_BUDGET
        job.action_budget = budget
        job.action_budget_resolved = True
        return budget

    def next_action(self) -> dict[str, Any] | None:
        if not self.adapter.connected():
            return None
        current_sequence = int(self.adapter.status()["sequence"])
        with self._lock:
            if self._awaiting_observation_sequence is not None:
                if current_sequence <= self._awaiting_observation_sequence:
                    return None
                self._awaiting_observation_sequence = None
            if self._current is None and self._queue:
                self._current = self._queue.popleft()
            job = self._current
        if job is None:
            return None
        if job.cancelled.is_set() and not (
            job.tool_name == "move_tracked_point" and job.generator is not None
        ):
            self._finish_current(
                {
                    "ok": False,
                    "error": "tool request was cancelled",
                    "code": "cancelled",
                    "tool": job.tool_name,
                    "tool_version": "official_v2",
                }
            )
            return None
        try:
            if job.generator is None:
                job.context = ToolContext(
                    world=self.world,
                    snapshots=self.snapshots,
                    plans=self.plans,
                    tracker=self.tracker,
                    task_name=self.task_name,
                    cancelled=job.cancelled,
                )
                function = PUBLIC_TOOL_FUNCTIONS[job.tool_name]
                job.generator = function(job.context, **job.arguments)
            for _ in range(64):
                try:
                    yielded = next(job.generator)
                except StopIteration:
                    result = None if job.context is None else job.context.result
                    self._finish_current(
                        result
                        or {
                            "ok": False,
                            "error": "tool completed without setting a result",
                            "tool": job.tool_name,
                            "tool_version": "official_v2",
                        }
                    )
                    return None
                if isinstance(yielded, DeferAction):
                    # Unlike WaitAction, a defer must leave next_action now. It
                    # lets a background planner span policy polls without an
                    # observation update or an evaluator action.
                    return None
                if isinstance(yielded, WaitAction) or yielded is None:
                    continue
                if not isinstance(yielded, dict):
                    raise TypeError(
                        f"tool {job.tool_name} yielded unsupported action {type(yielded).__name__}"
                    )
                if not self._allows_evaluator_action(job):
                    raise RuntimeError(
                        f"action-free tool {job.tool_name} attempted to emit an "
                        "evaluator action"
                    )
                action_budget = self._action_budget_for_job(job)
                if (
                    action_budget is not None
                    and job.evaluator_actions_dispatched >= action_budget
                ):
                    raise RuntimeError(
                        f"tool {job.tool_name} exceeded its hard evaluator-action "
                        f"budget of {action_budget}"
                    )
                job.evaluator_actions_dispatched += 1
                self.world.mark_action_dispatched(
                    yielded,
                    tool_name=job.tool_name,
                )
                with self._lock:
                    self._awaiting_observation_sequence = current_sequence
                return yielded
            raise RuntimeError(
                f"tool {job.tool_name} yielded too many internal wait events"
            )
        except Exception as exc:
            self._finish_current(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "failure_stage": "tool execution",
                    "traceback": traceback.format_exc(limit=8),
                    "tool": job.tool_name,
                    "tool_version": "official_v2",
                }
            )
            return None

    def _finish_current(self, result: dict[str, Any]) -> None:
        with self._lock:
            job = self._current
            self._current = None
        if job is not None:
            self._finish(job, result)

    def _finish(self, job: ToolJob, result: dict[str, Any]) -> None:
        if (
            job.tool_name == "plan_grasp_point_filter_rgbd_lite"
            and job.generator is not None
        ):
            # Release the background Future on reset/cancel. Only the pure
            # computation can outlive this generator, never plan publication.
            try:
                job.generator.close()
            except (RuntimeError, ValueError):
                # A concurrent poll may still be unwinding. Its cancellation
                # and epoch checks remain fail-closed before plan publication.
                pass
        context = job.context
        if context is None:
            context = ToolContext(
                world=self.world,
                snapshots=self.snapshots,
                plans=self.plans,
                tracker=self.tracker,
                task_name=self.task_name,
                cancelled=job.cancelled,
            )
        result_session_id = self._valid_capture_session_id(result.get("session_id"))
        argument_session_id = self._valid_capture_session_id(
            job.arguments.get("session_id")
        )
        with self._lock:
            capture_session_id = (
                result_session_id
                or argument_session_id
                or self._last_capture_session_id
            )
            if result_session_id or argument_session_id:
                self._last_capture_session_id = capture_session_id
        try:
            completed = attach_exit_capture(
                context,
                tool_name=job.tool_name,
                arguments=job.arguments,
                result=result,
                session_id=capture_session_id,
            )
        except Exception as exc:
            completed = {
                **dict(result),
                "exit_capture_attempted": True,
                "observation": None,
                "capture_error": f"{type(exc).__name__}: {exc}",
            }
        job.result = completed
        with self._lock:
            job.done.set()
            if job.release_requested:
                self._jobs.pop(job.request_id, None)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queued_jobs": len(self._queue),
                "current_request_id": None
                if self._current is None
                else self._current.request_id,
                "current_tool": None
                if self._current is None
                else self._current.tool_name,
                "known_jobs": len(self._jobs),
            }
