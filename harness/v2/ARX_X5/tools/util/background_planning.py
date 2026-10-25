"""Keep frozen RGB-D computation out of the evaluator's action-poll RPC.

Only the pure planner and its existing frozen inputs enter the worker. The
worker never receives a world, adapter, tool context, plan store or action
callback. Signing/rendering remain on the tool generator after revalidation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


BUILD = "frozen_rgbd_read_only_worker_v1"
# One worker bounds concurrent computation, including after a cancelled job.
# A pending job is cancelled on generator close; running NumPy work may finish
# but cannot publish a plan or emit any evaluator action.
_PLANNER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rgbd-read-only")


def run_frozen_rgbd_planner(ctx: Any, planner: Callable[..., Any], **inputs: Any):
    snapshot = inputs["snapshot"]
    episode_id = str(snapshot.episode_id)
    capture_epoch = int(snapshot.robot.get("motion_epoch", -1))

    def still_current():
        ctx.raise_if_cancelled("frozen RGB-D background planning")
        if str(ctx.world.episode_id()) != episode_id:
            raise ValueError("episode changed during frozen RGB-D planning")
        if int(ctx.world.motion_epoch()) != capture_epoch:
            raise ValueError("capture became stale during frozen RGB-D planning")

    still_current()
    future = _PLANNER.submit(planner, **inputs)
    try:
        while not future.done():
            still_current()
            # Defer, not hold: no simulation action, step or new observation.
            yield ctx.world.defer_action("frozen_rgbd_planning_pending")
        still_current()
        return future.result()
    finally:
        future.cancel()
