#!/usr/bin/env python3
"""Test-only official-v2 IK worker with a wider physical CUDA batch.

The planner request, logical batch size, seed count, seed generation, and
candidate order remain unchanged.  Only the number of rows submitted to one
``solve_batch`` call is widened.  The official worker is imported and patched
inside this child process; evaluator processes never import this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHYSICAL_BATCH_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_PHYSICAL_BATCH"
SOLVER_RESET_ENV = "OFFICIAL_V2_RGBD_LITE_TEST_SOLVER_RESET"
PREPARED_SIGNATURE_POOL_ENV = (
    "OFFICIAL_V2_RGBD_LITE_TEST_PREPARED_SIGNATURE_POOL"
)
_ACCELERATED_POLICIES = {
    "cuda_graph_split8_rewarm",
    "cuda_graph_split16_rewarm",
    "cuda_graph_warm32x6_rewarm",
    "cuda_graph_warm32x6_cold24_rewarm",
}
_SOURCE_SHAPE_RE = re.compile(r"solve_n=\d+ valid_n=\d+")


def configured_physical_batch() -> int:
    value = int(os.environ.get(PHYSICAL_BATCH_ENV, "0"))
    if value < 0:
        raise ValueError(f"{PHYSICAL_BATCH_ENV} must be non-negative")
    return value


def configured_solver_reset() -> str:
    value = str(os.environ.get(SOLVER_RESET_ENV, "none")).strip().lower()
    if value not in {"none", "optimizer", "graph", "optimizer_graph"}:
        raise ValueError(
            f"{SOLVER_RESET_ENV} must be none, optimizer, graph, or "
            "optimizer_graph"
        )
    return value


def configured_prepared_signature_pool() -> int:
    value = int(os.environ.get(PREPARED_SIGNATURE_POOL_ENV, "0"))
    if value < 0 or value > 4:
        raise ValueError(
            f"{PREPARED_SIGNATURE_POOL_ENV} must be in [0, 4]"
        )
    return value


def _persistent_signature_pool_main(arm: str, *, pool_size: int) -> None:
    """Keep fresh one-shot solver children ready for both Lite signatures."""
    from behavior_interface_eval_test.tool.official_v2 import ik_filter_worker

    preloaded = False
    prepared_by_signature: Dict[str, List[Dict[str, Any]]] = {}
    templates: Dict[str, Dict[str, Any]] = {}

    def dispose_all() -> None:
        for prepared in prepared_by_signature.values():
            for child in prepared:
                ik_filter_worker._dispose_prepared_solver(child)
        prepared_by_signature.clear()

    def ensure(signature: str, request: Dict[str, Any]) -> None:
        prepared = prepared_by_signature.setdefault(str(signature), [])
        while len(prepared) < int(pool_size):
            prepared.append(
                ik_filter_worker._spawn_prepared_solver(
                    request,
                    arm,
                    str(signature),
                )
            )

    def try_ensure(signature: str, request: Dict[str, Any]) -> None:
        try:
            ensure(signature, request)
        except Exception:
            pass

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request_id = None
            try:
                envelope = json.loads(line)
                request_id = envelope.get("request_id")
                request = envelope["request"]
                signature = str(
                    envelope.get("solver_signature") or "legacy"
                )
                if not preloaded:
                    ik_filter_worker._preload_forkserver_modules(request)
                    preloaded = True
                templates[signature] = request

                next_prepare = envelope.get("prepare_next")
                if isinstance(next_prepare, dict):
                    next_request = next_prepare.get("request")
                    next_signature = str(
                        next_prepare.get("solver_signature") or ""
                    )
                    if isinstance(next_request, dict) and next_signature:
                        templates[next_signature] = next_request

                if bool(envelope.get("prepare_only")):
                    try_ensure(signature, request)
                    result_json = '{"ok":true,"prepared":true}'
                else:
                    prepared = prepared_by_signature.setdefault(
                        signature,
                        [],
                    )
                    consumed = prepared.pop(0) if prepared else None

                    # Replenish before waiting on the current solve so solver
                    # construction overlaps the active fresh child.
                    for known_signature, template in tuple(templates.items()):
                        try_ensure(known_signature, template)

                    if consumed is not None:
                        try:
                            result_json = (
                                ik_filter_worker._prepared_solver_solve_request(
                                    consumed,
                                    request,
                                    signature,
                                )
                            )
                        except Exception:
                            ik_filter_worker._dispose_prepared_solver(consumed)
                            poses, transport_meta = (
                                ik_filter_worker._request_poses(request)
                            )
                            result_json = (
                                ik_filter_worker._forkserver_solve_request(
                                    request,
                                    arm,
                                    poses,
                                    transport_meta,
                                )
                            )
                    else:
                        poses, transport_meta = (
                            ik_filter_worker._request_poses(request)
                        )
                        result_json = (
                            ik_filter_worker._forkserver_solve_request(
                                request,
                                arm,
                                poses,
                                transport_meta,
                            )
                        )

                    if bool(json.loads(result_json).get("ok")):
                        for known_signature, template in tuple(
                            templates.items()
                        ):
                            try_ensure(known_signature, template)
            except Exception as exc:
                result_json = json.dumps(
                    ik_filter_worker._json_clean(
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=8),
                        }
                    ),
                    separators=(",", ":"),
                )
            response = (
                '{"request_id":'
                + json.dumps(request_id, separators=(",", ":"))
                + ',"result":'
                + result_json
                + "}\n"
            )
            sys.stdout.write(response)
            sys.stdout.flush()
    finally:
        dispose_all()


def official_physical_batch(req: Dict[str, Any]) -> int | None:
    """Return the unmodified official worker's physical solve width."""
    policy = str(req.get("solver_policy") or "").strip()
    logical_batch = max(1, int(req.get("batch_size", 1)))
    if policy == "cuda_graph_split8_rewarm":
        return min(8, logical_batch)
    if policy == "cuda_graph_split16_rewarm":
        return min(16, logical_batch)
    if policy in {
        "cuda_graph_warm32x6_rewarm",
        "cuda_graph_warm32x6_cold24_rewarm",
    }:
        return min(32, logical_batch)
    if policy in {
        "cuda_graph_fixed64",
        "cuda_graph_fixed64_rewarm",
        "fixed64_no_graph",
        "fixed64_rewarm_no_graph",
    }:
        return logical_batch
    return None


def accelerated_physical_batch(req: Dict[str, Any], target: int) -> int | None:
    policy = str(req.get("solver_policy") or "").strip()
    if target > 0 and policy in _ACCELERATED_POLICIES:
        official = official_physical_batch(req) or 1
        return min(
            max(int(target), int(official)),
            max(1, int(req.get("batch_size", 1))),
        )
    return official_physical_batch(req)


def rewrite_result_sources(
    results: List[Dict[str, Any]],
    *,
    req: Dict[str, Any],
    warm_flags: Sequence[bool],
) -> None:
    """Restore source diagnostics to the official physical slicing values."""
    physical_batch = official_physical_batch(req)
    if physical_batch is None or len(results) != len(warm_flags):
        return
    logical_batch = max(1, int(req.get("batch_size", 1)))
    for offset in range(0, len(results), logical_batch):
        chunk_results = results[offset : offset + logical_batch]
        chunk_warm = list(warm_flags[offset : offset + logical_batch])
        ranks = {True: 0, False: 0}
        totals = {
            True: sum(chunk_warm),
            False: len(chunk_warm) - sum(chunk_warm),
        }
        for result, is_warm in zip(chunk_results, chunk_warm):
            rank = ranks[is_warm]
            ranks[is_warm] += 1
            group_n = totals[is_warm]
            unit_start = (rank // physical_batch) * physical_batch
            valid_n = min(physical_batch, group_n - unit_start)
            source = result.get("source")
            if isinstance(source, str):
                result["source"] = _SOURCE_SHAPE_RE.sub(
                    f"solve_n={physical_batch} valid_n={valid_n}",
                    source,
                    count=1,
                )


def install_worker_patch(
    target: int,
    *,
    solver_reset: str = "none",
    prepared_signature_pool: int = 0,
) -> None:
    from behavior_interface_eval_test.tool.official_v2 import ik_filter_worker

    original_solve_pose_batch = ik_filter_worker._solve_pose_batch
    original_load_solver = ik_filter_worker._load_solver
    original_solve_arm_with_solver = ik_filter_worker._solve_arm_with_solver

    def physical_batch(req: Dict[str, Any]) -> int | None:
        return accelerated_physical_batch(req, target)

    def solve_pose_batch(
        req: Dict[str, Any],
        arm: str,
        poses: List[Dict[str, Any]],
        *,
        solver,
        tensor_args,
        q_names: List[str],
        pos_tol_m: float,
        ori_tol_deg: float,
        source_stage: str,
        runtime_cache=None,
    ):
        warm_flags = [
            ik_filter_worker._warm_start_q_for_pose(
                req,
                arm,
                pose,
                q_names,
            )
            is not None
            for pose in poses
        ]
        results, warm_count = original_solve_pose_batch(
            req,
            arm,
            poses,
            solver=solver,
            tensor_args=tensor_args,
            q_names=q_names,
            pos_tol_m=pos_tol_m,
            ori_tol_deg=ori_tol_deg,
            source_stage=source_stage,
            runtime_cache=runtime_cache,
        )
        rewrite_result_sources(results, req=req, warm_flags=warm_flags)
        return results, warm_count

    def load_solver(req: Dict[str, Any], arm: str, *, log_fn=print):
        solver, tensor_args, meta = original_load_solver(
            req,
            arm,
            log_fn=log_fn,
        )
        official_batch = official_physical_batch(req)
        if official_batch is not None:
            meta["fixed_solve_batch_size"] = int(official_batch)
        return solver, tensor_args, meta

    def solve_arm_with_solver(
        req,
        arm,
        poses,
        *,
        solver,
        tensor_args,
        solver_meta,
        runtime_cache=None,
    ):
        if solver_reset in {"optimizer", "optimizer_graph"}:
            solver.solver.reset()
        if solver_reset in {"graph", "optimizer_graph"}:
            solver.reset_cuda_graph()
        return original_solve_arm_with_solver(
            req,
            arm,
            poses,
            solver=solver,
            tensor_args=tensor_args,
            solver_meta=solver_meta,
            runtime_cache=runtime_cache,
        )

    ik_filter_worker._fixed_physical_batch_size = physical_batch
    ik_filter_worker._solve_pose_batch = solve_pose_batch
    ik_filter_worker._load_solver = load_solver
    ik_filter_worker._solve_arm_with_solver = solve_arm_with_solver
    if int(prepared_signature_pool) > 0:
        ik_filter_worker._persistent_forkserver_main = (
            lambda arm: _persistent_signature_pool_main(
                arm,
                pool_size=int(prepared_signature_pool),
            )
        )


def main() -> None:
    target = configured_physical_batch()
    solver_reset = configured_solver_reset()
    prepared_signature_pool = configured_prepared_signature_pool()
    if (
        target <= 0
        and solver_reset == "none"
        and prepared_signature_pool <= 0
    ):
        raise RuntimeError(
            f"{PHYSICAL_BATCH_ENV} must be positive or "
            f"{SOLVER_RESET_ENV} must enable a reset or "
            f"{PREPARED_SIGNATURE_POOL_ENV} must be positive"
        )
    install_worker_patch(
        target,
        solver_reset=solver_reset,
        prepared_signature_pool=prepared_signature_pool,
    )
    from behavior_interface_eval_test.tool.official_v2 import ik_filter_worker

    ik_filter_worker.main()


if __name__ == "__main__":
    main()
