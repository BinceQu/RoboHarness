#!/usr/bin/env python3
"""Probe exact concurrent RGBD Lite IK solves in one CUDA context.

This experiment is intentionally test-only.  It replays captured official IK
requests with one fresh solver per request, assigns each request an independent
CUDA stream, and compares every returned arm result with the captured result.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from behavior_interface_eval_test.benchmark_rgbd_lite_ik_trace_replay import (
    canonical_raw_result,
    first_difference,
)


def _load_trace(path: Path) -> Dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(trace.get("request"), dict):
        raise ValueError(f"trace has no request: {path}")
    if not isinstance(trace.get("result"), dict):
        raise ValueError(f"trace has no result: {path}")
    if trace["request"].get("pose_shared_memory") is not None:
        raise ValueError(f"ephemeral shared-memory trace: {path}")
    trace["path"] = str(path)
    return trace


def _solve_trace(
    trace: Dict[str, Any],
    *,
    stream,
    start_barrier: threading.Barrier,
    graph_capture_lock: threading.Lock,
    graph_ready_barrier: threading.Barrier,
) -> Dict[str, Any]:
    import torch

    from behavior_interface_eval_test.tool.official_v2 import ik_filter_worker

    request = copy.deepcopy(trace["request"])
    arm = str(trace["arm"])
    poses = copy.deepcopy(request.get("poses") or [])
    started = time.perf_counter()
    with torch.cuda.stream(stream):
        solver, tensor_args, solver_meta = ik_filter_worker._load_solver(
            request,
            arm,
            log_fn=lambda _message: None,
        )
        q_names = [
            str(name)
            for name in solver.rollout_fn.kinematics.joint_names
        ]
        runtime_cache = ik_filter_worker._IKRuntimeTensorCache(
            request,
            solver=solver,
            tensor_args=tensor_args,
            q_names=q_names,
        )
        solver.reset_seed()
        runtime_cache.reset_request_shape_state()
        original_solve_batch = solver.solve_batch
        first_solve = True

        def solve_batch_after_serial_capture(*args, **kwargs):
            nonlocal first_solve
            if not first_solve:
                return original_solve_batch(*args, **kwargs)
            first_solve = False
            # PyTorch CUDA graph capture is process-global. Capture the first
            # fixed-shape solve for each fresh solver serially, then release
            # every request together for concurrent graph replay.
            with graph_capture_lock:
                result = original_solve_batch(*args, **kwargs)
            graph_ready_barrier.wait(timeout=300.0)
            return result

        solver.solve_batch = solve_batch_after_serial_capture
        start_barrier.wait(timeout=300.0)
        arm_results, arm_meta = ik_filter_worker._solve_arm_with_solver(
            request,
            arm,
            poses,
            solver=solver,
            tensor_args=tensor_args,
            solver_meta=solver_meta,
            runtime_cache=runtime_cache,
        )
    stream.synchronize()
    expected = trace["result"]["arms"][arm]
    difference = first_difference(
        canonical_raw_result(expected),
        canonical_raw_result(arm_results),
    )
    return {
        "trace": Path(str(trace["path"])).name,
        "arm": arm,
        "policy": str(request.get("solver_policy") or ""),
        "poses": len(poses),
        "exact": difference is None,
        "first_difference": difference,
        "wall_s": float(time.perf_counter() - started),
        "solver_elapsed_s": float(arm_meta.get("elapsed_s", 0.0)),
    }


def run_probe(
    trace_paths: Sequence[Path],
    *,
    gpu: int,
    output: Path | None,
) -> Dict[str, Any]:
    if len(trace_paths) < 1:
        raise ValueError("at least one trace is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", str(int(gpu))):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected GPU {int(gpu)}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    os.environ["IK_FILTER_CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires exactly one visible CUDA device")
    traces = [_load_trace(path.resolve()) for path in trace_paths]
    streams = [torch.cuda.Stream() for _ in traces]
    barrier = threading.Barrier(len(traces))
    graph_capture_lock = threading.Lock()
    graph_ready_barrier = threading.Barrier(len(traces))
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(traces),
        thread_name_prefix="rgbd-lite-one-context-ik",
    ) as executor:
        futures = [
            executor.submit(
                _solve_trace,
                trace,
                stream=stream,
                start_barrier=barrier,
                graph_capture_lock=graph_capture_lock,
                graph_ready_barrier=graph_ready_barrier,
            )
            for trace, stream in zip(traces, streams)
        ]
        rows = [future.result() for future in futures]
    torch.cuda.synchronize()
    report = {
        "gpu": int(gpu),
        "requests": len(rows),
        "all_exact": all(bool(row["exact"]) for row in rows),
        "wall_s": float(time.perf_counter() - started),
        "sum_request_wall_s": float(sum(row["wall_s"] for row in rows)),
        "speedup_from_overlap": float(
            sum(row["wall_s"] for row in rows)
            / max(time.perf_counter() - started, 1e-12)
        ),
        "peak_allocated_mib": float(
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        ),
        "peak_reserved_mib": float(
            torch.cuda.max_memory_reserved() / (1024 ** 2)
        ),
        "rows": rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_probe(args.trace, gpu=args.gpu, output=args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["all_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
