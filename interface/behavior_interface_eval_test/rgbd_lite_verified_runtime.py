"""Install the V53 + Warp CUDA RGB-D Lite runtime in the eval interface."""

from __future__ import annotations

import os
from typing import Dict, MutableMapping


RUNTIME_ID = "rgbd_lite_v53_warp_cuda_v1_20260901"
_SAFE_IK_PHYSICAL_BATCH = "0"

# Keep the planner scheduling defaults aligned with the 136-case harness except
# for the unsafe physical-batch override. The new Warp occupancy path is
# fail-closed but has not been timed on this host because CUDA is unavailable in
# the current test environment.
VERIFIED_RUNTIME_DEFAULTS = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "CUDA_MODULE_LOADING": "LAZY",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OFFICIAL_V2_RGBD_LITE_TEST_CONCURRENCY": "6",
    "OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES": "4",
    "OFFICIAL_V2_RGBD_LITE_TEST_RENDER_PROCESSES": "8",
    "OFFICIAL_V2_RGBD_LITE_TEST_COMPONENT_CACHE": "1",
    "OFFICIAL_V2_RGBD_LITE_TEST_CAMERA_FACE_ORIENTATION_CACHE": "1",
    "OFFICIAL_V2_RGBD_LITE_TEST_NORMAL_ALIGNMENT_CACHE": "1",
    "OFFICIAL_V2_RGBD_LITE_TEST_POSE_GENERATION_CACHE": "1",
    "OFFICIAL_V2_RGBD_LITE_TEST_SIGNATURE_STICKY_LANES": "1",
    "OFFICIAL_V2_RGBD_LITE_TEST_SHARED_MEMORY": "0",
    "OFFICIAL_V2_RGBD_LITE_TEST_GLOBAL_SCHEDULE": "1",
    "OFFICIAL_V2_RGBD_LITE_TEST_PHYSICAL_BATCH": _SAFE_IK_PHYSICAL_BATCH,
    "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_COMPRESSION": "1",
    "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_COMPRESSION_SCOPE": "all",
    "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_COMPRESSION_STAGES": (
        "closure2,translation"
    ),
    "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_POST_LIMIT": "32",
    "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_STAGE_LIMIT": "32",
}

_BASE_LITE_SETTINGS = {
    "PLAN_GRASP_RGBD_LITE_PRODUCTION_FALLBACK": "0",
    "OFFICIAL_V2_LITE_PERSISTENT_IK": "1",
    "OFFICIAL_V2_LITE_PERSIST_BASELINE_IK": "0",
    "OFFICIAL_V2_LITE_FORKSERVER_IK": "1",
    "OFFICIAL_V2_LITE_PREPARED_IK": "1",
    "OFFICIAL_V2_LITE_EXACT_SOLVER_RELOAD": "1",
    "OFFICIAL_V2_LITE_IK_SHARED_MEMORY": "0",
    "OFFICIAL_V2_LITE_SCENE_CACHE": "1",
    "OFFICIAL_V2_LITE_SCENE_CACHE_SIZE": "2",
    "OFFICIAL_V2_LITE_OCCUPANCY_CACHE": "1",
    "OFFICIAL_V2_LITE_OCCUPANCY_CACHE_SIZE": "4",
    "OFFICIAL_V2_LITE_GEOMETRY_BACKEND": "v53_warp_cuda",
    "OFFICIAL_V2_LITE_OCCUPANCY_DEVICE": "cuda:0",
    "OFFICIAL_V2_LITE_GPU_OCCUPANCY_MAX_VOXELS": "4000000",
    "OFFICIAL_V2_LITE_GPU_OCCUPANCY_CHUNK_VOXELS": "262144",
    "OFFICIAL_V2_LITE_GPU_OCCUPANCY_MIN_FREE_MIB": "512",
    "OFFICIAL_V2_LITE_WARP_MAX_MESH_TRIANGLES": "4000000",
    "OFFICIAL_V2_LITE_WARP_MESH_CACHE_SIZE": "2",
    "OFFICIAL_V2_LITE_WARP_COMPONENT_LABEL_ITERATIONS": "64",
    "OFFICIAL_V2_LITE_WARP_ORIENTATION_REPAIR_ITERATIONS": "2048",
    "OFFICIAL_V2_V53_GPU_FAN_SPLIT": "1",
    "OFFICIAL_V2_V53_GPU_FAN_MAX_CORNERS": "6000000",
    "OFFICIAL_V2_V53_GPU_FAN_COMPONENT_ITERATIONS": "64",
    "OFFICIAL_V2_V53_GPU_SUPPORT_PLANE": "1",
    "OFFICIAL_V2_V53_SUPPORT_STRUCTURE_CACHE": "1",
    "OFFICIAL_V2_V53_SUPPORT_STRUCTURE_CACHE_SIZE": "2",
    "OFFICIAL_V2_LITE_ALLOW_CPU_REFERENCE": "0",
    "OFFICIAL_V2_LITE_PROJECTIVE_BACK_EXTRUSION_M": "0.060",
    "OFFICIAL_V2_LITE_PROJECTIVE_FRONT_TOLERANCE_M": "0.002",
    "OFFICIAL_V2_LITE_PROJECTIVE_PIXEL_RADIUS": "1",
    "OFFICIAL_V2_LITE_PROJECTIVE_THREE_VIEW": "0",
    "OFFICIAL_V2_LITE_OCCUPANCY_REGION_CACHE": "0",
    "OFFICIAL_V2_LITE_OCCUPANCY_REGION_CACHE_SIZE": "5",
    "OFFICIAL_V2_LITE_RENDER_CACHE": "1",
    "OFFICIAL_V2_LITE_RENDER_CACHE_SIZE": "4",
    "OFFICIAL_V2_LITE_ASYNC_THREE_VIEW": "0",
}


def apply_verified_runtime_defaults(
    environment: MutableMapping[str, str],
) -> None:
    for key, value in VERIFIED_RUNTIME_DEFAULTS.items():
        environment.setdefault(key, value)


def install_verified_rgbd_lite_runtime() -> Dict[str, object]:
    """Activate fail-closed Warp occupancy and the verified Lite IK runtime."""
    apply_verified_runtime_defaults(os.environ)
    os.environ.update(_BASE_LITE_SETTINGS)
    # The batch-64 wrapper changed strict-IK results on frozen live fixtures.
    # Keep it available to offline benchmarks, but use the official worker here.
    os.environ["OFFICIAL_V2_RGBD_LITE_TEST_PHYSICAL_BATCH"] = (
        _SAFE_IK_PHYSICAL_BATCH
    )
    os.environ.setdefault(
        "WARP_CACHE_PATH",
        os.path.join(
            os.environ.get("OMNIGIBSON_APPDATA_PATH", "/tmp"),
            "cache",
            "warp",
        ),
    )
    os.environ["OFFICIAL_V2_RGBD_LITE_TEST_MODE"] = "1"
    os.environ.setdefault(
        "IK_FILTER_CUDA_VISIBLE_DEVICES",
        os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )

    from behavior_interface_eval_test.test_support.rgbd_lite_isolated_runtime import (
        install_policy_pool_runtime,
    )

    settings = install_policy_pool_runtime()
    return {
        "enabled": True,
        "runtime_id": RUNTIME_ID,
        "result_contract": (
            "representative V53 geometry with local Warp CUDA winding "
            "occupancy; no occupancy mesh.split, trimesh.contains, "
            "pyembree, or CPU occupancy fallback"
        ),
        "verified_speedup": None,
        "ik_gpu": os.environ["IK_FILTER_CUDA_VISIBLE_DEVICES"],
        "settings": dict(settings),
    }


__all__ = [
    "RUNTIME_ID",
    "VERIFIED_RUNTIME_DEFAULTS",
    "apply_verified_runtime_defaults",
    "install_verified_rgbd_lite_runtime",
]
