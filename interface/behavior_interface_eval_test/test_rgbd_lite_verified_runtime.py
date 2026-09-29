from __future__ import annotations

import os

from behavior_interface_eval_test import benchmark_rgbd_lite_isolated_136
from behavior_interface_eval_test import rgbd_lite_verified_runtime
from behavior_interface_eval_test.test_support import rgbd_lite_isolated_runtime


def test_interface_defaults_match_benchmark_with_safe_ik_batch() -> None:
    expected = dict(
        benchmark_rgbd_lite_isolated_136.OPTIMIZED_RUNTIME_DEFAULTS
    )
    expected["OFFICIAL_V2_RGBD_LITE_TEST_PHYSICAL_BATCH"] = "0"
    assert rgbd_lite_verified_runtime.VERIFIED_RUNTIME_DEFAULTS == expected


def test_interface_geometry_settings_match_benchmark() -> None:
    for key, value in (
        benchmark_rgbd_lite_isolated_136.CURRENT_RUNTIME_SETTINGS.items()
    ):
        assert rgbd_lite_verified_runtime._BASE_LITE_SETTINGS[key] == value


def test_install_uses_verified_defaults_and_interface_gpu(monkeypatch) -> None:
    environment = {"CUDA_VISIBLE_DEVICES": "1"}
    monkeypatch.setattr(os, "environ", environment)
    observed = {}

    def fake_install():
        observed.update(os.environ)
        return {"runtime": "fake"}

    monkeypatch.setattr(
        rgbd_lite_isolated_runtime,
        "install_policy_pool_runtime",
        fake_install,
    )

    result = (
        rgbd_lite_verified_runtime.install_verified_rgbd_lite_runtime()
    )

    assert observed["OFFICIAL_V2_RGBD_LITE_TEST_MODE"] == "1"
    assert observed["IK_FILTER_CUDA_VISIBLE_DEVICES"] == "1"
    assert observed["OFFICIAL_V2_RGBD_LITE_TEST_IK_LANES"] == "4"
    assert observed["OFFICIAL_V2_RGBD_LITE_TEST_PHYSICAL_BATCH"] == "0"
    assert observed["OFFICIAL_V2_LITE_GEOMETRY_BACKEND"] == "v53_warp_cuda"
    assert observed["OFFICIAL_V2_LITE_OCCUPANCY_DEVICE"] == "cuda:0"
    assert observed["OFFICIAL_V2_LITE_ALLOW_CPU_REFERENCE"] == "0"
    assert observed["OFFICIAL_V2_LITE_OCCUPANCY_REGION_CACHE"] == "0"
    assert observed["OFFICIAL_V2_V53_GPU_FAN_SPLIT"] == "1"
    assert observed["OFFICIAL_V2_V53_GPU_SUPPORT_PLANE"] == "1"
    assert observed["WARP_CACHE_PATH"] == "/tmp/cache/warp"
    assert observed[
        "OFFICIAL_V2_RGBD_LITE_TEST_CANDIDATE_COMPRESSION_STAGES"
    ] == "closure2,translation"
    assert "no occupancy mesh.split" in result["result_contract"]
    assert "CPU occupancy fallback" in result["result_contract"]
    assert result["enabled"] is True
    assert result["ik_gpu"] == "1"
    assert result["settings"] == {"runtime": "fake"}


def test_install_rejects_inherited_batch64_override(monkeypatch) -> None:
    environment = {
        "CUDA_VISIBLE_DEVICES": "1",
        "OFFICIAL_V2_RGBD_LITE_TEST_PHYSICAL_BATCH": "64",
    }
    monkeypatch.setattr(os, "environ", environment)
    observed = {}

    def fake_install():
        observed.update(os.environ)
        return {}

    monkeypatch.setattr(
        rgbd_lite_isolated_runtime,
        "install_policy_pool_runtime",
        fake_install,
    )

    rgbd_lite_verified_runtime.install_verified_rgbd_lite_runtime()

    assert observed["OFFICIAL_V2_RGBD_LITE_TEST_PHYSICAL_BATCH"] == "0"


def test_explicit_ik_gpu_override_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(
        os,
        "environ",
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "IK_FILTER_CUDA_VISIBLE_DEVICES": "3",
        },
    )
    monkeypatch.setattr(
        rgbd_lite_isolated_runtime,
        "install_policy_pool_runtime",
        lambda: {},
    )

    result = (
        rgbd_lite_verified_runtime.install_verified_rgbd_lite_runtime()
    )

    assert result["ik_gpu"] == "3"
