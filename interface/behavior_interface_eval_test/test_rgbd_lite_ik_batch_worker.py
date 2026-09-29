from __future__ import annotations

from behavior_interface_eval_test.test_support import rgbd_lite_ik_batch_worker as worker


def test_configured_solver_reset_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv(worker.SOLVER_RESET_ENV, "unknown")

    try:
        worker.configured_solver_reset()
    except ValueError as error:
        assert worker.SOLVER_RESET_ENV in str(error)
    else:
        raise AssertionError("unknown reset mode must fail")


def test_configured_solver_reset_accepts_optimizer(monkeypatch):
    monkeypatch.setenv(worker.SOLVER_RESET_ENV, "optimizer")

    assert worker.configured_solver_reset() == "optimizer"


def test_configured_prepared_signature_pool_is_bounded(monkeypatch):
    monkeypatch.setenv(worker.PREPARED_SIGNATURE_POOL_ENV, "2")
    assert worker.configured_prepared_signature_pool() == 2

    monkeypatch.setenv(worker.PREPARED_SIGNATURE_POOL_ENV, "5")
    try:
        worker.configured_prepared_signature_pool()
    except ValueError as error:
        assert worker.PREPARED_SIGNATURE_POOL_ENV in str(error)
    else:
        raise AssertionError("prepared signature pool must be bounded")


def test_official_and_accelerated_physical_batch_widths():
    split8 = {"solver_policy": "cuda_graph_split8_rewarm", "batch_size": 64}
    warm32 = {
        "solver_policy": "cuda_graph_warm32x6_rewarm",
        "batch_size": 64,
    }

    assert worker.official_physical_batch(split8) == 8
    assert worker.official_physical_batch(warm32) == 32
    assert worker.accelerated_physical_batch(split8, 64) == 64
    assert worker.accelerated_physical_batch(warm32, 64) == 64
    assert worker.accelerated_physical_batch(split8, 16) == 16
    assert worker.accelerated_physical_batch(warm32, 16) == 32
    assert worker.official_physical_batch(
        {"solver_policy": "fixed64_no_graph", "batch_size": 64}
    ) == 64


def test_source_rewrite_restores_official_split_shape_for_mixed_warm_rows():
    req = {"solver_policy": "cuda_graph_split8_rewarm", "batch_size": 64}
    warm_flags = [False] * 9 + [True] * 3
    results = [
        {
            "source": (
                "gpu_worker stage=final arm=right offset=0 batch=0 "
                f"warm={int(is_warm)} solve_n=64 valid_n=12 "
                f"logical_n={sum(flag == is_warm for flag in warm_flags)} "
                "fixed_batch=1 cuda_graph=1"
            )
        }
        for is_warm in warm_flags
    ]

    worker.rewrite_result_sources(results, req=req, warm_flags=warm_flags)

    assert "solve_n=8 valid_n=8" in results[0]["source"]
    assert "solve_n=8 valid_n=1" in results[8]["source"]
    assert "solve_n=8 valid_n=3" in results[9]["source"]


def test_source_rewrite_handles_multiple_logical_chunks():
    req = {"solver_policy": "cuda_graph_warm32x6_rewarm", "batch_size": 64}
    warm_flags = [False] * 65
    results = [
        {
            "source": (
                f"gpu_worker stage=final arm=left offset={(i // 64) * 64} "
                f"batch={i % 64} warm=0 solve_n=64 valid_n=64 "
                f"logical_n={64 if i < 64 else 1} fixed_batch=1 cuda_graph=1"
            )
        }
        for i in range(65)
    ]

    worker.rewrite_result_sources(results, req=req, warm_flags=warm_flags)

    assert "solve_n=32 valid_n=32" in results[0]["source"]
    assert "solve_n=32 valid_n=32" in results[32]["source"]
    assert "solve_n=32 valid_n=1" in results[64]["source"]
