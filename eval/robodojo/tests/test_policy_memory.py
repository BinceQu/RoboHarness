import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

from tests.upstream.helpers import fake_observation
from tests.upstream.runtime_fixture import InterfaceRuntime


def test_consumed_receipts_release_context_and_preserve_snapshots(tmp_path):
    runtime = InterfaceRuntime(runs_root=tmp_path)
    runtime.update_observation(fake_observation(48, 64))
    for _ in range(32):
        request = runtime.submit("capture_head_camera", {"session_id": "memory-test"})
        assert runtime.get_action() == []
        receipt = runtime.wait(request, 1)
        context = weakref.ref(runtime.scheduler._jobs[request].context)
        runtime.scheduler.release(request)
        assert context() is None
        assert runtime.scheduler.status()["known_jobs"] == 0
        # The complete result remains usable after its execution state is gone.
        snapshot = runtime.snapshots.load("memory-test", receipt["image_id"])
        assert snapshot.rgb.shape == (48, 64, 3)


def test_timed_out_job_is_retained_until_cancellation_completes(tmp_path):
    runtime = InterfaceRuntime(runs_root=tmp_path)
    runtime.update_observation(fake_observation(48, 64))
    request = runtime.submit("capture_head_camera", {"session_id": "memory-test"})
    runtime.scheduler.cancel(request)
    runtime.scheduler.release(request)
    assert runtime.scheduler.status()["known_jobs"] == 1
    assert runtime.get_action() == []
    assert runtime.scheduler.status()["known_jobs"] == 0
    assert runtime.scheduler.status()["current_tool"] is None


def test_body_rendering_has_bounded_memory_and_stable_pixels(tmp_path):
    script = Path(__file__).resolve().parents[1] / "validation/render_memory.py"
    report = tmp_path / "memory.json"
    environment = {**os.environ, "MALLOC_ARENA_MAX": "2", "LP_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
    subprocess.run([sys.executable, str(script), "--iterations", "32", "--output", str(report)],
                   env=environment, check=True, timeout=240)
    data = json.loads(report.read_text())
    warm, final = data["memory"][4], data["memory"][-2]
    retained = lambda row: row["pss_kib"] + row["swap_pss_kib"]
    assert retained(final) - retained(warm) < 128 * 1024, data["memory"]
    assert final["threads"] <= 12
    assert len(set(data["image_hashes"][::2])) == 1
    assert len(set(data["image_hashes"][1::2])) == 1
    assert data["image_hashes"][0] != data["image_hashes"][1]
