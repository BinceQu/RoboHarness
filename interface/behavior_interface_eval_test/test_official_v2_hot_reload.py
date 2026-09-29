"""Regression tests for the official-v2 development reload boundary.

These tests are deliberately interface-free.  They exercise only policy-side
objects and Flask's in-process test client, so a failed test can never reset an
evaluator episode or mutate simulator state.
"""

from __future__ import annotations

import ast
import concurrent.futures
import importlib
import os
import queue
import sys
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from flask import Flask

from behavior_interface_eval_test.official_policy_interface import (
    OfficialPolicyRuntime,
    install_live_unit_test_routes,
)
from behavior_interface_eval_test.robot_contract import ACTION_DIM, PROPRIO_DIM
from behavior_interface_eval_test.tool.official_v2 import (
    human_track_object_distance_ui as human_ui,
)


class _StatusJob:
    def __init__(self, state: str, future=None) -> None:
        self._state = str(state)
        self._future = future

    def status(self):
        return {"state": self._state}


class _OwnershipLock:
    """Small reentrant lock which exposes ownership to race-boundary tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.depth = 0

    @property
    def held(self) -> bool:
        return self.depth > 0

    def __enter__(self):
        self._lock.acquire()
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.depth -= 1
        self._lock.release()
        return False


def _bare_runtime(*, job=None) -> OfficialPolicyRuntime:
    runtime = object.__new__(OfficialPolicyRuntime)
    runtime._live_test_lock = threading.RLock()
    runtime._live_test_job = job
    runtime._live_test_module_name = "unit.fake_live_reload_module"
    runtime._live_test_module_build = "old-build"
    return runtime


class ExistingLiveReloadSafetyTests(unittest.TestCase):
    """Properties the full official-v2 transaction must continue to satisfy."""

    def test_active_live_job_rejects_reload_before_import_mutation(self) -> None:
        for state in ("planning", "running", "settling", "verifying"):
            with self.subTest(state=state):
                runtime = _bare_runtime(job=_StatusJob(state))
                with mock.patch(
                    "behavior_interface_eval_test.official_policy_interface."
                    "importlib.invalidate_caches"
                ) as invalidate, mock.patch(
                    "behavior_interface_eval_test.official_policy_interface."
                    "importlib.import_module"
                ) as import_module:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "cannot reload live test module while a live test is running",
                    ):
                        runtime.reload_live_test_module()
                invalidate.assert_not_called()
                import_module.assert_not_called()
                self.assertEqual(runtime._live_test_module_build, "old-build")

    def test_cancelled_job_with_running_planner_rejects_reload(self) -> None:
        future = concurrent.futures.Future()
        runtime = _bare_runtime(job=_StatusJob("cancelled", future=future))
        with mock.patch(
            "behavior_interface_eval_test.official_policy_interface."
            "importlib.invalidate_caches"
        ) as invalidate:
            with self.assertRaisesRegex(RuntimeError, "planner thread"):
                runtime.reload_live_test_module()
        invalidate.assert_not_called()
        self.assertEqual(runtime._live_test_module_build, "old-build")

    def test_reload_preserves_policy_owned_episode_and_tracker_state(self) -> None:
        finished = concurrent.futures.Future()
        finished.set_result(None)
        runtime = _bare_runtime(job=_StatusJob("done", future=finished))

        adapter = SimpleNamespace(sequence=177)
        world = SimpleNamespace(episode="episode-live-17")
        tracker = SimpleNamespace(active_names=["head", "tail"])
        task_memory = {"tracked": ["head", "tail"]}
        initializer = SimpleNamespace(initialized=True)
        runtime.adapter = adapter
        runtime.server = SimpleNamespace(world=world)
        runtime.tracked_object_distances = tracker
        runtime.task_memory = task_memory
        runtime.episode_initializer = initializer

        replacement = SimpleNamespace(BUILD="new-build")
        # Patch ``reload`` before ``import_module``: both attributes belong to
        # Python's shared importlib module, and replacing import_module first
        # would also interfere with mock's dotted-target resolver.
        with mock.patch(
            "behavior_interface_eval_test.official_policy_interface.importlib.reload",
            return_value=replacement,
        ), mock.patch(
            "behavior_interface_eval_test.official_policy_interface."
            "importlib.invalidate_caches"
        ), mock.patch(
            "behavior_interface_eval_test.official_policy_interface."
            "importlib.import_module",
            return_value=replacement,
        ):
            loaded = runtime.reload_live_test_module()

        self.assertIs(loaded, replacement)
        self.assertEqual(runtime._live_test_module_build, "new-build")
        self.assertIs(runtime.adapter, adapter)
        self.assertIs(runtime.server.world, world)
        self.assertIs(runtime.tracked_object_distances, tracker)
        self.assertIs(runtime.task_memory, task_memory)
        self.assertIs(runtime.episode_initializer, initializer)
        self.assertEqual(runtime.adapter.sequence, 177)
        self.assertEqual(runtime.server.world.episode, "episode-live-17")
        self.assertEqual(runtime.tracked_object_distances.active_names, ["head", "tail"])

    def test_live_start_is_linearized_by_the_full_stack_reload_lock(self) -> None:
        runtime = object.__new__(OfficialPolicyRuntime)
        reload_lock = _OwnershipLock()
        runtime._official_reload_lock = reload_lock
        runtime._live_test_lock = threading.RLock()
        runtime._live_test_job = None
        runtime._live_test_module_name = "unit.fake_live_reload_module"
        runtime.evaluator_ready = lambda: True
        runtime._live_test_providers = lambda: {"provider": "current"}

        class Job:
            job_id = "live-lock-test"

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def start(self):
                self_test.assertTrue(reload_lock.held)
                return {"ok": True, "state": "planning"}

        self_test = self

        def reload_harness():
            self.assertTrue(reload_lock.held)
            return SimpleNamespace(BUILD="live-lock-build", LiveMoveTrackedPointJob=Job)

        runtime.reload_live_test_module = reload_harness
        runtime.server = SimpleNamespace(
            skill_lock=threading.RLock(),
            current_job=None,
            skill_queue=queue.Queue(),
            world=SimpleNamespace(episode_initialized=lambda: True),
            log=mock.Mock(),
        )

        result = OfficialPolicyRuntime.start_live_move_tracked_point_test(
            runtime, {"points": []}
        )

        self.assertTrue(result["ok"])
        self.assertFalse(reload_lock.held)
        self.assertEqual(runtime._live_test_job.kwargs["module_build"], "live-lock-build")

    def test_live_cancel_and_release_are_linearized_by_reload_lock(self) -> None:
        for operation in ("cancel", "release"):
            with self.subTest(operation=operation):
                runtime = object.__new__(OfficialPolicyRuntime)
                reload_lock = _OwnershipLock()
                runtime._official_reload_lock = reload_lock
                runtime._live_test_lock = threading.RLock()
                runtime._live_test_module_build = "live-lock-build"

                class Job:
                    def __init__(self):
                        self.state = "done" if operation == "release" else "running"

                    def status(self):
                        self_test.assertTrue(reload_lock.held)
                        return {"ok": True, "state": self.state}

                    def cancel(self, _reason):
                        self_test.assertTrue(reload_lock.held)
                        self.state = "cancelled"

                self_test = self
                runtime._live_test_job = Job()
                runtime.official_v2_reload_status = lambda: {"generation": 7}

                if operation == "cancel":
                    result = OfficialPolicyRuntime.cancel_live_test(runtime, "unit cancel")
                    self.assertEqual(result["state"], "cancelled")
                else:
                    OfficialPolicyRuntime.release_terminal_live_test(runtime)
                    self.assertIsNone(runtime._live_test_job)
                self.assertFalse(reload_lock.held)


class HotReloadedUiAssetTests(unittest.TestCase):
    def test_official_ui_assets_are_read_from_disk_on_every_request(self) -> None:
        cases = (
            (
                "track_object_distance_human_ui.js",
                "/__official__/assets/track_object_distance_human_ui.js",
                "track-js-v1",
                "track-js-v2",
            ),
            (
                "track_object_distance_human_ui.css",
                "/__official__/assets/track_object_distance_human_ui.css",
                "track-css-v1",
                "track-css-v2",
            ),
            (
                "move_tracked_point_human_ui.js",
                "/__official__/assets/move_tracked_point_human_ui.js",
                "move-js-v1",
                "move-js-v2",
            ),
            (
                "move_tracked_point_human_ui.css",
                "/__official__/assets/move_tracked_point_human_ui.css",
                "move-css-v1",
                "move-css-v2",
            ),
        )
        for filename, url, first_source, second_source in cases:
            with self.subTest(asset=filename):
                app = Flask(f"{__name__}.{filename}")

                @app.get("/")
                def index():
                    return "<html><head></head><body></body></html>"

                self._assert_asset_refreshes(
                    app,
                    filename=filename,
                    url=url,
                    first_source=first_source,
                    second_source=second_source,
                )

    def _assert_asset_refreshes(
        self,
        app: Flask,
        *,
        filename: str,
        url: str,
        first_source: str,
        second_source: str,
    ) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            asset_dir = Path(directory)
            required = {
                "track_object_distance_human_ui.js": "track-v1",
                "track_object_distance_human_ui.css": "track-css-v1",
                "move_tracked_point_human_ui.js": "move-js-v1",
                "move_tracked_point_human_ui.css": "move-css-v1",
            }
            required[filename] = first_source
            for name, source in required.items():
                (asset_dir / name).write_text(source, encoding="utf-8")

            with mock.patch.object(human_ui, "_ASSET_DIR", asset_dir):
                human_ui.install_track_object_distance_human_ui(app)
                client = app.test_client()
                first = client.get(url)
                self.assertEqual(first.get_data(as_text=True), first_source)
                self.assertEqual(
                    first.headers.get("Cache-Control"),
                    "no-cache, no-store, must-revalidate",
                )

                (asset_dir / filename).write_text(second_source, encoding="utf-8")
                second = client.get(url)

            self.assertEqual(second.get_data(as_text=True), second_source)
            self.assertEqual(
                second.headers.get("Cache-Control"),
                "no-cache, no-store, must-revalidate",
            )


class SourceReloadPrimitiveTests(unittest.TestCase):
    def test_same_second_same_size_source_edit_cannot_execute_stale_pyc(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        module_name = f"official_reload_same_tick_{id(self)}"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / f"{module_name}.py"
            source.write_text("BUILD = 'v42'\n", encoding="utf-8")
            sys.path.insert(0, directory)
            try:
                module = importlib.import_module(module_name)
                original_stat = source.stat()
                self.assertEqual(module.BUILD, "v42")

                # Timestamp-based pyc validation records only whole seconds and
                # source size. This is the exact edit pattern which previously
                # left an old build running even though the source digest was new.
                source.write_text("BUILD = 'v43'\n", encoding="utf-8")
                os.utime(
                    source,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )
                importlib.invalidate_caches()

                coordinator._reload_one(module, importlib.reload)

                self.assertEqual(module.BUILD, "v43")
            finally:
                sys.modules.pop(module_name, None)
                sys.path.remove(directory)


def _reload_receipt(generation: int = 4) -> dict:
    return {
        "ok": True,
        "generation": int(generation),
        "rollback": False,
        "modules": [
            {"name": name, "old_file_digest": "old", "new_file_digest": "new"}
            for name in (
                "contract",
                "dynamic_point_tracker",
                "tracked_object_distance",
                "grasp_geometry_local",
                "grasp_kinematics_local",
                "eef_adjustment_local",
                "tracked_point_constraints_local",
                "tracked_point_motion_local",
                "tracked_point_execution_local",
                "capabilities",
                "dispatch",
                "tools",
                "types",
                "registry",
                "official_v2",
            )
        ],
        "builds": {
            "move_tracked_point": "move-build",
            "live_runner": "live-build",
            "contract": "contract-digest",
            "capabilities": "capability-digest",
            "tracker": "tracker-build",
            "dynamic_tracker": "dynamic-tracker-build",
            "ui": "ui-build",
        },
        "stack_digest": "stack-digest",
        "source_tree": {
            "root": "/unit/official_v2",
            "digest": "source-tree-digest",
            "files": [],
        },
        "asset_tree": {
            "root": "/unit/official_v2/assets",
            "digest": "asset-tree-digest",
            "files": [],
        },
        "trajectory": {
            "schema": "official_v2_move_tracked_point_trajectory",
            "version": 4,
        },
        "registry": {
            "public_tools": ["move_tracked_point"],
            "callback_ids_before": {"move_tracked_point": 10},
            "callback_ids_after": {"move_tracked_point": 11},
            "installed": True,
        },
        "state_preserved": {
            "episode_id": True,
            "observation_sequence": True,
            "policy_motion_epoch": True,
            "last_action": True,
            "tracked_names": True,
            "tracked_state_digest": True,
            "manager_identity": True,
            "capture_bindings": True,
            "task_memory_identity": True,
            "task_memory_digest": True,
            "plan_integrity_key": True,
            "session_lock": True,
            "replay_executor": True,
            "ik_workers": True,
            "ik_workers_lock": True,
        },
        "state_snapshot": {
            "episode_id": "episode-live-17",
            "observation_sequence": 177,
            "policy_motion_epoch": 23,
            "last_action": [],
            "manager_identity": 1001,
            "tracked_names": ["head", "tail"],
            "tracked_state_digest": "tracked-state-digest",
            "capture_bindings": 1,
            "task_memory_identity": 1002,
            "task_memory_digest": "task-memory-digest",
        },
        "ui": {
            "metadata_revision": "metadata-digest",
            "asset_revision": {
                "move_tracked_point_human_ui.js": "js-digest",
                "move_tracked_point_human_ui.css": "css-digest",
            },
        },
        "workers": {
            "ik_filter": {
                "source_digest": "worker-source-digest",
                "source_path": "/unit/ik_filter_worker.py",
                "previous_source_digest": "previous-worker-source-digest",
                "invalidated_count": 0,
                "close_errors": [],
            }
        },
    }


class TransactionalReloadRouteContractTests(unittest.TestCase):
    def _app(self, runtime):
        app = Flask(f"{__name__}.transactional.{id(runtime)}")
        install_live_unit_test_routes(app, runtime)
        return app

    @staticmethod
    def _runtime(receipt=None):
        current = dict(receipt or _reload_receipt())

        class Runtime:
            _live_test_module_name = (
                "behavior_interface_eval_test.live_move_tracked_point_test"
            )

            def __init__(self):
                self.reload_calls = 0
                self.status_calls = 0
                self.legacy_reload_calls = 0

            def reload_official_v2_tool_stack(self):
                self.reload_calls += 1
                return dict(current)

            def official_v2_reload_status(self):
                self.status_calls += 1
                return dict(current)

            def reload_live_test_module(self):
                self.legacy_reload_calls += 1
                raise AssertionError("route must use the transactional reload API")

            def start_live_move_tracked_point_test(self, args):
                return {"ok": True, "state": "planning", "args": args}

            def live_test_status(self):
                return {"ok": True, "state": "idle"}

            def cancel_live_test(self, reason):
                return {"ok": False, "state": "cancelled", "error": reason}

        return Runtime()

    def test_post_reload_uses_full_stack_transaction_and_returns_receipt(self) -> None:
        runtime = self._runtime()
        response = self._app(runtime).test_client().post(
            "/__official__/dev/live/reload"
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), _reload_receipt())
        self.assertEqual(runtime.reload_calls, 1)
        self.assertEqual(runtime.legacy_reload_calls, 0)

    def test_get_reload_status_is_non_mutating(self) -> None:
        runtime = self._runtime()
        client = self._app(runtime).test_client()

        first = client.get("/__official__/dev/live/reload/status")
        second = client.get("/__official__/dev/live/reload/status")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.get_json(), _reload_receipt())
        self.assertEqual(second.get_json(), first.get_json())
        self.assertEqual(runtime.status_calls, 2)
        self.assertEqual(runtime.reload_calls, 0)

    def test_reload_conflict_is_http_409_without_legacy_fallback(self) -> None:
        runtime = self._runtime()

        def conflict():
            runtime.reload_calls += 1
            raise RuntimeError("cannot reload while a normal skill is running")

        runtime.reload_official_v2_tool_stack = conflict
        response = self._app(runtime).test_client().post(
            "/__official__/dev/live/reload"
        )

        self.assertEqual(response.status_code, 409)
        self.assertFalse(response.get_json()["ok"])
        self.assertIn("normal skill", response.get_json()["error"])
        self.assertEqual(runtime.reload_calls, 1)
        self.assertEqual(runtime.legacy_reload_calls, 0)


class TransactionalRuntimeReloadContractTests(unittest.TestCase):
    REQUIRED_MODULE_BASENAMES = {
        "contract",
        "dynamic_point_tracker",
        "tracked_object_distance",
        "grasp_geometry_local",
        "grasp_kinematics_local",
        "navigation_map_bridge",
        "map_navigation_local",
        "navigation_footprint_local",
        "navigation_route_overlay",
        "eef_adjustment_local",
        "tracked_point_constraints_local",
        "tracked_point_motion_local",
        "tracked_point_execution_local",
        "capabilities",
        "dispatch",
        "tools",
        "types",
        "registry",
        "official_v2",
    }

    def _runtime(self):
        navigation_bridge_module = importlib.import_module(
            "behavior_interface_eval_test.navigation_map_bridge"
        )
        registry_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.registry"
        )
        task_memory_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.task_memory"
        )
        tracker_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.tracked_object_distance"
        )
        skills_module = SimpleNamespace()
        navigation_occupancy = np.asarray(
            [[-1, 0, 100], [0, 0, 100]], dtype=np.int8
        )
        navigation_occupancy.setflags(write=False)
        adapter = SimpleNamespace(
            observation_metadata=lambda: (177, 1234.5),
            _last_action=np.linspace(-0.2, 0.2, ACTION_DIM, dtype=np.float32),
            _lock=threading.RLock(),
            _navigation_map={"occupancy": navigation_occupancy},
        )
        tracker = tracker_module.TrackedObjectDistanceMemory(
            replay_max_frames=2,
            replay_max_bytes=1024 * 1024,
            max_capture_bindings=2,
        )
        height, width = 72, 96
        rgb = np.random.default_rng(8192).integers(
            0,
            256,
            size=(height, width, 3),
            dtype=np.uint8,
        )
        frame = tracker_module.TrackerFrame(
            rgb=rgb,
            depth_linear=np.full((height, width), 1.2, dtype=np.float32),
            intrinsics=tracker_module.CameraIntrinsics(
                width=width,
                height=height,
                fx=80.0,
                fy=80.0,
                cx=width * 0.5,
                cy=height * 0.5,
            ),
            camera_to_robot_base=np.eye(4, dtype=np.float64),
            camera_to_policy=np.eye(4, dtype=np.float64),
            sequence=177,
            timestamp_s=1234.5,
            episode_id="episode-live-17",
            proprio=np.zeros(PROPRIO_DIM, dtype=np.float64),
        )
        tracker.ingest(frame)
        binding = tracker.register_capture(
            session_id="session-live",
            image_id="img-live",
            observation_sequence=177,
            episode_id="episode-live-17",
            image_shape=(height, width),
        )
        tracker.replace_points(
            [
                {"name": "head", "u": 350.0, "v": 500.0},
                {"name": "tail", "u": 650.0, "v": 500.0},
            ],
            session_id=binding["session_id"],
            image_id=binding["image_id"],
            capture_observation_sequence=binding["observation_sequence"],
            capture_episode_id=binding["episode_id"],
            capture_image_shape=tuple(binding["image_shape"]),
            capture_frame_digest=binding["frame_digest"],
        )
        world = SimpleNamespace(
            episode_id=lambda: "episode-live-17",
            motion_epoch=lambda: 23,
        )

        runtime = object.__new__(OfficialPolicyRuntime)
        runtime._official_reload_lock = threading.RLock()
        runtime._official_reload_generation = 0
        runtime._official_reload_receipt = None
        runtime._official_skills_module = skills_module
        runtime._live_test_lock = threading.RLock()
        runtime._live_test_job = None
        runtime._live_test_module_name = (
            "behavior_interface_eval_test.live_move_tracked_point_test"
        )
        runtime._live_test_module_build = None
        runtime._step_lock = threading.RLock()
        runtime.adapter = adapter
        runtime._navigation_map_bridge = (
            navigation_bridge_module.NavigationMapBridge()
        )
        runtime._navigation_map_bridge._cache_key = ("unit-test", 177)
        runtime._navigation_map_bridge._cached_snapshot = adapter._navigation_map
        provider_occupancy = np.asarray(
            [[100, 0], [-1, 0]], dtype=np.int8
        )
        provider_occupancy.setflags(write=False)
        runtime._spatial_live_mapper = SimpleNamespace(
            _state_lock=threading.RLock(),
            _latest=SimpleNamespace(
                frame_id=91,
                map_updated=True,
                node_count=12,
                loop_count=2,
                current_pose=SimpleNamespace(x_m=1.25, y_m=-0.5, yaw_rad=0.3),
                occupancy=provider_occupancy,
                low_obstacles=provider_occupancy,
                high_obstacles=np.zeros_like(provider_occupancy),
            ),
        )
        runtime.tracked_object_distances = tracker
        runtime.task_memory = task_memory_module.OfficialTaskMemory(
            task="chopping_wood",
            task_index=17,
            display_name="Chopping Wood",
            task_objective="Split the wood with the axe.",
            instruction="Use only the public task definition.",
            bddl_conditions=("condition-a", "condition-b"),
            instruction_source="unit-test",
        )
        runtime.episode_initializer = SimpleNamespace(initialized=True)
        runtime.server = SimpleNamespace(
            skill_lock=threading.RLock(),
            current_job=None,
            skill_queue=queue.Queue(),
            world=world,
        )
        task_memory_module.bind_official_task_memory(
            runtime.server,
            runtime.task_memory,
            dynamic_memory=runtime.tracked_object_distances,
        )
        runtime.tool_registry = registry_module.install_profile(
            skills_module,
            adapter,
        )

        spatial_map = importlib.import_module("behavior_interface.spatial_map")
        map_key = f"hot-reload-map-{id(runtime)}"
        ego = spatial_map.EgoMap(
            session_id=map_key,
            grid=spatial_map.OccupancyGrid(
                resolution_m=0.1,
                half_span_m=0.2,
            ),
        )
        ego.initialized = True
        ego.x = 1.5
        ego.y = -0.25
        ego.yaw_deg = 17.0
        ego.grid.low[1, 2] = np.float32(2.75)
        ego.grid.frames = 3
        with spatial_map._MAPS_LOCK:
            spatial_map._MAPS[map_key] = ego
        runtime._test_egomap_key = map_key
        runtime._test_egomap = ego

        def remove_test_map() -> None:
            with spatial_map._MAPS_LOCK:
                if spatial_map._MAPS.get(map_key) is ego:
                    spatial_map._MAPS.pop(map_key, None)

        self.addCleanup(remove_test_map)
        return runtime

    @staticmethod
    def _public_tracker_snapshot(runtime) -> dict:
        manager = runtime.tracked_object_distances
        return {
            "status": deepcopy(manager.status()),
            "observed": deepcopy(
                manager.observed_active_points_snapshot(
                    ["head", "tail"],
                    episode_id="episode-live-17",
                )
            ),
            "memory": deepcopy(manager.memory_fields()),
        }

    def test_policy_official_v2_runtime_imports_have_reload_declarations(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        policy = importlib.import_module(
            "behavior_interface_eval_test.official_policy_interface"
        )
        tree = ast.parse(Path(policy.__file__).read_text(encoding="utf-8"))
        package = "behavior_interface_eval_test.tool.official_v2"
        bridge_module = "behavior_interface_eval_test.navigation_map_bridge"
        runtime_imports: set[str] = set()
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if (
                node.module != bridge_module
                and node.module != package
                and not node.module.startswith(package + ".")
            ):
                continue
            runtime_imports.update(alias.asname or alias.name for alias in node.names)

        expected_startup_only = frozenset(
            {
                "install_official_task_memory",
                "install_track_object_distance_human_ui",
            }
        )
        startup_only = frozenset(
            getattr(coordinator, "_POLICY_STARTUP_ONLY_IMPORTS", ())
        )
        self.assertEqual(startup_only, expected_startup_only)
        declared = set(coordinator._POLICY_REBINDS) | set(startup_only)
        self.assertEqual(
            runtime_imports,
            declared,
            "every official_v2 symbol bound by the running policy module must "
            "either be rebound transactionally or be explicitly startup-only",
        )

    def test_reload_status_schema_is_stable_and_read_only(self) -> None:
        runtime = self._runtime()
        manager = runtime.tracked_object_distances
        task_memory = runtime.task_memory
        manager_state = self._public_tracker_snapshot(runtime)
        task_memory_state = deepcopy(task_memory.raw())
        generation = runtime._official_reload_generation
        stored_receipt = runtime._official_reload_receipt

        initial = runtime.official_v2_reload_status()
        repeated = runtime.official_v2_reload_status()

        self.assertEqual(repeated, initial)
        self.assertEqual(runtime._official_reload_generation, generation)
        self.assertIs(runtime._official_reload_receipt, stored_receipt)
        self.assertIs(runtime.tracked_object_distances, manager)
        self.assertIs(runtime.task_memory, task_memory)
        self.assertEqual(self._public_tracker_snapshot(runtime), manager_state)
        self.assertEqual(runtime.task_memory.raw(), task_memory_state)
        self.assertTrue(all(type(value) is bool for value in initial["state_preserved"].values()))
        self.assertEqual(
            set(initial["state_snapshot"]),
            {
                "episode_id",
                "observation_sequence",
                "policy_motion_epoch",
                "last_action",
                "manager_identity",
                "tracked_names",
                "tracked_state_digest",
                "capture_bindings",
                "task_memory_identity",
                "task_memory_digest",
                "navigation_bridge_identity",
                "navigation_provider_identity",
                "navigation_provider_latest_identity",
                "navigation_provider_content_digest",
                "navigation_adapter_map_identity",
                "navigation_adapter_occupancy_identity",
                "navigation_adapter_occupancy_digest",
                "navigation_bridge_snapshot_identity",
                "navigation_bridge_occupancy_identity",
                "navigation_bridge_occupancy_digest",
                "navigation_egomaps_container_identity",
                "navigation_egomap_identities",
                "navigation_egomap_content_digest",
            },
        )
        self.assertTrue(
            set(initial["state_snapshot"]).issubset(initial["state_preserved"])
        )
        self.assertEqual(
            set(initial["workers"]["ik_filter"]),
            {
                "source_digest",
                "source_path",
                "previous_source_digest",
                "invalidated_count",
                "close_errors",
            },
        )

        committed = runtime.reload_official_v2_tool_stack()
        committed_status = runtime.official_v2_reload_status()
        self.assertEqual(committed_status, committed)
        self.assertEqual(set(committed_status), set(initial))
        for mapping_name in (
            "source_tree",
            "asset_tree",
            "registry",
            "state_snapshot",
            "navigation",
            "ui",
        ):
            self.assertEqual(
                set(committed_status[mapping_name]),
                set(initial[mapping_name]),
                mapping_name,
            )
        self.assertEqual(
            set(committed_status["workers"]["ik_filter"]),
            set(initial["workers"]["ik_filter"]),
        )
        self.assertEqual(
            set(committed_status["workers"]),
            set(initial["workers"]),
        )
        self.assertTrue(
            all(
                type(value) is bool
                for value in committed_status["state_preserved"].values()
            )
        )

        committed_status["state_snapshot"]["tracked_names"].append("mutated")
        self.assertNotIn(
            "mutated",
            runtime.official_v2_reload_status()["state_snapshot"]["tracked_names"],
        )

    def test_committed_generation_live_start_reuses_harness_without_reload(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        live_module = importlib.import_module(
            "behavior_interface_eval_test.live_move_tracked_point_test"
        )
        runtime = self._runtime()
        receipt = runtime.reload_official_v2_tool_stack()
        runtime.evaluator_ready = lambda: True
        runtime.server.world.episode_initialized = lambda: True
        runtime.server.log = mock.Mock()
        providers = object()
        runtime._live_test_providers = lambda: providers
        created: list[object] = []

        class Job:
            job_id = "committed-generation-job"

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(self)

            def start(self):
                return {"ok": True, "state": "planning", "job_id": self.job_id}

        with mock.patch.object(live_module, "LiveMoveTrackedPointJob", Job), mock.patch.object(
            coordinator,
            "_reload_one",
            side_effect=AssertionError("per-case start attempted a module reload"),
        ) as reload_one:
            result = runtime.start_live_move_tracked_point_test(
                {"points": [{"name": "head", "target_xyz_m": ["?", "?", "?"]}]}
            )

        self.assertTrue(result["ok"])
        self.assertEqual(len(created), 1)
        self.assertIs(created[0].kwargs["providers"], providers)
        self.assertEqual(
            created[0].kwargs["module_build"],
            receipt["builds"]["live_runner"],
        )
        self.assertEqual(runtime._official_reload_generation, receipt["generation"])
        reload_one.assert_not_called()

    def test_changed_harness_source_rejects_live_start_before_job_creation(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        live_module = importlib.import_module(
            "behavior_interface_eval_test.live_move_tracked_point_test"
        )
        runtime = self._runtime()
        runtime.reload_official_v2_tool_stack()
        runtime.evaluator_ready = lambda: True
        runtime.server.world.episode_initialized = lambda: True
        runtime.server.log = mock.Mock()
        runtime._live_test_providers = lambda: object()
        committed_status = runtime.official_v2_reload_status()
        tracker_state = self._public_tracker_snapshot(runtime)
        task_memory_state = deepcopy(runtime.task_memory.raw())
        original_source_record = coordinator._source_record
        created = []

        class Job:
            def __init__(self, **_kwargs):
                created.append(self)

        def changed_live_source(module):
            record = original_source_record(module)
            if module is live_module:
                record = dict(record)
                record["digest"] = "uncommitted-live-source-digest"
            return record

        with mock.patch.object(live_module, "LiveMoveTrackedPointJob", Job), mock.patch.object(
            coordinator,
            "_source_record",
            side_effect=changed_live_source,
        ), mock.patch.object(
            coordinator,
            "_reload_one",
            side_effect=AssertionError("mismatched source must not be reloaded per case"),
        ) as reload_one:
            with self.assertRaisesRegex(
                RuntimeError,
                "source differs from the committed official-v2 generation",
            ):
                runtime.start_live_move_tracked_point_test(
                    {
                        "points": [
                            {
                                "name": "head",
                                "target_xyz_m": ["?", "?", "?"],
                            }
                        ]
                    }
                )

        self.assertEqual(created, [])
        self.assertIsNone(runtime._live_test_job)
        self.assertEqual(runtime.official_v2_reload_status(), committed_status)
        self.assertEqual(self._public_tracker_snapshot(runtime), tracker_state)
        self.assertEqual(runtime.task_memory.raw(), task_memory_state)
        reload_one.assert_not_called()

    def test_transaction_rebinds_policy_visualization_consumer(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        policy = importlib.import_module(
            "behavior_interface_eval_test.official_policy_interface"
        )
        visualization = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.visualization_local"
        )
        runtime = self._runtime()
        original_policy_renderer = policy.render_head_path_overlay_frame
        original_module_renderer = visualization.render_head_path_overlay_frame

        def reloaded_renderer(*_args, **_kwargs):
            return "reload-render-sentinel", {"generation": "fresh"}

        try:
            visualization.render_head_path_overlay_frame = reloaded_renderer
            receipt = coordinator.reload_official_v2_tool_stack(
                runtime,
                import_module=importlib.import_module,
                reload_module=lambda module: module,
                invalidate_caches=lambda: None,
            )
            self.assertTrue(receipt["ok"])
            self.assertIs(
                policy.render_head_path_overlay_frame,
                reloaded_renderer,
                "the already-running policy must not retain the old renderer",
            )
            self.assertEqual(
                policy.render_head_path_overlay_frame(),
                ("reload-render-sentinel", {"generation": "fresh"}),
            )
        finally:
            visualization.render_head_path_overlay_frame = original_module_renderer
            policy.render_head_path_overlay_frame = original_policy_renderer

    def test_bootstrap_can_patch_an_already_instantiated_runtime_class(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        policy = importlib.import_module(
            "behavior_interface_eval_test.official_policy_interface"
        )

        class LegacyRuntime:
            def __init__(self):
                self._live_test_lock = threading.RLock()
                self._live_test_job = None
                self._live_test_module_name = (
                    "behavior_interface_eval_test.live_move_tracked_point_test"
                )
                self.legacy_reload_calls = 0
                self.server = SimpleNamespace()
                self.tool_registry = {}

            def reload_live_test_module(self):
                # This is the only code path available to a process which was
                # started before the transaction endpoint existed. The method
                # deliberately knows nothing about the coordinator: reloading
                # the updated live harness must install the compatibility shim
                # through that module's import side effect.
                self.legacy_reload_calls += 1
                importlib.invalidate_caches()
                module = importlib.import_module(self._live_test_module_name)
                module = importlib.reload(module)
                return module

        runtime = LegacyRuntime()
        legacy_function = LegacyRuntime.reload_live_test_module
        app = Flask(f"{__name__}.legacy-bootstrap")

        @app.post("/__official__/dev/live/reload", endpoint="dev_live_reload")
        def legacy_reload_route():
            module = runtime.reload_live_test_module()
            return {"ok": True, "build": str(module.BUILD)}

        old_view = app.view_functions["dev_live_reload"]
        client = app.test_client()
        receipt = _reload_receipt(generation=1)
        with mock.patch.object(policy, "OfficialPolicyRuntime", LegacyRuntime), mock.patch.object(
            coordinator,
            "reload_official_v2_tool_stack",
            return_value=receipt,
        ) as transactional_reload:
            bootstrap_response = client.post("/__official__/dev/live/reload")
            transaction_response = client.post("/__official__/dev/live/reload")

        self.assertEqual(bootstrap_response.status_code, 200)
        self.assertEqual(
            bootstrap_response.get_json(),
            {
                "ok": True,
                "build": str(
                    importlib.import_module(
                        "behavior_interface_eval_test.live_move_tracked_point_test"
                    ).BUILD
                ),
            },
        )
        self.assertEqual(transaction_response.status_code, 200)
        self.assertEqual(transaction_response.get_json(), receipt)
        self.assertEqual(runtime.legacy_reload_calls, 1)
        transactional_reload.assert_called_once()
        self.assertIs(transactional_reload.call_args.args[0], runtime)
        self.assertIsNot(LegacyRuntime.reload_live_test_module, legacy_function)
        self.assertTrue(callable(runtime.reload_official_v2_tool_stack))
        self.assertTrue(callable(runtime.official_v2_reload_status))
        self.assertIsNot(app.view_functions["dev_live_reload"], old_view)

    def test_idle_reload_swaps_all_formal_callbacks_and_preserves_state(self) -> None:
        runtime = self._runtime()
        loaded_official_modules = {
            name
            for name, module in tuple(sys.modules.items())
            if (
                name == "behavior_interface_eval_test.tool.official_v2"
                or name.startswith(
                    "behavior_interface_eval_test.tool.official_v2."
                )
            )
            and str(getattr(module, "__file__", "")).endswith(".py")
            and name
            != "behavior_interface_eval_test.tool.official_v2.ik_filter_worker"
        }
        loaded_official_modules.add(
            "behavior_interface_eval_test.live_move_tracked_point_test"
        )
        old_registry = runtime.tool_registry
        old_callbacks = {
            name: spec.fn for name, spec in old_registry.items()
        }
        old_state_objects = (
            runtime.adapter,
            runtime.server.world,
            runtime.tracked_object_distances,
            runtime.task_memory,
            runtime.episode_initializer,
        )
        old_task_memory_class = runtime.task_memory.__class__
        old_task_memory_fields = deepcopy(runtime.task_memory.raw())
        old_memory_callbacks = (
            runtime.server.get_memory,
            runtime.server.get_memory_text,
            runtime.server.get_memory_summary,
        )
        old_memory_callback_results = (
            deepcopy(runtime.server.get_memory()),
            runtime.server.get_memory_text(),
            deepcopy(runtime.server.get_memory_summary()),
        )
        old_inner_tracker = runtime.tracked_object_distances._tracker
        old_track_states = dict(old_inner_tracker._tracks)
        old_inner_tracker_state = (
            old_inner_tracker._episode_id,
            old_inner_tracker._next_track_number,
        )
        old_capture_bindings = tuple(
            runtime.tracked_object_distances._capture_bindings.items()
        )
        old_registration_binding = (
            runtime.tracked_object_distances._registration_binding
        )
        tools_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.tools"
        )
        bridge_module = importlib.import_module(
            "behavior_interface_eval_test.navigation_map_bridge"
        )
        old_bridge = runtime._navigation_map_bridge
        old_bridge_class = old_bridge.__class__
        old_bridge_snapshot = old_bridge._cached_snapshot
        old_bridge_occupancy = old_bridge_snapshot["occupancy"]
        planner_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.map_navigation_local"
        )
        footprint_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2."
            "navigation_footprint_local"
        )
        route_overlay_module = importlib.import_module(
            "behavior_interface_eval_test.navigation_route_overlay"
        )
        old_planner_callback = planner_module.plan_clearance_path
        old_footprint_callback = (
            footprint_module.base_navigation_footprint_envelope
        )
        old_route_lock = route_overlay_module._LOCK
        old_routes = route_overlay_module._ROUTES
        route_session = f"hot-reload-route-{id(runtime)}"
        route_overlay_module.publish_route(
            route_session,
            "plan-before-reload",
            ((0.0, 0.0), (1.0, 0.5)),
        )
        self.addCleanup(route_overlay_module.clear_route, route_session)
        navigation_state_before = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )._navigation_state_identity(runtime)
        tracker_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.tracked_object_distance"
        )
        kinematics_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local"
        )
        old_plan_integrity_key = tools_module._PLAN_INTEGRITY_KEY
        old_session_lock = tools_module._SESSION_LOCK
        old_replay_executor = tracker_module._REPLAY_COMPRESSION_EXECUTOR
        old_ik_workers = kinematics_module._PERSISTENT_IK_WORKERS
        old_ik_workers_lock = kinematics_module._PERSISTENT_IK_WORKERS_LOCK
        fake_worker = SimpleNamespace(close=mock.Mock())
        fake_worker_key = ("unit-hot-reload-gpu", "left")
        old_ik_workers[fake_worker_key] = fake_worker
        self.addCleanup(old_ik_workers.pop, fake_worker_key, None)
        old_last_action = runtime.adapter._last_action.copy()
        old_public_tracker_state = self._public_tracker_snapshot(runtime)

        receipt = runtime.reload_official_v2_tool_stack()

        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["generation"], 1)
        self.assertFalse(receipt["rollback"])
        module_basenames = {
            str(item["name"]).rsplit(".", 1)[-1]
            for item in receipt["modules"]
        }
        self.assertTrue(
            self.REQUIRED_MODULE_BASENAMES.issubset(module_basenames),
            module_basenames,
        )
        receipt_module_names = {str(item["name"]) for item in receipt["modules"]}
        self.assertTrue(
            loaded_official_modules.issubset(receipt_module_names),
            sorted(loaded_official_modules - receipt_module_names),
        )
        self.assertIsNot(runtime.tool_registry, old_registry)
        self.assertIs(
            runtime._official_skills_module.SKILL_REGISTRY,
            runtime.tool_registry,
        )
        capabilities_after_reload = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.capabilities"
        )
        self.assertEqual(
            tuple(runtime.tool_registry),
            tuple(capabilities_after_reload.PUBLIC_TOOLS),
        )
        for name in set(runtime.tool_registry).intersection(old_callbacks):
            spec = runtime.tool_registry[name]
            self.assertIsNot(spec.fn, old_callbacks[name], name)
        self.assertIs(runtime.adapter, old_state_objects[0])
        self.assertIs(runtime.server.world, old_state_objects[1])
        self.assertIs(runtime.tracked_object_distances, old_state_objects[2])
        self.assertIs(runtime.task_memory, old_state_objects[3])
        self.assertIs(runtime.episode_initializer, old_state_objects[4])
        self.assertIs(runtime._navigation_map_bridge, old_bridge)
        self.assertIsNot(runtime._navigation_map_bridge.__class__, old_bridge_class)
        self.assertIs(
            runtime._navigation_map_bridge.__class__,
            bridge_module.NavigationMapBridge,
        )
        self.assertIs(runtime._navigation_map_bridge._cached_snapshot, old_bridge_snapshot)
        self.assertIs(
            runtime._navigation_map_bridge._cached_snapshot["occupancy"],
            old_bridge_occupancy,
        )
        task_memory_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.task_memory"
        )
        self.assertIs(
            runtime.task_memory.__class__,
            task_memory_module.OfficialTaskMemory,
            "an existing task-memory instance must execute the reloaded class",
        )
        self.assertIsNot(runtime.task_memory.__class__, old_task_memory_class)
        self.assertEqual(runtime.task_memory.raw(), old_task_memory_fields)
        new_memory_callbacks = (
            runtime.server.get_memory,
            runtime.server.get_memory_text,
            runtime.server.get_memory_summary,
        )
        for old_callback, new_callback in zip(
            old_memory_callbacks, new_memory_callbacks
        ):
            self.assertIsNot(old_callback, new_callback)
        self.assertEqual(
            (
                runtime.server.get_memory(),
                runtime.server.get_memory_text(),
                runtime.server.get_memory_summary(),
            ),
            old_memory_callback_results,
        )
        self.assertEqual(runtime.server.world.episode_id(), "episode-live-17")
        self.assertEqual(runtime.server.world.motion_epoch(), 23)
        self.assertEqual(runtime.adapter.observation_metadata()[0], 177)
        np.testing.assert_array_equal(runtime.adapter._last_action, old_last_action)
        self.assertEqual(
            list(runtime.tracked_object_distances._entries),
            ["head", "tail"],
        )
        self.assertIs(runtime.tracked_object_distances._tracker, old_inner_tracker)
        self.assertEqual(
            tuple(runtime.tracked_object_distances._capture_bindings.items()),
            old_capture_bindings,
        )
        self.assertIs(
            runtime.tracked_object_distances._registration_binding,
            old_registration_binding,
        )
        self.assertEqual(
            (
                old_inner_tracker._episode_id,
                old_inner_tracker._next_track_number,
            ),
            old_inner_tracker_state,
        )
        self.assertEqual(old_inner_tracker._tracks, old_track_states)
        self.assertEqual(
            self._public_tracker_snapshot(runtime), old_public_tracker_state
        )
        self.assertEqual(
            runtime.tracked_object_distances.status()["registered_names"],
            ["head", "tail"],
        )
        tracker_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.tracked_object_distance"
        )
        dynamic_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.dynamic_point_tracker"
        )
        kinematics_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.grasp_kinematics_local"
        )
        tools_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.tools"
        )
        planner_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.map_navigation_local"
        )
        footprint_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2."
            "navigation_footprint_local"
        )
        route_overlay_module = importlib.import_module(
            "behavior_interface_eval_test.navigation_route_overlay"
        )
        self.assertIs(
            runtime.tracked_object_distances.__class__,
            tracker_module.TrackedObjectDistanceMemory,
        )
        self.assertIs(old_inner_tracker.__class__, dynamic_module.DynamicPointTracker)
        self.assertIs(tools_module._PLAN_INTEGRITY_KEY, old_plan_integrity_key)
        self.assertIs(tools_module._SESSION_LOCK, old_session_lock)
        self.assertIsNot(planner_module.plan_clearance_path, old_planner_callback)
        self.assertIsNot(
            footprint_module.base_navigation_footprint_envelope,
            old_footprint_callback,
        )
        self.assertIs(
            tools_module.plan_clearance_path,
            planner_module.plan_clearance_path,
        )
        self.assertIs(
            tools_module.base_navigation_footprint_envelope,
            footprint_module.base_navigation_footprint_envelope,
        )
        self.assertIs(route_overlay_module._LOCK, old_route_lock)
        self.assertIs(route_overlay_module._ROUTES, old_routes)
        route_snapshot = route_overlay_module.get_route_snapshot(route_session)
        self.assertIsNotNone(route_snapshot)
        self.assertEqual(
            route_snapshot.points_xy_m,
            ((0.0, 0.0), (1.0, 0.5)),
        )
        self.assertEqual(
            importlib.import_module(
                "behavior_interface_eval_test.official_v2_hot_reload"
            )._navigation_state_identity(runtime),
            navigation_state_before,
        )
        self.assertIs(
            tracker_module._REPLAY_COMPRESSION_EXECUTOR,
            old_replay_executor,
        )
        self.assertIs(kinematics_module._PERSISTENT_IK_WORKERS, old_ik_workers)
        self.assertIs(
            kinematics_module._PERSISTENT_IK_WORKERS_LOCK,
            old_ik_workers_lock,
        )
        fake_worker.close.assert_called_once_with()
        self.assertNotIn(fake_worker_key, old_ik_workers)
        self.assertEqual(
            receipt["workers"]["ik_filter"]["invalidated_count"], 1
        )
        self.assertTrue(receipt["workers"]["ik_filter"]["source_digest"])
        self.assertEqual(receipt["workers"]["ik_filter"]["close_errors"], [])

        contract_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.contract"
        )
        constraints_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2."
            "tracked_point_constraints_local"
        )
        motion_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.tracked_point_motion_local"
        )
        execution_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2."
            "tracked_point_execution_local"
        )
        capabilities_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.capabilities"
        )
        dispatch_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.dispatch"
        )
        registry_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.registry"
        )
        facade_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2"
        )
        live_module = importlib.import_module(
            "behavior_interface_eval_test.live_move_tracked_point_test"
        )
        policy_module = importlib.import_module(
            "behavior_interface_eval_test.official_policy_interface"
        )

        self.assertIs(
            capabilities_module.validate_move_tracked_point_args,
            contract_module.validate_move_tracked_point_args,
        )
        self.assertIs(
            tools_module._evaluate_quick_constraint,
            constraints_module.evaluate_quick_constraint,
        )
        self.assertIs(
            tools_module._plan_tracked_endpoint,
            motion_module.plan_endpoint,
        )
        self.assertIs(
            tools_module._schedule_tracked_execution_waypoints,
            execution_module.schedule_execution_waypoints,
        )
        self.assertIs(
            registry_module.PUBLIC_TOOL_FUNCTIONS,
            tools_module.PUBLIC_TOOL_FUNCTIONS,
        )
        self.assertIs(
            facade_module.validate_submission,
            capabilities_module.validate_submission,
        )
        self.assertIs(
            facade_module.translate_submission,
            dispatch_module.translate_submission,
        )
        self.assertIs(
            facade_module.evaluate_quick_constraint,
            constraints_module.evaluate_quick_constraint,
        )
        self.assertIs(
            policy_module.validate_submission,
            capabilities_module.validate_submission,
        )
        self.assertIs(
            policy_module.translate_submission,
            dispatch_module.translate_submission,
        )
        self.assertIs(
            policy_module.validate_move_tracked_point_args,
            contract_module.validate_move_tracked_point_args,
        )
        self.assertIs(
            policy_module.validate_navigate_to_args,
            contract_module.validate_navigate_to_args,
        )
        reloaded_bridge_module = importlib.import_module(
            "behavior_interface_eval_test.navigation_map_bridge"
        )
        for bridge_name in (
            "NAVIGATION_MAP_SCHEMA",
            "NAVIGATION_MAP_SCHEMA_VERSION",
            "NavigationMapBridge",
            "copy_navigation_map_snapshot",
            "normalize_navigation_pose_freshness",
            "normalize_navigation_pose_source",
            "view_navigation_map_snapshot",
        ):
            self.assertIs(
                getattr(policy_module, bridge_name),
                getattr(reloaded_bridge_module, bridge_name),
                bridge_name,
            )
        self.assertIs(live_module.plan_endpoint, motion_module.plan_endpoint)
        self.assertIs(
            live_module.schedule_execution_waypoints,
            execution_module.schedule_execution_waypoints,
        )
        self.assertIs(
            live_module.evaluate_quick_constraint,
            constraints_module.evaluate_quick_constraint,
        )
        self.assertTrue(receipt["registry"]["installed"])
        self.assertIs(
            runtime.tool_registry["navigate_to"].fn,
            tools_module.navigate_to,
        )
        self.assertNotEqual(
            receipt["navigation"]["callback_id_before"],
            receipt["navigation"]["callback_id_after"],
        )
        self.assertTrue(receipt["navigation"]["callback_replaced"])
        self.assertTrue(receipt["navigation"]["dependencies_before_executor"])
        self.assertTrue(all(receipt["navigation"]["bindings_current"].values()))
        self.assertEqual(
            set(receipt["navigation"]["source_digests"]),
            {
                "contract",
                "map_bridge",
                "planner",
                "footprint",
                "executor",
                "route_overlay",
            },
        )
        self.assertEqual(
            receipt["navigation"]["bridge"],
            {
                "instance_id": id(old_bridge),
                "class_id_before": id(old_bridge_class),
                "class_id_after": id(bridge_module.NavigationMapBridge),
                "class_rebound": True,
                "class_current": True,
            },
        )
        self.assertTrue(
            all(
                receipt["navigation"]["source_changed_since_previous_commit"].values()
            )
        )
        self.assertTrue(
            receipt["navigation"]["renderer_hooks"][
                "state_identity_preserved"
            ]
        )
        self.assertNotEqual(
            receipt["registry"]["callback_ids_before"],
            receipt["registry"]["callback_ids_after"],
        )
        self.assertTrue(receipt["builds"]["move_tracked_point"])
        self.assertTrue(receipt["stack_digest"])
        self.assertTrue(receipt["trajectory"]["schema"])
        self.assertGreaterEqual(int(receipt["trajectory"]["version"]), 1)
        self.assertTrue(receipt["ui"]["metadata_revision"])
        self.assertIn(
            "move_tracked_point_human_ui.js",
            receipt["ui"]["asset_revision"],
        )
        self.assertIn(
            "move_tracked_point_human_ui.css",
            receipt["ui"]["asset_revision"],
        )
        for state_key in (
            "episode_id",
            "observation_sequence",
            "policy_motion_epoch",
            "last_action",
            "tracked_names",
            "manager_identity",
            "capture_bindings",
            "plan_integrity_key",
            "session_lock",
            "replay_executor",
            "navigation_bridge_identity",
            "navigation_provider_identity",
            "navigation_adapter_occupancy_identity",
            "navigation_adapter_occupancy_digest",
            "navigation_egomap_identities",
            "navigation_egomap_content_digest",
        ):
            self.assertTrue(receipt["state_preserved"][state_key], state_key)

        status = runtime.official_v2_reload_status()
        self.assertEqual(status["generation"], receipt["generation"])
        self.assertEqual(status["builds"], receipt["builds"])
        self.assertEqual(status["registry"]["callback_ids_after"], receipt["registry"]["callback_ids_after"])

    def test_reload_drains_background_tasks_and_retires_audit_fds(self) -> None:
        runtime = self._runtime()
        lite = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.rgbd_grasp_lite"
        )
        audit = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2."
            "grasp_prediction_audit_local"
        )
        old_lite_executor = lite._LITE_RENDER_EXECUTOR
        old_lite_futures = lite._LITE_RENDER_FUTURES
        old_lite_lock = lite._LITE_RENDER_EXECUTOR_LOCK
        old_writers = audit._writers
        old_writers_lock = audit._writers_lock

        render_future = mock.Mock()
        render_future.result.return_value = b"rendered"
        old_lite_futures.append(render_future)
        self.addCleanup(
            lambda: old_lite_futures.remove(render_future)
            if render_future in old_lite_futures
            else None
        )

        fd, path = tempfile.mkstemp(prefix="official-v2-hot-reload-audit-")
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        def close_fd_if_open() -> None:
            try:
                os.close(fd)
            except OSError:
                pass

        self.addCleanup(close_fd_if_open)
        legacy_writer = SimpleNamespace(_fd=fd, _lock=threading.Lock())
        old_writers[path] = legacy_writer
        self.addCleanup(old_writers.pop, path, None)

        receipt = runtime.reload_official_v2_tool_stack()

        render_future.result.assert_called_once()
        self.assertIs(lite._LITE_RENDER_EXECUTOR, old_lite_executor)
        self.assertIs(lite._LITE_RENDER_FUTURES, old_lite_futures)
        self.assertIs(lite._LITE_RENDER_EXECUTOR_LOCK, old_lite_lock)
        self.assertNotIn(render_future, old_lite_futures)
        self.assertEqual(
            receipt["workers"]["lite_render"],
            {"drained_count": 1, "pending_count_after": 0},
        )
        self.assertIs(audit._writers, old_writers)
        self.assertIs(audit._writers_lock, old_writers_lock)
        self.assertNotIn(path, old_writers)
        self.assertIsNone(legacy_writer._fd)
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.assertEqual(
            receipt["workers"]["prediction_audit"],
            {"closed_count": 1, "close_errors": []},
        )

    def test_replay_compression_is_drained_before_module_mutation(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        replay_future = mock.Mock()
        replay_future.result.return_value = "stored-frame"
        pending = SimpleNamespace(future=replay_future)
        manager = SimpleNamespace(_pending_replay=[pending])

        def drain_ready() -> None:
            manager._pending_replay.clear()

        manager._drain_ready_replay_locked = drain_ready
        report = coordinator._drain_background_tasks(
            SimpleNamespace(tracked_object_distances=manager),
            {},
        )

        replay_future.result.assert_called_once()
        self.assertEqual(
            report["replay_compression"],
            {"drained_count": 1, "pending_count_after": 0},
        )

    def test_background_failure_rejects_before_generation_mutation(self) -> None:
        runtime = self._runtime()
        lite = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.rgbd_grasp_lite"
        )
        old_registry = runtime.tool_registry
        old_status = runtime.official_v2_reload_status()
        failed_future = mock.Mock()
        failed_future.result.side_effect = RuntimeError("render generation failed")
        lite._LITE_RENDER_FUTURES.append(failed_future)
        self.addCleanup(
            lambda: lite._LITE_RENDER_FUTURES.remove(failed_future)
            if failed_future in lite._LITE_RENDER_FUTURES
            else None
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "RGB-D lite rendering failed before hot reload",
        ):
            runtime.reload_official_v2_tool_stack()

        self.assertIs(runtime.tool_registry, old_registry)
        self.assertEqual(runtime.official_v2_reload_status(), old_status)
        self.assertIn(failed_future, lite._LITE_RENDER_FUTURES)

    def test_legacy_tracker_fields_are_migrated_without_losing_public_state(self) -> None:
        runtime = self._runtime()
        manager = runtime.tracked_object_distances
        inner = manager._tracker
        track_state = next(iter(inner._tracks.values()))
        before = self._public_tracker_snapshot(runtime)

        del track_state.last_observation_rejections
        del track_state.pending_component_depth_roi
        del track_state.pending_component_origin_px
        del track_state.pending_component_anchor_px
        del track_state.pending_component_depth_m
        del inner._active_rigid_pair
        del inner._last_rigid_pair_report
        del inner._opencl_lk_state
        del inner._opencl_lk_device
        del inner._opencl_lk_disable_reason
        del inner._opencl_lk_validation_remaining
        del inner._opencl_lk_validation_passes
        del inner._opencl_lk_gpu_calls
        del inner._opencl_lk_cpu_calls
        del inner._opencl_umat_cache
        del manager._motion_retention_counter

        receipt = runtime.reload_official_v2_tool_stack()

        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["generation"], 1)
        self.assertEqual(track_state.last_observation_rejections, ())
        self.assertNotIn("pending_component_depth_roi", track_state.__dict__)
        self.assertNotIn("pending_component_origin_px", track_state.__dict__)
        self.assertNotIn("pending_component_anchor_px", track_state.__dict__)
        self.assertNotIn("pending_component_depth_m", track_state.__dict__)
        self.assertIsNone(inner._active_rigid_pair)
        self.assertEqual(inner._last_rigid_pair_report, {})
        self.assertEqual(inner._opencl_lk_state, "uninitialized")
        self.assertEqual(inner._opencl_lk_validation_remaining, 12)
        self.assertEqual(manager._motion_retention_counter, 0)
        self.assertEqual(self._public_tracker_snapshot(runtime), before)

        lease = manager.begin_motion_retention(
            ["head", "tail"],
            episode_id="episode-live-17",
            timeout_s=5.0,
        )
        self.assertTrue(lease["ok"])
        self.assertEqual(lease["lease_id"], "motion-retention-00000001")
        self.assertTrue(manager.end_motion_retention(lease["lease_id"])["released"])

    def test_loaded_new_helper_reloads_and_unloaded_source_changes_stack_digest(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        package = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2"
        )
        token = f"reload_synthetic_{id(self)}"
        loaded_name = f"{package.__name__}.{token}.loaded_helper"
        consumer_name = f"{package.__name__}.{token}.consumer"

        with tempfile.TemporaryDirectory(
            prefix=f"_{token}_",
        ) as directory:
            directory_path = Path(directory)
            loaded_source = directory_path / "loaded_helper.py"
            consumer_source = directory_path / "consumer.py"
            unloaded_source = directory_path / "unloaded_helper.py"
            loaded_source.write_text("BUILD = 'v42'\n", encoding="utf-8")
            consumer_source.write_text(
                f"from {loaded_name} import BUILD\n\n"
                "def read_build():\n"
                "    return BUILD\n",
                encoding="utf-8",
            )
            unloaded_source.write_text("BUILD = 'v42'\n", encoding="utf-8")
            loaded_stat = loaded_source.stat()
            unloaded_stat = unloaded_source.stat()
            spec = importlib.util.spec_from_file_location(
                loaded_name, loaded_source
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            loaded_module = importlib.util.module_from_spec(spec)
            sys.modules[loaded_name] = loaded_module
            spec.loader.exec_module(loaded_module)
            self.assertEqual(loaded_module.BUILD, "v42")
            consumer_spec = importlib.util.spec_from_file_location(
                consumer_name, consumer_source
            )
            self.assertIsNotNone(consumer_spec)
            self.assertIsNotNone(consumer_spec.loader)
            consumer_module = importlib.util.module_from_spec(consumer_spec)
            sys.modules[consumer_name] = consumer_module
            consumer_spec.loader.exec_module(consumer_module)
            self.assertEqual(consumer_module.read_build(), "v42")

            def reload_selected(module):
                if module.__name__ in {loaded_name, consumer_name}:
                    return coordinator._reload_one(module, importlib.reload)
                return module

            original_loaded_modules = coordinator._loaded_official_v2_modules
            original_source_tree = coordinator._package_source_tree

            def loaded_modules_with_test_helpers(package_root):
                result = original_loaded_modules(package_root)
                result[loaded_name] = loaded_module
                result[consumer_name] = consumer_module
                return result

            def source_tree_with_test_helpers(modules):
                result = original_source_tree(modules)
                files = list(result["files"])
                for source, module_name, execution in (
                    (loaded_source, loaded_name, "in_process"),
                    (consumer_source, consumer_name, "in_process"),
                    (unloaded_source, None, "lazy"),
                ):
                    record = coordinator._path_source_record(
                        source,
                        name=f"__test_external__/{token}/{source.name}",
                    )
                    files.append(
                        {
                            "relative_path": record["name"],
                            "digest": record["digest"],
                            "execution": execution,
                            "module": module_name,
                        }
                    )
                files.sort(key=lambda item: str(item["relative_path"]))
                return {
                    "root": result["root"],
                    "digest": coordinator._canonical_digest(
                        [
                            (item["relative_path"], item["digest"])
                            for item in files
                        ]
                    ),
                    "files": files,
                }

            loaded_modules_patch = mock.patch.object(
                coordinator,
                "_loaded_official_v2_modules",
                side_effect=loaded_modules_with_test_helpers,
            )
            source_tree_patch = mock.patch.object(
                coordinator,
                "_package_source_tree",
                side_effect=source_tree_with_test_helpers,
            )
            loaded_modules_patch.start()
            source_tree_patch.start()
            try:
                runtime = self._runtime()
                baseline = coordinator.reload_official_v2_tool_stack(
                    runtime,
                    import_module=importlib.import_module,
                    reload_module=reload_selected,
                    invalidate_caches=importlib.invalidate_caches,
                )
                unchanged = coordinator.reload_official_v2_tool_stack(
                    runtime,
                    import_module=importlib.import_module,
                    reload_module=reload_selected,
                    invalidate_caches=importlib.invalidate_caches,
                )
                self.assertEqual(
                    unchanged["stack_digest"], baseline["stack_digest"]
                )

                unloaded_source.write_text("BUILD = 'v43'\n", encoding="utf-8")
                os.utime(
                    unloaded_source,
                    ns=(unloaded_stat.st_atime_ns, unloaded_stat.st_mtime_ns),
                )
                unloaded_changed = coordinator.reload_official_v2_tool_stack(
                    runtime,
                    import_module=importlib.import_module,
                    reload_module=reload_selected,
                    invalidate_caches=importlib.invalidate_caches,
                )
                self.assertNotEqual(
                    unloaded_changed["stack_digest"],
                    unchanged["stack_digest"],
                    "an unimported official_v2 source edit was absent from stack identity",
                )

                loaded_source.write_text("BUILD = 'v43'\n", encoding="utf-8")
                os.utime(
                    loaded_source,
                    ns=(loaded_stat.st_atime_ns, loaded_stat.st_mtime_ns),
                )
                loaded_changed = coordinator.reload_official_v2_tool_stack(
                    runtime,
                    import_module=importlib.import_module,
                    reload_module=reload_selected,
                    invalidate_caches=importlib.invalidate_caches,
                )
                self.assertEqual(loaded_module.BUILD, "v43")
                self.assertEqual(
                    consumer_module.read_build(),
                    "v43",
                    "a live direct-import consumer retained the old helper binding",
                )
                self.assertIn(
                    loaded_name,
                    {item["name"] for item in loaded_changed["modules"]},
                )
                self.assertIn(
                    consumer_name,
                    {item["name"] for item in loaded_changed["modules"]},
                )
                self.assertNotEqual(
                    loaded_changed["stack_digest"],
                    unloaded_changed["stack_digest"],
                )
            finally:
                source_tree_patch.stop()
                loaded_modules_patch.stop()
                sys.modules.pop(consumer_name, None)
                sys.modules.pop(loaded_name, None)

    def test_transaction_rebinds_existing_http_contract_and_metadata_once(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        policy = importlib.import_module(
            "behavior_interface_eval_test.official_policy_interface"
        )
        capabilities = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.capabilities"
        )
        registry_module = importlib.import_module(
            "behavior_interface_eval_test.tool.official_v2.registry"
        )
        runtime = self._runtime()
        app = Flask(f"{__name__}.http-rebind")
        runtime._official_http_app = app
        submitted: list[tuple[str, dict]] = []
        base_calls: list[int] = []

        original_validate = capabilities.validate_submission
        original_policy_validate = policy.validate_submission
        original_registry_builder = registry_module.build_registry

        def sentinel_validate(name, args):
            normalized = dict(args or {})
            sentinel = normalized.pop("__reload_sentinel", None)
            result = original_validate(name, normalized)
            if sentinel is not None:
                result["__reload_sentinel"] = sentinel
                result["timeout_s"] = 321.0
            return result

        def sentinel_registry(adapter):
            entries = original_registry_builder(adapter)
            spec = entries["move_tracked_point"]
            for parameter in spec.params:
                if parameter.get("name") == "timeout_s":
                    parameter["default"] = 321.0
            spec.params.append(
                {
                    "name": "__reload_sentinel",
                    "type": "str",
                    "default": None,
                    "required": False,
                }
            )
            return entries

        def submit_skill(name, args):
            submitted.append((str(name), deepcopy(args)))
            return "request-hot-reload"

        def wait_for_skill_result(name, **_kwargs):
            self.assertEqual(name, "move_tracked_point")
            return {
                "ok": True,
                "validated_timeout_s": submitted[-1][1]["timeout_s"],
                "sentinel": submitted[-1][1].get("__reload_sentinel"),
            }

        runtime.server.submit_skill = submit_skill
        runtime.server.wait_for_skill_result = wait_for_skill_result

        @app.post(
            "/api/v2/move_tracked_point",
            endpoint="official_move_tracked_point",
        )
        def stale_move_route():
            from flask import jsonify, request

            body = request.get_json(force=True, silent=True) or {}
            raw = {key: value for key, value in body.items() if key != "session_id"}
            try:
                original_validate("move_tracked_point", raw)
            except (TypeError, ValueError) as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
            return jsonify({"ok": True, "stale_handler": True})

        @app.get("/api/v2/tools", endpoint="api_v2_tools")
        def stale_tools_metadata():
            from flask import jsonify

            base_calls.append(1)
            return jsonify(
                {
                    "tool_version": "v2",
                    "tools": [
                        {"name": "track_object_distance", "args": []},
                        {
                            "name": "move_tracked_point",
                            "args": [
                                {
                                    "name": "points",
                                    "required": True,
                                    "min_points": 1,
                                    "max_points": 2,
                                },
                                {
                                    "name": "timeout_s",
                                    "default": 90.0,
                                    "required": False,
                                },
                            ],
                        },
                    ],
                }
            )

        @app.get("/__official__/tools", endpoint="official_tools")
        def stale_official_tools():
            from flask import jsonify

            return jsonify({"ok": True, **policy.capability_report()})

        request_body = {
            "session_id": "web-hot-reload",
            "points": [
                {
                    "name": "head",
                    "target_xyz_m": ["x", "y", "z"],
                }
            ],
            "__reload_sentinel": "accepted-after-reload",
        }
        client = app.test_client()
        self.assertEqual(
            client.post("/api/v2/move_tracked_point", json=request_body).status_code,
            400,
        )
        base_view = app.view_functions["api_v2_tools"]

        capability_table = deepcopy(capabilities.TOOL_CAPABILITIES)
        capability_table["move_tracked_point"][
            "reload_test_sentinel"
        ] = "new-capability"
        with mock.patch.object(
            capabilities, "validate_submission", sentinel_validate
        ), mock.patch.object(
            capabilities, "TOOL_CAPABILITIES", capability_table
        ), mock.patch.object(
            registry_module, "build_registry", sentinel_registry
        ), mock.patch.object(
            policy, "validate_submission", original_policy_validate
        ), mock.patch.object(
            policy, "_attach_official_head_capture", return_value=None
        ):
            first_receipt = coordinator.reload_official_v2_tool_stack(
                runtime,
                import_module=importlib.import_module,
                reload_module=lambda module: module,
                invalidate_caches=lambda: None,
            )
            accepted = client.post(
                "/api/v2/move_tracked_point", json=request_body
            )
            first_metadata = client.get("/api/v2/tools").get_json()
            first_capabilities = client.get("/__official__/tools").get_json()

            second_receipt = coordinator.reload_official_v2_tool_stack(
                runtime,
                import_module=importlib.import_module,
                reload_module=lambda module: module,
                invalidate_caches=lambda: None,
            )
            second_metadata = client.get("/api/v2/tools").get_json()

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(accepted.get_json()["sentinel"], "accepted-after-reload")
        self.assertEqual(accepted.get_json()["validated_timeout_s"], 321.0)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(first_receipt["generation"], 1)
        self.assertEqual(second_receipt["generation"], 2)
        self.assertIs(runtime._official_v2_tools_base_view, base_view)
        self.assertEqual(len(base_calls), 2)
        for payload, receipt in (
            (first_metadata, first_receipt),
            (second_metadata, second_receipt),
        ):
            move_entries = [
                item
                for item in payload["tools"]
                if item.get("name") == "move_tracked_point"
            ]
            self.assertEqual(len(move_entries), 1)
            params = {
                item["name"]: item for item in move_entries[0]["args"]
            }
            self.assertIn("__reload_sentinel", params)
            self.assertEqual(params["timeout_s"]["default"], 321.0)
            self.assertEqual(
                payload["official_v2_reload"],
                {
                    "generation": receipt["generation"],
                    "stack_digest": receipt["stack_digest"],
                    "metadata_revision": receipt["ui"]["metadata_revision"],
                },
            )
            self.assertEqual(
                move_entries[0]["metadata_revision"],
                receipt["ui"]["metadata_revision"],
            )
        self.assertEqual(
            first_capabilities["tools"]["move_tracked_point"][
                "reload_test_sentinel"
            ],
            "new-capability",
        )

    def test_normal_job_and_queue_reject_before_registry_mutation(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        cases = ("current_job", "queue")
        for case in cases:
            with self.subTest(case=case):
                runtime = self._runtime()
                old_registry = runtime.tool_registry
                if case == "current_job":
                    runtime.server.current_job = SimpleNamespace(request_id="job-live")
                else:
                    runtime.server.skill_queue.put(SimpleNamespace(request_id="job-queued"))

                import_module = mock.Mock(
                    side_effect=AssertionError("busy reload must not import modules")
                )
                reload_module = mock.Mock(
                    side_effect=AssertionError("busy reload must not reload modules")
                )
                invalidate_caches = mock.Mock()
                with self.assertRaisesRegex(RuntimeError, "running|queued|normal skill"):
                    coordinator.reload_official_v2_tool_stack(
                        runtime,
                        import_module=import_module,
                        reload_module=reload_module,
                        invalidate_caches=invalidate_caches,
                    )

                self.assertEqual(runtime._official_reload_generation, 0)
                self.assertIs(runtime.tool_registry, old_registry)
                self.assertIs(runtime._official_skills_module.SKILL_REGISTRY, old_registry)
                invalidate_caches.assert_not_called()
                import_module.assert_not_called()
                reload_module.assert_not_called()

    def test_live_job_or_unfinished_future_rejects_before_registry_mutation(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        pending = concurrent.futures.Future()
        jobs = (
            _StatusJob("running"),
            _StatusJob("cancelled", future=pending),
        )
        for job in jobs:
            with self.subTest(state=job.status()["state"]):
                runtime = self._runtime()
                runtime._live_test_job = job
                old_registry = runtime.tool_registry

                import_module = mock.Mock(
                    side_effect=AssertionError("busy reload must not import modules")
                )
                reload_module = mock.Mock(
                    side_effect=AssertionError("busy reload must not reload modules")
                )
                invalidate_caches = mock.Mock()
                with self.assertRaisesRegex(RuntimeError, "live test|planner thread"):
                    coordinator.reload_official_v2_tool_stack(
                        runtime,
                        import_module=import_module,
                        reload_module=reload_module,
                        invalidate_caches=invalidate_caches,
                    )

                self.assertEqual(runtime._official_reload_generation, 0)
                self.assertIs(runtime.tool_registry, old_registry)
                self.assertIs(runtime._official_skills_module.SKILL_REGISTRY, old_registry)
                invalidate_caches.assert_not_called()
                import_module.assert_not_called()
                reload_module.assert_not_called()

    def test_reload_failure_rolls_back_registry_and_generation(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        runtime = self._runtime()
        old_registry = runtime.tool_registry
        old_status = runtime.official_v2_reload_status()
        old_manager = runtime.tracked_object_distances
        old_inner_tracker = old_manager._tracker
        old_manager_class = old_manager.__class__
        old_inner_class = old_inner_tracker.__class__
        old_manager_containers = {
            name: getattr(old_manager, name)
            for name in (
                "_capture_bindings",
                "_binding_failures",
                "_frame_history",
                "_pending_replay",
                "_name_by_track_id",
                "_entries",
                "_registration_entries",
                "_temporarily_unobserved",
                "_motion_retention_leases",
            )
        }
        old_registration_binding = old_manager._registration_binding
        old_latest_frame = old_manager._latest_frame
        old_entries = dict(old_manager._entries)
        old_capture_bindings = dict(old_manager._capture_bindings)
        touched_modules = []

        def fail_on_execution(module):
            module.__dict__["_OFFICIAL_V2_RELOAD_TEST_SENTINEL"] = module.__name__
            touched_modules.append(module)
            if module.__name__.endswith("tracked_point_execution_local"):
                raise RuntimeError("injected reload failure")
            return module

        with self.assertRaisesRegex(RuntimeError, "injected reload failure"):
            coordinator.reload_official_v2_tool_stack(
                runtime,
                import_module=importlib.import_module,
                reload_module=fail_on_execution,
                invalidate_caches=lambda: None,
            )

        self.assertEqual(runtime._official_reload_generation, 0)
        self.assertIs(runtime.tool_registry, old_registry)
        self.assertIs(runtime._official_skills_module.SKILL_REGISTRY, old_registry)
        after_status = runtime.official_v2_reload_status()
        self.assertEqual(after_status["generation"], old_status["generation"])
        self.assertEqual(after_status["registry"], old_status["registry"])
        self.assertIs(runtime.tracked_object_distances, old_manager)
        self.assertIs(old_manager._tracker, old_inner_tracker)
        self.assertIs(old_manager.__class__, old_manager_class)
        self.assertIs(old_inner_tracker.__class__, old_inner_class)
        for name, container in old_manager_containers.items():
            self.assertIs(getattr(old_manager, name), container, name)
        self.assertIs(old_manager._registration_binding, old_registration_binding)
        self.assertIs(old_manager._latest_frame, old_latest_frame)
        self.assertEqual(old_manager._entries, old_entries)
        self.assertEqual(old_manager._capture_bindings, old_capture_bindings)
        self.assertGreaterEqual(len(touched_modules), 1)
        for module in touched_modules:
            self.assertNotIn(
                "_OFFICIAL_V2_RELOAD_TEST_SENTINEL",
                module.__dict__,
                module.__name__,
            )

    def test_precommit_failure_after_http_rebind_rolls_back_every_publication(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        policy = importlib.import_module(
            "behavior_interface_eval_test.official_policy_interface"
        )
        runtime = self._runtime()
        app = Flask(f"{__name__}.late-rollback")
        runtime._official_http_app = app

        @app.post(
            "/api/v2/move_tracked_point",
            endpoint="official_move_tracked_point",
        )
        def old_move_view():
            return {"ok": True, "generation": "old"}

        @app.get("/api/v2/tools", endpoint="api_v2_tools")
        def old_tools_view():
            return {"tools": []}

        old_views = dict(app.view_functions)
        old_registry = runtime.tool_registry
        old_status = runtime.official_v2_reload_status()
        old_skills_state = {
            name: getattr(runtime._official_skills_module, name, None)
            for name in ("SKILL_REGISTRY", "PUBLIC_SKILLS", "TOOL_VERSION")
        }
        old_policy_globals = {
            name: getattr(policy, name)
            for name in coordinator._POLICY_REBINDS
        }
        old_manager = runtime.tracked_object_distances
        old_tracker = old_manager._tracker
        old_tracker_public_state = self._public_tracker_snapshot(runtime)
        old_task_memory = runtime.task_memory
        old_task_memory_class = old_task_memory.__class__
        old_task_memory_fields = deepcopy(old_task_memory.raw())
        old_navigation_bridge = runtime._navigation_map_bridge
        old_navigation_bridge_class = old_navigation_bridge.__class__
        old_navigation_bridge_snapshot = old_navigation_bridge._cached_snapshot
        old_navigation_bridge_occupancy = old_navigation_bridge_snapshot[
            "occupancy"
        ]
        old_memory_callbacks = {
            name: getattr(runtime.server, name)
            for name in (
                "get_memory",
                "get_memory_text",
                "get_memory_summary",
            )
        }
        route_overlay = importlib.import_module(
            "behavior_interface_eval_test.navigation_route_overlay"
        )
        old_renderer_hooks = route_overlay.capture_renderer_hook_state()
        navigation_state_before = coordinator._navigation_state_identity(runtime)

        with mock.patch.object(
            coordinator,
            "_builds",
            side_effect=RuntimeError("injected failure after HTTP publication"),
        ):
            with self.assertRaisesRegex(RuntimeError, "after HTTP publication"):
                coordinator.reload_official_v2_tool_stack(
                    runtime,
                    import_module=importlib.import_module,
                    reload_module=lambda module: module,
                    invalidate_caches=lambda: None,
                )

        self.assertEqual(runtime.official_v2_reload_status(), old_status)
        self.assertIs(runtime.tool_registry, old_registry)
        self.assertIs(
            runtime._official_skills_module.SKILL_REGISTRY, old_registry
        )
        for name, value in old_skills_state.items():
            self.assertIs(getattr(runtime._official_skills_module, name, None), value)
        for name, value in old_policy_globals.items():
            self.assertIs(getattr(policy, name), value, name)
        self.assertEqual(set(app.view_functions), set(old_views))
        for name, view in old_views.items():
            self.assertIs(app.view_functions[name], view, name)
        self.assertFalse(hasattr(runtime, "_official_v2_tools_base_view"))
        self.assertIs(runtime.tracked_object_distances, old_manager)
        self.assertIs(old_manager._tracker, old_tracker)
        self.assertEqual(
            self._public_tracker_snapshot(runtime), old_tracker_public_state
        )
        self.assertIs(runtime.task_memory, old_task_memory)
        self.assertIs(runtime.task_memory.__class__, old_task_memory_class)
        self.assertEqual(runtime.task_memory.raw(), old_task_memory_fields)
        self.assertIs(runtime._navigation_map_bridge, old_navigation_bridge)
        self.assertIs(
            runtime._navigation_map_bridge.__class__,
            old_navigation_bridge_class,
        )
        self.assertIs(
            runtime._navigation_map_bridge._cached_snapshot,
            old_navigation_bridge_snapshot,
        )
        self.assertIs(
            runtime._navigation_map_bridge._cached_snapshot["occupancy"],
            old_navigation_bridge_occupancy,
        )
        for name, callback in old_memory_callbacks.items():
            self.assertIs(getattr(runtime.server, name), callback, name)
        restored_renderer_hooks = route_overlay.capture_renderer_hook_state()
        self.assertIs(
            restored_renderer_hooks.spatial_render_minimap_rgba,
            old_renderer_hooks.spatial_render_minimap_rgba,
        )
        self.assertIs(
            restored_renderer_hooks.spatial_map_version,
            old_renderer_hooks.spatial_map_version,
        )
        self.assertIs(
            restored_renderer_hooks.live_map_snapshot_png,
            old_renderer_hooks.live_map_snapshot_png,
        )
        self.assertEqual(
            coordinator._navigation_state_identity(runtime),
            navigation_state_before,
        )

    def test_failed_reload_keeps_last_commit_live_and_can_retry(self) -> None:
        coordinator = importlib.import_module(
            "behavior_interface_eval_test.official_v2_hot_reload"
        )
        runtime = self._runtime()
        no_reload = lambda module: module

        first = coordinator.reload_official_v2_tool_stack(
            runtime,
            import_module=importlib.import_module,
            reload_module=no_reload,
            invalidate_caches=lambda: None,
        )
        first_registry = runtime.tool_registry
        first_callbacks = {
            name: spec.fn for name, spec in first_registry.items()
        }
        first_status = runtime.official_v2_reload_status()
        touched_modules = []

        def fail_after_first_generation(module):
            module.__dict__["_OFFICIAL_V2_RELOAD_RETRY_SENTINEL"] = True
            touched_modules.append(module)
            if module.__name__.endswith("tracked_point_execution_local"):
                raise RuntimeError("generation two is invalid")
            return module

        with self.assertRaisesRegex(RuntimeError, "generation two is invalid"):
            coordinator.reload_official_v2_tool_stack(
                runtime,
                import_module=importlib.import_module,
                reload_module=fail_after_first_generation,
                invalidate_caches=lambda: None,
            )

        self.assertIs(runtime.tool_registry, first_registry)
        self.assertEqual(runtime.official_v2_reload_status(), first_status)
        self.assertEqual(runtime._official_reload_generation, first["generation"])
        for name, callback in first_callbacks.items():
            self.assertIs(runtime.tool_registry[name].fn, callback, name)
        for module in touched_modules:
            self.assertNotIn(
                "_OFFICIAL_V2_RELOAD_RETRY_SENTINEL", module.__dict__
            )

        second = coordinator.reload_official_v2_tool_stack(
            runtime,
            import_module=importlib.import_module,
            reload_module=no_reload,
            invalidate_caches=lambda: None,
        )
        self.assertEqual(second["generation"], first["generation"] + 1)
        self.assertEqual(
            runtime.official_v2_reload_status()["stack_digest"],
            second["stack_digest"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
