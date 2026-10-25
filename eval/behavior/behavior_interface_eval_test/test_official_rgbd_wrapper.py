from __future__ import annotations

import importlib.util
import json
import base64
import hashlib
import os
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np


class OfficialRGBDWrapperTest(unittest.TestCase):
    def test_refreshes_handles_before_reloading_observation_space(self) -> None:
        events = []

        fake_og = types.ModuleType("omnigibson")
        fake_og.sim = types.SimpleNamespace(
            update_handles=lambda: events.append("update_handles"),
            editing_usd=nullcontext,
        )

        fake_envs = types.ModuleType("omnigibson.envs")

        class Environment:
            pass

        class EnvironmentWrapper:
            def __init__(self, env):
                self.env = env

        fake_envs.Environment = Environment
        fake_envs.EnvironmentWrapper = EnvironmentWrapper

        fake_eval_utils = types.ModuleType(
            "omnigibson.eval.utils.eval_utils"
        )
        fake_eval_utils.HEAD_RESOLUTION = (720, 720)
        fake_eval_utils.WRIST_RESOLUTION = (480, 480)
        fake_eval_utils.get_robot_camera_names = lambda name, config: {
            "head": f"{name}::head_sensor"
        }

        def set_sensor_modalities(sensor, modalities):
            events.append(("modalities", frozenset(modalities)))

        fake_eval_utils.set_sensor_modalities = set_sensor_modalities

        fake_ui_utils = types.ModuleType("omnigibson.utils.ui_utils")
        fake_ui_utils.create_module_logger = lambda module_name: types.SimpleNamespace(
            info=lambda message: None
        )

        fake_websocket_client = types.ModuleType("websockets.sync.client")

        def sync_websocket_connect(*args, **kwargs):
            raise AssertionError("the threaded sync websocket must not be used")

        fake_websocket_client.connect = sync_websocket_connect
        fake_websocket_sync = types.ModuleType("websockets.sync")
        fake_websocket_sync.client = fake_websocket_client

        fake_websockets = types.ModuleType("websockets")
        fake_websockets.sync = fake_websocket_sync

        class FakeSocket:
            def __init__(self):
                self.incoming = bytearray()
                self.sent = []
                self.closed = False
                self.timeout = "unset"

            def settimeout(self, timeout):
                self.timeout = timeout

            def sendall(self, data):
                data = bytes(data)
                self.sent.append(data)
                if not data.startswith(b"GET "):
                    return
                key_line = next(
                    line
                    for line in data.split(b"\r\n")
                    if line.lower().startswith(b"sec-websocket-key:")
                )
                key = key_line.split(b":", 1)[1].strip().decode("ascii")
                accept = base64.b64encode(
                    hashlib.sha1(
                        (
                            key
                            + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
                        ).encode("ascii")
                    ).digest()
                ).decode("ascii")
                response = (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                ).encode("ascii")
                self.incoming.extend(
                    response
                    + b"\x89\x02hi"
                    + b"\x82\x05reply"
                )

            def recv(self, size):
                if not self.incoming:
                    return b""
                chunk = bytes(self.incoming[:size])
                del self.incoming[:size]
                return chunk

            def close(self):
                self.closed = True
                events.append("websocket_close")

        fake_socket = FakeSocket()

        class Sensor:
            def __init__(self):
                self._height = 1
                self._width = 1

            @property
            def image_height(self):
                return self._height

            @image_height.setter
            def image_height(self, value):
                self._height = value
                events.append(("height", value))

            @property
            def image_width(self):
                return self._width

            @image_width.setter
            def image_width(self, value):
                self._width = value
                events.append(("width", value))

        robot = types.SimpleNamespace(
            name="robot_r1",
            sensors={"head_sensor": Sensor()},
        )
        task = types.SimpleNamespace(
            compiled_task=types.SimpleNamespace(
                conditions=types.SimpleNamespace(
                    parsed_goal_conditions=[
                        ["real", "cooked__popcorn.n.01_1"],
                        [
                            "contains",
                            "popcorn__bag.n.01_1",
                            "cooked__popcorn.n.01_1",
                        ],
                    ]
                )
            ),
            activity_natural_language_goal_conditions=[],
        )

        def env_step(action, n_render_iterations=1):
            events.append(("env_step", tuple(action), n_render_iterations))
            return (
                {"robot_r1": {"proprio": [0.0]}},
                0.0,
                False,
                False,
                {"goal_status": {"satisfied": [1], "unsatisfied": [0]}},
            )

        def env_reset():
            events.append("env_reset")
            return ({"reset": True}, {"obs_info": {}})

        env = types.SimpleNamespace(
            robots=[robot],
            task=task,
            _eval_robot_config={},
            load_observation_space=lambda: events.append(
                "load_observation_space"
            ),
            step=env_step,
            reset=env_reset,
        )

        module_path = (
            Path(__file__).with_name("official_rgbd_wrapper.py")
        )
        spec = importlib.util.spec_from_file_location(
            "_official_rgbd_wrapper_test_target",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)

        fake_modules = {
            "omnigibson": fake_og,
            "omnigibson.envs": fake_envs,
            "omnigibson.eval": types.ModuleType("omnigibson.eval"),
            "omnigibson.eval.utils": types.ModuleType(
                "omnigibson.eval.utils"
            ),
            "omnigibson.eval.utils.eval_utils": fake_eval_utils,
            "omnigibson.utils": types.ModuleType("omnigibson.utils"),
            "omnigibson.utils.ui_utils": fake_ui_utils,
            "websockets": fake_websockets,
            "websockets.sync": fake_websocket_sync,
            "websockets.sync.client": fake_websocket_client,
        }
        with mock.patch.dict(sys.modules, fake_modules), mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            spec.loader.exec_module(module)
            self.assertEqual(module._rgbd_annotator_device(), "cpu")
            with mock.patch.object(
                module,
                "_prepare_rgbd_annotator_device",
                wraps=module._prepare_rgbd_annotator_device,
            ) as prepare_annotator_device:
                wrapper = module.OfficialRGBDFullResWrapper(env)
            prepare_annotator_device.assert_called_once_with(
                [robot.sensors["head_sensor"]],
                "cpu",
            )
            # v3.9.3 builds full-resolution cameras before wrapping. Setting
            # an equal dimension still destroys the live Replicator graph.
            # Exercise its nested one-environment robot layout as well.
            before = len(events)
            env.robots = [[robot]]
            module.OfficialRGBDFullResWrapper(env)
            self.assertFalse(any(
                isinstance(event, tuple) and event[0] in {"height", "width"}
                for event in events[before:]
            ))
            del events[before:]
            env.robots = [robot]
            sensor = robot.sensors["head_sensor"]
            annotator_calls = []

            class Annotator:
                def __init__(self, value, *, wrapped=False):
                    self.value = value
                    self.wrapped = wrapped

                def get_data(self, *, device, do_array_copy):
                    annotator_calls.append((device, do_array_copy))
                    return {"data": self.value} if self.wrapped else self.value

            sensor.initialized = True
            sensor._modalities = {"rgb", "depth_linear"}
            sensor._annotators = {
                "rgb": Annotator("rgb-data"),
                "depth_linear": Annotator("depth-data", wrapped=True),
            }
            sensor._preprocess_cpu_obs = lambda data, modality: (
                modality,
                data,
            )
            camera_obs, camera_info = sensor._get_obs()
            step_result = wrapper.step([1.0, 2.0], n_render_iterations=3)
            reset_result = wrapper.reset()
            with mock.patch.object(
                module.socket,
                "create_connection",
                return_value=fake_socket,
            ) as create_connection:
                connection = fake_websocket_client.connect(
                    "ws://test:1234/policy",
                    compression=None,
                    max_size=None,
                    ping_interval=60,
                    ping_timeout=300,
                )
                connection.send(b"request")
                self.assertEqual(connection.recv(), b"reply")
                connection.close()
                create_connection.assert_called_once_with(
                    ("test", 1234),
                    timeout=10,
                )
            self.assertFalse(wrapper.rigid_brake_status["direct_state_mutation"])

            with mock.patch.dict(
                os.environ,
                {module._RGBD_ANNOTATOR_DEVICE_ENV: " CUDA:2 "},
                clear=True,
            ):
                self.assertEqual(module._rgbd_annotator_device(), "cuda:2")
            for invalid_device in ("", "gpu", "cuda:", "cuda:-1", "cuda:one"):
                with self.subTest(invalid_device=invalid_device), mock.patch.dict(
                    os.environ,
                    {module._RGBD_ANNOTATOR_DEVICE_ENV: invalid_device},
                    clear=True,
                ):
                    with self.assertRaises(ValueError):
                        module._rgbd_annotator_device()

            lifecycle_events = []
            render_product = object()
            raw_types = {
                "rgb": "rgb",
                "depth_linear": "distance_to_image_plane",
            }
            arrays = {
                "rgb": np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
                "depth_linear": np.arange(6, dtype=np.float32).reshape(2, 3),
            }
            replacements = {}
            case = self

            class LifecycleAnnotator:
                def __init__(self, modality, *, candidate):
                    self.modality = modality
                    self.candidate = candidate
                    self.template_name = (
                        f"{modality}buffPtr" if candidate else f"{modality}hostPtr"
                    )
                    self.precommit_reads = 0

                def attach(self, attached_render_product):
                    self.assert_render_product(attached_render_product)
                    lifecycle_events.append(("attach", self.modality))

                def detach(self, detached_render_product):
                    self.assert_render_product(detached_render_product)
                    if not self.candidate:
                        self.assert_is_active(replacements[self.modality])
                    lifecycle_events.append(
                        (
                            "detach_candidate" if self.candidate else "detach_old",
                            self.modality,
                        )
                    )

                def get_data(self, *, device, do_array_copy):
                    self.assertEqualCopyRequest(device, do_array_copy)
                    if self.candidate and not self.is_active():
                        self.precommit_reads += 1
                    lifecycle_events.append(
                        ("get_data", self.modality, self.candidate, self.is_active())
                    )
                    data = arrays[self.modality].copy()
                    return {"data": data} if self.candidate else data

                def is_active(self):
                    return cuda_sensor._annotators[self.modality] is self

                def assert_is_active(self, expected):
                    case.assertIs(cuda_sensor._annotators[self.modality], expected)

                def assert_render_product(self, value):
                    case.assertIs(value, render_product)

                def assertEqualCopyRequest(self, device, do_array_copy):
                    case.assertEqual((device, do_array_copy), ("cpu", True))

            original_annotators = {
                modality: LifecycleAnnotator(modality, candidate=False)
                for modality in arrays
            }
            cuda_sensor = types.SimpleNamespace(
                initialized=True,
                _modalities=set(arrays),
                _annotators=dict(original_annotators),
                _RAW_SENSOR_TYPES=raw_types,
                _render_product=render_product,
                _preprocess_cpu_obs=lambda data, modality: (modality, data),
            )
            registry_calls = []

            class Registry:
                @staticmethod
                def get_annotator(name, *, device, do_array_copy):
                    registry_calls.append((name, device, do_array_copy))
                    modality = next(
                        modality
                        for modality, raw_type in raw_types.items()
                        if raw_type == name
                    )
                    candidate = LifecycleAnnotator(modality, candidate=True)
                    replacements[modality] = candidate
                    return candidate

            with mock.patch.object(
                module,
                "_replicator_annotator_registry",
                return_value=Registry,
            ):
                plans = module._prepare_rgbd_annotator_device(
                    [cuda_sensor],
                    "cuda:0",
                )

            unexpected_originals = dict(original_annotators)
            unexpected_originals["rgb"] = LifecycleAnnotator(
                "rgb",
                candidate=True,
            )
            unexpected_sensor = types.SimpleNamespace(
                _modalities=set(arrays),
                _annotators=unexpected_originals,
                _RAW_SENSOR_TYPES=raw_types,
                _render_product=render_product,
            )
            with mock.patch.object(
                module,
                "_replicator_annotator_registry",
                return_value=Registry,
            ), self.assertRaisesRegex(RuntimeError, "expected a CPU hostPtr"):
                module._prepare_rgbd_annotator_device(
                    [unexpected_sensor],
                    "cuda:0",
                )
            self.assertEqual(
                registry_calls,
                [
                    ("distance_to_image_plane", "cuda:0", True),
                    ("rgb", "cuda:0", True),
                ],
            )
            for modality in arrays:
                self.assertIs(
                    cuda_sensor._annotators[modality],
                    original_annotators[modality],
                )
            self.assertTrue(module._finalize_rgbd_annotator_device(plans))
            for modality in arrays:
                self.assertEqual(replacements[modality].precommit_reads, 1)
                self.assertIs(cuda_sensor._annotators[modality], replacements[modality])
                self.assertLess(
                    lifecycle_events.index(("get_data", modality, True, False)),
                    lifecycle_events.index(("detach_old", modality)),
                )

            module._install_cpu_rgbd_reader(cuda_sensor)
            self.assertFalse(cuda_sensor._eval_test_profile_rgbd)
            with mock.patch.object(
                module.time,
                "perf_counter",
                side_effect=AssertionError(
                    "disabled profiling must not read the clock"
                ),
            ):
                cuda_obs, cuda_info = cuda_sensor._get_obs()
            self.assertEqual(cuda_info, {})
            np.testing.assert_array_equal(cuda_obs["rgb"][1], arrays["rgb"])
            self.assertIn(("get_data", "rgb", True, True), lifecycle_events)

            natural_events = []

            class NaturalResetAnnotator:
                def __init__(self, *, candidate):
                    self.candidate = candidate

                def get_data(self, *, device, do_array_copy):
                    case.assertEqual((device, do_array_copy), ("cpu", True))
                    natural_events.append(("get_data", self.candidate))
                    return arrays["rgb"].copy()

                def detach(self, detached_render_product):
                    case.assertIs(detached_render_product, render_product)
                    case.assertIs(
                        natural_sensor._annotators["rgb"],
                        natural_candidate,
                    )
                    natural_events.append(("detach", self.candidate))

            natural_original = NaturalResetAnnotator(candidate=False)
            natural_candidate = NaturalResetAnnotator(candidate=True)
            natural_sensor = types.SimpleNamespace(
                _annotators={"rgb": natural_original},
            )
            natural_plan = {
                "sensor": natural_sensor,
                "render_product": render_product,
                "modalities": ("rgb",),
                "originals": {"rgb": natural_original},
                "replacements": {"rgb": natural_candidate},
            }
            wrapper._pending_rgbd_migration = [natural_plan]
            event_count = len(events)
            wrapper.reset()
            self.assertEqual(events[event_count], "env_reset")
            self.assertEqual(
                natural_events,
                [("get_data", False), ("get_data", True), ("detach", False)],
            )
            self.assertIs(
                natural_sensor._annotators["rgb"],
                natural_candidate,
            )
            self.assertEqual(wrapper._pending_rgbd_migration, [])

            mismatch_events = []
            mismatch_replacements = {}

            class MismatchAnnotator:
                def __init__(self, modality, *, candidate):
                    self.modality = modality
                    self.candidate = candidate
                    self.template_name = (
                        f"{modality}buffPtr" if candidate else f"{modality}hostPtr"
                    )

                def attach(self, attached_render_product):
                    self.assert_render_product(attached_render_product)
                    mismatch_events.append(("attach", self.modality))

                def detach(self, detached_render_product):
                    self.assert_render_product(detached_render_product)
                    if not self.candidate:
                        raise AssertionError("old annotator must remain attached")
                    mismatch_events.append(("detach_candidate", self.modality))

                def get_data(self, *, device, do_array_copy):
                    self.assertEqualCopyRequest(device, do_array_copy)
                    case.assertIs(
                        mismatch_sensor._annotators[self.modality],
                        mismatch_originals[self.modality],
                    )
                    data = arrays[self.modality].copy()
                    if self.candidate and self.modality == "depth_linear":
                        data.flat[0] += 1
                    return {"data": data} if self.candidate else data

                def assert_render_product(self, value):
                    case.assertIs(value, render_product)

                def assertEqualCopyRequest(self, device, do_array_copy):
                    case.assertEqual((device, do_array_copy), ("cpu", True))

            mismatch_originals = {
                modality: MismatchAnnotator(modality, candidate=False)
                for modality in arrays
            }
            mismatch_sensor = types.SimpleNamespace(
                _modalities=set(arrays),
                _annotators=dict(mismatch_originals),
                _RAW_SENSOR_TYPES=raw_types,
                _render_product=render_product,
            )

            class MismatchRegistry:
                @staticmethod
                def get_annotator(name, *, device, do_array_copy):
                    self.assertEqual((device, do_array_copy), ("cuda:0", True))
                    modality = next(
                        modality
                        for modality, raw_type in raw_types.items()
                        if raw_type == name
                    )
                    candidate = MismatchAnnotator(modality, candidate=True)
                    mismatch_replacements[modality] = candidate
                    return candidate

            with mock.patch.object(
                module,
                "_replicator_annotator_registry",
                return_value=MismatchRegistry,
            ):
                mismatch_plans = module._prepare_rgbd_annotator_device(
                    [mismatch_sensor],
                    "cuda:0",
                )
            wrapper._pending_rgbd_migration = mismatch_plans
            with self.assertRaisesRegex(RuntimeError, "parity check failed"):
                wrapper._finalize_pending_rgbd_migration()
            self.assertEqual(wrapper._pending_rgbd_migration, [])
            for modality in arrays:
                self.assertIs(
                    mismatch_sensor._annotators[modality],
                    mismatch_originals[modality],
                )
            self.assertEqual(
                mismatch_events[-2:],
                [
                    ("detach_candidate", "rgb"),
                    ("detach_candidate", "depth_linear"),
                ],
            )

            atomic_events = []

            class AtomicAnnotator:
                def __init__(
                    self,
                    sensor_id,
                    *,
                    candidate,
                    mismatch=False,
                    detach_fails=False,
                ):
                    self.sensor_id = sensor_id
                    self.candidate = candidate
                    self.mismatch = mismatch
                    self.detach_fails = detach_fails

                def get_data(self, *, device, do_array_copy):
                    case.assertEqual((device, do_array_copy), ("cpu", True))
                    data = arrays["rgb"].copy()
                    if self.mismatch:
                        data.flat[0] += 1
                    return data

                def detach(self, detached_render_product):
                    case.assertIs(detached_render_product, render_product)
                    if not self.candidate:
                        raise AssertionError("uncommitted old graph was detached")
                    atomic_events.append(("detach", self.sensor_id))
                    if self.detach_fails:
                        raise RuntimeError("candidate detach failed")

            atomic_plans = []
            atomic_sensors = []
            for sensor_id in ("head", "wrist"):
                original = AtomicAnnotator(sensor_id, candidate=False)
                candidate = AtomicAnnotator(
                    sensor_id,
                    candidate=True,
                    mismatch=sensor_id == "wrist",
                )
                sensor = types.SimpleNamespace(_annotators={"rgb": original})
                atomic_sensors.append((sensor, original))
                atomic_plans.append(
                    {
                        "sensor": sensor,
                        "render_product": render_product,
                        "modalities": ("rgb",),
                        "originals": {"rgb": original},
                        "replacements": {"rgb": candidate},
                    }
                )
            wrapper._pending_rgbd_migration = atomic_plans
            with self.assertRaisesRegex(RuntimeError, "parity validation failed"):
                wrapper._finalize_pending_rgbd_migration()
            self.assertEqual(atomic_events, [("detach", "wrist"), ("detach", "head")])
            for sensor, original in atomic_sensors:
                self.assertIs(sensor._annotators["rgb"], original)
            self.assertEqual(wrapper._pending_rgbd_migration, [])

            cleanup_original = AtomicAnnotator("cleanup", candidate=False)
            cleanup_candidate = AtomicAnnotator(
                "cleanup",
                candidate=True,
                mismatch=True,
                detach_fails=True,
            )
            cleanup_sensor = types.SimpleNamespace(
                _annotators={"rgb": cleanup_original},
            )
            cleanup_plan = {
                "sensor": cleanup_sensor,
                "render_product": render_product,
                "modalities": ("rgb",),
                "originals": {"rgb": cleanup_original},
                "replacements": {"rgb": cleanup_candidate},
            }
            wrapper._pending_rgbd_migration = [cleanup_plan]
            with self.assertRaisesRegex(RuntimeError, "candidate cleanup also failed"):
                wrapper._finalize_pending_rgbd_migration()
            self.assertEqual(wrapper._pending_rgbd_migration, [])
            self.assertIs(cleanup_sensor._annotators["rgb"], cleanup_original)
            self.assertIs(wrapper._failed_rgbd_migration[-1], cleanup_plan)

            partial_events = []

            class PartialAttachAnnotator:
                def __init__(self, modality):
                    self.modality = modality
                    self.template_name = f"{modality}buffPtr"

                def attach(self, attached_render_product):
                    case.assertIs(attached_render_product, render_product)
                    partial_events.append(("attach", self.modality))
                    if self.modality == "rgb":
                        raise RuntimeError("partial attach")

                def detach(self, detached_render_product):
                    case.assertIs(detached_render_product, render_product)
                    partial_events.append(("detach", self.modality))

            class PartialRegistry:
                @staticmethod
                def get_annotator(name, *, device, do_array_copy):
                    case.assertEqual((device, do_array_copy), ("cuda:0", True))
                    modality = next(
                        modality
                        for modality, raw_type in raw_types.items()
                        if raw_type == name
                    )
                    return PartialAttachAnnotator(modality)

            partial_sensor = types.SimpleNamespace(
                _modalities=set(arrays),
                _annotators=dict(original_annotators),
                _RAW_SENSOR_TYPES=raw_types,
                _render_product=render_product,
            )
            with mock.patch.object(
                module,
                "_replicator_annotator_registry",
                return_value=PartialRegistry,
            ):
                with self.assertRaisesRegex(RuntimeError, "partial attach"):
                    module._prepare_rgbd_annotator_device(
                        [partial_sensor],
                        "cuda:0",
                    )
            self.assertEqual(
                partial_events[-2:],
                [("detach", "rgb"), ("detach", "depth_linear")],
            )
            for modality in arrays:
                self.assertIs(
                    partial_sensor._annotators[modality],
                    original_annotators[modality],
                )

            retained_array = arrays["rgb"]

            class RetainedAnnotator:
                def __init__(self, *, candidate):
                    self.candidate = candidate

                def get_data(self, *, device, do_array_copy):
                    case.assertEqual((device, do_array_copy), ("cpu", True))
                    return retained_array.copy()

                def detach(self, detached_render_product):
                    case.assertIs(detached_render_product, render_product)
                    if not self.candidate:
                        raise RuntimeError("old cleanup failed")

            retained_original = RetainedAnnotator(candidate=False)
            retained_candidate = RetainedAnnotator(candidate=True)
            retained_sensor = types.SimpleNamespace(
                _annotators={"rgb": retained_original},
            )
            retained_plan = {
                "sensor": retained_sensor,
                "render_product": render_product,
                "modalities": ("rgb",),
                "originals": {"rgb": retained_original},
                "replacements": {"rgb": retained_candidate},
            }
            wrapper._pending_rgbd_migration = [retained_plan]
            with self.assertRaisesRegex(RuntimeError, "old annotator cleanup"):
                wrapper._finalize_pending_rgbd_migration()
            self.assertEqual(wrapper._pending_rgbd_migration, [])
            self.assertIs(
                retained_sensor._annotators["rgb"],
                retained_candidate,
            )
            self.assertIs(wrapper._failed_rgbd_migration[-1], retained_plan)

        self.assertEqual(annotator_calls, [("cpu", True), ("cpu", True)])
        self.assertEqual(camera_info, {})
        self.assertEqual(camera_obs["rgb"], ("rgb", "rgb-data"))
        self.assertEqual(
            camera_obs["depth_linear"],
            ("depth_linear", "depth-data"),
        )

        step_progress = json.loads(
            step_result[0][module.UI_BDDL_PROGRESS_KEY]
        )
        reset_progress = json.loads(
            reset_result[0][module.UI_BDDL_PROGRESS_KEY]
        )
        self.assertEqual(step_progress["satisfied"], 1)
        self.assertEqual(step_progress["total"], 2)
        self.assertFalse(step_progress["complete"])
        self.assertEqual(reset_progress["satisfied"], 0)
        self.assertEqual(reset_progress["total"], 2)
        self.assertIn(b"GET /policy HTTP/1.1", fake_socket.sent[0])
        frame_header = fake_socket.sent[1]
        self.assertEqual(frame_header[0] & 0x0F, 0x2)
        self.assertTrue(frame_header[1] & 0x80)
        self.assertEqual(
            module._mask_websocket_payload(
                fake_socket.sent[2],
                frame_header[2:6],
            ),
            b"request",
        )
        pong_header = fake_socket.sent[3]
        self.assertEqual(pong_header[0] & 0x0F, 0xA)
        self.assertEqual(
            module._mask_websocket_payload(
                fake_socket.sent[4],
                pong_header[2:6],
            ),
            b"hi",
        )
        self.assertIsNone(fake_socket.timeout)
        self.assertTrue(fake_socket.closed)
        self.assertIn("websocket_close", events)

        update_index = events.index("update_handles")
        load_index = events.index("load_observation_space")
        self.assertLess(update_index, load_index)
        self.assertIn(("env_step", (1.0, 2.0), 3), events)
        self.assertGreater(events.count("update_handles"), 1)
        self.assertIn(("modalities", frozenset({"rgb", "depth_linear"})), events)
        self.assertIn(("height", 720), events)
        self.assertIn(("width", 720), events)

        source = module_path.read_text(encoding="utf-8")
        self.assertNotIn("get_joint_positions", source)
        self.assertNotIn("RigidBrakeEffortManager", source)
