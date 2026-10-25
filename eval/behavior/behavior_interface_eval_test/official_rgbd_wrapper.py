"""Challenge RGB-D and BDDL UI wrapper for the evaluator.

The v3.9.1 stock RGBDFullResWrapper rebuilds camera render products and then
reloads the full observation space. Rebuilding render products invalidates the
Isaac articulation handles on some headless systems, so the proprioception
space reload can observe a missing physics view. Refreshing handles after the
camera-only USD edits preserves the stock observation contract without
changing task or simulator state. The normal wrapper does not read simulator
joint state or mutate actuator, robot, object, task, or simulator state. An
explicitly enabled, test-only oracle may read private assisted-grasp state into
an isolated JSONL file; it never alters the returned observation.
"""

import base64
import hashlib
import os
import socket
import ssl
import struct
import time
from types import MethodType
from urllib.parse import urlparse

import numpy as np
import omnigibson as og
from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import (
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    get_robot_camera_names,
    set_sensor_modalities,
)
from omnigibson.utils.ui_utils import create_module_logger

from behavior_interface_eval_test.official_bddl_progress import (
    UI_BDDL_PROGRESS_KEY,
    build_evaluator_bddl_progress,
    encode_bddl_progress,
)
from behavior_interface_eval_test.privileged_grasp_oracle import (
    privileged_grasp_oracle_from_env,
)


logger = create_module_logger(module_name=__name__)


_WEBSOCKET_ACCEPT_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_OFFICIAL_RGBD_MODALITIES = frozenset({"rgb", "depth_linear"})
_RGBD_ANNOTATOR_DEVICE_ENV = "BEHAVIOR_EVAL_TEST_RGBD_ANNOTATOR_DEVICE"
_STEP_PROFILE_INTERVAL_ENV = "BEHAVIOR_EVAL_TEST_STEP_PROFILE_INTERVAL"
_RGBD_PROFILE_TOTALS = {
    "get_data_s": 0.0,
    "preprocess_s": 0.0,
    "calls": 0,
}


def _rgbd_annotator_device():
    """Return the validated Replicator graph device for official RGB-D."""
    device = os.environ.get(_RGBD_ANNOTATOR_DEVICE_ENV, "cpu").strip().lower()
    if device in {"cpu", "cuda"}:
        return device
    prefix, separator, ordinal = device.partition(":")
    if prefix == "cuda" and separator and ordinal.isdigit():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible.isdigit() and int(ordinal) != 0:
            raise ValueError(
                f"{_RGBD_ANNOTATOR_DEVICE_ENV}={device} is invalid with "
                f"CUDA_VISIBLE_DEVICES={visible}; use local cuda:0"
            )
        return device
    raise ValueError(
        f"{_RGBD_ANNOTATOR_DEVICE_ENV} must be cpu, cuda, or cuda:<index>"
    )


def _step_profile_interval():
    """Return the opt-in evaluator timing report interval."""
    raw = os.environ.get(_STEP_PROFILE_INTERVAL_ENV, "0").strip()
    try:
        interval = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_STEP_PROFILE_INTERVAL_ENV} must be an integer"
        ) from exc
    if not 0 <= interval <= 10000:
        raise ValueError(
            f"{_STEP_PROFILE_INTERVAL_ENV} must be between 0 and 10000"
        )
    return interval


def _replicator_annotator_registry():
    import omnigibson.lazy as lazy

    return lazy.omni.replicator.core.AnnotatorRegistry


def _raw_annotator_data(raw_observation):
    return (
        raw_observation["data"]
        if isinstance(raw_observation, dict)
        else raw_observation
    )


def _raw_array_signature(data):
    """Return the observable NumPy layout and content of an annotator array."""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"annotator returned unsupported CPU data type: {type(data)!r}")
    if getattr(data, "size", 0) == 0:
        raise ValueError("annotator returned an empty CPU array")
    flags = getattr(data, "flags", None)
    return (
        type(data),
        data.dtype.str,
        tuple(data.shape),
        tuple(data.strides),
        bool(flags.c_contiguous) if flags is not None else None,
        bool(flags.f_contiguous) if flags is not None else None,
        bool(flags.writeable) if flags is not None else None,
        data.tobytes(order="C"),
    )


class _RGBDAnnotatorMigrationError(RuntimeError):
    """Annotator migration failure with explicit cleanup state."""

    def __init__(self, message, *, cleanup_complete, committed=False):
        super().__init__(message)
        self.cleanup_complete = cleanup_complete
        self.committed = committed


def _detach_rgbd_candidates(attached):
    rollback_errors = []
    with og.sim.editing_usd():
        for plan, modality in reversed(attached):
            try:
                plan["replacements"][modality].detach(plan["render_product"])
            except Exception as exc:
                rollback_errors.append(f"detach candidate {modality}: {exc}")
    if rollback_errors:
        raise RuntimeError(
            "failed to clean up uncommitted CUDA RGB-D annotators: "
            + "; ".join(rollback_errors)
        )


def _prepare_rgbd_annotator_device(sensors, device):
    """Attach CUDA candidates while leaving the active CPU annotators untouched."""
    if device == "cpu":
        return []

    registry = _replicator_annotator_registry()
    plans = []
    for sensor in sensors:
        modalities = tuple(sorted(sensor._modalities))
        unsupported = set(modalities) - _OFFICIAL_RGBD_MODALITIES
        missing = _OFFICIAL_RGBD_MODALITIES - set(modalities)
        if unsupported or missing:
            raise RuntimeError(
                "official RGB-D annotator setup requires exactly rgb and "
                "depth_linear; "
                f"missing={sorted(missing)}, unsupported={sorted(unsupported)}"
            )
        originals = {
            modality: sensor._annotators[modality]
            for modality in modalities
        }
        for modality, original in originals.items():
            if not original.template_name.endswith("hostPtr"):
                raise RuntimeError(
                    "official RGB-D migration expected a CPU hostPtr graph for "
                    f"{modality}, got {original.template_name}"
                )
        plans.append(
            {
                "sensor": sensor,
                "render_product": sensor._render_product,
                "modalities": modalities,
                "originals": originals,
                "replacements": {},
            }
        )
        plan = plans[-1]
        for modality in modalities:
            candidate = registry.get_annotator(
                sensor._RAW_SENSOR_TYPES[modality],
                device=device,
                do_array_copy=True,
            )
            if not candidate.template_name.endswith("buffPtr"):
                raise RuntimeError(
                    "CUDA RGB-D annotator did not select a buffPtr graph for "
                    f"{modality}: {candidate.template_name}"
                )
            plan["replacements"][modality] = candidate

    attached = []

    try:
        with og.sim.editing_usd():
            for plan in plans:
                for modality in plan["modalities"]:
                    # Track the attempt first. Replicator can activate a graph
                    # before a later attach callback raises.
                    attached.append((plan, modality))
                    plan["replacements"][modality].attach(plan["render_product"])
    except Exception as exc:
        try:
            _detach_rgbd_candidates(attached)
        except Exception as cleanup_exc:
            raise _RGBDAnnotatorMigrationError(
                "failed to clean up a partially attached CUDA RGB-D graph",
                cleanup_complete=False,
            ) from cleanup_exc
        raise
    return plans


def _finalize_rgbd_annotator_device(plans):
    """Commit candidates only after byte-identical same-frame CPU materialization."""
    if not plans:
        return False

    attached = [
        (plan, modality)
        for plan in plans
        for modality in plan["modalities"]
    ]
    try:
        for plan in plans:
            for modality in plan["modalities"]:
                original = _raw_annotator_data(
                    plan["originals"][modality].get_data(
                        device="cpu",
                        do_array_copy=True,
                    )
                )
                candidate = _raw_annotator_data(
                    plan["replacements"][modality].get_data(
                        device="cpu",
                        do_array_copy=True,
                    )
                )
                if _raw_array_signature(candidate) != _raw_array_signature(original):
                    raise RuntimeError(
                        "CUDA RGB-D parity check failed for modality "
                        f"{modality}"
                    )
    except Exception as exc:
        try:
            _detach_rgbd_candidates(attached)
        except Exception as cleanup_exc:
            raise _RGBDAnnotatorMigrationError(
                "CUDA RGB-D parity validation failed and candidate cleanup "
                "also failed",
                cleanup_complete=False,
            ) from cleanup_exc
        raise _RGBDAnnotatorMigrationError(
            f"CUDA RGB-D parity validation failed: {exc}",
            cleanup_complete=True,
        ) from exc

    for plan in plans:
        plan["sensor"]._annotators.update(plan["replacements"])

    try:
        with og.sim.editing_usd():
            for plan in plans:
                for modality in plan["modalities"]:
                    plan["originals"][modality].detach(plan["render_product"])
    except Exception as exc:
        raise _RGBDAnnotatorMigrationError(
            "CUDA RGB-D candidates passed parity, but old annotator cleanup failed",
            cleanup_complete=False,
            committed=True,
        ) from exc
    return True


def _cpu_rgbd_sensor_get_obs(sensor):
    """Materialize official RGB-D as owned CPU arrays for the evaluator."""
    if not sensor.initialized:
        raise RuntimeError("vision sensor must be initialized before observation")
    modalities = set(sensor._modalities)
    unsupported = modalities - _OFFICIAL_RGBD_MODALITIES
    if unsupported:
        raise RuntimeError(
            "official CPU camera reader received unsupported modalities: "
            f"{sorted(unsupported)}"
        )

    observation = {}
    profile_enabled = bool(
        getattr(sensor, "_eval_test_profile_rgbd", False)
    )
    for modality in modalities:
        # Evaluator keeps the prior observation alive while the next frame is
        # rendered. Copy here so Replicator cannot recycle a live render buffer.
        if profile_enabled:
            started = time.perf_counter()
            raw_observation = sensor._annotators[modality].get_data(
                device="cpu",
                do_array_copy=True,
            )
            _RGBD_PROFILE_TOTALS["get_data_s"] += (
                time.perf_counter() - started
            )
        else:
            raw_observation = sensor._annotators[modality].get_data(
                device="cpu",
                do_array_copy=True,
            )
        data = _raw_annotator_data(raw_observation)
        if profile_enabled:
            started = time.perf_counter()
            observation[modality] = sensor._preprocess_cpu_obs(data, modality)
            _RGBD_PROFILE_TOTALS["preprocess_s"] += (
                time.perf_counter() - started
            )
            _RGBD_PROFILE_TOTALS["calls"] += 1
        else:
            observation[modality] = sensor._preprocess_cpu_obs(data, modality)
    return observation, {}


def _install_cpu_rgbd_reader(sensor, *, profile_enabled=False):
    sensor._eval_test_profile_rgbd = bool(profile_enabled)
    sensor._get_obs = MethodType(_cpu_rgbd_sensor_get_obs, sensor)


def _mask_websocket_payload(payload, mask):
    try:
        from websockets.speedups import apply_mask

        return apply_mask(payload, mask)
    except (ImportError, ModuleNotFoundError):
        return bytes(value ^ mask[index & 3] for index, value in enumerate(payload))


class _MainThreadWebSocketConnection:
    """Small RFC 6455 client with no worker threads or asyncio integration."""

    def __init__(self, uri, **kwargs):
        parsed = urlparse(uri)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise ValueError(f"unsupported evaluator websocket URI: {uri!r}")
        if kwargs.get("compression") not in {None, ""}:
            raise ValueError("evaluator websocket compression must be disabled")

        self._socket = None
        self._buffer = bytearray()
        self._closed = False
        self._close_sent = False
        self._max_size = kwargs.get("max_size")
        self._open(
            parsed,
            additional_headers=kwargs.get("additional_headers"),
            origin=kwargs.get("origin"),
            open_timeout=kwargs.get("open_timeout", 10),
        )

    def _open(
        self,
        parsed,
        *,
        additional_headers,
        origin,
        open_timeout,
    ):
        secure = parsed.scheme == "wss"
        port = parsed.port or (443 if secure else 80)
        raw_socket = socket.create_connection(
            (parsed.hostname, port),
            timeout=open_timeout,
        )
        if secure:
            context = ssl.create_default_context()
            raw_socket = context.wrap_socket(
                raw_socket,
                server_hostname=parsed.hostname,
            )
        raw_socket.settimeout(None)
        self._socket = raw_socket

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        default_port = 443 if secure else 80
        host = parsed.hostname if port == default_port else f"{parsed.hostname}:{port}"
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        headers = {
            "Host": host,
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Key": key,
            "Sec-WebSocket-Version": "13",
        }
        if origin:
            headers["Origin"] = str(origin)
        if additional_headers:
            items = (
                additional_headers.items()
                if hasattr(additional_headers, "items")
                else additional_headers
            )
            for name, value in items:
                headers[str(name)] = str(value)
        request = [f"GET {target} HTTP/1.1"]
        request.extend(f"{name}: {value}" for name, value in headers.items())
        raw_socket.sendall(("\r\n".join(request) + "\r\n\r\n").encode("ascii"))

        response = self._recv_until(b"\r\n\r\n", limit=65536)
        header_block, remainder = response.split(b"\r\n\r\n", 1)
        self._buffer.extend(remainder)
        lines = header_block.decode("latin-1").split("\r\n")
        if not lines or " 101 " not in f" {lines[0]} ":
            self.close()
            raise ConnectionError(f"websocket upgrade failed: {lines[0] if lines else 'empty response'}")
        response_headers = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            response_headers[name.strip().lower()] = value.strip()
        expected_accept = base64.b64encode(
            hashlib.sha1((key + _WEBSOCKET_ACCEPT_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if response_headers.get("sec-websocket-accept") != expected_accept:
            self.close()
            raise ConnectionError("websocket server returned an invalid accept key")

    def _recv_until(self, delimiter, *, limit):
        data = bytearray()
        while delimiter not in data:
            chunk = self._socket.recv(4096)
            if not chunk:
                raise EOFError("websocket closed during HTTP upgrade")
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError("websocket HTTP upgrade response is too large")
        return bytes(data)

    def _recv_exact(self, size):
        data = bytearray()
        if self._buffer:
            take = min(size, len(self._buffer))
            data.extend(self._buffer[:take])
            del self._buffer[:take]
        while len(data) < size:
            chunk = self._socket.recv(size - len(data))
            if not chunk:
                self._closed = True
                raise EOFError("websocket connection closed")
            data.extend(chunk)
        return bytes(data)

    def _send_frame(self, opcode, payload=b""):
        if self._closed or self._socket is None:
            raise EOFError("websocket connection is closed")
        payload = bytes(payload)
        length = len(payload)
        first = 0x80 | int(opcode)
        if length < 126:
            header = struct.pack("!BB", first, 0x80 | length)
        elif length <= 0xFFFF:
            header = struct.pack("!BBH", first, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", first, 0x80 | 127, length)
        mask = os.urandom(4)
        self._socket.sendall(header + mask)
        if payload:
            self._socket.sendall(_mask_websocket_payload(payload, mask))

    def send(self, message):
        if isinstance(message, str):
            self._send_frame(0x1, message.encode("utf-8"))
            return
        self._send_frame(0x2, message)

    def recv(self):
        fragmented_opcode = None
        fragmented_payload = bytearray()
        while True:
            first, second = self._recv_exact(2)
            final = bool(first & 0x80)
            if first & 0x70:
                raise ValueError("compressed websocket frames are not supported")
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            if self._max_size is not None and length > int(self._max_size):
                raise ValueError("websocket message exceeds max_size")
            mask = self._recv_exact(4) if masked else None
            payload = self._recv_exact(length)
            if mask is not None:
                payload = _mask_websocket_payload(payload, mask)

            if opcode == 0x8:
                if not self._close_sent:
                    self._send_frame(0x8, payload[:125])
                    self._close_sent = True
                self._closed = True
                self._socket.close()
                raise EOFError("websocket peer closed the connection")
            if opcode == 0x9:
                self._send_frame(0xA, payload[:125])
                continue
            if opcode == 0xA:
                continue
            if opcode in {0x1, 0x2}:
                if fragmented_opcode is not None:
                    raise ValueError("received a new message during fragmentation")
                if final:
                    return payload.decode("utf-8") if opcode == 0x1 else payload
                fragmented_opcode = opcode
                fragmented_payload.extend(payload)
                continue
            if opcode == 0x0 and fragmented_opcode is not None:
                fragmented_payload.extend(payload)
                if self._max_size is not None and len(fragmented_payload) > int(self._max_size):
                    raise ValueError("websocket message exceeds max_size")
                if final:
                    payload = bytes(fragmented_payload)
                    return payload.decode("utf-8") if fragmented_opcode == 0x1 else payload
                continue
            raise ValueError(f"unsupported websocket opcode: {opcode}")

    def close(self):
        if self._closed:
            return
        try:
            if self._socket is not None and not self._close_sent:
                self._send_frame(0x8, struct.pack("!H", 1000))
                self._close_sent = True
        finally:
            self._closed = True
            if self._socket is not None:
                self._socket.close()


def _install_main_thread_evaluator_websocket():
    """Avoid Isaac/Carb mutex corruption from websocket threads and asyncio."""
    import websockets.sync.client as websocket_client

    current_connect = websocket_client.connect
    if getattr(current_connect, "_eval_test_main_thread", False):
        return

    def connect_main_thread(uri, **kwargs):
        return _MainThreadWebSocketConnection(uri, **kwargs)

    connect_main_thread._eval_test_main_thread = True
    websocket_client.connect = connect_main_thread


class OfficialRGBDFullResWrapper(EnvironmentWrapper):
    """Expose only official full-resolution RGB, depth, and proprioception."""

    @classmethod
    def camera_spec(cls):
        # v3.9.3 creates cameras before constructing the wrapper.
        return {
            "modalities": ["rgb", "depth_linear"],
            "resolution": {
                "head": HEAD_RESOLUTION,
                "left_wrist": WRIST_RESOLUTION,
                "right_wrist": WRIST_RESOLUTION,
            },
        }

    @staticmethod
    def _route_robot(env):
        robots = env.robots
        if robots and isinstance(robots[0], (list, tuple)):
            if len(robots) != 1:
                raise ValueError("One interface port requires exactly one logical environment")
            robots = robots[0]
        if len(robots) != 1:
            raise ValueError("One interface port requires exactly one robot")
        return robots[0]

    def __init__(self, env: Environment):
        super().__init__(env=env)
        _install_main_thread_evaluator_websocket()
        annotator_device = _rgbd_annotator_device()
        self._step_profile_interval = _step_profile_interval()
        robot = self._route_robot(env)
        self._privileged_grasp_oracle = privileged_grasp_oracle_from_env()
        robot_eval_config = getattr(env, "_eval_robot_config", {})
        camera_roles_by_sensor_name = {
            camera_name.split("::")[1]: camera_id
            for camera_id, camera_name in get_robot_camera_names(
                robot.name,
                robot_eval_config,
            ).items()
        }

        rgbd_sensors = []
        for sensor_name, sensor in robot.sensors.items():
            if not hasattr(sensor, "image_height") or not hasattr(
                sensor,
                "image_width",
            ):
                continue
            set_sensor_modalities(sensor, {"rgb", "depth_linear"})
            camera_id = camera_roles_by_sensor_name.get(sensor_name)
            resolution = (
                HEAD_RESOLUTION
                if camera_id == "head"
                else WRIST_RESOLUTION
            )
            # v3.9.3 already applies camera_spec at scene creation. Even an
            # equal-value assignment tears down the Replicator render product
            # and can crash while detaching graph nodes. Resize only legacy
            # cameras which actually have a different resolution.
            if sensor.image_height != resolution[0]:
                sensor.image_height = resolution[0]
            if sensor.image_width != resolution[1]:
                sensor.image_width = resolution[1]
            _install_cpu_rgbd_reader(
                sensor,
                profile_enabled=self._step_profile_interval > 0,
            )
            rgbd_sensors.append(sensor)

        self._failed_rgbd_migration = []
        self._pending_rgbd_migration = _prepare_rgbd_annotator_device(
            rgbd_sensors,
            annotator_device,
        )
        self._step_profile_count = 0
        self._step_profile_last_exit = None
        self._step_profile_totals = {
            "loop_gap_s": 0.0,
            "env_step_s": 0.0,
            "sim_step_s": 0.0,
            "rgbd_get_data_s": 0.0,
            "rgbd_preprocess_s": 0.0,
            "rgbd_calls": 0,
            "wrapper_post_s": 0.0,
        }
        self._step_profile_rgbd_baseline = dict(_RGBD_PROFILE_TOTALS)

        og.sim.update_handles()
        env.load_observation_space()
        logger.info(
            "Reloaded official RGB-D observation space after refreshing "
            "physics handles; evaluator websocket I/O uses a main-thread raw "
            "socket with no worker thread or Kit asyncio integration. RGB-D "
            f"annotators requested the {annotator_device} graph and materialize "
            "owned CPU copies; CUDA candidates must pass a same-frame byte parity "
            "gate before activation. Physics configuration is unchanged. "
            "The wrapper performs no runtime joint-state reads or actuator writes."
        )
        if self._privileged_grasp_oracle is not None:
            self._privileged_grasp_oracle.record(
                robot,
                event="wrapper_initialized",
            )
            logger.warning(
                "PRIVILEGED TEST-ONLY assisted-grasp oracle enabled; labels "
                "are written only to its isolated evaluator JSONL trace and "
                "are not attached to policy observations."
            )

    def step(self, action, n_render_iterations=1):
        profile_enabled = self._step_profile_interval > 0
        started = time.perf_counter() if profile_enabled else 0.0
        if profile_enabled and self._step_profile_last_exit is not None:
            self._step_profile_totals["loop_gap_s"] += (
                started - self._step_profile_last_exit
            )
        result = self.env.step(action, n_render_iterations=n_render_iterations)
        env_finished = time.perf_counter() if profile_enabled else 0.0
        if self._finalize_pending_rgbd_migration():
            og.sim.update_handles()
        if self._privileged_grasp_oracle is not None:
            self._privileged_grasp_oracle.record(
                self._route_robot(self.env),
                event="step",
                action=action,
            )
        info = result[4] if isinstance(result, tuple) and len(result) >= 5 else {}
        goal_status = info.get("goal_status") if isinstance(info, dict) else None
        result = self._attach_bddl_progress(result, goal_status=goal_status)
        if profile_enabled:
            self._record_step_profile(started, env_finished)
        return result

    def _record_step_profile(self, started, env_finished):
        """Record timing only; never reads or mutates simulator/task state."""
        finished = time.perf_counter()
        totals = self._step_profile_totals
        totals["env_step_s"] += env_finished - started
        totals["wrapper_post_s"] += finished - env_finished

        simulator_profiler = getattr(og.sim, "_step_profiler", None)
        if simulator_profiler is not None:
            totals["sim_step_s"] += float(
                getattr(simulator_profiler, "last_dt", 0.0)
            )

        baseline = self._step_profile_rgbd_baseline
        totals["rgbd_get_data_s"] += (
            _RGBD_PROFILE_TOTALS["get_data_s"] - baseline["get_data_s"]
        )
        totals["rgbd_preprocess_s"] += (
            _RGBD_PROFILE_TOTALS["preprocess_s"] - baseline["preprocess_s"]
        )
        totals["rgbd_calls"] += int(
            _RGBD_PROFILE_TOTALS["calls"] - baseline["calls"]
        )
        self._step_profile_rgbd_baseline = dict(_RGBD_PROFILE_TOTALS)
        self._step_profile_count += 1
        count = self._step_profile_count

        if count % self._step_profile_interval == 0:
            samples = self._step_profile_interval
            scale = 1000.0 / samples
            print(
                "[official-test-profile] "
                f"steps={samples} "
                f"loop_gap_ms={totals['loop_gap_s'] * scale:.3f} "
                f"env_step_ms={totals['env_step_s'] * scale:.3f} "
                f"sim_step_ms={totals['sim_step_s'] * scale:.3f} "
                f"env_post_ms={(totals['env_step_s'] - totals['sim_step_s']) * scale:.3f} "
                f"rgbd_get_data_ms={totals['rgbd_get_data_s'] * scale:.3f} "
                f"rgbd_preprocess_ms={totals['rgbd_preprocess_s'] * scale:.3f} "
                f"rgbd_calls_per_step={totals['rgbd_calls'] / samples:.1f} "
                f"wrapper_post_ms={totals['wrapper_post_s'] * scale:.3f}",
                flush=True,
            )
            for key in totals:
                totals[key] = 0 if key == "rgbd_calls" else 0.0
        self._step_profile_last_exit = finished

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self._finalize_pending_rgbd_migration()
        og.sim.update_handles()
        if self._privileged_grasp_oracle is not None:
            self._privileged_grasp_oracle.record(
                self._route_robot(self.env),
                event="reset",
            )
        return self._attach_bddl_progress(result, goal_status=None)

    def _finalize_pending_rgbd_migration(self):
        plans = self._pending_rgbd_migration
        if not plans:
            return False
        try:
            migrated = _finalize_rgbd_annotator_device(plans)
        except _RGBDAnnotatorMigrationError as exc:
            if not exc.cleanup_complete:
                # Preserve strong references for postmortem cleanup. Continuing
                # the evaluator after an incomplete graph cleanup is unsafe.
                self._failed_rgbd_migration.extend(plans)
            self._pending_rgbd_migration = []
            raise
        except Exception:
            self._failed_rgbd_migration.extend(plans)
            self._pending_rgbd_migration = []
            raise
        self._pending_rgbd_migration = []
        if migrated:
            logger.info(
                "Activated CUDA RGB-D annotators after same-frame CPU byte parity "
                "validation."
            )
        return migrated

    def _attach_bddl_progress(self, result, *, goal_status):
        # v3.9.3 returns a list of per-environment observations. Its task
        # accessor is batched too; do not apply the scalar legacy UI helper
        # to that object or expose privileged state to the policy.
        if isinstance(result, tuple) and result and isinstance(result[0], list):
            return result
        progress = encode_bddl_progress(
            build_evaluator_bddl_progress(self.env.task, goal_status)
        )
        if isinstance(result, tuple):
            if not result or not isinstance(result[0], dict):
                return result
            observation = dict(result[0])
            observation[UI_BDDL_PROGRESS_KEY] = progress
            return (observation, *result[1:])
        if isinstance(result, dict):
            observation = dict(result)
            observation[UI_BDDL_PROGRESS_KEY] = progress
            return observation
        return result

    @property
    def rigid_brake_status(self):
        return {
            "enabled": False,
            "direct_simulator_access": False,
            "direct_state_mutation": False,
        }
