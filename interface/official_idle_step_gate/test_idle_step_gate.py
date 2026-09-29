from __future__ import annotations

import asyncio
import threading
import time
import unittest
from typing import Any
from urllib.error import HTTPError

import numpy as np
import websockets

from official_idle_step_gate.protocol import (
    has_reset_key,
    is_reset_payload,
    packb,
    unpackb,
)
from official_idle_step_gate.proxy import (
    ActivitySnapshot,
    BackendActivityReader,
    GateCancelled,
    IdleStepGate,
    IdleStepGateProxy,
    ProxyConfig,
)


def _health(*, source: str = "hold", downstream: bool = False, live: str = "idle"):
    return {
        "ok": True,
        "action_source": source,
        "downstream": {"configured": downstream},
        "episode_initialization": {"ready": True},
        "live_unit_test": {"state": live},
        "last_error": "",
    }


class ProtocolTest(unittest.TestCase):
    def test_official_numpy_frame_roundtrip(self) -> None:
        value = {
            "action": np.arange(23, dtype=np.float32),
            "nested": {"x": np.array([[1, 2]], dtype=np.int64)},
        }
        decoded = unpackb(packb(value), strict_map_key=False)
        np.testing.assert_array_equal(decoded["action"], value["action"])
        np.testing.assert_array_equal(decoded["nested"]["x"], value["nested"]["x"])

    def test_reset_control_is_key_presence_based(self) -> None:
        self.assertTrue(is_reset_payload({"reset": False}))
        self.assertTrue(is_reset_payload({b"reset": 0}))
        self.assertFalse(is_reset_payload({"resetting": True}))

    def test_reset_scanner_skips_large_observation_values(self) -> None:
        observation = packb(
            {"RGB": np.zeros((128, 128, 3), dtype=np.uint8), "depth": np.ones(7)}
        )
        self.assertFalse(has_reset_key(observation))
        self.assertTrue(has_reset_key(packb({"reset": False, "payload": observation})))

    def test_reset_scanner_fails_open_on_malformed_msgpack(self) -> None:
        # The evaluator/backend remains the protocol authority for malformed
        # input; the sidecar must not close a valid websocket just while
        # deciding whether a frame is a reset control message.
        self.assertFalse(has_reset_key(b"\x81\xa5reset"))
        self.assertFalse(has_reset_key(packb({"reset": False}) + b"junk"))


class ClassificationTest(unittest.TestCase):
    def test_only_complete_idle_hold_is_suppressible(self) -> None:
        state = {
            "active_skill": None,
            "reset_pending": False,
            "task_switch_pending": False,
            "vision_degraded": False,
            "simulation_degraded": False,
        }
        activity = BackendActivityReader.classify_payloads(_health(), state)
        self.assertTrue(activity.idle)
        self.assertTrue(activity.diagnostic_ok)
        self.assertEqual(activity.reason, "idle_hold")

    def test_current_runtime_none_reset_request_is_idle(self) -> None:
        state = {
            "active_skill": None,
            "reset_pending": None,
            "task_switch_pending": False,
            "vision_degraded": False,
            "simulation_degraded": False,
        }
        activity = BackendActivityReader.classify_payloads(_health(), state)
        self.assertTrue(activity.idle)

    def test_missing_or_pending_state_flags_fail_open(self) -> None:
        missing = BackendActivityReader.classify_payloads(
            _health(), {"active_skill": None}
        )
        self.assertFalse(missing.idle)
        self.assertFalse(missing.diagnostic_ok)

        pending = BackendActivityReader.classify_payloads(
            _health(),
            {
                "active_skill": None,
                "reset_pending": True,
                "task_switch_pending": False,
                "vision_degraded": False,
                "simulation_degraded": False,
            },
        )
        self.assertFalse(pending.idle)
        self.assertEqual(pending.reason, "reset_pending")

    def test_active_skill_and_downstream_fail_open(self) -> None:
        state = {"active_skill": {"state": "running"}}
        active = BackendActivityReader.classify_payloads(_health(), state)
        self.assertFalse(active.idle)
        self.assertEqual(active.reason, "skill_active_or_queued")

        downstream = BackendActivityReader.classify_payloads(
            _health(downstream=True), {}
        )
        self.assertFalse(downstream.idle)
        self.assertEqual(downstream.reason, "downstream_policy_configured")

    def test_all_live_states_are_observation_driven(self) -> None:
        state = {"active_skill": None}
        planning = BackendActivityReader.classify_payloads(
            _health(live="planning"), state
        )
        self.assertFalse(planning.idle)
        self.assertEqual(planning.reason, "live_test_planning")
        running = BackendActivityReader.classify_payloads(
            _health(live="running"), state
        )
        self.assertFalse(running.idle)

    def test_missing_diagnostics_never_suppresses(self) -> None:
        activity = BackendActivityReader.classify_payloads(
            {"ok": True, "action_source": "hold"}, {}
        )
        self.assertFalse(activity.idle)
        self.assertFalse(activity.diagnostic_ok)

    def test_transitions_errors_and_degradation_fail_open(self) -> None:
        base_state = {
            "active_skill": None,
            "reset_pending": False,
            "task_switch_pending": False,
            "vision_degraded": False,
            "simulation_degraded": False,
        }
        cases = (
            ("initialization", {"ready": False}, "episode_initialization_active"),
            ("error", "backend failure", "backend_error_present"),
            ("task_switch", True, "task_switch_pending"),
            ("vision", True, "backend_degraded"),
            ("simulation", True, "backend_degraded"),
            ("live_unknown", "unexpected", "live_test_state_unknown"),
        )
        for name, value, reason in cases:
            with self.subTest(name=name):
                health = _health()
                state = dict(base_state)
                if name == "initialization":
                    health["episode_initialization"] = value
                elif name == "error":
                    health["last_error"] = value
                elif name == "task_switch":
                    state["task_switch_pending"] = value
                elif name == "vision":
                    state["vision_degraded"] = value
                elif name == "simulation":
                    state["simulation_degraded"] = value
                else:
                    health["live_unit_test"] = {"state": value}
                activity = BackendActivityReader.classify_payloads(health, state)
                self.assertFalse(activity.idle)
                self.assertEqual(activity.reason, reason)

    def test_compact_idle_probe_requires_explicit_protocol_and_flags(self) -> None:
        idle = BackendActivityReader.classify_idle_probe(
            {
                "ok": True,
                "protocol": "behavior-interface-idle-probe-v1",
                "diagnostic_ok": True,
                "idle": True,
                "reason": "idle_hold",
            }
        )
        self.assertTrue(idle.idle)
        self.assertTrue(idle.diagnostic_ok)
        self.assertEqual(idle.reason, "idle_hold")

        malformed = BackendActivityReader.classify_idle_probe(
            {
                "ok": True,
                "protocol": "behavior-interface-idle-probe-v1",
                "diagnostic_ok": True,
                "idle": "yes",
            }
        )
        self.assertFalse(malformed.idle)
        self.assertFalse(malformed.diagnostic_ok)

    def test_compact_probe_failure_is_fail_open(self) -> None:
        activity = BackendActivityReader.classify_idle_probe(
            {
                "ok": True,
                "protocol": "behavior-interface-idle-probe-v1",
                "diagnostic_ok": False,
                "idle": True,
                "reason": "idle_probe_read_failed:RuntimeError",
            }
        )
        self.assertFalse(activity.idle)
        self.assertFalse(activity.diagnostic_ok)
        self.assertEqual(activity.reason, "idle_probe_read_failed:RuntimeError")


class GateWaitTest(unittest.IsolatedAsyncioTestCase):
    class SequenceReader:
        def __init__(self, values: list[ActivitySnapshot]):
            self.values = list(values)
            self.calls = 0

        async def read(self) -> ActivitySnapshot:
            self.calls += 1
            if self.values:
                return self.values.pop(0)
            return ActivitySnapshot(True, True, "idle_hold")

    async def test_idle_wait_does_not_return_until_activity_appears(self) -> None:
        reader = self.SequenceReader(
            [
                ActivitySnapshot(True, True, "idle_hold"),
                ActivitySnapshot(False, True, "skill_active_or_queued"),
            ]
        )
        gate = IdleStepGate(reader, poll_interval_s=0.01)
        cancel = asyncio.Event()
        started = time.monotonic()
        released = await gate.wait_until_released(cancel_event=cancel)
        self.assertFalse(released.idle)
        self.assertEqual(released.reason, "skill_active_or_queued")
        self.assertGreaterEqual(reader.calls, 2)
        self.assertGreaterEqual(time.monotonic() - started, 0.009)

    async def test_cancel_releases_wait_without_backend_poll_loop(self) -> None:
        reader = self.SequenceReader([ActivitySnapshot(True, True, "idle_hold")])
        gate = IdleStepGate(reader, poll_interval_s=0.01)
        cancel = asyncio.Event()

        async def cancel_soon() -> None:
            await asyncio.sleep(0.015)
            cancel.set()

        asyncio.create_task(cancel_soon())
        with self.assertRaises(GateCancelled):
            await gate.wait_until_released(cancel_event=cancel)

    async def test_positive_bound_fails_open(self) -> None:
        reader = self.SequenceReader([ActivitySnapshot(True, True, "idle_hold")])
        gate = IdleStepGate(reader, poll_interval_s=0.005, max_wait_s=0.02)
        result = await gate.wait_until_released(cancel_event=asyncio.Event())
        self.assertFalse(result.idle)
        self.assertEqual(result.reason, "idle_wait_timeout")


class GateInspectTest(unittest.TestCase):
    def test_only_action_response_can_be_deferred(self) -> None:
        reader = _MutableReader(ActivitySnapshot(True, True, "idle_hold"))
        gate = IdleStepGate(reader)
        no_action = gate.inspect(packb({"server_timing": {}}))
        self.assertFalse(no_action.defer)
        self.assertEqual(no_action.activity.reason, "response_has_no_action")

        action = gate.inspect(packb({"action": np.zeros(23, dtype=np.float32)}))
        self.assertTrue(action.defer)
        self.assertEqual(action.activity.reason, "idle_hold")


class ReaderIOTest(unittest.TestCase):
    def test_reader_uses_compact_probe_without_fetching_large_state(self) -> None:
        calls: list[str] = []
        payload = {
            "ok": True,
            "protocol": "behavior-interface-idle-probe-v1",
            "diagnostic_ok": True,
            "idle": True,
            "reason": "idle_hold",
        }

        def fetch(url: str, _timeout: float):
            calls.append(url)
            return payload

        reader = BackendActivityReader("http://backend", fetch_json=fetch)
        # Production urllib readers opt in automatically. Enabling the same
        # mode here keeps the test deterministic without opening a socket.
        reader._prefer_idle_probe = True
        result = reader.read()
        self.assertTrue(result.idle)
        self.assertEqual(calls, ["http://backend/__official__/idle_probe"])

    def test_reader_falls_back_to_legacy_endpoints_on_probe_404(self) -> None:
        calls: list[str] = []
        payloads: dict[str, Any] = {
            "http://backend/__official__/healthz": _health(),
            "http://backend/api/state": {
                "active_skill": None,
                "reset_pending": None,
                "task_switch_pending": False,
                "vision_degraded": False,
                "simulation_degraded": False,
            },
        }

        def fetch(url: str, _timeout: float):
            calls.append(url)
            if url.endswith("/__official__/idle_probe"):
                raise HTTPError(url, 404, "missing", hdrs=None, fp=None)
            return payloads[url]

        reader = BackendActivityReader("http://backend", fetch_json=fetch)
        reader._prefer_idle_probe = True
        result = reader.read()
        self.assertTrue(result.idle)
        self.assertEqual(
            calls,
            [
                "http://backend/__official__/idle_probe",
                "http://backend/__official__/healthz",
                "http://backend/api/state",
            ],
        )

    def test_reader_fetches_state_only_for_candidate_hold(self) -> None:
        calls: list[str] = []
        payloads: dict[str, Any] = {
            "http://backend/__official__/healthz": _health(source="downstream-model"),
        }

        def fetch(url: str, _timeout: float):
            calls.append(url)
            return payloads[url]

        reader = BackendActivityReader("http://backend", fetch_json=fetch)
        result = reader.read()
        self.assertFalse(result.idle)
        self.assertEqual(calls, ["http://backend/__official__/healthz"])

    def test_reader_fail_open_on_http_error(self) -> None:
        def fetch(_url: str, _timeout: float):
            raise OSError("offline")

        reader = BackendActivityReader("http://backend", fetch_json=fetch)
        result = reader.read()
        self.assertFalse(result.idle)
        self.assertFalse(result.diagnostic_ok)
        self.assertTrue(result.reason.startswith("health_fetch_failed:"))


class _FakeSocket:
    """Minimal async socket used to exercise the proxy handler in-process."""

    _CLOSE = object()

    def __init__(self, messages: list[Any] | None = None) -> None:
        self._incoming: asyncio.Queue[Any] = asyncio.Queue()
        for message in messages or []:
            self._incoming.put_nowait(message)
        self.sent: list[Any] = []
        self._closed = asyncio.Event()

    async def recv(self) -> Any:
        message = await self._incoming.get()
        if message is self._CLOSE:
            raise websockets.ConnectionClosedOK(None, None)
        return message

    async def send(self, message: Any) -> None:
        self.sent.append(message)

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def close(self) -> None:
        if not self._closed.is_set():
            self._closed.set()
            self._incoming.put_nowait(self._CLOSE)


class _FakeBackend(_FakeSocket):
    def __init__(self, response: bytes | list[bytes]) -> None:
        super().__init__([packb({"layer": "fake-backend"})])
        self.responses = list(response) if isinstance(response, list) else [response]

    async def send(self, message: Any) -> None:
        await super().send(message)
        if not has_reset_key(message):
            response = self.responses.pop(0) if self.responses else packb({})
            self._incoming.put_nowait(response)


class _MutableReader:
    def __init__(self, activity: ActivitySnapshot) -> None:
        self.activity = activity
        self.calls = 0

    def read(self) -> ActivitySnapshot:
        self.calls += 1
        return self.activity


class ProxyProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def _wait_for(self, predicate, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("condition was not reached before timeout")
            await asyncio.sleep(0.005)

    async def test_idle_response_is_deferred_then_exact_bytes_are_forwarded(self) -> None:
        response = packb({"action": np.arange(23, dtype=np.float32), "tag": "hold"})
        backend = _FakeBackend(response)
        reader = _MutableReader(ActivitySnapshot(True, True, "idle_hold"))
        gate = IdleStepGate(reader, poll_interval_s=0.01)
        proxy = IdleStepGateProxy(
            ProxyConfig(
                backend_connect_timeout_s=0.1,
                poll_interval_s=0.01,
            ),
            reader=reader,
            gate=gate,
        )

        async def connect_backend():
            return backend

        proxy._connect_backend = connect_backend  # type: ignore[method-assign]
        client = _FakeSocket([packb({"robot_r1::proprio": np.zeros(1)})])
        task = asyncio.create_task(proxy.handler(client))
        await self._wait_for(lambda: len(client.sent) >= 1)
        self.assertEqual(unpackb(client.sent[0])["layer"], "fake-backend")
        await asyncio.sleep(0.03)
        # Only metadata has crossed the evaluator-facing boundary so far.
        self.assertEqual(len(client.sent), 1)

        reader.activity = ActivitySnapshot(False, True, "skill_active_or_queued")
        await self._wait_for(lambda: len(client.sent) >= 2)
        self.assertEqual(client.sent[1], response)
        self.assertEqual(proxy.status()["counters"]["deferred_frames"], 1)
        self.assertEqual(proxy.status()["counters"]["released_deferred_frames"], 1)
        await client.close()
        await asyncio.wait_for(task, timeout=1.0)

    async def test_active_response_and_reset_are_forwarded_without_suppression(self) -> None:
        response = packb({"action": np.zeros(23, dtype=np.float32)})
        backend = _FakeBackend(response)
        reader = _MutableReader(ActivitySnapshot(False, True, "skill_active_or_queued"))
        proxy = IdleStepGateProxy(
            ProxyConfig(backend_connect_timeout_s=0.1),
            reader=reader,
            gate=IdleStepGate(reader, poll_interval_s=0.01),
        )

        async def connect_backend():
            return backend

        proxy._connect_backend = connect_backend  # type: ignore[method-assign]
        reset = packb({"reset": True})
        observation = packb({"observation": np.zeros(2, dtype=np.float32)})
        client = _FakeSocket([reset, observation])
        task = asyncio.create_task(proxy.handler(client))
        await self._wait_for(lambda: len(client.sent) >= 2)
        self.assertEqual(client.sent[1], response)
        self.assertEqual(backend.sent[:2], [reset, observation])
        await client.close()
        await asyncio.wait_for(task, timeout=1.0)

    async def test_first_idle_frame_after_activity_is_transition_then_later_idle_defers(
        self,
    ) -> None:
        active_response = packb(
            {"action": np.zeros(23, dtype=np.float32), "tag": "active"}
        )
        transition_response = packb(
            {"action": np.ones(23, dtype=np.float32), "tag": "transition"}
        )
        deferred_response = packb(
            {"action": np.full(23, 2.0, dtype=np.float32), "tag": "deferred"}
        )
        backend = _FakeBackend(
            [active_response, transition_response, deferred_response]
        )
        reader = _MutableReader(
            ActivitySnapshot(False, True, "skill_active_or_queued")
        )
        proxy = IdleStepGateProxy(
            ProxyConfig(
                backend_connect_timeout_s=0.1,
                poll_interval_s=0.01,
            ),
            reader=reader,
            gate=IdleStepGate(reader, poll_interval_s=0.01),
        )

        async def connect_backend():
            return backend

        proxy._connect_backend = connect_backend  # type: ignore[method-assign]
        observations = [
            packb({"observation": np.array([index], dtype=np.float32)})
            for index in range(3)
        ]
        # Feed observations one at a time so the test controls the activity
        # state seen by each response rather than racing the handler loop.
        client = _FakeSocket([observations[0]])
        task = asyncio.create_task(proxy.handler(client))

        await self._wait_for(lambda: len(client.sent) >= 2)
        self.assertEqual(unpackb(client.sent[1])['tag'], 'active')

        # The first idle response after an active response is the one
        # post-action observation/termination transition and must pass.
        reader.activity = ActivitySnapshot(True, True, "idle_hold")
        client._incoming.put_nowait(observations[1])
        await self._wait_for(lambda: len(client.sent) >= 3)
        self.assertEqual(unpackb(client.sent[2])['tag'], 'transition')
        self.assertEqual(
            proxy.status()['counters']['transition_release_frames'], 1
        )

        # A subsequent pure idle response is the one that may be held.
        client._incoming.put_nowait(observations[2])
        await asyncio.sleep(0.03)
        self.assertEqual(len(client.sent), 3)
        self.assertEqual(proxy.status()['counters']['deferred_frames'], 1)
        reader.activity = ActivitySnapshot(False, True, "skill_active_or_queued")
        await self._wait_for(lambda: len(client.sent) >= 4)
        self.assertEqual(unpackb(client.sent[3])['tag'], 'deferred')
        self.assertEqual(
            proxy.status()['counters']['released_deferred_frames'], 1
        )
        await client.close()
        await asyncio.wait_for(task, timeout=1.0)


if __name__ == "__main__":
    unittest.main()
