"""Protocol-transparent evaluator idle-step gate.

The official evaluator owns the simulator and follows a synchronous loop:
receive observation -> wait for policy action -> ``env.step(action)``.  An
unchanged policy interface normally answers an idle observation with a hold
action, which makes the evaluator advance a useless simulation step.  This
sidecar withholds that already-computed hold response while the interface is
provably idle.  A tool/initialization/action state, a diagnostic failure, or a
disconnect releases the response or closes the connection.

The sidecar is intentionally conservative:

* it forwards the exact backend bytes and never replays an observation;
* it suppresses a response only after an explicit, read-only health/state
  check proves the backend is in an idle hold state;
* uncertainty is fail-open (the original response is sent);
* it has no simulator, Isaac, CUDA, or evaluator imports.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import inspect
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import websockets

try:  # websockets >= 13 (the supported official environments use this path)
    import websockets.asyncio.client as websocket_client
    import websockets.asyncio.server as websocket_server
except ImportError:  # pragma: no cover - compatibility for older deployments
    websocket_client = websockets
    websocket_server = websockets

from .protocol import has_reset_key, mapping_get, unpackb

LOGGER = logging.getLogger("official_idle_step_gate")

_ACTIVE_LIVE_STATES = frozenset(
    {"start_settling", "planning", "running", "settling"}
)
_TERMINAL_LIVE_STATES = frozenset({"idle", "done", "failed", "cancelled"})


async def _run_blocking_daemon(fn: Callable[..., Any], *args: Any) -> Any:
    """Run one bounded blocking probe in a daemon worker.

    Some supported Conda/Python combinations do not reliably join asyncio's
    default executor during interpreter shutdown. Health probes are already
    bounded by an HTTP timeout, so a tiny daemon worker is both safer for the
    sidecar and easier to cancel when a websocket closes.
    """

    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()

    def complete(result: Any = None, error: BaseException | None = None) -> None:
        def publish() -> None:
            if future.done():
                return
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)

        try:
            loop.call_soon_threadsafe(publish)
        except RuntimeError:
            # The event loop may be closing after a websocket cancellation; the
            # daemon worker's result is intentionally dropped in that case.
            pass

    def worker() -> None:
        try:
            complete(result=fn(*args))
        except BaseException as exc:
            complete(error=exc)

    threading.Thread(
        target=worker,
        name="official-idle-gate-probe",
        daemon=True,
    ).start()
    return await future


@dataclass(frozen=True)
class ActivitySnapshot:
    """Read-only classification of the unchanged backend interface."""

    idle: bool
    diagnostic_ok: bool
    reason: str
    health: Mapping[str, Any] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateDecision:
    """Decision for one backend response."""

    defer: bool
    activity: ActivitySnapshot


class BackendActivityReader:
    """Fetch and classify the interface's read-only HTTP diagnostics.

    ``/healthz`` on the policy WebSocket port only reports transport health.
    Newer interfaces expose a compact ``/__official__/idle_probe`` on their
    public HTTP port.  It contains the same conservative idle decision inputs
    without copying camera/replay/scene-graph state.  Older interfaces are
    supported through the original ``/__official__/healthz`` + ``/api/state``
    fallback.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 1.5,
        fetch_json: Callable[[str, float], Mapping[str, Any]] | None = None,
        fail_closed: bool | None = None,
    ) -> None:
        normalized = str(base_url or "").strip().rstrip("/")
        if not normalized:
            raise ValueError("backend HTTP URL is required")
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "backend HTTP URL must be an absolute http(s) URL: "
                f"{base_url!r}"
            )
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("backend HTTP timeout must be positive") from exc
        if timeout <= 0.0:
            raise ValueError("backend HTTP timeout must be positive")
        self.base_url = normalized + "/"
        self.timeout_s = timeout
        if fail_closed is None:
            fail_closed = os.environ.get("BEHAVIOR_IDLE_GATE_FAIL_CLOSED", "0").strip() == "1"
        self.fail_closed = bool(fail_closed)
        # Custom fetchers are used by the isolated unit tests and by a few
        # offline callers that intentionally model the legacy endpoint. Keep
        # their call contract unchanged; the production urllib path opts into
        # the compact probe automatically.
        self._production_fetcher = fetch_json is None
        self._prefer_idle_probe = self._production_fetcher
        self._fetch_json = fetch_json or self._default_fetch_json

    @staticmethod
    def _default_fetch_json(url: str, timeout_s: float) -> Mapping[str, Any]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Cache-Control": "no-cache",
            },
        )
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            status = int(getattr(response, "status", 200))
            if status < 200 or status >= 300:
                raise RuntimeError(f"HTTP {status} from {url}")
            payload = json.load(response)
        if not isinstance(payload, Mapping):
            raise TypeError(f"non-object JSON response from {url}")
        return payload

    def _get(self, path: str) -> Mapping[str, Any]:
        return self._fetch_json(urljoin(self.base_url, path.lstrip("/")), self.timeout_s)

    def _uncertain(
        self,
        reason: str,
        *,
        health: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> ActivitySnapshot:
        """Return the safe decision for a diagnostic failure.

        Ordinary one-off rollouts keep the historical fail-open behavior.
        A resident route opts into fail-closed so an HTTP probe timeout cannot
        turn into an unsolicited simulator step or a video frame.
        """
        if self.fail_closed:
            return ActivitySnapshot(
                True,
                False,
                f"fail_closed:{reason}",
                health=health or {},
                state=state or {},
            )
        return ActivitySnapshot(
            False,
            False,
            reason,
            health=health or {},
            state=state or {},
        )

    def _apply_failure_policy(self, activity: ActivitySnapshot) -> ActivitySnapshot:
        if activity.diagnostic_ok or not self.fail_closed:
            return activity
        return ActivitySnapshot(
            True,
            False,
            f"fail_closed:{activity.reason}",
            health=activity.health,
            state=activity.state,
        )

    @classmethod
    def classify_payloads(
        cls,
        health: Mapping[str, Any],
        state: Mapping[str, Any] | None,
    ) -> ActivitySnapshot:
        """Classify already-fetched payloads without I/O.

        Missing fields are treated as *not safe to suppress*.  This makes the
        sidecar compatible with future interface versions without silently
        changing their action cadence.
        """

        if not isinstance(health, Mapping):
            return ActivitySnapshot(False, False, "health_not_an_object")
        # The interface's health endpoint currently emits ``ok: true``.  Treat
        # an absent/non-boolean value as unknown rather than allowing a future
        # endpoint revision to silently widen the suppression condition.
        if health.get("ok") is not True:
            return ActivitySnapshot(False, False, "health_not_ok", health=health)
        if health.get("action_source") != "hold":
            return ActivitySnapshot(
                False,
                True,
                f"action_source={health.get('action_source')!r}",
                health=health,
                state=state or {},
            )

        downstream = health.get("downstream")
        if not isinstance(downstream, Mapping):
            return ActivitySnapshot(
                False,
                False,
                "downstream_status_missing",
                health=health,
                state=state or {},
            )
        if downstream.get("configured") is not False:
            return ActivitySnapshot(
                False,
                False,
                "downstream_policy_configured",
                health=health,
                state=state or {},
            )

        initialization = health.get("episode_initialization")
        if not isinstance(initialization, Mapping):
            return ActivitySnapshot(
                False,
                False,
                "episode_initialization_status_missing",
                health=health,
                state=state or {},
            )
        if initialization.get("ready") is not True:
            return ActivitySnapshot(
                False,
                True,
                "episode_initialization_active",
                health=health,
                state=state or {},
            )

        live = health.get("live_unit_test")
        if not isinstance(live, Mapping):
            return ActivitySnapshot(
                False,
                False,
                "live_test_status_missing",
                health=health,
                state=state or {},
            )
        live_state = str(live.get("state") or "")
        if live_state in _ACTIVE_LIVE_STATES:
            return ActivitySnapshot(
                False,
                True,
                f"live_test_{live_state}",
                health=health,
                state=state or {},
            )
        # Every non-terminal live state is observation-driven.  In particular,
        # ``planning`` only installs its completed background future when the
        # next evaluator observation reaches ``job.step()``; suppressing that
        # callback would deadlock the live runner.
        if live_state not in _ACTIVE_LIVE_STATES | _TERMINAL_LIVE_STATES:
            return ActivitySnapshot(
                False,
                False,
                "live_test_state_unknown",
                health=health,
                state=state or {},
            )

        # A reported error is deliberately fail-open.  The unchanged backend
        # may be returning a safety hold while waiting for operator recovery.
        if "last_error" not in health:
            return ActivitySnapshot(
                False,
                False,
                "last_error_status_missing",
                health=health,
                state=state or {},
            )
        if health.get("last_error") not in ("", None):
            return ActivitySnapshot(
                False,
                True,
                "backend_error_present",
                health=health,
                state=state or {},
            )

        if state is None or not isinstance(state, Mapping):
            return ActivitySnapshot(
                False,
                False,
                "state_not_an_object",
                health=health,
                state={},
            )
        if "active_skill" not in state:
            return ActivitySnapshot(
                False,
                False,
                "state_fields_missing:active_skill",
                health=health,
                state=state,
            )
        if state.get("active_skill") is not None:
            return ActivitySnapshot(
                False,
                True,
                "skill_active_or_queued",
                health=health,
                state=state,
            )
        required_state_flags = (
            "reset_pending",
            "task_switch_pending",
            "vision_degraded",
            "simulation_degraded",
        )
        missing_flags = [key for key in required_state_flags if key not in state]
        if missing_flags:
            return ActivitySnapshot(
                False,
                False,
                "state_fields_missing:" + ",".join(missing_flags),
                health=health,
                state=state,
            )
        # ``BehaviorInterface.snapshot_state`` represents no reset request as
        # either ``None`` (current runtime) or ``False`` (older runtime).
        if state.get("reset_pending") not in (None, False):
            return ActivitySnapshot(
                False,
                False,
                "reset_pending",
                health=health,
                state=state,
            )
        if state.get("task_switch_pending") is not False:
            return ActivitySnapshot(
                False,
                False,
                "task_switch_pending",
                health=health,
                state=state,
            )
        if state.get("vision_degraded") is not False or state.get(
            "simulation_degraded"
        ) is not False:
            return ActivitySnapshot(
                False,
                False,
                "backend_degraded",
                health=health,
                state=state,
            )

        reason = "idle_hold"
        return ActivitySnapshot(
            True,
            True,
            reason,
            health=health,
            state=state,
        )

    @classmethod
    def classify_idle_probe(
        cls,
        probe: Mapping[str, Any],
    ) -> ActivitySnapshot:
        """Classify the constant-size interface idle probe.

        The interface computes this payload from its existing state machine;
        the sidecar still treats missing or malformed fields as fail-open.  A
        compact probe is therefore only an optimization of observation, not a
        second source of policy semantics.
        """

        if not isinstance(probe, Mapping):
            return ActivitySnapshot(False, False, "idle_probe_not_an_object")
        if probe.get("ok") is not True:
            return ActivitySnapshot(False, False, "idle_probe_not_ok", health=probe)
        if probe.get("protocol") != "behavior-interface-idle-probe-v1":
            return ActivitySnapshot(
                False,
                False,
                "idle_probe_protocol_missing",
                health=probe,
            )
        if probe.get("diagnostic_ok") is not True:
            return ActivitySnapshot(
                False,
                False,
                str(probe.get("reason") or "idle_probe_diagnostic_failed"),
                health=probe,
            )
        idle = probe.get("idle")
        if not isinstance(idle, bool):
            return ActivitySnapshot(
                False,
                False,
                "idle_probe_idle_flag_missing",
                health=probe,
            )
        return ActivitySnapshot(
            idle,
            True,
            str(probe.get("reason") or ("idle_hold" if idle else "active")),
            health=probe,
            state={},
        )

    def read(self) -> ActivitySnapshot:
        """Read backend state, using the configured failure policy."""

        if self._prefer_idle_probe:
            try:
                probe = self._get("/__official__/idle_probe")
            except HTTPError as exc:
                # A 404 is the expected compatibility signal from an older
                # interface. Other HTTP failures are real diagnostic
                # uncertainty and must fail open without immediately issuing
                # the much larger legacy state request.
                if int(getattr(exc, "code", 0) or 0) != 404:
                    return self._uncertain(
                        f"idle_probe_fetch_failed:HTTP{getattr(exc, 'code', 'error')}"
                    )
            except Exception as exc:  # pragma: no cover - network errors vary
                return self._uncertain(
                    f"idle_probe_fetch_failed:{type(exc).__name__}"
                )
            else:
                if isinstance(probe, Mapping) and (
                    probe.get("protocol") == "behavior-interface-idle-probe-v1"
                ):
                    activity = self._apply_failure_policy(
                        self.classify_idle_probe(probe)
                    )
                    # A live external session is the model thinking, not
                    # motion. The evaluator steps only when this gate
                    # forwards a frame, so releasing every frame for the
                    # whole session burns the tick budget between tools.
                    # Hold stays in place until a tool makes the probe
                    # non-idle. The controller's observation window is
                    # already long enough for the deferred first frame.
                    return activity

        try:
            health = self._get("/__official__/healthz")
        except Exception as exc:  # pragma: no cover - exact network errors vary
            return self._uncertain(
                f"health_fetch_failed:{type(exc).__name__}"
            )

        # Avoid serializing the larger /api/state response for clearly active
        # action sources and configured downstream policies.
        if health.get("action_source") != "hold":
            return self._apply_failure_policy(self.classify_payloads(health, {}))
        downstream = health.get("downstream")
        initialization = health.get("episode_initialization")
        live = health.get("live_unit_test")
        if (
            isinstance(downstream, Mapping)
            and bool(downstream.get("configured"))
        ) or (
            isinstance(initialization, Mapping)
            and initialization.get("ready") is not True
        ) or (
            isinstance(live, Mapping)
            and str(live.get("state") or "") in _ACTIVE_LIVE_STATES
        ):
            return self._apply_failure_policy(self.classify_payloads(health, {}))
        try:
            state = self._get("/api/state")
        except Exception as exc:  # pragma: no cover - exact network errors vary
            return self._uncertain(
                f"state_fetch_failed:{type(exc).__name__}",
                health=health,
            )
        return self._apply_failure_policy(self.classify_payloads(health, state))


class GateCancelled(RuntimeError):
    """The evaluator or backend websocket closed while a hold was deferred."""


class IdleStepGate:
    """Core response-defer state machine, independent of websocket objects."""

    def __init__(
        self,
        reader: BackendActivityReader,
        *,
        poll_interval_s: float = 0.25,
        max_wait_s: float = 0.0,
    ) -> None:
        self.reader = reader
        try:
            poll = float(poll_interval_s)
            max_wait = float(max_wait_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("gate intervals must be numeric") from exc
        if poll <= 0.0:
            raise ValueError("poll_interval_s must be positive")
        if max_wait < 0.0:
            raise ValueError("max_wait_s cannot be negative")
        self.poll_interval_s = poll
        self.max_wait_s = max_wait

    def inspect(self, response: bytes | str) -> GateDecision:
        """Decide whether one backend response may be deferred."""

        try:
            payload = unpackb(response, strict_map_key=False)
        except Exception:
            return GateDecision(
                False,
                ActivitySnapshot(False, False, "response_decode_failed"),
            )
        if mapping_get(payload, "action", None) is None:
            return GateDecision(
                False,
                ActivitySnapshot(False, True, "response_has_no_action"),
            )
        activity = self.reader.read()
        return GateDecision(activity.idle, activity)

    async def _read_activity(self) -> ActivitySnapshot:
        """Read diagnostics without blocking the event loop.

        Test/demonstration readers may expose an async ``read`` method.  The
        production HTTP reader is synchronous and is isolated in the default
        executor, just like the existing interface's non-simulator work.
        """

        read = self.reader.read
        if inspect.iscoroutinefunction(read):
            value = await read()
        else:
            value = await _run_blocking_daemon(read)
        if not isinstance(value, ActivitySnapshot):
            raise TypeError("activity reader returned an invalid snapshot")
        return value

    async def wait_until_released(
        self,
        *,
        cancel_event: asyncio.Event,
        initial: ActivitySnapshot | None = None,
    ) -> ActivitySnapshot:
        """Wait without sending any evaluator action.

        ``max_wait_s=0`` means unbounded.  A positive bound is useful during a
        deployment transition when the backend's observation freshness window
        has not yet been increased; expiry releases the cached hold (fail-open)
        rather than risking an evaluator deadlock.
        """

        started = time.monotonic()
        activity = initial
        while True:
            if cancel_event.is_set():
                raise GateCancelled("websocket closed while waiting for action")
            if activity is not None and not activity.idle:
                return activity
            if (
                self.max_wait_s > 0.0
                and time.monotonic() - started >= self.max_wait_s
            ):
                return ActivitySnapshot(
                    False,
                    True,
                    "idle_wait_timeout",
                    health=(activity.health if activity else {}),
                    state=(activity.state if activity else {}),
                )
            try:
                await asyncio.wait_for(
                    cancel_event.wait(),
                    timeout=self.poll_interval_s,
                )
            except asyncio.TimeoutError:
                pass
            if cancel_event.is_set():
                raise GateCancelled("websocket closed while waiting for action")
            activity = await self._read_activity()
            if not activity.idle:
                return activity


@dataclass(frozen=True)
class ProxyConfig:
    """Runtime configuration for one sidecar instance."""

    listen_host: str = "127.0.0.1"
    listen_port: int = 18091
    backend_policy_uri: str = "ws://127.0.0.1:18081"
    backend_http_url: str = "http://127.0.0.1:15060"
    poll_interval_s: float = 0.25
    max_idle_wait_s: float = 0.0
    backend_connect_timeout_s: float = 30.0
    backend_http_timeout_s: float = 1.5
    fail_closed: bool = False


class IdleStepGateProxy:
    """WebSocket sidecar that applies :class:`IdleStepGate` per connection."""

    def __init__(
        self,
        config: ProxyConfig,
        *,
        reader: BackendActivityReader | None = None,
        gate: IdleStepGate | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.log = logger or LOGGER
        self.reader = reader or BackendActivityReader(
            config.backend_http_url,
            timeout_s=config.backend_http_timeout_s,
            fail_closed=config.fail_closed,
        )
        self.gate = gate or IdleStepGate(
            self.reader,
            poll_interval_s=config.poll_interval_s,
            max_wait_s=config.max_idle_wait_s,
        )
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = {
            "connections": 0,
            "closed_connections": 0,
            "backend_frames": 0,
            "forwarded_frames": 0,
            "deferred_frames": 0,
            "released_deferred_frames": 0,
            "transition_release_frames": 0,
            "diagnostic_fail_open_frames": 0,
            "diagnostic_uncertain_frames": 0,
        }

    def _count(self, key: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + int(amount)

    def status(self) -> dict[str, Any]:
        with self._stats_lock:
            counters = dict(self._stats)
        return {
            "ok": True,
            "mode": "official-idle-step-gate",
            "listen": f"{self.config.listen_host}:{self.config.listen_port}",
            "backend_policy_uri": self.config.backend_policy_uri,
            "backend_http_url": self.config.backend_http_url,
            "poll_interval_s": self.config.poll_interval_s,
            "max_idle_wait_s": self.config.max_idle_wait_s,
            "fail_closed": self.config.fail_closed,
            "counters": counters,
        }

    async def _connect_backend(self):
        deadline = time.monotonic() + float(self.config.backend_connect_timeout_s)
        last_error: Exception | None = None
        while True:
            try:
                return await websocket_client.connect(
                    self.config.backend_policy_uri,
                    compression=None,
                    max_size=None,
                    # The supported deployment is a local sidecar/backend
                    # pair. Never route that hop through HTTP_PROXY/HTTPS_PROXY.
                    proxy=None,
                    open_timeout=min(5.0, max(0.1, self.config.backend_connect_timeout_s)),
                    ping_interval=60,
                    ping_timeout=300,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "could not connect to backend policy websocket within "
                        f"{self.config.backend_connect_timeout_s}s: {exc}"
                    ) from exc
                await asyncio.sleep(0.25)
        assert last_error is not None

    @staticmethod
    async def _watch_closed(websocket, event: asyncio.Event) -> None:
        """Set ``event`` when a modern or legacy websocket closes."""

        try:
            wait_closed = getattr(websocket, "wait_closed", None)
            if callable(wait_closed):
                await wait_closed()
            else:  # pragma: no cover - old websockets fallback
                while not getattr(websocket, "closed", False):
                    await asyncio.sleep(0.25)
        finally:
            event.set()

    @staticmethod
    async def _close_peers_when_event(
        event: asyncio.Event,
        *websockets_to_close: Any,
    ) -> None:
        """Wake a blocked recv on the other side of a half-closed proxy."""

        await event.wait()
        for websocket in websockets_to_close:
            try:
                await websocket.close()
            except Exception:
                pass

    async def handler(self, client_websocket) -> None:
        """Handle one official evaluator connection."""

        self._count("connections")
        backend = None
        # Both sides share one cancellation event.  A backend disconnect while
        # a hold is deferred must wake the same wait as an evaluator disconnect;
        # otherwise the gate could poll an idle HTTP endpoint forever.
        connection_closed = asyncio.Event()
        client_watch = asyncio.create_task(
            self._watch_closed(client_websocket, connection_closed)
        )
        backend_watch = None
        peer_closer = None
        # Keep one post-activity hold step.  It preserves the stock evaluator
        # transition where a completed tool's last action is followed by one
        # observation/termination check before a long idle freeze.
        passed_since_idle = False
        try:
            backend = await self._connect_backend()
            backend_watch = asyncio.create_task(
                self._watch_closed(backend, connection_closed)
            )
            # A close watcher only sets an event; a handler blocked in the
            # opposite ``recv`` wouldn't notice that event by itself. Close
            # both endpoints when either watcher fires so ordinary and
            # deferred paths have the same bounded shutdown behavior.
            peer_closer = asyncio.create_task(
                self._close_peers_when_event(
                    connection_closed,
                    client_websocket,
                    backend,
                )
            )
            metadata = await backend.recv()
            await client_websocket.send(metadata)
            self._count("forwarded_frames")
            while True:
                message = await client_websocket.recv()
                # Reset is a key-presence control frame.  Scan only its
                # top-level msgpack keys so a normal RGB-D observation is not
                # decoded and materialized a second time in the sidecar.
                reset_frame = has_reset_key(message)
                await backend.send(message)
                if reset_frame:
                    # The unchanged interface intentionally sends no response
                    # to reset.  Preserve that exact protocol behavior.
                    passed_since_idle = False
                    continue

                response = await backend.recv()
                self._count("backend_frames")
                decision = await _run_blocking_daemon(self.gate.inspect, response)
                if not decision.activity.diagnostic_ok:
                    self._count("diagnostic_uncertain_frames")
                    if not decision.defer:
                        self._count("diagnostic_fail_open_frames")
                if not decision.defer:
                    await client_websocket.send(response)
                    self._count("forwarded_frames")
                    passed_since_idle = True
                    continue

                if passed_since_idle:
                    await client_websocket.send(response)
                    self._count("forwarded_frames")
                    self._count("transition_release_frames")
                    passed_since_idle = False
                    continue

                self._count("deferred_frames")
                released = await self.gate.wait_until_released(
                    cancel_event=connection_closed,
                    initial=decision.activity,
                )
                if connection_closed.is_set():
                    raise GateCancelled("backend websocket closed while waiting")
                # Always send the exact response generated for this
                # observation.  In particular, never call the backend again
                # with a duplicate observation to obtain a fresher action.
                await client_websocket.send(response)
                self._count("forwarded_frames")
                self._count("released_deferred_frames")
                passed_since_idle = True
                self.log.debug("released deferred hold: %s", released.reason)
        except GateCancelled:
            self.log.info("idle gate connection ended while waiting")
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception:
            self.log.exception("idle gate connection failed")
        finally:
            connection_closed.set()
            if backend is not None:
                try:
                    await backend.close()
                except Exception:
                    pass
            if backend_watch is not None:
                backend_watch.cancel()
                await asyncio.gather(backend_watch, return_exceptions=True)
            if peer_closer is not None:
                peer_closer.cancel()
                await asyncio.gather(peer_closer, return_exceptions=True)
            client_watch.cancel()
            await asyncio.gather(client_watch, return_exceptions=True)
            self._count("closed_connections")

    async def process_request(self, connection, request):
        path = str(getattr(request, "path", "") or "").split("?", 1)[0]
        if path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if path == "/status":
            body = json.dumps(self.status(), separators=(",", ":")) + "\n"
            return connection.respond(http.HTTPStatus.OK, body)
        return None

    async def serve_forever(self) -> None:
        async with websocket_server.serve(
            self.handler,
            self.config.listen_host,
            int(self.config.listen_port),
            compression=None,
            max_size=None,
            ping_interval=None,
            process_request=self.process_request,
        ) as server:
            self.log.info(
                "idle step gate listening on %s:%s -> %s",
                self.config.listen_host,
                self.config.listen_port,
                self.config.backend_policy_uri,
            )
            await server.serve_forever()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Proxy the official evaluator websocket and suppress idle hold steps."
    )
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18091)
    parser.add_argument(
        "--backend-policy-uri",
        default="ws://127.0.0.1:18081",
        help="unchanged interface policy websocket URI",
    )
    parser.add_argument(
        "--backend-http-url",
        required=False,
        default="http://127.0.0.1:15060",
        help="unchanged interface public HTTP base URL",
    )
    parser.add_argument("--poll-interval-s", type=float, default=0.25)
    parser.add_argument(
        "--max-idle-wait-s",
        type=float,
        default=0.0,
        help="0 means unbounded; positive values fail-open after the bound",
    )
    parser.add_argument("--backend-connect-timeout-s", type=float, default=30.0)
    parser.add_argument("--backend-http-timeout-s", type=float, default=1.5)
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help="defer action frames when the idle diagnostic is uncertain",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    config = ProxyConfig(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        backend_policy_uri=args.backend_policy_uri,
        backend_http_url=args.backend_http_url,
        poll_interval_s=args.poll_interval_s,
        max_idle_wait_s=args.max_idle_wait_s,
        backend_connect_timeout_s=args.backend_connect_timeout_s,
        backend_http_timeout_s=args.backend_http_timeout_s,
        fail_closed=args.fail_closed,
    )
    proxy = IdleStepGateProxy(config)
    try:
        asyncio.run(proxy.serve_forever())
    except KeyboardInterrupt:
        LOGGER.info("idle step gate stopped")


if __name__ == "__main__":  # pragma: no cover
    main()
