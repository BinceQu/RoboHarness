"""Direct evaluator connection: allowed observations in, 27D actions out.

No macro-tool dispatcher, simulator handle, web interface, agent or model
client exists here. Python tools execute in the agent's own workspace.
"""
import argparse
import asyncio
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
import json
import os
from pathlib import Path
import socketserver
import threading
import time

import numpy as np

from .controller import JointController
from .controller_state import Pose
from .initialization import EpisodeGraspPrepController
from .model import validate_action
from .budget import sanitize_budget
from .observation import Observation, Snapshot, filter_observation
from .protocol import pack, unpack, send, receive
from .geometry.tracked_object_distance import TrackedObjectDistanceMemory, tracker_frame_from_allowed_observation


@dataclass
class Command:
    action: np.ndarray
    event: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: str | None = None


class Driver:
    def __init__(self, status_path=None):
        self.condition = threading.Condition(threading.RLock())
        self.adapter = Observation()
        self.world = JointController(proprio_provider=self.adapter.proprio_vector,
                                     hold_action_provider=self.adapter.hold_action,
                                     eef_pose_provider=self.adapter.eef_pose)
        self.tracker = TrackedObjectDistanceMemory()
        self.initializer = EpisodeGraspPrepController(arm_dof=8)
        self.pending = None
        self.inflight = None
        self.connected = False
        self.binding = None
        self.sequence = 0
        self.ticks = 0
        self.budget = {}
        self.status_path = Path(status_path) if status_path else None
        self.finishing = False
        self.world_methods = {name for name in dir(self.world)
                              if not name.startswith("_") and callable(getattr(self.world, name))}
        self.tracker_methods = {name for name in dir(self.tracker)
                                if not name.startswith("_") and callable(getattr(self.tracker, name))}

    def status(self):
        return {"connected": self.connected, "episode_id": self.world.episode_id(),
                "sequence": self.sequence, "ticks": self.ticks,
                "initialization": self.initializer.status(), "session_id": self.binding,
                "pending_action": self.pending is not None, "inflight_action": self.inflight is not None,
                "architecture": "workspace Python -> 27D action socket -> evaluator WebSocket"}

    def write_status(self):
        if self.status_path:
            temp = self.status_path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.status(), indent=2) + "\n")
            temp.replace(self.status_path)

    def cancel_commands(self, message):
        for command in (self.pending, self.inflight):
            if command is not None:
                command.error = message
                command.event.set()
        self.pending = self.inflight = None

    def reset(self):
        with self.condition:
            self.cancel_commands("Evaluator reset ended this episode")
            self.world.reset_observation_state()
            self.tracker.reset()
            self.initializer.reset()
            self.adapter.current = Snapshot({}, self.sequence, 0., self.world.episode_id())
            self.binding = None
            self.ticks = 0
            self.budget = {}
            self.finishing = False
            self.write_status()
            self.condition.notify_all()

    def disconnected(self):
        with self.condition:
            self.connected = False
            self.cancel_commands("Evaluator disconnected")
            self.write_status()
            self.condition.notify_all()

    def observe(self, raw):
        """Called exactly once per native evaluator observation."""
        with self.condition:
            self.sequence += 1
            self.adapter.current = Snapshot(filter_observation(raw), self.sequence, time.time(), self.world.episode_id())
            # Privileged goal/state fields are discarded, never copied into
            # snapshots, the workspace or robot responses.
            budget = raw.get("__behavior_rollout_budget__", {})
            self.budget = sanitize_budget(budget)
            self.world.reconcile_base_odometry_from_proprio(1. / 30.)
            try:
                frame = tracker_frame_from_allowed_observation(
                    self.adapter.current.observation, sequence=self.sequence,
                    timestamp_s=self.adapter.current.received_ts, episode_id=self.world.episode_id(),
                    base_xy_yaw=self.world.policy_local_base_pose(), copy_arrays=False)
                self.tracker.ingest(frame, copy_frame=False)
            except (ValueError, TypeError) as exc:
                self.tracker.note_observation_error(exc)
            if self.inflight is not None:
                self.inflight.result = self.payload()
                self.inflight.event.set()
                self.inflight = None
            if not self.initializer.ready():
                action = self.initializer.step(self.world)
                self.world.set_episode_initialized(self.initializer.ready())
                self.ticks += 1
                self.write_status()
                return action
            self.world.set_episode_initialized(True)
            self.write_status()
            self.condition.notify_all()
            # The evaluator pauses here while the model thinks. No wall-time
            # idle actions burn physics budget or change the observed scene.
            while self.pending is None and self.connected and not self.finishing:
                self.condition.wait(timeout=1.)
            if not self.connected:
                raise RuntimeError("Evaluator disconnected while awaiting action")
            if self.finishing:
                self.ticks += 1
                return self.world.hold_action()
            command = self.pending
            self.pending = None
            self.inflight = command
            self.ticks += 1
            self.write_status()
            return command.action

    def payload(self):
        return {"snapshot": asdict(self.adapter.current), "tracking": self.tracker.memory_fields(),
                "ticks": self.ticks, "budget": self.budget}

    def check_session(self, request):
        if not self.connected or not self.initializer.ready():
            raise RuntimeError("Evaluator is not ready")
        if not self.binding or request.get("session_id") != self.binding:
            raise ValueError("Request does not own this session")
        if request.get("episode_id") != self.world.episode_id():
            raise ValueError("Request belongs to an expired episode")

    def request(self, request):
        operation = request.get("operation")
        with self.condition:
            if operation == "status":
                return self.status()
            if operation == "bind":
                if not self.connected or not self.initializer.ready() or self.binding:
                    raise RuntimeError("Episode is not ready or is already bound")
                if request.get("episode_id") != self.world.episode_id():
                    raise ValueError("Cannot bind an expired episode")
                self.binding = str(request["session_id"])
                self.write_status()
                return self.payload()
            if operation == "finish":
                self.finishing = True
                self.condition.notify_all()
                return {"finishing": True}
            self.check_session(request)
            if operation == "observe":
                return self.payload()
            if operation in ("joints", "tracking"):
                name = request.get("method")
                allowed = self.world_methods if operation == "joints" else self.tracker_methods
                if name not in allowed or name in {"reset", "reset_observation_state", "set_episode_initialized"}:
                    raise ValueError("Unsupported atomic/state operation")
                target = self.world if operation == "joints" else self.tracker
                result = getattr(target, name)(*request.get("args", []), **request.get("kwargs", {}))
                return {"__pose__": result.as_dict()} if isinstance(result, Pose) else result
            if operation != "step":
                raise ValueError("Unknown control operation")
            if self.pending is not None or self.inflight is not None:
                raise RuntimeError("Another action is in flight; concurrent robot writers are forbidden")
            if request.get("sequence") != self.sequence:
                raise ValueError("Action was prepared from an outdated observation")
            command = Command(validate_action(request["action"]))
            self.pending = command
            self.condition.notify_all()
        # Return only once a new observation confirms the action was consumed.
        # A disconnected/reset evaluator wakes this waiter with an error.
        if not command.event.wait(timeout=180):
            with self.condition:
                if self.pending is command:
                    self.pending = None
            raise TimeoutError("No evaluator observation arrived after the submitted action")
        if command.error:
            raise RuntimeError(command.error)
        return command.result


class ControlServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class ControlHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            value = self.server.driver.request(receive(self.request))
            send(self.request, {"ok": True, "value": value})
        except Exception as exc:
            try:
                send(self.request, {"ok": False, "error": str(exc),
                    "error_type": type(exc).__name__, "reason": getattr(exc, "reason", None)})
            except OSError:
                pass


async def serve(driver, host, port):
    from websockets.asyncio.server import serve as websocket_serve

    def health_check(connection, request):
        # The official WebsocketPolicy performs this HTTP probe before its
        # first connection. It is part of the evaluator protocol, not a
        # separate robot-control HTTP API.
        if request.path == "/healthz":
            return connection.respond(HTTPStatus.OK, "OK\n")
        return None

    async def evaluator(socket):
        with driver.condition:
            if driver.connected:
                await socket.close(code=1013, reason="An evaluator already owns this driver")
                return
            driver.connected = True
        try:
            async def watch_disconnect():
                await socket.wait_closed()
                driver.disconnected()
            disconnect_watcher = asyncio.create_task(watch_disconnect())
            await socket.send(pack({"policy": "R1 Pro workspace atomic driver", "action_dim": 27}))
            async for message in socket:
                observation = unpack(message)
                if "reset" in observation:
                    driver.reset()
                    continue
                started = time.monotonic()
                action = await asyncio.to_thread(driver.observe, observation)
                await socket.send(pack({"action": action, "server_timing": {"infer_ms": (time.monotonic() - started) * 1000}}))
        finally:
            if "disconnect_watcher" in locals():
                disconnect_watcher.cancel()
            driver.disconnected()

    async with websocket_serve(evaluator, host, port, compression=None, max_size=None,
                              ping_interval=None, process_request=health_check):
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    if args.socket.exists():
        raise RuntimeError("Control socket already exists; use a fresh session/runtime directory")
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    driver = Driver(args.status)
    with ControlServer(str(args.socket), ControlHandler) as server:
        os.chmod(args.socket, 0o600)
        server.driver = driver
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            asyncio.run(serve(driver, "127.0.0.1", args.policy_port))
        finally:
            driver.disconnected()
            server.shutdown()


if __name__ == "__main__":
    main()
