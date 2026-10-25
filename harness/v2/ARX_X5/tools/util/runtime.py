"""Session-local state worker. Physics is exclusively owned by the evaluator.

The evaluator calls reset/update_obs/get_action. Bash requests enter a private
Unix socket. The original generators run from tools/NAME.py and yield the
atomic joint dictionaries consumed by the evaluator's take_action method.
"""

import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import struct
import threading
import time

from tools.util.action_world import RoboDojoActionWorld
from tools.util.context import ToolContext
from tools.util.observation import ObservationAdapter
from tools.util.storage import PlanStore, SnapshotStore
from tools.util.tracking import PointTracker
from tools.util.scheduler import ToolScheduler
from tools.util.settings import initialize_state, write_json
from tools.util.history import History
from tools.joint_command import validate_command


def socket_address(root):
    # Linux abstract sockets avoid Unix's 108-byte path limit in long sessions.
    # SO_PEERCRED below restricts access to the workspace owner.
    return "\0arx-x5-" + hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()


class Runtime:
    def __init__(self, root):
        self.root = Path(root).resolve()
        initialize_state(self.root)
        self.config = json.loads((self.root / ".session.json").read_text())
        self.session_id = self.config["session_id"]
        self.operation = threading.Lock()
        self.history = None
        self.server = None
        self.body_renderer = None
        self.adapter = ObservationAdapter()
        self.world = RoboDojoActionWorld(self.adapter)
        self.tracker = PointTracker(base_rotation=self.world.base_rotation)
        self.world._official_tracked_object_distances = self.tracker
        key_path = self.root / ".runtime/plan.key"
        if not key_path.exists():
            key_path.write_bytes(secrets.token_bytes(32))
        secret = key_path.read_bytes()
        self.snapshots = SnapshotStore(self.root / ".runtime/artifacts", secret=secret)
        self.plans = PlanStore(self.root / ".runtime/artifacts", secret=secret)
        self.scheduler = ToolScheduler(
            adapter=self.adapter,
            world=self.world,
            tracker=self.tracker,
            snapshots=self.snapshots,
            plans=self.plans,
            task_name=self.config.get("task", ""),
        )
        self.scheduler.set_capture_session_id(self.session_id)
        dispatched = self.world.mark_action_dispatched

        def record_action(action, **kwargs):
            validate_command(action)
            if self.history is None:
                raise RuntimeError("No active tool owns this action")
            self.history.action(
                action, self.adapter.status()["sequence"], self.world.episode_id()
            )
            return dispatched(action, **kwargs)

        self.world.mark_action_dispatched = record_action

    def context(self):
        return ToolContext(
            world=self.world,
            snapshots=self.snapshots,
            plans=self.plans,
            tracker=self.tracker,
            task_name=self.config.get("task", ""),
            cancelled=threading.Event(),
        )

    def reset(self):
        self.scheduler.reset()
        self.tracker.reset()
        self.world.reset()
        self.adapter.reset()
        state = json.loads((self.root / ".runtime/state.json").read_text())
        state.update(latest_image_id=None, active_skill=None)
        write_json(self.root / ".runtime/state.json", state)

    def update_obs(self, obs):
        self.scheduler.ingest(dict(obs))

    def get_action(self):
        self.adapter.touch()
        action = self.scheduler.next_action()
        return [] if action is None else [validate_command(action)]

    def status(self):
        return {
            "ok": True,
            "workspace": str(self.root),
            "robot": "ARX_X5",
            "evaluator": self.adapter.status(),
            "scheduler": self.scheduler.status(),
        }

    def call(self, name, arguments):
        from tools.util.capabilities import PUBLIC_TOOLS

        if name not in (
            *PUBLIC_TOOLS,
            "joint_command",
            "read_proprioception",
            "activate_skill",
            "deactivate_skill",
        ):
            return {"error": "Unknown workspace tool"}
        with self.operation:
            arguments = dict(arguments)
            if arguments.get("session_id", self.session_id) != self.session_id:
                return {"error": "Tool cannot access another session"}
            arguments["session_id"] = self.session_id
            self.history = History(self.root, name, arguments)
            result = {}
            try:
                if name in ("activate_skill", "deactivate_skill"):
                    from importlib import import_module

                    result = getattr(import_module("tools." + name), name)(
                        self.root, arguments
                    )
                elif name == "read_proprioception":
                    from tools.read_proprioception import read_proprioception

                    result = read_proprioception(self, **arguments)
                else:
                    if not self.adapter.connected():
                        raise RuntimeError(
                            "Evaluator is not connected; no command was sent"
                        )
                    self.history.crosshairs(arguments, self.snapshots, self.session_id)
                    job = self.scheduler.submit(name, arguments)
                    try:
                        result = self.scheduler.wait(job, timeout_s=3600)
                    except TimeoutError:
                        self.scheduler.cancel(job)
                        raise
                    finally:
                        self.scheduler.release(job)
                    from tools.util.perception import publish

                    result = publish(self, self.history, result)
            except Exception as exc:
                result = {
                    **result,
                    "tool": name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            try:
                return self.history.finish(result)
            finally:
                self.history = None

    def start(self):
        runtime = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                _pid, uid, _gid = struct.unpack(
                    "3i",
                    self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12),
                )
                if uid not in (os.getuid(), 0):
                    return
                try:
                    payload = json.loads(self.rfile.readline(2 * 1024 * 1024))
                    if payload.get("method") == "status":
                        result = runtime.status()
                    elif payload.get("method") == "call":
                        result = runtime.call(
                            payload["tool"], payload.get("arguments", {})
                        )
                    else:
                        raise ValueError(
                            "Only status and call are public workspace methods"
                        )
                    self.wfile.write(
                        json.dumps(result, allow_nan=False).encode() + b"\n"
                    )
                except Exception as exc:
                    self.wfile.write(
                        json.dumps({"error": str(exc)}).encode() + b"\n"
                    )

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True

        self.server = Server(socket_address(self.root), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.scheduler.reset()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.body_renderer is not None:
            self.body_renderer.close()
            self.body_renderer = None
