"""Agent-side client for atomic evaluator actions and allowed observations."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import fcntl
from pathlib import Path
import time
import numpy as np

from .controller import JointController
from .controller_state import Pose
from .observation import Observation
from .protocol import request
from .settings import configure


class Tracking:
    def __init__(self, robot):
        self.robot = robot

    def replace_points(self, points, *, check_cancelled=None, commit_guard=None,
                       commit_result=None, **binding):
        """Keep caller callbacks local; only point/binding data crosses IPC.

        The driver holds its episode lock throughout replacement. The local
        operation lock prevents a second tool from acting between that commit
        and publication of the result in this Python process.
        """
        for name, callback in (("check_cancelled", check_cancelled),
                               ("commit_guard", commit_guard),
                               ("commit_result", commit_result)):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable")
        guard = commit_guard() if commit_guard is not None else nullcontext(True)
        with guard as allowed:
            if allowed is False:
                raise ValueError("track_object_distance request was cancelled before commit")
            if check_cancelled is not None:
                check_cancelled()
            committed = self.robot.rpc("tracking", method="replace_points", args=[points], kwargs=binding)
            if commit_result is not None:
                commit_result(deepcopy(committed))
            return committed

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.robot.rpc("tracking", method=name, args=args, kwargs=kwargs)


class Robot:
    """One session's joint API. All state comes from evaluator observations.

    `step(action)` sends exactly one 27-value controller command and waits for
    the resulting observation. `make_action` only builds a command; it does
    not move the robot. Use `exclusive()` around a complete operation.
    """
    control_hz = 30.
    robot_dof = 8
    _codex_pinned_actions_v10 = True
    _official_async_capture_artifacts = False

    def __init__(self):
        self.config = configure()
        self.workspace = Path(self.config["workspace"])
        self.state_dir = Path(self.config["state_dir"])
        self.session_id = self.config["session_id"]
        self.task_name = self.config.get("task_name", "")
        self._official_adapter = Observation()
        self._official_tracked_object_distances = Tracking(self)
        self.world = self
        self.result = None
        self.results = []
        self.logs = []
        self.step_count = 0
        self._lock_handle = None
        self.history = None
        self.update(self.rpc("observe"))

    def rpc(self, operation, **fields):
        value = request(self.config["socket"], {"operation": operation, "session_id": self.session_id,
            "episode_id": self.config["episode_id"], **fields})
        if isinstance(value, dict) and "__pose__" in value:
            value = value["__pose__"]
            return Pose(np.asarray(value["pos"]), np.asarray(value["quat"]))
        return value

    def update(self, payload):
        self._official_adapter.update(payload["snapshot"])
        self.tracking = payload.get("tracking", {})
        self.budget = payload.get("budget", {})
        self.ticks = payload["ticks"]

    def observe(self):
        self.update(self.rpc("observe"))
        return self._official_adapter.snapshot()

    @contextmanager
    def exclusive(self):
        if self._lock_handle is not None:
            yield self
            return
        with (self.state_dir / "robot.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another tool is controlling this robot; wait for its result") from None
            self._lock_handle = handle
            try:
                self.observe()
                yield self
            finally:
                self._lock_handle = None
                fcntl.flock(handle, fcntl.LOCK_UN)

    def step(self, action):
        if self._lock_handle is None:
            raise RuntimeError("Use `with robot.exclusive():` to own a complete control operation")
        vector = np.asarray(action)
        if vector.shape != (27,):
            raise ValueError("A controller action contains exactly 27 values")
        before = self._official_adapter.current.sequence
        self.update(self.rpc("step", action=vector.tolist(), sequence=before))
        if self.history is not None:
            self.history.action(vector, before, self._official_adapter.current.sequence)
        self.step_count += 1
        return self._official_adapter.snapshot()

    def __getattr__(self, name):
        # Only genuine joint/odometry methods are delegated. Numerical
        # generators and planning always execute in the local Python process.
        if not name.startswith("_") and callable(getattr(JointController, name, None)):
            return lambda *args, **kwargs: self.rpc("joints", method=name, args=args, kwargs=kwargs)
        raise AttributeError(name)

    def set_result(self, result):
        self.result = result
        self.results.append(result)
        return result

    def log(self, message):
        self.logs.append({"time": time.time(), "message": str(message)})

    def set_status(self, message):
        self.log(message)

    def get_last_result(self):
        return self.result

    def is_cancelled(self):
        return False

    def raise_if_cancelled(self, where=""):
        return None

    def execute(self, generator, **arguments):
        for action in generator(self, **arguments):
            self.step(action)
        return self.result

    def _workspace_tool(self, name, arguments):
        """Execute this session's readable file, with the same history as Bash."""
        import runpy
        from .tool import run
        module = runpy.run_path(str(self.workspace / "tools" / (name + ".py")))
        self.result, self.results, self.logs, self.step_count = None, [], [], 0
        return run(name, module["SCHEMA"], module["actions"], arguments, robot=self)

    def command(self, action, *, steps=1):
        """Recorded direct 27D control, followed by refreshed perception/images."""
        return self._workspace_tool("robot_step", {"action": list(action), "steps": steps})

    def capture(self, role="head"):
        """Recorded read-only RGB-D capture; returns native-reader image paths."""
        if role not in {"head", "left_wrist", "right_wrist"}:
            raise ValueError("Unknown camera role")
        return self._workspace_tool(f"capture_{role}_camera", {})
