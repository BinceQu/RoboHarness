"""Model-visible call records (input, receipt, returned/crosshair images); bulk evidence goes to the trajectory dir."""
import json
import fcntl
from pathlib import Path
import time
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFont

from .files import atomic_json, load_json
from .receipt import without_ok


def points(value, image_id=None, label="point"):
    if isinstance(value, dict):
        image_id = value.get("image_id", image_id)
        if {"u", "v"}.issubset(value):
            yield {"image_id": image_id, "u": value["u"], "v": value["v"], "label": value.get("name", label)}
        for key, item in value.items():
            yield from points(item, image_id, str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from points(item, image_id, f"{label}[{index}]")


class CallHistory:
    def __init__(self, robot, name, arguments):
        self.robot, self.name = robot, name
        state_path = robot.state_dir / "state.json"
        state = load_json(state_path, {"counter": 0, "latest_image_id": "", "active_skill": None})
        number = state["counter"]
        while True:
            number += 1
            self.path = robot.workspace / "tool_call_history" / f"{number:06d}_{name}"
            try:
                self.path.mkdir()
                self.calls = Path(robot.config["trajectory_dir"]) / "calls" / self.path.name
                self.calls.mkdir(parents=True, exist_ok=True)
                break
            except FileExistsError:
                continue
        state["counter"] = number
        atomic_json(state_path, state)
        self.number, self.started = number, time.time()
        atomic_json(self.path / "input.json", {"tool": name, "arguments": arguments,
            "session_id": robot.session_id, "episode_id": robot.config["episode_id"],
            "started_at": self.started,
            "observation_sequence": getattr(robot._official_adapter.current, "sequence", None)})
        self.selected_points = list(points(arguments))
        self.annotate_clicks()

    def annotate_clicks(self):
        grouped = {}
        for point in self.selected_points:
            image_id = point["image_id"]
            if not isinstance(image_id, str) or not image_id.replace("_", "").isalnum():
                continue
            grouped.setdefault(image_id, []).append(point)
        for image_id, selected in grouped.items():
            source = (Path(self.robot.config["trajectory_dir"]) / "recordings" / self.robot.session_id
                      / "images" / (image_id + ".png"))
            if not source.is_file():
                continue
            with Image.open(source) as source_image:
                image = source_image.convert("RGB")
            draw = ImageDraw.Draw(image)
            for index, point in enumerate(selected):
                u, v = point["u"], point["v"]
                if type(u) is not int or type(v) is not int or not (0 <= u < image.width and 0 <= v < image.height):
                    continue
                color = (255, 0, 100) if index % 2 == 0 else (0, 220, 230)
                for a, b in (((u - 16, v), (u - 4, v)), ((u + 4, v), (u + 16, v)),
                             ((u, v - 16), (u, v - 4)), ((u, v + 4), (u, v + 16))):
                    draw.line((a, b), fill="white", width=5)
                    draw.line((a, b), fill=color, width=3)
                draw.ellipse((u - 3, v - 3, u + 3, v + 3), outline=color, width=2)
                text = f"{point['label']}: ({u}, {v})"
                x, y = min(u + 19, image.width - 230), max(4, v - 24)
                draw.text((max(4, x), y), text, font=ImageFont.load_default(size=16), fill=color, stroke_width=2, stroke_fill="white")
            image.save(self.path / f"click_{image_id}_crosshair.png")
        atomic_json(self.calls / "selected_points.json", self.selected_points)

    def action(self, action, before, after):
        with (self.calls / "joint_commands.jsonl").open("a") as stream:
            stream.write(json.dumps({"sequence_before": before, "sequence_after": after,
                                    "action": [float(value) for value in action]}, allow_nan=False) + "\n")

    def finish(self, receipt, full_result=None):
        receipt = without_ok(receipt)
        if full_result is not None:
            atomic_json(self.calls / "output.json", without_ok(full_result))
        receipt.update(record=str(self.path / "result.json"), duration_s=time.time() - self.started)
        atomic_json(self.path / "result.json", receipt)
        if self.robot.logs:
            atomic_json(self.calls / "execution_log.json", without_ok(self.robot.logs))
        atomic_json(self.robot.state_dir / "last_result.json", receipt)
        return receipt


def record_failure(name, arguments, receipt):
    """Record a rejected CLI call even when no robot connection was possible.

    Reuse the operation lock so allocating its number cannot overwrite another
    call's state. No observation is attached: a connection failure cannot prove
    that the previously saved image still represents the current robot.
    """
    from .settings import settings
    receipt = without_ok(receipt)
    try:
        config = settings()
        context = SimpleNamespace(config=config, workspace=Path(config["workspace"]),
            state_dir=Path(config["state_dir"]), session_id=config["session_id"],
            _official_adapter=SimpleNamespace(current=None), logs=[])
        with (context.state_dir / "robot.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            history = CallHistory(context, name, arguments)
            return history.finish(receipt, receipt.copy())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        # An uninitialized template has nowhere legitimate to store session
        # history. Preserve the original error and make this limitation visible.
        receipt["history_error"] = f"{type(exc).__name__}: {exc}"
        return receipt
